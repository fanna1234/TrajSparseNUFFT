#!/usr/bin/env python3
"""Direction-balanced native production/cuFINUFFT full-CG campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path


CASES = {
    "spiral": {
        "trajectory": "experiments/data_prep/trajectories/spiral_standard_256.f32xy.bin",
        "pack": "spiral65",
        "raw_scale": 1185.363037109375,
    },
    "radial": {
        "trajectory": "experiments/data_prep/trajectories/radial_256x256.f32xy.bin",
        "pack": "radial65",
        "raw_scale": 1187.88720703125,
    },
    "golden": {
        "trajectory": "experiments/data_prep/trajectories/golden_256x256.f32xy.bin",
        "pack": "golden65",
        "raw_scale": 1187.2781982421875,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_geomean_ci(ratios: list[float], seed: int) -> dict:
    logs = [math.log(value) for value in ratios]
    rng = random.Random(seed)
    samples = []
    for _ in range(50000):
        draw = [logs[rng.randrange(len(logs))] for _ in logs]
        samples.append(math.exp(statistics.fmean(draw)))
    return {
        "geomean": math.exp(statistics.fmean(logs)),
        "ci95_low": quantile(samples, 0.025),
        "ci95_high": quantile(samples, 0.975),
    }


def parse_result(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("{"):
            payload = json.loads(line)
            if payload.get("correct") and "us" in payload:
                return payload
    raise RuntimeError("missing benchmark result")


def foreign_processes() -> list[str]:
    process = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    )
    allowed = ("gnome-remote-desktop-daemon", "Xorg")
    return [
        line for line in process.stdout.splitlines()
        if line.strip() and not any(name in line for name in allowed)
    ]


def gpu_state() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,power.limit,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--production-binary", type=Path, required=True)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--quality-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--method", type=int, default=2)
    parser.add_argument("--forward-method", type=int)
    parser.add_argument("--adjoint-method", type=int)
    parser.add_argument("--gpu-sort", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--raw-scale",
        type=float,
        help="optional truth-free raw scale shared by every case",
    )
    args = parser.parse_args()
    if args.raw_scale is not None and args.raw_scale <= 0.0:
        raise ValueError("raw-scale must be positive")

    repo = args.repo_root.resolve()
    production_binary = args.production_binary.resolve()
    baseline_binary = args.baseline_binary.resolve()
    runtime_root = args.runtime_root.resolve()
    packed_root = args.packed_root.resolve()
    quality_summary = args.quality_summary.resolve()
    output = args.output.resolve()
    quality = json.loads(quality_summary.read_text())
    quality_admitted = quality.get(
        "all_output_application_pass", quality.get("all_application_pass", False)
    )
    if not quality_admitted:
        raise RuntimeError("quality gate is not closed")
    forward_method = args.forward_method or args.method
    adjoint_method = args.adjoint_method or args.method
    if quality.get("binary_sha256") != sha256(baseline_binary):
        raise RuntimeError("quality admission binary does not match timing binary")
    if quality.get("forward_method") != forward_method:
        raise RuntimeError("quality admission forward method does not match timing")
    if quality.get("adjoint_method") != adjoint_method:
        raise RuntimeError("quality admission adjoint method does not match timing")
    if quality.get("gpu_sort") != args.gpu_sort:
        raise RuntimeError("quality admission sort setting does not match timing")
    if args.raw_scale is not None and quality.get("raw_scale") != args.raw_scale:
        raise RuntimeError("quality admission scale does not match timing")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    raw = output / "raw_processes"
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    snapshots = []
    with open("/tmp/trajsparsenufft_gpu.lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for pair in range(args.pairs):
            for case_index, (case, specification) in enumerate(CASES.items()):
                order = (
                    ("production", "cufinufft")
                    if (pair + case_index) % 2 == 0
                    else ("cufinufft", "production")
                )
                pair_payloads = {}
                for variant in order:
                    foreign = foreign_processes()
                    snapshots.append(
                        {
                            "pair": pair,
                            "case": case,
                            "variant": variant,
                            "foreign": foreign,
                            "gpu_state": gpu_state(),
                        }
                    )
                    if foreign:
                        raise RuntimeError(f"foreign CUDA process before {pair}/{case}/{variant}: {foreign}")
                    runtime = runtime_root / "frame00" / f"{case}65"
                    raw_scale = (
                        args.raw_scale
                        if args.raw_scale is not None
                        else specification["raw_scale"]
                    )
                    if variant == "production":
                        pack = packed_root / specification["pack"]
                        command = [
                            str(production_binary),
                            str(pack / "G/real"), str(pack / "G/imag"),
                            str(pack / "G_T/real"), str(pack / "G_T/imag"),
                            str(runtime), repr(raw_scale / 512.0),
                            str(args.warmup), str(args.iters), "0", "0", "10",
                        ]
                    else:
                        command = ["env"]
                        if args.raw_scale is not None:
                            command.append(
                                f"JKF_CUFINUFFT_OPERATOR_SCALE={args.raw_scale}"
                            )
                        command.extend([
                            f"JKF_CUFINUFFT_FORWARD_METHOD={forward_method}",
                            f"JKF_CUFINUFFT_ADJOINT_METHOD={adjoint_method}",
                            f"JKF_CUFINUFFT_GPU_SORT={args.gpu_sort}",
                            str(baseline_binary),
                            str(repo / specification["trajectory"]), str(runtime),
                            str(args.warmup), str(args.iters), "0", "10",
                            str(args.method), "5e-4",
                        ])
                    started_ns = time.time_ns()
                    process = subprocess.run(
                        command, text=True, capture_output=True, check=False
                    )
                    ended_ns = time.time_ns()
                    log = raw / f"pair{pair}_{case}_{variant}.log"
                    log.write_text(
                        "$ " + " ".join(command) + "\n\n[stdout]\n" + process.stdout
                        + "\n[stderr]\n" + process.stderr
                    )
                    if process.returncode != 0:
                        raise RuntimeError(
                            f"{variant} failed at {pair}/{case}: {process.returncode}"
                        )
                    payload = parse_result(process.stdout)
                    pair_payloads[variant] = payload
                    rows.append(
                        {
                            "pair": pair, "case": case, "variant": variant,
                            "order": list(order), "command": command,
                            "started_ns": started_ns, "ended_ns": ended_ns,
                            "payload": payload, "raw_log": str(log),
                        }
                    )
                production_us = float(pair_payloads["production"]["us"])
                baseline_us = float(pair_payloads["cufinufft"]["us"])
                with (output / "paired_rows.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "pair": pair, "case": case, "order": list(order),
                                "production_us": production_us,
                                "cufinufft_us": baseline_us,
                                "cufinufft_over_production": baseline_us / production_us,
                            },
                            sort_keys=True,
                        ) + "\n"
                    )

    paired_rows = [
        json.loads(line)
        for line in (output / "paired_rows.jsonl").read_text().splitlines()
        if line
    ]
    aggregate_rows = []
    for pair in range(args.pairs):
        selected = [row for row in paired_rows if row["pair"] == pair]
        production_sum = sum(row["production_us"] for row in selected)
        baseline_sum = sum(row["cufinufft_us"] for row in selected)
        aggregate_rows.append(
            {
                "pair": pair, "production_sum_us": production_sum,
                "cufinufft_sum_us": baseline_sum,
                "cufinufft_over_production": baseline_sum / production_sum,
            }
        )
    ratios = [row["cufinufft_over_production"] for row in aggregate_rows]
    interval = bootstrap_geomean_ci(ratios, 20260825)
    case_summaries = []
    for case in CASES:
        selected = [row for row in paired_rows if row["case"] == case]
        case_ratios = [row["cufinufft_over_production"] for row in selected]
        case_interval = bootstrap_geomean_ci(case_ratios, 20260900 + len(case_summaries))
        case_summaries.append(
            {
                "case": case,
                "paired_geomean_speedup": case_interval["geomean"],
                "paired_ci95_low": case_interval["ci95_low"],
                "paired_ci95_high": case_interval["ci95_high"],
                "wins": sum(value > 1.0 for value in case_ratios),
            }
        )
    all_wins = all(value > 1.0 for value in ratios)
    summary = {
        "experiment_id": "cufinufft_baseline",
        "production_binary": str(production_binary),
        "production_binary_sha256": sha256(production_binary),
        "baseline_binary": str(baseline_binary),
        "baseline_binary_sha256": sha256(baseline_binary),
        "quality_summary": str(quality_summary),
        "quality_summary_sha256": sha256(quality_summary),
        "quality_admission_field": (
            "all_output_application_pass"
            if "all_output_application_pass" in quality
            else "all_application_pass"
        ),
        "pairs": args.pairs,
        "warmup": args.warmup,
        "iters": args.iters,
        "method": args.method,
        "forward_method": args.forward_method or args.method,
        "adjoint_method": args.adjoint_method or args.method,
        "gpu_sort": args.gpu_sort,
        "scale_rule": (
            {
                "kind": "fixed_truth_free",
                "raw_scale": args.raw_scale,
                "production_native_scale": args.raw_scale / 512.0,
            }
            if args.raw_scale is not None
            else {"kind": "legacy_per_case"}
        ),
        "case_summaries": case_summaries,
        "aggregate_rows": aggregate_rows,
        "paired_geomean_speedup": interval["geomean"],
        "paired_ci95_low": interval["ci95_low"],
        "paired_ci95_high": interval["ci95_high"],
        "wins": sum(value > 1.0 for value in ratios),
        "all_wins": all_wins,
        "snapshots": snapshots,
        "decision": (
            "ACCEPT" if all_wins and interval["ci95_low"] >= 1.20
            else "MEASURED-NONHEADLINE"
        ),
    }
    (output / "records.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
