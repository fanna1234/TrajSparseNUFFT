#!/usr/bin/env python3
"""Direction-balanced sparse versus dense residual-panel full-CG campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from pathlib import Path


CASES = {
    "spiral": 1185.363037109375,
    "radial": 1187.88720703125,
    "golden": 1187.2781982421875,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_timing(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("{"):
            payload = json.loads(line)
            if "us" in payload:
                return payload
    raise RuntimeError("process emitted no timing JSON")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict:
    return {
        "p10_us": quantile(values, 0.10),
        "median_us": statistics.median(values),
        "p90_us": quantile(values, 0.90),
    }


def bootstrap_geomean_ci(ratios: list[float], seed: int) -> dict:
    logs = [math.log(value) for value in ratios]
    rng = random.Random(seed)
    draws = []
    for _ in range(50000):
        sample = [logs[rng.randrange(len(logs))] for _ in logs]
        draws.append(math.exp(statistics.fmean(sample)))
    return {
        "geomean": math.exp(statistics.fmean(logs)),
        "ci95_low": quantile(draws, 0.025),
        "ci95_high": quantile(draws, 0.975),
    }


def audit_compute_processes() -> list[str]:
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip().splitlines()
    foreign = [
        row
        for row in rows
        if "/usr/libexec/gnome-remote-desktop-daemon" not in row
    ]
    if foreign:
        raise RuntimeError("shared GPU is not idle: " + "; ".join(foreign))
    return rows


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
    parser.add_argument("--sparse-binary", type=Path, required=True)
    parser.add_argument("--dense-binary", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--packed-root", type=Path, required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--quality-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--raw-scale",
        type=float,
        help="optional truth-free raw scale shared by every case",
    )
    args = parser.parse_args()
    if args.raw_scale is not None and args.raw_scale <= 0.0:
        raise ValueError("raw-scale must be positive")

    repo = args.repo_root.resolve()
    sparse_binary = args.sparse_binary.resolve()
    dense_binary = args.dense_binary.resolve()
    runtime_root = args.runtime_root.resolve()
    packed_root = args.packed_root.resolve()
    dense_root = args.dense_root.resolve()
    quality_summary = args.quality_summary.resolve()
    output = args.output.resolve()
    quality = json.loads(quality_summary.read_text(encoding="utf-8"))
    if not quality.get("all_application_pass", False):
        raise RuntimeError("dense-control quality gate is not closed")
    if not quality.get("all_eager_graph_bitwise", False):
        raise RuntimeError("dense-control eager/Graph determinism is not closed")
    if quality.get("binary_sha256") != sha256(dense_binary):
        raise RuntimeError("dense-control quality binary does not match timing binary")
    if Path(quality.get("dense_root", "")).resolve() != dense_root:
        raise RuntimeError("dense-control quality pack does not match timing pack")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {output}")
    compute_processes = audit_compute_processes()
    raw = output / "raw_processes"
    raw.mkdir(parents=True, exist_ok=True)

    production_source = (
        repo
        / "experiments/production/src"
    )
    source_paths = [
        production_source / "nufft_spmma_bench.cu",
        production_source / "nufft_dense_control.cuh",
        production_source / "nufft_integrated_bench.cu",
        production_source / "nufft_fp16x2_cg_bench.cu",
    ]

    environment = {
        "experiment_id": "dense_control",
        "started_ns": time.time_ns(),
        "sparse_binary": str(sparse_binary),
        "sparse_binary_sha256": sha256(sparse_binary),
        "dense_binary": str(dense_binary),
        "dense_binary_sha256": sha256(dense_binary),
        "quality_summary": str(quality_summary),
        "quality_summary_sha256": sha256(quality_summary),
        "runtime_root": str(runtime_root),
        "packed_root": str(packed_root),
        "dense_root": str(dense_root),
        "pairs": args.pairs,
        "warmup": args.warmup,
        "iters": args.iters,
        "scale_rule": (
            {
                "kind": "fixed_truth_free",
                "raw_scale": args.raw_scale,
                "native_scale": args.raw_scale / 512.0,
            }
            if args.raw_scale is not None
            else {"kind": "legacy_per_case"}
        ),
        "pre_run_compute_processes": compute_processes,
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "repository_state": "workspace-not-git",
        "nvidia_smi": gpu_state(),
    }

    rows = []
    process_audits = []
    with open("/tmp/trajsparsenufft_gpu.lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for graph in (0, 1):
            mode = "graph" if graph else "eager"
            for pair in range(args.pairs):
                for case_index, (case, legacy_raw_scale) in enumerate(CASES.items()):
                    raw_scale = (
                        args.raw_scale
                        if args.raw_scale is not None
                        else legacy_raw_scale
                    )
                    order = (
                        ("sparse", "dense")
                        if (pair + case_index) % 2 == 0
                        else ("dense", "sparse")
                    )
                    pack = packed_root / f"{case}65"
                    runtime = runtime_root / "frame00" / f"{case}65"
                    control = dense_root / f"{case}65"
                    payloads = {}
                    for variant in order:
                        process_audits.append(
                            {
                                "mode": mode,
                                "pair": pair,
                                "case": case,
                                "variant": variant,
                                "processes": audit_compute_processes(),
                                "gpu_state": gpu_state(),
                            }
                        )
                        binary = sparse_binary if variant == "sparse" else dense_binary
                        command = [
                            str(binary),
                            str(pack / "G/real"),
                            str(pack / "G/imag"),
                            str(pack / "G_T/real"),
                            str(pack / "G_T/imag"),
                            str(runtime),
                            repr(raw_scale / 512.0),
                            str(args.warmup),
                            str(args.iters),
                            "0",
                            str(graph),
                            "10",
                        ]
                        process_env = os.environ.copy()
                        if variant == "dense":
                            process_env.update(
                                JKF_DENSE_FWD_CONTROL=str(control / "g"),
                                JKF_DENSE_ADJ_CONTROL=str(control / "gt"),
                            )
                        process = subprocess.run(
                            command,
                            text=True,
                            capture_output=True,
                            check=False,
                            env=process_env,
                        )
                        log = raw / f"{mode}_pair{pair}_{case}_{variant}.log"
                        log.write_text(
                            "$ "
                            + " ".join(command)
                            + "\n\n[stdout]\n"
                            + process.stdout
                            + "\n[stderr]\n"
                            + process.stderr
                        )
                        if process.returncode != 0:
                            raise RuntimeError(
                                f"{variant} failed at {mode}/{case}/pair{pair}"
                            )
                        payload = parse_timing(process.stdout)
                        if not payload.get("correct", False):
                            raise RuntimeError(f"{variant} correctness failed")
                        payloads[variant] = payload

                    sparse_us = float(payloads["sparse"]["us"])
                    dense_us = float(payloads["dense"]["us"])
                    row = {
                        "mode": mode,
                        "pair": pair,
                        "case": case,
                        "order": list(order),
                        "sparse_us": sparse_us,
                        "dense_us": dense_us,
                        "sparse_speedup": dense_us / sparse_us,
                        "sparse": payloads["sparse"],
                        "dense": payloads["dense"],
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

    (output / "paired_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )

    summaries = []
    for mode_index, mode in enumerate(("eager", "graph")):
        mode_rows = [row for row in rows if row["mode"] == mode]
        case_summaries = []
        for case_index, case in enumerate(CASES):
            selected = [row for row in mode_rows if row["case"] == case]
            ratios = [row["sparse_speedup"] for row in selected]
            interval = bootstrap_geomean_ci(
                ratios, 20260825 + mode_index * 100 + case_index
            )
            sparse_values = [row["sparse_us"] for row in selected]
            dense_values = [row["dense_us"] for row in selected]
            case_summaries.append(
                {
                    "case": case,
                    "sparse": distribution(sparse_values),
                    "dense": distribution(dense_values),
                    "marginal_ratio": statistics.median(dense_values)
                    / statistics.median(sparse_values),
                    "paired_geomean_speedup": interval["geomean"],
                    "paired_ci95_low": interval["ci95_low"],
                    "paired_ci95_high": interval["ci95_high"],
                    "wins": sum(ratio > 1.0 for ratio in ratios),
                }
            )

        aggregate_rows = []
        for pair in range(args.pairs):
            selected = [row for row in mode_rows if row["pair"] == pair]
            sparse_sum = sum(row["sparse_us"] for row in selected)
            dense_sum = sum(row["dense_us"] for row in selected)
            aggregate_rows.append(
                {
                    "pair": pair,
                    "sparse_sum_us": sparse_sum,
                    "dense_sum_us": dense_sum,
                    "sparse_speedup": dense_sum / sparse_sum,
                }
            )
        interval = bootstrap_geomean_ci(
            [row["sparse_speedup"] for row in aggregate_rows],
            20260925 + mode_index,
        )
        summaries.append(
            {
                "mode": mode,
                "case_summaries": case_summaries,
                "aggregate": {
                    "rows": aggregate_rows,
                    "paired_geomean_speedup": interval["geomean"],
                    "paired_ci95_low": interval["ci95_low"],
                    "paired_ci95_high": interval["ci95_high"],
                    "wins": sum(
                        row["sparse_speedup"] > 1.0 for row in aggregate_rows
                    ),
                },
            }
        )

    environment["ended_ns"] = time.time_ns()
    environment["process_audits"] = process_audits
    environment["post_run_compute_processes"] = audit_compute_processes()
    summary = {
        "experiment_id": "dense_control",
        "environment": environment,
        "summaries": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
