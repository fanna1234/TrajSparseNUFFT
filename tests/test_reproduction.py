"""Exercise reproduction routing and fail-closed checks without CUDA or MRI data."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproductionRoutingTest(unittest.TestCase):
    def setUp(self):
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("TRAJTC_", "TRAJSPARSE_"))}

    def run_group(self, *args, env=None, cwd=None):
        return subprocess.run(
            ["bash", str(REPO / "reproduce.sh"), *args], cwd=cwd,
            env=self.env if env is None else env, text=True, capture_output=True,
        )

    def test_help_is_independent_of_readme_headings(self):
        result = self.run_group("help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Usage:", result.stdout)
        self.assertIn("main-performance", result.stdout)

    def test_invalid_group_and_arguments_fail(self):
        for arguments in (("unknown",), ("smoke", "--unknown")):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.run_group(*arguments).returncode, 2)

    def test_missing_data_configuration_fails_before_build(self):
        result = self.run_group("quality")
        self.assertEqual(result.returncode, 2)
        self.assertIn("required environment variable", result.stderr)
        self.assertNotIn("build.sh", result.stdout)

    def test_every_group_has_a_nonmutating_dry_run(self):
        groups = ("smoke", "evidence", "doctor", "prepare-trajectories", "get-data",
                  "prepare-data", "build", "build-baseline", "baseline-quality",
                  "quality", "main-performance", "design-evidence", "hardening", "long-cg")
        with tempfile.TemporaryDirectory() as directory:
            for group in groups:
                with self.subTest(group=group):
                    result = self.run_group(group, "--dry-run", cwd=directory)
                    self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_gpu_interpreter_is_used_by_all_validation_drivers(self):
        env = dict(self.env, TRAJTC_GPU_PYTHON="/a path with spaces/python")
        for group in ("quality", "baseline-quality", "main-performance", "design-evidence", "long-cg"):
            with self.subTest(group=group):
                result = self.run_group(group, "--dry-run", env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("/a\\ path\\ with\\ spaces/python", result.stdout)
                self.assertNotIn("+ python3 ", result.stdout)

    def test_smoke_dry_run_does_not_require_uv(self):
        result = self.run_group("smoke", "--dry-run", env=dict(self.env, PATH="/usr/bin:/bin"))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_environment_is_supported_but_conflicts_fail(self):
        env = dict(self.env, TRAJSPARSE_GPU_PYTHON="/legacy/python")
        self.assertIn("/legacy/python", self.run_group("quality", "--dry-run", env=env).stdout)
        env["TRAJTC_GPU_PYTHON"] = "/different/python"
        result = self.run_group("quality", "--dry-run", env=env)
        self.assertEqual(result.returncode, 2)
        self.assertIn("conflicting settings", result.stderr)

    def test_environment_failure_precedes_data_download(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / "prepared"
            data = Path(directory) / "data"
            env = dict(self.env, TRAJTC_GPU_PYTHON="/usr/bin/false",
                       TRAJTC_PREPARED_ROOT=str(prepared), TRAJTC_DATA_ROOT=str(data))
            result = self.run_group("prepare-data", env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(prepared.exists())
            self.assertFalse(data.exists())

    def test_runtime_manifest_rejects_measurement_rescaling(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame00"
            frame.mkdir()
            record = [{"name": name, "measurement_scale": 512, "entries": []}
                      for name in ("spiral65", "radial65", "golden65")]
            (frame / "manifest.json").write_text(json.dumps(record))
            result = subprocess.run(
                [sys.executable, str(REPO / "tools/build_runtime_manifest.py"),
                 directory, "--expected", "1"], text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be unscaled", result.stderr)
            self.assertFalse((Path(directory) / "manifest.json").exists())


class ResultGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.performance = load_tool("check_reproduced_performance")
        cls.quality = load_tool("quality_contract")

    def source(self, name):
        return json.loads((REPO / "evidence/source" / f"{name}.json").read_text())

    def test_frozen_rows_satisfy_row_level_checks(self):
        self.quality.validate_rows(self.source("development_quality")["rows"])
        self.quality.validate_rows(self.source("heldout_quality")["rows"], ("fs0005", "fs0016"))
        self.quality.validate_rows(self.source("dense_quality")["rows"])

    def test_missing_duplicate_failed_and_nonfinite_rows_are_rejected(self):
        rows = self.source("development_quality")["rows"]
        mutations = [rows[:-1], rows[:-1] + [copy.deepcopy(rows[0])]]
        for key, value in (("application_vs_fp32_pass", False),
                           ("native_vs_fp32_ssim", float("nan")), ("operator_scale", 1024)):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            mutations.append(changed)
        for changed in mutations:
            with self.assertRaises(RuntimeError):
                self.quality.validate_rows(changed)

    def test_frozen_pairs_recompute_the_published_aggregate(self):
        self.performance.validate_pairs(self.source("cufinufft_performance"), "cufinufft")
        for mode in self.source("dense_performance")["summaries"]:
            self.performance.validate_pairs(mode["aggregate"], "dense")

    def test_truncated_duplicate_and_tampered_pairs_are_rejected(self):
        source = self.source("cufinufft_performance")
        variants = []
        changed = copy.deepcopy(source)
        changed["aggregate_rows"].pop()
        variants.append(changed)
        changed = copy.deepcopy(source)
        changed["aggregate_rows"][0]["pair"] = 1
        variants.append(changed)
        changed = copy.deepcopy(source)
        changed["aggregate_rows"][0]["production_sum_us"] = float("nan")
        variants.append(changed)
        changed = copy.deepcopy(source)
        changed["paired_geomean_speedup"] *= 2
        variants.append(changed)
        for changed in variants:
            with self.assertRaises(RuntimeError):
                self.performance.validate_pairs(changed, "cufinufft")

    def test_low_and_nonfinite_speedups_are_never_successes(self):
        for value in (0.1, -1, float("inf"), float("nan")):
            self.assertFalse(self.performance.classify(value, 2.5)[1])

    def test_evidence_can_be_verified_outside_the_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(REPO / "tools/verify_evidence.py"), "--repo-root", str(REPO)],
                cwd=directory, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
