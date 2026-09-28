"""Protect configuration aliases, the shared lock identity, and frozen metadata."""

import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
PREFIXES = ("TRAJTC_", "TRAJSPARSE_")
SUFFIXES = (
    "GPU_PYTHON", "DATA_ROOT", "PREPARED_ROOT", "RESULTS_ROOT",
    "DEVELOPMENT_CASE", "DEVELOPMENT_RUNTIME_ROOT", "HELDOUT_CASES_ROOT",
    "HELDOUT_RUNTIME_ROOT", "PERFORMANCE_RUNTIME_ROOT", "PACKED_ROOT",
    "DENSE_ROOT", "CUFINUFFT_QUALITY_SUMMARY", "ALLOW_VERSION_DRIFT",
)


class RenameCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(PREFIXES)}
        self.values = {suffix: f"/paths with spaces/{suffix}" for suffix in SUFFIXES}
        self.values["ALLOW_VERSION_DRIFT"] = "1"

    def settings(self, prefix, values=None):
        values = self.values if values is None else values
        return {prefix + suffix: value for suffix, value in values.items()}

    def resolve(self, settings):
        return subprocess.run(
            ["bash", "-c", 'set -euo pipefail; source "$1"; exec "$2" -c "$3"',
             "_", str(REPO / "tools/runtime_env.sh"), sys.executable,
             "import json, os; print(json.dumps({k: v for k, v in os.environ.items() "
             "if k.startswith(('TRAJTC_', 'TRAJSPARSE_'))}))"],
            env=dict(self.env, **settings), text=True, capture_output=True,
        )

    def test_all_aliases_export_the_same_values_to_children(self):
        current = self.settings(PREFIXES[0])
        legacy = self.settings(PREFIXES[1])
        empty = {suffix: "" for suffix in SUFFIXES}
        both = dict(current, **legacy)
        cases = (
            ({}, {}),
            (current, both),
            (legacy, both),
            (both, both),
            (dict(current, **self.settings(PREFIXES[1], empty)), both),
            (dict(legacy, **self.settings(PREFIXES[0], empty)), both),
        )
        for settings, expected in cases:
            with self.subTest(settings=settings):
                result = self.resolve(settings)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), expected)

    def test_each_conflicting_alias_fails_before_child_execution(self):
        for suffix in SUFFIXES:
            with self.subTest(suffix=suffix):
                result = self.resolve({f"TRAJTC_{suffix}": "current",
                                       f"TRAJSPARSE_{suffix}": "legacy"})
                self.assertEqual(result.returncode, 2)
                self.assertIn(f"conflicting settings: TRAJTC_{suffix} and TRAJSPARSE_{suffix}",
                              result.stderr)
                self.assertEqual(result.stdout, "")

    def test_legacy_and_current_configuration_route_identical_commands(self):
        for group in ("doctor", "quality", "baseline-quality", "main-performance",
                      "design-evidence", "hardening", "long-cg"):
            with self.subTest(group=group):
                results = [subprocess.run(
                    ["bash", str(REPO / "reproduce.sh"), group, "--dry-run"],
                    env=dict(self.env, **self.settings(prefix)), text=True, capture_output=True,
                ) for prefix in PREFIXES]
                for result in results:
                    self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(results[0].stdout, results[1].stdout)
                self.assertEqual(results[0].stderr, results[1].stderr)

    def test_gpu_entrypoints_keep_the_shared_legacy_lock_path(self):
        # Pin the cross-version coordination name without acquiring a real GPU lock.
        expected = "/tmp/trajsparsenufft_gpu.lock"
        drivers = (
            "production/run_cg_quality_matrix.py", "quality/run_heldout_quality.py",
            "cufinufft_baseline/run_quality_matrix.py", "cufinufft_baseline/run_paired_fullcg.py",
            "dense_control/run_dense_quality_matrix.py", "dense_control/run_paired_dense_fullcg.py",
            "frozen_evidence/run_long_cg.py",
        )
        for driver in drivers:
            with self.subTest(driver=driver):
                tree = ast.parse((REPO / "experiments" / driver).read_text())
                lock_paths = [node.args[0].value for node in ast.walk(tree)
                              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                              and node.func.id == "open" and node.args
                              and isinstance(node.args[0], ast.Constant)
                              and isinstance(node.args[0].value, str)
                              and node.args[0].value.endswith(".lock")]
                self.assertEqual(lock_paths, [expected])
        hardening = (REPO / "experiments/frozen_evidence/run_hardening.sh").read_text()
        self.assertEqual(re.findall(r"^\s*flock\s+(\S+)", hardening, re.M), [expected])

    def test_frozen_evidence_rejects_identifier_only_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(REPO / "evidence", root / "evidence")
            source = root / "evidence/source/development_quality.json"
            original = source.read_text()
            for old, new in (
                ('"experiment_id": "production"', '"experiment_id": "trajtc_production"'),
                ("repo://experiments/production/build/nufft_fp16x2_cg",
                 "repo://experiments/production/build/trajtc_cg"),
            ):
                with self.subTest(identifier=old):
                    self.assertIn(old, original)
                    source.write_text(original.replace(old, new, 1))
                    result = subprocess.run(
                        [sys.executable, str(REPO / "tools/verify_evidence.py"),
                         "--repo-root", str(root)], text=True, capture_output=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("frozen result record or source hash changed", result.stderr)


if __name__ == "__main__":
    unittest.main()
