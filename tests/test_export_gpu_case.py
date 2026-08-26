import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from src.export_gpu_case import MAGIC, export_case


class ExportGpuCaseTest(unittest.TestCase):
    def test_small_export_has_consistent_header_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with redirect_stdout(StringIO()):
                export_case(
                    Namespace(
                        trajectory="spiral",
                        trajectory_file=None,
                        periodic=False,
                        view="G_T",
                        grid=24,
                        width=4,
                        samples=192,
                        policy="overlap",
                        illegal_mode="csr_residual",
                        seed=23,
                        output_dir=output,
                    )
                )
            header = np.fromfile(output / "header.bin", dtype=np.uint64, count=1)
            self.assertEqual(int(header[0]), MAGIC)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertGreater((output / "tile_a_comp.bin").stat().st_size, 0)
            self.assertGreater((output / "reference.bin").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
