import unittest

import numpy as np

from src.nufft_layout_oracle import (
    apply_packed,
    apply_sparse,
    build_interpolation,
    canonical_operator_checksum,
    generate_trajectory,
    pack_view,
    packed_checksum,
    sparse_checksum,
    transpose_sparse,
    validate_24,
)


class LayoutOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trajectory = generate_trajectory("radial", 24, 96)
        self.forward = build_interpolation(self.trajectory, 24, 4)
        self.adjoint = transpose_sparse(self.forward, 24)

    def test_transpose_preserves_coefficients(self) -> None:
        self.assertEqual(
            canonical_operator_checksum(self.forward, False),
            canonical_operator_checksum(self.adjoint, True),
        )

    def test_forward_pack_is_lossless_and_legal(self) -> None:
        packed = pack_view(self.forward, "overlap", 32, 7)
        self.assertEqual(sparse_checksum(self.forward), packed_checksum(packed))
        self.assertTrue(all(validate_24(tile)[0] for tile in packed.tiles))
        rng = np.random.default_rng(9)
        dense = rng.standard_normal((self.forward.n_cols, 2)) + 1j * rng.standard_normal(
            (self.forward.n_cols, 2)
        )
        np.testing.assert_allclose(
            apply_sparse(self.forward, dense),
            apply_packed(packed, dense),
            rtol=1e-13,
            atol=1e-13,
        )

    def test_dual_view_adjointness(self) -> None:
        packed_forward = pack_view(self.forward, "overlap", 32, 11)
        packed_adjoint = pack_view(self.adjoint, "overlap", 32, 13)
        rng = np.random.default_rng(17)
        x = rng.standard_normal((self.forward.n_cols, 2)) + 1j * rng.standard_normal(
            (self.forward.n_cols, 2)
        )
        y = rng.standard_normal((self.forward.output_size, 2)) + 1j * rng.standard_normal(
            (self.forward.output_size, 2)
        )
        left = np.vdot(apply_packed(packed_forward, x), y)
        right = np.vdot(x, apply_packed(packed_adjoint, y))
        self.assertLessEqual(abs(left - right) / max(abs(left), abs(right)), 1e-12)


if __name__ == "__main__":
    unittest.main()
