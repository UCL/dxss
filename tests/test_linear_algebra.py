"""Tests for linear algebra routines in the dxss package."""

import pytest

from dxss._utils import givens_rotation


class TestLinearAlgebraRoutines:
    def test_givens_rotation_zero_v1_zero_v2(self):
        v1 = 0.0
        v2 = 0.0
        with pytest.raises(
            ValueError,
            match="v1 & v2 cannot both be zero for Givens rotation",
        ):
            givens_rotation(v1, v2)

    def test_givens_rotation_zero_v1_negative_v2(self):
        v1 = 0.0
        v2 = -18.0
        cs, sn = givens_rotation(v1, v2)
        assert cs == 0
        assert sn == -1

    def test_givens_rotation_zero_v1_positive_v2(self):
        v1 = 0.0
        v2 = 53.0
        cs, sn = givens_rotation(v1, v2)
        assert cs == 0
        assert sn == 1

    def test_givens_rotation_zero_v2(self):
        v1 = 5.0
        v2 = 0.0
        cs, sn = givens_rotation(v1, v2)
        assert cs == 1
        assert sn == 0

    def test_givens_rotation_nonzero_v1_and_v2(self):
        v1 = 6.0
        v2 = 5.0
        cs, sn = givens_rotation(v1, v2)
        assert cs == pytest.approx(0.7682, rel=1e-4)
        assert sn == pytest.approx(0.6402, rel=1e-4)
