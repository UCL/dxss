"""Tests for dxss package."""

import pytest


def test_dummy():
    """Dummy test - just try importing dxss."""
    import dxss  # noqa: F401


def test_pardiso():
    """Test for PyPardiso. Print info for `pytest -s` logging."""
    from dxss.solve_1d import PySolver, pypardiso

    if pypardiso is not None:
        import pypardiso as _  # noqa: F401

        print("PyPardiso is available.")
    else:
        print("PyPardiso is not available.")
        with pytest.warns(UserWarning):
            PySolver(Asp=None, psolver=None)
