"""Tests for dxss package."""

import importlib
import sys

import pytest


def test_dummy():
    """Dummy test - just try importing dxss."""
    import dxss  # noqa: F401


def test_pardiso():
    """Test for PyPardiso. Print info for `pytest -s` logging."""
    import dxss._solver_backend

    if dxss._solver_backend.pypardiso is not None:
        import pypardiso as _  # noqa: F401

        print("PyPardiso is available.")
    else:
        print("PyPardiso is not available.")


def test_mock_no_pypardiso_for_solver_warns(mocker):
    # mock no pypardiso installed (even on systems where it's installed)
    mocker.patch.dict(sys.modules, {"pypardiso": None})

    # need to reload the modules in case they where cached
    import dxss._solvers  # fmt: skip

    importlib.reload(sys.modules["dxss._solver_backend"])
    importlib.reload(sys.modules["dxss._solvers"])

    # now check that solve_1d gracefully handles missing pypardiso and warns
    with pytest.warns(UserWarning):
        dxss._solvers.PySolver(Asp=None, psolver=None)
