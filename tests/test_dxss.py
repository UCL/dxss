"""Tests for dxss package."""

import importlib
import sys

import pytest


def test_dummy():
    """Dummy test - just try importing dxss."""
    import dxss  # noqa: F401


def test_pardiso():
    """Test for PyPardiso. Print info for `pytest -s` logging."""
    from dxss.solve import PySolver, pypardiso

    if pypardiso is not None:
        import pypardiso as _  # noqa: F401

        print("PyPardiso is available.")
    else:
        print("PyPardiso is not available.")
        with pytest.warns(UserWarning):
            PySolver(Asp=None, psolver=None)


def test_mock_no_pypardiso_for_solve_1d(mocker):
    # mock no pypardiso installed (even on systems where it's installed)
    mocker.patch.dict(sys.modules, {"pypardiso": None})
    if "dxss.solve" in sys.modules:
        importlib.reload(sys.modules["dxss.solve"])

    # check that trying to import raises an ImportError
    with pytest.raises(ImportError):
        import pypardiso

    # now check that solve_1d gracefully handles missing pypardiso and warns
    from dxss.solve import PySolver, pypardiso  # noqa: F811

    assert pypardiso is None
    with pytest.warns(UserWarning):
        PySolver(Asp=None, psolver=None)
