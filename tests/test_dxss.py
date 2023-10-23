"""Tests for dxss package."""

import importlib
import sys

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


def test_mock_no_pypardiso_for_solve_1d(mocker):
    # mock no pypardiso installed (even on systems where it's installed)
    mocker.patch.dict(sys.modules, {"pypardiso": None})
    importlib.reload(sys.modules["dxss.solve_1d"])

    # check that trying to import raises an ImportError
    with pytest.raises(ImportError):
        import pypardiso

    # now check that solve_1d gracefully handles missing pypardiso and warns
    from dxss.solve_1d import PySolver, pypardiso  # noqa: F811

    assert pypardiso is None
    with pytest.warns(UserWarning):
        PySolver(Asp=None, psolver=None)


def test_mock_no_pypardiso_for_solve_2d(mocker):
    mocker.patch.dict(sys.modules, {"pypardiso": None})
    with pytest.raises(ImportError):
        import pypardiso
    from dxss.solve_2d import PySolver, pypardiso  # noqa: F811

    assert pypardiso is None
    with pytest.warns(UserWarning):
        PySolver(Asp=None, psolver=None)


def test_mock_no_pypardiso_for_solve_3d(mocker):
    mocker.patch.dict(sys.modules, {"pypardiso": None})
    with pytest.raises(ImportError):
        import pypardiso
    from dxss.solve_3d import PySolver, pypardiso  # noqa: F811

    assert pypardiso is None
    with pytest.warns(UserWarning):
        PySolver(Asp=None, psolver=None)
