import pytest

from dxss.space_time import ProblemParameters, SpaceTime


def test_defaults():
    p = ProblemParameters()
    assert p.jumps_in_fw_problem is False
    assert p.well_posed is False
    assert p.data_domain_fitted is True


def test_setting_data_domain():
    p = ProblemParameters(data_domain_fitted=False)
    assert p.jumps_in_fw_problem is False
    assert p.well_posed is False
    assert p.data_domain_fitted is False


def test_setting_well_posed():
    p = ProblemParameters(well_posed=True)
    assert p.jumps_in_fw_problem is False
    assert p.well_posed is True
    assert p.data_domain_fitted is True


def test_setting_jumps_in_fw_problem():
    p = ProblemParameters(jumps_in_fw_problem=True)
    assert p.jumps_in_fw_problem is True
    assert p.well_posed is False
    assert p.data_domain_fitted is True


@pytest.mark.skip(
    "Skip for now, enable when SpaceTime refactored and easier to mock up.",
)
def test_spacetime_defaults():
    s = SpaceTime()
    assert s.jumps_in_fw_problem is False
    assert s.well_posed is False
