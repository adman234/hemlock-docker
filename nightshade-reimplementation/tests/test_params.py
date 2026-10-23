import pytest

from nsre.params import (
    OPT_SETTINGS,
    PROTECTION_LEVELS,
    compute_level_to_opt_setting,
    get_params,
    intensity_to_level,
)


@pytest.mark.parametrize(
    "level,max_change,lr",
    [("LOW", 0.035, 0.03), ("DEFAULT", 0.07, 0.1), ("HIGH", 0.12, 0.2)],
)
def test_level_table(level, max_change, lr):
    p = get_params(level, "1")
    assert (p.max_change, p.lr_initial) == (max_change, lr)


@pytest.mark.parametrize(
    "opt,eot,steps,pen",
    [("1", 1, 100, 1000), ("2", 2, 150, 500), ("3", 3, 300, 500), ("4", 6, 500, 800)],
)
def test_render_quality_table(opt, eot, steps, pen):
    p = get_params("DEFAULT", opt)
    assert (p.eot, p.tot_steps, p.penalty_initial) == (eot, steps, pen)


def test_high_adds_200_steps():
    assert get_params("HIGH", "4").tot_steps == 700
    assert get_params("LOW", "4").tot_steps == 500


def test_smoke_mode_defaults_eot():
    p = get_params("LOW", "0")
    assert p.tot_steps == 4 and p.eot == 1


def test_gui_slider_mapping():
    assert [compute_level_to_opt_setting(i) for i in (-1, 0, 1, 2, 3)] == ["0", "1", "2", "3", "4"]
    assert [intensity_to_level(i) for i in (0, 1, 2)] == ["LOW", "DEFAULT", "HIGH"]


@pytest.mark.parametrize("bad", [4, -2, 1.5])
def test_invalid_compute_level(bad):
    with pytest.raises(ValueError):
        compute_level_to_opt_setting(bad)


def test_invalid_names():
    with pytest.raises(ValueError):
        get_params("EXTREME", "1")
    with pytest.raises(ValueError):
        get_params("LOW", "9")
    with pytest.raises(ValueError):
        intensity_to_level(3)


def test_tables_complete():
    assert set(PROTECTION_LEVELS) == {"LOW", "DEFAULT", "HIGH"}
    assert set(OPT_SETTINGS) == {"0", "1", "2", "3", "4"}
