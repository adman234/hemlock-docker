"""Parameter tables.

Source: utils.get_app_parameters and nightshade.Nightshade.update_params in the
Nightshade 1.1 Windows build (see research/parameter-table.md, evidence E-020..E-024).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

# protection level -> (max_change, lr_initial); GUI "Intensity" slider 0..2
PROTECTION_LEVELS = {
    "LOW": (0.035, 0.03),
    "DEFAULT": (0.07, 0.1),
    "HIGH": (0.12, 0.2),
}
INTENSITY_INDEX_TO_LEVEL = {0: "LOW", 1: "DEFAULT", 2: "HIGH"}

# opt_setting -> (eot, tot_steps, penalty_initial); GUI "Render quality" slider 0..3
# maps to opt_setting "1".."4" via update_params (compute_level + 1).
# "0" is a reference-side smoke-test mode (compute_level == -1): 4 steps, no eot key.
OPT_SETTINGS = {
    "0": (None, 4, 500),
    "1": (1, 100, 1000),
    "2": (2, 150, 500),
    "3": (3, 300, 500),
    "4": (6, 500, 800),
}
HIGH_EXTRA_STEPS = 200  # added to tot_steps for every opt_setting when the level is HIGH

STYLE_TRANSFER_ITER = 30   # SD img2img num_inference_steps
STYLE_STRENGTH = 0.8       # SD img2img strength
GUIDANCE_SCALE = 7.5
LR_FINAL_FRACTION = 0.01   # lr decays linearly from lr_initial to lr_initial/100
CLAMP_SLACK = 1.3          # modifier clamp is +-1.3 * max_change
PROCESS_SIZE = 512
MAX_RESOLUTION = 5120
MASK_DIFF_THRESHOLD = 0.3
MASK_FRAC_HIGH = 0.3
MASK_FRAC_LOW = 0.05
CONCEPT_SEED = 678121
FALLBACK_CONCEPT = "painting"
CONCEPT_BLOCKLIST = ("image", "photo", "white", "new", "black", "design", "painting")
BLIP_PREFIX = "a photography of"
TARGET_RANK_SLICE = (4, 10)  # ranks 4..9 of descending similarity, then random.choice


@dataclass(frozen=True)
class Params:
    protection_level: str
    max_change: float
    lr_initial: float
    opt_setting: str
    eot: int
    tot_steps: int
    penalty_initial: float
    style_transfer_iter: int = STYLE_TRANSFER_ITER
    lpips: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def get_params(protection_level: str, opt_setting: str) -> Params:
    """Equivalent of utils.get_app_parameters(protection_level, opt_setting)."""
    if protection_level not in PROTECTION_LEVELS:
        raise ValueError(f"unknown protection level {protection_level!r}")
    opt_setting = str(opt_setting)
    if opt_setting not in OPT_SETTINGS:
        raise ValueError(f"unknown opt_setting {opt_setting!r}")
    max_change, lr_initial = PROTECTION_LEVELS[protection_level]
    eot, steps, penalty = OPT_SETTINGS[opt_setting]
    if protection_level == "HIGH":
        steps += HIGH_EXTRA_STEPS
    if eot is None:
        # The original has no 'eot' key in this mode and would raise KeyError in
        # core_optimize. We default to 1 (deliberate deviation, documented).
        eot = 1
    return Params(protection_level, max_change, lr_initial, opt_setting, eot, steps, float(penalty))


def compute_level_to_opt_setting(compute_level: int) -> str:
    """update_params: -1 -> '0', 0 -> '1', 1 -> '2', 2 -> '3', 3 -> '4'."""
    if compute_level not in (-1, 0, 1, 2, 3):
        raise ValueError(f"Unknown render level of {compute_level:.2f}")
    return str(compute_level + 1)


def intensity_to_level(p_level: int) -> str:
    if p_level not in INTENSITY_INDEX_TO_LEVEL:
        raise ValueError(f"incorrect protection level: {p_level}")
    return INTENSITY_INDEX_TO_LEVEL[p_level]
