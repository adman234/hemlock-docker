import pytest

from nsre.engine import Engine


def make(pool=None, mapping=None):
    return Engine("unused", device="cpu", use_concept_models=False, pool=pool, mapping=mapping)


def test_mapping_wins():
    assert make(pool=["a", "b"], mapping={"cat": "paul"}).pick_target("cat") == "paul"


def test_degraded_pick_is_deterministic_and_avoids_the_concept():
    eng = make(pool=[f"w{i}" for i in range(20)] + ["cat"])
    picks = {eng.pick_target("cat") for _ in range(5)}
    assert len(picks) == 1 and "cat" not in picks
    assert eng.pick_target("dog") in eng.pool


def test_no_pool_no_clip_is_an_error():
    with pytest.raises(RuntimeError):
        make().pick_target("cat")


def test_detect_concept_without_models_falls_back():
    from PIL import Image

    assert make().detect_concept(Image.new("RGB", (8, 8))) == "painting"
