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


def test_unload_releases_state():
    eng = make(pool=["a", "b"])
    eng._loaded = True
    eng._pipe = object()
    eng._clip = object()
    eng.unload()
    assert not eng.loaded and eng._pipe is None and eng._clip is None


def test_description_is_written_to_png_and_jpeg(tmp_path):
    from PIL import Image

    from nsre import imageio

    img = Image.new("RGB", (16, 16), (1, 2, 3))
    png = imageio.save_image(img, "a.png", str(tmp_path), "LOW", description="a guitar player")
    assert Image.open(png).text.get("Description") == "a guitar player"
    jpg = imageio.save_image(img, "b.jpg", str(tmp_path), "LOW", description="a guitar player")
    assert Image.open(jpg).getexif().get(0x010E) == "a guitar player"


def test_shade_end_to_end_with_standins_writes_description(tmp_path, encoder, small_params, img_factory):
    """Whole Engine.shade path with stand-in models: the description may be a function of the concept."""
    import types

    from PIL import Image

    from conftest import fake_lpips

    class FakePipe:
        def __call__(self, **kw):
            return types.SimpleNamespace(images=[Image.new("RGB", (512, 512), (9, 9, 9))])

    eng = Engine("unused", device="cpu", use_concept_models=False, pool=["dog", "car", "bird"])
    eng._loaded, eng._encode, eng._perceptual, eng._pipe = True, encoder, fake_lpips, FakePipe()
    src = img_factory("a.png", size=(640, 512))
    stages = []
    res = eng.shade(src, str(tmp_path / "out"), small_params, tag="cat", description=lambda t: f"a photo of a {t}",
                    on_stage=stages.append, write_json=False)
    assert res.concept == "cat" and res.target in ("dog", "car", "bird")
    assert Image.open(res.output).text.get("Description") == "a photo of a cat"
    assert stages == ["Reading the image", "Making the target image", "Crafting the protection", "Saving"]