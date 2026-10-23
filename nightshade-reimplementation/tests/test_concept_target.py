import numpy as np
from PIL import Image

from nsre import concept, target


def fake_tagger(words):
    nouns = {"cat", "sofa", "image", "photo", "dog"}
    return [(w, "NN" if w in nouns else "DT") for w in words]


def test_extract_nouns_skips_first_three_tokens():
    cap = "a photography of a cat on a sofa"
    # tokens: a photography of a cat on a sofa; drop first 3 -> a cat on a sofa
    assert concept.extract_nouns(cap, fake_tagger) == ["cat", "sofa"]


def test_pick_concept_rules():
    assert concept.pick_concept([]) == "painting"
    assert concept.pick_concept(["image", "photo"]) == "painting"  # blocklist
    assert concept.pick_concept(["cat"]) == "cat"
    scores = lambda prompts: np.array([0.1, 0.9]) if len(prompts) == 2 else None
    assert concept.pick_concept(["cat", "sofa"], scores) == "sofa"


def test_pick_target_rank_window_and_determinism():
    pool = [f"c{i}" for i in range(30)]
    sim = np.arange(30, dtype=float)  # c29 is most similar
    expected = {f"c{29 - r}" for r in range(4, 10)}
    picks = {concept.pick_target("x", pool, sim) for _ in range(5)}
    assert len(picks) == 1 and picks <= expected


def test_pad_and_crop_roundtrip():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8))
    canvas = target.pad_to_canvas(img)
    assert canvas.shape == (512, 512, 3) and canvas[300:].sum() == 0
    back = target.crop_from_canvas(canvas, img.size)
    assert np.array_equal(np.array(back), np.array(img))


def test_generate_target_image_wiring():
    seen = {}

    def fake_pipe(**kw):
        seen.update(kw)
        return Image.new("RGB", (512, 512), (9, 9, 9))

    cur = Image.new("RGB", (1024, 512))
    out = target.generate_target_image(cur, "dog", fake_pipe)
    assert out.size == cur.size
    assert seen["prompt"] == ["a photo of a dog"]
    assert seen["strength"] == 0.8 and seen["guidance_scale"] == 7.5 and seen["num_inference_steps"] == 30
    assert seen["image"].shape == (1, 3, 512, 512)
