import json
import os

import numpy as np
import pytest
from PIL import Image

from nsre import cli
from nsre.pipeline import Components, shade_image
from conftest import fake_lpips


def fake_target(small, concept):
    return Image.fromarray(255 - np.array(small))  # strongly different target


@pytest.mark.parametrize("name,fmt,size", [("a.png", "PNG", (640, 512)), ("b.jpg", "JPEG", (700, 900))])
def test_end_to_end_with_stand_ins(tmp_path, encoder, small_params, img_factory, name, fmt, size):
    src = img_factory(name, size=size, smooth=(fmt == "JPEG"))
    out_dir = tmp_path / "out"
    comps = Components(encoder, fake_lpips, fake_target)
    out = shade_image(src, str(out_dir), small_params, comps, "dog", seed=3)
    assert os.path.basename(out).startswith(name.split(".")[0] + "-protected-intensity-DEFAULT-V1")
    img = Image.open(out)
    assert img.size == size and img.format == fmt
    diff = np.abs(np.array(img).astype(int) - np.array(Image.open(src).convert("RGB")).astype(int))
    assert diff.max() > 0  # something changed
    # perturbation is bounded: 1.3*0.07*127.5 ~ 11.6 levels, plus resampling and jpeg slack
    slack = 40 if fmt == "JPEG" else 25
    assert diff.max() <= slack
    meta = json.load(open(out + ".json"))
    assert meta["input_sha256"] and meta["params"]["max_change"] == 0.07 and meta["target_concept"] == "dog"


def test_square_input_runs(tmp_path, encoder, small_params, img_factory):
    src = img_factory("sq.png", size=(300, 300))
    out = shade_image(src, str(tmp_path), small_params, Components(encoder, fake_lpips, fake_target), "x")
    assert Image.open(out).size == (300, 300)


def test_invalid_input_raises(tmp_path, encoder, small_params):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError):
        shade_image(str(bad), str(tmp_path), small_params, Components(encoder, fake_lpips, fake_target), "x")


def test_cli_dry_run(tmp_path, img_factory, capsys):
    img_factory("a.png")
    rc = cli.main(["-d", str(tmp_path), "-m", "2", "-rq", "3", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "'max_change': 0.12" in out and "'tot_steps': 700" in out and "1 image(s)" in out


def test_cli_rejects_bad_levels():
    with pytest.raises(SystemExit):
        cli.main(["-m", "3", "--dry-run"])
    with pytest.raises(SystemExit):
        cli.main(["-rq", "4", "--dry-run"])


def test_cli_missing_output_dir(tmp_path, img_factory):
    img_factory("a.png")
    assert cli.main(["-d", str(tmp_path), "-od", str(tmp_path / "nope")]) == 2


def test_cli_skips_own_outputs_and_non_images(tmp_path, img_factory, capsys):
    img_factory("a.png")
    img_factory("a-protected-intensity-LOW-V1.png")
    (tmp_path / "notes.txt").write_text("x")
    cli.main(["-d", str(tmp_path), "--dry-run"])
    assert "1 image(s)" in capsys.readouterr().out
