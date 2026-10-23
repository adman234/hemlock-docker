import os

import numpy as np
import pytest
import torch
from PIL import Image

from nsre import imageio


def test_contain_image_longer_side_is_512():
    assert imageio.contain_image(Image.new("RGB", (1000, 500))).size == (512, 256)
    assert imageio.contain_image(Image.new("RGB", (500, 1000))).size == (256, 512)
    assert imageio.contain_image(Image.new("RGB", (300, 300))).size == (512, 512)


def test_shrink_shortside_is_512():
    assert imageio.shrink_shortside(Image.new("RGB", (1000, 500))).size == (1024, 512)
    assert imageio.shrink_shortside(Image.new("RGB", (400, 800))).size == (512, 1024)


def test_img2tensor_range_and_shape():
    img = Image.fromarray(np.array([[[0, 127, 255]]], dtype=np.uint8))
    t = imageio.img2tensor(img)
    assert t.shape == (1, 3, 1, 1) and t.dtype == torch.float32
    assert t[0, 0, 0, 0].item() == pytest.approx(-1.0)
    assert t[0, 2, 0, 0].item() == pytest.approx(255 / 127.5 - 1)


def test_tensor2img_truncates_not_rounds():
    t = torch.tensor([[[[0.0]], [[0.0]], [[0.0]]]])  # mid-grey: (0+1)/2*255 = 127.5
    assert np.array(imageio.tensor2img(t)).ravel().tolist() == [127, 127, 127]


def test_roundtrip_within_one_level():
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    back = np.array(imageio.tensor2img(imageio.img2tensor(Image.fromarray(arr)))).astype(int)
    assert np.abs(back - arr.astype(int)).max() <= 1


def test_reduce_quality_caps_long_side():
    img = Image.new("RGB", (6000, 3000))
    assert imageio.reduce_quality(img).size == (5120, 2560)
    small = Image.new("RGB", (100, 80))
    assert imageio.reduce_quality(small).size == (100, 80)


def test_load_img_invalid_inputs(tmp_path):
    assert imageio.load_img(str(tmp_path / "missing.png")) is None
    assert imageio.load_img(str(tmp_path)) is None
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert imageio.load_img(str(bad)) is None
    assert not imageio.check_is_img(str(bad))


def test_load_img_rgba_and_palette_become_rgb(tmp_path):
    for mode in ("RGBA", "L", "P"):
        p = tmp_path / f"{mode}.png"
        Image.new(mode, (40, 30)).save(p)
        assert imageio.load_img(str(p)).mode == "RGB"


def test_load_img_exif_orientation_6_rotates(tmp_path):
    img = Image.new("RGB", (60, 30), (255, 0, 0))
    ex = Image.Exif()
    ex[0x0112] = 6
    p = tmp_path / "o6.jpg"
    img.save(p, exif=ex)
    assert imageio.load_img(str(p)).size == (30, 60)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("cat.png", "cat-protected-intensity-HIGH-V1.png"),
        ("my.photo.jpg", "my.photo-protected-intensity-HIGH-V1.jpg"),
        ("noext", "noext-protected-intensity-HIGH-V1"),
    ],
)
def test_output_name(name, expected):
    assert imageio.output_name("/x/" + name, "HIGH") == expected


def test_save_image_format_by_extension(tmp_path):
    img = Image.new("RGB", (16, 16), (10, 20, 30))
    png = imageio.save_image(img, "a.png", str(tmp_path), "LOW")
    jpg = imageio.save_image(img, "b.jpeg", str(tmp_path), "LOW")
    upper = imageio.save_image(img, "c.PNG", str(tmp_path), "LOW")
    assert Image.open(png).format == "PNG"
    assert Image.open(jpg).format == "JPEG"
    assert Image.open(upper).format == "JPEG"  # case-sensitive check, as in the original
    assert os.path.basename(png) == "a-protected-intensity-LOW-V1.png"


def test_add_perturbation_preserves_size_and_clips():
    full = Image.new("RGB", (800, 600), (250, 5, 128))
    src = imageio.img2tensor(Image.new("RGB", (684, 512), (250, 5, 128)))
    adv = torch.clamp(src + 0.5, -1, 1)
    out = imageio.add_perturbation_full_size(full, src, adv)
    assert out.size == (800, 600)
    arr = np.array(out)
    assert arr.dtype == np.uint8 and arr[..., 0].max() == 255  # clipped at the top
