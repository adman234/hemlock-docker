import random

import numpy as np
import pytest
import torch
from PIL import Image

from nsre.params import Params


class TinyEncoder(torch.nn.Module):
    """Stand-in for the SD VAE encoder: strided conv, [1,3,512,512] -> [1,4,64,64]."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=8, stride=8, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g) * 0.2)
        self.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


def fake_lpips(a, b):
    """Differentiable perceptual stand-in: mean abs diff per image, shape [N,1,1,1]."""
    return (a - b).abs().mean(dim=(1, 2, 3), keepdim=True)


@pytest.fixture
def encoder():
    return TinyEncoder()


@pytest.fixture
def small_params():
    return Params("DEFAULT", 0.07, 0.1, "1", 1, 3, 1000.0)


def make_image(path, size=(640, 512), mode="RGB", seed=0, fmt=None, smooth=False):
    rng = np.random.default_rng(seed)
    if smooth:  # codec-friendly content so JPEG round-trip error stays small
        yy, xx = np.mgrid[0 : size[1], 0 : size[0]]
        arr = np.stack([xx * 255 // size[0], yy * 255 // size[1], (xx + yy) * 255 // sum(size)], -1).astype(np.uint8)
    else:
        arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr).convert(mode)
    img.save(path, format=fmt)
    return str(path)


@pytest.fixture
def img_factory(tmp_path):
    def _f(name="a.png", **kw):
        return make_image(tmp_path / name, **kw)

    return _f
