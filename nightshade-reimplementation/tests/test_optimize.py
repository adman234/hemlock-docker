import random

import numpy as np
import pytest
import torch

from nsre.optimize import core_optimize, crop_geometry, crop_for_latent, get_mask, step_size
from nsre.params import CLAMP_SLACK, Params
from conftest import fake_lpips


def rand_t(h, w, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, h, w, generator=g) * 2 - 1


def test_step_size_endpoints():
    assert step_size(0.1, 0, 100) == pytest.approx(0.1)
    assert step_size(0.1, 100, 100) == pytest.approx(0.001)
    assert step_size(0.2, 50, 100) == pytest.approx(0.2 - (0.2 - 0.002) / 2)


def test_crop_geometry_cases():
    assert crop_geometry(512, 768) == (256, True, None)
    assert crop_geometry(900, 512) == (388, False, None)
    assert crop_geometry(512, 512) == (0, False, 1)
    with pytest.raises(ValueError):
        crop_geometry(600, 600)


def test_crop_for_latent_shapes_and_axis():
    x = torch.arange(512 * 768, dtype=torch.float32).reshape(1, 1, 512, 768)
    c = crop_for_latent(x, 100, True)
    assert c.shape == (1, 1, 512, 512) and c[0, 0, 0, 0] == 100
    y = torch.arange(900 * 512, dtype=torch.float32).reshape(1, 1, 900, 512)
    d = crop_for_latent(y, 100, False)
    assert d.shape == (1, 1, 512, 512) and d[0, 0, 0, 0] == 100 * 512


def test_get_mask_forced_full_when_fraction_out_of_range():
    s = rand_t(512, 512)
    mask, mt = get_mask(s, s.clone())  # no change -> 0% < 5% -> all ones
    assert mask.min() == 1 and torch.equal(mt, s)
    mask2, _ = get_mask(s, -s)  # nearly everything changes -> >30% -> all ones
    assert mask2.min() == 1


def test_get_mask_partial_region_kept():
    s = torch.zeros(1, 3, 512, 512)
    t = s.clone()
    t[:, :, :128, :] = 1.0  # 25% of pixels differ by 3.0 summed
    mask, mt = get_mask(s, t)
    assert mask[0, 0, :128].min() == 1 and mask[0, 0, 128:].max() == 0
    assert torch.equal(mt, t)


@pytest.mark.parametrize("shape", [(512, 512), (512, 640), (640, 512)])
def test_perturbation_bounded_and_finite(encoder, small_params, shape):
    src, tgt = rand_t(*shape, 1), rand_t(*shape, 2)
    res = core_optimize(src, tgt, small_params, encoder, fake_lpips, rng=random.Random(0))
    bound = small_params.max_change * CLAMP_SLACK
    assert res.modifier.abs().max().item() <= bound + 1e-6
    assert res.adv.min() >= -1 and res.adv.max() <= 1
    assert torch.isfinite(res.adv).all() and all(np.isfinite(res.losses))
    assert res.adv.shape == src.shape


def test_deterministic_given_same_rng(encoder, small_params):
    src, tgt = rand_t(512, 640, 1), rand_t(512, 640, 2)
    a = core_optimize(src, tgt, small_params, encoder, fake_lpips, rng=random.Random(7))
    b = core_optimize(src, tgt, small_params, encoder, fake_lpips, rng=random.Random(7))
    assert torch.equal(a.adv, b.adv)


def test_crop_randomness_changes_result_on_nonsquare(encoder, small_params):
    src, tgt = rand_t(512, 768, 1), rand_t(512, 768, 2)
    a = core_optimize(src, tgt, small_params, encoder, fake_lpips, rng=random.Random(1))
    b = core_optimize(src, tgt, small_params, encoder, fake_lpips, rng=random.Random(2))
    assert not torch.equal(a.adv, b.adv)


def test_og_size_roundtrip_keeps_shape_and_bounds(encoder, small_params):
    src, tgt = rand_t(512, 640, 1), rand_t(512, 640, 2)
    res = core_optimize(src, tgt, small_params, encoder, fake_lpips, rng=random.Random(0), og_size=(1500, 1875))
    assert res.adv.shape == src.shape
    assert res.modifier.abs().max().item() <= small_params.max_change * CLAMP_SLACK + 1e-6


def test_square_image_is_deterministic_without_rng_state(encoder, small_params):
    src, tgt = rand_t(512, 512, 1), rand_t(512, 512, 2)
    a = core_optimize(src, tgt, small_params, encoder, fake_lpips, rng=random.Random(1))
    b = core_optimize(src, tgt, small_params, encoder, fake_lpips, rng=random.Random(2))
    assert torch.equal(a.adv, b.adv)  # r_range == 0: no random crops, eot forced to 1


def test_loss_moves_towards_target(encoder):
    p = Params("HIGH", 0.12, 0.2, "1", 1, 25, 1000.0)
    src, tgt = rand_t(512, 512, 1), rand_t(512, 512, 2)
    res = core_optimize(src, tgt, p, encoder, fake_lpips, rng=random.Random(0))
    assert res.losses[-1] < res.losses[0]


def test_zero_penalty_weight_allows_bigger_lpips(encoder):
    src, tgt = rand_t(512, 512, 1), rand_t(512, 512, 2)
    lo = Params("HIGH", 0.12, 0.2, "1", 1, 10, 0.0)
    hi = Params("HIGH", 0.12, 0.2, "1", 1, 10, 1e6)
    r0 = core_optimize(src, tgt, lo, encoder, fake_lpips, rng=random.Random(0))
    r1 = core_optimize(src, tgt, hi, encoder, fake_lpips, rng=random.Random(0))
    assert r1.lpips_losses[-1] <= r0.lpips_losses[-1] + 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cpu_gpu_consistency(small_params):
    from conftest import TinyEncoder

    src, tgt = rand_t(512, 512, 1), rand_t(512, 512, 2)
    enc = TinyEncoder()
    cpu = core_optimize(src, tgt, small_params, enc, fake_lpips, rng=random.Random(0))
    enc_g = TinyEncoder().cuda()
    gpu = core_optimize(src.cuda(), tgt.cuda(), small_params, enc_g, fake_lpips, rng=random.Random(0))
    assert (cpu.adv - gpu.adv.cpu()).abs().max().item() < 1e-3
