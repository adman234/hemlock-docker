# Parity report

## Status

| Layer | Method | Result |
|---|---|---|
| Parameter tables | unit tests encode every row of `parameter-table.md` | 100% of the recovered values (tests/test_params.py, 15 cases pass) |
| Deterministic I/O stages | unit tests vs behaviour read from bytecode (naming, format choice, EXIF 3/6/8, 5120 cap, `/127.5 - 1`, truncating uint8 cast, contain/shrink geometry) | pass (tests/test_imageio.py) |
| Optimizer structure | stand-in encoder and LPIPS; checks step schedule, crop geometry, clamp bound 1.3 x max_change, finiteness, determinism with equal RNG, crop-randomness effect, loss decrease | pass (tests/test_optimize.py) |
| Pipeline and CLI | stand-in models end to end on PNG/JPEG, square and non-square; bounded perturbation; error paths | pass (tests/test_pipeline_cli.py) |
| CPU vs GPU consistency | written, skipped on this machine's test env (CPU torch only) | not run |
| **Numerical parity with the reference app** | not measured | **not measured** (reference not run, assets absent) |
| Reference-vs-reimplementation image metrics | `research/tools/metrics.py` ready | not measured |

Test command (from `nightshade-reimplementation/`, with torch, torchvision, numpy, pillow, einops, pytest installed):

```
PYTHONPATH=src:tests python -m pytest -q
```

Last run: `55 passed, 1 skipped`.

## What the passing tests do and do not establish

- They establish that the reimplementation matches the **recovered specification** (E-0xx) for everything that does not depend on model weights.
- They do not establish that it reproduces the reference's pixels or its poisoning strength. Differences expected even after assets are available:
  1. SD target synthesis is stochastic in the reference (no torch seed), so only distributions are comparable, unless the reference target image is injected.
  2. Substitute models: SD v1.5 weights from a public mirror should match the original `runwayml` files if the md5 sums in E-010 match (to be checked); BLIP, CLIP and LPIPS variants are unknown (U-003..U-005), and the target pool is absent (U-001).
  3. Library versions: the reference pins diffusers 0.14.0, transformers 4.29.0, torch 2.8.0; newer diffusers changes scheduler and VAE details slightly.
  4. fp16 versus fp32 and cuDNN algorithm choice.

Fill the table below once the experiments run.

| Experiment | Metric | Reference | Reimplementation | Delta | Explanation |
|---|---|---|---|---|---|
| (pending) | | | | | |
