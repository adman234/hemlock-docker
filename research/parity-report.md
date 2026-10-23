# Parity report

Reference: the original Nightshade 1.1 `nightshade`/`opt`/`utils` modules unpacked from the shipped build and run unmodified
in a Python 3.11 environment pinned to the app's library versions (torch 2.8.0+cu128, diffusers 0.14.0, transformers 4.29.0,
Pillow 9.3.0, numpy 1.24.1), with the application's own downloaded assets (all ten MD5s verified).
Reimplementation: `nightshade-reimplementation` (`nsre`), same assets (SD v1.5 weights from the app's `base/base`, and for
LPIPS the app's `lpips_fn.p`, loaded locally for parity only).
Hardware: RTX 4070 SUPER 12 GB. Raw data: `research/results/`, tables: `research/measured-results.md`, `experiments.csv`.

## Verdict

| Layer | Method | Result |
|---|---|---|
| Parameter tables | all 15 (level x render setting) combinations, original vs nsre | **15/15 identical** |
| Deterministic stages | `load_img`, `img2tensor`, `tensor2img`, `contain_image`, shrink, `add_perturbation_full_size`, `get_mask`, `save_image` naming and format, on 17 corpus files | **114 identical, plus 1 documented deviation (115 checks)**: nsre accepts GIF, the original crashes |
| Optimizer core | original `core_optimize` vs nsre, identical source, target tensor, models and RNG state | **bit-identical (cosine 1.0, max diff 0.0) in 4 of 5 cases** (square, wide, tall, eot 1/2/3, LOW/DEFAULT). The 5th case (2000x1500, HIGH, 300 steps) has cosine 0.642 between the two, and 0.6417 between two runs of the original: that is CUDA nondeterminism in the resize backward pass, not an implementation difference |
| Concept and target selection | original `ImageAnalyzer` vs nsre rules on the same caption, CLIP scores and embeddings | **13/13 concepts, 20/20 similarity-ranked targets, 60/60 mapped targets** |
| End to end (statistics) | 22 matched runs (sizes, formats, intensities, render qualities, targets), same input and settings | mean absolute error **2.74 (nsre) vs 2.72 (original)**, PSNR **36.96 vs 37.01 dB**, mean max error 14.27 vs 14.23, output names and formats identical in all 22 |
| Run-to-run variability | 5 identical unseeded runs each | MAE sd **0.051 (nsre) vs 0.077 (original)**; perturbation cosine between runs 0.090 vs 0.096; cosine between nsre and original runs about 0.09 (same as original vs original) |
| CPU vs GPU | nsre CPU fp32 vs GPU fp16 (seeded), original CPU vs GPU | statistically equal (MAE 2.68 vs 2.73 nsre, 2.60 original CPU); pixel level cosine 0.108, as expected for a stochastic target |
| Speed and memory | same GPU | nsre 25.7 s mean vs 33.2 s for the original, peak VRAM 4.3 GB vs 7.5 GB (cause not isolated; hypothesis: the original keeps more models resident on cards above 10 GB) |

So: the reimplementation reproduces every deterministic behaviour exactly and the optimizer bit-for-bit, and its
output statistics are indistinguishable from the original's run-to-run spread. It does **not** reproduce the original's
pixels run for run, and neither does the original reproduce itself: the Stable Diffusion target is sampled without a
seed, so two runs of the original differ by about 3.3 grey levels on average. Use `--seed` (nsre) to get repeatability.

## What is still not verified

- Poisoning effectiveness against any model (out of scope; no training or evaluation was done).
- The default web/CLI path uses the Hugging Face CLIP and BLIP checkpoints. Their architectures match the originals
  (E-092, E-093) but the weights were not compared, and the automatic tag-detection path has been run only through the
  selection-rule parity test (with the original's own BLIP/CLIP objects), not with the Hugging Face copies.
- Library versions: the parity numbers above were measured on the original's own pins (diffusers 0.14.0, transformers 4.29.0, Pillow 9.3.0). The container uses diffusers 0.25.1 / transformers 4.36.2 / Pillow 10.4 (the same as mist-docker) because 0.14 cannot read the current Stable Diffusion VAE file. Under the new pins, with the container's exact files, 5 end-to-end cases give the same statistics (MAE 2.65/2.46/1.16/5.01/3.47 vs 2.72/2.39/1.08/5.24/3.31), deterministic image stages are byte-identical across Pillow 9.3 and 10.4, but fixed-input optimizer outputs are **not** bit-identical across the two diffusers versions (VAE latent differs by up to 0.008 in fp16, which the 100-step optimisation amplifies: cosine 0.59 and 0.42 for the two cases). See E-103 to E-105.
- The container image itself is built and smoke-tested by GitHub Actions, not on this machine.

## Differences and explanations

| Difference | Cause |
|---|---|
| Per-pixel results differ from run to run (both implementations) | SD img2img sampling noise (torch never seeded in the original) and random crop offsets |
| 2000x1500 HIGH optimizer case not bit-identical | bilinear-resize backward on a large image uses atomic adds; the original also differs from itself |
| GIF input | the original raises `AttributeError`; nsre accepts it (documented deviation) |
| Output EXIF orientation | original keeps the tag on already-rotated pixels; nsre keeps parity in the library, the web app resets it to 1 |
| Peak VRAM and speed | not isolated; hypothesis: the original holds more model copies resident (it skips unloading on cards above 10 GB) |
| Targets with the web app's default word list | the original pool and mapping are proprietary; the web app ships its own list, so chosen targets differ for the same tag |

## Reproduce

```
research/tools/stage_parity.py      deterministic stages
research/tools/optimizer_parity.py  optimizer core on GPU
research/tools/concept_parity.py    selection rules
research/tools/run_matrix.ps1, run_re_matrix.ps1, run_dev.ps1, analyze.py   end-to-end matrix and tables
```

Unit tests (no GPU, no weights): `PYTHONPATH=src:tests python -m pytest -q` in `nightshade-reimplementation/` gives
`63 passed, 1 skipped` (the skipped test compares CPU and GPU and needs CUDA).
