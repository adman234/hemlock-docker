# Experiment protocol (Phase 3)

Status: **designed, not executed.** The reference app needs several GB of assets that have not been downloaded (E-007) and
that load unpickled objects (E-012). Execution waits for approval (see analysis-plan.md). `experiments.csv` lists the planned runs;
`reference-observations.md` is the place for measured results.

## Setup

- Reference: unmodified `Nightshade.exe` 1.1, run from a copy of the install in a throwaway Windows user profile or Windows Sandbox. `%USERPROFILE%\.nightshade` is the only writable location. CLI mode (`python -m nightshade`-style invocation is not available in a frozen build), so runs use the GUI through `capture_native_ui_scenario`-style automation, or a source-equivalent harness that imports the extracted `nightshade` module with the same bundled libraries.
- Reimplementation: `nsre` with the public substitutes in `models.py`.
- Corpus: images the owner is authorized to use. Fixtures are generated deterministically by `tests/conftest.py::make_image` (seeded noise and gradients) plus 3 to 5 owner-supplied photographs. No third-party artwork.
- Record for every run: input SHA-256, app version, intensity, rq, target/concept override, device, GPU name and driver, torch version, wall time, peak VRAM (`nvidia-smi` sampling), output SHA-256, output size/format/mode, EXIF presence, stderr.
- Hardware available: RTX 4070 SUPER 12 GB (> 10000 MB, so the original takes the `extra_large_gpu` path), driver 616.56.

## One-factor-at-a-time matrix

| ID | Factor varied | Levels | Others fixed |
|---|---|---|---|
| X-DIM | size / aspect | 512x512, 768x512, 512x768, 1024x1024, 4000x3000, 5200x300 (cap and degenerate) | DEFAULT, rq 0, PNG, forced target |
| X-FMT | format and mode | PNG RGB, PNG RGBA, PNG L, PNG P, JPEG, JPEG with EXIF orientation 6, GIF, `.PNG` upper-case | 768x512, DEFAULT, rq 0 |
| X-INT | intensity | LOW, DEFAULT, HIGH | 768x512, rq 0 |
| X-RQ | render quality | 0, 1, 2, 3 (time boxed; 3 only for 512x512) | 512x512, DEFAULT |
| X-TGT | target | 3 forced targets, 1 auto from pool | 768x512 |
| X-DEV | device | CUDA fp16 vs CPU fp32 (CPU only for rq 0 and 512x512) | DEFAULT |
| X-REP | repeats | 5 identical runs | 768x512, DEFAULT, rq 0, forced target |
| X-BAD | invalid input | missing path, directory, zero-byte file, truncated PNG, text file named `.png`, output dir missing | n/a |
| X-SKIP | filename filter | input named `a-nightshade-intensity.png` vs `a-protected-intensity-LOW-V1.png` | n/a (tests E-024) |

## Metrics (computed by `research/tools/metrics.py` once outputs exist)

- Output size, format, mode, EXIF equality, filename.
- Per-pixel difference vs input: max abs error, mean abs error, fraction of pixels changed, per-channel stats, PSNR, SSIM, LPIPS (alex).
- Perturbation spectrum summary (low vs high frequency energy) to compare reference and reimplementation texture.
- Run-to-run: pairwise difference between repeats; separate deterministic stages (resize, EXIF, naming, truncating cast) from stochastic ones (SD sampling, crops).
- VAE-latent distance to the target (using the SD VAE) as a proxy for attack strength.

## Parity method

1. Stage parity: feed identical inputs to deterministic stages (load, orientation, resize, tensor conversion, lift-to-full-size, naming, save) and require exact equality.
2. Optimizer parity: inject the **same target image** (captured from the reference or a fixed one) and the same crop offsets (`random.Random(678121)` stream), compare `δ`.
3. End-to-end: compare distributions (mean/max error, PSNR, latent distance) over N runs since SD sampling is unseeded in the reference.

## Safety

- Reference runs are offline after the assets are staged; network allow-list only for the staging step.
- Pickled assets are first inspected with `pickletools.dis` (no execution) before the app ever loads them.
