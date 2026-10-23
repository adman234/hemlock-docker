# Hemlock (hemlock-docker)

A self-hosted web app that protects artwork from AI scraping with an independent, open-source take on the
[Nightshade](https://nightshade.cs.uchicago.edu/) method, built from a reverse-engineering study of the Nightshade 1.1
Windows release and verified against it.

- Upload one image or a whole batch from a phone or a computer, wait a little per image, download the protected images (or a zip)
- Press and hold a result to see your original, or switch to a slider, to judge how visible the changes are
- Per-image poison tag (with Copy buttons) and a ready-made ALT-text list, as Nightshade recommends posting them with the image
- Strength (Light, Balanced, Strong) and render quality (Fast to Maximum), optional tag, target concept and seed
- Live per-image progress, a job queue, automatic cleanup after an hour, optional password
- Runs on an NVIDIA GPU (about 20 s per image at the default quality on a mid-range card)

Hemlock raises the cost of training on your work. It is not a guarantee, and its protective effect has **not** been
evaluated here. Keep your originals. Not affiliated with the Nightshade team.

## Unraid

Install the Nvidia Driver plugin, then fetch the template from the Unraid terminal:

```bash
wget -O /boot/config/plugins/dockerMan/templates-user/my-hemlock-docker.xml https://raw.githubusercontent.com/adman234/hemlock-docker/main/unraid/hemlock-docker.xml
```

Then Docker tab, Add Container, pick `hemlock-docker` from the template list. The default web port is 7861 so it can sit next to mist-docker.
The first start downloads Stable Diffusion 1.5 (about 2.5 GB) and two small tagging models into the data folder.

## Docker

```bash
docker run -d --name hemlock --gpus all -p 7861:7860 -v /path/to/appdata:/data ghcr.io/adman234/hemlock-docker:latest
```

Settings (environment variables): `HEMLOCK_PASSWORD`, `HEMLOCK_USERNAME`, `HEMLOCK_MAX_FILES` (20), `HEMLOCK_QUEUE_MAX` (20),
`HEMLOCK_JOB_TTL_MINUTES` (60), `HEMLOCK_CONCEPT_MODELS` (1; set 0 to skip the tag-detection models), `PUID`/`PGID`/`UMASK`.
Put `concepts.txt` (one word per line) or `mapping.json` (concept to target) in the data folder to override the built-in word list.
The image is built by GitHub Actions on every push to `main` (`ghcr.io/adman234/hemlock-docker:latest`).

## Repository layout

| Path | Content |
|---|---|
| `docker/` | web app (`app.py`), entrypoint, smoke test, requirements, word list |
| `unraid/hemlock-docker.xml` | Unraid template |
| `nightshade-reimplementation/` | the `nsre` library and CLI (independent implementation) with 63 tests |
| `research/` | the reverse-engineering notes, evidence ledger, experiments and parity report |

No vendor code, disassembly, model weights or proprietary assets are stored here. `research/tools/` has the scripts that unpack and
inspect your own copy of the application.

## CLI and tests

```bash
cd nightshade-reimplementation
pip install torch torchvision numpy pillow einops pytest
PYTHONPATH=src:tests python -m pytest -q          # 63 passed, 1 skipped (GPU test)
```

## What the research found

Full details in `research/` (start with `evidence.md` and `parity-report.md`).

- The Windows app is a PyInstaller build with plain Python 3.11 bytecode. No weights ship; nine assets download on first run.
- The method: Stable Diffusion img2img makes a target image of a different concept, then a perturbation is optimized so the
  **SD VAE encoder latent** of the perturbed image matches the target's, under an LPIPS (vgg16) hinge penalty and a clamp of
  1.3 x max_change (0.035 / 0.07 / 0.12 for the three intensities; 100 / 150 / 300 / 500 steps with 1 / 2 / 3 / 6 crop samples for the four render qualities, HIGH adds 200).
- Parity with the original, measured on the same GPU and assets: parameters and deterministic stages identical (115/115 checks,
  one deliberate deviation), the optimizer bit-identical for the same inputs in 4 of 5 cases (the fifth differs only by CUDA
  nondeterminism that the original shows against itself), selection rules identical, and end-to-end statistics equal (MAE 2.74 vs 2.72,
  PSNR 36.96 vs 37.01 dB over 22 matched runs).
- Defects found in the original: GIF input crashes, EXIF orientation is applied twice by viewers, no-valid-input crashes with
  `UnboundLocalError`, the "already shaded" skip filter never matches its own output names, and render level -1 raises `KeyError`.
- The original is not repeatable run to run (torch is never seeded); here you can pass a seed.
- Open: whether the protection actually works against real model training, and the licence terms for derived work.
