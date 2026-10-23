# Hemlock (hemlock-docker)

One self-hosted web app, one Docker image, two image-protection methods. Pick a method, upload one image or a whole batch from a
phone or a computer, compare before and after, download the results.

| Method | What it does | Speed |
|---|---|---|
| **Hemlock** | An independent, open-source take on the [Nightshade](https://nightshade.cs.uchicago.edu/) method. Adds faint, targeted changes so a model *trained* on the image learns the wrong thing. Gives you an ALT-text tag to post with the image. | about 20 s per image |
| **Mist v2** | [Mist v2](https://github.com/psyker-team/mist-v2) (Apache-2.0, kept under `mist/`). Adds noise that disrupts copying an artist's style. | several minutes per image |

Both are deterrents, not guarantees. Hemlock's protective effect has not been measured here (see `research/`). Keep your originals.
Not affiliated with the Nightshade or Mist teams.

## Features

- Bulk upload (Hemlock 20 per run, Mist 4 per run by default), per-image download, and a zip of everything
- Before/after view: press and hold, or a slider. On desktop a **Full size** button opens the images at native resolution with Fit and 100% zoom
- One live progress bar, a Stop button, a job queue, and automatic cleanup of uploads
- Models are loaded only while needed. Hemlock's engine runs in its own process that is shut down after an idle timeout, so an idle
  container holds a small web process and no VRAM; the next job pays a short load (about 10 s)
- Layout adapts: phone-first, wider and larger on desktop
- Optional password

## Unraid

Install the Nvidia Driver plugin, then fetch the template from the Unraid terminal:

```bash
wget -O /boot/config/plugins/dockerMan/templates-user/my-hemlock-docker.xml https://raw.githubusercontent.com/adman234/hemlock-docker/main/unraid/hemlock-docker.xml
```

Then Docker tab, Add Container, choose `hemlock-docker` from the template list. The default web port is 7861 so it can sit next to mist-docker.
The first start downloads Stable Diffusion 1.5 (about 5 GB, shared by both methods) and two small tagging models into the data folder.
If you used mist-docker you can point the Data path at its appdata folder to reuse the Stable Diffusion download.

## Docker

```bash
docker run -d --name hemlock --gpus all -p 7861:7860 -v /path/to/appdata:/data ghcr.io/adman234/hemlock-docker:latest
```

### Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `HEMLOCK_IDLE_UNLOAD_MINUTES` | 10 | Unload models after this many idle minutes. 0 unloads after every job, -1 never. |
| `HEMLOCK_JOB_TTL_MINUTES` | 60 | Delete uploads and results after this long (checked every 5 minutes). |
| `HEMLOCK_PASSWORD`, `HEMLOCK_USERNAME` | empty | Optional login (username defaults to `hemlock`). |
| `HEMLOCK_MAX_FILES` / `MIST_MAX_FILES` | 20 / 4 | Images per run for each method. |
| `MIST_MAX_SIDE`, `MIST_CLASS_IMAGES` | 4096 / 20 | Mist upload size cap, and its first-run comparison images. |
| `HEMLOCK_QUEUE_MAX` | 20 | How many runs may wait. |
| `HEMLOCK_CONCEPT_MODELS` | 1 | 0 skips the two tagging models (type the tag yourself). |
| `PUID`, `PGID`, `UMASK` | 99, 100, 022 | File ownership. |

Put `concepts.txt` (one word per line) or `mapping.json` (concept to target) in the data folder to override Hemlock's built-in word list.

## Choosing options (Hemlock)

Measured on an RTX 4070 SUPER at 512x512 (values are per-pixel changes on a 0 to 255 scale):

| Strength | Largest change | Average | Looks like |
|---|---|---|---|
| Light | 6 | 1.1 | nearly invisible (about 44 dB) |
| Balanced | 12 | 2.4 to 2.9 | faint grain when zoomed in (about 37 dB) |
| Strong | 20 | 5.1 | texture on flat areas (about 31 dB), about twice as slow |

| Render quality | Time per image | What changes |
|---|---|---|
| Fast | about 20 s | 100 passes, 1 random crop per pass |
| Better | about 30 s | 150 passes, 2 crops |
| Thorough | about 1 min | 300 passes, 3 crops |
| Maximum | about 1.5 min | 500 passes, 6 crops |

Strength is how much a pixel may change; render quality is how long it optimises, and slower options test more crops so the effect should
hold up better after cropping or resizing. Defaults are Light and Better. Whether any setting poisons a model more was not measured.

## ALT text

Hemlock names each image with a tag (detected automatically, or typed for a single image). Nightshade's advice is to include the tag in the
image's ALT text when you post it. The results show the suggested ALT text per image with a **Copy ALT** button, plus one combined list
(and a `hemlock-tags.txt` inside the zip). It is also written into the image file's metadata by default, but most sites strip metadata, so
paste it into the ALT field yourself. An optional template (for example `Digital painting of a {tag}`) turns the bare tag into a sentence.

## Image size

One image holds both methods, so torch and the CUDA libraries (about 4 GB compressed) are stored once instead of twice. Models and caches
live in the data volume, not in the image.

## Repository layout

| Path | Content |
|---|---|
| `docker/` | web app, model manager, Hemlock worker process, viewer, entrypoint, smoke test |
| `mist/` | the mist-docker repository, merged with its history (Apache-2.0, see `mist/LICENSE`) |
| `nightshade-reimplementation/` | the `nsre` library and CLI (independent Nightshade-style implementation) with tests |
| `research/` | reverse-engineering notes, evidence ledger, experiments and parity report |
| `unraid/hemlock-docker.xml` | Unraid template |

No Nightshade code, disassembly, weights or proprietary assets are stored here.

## What the research found

Details in `research/` (start with `evidence.md` and `parity-report.md`).

- The Nightshade 1.1 Windows app is a PyInstaller build with plain Python 3.11 bytecode. No weights ship; nine assets download on first run.
- The method: Stable Diffusion img2img makes a target image of a different concept, then a perturbation is optimised so the SD VAE encoder
  latent of the perturbed image matches the target's, under an LPIPS (vgg16) penalty and a clamp of 1.3 x max_change.
- Parity with the original, on the same GPU and assets: parameters and deterministic stages identical (115/115 checks, one deliberate
  deviation), the optimizer bit-identical for the same inputs in 4 of 5 cases (the fifth differs only by CUDA nondeterminism that the original
  shows against itself), selection rules identical, and end-to-end statistics equal.
- The container uses diffusers 0.25.1, because the original's 0.14 cannot read the current Stable Diffusion VAE file (E-103).
- Defects found in the original: GIF input crashes, EXIF orientation is applied twice, no valid input ends in `UnboundLocalError`, the
  "already shaded" filter never matches its own names, render level -1 raises `KeyError`. The original is not repeatable run to run.
- Open: whether the protection works against real model training, and licence terms for derived work.
