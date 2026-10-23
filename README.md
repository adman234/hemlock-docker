# Hemlock

A small web app you run yourself that puts protective noise on artwork before you post it online. It bundles two methods in one container: Hemlock, a Nightshade-style method written for this project, and Mist v2.

It was built on a home server with a single GPU, so it does one job at a time and everyone else waits in line. Upload one image or a batch, wait, then press and hold a result to flip back to your original. On a desktop there's also a full-size view for checking pixels.

Neither method is a guarantee. Nobody has measured here whether Hemlock actually changes what a model learns, and Mist's effect is whatever its authors report. Keep your originals.

## The two methods

**Hemlock** nudges an image towards a different concept (say "dog" towards "car"), so that a model trained on lots of images like it could learn the wrong thing. On an RTX 4070 SUPER it takes about 20 seconds to a minute and a half per image (roughly 1.5 to 7 minutes on a GTX 1070, my estimate), and it gives you a tag to put in the image's ALT text when you post it.

**Mist v2** ([psyker-team/mist-v2](https://github.com/psyker-team/mist-v2), Apache-2.0, included under `mist/`) adds noise aimed at tools that copy an artist's style. It takes a few minutes per image on a fast card, and would be much slower on a GTX 1070 and may not fit in its 8 GB.

Neither will change what a chat assistant says about a single image. They work on what a model learns from training data, not on what it sees at inference time.

## Running it

On Unraid, install the Nvidia Driver plugin and fetch the template from the terminal:

```bash
wget -O /boot/config/plugins/dockerMan/templates-user/my-hemlock-docker.xml https://raw.githubusercontent.com/adman234/hemlock-docker/main/unraid/hemlock-docker.xml
```

Then add the container from the Docker tab. The web port defaults to 7861 so it can sit beside mist-docker. If you already have mist-docker's data folder, point the Data path at it and the Stable Diffusion download gets reused.

With plain Docker:

```bash
docker run -d --name hemlock --gpus all -p 7861:7860 -v /path/to/appdata:/data ghcr.io/adman234/hemlock-docker:latest
```

For older cards there is a separate Unraid template that already uses the `pascal` tag and slow-card defaults:

```bash
wget -O /boot/config/plugins/dockerMan/templates-user/my-hemlock-docker-pascal.xml https://raw.githubusercontent.com/adman234/hemlock-docker/main/unraid/hemlock-docker-pascal.xml
```

Older cards (GTX 10 series, Pascal) need the `pascal` tag instead of `latest`: `ghcr.io/adman234/hemlock-docker:pascal`. It is built from the same code with CUDA 12.6, and it is updated whenever `latest` is. Don't use it on RTX 50 series cards. On Unraid, change the Repository field to that tag. An 8 GB card should handle Hemlock but probably not Mist.

The first start downloads Stable Diffusion 1.5 (about 5 GB, shared by both methods) and two small captioning models. The image itself is about 4.4 GB to pull, most of it PyTorch and the CUDA libraries.

### Settings

All environment variables, all optional:

| Variable | Default | What it does |
|---|---|---|
| `HEMLOCK_IDLE_UNLOAD_MINUTES` | 10 | Frees the GPU and RAM after this long without a job. 0 frees after every job, -1 never. |
| `HEMLOCK_JOB_TTL_MINUTES` | 60 | Uploads and results are deleted after this long. |
| `HEMLOCK_PASSWORD` | none | Turns on a password page (password only, no username). |
| `HEMLOCK_LOGIN_MAX_FAILS`, `HEMLOCK_LOGIN_LOCK_MINUTES` | 5, 15 | After this many wrong passwords from one address it is locked out for this long. |
| `HEMLOCK_MAX_FILES`, `MIST_MAX_FILES` | 20, 20 | Images allowed per run for each method. |
| `HEMLOCK_QUEUE_MAX` | 20 | How many runs can wait before new ones are refused. |
| `HEMLOCK_DEFAULT_STRENGTH`, `HEMLOCK_DEFAULT_QUALITY` | Balanced, Better | What the Hemlock form starts on: Light, Balanced or Strong, and Fast, Better, Thorough or Maximum. |
| `MIST_DEFAULT_STRENGTH`, `MIST_DEFAULT_PASSES` | Balanced, 5 | What the Mist form starts on: Light, Balanced or Strong, and 1 to 10 passes. |
| `HEMLOCK_ENABLE_MIST` | 1 | Set to `0` to turn Mist off. The method picker disappears and the page is Hemlock only. Sensible on cards with 8 GB or less. |
| `HEMLOCK_DEFAULT_METHOD` | hemlock | Set to `mist` to open on Mist instead. |
| `HEMLOCK_SPEED_FACTOR` | 4.5 | Scales the starting time guesses. 4.5 is about a GTX 1070, 1 is about an RTX 4070 SUPER. After a few runs the server uses the times it measured itself. |
| `HEMLOCK_CONCEPT_MODELS` | 1 | Set to 0 to skip the captioning models and type tags yourself. |
| `MIST_MAX_SIDE`, `MIST_CLASS_IMAGES` | 4096, 20 | Mist's resize cap and the number of comparison images it makes on first run. |
| `PUID`, `PGID`, `UMASK` | 99, 100, 022 | File ownership. |

Drop a `concepts.txt` (one word per line) or a `mapping.json` into the data folder to replace Hemlock's built-in word list.

With a password set, wrong guesses are rate limited by the app itself (see above). Behind a reverse proxy such as Nginx Proxy Manager it reads the visitor's address from `X-Real-IP`, which nginx sets. If you use Cloudflare in front of it, make nginx see the real visitor address, otherwise everyone shares one limit.

Each run shows how many are ahead of it and a time estimate, which is refined from the run's own progress as it goes.

Models are only in memory while a job needs them. Hemlock runs in its own process that gets shut down when idle, which is what actually gives the memory back. The next job after that takes about ten seconds longer to start.

## Picking Hemlock settings

These numbers are from an RTX 4070 SUPER on 512 pixel images. "Change" means how far a pixel moves on a 0 to 255 scale.

| Strength | Biggest change | Average change | In practice |
|---|---|---|---|
| Light | 6 | 1.1 | Close to invisible |
| Balanced | 12 | 2.4 to 2.9 | A little grain if you zoom in |
| Strong | 20 | 5.1 | Texture on flat areas, and it takes about twice as long |

| Render quality | 4070 SUPER | GTX 1070 (estimate) | What it does |
|---|---|---|---|
| Fast | about 20 s | about 1.5 min | 100 passes, one random crop each |
| Better | about 30 s | about 2.5 min | 150 passes, two crops |
| Thorough | about 1 min | about 4.5 min | 300 passes, three crops |
| Maximum | about 1.5 min | about 7 min | 500 passes, six crops |

Strength limits how far pixels can move. Render quality is how long it keeps optimising, and the slower settings test more crops, which should help the effect survive cropping and resizing. The defaults are Balanced and Better. Nobody has checked whether a slower setting poisons any better.

## ALT text

Hemlock tags each image with a word (found automatically, or typed in if you upload one image). Nightshade's advice is to put that word in the image's ALT text when you post. Each result has a Copy ALT button, and the zip includes a `hemlock-tags.txt`. A template like `Digital painting of a {tag}` turns the bare word into a sentence.

The ALT text is also written into the image's metadata by default, but most sites strip metadata, so paste it into the ALT field yourself.

## How Hemlock came about

The method was worked out by reading the Nightshade 1.1 Windows app's Python bytecode and running it. The notes, the evidence behind each claim, and the comparison against the original are in `research/`. The short version:

- Stable Diffusion draws a target picture of a different concept, then the image is nudged until the Stable Diffusion image encoder sees it much like the target, while a perceptual loss keeps the changes small.
- On the same GPU and assets, the optimiser gives bit-identical results to the original in four of five test cases. The fifth differs only by GPU nondeterminism, which the original also shows against itself. Output quality statistics match over 22 paired runs.
- The original has some bugs (GIF files crash it, EXIF orientation gets applied twice, the "already shaded" check never matches its own filenames). Hemlock handles those differently, and `research/known-limitations.md` lists every deliberate difference.
- The container runs a newer diffusers than the original because the old one can't read the current Stable Diffusion VAE file.

It contains no Nightshade code or weights, and it's not affiliated with the Nightshade or Mist teams.

## What's in the repo

| Path | What's there |
|---|---|
| `docker/` | The web app, model manager, Hemlock worker, entrypoint and smoke test |
| `mist/` | The mist-docker repository, merged in with its history (Apache-2.0, see `mist/LICENSE`) |
| `nightshade-reimplementation/` | The `nsre` library and command line tool, with tests |
| `research/` | Reverse-engineering notes, experiments and the parity report |
| `unraid/hemlock-docker.xml` | The Unraid template |
