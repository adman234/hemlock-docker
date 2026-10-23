# Evidence ledger

Target: Nightshade 1.1 for Windows, `C:\Users\Adam\Downloads\Nightshade-1.1-Windows\Nightshade` (left unmodified).
All analysis was done on copies under `C:\ns\work` using the scripts in `research/tools/`.

Confidence classes: **O** = observed directly in a shipped artifact, **D** = derived mechanically from observed bytecode,
**I** = inference, **H** = hypothesis, **U** = unknown.

Module identifiers below refer to modules inside the `PYZ.pyz` archive embedded in `Nightshade.exe`
(`nightshade`, `opt`, `utils`, `downloader`) and to the `app` entry script. "dis" means the CPython 3.11
disassembly produced by `research/tools/dis.py`. Raw disassembly is not committed (it is vendor code); rerun the tools to regenerate it.

## Artifact identity

| ID | Class | Finding | Evidence |
|---|---|---|---|
| E-001 | O | `Nightshade.exe` is a 63,380,224-byte PE32+ x64 GUI executable, unsigned (no certificate directory), file time 2026-04-15, SHA-256 `07723B4C5890B9CD948551055E5256B58F148839D061639DEC0D48367C6CD6F9`. | `research/tools/pe.py`, Get-FileHash |
| E-002 | O | It is a PyInstaller one-directory build. The CArchive contains `struct`, `pyimod01..04`, `pyiboot01_bootstrap`, runtime hooks (inspect, pkgutil, setuptools, pkgres, cryptography_openssl, _tkinter, nltk, pyqt6, multiprocessing), the entry script `app`, and `PYZ.pyz` (63,014,391 bytes). | `research/tools/carch.py` TOC listing |
| E-003 | O | Python 3.11 runtime (`_internal/python311.dll`, bytecode parses and disassembles with CPython 3.11). | file listing; `dis.py` |
| E-004 | O | App bytecode is plain marshal: no encryption, no obfuscation, no stripped names. Local names, string constants and keyword-argument tuples are all recoverable. | `dis.py`, `tuples.py` |
| E-005 | O | PYZ holds 7,235 modules. App-specific code is `nightshade` (13,083 B marshal), `opt` (17,993 B), `utils` (23,294 B), `downloader` (10,310 B), plus bundled `clip`, `lpips`, `einops`. Entry script `app` (40,263 B). | `pyz.py` |
| E-006 | O | Bundled libraries (dist-info): torch 2.8.0+cu128, torchvision 0.23.0+cu128, diffusers 0.14.0, transformers 4.29.0, tokenizers 0.13.3, safetensors 0.4.1, huggingface_hub 0.19.4, openai-clip 1.0.1, lpips 0.1.4, nltk 3.8, numpy 1.24.1, Pillow 9.3.0, scikit-learn 1.3.2, scipy 1.15.3, PyQt6, cryptography 46.0.6, requests 2.28.1. | `_internal/*.dist-info` |
| E-007 | O | No model weights are shipped. `_internal` has no checkpoint, `.pt`, `.p`, `.bin` or `.safetensors` file; the largest files are CUDA/cuDNN DLLs. `~/.nightshade` does not exist on this machine, so the application has never been run here. | directory listing |

## Resource acquisition (module `downloader`)

| ID | Class | Finding |
|---|---|---|
| E-010 | O | `download_all_resources` fetches 9 resources into `%USERPROFILE%\.nightshade` and validates each by MD5: `base.zip` (mirror.cs.uchicago.edu/fawkes/files/glaze/, md5 `0404aa8a...e6`), SD unet / text_encoder / vae `.bin` from `huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/...` (md5 `f5489682...`, `167df822...`, `a90d1567...`), `bpe_simple_vocab_16e6.txt.gz`, `clip_model.p`, `lpips_fn.p`, `full_asset3.p`, `blip_model.pt`, `blip_preprocessor2.pt` (all from the UChicago mirror). Plain `http://`, MD5 only. |
| E-011 | O | `utils.load_pickle` decrypts a file with `cryptography.fernet.Fernet` using a key embedded as a bytes constant in `utils`, then `pickle.loads`. Used for `full_asset3.p`, which provides `target_concepts`, `target_concepts_embs`, `precompute_mapping`. |
| E-012 | O | `clip_model.p`, `blip_model.pt`, `blip_preprocessor2.pt` are loaded with `torch.load(..., weights_only=False)`; `nightshade` also monkeypatches `torch.load` to force `weights_only=False`. `lpips_fn.p` is loaded with `pickle.load`. These are unpickled code paths. |
| E-013 | O | App checks version via `http://glaze.cs.uchicago.edu/software/glaze_version.json` (`check_glaze_update`). |

## Pipeline entry points

| ID | Class | Finding |
|---|---|---|
| E-020 | O | GUI `GlazeAPP` (module `app`): intensity QSlider min 0 max 2 default 1; render-quality QSlider min 0 max 3 default 0. Slider values are passed as `intensity` and `rq` to `Worker`, which calls `Nightshade.update_params(output_dir, intensity, rq, signal)` then `run_protection_prod(image_paths, overwrite_concept)`. File dialog filter `Images (*.png *.xpm *.jpg *jpeg *.gif)`. Custom tag is only allowed when exactly one image is selected and must be 3 English words or less. |
| E-021 | O | `Nightshade.update_params`: compute_level -1 -> opt_setting `'0'`, 0 -> `'1'`, 1 -> `'2'`, 2 -> `'3'`, 3 -> `'4'`, otherwise raises `Unknown render level`. If level != `'0'` an output folder is mandatory. |
| E-022 | O | `run_protection_prod`: p_level 0/1/2 -> `LOW`/`DEFAULT`/`HIGH`; calls `get_app_parameters`; filters paths with `os.path.isfile` and `check_is_img`; refuses more than 3 images on `mps`; per image skips names containing `-nightshade-intensity`; `load_img`; `get_current_concept`; optional `overwrite_concept`; `get_target_style`; optional `force_target`; style dict `{'style': 'a photo of a {target}', 'strength': 0.8}`; `Optimizer.generate_perturbation(image_paths=[p], parameter=..., target_params=...)`; appends `"{out_file}: {concept}"` to the guide list; writes `nightshade-result.txt` in the output dir with a header telling users to add the tags to ALT text. |
| E-023 | O | CLI `main`: `-d/--directory` default `imgs/`, `-od/--out-dir`, `-m/--mode` int default 0, `-rq/--rq` int default 0, `-t/--target`. Note `-rq` 0 maps to opt_setting `'1'` (fastest real mode). |
| E-024 | O | The skip filter string `-nightshade-intensity` never matches the app's own output suffix `-protected-intensity-<LEVEL>-V1` (E-043), so shaded outputs are not skipped on a re-run. Quirk, not a bug in the reimplementation. |

## Parameters (module `utils.get_app_parameters`)

| ID | Class | Finding |
|---|---|---|
| E-030 | O | LOW: max_change 0.035, lr_initial 0.03. DEFAULT: 0.07 / 0.1. HIGH: 0.12 / 0.2. Other level raises. |
| E-031 | O | Common: n_runs 1, style_transfer_iter 30, lr_const 0.005, penalty_initial 500, lpips True. |
| E-032 | O | opt_setting: `'0'` tot_steps 4 (no eot key); `'1'` eot 1, steps 100, penalty 1000; `'2'` eot 2, 150, 500; `'3'` eot 3, 300, 500; `'4'` eot 6, 500, 800. HIGH adds 200 to tot_steps. |
| E-033 | I | `lr_const` and `n_runs` are stored but no read of them was found in `opt` or `nightshade` (searched by name in both disassemblies). |
| E-034 | I | opt_setting `'0'` has no `eot` entry, and `core_optimize` reads `parameters['eot']`, so `'0'` likely raises KeyError for non-square inputs; for 512x512 inputs `eot` is overwritten to 1 before use. Untested at runtime. |

## Preprocessing and I/O (module `utils`, `opt`)

| ID | Class | Finding |
|---|---|---|
| E-040 | O | `load_img`: returns None for missing path, `UnidentifiedImageError`, `IsADirectoryError`, `OSError` on `_getexif`; applies EXIF orientation 3 -> rotate(180), 6 -> rotate(270), 8 -> rotate(90), all `expand=True`; `.convert('RGB')`; `reduce_quality`. |
| E-041 | O | `reduce_quality`: if max(size) > 5120, `thumbnail((5120, 5120), LANCZOS)` in place. |
| E-042 | O | `generate_perturbation`: `getexif()` kept; `og_size = (height, width)`; `shrink_shortside_transform = Resize(512, BILINEAR)` (shorter side to 512); `img2tensor` = `np.array(img)/127.5 - 1.0` as float32, `h w c -> c h w`, unsqueeze(0); both source and target tensors moved to the device and `.half()` on cuda/mps. |
| E-043 | O | Output name `<stem>-protected-intensity-<LOW|DEFAULT|HIGH>-V1.<ext>`; saved with `format='PNG'` when the original path ends with the exact string `.png`, else `format='JPEG'`; `exif=<original exif>`, `quality=100`. No subsampling argument. |
| E-044 | O | `add_perturbation_full_size`: `tensor2img` of source and perturbed tensors, each `.resize(og_size)` (PIL default filter), perturbation = float(perturbed) - float(source), added to float32 full image, `np.clip(0,255)`, `astype(uint8)`. `tensor2img` is `clamp((x+1)/2, 0, 1)*255` then `astype(uint8)` (truncation). |
| E-045 | O | `get_target_image`: `contain_image` (longer side to 512), pastes into a 512x512 black canvas at top-left, builds a tensor with `img2tensor`, runs `StableDiffusionImg2ImgPipeline(prompt=[style], image=tensor, strength=0.8, guidance_scale=7.5, num_inference_steps=style_transfer_iter(30))` under `torch.no_grad`, takes `.images[0]`, crops the pasted region, resizes to the working image size, then `unload_sd_model`. |
| E-046 | O | Pipeline load: cuda/mps `from_pretrained(<root>/base/base, torch_dtype=float16)`; CPU `from_pretrained(...).to(device)` plus `enable_attention_slicing()`. VAE: `AutoencoderKL.from_pretrained(<root>/base/base, subfolder='vae')`, `.half()` on cuda/mps. |
| E-047 | O | `unload_sd_model` is a no-op when `extra_large_gpu` (cuda with > 10000 MB); otherwise moves the pipeline to CPU, drops it, empties the cache, `gc.collect()`. |

## Optimization core (module `opt`, `Optimizer.core_optimize`)

| ID | Class | Finding |
|---|---|---|
| E-050 | O | Geometry: width > 512 and height == 512 -> `r_range = width - 512`, `long_width = True`; height > 512 and width == 512 -> `r_range = height - 512`, `long_width = False`; both 512 -> `r_range = 0`, `eot = 1`; otherwise raises. |
| E-051 | O | `modifier = clone(source) * 0.0` (zero perturbation start). Step size `a_i = lr - ((lr - lr/100)/T) * i`. |
| E-052 | O | Per step, per eot sample: crop index `random.randrange(0, r_range)` (module-level `random`, 0 when `r_range == 0`); target latent = `vae.encode(crop(target))` `.latent_dist.mean`; `adv = clamp(modifier + source, -1, 1)`; `adv = Resize(512)(Resize(og_size)(adv))`; adv latent from the same crop; `loss = (adv_latent - target_latent).norm()` (L2 norm over the whole latent); `grad = autograd.grad(loss, modifier)`; gradients are summed and divided by `eot`. |
| E-053 | O | LPIPS branch (`is_lpips = True` hard-coded): `adv = clamp(modifier + source)`, `lp = loss_fn(adv, source)`, `penalty = maximum(lp - max_change, lp*0).sum() * penalty_initial`, `grad_lp = autograd.grad(penalty, modifier)`, added to the averaged gradient. Update `modifier = modifier - grad * a_i` (raw gradient, not sign), then `clamp(modifier, -1.3*max_change, 1.3*max_change)`. A sign-gradient (`modifier - sign(grad)*a_i`, clamp to `+-max_change`) branch exists but is dead code because `is_lpips` is always True. |
| E-054 | O | Result: `final = clamp(modifier.detach() + source, -1, 1)`. |
| E-055 | O | `get_mask`: `diff = (source - target).sum(1, keepdims=True)`, `mask = (|diff| > 0.3).int()` repeated over 3 channels, `masked_target = target*mask + (1-mask)*source`; if mean(mask) > 0.3 or < 0.05 the mask becomes all ones. `core_optimize` receives the mask as an argument but contains no read of it; only `masked_target` is used. |
| E-056 | O | The module flag `CHECK_RUN = False`, so `torch.manual_seed(1234)` is never executed: no torch seeding anywhere in `opt`. |

## Concept and target selection (module `utils.ImageAnalyzer`)

| ID | Class | Finding |
|---|---|---|
| E-060 | O | BLIP captioning: conditional prefix text `'a photography of'`, `blip_model.generate(**inputs)`, `decode(skip_special_tokens=True)`. |
| E-061 | O | `get_current_concept`: split caption on `' '`, `nltk.pos_tag`, drop the first 3 tagged tokens, keep tags whose first two characters are `NN`, drop words in `('image','photo','white','new','black','design','painting')`; 0 left -> `'painting'`, 1 -> that word, more -> CLIP zero-shot over `'a photo of {noun}'` and `argmax`. |
| E-062 | O | `get_target_style`: if the concept is a key of `precompute_mapping` return that. Otherwise CLIP text embedding of the concept, cosine similarity against `target_concepts_embs`, descending `argsort`, ranks `[4:10]`, `random.seed(678121)`, `random.choice`. |
| E-063 | I | Because `random.seed(678121)` reseeds the global generator once per image just before the optimizer draws crop offsets, the crop sequence is deterministic per image in the original when the target comes from this path. With `-t/--target` or `precompute_mapping` hits the seed is not set. |
| E-064 | O | CLIP image preprocessing: Resize 224 BICUBIC, CenterCrop 224, RGB, ToTensor, Normalize mean (0.48145466, 0.4578275, 0.40821073) std (0.26862954, 0.26130258, 0.27577711). |

## Device handling (module `nightshade`)

| ID | Class | Finding |
|---|---|---|
| E-070 | O | `detect_device`: CUDA with total memory > 4500 MB -> `cuda`; MPS (non-i386 processor) -> `mps`; else `cpu`. Accelerators must pass `check_device_health` (fp16 `9*3 == 27`), else CPU. Messages say Nightshade needs at least 5 GB GPU memory. |
| E-071 | O | fp16 is used on cuda/mps (`Optimizer.half`); CPU is fp32. `loss_fn` (LPIPS) moved to device and `.half()` on cuda/mps. |
| E-072 | O | GUI `check_memory` warns when RAM is low ("needs around 5Gb of RAM"). |

## Unknowns

See `research/known-limitations.md` for the unknown register (U-001...).

## Runtime and asset findings (added after the assets were downloaded and the original was run)

Reference runs used the original `nightshade`, `opt` and `utils` modules unpacked from the shipped PYZ (`research/tools/mkpyc.py`),
driven by `research/tools/ref_run.py` in a Python 3.11 environment pinned to the app's library versions, with
`USERPROFILE` pointed at a scratch home holding the downloaded assets. Hardware: RTX 4070 SUPER 12 GB, driver 616.56.
Raw per-run data: `research/results/*.jsonl`; derived tables: `research/measured-results.md`, `experiments.csv`.
Class **R** = observed at run time.

| ID | Class | Finding | Evidence |
|---|---|---|---|
| E-080 | R | GIF input crashes the original with `AttributeError('_getexif')` in `load_img` although the file dialog offers `*.gif`. | X-FMT-6, stage_parity |
| E-081 | R | The original writes the already-rotated pixels but keeps the source EXIF verbatim, so an Orientation tag of 6 survives and viewers rotate the output again. | X-FMT-5 (both implementations) |
| E-082 | R | When no input file survives the filters (unreadable, or skipped by name) `run_protection_prod` ends with `UnboundLocalError: out_file_name`. | X-BAD-* |
| E-083 | R | Skip filter confirmed: `a-nightshade-intensity.png` is skipped, `a-protected-intensity-LOW-V1.png` is processed again and gets a doubled suffix. | X-BAD-a-* (E-024) |
| E-084 | R | `load_img` returns `None` for a truncated PNG (the `OSError` raised by `_getexif`), so it is skipped silently. | stage_parity |
| E-085 | R | render level -1 raises `KeyError('eot')` even for a square image (the key is read before the square override). Confirms and strengthens E-034. | smoke run |
| E-086 | R | Measured limits: perturbation max error 12 levels at DEFAULT, 6 at LOW, 20 at HIGH on 512-scale content, up to 16 or 17 on larger images and 20 to 35 on JPEG output (codec noise). Predicted about 11.6, 5.8, 19.9. | X-INT, X-DIM, X-FMT |
| E-087 | R | Timing on the 4070 SUPER: about 26 s for rq 0, 33 s rq 1, 56 s rq 2, 92 s rq 3 at 512x512 DEFAULT; HIGH rq 0 about 59 s. Peak VRAM 7.4 to 8.1 GB (the app keeps its models resident because the card has more than 10000 MB). On CPU (fp32) rq 0 at 512x512 took 641 s. These are far below the GUI's "5 to 20 min" and "30 to 180 min" labels. | ref_runs, dev_runs |
| E-088 | R | Output container: `.png` input gives PNG; `.jpg` gives JPEG; `.PNG` and `.gif` get JPEG data inside a file with that extension. RGBA, L and P PNG inputs all give RGB PNG (alpha dropped). | X-FMT |
| E-089 | R | Run-to-run variability of the original is large: five identical runs have perturbation cosine similarity of about 0.096 and mean absolute output difference of about 3.3 levels. The perturbation is dominated by the random SD target and crop offsets. | X-REP |

### Assets (asset files inspected with `pickletools`, then loaded after the safe-global check)

| ID | Class | Finding |
|---|---|---|
| E-090 | O | All ten downloads still resolve (the Hugging Face URLs redirect to a successor repository) and match the app's MD5s. The UChicago mirror returns HTTP 503 under parallel connections. |
| E-091 | O | `lpips_fn.p` is `lpips.lpips.LPIPS` with a **vgg16** backbone, version 0.1, `pnet_rand=False`, `lpips=True`, fp32 weights embedded in the pickle (resolves U-005). |
| E-092 | O | `clip_model.p` is OpenAI CLIP **ViT-B/32** (patch 32, width 768, 12 layers, 512-d embeddings, context 77), so `openai/clip-vit-base-patch32` is the right architecture (resolves U-004; weight identity not verified). |
| E-093 | O | `blip_model.pt` is `BlipForConditionalGeneration` whose config names `Salesforce/blip-image-captioning-base` (12+12 layers, hidden 768, image size 384); `blip_preprocessor2.pt` is a `BlipProcessor` (resolves U-003). |
| E-094 | O | `full_asset3.p` decrypts (Fernet, key from E-011) to a pickle that references only numpy array types. Contents: 178 `target_concepts`, a 178x512 float16 embedding matrix (unnormalised, norm about 10.5, so CLIP ViT-B/32 text features), and a 393-entry `precompute_mapping` dict. The word lists are vendor data and are not reproduced in this repository (resolves U-001 structurally). |
| E-095 | O | `base.zip` holds only SD v1.5 configs: `model_index.json`, tokenizer, scheduler, feature_extractor and unet/vae/text_encoder `config.json` (resolves U-006). The three weight files come from the Hugging Face URLs. |

### Parity measurements (see `research/parity-report.md`)

| ID | Class | Finding |
|---|---|---|
| E-100 | R | Deterministic stages (parameters, `load_img`, `img2tensor`, `tensor2img`, `contain_image`, shrink, `add_perturbation_full_size`, `get_mask`, `save_image`) match the original on 115 of 115 checks over the whole corpus (114 identical plus one documented deviation: GIF). |
| E-101 | R | `core_optimize` of the original and of nsre given the same source, target tensor, VAE, LPIPS and RNG state give bit-identical results (cosine 1.0, max difference 0.0) in 4 of 5 cases, including non-square crops and eot of 1, 2 and 3. The fifth (2000x1500, HIGH, 300 steps) differs with cosine 0.642, and the original differs from its own rerun by the same amount (0.6417): CUDA nondeterminism, not an implementation difference. |
| E-102 | R | Concept and target selection rules match: 13 of 13 concept detections, 20 of 20 similarity-ranked targets (ranks 4 to 9, seed 678121), 60 of 60 mapped targets. |
