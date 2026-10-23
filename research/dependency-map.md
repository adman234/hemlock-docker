# Dependency map

## Module graph (application code)

```
Nightshade.exe (bootloader)
  └─ app            GUI: GlazeAPP, Worker, BLIP_Worker, Downloader, ScrollLabel, ReminderClock
       ├─ downloader   download_all_resources, get_file, validate_file, _extract_archive
       ├─ nightshade   Nightshade, main (CLI)
       │    ├─ opt        Optimizer (SD img2img target, VAE-latent attack, LPIPS penalty, save)
       │    └─ utils      get_app_parameters, ImageAnalyzer, CLIP wrapper, load_img, img2tensor, tensor2img, contain_image
       └─ utils         ImageAnalyzer, load_img, resource_path
```

## Third-party libraries and their role

| Library (bundled version) | Used for | Where |
|---|---|---|
| torch 2.8.0+cu128 | tensors, autograd, device dispatch | `opt`, `utils`, `nightshade` |
| torchvision 0.23.0 | `Resize`, `functional.crop`, CLIP-style transforms | `opt`, `utils` |
| diffusers 0.14.0 | `AutoencoderKL`, `StableDiffusionImg2ImgPipeline` | `opt` |
| transformers 4.29.0 / tokenizers 0.13.3 | BLIP captioner (unpickled model) | `utils` |
| openai-clip 1.0.1 (`clip`) | tokenizer and CLIP model wrapper | `utils` |
| lpips 0.1.4 | perceptual penalty (`lpips_fn.p` is a pickled LPIPS module) | `opt` |
| nltk 3.8 (`nltk_data`) | POS tagging (`averaged_perceptron_tagger_eng`) | `utils` |
| scikit-learn 1.3.2 | `cosine_similarity` | `utils` |
| einops | `rearrange` in `img2tensor`/`tensor2img` | `utils` |
| Pillow 9.3.0 | load, EXIF, resize, save | `utils`, `opt` |
| numpy 1.24.1 | array math | everywhere |
| cryptography | Fernet decrypt of `full_asset3.p` | `utils.load_pickle` |
| requests | resource download, update check | `downloader`, `app` |
| PyQt6 | GUI, QThread workers | `app` |
| psutil | RAM check | `app` |
| exiftool (`/usr/local/bin/exiftool`) | `add_meta_data`, a macOS path; not called by `run_protection_prod` | `utils` |

## Model assets (downloaded, not shipped)

| Asset | Source | Role | Reimplementation substitute |
|---|---|---|---|
| SD v1.5 unet/text_encoder/vae (`base/base`) | runwayml/stable-diffusion-v1-5 (HF) | img2img target synthesis; **VAE encoder is the attacked feature space** | public SD v1.5 mirror (CreativeML OpenRAIL-M) |
| `lpips_fn.p` | UChicago mirror | perceptual penalty | `lpips` package (backbone unknown, U-005) |
| `clip_model.p`, `bpe_simple_vocab_16e6.txt.gz` | UChicago mirror | concept disambiguation, target ranking | HF `openai/clip-vit-base-patch32` (variant unknown, U-004) |
| `blip_model.pt`, `blip_preprocessor2.pt` | UChicago mirror | caption to concept | HF BLIP captioning model (variant unknown, U-003) |
| `full_asset3.p` (Fernet) | UChicago mirror | target concept pool, embeddings, precomputed mapping | **none**: caller supplies a pool (U-001) |
| `base.zip` | UChicago mirror | SD config files (json, tokenizer) extracted to `base/base` | diffusers hub files |

All substitutes are fetched by the user at run time from their original distributors. This repository contains no weights and no vendor code.
