# Architecture

## Summary (recovered, E-020..E-072)

Nightshade 1.1 is a PyQt6 front end over a single-image optimizer. For each image it
1. detects a content "concept" word (BLIP caption, noun filter, CLIP tie-break),
2. picks a different "target" concept from a pool by CLIP text similarity (ranks 4 to 9, seeded choice),
3. synthesizes a target image of that concept with Stable Diffusion v1.5 img2img (strength 0.8, 30 steps, guidance 7.5),
4. optimizes an additive perturbation on a 512-short-side copy so that the **SD VAE encoder latent** of the perturbed image matches the latent of the target image (L2 norm, random 512x512 crops, expectation over `eot` crops), under an LPIPS hinge penalty and an element-wise clamp of 1.3 x max_change,
5. transfers the perturbation to the original resolution by resizing (perturbed - source) and adding it, then saves PNG or JPEG with original EXIF and writes the concept tags to `nightshade-result.txt`.

The UNet and text encoder are used only in step 3. The attacked feature space is the VAE encoder (step 4).

## Diagram

```
 GUI (app.py)                    CLI (nightshade.main)
 intensity 0..2, rq 0..3            -d -od -m -rq -t
        │                                 │
        └───────────────┬─────────────────┘
                        v
             Nightshade.update_params / run_protection_prod
                        │   get_app_parameters(LEVEL, opt_setting)
                        v
   ┌────────────── per image ─────────────────────────────────────────┐
   │ load_img (EXIF 3/6/8, RGB, cap 5120)                              │
   │      │                                                            │
   │      ├─> ImageAnalyzer.get_current_concept  (BLIP -> POS -> CLIP) │
   │      │        └─> get_target_style (CLIP sim, ranks 4..9, seed)   │
   │      v                                                            │
   │ Optimizer.generate_perturbation                                   │
   │   shrink short side to 512 (bilinear) ──> source tensor [-1,1]    │
   │   get_target_image: contain 512 + black canvas                    │
   │        └─> SD v1.5 img2img ("a photo of a T", 0.8, 7.5, 30)       │
   │        └─> crop + resize ──> target tensor                        │
   │   get_mask (computed; unused by the loop)                         │
   │   core_optimize ─ steps x eot crops:                              │
   │        VAE.encode(crop(resize(clamp(src+δ)))) vs VAE.encode(crop(tgt))  │
   │        loss = ||Δlatent||₂  +  λ·Σ max(LPIPS(adv,src) − max_change, 0)   │
   │        δ ← clamp(δ − lr_i·mean(∇), ±1.3·max_change)               │
   │   add_perturbation_full_size (resize δ-image to original, uint8)  │
   │   save_image (PNG if path endswith ".png" else JPEG q=100, EXIF)  │
   └───────────────────────────────────────────────────────────────────┘
                        v
              nightshade-result.txt (path: concept lines)
```

## Reimplementation mapping

| Original | Reimplementation (`nightshade-reimplementation/src/nsre`) |
|---|---|
| `utils.get_app_parameters`, `update_params` | `params.py` |
| `utils.load_img/reduce_quality/contain_image/img2tensor/tensor2img`, `opt.save_image`, `add_perturbation_full_size` | `imageio.py` |
| `opt.core_optimize/get_mask/get_latent_with_crop` | `optimize.py` (models injected) |
| `utils.ImageAnalyzer` selection rules | `concept.py` (pool is caller supplied) |
| `opt.get_target_image` | `target.py` |
| model loading | `models.py` (public substitutes, lazy imports) |
| `detect_device`, `check_device_health` | `device.py` |
| logging, hashes, manifests, optional seeding | `repro.py` |
| `generate_perturbation` | `pipeline.py` |
| `nightshade.main` | `cli.py`, `runner.py` |
