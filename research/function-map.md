# Function map

Original (Nightshade 1.1 bytecode, line numbers from the recovered `co_firstlineno`) to reimplementation.

| Original | Line | Role | Calls | Reimplementation | Evidence |
|---|---|---|---|---|---|
| `app.GlazeAPP.protect_images` | app:588 | collects sliders, starts `Worker` | `Worker.run` | n/a (GUI) | E-020 |
| `app.Worker.run` | app:71 | saves `saved_params.p`, runs pipeline | `update_params`, `run_protection_prod` | `cli.main` / `runner.run_batch` | E-020 |
| `nightshade.Nightshade.__init__` | nightshade:39 | builds `Optimizer`, `ImageAnalyzer('cpu')` | `detect_device` | `runner.run_batch` | E-070 |
| `Nightshade.update_params` | 68 | slider to opt_setting; validates output dir | | `params.compute_level_to_opt_setting` | E-021 |
| `Nightshade.run_protection_prod` | 98 | per-image driver | `get_app_parameters`, `load_img`, `get_current_concept`, `get_target_style`, `generate_perturbation` | `runner.run_batch` + `pipeline.shade_image` | E-022 |
| `Nightshade.detect_device` / `check_device_health` | 180 / 229 | device choice and fp16 smoke test | | `device.py` | E-070 |
| `nightshade.main` | 243 | argparse CLI | | `cli.py` | E-023 |
| `utils.get_app_parameters` | utils:35 | parameter dict | | `params.get_params` | E-030..E-032 |
| `utils.ImageAnalyzer.get_blip_caption` | 130 | BLIP caption | | `runner._blip_caption` | E-060 |
| `ImageAnalyzer.get_current_concept` | 141 | noun extraction and CLIP tie-break | `concept_candidate_filter` | `concept.extract_nouns/pick_concept` | E-061 |
| `ImageAnalyzer.get_target_style` | 117 | target pick | CLIP text embedding | `concept.pick_target` | E-062 |
| `utils.load_img` | 304 | open, orient, RGB, cap | `reduce_quality` | `imageio.load_img` | E-040 |
| `utils.reduce_quality` | 340 | long side cap 5120 | | `imageio.reduce_quality` | E-041 |
| `utils.contain_image` | 371 | fit long side to 512 | | `imageio.contain_image` | E-045 |
| `utils.img2tensor` / `tensor2img` | 348 / 357 | PIL <-> tensor | | `imageio.img2tensor/tensor2img` | E-042, E-044 |
| `utils.check_is_img` | 421 | `Image.open().verify()` style test | | `imageio.check_is_img` | E-022 |
| `utils.load_pickle` | 188 | Fernet + pickle | | not reimplemented (proprietary asset) | E-011 |
| `opt.Optimizer.generate_perturbation` | opt:106 | image pipeline | `load_model`, `get_target_image`, `get_mask`, `core_optimize`, `add_perturbation_full_size`, `save_image` | `pipeline.shade_image` | E-042 |
| `Optimizer.load_model` | 332 | SD img2img pipeline | | `models.load_img2img_pipeline` | E-046 |
| `Optimizer.get_target_image` | 348 | SD target | `contain_image`, `img2tensor`, `unload_sd_model` | `target.generate_target_image` | E-045 |
| `Optimizer.get_mask` | 277 | change mask | | `optimize.get_mask` | E-055 |
| `Optimizer.get_latent_with_crop` | 173 | crop and VAE encode | | `optimize.crop_for_latent` + injected encoder | E-052 |
| `Optimizer.core_optimize` | 181 | attack loop | | `optimize.core_optimize` | E-050..E-054 |
| `Optimizer.add_perturbation_full_size` | 155 | lift to full res | | `imageio.add_perturbation_full_size` | E-044 |
| `Optimizer.save_image` | 292 | naming and format | | `imageio.output_name/save_image` | E-043 |
| `Optimizer.unload_sd_model` | 92 | free VRAM | | not needed (pipeline lifetime owned by caller) | E-047 |
| `downloader.download_all_resources` | downloader:18 | fetch assets with MD5 | | not reimplemented | E-010 |
