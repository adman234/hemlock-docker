# Known limitations and unknowns

## Limitations of the reimplementation

| ID | Limitation | Class |
|---|---|---|
| L-001 | The target concept pool and the precomputed concept mapping are vendor data (E-094) and are not reproduced. nsre takes a caller-supplied pool (`--concepts-file`) and optional mapping (`--mapping-file`); the web app ships its own word list. The selection rule is identical (13/13, 20/20, 60/60 parity), the candidates are not, so targets differ for the same tag. | approximation |
| L-002 | BLIP and CLIP weights come from Hugging Face (`Salesforce/blip-image-captioning-base`, `openai/clip-vit-base-patch32`). Architectures match the originals (E-092, E-093); weight identity was not compared. | unverified |
| L-003 | LPIPS: nsre builds `lpips.LPIPS(net='vgg')` (the same architecture as the original, E-091); ImageNet backbone weights come from torchvision's download. Parity runs used the original's `lpips_fn.p` locally. | approximation |
| L-004 | Concept re-ranking with CLIP when several nouns remain is implemented in `Engine.detect_concept` and its rule is parity tested, but the Hugging Face BLIP/CLIP path has not been executed end to end here (no download of those models). | unverified |
| L-005 | opt_setting `'0'` (render level -1) defaults `eot` to 1 instead of raising `KeyError` (E-034, E-085). | deliberate deviation |
| L-006 | The CLI and web app skip names containing `-protected-intensity`, the marker the output files really carry, instead of the original's non-matching `-nightshade-intensity` (E-024, E-083). | deliberate deviation |
| L-007 | GIF and other formats without `_getexif` are accepted (E-080). | deliberate deviation |
| L-008 | The web app resets the EXIF Orientation tag to 1 on output (E-081); the library default keeps parity with the original. | deliberate deviation |
| L-009 | No Windows/macOS GUI, download manager, update check or MPS batch limit. The web app replaces the GUI. | out of scope |
| L-010 | Poisoning effectiveness against any model is not evaluated, for the original or the reimplementation. | out of scope |
| L-011 | Output is not pixel-reproducible across runs unless `--seed` is used, matching the original's nondeterminism in spirit (the original has no seed at all). | by design |
| L-013 | The container runs on diffusers 0.25.1 / transformers 4.36.2 / Pillow 10.4 (E-103), not the original's 0.14 / 4.29 / 9.3. Statistics match; fixed-input optimizer outputs are not bit-identical across the two sets (E-104). The bit-identical optimizer parity (E-101) was measured on the original's pins. | deliberate deviation |
| L-012 | The Docker image has not been built on the development machine (no Docker); it is built and smoke-tested by GitHub Actions. CUDA vs CPU unit test is skipped without a GPU in the test environment. | not verified locally |

## Unknown register

| ID | Question | Status |
|---|---|---|
| U-001 | Size and content of the target pool, embedding model. | **resolved structurally**: 178 words, 178x512 float16 CLIP ViT-B/32 text features, 393 mapped concepts (E-094). Content intentionally not reproduced. |
| U-002 | Message of the bare exception for unsupported shapes. | **resolved**: no message (E-050). |
| U-003 | BLIP variant. | **resolved**: base (E-093). |
| U-004 | CLIP variant. | **resolved**: ViT-B/32 (E-092). |
| U-005 | LPIPS backbone. | **resolved**: vgg16 (E-091). |
| U-006 | Contents of `base.zip`. | **resolved**: SD configs only (E-095). |
| U-007 | Run-to-run variance, VRAM and time per setting. | **resolved**: measured (E-086..E-089). |
| U-008 | Which GUI time-label set is GPU vs CPU. | open, low value: both are far above measured times (E-087). |
| U-009 | Whether download URLs still resolve. | **resolved**: yes, Hugging Face ones redirect to a successor repo; the UChicago mirror returns 503 under parallel load (E-090). |
| U-010 | Terms of use of Nightshade and the assets with respect to reimplementation. | open: not reviewed. No vendor code, weights, pool or mapping are redistributed here. |
| U-011 | Whether `check_glaze_update` and `app` contact other hosts at runtime. | open: only the update-check URL and the downloader hosts are present in the bytecode; no runtime capture was done. |
| U-012 | Why the original uses about 3 GB more VRAM than nsre on the same card. | open: hypothesis only. |
| U-013 | Whether protection survives training, scaling or JPEG on any real model. | open and out of scope. |
