# Known limitations and unknowns

## Limitations of the reimplementation

| ID | Limitation | Class |
|---|---|---|
| L-001 | Target concept pool is a caller-supplied list. The original pool, its embeddings and `precompute_mapping` are in the encrypted asset `full_asset3.p`, not reproduced. Target choice therefore differs from the original even though the selection rule (ranks 4..9, seed 678121) is the same. | approximation |
| L-002 | BLIP model: `Salesforce/blip-image-captioning-large` is a guess for `blip_model.pt`. | approximation |
| L-003 | CLIP model: `openai/clip-vit-base-patch32` is a guess for `clip_model.p`. | approximation |
| L-004 | `runner.py` omits the CLIP zero-shot re-ranking when several nouns remain (it takes the first). `concept.pick_concept` implements it and is tested; wiring needs the CLIP image-text path. | gap |
| L-005 | LPIPS backbone default is `alex`; original unknown (U-005). | approximation |
| L-006 | opt_setting `'0'` defaults `eot` to 1 instead of failing (E-034). | deliberate deviation |
| L-007 | The skip filter in `cli.py` uses the real output marker `-protected-intensity`, not the original's non-matching `-nightshade-intensity` (E-024). | deliberate deviation |
| L-008 | No GUI, no download manager, no update check, no MPS batch limit. | out of scope |
| L-009 | GPU consistency test exists but was not run (no CUDA in the test environment); no end-to-end run with real weights has been done. | not verified |
| L-010 | Stand-in tests use a tiny conv encoder, not the SD VAE; they prove algorithm structure, not attack effectiveness. | not verified |
| L-011 | Poisoning effectiveness against any model is not evaluated. | out of scope |

## Unknown register

| ID | Question | Why it matters | Next step |
|---|---|---|---|
| U-001 | Size and content of the target concept pool; embedding model used for `target_concepts_embs`. | determines which targets are chosen | decrypt locally and inspect with `pickletools` (no unpickling) |
| U-002 | Resolved by static reading: the unsupported-shape branch raises `Exception` with no message (E-050). Reimplementation raises `ValueError` with a descriptive message. | error-handling parity | none |
| U-003 | BLIP variant. | caption/concept parity | `pickletools` on `blip_model.pt`, check config |
| U-004 | CLIP variant. | ranking parity | `pickletools` on `clip_model.p` |
| U-005 | LPIPS backbone and whether it is fp16-safe. | penalty magnitude | `pickletools` on `lpips_fn.p` |
| U-006 | Whether `base.zip` contains anything beyond SD config files. | asset completeness | list zip contents after approved download |
| U-007 | Real run-to-run variance and VRAM/time per setting. | parity metrics | experiments X-REP, X-RQ |
| U-008 | Which of the two GUI time-label sets applies to GPU versus CPU. | documentation only | read the GUI branch that picks the label set |
| U-009 | Whether the shipped download URLs still resolve (the SD repo URL points at `runwayml/stable-diffusion-v1-5`). | first-run reproducibility | approved dry download of HEAD requests only |
| U-010 | Terms of use of Nightshade and licenses of the downloaded assets with respect to reimplementation. | legal | read the download page and EULA; no vendor code or weights are redistributed here |
| U-011 | Whether `check_glaze_update` and `app` contact other hosts at runtime. | privacy | process capture of the app with network disabled |
