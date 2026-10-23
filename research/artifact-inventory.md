# Artifact inventory

| Item | Value | Evidence |
|---|---|---|
| Application | Nightshade 1.1 (GUI banner "Welcome to Nightshade 1.1!", title "Nightshade - Proactive Copyright Protection"; output tag `V1`) | E-020, E-043 |
| Platform | Windows x64, PE32+ GUI, unsigned | E-001 |
| Installed at | `C:\Users\Adam\Downloads\Nightshade-1.1-Windows\Nightshade` (unmodified; 16,096 files, 7.85 GB) | file listing |
| Launcher | `Nightshade.exe`, 63,380,224 B, SHA-256 `07723B4C...D6F9`, PyInstaller bootloader plus CArchive | E-001, E-002 |
| Runtime | CPython 3.11 (`python311.dll`), `_internal/` one-dir layout | E-003 |
| GUI | PyQt6, layout from `_internal/nightshade.ui` (3,868 B) | file listing, E-020 |
| App code | Python bytecode in `PYZ.pyz`: `nightshade`, `opt`, `utils`, `downloader`; script `app` | E-005 |
| Source / symbols / debug info | None shipped as source. Bytecode retains all names, so function-level recovery is possible. No `.pdb`. | E-004 |
| Obfuscation | None found for code. One proprietary asset (`full_asset3.p`) is Fernet-encrypted with a key embedded in `utils`. | E-004, E-011 |
| ML stack | torch 2.8.0+cu128, torchvision 0.23.0+cu128, diffusers 0.14.0, transformers 4.29.0, openai-clip 1.0.1, lpips 0.1.4, nltk 3.8 | E-006 |
| GPU components | CUDA 12.x runtime DLLs in `torch/lib` (cublas, cublasLt, cudnn, cufft, curand, cusolver, cusparse, nvrtc, nvJitLink), `torch_cuda.dll` 1.03 GB | file listing |
| Native extensions | 216 `.pyd` (PyQt6, numpy, scipy, sklearn, PIL, cryptography, etc.); none are Nightshade-specific | file listing |
| Model weights shipped | None | E-007 |
| Model assets downloaded at first run | 9 resources into `%USERPROFILE%\.nightshade` | E-010 |
| Model cache layout | `<root>/base/base/{unet,text_encoder,vae}`, `<root>/{clip_model.p, lpips_fn.p, full_asset3.p, blip_model.pt, blip_preprocessor2.pt, bpe_simple_vocab_16e6.txt.gz}`, `<root>/tmp`, `<root>/saved_params.p` | E-010, E-046, E-020 |
| License notices | Third-party dist-info licenses present (click, cryptography, diffusers, ...). No Nightshade EULA file found in the install directory. Not verified against the download page terms. | file listing (U-010) |
| Update / network behaviour | Version check against `glaze.cs.uchicago.edu`; resource download over plain HTTP with MD5 checks | E-010, E-013 |

Counts by extension in `_internal`: `.h` 9160, `.py` 4623, `.pyi` 221, `.pyd` 216, `.msg` 143, `.dll` 92, `.qm` 89, `.gz` 83, `.enc` 80, `.tcl` 69.

## Tooling note (REA)

REA tools are advertised in this session (native Hopper/Ghidra bridge, JS/Electron, Android, managed .NET, firmware, browser,
process capture). None of them targets PyInstaller or CPython bytecode, and native decompilation of the bootloader would add
nothing, because the application logic is recoverable from higher-level artifacts (rules 3 and 4 of the mission). REA was
therefore not used for analysis. The extraction and disassembly used `PyInstaller.archive.readers`, `marshal` and `dis`
(`research/tools/`). If native components need analysis later (for example a custom `.pyd`), REA `open_binary` is the entry point.
