# Local MOSS-Image Training Foundation

Date: 2026-11-17

## Source and provenance

- Trainer source: `D:\Develop\nigenmame-LoRA\KohakuMossTrainer` (read-only during this work).
- Destination trainer: `D:\Develop\KohakuMossTrainer-LoRI\KohakuMossTrainer`.
- The source tree has no `.git` directory; `git rev-parse --show-toplevel` reports that it is not a repository, so no Git working-tree status can be reported.
- The source workspace's `.setup/source-provenance.json` identifies the snapshot as official commit `e60bf3260d24cea6fdd61be53893dcddad529652`. Its prior GitHub blob audit recorded 1,093 files checked, zero differences, and zero missing files. The source workspace README gives the same commit.
- No local source-code delta from that official snapshot was identified. Six additional files under `src/kohakumoss.egg-info/` were generated editable-install metadata, not trainer source, and were omitted. The trainer code and low-memory features in the official snapshot were preserved as-is.

## Low-memory capabilities preserved

The copied trainer includes its upstream MOSS-Image memory profiles and recipes, including:

- `configs/moss/vram/vram_12g.py`
- `configs/moss/recipes/recipes_lora_12g_stream.py`
- 8 GB and 6 GB profile configuration files
- Block weight streaming, activation and gradient offload, host optimizer states, gradient checkpointing, and offline latent/text cache code documented by the trainer.

These are upstream KohakuMossTrainer capabilities. This copy does not add LoRI support or modify training parameters.

## Files copied and verification

### Trainer

Copied 1,093 files (9,355,314 bytes) from the source trainer's `.github`, `configs`, `docs`, `scripts`, `src`, and `tests` trees, plus the repository root's source/readme/build files. Destination SHA-256 was checked against both the source file and the recorded provenance SHA for every official file: 1,093/1,093 matched, with zero missing or extra files.

Excluded the source `.venv`, Git metadata (not present), generated `src/kohakumoss.egg-info/`, and cache/build artifacts. The two directories named `cache` that remain are source modules and their Python tests (`src/kohakumoss_data/cache` and `tests/data/cache`), not generated cache contents. The copied `tests/data` tree contains Python test code only.

### Base model

Copied exactly the three ComfyUI-format MOSS-Image files that the trainer's `docs/guides/weights.md` lists for its default base layout. Destination SHA-256 was checked against the pre-copy source SHA-256 for each file; all three matched. The destination contains only these three model files.

| Component | Destination-relative path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| MOSS-Image Base DiT, int8 ConvRot | `models/ComfyUI/models/diffusion_models/moss_image_base_int8_convrot.safetensors` | 5,413,698,958 | `f601c1e1161915fb3fea003da0c1782e374dd960fbb3978bc0e4e3fa1f2d8e6f` |
| Qwen3.5-9B text encoder, int8 ConvRot | `models/ComfyUI/models/text_encoders/qwen3.5_9b_int8_convrot.safetensors` | 9,748,121,532 | `32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a` |
| MOSS VAE | `models/ComfyUI/models/vae/flux2-vae_small.safetensors` | 249,521,340 | `d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b` |

The trainer package includes Qwen tokenizer/config assets under `KohakuMossTrainer/src/kohakumoss/encoders/text/assets/qwen3.5-9b/` (`config.json`, `tokenizer.json.gz`, `tokenizer_config.json`, `chat_template.jinja`, license and notice). The ComfyUI weight layout does not require separate tokenizer files. No unrelated CLIP model, training data, reports, outputs, backups, or source-workspace configuration/data trees were copied.

## Available machine and Python environment

Inventory at copy time:

- Windows 11 Home, build `10.0.26200`.
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12,227 MiB, compute capability 12.0, driver 610.62.
- CPU: AMD Ryzen 9 9955HX, 16 cores / 32 logical processors.
- Physical RAM: 61.68 GiB total; approximately 41.64 GiB free at the time of inventory.
- D: drive: 953.56 GiB total; 221.73 GiB free before copying and 207.36 GiB after copying the code and three model files.
- System Python launcher exposes Python 3.12, 3.13, and 3.14. The reusable external environment is `D:\Develop\nigenmame-LoRA\KohakuMossTrainer\.venv\Scripts\python.exe` (Python 3.12.7); no virtual environment or package files were copied.

The external environment was import-checked without starting training: PyTorch `2.14.1+cu130`, torchvision `0.29.1+cu130`, CUDA runtime `13.0`, Transformers `5.20.1`, Diffusers `0.41.0`, Accelerate `1.15.0`, LyCORIS `4.1.2`, and bitsandbytes `0.50.2`. The source environment report also records Triton for Windows `3.8.0.post29` and TileLang `0.1.15`.

The external venv's editable `kohakumoss` package currently resolves to the source trainer at `D:\Develop\nigenmame-LoRA\KohakuMossTrainer\src\kohakumoss`. The dependencies are available there, but the new local trainer copy has not been installed into that environment.

## Work boundary

This was a copy and inventory only. No training was run, no training parameters were generated or changed, and no source files were modified.
