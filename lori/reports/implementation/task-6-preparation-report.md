# Task 6 preparation report

Update 2026-11-18: Task 5 full12g gate has passed at `285f4bd`. Private implementation and final native resolve PASS are now available under `experiment/task6/session-20261118-a/`; its README documents exact commands and monitor protocol. No catalog/cache/training has started. The original preparation findings below are retained as historical proposals; the session's resolved JSON files record actual choices (including cache TOKEN_BUDGET estimate before text lengths exist, training exact, NUM_WORKERS0/DATA_SCAN_WORKERS1).

Status: documentation-only preparation, 2026-11-17. Task 5 has not passed its final gate. No Task 6 configuration, code, cache, or training process has been created or launched. All recipes and commands below are proposals for later implementation.

## Scope and existing inputs

- Workspace: `D:/Develop/KohakuMossTrainer-LoRI`; copied trainer: `KohakuMossTrainer/` beneath it. Run copied source with the read-only source-venv interpreter, not the source trainer.
- Existing manifest: `experiment/data/round2-caption-12/dataset.jsonl`, with 12 records (10 portrait, 2 landscape). Reuse its existing inventory/source snapshot; do not select again, copy source images, or change captions.
- Source image paths remain absolute and read-only. Set the JSONL entry's `root` to the workspace-owned manifest directory, and set an explicit workspace-owned `CACHE_DIR` for catalog and tensor vaults. Do not use the source dataset as a cache destination.
- Local weights already exist under `models/ComfyUI/models/`: DiT `diffusion_models/moss_image_base_int8_convrot.safetensors` (5,413,698,958 bytes); TE `text_encoders/qwen3.5_9b_int8_convrot.safetensors` (9,748,121,532 bytes); VAE `vae/flux2-vae_small.safetensors` (249,521,340 bytes).
- The copied source packages Comfy TE config/tokenizer assets; the compressed tokenizer is expanded in memory. No mirror download or source-venv modification is required by this route.
- External interpreter exists at `D:/Develop/nigenmame-LoRA/KohakuMossTrainer/.venv/Scripts/python.exe`. Its bundled Triton TinyCC exists at `.venv/Lib/site-packages/triton/runtime/tcc/tcc.exe`.

## CLI and knob distinctions

The installed console entry point is `kogine = kohakuengine.cli:main`. To bind execution explicitly to the external interpreter, use `python -m kohakuengine.cli run SCRIPT --config CONFIG`. There is no `kohakuengine/__main__.py`; do not use `python -m kohakuengine`.

`scripts/train/moss.py`, `scripts/data/cache_text.py`, and `scripts/data/cache_latents.py` consume KohakuEngine Python configs. Config dictionaries replace whole dictionaries when composed with `use_config`, so each override must retain the required fields.

The train script's `TEXT_ENCODER` knob is a weights path. The text-cache script's same-named knob is a dotted factory. Therefore a dedicated `data.py` should inherit the common train recipe and then set `TEXT_ENCODER = "kohakumoss.encoders.text_encoder"`, with the actual local TE file passed through `TEXT_ENCODER_KWARGS["path"]`. Do not reuse the factory string as the training weights path.

The text-cache kwargs helper inherits only training `TEXT_ENCODE` compute fields (`dtype`, `precision`, `quant_group`, `fp8_recipe`). It does not inherit ring placement, resident slots, batch size, or token cap. Set these explicitly in the data config's `TEXT_ENCODER_KWARGS["plan"]`:

- `memory="ring"`, `stream_resident=2`, `batch=1`, `max_tokens=4096`;
- `dtype="bf16"`, `precision="auto"`, matching the training recipe's compute settings and the local INT8 stored layout;
- `device="cuda"` and the explicit workspace-local TE file path.

Set `ENCODE_BATCH=1` for both cache passes. The text script defaults to 512 captions and the latent script to 8 images. The VAE factory should receive the explicit local VAE file and `mode="mode"`; `VAE_DTYPE="bf16"` and `VAE_ENCODE` flow through the latent helper. Use the same VAE placement/compute settings as the common recipe.

## Proposed private files and shared recipe

Keep private implementation under ignored `experiment/task6/`: `common.py`, `data.py`, `calibration.py`, `sparse.py`, a launcher, and an audit helper/report. Keep model/tensor/kernel/temp/output caches inside the workspace. These files do not exist as a result of this preparation task.

The common recipe would compose the shipped `recipes_lora_12g_stream.py`, then override the following requirements. These are recorded choices for later implementation, not an active configuration:

| Area | Proposed settings |
| --- | --- |
| Weights/data | `WEIGHTS=<workspace>/models/ComfyUI/models`; JSONL entry points to the existing manifest and workspace manifest root; `CACHE_DIR=<workspace>/experiment/task6/cache-256` |
| Caption | `CAPTION={"template":"nl"}`, `CAPTION_VARIANTS=1`, `TEXT_DROP=0.0`, `SEED=42`; preserve manifest `caption_nl` |
| Geometry | `RESOLUTION_MODE="resize"`, `TARGET_AREA=256*256`, `SIZE_STEP=16`, `ASPECT_RANGE=(0.25,4.0)`, `UPSCALE="never"`, `CROP="center"`, `FLIP=False`, `BACKGROUND="white"`, `RESAMPLE="lanczos"`, `LATENT_DRAW="mode"` |
| Caches/compute | Both caches `"offline"`; cache storage, VAE and training compute `"bf16"`; explicit identical text compute settings in cache and train; `BASE_PRECISION="int8"` |
| Batching | `BATCHING="count"`, `BATCH_SIZE=1`, `GRAD_ACC=1`, `MAX_TOKENS=6144`, `TOKEN_BUDGET="exact"`, `OVERLONG="drop"`; verify all 12 samples fit rather than silently dropping/truncating |
| Adapter | `ALGO="lori"`, `PRESET="attn-mlp"`, `DIM=8`, `ALPHA=16.0`; input/rank/module dropout zero; no DoRA or train norms |
| Optimizer | `OPTIMIZER="adamw"`, `LR=1e-4`, `WEIGHT_DECAY=0.0`, `SCHEDULE={"mode":"constant","warmup":0,"end":-1}`, `EMA=0.0`; otherwise retain native defaults |
| Memory | `MEMORY_PROFILE="ckpt-ring4-actoff-optstream-gradoff"`; do not mix manual memory knobs into the named profile |
| Runtime/artifacts | `COMPILE="off"`, `TUNE="off"`, empty sample prompts, sampling/validation off; periodic saves off, `FINAL_CHECKPOINT=True`, `SAVE_FORMAT="both"`, merged export off; separate output directories and `RESUME=""` for the two phases |

The 12g profile resolves to block checkpointing, four resident ring blocks, activation offload, streamed host optimizer state, **`fused_step=True`**, and gradient offload. A generic non-fused optimizer/offload gate is insufficient: Task 5's final evidence must cover this actual path before it is adopted. The parent has asked the Task 5 owner to verify that coverage.

Calibration overrides: `STEPS=12`, `ALGO_KWARGS={"phase":"calibration","sparsity":0.9}`. Sparse overrides: `STEPS=32`, `ALGO_KWARGS={"phase":"sparse","sparsity":0.9,"mask_path":<sparse-init>}`. Sparse starts in a new trainer/optimizer process with `RESUME=""`; never restore calibration optimizer moments into it.

## Exact command route for later execution

In the copied trainer working directory, with `$py` set to the external interpreter and the config variables set to their absolute private paths:

```powershell
& $py -m kohakuengine.cli run scripts/data/catalog_build.py --config $data
& $py -m kohakuengine.cli run scripts/data/cache_text.py --config $data
& $py -m kohakuengine.cli run scripts/data/cache_latents.py --config $data
```

Audit complete offline coverage before starting calibration. Root must arrange Luna monitoring before an actual training launch and sleep during the run:

```powershell
& $py -m kohakuengine.cli run scripts/train/moss.py --config $calibration
```

After calibration and its assertions pass, extract the mask on CPU using the **native** checkpoint, not a portable LoRA export:

```powershell
& $py scripts/adapters/lori_mask.py `
  '<calibration-output>/checkpoints/step_00000012/trainable.safetensors' `
  '<workspace>/experiment/task6/sparse-init.safetensors' --sparsity 0.9
```

Validate exact A reuse, global abs-B quantile threshold, `>=` ties, retained counts and zero compact B before launching the independent sparse phase with Luna ready:

```powershell
& $py -m kohakuengine.cli run scripts/train/moss.py --config $sparse
```

The private launcher must apply the compiler report's process-local `CC` override before Python starts, preserve normal CXX setup, use `PYTHONDONTWRITEBYTECODE=1`, and point `PYTHONPATH` to copied `src` plus workspace `.tools/python`. Give every Triton/kernel/Inductor/CUDA/temp/cache/output location a workspace-owned path. Record the resolved environment and knobs. Do not change system environment or source venv, terminate unrelated processes, or start duplicate runs after observation timeouts.

## Cache completeness and provenance assertions

File existence alone is insufficient. Both cache passes must use the same common recipe so caption rendering, variants, seed, resize geometry, decode settings, encoder identity/spec and storage match training.

1. Resolve `CACHE_DIR`, catalog destination, tensor vaults, manifest root and all runtime caches; assert containment in the workspace before any constructor that can create a cache.
2. Record manifest/model hashes and resolved knobs; compare `latents.settings.json` and `text.settings.json` with the training-derived settings, including stored-layout text compute spec. Reuse the existing source inventory; do not expose caption bodies in reports.
3. Enumerate native `latent_items(dataset)` and check every requested key is present/readable with finite, valid-shaped tensors. Flip off implies 12 image/geometry entries for this manifest.
4. Enumerate native `caption_set(dataset)` and check every deduplicated caption plus the empty prompt; at most 13 keys for 12 records and one NL variant. Check tensor shape/dtype/finiteness and exact cached lengths.
5. Traverse every epoch-0 batch under the final offline training config; assert 12 samples, count batch1, no dropped samples, no truncation, and no offline misses. Native offline lookup raises on misses rather than encoding them.

Both native cache scripts skip existing keys and record settings; a successful rerun alone does not prove correct provenance or all training keys. These assertions have not yet been executed because cache creation is not authorized before the Task 5 gate.

## Resolution recommendation and remaining gates

Read-only resource snapshot during preparation: RTX 5070 Ti Laptop GPU, 12,227 MiB total / 1,878 MiB used / 10,067 MiB free; host physical memory free 43,120,188 KiB (about 41.1 GiB). This is transient and must be measured again immediately before launch.

Recommend 256 for the first bounded correctness run. The 384 target has approximately 2.25 times the image-token area; it is a later quality comparison, not needed to establish the two-phase state contract. Neither resolution has been proven feasible by this preparation report.

Outstanding: Task 5 resume root cause and final verified gate, coverage of the 12g fused optimizer path, and pre-launch resource checks. No missing model/download was identified by static inspection. Task 6 recipes, caches, resolved specs and real-image training results remain uncreated/unverified until explicit follow-up.
