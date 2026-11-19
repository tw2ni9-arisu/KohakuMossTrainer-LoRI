# MOSS-Image DiT LoRI Design

Date: 2026-11-17. Scope: paper-correct LoRI, compact trainable storage, one bounded local training experiment, paired base/adapter comparison. No training was performed in this design task.

## Workspace

Implement in `D:/Develop/KohakuMossTrainer-LoRI/KohakuMossTrainer`. Source trainer/model/data under `D:/Develop/nigenmame-LoRA` remain read-only. All caches, manifests, machine configurations and outputs belong under the current workspace and stay gitignored. Future git covers inner source only; retain licensing/naming. Follow CLAUDE.md: ASCII Python, what-comments, black/ruff, no fake-model tests. Adapter math/pure functions can be model-free; full integration uses actual MOSS weights.

Setup verified official commit `e60bf3260d24cea6fdd61be53893dcddad529652`, 1093 matching official files and six extra egg-info files: no established private optimization patch. Source has no .git. Reported hardware: RTX 5070 Ti Laptop 12227 MiB, sm120; 61.68 GiB RAM. External Python environment is source `.venv/Scripts/python.exe` (3.12.7, torch 2.14.1+cu130, LyCORIS 4.1.2); copied src must take import precedence. Copied model root: workspace `models/ComfyUI/models`.

## Algorithm

Research verified [LoRI, arXiv:2504.07448](https://arxiv.org/abs/2504.07448), COLM 2025, against official code. Use research report for the precise repository pointer.

- Paper orientation: W[in,out], A[in,r], B[r,out], xW+xA(B*M). PyTorch orientation: A[r,in], B[out,r], delta_W=scale*(B*M)@A.
- Initialize A ONCE Kaiming uniform in [-sqrt(3/in),sqrt(3/in)], freeze A; B=0.
- Calibration trains dense B on the normal image flow-matching objective, with no mask. This is not gradient/Fisher/SNIP ranking.
- Concatenate absolute B across ALL selected layers/projections, compute ONE global sparsity s quantile tau. M=abs(B)>=tau. Ties can retain more than the target fraction; record actual nnz. No per-layer top-k or arbitrary exact-count tie breaker.
- Preserve exact A tensors, reset B=0, fix mask and start sparse training with fresh optimizer/scheduler state. Official two-process scripts do not clearly guarantee A reuse; explicit preservation follows paper Algorithm 1.
- Paper equation has no explicit alpha/r; official PEFT code uses alpha/r, reported alpha=2r. Reuse DIM/ALPHA convention, record scale and apply once.
- Author code uses dense masked B; compact storage is our optimizer-state optimization. Dense scratch/activations remain; no proportional speed or total-VRAM claim.

## Verified seams and mechanisms

Paths below are relative to inner source repository.

| File | Existing behavior |
| --- | --- |
| src/kohakumoss/adapters/build/config.py | AdapterConfig/config_from_knobs; constructor introspection validates ALGO_KWARGS. |
| adapters/kinds/record.py, kinds/__init__.py | Kind registry by algo/class; module_forward, diff_weight, comfy_tensors hooks. |
| adapters/build/network.py | build_targets uses LycorisNetwork preset discovery then creates Target. |
| adapters/build/apply.py | _place registers adapter as linear.lyco and moves dtype/device. |
| adapters/train/adapter_set.py | Trainables come from requires_grad; native state includes buffers; groups/save/restore/export/merge exist. |
| training/build/model.py | Host load -> memory placement -> build_trainable -> parameter dtype -> engines.exclude_params -> bind_ops -> encoders. |
| training/build/loop.py | build_step uses existing image loss; build_optim consumes final groups. |
| training/entry.py | Optimizer currently builds BEFORE checkpoint restoration. Compact shapes must initialize earlier. |
| offload/weights/layers.py | module_tensors enumerates parameters and buffers; inspect actual adapter placement. |
| src/kohakumoss_data/config.py | Explicit CACHE_DIR controls tensor vault and catalog; default source/cache is forbidden here. |
| scripts/data/cache_latents.py, cache_text.py | Existing separate precomputation. |
| scripts/sample/compare_grid.py | Base plus adapter grid, singles and metadata with matched sampler/prompts/seeds. |

Abbreviated adapters/training/offload paths are under src/kohakumoss.
Verified 12GB profile is `ckpt-ring4-actoff-optstream-gradoff`: checkpointing, H2D weight streaming/four resident blocks, activation/gradient offload and host AdamW state. Text supports layer ring/encode-ahead; VAE offload exists. ConvRot INT8 frozen model files exist. Do not add foreign block-swap APIs. FP8 exists but is outside first experiment.

## Minimal implementation

Keep existing preset discovery. For ALGO=lori, discover standard LoRA targets (LoRI kwargs must not leak to third-party factory), immediately replace temporary LoCon modules with local LoRIModule BEFORE creating Target. Retain path/name/linear and ordinary weak reference binding. Do not fork installed LyCORIS or introduce another injection engine.

Create `adapters/kinds/lori.py`: LoRIModule plus Kind. Public config accepts phase calibration/sparse, sparsity and mask_path alongside lora_name/org_module/lora_dim/alpha. Builder reads initialization state once, passes per-layer tensors through private factory parameters. Maintain attributes consumed by Target/export/bypass. Initially support selected Linear/MossLinear; reject unsupported modules and DoRA/norm training/nonzero dropout. First experiment uses AdamW, no LoRA+ or Muon.

Calibration: persistent A[r,in] and alpha buffers, dense trainable b_dense[out,r]. Sparse: persistent A and alpha buffers, sorted unique int64 active_indices[nnz], ONLY trainable b_values[nnz]. Reconstruct ephemeral B by differentiable scatter for nested F.linear; this gives compact parameter/moment storage without custom kernels or token-by-nnz intermediates. Kind.module_forward reuses base-plus-bypass; diff_weight materializes scale*B@A only for explicit export. comfy_tensors expands to ordinary LoRA up/down/alpha. Native export remains compact, portable Comfy export intentionally grows.

WD_RULES=moss exempts 1-D values from decay. Choose decay=0 initially; if nonzero decay is requested, WD_RULES=all must preserve dense/compact AdamW equivalence.

## Lifecycle and artifacts

Three separate operations: calibration trainer process -> CPU mask extractor -> sparse trainer process. NEVER replace Parameter after optimizer/gradient ring/EMA capture. Sparse modules exist with final shapes before engines.exclude_params/build_optim. No dense calibration moments transfer.

Create `adapters/train/lori.py` helpers:

- read_lori_state(path) -> (dict[str,Tensor], dict[str,str]).
- make_sparse_init(calibration_state,sparsity) -> (dict[str,Tensor],dict[str,str]): float32 CPU global quantile, >= ties, preserved A, zero values.
- lori_layer_init(state,lora_name) -> dict[str,Tensor]: validate phase/version/rank/shape/indices and select layer state.

Initialization artifact contains exact A, indices, zero values, alpha, module keys/shapes/rank/scale/seed, algorithm version, calibration digest, threshold/sparsity and actual per-layer/total nnz. Reject nonfinite scores, wrong target sets/shapes/rank and duplicate/out-of-range indices. Individual empty masks are legal; test empty parameters with host optimizer and omit empty trainables if necessary.

Resume/native inference must not require the original mask_path. Add optional initial_state plumbing to local apply/build_targets. Read checkpoint trainable tensors early for LoRI; instantiate final parameter shapes before optimizer capture, then existing parts.restore copies state in place. Do not resize Parameters during later load_state_dict or load the entire optimizer twice to learn indices. AdapterStack._raw_tensors supplies loaded raw state to construction; _build recognizes native LoRI before third-party get_module. Reload must work after deleting/moving initialization artifact.

A/indices are persistent buffers, never optimizer members. Integer indices retain dtype on moves. Verify groups, gradient staging and forward reference live b_values; moment/staging sizes scale with nnz. engines.exclude_params may keep adapters resident: measure actual behavior, do not claim all buffers stream.

## Dataset and first experiment

Read-only inventory: `D:/Develop/nigenmame-LoRA/dataset-raw/round2-caption`: 285 images (279 PNG/6 JPG), 285 nonempty UTF-8 prose captions; 273 portrait/12 landscape, mostly ~1MP. Most frequent size 704x1440 (28); captions begin nigenmame. Natural prose must not be shuffled as tags.

Workspace JSONL selects 12 images deterministically: sort names by orientation, take 10 portrait/2 landscape at evenly spaced indices including endpoints. Each row uses absolute original image and original caption as caption_nl. Save selection/dimensions/image+caption hashes in inventory JSON. JsonlSource supports absolute paths: no copies/symlinks required. CAPTION={template:nl}, CAPTION_VARIANTS=1, TEXT_DROP=0, FLIP=False, center resize. Explicit workspace CACHE_DIR even when caches off.

Initial engineering smoke recipe: 256 or 384 target area, batch=1/count batching/accumulation=1, rank8/alpha16, sparsity=.9, AdamW LR1e-4/decay0, constant schedule, seed42, bf16, INT8 base, verified 12GB profile. Execution owner chooses exact values based on resource checks and records resolved config. Calibration covers selected records once (~12 steps), then ~32 sparse steps; do not impose language-paper 1000 calibration steps. Cache latents/text separately and train both phases offline. Disable periodic sampling; save final checkpoint/native states.

## Acceptance

Math/model-free: zero-init equals base; compact outputs/input gradients/active gradients match dense masked reference; A frozen; outside-mask B zero; global quantile differs from per-layer fixture; >= tie fixture; scale once; deterministic same-state construction; empty/all-selected masks.

State: exact native roundtrip and resume without mask file; shapes before optimizer; native/converted LoRA/merged outputs agree at multipliers0/.5/1; quantized base uses bypass without mutation; int buffers survive dtype moves. Existing LoRA regressions pass.

Real MOSS integration: bounded resident and weight-ring/checkpoint runs with CPU optimizer and gradient offload; AdamW moment total2*nnz excluding step scalars; A/indices have no states; gradient staging uses nnz; forward sees same live Parameter ids. Test recomputation/resume with actual MOSS weights per CLAUDE.md.

Monitor cache/startup/calibration/train/sample peaks separately, finite losses, first completed step, exit/stall, allocated/reserved GPU peaks, nvidia-smi usage, RAM/time and artifacts. Luna monitors background training; primary sleeps until error/completion. OOM/nonfinite/stall returns evidence before adjustments. No unmeasured 6/8GB claim.

Compare with existing compare_grid: identical base/precision/resolution/sampler/CFG/shift/prompts/seeds; multiplier0/base vs trained multiplier1. Four nonexplicit prompts, matched with/without nigenmame, seeds42/43 as budget allows. Save grid/singles/metadata/config/checkpoint hashes and inspect/show images. Success means finite training, changed active B, frozen A/base, exact reload, nonzero paired effect. A weak effect is reported honestly; short-run difference does not prove quality/generalization/LoRA parity.
