# Task 3 independent review

Reviewed range: `46d84ada69638b2a7eb78450eb1663b4019c772a..486dfa2b88abcac71b68c3b36fdbf12d2e916ac4`.

Spec compliance: PASS for Task 3.
Code quality: PASS; no actionable Critical, Important, or Minor findings identified.

## Scope and evidence

Read the Task 3 brief, implementation report, and complete review-package diff. Focused owning-source checks covered LoRIModule constructor validation, raw-state/artifact validation, Target registration, adapter binding, and parameter placement. Verified the inner repository HEAD equals the reviewed head and its working tree is clean. No source, index, HEAD, environment, or training changes were made. This review report is the only written artifact.

The implementation report records 157 passing relevant regression tests, RED/GREEN evidence, black/ruff/ASCII checks, and clean diff checks. Those tests were not rerun, per review instructions. This is an independent code review, not an independent repeat of the reported test results.

## Contract assessment

- Existing `apply` and `build_targets` calls remain valid; `initial_state` is optional. LoRI options are consumed locally and never forwarded to the discovery network. Ordinary LoRA module kwargs retain their existing behavior.
- Standard LoRA preset discovery supplies native names and operators. LoRIModule replacement occurs before weak binding and Target construction (`src/kohakumoss/adapters/build/network.py:123`, `:131`). The prior Task 1 registry already provides the LoRI kind, so leaving kinds/__init__.py unchanged is consistent with the brief.
- Unsupported DoRA, DORA_DETACH, input/rank/module dropout, TRAIN_NORMS and non-unit LoRA+ settings are refused by config validation (`src/kohakumoss/adapters/build/config.py:131`). LoRI + Muon is refused at the trainer knob boundary (`src/kohakumoss/training/knobs.py:254`). The small additional knobs.py edit is necessary because its static algorithm list otherwise rejects LoRI before construction.
- Explicit `initial_state` wins over mask_path without opening the stale path. When only mask_path is used for sparse construction, one `read_lori_state` call enforces the existing artifact version/phase/digest/count validation, and metadata target dimensions are compared to discovered targets (`src/kohakumoss/adapters/build/network.py:139`, `:175`).
- Supplied state target names must exactly match discovery. Raw state is checked for valid fields, rank/input shape, finite tensors, dense B output shape, sparse values/index count, sorted unique int64 indices, actual-target index bounds, and phase. Fixed A and alpha are initialized from supplied tensors; B starts at zero as required. Trained B is subsequently copied by the existing loader into the established Parameter.
- Final dense/compact Parameters are created by `_lori_module` before `_place` registers the module (`src/kohakumoss/adapters/build/network.py:162`; `src/kohakumoss/adapters/build/apply.py:64`). Thus the factory returns final Parameters before downstream engine exclusion/optimizer capture. No restore-time resizing was introduced.
- The direct parameter-group regression checks only compact values, LR, no Muon flag, zero default decay and explicit `rules="all"` nonzero decay. The AdamW comparison checks the meaningful compact/dense update equivalence.

## Boundaries and follow-up gates

These are documented interface limits and subsequent-task requirements, not Task 3 defects:

1. Raw sparse tensors do not encode the original output dimension. A supplies rank/input width; legal active offsets establish only a lower bound on output size. Current construction correctly rejects offsets outside the actual target, but cannot prove the original full output size from raw state alone. Strict mask artifacts carry and check the full shape. Task 4 must retain and compare checkpoint metadata target shapes before registration/capture when resuming from raw checkpoint tensors; it must not treat legal offsets as proof of complete shape identity.
2. `initial_state` prepares immutable tensors and final Parameter shapes, while B deliberately remains zero at construction. Task 4 must invoke the established state loader to restore trained B in place, and route resume construction through checkpoint state rather than an old mask_path. Task 4 owns native checkpoint metadata and restore wiring.
3. Actual MOSS target matching and real training/VRAM measurements belong to Task 5. Model-free target discovery tests do not establish those results.
4. The separate doctor/CUDA toolchain failure is outside this review's proven attribution. The report records the stdalign.h compile failure and log; no conclusion is made here about whether it predates this change or its environment ownership. Its independent diagnosis remains a separate gate.

## Findings

Critical: none.
Important: none.
Minor: none.
