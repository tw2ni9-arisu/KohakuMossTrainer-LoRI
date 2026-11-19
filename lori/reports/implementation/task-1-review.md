# Task 1 independent review

Base: `7a5b94a`. Reviewed head: `f4a2c65a65a3dee7cedb3f1bf5d1fdc630737164`.

Spec verdict: **Compliant within Task 1 scope**.

Code quality verdict: **Approved**. No blocking findings.

## Scope and evidence

Read the complete supplied review-package diff once, the Task 1 brief and implementation report. Inspected the unchanged owning contracts in `kinds/record.py`, `kinds/common.py`, `bind/eager.py`, and `build/target.py` specifically to check registry lookup, multiplier ownership, weak layer access, and the module-forward signature. Read the inner `CLAUDE.md` conventions.

This review did not change the inner code tree, HEAD, or index; run training; load model weights; inspect source dependency code; or rerun the reported test suite. The implementation report records 78 passing focused/regression tests, successful black/ruff and diff checks, and a clean post-commit status. Those execution results are supplied evidence, not independently reproduced results.

## Satisfied contracts and strengths

- `lori.py:40-66`: A is initialized with the stated shape and uniform range, or preserved by detached clone from a supplied projection. A and alpha are persistent buffers. Calibration owns only `b_dense`; sparse owns only compact `b_values` plus persistent integer indices. Neither A nor indices is an optimizer Parameter.
- `lori.py:54-66`: Sparse offsets must be one-dimensional, sorted, unique, in bounds, and int64. Empty and full index sets are accepted. Cloning supplied A and indices avoids storage aliasing with construction inputs.
- `lori.py:73-92`: Flattened scatter reconstructs exactly B[out, rank], preserving gradients to the selected values. The empty index case still has a differentiable path to the empty Parameter; the mathematical delta and input gradient are zero. Dense calibration exposes the same B representation without scatter.
- `lori.py:83-92`: Nested linear operations compute `(alpha/rank) * multiplier * B(Ax)`. Materialized delta computes the same scaled `B @ A`. The adapter applies scale once; the unchanged `unit_diff` requests multiplier 1 and `delta_builder` applies the external multiplier once. No double multiplication occurs through Target.
- `lori.py:95-122`: Registration matches the actual Kind protocol, including exact module-class lookup, `module_forward(t, ctx)`, `unit_diff(t)`, and additive stacking. The bypass delegates to the existing weak-layer `module_bypass` contract and adds the base output/bias once. The three ordinary LoRA Comfy tensors reproduce the same delta, with alpha retained separately from unscaled factors.
- `tests/adapters/test_lori_math.py`: Tests cover dense and sparse factor math, empty/partial/full masks, input/value gradients, multipliers 0/0.5/1, initial zeros, buffer persistence keys, supplied A preservation, integer dtype retention, invalid offsets, real nn.Linear/Target integration, and Comfy scale equivalence. They use direct adapter mathematics and real nn.Linear operators rather than a fake MOSS model. A deliberately double-scaled result is rejected.
- The implementation is small, phase dispatch is selected once at construction, and it introduces no custom kernels, dependencies, source-environment changes, or unrelated refactors. Changed Python shown in the package is ASCII and matches local style.

## Findings

### Critical

None.

### Important

None.

### Minor

None requiring a change for Task 1.

## Cannot verify here / subsequent-task requirements

These are scope boundaries, not Task 1 defects or approval blockers:

- Registry registration alone does not establish production factory construction. Task 3 must instantiate the local LoRIModule, preserve A across phases, select supported linear targets, and reject unsupported options rather than relying on the existing external LoRA builder.
- Global absolute-B quantile calibration, threshold ties, phase transition and zeroing sparse B are Task 2 responsibilities.
- Matching native state keys are present, but production restore must construct the final phase/indices/Parameter shapes before optimizer, offload, or EMA capture and restore without shape replacement. That Task 4 path was not exercised here.
- The adapter physically stores only nnz trainable values in sparse mode, but reconstructs a dense B transiently. Optimizer-state allocation, offload compatibility, GPU peak memory, throughput, and real MOSS training behavior require their later scoped checks. The current review establishes no total-VRAM or speed claim.
- GPU assertions in the reported tests are conditional. The report does not establish that a CUDA device was available for those assertions; real-device integration remains to be verified later.

Task 1 may pass its review gate and proceed to the subsequent scoped tasks.

## Focused follow-up: CUDA scalar synchronization

Rechecked `lori.py:70-73` and `lori.py:88-92` for per-layer host synchronization. `scale` returns the Tensor expression `self.alpha / self.lora_dim`, and bypass multiplies that Tensor by the Python multiplier and the output. Neither path calls `.item()`, `float(tensor)`, tensor truth conversion, or a host transfer. When alpha is on CUDA, these expressions stay on the device; no explicit device-to-host scalar synchronization is introduced per forward. The property does perform a small Tensor division each call, but that is device arithmetic rather than a host scalar read and does not require a Task 1 fix. Constructor-only index validation uses tensor truth checks; these are outside the per-forward path. This is a static source conclusion, not a CUDA profiler measurement. Approval and findings remain unchanged.
