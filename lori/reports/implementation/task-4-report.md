# Task 4 implementation report

Status: implemented and committed in the copied inner repository.
Commit: `623629b39ef01be23b6d28ec2aad95a7ddb35c73`.
Baseline: `486dfa2b88abcac71b68c3b36fdbf12d2e916ac4`.

## Restore and initialization

- `apply` and `build_targets` accept optional `initial_metadata` alongside the existing optional raw `initial_state`. Existing callers remain valid.
- Extracted the Task 2 validator into `validate_lori_state(state, metadata)`; `read_lori_state` delegates to it. There is one validation implementation, including version, full target shape, rank, counts, phase and exact A/alpha/index digests.
- Factory propagates validated metadata target shapes and compares each full [out,in] shape against the actual discovered target before `_place`. Tests include a larger metadata output dimension with indices still legal, with an explicit placement rejection sentinel.
- Fresh sparse `apply` reads the mask artifact once and retains its versioned metadata. Explicit checkpoint initialization takes precedence over a stale mask path.
- Training resolves the resume directory once before model construction, reads only `trainable.safetensors` through `load_lori_initialization`, and forwards raw tensors and metadata through `build_model` / `build_trainable`. Final compact Parameters exist before engine trainable exclusion, optimizer construction, EMA capture and parallel hooks.
- Regular `parts.restore(ckpt.load(...))` still restores all values and moments normally. Early initialization never opens optimizer state; the ordinary checkpoint load reads it once. Trainable tensors are reread at ordinary restore, rather than retaining an unnecessary full checkpoint/optimizer copy during model construction.
- `load_state_dict` prechecks every shape before any copy, rejecting broadcasting and partial updates. It copies into existing Parameters; no trained B Parameter is replaced or resized after capture.

## Native metadata, precision, inference and export

- Sparse state metadata reuses the strictly validated initialization metadata and changes phase to `sparse`. Full target shape and frozen A/alpha/index digests remain embedded. Native restore and inference work after the original artifact file is deleted.
- LoRI placement moves frozen buffers to the selected device while preserving their original dtype and exact values across master dtype selection. Trainable B still uses the requested master dtype.
- Native `lycoris` export for LoRI writes self-contained local raw state with `kohakumoss_format=kohakumoss_lori`, its saved spec/fold and versioned LoRI metadata. It does not advertise `ss_network_module=lycoris.kohya` compatibility. Native tensors retain their existing dtype regardless of requested export dtype; Comfy exports retain the ordinary export dtype convention.
- When native export receives row scales, it unfolds compact B values using the selected flattened B row offsets (dense calibration B uses its rows), preserves A/alpha/index bytes, and marks the output fold as null. No dense B allocation is needed for this native export conversion.
- Inference passes loaded raw tensors and metadata into early construction. `_build` recognizes LoRI fields before third-party `get_module`, constructs the local LoRIModule and copies trained values. Raw sparse inference no longer reads the recorded mask path.
- LoRI does not read the base for its additive delta; the raw inference source-base helper avoids unnecessary base materialization for this kind.

## TDD and verification

All Python invocations used the external existing venv interpreter with copied `src` and outer `.tools/python` on PYTHONPATH and PYTHONDONTWRITEBYTECODE=1. No dependency installation or source-venv edits occurred. GPU-capable regression processes selected the existing bundled TCC through process-local CC, with private workspace Triton/Inductor/kernel/CUDA/temp caches. CXX and system configuration were not changed.

RED before production changes: `pytest tests/adapters/test_lori_state.py -q` produced 11 failures / 3 passes. The first fixture attempt additionally hit an existing fold-record model contract; after changing the fixture to stub the pure metadata function, the meaningful RED remained 11 failed / 3 passed: missing native sparse metadata and unsupported third-party LoRI inference. No stand-in model classes were introduced: tests use real nn.Linear operators and adapter tensors.

An intermediate AdamW continuation check exposed shared moment tensors in the test's in-memory state transfer; deep-copying optimizer state reproduces independent disk restore and gives bit-exact continuation.

Final current-tree regression:

```
python -m pytest tests/adapters/test_lori_state.py tests/adapters/test_lori_build.py tests/adapters/test_lori_calibration.py tests/adapters/test_lori_math.py tests/adapters/test_state_refusal.py tests/adapters/test_stacking.py tests/adapters/test_stored_bypass.py tests/adapters/test_fused_qkv.py tests/training/checkpoint/test_checkpoint_meta.py tests/training/checkpoint/test_checkpoint_stream.py tests/training/test_checkpoint_safe_load.py tests/training/test_final_checkpoint.py tests/training/knobs/test_knob_rules.py -q -p no:cacheprovider
```

Result: **194 passed in 45.90s**, exit 0.

The earlier broader regression, before the final compact native unfolding branch and shape-broadcast regression were added, ran all `tests/adapters`, the checkpoint safe/stream/meta/final cases and all `tests/training/knobs`: **345 passed in 240.30s**, exit 0. Final changes were covered by the subsequent 194-case command above; no claim is made that every kernel/compiler combination was rechecked after that final change.

Focused native/fold/checkpoint suite before the last broadcast case: **30 passed in 0.78s**. Native state/output/A/index equality, artifact removal, wrong version/rank/actual target shape, pre-optimizer compact shapes and stable Parameter IDs, independent AdamW continuation, native/Comfy/dense-delta multipliers 0/.5/1, nonuniform row scales, and fp16/bf16 master/export precision are covered.

Final black --check: 11 files unchanged; ruff: all checks passed. Explicit ASCII and <=600-line scan passed for all 11 changed Python files. `git diff --check` and staged diff check passed. Post-commit inner git status is clean.

## Concerns and Task 5 handoff

- No real MOSS loading, training, encoder/VAE emulation, integration, VRAM or speed measurement was performed. Task 5 owns actual MOSS targets, offload and training behavior.
- Raw tensor-only factory initialization remains supported for Task 3 compatibility. Sparse native save explicitly requires the accompanying versioned metadata; raw indices alone cannot establish the original full output dimension or calibration provenance. Actual training resume/native files supply and validate that metadata.
- Native LoRI files are local KohakuMoss files; Comfy LoRA conversion is the compatibility export. No third-party source or loader was modified.
- Existing sparse compute still reconstructs dense B during forward; this task changes portable storage/restore/export, not sparse compute or activation memory.

## Task 4 review fix, round 1

Review baseline: `623629b39ef01be23b6d28ec2aad95a7ddb35c73`.
Fix commit: `f34464acf6eaa87ad11883ad700053d101249c9d`.
Status: the single Important mixed-dtype delta/fold export defect is fixed.

Cause confirmed: placement preserves FP32 A while casting trainable B to fp16/bf16; the direct `B @ A` in `LoRIModule.get_diff_weight` consequently raises outside autocast. The fix materializes B once and casts only the temporary A operand to B's dtype. With low-precision B, the matmul remains fp16/bf16 rather than promoting either operand to FP32. No persistent buffer or Parameter is cast/replaced by this operation; the preexisting all-FP32 model-free reference behavior is unchanged.

Added focused regressions cover dense calibration and sparse compact B with FP32 A and fp16/bf16 B; multipliers 0/.5/1; nonzero values; differentiability and B identity; frozen A/alpha/index dtype/value preservation; Target dense deltas; nonuniform fold conversion; actual native and Comfy file exports; and strict rereading/digests of native files. No other production path was edited.

Commands ran from the copied inner repository using the existing external interpreter. Both test commands had `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH=D:/Develop/KohakuMossTrainer-LoRI/KohakuMossTrainer/src;D:/Develop/KohakuMossTrainer-LoRI/.tools/python`.

RED, before the production fix:

```
python -m pytest tests/adapters/test_lori_math.py tests/adapters/test_lori_state.py -k 'low_precision_b or master_folded' -q -p no:cacheprovider --basetemp ../.tools/task4-fix-red
```

Output: **8 failed, 55 deselected in 0.27s**, exit 1. All eight fail on the expected original dtype mismatch in get_diff_weight, including both fp16 and bf16.

GREEN, after the two-line production correction:

```
python -m pytest tests/adapters/test_lori_math.py tests/adapters/test_lori_state.py tests/adapters/test_lori_build.py tests/adapters/test_lori_calibration.py tests/adapters/test_state_refusal.py tests/adapters/test_stacking.py -q -p no:cacheprovider --basetemp ../.tools/task4-fix-green
```

Output: **133 passed in 11.09s**, exit 0.

Final checks:

```
python -m black --check src/kohakumoss/adapters/kinds/lori.py tests/adapters/test_lori_math.py tests/adapters/test_lori_state.py
python -m ruff check src/kohakumoss/adapters/kinds/lori.py tests/adapters/test_lori_math.py tests/adapters/test_lori_state.py
```

Output: **3 files would be left unchanged**; **All checks passed!**, exit 0. An initial repeated formatting-check shell omitted the outer tools PYTHONPATH and reported that black/ruff modules were unavailable; rerunning with the same documented PYTHONPATH as the test process produced the successful final results above. No package installation or dependency edit was required.

ASCII and <=600-line checks: **3 files passed**. Staged diff check passed. Post-commit working tree is clean. No real model loading, training, third-party modifications, or additional unrelated cases were performed. Real MOSS/offload integration remains Task 5.
