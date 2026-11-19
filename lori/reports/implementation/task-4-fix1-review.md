# Task 4 fix round 1 scoped review

Base: `623629b39ef01be23b6d28ec2aad95a7ddb35c73`.
Head: `f34464acf6eaa87ad11883ad700053d101249c9d`.

## Finding verdict

**ADDRESSED** — “preserved projection dtype breaks low-precision delta and folded export.”

`src/kohakumoss/adapters/kinds/lori.py:96-97` now materializes B once and converts the temporary A operand to B's dtype before matrix multiplication. FP32 A with FP16/BF16 B therefore no longer invokes the invalid mixed-dtype matmul. The computation retains B's working dtype. The conversion does not assign to A, alter any persistent A/alpha/index bytes, or replace the trainable B Parameter. Gradient propagation through dense/compact B is retained. The all-FP32 path remains equivalent to the original calculation.

The added regressions cover the original failure combinations directly:

- `tests/adapters/test_lori_math.py:164`: dense calibration and sparse compact phases, both low-precision B dtypes with FP32 A, multipliers 0/.5/1, backward propagation, stable B identity and unchanged frozen buffers.
- `tests/adapters/test_lori_state.py:225`: nonzero sparse state, Target deltas, nonuniform row-scale conversion, actual native/Comfy file exports, native rereading with digest validation, unchanged frozen tensor dtypes/values and stable trainable Parameter identities.

The implementation report records the focused tests failing before the fix (**8 failed, 55 deselected**) on the original dtype error, followed by the affected math/state/build/calibration/refusal/stacking suites passing after the fix (**133 passed**). These results were read as implementation-owner evidence; the already-passing suites were not rerun during review.

## New findings

- Critical: none.
- Important: none.
- Minor: none.

The production diff is limited to the two-line operand correction. Its dtype conversion is temporary and leaves the Task 4 storage and restore identity constraints intact. No new concrete failure hypothesis requiring a focused execution arose from this diff.

## Out-of-scope observations

No additional observations. Unchanged code was not re-reviewed. Real MOSS/offload integration, VRAM/performance and actual training remain Task 5/6 verification and are not findings against this fix. This review does not extend the earlier Task 4 verification claims to those tasks.

## Round verdict

**PASS — Spec and Quality accepted for this fix round; the sole prior Important finding is resolved.**

Read the complete fix review package and the appended fix report. Independently confirmed the stated HEAD and clean inner working tree. No source, HEAD or index changes, training, subagents or additional test executions were performed. Only this outer review report was written.
