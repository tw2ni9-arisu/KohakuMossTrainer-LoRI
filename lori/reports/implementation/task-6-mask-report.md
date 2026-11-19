# Task 6 approved audit recovery and CPU mask

Date: 2026-11-18 local. Session: `experiment/task6/session-20261118-a/`. Root authorized only the minimal private audit-path fix, CPU posthoc calibration verification and conditional CPU mask extraction. No model was reloaded, no training was repeated, and no sparse training has started.

## Explicit audit correction

Native training writes under `Path(OUTPUT_DIR) / run_name(knobs)`, with `run_name` falling back to `run`. The private audit had omitted that component. `audit.py` now uses native `run_name` with the exact phase config to resolve each phase output and step checkpoint; mask extraction uses the same explicit calibration checkpoint. It does not discover a checkpoint by glob.

The original seven scripts, original `resolved.json` source-hash manifest, and calibration `result.json`/`exit.json` were preserved under `audit-path-fix-originals/`. The historical calibration stage remains exit1 with its original traceback. `approved-audit-path-fix.json` records the explicit approval, every old/new script hash, resolved corrected paths and unchanged configuration hashes. Only audit/worker/launcher changed: a separately named CPU posthoc stage was added, and mask now requires that posthoc success rather than silently treating the original exit1 as success. The four common/data/calibration/sparse recipe files and original resolved manifest were not modified.

Black and Ruff passed on the three changed files. CPU dry checks confirmed the exact paths `calibration/lori-calibration/checkpoints/step_00000012` and `sparse/lori-sparse/checkpoints/step_00000032`, preserved exit1, unchanged configuration hashes and no CUDA initialization.

## Posthoc calibration verification

Unique CPU stage `posthoc-calibration` ran as actual PID50228 (redirector82444). It genuinely exited with worker/launcher/exec status0, while explicitly reporting `original_wrapper_exit_code=1`. It re-read the correct native checkpoint, confirmed unchanged A/alpha hashes and 252 changed B layers, and accepted the detailed twelve-finite-step/initial-zero-B/frozen-base evidence in `calibration-posthoc-audit.json`. Checkpoint state remains step12/samples12/seed42.

Native checkpoint SHA256: `b2cd61ba60307f10e10f4121e8cbccf922051516163dbb603ae5fa7a851f981d`. Posthoc work took 0.394 seconds and complete launch 17.899 seconds, with peak working set 967,598,080 bytes. CUDA was not initialized. No full 5B post-training base hash or new resume test is claimed; base evidence remains initial requires_grad=False plus optimizer membership limited to B.

## CPU mask result

Unique `mask` stage invoked the original `scripts/adapters/lori_mask.py` on that explicit native step12 checkpoint with `--sparsity 0.9`. Actual PID48744 (redirector57364) genuinely exited; worker, launcher and exec status are all0. CUDA remained uninitialized. No retries or overwrites occurred.

| Field | Result |
| --- | --- |
| Global score | absolute dense calibration B, float32 |
| Quantile sparsity | 0.9 |
| Threshold | 0.0006955114076845348 |
| Selection comparison | score >= threshold, retaining ties |
| Original B values | 10,665,984 |
| Retained compact B values | 1,066,599 |
| Actual density | 0.10000005625360023 |
| Layers | 252 |
| Artifact phase | sparse_init, version1 |
| Output size | 44,654,588 bytes |
| Output SHA256 | 37f38c65bc19c63e0ab0ea9238f0a0aa8cb9a958a41664d66af4d877ef88ee74 |
| Calibration state digest | 2bc5c2abf51df43dec2bd573914456b88ef24e4340701b7349428488d9255280 |

The audit regenerated expected CPU sparse initialization from the calibrated state, compared every output tensor exactly and compared all generated mask metadata. A and alpha are therefore preserved exactly; compact B values are zero. Native artifact reading separately validates finite values, sorted unique in-range indices, target/rank shapes, per-layer digests, total/nnz/density and provenance metadata. The sparse phase will have to construct a new optimizer; this operation created no optimizer state.

Mask work took 1.776 seconds; complete launch took 19.279 seconds. Host lifetime peak working set was 1,159,544,832 bytes. CPU stages have no CUDA allocator peak measurement. Both stages retain their actual PID/start/command/log/result/waited-exit files; Luna independently monitors those identities and artifacts.

The output is `experiment/task6/session-20261118-a/sparse-init.safetensors`. Sparse stage/output directories are absent. The source dataset/venv were not written, the inner source git working tree remains clean, and all original failure evidence remains preserved. This report establishes mask readiness, not sparse-training or image-quality success.
