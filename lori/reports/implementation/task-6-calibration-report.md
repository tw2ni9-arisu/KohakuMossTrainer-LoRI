# Task 6 calibration execution

Approved recovery update: Root subsequently authorized a minimal private audit-path correction and CPU-only posthoc audit, without retraining. `stages/posthoc-calibration` has passed independently with exit0 while the original `stages/calibration` result/exit1 remains unchanged. Old scripts and hashes are archived, and new hashes are recorded in `approved-audit-path-fix.json`; all four recipe files are unchanged. Native calibration is now validated by this explicit posthoc result. See `task-6-mask-report.md` for the authorized CPU mask follow-up, not a sparse-training result.

Date: 2026-11-18 local. Session: `experiment/task6/session-20261118-a/`. Only the explicitly authorized calibration stage ran once, using the frozen recipe, default Triton and full named fused 12g plan. No mask, sparse training or generated-image phase has started.

## Native training completed; wrapper audit failed

The native trainer completed all 12 steps, saved its checkpoint and exports, printed `Script executed successfully`, and returned `TrainState(step=12, samples=12, ok=True, skipped=0, nonfinite=0, nonfinite_params=0)`.

The private wrapper then failed during its post-training artifact audit. The audit incorrectly expected `OUTPUT_DIR/checkpoints/step_00000012`; the native trainer appends `RUN_NAME`, and the real checkpoint is under `calibration/lori-calibration/checkpoints/step_00000012/`. The worker, launcher and exec session therefore have exit code 1. This is a private audit-path error after successful native training, not an OOM or training-numerics failure. It was reported immediately; all artifacts/error logs remain intact, with no automatic retry or configuration/worker change.

The same mistaken path is present in the prepared mask route and the sparse-phase audit. Those phases must not be launched until their private path handling is corrected and reviewed; no such correction or launch occurred within this authorization.

## CPU-only post-hoc verification

An independent read-only inspection of the actual checkpoint, original initial-state hashes and `metrics.jsonl` was saved to `calibration-posthoc-audit.json`. It does not replace or rewrite the original failing stage result.

- Exactly 12 metric rows cover steps 1 through 12; all losses and gradient norms are finite. Every row reports zero skipped steps, nonfinite steps and nonfinite parameters. Loss range: 0.53758955 to 0.91204971; final loss 0.59829611. Twelve steps are a bounded correctness check, not a convergence or image-quality result.
- The frozen worker's initialization hook successfully asserted that all base parameters had `requires_grad=False` before adapter construction and every initial B was zero, then wrote `stages/calibration/initial.json`.
- All 504 final A/alpha tensor hashes match the initial hashes exactly. All 252 dense B tensors changed, and all native adapter tensors are finite. Dense calibration has 10,665,984 trainable B values.
- `optim.pt` has 252 streamed optimizer entries, each with three same-shaped FP32 tensors and a scalar; all inspected tensors are finite. Total optimizer tensor elements are 31,998,204, equal to three times the dense B values plus 252 scalars. This is dense calibration storage, not sparse-phase memory.
- Native `state.json` records `step=12`, `samples=12`, loader seed42/start12, world1. Native trainable state is 74,397,040 bytes; optimizer file is 128,213,819 bytes. RNG and state files are present.
- Checkpoint-local and final Comfy and LyCORIS/HF portable exports are present; their sizes and SHA256 values are recorded in the post-hoc audit. This inspection confirms saved artifacts and native read/validation. It does not claim a new real-model resume or generated-output comparison.
- Base-freezing evidence consists of the initialization assertion and optimizer membership limited to adapter B. The exited worker did not record a post-training full 5B in-memory base hash, so a bitwise final base comparison is not claimed.

## Process and resource evidence

Launcher redirector PID57212 led to actual worker PID86712. Worker start was `2026-11-17T19:08:39.421685Z`, end `19:09:19.040292Z`. Both worker and launcher genuinely exited; PID/start/command matching was independently monitored by Luna.

| Measurement | Value |
| --- | --- |
| Complete launcher wall time | 58.395 seconds |
| Worker stage work after imports | 39.619 seconds |
| Host process lifetime peak working set | 14,241,726,464 bytes (about 13.26 GiB) |
| PyTorch CUDA peak allocated | 2,612,107,776 bytes (about 2.433 GiB) |
| PyTorch CUDA peak reserved | 2,925,527,040 bytes (about 2.725 GiB) |
| Step 1 | 8.387 seconds |
| Steps 2-10 median from native log | 1.677 seconds |

The CUDA statistics are from the native outer memory window around stage work after imports, absorbing nested peak resets and including training/export/post-stage work. Complete wall time includes startup/imports. These allocated/reserved figures exclude CUDA driver/context and other processes; Luna's nvidia-smi samples are observed global usage and must remain separately labelled.

Detailed evidence: `stages/calibration/{launch,start,result,exit}.json`, `stdout.log`, `initial.json`; native `calibration/lori-calibration/{config.json,metrics.jsonl,log.txt}` and checkpoints/exports; `calibration-posthoc-audit.json`; `monitor/monitor.jsonl` and monitor report. The stage remains marked ERROR at wrapper level until a separately authorized audit correction/recovery decision; the already completed training must not be restarted merely to fix its audit path.
