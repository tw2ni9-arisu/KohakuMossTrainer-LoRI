# Task 6 bounded sparse training

Date: 2026-11-18 local. Session: `experiment/task6/session-20261118-a/`. The authorized 32-step sparse phase completed once with native trainer, wrapper, waited launcher and exec-session exit0. No sampling, Task 7 run, extra real-model reload or additional training was started.

## Result and invariants

The existing frozen recipe and approved private audit-path revision were used unchanged: original twelve-image NL subset, 256 target area, rank8/alpha16, sparsity0.9, bf16/INT8, default Triton, named `ckpt-ring4-actoff-optstream-gradoff` with fused_step=True, AdamW1e-4/zero decay/constant schedule, seed42, count batch1 and accumulation1. The phase started in a new process with an empty RESUME, zero compact B and newly created optimizer state. The calibration optimizer was not restored.

- Metrics contain exactly steps 1..32. Every loss and gradient norm is finite; skipped/nonfinite/nonfinite-parameter counters are zero throughout. Native final TrainState and checkpoint state both record step32/samples32, loader seed42/start32, world1.
- Initial hook successfully asserted zero B and requires_grad=False for base parameters. All 756 A/alpha/active-index tensor hashes remain exactly unchanged. All 252 compact B vectors changed and remain finite.
- There are exactly 1,066,599 trainable B values, equal to mask nnz. The optimizer has 252 entries. Master, first moment and second moment are FP32 vectors, each totaling 1,066,599 elements. The two moment arrays therefore contain exactly 2,133,198 elements, twice nnz; no full dense calibration-B moment arrays are present. Every optimizer counter is32 and all tensors are finite.
- This is evidence of physical compact adapter/optimizer storage. It does not imply proportional total VRAM or runtime savings, because the base model, activations, fixed A and index storage remain.

Loss first/final/min/max: 0.54284686 / 0.81080884 / 0.42006013 / 0.91301197. These 32 stochastic steps establish a bounded functional run; no convergence, character/style quality or LoRA-equivalence claim follows from this loss trace.

## Artifacts and CPU self-contained reload

Actual output: `sparse/lori-sparse/`. Native step32 checkpoint contains `trainable.safetensors`, `optim.pt`, `rng.0.pt` and `state.json`; checkpoint-local and final Comfy/LyCORIS exports are present.

| Final artifact | Bytes | SHA256 |
| --- | --- | --- |
| Native checkpoint trainable | 44,654,812 | d5658e14b3698e0965382259a55aaadb0f93fb08edf63c8f36d21db20e7b3c93 |
| Native optimizer | 13,039,483 | retained in checkpoint |
| Final Comfy export | 49,601,320 | 5dc2278c9c2397b0c0ebbba0bd081fa76a24835915d22e9c27b28cee70bdedba |
| Final LyCORIS/HF export | 44,654,796 | 2c403855d340bea8460b492ad7999fbc07ad6cd10c3043cc99a6d8e3bdf17e19 |

The lightweight CPU reload called native `load_lori_initialization` on the saved checkpoint. A guard prohibited opening the original `sparse-init.safetensors`; the only actual safetensors read was the native checkpoint itself. Native metadata/shape/digest validation passed. Every layer's constructor-sizing inputs (`A`, alpha and active indices/rank) were derived from that loaded native state through `lori_layer_init`, without the original mask. The loaded B values matched the saved final-state invariants. CUDA remained uninitialized.

This verifies native state reading, self-contained constructor inputs and exact saved tensors. It is not a new whole-model reload, optimizer-resumed training step, generated-output comparison or deterministic training-continuation experiment. Task 5 separately provides the real-model resume gate; Task 6 has retained its complete native resume artifact.

## Resource measurements and process scope

Actual worker PID85176 (redirector84188), start `2026-11-17T19:26:07.325002Z`, end `19:27:15.405480Z`. Luna independently checked process identity, real exit, stage result and checkpoint evidence.

| Measurement | Value |
| --- | --- |
| Complete launcher wall time, including imports | 84.134 seconds |
| Worker stage work after imports | 68.081 seconds |
| Step1 time from metrics | 7.755 seconds |
| Steps2..32 median | 1.459 seconds |
| Host process lifetime peak working set | 14,242,959,360 bytes, about 13.27 GiB |
| Final host RSS | 9,768,771,584 bytes |
| PyTorch CUDA peak allocated | 2,576,283,136 bytes, about 2.40 GiB |
| PyTorch CUDA peak reserved | 2,889,875,456 bytes, about 2.69 GiB |

The native outer CUDA memory window surrounds stage work after imports and includes training, exports and final assertions. Its bookkeeping absorbs internal nested peak-counter resets. These are process-local PyTorch figures, excluding context/driver/non-PyTorch allocations and other workloads. Luna's highest reported live nvidia-smi sample during sparse was 4772 MiB global used (7173 MiB free, 99% utilization at step29); that is observed machine-wide usage, not a process allocation peak or guaranteed true global peak. Neither scope certifies 4GB/6GB operation or other resolutions.

## Integrity and retained history

Approved private Python hashes still match `approved-audit-path-fix.json`; all four recipe files retain their original hashes. The 570 source-file names/sizes/nanosecond mtimes and source directory mtime remain unchanged. Luna also checks selected image/caption contents, exact manifest text and cache fingerprints against the previous baseline. Inner trainer git status remains clean. No unrelated process, source image/caption or source venv was modified.

The original calibration wrapper exit1 and traceback remain intact; its already successful twelve-step native training was accepted only through the separately authorized CPU posthoc audit, not a rerun. Mask creation and its approved path fix remain explicitly documented in `task-6-mask-report.md`.

Machine-readable evidence: `sparse-final-audit.json`, `stages/sparse/{launch,start,result,exit}.json`, `stages/sparse/initial.json`, native `metrics.jsonl`, checkpoint/export artifacts, and `monitor/monitor.jsonl`. Task 6's bounded data/cache/calibration/mask/sparse sequence is verified within these stated scopes. Image-quality comparison remains unexecuted.
