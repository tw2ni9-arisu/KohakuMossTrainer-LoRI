# Task 5 compatibility gate

Status: revised Task 5 gate PASSED on real stored-INT8 MOSS with the scoped backend oracle described below; verified tests committed. The original run-1 failure is retained unchanged.
Baseline: `f34464acf6eaa87ad11883ad700053d101249c9d`.
Verified source commit: `285f4bd2b5c29378ca75797dd247690f6a06a21a`.

## Original implementation (run 1)

Only three inner-repository test files were added; no production implementation was changed:

- `tests/training/run/test_lori_real_weights.py`: full stored-INT8 MOSS, actual `build_model`, `build_step`, memory optimizer wiring and checkpoint manager. Uses four weight slots, block checkpointing, activation offload, CPU AdamW and gradient offload.
- `tests/training/run/_lori_checks.py`: live compact Parameter identity, CPU master/moment shape, exactly `2*nnz` moments excluding scalar steps, host gradient shape, CPU optimizer alias and bounded host/device gradient staging assertions.
- `tests/memory/offload/test_lori_compact.py`: model-free real linear operators with compact and empty-mask LoRI, host updates and a deliberately dense-moment negative control.

The actual-weight gate performs four training-style forward/backward/update steps across three sequential full-model constructions: one calibration step, sparse step one, continuous sparse step two, then checkpoint restore and repeated sparse step two. The native checkpoint is written through CheckpointManager; mask initialization is removed before restore. Restored final adapter tensors and optimizer moments must exactly equal the continuous branch. Final compact Parameters must exist before optimizer/offload capture and retain their identities through restore and updates.

Frozen A/alpha/index snapshots and an SHA256 digest of the frozen backbone parameters, buffers and streamed host weight tensors must remain unchanged. Actual stored-INT8 target bases must remain StreamedQuantWeight with empty dense weight placeholders. A target base-materialization sentinel rejects a LoRI merge/delta path. The expected attn-mlp target count is `6*actual depth`, with unique paths and compact B values matching int64 indices on the correct device. Block forward hooks must prove recomputation, stream counters must prove block loads, and activation offload must pack tensors and release pending buffers.

## Input boundary

This compatibility gate uses the complete workspace INT8 MOSS DiT, not a toy model. Its numerical inputs are synthetic: one packed 16x16 latent grid (256 tokens), eight text tokens, dimensions read from the real architecture. Offline cache knobs suppress encoder/VAE construction; no encoder or VAE is emulated. It does not measure real image quality, caption conditioning or dataset learning and cannot replace Task 6.

After the actual calibration step, one target's calibration B in a CPU test copy is deliberately zeroed before global quantile mask extraction, ensuring an empty target is exercised. This is a compatibility edge-case input, not a proposed calibration procedure for Task 6. The generated sparse initialization must retain exact A, zero B and fresh moments.

## Model-free evidence

Environment: external source-venv Python used read-only; copied src and outer `.tools/python` on PYTHONPATH; PYTHONDONTWRITEBYTECODE=1; process-local CC selects the existing bundled TCC. All temporary/cache outputs are under outer `.tools/task5/checks`.

Command from the inner repository:

```
python -m pytest tests/memory/offload/test_lori_compact.py tests/memory/offload/test_grad_offload_steps.py tests/memory/offload/test_optim_offload.py tests/adapters/test_lori_state.py tests/adapters/test_lori_math.py tests/adapters/test_lori_build.py tests/adapters/test_lori_calibration.py tests/training/checkpoint/test_checkpoint_meta.py tests/training/checkpoint/test_checkpoint_stream.py -q -p no:cacheprovider --basetemp <workspace>/.tools/task5/checks/pytest-temp
```

Result: **137 passed, 1 skipped in 23.88s, exit 0**. The skip is the existing CPUAdamW test's XPU device parameter (`no XPU`), not the MOSS gate. Log: `.tools/task5/checks/model-free.log`.

Collection only of the new actual-weight file: **one test collected in 10.78s, exit 0**. This executed no model fixture or training. Log: `.tools/task5/checks/collect.log`.

Black and ruff passed for all three added Python files. AST parse, ASCII and file-length checks passed for the test files and private launcher; all are below 600 lines. The inner diff contains only untracked additions at this preparation stage.

## Launch and monitoring protocol

Private launcher: outer `.tools/task5/launch.py`. Launch exactly once, only after root supplies a ready Luna monitor:

```powershell
Start-Process -FilePath 'D:\Develop\nigenmame-LoRA\KohakuMossTrainer\.venv\Scripts\python.exe' -ArgumentList 'D:\Develop\KohakuMossTrainer-LoRI\.tools\task5\launch.py' -WindowStyle Hidden -PassThru
```

The launcher uses a new `.tools/task5/run-<UTC timestamp>` directory and writes its location to `.tools/task5/latest-run.txt`. `pid.json` contains the actual pytest child Python PID, launcher PID, UTC start, exact command, cwd, log path and resolved child environment overrides. Monitor the child PID, not a lock file. `run.log` captures initial nvidia-smi and all pytest output/tracebacks. `exit.json` records actual child exit and end time. Exit 0 without `artifacts/result.json` is converted to exit 3, preventing a skipped test from appearing successful.

Each run has private TEMP/TMP, Triton, Inductor, project/LyCORIS kernel, CUDA and HF cache directories. CC is overridden only in the child environment; CXX is not changed by the launcher. The gate records compiler environment after backend/model initialization. Source venv, external models and source data stay read-only. No model download is needed. Unrelated processes must not be changed or terminated.

Artifacts: `artifacts/progress.json` at completed phase boundaries, `artifacts/checkpoints/step_00000001/` (trainable/native metadata, optimizer, RNG and state), `artifacts/final-native.safetensors`, and final `artifacts/result.json`. The final report includes per-build and per-step peak allocated/reserved VRAM, loss, gradient norm, timing, ring/offload counters, compact counts and exact continuation verdict.

## Actual-weight evidence

Run: `.tools/task5/run-20261117T150616Z`, launched once after root authorization and Luna READY. Actual pytest child PID 10580; internal launcher PID 66888 (Start-Process entry PID 78156). Start UTC `2026-11-17T15:06:16.164105+00:00`, exit UTC `2026-11-17T15:07:19.636366+00:00`. Exit 1, **1 failed in 45.86s**, not skipped. Both execution owner and Luna independently read exit.json/run.log; the monitored processes ended. No retry or code/parameter change followed failure.

Failure: `test_lori_real_weights.py:349` compared final resumed adapter state to the continuous branch with `rtol=0, atol=0`. The first B-values comparison had 12/12 differing elements, greatest absolute difference `1.6914214938879013e-05`, greatest relative difference `0.009535974822938442`. The failure occurred before comparing final optimizer moments. Cause is not established: checkpoint behavior versus numerical repeatability needs focused diagnosis before any fix or second actual run. This is not a compatibility PASS and no final result.json/native-final was produced.

Completed phase evidence in `artifacts/progress.json`:

| Phase | Loss | Gradient norm | Step seconds | Peak allocated bytes | Peak reserved bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibration | 2.5738911628723145 | 0.5740384459495544 | 8.052695900027175 | 2664428544 | 2908749824 |
| Sparse step 1 | 2.5738911628723145 | 0.5437141060829163 | 1.2786907000117935 | 1926782464 | 2004877312 |

Both completed steps recorded two block-0 calls (checkpoint recompute), 80 streamed block loads, 9,069,854,720 H2D weight bytes, zero weight-engine host syncs, and 42 packed activation tensors / 68,124,672 packed bytes. Sparse state had `nnz=1,070,251`, `moment_numel=2,140,502`, `host_grad_numel=1,070,251`; the host staging bound was 578,384 bytes (initial pre-step staging allocation was zero). The deliberate empty target was `lora_unet_layers_0_proj_ffn_down`. Actual target construction logged 252 modules.

Execution reached checkpoint save, mask deletion, full-model reconstruction, optimizer/RNG restore and a finite resumed second step; live compact/offload checks after that step passed before the final equality assertion. Phase metrics after sparse step 1 were held in memory and not flushed before the failure, so no later timing/VRAM claims are made from this run. Frozen A/base consistency passed on the calibration and continuous branches; the last resumed frozen checks occurred after the failed equality assertion and therefore were not reached.

Evidence retained: `run.log` (complete traceback), `exit.json`, `artifacts/progress.json`, and native `artifacts/checkpoints/step_00000001/` including trainable metadata, moments, RNG and state. Optional resident comparison was not run; no test matrix expansion or unrelated process intervention occurred.

## Run 1 diagnosis (read-only source/checkpoint and model-free probes)

No further full MOSS load/forward/backward was performed. The original run remains failed. Diagnosis used systematic-debugging and did not change production or gate code or loosen comparison tolerances.

Native checkpoint inspection (`.tools/task5/diagnose-resume.log` and `diagnose-resume-cuda.log`): all 252 compact B tensors exactly equal the corresponding CPU master in optimizer parameter order (q/k/v/o/ffn_up_gate/ffn_down per block). Every master/m/v shape matches B; scalar steps are all 1; nnz and moment counts remain 1,070,251 / 2,140,502. The saved group has LR 0.001, weight decay 0, betas (0.9,0.95), epsilon 1e-8. Schedule epoch is 1 and its restored LR is 0.001. Step RNG has separate sigma and noise generator states.

A model-free actual LoRI linear operator plus CPUAdamW, GradOffload, schedule and CheckpointManager was saved and independently restored on CPU and CUDA. Before the next step, p/master/m/v/step were exactly equal; Parameter identity, gradient-host lane alias and optimizer-group alias were preserved. After an identical operator step both parameter and all optimizer tensors were exactly equal (max absolute difference zero). The CUDA diagnostic initially used an invalid Windows folder containing `cuda:0`; that diagnostic-only path was corrected to `cuda-0`, then its probe passed. This was not a MOSS retry.

Source trace: the gate uses a fresh local CPU generator with seed 17 for identical batch tensors. TrainStep owns separately seeded CPU sigma and CUDA noise generators, restored through `load_rng_state`; noising is out-of-place. Native adapter load copies into existing Parameters and rebinds the same owning operations. CPUAdamW restores fp32 host master/m/v after torch optimizer load and rebinds each lane to its restored group; the model-free probe covers these aliases. The run-1 log did not persist pre-step restored-state or prepared-input hashes, so a true real-integration boundary mismatch cannot yet be ruled out from artifacts alone.

One concrete faulty test assumption was reproduced: **the default Triton fused attention backward is not bit-exact on identical inputs**, so final B bit equality is not a valid default-kernel continuation oracle without a repeatability precondition. `.tools/task5/diagnose_attention.py` used the gate's joint attention shape [264,24,128], BF16, fixed q/k/v and fixed output gradient, with tuning off and the default registry backend (Triton). Four model-free repeats produced identical forwards and identical dK/dV, but repeats 1/2/3 differed in 41/46/42 dQ elements from repeat 0; greatest dQ absolute difference was 0.00048828125. Evidence: `diagnose-attention.log`, exit 0. Source `backends/triton/attention/_kernels.py::_fused_tile` accumulates dQ with relaxed atomic_add; backward strategy's first untuned candidate is fused in `varlen.py::_build_backward`.

The same fixed-input repeated probe with the existing eager attention backend and PyTorch math SDPA produced exact forward/dQ/dK/dV equality in all four repeats (`diagnose-attention-math.log`, exit 0). This establishes a tested deterministic attention option for an exact continuation oracle; it does not alone prove that all of run 1's B difference came from attention.

Proposed minimal change, pending root approval: test-layer boundary assertions and immediate phase evidence for saved/restored p/master/m/v/step, LR/schedule, IDs/aliases, batch and prepared-input hashes; keep `rtol=0,atol=0` for continuation but select the existing eager + math SDPA attention path for that strict oracle. Retain run 1's default-Triton finite/offload evidence separately and report the precise backend scope. No production optimizer, adapter or attention change is justified by current evidence. Any second real gate still requires new root launch authorization and Luna READY.

## Revised gate preparation (test-only, not yet run on MOSS)

Root approved the scoped test correction and required the actual Task 6 profile route. The revised gate now uses the named `ckpt-ring4-actoff-optstream-gradoff` profile unchanged: block checkpointing, four weight slots, activation offload, streamed host optimizer states, fused step and gradient offload. BASE_PRECISION is int8 and GRAD_CLIP is the native default 1.0. GRAD_ACC remains the Task 6 value 1. It builds the real FusedStepBundle with shared HostStateAdamW via build_optim, invokes accumulate(True) / bundle.backward / bundle.step as the trainer does, and asserts that all live B names reached the bundle's in-backward step callback. Finite/nonzero gradients are captured before accumulation hooks consume them. It also requires bundle.last_ok (the fused step API alone returns True even on a flagged step).

The actual calibration and sparse step 1 keep default Triton attention and now must pass finite, compact-update, exact frozen-A/base and complete memory-path checks. Both continuous step 2 and resumed step 2 then switch **only** the attention_varlen callable in the existing OpSet, using dataclasses.replace, to existing eager attention inside a math-SDPA context. Every other resolved primitive/fused callable, the complete MOSS backbone, weight ring, checkpoint, activation offload, HostStateAdamW and fused optimizer route remains unchanged. Zero-tolerance final parameter and optimizer continuation assertions remain. The result names the backend for each phase; this is not a claim that default Triton is bit-exact.

`_lori_checks.py` now covers both the original CPUAdamW and actual HostStateAdamW compact state forms. For the latter it checks live host/state/group IDs, exactly 2*nnz m/v, three optimizer state slots each shaped [4,max_compact_numel], and gradient ring / host accumulator allocation bounds. With GRAD_ACC=1 the final gradient hook directly steps each parameter; host gradient sums are not filled. The actual metrics explicitly report `host_gradient_sum_exercised=false`. This is an honest scope boundary, not disabled gradient offload.

New `_lori_evidence.py` creates immutable CPU snapshots, recursive exact assertions and dtype/shape/SHA256 JSON evidence. Before strict continuation and immediately after restore, the gate writes `boundary-continuous.json` / `boundary-resumed.json` with trainable parameters and buffers, master/m/v/step, full bundle scheduler/scaler/clip state, optimizer LR, sigma/noise RNG, global torch CPU/CUDA RNG, and batch/SeqInfo inputs. All must match exactly before the resumed forward starts. The real TrainStep.keep callback writes `prepared-continuous.json` / `prepared-resumed.json` and compares noised latent/text/sigma/target/weight/mask/denominator inputs exactly before forward. Phase metrics are flushed after each completed step; frozen A/base checks run before final continuation comparisons.

New model-free `tests/memory/offload/test_lori_fused.py` exercises the actual named profile wiring on real LoRI linear operators, an empty compact target, HostStateAdamW, GradOffload, FusedStepBundle, clipping/scheduler state and CheckpointManager. Two micro-batches in the model-free first step prove host sum/staging activity; the continuation step uses one micro-batch as Task 6 does. Parameter identities, pre-step state, final weights and all optimizer/scheduler state are exact through disk restore. A modified clip-state negative control is rejected. No fake MOSS/encoder/VAE classes were created.

Revised relevant model-free command:

```
python -m pytest tests/memory/offload/test_lori_fused.py tests/memory/offload/test_lori_compact.py tests/memory/offload/test_grad_offload_steps.py tests/memory/test_optim_wiring.py tests/adapters/test_lori_state.py tests/training/checkpoint/test_checkpoint_meta.py tests/training/checkpoint/test_checkpoint_stream.py -q -p no:cacheprovider --basetemp <workspace>/.tools/task5/checks/fused-pytest-temp
```

Result: **44 passed in 12.21s, exit 0**, no skips, one AnySchedule warning about scheduler.step before optimizer.step on its non-torch host-optimizer LR holder. The model-free oracle independently verified unchanged LR/scheduler/clip state and exact continuation despite this existing warning. No schedule production code was changed. Log: `.tools/task5/checks/fused-model-free.log`.

All five added test files pass black/ruff. Updated real gate and helpers pass AST/ASCII/size checks; actual gate is below 600 lines. Only tests and this private outer report/diagnostic tooling changed. Launcher/PID/exit/cache protocol is unchanged. The revised gate is READY for independent review; no second MOSS launch is authorized or performed yet.

## Run 2 actual-weight verification and committed result

After independent review PASS (`task-5-review.md`) and fresh Luna READY, root explicitly authorized exactly one revised launch. New run directory: `.tools/task5/run-20261117T182252Z`. Actual pytest child PID 41440, internal launcher PID 38932 (Start-Process entry PID 77380), start UTC `2026-11-17T18:22:52.559880+00:00`; exit UTC `2026-11-17T18:24:02.527520+00:00`. Luna independently validated the child command/cwd and its compute descendant, monitored completion and preserved `monitor.jsonl` / `monitor-report.md`. No code or parameter changed during this run, no duplicate run occurred, and no unrelated process was touched.

Result: **1 passed, 1 warning in 52.46s; exit 0, not skipped**. Execution owner and root independently read exit.json, result.json and complete pytest log. The warning is the reviewed AnySchedule/LR-holder warning already documented above; this result certifies the tested constant schedule only.

### Default Triton instrumented gate measurements

These are instrumented compatibility-gate measurements on synthetic 256 latent / eight text tokens, including gradient reductions and synchronizing input/gradient audit overhead. They are not production throughput or real-image training performance.

| Phase | Attention | Loss | Gradient norm | Seconds | Peak allocated MiB | Peak reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Calibration | default Triton | 2.574495553970337 | 0.5720269083976746 | 8.336172099981923 | 2318.85 | 2538.00 |
| Sparse step 1 | default Triton | 2.574495553970337 | 0.5424540638923645 | 1.191906799969729 | 1610.04 | 1774.00 |

Both steps used the actual full MOSS stored-INT8 backbone and unchanged named 12g profile, FusedStepBundle / HostStateAdamW and GRAD_ACC=1. Both proved two block-0 calls (checkpoint recomputation), 80 streamed weight loads / 9,069,854,720 H2D bytes / zero weight-engine host syncs, and 42 activation packets / 68,124,672 packed bytes. Frozen A/base checks and finite compact updates passed. The first-step timing includes first-use setup/compilation overhead and is not a warmed benchmark.

### Strict continuation oracle instrumented gate measurements

Only attention_varlen changed to eager with math SDPA; all other complete-model/profile execution owners and resolved ops remained as reviewed. These figures must not be attributed to default Triton throughput or memory.

| Phase | Attention | Loss | Gradient norm | Seconds | Peak allocated MiB | Peak reserved MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Continuous sparse step 2 | eager/math oracle | 2.1443419456481934 | 0.8194494843482971 | 1.3021550999837928 | 1621.00 | 1774.00 |
| Resumed sparse step 2 | eager/math oracle | 2.1443419456481934 | 0.8194494843482971 | 1.3588530999841169 | 1621.28 | 2356.00 |

Canonical frozen base digest after all branches: `6fc1035bee7190e43155dd5b48dec9e7901d72664ff836f0583bec9e8c583182`. Frozen A/alpha/indices, target shape/device/index contracts, activation exclusions and live compact Parameter identities all passed. The 252 targets include deliberate empty `lora_unet_layers_0_proj_ffn_down`.

Physical adapter state at sparse construction and after resumed update: **nnz 1,070,770; m+v 2,141,540 elements**, exactly 2*nnz excluding scalar steps. Host gradient sums have 1,070,770 elements. Compact optimizer state-ring allocation is 1,755,696 bytes; host staging is zero with a 585,232-byte compact bound. As required for Task 6's GRAD_ACC=1, host sums were not filled in this actual run; model-free multi-microbatch coverage is reported separately.

Strict pre-step trainable/master/m/v/step, scheduler/scaler/clip/LR, torch CPU/CUDA and sigma/noise RNG, batch and SeqInfo snapshots all compared exactly. The separately persisted `boundary-continuous.json` and `boundary-resumed.json` are also identical as parsed JSON; likewise `prepared-continuous.json` and `prepared-resumed.json`. Prepared inputs matched before resumed forward, and final adapter and complete optimizer/bundle state matched at rtol=0, atol=0. `restore_boundary_exact`, `exact_resume`, `live_parameter_ids_preserved` and `frozen_projection_unchanged` are true in result.json.

Artifacts retained under the run's `artifacts/`: progress/result JSON, both boundary and prepared-input JSON pairs, native `checkpoints/step_00000001/{trainable.safetensors,optim.pt,rng.0.pt,state.json}`, and `final-native.safetensors` (44,703,856 bytes). The original mask was deleted before resumed construction. `run.log`, `pid.json`, `exit.json`, `monitor.jsonl` and `monitor-report.md` are retained at the run root.

Post-run black --check, ruff check, unstaged and staged diff checks passed for all five test files. Commit `285f4bd2b5c29378ca75797dd247690f6a06a21a` (`Add real MOSS LoRI fused compatibility gate`) contains only the five reviewed tests/helpers; post-commit inner git status is clean. No production code, dependency, source venv, external data or model file was edited.

Task 5 establishes finite default-Triton compatibility and exact continuation under the explicitly deterministic attention oracle for this bounded synthetic-input gate. It does not establish default-Triton bit-exact continuation, arbitrary schedules/token counts, resident-vs-ring equivalence, encoder startup/cache feasibility, dataset learning or image/style/character quality. Task 6's real-image workflow remains subsequent work.
