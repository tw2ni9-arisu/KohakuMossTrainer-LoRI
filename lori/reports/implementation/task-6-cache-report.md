# Task 6 offline preprocessing

Date: 2026-11-18 local (2026-11-17 UTC). Session: `experiment/task6/session-20261118-a/`. Scope: the conditionally authorized catalog -> text -> latent -> offline-audit group only. No calibration, mask extraction, sparse training, or image generation has started.

## Result

All four native stage processes completed with worker `result.json.exit_code=0`, waited launcher `exit.json.exit_code=0`, and successful exec-session completion. The cache has 12 latent entries and 13 text entries (12 NL captions and the empty prompt). Offline audit materialized all 12 count-batch1 epoch-0 batches using the final calibration DataConfig, with no missing cache keys, dropped rows, text truncation, or image-token dropping. It checked finite bf16 tensors, shapes, cached text lengths and recorded provenance. The independently resolved sparse DataConfig is identical to calibration.

The frozen recipes use 256 target area, original NL captions, seed42, one caption variant, center resize, no flip/dropout, bf16 storage/compute, VAE mode draw, and local INT8 weights. Text encoder compute spec is `bf16/int8/g0/src:int8-convrot256`; the text cache used ring2, batch1 and max_tokens4096. Each encoder ran in a separate process and exited before the next stage.

## Evidence and resource measurements

For every stage, `stages/<stage>/` retains `launch.json`, actual-worker `start.json`, `stdout.log`, `result.json` and waited `exit.json`. The Windows venv redirector PID can differ from the actual worker PID; identity was checked using PID, creation/start time and command. Catalog worker PID74972 was subsequently reused by an unrelated Edge process, which was correctly distinguished and left alone.

| Stage | Actual PID | Worker work seconds | Complete launch wall seconds | Host peak working set bytes | CUDA peak allocated / reserved bytes |
| --- | --- | --- | --- | --- | --- |
| Catalog | 74972 | 0.321 | 18.006 | 813,600,768 | not measured; CPU-only |
| Text | 85600 | 201.472 | 220.202 | 18,247,872,512 | 1,308,635,136 / 1,344,274,432 |
| Latents | 54192 | 1.242 | 19.018 | 2,027,782,144 | 260,576,768 / 301,989,888 |
| Offline audit | 86072 | 0.231 | 17.597 | 885,780,480 | not measured; CPU-only |

Text's host peak is approximately 17.0 GiB, with PyTorch allocated/reserved peaks of 1.219/1.252 GiB. VAE allocated/reserved peaks are approximately 0.243/0.281 GiB. These are encoder-cache measurements, not DiT training measurements or guarantees for a particular VRAM tier.

The CUDA figures come from the trainer's native outer memory window around stage work after imports, including the CLI and post-stage assertions. Its nested-window bookkeeping absorbs internal peak-counter resets. Complete wall time includes imports/startup; worker work time starts after imports. Host peak_wset covers the process lifetime. Catalog and offline audit had CUDA hidden, opened no GPU window and report `cuda_initialized=false`.

Luna's monitoring records live nvidia-smi values as observed global usage, including unrelated workloads, never as worker allocation peaks. Its text sampling observed a global maximum of 3015 MiB used. The latent stage was too short to obtain a live worker sampling point before it finished; native process-local peaks remain available. The text log contained the non-fatal Inductor message `Not enough SMs to use max_autotune_gemm mode`, followed by all 13 completed embeddings; no retry or parameter change was made.

## Provenance and read-only checks

- `resolved.json` records source revision `285f4bd2b5c29378ca75797dd247690f6a06a21a`, private source hashes, manifest digest, shared DataConfig and the actual fused 12g plan. `resolved-*.json` retain native script knobs. Frozen configuration/worker Python hashes still match this record after preprocessing.
- Catalog and both vaults/settings remain under this session's `cache/`: `catalog.db`, `latents.db`, `text.db`, `latents.settings.json`, `text.settings.json`. Both recorded settings matched final training-derived key settings during offline audit. Cache/key details are retained in `stages/offline/result.json`.
- Manifest SHA256 remains `a3c9b275d7fff7057e683dc9838b7eef9597fd0d67539ba17b28fe9ee5e791bf`. The source dataset was not reselected or copied and captions were not rewritten.
- `source-readonly.json` records an independent post-group comparison: all 570 original source files have exactly the original names, byte sizes and nanosecond mtimes, and the source directory mtime is unchanged. The frozen private Python hashes and manifest digest are unchanged.
- Luna independently checked selected image/caption hashes and exact JSONL caption text against the original files before the group; final selected-content verification and monitor verdict are retained in `monitor/monitor-report.md` and `monitor/monitor.jsonl`.
- All runtime/kernel/temp/output caches were routed to the workspace by the private launcher. The source venv was used read-only. Inner trainer git status remains clean. No unrelated process was terminated or modified, no failed stage was retried, and no network/dependency action was needed.

Machine-readable aggregate measurements are in `cache-group-metrics.json`; monitored sampling and stage verdicts are in `monitor/`. Actual training remains a separately authorized next phase and has not been launched.
