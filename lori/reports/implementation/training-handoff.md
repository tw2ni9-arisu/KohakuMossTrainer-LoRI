# Real-model execution handoff

This note supplements Tasks 5–7 without replacing their requirements.

- Use copied models under the current workspace, and source Python dependencies read-only. `compiler-diagnosis-report.md` gives the verified process-only CC override and workspace cache variables; capture the resolved environment in each run.
- The prepared 12-record manifest is `experiment/data/round2-caption-12/dataset.jsonl`, relative to the outer workspace. Original files remain read-only. Set all catalog/tensor/kernel/output caches explicitly inside this workspace. Dataset preparation and source snapshots are already complete; do not repeat the selection or copy the whole dataset.
- Four source samples were visually reviewed in `data-visual-review.md`; use representative character/style prompts for comparison. Existing scene-only suggestions are provisional.
- Check available GPU/RAM immediately before execution. Unrelated moss-controlnet processes must not be terminated or changed.
- A Sol execution owner prepares integration assertions, commands/configs and launch/log paths, but does not start actual model training until a Luna monitor is ready.
- Root dispatches Luna monitoring first. Sol and Luna coordinate launch PID/session, start time and output paths directly; root enters sleep and remains there during training. Luna sends root only a genuine error or completion, and monitors the actual process/session rather than a lock or stale state file.
- Treat temporary observation failure as a reason to recheck the same handle, not restart. Never duplicate a run just because a timeout elapsed. Capture exit status, traceback, final artifacts, finite loss, optimizer state sizes, step timing and phase-specific memory. Return evidence before changing parameters on failure.
- The bounded real-MOSS optimizer/offload gate remains required before the small dataset training. Synthetic adapter tests alone do not certify MOSS compatibility. Real-model integration must not be silently skipped for missing environment hooks.
- Calibration, CPU mask extraction and sparse training are distinct operations with exact A reuse, B reset and new optimizer moments. Save the actual resolved recipes and final native/portable weights. Compare outputs with identical base/seed/sampler settings and show the images.
