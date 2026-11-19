## Global Constraints

- Implement only in copied `D:/Develop/KohakuMossTrainer-LoRI/KohakuMossTrainer`; source trainer/model/data remain read-only.
- Machine paths/private data/configs/manifests/caches/outputs stay gitignored under current workspace; future git covers inner source only.
- Follow inner CLAUDE.md: ASCII Python, comments explain what, black/ruff; model-free adapter/algorithm tests are allowed; integration uses actual MOSS weights, never fake toy model classes.
- Preserve exact A across phases; global abs-B quantile and >= threshold ties; zero sparse B and fresh moments.
- Final compact Parameters exist before any optimizer/offload/EMA capture. Do not change parameter shapes during restore.
- No new third-party fork, custom CUDA kernel, unrelated refactor or FP8 experiment.
- User selected subagent execution. All training uses Luna background monitoring; primary sleeps until completion/error. No additional routine permission gate is inferred.


### Task 6: Private subset, offline cache, calibration and sparse training

**Files:** Ignored workspace manifests/inventory/configs/cache/runs; report under ignored output or outer docs as appropriate. No source dataset write.

**Interfaces:** Existing scripts/data/cache_latents.py and cache_text.py -> scripts/train/moss.py; Task2 mask CLI bridges separate phase processes. All invocations use copied src with external interpreter.

- [ ] Create deterministic 12-image JSONL/inventory: 10 portrait/2 landscape equally spaced in sorted orientation groups, exact caption_nl text, source hashes/dimensions. Verify all source references read-only and CACHE_DIR resolves inside workspace for catalog and vault.
- [ ] Prepare local 256/384 recipe, rank8/alpha16/sparsity.9, bf16/INT8/verified12GB profile, AdamW1e-4/decay0, count batch1/accumulation1, seed42, constant schedule, no dropout/periodic sampling. Record exact final choices after resource checks, aiming 12 calibration and32 sparse optimizer steps.
- [ ] Precompute both caches with identical NL caption/resize/encoder settings; verify cache provenance and offline completeness. Do not start all285-image processing.
- [ ] Dispatch Luna monitor, launch calibration as bounded process, primary sleeps; on completion inspect finite loss/changed B/frozen A/peak resources/final state. On error capture evidence and dispatch fix instead of continuing blindly.
- [ ] Run CPU mask extraction and verify global threshold/retained counts/A hashes/zero values. Start a NEW sparse trainer and NEW optimizer process; Luna monitors and primary sleeps.
- [ ] Verify completed sparse loss/parameter changes/frozen A/base, exact save/reload and resume artifact. Report phase-specific time/RAM/GPU peaks, config/digests and any limitations.


