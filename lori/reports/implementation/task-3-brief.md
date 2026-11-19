## Global Constraints

- Implement only in copied `D:/Develop/KohakuMossTrainer-LoRI/KohakuMossTrainer`; source trainer/model/data remain read-only.
- Machine paths/private data/configs/manifests/caches/outputs stay gitignored under current workspace; future git covers inner source only.
- Follow inner CLAUDE.md: ASCII Python, comments explain what, black/ruff; model-free adapter/algorithm tests are allowed; integration uses actual MOSS weights, never fake toy model classes.
- Preserve exact A across phases; global abs-B quantile and >= threshold ties; zero sparse B and fresh moments.
- Final compact Parameters exist before any optimizer/offload/EMA capture. Do not change parameter shapes during restore.
- No new third-party fork, custom CUDA kernel, unrelated refactor or FP8 experiment.
- User selected subagent execution. All training uses Luna background monitoring; primary sleeps until completion/error. No additional routine permission gate is inferred.


### Task 3: Existing trainer configuration and target construction

**Files:** Modify `adapters/build/config.py`, `build/network.py`, `build/apply.py`, `kinds/__init__.py`, relevant `scripts/train/moss.py` ALGO comment/knob validation if needed; create `tests/adapters/test_lori_build.py` and extend existing knob tests.

**Interfaces:** Existing `apply(model,config,dtype,device,initial_state=None)` and `build_targets(model,config,initial_state=None)` accept optional raw state; standard callers are unchanged. LoRI phase/sparsity/mask_path live in ALGO_KWARGS. Native target map matches saved state.

- [ ] Write failing config tests for valid lori and clear rejection of unsupported DoRA/dropout/norm/Muon/LoRA+ combinations. Verify ordinary lora config unchanged. Write direct parameter-group test showing only compact values and intended decay/LR.
- [ ] Run tests, confirm expected failures.
- [ ] Build LoRI through existing standard-LoRA preset discovery then immediate module replacement before Target creation. Read mask state once; do not pass local-only kwargs to LycorisNetwork. Pass initial_state tensors before _place and before engines.exclude_params/build_optim.
- [ ] Verify existing adapter build/config/state and optimizer-group regressions; real target matching verification belongs to Task5. Initial recipe uses decay0; explicit nonzero-decay test uses WD_RULES=all.
- [ ] Commit integration.


