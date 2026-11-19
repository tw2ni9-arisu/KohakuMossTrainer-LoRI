## Global Constraints

- Implement only in copied `D:/Develop/KohakuMossTrainer-LoRI/KohakuMossTrainer`; source trainer/model/data remain read-only.
- Machine paths/private data/configs/manifests/caches/outputs stay gitignored under current workspace; future git covers inner source only.
- Follow inner CLAUDE.md: ASCII Python, comments explain what, black/ruff; model-free adapter/algorithm tests are allowed; integration uses actual MOSS weights, never fake toy model classes.
- Preserve exact A across phases; global abs-B quantile and >= threshold ties; zero sparse B and fresh moments.
- Final compact Parameters exist before any optimizer/offload/EMA capture. Do not change parameter shapes during restore.
- No new third-party fork, custom CUDA kernel, unrelated refactor or FP8 experiment.
- User selected subagent execution. All training uses Luna background monitoring; primary sleeps until completion/error. No additional routine permission gate is inferred.


### Task 5: Real low-VRAM compatibility and bounded test gate

**Files:** Extend existing actual-weight integration tests under `tests/training/run/` or adapters suite following CLAUDE.md; add LoRI optimizer assertions to existing offload test harness without fake model classes. Generic repo config may be added under configs/moss/algo; all local paths stay ignored.

**Interfaces:** Existing MOSS weights/profile/TrainStep/optimizer remain owners of training. Tests consume Task3/4 module and restore interfaces.

- [ ] Add failing actual-MOSS integration assertions: valid target set, stored INT8 base bypass, frozen base, finite one-step gradients, recompute, host optimizer+gradient-offload compact update, dtype/device indices, zero-nnz handling.
- [ ] Run bounded weight-aware tests using existing test selection/weight hooks; report skipped weight requirements honestly.
- [ ] Fix only evidenced compatibility issues. Verify moment exp_avg+exp_avg_sq numel=2*nnz excluding scalar steps; no A/index optimizer state; gradient staging sizes and live b_values identities match. Compare resident versus four-slot ring path only if resource budget supports both.
- [ ] Run black/ruff checks as configured plus relevant adapters/checkpoint/offload regressions once; rerun only when changes/failures justify it. Record commands and results.
- [ ] Commit verified source before full-size experiment. No success claim from skipped tests.


