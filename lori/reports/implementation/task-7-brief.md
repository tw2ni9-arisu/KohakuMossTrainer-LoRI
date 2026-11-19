## Global Constraints

- Implement only in copied `D:/Develop/KohakuMossTrainer-LoRI/KohakuMossTrainer`; source trainer/model/data remain read-only.
- Machine paths/private data/configs/manifests/caches/outputs stay gitignored under current workspace; future git covers inner source only.
- Follow inner CLAUDE.md: ASCII Python, comments explain what, black/ruff; model-free adapter/algorithm tests are allowed; integration uses actual MOSS weights, never fake toy model classes.
- Preserve exact A across phases; global abs-B quantile and >= threshold ties; zero sparse B and fresh moments.
- Final compact Parameters exist before any optimizer/offload/EMA capture. Do not change parameter shapes during restore.
- No new third-party fork, custom CUDA kernel, unrelated refactor or FP8 experiment.
- User selected subagent execution. All training uses Luna background monitoring; primary sleeps until completion/error. No additional routine permission gate is inferred.


### Task 7: Base/adapter comparison and delivery

**Files:** Ignored compare_grid configuration and generated grid/singles/metadata; outer workspace final report.

**Interfaces:** Existing scripts/sample/compare_grid.py consumes native/portable saved adapter. Use trained multiplier1 and matched multiplier0/base, same precision/base/sampler/seed.

- [ ] Prepare four nonexplicit prompts, paired with/without nigenmame, seeds42/43 as budget permits, same resolution/steps/CFG/shift. Reuse compare_grid rather than add sampling infrastructure.
- [ ] Run bounded inference with monitoring, inspect saved image/grid and metadata, confirm no adapter contamination between columns.
- [ ] Compare outputs and report evidence: nonzero adapter effect, whether style learning is visibly plausible, failures/artifacts/weak effects. Do not claim quality parity/generalization from this small smoke experiment.
- [ ] Deliver base and adapter images, artifact links, exact source revision/config/checkpoint references, tests and measured resources. No publishing/pushing is included without separate user instruction.


