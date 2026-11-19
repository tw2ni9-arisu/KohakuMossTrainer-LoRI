## Global Constraints

- Implement only in copied `D:/Develop/KohakuMossTrainer-LoRI/KohakuMossTrainer`; source trainer/model/data remain read-only.
- Machine paths/private data/configs/manifests/caches/outputs stay gitignored under current workspace; future git covers inner source only.
- Follow inner CLAUDE.md: ASCII Python, comments explain what, black/ruff; model-free adapter/algorithm tests are allowed; integration uses actual MOSS weights, never fake toy model classes.
- Preserve exact A across phases; global abs-B quantile and >= threshold ties; zero sparse B and fresh moments.
- Final compact Parameters exist before any optimizer/offload/EMA capture. Do not change parameter shapes during restore.
- No new third-party fork, custom CUDA kernel, unrelated refactor or FP8 experiment.
- User selected subagent execution. All training uses Luna background monitoring; primary sleeps until completion/error. No additional routine permission gate is inferred.


### Task 4: Portable native restore, inference and export

**Files:** Modify `training/entry.py`, `training/build/model.py` and checkpoint-reader helper only as needed for early trainable initialization; modify `adapters/infer/inference.py`, native state metadata/export helpers if needed; create `tests/adapters/test_lori_state.py`, extend checkpoint tests.

**Interfaces:** Training resolves LoRI trainable initialization from mask artifact for a fresh run OR checkpoint trainable tensors for resume before module creation. Regular parts.restore still restores values/moments in place. Inference raw construction receives loaded raw tensors. AdapterStack._build detects native LoRI before third-party get_module.

- [ ] Write failing model-free tests: native state roundtrip output/A/index equality; save then remove original artifact path and reload; wrong version/target/rank rejected; compact Parameter ids established before optimizer and unchanged by load; continuation matches uninterrupted AdamW reference.
- [ ] Run tests, confirm failures; inspect existing CheckpointManager/io helpers to avoid reading full optimizer twice.
- [ ] Implement early LoRI tensor-only initialization and local native inference branch; strict copying never replaces parameter identity after capture. Native output does not falsely advertise unsupported third-party native LyCORIS loading.
- [ ] Verify native, Comfy-LoRA conversion and dense delta agree at multipliers0/.5/1, including fold/row-scale existing conventions. Run existing state/export/checkpoint tests affected by optional initial_state signature.
- [ ] Commit restore/export changes.


