## Global Constraints

- Implement only in copied `D:/Develop/KohakuMossTrainer-LoRI/KohakuMossTrainer`; source trainer/model/data remain read-only.
- Machine paths/private data/configs/manifests/caches/outputs stay gitignored under current workspace; future git covers inner source only.
- Follow inner CLAUDE.md: ASCII Python, comments explain what, black/ruff; model-free adapter/algorithm tests are allowed; integration uses actual MOSS weights, never fake toy model classes.
- Preserve exact A across phases; global abs-B quantile and >= threshold ties; zero sparse B and fresh moments.
- Final compact Parameters exist before any optimizer/offload/EMA capture. Do not change parameter shapes during restore.
- No new third-party fork, custom CUDA kernel, unrelated refactor or FP8 experiment.
- User selected subagent execution. All training uses Luna background monitoring; primary sleeps until completion/error. No additional routine permission gate is inferred.


### Task 2: Global calibration mask artifact

**Files:** Create `src/kohakumoss/adapters/train/lori.py`, `scripts/adapters/lori_mask.py`, `tests/adapters/test_lori_calibration.py` (create scripts/adapters only if no better existing export/data command location fits).

**Interfaces:** `read_lori_state(path) -> (dict[str,Tensor],dict[str,str])`; `make_sparse_init(calibration_state,sparsity) -> (dict[str,Tensor],dict[str,str])`; `lori_layer_init(state,lora_name) -> dict[str,Tensor]`. CLI consumes dense calibration raw safetensors and writes self-contained compact raw initialization plus metadata; no GPU required.

- [ ] Write failing pure-function tests: cross-layer global threshold fixture vs per-layer threshold; >= ties; zero/tied scores; nonfinite rejection; invalid sparsity/index/rank/target shapes; preserved A byte equality; zero values; count/metadata correctness; individual zero-nnz layer.
- [ ] Run calibration tests and confirm expected failures.
- [ ] Implement CPU float32 concatenated abs-B quantile and metadata/version/digest handling. Select output-major flattened indices, sorted/unique. Never transfer calibrated values/moments.
- [ ] Run tests and CLI on generated adapter-only fixtures. Reload saved artifact, assert exact tensors/metadata and deterministic output.
- [ ] Commit tested artifact pipeline.


