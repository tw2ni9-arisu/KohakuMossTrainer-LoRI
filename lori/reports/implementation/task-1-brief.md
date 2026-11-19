## Global Constraints

- Implement only in copied `D:/Develop/KohakuMossTrainer-LoRI/KohakuMossTrainer`; source trainer/model/data remain read-only.
- Machine paths/private data/configs/manifests/caches/outputs stay gitignored under current workspace; future git covers inner source only.
- Follow inner CLAUDE.md: ASCII Python, comments explain what, black/ruff; model-free adapter/algorithm tests are allowed; integration uses actual MOSS weights, never fake toy model classes.
- Preserve exact A across phases; global abs-B quantile and >= threshold ties; zero sparse B and fresh moments.
- Final compact Parameters exist before any optimizer/offload/EMA capture. Do not change parameter shapes during restore.
- No new third-party fork, custom CUDA kernel, unrelated refactor or FP8 experiment.
- User selected subagent execution. All training uses Luna background monitoring; primary sleeps until completion/error. No additional routine permission gate is inferred.


### Task 1: Local LoRI module and mathematical contract

**Files:** Create `src/kohakumoss/adapters/kinds/lori.py`; modify `kinds/__init__.py`; create `tests/adapters/test_lori_math.py`.

**Interfaces:** LoRIModule compatibility with Target/module_bypass: lora_name, shape, lora_dim, scale, bypass_forward_diff(x,multiplier). Calibration contains b_dense; sparse contains b_values and persistent A/active_indices/alpha buffers. `dense_b() -> Tensor`, native state and `get_diff_weight(multiplier=1.0)` expose exact semantics. Kind.diff_weight/module_forward/comfy_tensors are registered locally.

- [ ] Write failing model-free tests: A shape/range/frozen, B zero, zero diff; sparse vs explicit masked B outputs/input gradients/value gradients; scale applied once at multipliers0/.5/1; empty/full indices; A/index device/dtype behavior. Use direct adapter tensors/linear operations, no fake MOSS class.
- [ ] Run `python -m pytest tests/adapters/test_lori_math.py -q`, confirm expected failure.
- [ ] Implement minimal dense calibration and compact sparse branches; differentiable scatter reconstructs B, nested F.linear computes diff. Comfy conversion expands to ordinary LoRA and preserves scale.
- [ ] Rerun math tests and compare converted tensors/dense delta. Confirm named_parameters contains only phase-appropriate B and no A/indices.
- [ ] Commit only after repo exists and checks pass; preserve license/header style.


