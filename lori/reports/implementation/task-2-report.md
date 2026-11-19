# Task 2 implementation report

Status: implemented and committed in copied inner repository only.
Commit: `46d84ada69638b2a7eb78450eb1663b4019c772a`.

## Scope and algorithm

Added only:
- `src/kohakumoss/adapters/train/lori.py`
- `scripts/adapters/lori_mask.py`
- `tests/adapters/test_lori_calibration.py`

`make_sparse_init(calibration_state, sparsity)` validates raw dense LoRI layers, concatenates all absolute B values on CPU in float32, computes one `torch.quantile`, and selects `>= threshold`. It preserves ties, including all-zero scores, and permits individual empty layers. Sparsity is finite in `[0,1)`; requested sparsity does not imply an exact nnz count. Sorted unique int64 offsets address output-major flattened B[out,rank]. Output B values are zero with calibration B dtype; A and alpha are exact detached clones. No optimizer state is accepted or transferred.

Native raw names remain `<lora_name>.A`, `.alpha`, `.b_dense` for calibration; output uses `.A`, `.alpha`, `.active_indices`, `.b_values`. There are no auxiliary tensor keys that would break exact native state loading.

`read_lori_state(path)` returns CPU raw tensors and string metadata. Dense raw state can omit all new metadata and derives shape from A/B. Sparse state requires versioned metadata and checks target dimensions, rank, scale, count, sorted unique in-range indices, finite tensors, and tensor digests. Unexpected raw state keys and mixed phases are refused.

`lori_layer_init(state, lora_name)` returns constructor kwargs `a`, `alpha`, `lora_dim`, and (sparse only) `active_indices`. It never copies trained B into the constructor. `phase` remains the factory's decision. Sparse output size must also be checked against the actual target by the factory: nnz alone cannot supply it. Returned alpha is a scalar tensor accepted by the module's float conversion; lora_dim is a Python integer despite the requested broad dict-of-Tensor interface annotation.

## Metadata contract for Tasks 3/4

`META_KEY = "kohakumoss_lori"`, `VERSION = 1`. Its string value is sorted compact JSON:

- `version`: 1
- `phase`: `sparse_init` for fresh initialization, `sparse` for trained native state
- `sparsity`, `threshold`, `total` (all dense B elements), `nnz`, `actual_density`
- `calibration_digest`: SHA256 of a sorted JSON map of all calibration raw keys to exact tensor digests (provenance; original dense B is not embedded)
- `layers`: mapping exact raw lora_name -> `target_shape` [out,in], `rank`, `alpha`, `scale`, `nnz`, `a_digest`, `alpha_digest`, `indices_digest`

Tensor digest hashes an ASCII dtype/shape prefix plus contiguous exact tensor bytes. Calibration digest changes with calibrated values and does not depend on dictionary order. Read validates digest syntax but cannot recompute the calibration source digest without the original dense calibration; it verifies A/alpha/index digests against embedded tensors.

The CLI preserves existing non-LoRI metadata (`kohakumoss_adapter_state`, `kohakumoss_spec`, fold metadata, etc.) and replaces only META_KEY. The preserved calibration spec is provenance, not an instruction to use calibration phase for sparse construction; factory/current config owns phase selection.

Native trained sparse save can reuse the initialization metadata after parsing META_KEY JSON and changing only `phase` from `sparse_init` to `sparse`. Frozen A/alpha/indices and all shapes/counts stay identical, so no digest logic needs duplication. B is not hashed: sparse_init rejects any nonzero B; sparse permits finite trained B. Task4 should retain this metadata within each native checkpoint. Restore then needs only checkpoint raw tensors and metadata, with no original mask_path.

CLI: `python scripts/adapters/lori_mask.py CALIBRATION.safetensors INITIALIZATION.safetensors --sparsity 0.9`. No sys.path modification, GPU, model, or optimizer required. Input/output same path is refused. The CLI rereads its output through the strict validator.

## Verification evidence

All Python commands used `D:/Develop/nigenmame-LoRA/KohakuMossTrainer/.venv/Scripts/python.exe`, with `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH` beginning with copied inner `src`, followed by outer `.tools/python`. Source environment packages were not modified.

Real RED before production implementation:
`python -m pytest tests/adapters/test_lori_calibration.py -q`
failed collection with `ModuleNotFoundError: No module named 'kohakumoss.adapters.train.lori'`, exit 1.

The initial implementation's 25 cases passed their assertions but the CLI roundtrip case exceeded the 10-second per-test budget because it ran two new Python processes (11.3s). Split those runs into separate parameterized cases; did not raise or disable the budget.

Subsequent RED for the native phase contract: `-k sparse_init_requires` failed `DID NOT RAISE ValueError` on nonzero initialization B. Added the zero-B check; trained phase accepts unchanged compact values.

Subsequent RED for calibration provenance: `-k calibration_digest` failed `KeyError: calibration_digest`. Added deterministic calibration source digest generation/validation.

Final checks:
- `python -m black --check src/kohakumoss/adapters/train/lori.py scripts/adapters/lori_mask.py tests/adapters/test_lori_calibration.py`: 3 files unchanged, exit 0.
- `python -m ruff check` same three files: all checks passed, exit 0.
- `python -m pytest tests/adapters/test_lori_calibration.py tests/adapters/test_lori_math.py tests/adapters/test_state_refusal.py -q`: **70 passed in 12.03s**, exit 0 (41 calibration cases and 29 related cases).
- CLI executes against generated adapter-only raw fixtures in two independent subprocess cases. Reloaded metadata/tensors exactly equal direct helper output.
- Pure negative helper proves the global fixture rejects deliberately substituted per-layer quantiles.
- Explicit ASCII source scan and `git diff --cached --check`: pass.
- Post-commit `git status --short`: empty.

## Concerns and deferred work

No real model, training, source trainer edits, model/data writes, factory changes, restore changes, or optimizer/offload changes were performed. Sparse artifacts are self-contained; factory must still verify actual selected target sets/shapes before construction. Seed is not invented: exact A and its digest carry projection identity. Global quantile materializes concatenated CPU float32 B scores as required; no total-VRAM or throughput claims are made. Metadata carries threshold/count/provenance evidence, not a proof of calibration quality. Follow-up tasks own parameter allocation ordering and native checkpoint metadata retention.
