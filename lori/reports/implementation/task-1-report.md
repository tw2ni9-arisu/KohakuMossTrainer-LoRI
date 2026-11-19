# Task 1 implementation report

Status: implemented and committed in copied inner repository only.

Commit: `f4a2c65a65a3dee7cedb3f1bf5d1fdc630737164`

## Changes and interface

- Added `src/kohakumoss/adapters/kinds/lori.py`, local `LORI` Kind registration via `kinds/__init__.py`, and model-free `tests/adapters/test_lori_math.py`.
- Constructor: `LoRIModule(lora_name, org_module, lora_dim=8, alpha=16.0, *, phase="calibration", a=None, active_indices=None)`.
- `a` optionally supplies an exact projection (detached clone, no reinitialization). Otherwise initialize uniform `[-sqrt(3/in), sqrt(3/in)]` once. Buffer name is uppercase `A`.
- Calibration has sole Parameter `b_dense[out,rank]`. Sparse requires sorted unique int64 flattened B offsets and has sole Parameter `b_values[nnz]`; `active_indices` is persistent. Both have persistent `A` and `alpha` buffers, and zero-initialized B.
- `dense_b()` returns a differentiable scatter reconstruction in sparse mode. `bypass_forward_diff(x,multiplier=1.0)` performs nested linear operations. `get_diff_weight(multiplier=1.0)` returns `(scale*multiplier*B@A, None)`. `scale` reads persistent `alpha/rank`.
- Exposes `lora_name`, `shape`, `lora_dim`, `scale` for Target. Registered Kind uses existing `unit_diff` and `module_bypass`; Comfy tensors expand to ordinary `lora_up.weight=B`, `lora_down.weight=A`, `alpha=alpha`, preserving scale once.
- Dense/sparse reconstruction callable is selected once at construction without a bound-self reference cycle.

## Environment and commands

All Python runs used `D:/Develop/nigenmame-LoRA/KohakuMossTrainer/.venv/Scripts/python.exe`; no source environment packages were changed. `PYTHONDONTWRITEBYTECODE=1` throughout. `PYTHONPATH` starts with copied inner `src`, followed by outer `.tools/python` for tooling.

Import verification printed:
`D:\Develop\KohakuMossTrainer-LoRI\KohakuMossTrainer\src\kohakumoss\__init__.py`.

Initial pytest attempt reported `No module named pytest`; this is an environment issue, not algorithm red evidence. Installed isolated tooling with:

```powershell
uv pip install --python D:/Develop/nigenmame-LoRA/KohakuMossTrainer/.venv/Scripts/python.exe --target D:/Develop/KohakuMossTrainer-LoRI/.tools/python pytest black ruff
$env:PYTHONPATH=(Resolve-Path src).Path+';D:/Develop/KohakuMossTrainer-LoRI/.tools/python'
$env:PYTHONDONTWRITEBYTECODE='1'
```

Tool versions: pytest 9.2.0, black 26.10.1, ruff 0.17.5.

Real RED before implementation:

```text
python -m pytest tests/adapters/test_lori_math.py -q
ModuleNotFoundError: No module named 'kohakumoss.adapters.kinds.lori'
1 error in 0.10s (collection), exit 1
```

Initial GREEN: same command, `22 passed in 0.25s`. Added dense low-precision/gradient cases: `25 passed in 0.28s`. Added negative-case helper rejecting deliberately double-scaled delta.

Final checks (each `python` means the explicit interpreter above):

```text
python -m black --check src/kohakumoss/adapters/kinds tests/adapters
21 files would be left unchanged, exit 0
python -m ruff check src/kohakumoss/adapters/kinds tests/adapters
All checks passed!, exit 0
python -m pytest tests/adapters/test_lori_math.py tests/adapters/test_state_refusal.py tests/adapters/test_stacking.py tests/adapters/test_stored_bypass.py -q
78 passed in 4.90s, exit 0
git diff --check
exit 0
git status --short (after commit)
empty, exit 0
```

Coverage includes A shape/range/frozen status, zero B/delta, sparse empty/partial/full masks, input and value gradients, multipliers 0/.5/1, dtype/device movement with integer indices retained, fp16/bf16 dense gradients, Target delta/module_forward and Comfy conversion, invalid indices, and a broken double-scale negative check. Tests use direct adapter tensors and actual nn.Linear operators, with no fake MOSS/encoder/VAE classes.

## Concerns and scope limits

- No training, model-weight loading, calibration mask computation, network factory integration, restore plumbing, or optimizer/offload implementation was performed. Subsequent tasks own these.
- Native persistence is verified through state keys and dtype/device behavior; native resume/roundtrip across factory construction is deferred as scoped.
- Sparse parameters are compact, but reconstruct full dense B temporarily; no throughput or total-VRAM reduction is claimed. Optimizer state scaling has not yet been measured.
- The constructor checks the linear dimension interface; production factory remains responsible for supported target selection and rejecting unsupported adapter options.
- Source dependency environment was missing dev tooling; isolated outer-workspace tools supply it. Only the three intended inner source/test files were committed.
