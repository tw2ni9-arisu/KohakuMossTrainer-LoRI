# Task 3 implementation report

Status: implemented and committed in the copied inner repository only.
Commit: `486dfa2b88abcac71b68c3b36fdbf12d2e916ac4` (baseline `46d84ada69638b2a7eb78450eb1663b4019c772a`).

## Changes and interfaces

- `apply(model, config, dtype=torch.float32, device=None, initial_state=None)` and `build_targets(model, config, initial_state=None)` accept optional raw tensor dictionaries. Existing callers remain valid.
- LoRI retains the existing Kind mapping to standard `lora`. The same LycorisNetwork and resolved preset discover native names and target operators; each discovered module is replaced with local LoRIModule immediately before weak binding and Target construction. No third-party edits or second injection engine.
- `ALGO_KWARGS` for LoRI accepts exactly `phase`, `sparsity`, `mask_path`. Phase defaults to calibration; sparsity defaults to 0.9 for validation and must be finite in [0,1). Factory does not recalculate masks: Task 2 owns actual quantile selection.
- LoRI rejects DoRA, DORA_DETACH, input/rank/module dropout, TRAIN_NORMS and non-unit LORAPLUS_RATIO. `training/knobs.py` required a minimal additional edit: add lori to its static ALGOS and reject LoRI + muon before construction. The script ALGO comment now lists lori.
- `config_from_knobs` retains LoRI `wd_on_out` in ALGO_KWARGS so it is explicitly rejected, rather than silently popped as a DoRA setting. Ordinary algorithms preserve existing consumption and module kwargs.
- Explicit initial_state takes precedence and never reads mask_path, including a stale/nonexistent path. Without initial_state, sparse phase requires mask_path and reads it once through Task 2 read_lori_state, including strict metadata/digest validation. Its metadata target shapes are compared to actual discovered targets.
- Target sets, rank, A input width, dense B output shape, sparse indices/values and phase are validated. Supplied A and alpha are cloned from state; constructor B remains zero. Existing load_state_dict later copies trained B into the already-sized Parameter without replacing it.
- Compact b_values exist before `_place`, and apply returns them before existing engine exclusion and optimizer construction. No engine, optimizer, EMA or checkpoint code was modified.
- kinds/__init__.py already registered the LoRI Kind in Task 1 and required no additional change.

## Verification

All Python commands used:
`D:/Develop/nigenmame-LoRA/KohakuMossTrainer/.venv/Scripts/python.exe`.
Environment: `PYTHONDONTWRITEBYTECODE=1`; PYTHONPATH begins with `D:/Develop/KohakuMossTrainer-LoRI/KohakuMossTrainer/src`, followed by `D:/Develop/KohakuMossTrainer-LoRI/.tools/python`. No package installation or source dependency environment edits.

TDD RED before production changes:

```
python -m pytest tests/adapters/test_lori_build.py tests/training/knobs/test_knob_rules.py -q
18 failed, 12 passed in 19.78s; exit 1
```

Failures included missing optional initial_state, missing LoRI ALGOS registration, discovery still creating LoRA modules, and unsupported config options being accepted. First GREEN: 30 passed in 11.31s.

Additional RED for silent DoRA argument consumption:

```
python -m pytest tests/adapters/test_lori_build.py -q -k dora_network
1 failed, 21 deselected in 0.09s; exit 1
```

Final relevant regression command:

```
python -m pytest tests/adapters/test_lori_build.py tests/adapters/test_lori_calibration.py tests/adapters/test_lori_math.py tests/adapters/test_state_refusal.py tests/adapters/test_stacking.py tests/adapters/test_stored_bypass.py tests/training/knobs/test_knobs.py tests/training/knobs/test_knob_rules.py -q
157 passed in 29.49s; exit 0
```

Coverage uses actual nn.Linear operators in an nn.Sequential container, adapter-only state, and pure knob validation; no fake MOSS/encoder/VAE classes. It covers exact native target names, ordinary LoRA discovery, initial_state precedence, zero compact initialization followed by in-place trained B load, one artifact read, exact A/alpha, target/rank/shape/phase/index refusal, and metadata output-shape mismatch. Parameter groups contain only compact Parameters at intended LR. The default moss WD rule exempts 1D values; initial recipe uses zero decay. Explicit nonzero decay uses `rules="all"`; compact AdamW updates equal selected entries of dense AdamW updates. A constructed group containing frozen A is rejected by the same group identity/shape check.

Final formatting/lint checks on all seven changed files: black --check (7 unchanged), ruff check (all checks passed), explicit ASCII source scan, git diff --check, git diff --cached --check: exit 0. Post-commit git status --short: empty.

## Broader environment failure (not a complete environment pass)

Broader command:
`python -m pytest tests/adapters/test_lori_build.py tests/training/knobs -q`
result: 86 passed, 1 failed in 12.89s, exit 1. The failure was existing `tests/training/knobs/test_entry_features.py::test_doctor_json`, whose doctor report probes actual CUDA kernel availability. It reached Triton driver compilation and failed before any LoRI path.

Single-test reproduction, same interpreter/PYTHONPATH/environment:
`python -m pytest tests/training/knobs/test_entry_features.py::test_doctor_json -q`
result: 1 failed in 0.71s, pytest exit 1. Combined stdout/stderr saved to:
`D:/Develop/KohakuMossTrainer-LoRI/docs/execution/dit-lori/task-3-doctor-failure.log`.

Actual selected compiler:
`C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe`.
The logged compile command uses `/std:c11`, MSVC 14.29 include/lib directories, Windows SDK 10.0.19041.0, the source venv Triton nvidia driver.c/include/lib, and Python 3.12 include/libs. It fails at Triton `driver.c(13)` with fatal error C1083, cannot open `stdalign.h`. The log contains the full compiler argv and traceback. This is evidence of the current CUDA probe toolchain failing, not a claim that all GPU prerequisites work. No compiler/environment/unrelated source changes were attempted. Parent will diagnose separately.

## Concerns and Task 4 handoff

- No real training or real MOSS target matching was run; Task 5 owns real target verification. No total VRAM or speed claim.
- Raw sparse state contains no original output dimension; A gives rank/input width and indices give a lower bound only. Factory validates the actual target rank/input and every index against actual out*rank, but cannot prove original full output shape from raw tensors alone. Dense raw state and mask artifact metadata do provide full output dimensions and are checked.
- Task 4 must keep versioned LoRI metadata in native checkpoints, call read_lori_state(checkpoint_path) for strict digest/phase/count validation before passing raw state, and compare metadata `layers[name].target_shape` with the actual target shape (and target key map) before `_place` / optimizer/offload capture. A natural integration entry is the resume/native loader immediately before apply(... initial_state=raw_state); expose/check the validated metadata shape map at target construction rather than trusting legal indices as proof of complete shape equality. Do not route resume back through old mask_path. This task intentionally keeps the requested raw-only initial_state interface.
- Current factory uses supplied state only to size/initialize fixed tensors and final Parameters; Task 4 must still call existing load_state_dict to copy trained B. It must construct from final checkpoint state before engine exclude_params/build_optim capture and preserve A/alpha/indices metadata. Task 4 also owns inference/export native restore phase selection.
- Strict artifact reads do not transfer optimizer state. Sparse B is zero at construction; fresh sparse run moments are naturally fresh when its optimizer is built.
- Existing sparse math temporarily reconstructs dense B; compact optimizer storage does not imply sparse compute or lower activation memory.

Compiler environment follow-up: the PowerShell startup command set only PYTHONPATH and PYTHONDONTWRITEBYTECODE. A fresh exec environment query of CC, CXX, VCINSTALLDIR, VCToolsVersion, VCToolsInstallDir, VSINSTALLDIR, CUDA_PATH, INCLUDE, LIB, LIBPATH returned no variables set. No test-launch helper modified those shell values. Python import-time toolchain initialization may select MSVC; the separate compiler diagnostic agent is investigating that possibility.
