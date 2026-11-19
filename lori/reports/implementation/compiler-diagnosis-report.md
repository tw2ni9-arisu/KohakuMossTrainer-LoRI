# Compiler diagnosis: Triton C launcher on Windows

Date: 2026-11-17. Scope: static environment/source inspection, exactly one 16-element CUDA smoke check, and one explicitly requested regression of the original pytest case; no model training, trainer-code changes, source-venv writes, system changes, downloads, cache deletion, or process termination.

## Result

A process-local `CC` selecting the TinyCC already bundled in the source venv resolves the observed `stdalign.h` failure. The existing project Triton `_status.check(cuda:0)` completed and validated its DOUBLE result, exit code 0. The parent subsequently explicitly requested one regression of the original doctor JSON pytest case: it also passed (1 passed in 0.78s, exit code 0). These checks do not establish every backend or training workload works.

Evidence: `task-3-doctor-failure.log` (original failure), `compiler-probe.log` (successful check, environment before/after setup), `compiler-doctor-regression.log` (original failing-case regression), and temporary `.tools/compiler-diagnosis/probe.py` plus its private caches.

## Root cause and selection chain

1. Original test command, supplied by its execution owner, ran from the inner repository with only `PYTHONPATH=<copy>/src;<workspace>/.tools/python` and `PYTHONDONTWRITEBYTECODE=1`, using source-venv Python and `-m pytest tests/training/knobs/test_entry_features.py::test_doctor_json -q`. Its parent shell had no CC, CXX, VCINSTALLDIR, VCToolsVersion, INCLUDE, or LIB. The failure was not caused by an explicit shell compiler override.
2. `src/kohakumoss_kernels/backends/triton/__init__.py` calls `ensure_toolchain()`. `hardware/toolchain.py::_windows_toolchain` discovers Visual Studio, runs vcvars64, applies the compiler/include/lib environment, and sets CXX to cl.EXE. This behavior is documented in `docs/guides/install-windows.md`, Compiler setup.
3. Installed Triton `runtime/build.py::_find_compiler('c')` checks CC first, then environment-discovered MSVC, then bundled TinyCC. After project setup it therefore selects installed MSVC 14.29.30133 rather than the bundled compiler.
4. The full original log shows cl.exe compiling `triton/backends/nvidia/driver.c` with `/std:c11`. Its explicit include paths are Triton NVIDIA include (twice), temporary build directory, Python312/Include, MSVC14.29/include, and Windows SDK10.0.19041.0 shared/ucrt/um. Compilation fails at driver.c:13, `#include <stdalign.h>`, with C1083. The installed MSVC include directory was explicitly checked and has no stdalign.h. This is the first fatal error; the SDK winbase.h warning is not the failure.
5. Static inventory found only MSVC14.29.30133 under the installed VS2019 BuildTools and only Windows SDK10.0.19041.0. No VS installation was present under the standard 64-bit Program Files Visual Studio root. Crucially, installed Triton already supplies `runtime/tcc/tcc.exe`, libtcc.dll, include/stdalign.h and libraries. No download or upgrade is needed for this remedy.

Single tested hypothesis: forcing CC to the existing bundled TinyCC bypasses the incompatible MSVC C-launcher selection while allowing the project to preserve its normal MSVC C++ setup. No header shims or third-party patches were introduced.

## Reproduction command

Run in a fresh PowerShell child process. Set CC before importing project backends or making the first Triton `_find_compiler('c')` call; that function caches its selection. Do not change CXX. For subsequent execution the owner should use a new private run-cache directory instead of reusing this diagnostic directory.

```powershell
$taskRoot = 'D:\Develop\KohakuMossTrainer-LoRI\.tools\compiler-diagnosis'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = 'D:\Develop\KohakuMossTrainer-LoRI\KohakuMossTrainer\src;D:\Develop\KohakuMossTrainer-LoRI\.tools\python'
$env:CC = 'D:\Develop\nigenmame-LoRA\KohakuMossTrainer\.venv\Lib\site-packages\triton\runtime\tcc\tcc.exe'
$env:TRITON_CACHE_DIR = Join-Path $taskRoot 'triton-cache'
$env:TRITON_DUMP_DIR = Join-Path $taskRoot 'triton-dump'
$env:TRITON_OVERRIDE_DIR = Join-Path $taskRoot 'triton-override'
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $taskRoot 'inductor-cache'
$env:KOHAKUMOSS_KERNEL_CACHE = Join-Path $taskRoot 'kernels-cache'
$env:CUDA_CACHE_PATH = Join-Path $taskRoot 'cuda-cache'
$env:TEMP = Join-Path $taskRoot 'temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
& 'D:\Develop\nigenmame-LoRA\KohakuMossTrainer\.venv\Scripts\python.exe' (Join-Path $taskRoot 'probe.py')
```

The existing probe prints selected environment fields before setup, calls ensure_toolchain, prints the result and post-setup environment, prints `_find_compiler('c')`, torch/Triton versions and device, then calls the project's `_status.check(torch.device('cuda:0'))` exactly once and synchronizes.

## Actual verified environment and outcome

- Python: source `.venv/Scripts/python.exe`, CPython 3.12 ABI (cp312 in original build artifacts).
- torch: 2.14.1+cu130; Triton module: 3.8.0; installed distribution metadata: triton-windows 3.8.0.post29.
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU; CUDA available True.
- Before setup: CC points to bundled tcc.exe; CXX, VCINSTALLDIR, VCToolsVersion, INCLUDE, LIB and Windows SDK fields unset.
- ensure_toolchain: ok=True, source=vcvars; CXX becomes `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64\cl.EXE`; VCToolsVersion=14.29.30133; WindowsSDKVersion=10.0.19041.0.
- After setup: CC remains exactly the explicit bundled tcc.exe path. `_find_compiler('c')` returns that tcc.exe. Thus project setup does not override this solution.
- Fresh workspace Triton cache produced compiled `cuda_utils.cp312-win_amd64.pyd` and DOUBLE kernel artifacts (`.cubin`, `.ptx`, `.llir`, `.ttir`, `.ttgir`, JSON/source metadata).
- Final output: `PASS: project Triton 16-element DOUBLE smoke check`; process exit code 0. The standalone smoke was run once. The original failing pytest case was then run once on explicit parent request; no model training occurred.

## Handoff

The execution owner can apply the same CC-only compiler override in its private launcher before Python starts, retain source-venv read-only and per-run workspace caches, and capture resolved CC/CXX/toolchain environment after ensure_toolchain. The original doctor test has now passed under that environment. It was the only additional test run, explicitly authorized by the parent after the smoke passed. Any later C++/torch.compile failure needs its own evidence: this result establishes Triton CUDA C launcher availability only.

## Original failing-case regression

With the same CC, PYTHONPATH and cache environment as above, from `D:\Develop\KohakuMossTrainer-LoRI\KohakuMossTrainer`, executed exactly once:

```powershell
& 'D:\Develop\nigenmame-LoRA\KohakuMossTrainer\.venv\Scripts\python.exe' -m pytest tests/training/knobs/test_entry_features.py::test_doctor_json -q -p no:cacheprovider --basetemp 'D:\Develop\KohakuMossTrainer-LoRI\.tools\compiler-diagnosis\doctor-pytest-temp'
```

The pytest cache provider was disabled and its temporary output redirected to a new workspace-owned directory. stdout/stderr and exit code were saved to `compiler-doctor-regression.log`:

```text
.                                                                        [100%]
1 passed in 0.78s
exit_code=0
```

This closes the initially observed doctor_json failure with the process-local CC override; no source or dependency patch was required. The standalone smoke verifies actual CUDA compilation, launch and tensor result; the pytest result separately verifies the original test now completes and accepts its doctor JSON output.

