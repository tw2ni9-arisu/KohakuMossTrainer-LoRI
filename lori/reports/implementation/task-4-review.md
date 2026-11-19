# Task 4 independent review

Reviewed baseline `486dfa2b88abcac71b68c3b36fdbf12d2e916ac4` to commit `623629b39ef01be23b6d28ec2aad95a7ddb35c73`.

## Verdict

- **Spec: changes required.** Portable native sparse state, early tensor-only sizing, metadata validation and in-place restore are implemented. The required fold/dense-delta/export contract fails when the supported trainable master dtype differs from the frozen projection dtype.
- **Quality: changes required.** One Important regression is independently reproduced. No Critical or additional Minor findings.

## Findings

### Important / P1: preserved projection dtype breaks low-precision delta and folded export

Location: `src/kohakumoss/adapters/build/apply.py:32-34`; affected consumer: `src/kohakumoss/adapters/kinds/lori.py:94-96`.

The new placement logic casts trainable B to the requested master dtype, then restores A to its original dtype. With a normal FP32 calibration projection and FP16/BF16 master parameters, B and A consequently have different dtypes. `get_diff_weight` still computes `self.dense_b() @ self.A` directly. Outside autocast this raises a RuntimeError instead of producing a delta. This is a regression introduced by preserving buffer dtype without updating its consumer.

An independent CPU/model-free probe, using the existing `dense_state`, `operators` and `config` helpers and `make_sparse_init`, confirmed both cases:

Reproduction command (PowerShell, from the copied inner repository):

```powershell
$env:PYTHONPATH='D:\Develop\KohakuMossTrainer-LoRI\KohakuMossTrainer\src;D:\Develop\KohakuMossTrainer-LoRI\.tools\python'
$env:PYTHONDONTWRITEBYTECODE='1'
@'
import torch
from kohakumoss.adapters.build.apply import apply
from kohakumoss.adapters.train.lori import make_sparse_init
from tests.adapters.test_lori_build import config, dense_state, operators
state, meta = make_sparse_init(dense_state(), 0.6)
for dtype in (torch.float16, torch.bfloat16):
    aset = apply(operators(), config(algo_kwargs={'phase':'sparse'}), dtype=dtype, initial_state=state, initial_metadata=meta)
    module = aset.targets[0].lyco
    print('master',dtype,'A',module.A.dtype,'B',module.b_values.dtype)
    for name, call in [('delta',lambda:module.get_diff_weight()), ('folded comfy',lambda:aset.tensors('comfy',row_scales={'0':torch.tensor([0.5,1.,2.])}))]:
        try:
            call()
            print(name,'OK')
        except Exception as error:
            print(name,type(error).__name__,str(error))
'@ | & 'D:\Develop\nigenmame-LoRA\KohakuMossTrainer\.venv\Scripts\python.exe' -
```

Output (LyCORIS discovery logs omitted):

```
master torch.float16 A torch.float32 B torch.float16
delta RuntimeError expected m1 and m2 to have the same dtype, but got: struct c10::Half != float
folded comfy RuntimeError expected m1 and m2 to have the same dtype, but got: struct c10::Half != float
master torch.bfloat16 A torch.float32 B torch.bfloat16
delta RuntimeError expected m1 and m2 to have the same dtype, but got: struct c10::BFloat16 != float
folded comfy RuntimeError expected m1 and m2 to have the same dtype, but got: struct c10::BFloat16 != float
```

The folded-export probe called `aset.tensors('comfy', row_scales={'0': torch.tensor([0.5, 1., 2.])})`. LoRI has no row-scale factor callback, so the existing converter falls back to `scaled_diff` and reaches this failing matrix multiplication. The real export caller `training/exports.py:135` supplies the backbone's row scales through this same contract. Dense delta/merge consumers likewise use `get_diff_weight`. This is an actual callable-contract failure, independent of the deferred real-MOSS integration work.

Fix by using a common working dtype when materializing the delta, preserving the stored A bytes and B Parameter identity. Add focused coverage combining FP16/BF16 master parameters with nonuniform fold scales and delta/merged export. The new preservation test (`tests/adapters/test_lori_state.py:144`) only writes the native format, while the fold/delta test (`:169`) only uses FP32 master parameters; neither covers this combination.

## Confirmed by source inspection

- `load_lori_initialization` reads only native trainable tensors/metadata; it never opens the optimizer. The regular checkpoint loader remains responsible for the optimizer read. Early state is released after model construction rather than retaining an optimizer copy.
- The entry/build call chain supplies native initialization before trainable exclusion, optimizer/EMA capture and ordinary restore. Sparse shapes are established during construction; the restore implementation checks all tensor shapes before copying any values and does not replace compact Parameters.
- The native validator checks version, phase, full target shapes, rank, global counts and exact A/alpha/index digests. The factory checks discovered names and actual target dimensions before placement. Sparse native metadata is embedded in saved files and explicit initialization takes precedence over a stale mask path.
- The native inference branch precedes third-party module discovery. Raw inference passes loaded state and metadata into construction. Native export identifies itself as local KohakuMoss state without claiming third-party native LyCORIS support; Comfy remains the conversion path.
- Native compact row unfolding changes B only, preserves fixed projection/index state, and marks the unfolded file's fold as null.

## Evidence and limits

Read the task brief, implementation report and complete change package, then inspected only unchanged helpers needed to resolve specific precision, metadata, export and checkpoint-call risks. The reviewed inner HEAD was confirmed as `623629b39ef01be23b6d28ec2aad95a7ddb35c73`; git status was clean. No source, HEAD or index changes were made, and no training or subagents were launched.

The reported final 194-case suite and earlier 345-case suite were not rerun; they are implementation-owner evidence, not independent review execution. Only the concrete dtype-risk probe above was executed, using the existing external interpreter with copied source first on PYTHONPATH and bytecode writing disabled. It loaded no model weights and wrote no artifacts.

Real MOSS targets, offload behavior, GPU peak memory, training quality and speed are not verified here and remain Task 5/6 work, not additional Task 4 defects. Compiler/TCC diagnostics are covered by the separate existing diagnosis report and were not repeated.
