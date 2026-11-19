# LoRI 导出自描述改造：独立审计（Task 13）

**结论先行。** 需求已满足，**无阻断项，可放行到 90% 稀疏 900 步重训**。两种格式现在都能自描述：`phase / mechanism / trainable / a=frozen_buffer / a_sha256` 全部落盘，sparse 另带 `kept_total` 与 `rank`，comfy 另带 `kohakumoss_lori_comfy=inference_only_dense_b` 且稀疏导出会打警告；dense calibration 与 sparse LoRI 可区分；原生文件保留 mask、comfy 只在容器层展开。

证据链：真实模型 3 个检查点 × 2 格式共 **6 个导出**，键集合 / 元数据 / `a_sha256`（独立复算）/ 与旧导出的逐位一致 / comfy 的 A 与 B 映射 / 警告次数全部符合预期；训练 2 步（dense、sparse90）与 resume（1 步、旧格式检查点）回归通过；测试 15 passed（新文件）、299 passed（GPU 全套）；black/ruff 通过。

非阻断发现 5 条（1×P2、4×P3），见第 5 节；其中 F1 建议在 900 步长跑前处理。

结构化证据：[lori-export-audit.json](../../../experiment/task13-audit/lori-export-audit.json)。

## 1. 改动与代码审查

工作树基于 HEAD `957ccc3`，8 个文件 +157/−5，另有未跟踪的 `tests/adapters/test_lori_export_facts.py`。diffstat 记为 5 处删除，逐行核对实际为 6 行删除，**全部非行为**：`lori.py` 模块 docstring 1 行、`adapter_set.py` 的 `import META_KEY` 1 行与 `state_metadata` docstring 2 行、`protocols.py` docstring 2 行。

| 检查项 | 结论 | 依据 |
| --- | --- | --- |
| 训练数学零改动 | ✅ | diff 不含 loss/forward/optimizer/数据/RNG 路径；首步 loss 实测逐位相同（见 §4） |
| A 仍是 buffer | ✅ | `adapters/kinds/lori.py:47-48` 未改；运行时 252 个 A `requires_grad=False`、非 Parameter |
| 没有新增可训练参数 | ✅ | 无新 `nn.Parameter`；`trainable_is_B=True`，A/alpha 不在可训练集合 |
| 没有把 A 纳入 optimizer 的路径 | ✅ | `build_optim` 仍只取 `trainable_params()`（`training/build/loop.py:173-202`）；`optim.pt` 仍 252 entry |
| resume 不受新键影响 | ✅ | `Parts.restore`（`drivers/common.py:136-145`）只读已知键；`lori` 键被忽略 |
| 旧 checkpoint 兼容 | ✅ | 用 task10 step_900 的**副本**续训 1 步（step 901）成功，无报错 |
| `checkpoint_metadata` 三态 | ✅ | LycoAdapterSet → `{'lori': …}`；FullTrainable → `{}`（`training/full.py:63-65`）；pipeline 传 gather 后的状态（`drivers/pipeline.py:168-172`） |
| 风格 | ✅ | black/ruff 通过；`adapter_set.py` 572 行 < 600；新增代码全 ASCII；无函数内 import；无 rationale 注释 |
| `ss_network_args` | ✅ | 仍为 `{"algo": "lora"}`，未注入 LoRI 键；6 个真实导出全部解析为该值 |
| 文档与实现一致 | ✅ | `docs/concepts/adapters.md` 的 LoRI exports 段与代码逐项对应（container 仅 comfy、lycoris 无 ss_network_*、comfy 为 bf16 故 digest 从 state 复算） |

## 2. 测试复跑

| 命令 | 结果 | 说明 |
| --- | --- | --- |
| `pytest tests/adapters/test_lori_export_facts.py -q` | **15 passed**（8.32 s） | 新增测试全过 |
| `pytest tests/adapters -q`（GPU） | **299 passed**（210.59 s） | 与本改动相关的路径全过 |
| `pytest tests/adapters -q`（`CUDA_VISIBLE_DEVICES=''`） | 244 passed / **55 failed** | 55 个全在 `test_kernel_ops.py`（该文件 `pytestmark = pytest.mark.cuda`，无 CUDA 时构造 cuda tensor 失败）→ **环境失败，非改动** |
| 同上，排除 `test_kernel_ops.py` | 243 passed / 1 failed | 失败为 `test_compiled_body_checkpoint`，同样需要 CUDA 编译 → 环境 |
| `pytest tests/training -q -k 'checkpoint or state or parts'` | 22 passed / 2 skipped | resume/parts 相关通过 |
| `black --check`（7 个文件） | 7 files would be left unchanged | ✅ |
| `ruff check`（7 个文件） | All checks passed! | ✅ |
| `pytest tests/training -q -k doctor_json` | **1 passed** | 实现者报告的失败在本机**未复现**，如实记录为通过 |
| `pytest tests/rules -q` | 18 passed / 1 failed | `test_no_raw_peak_counter_use` 指向 `tests/training/run/test_lori_real_weights.py` 的 [85,102,103,206,259,260] 行；该文件**不在本次 diff** 中（既有问题，需另行修复） |

## 3. 真实模型导出验证（新代码，3 检查点 × 2 格式）

命令模板（DEVICE=cuda，逐份记录退出码与日志）：

```text
kogine run scripts/export/export.py --set WEIGHTS=<models/ComfyUI/models> --set ADAPTER=<ckpt>/trainable.safetensors \
  --set OUT=<exports>/<case>-<fmt>.safetensors --set FORMAT=lycoris|comfy --set SAVE_DTYPE=fp32|bf16 --set DEVICE=cuda
```

| 检查项 | dense900 | sparse90-900 | sparse60-570 |
| --- | --- | --- | --- |
| 来源检查点 | task10 dense step 900 | task11 sparse90 step 900 | task11 sparse60 step **570** |
| facts 6 键 | ✅ | ✅ | ✅ |
| phase / mechanism / trainable | calibration / dense_calibration / b_dense | sparse / sparse_mask / b_values | sparse / sparse_mask / b_values |
| sparsity | 0.0 | 0.9 | 0.6 |
| kept_total / rank | 仅 calibration：无（符合规格） | **1,066,599** / 8 | **4,266,394** / 8 |
| 原生键集合（252 层） | 恰 `{A, alpha, b_dense}`，无 b_values | 恰 `{A, alpha, b_values, active_indices}`，**无 b_dense** | 同左 |
| `kohakumoss_lori_a_sha256` | `976522f0…`（独立复算一致） | 同 | 同 |
| 同一 run 不同 step 的 a_sha256 | 285/570/855/900 全部相同 | 285/570/855/900 全部相同 | 285/570/855/900 全部相同 |
| 与旧导出的张量逐位一致 | lycoris 756/756、comfy 504/504 | 1008/1008、504/504 | 1008/1008、504/504 |
| comfy 标记 / container | `inference_only_dense_b` / `lora` | 同 | 同 |
| `ss_network_args` | `{"algo": "lora"}` | 同 | 同 |
| 原生 `ss_network_*` | 无（按设计） | 无 | 无 |

补充映射验证（comfy 张量 vs 原生 A/B，bf16 位级比较）：

- **126 个非融合层**：`lora_down.weight` == A、`lora_up.weight` == B（dense 用 `b_dense`，sparse 用 `b_values` 按 `active_indices` 零填充重建），**126/126 全部位级相等**，三个检查点均如此。
- **42 个融合 qkv 层**：`lora_down` == `cat(A_q, A_k, A_v, dim=0)`，`lora_up` == `2 × block_diag(B_q, B_k, B_v)`；**42/42 位级相等**。系数 2 = `alpha/rank`，是 comfy 融合 qkv 的既有布局约定（旧导出同样如此，非本次改动引入）。
- 三份原生文件的 canonical A/alpha 摘要（`name:digest` 排序串联）仍为 `7f8b433f…` / `930ce824…`；sparse 的 `b_values` 非零数分别 1,066,599 / 4,266,394，与 `kept_total` 一致。
- 警告次数（按日志精确计数）：sparse90-comfy **1**、sparse60-comfy **1**、dense-comfy **0**、三份 lycoris **0**。

## 4. 训练语义与 resume 回归（GPU，2 步 × 2 + 1 + 1）

| 回归 | 结果 |
| --- | --- |
| dense 2 步 | 首步 loss **0.7369208931922913**（与既有记录逐位相同）；`state.json` 有 `lori` 块（calibration / dense_calibration / b_dense / sparsity 0.0 / kept_total 10,665,984 / a_sha256 `976522f0…`）；`optim.pt` 252 entry、矩 21,331,968；A 训练前后不变；B 252/252 变化 |
| sparse90 2 步 | 首步 loss 同为 **0.7369208931922913**；`lori` 块为 sparse / sparse_mask / b_values / 0.9 / **1,066,599**；`optim.pt` 252 entry、矩 **2,133,198**；A 不变；B 252/252 变化 |
| 自动导出 | 两个 run 的 checkpoint 与 run 级导出都带新元数据：lycoris 有 facts、comfy 有 marker 与 `{"algo": "lora"}` |
| resume（新检查点） | 从 dense 2 步 run 的 `step_00000002` 续训 1 步：`metrics` 3 行、steps 到 3；`lori` 块仍在且字段正确；`optim.pt` step=3、lr=1e-4；无报错 |
| resume（**旧格式**检查点） | task10 step_900 的副本（`state.json` 无 `lori` 块）续训 1 步到 901：加载正常、写出新的 `lori` 块、`optim.pt` step=901、lr=1e-4 |

**环境注意（重要）**：首步 loss 的逐位复现依赖 kernel/extension cache 隔离。使用 per-run 的 `KOHAKUMOSS_KERNEL_CACHE`/`TORCH_EXTENSIONS_DIR`/`CUDA_CACHE_PATH`（与 task10/task11 launch.json 相同做法）时，首步 loss = 0.7369208931922913；若沿用共享的旧 cache，首步 loss = 0.7371941208839417（+3.7e-4），属于 kernel 选择差异，**不是代码改动**。

## 5. 发现与建议（分级）

| ID | 级别 | 位置 | 问题 | 建议 |
| --- | --- | --- | --- | --- |
| F1 | **P2** | `lori.py` 的 `a_sha256` / `kept_total` | 两者都经 `_layers()` 把每个字段（含 42 MB 的 `b_dense`）搬到 CPU；`export_facts` 与 `checkpoint_block` 连续调用两者 → 每次保存/导出多两趟全量拷贝 | 一次 `_layers` 同时算两值，或加只遍历 A / active_indices 的路径 |
| F2 | P3 | `adapter_set.py: state_metadata()` | facts 由本地 `_live()` 计算；pipeline/DDP 下每 rank 的 trainable 元数据只描述本地分片（`state.json` 已用 gather 后的状态，是正确的） | 让 `state_metadata` 也接受 state，或在 pipeline 分支覆盖 |
| F3 | P3 | `tests/adapters/test_lori_export_facts.py` | 15 个 case 未覆盖 `FullTrainable.checkpoint_metadata()=={}` 与「旧 checkpoint 续训」 | 补两个 case（本次审计已外部验证） |
| F4 | P3 | `adapter_set.py: export()` | 只有 lycoris/comfy 带 facts；merged/diffusers 的 LoRI 导出没有自描述键 | 需求只要求两种格式；若要覆盖，给 merged 也写同一组键 |
| F5 | P3 | `adapter_set.py: metadata()` | sparse+comfy 的警告在 `metadata()` 内发出，仅检查不落盘也会打 | 移到写盘分支或加 once 标记 |

没有 P0/P1：未发现训练数学变化、可训练参数泄漏、A 进入优化器、resume 破坏或元数据错误。

## 6. 裁决

- **需求满足**：两种格式自描述 ✅；dense calibration 与 sparse LoRI 可区分 ✅；comfy 标 inference-only ✅；A 事实入文件 ✅；`state.json` 带 `lori` 块 ✅。
- **无阻断问题**，可以放行到 90% 稀疏 900 步重训。
- 建议在长跑前处理 **F1**（保存路径的两趟全量拷贝）；F2–F5 可后置。

## 7. 限制

- 未跑完整 `tests/training`（只跑 checkpoint/state/parts 子集）；`tests/rules` 的既有失败不属本改动。
- 导出只覆盖 3 个检查点（dense 900、sparse90 900、sparse60 570），其余 step 只做了 `a_sha256` 复算。
- 融合 qkv 的 `lora_up = 2 × block_diag(B)` 是按 comfy 既有布局约定验证的；该系数与旧导出逐位一致，不是本次改动新增。
- 未修改实现与其它人产物、未删除旧文件、未 `git commit`；`tmp/` 与 `tmp2/` 为本次审计的临时目录。
