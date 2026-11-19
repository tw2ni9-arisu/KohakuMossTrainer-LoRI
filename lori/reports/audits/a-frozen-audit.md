# A 是否在训练中被冻结：Task 12 取证审计

**结论先行。** A 在训练中**确实冻结**，而且是**构造性冻结**——`A`/`alpha` 用 `register_buffer` 注册（`adapters/kinds/lori.py:47-48`），永远不进 `named_parameters()`，也永远不被 `trainable_params()` 枚举（`adapters/train/adapterset.py:246-262`），优化器只从该枚举建组（`training/build/loop.py:173-202`）。这不是靠约定、注释或断言，而是数据结构上不可能。**但是**：交付的 comfy 导出把 A 与 B 平权写成 `lora_down.weight`/`lora_up.weight`，通用加载器会把两者都建为 `requires_grad=True` 的 `nn.Parameter`（我实测），文件里没有任何 frozen 标记——**「训练时 A 冻结」与「导出文件表达 A 冻结」是两件事，前者成立，后者不成立**。

因此「dense 900 那个 run 是不是论文意义的 LoRI」的答案是：**不是**。它是固定 A 的稠密 B 校准训练（论文 `s=0` 极限，称 LoRA-FA 形态）；只有 Task 11 的三档稀疏（90/60/30%）才具备 LoRI 的定义性机制（`active_indices` + 紧凑 `b_values` + 紧凑优化器状态）。

结构化证据：[a-frozen-audit.json](../../../experiment/task12-audit/a-frozen-audit.json)（24 个检查点/掩码文件、4 份优化器、2 个 2 步运行时、4 份导出语义实验）。

## 0. 审计口径与代码状态

| 项 | 值 |
| --- | --- |
| 训练器 HEAD | `957ccc3f89c1506db8e41ce4d260f529446e142b`（2026-11-18 13:07 +08:00，*Restore native LoRI state in the standalone exporter*） |
| 交付产物生成时 | 工作树干净（本轮复核开始时 `git status --porcelain` 为空） |
| 复核期间的工作树 | **有未提交改动**：8 个文件 +139/−4，另有未跟踪的 `tests/adapters/test_lori_export_facts.py` |
| 该改动归属 | **不是我改的**；内容是给导出/检查点补 LoRI 元数据（`kohakumoss_lori_phase/mechanism/trainable/a=frozen_buffer/a_sha256`、`kohakumoss_container`、`kohakumoss_lori_comfy=inference_only_dense_b`，以及稀疏 comfy 导出的警告） |
| 对结论的影响 | 无。`git diff` 里 `lori.py` 只在第 124 行后新增 `a_sha256`/`kept_total` 两个纯函数，`LoRIModule.__init__` 的 buffer 注册（47-48 行）未被改动 |
| 2 步实验环境 | 同一工作树（HEAD + 上述补丁）；`CUDA_VISIBLE_DEVICES` 未屏蔽、单卡 RTX 5070 Ti Laptop 12 GB |

## 1. 静态取证：A 为什么不可能被更新

| 位置 | 事实 |
| --- | --- |
| `adapters/kinds/lori.py:47-48` | `self.register_buffer("A", a.detach().clone())`、`register_buffer("alpha", ...)` —— A/alpha 是 **buffer**，不进 `nn.Parameter` 集合 |
| `adapters/kinds/lori.py:52` | calibration：`self.b_dense = nn.Parameter(self.A.new_zeros((out, dim)))` —— 唯一可训练张量 |
| `adapters/kinds/lori.py:66-67` | sparse：`register_buffer("active_indices")` + `nn.Parameter("b_values")` —— 掩码外根本没有可训练存储 |
| `adapters/train/adapter_set.py:246-262` | `trainable_params() → named_trainable() → _entries()` 只遍历 `t.lyco.named_parameters()` 且要求 `requires_grad`；buffer 永不出现在枚举里 |
| `training/build/loop.py:173-202` | `build_optim` 只从 `built.trainable.param_groups(...)` 取参数 —— 优化器构造上不可能持有 A |
| `training/entry.py:300` | 训练入口只有 `build_optim(k, built, ...)` 这一条路径，没有第二条注入通道 |
| `adapters/build/apply.py:25-36` | 唯一写 buffer 的地方是构建期 `_place()`（把 A/alpha 搬到目标 device），训练循环不触碰 |
| `adapters/train/lori.py:180-186` | `read_lori_state` 返回普通 `Tensor`，不包 `Parameter`、不设 `requires_grad` —— 冻结由**加载方**的 `register_buffer` 保证 |
| `adapters/infer/inference.py:374-382, 383-403` | 推理加载：原生 LoRI 走 `LoRIModule`（A 变 buffer）；comfy 走通用 `get_module`/`make_module`（A 就是 `lora_down`），随后统一 `requires_grad_(False)` —— 冻结是加载方施加的 |

一句话：**A 的冻结靠「不是 Parameter」实现，而不是靠「梯度被清零」。** 任何绕过 `named_parameters()` 的路径（例如直接把 `lora_down` 当普通权重训练）都不会经过这套保护。

## 2. 产物取证：24 个文件的 A/alpha 是否逐字节相同

覆盖 task6（calibration 12 步、sparse 32 步）、task8（sparse 232 步）、task9（dense 232 步）、task10（dense 285/570/855/900）、task11（calibration 285、sparse90/60/30 × 4 步、3 份掩码）。

| 统计 | 结果 |
| --- | --- |
| 文件数 | **24**（21 个检查点 + 3 个掩码；三份导出另在 §4 单列） |
| A 组合摘要 | **24/24 相同**：`7f8b433fe34e4caa789e9923e44227acdef677e04edc1852a71d6242b884db19` |
| alpha 组合摘要 | **24/24 相同**：`930ce824c0831386ee6bad65542d768c8b56d2c5a3264511be2d7ae02207c991` |
| B 摘要 | **24/24 互不相同**（含掩码，其 `b_values` 恒零）——B 在每个阶段都真的变了 |
| 步数跨度 | step 12 / 32 / 232 / 285 / 570 / 855 / 900，跨 4 个 session（task6/8/9/10/11） |

组合摘要的构造是 `sha256("<name>:<sha256(dtype:shape:rawbytes)>\n" …)`（名字排序），与训练器 `_digest` 同构；我另外逐张量比较了参考文件的 252 个 A，全部命中，不是只看一个汇总值。

## 3. 优化器取证：状态里有没有 A

| run | entry | 状态形状 | 矩元素 | 2×B | master 多重集 = B | A 形状张量 |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| task10 dense 900 | 252 | `[out,8]`（3072/16384 × 8） | 21,331,968 | ✓ | ✓ | **0** |
| task11 sparse90 900 | 252 | 1-D，长度 = 保留数 | 2,133,198 | ✓ | ✓ | **0** |
| task11 sparse60 900 | 252 | 1-D，长度 = 保留数 | 8,532,788 | ✓ | ✓ | **0** |
| task11 sparse30 900 | 252 | 1-D，长度 = 保留数 | 14,932,378 | ✓ | ✓ | **0** |

`optim.pt` 的每个 state 元组是 `(master, exp_avg, exp_avg_sq, step)`；**没有任何 state 张量的形状是 `[8, in_features]`**（A 的形状），且 `master` 与同 step 的 `b_dense`/`b_values` 逐张量摘要多重集完全相等——优化器只持有 B 的副本。

## 4. 运行时取证：受控 2 步（GPU，唯一一次训练）

克隆 task10 的 `train.py` 与 task11 的 `sparse-90.py` 为 `STEPS=2`、`SAVE_EVERY=0`、`FINAL_CHECKPOINT=True`，输出到 `experiment/task12-audit/tmp/`，在 `apply_adapters` 之后与优化器构造处打补丁，`register_hook` 覆盖全部 252 个可训练张量，并在训练前后比对 A 的字节与 autograd 版本号。两轮各约 10 s 训练墙钟（模型加载在外），跑完即删除 `tmp/`。

| 断言 | dense（calibration） | sparse90 |
| --- | --- | --- |
| 模块 / A / alpha / B / 可训练 | 252 / 252 / 252 / 252 / 252 | 同 |
| `A.requires_grad` / `A` 是 Parameter | **0 / 0** | **0 / 0** |
| A/alpha 出现在可训练集合 | **0 / 0** | **0 / 0** |
| 可训练集合 == B 集合 | ✓（`b_dense`） | ✓（`b_values`） |
| 优化器实现与参数数 | `HostStateAdamW` 252（形状 `[3072,8]`、`[16384,8]`） | `HostStateAdamW` 252（1-D，长度=保留数） |
| A 在优化器参数组 | **0** | **0** |
| A 字节 / 版本号 / 摘要变化 | 全部不变 | 全部不变 |
| A/alpha 的 hook 命中 | 0 / 0 | 0 / 0 |
| B 的 hook 命中 / 字节变化 | **252 / 252** | **252 / 252** |
| 步后 A 摘要（磁盘） | `7f8b433f…`（与初始一致） | 同 |
| optim.pt entry / 矩元素 | 252 / 21,331,968 | 252 / 2,133,198 |
| master 多重集 = 当步 B | ✓ | ✓ |
| CUDA 峰值 allocated / reserved | 1,889,433,088 / 2,195,718,144 B | 1,816,400,384 / 2,122,317,824 B |
| loss 首步 / 末步 | 0.7369208931922913 / 0.724826953 | 0.7369208931922913 / 0.724192262 |

两个 run 的首步 loss 逐位相同（同数据、同初始 A、B 全零 → 首步前向相同），但首步 `grad_norm` 在四次独立运行中分别为 0.143461 / 0.143685 / 0.144279 / 0.143…（小数位不同）——这也**独立复现**了 Task 11 复核里「calibration 与 dense285 从第一次反向就分叉」的现象：前向确定、反向不确定。

## 5. 导出语义取证：「文件能不能被当普通 LoRA 继续训练」

| 文件 | 张量数 | 读回结果 | 能否当普通 LoRA 续训 |
| --- | ---: | --- | --- |
| `lori-dense-step900-comfy.safetensors` | 504（168 层 × down/up/alpha） | 通用 lycoris 路径 → `LoConModule`，`lora_down.weight [8,8192]` 与 `lora_up.weight [3072,8]` **都是 `nn.Parameter(requires_grad=True)`** | **能**（A 会被一起训） |
| `lori-sparse90-comfy.safetensors` | 504 | 同上；且 `lora_up` 已是**展开的稠密 B**（49,601,320 B，与 dense 同尺寸；非零 1,066,599 = 保留数） | **能**（既训 A，也丢失稀疏性） |
| `lori-dense-step900-lycoris.safetensors` | 756 | `read_lori_state` 返回普通 `Tensor`（不包 Parameter）；元数据有 phase/sparsity | 通用 lycoris 算法匹配返回 `None`，**读不了** |
| `lori-sparse90-lycoris.safetensors` | 1008 | 同上；元数据含 `a_digest`/`alpha_digest`/`indices_digest`/`nnz`/`rank` | 同上，**读不了** |

`comfy` 元数据自相矛盾地同时写着 `kohakumoss_algo="lori"` 与 `ss_network_args={"algo": "lora"}`——后者来自 `Kind(algos={"lori": "lora"})`（`lori.py:117`），是给通用加载器看的；前者是私有标签。**没有任何一个键表达「A 冻结」**。

工作树里那份未提交补丁恰好补的就是这一点：`kohakumoss_lori_phase`、`kohakumoss_lori_mechanism`、`kohakumoss_lori_trainable`、`kohakumoss_lori_a="frozen_buffer"`、`kohakumoss_lori_a_sha256`、`kohakumoss_lori_comfy="inference_only_dense_b"`（稀疏 comfy 导出还会打警告）。**它不在任何已交付文件里**——我逐个读取了四个导出的 metadata 证实。

## 6. 概念界定与需要修正的过度表述

**论文口径**（本项目自己的调研文档已记录，`docs/research/2026-11-17-lori-dit-investigation.md:39`）：`s=0` 的单任务极限是 LoRA-FA（固定 A、稠密 B），LoRI-S 是稀疏版；两者应分别报告，不能混成一个 checkpoint。

| 文件 | 行 | 现表述 | 建议改写 |
| --- | ---: | --- | --- |
| `task-10-comparison-report.md` | 1 | `# LoRI 稠密 B 第 285 步与第 900 步…` | `# 固定 A 的稠密 B（LoRA-FA 形态）第 285 步与第 900 步…` |
| `task-10-570-comparison-report.md` | 1 | `# LoRI 稠密 B 第 570 步…` | `# 固定 A 的稠密 B（LoRA-FA 形态）第 570 步…` |
| `task-10-data-preparation-report.md` | 3 | `## MOSS-Image DiT LoRI · 全量 285 图文对 · 稠密 B · 900 步` | `## MOSS-Image DiT 固定 A 的稠密 B（LoRA-FA 形态）· 全量 285 图文对 · 900 步` |
| `task-10-data-preparation-report.md` | 218 | `\| LoRI 形态 \| ALGO="lori" …（稠密 B，不建掩码） \| 通过 \|` | `\| 固定 A + 稠密 B（LoRA-FA 形态，非 LoRI-S） \| … \| 通过 \|` |
| `task-10-independent-review.md` | 1 | `# Task 10 独立复核（LoRI 稠密 B …）` | `# Task 10 独立复核（固定 A 的稠密 B …）`（本复核自己的标题，未改动，仅在 §10 说明） |
| `task-10-dense-training-report.md` | 203 | `…把稀疏 LoRI 也训到 900 步…` | **无需改**（此处确实指稀疏 LoRI） |

另外两处**交付物命名/元数据**也应按同一口径修正：`lori-dense-step900-comfy.safetensors` 建议改为 `frozen-A-dense-B-step900-comfy-inference-only.safetensors` 或至少在 metadata 标明 `inference_only`；`ss_network_args` 的 `{"algo": "lora"}` 应同时给出私有 `algo=lori` 与 A 冻结事实（工作树补丁已做）。

## 7. 建议

1. **对外表述**：dense 900 一律称「固定 A 的稠密校准训练（dense-B / LoRA-FA 形态）」；「LoRI」只用于 `phase=sparse` 的三档。
2. **导出元数据**：把工作树里那套事实键合入并重新导出所有交付文件；comfy 版必须带 `inference_only_dense_b` 与 A 摘要。
3. **加载侧护栏**：若加载的 comfy 文件带 LoRI 标记，`lora_down` 应按 buffer 装载或在训练模式下直接拒绝——否则「A 冻结」只在原生路径成立。
4. **命名/目录**：comfy 导出放到 `inference-only/` 或在文件名标注，避免被当作可续训权重。
5. **测试**：保留工作树里的 `tests/adapters/test_lori_export_facts.py`，并补一条「comfy 文件经通用加载器得到 requires_grad=True」的负向断言，防止未来有人误以为文件自带冻结。

## 8. 限制

- 运行时只跑了 **2 步**（受控实验），不是对 900 步的重新训练；长跑的 A 不变性由 24 个检查点的逐张量摘要覆盖。
- 单卡、单次运行；未做多卡/DDP 复核。
- 2 步实验用的是带未提交元数据补丁的工作树；补丁不涉及 A 冻结路径（已核 diff）。
- 未识别运行时第二个 optimizer（`torch.optim.Optimizer`，1 个 `[1]` 形状参数）的来源；它与 A 无关、也不在 `optim.pt` 的 252 个 entry 里，建议维护者确认。
- 未修改训练器源码、未改他人报告、未删除任何实验产物；2 步实验的 `tmp/` 已删除。
