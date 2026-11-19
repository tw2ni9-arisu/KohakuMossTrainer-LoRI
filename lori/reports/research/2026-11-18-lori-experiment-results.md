# MOSS-Image DiT LoRI 首轮实验结果

2026-11-18（Asia/Shanghai）

在 RTX 5070 Ti Laptop（12,227 MiB 显存）上，KohakuMossTrainer 使用本地 MOSS-Image INT8 主干完成了用户 12 张标注图像的离线缓存、12 步 LoRI 校准、全局掩码和 32 步稀疏训练。稀疏阶段有 **1,066,599 个可训练值**，AdamW 一阶和二阶矩合计 **2,133,198 个元素**。32 步训练的 PyTorch 峰值为 **2.40 GiB allocated / 2.69 GiB reserved**，第 2–32 步的中位步时为 **1.459 秒**。

使用最终原生 LoRI 权重生成了 **24 张 512×512 图像**。8 行中，LoRI ×1 均改变输出，LoRI ×0 均逐像素恢复 Base。服装、配饰、表情和姿态出现可见变化。这一轮的单种子结果尚无法确认稳定的 `nigenmame` 风格或触发词特异性。 [对照网格](../../experiment/task7/session-20261118-a/images/grid.png) · [逐像素审计](../../experiment/task7/session-20261118-a/pixel-audit.json)

## 论文移植与本地基础

LoRI 原论文和作者代码在语言模型上验证了固定随机投影 A 与稀疏 B 的方法。本项目保留其训练顺序。先用 MOSS 原生 MSE 目标校准零初始化的稠密 B，再对所有目标层的 `abs(B)` 取一个全局分位阈值，以 `>=` 保留阈值并列项。之后复用同一组 A，把活跃位置的 B 清零，在新进程中创建优化器并开始稀疏训练。DiT 的 `attn-mlp` 预设覆盖 252 个目标模块，rank 8、alpha 16，有效缩放为 2。 [论文和作者实现调查](2026-11-17-lori-dit-investigation.md) · [DiT 设计](../superpowers/specs/2026-11-17-dit-lori-design.md)

稀疏阶段只将活跃位置的 `b_values` 注册为可训练参数，索引和 A 保存在原生状态中。前向会临时重建稠密 B，因此本次证据支持优化器状态按活跃元素数分配。总显存和吞吐的净收益需要同配置的稠密基线对照。 [数学模块验证](../execution/dit-lori/task-1-report.md) · [全局掩码验证](../execution/dit-lori/task-2-report.md) · [稀疏状态审计](../../experiment/task6/session-20261118-a/sparse-final-audit.json)

独立代码副本基于本地官方快照 `e60bf3260d24cea6fdd61be53893dcddad529652`，内层 Git 基线 `7a5b94a`，实验版本 `lori-development` 的 `285f4bd2b5c29378ca75797dd247690f6a06a21a`。复用三件已校验的本地权重（MOSS-Image INT8 ConvRot DiT、Qwen3.5-9B INT8 ConvRot、MOSS VAE）、现有 Python 环境与训练器的 block checkpointing、四槽权重流送、激活卸载、流式主机优化器状态、融合更新和梯度卸载。用户数据以只读引用进入清单，选中的 12 对图文内容哈希和源目录 570 个文件的元数据均通过复核。 [本地基础与环境](2026-11-17-local-foundation.md) · [数据来源与独立审查](../execution/dit-lori/task-6-review.md)

## 训练与核验

两段真实图像训练均使用 256×256 目标面积（保留样本宽高比）、BF16、INT8 主干、默认 Triton、`ckpt-ring4-actoff-optstream-gradoff`（`fused_step=True`）、AdamW `1e-4` 恒定学习率、零权重衰减、batch 1、梯度累积 1 和 seed 42。文本与图像编码在独立预处理进程完成，离线审计读到 12 个 latent、13 个文本 embedding（含空提示）及 12 个完整训练批次，未发现丢样或截断。 [配方与缓存报告](../execution/dit-lori/task-6-cache-report.md)

- **12 步校准。** 原生训练完成 12/12，损失和梯度范数均有限，跳过、非有限步骤与非有限参数计数均为零。504 个 A/alpha 张量哈希与初始化一致，252 个稠密 B 张量发生更新。训练后的私有审计漏拼了 `RUN_NAME`，导致包装器退出 1。保留原错误和退出记录，修正私有路径后使用已有 checkpoint 单独完成 CPU 补验，退出 0。校准只训练了一次。 [校准报告](../execution/dit-lori/task-6-calibration-report.md) · [原退出记录](../../experiment/task6/session-20261118-a/stages/calibration/exit.json) · [CPU 补验](../../experiment/task6/session-20261118-a/calibration-posthoc-audit.json)
- **全局掩码。** CPU 计算 `abs(B)` 的 0.9 分位阈值 `0.0006955114076845348`，从 10,665,984 个稠密 B 元素中保留 1,066,599 个位置，密度约 10.00%。独立复算验证了每层保留位置、原样复用的 A/alpha 和全零的紧凑 B。 [掩码报告](../execution/dit-lori/task-6-mask-report.md) · [稀疏初始化权重](../../experiment/task6/session-20261118-a/sparse-init.safetensors)
- **32 步稀疏训练。** 实际 worker PID `85176`，worker 和启动器均退出 0。32 步的损失与梯度范数均有限，跳过和非有限计数为零。756 个 A/alpha/索引张量哈希一致，252 个紧凑 B 向量均有更新。252 组优化器状态的 FP32 master、一阶矩和二阶矩各含 1,066,599 个值，步数计数均为 32。损失首步 0.54285、末步 0.81081，区间 0.42006–0.91301。这条短跑曲线记录训练状态，收敛和视觉质量尚待验证。 [训练报告](../execution/dit-lori/task-6-sparse-report.md) · [机器可读指标](../../experiment/task6/session-20261118-a/sparse/lori-sparse/metrics.jsonl)
- **保存与恢复。** 真实模型门禁的首次运行在默认 Triton 的严格续训相等断言处退出 1。随后固定输入探针复现了默认 Triton attention 反向的逐位差异。第二次真实门禁保留默认 Triton 的校准和稀疏步覆盖，仅在合成批次的严格续训对照中选用 eager + math SDPA。在该确定性路径中，恢复边界、准备后输入、最终 adapter 和优化器状态逐位一致。第二次门禁为 1 passed、1 warning、退出 0，相关无模型回归记录为 44 passed。用户图像实验另完成了自包含原生权重的 CPU 读取与构造参数校验，并保留完整可续训 checkpoint。用户图像的整模型续训和默认 Triton 的逐位续训均未获验证。 [首次门禁退出](../../.tools/task5/run-20261117T150616Z/exit.json) · [通过的门禁结果](../../.tools/task5/run-20261117T182252Z/artifacts/result.json) · [门禁与测试报告](../execution/dit-lori/task-5-report.md)

### 资源实测

| 阶段 | Worker / 启动器墙钟时间（秒） | PyTorch CUDA allocated / reserved 峰值（GiB） | 主机进程峰值工作集（GiB） |
| --- | ---: | ---: | ---: |
| 文本缓存 | 201.472 / 220.202 | 1.219 / 1.252 | 16.99 |
| VAE latent 缓存 | 1.242 / 19.018 | 0.243 / 0.281 | 1.89 |
| 12 步校准 | 39.619 / 58.395 | 2.433 / 2.725 | 13.26 |
| 32 步稀疏训练 | 68.081 / 84.134 | 2.399 / 2.691 | 13.26 |
| 24 张对照图 | 1515.797 / 1521.579 | 1.974 / 2.039 | 23.34 |

校准首步 8.387 秒，第 2–10 步原生日志中位数 1.677 秒。稀疏训练首步 7.755 秒，第 2–32 步中位数 1.459 秒。采样原生日志记录 Base、LoRI ×1、LoRI ×0 三列各 8 张，分别耗时 449.84、504.77、499.99 秒。

表中 CUDA 数值取原生外层分配器窗口，覆盖该阶段加载或训练、缓存、导出以及审计所用的 PyTorch 内存。采样窗口覆盖加载、编码、三列采样和解码。启动器墙钟时间还包括导入，主机工作集覆盖进程全生命周期。稀疏训练阶段另有 `nvidia-smi` 观测样本 4,772 MiB，这是包括其他进程的整机占用读数。12 GB 设备上成功运行的是上述配方与输入，8 GB、6 GB 和更高训练分辨率尚无实测。 [缓存资源](../execution/dit-lori/task-6-cache-report.md) · [训练资源](../execution/dit-lori/task-6-sparse-report.md) · [采样日志](../../experiment/task7/session-20261118-a/stages/generate/stdout.log)

## 同种子图像对照

4 个场景各有一对提示，只增删开头的 `nigenmame, `。所有行使用 seed 42、512×512、Euler 50 步、CFG 3、shift 5、two-pass CFG、batch 1，复用文本 embedding。三列依次是 Base、最终原生 LoRI ×1、同权重 LoRI ×0。使用 BF16、INT8 基座、`minimal` 权重流送和 adapter bypass。最初的准备阶段因把正常 CUDA context 初始化视为失败而退出 1，修正准备检查后退出 0。实际采样只运行了一次。 [实际配置](../../experiment/task7/session-20261118-a/config.py) · [元数据](../../experiment/task7/session-20261118-a/images/metadata.json) · [首次准备退出](../../experiment/task7/session-20261118-a/stages/prepare-failed-context-check/exit.json)

实际采样 worker PID `76176`。Luna 的后台监控记录了真实进程和前两列完成，随后因用量限制结束。主代理在原进程结束后核对原生成功日志、worker result 和启动器 exit（均退出 0）以及全部 24 张单图。解码后的 RGB 数据验证了 8/8 行 Base = ×0 和 8/8 行 ×1 ≠ Base。网格为 2048×4132。 [监控记录](../../experiment/task7/session-20261118-a/monitor/monitor-report.md) · [Worker result](../../experiment/task7/session-20261118-a/stages/generate/result.json) · [启动器 exit](../../experiment/task7/session-20261118-a/stages/generate/exit.json)

### 逐行观察

差值为 LoRI ×1 相对同一行 Base 的逐通道平均绝对差，按 RGB 8-bit 的 0–255 计。变化像素统计至少一个通道变化的位置，每行共有 262,144 个像素。视觉描述来自网格及第 4、5 行的代表性单图复核。

| 行 | 场景 | 触发词 | 变化像素 / 平均差 | LoRI ×1 的可见变化 |
| --- | --- | --- | ---: | --- |
| 0 | 紫发星空半身 | 有 | 260,611 / 12.38 | 白色上衣面积增大，袖口和胸前绑带重绘，发梢与小行星布局变化。金眼、托腮和星空构图保留。 |
| 1 | 紫发星空半身 | 无 | 260,028 / 9.45 | 右侧袖口由多道细带变为单条深色带并伴随褶皱变化，发丝、脸部阴影和衣褶有小幅重绘。 |
| 2 | 青发角色与花束 | 有 | 262,019 / 7.08 | 领口出现交叉细带，腰侧蝴蝶结和飘带轮廓变化。深蓝背景、双马尾和发光花束保留。 |
| 3 | 青发角色与花束 | 无 | 262,087 / 9.50 | 深色前襟变浅，领口增加交叉细带，袖口、花束和飘带重绘。 |
| 4 | 红白服装双人和布偶 | 有 | 261,993 / 16.66 | 左侧人物换成发间浅色饰件、露出颈部，布偶头饰消失且发色变浅，手指握法、布偶裙褶和面部线条变化。 |
| 5 | 红白服装双人和布偶 | 无 | 261,802 / 11.69 | 左侧人物出现浅色发饰，右侧人物眼形与刘海变化，布偶面部和头发重绘。双人近景和红白配色保留。 |
| 6 | 海边黄昏与手持烟花 | 有 | 262,138 / 10.68 | 左侧人物头发、手势与外套位置变化，右侧表情和衣褶重绘。两人的站位、栏杆和日落色调接近。 |
| 7 | 海边黄昏与手持烟花 | 无 | 262,130 / 12.37 | 左侧人物头发向外飘起，姿态与袖口变化，左侧亮焰减少。右侧人物和海边构图保留。 |

加入触发词时，Base 的紫发星空图也把长袖和伸指换成短袖与曲手，花束图改变腰侧飘带形状。红白双人图换了袖型、布偶衣裙和人物发色，海边图变更右侧人物的手势及两人的衣裙。这些场景差异在 ×1 列中仍可见。双人近景的带词与无词 ×1 都出现浅色发饰，花束场景的带词与无词 ×1 都出现交叉细带。第 4 行的平均差最大，手指、脸部轮廓和服饰变化在单图中也可见。 [网格](../../experiment/task7/session-20261118-a/images/grid.png) · [第 4 行 Base](../../experiment/task7/session-20261118-a/images/r04_c00.png) · [第 4 行 LoRI ×1](../../experiment/task7/session-20261118-a/images/r04_c01.png) · [第 5 行 Base](../../experiment/task7/session-20261118-a/images/r05_c00.png) · [第 5 行 LoRI ×1](../../experiment/task7/session-20261118-a/images/r05_c01.png)

4 对场景中，有词行的 ×1 与 Base 的平均像素差较大和较小各有 2 对。各组总体仍接近提示指定的构图，未观察到横跨 4 个场景的统一画法变化。像素差衡量输出变化幅度。特定训练集风格的相似度仍需另行评估。这轮使用一个采样种子、无 LoRA 对照和无独立留出评估，风格学习、角色一致性、泛化和与 LoRA 的质量对齐均待验证。

## 权重、图像与复现入口

| 工件 | 用途 | 字节数或内容 |
| --- | --- | --- |
| [原生 LoRI 推理权重](../../experiment/task6/session-20261118-a/sparse/lori-sparse/export/lori-sparse-lycoris.safetensors) | 本次 `compare_grid` 使用的 A、活跃索引、紧凑 B 和元数据 | 44,654,796 bytes |
| [Comfy 导出权重](../../experiment/task6/session-20261118-a/sparse/lori-sparse/export/lori-sparse-comfy.safetensors) | 密集 LoRA 兼容表示，已导出，实际 ComfyUI 加载尚待验证 | 49,601,320 bytes |
| [第 32 步训练 checkpoint](../../experiment/task6/session-20261118-a/sparse/lori-sparse/checkpoints/step_00000032/) | 完整训练恢复工件 | [权重](../../experiment/task6/session-20261118-a/sparse/lori-sparse/checkpoints/step_00000032/trainable.safetensors)、[优化器](../../experiment/task6/session-20261118-a/sparse/lori-sparse/checkpoints/step_00000032/optim.pt)、[随机数状态](../../experiment/task6/session-20261118-a/sparse/lori-sparse/checkpoints/step_00000032/rng.0.pt)、[训练状态](../../experiment/task6/session-20261118-a/sparse/lori-sparse/checkpoints/step_00000032/state.json) |
| [24 张单图及网格](../../experiment/task7/session-20261118-a/images/) | 单图 `rNN_cNN.png`，`r` 为提示行号，`c=00/01/02` 对应 Base/×1/×0 | [网格](../../experiment/task7/session-20261118-a/images/grid.png)、[元数据](../../experiment/task7/session-20261118-a/images/metadata.json)、[像素及文件 SHA-256 审计](../../experiment/task7/session-20261118-a/pixel-audit.json) |

训练配置由 [common.py](../../experiment/task6/session-20261118-a/common.py)、[data.py](../../experiment/task6/session-20261118-a/data.py)、[calibration.py](../../experiment/task6/session-20261118-a/calibration.py) 和 [sparse.py](../../experiment/task6/session-20261118-a/sparse.py) 组成。[训练启动器](../../experiment/task6/session-20261118-a/launch.py) 使用本地 Python 调用仓库原生缓存/训练/掩码入口，实际解析参数见 [训练解析记录](../../experiment/task6/session-20261118-a/resolved.json)。[采样配置](../../experiment/task7/session-20261118-a/config.py) 与 [采样启动器](../../experiment/task7/session-20261118-a/launch.py) 调用 `scripts/sample/compare_grid.py`。[采样冻结清单](../../experiment/task7/session-20261118-a/freeze.json) 保存输入和源代码哈希。未来复跑应保留现有一次性历史目录，为新的配置、运行记录、缓存和输出选用新会话路径。

### 独立导出修复

整分支审查发现，独立导出在构建稀疏适配器时遗漏了已读取的 checkpoint 张量和元数据。配置中的原掩码路径不可用时，导出会因重读掩码而失败。交付版本 `957ccc3f89c1506db8e41ce4d260f529446e142b` 将两者传给适配器构建入口。新增回归在修复前重现 `FileNotFoundError`，修复后通过，相关 CPU 测试记录为 256 passed、56 skipped。 [整分支审查](../execution/dit-lori/final-review.md) · [独立导出入口](../../KohakuMossTrainer/src/kohakumoss/adapters/export/cli.py) · [回归测试](../../KohakuMossTrainer/tests/adapters/test_lori_state.py)

使用真实第 32 步 checkpoint 的副本，把掩码路径指向一个不存在的文件，再通过 `scripts/export/export.py` 用同一套 INT8 主干在 CPU 上重导原生格式。重导工件的 1,008 个张量与训练导出的张量在名称、形状、dtype 和内容上完全一致，文件容器的元数据有差别。历史训练、采样和本报告中的资源实测均来自 `285f4bd2b5c29378ca75797dd247690f6a06a21a`。 [重导复核脚本](../../.tools/final-export-check/smoke.py)
