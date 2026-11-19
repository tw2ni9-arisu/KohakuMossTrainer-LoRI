# DiT LoRI 工作记录

日期：2026-11-17（Asia/Shanghai）

首轮实验完成记录更新于 2026-11-18。最终结论、复现配置和交付工件见 [综合结果](research/2026-11-18-lori-experiment-results.md)。

## 目标与验收

1. Luna 根据 LoRI 原论文与官方实现形成可追溯调查报告，明确 DiT 迁移的假设。
2. Luna 从 `D:\Develop\nigenmame-LoRA` 复制本地优化后的 KohakuMossTrainer 和必要 MOSS base 组件，记录版本与来源。
3. Sol 根据调查与真实代码制定接口、实现和验证计划，实现校准、紧凑参数、训练、保存加载及推理。
4. 通过算法与集成测试，独立审查影响正确性的问题。
5. 运行一次实际 MOSS-Image 小型 LoRI 训练，记录 GPU 峰值、步时、损失与权重，并进行同种子 base/LoRI 视觉对照。

## 约束与已确认决定

- 源目录只读：不移动、删除或修改源文件；基础复制排除训练数据、报告、缓存与输出。
- 用户随后明确授权首轮训练使用 `D:\Develop\nigenmame-LoRA\dataset-raw\round2-caption` 下已标注图像。
- 训练缓存、配置和输出置于当前工作区，不能污染源数据目录。
- 优先复用本地模型、仓库、环境，仅必要缺失项联网获取。
- 代码和训练参数由 GPT6.1 Sol high 子代理处理；调研、复制及训练监控由 GPT6 Luna max 子代理处理。
- 实际训练启动后，主代理进入 sleep 等待，只有监控报告错误或完成才恢复处理。
- 首版以论文定义正确性及实测为准；背景说明中的显存估计不视作已验证结论。

## 当前分工

- `/root/lori_research`：LoRI 原论文与官方代码调查、DiT 迁移报告。
- `/root/local_base_setup`：本地安全复制、模型与硬件环境清单。
- `/root/dit_design`：只读代码审查、可行性评估、设计与实现计划。

## 进度

- 已读当前 AGENTS.md；工作开始时工作区仅有该文件，无残留 Python/git/复制训练进程。
- 已向用户确认首轮训练数据来源。
- 三项准备任务已完成，报告/设计/计划已落地；源码及三件权重已复制和核验。
- 内层源码仓库已建立 `lori-development` 分支，基线 `7a5b94a`。
- Task1 数学模块已提交 `f4a2c65`，78 个相关测试通过，独立代码审查已批准。
- Task2 全局掩码/工件/CLI 已提交 `46d84ad`，70 个相关测试通过，独立代码审查已批准；尚未启动真实训练。
- Task3 配置/工厂接入已提交 `486dfa2`，157 个相关测试通过，独立代码审查已批准。
- 12 张图的只读训练清单已准备；Triton 编译环境问题通过进程 CC 选择已有 TCC 解决，原失败 doctor 用例通过。尚未运行模型训练。
- Task4 自包含保存/恢复/推理及混合精度审查修复已完成至 `f34464a`，133 个覆盖修复的相关测试通过，独立复审已批准；真实 MOSS 卸载兼容性待验证。
- Task5 真实 MOSS 门禁已通过并提交 `285f4bd`；使用完整 12GB fused/stream/offload 配置，精确恢复与默认非确定性 attention 的边界单独记录。开始准备用户图像缓存和12步校准、32步稀疏训练。
- Task6 缓存组已全部通过：12 latent、13 text、完整12批离线审计无丢样/截断，原570个源文件元数据与选中图文内容不变。正准备12步真实图像校准；稀疏训练及效果对照仍待完成。
- Task6 12 步原生校准、全局掩码与 32 步稀疏训练已完成，独立证据审查通过。原校准包装器因漏拼 `RUN_NAME` 的审计路径退出 1，现有 checkpoint 经单独 CPU 补验退出 0，未重训。稀疏阶段退出 0，1,066,599 个可训练值对应 2,133,198 个一阶和二阶矩元素，756 个 A/alpha/索引哈希一致。[Task6 审查](execution/dit-lori/task-6-review.md)
- Task7 实际采样 worker `76176` 已结束，worker 和启动器退出 0。生成 24 张 512×512 单图，8/8 行 Base 与 LoRI ×0 的解码 RGB 完全一致，8/8 行 LoRI ×1 相对 Base 有变化。Luna 监控记录到前两列完成，随后因用量限制结束。主代理核验原 worker 退出记录与图像产物。[网格](../experiment/task7/session-20261118-a/images/grid.png) · [像素审计](../experiment/task7/session-20261118-a/pixel-audit.json)
- 实测覆盖本机 RTX 5070 Ti Laptop、256×256 目标面积训练和 512×512 采样。逐位恢复在 Task5 的确定性 attention 门禁中成立。Task7 的目视对照呈现服饰、配饰、表情和姿态变化，稳定画风、触发词特异性和 LoRA 质量对齐尚待验证。[综合结果与产物](research/2026-11-18-lori-experiment-results.md)
- 训练与采样完成后，整分支审查发现独立导出入口会重读原掩码。交付提交 `957ccc3f89c1506db8e41ce4d260f529446e142b` 用 checkpoint 已保存的张量与元数据构建适配器。针对性回归从 `FileNotFoundError` 转为通过，相关 CPU 测试为 256 passed、56 skipped。真实第 32 步 checkpoint 副本在失效掩码路径下重导，1,008 个张量与原训练导出精确一致，容器元数据有差别。原训练与采样证据对应 `285f4bd`。[整分支审查记录](execution/dit-lori/final-review.md)

## 调查确认的实现约束

- 论文主流程：固定随机 A，按目标任务损失校准无掩码 B，全局拼接所有目标 B 的绝对值，以分位数阈值保留大幅值位置；B 清零后正式训练。
- 校准与正式训练必须使用同一个 A；掩码产物保存 A 张量及模块形状信息，不依赖两次运行隐式 RNG 一致。
- 官方代码的稠密 B 加梯度掩码仍产生完整 AdamW 状态；本项目需要紧凑 `b_values` 参数。
- 真实训练器原生支持外置 CACHE_DIR 和 jsonl 引用原始文件，无需搬运源数据。
- 初步低显存路线：复用训练器现有 ckpt/ring4/actoff/optstream/gradoff 管线。校准和正式训练分进程构建参数与优化器，避免替换 Parameter 后的过期状态。
- 本地来源经复制组核查为官方快照 `e60bf3260d24cea6fdd61be53893dcddad529652`，1093 个官方文件匹配；“优化后的训练器”指该快照自带的优化机制，未发现另加私有补丁。
- 本机由复制组核实为 RTX 5070 Ti Laptop 12227 MiB、61.68 GiB RAM。12 GB 实测结论适用于上面的输入与配置。

## Task 10：全量 285 图、稠密 B、900 步（2026-11-18）

- Task 10 用 `D:\Develop\nigenmame-LoRA\dataset-raw\round2-caption` 的全部 285 对图文，从零训练固定 A 的稠密 B：900/900 次有效更新、3.158 个数据轮次，rank 8 / alpha 16 / `attn-mlp` 252 个目标、INT8 ConvRot 主干 + BF16、AdamW `1e-4` 恒定、`ckpt-ring4-actoff-optstream-gradoff`，trainer `957ccc3f89c1506db8e41ce4d260f529446e142b`、git status 干净、源码零改动。保留 step 285 / 570 / 855 / 900 四个检查点。[数据与冻结报告](execution/dit-lori/task-10-data-preparation-report.md) · [训练报告](execution/dit-lori/task-10-dense-training-report.md)
- 训练终态独立复核 84 项（通过 82、无法核对 2、失败 0）：`metrics.jsonl` 恰 900 行、step 连续、无跳步无非法值，loss 均值 0.70493；CUDA 峰值 2,432.75 / 2,724.61 MiB allocated/reserved；`optim.pt` 为 252 个 entry 的四元组，两列 AdamW 矩各 10,665,984（合计 21,331,968）全 fp32 有限、更新计数全 900；A/alpha 与 task9 step 232 逐张量 504/504 相同，稠密 B 252/252 非零并被更新（全局 L2 5.4967 → 9.7500）；lycoris 与 comfy 导出均与检查点逐张量一致。[review-phase-a.json](../experiment/task10/review-phase-a.json)
- 出图：6 组提示 × seed 42/43 × 有/无 `nigenmame, ` 前缀 = 24 行 × 3 列（Base / step 285 / step 900）= 72 张 512²，生成启动器 4,337.65 s、CUDA 峰值 2,120,034,816 / 2,212,495,360 B。三列 MAE 均值 30.36 / 34.15 / 24.62，21/24 行 step 900 偏离更大；6 个 task9 锚点的 Base 解码 RGB sha256 与归档逐位相同，采样条件零漂移。[对照报告](execution/dit-lori/task-10-comparison-report.md) · [三列网格](../experiment/task10/comparison-20261118-a/matched-grid.png) · [一屏总览](../experiment/task10/summary-contact-sheet.png)
- 图像产物独立复核 34 项（通过 33、失败 1）：原生 `grid.png` 抽 30 个图块位级一致、锚点 6/6、逐行 MAE 24/24 重算零偏差、blind-grid 列序 [2,0,1] 与 key 自洽；唯一失败项是盲评可被同目录 `pixel-audit.json` 的逐列哈希在 1 秒内旁路（表本身不泄露列名、24 行齐全、三维度齐全）。[review-phase-b.json](../experiment/task10/review-phase-b.json)
- 裁决：训练完整性达标；视觉信号**未达**方案第 73 行「小样本视觉增益」门槛——本轮没有真人盲评数据，且 G1（粉色，r00/r01）与 G4（图书馆，r12–r15）在 step 900 出现可重复的面部（细缝/无瞳眼）与手部（手掌融合成片、布偶分离的双手套）退化。三条边界：无独立留出集、触发词不可证伪、285 vs 900 混淆步数与数据轮次。[独立复核](execution/dit-lori/task-10-independent-review.md)
- 第四列 step 570 已补齐并独立复核 31 项（通过 30、失败 1）：归档 Base/285/900 只读复用（72 张 PNG mtime 仍为 19:12），24 张 570 由自写 `render_single.py` 渲染（worker exit 0，1,562.74 s，CUDA 峰值 2,120,034,816 / 2,212,495,360 B）；单列 canary 在 r00/r14 与归档 step285 列**逐位相同**（`a77ac81d…` / `a8df90e2…`）；quad-grid 96/96、contact sheet 96/96 图块位级一致；144 个 MAE 单元重算零偏差。四列均值 Base→285 **30.36** / Base→570 **31.44** / Base→900 **34.15**，285→570 21.93、570→900 20.78；**13/24 行 570 居中、12/24 行严格递增、5/6 组单调**（图书馆组 570 回落为唯一非单调）。[570 报告](execution/dit-lori/task-10-570-comparison-report.md) · [四列总览](../experiment/task10/comparison-570-20261118-a/quad-contact-sheet.png)
- 570 的视觉退化**行相关不均匀**：r08 右侧脸空白、r12 闭眼 + 手部融合、r00 布偶分离手套手、r06 提示外光环在 570 已出现，而 r14/r20/r01 仍结构可接受；说明退化随训练累积且集中在特定行，不是 900 独有。唯一失败项是盲评的**目录级隔离**：未打乱的 `quad-grid.png` 与 `blind-quad-grid.png` 同目录，数秒内即可反解列序（文件级的 `audit-hashes/` 隔离本身有效）。另记录 `sample.py` 的 adapter 路径不等价（Base 2/2 相同、adapter 2/2 不同，MAE 3.23/4.52），建议 trainer 侧补 canary。[review-phase-c.json](../experiment/task10/review-phase-c.json) · [独立复核 §8](execution/dit-lori/task-10-independent-review.md)

## Task 11：稀疏度对比（2026-11-19）

- 在上面的 285 图配方上，用一份 285 步 calibration 生成三档掩码：移除 90% / 60% / 30% → 保留 **1,066,599 / 4,266,394 / 7,466,189** 个 B 值（密度 0.10000006 / 0.40000004 / 0.70000002），逐层 90% ⊆ 60% ⊆ 30%（零违反），掩码外 B 全零、三份 A/alpha 与 calibration 及初始摘要逐张量一致。三条链各训练 900 步 / 4 检查点，优化器两列矩 **2,133,198 / 8,532,788 / 14,932,378**（恰为 2× 保留数，fp32 有限、计数 900），loss 均值 0.70588 / 0.70404 / 0.70424，步时中位 1.7166 / 1.7220 / 1.7201 s；12 份 lycoris 导出可读且 A/alpha 一致。[稀疏度对比报告](execution/dit-lori/task-11-sparsity-comparison-report.md)
- 出图 13 列 × 12 行 = 156 格（只新渲染 9 个稀疏列 = 108 张，Base/D285/D570/D900 复用归档），生成 7,385.72 s、CUDA 峰值 2,120,034,816 / 2,193,620,992 B；`grid-13col.png` 156/156、3 张 group sheet 各 52/52、overview 156/156 与单图位级一致，canary 4/4，归档 48 图块未变。每列 vs Base 均值：Dense 28.58 / 30.78 / 33.60，**90% 28.28 / 28.28 / 29.06**、60% 32.83 / 30.82 / 31.27、30% 32.37 / 31.58 / 33.76；同档跨步 570→900：Dense 22.22、**90% 12.18**、60% 17.79、30% 20.57。[13 列网格](../experiment/task11/comparison-20261119-a/visual/grid-13col.png) · [总览](../experiment/task11/comparison-20261119-a/visual/sheet-overview.png)
- 独立复核 30 项全部通过。[review-phase-d.json](../experiment/task11/review-phase-d.json) · [独立复核 §9](execution/dit-lori/task-10-independent-review.md)
- 三条新增限制：① **90% 是早期饱和而非没学到** —— 像素三列几乎一样（570→900 仅 12.18），但 B 的 L2 仍从 3.324 长到 5.225（+57%），且 285 步已达 vs-Base 28.28；② **基线不同源** —— 本轮 calibration 的 B 与 task10 dense285 非逐值复现（0/252 层、最大差 8.99e-3、L2 5.2526 vs 5.4986），差异从首步 grad_norm 就出现（0.144279 对 0.143461，指向反向非确定性），而 Dense 四列取自 task10 归档；③ 仍无独立留出集与真人盲评（13 列未做盲评），像素差不是质量排名。

## Task 12：A 冻结取证审计（2026-11-19）

- 用户质疑「dense 900 是否真是 LoRI、训练中 A 是否冻结」。四层取证（静态代码 / 24 个检查点 / 4 份优化器 / 受控 2 步 GPU 运行）结论：**A 确实冻结，且是构造性冻结** —— `A`/`alpha` 用 `register_buffer` 注册（`adapters/kinds/lori.py:47-48`），不进 `named_parameters()`，`trainable_params()` 只枚举参数（`adapter_set.py:246-262`），优化器只从该枚举建组（`loop.py:173-202`、`entry.py:300`）。24 个文件（21 检查点 + 3 掩码，跨 task6/8/9/10/11）的 A 组合摘要 **24/24** 相同（`7f8b433f…`）、alpha **24/24**（`930ce824…`）、B **24/24** 互不相同；四份 `optim.pt` 各 **252** entry，矩元素 21,331,968 / 2,133,198 / 8,532,788 / 14,932,378（=2×B），**无 A 形状状态**、`master` 与同 step 的 B 摘要多重集相等。[a-frozen-audit.md](execution/dit-lori/a-frozen-audit.md) · [a-frozen-audit.json](../experiment/task12-audit/a-frozen-audit.json)
- 受控 2 步运行时（dense 与 sparse90 各一次，输出到 `experiment/task12-audit/tmp/` 后删除）：A/alpha 252+252 全部 `requires_grad=False`、非 Parameter、不在可训练集合与优化器参数组，A 的字节/版本号/摘要前后不变、grad hook 零命中；B 252/252 收到梯度且字节改变；`optim.pt` 252 entry、矩元素 21,331,968 / 2,133,198。四次独立运行首步 loss 完全相同但首步 `grad_norm` 每次略异（0.143461 / 0.143685 / 0.144279 / …），独立复现了「前向确定、反向不确定」。
- **导出语义是否编码冻结：否。** comfy 导出（`lora_down`/`lora_up`/`alpha`，504 张量）经通用 lycoris 路径加载为 `LoConModule`，两侧都是 `nn.Parameter(requires_grad=True)` —— 可被当普通 LoRA 续训并训到 A；稀疏 comfy 还把 mask 展开成同尺寸稠密 B（49,601,320 B，非零 1,066,599）。原生 lycoris（756/1008 张量）读回为普通 `Tensor`，通用加载器算法匹配返回 `None`（私有格式）。
- 概念界定：dense 900 是 `phase=calibration`、`sparsity=0.0`、全量 AdamW 状态，对应论文 `s=0` 的 **LoRA-FA 形态**，**不是** LoRI-S；只有三档稀疏（`active_indices`+`b_values`+紧凑状态）才是论文定义的 LoRI。建议措辞修正 5 处（两份 task-10 对照报告标题、数据准备报告第 3/218 行、复核文档标题），未改动他人报告。
- 附注：审计期间发现训练器工作树存在**未提交**的导出/检查点元数据补丁（8 文件 +139/−4 + 未跟踪测试，新增 `kohakumoss_lori_phase/mechanism/trainable/a=frozen_buffer/a_sha256`、`kohakumoss_container`、`kohakumoss_lori_comfy=inference_only_dense_b` 等）——**不是我的改动**，也不在任何已交付产物里（四个导出文件均缺这些键），交付文件需重新导出。

## Task 13：LoRI 导出自描述改造审计（2026-11-19）

- 对工作树里的导出/检查点自描述改造（8 文件 +157/−5，新增 `tests/adapters/test_lori_export_facts.py`）做代码审查 + 真实模型集成 + 训练/resume 回归：**需求满足、无阻断项、可放行 90% 稀疏 900 步重训**。[lori-export-audit.md](execution/dit-lori/lori-export-audit.md) · [lori-export-audit.json](../experiment/task13-audit/lori-export-audit.json)
- 真实模型 3 检查点 × 2 格式共 6 个导出：facts 6 键齐全（sparse 另有 kept_total 1,066,599 / 4,266,394 与 rank 8，comfy 另有 `inference_only_dense_b`），原生键集合恰为 `{A,alpha,b_dense}` 或 `{A,alpha,b_values,active_indices}`、无交叉污染；`a_sha256` 独立复算一致且跨 step 稳定（`976522f0…`）；与旧导出**张量逐位相同**（756/756、1008/1008、504/504）；comfy 126 个非融合层 down==A、up==B，42 个融合 qkv down==cat(A_q,A_k,A_v)、up==2×block_diag(B)；sparse+comfy 警告各 1 次、calibration 0 次。
- 训练回归：dense/sparse 2 步首步 loss 均为 **0.7369208931922913**（需 per-run kernel cache 隔离；共享旧 cache 时为 0.7371941，属 kernel 选择差异）；`state.json` lori 块字段正确、`optim.pt` 252 entry、矩 21,331,968 / 2,133,198；resume 新检查点 1 步与**旧格式**检查点 1 步（step 901）均正常。测试：新文件 15 passed、GPU 全套 299 passed、training 子集 22 passed/2 skipped、black/ruff 通过。
- 非阻断发现 5 条（1×P2 + 4×P3）：F1 保存/导出路径对全量 state 做了两趟 CPU 拷贝（a_sha256 与 kept_total 各调一次 `_layers`）；F2 `state_metadata()` 在多卡下只描述本地分片；F3 缺 full 与旧检查点续训的回归 case；F4 merged/diffusers 不带 facts；F5 警告在 metadata() 内发出。

## Task 14：原生分辨率 90% 稀疏 900 步 + 6 列出图（2026-11-19）

> 会话目录仍沿用 `experiment/task13/`；本节按时间顺序记为 Task 14。

- 把训练分辨率从 256² 提到**原生零缩放**（`TARGET_AREA=1048576`、`UPSCALE=never`）：285/285 的 target 等于原始尺寸、面积比恒 1.0、裁掉像素 0；缓存重建 catalog 285 / text 286 / latents 285、图像 token 1,135,799。四段续训（250/550/850/900）全 exit 0，900 步调用 **5,613.6 s**（四段 5,582.1 s + 探针 31.5 s）、中位步时 **6.0546 s**、CUDA 峰值 2,605,227,520 / 2,925,527,040 B，无 OOM。
- 4 个检查点 `state.json` 均含正确 lori 块，optim.pt 252 entry / 矩 2,133,198 / 计数 900，A/alpha canonical 与 task11 一致，`b_values` 1,066,599 全非零且 `active_indices` 与 90% 掩码 252/252 相同；B 的 L2 2.105→3.327→4.308→4.398（252/252 层变化）。4×2 导出 + 2 份 run 级 facts 完整。[原生对比报告](execution/dit-lori/task-13-native-comparison-report.md)
- 6 列 × 12 行出图：只新渲染 48 张（Base 与 256² 稠密参考复用 task10），生成 3,415.7 s；`grid-6col.png` 72/72、3 张 sheet 各 24/24、overview 72/72、canary 4/4、归档 24 块未变。像素（vs Base）：稠密 900@256 **33.60**、N250 **16.16**、N550 **23.26**、N850 **26.45**、N900 **26.07**；相邻位移 18.00/14.46/**6.09**；与 256² 参考距离 32.30→29.88→28.12→**28.04**。[6 列网格](../experiment/task13/comparison-20261119-a/visual/grid-6col.png) · [总览](../experiment/task13/comparison-20261119-a/visual/sheet-overview.png)
- 独立复核 26 项全部通过。[review-phase-e.json](../experiment/task13/review-phase-e.json) · [独立复核 §11](execution/dit-lori/task-10-independent-review.md)
- 三条限制：① **饱和而非单调增强**——参考距离与相邻位移单调收敛，但 vs-Base 最后一步 26.45→26.07 略回落、B 仍有 252 层在动；② **不能归因于尺度不匹配**——没有原生采样/512² 训练的对照臂，训练分辨率、loss 尺度（0.375 vs 0.705）、采样分辨率与掩码来源四重混杂；③ **掩码来自 256² 校准**（task11 90%），本轮只改变了训练分辨率。
- 过程异常均已坐实：cache attempt1 exit 1（审计统计 bug，native 三段 exit 0，recheck 18.2 s 不重建）、seg550 attempt1 exit 1（构建期 B 非零断言，未跑步，run.py 两次 amendment `160af90f`→`948aea1c`→`f411c940` 末值与磁盘一致）、step 251/551/851 计时伪影（0.046/0.0215/0.0161 s，非跳步）、comparison canary-attempt1 exit 1 后成功。
- **task10 文件数差异**：当前递归 3,350 个文件；`images/` 74、`images-570/` 25、两份 freeze 18/18 与 11/11、session freeze 9/9 全部命中，关键产物 mtime 仍是 2026-11-18；差异只能落在未被 manifest 记录的 `runtime/`（372+1,386）与 `__pycache__` 缓存里，**没有关键产物缺失**。
