# DiT LoRI 整分支审查

2026-11-18。审查基线为 `7a5b94a`，原评审 HEAD 为 `285f4bd`。依据设计与实现计划、整分支 review package、独立阶段评审和实际运行记录。开始审查时源码工作树干净，`git diff --check 7a5b94a..285f4bd` 通过。

## 最终结论

**通过。** `957ccc3` 修复并关闭独立导出问题，整分支没有待处理的 Critical 或 Important 项。训练及配对推理的实际运行基于 `285f4bd`。后续提交仅变更独立导出入口及其回归测试。

## 原始审查结论

原评审 HEAD 有 **1 项 Important、0 项 Critical**。训练、CPU 掩码提取、训练内导出和配对推理有通过证据。独立导出命令需要修复后复审。

**Important — 独立导出无法在原掩码不可用时读取稀疏原生检查点。** `scripts/export/export.py` 调用的 `src/kohakumoss/adapters/export/cli.py:96` 已读取检查点的张量与元数据，但调用 `apply(model, config)` 时遗漏两者。`adapters/build/apply.py:75` 因此重新读取配置中的 `mask_path`，在加载检查点张量前失败。使用实际 Task 6 step32 的 `trainable.safetensors` 和 CPU `nn.Linear`，将原掩码模拟为不可用，通过 `export_main` 重现 `FileNotFoundError`，确认只尝试了原掩码路径。该探针没有修改或移动原文件，也未加载 MOSS 权重或写入输出。最小修复是传递已有 `initial_state=tensors` 与 `initial_metadata=meta`，沿用 `AdapterStack._raw_tensors` 的构建方式，并从独立导出入口覆盖缺失掩码的回归场景。

## 跨阶段证据

- LoRI 数学、全局绝对值分位数及并列保留规则、A 与索引的持久缓冲区、压缩 B 的构建时机和恢复身份经阶段审查。Task 4 的低精度折叠导出问题已由 `f34464a` 修复并复审。[阶段 1](task-1-review.md) · [阶段 2](task-2-review.md) · [阶段 3](task-3-review.md) · [阶段 4 修复](task-4-fix1-review.md)
- 实际 MOSS INT8 兼容门禁在 `ckpt-ring4-actoff-optstream-gradoff` 下使用合成 latent/text 输入，退出 0，pytest 记录 1 passed、1 warning。默认 Triton 校准和稀疏更新均有限。确定性 math-SDPA 分支在实际权重上精确恢复继续，核对 B 参数身份、固定 A 和基础模型摘要。常数调度的 AnySchedule 警告及带探针的时间/显存测量范围按原报告记录。[阶段 5 评审](task-5-review.md) · [实际门禁结果](../../../.tools/task5/run-20261117T182252Z/artifacts/result.json)
- 用户数据流程经离线缓存、12 步校准、CPU 全局掩码提取和新进程的 32 步稀疏训练。校准 launcher 原始 exit 1 来自训练结束后的审计路径错误，保留了原始记录并通过独立 CPU 审计恢复。稀疏运行退出 0，252 个 B 向量更新，A/alpha/索引摘要一致，实际 AdamW 两份动量共 2,133,198 个元素，等于 `2 × 1,066,599 nnz`。确认了最终检查点与原生导出的文件摘要。[阶段 6 独立评审](task-6-review.md) · [稀疏训练报告](task-6-sparse-report.md)
- `scripts/sample/compare_grid.py` 成功读取该原生导出。Task 7 退出 0，四个提示词分别带和不带触发词，同种子生成 24 张。重新核对了全部 26 个交付文件的 SHA256（含网格及元数据），并在 CPU 逐图解码复核 8 行 RGB 摘要。每行 ×0 与 base 一致，×1 均产生像素差异。[采样结果](../../../experiment/task7/session-20261118-a/stages/generate/result.json) · [像素审计](../../../experiment/task7/session-20261118-a/pixel-audit.json)

## 证据边界

实跑覆盖 RTX 5070 Ti Laptop（12GB 档）、`256²` 目标面积的训练配方及 `512 × 512` 的配对采样。默认 Triton 的逐位续训、实际用户数据全模型续训、Task 6 训练后全量基础模型摘要、ComfyUI 实际加载和 8GB/6GB 硬件均未实测。配对图验证适配器有输出效应，风格、角色一致性和 LoRA 质量平价仍需专门实验。稀疏 B 的物理参数和动量按 nnz 储存，前向仍重建临时稠密 B。本审查没有重新训练或生成。

## 独立导出修复复审

`285f4bd..957ccc3` 的生产变更仅在 `export_main` 将已经读取的张量与元数据传给 `apply`。稀疏 LoRI 以检查点中的索引尺寸构建参数并验证元数据，再原地加载数值。LoRA 等其他算法在 `_lori_state` 直接返回，仍走原来的构建和加载路径。复审时 HEAD 为 `957ccc3f89c1506db8e41ce4d260f529446e142b`，源码工作树干净，整段 `git diff --check` 通过。

修复者记录的新回归测试在修复前因缺失掩码报 `FileNotFoundError`，修复后通过。相关 adapter/checkpoint 套件记录 256 通过、56 跳过，跳过项未计入通过。复审读取了测试实现和修复者的执行记录，并独立检查 CPU 实权重导出产物。临时的 step32 复制件仅改写 `mask_path` 为不存在的路径，`scripts/export/export.py` 使用实际 INT8 底模进行 CPU 导出。复审重新计算所有 1,008 个张量的数据摘要，名称、形状、dtype 和数据均与原生导出一致，两个原件的文件 SHA256 亦一致。[CPU 导出探针](../../../.tools/final-export-check/smoke.py)
