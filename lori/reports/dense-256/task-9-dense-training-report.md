# LoRI 稠密 B 对照训练

稠密组从零开始完成 232 次更新。沿用 12 张图像、256² 目标面积、INT8 主干、BF16、rank 8、alpha 16、seed 42 和相同的流式融合 AdamW，学习率保持 `1e-4`。原生配方用 `phase="calibration"` 构建完整 B，这次将它训练到最终 checkpoint 用作稠密对照。CPU 和实际 GPU 初始化均验证全部 252 个 A 和 252 个 alpha 与稀疏组一致，GPU 还验证了 B 全零、主干冻结。训练使用独立缓存与全新优化器，数据游标从 0 开始。 [配方](../../../experiment/task9/session-20261118-a/resolved.json) · [GPU 初始状态](../../../experiment/task9/session-20261118-a/stages/dense/initial.json)

| 指标 | 稠密 B | 稀疏 B |
| --- | ---: | ---: |
| 更新次数 | 232 | 232 |
| 可训练 B 值 | 10,665,984 | 1,066,599 |
| FP32 一阶和二阶矩元素 | 21,331,968 | 2,133,198 |
| 原生 `optim.pt` 字节数 | 128,213,819 | 13,039,483 |
| 完整 232 步平均 loss | 0.71226 | 0.72289 |
| steps 1–32 平均 loss | 0.72541 | 0.72731 |
| steps 33–232 平均 loss | 0.71016 | 0.72218 |
| steps 201–232 平均 loss | 0.66561 | 0.68294 |
| step 232 loss | 0.42150 | 0.43177 |
| steps 33–232 中位步时 | 1.519 s | 1.608 s |
| PyTorch 峰值分配 / 保留 | 2.43 / 2.72 GiB | 2.40 / 2.69 GiB |

用三次实际运行的配置和只读缓存复放数据计划，稠密 1–232 与稀疏 1–32、33–232 的每步源行、裁切、caption 及 token 数均一致。复放结果还与三个 checkpoint 的数据计数、游标和日志中最后一批的行 ID 相符。两组首步 loss 都是 0.5428468585，232 个记录的 sigma 值及最终 sigma/noise 生成器状态一致。稀疏组另用 12 步校准来选掩码，稀疏训练开始前清零 B，优化器也重新初始化。 [匹配审计](../../../experiment/task9/session-20261118-a/matched-training-audit.json)

稠密组 232 个 loss 和梯度范数均有限，跳过次数为零。252 个稠密 B 全部更新，252 组优化器计数达到 232，两列 AdamW 矩张量均保留完整 B 的二维形状分布。表中 loss 来自两组反复使用的同 12 张训练图片；这次训练没有留出集，不能用这些 loss 判断泛化。 [稠密指标](../../../experiment/task9/session-20261118-a/training/lori-dense/metrics.jsonl) · [稠密审计](../../../experiment/task9/session-20261118-a/final-audit.json) · [稀疏续训审计](../../../experiment/task8/session-20261118-a/final-audit.json)

以最终原生 checkpoint 的训练坐标计算 `|| (alpha/rank) BA ||_F`，稠密组 252 层均高于稀疏组，层级比值的中位数为 2.60。所有层合并的 Frobenius 范数分别为 11.86 和 5.45。这个量衡量参数更新幅度，生成效果由匹配图像评估。 [逐层范数](../../../experiment/task9/session-20261118-a/matched-training-audit.json)

稠密运行的实际 worker、启动器和等待进程均退出 0，墙钟耗时 388.14 秒。两组均使用主机流式优化器，稀疏组的状态元素约为稠密组的十分之一，本次总峰值显存仅下降约 34 MiB。原生 LyCORIS 导出分别为 74,397,152 和 44,654,660 字节。两份 ComfyUI 兼容导出均为 49,601,320 字节。源数据与参考 checkpoint 的哈希在训练完成后复核一致。 [进程结果](../../../experiment/task9/session-20261118-a/stages/dense/result.json) · [启动器退出](../../../experiment/task9/session-20261118-a/stages/dense/exit.json)

[稠密 step 232 原生权重](../../../experiment/task9/session-20261118-a/training/lori-dense/export/lori-dense-lycoris.safetensors) · [稠密 step 232 完整 checkpoint](../../../experiment/task9/session-20261118-a/training/lori-dense/checkpoints/step_00000232/) · [稀疏 step 232 原生权重](../../../experiment/task8/session-20261118-a/training/lori-sparse/export/lori-sparse-lycoris.safetensors)
