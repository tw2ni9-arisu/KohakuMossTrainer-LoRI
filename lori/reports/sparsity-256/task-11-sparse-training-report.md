# Task 11 稀疏度对比：固定 A 的 90% / 60% / 30% 移除率 900 步训练报告

日期 2026-11-19（本地）。会话 experiment/task11/session-20261119-a/。在 RTX 5070 Ti Laptop 12GB 上，用同一批 285 图、同一固定 A、同一套配方，先用 285 步校准得到稠密 B，再用校准结果生成三档稀疏掩码（移除 90% / 60% / 30%），各从零训练 900 次有效更新，并完成指标、检查点、优化器、张量、导出与冻结复核的终态审计。训练器源码零改动（HEAD 957ccc3f89c1506db8e41ce4d260f529446e142b，git status 干净），task10 的缓存与清单只读复用、未改动，未出图、未删除任何检查点或导出。

**结论边界（先写在前面）**：三档都完成了 900 次有效更新，A/alpha 504 张量与初始哈希完全一致，紧凑 B 全部更新，优化器矩元素严格等于该档保留数；但本轮只比较「同一 A、同一数据、同一预算下不同移除率」的训练可观测量，**不与稠密做质量排名，也不声称任何画质结论**——没有留出集，也没有出图。

| 指标 | 校准（285 步） | 移除 90% | 移除 60% | 移除 30% |
| --- | ---: | ---: | ---: | ---: |
| 保留 B 值 | 10,665,984（稠密） | 1,066,599 | 4,266,394 | 7,466,189 |
| 有效更新 | 285 | 900 | 900 | 900 |
| 启动器墙钟 | 526.172 s | 1,591.890 s（+2.871 s 复审） | 1,599.636 s | 1,602.838 s |
| 全量平均 loss | 0.71292 | 0.70588 | 0.70404 | 0.70424 |
| 步时中位数 | 1.7039 s | 1.7166 s | 1.7220 s | 1.7201 s |
| CUDA 峰值 allocated / reserved（B） | 2,612,148,736 / 2,925,527,040 | 2,576,582,144 / 2,891,972,608 | 2,620,938,752 / 2,980,052,992 | 2,669,804,032 / 3,101,687,808 |
| host peak_wset（B） | 14,245,687,296 | 14,294,994,944 | 14,332,936,192 | 14,369,595,392 |
| optim.pt 字节 | 128,213,819 | 13,038,715 | 51,436,795 | 89,835,963 |
| AdamW 矩元素 | 21,331,968 | 2,133,198 | 8,532,788 | 14,932,378 |
| A / alpha 与初始一致 | 252 + 252 | 252 + 252 | 252 + 252 | 252 + 252 |
| 紧凑 B 全部更新 | — | 252 / 252 | 252 / 252 | 252 / 252 |

[run.py（launcher + worker）](../../../experiment/task11/session-20261119-a/run.py) · [freeze](../../../experiment/task11/session-20261119-a/freeze.json) · [resolved](../../../experiment/task11/session-20261119-a/resolved.json) · [校准审计](../../../experiment/task11/session-20261119-a/stages/calibration/result.json) · [掩码审计](../../../experiment/task11/session-20261119-a/stages/masks/masks.json) · [sparse90 审计](../../../experiment/task11/session-20261119-a/sparse90-final-audit.json) · [sparse60 审计](../../../experiment/task11/session-20261119-a/sparse60-final-audit.json) · [sparse30 审计](../../../experiment/task11/session-20261119-a/sparse30-final-audit.json)

---

## 1. 术语约定

用户说的「稀疏度 90% / 60% / 30%」按项目既有约定是**移除比例**（训练器的 sparsity 参数，阈值取全局 abs(B) 的该分位数），不是保留比例。本报告一律同时写清两者：

| 档位名 | 移除 | 保留 | 保留比例（实测） |
| --- | ---: | ---: | ---: |
| sparse90 | 90% | 10%（约 1.07M 个 B 值） | 10.0000056% |
| sparse60 | 60% | 40%（约 4.27M 个 B 值） | 40.0000038% |
| sparse30 | 30% | 70%（约 7.47M 个 B 值） | 70.0000019% |

选择规则是 abs(B) >= 全局分位阈值，阈值并列全部保留，因此保留数是整数、彼此严格递增，且三档索引集合严格嵌套（90% 属于 60% 属于 30%，实测均为 true）。

## 2. 范围、配方与边界

配方与 Task 10 稠密 900 步逐项一致，只改必要项：rank 8 / alpha 16、attn-mlp 252 个目标模块、256² 目标面积、batch 1、梯度累积 1、INT8 ConvRot 主干、BF16、AdamW 1e-4 恒定（零权重衰减）、seed 42、ckpt-ring4-actoff-optstream-gradoff、RESUME 空串。三档稀疏 run 的差别只有 ALGO_KWARGS 的 sparsity / mask_path 与 RUN_NAME（resolved.json 的 differing_knobs 显示：sparse90 与 sparse60 的差异键恰为 ALGO_KWARGS、RUN_NAME 两项）。校准 run 与其他三项的差异键为 ALGO_KWARGS、KEEP_LAST、OUTPUT_DIR、RUN_NAME、SAVE_EVERY、STEPS——即只在步数、保存节奏、输出目录与阶段参数上不同。

- 数据与缓存：只读复用 experiment/task10/session-20261118-a/cache/（catalog 285、text 286、latents 285）与 experiment/data/round2-caption-285/dataset.jsonl（285 行，sha256 4030fdf3094b91cff164257f70fd5290674dbf5b24f4a1e4a75d19036e869ead）。本轮自己的 common.py 与 task10 的逐项相同，只把 CACHE_DIR 与 DATASETS[0].catalog 指向 task10 的缓存；缓存记录的 text / latents settings 与本轮训练配置解析结果逐键相等（启动前置断言）。
- 未做：任何出图、图像网格、盲评；任何训练器源码改动；任何缓存重建；与稠密的质量排名。

## 3. 执行链与退出码

五个 stage 串行、各自独立进程（launcher 起子进程，上游 result/exit 任一非 0 即停）：

| stage | 退出码 | launcher 墙钟 | worker 墙钟 | 训练调用墙钟 | 产物 |
| --- | ---: | ---: | ---: | ---: | --- |
| calibration | 0 | 526.172 s | 524.46 s | 506.61 s | calibration/lori-calibration/ |
| masks | 0 | 14.355 s | 13.51 s | —（CPU，CUDA_VISIBLE_DEVICES 为空） | masks/sparsity-90/60/30.safetensors |
| sparse90 | 0 | 1,591.890 s + 2.871 s 复审 | 1,590.11 s + 2.87 s 复审 | 1,575.37 s | runs/lori-sparse90/ |
| sparse60 | 0 | 1,599.636 s | 1,597.87 s | 1,580.96 s | runs/lori-sparse60/ |
| sparse30 | 0 | 1,602.838 s | 1,601.03 s | 1,580.57 s | runs/lori-sparse30/ |

每个 stage 目录含 launch.json / start.json / result.json / exit.json（另有 initial.json、audit.json 或 masks.json、stdout.log）。GPU 作业严格串行：同一时刻只有一个训练进程，未结束任何无关进程。sparse90 的第一次执行在训练完成后被审计脚本自身的断言顺序错误打断（见第 13 节），随后以「复审现有产物、不重训」的方式通过，因此该 stage 目录同时保留 result.attempt1.json / exit.attempt1.json 与 recheck.json。

## 4. 校准：285 步稠密 B（1 个数据轮次）

| 项目 | 实测 |
| --- | ---: |
| 有效更新 | 285 / 285 |
| loss 均值 / 最小 / 最大 | 0.71292 / 0.36593 / 0.95810 |
| loss 首 / 末 / 中位 | 0.73692 / 0.88961 / 0.72479 |
| grad_norm 均值 / 最大 / 末值 | 0.22701 / 3.57082 / 0.08636 |
| 步时中位 / 均值 / 最小 / 最大 | 1.7039 / 1.7372 / 1.4965 / 7.9809 s（最大为 step 1） |
| CUDA 峰值 allocated / reserved | 2,612,148,736 / 2,925,527,040 B（2.4328 / 2.7246 GiB） |
| host peak_wset | 14,245,687,296 B（13.2674 GiB） |
| 指标行数 / 连续步 | 285 行，step 1..285 |
| skipped / nonfinite / nonfinite_params | 全 0 |
| lr | 恒 1e-4 |

metrics.jsonl 260,371 B，sha256 25c3c7c60e63e8c12c0152143c2834cac760bd1c6da21d066cea8943fdc5270a。

最终检查点 calibration/lori-calibration/checkpoints/step_00000285/：state 为 step 285、samples 285、loader {"seed":42,"start":285}、tokens 66,900、text_tokens 112,490、joint_sq 113,361,806、world 1。文件：

- state.json 181 / 38b355b89faa44f59ef6fd97f5568b845c0c8affc1833817cdc653891b395597
- trainable.safetensors 74,397,040 / c9527e6b1152733fa0d43c5f3952fd752e36e16b6364029d83910b58ef8de228
- optim.pt 128,213,819 / 7a223e796af359e0fac9948ee08adcfeab223de8e8668a62741aef156b85ccaf
- rng.0.pt 20,105 / 3f146c88914d2bea4e75f6b4e3616bf8f3c3781bf0375139d4153f6432c23706
- export/lori-calibration-lycoris.safetensors 74,397,152 / ec342fbe035ef018d5d473b4cc2eb027351372a20b9266b9eb314d4df50b5958
- export/lori-calibration-comfy.safetensors 49,601,320 / 6c0aae418f6ae372d94dddcbd13fb5d30441ed868ce2d44cb8812705289790bf

稠密 B：252 层、10,665,984 个值全部非零且有限（单层非零最小值 24,576、最大值 131,072），全局 L2 范数 5.25066。形状集合 {[3072, 8], [16384, 8]}。

## 5. 三档掩码（CPU，CUDA 未初始化）

掩码由训练器自带的 KohakuMossTrainer/scripts/adapters/lori_mask.py 生成（以 calibration 的 trainable.safetensors 为唯一输入，runpy 调用，--sparsity 分别 0.9 / 0.6 / 0.3），CPU 阶段 CUDA_VISIBLE_DEVICES 置空，start 与 end 两次断言 torch.cuda.is_initialized() 均为 False。三次生成分别耗时 0.628 / 0.743 / 0.777 s。

| 档位 | 移除 | 保留数 | 保留比例 | 精确移除比例 | 阈值 | 单层保留 min / max | 文件字节 | sha256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sparsity-90 | 90% | **1,066,599** | 0.1000000562536 | 0.8999999437464 | 0.002638536039739847 | 1,197 / 21,142 | 44,654,540 | 23df6144faa11c7a561da9d3ad8170cd6c70dfc0d4358702d1d02f3cbf2f5e40 |
| sparsity-60 | 60% | **4,266,394** | 0.4000000375024 | 0.5999999624976 | 0.0013388432562351227 | 7,790 / 62,463 | 83,052,792 | 9af51d994227640661bb0b9cf4a90b8a1ccdbddc647b42bd904571d12cb34485 |
| sparsity-30 | 30% | **7,466,189** | 0.7000000187512 | 0.2999999812488 | 0.0006112979608587921 | 15,515 / 97,603 | 121,451,356 | 53c3e2f0d6103192829b8c4ede549eb093f394f134ba0478942763e3041f9ab1 |

实测保留数比 10% / 40% / 70% × 10,665,984 的理论值分别多 0.6 / 0.4 / 0.2 个（阈值并列按 >= 保留），与 task6 记录的 90% 档 1,066,599 完全一致。

逐条断言（全部通过）：

1. **A / alpha 一致**：三份掩码的 504 个 .A / .alpha 张量逐张量 digest 与 calibration 输入完全相同（组合 digest A 7f8b433fe34e4caa789e9923e44227acdef677e04edc1852a71d6242b884db19、alpha 930ce824c0831386ee6bad65542d768c8b56d2c5a3264511be2d7ae02207c991），彼此也一致；A 形状逐层与输入一致。
2. **保留数单调**：1,066,599 < 4,266,394 < 7,466,189；精确比例见表。
3. **掩码外 B 全零**：每层 b_values 非零计数为 0 且有限；索引为 int64、已排序、唯一、落在 [0, out*rank) 内，长度与 b_values 相等；模块数 252、层名集合与输入完全相同。
4. **可复算**：用 make_sparse_init(calibration_state, sparsity) 从校准状态重新生成后，与落盘文件逐张量 torch.equal 全部相等、元数据 JSON 全等。
5. **元数据自洽**：total = 10,665,984、nnz = 该档保留数、actual_density = nnz / total、sparsity = 0.9/0.6/0.3、阈值有限且非负、三档 calibration_digest 同为 7e883962374ace94f9f612f47cf6d470cb16fd27180843968ee011ac106a8ac8。
6. **嵌套**：90% 档索引集合是 60% 档的子集、60% 档是 30% 档的子集，逐层为 true。

## 6. 三个稀疏 run 的训练指标

三个 run 各自全新优化器、全新 RNG、数据游标从 0 开始（RESUME 空串；worker 在 apply_adapters 处断言 initial_state 与 initial_metadata 均为 None，即从 mask_path 载入掩码而不是恢复状态；252 个 entry 的更新计数全为 900）。metrics.jsonl 各恰 900 行、step 1..900 连续、loss / grad_norm / step_time 全有限、skipped_steps / nonfinite_steps / nonfinite_params 全 0、lr 恒 1e-4。

### 6.1 分段 loss 与步时

| 区段 | 移除 90%：loss 均值 / 最小 / 最大 / 首 / 末 / 步时中位 | 移除 60% | 移除 30% |
| --- | --- | --- | --- |
| 全量 1–900 | 0.70588 / 0.36355 / 0.95945 / 0.73692 / 0.85300 / 1.7166 s | 0.70404 / 0.36330 / 0.95265 / 0.73692 / 0.85479 / 1.7220 s | 0.70424 / 0.36407 / 0.95240 / 0.73692 / 0.85639 / 1.7201 s |
| 1–285 | 0.71152 / 0.36518 / 0.95945 / 0.73692 / 0.89369 / 1.710 s | 0.70933 / 0.36403 / 0.95265 / 0.73692 / 0.89254 / 1.719 s | 0.71073 / 0.36469 / 0.95240 / 0.73692 / 0.89104 / 1.720 s |
| 286–570 | 0.69851 / 0.36355 / 0.89802 / 0.67089 / 0.43986 / 1.718 s | 0.69722 / 0.36330 / 0.89800 / 0.67243 / 0.43963 / 1.725 s | 0.69697 / 0.36407 / 0.89999 / 0.67170 / 0.43766 / 1.724 s |
| 571–900 | 0.70737 / 0.42360 / 0.92500 / 0.68891 / 0.85300 / 1.720 s | 0.70538 / 0.42413 / 0.92884 / 0.68817 / 0.85479 / 1.722 s | 0.70492 / 0.42464 / 0.92657 / 0.68749 / 0.85639 / 1.715 s |
| 801–900 | 0.70421 / 0.47160 / 0.92500 / 0.65662 / 0.85300 / 1.727 s | 0.70195 / 0.47185 / 0.92537 / 0.64448 / 0.85479 / 1.721 s | 0.70180 / 0.47116 / 0.92657 / 0.64243 / 0.85639 / 1.718 s |

补充：grad_norm 均值 / 最大 / 末值分别为 0.06067 / 1.23233 / 0.04510（90%）、0.11992 / 2.38337 / 0.08383（60%）、0.15825 / 1.32912 / 0.13358（30%）——保留参数越多，从全零初始的 B 上累计出的梯度越大，与参数量一致，不构成质量结论。步时最小 / 最大：1.5596 / 7.7531（90%）、1.5479 / 7.7112（60%）、1.5589 / 7.8187（30%）s；最大都出现在 step 1（编译/预热）。后 100 步步时中位分别为 1.7274 / 1.7208 / 1.7179 s。

### 6.2 显存与优化器规模

| 项目 | 移除 90% | 移除 60% | 移除 30% |
| --- | ---: | ---: | ---: |
| 保留 B 值 | 1,066,599 | 4,266,394 | 7,466,189 |
| CUDA 峰值 allocated | 2,576,582,144 B（2.3996 GiB） | 2,620,938,752 B（2.4409 GiB） | 2,669,804,032 B（2.4864 GiB） |
| CUDA 峰值 reserved | 2,891,972,608 B（2.6934 GiB） | 2,980,052,992 B（2.7754 GiB） | 3,101,687,808 B（2.8887 GiB） |
| host peak_wset | 14,294,994,944 B（13.3133 GiB） | 14,332,936,192 B（13.3486 GiB） | 14,369,595,392 B（13.3827 GiB） |
| optim.pt 字节 | 13,038,715 | 51,436,795 | 89,835,963 |
| optimizer entry | 252 | 252 | 252 |
| master / exp_avg / exp_avg_sq 元素 | 各 1,066,599 | 各 4,266,394 | 各 7,466,189 |
| AdamW 矩元素合计 | 2,133,198 | 8,532,788 | 14,932,378 |
| 更新计数 | 全 900 | 全 900 | 全 900 |
| 吞吐（稳态） | 0.5788 样本/s，11.085 TFLOP/s | 0.5772 样本/s，11.053 TFLOP/s | 0.5774 样本/s，11.058 TFLOP/s |

显存读法：三档的 optimizer 状态在主机侧（optstream），但 GPU 侧要驻留 b_values、active_indices 与每步重建的稠密 B，所以 allocated 随保留数单调上升，从 2.3996 到 2.4864 GiB 只增加 88.8 MiB，而 optim.pt 从 13.0 MB 增到 89.8 MB（6.90 倍）。也就是说：**这档稀疏度的差别主要落在适配器与优化器状态的存储上，而不是总显存**——总显存由 INT8 主干主导，与 task6 的既有结论一致（未测稀疏 FLOPs，也不声称总显存按比例下降）。

每 100 步 loss 轨迹（块均值）：

| 步段 | 1–100 | 101–200 | 201–300 | 301–400 | 401–500 | 501–600 | 601–700 | 701–800 | 801–900 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 移除 90% | 0.73143 | 0.70752 | 0.69765 | 0.71358 | 0.69072 | 0.69042 | 0.70525 | 0.71216 | 0.70421 |
| 移除 60% | 0.72933 | 0.70491 | 0.69591 | 0.71238 | 0.68942 | 0.68874 | 0.70364 | 0.71013 | 0.70195 |
| 移除 30% | 0.72957 | 0.70668 | 0.69785 | 0.71208 | 0.68935 | 0.68832 | 0.70315 | 0.70940 | 0.70180 |

三条轨迹在每 100 步块均值上的差都在 0.003 以内，且都呈现「前三块下降、第四块回升、末三块基本平坦」的形状。900 步 = 3.158 个数据轮次，样本几乎不重复，因此看不见持续下降是数据池性质的结果，不能读成学习失败，也不能读成三档等价。

## 7. 检查点清单（4 个 step × 3 档 = 12 个，路径 + sha256）

环形保留后每个 run 恰 4 个目录：step_00000285 / step_00000570 / step_00000855 / step_00000900，每个含 state.json、trainable.safetensors、optim.pt、rng.0.pt 与 export/ 下两份导出；state 的 step = samples = 目录步数，loader = {"seed":42,"start":<同值>}，world = 1，tokens 依次 66,900 / 133,800 / 200,700 / 211,287，text_tokens 112,490 / 224,980 / 337,470 / 355,358，joint_sq 113,361,806 / 226,723,612 / 340,085,418 / 358,156,325（三档完全一致，因为数据游标与采样随机流相同）。

**移除 90%**（runs/lori-sparse90/checkpoints/）：

- step_00000285/：state.json 181 / 38b355b89faa44f59ef6fd97f5568b845c0c8affc1833817cdc653891b395597；trainable.safetensors 44,654,764 / 340a2e3a23318a316cdd1b2a8dc2a326d508053608e28ce872ae3bdae27d6a07；optim.pt 13,038,715 / 03af96df420d9c3f8bbe306fe4e3d5bc04ada5be232a7699ea02e4ed18e14d8c；rng.0.pt 20,169 / 84170fe91f16a6c0d974b16bf3e3b28209125d1d84a4edf3bce3f343b3027727；export/lori-sparse90-lycoris.safetensors 44,654,756 / 2318a8557e7e2f4f473c6bee2b9ce99e2f1599b2a730770d97fe9e6ccfa71f49；export/lori-sparse90-comfy.safetensors 49,601,320 / b6219bd39f9b0dced4a6a869ef087fc9374142d38d1f07bd5d6ba0294b121312
- step_00000570/：state.json 182 / e0bb05c6dd4a4836adecfb91b8549b148425333639356fe34f739f91fbc9eec6；trainable.safetensors 44,654,764 / db8c6f7019e55d98ba768e9d67e0a79fc163ec0116ebadca2b9288836b9b679f；optim.pt 13,038,715 / fcdb2562b6c8a1d30cacabe1634585ffa07d3be1770e0e34697d8aebacd800bb；rng.0.pt 20,169 / 56b5886e18ae503bbb497ddd0e675d4ab4aa0836630b10ff24d85b4d3cbc8af5；export/lori-sparse90-lycoris.safetensors 44,654,756 / 01a1d40f2187484c639e4498d3bb367e4e91d2b3fa5c1b84ab97fe3d65b051fc；export/lori-sparse90-comfy.safetensors 49,601,320 / fe5f7cb837728fe1a70edd8d98b6d64af87399ebe394815ede78722629529db7
- step_00000855/：state.json 182 / 355e284248764ae115204903a80bccb7cfa40036ebad44fd516c7c633da3b62c；trainable.safetensors 44,654,764 / 5f53d1d53cd1466ac61827d36dde2b1586f9e2c618bc1aad0e36190b115d51da；optim.pt 13,038,715 / 4b224ac1424fb154dcb10081564cedc45e629f4400ea3971be64915ff51042ed；rng.0.pt 20,169 / 0c44bf2728a2dd973b04ceaf5e3a60eadce9834ae86b6f13c063a21b4eae5684；export/lori-sparse90-lycoris.safetensors 44,654,756 / c1f35ffdca3bd0b5d7f15ffa5d9f7ba927095126d67361cd614eb28287355565；export/lori-sparse90-comfy.safetensors 49,601,320 / 24e240923818c5048e01faec6aea5972aa134fa0bbbf65e461ecd658c6f6d1ad
- step_00000900/：state.json 182 / e2126ee07eab176d962b65b93e18a3c508b623ff293318c7df38372232b64def；trainable.safetensors 44,654,764 / fec9b71926355e1ce2a0c646b3e982b671697ed30457e8b4c798b0a6c5bec01c；optim.pt 13,038,715 / 898aa809a9240af899cd69c35f591a48b4c3be32e82c990d4b02bf330ff8da95；rng.0.pt 20,169 / 783f21d65a5452c1adeae60bd1dccca7204eb73920cc8b4fc741040b99a8ca43；export/lori-sparse90-lycoris.safetensors 44,654,756 / fcfef742c16e19db1bad3735dcc3a9ac10d2aa3183c79ae773d4dfc94500fa54；export/lori-sparse90-comfy.safetensors 49,601,320 / ce9ba0ffc798af023c56f51c5690ca137830337d4faf682f3d462e47e2fb4d59

**移除 60%**（runs/lori-sparse60/checkpoints/）：

- step_00000285/：state.json 181 / 38b355b89faa44f59ef6fd97f5568b845c0c8affc1833817cdc653891b395597；trainable.safetensors 83,052,792 / e80131e33d271877913dc54f7ecc5c19e73e4278ce9ca549624c85c7a9dfed71；optim.pt 51,436,795 / 622d494375a9153ef536a449f124d7043d1678be12f21480d2412663333195db；rng.0.pt 20,105 / 9d20415ffc674b5fa4e7d6df357584a0dfe788f121819e3d7ac9bbf84de9b9b9；export/lori-sparse60-lycoris.safetensors 83,053,008 / 1ad98288bcc6a3d0d3b19f96195f378fc49883b134e8909c6455f5f830b46e98；export/lori-sparse60-comfy.safetensors 49,601,320 / 61a12254517c3ecdda7c03ba7a427b22f56ab72f3b9a3bea793f4174d5a38ca3
- step_00000570/：state.json 182 / e0bb05c6dd4a4836adecfb91b8549b148425333639356fe34f739f91fbc9eec6；trainable.safetensors 83,052,792 / bfc016902e358850181d762d543eeb12881b30548938f9328b4b90656a4b543a；optim.pt 51,436,795 / 0e59d7dbcdc4b32d472a51fbc769d7b885099cee4ab81b5da85f67f62291306e；rng.0.pt 20,105 / c9ede3f88e6d8e169db78c853ca5d7da4938bf7ec318536686d5ad73e5d69466；export/lori-sparse60-lycoris.safetensors 83,053,008 / 627c900cb8e5f544d671e975e5717942ab885ba7214c1646c3e278942e055a5c；export/lori-sparse60-comfy.safetensors 49,601,320 / 2dd4e5f7249007967b721e7f2b76403b902a01e74aa005136c9f1c2a3feee6e0
- step_00000855/：state.json 182 / 355e284248764ae115204903a80bccb7cfa40036ebad44fd516c7c633da3b62c；trainable.safetensors 83,052,792 / c58573d3c7bc5782fd04742cee6f1b53d1ad38ec4be77219f2036bec2578480c；optim.pt 51,436,795 / 9c4134b14d3cc29c7bf0c422316bab1e9f0d5778ccb84e2329c52e0f0dfbcc81；rng.0.pt 20,105 / 604279cb94208c0f1ff3adc82dfe95941c42d0305d75606ca505173a23e458ad；export/lori-sparse60-lycoris.safetensors 83,053,008 / 6a95e8687fe9da7c8b39c42428d553601d7fd54cb6d893e1c5f21c4d3569d4bf；export/lori-sparse60-comfy.safetensors 49,601,320 / c51fe3a2ea95e511ace2b3876320acd7adedfc7eb5c0bb5c0c99b565ab48861e
- step_00000900/：state.json 182 / e2126ee07eab176d962b65b93e18a3c508b623ff293318c7df38372232b64def；trainable.safetensors 83,052,792 / cc19a269e6b580b46b00add91a3bbbadae9fabf7eff0321dc65cbaa85f90f0da；optim.pt 51,436,795 / 1d7f044d3c59b13c37b5bb20a78fc20d1b455db2241ac43d4d1e0561e6cc7d75；rng.0.pt 20,105 / e9fa2dd962c0c0eaf9a6ac4eafc2ca8b3f07edb0dbaa2d3b69841cfbeab6c190；export/lori-sparse60-lycoris.safetensors 83,053,008 / 7c4f7223082d8a33814dab82b02c9c1e5601c2210c1797558608466940729b50；export/lori-sparse60-comfy.safetensors 49,601,320 / 2e7fc6bee8a8281d8c57d76649928c7a2e4b86faed84d95cc52ca48bdde9aba9

**移除 30%**（runs/lori-sparse30/checkpoints/）：

- step_00000285/：state.json 181 / 38b355b89faa44f59ef6fd97f5568b845c0c8affc1833817cdc653891b395597；trainable.safetensors 121,451,196 / 28fe2bb0e8afbaeb7abcefd9efd6c3b8a836dbda34c4f50dc1cc04a9ca3345a1；optim.pt 89,835,963 / 7f7a0639b389a301da41630f87a7b5698907629757e618a0078cf46e5238f4fc；rng.0.pt 20,105 / eef9bbdc7822104c2c4eb5ca1e9177cc285f8ead536bfe8b7c291136e7e0a7d9；export/lori-sparse30-lycoris.safetensors 121,451,580 / 8ac750ba42ddf0b6d11d1347891457a87c105171277bf2dffdf6f915fd30d6de；export/lori-sparse30-comfy.safetensors 49,601,320 / 1bf4eac9c629e2cdb15c6e7faa47ea26cb2cf090010bb7c4d49adb0ef09f22bf
- step_00000570/：state.json 182 / e0bb05c6dd4a4836adecfb91b8549b148425333639356fe34f739f91fbc9eec6；trainable.safetensors 121,451,196 / 511529a7391920e95889af3e4f532bd2fb1b25f6b9b19ccee742ad3bfa0f0565；optim.pt 89,835,963 / 7b401ccdd7d894d4ebdcf40460b3abf994b96bd1cf130fdd7c3bd5fc72e1e728；rng.0.pt 20,105 / 2c4009195a0c38c8ab09e7a401b87f0aecba97e15574e2a475d454b2b1d72e5c；export/lori-sparse30-lycoris.safetensors 121,451,580 / 58a9b60945809708dc055cdfc7e98fdecf48a710625a7b2385527e0e030bb256；export/lori-sparse30-comfy.safetensors 49,601,320 / 3f4ac2ac00833a32fc33fba42a8339ace3f382c9595a00070ee51b7514b94799
- step_00000855/：state.json 182 / 355e284248764ae115204903a80bccb7cfa40036ebad44fd516c7c633da3b62c；trainable.safetensors 121,451,196 / d685fdd6890a0a307b4aba46673621fe8260d4e3f2b34e8f39cb23924b1226ee；optim.pt 89,835,963 / 6b1f0ba82cdc29294645b4046ee8c7e5b8a470f99eb2e39dba9d41a766ed8f34；rng.0.pt 20,105 / cdaa817c2aa86b746a41e32a36abc3c297f0d99c385ab270b0641bee54e53e52；export/lori-sparse30-lycoris.safetensors 121,451,580 / 46aee2e5d72a50acd90fbf21bddbe174625ab05c6fa0b8145de7ff608c4bcf5c；export/lori-sparse30-comfy.safetensors 49,601,320 / 57f32b8cecd6f50311cddf80ce6f7c804cfa62c17a07f25f64f928ec23f24b9e
- step_00000900/：state.json 182 / e2126ee07eab176d962b65b93e18a3c508b623ff293318c7df38372232b64def；trainable.safetensors 121,451,196 / 7e7dd837c75a15cab5a2dc9cbd9f728709cccfda43e5996b98ed2b7d9d415957；optim.pt 89,835,963 / 1324f1a87026a2104b1790df0fd81cfab9d44e4f165d818e4649ebf586506562；rng.0.pt 20,105 / f1246d1883373dd4ed571f64009529eaff950a71a6c0ac0560f55a2774b41530；export/lori-sparse30-lycoris.safetensors 121,451,580 / 78da1a49eb48c556c4352de28ba5e82c009ed02f9bd502badfd3d7a80f31edcd；export/lori-sparse30-comfy.safetensors 49,601,320 / be86e1f2f4c9736bfee283a5ffcf26a20d53701ac28f4893f147151cf6d65562

运行级 export/ 另有终态两份（训练器既有路径，与 step 900 检查点内导出字节不同）：

| run | 文件 | 字节 | sha256 |
| --- | --- | ---: | --- |
| sparse90 | export/lori-sparse90-lycoris.safetensors | 44,654,756 | aef913a536d9979994fe7156d9750ae985b31a854cbc0073aac1c9ed6c05fc68 |
| sparse90 | export/lori-sparse90-comfy.safetensors | 49,601,320 | 9457f0de0de8bfbd5fe3b0b37594e3ab414087c4f6758a107c018ad725cda68e |
| sparse60 | export/lori-sparse60-lycoris.safetensors | 83,053,008 | 见 sparse60-final-audit.json 的 run_level_export |
| sparse60 | export/lori-sparse60-comfy.safetensors | 49,601,320 | 见 sparse60-final-audit.json 的 run_level_export |
| sparse30 | export/lori-sparse30-lycoris.safetensors | 121,451,580 | fa703255184b3d3fb8327e461d4e11ba4de5cb5e214f765b790f97a1fe657f57 |
| sparse30 | export/lori-sparse30-comfy.safetensors | 49,601,320 | 89290af3a2582ab26c63581ec6d25351202e1c65eb9e5ed3e736a9e7c300ef80 |

## 8. A 不变与 B 更新证据

三档的 trainable.safetensors 都是 1008 张量（.A 252 + .alpha 252 + .b_values 252 + .active_indices 252），step 285 与 step 900 两个检查点、以及两份 lycoris 导出都做了逐张量重新签名：

| 证据 | 移除 90% | 移除 60% | 移除 30% |
| --- | --- | --- | --- |
| A 组合 digest（252 张量） | 7f8b433fe34e4caa789e9923e44227acdef677e04edc1852a71d6242b884db19 | 同 | 同 |
| alpha 组合 digest（252 张量） | 930ce824c0831386ee6bad65542d768c8b56d2c5a3264511be2d7ae02207c991 | 同 | 同 |
| 与初始哈希一致 | 252 / 252 与 252 / 252 | 同 | 同 |
| 紧凑 B 非零值（step 285） | 1,066,599（全部非零） | 4,266,394（全部非零） | 7,466,189（全部非零） |
| 紧凑 B 非零值（step 900） | 1,066,599（全部非零） | 4,266,394（全部非零） | 7,466,189（全部非零） |
| B 全局 L2：285 到 900 | 3.32411 到 5.22539 | 4.41406 到 7.17537 | 4.99930 到 8.58530 |
| 单层非零 min / max | 1,197 / 21,142 | 7,790 / 62,463 | 15,515 / 97,603 |
| step 285 vs 900 不同的层 | 252 / 252 | 252 / 252 | 252 / 252 |
| 900/285 逐层 L2 比值 中位（min–max） | 1.6215（1.2864–1.8870） | 1.6525（1.3919–1.8753） | 1.7281（1.5273–1.8492） |
| 索引与掩码逐层相同 | 是（trainable 与 lycoris 均逐层 digest 相等） | 是 | 是 |

也就是说：**A 与 alpha 在三档、两个步点、两种导出格式里完全不动，唯一变化的是紧凑 B 的数值**；B 的条数恒等于掩码保留数，且每一层都被更新过（没有冻结层、没有全零层）。

## 9. 优化器状态的紧凑性

optim.pt 的 schedule.last_epoch 为 900，skipped 与 nonfinite 均为 0，optimizers 只有 1 组、恰 252 个 entry（键 0..251 连续）。每个 entry 是四项：master、exp_avg、exp_avg_sq、更新计数；三列都是一维 fp32、全部有限。

| 项目 | 移除 90% | 移除 60% | 移除 30% |
| --- | ---: | ---: | ---: |
| 每列元素数（master / exp_avg / exp_avg_sq） | 1,066,599 | 4,266,394 | 7,466,189 |
| 矩元素合计 = 2 × 保留数 | 2,133,198 | 8,532,788 | 14,932,378 |
| 更新计数 | 900 | 900 | 900 |
| optim.pt 字节 / sha256 | 13,038,715 / 898aa809… | 51,436,795 / 1d7f044d… | 89,835,963 / 1324f1a8… |

三档的矩元素合计都**恰好等于 2 × 该档保留数**（断言为相等，而不是上界），说明稀疏是物理紧凑存储，而不是「稠密矩阵加掩码」——这与 task6 的 90% 档结论一致，并首次给出 60% / 30% 的对照：保留 7 倍的参数，优化器状态也正好是 6.90 倍（89,835,963 / 13,038,715）。

## 10. 导出可读性与元数据

kohakumoss.adapters.train.lori.read_lori_state 能直接读取 step 285 与 step 900 的 export/*-lycoris.safetensors（1008 张量）；读取过程本身会重算每层的 nnz、A/alpha/索引 digest 与形状并与元数据比对，全部通过。A / alpha 的组合 digest 与初始哈希一致；252 个 B 非零且有限；索引与掩码逐层 torch.equal。

元数据（kohakumoss_lori）与实际一致：

| run | phase | sparsity | total | nnz | actual_density | layers | calibration_digest |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| sparse90（trainable 与 lycoris） | sparse | 0.9 | 10,665,984 | 1,066,599 | 0.1000000562536 | 252 | 7e883962…6a8ac8 |
| sparse60 | sparse | 0.6 | 10,665,984 | 4,266,394 | 0.4000000375024 | 252 | 同 |
| sparse30 | sparse | 0.3 | 10,665,984 | 7,466,189 | 0.7000000187512 | 252 | 同 |

kohakumoss_format 为 kohakumoss_lori，kohakumoss_step 为 285 / 900。ComfyUI 导出能被 safetensors.safe_open 打开，504 张量（q/k/v 已融合，lora_up.weight 是每步重建的稠密 B 视图），metadata 记 kohakumoss_format=comfy、kohakumoss_algo=lori。

另注：稀疏 run 的 comfy 导出固定 49,601,320 B（稠密 B 视图，三档同尺寸），lycoris 导出随保留数增长（44.65 / 83.05 / 121.45 MB），trainable.safetensors 同理（44.65 / 83.05 / 121.45 MB）。

## 11. 冻结复核与只读输入

每次 stage 的 worker 在训练前与训练后各跑一次同一套冻结校验，两次结果逐字段相同（cache 文件 sha256、三个权重的 size/mtime、570 个源文件快照、训练器 HEAD 与干净状态）：

| 项目 | 值 |
| --- | --- |
| manifest | 285 行，4030fdf3094b91cff164257f70fd5290674dbf5b24f4a1e4a75d19036e869ead |
| catalog.db | 3,686,400 B，fd4bd5debed5dc470edf0e68c676e9f76d3c31c14df8b2bc68882f22adeff0aa |
| text.db | 923,926,528 B，180f536bc53e8bc481709057357ec6c048909716abffa1d36231908e9ed2890c |
| latents.db | 18,841,600 B，4cb07744c0d662d7a177e70dab5d728693fa40cc0c4195b4c62efec2ed409fdc |
| text.settings.json / latents.settings.json | 64e1a048…d7b0847 / 3db09a06…e32e906 |
| 三个权重 | size 5,413,698,958 / 9,748,121,532 / 249,521,340 B，mtime_ns 与 task10 冻结值一致 |
| 570 个源文件快照 | 摘要 e7022c689cfd18240530b83bed5447eb7759a46e37a00a287ee65c491f40d4cd |
| 训练器 | HEAD 957ccc3f89c1506db8e41ce4d260f529446e142b，git status 干净 |
| 本轮脚本 | 6 个会话脚本 + moss.py + lori_mask.py 的 sha256 记入 freeze.json |

缓存没有重建、也没有被改写（训练前后 sha256 相同），task10 目录只被读取。

## 12. 校准 B 与 task10 稠密 285 的交叉核对

| 比较项 | 结果 |
| --- | --- |
| A / alpha 逐张量 | **504 / 504 完全相同**（task10 的 step_00000285、task10 prepare 的初始哈希、本轮初始哈希三者一致） |
| 稠密 B 逐张量 | **0 层相同 / 252 层不同** |
| 最大绝对差 | 8.9935e-3 |
| 本轮 B 全局 L2 | 5.25066（task10 为 5.49669） |
| 交叉核对对象 | training/lori-dense/checkpoints/step_00000285/trainable.safetensors（74,397,040 B，sha256 574be276…6d0984） |

按任务约定，这**不是失败**，记录为「同配方、同 A、同数据顺序，但不是逐张量复现」。可支持的判断是：A 的初始化与 task10 完全一致（A 由 seeded_rng(SEED) 在构建时独立播种，与之前的随机消耗无关，因此跨进程可复现）；B 的数值则不能跨进程逐位复现——三档配置、数据顺序与 seed 都相同，首步 loss 与 task10 逐位相同（0.7369208931922913），此后由于编译缓存/内核选择不同带来的舍入差异在 285 步里累积到 1e-2 量级。本轮三档之间的可比性不受影响（三档用的是本轮同一份校准的同一份 B 与同一套掩码），但与 task10 稠密 step 285 只可比统计量、不可比逐值。

## 13. 偏差与异常

1. **sparse90 第一次执行 exit 1（训练已完成）**：900 步训练正常跑完（native_completed=true，训练调用 1,575.37 s），随后本会话审计脚本 audit_trainable 在断言训练态元数据键集合时把四个键按非字母序书写，sorted() 结果不匹配而抛错——即**审计脚本的期望常量写错，不是训练或产物问题**。处理方式：(a) 把断言改成同样的四个键的正确顺序（键集合未变，未放宽任何检查）；(b) 新增 --recheck 模式，只对已存在产物重跑完整审计、不重训、不改配方；(c) 由于 run.py 被改动而 freeze 正确报告 frozen files changed，另写 freeze-amendment.json 记录 run.py 的原/新 sha256（ba73721e979635b48fc420288b254929267604f91b5029241a962a59d88f2d34 到 c9478a0d5a8540a4333922f0b824b0054a6852d091a33288a388b97865165b8d）、原因，以及四个配方文件与两个训练器脚本哈希未变的事实；freeze.json 的 sha256 表只更新 run.py 一项并留下 amended_files 记录。原证据完整保留：stages/sparse90/result.attempt1.json（exit 1 + traceback）、exit.attempt1.json、run.py.attempt1.sha256、recheck.json。复审在 2.87 s 内通过，stage 的 result.json / exit.json 记录 exit 0 并注明 recheck 与 attempt1 指针。sparse60 / sparse30 用同一份修正后的脚本一次通过，未再出现同类问题。
2. **校准 B 与 task10 稠密 285 不同**（第 12 节）：记录差异数量与幅度，不作为失败、不更换数据、不强行恢复旧检查点。
3. **三档的 loss / 步时几乎相同**：预期而非异常——损失对同样的数据与同样的 A 计算，差别只来自 B 的稀疏结构与参数量；900 步 = 3.158 轮、样本几乎不重复，所以块均值平坦。不要把「三档 loss 接近」读成「三档效果相同」。
4. **显存差远小于参数差**：optimizer 在主机侧、总显存由 INT8 主干主导，见 6.2 节的读法。
5. 无 traceback 残留、无非有限步、无跳步、无「改配方重跑」；GPU 作业全程串行。

## 14. 限制

1. **仍然没有独立留出集**：285 张全部入训（900 步 = 3.158 轮），本轮不能判断泛化，也不能把 loss 变化当成画质或风格改善。
2. **只比较稀疏度，不与稠密做质量排名**：本轮没有出图、没有网格、没有盲评；三档之间以及三档与 task10 稠密之间的比较仅限于参数规模、优化器规模、步时、显存、loss 这些可测量，不构成画质结论，也不能据此说某个稀疏率更好。
3. **稀疏不等于省显存/省时间**：前向仍每步重建稠密 B，本轮步时中位 1.7166 / 1.7220 / 1.7201 s 相对稠密 900 步的 1.602 s 没有变快（迭代开销略高），总显存差别只有 88.8 MiB。真正随保留数线性变化的是适配器权重文件与主机侧优化器状态。
4. **A 的跨进程可复现、B 的跨进程不可复现**（第 12 节），凡需要逐值对照的实验都必须落在同一进程/同一份校准产物上；本轮三档满足这一点。
5. 单机单卡、单一 seed、单一配方；结论不外推到其它分辨率、其它 rank/alpha、其它 memory profile 或其它数据集。

## 15. 工件清单

| 工件 | 路径（相对 experiment/task11/session-20261119-a/） |
| --- | --- |
| launcher + worker | run.py（含 --stage / --worker / --check / --recheck） |
| 配方 | common.py、calibration.py、sparse-90.py、sparse-60.py、sparse-30.py |
| 冻结与解析 | freeze.json、freeze-amendment.json、resolved.json、resolved-calibration.json、resolved-sparse-90/60/30.json、resolved-cache_text.json、resolved-cache_latents.json |
| 校准 | calibration/lori-calibration/（metrics.jsonl、log.txt、config.json、checkpoints/step_00000285/{state.json,trainable.safetensors,optim.pt,rng.0.pt,export/}） |
| 掩码 | masks/sparsity-90.safetensors、masks/sparsity-60.safetensors、masks/sparsity-30.safetensors |
| 三个 run | runs/lori-sparse90/、runs/lori-sparse60/、runs/lori-sparse30/（各含 metrics.jsonl、log.txt、config.json、checkpoints/step_285,570,855,900/、export/、ranks/、startup/、final-audit.json） |
| 终态审计 | sparse90-final-audit.json、sparse60-final-audit.json、sparse30-final-audit.json |
| stage 记录 | stages/calibration/{launch,start,result,exit,initial,audit}.json、stages/masks/{launch,start,result,exit,masks}.json、stages/sparse90|sparse60|sparse30/{launch,start,result,exit,initial,audit}.json（sparse90 另有 result.attempt1.json、exit.attempt1.json、run.py.attempt1.sha256、recheck.json） |

## 16. 下一步（不在本任务范围）

1. 若要比较稀疏与稠密的同预算质量，需要按同一 285 图配方出图：Base、稠密 step 285 / 900、三档稀疏 step 285 / 900，用同一提示组、同一种子、隐藏列名盲评；本轮的 12 个检查点已就位，可直接读取。
2. 掩码阈值来自 285 步稠密校准；若要回答「校准步数/校准数据是否影响掩码质量」，需要另开校准步数消融。
3. 无论如何都仍需要训练目录之外的独立留出集，才能谈泛化与「学会画风 vs 记住样本」。
