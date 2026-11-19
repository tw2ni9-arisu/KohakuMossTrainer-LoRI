# Task 13 原生分辨率 90% 稀疏 LoRI：cache、探针、四段 900 步与终态审计

日期 2026-11-19（本地）。会话 experiment/task13/session-20261119-a/。在 RTX 5070 Ti Laptop 12GB 上，用 Task 13 的**工作树改动**（HEAD 仍为 957ccc3f89c1506db8e41ce4d260f529446e142b，工作树含 8 个已改文件 + 1 个未跟踪测试，全程未 stash / 未 commit / 未改任何源码）在**原生分辨率**训练 90% 稀疏 LoRI 900 次有效更新，分四段续训，保留 250 / 550 / 850 / 900 四个检查点，并完成 cache、探针、续训、优化器、导出元数据与工作树的终态审计。

**结论边界**：训练完整性、A 不变、紧凑 B 更新、新元数据字段全部达标；掩码来自 **256² 校准**，本轮只改分辨率与训练，不与 256² 的 Task 11 run 做同配方对照，也没有留出集与出图，**不构成任何画质结论**。

| 指标 | 实测 |
| --- | ---: |
| 分辨率 | 原生（目标面积上限 1,048,576 px，零缩放） |
| 图像 tokens | 1,135,799（285 图，2728–4089/图，中位 4000） |
| 有效更新 | 900（250 + 300 + 300 + 50，四段串行续训） |
| cache 计数 | catalog 285 / text 286 / latents 285 |
| 探针 2 步 | 10.98 s / 8.33 s，无 OOM |
| 900 步训练调用墙钟 | 5,613.6 s（1.56 h），平均 6.10 s/步，中位 6.05 s |
| CUDA 峰值 allocated / reserved | 2,605,227,520 / 2,925,527,040 B（2.4263 / 2.7246 GiB），四段完全相同 |
| host peak_wset | 14,229,921,792 – 14,295,261,184 B（13.25–13.31 GiB） |
| 保留 B 值 | 1,066,599（掩码 kept_total，四检查点一致） |
| optim.pt | 252 entry，矩 2,133,198，计数 900，13,038,715 B |
| A / alpha | 与 canonical 摘要完全一致（7f8b433f… / 930ce824…），a_sha256 976522f0… |
| B 更新 | 250 与 900 逐层 252/252 不同，L2 2.10500 到 4.39752 |

[run.py](../../../experiment/task13/session-20261119-a/run.py) · [freeze](../../../experiment/task13/session-20261119-a/freeze.json) · [resolved](../../../experiment/task13/session-20261119-a/resolved.json) · [cache 审计](../../../experiment/task13/session-20261119-a/stages/cache/result.json) · [探针](../../../experiment/task13/session-20261119-a/stages/probe/result.json) · [终态审计](../../../experiment/task13/session-20261119-a/final-audit.json)

---

## 1. 分辨率决策与「零缩放」证明

配方只改两处：TARGET_AREA = 1024 × 1024 = 1,048,576，UPSCALE 保持 "never"（RESOLUTION_MODE 仍是 resize，SIZE_STEP 仍是 16）。规划器 fit() 的算法是：先按 ASPECT_RANGE 取最大居中子尺寸 (cw, ch)，再取 scale = min(sqrt(area / (cw*ch)), 1.0)，乘以原尺寸后向下取 16 的倍数。

证明链（逐图断言，285/285 通过）：

1. 目录里最大的图是 **1,046,784 px**（最小 698,368 px），全部 ≤ 1,048,576，因此 sqrt(area/(cw*ch)) ≥ 1；再配合 upscale="never" 的 min(scale, 1.0)，每张图的 scale **恰为 1.0**。
2. 285 张图里 **没有一张**触发 ASPECT_RANGE 裁剪（clamped = 0）：宽高比全部落在 [0.25, 4.0]。
3. 目标尺寸与原始尺寸**逐张完全相同**：285 / 285 identical；高度、宽度的最大下降都是 **0 px**（不是「不超过 16」，而是 0），面积比 min = median = max = **1.0**。
4. 每张图都重新调用同一规划器算一遍并与缓存里记录的 target_hw 比对，全部相等。

原因很简单：这批图的宽高本来就都是 16 的倍数，所以向下对齐不掉任何像素。缓存里的 target_hw 分布因此就是原始尺寸分布本身（55 个不同尺寸，常见：704x1440 × 28、720x1408 × 15、720x1440 × 15、688x1472 × 12、736x1376 × 14、656x1552 × 3、1360x768 × 3 等；极值 624x1616 与 1584x640）。**本轮没有任何缩放、裁剪或重采样发生。**

## 2. cache（本会话新建，未复用 task10/11 的缓存）

用 scripts/data/catalog_build.py、cache_text.py、cache_latents.py 配本会话的 data.py 依次在独立子进程里构建，每次 pass 各自开 CUDA window 并记录峰值。

| pass | 墙钟 | 原生用时 | CUDA allocated / reserved | host peak_wset |
| --- | ---: | ---: | --- | ---: |
| catalog_build | 7.2 s | 4.3 s | 0 / 0（不用 GPU） | 594,427,904 B |
| cache_text | 283.1 s | 278.9 s | 1,308,635,136 / 1,344,274,432 B | 18,333,892,608 B |
| cache_latents | 82.1 s | 78.4 s | 1,777,176,576 / 3,900,702,720 B | 2,157,637,632 B |

| 缓存 | 计数 | 内容 |
| --- | ---: | --- |
| catalog.db | 285 行（live 285） | locator 与 caption 逐条等于 manifest；图像像素 698,368–1,046,784 |
| text.db | 286 条 | 285 条 caption + 空提示；长度 10–552，中位 392，合计 **112,500** text tokens；4096 通道 |
| latents.db | 285 条 | 128 通道，bf16；每图 **2728–4089 tokens**（中位 4000、均值 3985.26），合计 **1,135,799** image tokens |

对比 256²（Task 11 复用的 task10 缓存：66,900 image tokens、235 tokens/图）：本轮每图 tokens 是 **17.0 倍**，总量 1,135,799 是 256² 的 17.0 倍，符合 1024²/256² = 16 倍再加宽高比差异。latents.db 292,683,776 B（约 1.03 MB/图，4000 × 128 × 2 B）。

缓存文件（sha256 全部钉入 freeze.json，训练前后复核不变）：

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| catalog.db | 3,653,632 | 见 freeze.json cache_files |
| latents.db | 292,683,776 | 见 freeze.json cache_files |
| text.db | 923,926,528 | 见 freeze.json cache_files |
| latents.settings.json | 408 | 见 freeze.json cache_files |
| text.settings.json | 236 | 见 freeze.json cache_files |

每题尺寸与 settings：data.py 解析出的 DataConfig 与训练配置（probe.py）的 latents / text settings 逐键相等（启动前置断言），因此离线缓存必命中；data.py 的 TOKEN_BUDGET 为 "estimate"，训练配置为 "exact"，token 预算不参与缓存键。

## 3. 2 步探针（不改内存档位）

| 项目 | 实测 |
| --- | ---: |
| step 1 / step 2 步时 | 10.981 s / 8.333 s |
| 第 2 步 loss / grad_norm | 0.46324 / 0.013613 |
| 训练调用墙钟 | 31.5 s（含模型加载与 252 个适配器构建） |
| CUDA 峰值 allocated / reserved | 2,605,227,520 / 2,925,527,040 B（2.4263 / 2.7246 GiB） |
| 逐 step vram_peak_alloc / reserved | 2.4263 / 2.5918 GiB |
| host peak_wset | 14,229,921,792 B（13.2524 GiB） |
| OOM | **否**（未改内存档位：仍是 ckpt-ring4-actoff-optstream-gradoff） |
| ETA（900 × 8.333 s + 4 × 45 s） | 7,679.9 s ≈ 2.13 h |

探针已经带出新元数据：step_00000002 的 state.json 有完整 lori 块，lycoris / comfy 导出各自带 facts（见第 7 节）。注意探针的 8.33 s/步仍含首段预热，稳态实际更快（下文 6.05 s/步），所以 ETA 高估了约 37%（**实际 900 步训练调用 5,613.6 s = 1.56 h**）。

## 4. 四段续训

RUN_NAME = lori-sparse90-native，OUTPUT_DIR = <session>/runs，SAVE_EVERY = 0、FINAL_CHECKPOINT = True、KEEP_LAST = 4，每段结束时写该段的最终检查点；段 2–4 的 RESUME 指向前一段的检查点目录。

| 段 | STEPS | RESUME | 退出码 | launcher 墙钟 | 训练调用 | loss 均值（段内） | loss 首 / 末 | 步时中位 | CUDA 峰值 A/R（B） | host peak（B） |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |
| seg250 | 250 | ""（全新） | 0 | 1,483.9 s | 1,463.0 s | 0.38143 | 0.30510 / 0.43097 | 5.7346 s | 2,605,227,520 / 2,925,527,040 | 14,249,959,424 |
| seg550 | 550 | step_00000250 | 0 | 1,852.4 s | 1,831.9 s | 0.37435 | 0.33547 / 0.65920 | 6.0953 s | 同上（逐字节相同） | 14,269,931,520 |
| seg850 | 850 | step_00000550 | 0 | 1,953.1 s | 1,931.8 s | 0.37824 | 0.37506 / 0.28371 | 6.3954 s | 同上 | 14,295,261,184 |
| seg900 | 900 | step_00000850 | 0 | 381.5 s | 355.4 s | 0.33041 | 0.39540 / 0.56316 | 6.5162 s | 同上 | 14,294,302,720 |

全量 900 步（metrics.jsonl 恰 900 行、step 1..900 连续、loss / grad_norm / step_time 全有限、skipped_steps / nonfinite_steps / nonfinite_params 全 0、lr 恒 1e-4）：loss 均值 0.37517，最小 0.13759，最大 0.79118，首 0.30510，末 0.56316；grad_norm 均值 0.02136、最大 0.25619、末 0.04276；步时中位 6.0546 s、均值 6.1049 s、最小 0.0161 s、最大 11.3214 s（step 1）。

续训证据（每段都断言，不只依赖配置）：

1. 训练器自己的日志行 "resuming from <上一段检查点>" 出现在该段 stdout.log（段 2–4 均为 true）。
2. 适配器构建时 initial_state 非 None、键数 **1008**（252 × A/alpha/b_values/active_indices），且索引与掩码逐层一致。
3. 构建时紧凑 B 仍全零（trainer 的设计：先按 A/alpha/索引建模块，再由 Parts.restore 载入检查点的 B 与优化器），因此断言改成「构建期 B 为零」，而恢复的证据由下一条与终态 B 提供。
4. 段末 optim.pt 的更新计数恰等于该段 STEPS（550 / 850 / 900），说明优化器状态是**累积**而不是重置。
5. 段末 state.json 的 loader.start 依次为 250 / 550 / 850 / 900，tokens 依次 995,490 / 2,191,376 / 3,387,387 / 3,587,520（每样本 3,982–3,986 tokens）。

一处可解释的计时异常：每个续训段的**第一步**（step 251 / 551 / 851）step_time 分别只有 0.0463 / 0.0215 / 0.0161 s，而其 loss 与 grad_norm 正常（0.335 / 0.375 / 0.395），后续各步回到 6 s 量级。这是续训后首步的计时归属落在恢复阶段所致（不是跳步：900 行齐全、loss 无重复、grad 正常），统计时以中位数为主。

## 5. 检查点清单（路径 + sha256）

四个检查点都在 runs/lori-sparse90-native/checkpoints/，各含 state.json、trainable.safetensors、optim.pt、rng.0.pt 与 export/ 下两份导出；state.json 的 step = samples = 目录步数，loader = {"seed":42,"start":<同值>}，world = 1。所有文件字节与 sha256：

**step_00000250/**（tokens 995,490，text_tokens 98,518，joint_sq 4,793,810,608）

- state.json 422 / 7ecd8e85c530cce5faa3c9f276d4efda70bd082c3e7f2ffca3a8cfe839d0eb42
- trainable.safetensors 44,655,108 / 78068e73369f2fc1f0b0f4ee60954ffe402820e03f970698fdd1acda301afaff
- optim.pt 13,038,715 / 05b7e6e37f8f4fd4daf830ca66065cf8ae7159941476798f41496873ebf1f322
- rng.0.pt 20,105 / 865811bfbd3cf10e83da4bc6789c89b9c22c3c2f8a4cadb9cf81409106da9f8d
- export/lori-sparse90-native-lycoris.safetensors 44,655,100 / 900fcad9a14a9ff52165937ba56e58e6299cccbd95f0428765803ac18e7d924f
- export/lori-sparse90-native-comfy.safetensors 49,601,744 / f67285d3e487fce6b4d3fa93b52bf29ae87e4a44a7b7b6d62dfc99a209fa7535

**step_00000550/**（tokens 2,191,376，text_tokens 217,076，joint_sq 10,559,723,694）

- state.json 425 / ca75c54f778b23a1f9c360407288170b60f5e17b74452728020c364d03a53a14
- trainable.safetensors 44,655,108 / 41e410100e486f966aee6496ebcb9271b3777218537b73e3fc211d0ef15b29f7
- optim.pt 13,038,715 / ed8d291badbce7bdd05516f2d632d2ec336c730041637af335f8821b29fb831f
- rng.0.pt 20,105 / 352a83523c448f09b49e518da729cc3983b2f3bcddb80d7d6cf1f54af5c6d45c
- export/lori-sparse90-native-lycoris.safetensors 44,655,100 / ae04510312b23be33731490a4f61f5a1c02596423cd03ac587586c02a9cffd17
- export/lori-sparse90-native-comfy.safetensors 49,601,744 / 961d02d82715ef99cfd7a279dbf5738375be554ad0fdf3c8c8e09f3766871fc3

**step_00000850/**（tokens 3,387,387，text_tokens 335,696，joint_sq 16,327,235,305）

- state.json 425 / 8223752d90e627ec44298f4cd7a768a3cdc18815bbabd56fa79a995fa5ba4ec4
- trainable.safetensors 44,655,108 / dbbd791573f0d1c3746510ae7bd652af8ff76295a04ba47bb982e7129a7286a1
- optim.pt 13,038,715 / 553b23856482b49637bc9c022abc6a021221ba64e22ab168776fa4ea731f3739
- rng.0.pt 20,105 / ac6d8f22b781f72b4e594e84a72992b22d2fbcfea1a48cbbd26f646cd72bf0ae
- export/lori-sparse90-native-lycoris.safetensors 44,655,100 / 7a4657284c8e7dc3fe69f0f9caecc1ef43ee62a73a6beb0df156c1ef56de80ea
- export/lori-sparse90-native-comfy.safetensors 49,601,744 / f509acb0c5397b820004a4602814a33faf5846392593d8c134f761dd514d6a44

**step_00000900/**（tokens 3,587,520，text_tokens 355,358，joint_sq 17,293,611,542）

- state.json 425 / 2a78acb6aec05bf1009aa93c781563bf5117b40617555df6b175e3ad0db82563
- trainable.safetensors 44,655,108 / d0b0079d27ec898f1fa2629f5d7f1542ee5d5feb4f58466e158b08a47fff7382
- optim.pt 13,038,715 / 7485c93c5ef5b09ce2f4c4f056eb6bac1ea90e39f65d3ec090f41fad08ec651e
- rng.0.pt 20,105 / 395320fa3f6369e02b889d697fd9ab02c0c7517b520984cbe60886c2faab9451
- export/lori-sparse90-native-lycoris.safetensors 44,655,100 / 30d87ca202c94b32bc29c13d7ea1368e7bcd303d48002587ea02af0f7bf6beed
- export/lori-sparse90-native-comfy.safetensors 49,601,744 / 307f43baa3a2a8773c044967a37851836437aa4c552a7405069a82ae59e3ff66

注意：由于每段都是同一个 run 目录、SAVE_EVERY = 0，第 4 段结束时环形保留的 4 个目录恰是 250 / 550 / 850 / 900，没有中间检查点被删（也从未生成过）。

## 6. 优化器规模与 A 不变证据

第 900 步 optim.pt：schedule.last_epoch = 900，skipped = 0、nonfinite = 0，1 组优化器、252 个 entry（键 0..251 连续），每个 entry 为 (master, exp_avg, exp_avg_sq, 更新计数)。

| 列 | 张量数 | 元素数 | dtype | 有限 |
| --- | ---: | ---: | --- | --- |
| master | 252 | 1,066,599 | fp32 | 是 |
| exp_avg | 252 | 1,066,599 | fp32 | 是 |
| exp_avg_sq | 252 | 1,066,599 | fp32 | 是 |
| AdamW 矩合计 | 504 | **2,133,198** | fp32 | 是 |

更新计数全为 900，optim.pt 13,038,715 B（与 256² 的 Task 11 稀疏 run 同尺寸——状态规模由 kept_total 决定，与分辨率无关）。

A / alpha 不变（四个检查点逐一验证，不只是第 900 步）：

| 检查点 | A 组合 digest（252 张量） | alpha 组合 digest（252 张量） | a_sha256 | 与初始一致 |
| --- | --- | --- | --- | --- |
| step_00000250 | 7f8b433fe34e4caa789e9923e44227acdef677e04edc1852a71d6242b884db19 | 930ce824c0831386ee6bad65542d768c8b56d2c5a3264511be2d7ae02207c991 | 976522f088b6a717f10f344f2a5fe843742ccb180b4b5804600f1631eebd73fb | 是 |
| step_00000550 | 同 | 同 | 同 | 是 |
| step_00000850 | 同 | 同 | 同 | 是 |
| step_00000900 | 同 | 同 | 同 | 是 |

紧凑 B：四检查点均为 252 层、1,066,599 个值、全部非零且有限（单层非零 min / max = 1197 / 21142，恰等于掩码逐层保留数），索引与掩码逐层 torch.equal。

| 检查点 | B 全局 L2 | 与零初始相比更新的层 |
| --- | ---: | ---: |
| step_00000250 | 2.10500 | 252 / 252 |
| step_00000550 | 3.32641 | 252 / 252 |
| step_00000850 | 4.30828 | 252 / 252 |
| step_00000900 | 4.39752 | 252 / 252 |

step 250 与 step 900 逐层比较：**252 / 252 全部不同**；900/250 的逐层 L2 比值中位 1.9317（min 1.5004、max 2.4400）。B 的 L2 从 250 到 850 一路上升、850 到 900 只涨 2.1%，与最后一段只有 50 步一致。

## 7. 新元数据（state.json lori 块 + 文件 facts）

**state.json 的 lori 块**（四个检查点相同，step 不同）：

    "lori": {
      "phase": "sparse",
      "mechanism": "sparse_mask",
      "trainable": "b_values",
      "sparsity": 0.9,
      "kept_total": 1066599,
      "a_sha256": "976522f088b6a717f10f344f2a5fe843742ccb180b4b5804600f1631eebd73fb"
    }

审计对每个检查点逐字段断言：phase = sparse、mechanism = sparse_mask、trainable = b_values、sparsity = 0.9、kept_total = 1,066,599、a_sha256 = 掩码的 canonical 值（掩码 23df6144…、其 a_sha256 976522f0…、calibration_digest 7e883962…）。

**trainable.safetensors 的 safetensors metadata**：除 kohakumoss_adapter_state / kohakumoss_spec / kohakumoss_fold 与 kohakumoss_lori（版本化稀疏元数据）外，新增 8 个 facts：

    kohakumoss_lori_phase=sparse, kohakumoss_lori_mechanism=sparse_mask,
    kohakumoss_lori_trainable=b_values, kohakumoss_lori_a=frozen_buffer,
    kohakumoss_lori_sparsity=0.9, kohakumoss_lori_a_sha256=976522f0…,
    kohakumoss_lori_kept_total=1066599, kohakumoss_lori_rank=8

**lycoris 导出**（每检查点 1008 张量，可直接被 read_lori_state 读取并通过其版本/形状/摘要校验）：facts 与上相同（8 项），另有 kohakumoss_format=kohakumoss_lori。注意 lycoris 路径不写 kohakumoss_container（只有 comfy 路径写），因此审计对该字段只要求 comfy。

**comfy 导出**（每检查点 504 张量，q/k/v 已融合，lora_up.weight 是稠密 B 视图）：facts + kohakumoss_container=lora + **kohakumoss_lori_comfy=inference_only_dense_b**，格式标签为 comfy。这正是本任务要验证的「稀疏掩码被展开为稠密 LoRA、只能用于推理」的显式标记。

四个检查点的这两份导出，facts 的 8 个字段值完全一致（a_sha256 都是掩码的 canonical 值，kept_total 都是 1,066,599），只有导出文件本身随步数变化（sha256 见第 5 节）。

## 8. 冻结与只读输入复核

每个 stage 在训练前后各跑一次同一套校验，两次结果逐字段相同：

| 项目 | 值 |
| --- | --- |
| manifest | 285 行，4030fdf3094b91cff164257f70fd5290674dbf5b24f4a1e4a75d19036e869ead |
| 掩码 | 44,654,540 B，23df6144faa11c7a561da9d3ad8170cd6c70dfc0d4358702d1d02f3cbf2f5e40（task11 的 256² 校准产物，只读） |
| 掩码 kept / a_sha256 | 1,066,599 / 976522f088b6a717f10f344f2a5fe843742ccb180b4b5804600f1631eebd73fb |
| 三个权重 | size 5,413,698,958 / 9,748,121,532 / 249,521,340 B，mtime_ns 与 task10 冻结值一致 |
| 570 个源文件快照 | 目录 mtime 与逐文件 (size, mtime_ns) 未变 |
| 本轮 cache | 5 个文件的 sha256 钉入 freeze.json，训练前后相同（cache 未被改写） |
| 训练器 HEAD | 957ccc3f89c1506db8e41ce4d260f529446e142b |
| 工作树 | status sha256 e044945b…、diff sha256 556eb6f8…（8 个已改文件）、未跟踪 tests/adapters/test_lori_export_facts.py 的 sha256，训练前后完全相同；**未 stash、未 commit、未改源码** |
| 本轮脚本 | 8 个会话脚本 + 5 个训练器脚本的 sha256 钉入 freeze.json（run.py 有两次记录在案的 amendment，见第 10 节） |

## 9. 耗时汇总

| 阶段 | 墙钟 |
| --- | ---: |
| cache（首次失败 391.5 s + 复审 18.2 s） | 409.8 s |
| 探针 2 步 | 50.5 s |
| seg250（250 步） | 1,483.9 s |
| seg550（300 步，含一次未训练的重启尝试 29.8 s） | 1,852.4 s |
| seg850（300 步） | 1,953.1 s |
| seg900（50 步） | 381.5 s |
| 终态审计 | 15.4 s |
| **合计** | **6,146.6 s ≈ 102.4 min**（另加首次 seg550 尝试 29.8 s ≈ 103 min） |

900 步的训练调用合计 5,613.6 s = 93.6 min；平均 6.10 s/步、中位 6.05 s/步，是 256² 稀疏 run（1.72 s/步）的 3.5 倍——tokens 是 17 倍，所以每 token 反而更便宜（注意力在大序列上利用率更高）。

## 10. 偏差与异常

1. **cache 首次执行 exit 1（三个 native pass 全部成功）**：catalog 7.2 s、text 283.1 s、latents 82.1 s 都 exit 0，随后审计函数读 LengthTable 的 tokens 统计（min/mean of values()）写错而抛错。处理：改成用逐图 tokens 计算；同时发现 lycoris 导出路径不写 kohakumoss_container（只有 comfy 写），把该断言收窄到 comfy；新增 --recheck-cache 对**同一份已建缓存**重审并 pin cache（**未重建缓存**），2.87 s 通过。原证据保留为 stages/cache/result.attempt1.json、exit.attempt1.json、三个 native-*.json 与 *.log，另存 recheck.json。
2. **seg550 首次执行 exit 1（未跑任何训练步）**：我的 describe_initial 断言「续训段的紧凑 B 在适配器构建时就非零」，而训练器的设计是先按 A/alpha/索引建模块、再由 Parts.restore 载入检查点的 B 与优化器，因此构建期 B 恒为零。改成：构建期对全新与续训都断言 B 为零，并新增两条恢复证据（训练器日志 resuming from <path>、段末优化器更新计数 = 段步数）。该次尝试没有消耗任何训练步、没有写入检查点（目录里只有 step_00000250），随后用 --restage 在**同一配置、同一 RESUME** 下重跑，原记录保留为 result.attempt1.json / exit.attempt1.json / stdout.attempt1.log。
3. **两次 freeze amendment**：run.py 因上述两处修复而改动，freeze 正确报告 frozen files changed；分别写 freeze-amendment.json（cache 修复，160af90f → 948aea1c）与 freeze-amendment-2.json（续训断言修复，948aea1c → f411c940），两者都记录「四个配方文件与五个训练器脚本哈希未变」并保留触发证据。freeze.json 的 sha256 表只更新 run.py 一项，且留下 amended_files 记录。
4. **续训首步计时异常**：step 251 / 551 / 851 的 step_time 为 0.0463 / 0.0215 / 0.0161 s（loss 与 grad_norm 正常）。这是计时归属落在恢复阶段所致，不是跳步；全量 900 行连续、无重复、无非有限值。
5. 没有出现 OOM；内存档位未被修改；GPU 全程独占串行；未删任何文件；未 git commit。

## 11. 限制

1. **掩码来自 256² 校准**：阈值与 kept 坐标是 Task 11 在 256² 校准出的稠密 B 上按全局分位数选的；本轮只是把同一份掩码用在原生分辨率上，未重新校准，因此不能回答「原生分辨率下最优稀疏率」或「校准分辨率是否影响掩码质量」。
2. **不同分辨率，不与 256² run 做同配方对照**：本轮每图 tokens 是 256² 的 17 倍，loss 水平（均值 0.37517）与 256² 稀疏 run（0.70588）不是同一尺度，step 时间也不是同一工作量；两者不能当成「同配方不同分辨率」的对照实验，更不能据此说哪个更好。
3. **仍然没有留出集**：285 张全部入训，900 步 = 3.158 个数据轮次（同一 loader 游标跨四段连续），不能判断泛化；没有出图、没有盲评，本报告不含任何画质结论。
4. **分段续训不是端到端单进程**：四段各自重启进程、各自加载一次模型（每段有 20–30 s 启动开销），检查点与优化器状态经过磁盘往返；数值上等价性依赖训练器的 resume 路径，本轮以日志行、优化器计数与 loader 起点作为证据。
5. 单机单卡、单一 seed、单一配方、单一 10% 保留率；comfy 导出是推理专用（稠密 B 视图），不能用于续训（续训会训练 A，导出元数据里已显式标注）。

## 12. 工件清单（相对 experiment/task13/session-20261119-a/）

| 工件 | 路径 |
| --- | --- |
| launcher + worker | run.py（--stage / --worker / --native / --check / --restage / --recheck-cache） |
| 配方 | common.py（TARGET_AREA = 1048576）、data.py、probe.py、train-250/550/850/900.py |
| 冻结与解析 | freeze.json、freeze-amendment.json、freeze-amendment-2.json、resolved.json、resolved-probe.json、resolved-seg*.json、resolved-data.json |
| cache | cache/catalog.db、cache/latents.db、cache/text.db、cache/*.settings.json（本会话新建） |
| 探针 | probe/lori-sparse90-native-probe/（checkpoints/step_00000002 + 两份导出） |
| 训练 | runs/lori-sparse90-native/（metrics.jsonl 900 行、log.txt、config.json、checkpoints/step_00000250|550|850|900/、export/） |
| 终态审计 | final-audit.json（会话根）与 runs/lori-sparse90-native/final-audit.json（内容相同） |
| stage 记录 | stages/{cache,probe,seg250,seg550,seg850,seg900,audit}/{launch,start,result,exit}.json（cache 与 seg550 另有 attempt1 记录） |
