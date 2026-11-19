# Task 10 稠密 B 全量 900 步训练报告

本轮在 RTX 5070 Ti Laptop 12GB 上，用全部 285 对图文把固定 A 的稠密 B 从零训练到 900 次有效更新，并完成进程、检查点、优化器、张量、导出与冻结复核的终态审计。训练配方与 Task 9 稠密对照完全一致（rank 8 / alpha 16、attn-mlp 252 个目标、calibration 阶段 + sparsity=0.0、INT8 ConvRot 主干、BF16、AdamW 1e-4 恒定、batch 1、梯度累积 1、256² 目标面积、ckpt-ring4-actoff-optstream-gradoff），差别只有数据池（285 图 vs 12 图）、步数（900 vs 232）与全新优化器。训练器源码零改动（HEAD 957ccc3f89c1506db8e41ce4d260f529446e142b，git status 干净），缓存与源数据未改动，未出图、未删除任何检查点或导出。

| 指标 | 实测 |
| --- | ---: |
| 有效更新 | 900 / 900 |
| 启动器墙钟 | 1,497.920 s（24 min 57.9 s） |
| 训练调用墙钟（原生 CLI） | 1,472.873 s（24 min 32.9 s） |
| 全量 900 步平均 loss | 0.70493 |
| steps 1–285 / 286–570 / 571–900 平均 loss | 0.71280 / 0.69705 / 0.70494 |
| steps 801–900 平均 loss | 0.70163 |
| 中位步时（全量 / 后 100 步） | 1.602 s / 1.606 s |
| CUDA 峰值 allocated / reserved | 2,612,148,736 / 2,925,527,040 B（2.4328 / 2.7246 GiB） |
| host peak_wset | 14,241,509,376 B（13.2634 GiB） |
| 保留检查点 | 4 个（step 285 / 570 / 855 / 900） |
| 优化器 entry / 矩元素 | 252 / 21,331,968 |
| A、alpha 是否与初始一致 | 252 + 252 全部一致 |
| 稠密 B 是否全部更新 | 252 / 252 |

[启动器](../../../experiment/task10/session-20261118-a/train_launch.py) · [worker](../../../experiment/task10/session-20261118-a/train_worker.py) · [终态审计](../../../experiment/task10/session-20261118-a/final-audit.json) · [worker 结果](../../../experiment/task10/session-20261118-a/stages/train/result.json) · [启动器退出](../../../experiment/task10/session-20261118-a/stages/train/exit.json) · [GPU 初始态](../../../experiment/task10/session-20261118-a/stages/train/initial.json) · [指标](../../../experiment/task10/session-20261118-a/training/lori-dense/metrics.jsonl)

---

## 1. 范围与边界

| 类别 | 内容 | 状态 |
| --- | --- | --- |
| 已完成 | 训练 stage 的 launcher / worker，900 次有效更新的真实 GPU 运行 | ✅ |
| 已完成 | 终态审计（指标、4 个检查点、优化器、A/alpha、稠密 B、导出、冻结复核） | ✅ |
| 已完成 | final-audit.json 写入会话根目录并复制到 training/lori-dense/ | ✅ |
| 未做 | 任何出图、图像网格、视觉评分 | ❌ 不属本任务 |
| 未做 | 改配方、改训练器源码、改缓存、删除检查点或导出 | ❌ 禁止项 |

训练进程为全新 run：RESUME 为空串，stages/train/launch.json 记录了启动前 training/ 不存在，worker 在 apply_adapters 处断言 initial_state 为 None，宿主 optim.pt 的 252 个 entry 全部计数到 900（不是续训后的部分计数）。

## 2. 启动与执行

train_launch.py 与 task9 launch.py 同构：PYTHONDONTWRITEBYTECODE=1、PYTHONPATH=<ws>/KohakuMossTrainer/src;<ws>/.tools/python、CC=<venv>/Lib/site-packages/triton/runtime/tcc/tcc.exe、HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1，并把 12 个缓存变量（TRITON_CACHE_DIR、TRITON_DUMP_DIR、TRITON_OVERRIDE_DIR、TORCHINDUCTOR_CACHE_DIR、KOHAKUMOSS_KERNEL_CACHE、CUDA_CACHE_PATH、TORCH_EXTENSIONS_DIR、HF_HOME、TORCH_HOME、TEMP、TMP、XDG_CACHE_HOME）全部指向 <session>/runtime/train/<key 小写>。启动前置断言：stages/prepare/result.json 与 stages/prepare/exit.json 的 exit_code 均为 0、<session>/training 与 training/lori-dense 均不存在；launch.json 另记录 train.py、common.py、train_launch.py、train_worker.py 的 sha256。

| 时间点（UTC） | 事件 |
| --- | --- |
| 2026-11-18T09:16:53.209 | worker 进程启动，写 start.json |
| 09:17:13 | 原生 CLI 进入 training.entry：900 步、bf16、cuda:0 |
| 09:17:24 | LyCORIS 建立 252 个 LoConModule（bypass_mode 自动转 force-bypass） |
| 09:41:46 | step 900 完成并写出 4 个导出文件 |
| 09:41:49.389 | worker 审计结束，写 result.json（exit_code=0） |
| 09:41:51.013 | 启动器写 exit.json（exit_code=0） |

worker 前置校验逐项通过：freeze.json 的 9 个脚本/配置 sha256、manifest 285 行与 sha256、5 个缓存文件 sha256、3 个权重文件的 size 与 mtime_ns、570 个源文件的 (size, mtime_ns) 快照（摘要 e7022c689cfd18240530b83bed5447eb7759a46e37a00a287ee65c491f40d4cd）、训练器 HEAD 与干净的 git status。train.py 的 143 个 knob 经 validate 与 config_from_knobs(...).validate() 均为空，并逐项比对 STEPS/RUN_NAME/OUTPUT_DIR/SAVE_EVERY/KEEP_LAST/FINAL_CHECKPOINT/SAVE_FORMAT/RESUME/SEED/ALGO_KWARGS/DIM/ALPHA/LR/SCHEDULE/BATCH_SIZE/GRAD_ACC/精度/缓存/数据池。

apply_adapters 被 monkeypatch 捕获 GPU 侧构建的适配器，得到：252 个模块、b_dense 10,665,984 个值且全零、主干参数全部 requires_grad=False、可训练集合恰为 252 个稠密 B。A 与 alpha 逐张量 digest 与 stages/prepare/initial-hashes.json 完全相同：A 7f8b433fe34e4caa789e9923e44227acdef677e04edc1852a71d6242b884db19、alpha 930ce824c0831386ee6bad65542d768c8b56d2c5a3264511be2d7ae02207c991、组合 3731500a5202c3068d1a8a749204c9955b52f091b0fee6139042243aaa6da108。这些值同时与 task9 的 504/504 一致，说明本轮 A 初始化没有漂移。

峰值显存以 torch.cuda.init() 加 kohakumoss_kernels.hardware.memory.open_window("cuda:0") 包裹整个原生 CLI 取得，close_window 读出的 allocated / reserved 分别为 2,612,148,736 / 2,925,527,040 字节（2.4328 / 2.7246 GiB）。逐 step 指标里的 vram_peak_alloc / vram_peak_reserved 最大值同为 2.43275 / 2.72461 GiB，两条独立路径互相印证；ranks/summary.json 另记录 device_used_gib=3.9439、smi_mem_peak_gib=5.4941（含 CUDA 上下文与其它进程的整卡占用）。

## 3. 900 步 loss 与步时

metrics.jsonl 恰 900 行，step 1..900 连续，loss、grad_norm、step_time 全部有限，skipped_steps、nonfinite_steps、nonfinite_params 全为 0，lr 恒为 1e-4。

| 区段 | 步数 | loss 均值 | 最小 | 最大 | 首步 | 末步 | 步时中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全量 1–900 | 900 | 0.70493 | 0.36417 | 0.95832 | 0.73692 | 0.86007 | 1.602 s |
| 1–285 | 285 | 0.71280 | 0.36546 | 0.95832 | 0.73692 | 0.88948 | 1.639 s |
| 286–570 | 285 | 0.69705 | 0.36417 | 0.89776 | 0.66842 | 0.43913 | 1.568 s |
| 571–900 | 330 | 0.70494 | 0.42451 | 0.92787 | 0.68701 | 0.86007 | 1.606 s |
| 801–900 | 100 | 0.70163 | 0.47299 | 0.92787 | 0.63839 | 0.86007 | 1.606 s |

grad_norm 均值 0.18678、末值 0.17973、最大值 2.15106（出现在 step 5，属启动期探针/编译区）；步时中位 1.602 s、均值 1.624 s、最小 1.249 s、最大 8.606 s（step 1）。超过 3 s 的只有 step 1 和 step 2，其余 898 步均进入稳态。

每 100 步 loss 轨迹（块均值）：

| 步段 | 1–100 | 101–200 | 201–300 | 301–400 | 401–500 | 501–600 | 601–700 | 701–800 | 801–900 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 均值 | 0.73050 | 0.70867 | 0.70093 | 0.71212 | 0.68936 | 0.68805 | 0.70370 | 0.70943 | 0.70163 |

抽点 loss（step: loss）：1: 0.73692 · 100: 0.76598 · 200: 0.62802 · 300: 0.74799 · 400: 0.60430 · 500: 0.77175 · 600: 0.84460 · 700: 0.70508 · 800: 0.58838 · 900: 0.86007。

启动期开销（startup/step1_breakdown.json）：step 1 墙钟 8.60406 s，其中 torch_compile 1.475 s、optimizer 0.765 s、compile_probe 0.28 s、forward_backward 0.085 s、triton_jit_load 0.05 s、data 0.01 s；线程侧 kohakumoss-grad-accumulate 5.33 s、读线程 2.665 s。897 步进入 steady_steps，稳态 step_time 1.61079 s（ranks/summary.json），吞吐 0.62081 样本/秒、145.75 token/秒、11.889 TFLOP/s，每步 int8 主干流式 H2D 峰值 9.357 GB。

## 4. 检查点

SAVE_EVERY=285、KEEP_LAST=4、FINAL_CHECKPOINT=True，环形保留后目录里恰 4 个 step 目录，每个都含 state.json、trainable.safetensors、optim.pt、rng.0.pt 与 export/ 下两份 safetensors。运行目录合计 33 个文件、1,432,412,779 字节（1.33 GiB）。

| 检查点 | step = samples | loader | tokens | text_tokens | world |
| --- | ---: | --- | ---: | ---: | ---: |
| step_00000285 | 285 | {"seed":42,"start":285} | 66,900 | 112,490 | 1 |
| step_00000570 | 570 | {"seed":42,"start":570} | 133,800 | 224,980 | 1 |
| step_00000855 | 855 | {"seed":42,"start":855} | 200,700 | 337,470 | 1 |
| step_00000900 | 900 | {"seed":42,"start":900} | 211,287 | 355,358 | 1 |

step 900 的 joint_sq 为 358,156,325，tokens 211,287（平均 234.76 token/样本），与离线阶段记录的 900 batch、3 个完整 epoch 加第 4 轮前 45 条、loader 末态 {"seed":42,"start":900} 完全吻合。

各文件字节数与 sha256：

- step_00000285/：state.json 181 / 38b355b89faa44f59ef6fd97f5568b845c0c8affc1833817cdc653891b395597；trainable.safetensors 74,397,040 / 574be2763b7ce6d752c65c55466735cfd808bc5e4a7d3c60cb1f07c1466d0984；optim.pt 128,213,819 / b0cbbfaf05f1cf0fd4c2c7fdd700e28de8d8a2350b0909827998da6fb0f9300e；rng.0.pt 20,169 / 12d4f46a739908d767a1d40cda20433949063f10f3acdf49c208bb6029349638；export/lori-dense-lycoris.safetensors 74,397,152 / 520ac04443e14918427a8f9de55a250c877c641207959c5a17ccf480c72f6d50；export/lori-dense-comfy.safetensors 49,601,320 / 81d4797d5e7e709c7838c1a161b3f13e438b2e63c71b93c57e6d0657a9901489
- step_00000570/：state.json 182 / e0bb05c6dd4a4836adecfb91b8549b148425333639356fe34f739f91fbc9eec6；trainable.safetensors 74,397,040 / c74e53204fdd85737c313457c153e5305d80d8ac0a8a5ff0f092d971a4e5998d；optim.pt 128,213,819 / 7bf204129c8b060c7917e059796445d6ba97bc6aa94ffb0d13f5f6ef7dc00cc6；rng.0.pt 20,169 / 908610e998e11812a60d9f3aaf2c91f4dcd31179b32e55d4b15b8f9c170ae422；export/lori-dense-lycoris.safetensors 74,397,152 / 0ff41b5da4a436d73c68014025265ded0c47dedadf7588e1838874ad09e884bc；export/lori-dense-comfy.safetensors 49,601,320 / d0e9949354da1c9afeec96d0c2afcb5326b120ddb7f0cf7406bc5262324582e0
- step_00000855/：state.json 182 / 355e284248764ae115204903a80bccb7cfa40036ebad44fd516c7c633da3b62c；trainable.safetensors 74,397,040 / 4b3da8b214e856449ce2a180e1a9132a5b060c8dafe2b8421c7be822043098fc；optim.pt 128,213,819 / d0ad75697a913e2024580ce61359ee79b3d4da1e314be22e18922080783a0e65；rng.0.pt 20,169 / dc3bfacd5f5d3e94333dc6e8936f26694a79d1b02db2c393a80c09d687c6528f；export/lori-dense-lycoris.safetensors 74,397,152 / 922ad3a662c41432fda17c7ff7e6ea2d646ec835292ec47002124b1f36293c93；export/lori-dense-comfy.safetensors 49,601,320 / 92a9533d8e6c602118c0a5c3a2e11e3e4533af2f4a69ed6d9ca5a36c6f57f5ff
- step_00000900/：state.json 182 / e2126ee07eab176d962b65b93e18a3c508b623ff293318c7df38372232b64def；trainable.safetensors 74,397,040 / 8200a308953e8f6e81f50865161e90a5a99d6b0465d81aa81608e5516b3725de；optim.pt 128,213,819 / fc580720364a0c2a65c69fcc1685792065f8985f97a1bf4668cabbb9dbce344c；rng.0.pt 20,169 / ede110fb2a93858f482321f57cab80dde33530b0e77157e3077f5aa5ff7562db；export/lori-dense-lycoris.safetensors 74,397,152 / c4309ca6a4e57c49b1354f96f1eb8ef894cf991d7226072fbe765471105c400f；export/lori-dense-comfy.safetensors 49,601,320 / c8e45d41fee8db1d5df883ece4928d81cc047e2260ea4995e5fb161455159cab

每个检查点 6 个文件、326,629,681 至 326,629,682 字节。运行级 export/ 另有终态两份：lori-dense-lycoris.safetensors 74,397,152 / b7f5eb34185d01a5cf81bcbf0eda59065d6a28bfd7a4e69d2cb1f49439eaec8c、lori-dense-comfy.safetensors 49,601,320 / 477768fae32880ef8aa1d796a7c74323cab762e63595580410329081ae52209f。这两份与 step 900 检查点内导出字节不同，属训练器既有的运行级导出路径（写在同一 run 的 export/ 下），两份都在，未删除。

## 5. 优化器状态（step 900）

optim.pt 的 schedule.last_epoch 为 900，skipped 与 nonfinite 均为 0，optimizers 只有 1 组、恰 252 个 entry（键 0..251 连续）。每个 entry 是四个元素：master、exp_avg、exp_avg_sq、更新计数。

| 列 | 张量数 | 元素数 | dtype | 有限 |
| --- | ---: | ---: | --- | --- |
| master | 252 | 10,665,984 | fp32 | ✅ |
| exp_avg | 252 | 10,665,984 | fp32 | ✅ |
| exp_avg_sq | 252 | 10,665,984 | fp32 | ✅ |
| 两列 AdamW 矩合计 | 504 | 21,331,968 | fp32 | ✅ |

三列的形状多重集合都等于 252 个稠密 B 的形状集合（[3072, 8] 与 [16384, 8] 共 252 项），每个 entry 的更新计数都恰为 900。optim.pt 为 128,213,819 字节，与 task9 稠密 232 步的同名文件同尺寸，说明状态规模由 B 的形状决定，不由步数决定。

## 6. A / alpha 不变、稠密 B 更新

trainable.safetensors 与 export/*-lycoris.safetensors 的键名与 initial-hashes.json 一致（后缀 .A 252 个、.alpha 252 个、.b_dense 252 个，共 756 张量），因此本轮不需要按语义重映射键名；审计同时记录了该键名集合。step 285 与 step 900 均重新逐张量签名：

| 张量组 | 数量 | combined digest（285 与 900 均相同） | 与初始一致 |
| --- | ---: | --- | ---: |
| A | 252 | 7f8b433fe34e4caa789e9923e44227acdef677e04edc1852a71d6242b884db19 | 252 / 252 |
| alpha | 252 | 930ce824c0831386ee6bad65542d768c8b56d2c5a3264511be2d7ae02207c991 | 252 / 252 |

稠密 B 侧（初始全零）：

| 项目 | step 285 | step 900 |
| --- | ---: | ---: |
| 层数 | 252 | 252 |
| 非零值 | 10,665,984（全部非零） | 10,665,984（全部非零） |
| 单层非零最小值 / 最大值 | 24,576 / 131,072 | 24,576 / 131,072 |
| 合并 L2 范数 | 5.49669 | 9.75000 |
| 与零初始相比有更新的层 | 252 / 252 | 252 / 252 |

step 285 与 step 900 逐层比较：252 / 252 层全部不相等（无一层停留在相同数值）。900/285 的逐层 L2 比值中位数 1.7583（最小 1.6181、最大 1.8926），即第二轮与第三轮的更新仍在稳定累积，没有出现某层被冻结或数值爆炸。全部 B 非零且有限。

## 7. 导出可读性

read_lori_state 能直接读取两个检查点的 export/*-lycoris.safetensors：756 张量、A/alpha 与初始 digest 完全一致、252 个 B 全部非零有限，元数据 kohakumoss_format=kohakumoss_lori、kohakumoss_algo=lori、kohakumoss_step 分别为 285 / 900、kohakumoss_adapter_state=1、kohakumoss_spec 记录 dim=8、alpha=16.0、algo_kwargs={"phase":"calibration","sparsity":0.0}、seed=42。ComfyUI 兼容导出能被 safetensors.safe_open 打开，504 张量（lora_down.weight / lora_up.weight / alpha，q/k/v 已融合），元数据 kohakumoss_format=comfy、kohakumoss_fused_qkv=1、ss_network_dim=8、ss_network_alpha=16.0、ss_network_module=lycoris.kohya，首张量 lora_unet_layers_0_proj_ffn_down.alpha（标量 fp32）。两份导出的字节数与 sha256 见第 4 节。

## 8. 训练后的冻结复核

训练结束后重跑了与启动前相同的校验，两次结果逐字段相同（freeze_before 与 freeze_after 完全相同）：

| 项目 | 值 |
| --- | --- |
| 9 个脚本/配置 sha256 | 全部一致 |
| manifest | 285 行，4030fdf3094b91cff164257f70fd5290674dbf5b24f4a1e4a75d19036e869ead |
| catalog.db | 3,686,400 B，fd4bd5debed5dc470edf0e68c676e9f76d3c31c14df8b2bc68882f22adeff0aa |
| text.db | 923,926,528 B，180f536bc53e8bc481709057357ec6c048909716abffa1d36231908e9ed2890c |
| latents.db | 18,841,600 B，4cb07744c0d662d7a177e70dab5d728693fa40cc0c4195b4c62efec2ed409fdc |
| text.settings.json / latents.settings.json | 64e1a048c995ccef9f8de2efc47b5d914b771527703ca3665a36c8c94d7b0847 / 3db09a06b57aeb0e60aff1f184175c7bfc7f2027675c50cbac47e4e61e32e906 |
| 3 个权重 | size 与 mtime_ns 未变（5,413,698,958 / 9,748,121,532 / 249,521,340 B） |
| 570 个源文件快照 | 摘要 e7022c689cfd18240530b83bed5447eb7759a46e37a00a287ee65c491f40d4cd，目录 mtime_ns 未变 |
| 训练器 | HEAD 957ccc3f89c1506db8e41ce4d260f529446e142b，git status 干净 |

## 9. 与 task9 232 步的同配方可比性

两组配方逐字段相同（同一 common.py 血统、rank 8 / alpha 16、attn-mlp 252 个目标、INT8 ConvRot 加 BF16、AdamW 1e-4 恒定、batch 1、256²、ckpt-ring4-actoff-optstream-gradoff、seed 42），A/alpha 504 张量与 task9 step 232 完全一致，因此初始化不是变量。但数据池不同：本轮 285 图，task9 稠密对照 12 图，同一 900 步在两个数据池上不是同一件事，不能把步数当对照。

| 指标 | 本轮（285 图，900 步） | task9 稠密（12 图，232 步） |
| --- | ---: | ---: |
| 数据集轮次 | 900 / 285 = 3.158 | 232 / 12 = 19.33 |
| 平均 loss | 0.70493（全量） | 0.71226（全量） |
| 首 / 末 loss | 0.73692 / 0.86007 | 0.54285 / 0.42150 |
| loss 最小 / 最大 | 0.36417 / 0.95832 | 0.40382 / 0.93787 |
| 中位步时 | 1.602 s | 1.519 s（33–232） |
| step 1 墙钟 | 8.604 s | 7.661 s |
| CUDA allocated / reserved | 2.43 / 2.72 GiB | 2.43 / 2.59 GiB |
| 稠密 B / 矩元素 | 10,665,984 / 21,331,968 | 10,665,984 / 21,331,968 |

三点读法：(1) task9 的 12 图在 232 步里被反复看 19 轮，loss 会明显下降；本轮 3.16 轮看到的样本几乎都不重复，分段均值 0.7128 → 0.6971 → 0.7049 基本平坦，这是数据池差异的直接结果，不能读成训练失败或没有学到东西。(2) 本轮内部可比的是同 run 的 step 285 与 step 900（同配方、同数据顺序、同优化器），两者 B 252/252 不同、L2 从 5.49669 长到 9.75000，是干净的训练进度比较。(3) 视觉收益要靠 step 285 / step 900 / Base 的盲评网格判断；285 图全部参与训练、没有留出集，loss 与 RGB 差异都不能当作泛化证据。

## 10. 偏差、异常与需要注意的点

1. 无失败、无重试：worker 与 launcher 均 exit_code=0，traceback 为空，native_completed=true；原生 TrainState 终值为 step=900、samples=900、tokens=211287、ok=True、skipped=0、nonfinite=0。
2. final-audit.json 的 peaks.host_peak_working_set_bytes 为 null：psutil 的 peak_wset 只有进程退出时才最终确定，而审计在退出前写出，所以该字段留空。真实值在 stages/train/result.json：14,241,509,376 字节（13.2634 GiB）。这是记录时点问题，不是训练异常，也不影响任何断言。
3. reserved 略高于 task9：2.7246 GiB 对 2.5898 GiB（加 138 MiB，加 5.2%），allocated 基本不变（2.43275 对 2.43271 GiB）。本轮离线文本缓存 881 MB、token 分布更长（355,358 text token），分配器保留块相应更多，属预期量级。
4. grad_norm 最大值 2.15106 出现在 step 5，处于启动期编译/探针区；进入稳态后末值 0.17973，未见尖峰或非有限值。
5. 末步 loss 0.86007 是单批噪声，最后 100 步均值 0.70163、全量中位 0.71480，与第 1 段（0.71280）差别很小；不要用单步末值代表趋势。
6. A 哈希、loader 末态、检查点数量、导出键名均无偏差：A/alpha digest 与 prepare 的初始哈希、task9 的 504/504 逐张量一致；loader 末态 {"seed":42,"start":900} 与离线规划一致；环形保留后恰 4 个检查点；检查点/导出的键名就是 .A / .alpha / .b_dense（Comfy 导出另有 lora_down.weight / lora_up.weight 视图），没有出现需要按语义重映射的别名。

## 11. 工件清单

| 工件 | 路径 |
| --- | --- |
| 终态审计（会话根） | experiment/task10/session-20261118-a/final-audit.json（22,349 B，sha256 6447314f71ecb42eb5f02af62a0e4e0c56f12db32917e413740467a42e164884） |
| 终态审计（运行目录副本） | experiment/task10/session-20261118-a/training/lori-dense/final-audit.json（内容相同，sha256 相同） |
| 指标 | experiment/task10/session-20261118-a/training/lori-dense/metrics.jsonl（822,050 B，sha256 ab45aa0ea057a1acc6796e6e28b1c2b08ca83f24f8fa18916d446f322dad355c） |
| 运行日志 | experiment/task10/session-20261118-a/training/lori-dense/log.txt（100,267 B） |
| 运行配置 | experiment/task10/session-20261118-a/training/lori-dense/config.json（5,055 B） |
| 检查点 | experiment/task10/session-20261118-a/training/lori-dense/checkpoints/（step_00000285、step_00000570、step_00000855、step_00000900） |
| 终态导出 | experiment/task10/session-20261118-a/training/lori-dense/export/ |
| step 1 分解 | experiment/task10/session-20261118-a/training/lori-dense/startup/step1_breakdown.json |
| 性能汇总 | experiment/task10/session-20261118-a/training/lori-dense/ranks/summary.json（含 rank0.jsonl） |
| stage 记录 | experiment/task10/session-20261118-a/stages/train/（start.json、initial.json、result.json、launch.json、stdout.log、exit.json） |

final-audit.json 的内容包含了本节所有终态证据：metrics（900 行统计、分段、每 100 步轨迹、抽点 loss、逐 step 显存峰值）、checkpoints（4 个目录的 state 与 6 个文件的字节/sha256）、step_900_state、optimizer、trainable（A/alpha digest、B 非零与 285 对 900 差异）、exports（lycoris/comfy 的可读性、元数据、sha256）、startup、ranks、peaks。出图阶段可直接读 training/lori-dense/final-audit.json 与 checkpoints/step_00000285、checkpoints/step_00000900。

## 12. 下一步

1. 按方案用 step 285 与 step 900 及 Base 出 6 组提示 × 2 seed × 有词/无词，共 24 行、72 张图，盲评风格线索、提示遵循与结构质量。
2. 只有当多数场景、两个 seed 方向一致且提示遵循/结构质量没有重复退化时，才记录为小样本视觉增益。
3. 若要比较稀疏与稠密的同预算质量，需要用同一 285 图配方把稀疏 LoRI 也训到 900 步；普通 LoRA 同预算基线与稀疏率/校准步数消融在更后面。
