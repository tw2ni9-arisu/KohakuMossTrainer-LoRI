# Task 10 数据准备与训练前冻结报告

## MOSS-Image DiT LoRI · 全量 285 图文对 · 稠密 B · 900 步

> **边界声明（先说清楚）**：本报告只覆盖**数据工件（manifest/inventory/source-snapshot）、离线缓存（catalog/text/latents）、训练前 CPU 预检（resolve/offline/prepare 断言）与冻结快照**这四个环节。
> **900 步训练尚未启动，也没有生成任何图片。** 训练进程、优化器、step 285/570/855/900 检查点、LoRA 导出与 24 行 × 3 列评估图全部没有开始；**900 步训练的启动不在本任务范围内**，属于后续单独授权的工作。
> 报告中所有显存与耗时数字都来自**缓存编码与预检阶段**，不是 DiT 训练期测量，不能直接外推训练可用性。

| 元信息 | 值 |
| --- | --- |
| 任务 | Task 10：全量 285 图、稠密 B（rank 8 / alpha 16）、900 次有效更新前的数据与缓存准备 |
| 会话目录 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\` |
| 执行窗口（UTC） | 2026-11-18T08:32:42 → 2026-11-18T08:41:16（本地 16:32–16:41） |
| 训练器 | `KohakuMossTrainer` @ `957ccc3f89c1506db8e41ce4d260f529446e142b`，`trainer_clean=true` |
| 解释器 | `D:\Develop\nigenmame-LoRA\KohakuMossTrainer\.venv\Scripts\python.exe`（只读使用；未 pip install、未升级包） |
| 阶段链 | `resolve → catalog → text → latents → offline → prepare`，六个 stage 全部 `exit_code=0` |
| 训练/出图 | **未启动、未出图**（会话目录下不存在 `training/`） |
| 报告文件 | `D:\Develop\KohakuMossTrainer-LoRI\docs\execution\dit-lori\task-10-data-preparation-report.md` |

---

## 0. 结论摘要

| 结论项 | 结果 |
| --- | --- |
| 数据工件 | `dataset.jsonl` 285 行、549,396 字节、sha256 `4030fdf3094b91cff164257f70fd5290674dbf5b24f4a1e4a75d19036e869ead` |
| 源数据只读 | 570 个源文件（285 图 + 285 txt）读取前后 `(name,size,mtime_ns)` 完全一致，元数据摘要哈希相同 |
| 离线缓存 | catalog 285 行、text 286 条、latents 285 条；5 个缓存工件合计 946,455,170 字节（902.6 MiB） |
| 跨脚本一致性 | 37 个 DataConfig 字段在 train.py 与三个 data.py 之间逐字段相等（3 个设计性排除项）；`validate` / `config_from_knobs(knobs).validate()` / `data_problems` 全为空 |
| 900 batch 编排 | 900 个 batch 全部物化：3 轮 × 285 + 第 4 轮前 45；无裁剪、无丢弃、无缺失缓存键 |
| LoRI 初始态 | 主干全冻结；252 个稠密 B 参数合计 10,665,984 且全零；A/alpha 共 504 张量与 task9 逐张量一致 |
| 冻结快照 | 9 个脚本/配置、manifest、570 源文件、5 个缓存文件、3 个权重、trainer HEAD 全部记录并校验 |
| 总墙钟 | 六个 stage 的 launcher 墙钟合计 **434.9 秒**（worker 合计 429.5 秒），约 7 分 15 秒 |
| 本轮未做的部分 | 900 步训练本身、检查点、LoRA 导出、任何出图与视觉评估 |

---

## 1. 边界：本轮做了什么、没做什么

| 类别 | 内容 | 状态 |
| --- | --- | --- |
| 已完成 | 全量 285 图文对的 manifest 重建（含 inventory 与只读快照） | ✅ |
| 已完成 | catalog / text / latents 三份离线缓存（285 / 286 / 285 条） | ✅ |
| 已完成 | 解析期断言：37 字段跨脚本一致、encoder spec 一致、`validate`/`data_problems` 为空、路径全部在工作区内 | ✅ |
| 已完成 | 900 batch 的完整离线编排审计（CPU、CUDA 隐藏），含 loader 末态与丢弃计数 | ✅ |
| 已完成 | 参数冻结与 LoRI 初始态证明、三份权重与脚本哈希冻结 | ✅ |
| **未开始** | **900 步训练进程、AdamW 优化器、融合 step 实跑** | ❌ 不在本任务范围 |
| **未开始** | **step 285 / 570 / 855 / 900 检查点与导出** | ❌ 不在本任务范围 |
| **未开始** | **任何图像生成、Base/step285/step900 三列对比图** | ❌ 不在本任务范围 |

本轮工作**没有任何一项**涉及 DiT 前向/反向、梯度更新或采样解码；`text` 与 `latents` 两个阶段虽然使用 GPU，但只运行文本编码器与 VAE，不加载 DiT 训练图。

---

## 2. 数据工件

### 2.1 工作区数据工件（`experiment/data/round2-caption-285/`）

| 工件 | 行/条 | 字节 | sha256 | 说明 |
| --- | --- | ---: | --- | --- |
| `dataset.jsonl` | 285 | 549,396 | `4030fdf3094b91cff164257f70fd5290674dbf5b24f4a1e4a75d19036e869ead` | 每行 `{"image": 绝对路径, "caption_nl": 同名 txt 原文}`；UTF-8、LF、`ensure_ascii=False`、`separators=(",",":")` |
| `inventory.json` | 285 条记录 | 223,882 | `0fbd65ef6ae23ae9e14f489dd22358ce51055735f6eecb0c6b1fe5b7d11ae0c7` | 源清单、方向分组、像素范围、caption 统计、逐图逐 caption 哈希、8 项校验 |
| `source-snapshot.json` | 570 文件 × 2 次 | 160,106 | `c5b1789f66538b0aa7a9296422d1a8932453ecfbcae93b9ef27553dc079ae0f3` | 写入前后 `(name,size,mtime_ns)` 与目录 mtime 快照，`read_only_result=PASS` |
| `build_manifest.py` | — | 25,986 | `f8c2a661f7cf2bf544f01d394cb96551eb83fe260ef00b8669768cae135ed2dc` | 可重跑生成器（本轮未重跑，仅记录） |

- 行序 = 文件名**大小写不敏感升序**；本次独立复核 `names == sorted(names, key=lower)` 为真。
- 与 task6 的 `round2-caption-12` **格式字节级一致**：inventory 记录 `format_parity_with_round2_caption_12 = "all 12 reference lines reproduce byte-for-byte"`（12 条参照样本重放比对）。
- 每行 `caption_nl` 与源 `.txt` 逐字节相等：本次独立复核 285/285 相等，**0 mismatch、0 缺失**。

### 2.2 源数据清单（只读引用）

| 项目 | 值 |
| --- | --- |
| 源目录 | `D:\Develop\nigenmame-LoRA\dataset-raw\round2-caption` |
| 文件总数 | 570（285 图 + 285 同名 txt） |
| 图像扩展名 | `.png` 279 / `.jpg` 6 / `.jpeg` 0 |
| 图像字节合计 | 277,264,497 |
| caption 字节合计 | 518,771（UTF-8） |
| 方向分组 | portrait **273**（`height > width`，定义同 task6）/ landscape **12** / square **0** |
| 宽度范围 | 全局 624–1600；portrait 624–960；landscape 1040–1600 |
| 高度范围 | 全局 640–1648；portrait 992–1648；landscape 640–992 |
| 选择规则 | 全量 285，无二次抽样（`selection: "all 285, no subsampling"`） |

### 2.3 Caption 统计（285 条）

| 指标 | 字符数（codepoints） | UTF-8 字节 |
| --- | ---: | ---: |
| min | 1,275 | 1,279 |
| max | 2,529 | 2,529 |
| mean | 1,815.291 | 1,820.249 |
| median | 1,809 | 1,813 |
| sum | 517,358 | 518,771 |

- 285/285 以 `nigenmame` 开头（`caption_prefix_check.all_start_with_prefix=true`），因此训练 `TEXT_DROP=0` 时触发词无法作为独立对照变量。
- 285 条 caption 两两不同（无重复文本）。
- 每条 caption 均非空；本轮未裁剪、未改写任何 caption。

### 2.4 只读证明

| 证据 | 值 | 结果 |
| --- | --- | --- |
| 570 文件 `(name,size,mtime_ns)` 读前摘要 | `69bbc5434c7716fb9a23e9b9aabea7cb1069927a25704698546744452cc02e25` | 与读后相同 |
| 570 文件读后摘要 | `69bbc5434c7716fb9a23e9b9aabea7cb1069927a25704698546744452cc02e25` | 一致 |
| 源目录 mtime_ns | 实测 **1793169777864695900** | 未变（见 §7 关于 100 ns 记录差异） |
| 285 图 + 285 caption 复算哈希 | `selected_content_hashes_unchanged_after_manifest_write=true` | 一致 |
| inventory 校验 1–7 | PASS | 全通过 |
| inventory 校验 8（`PENDING_POST_WRITE`） | 由 `source-snapshot.json` 的 `read_only_result="PASS"` 补全 | 已闭环 |

源图与源 txt **未复制、未移动、未写入**；本轮所有产物在 workspace 内，训练器源码目录零改动。

---

## 3. 六个 stage 的执行与资源测量

所有 stage 均由 `worker.py` 以独立进程运行，`stages/<stage>/` 保留 `launch.json`、`start.json`、`result.json`、等待式 `exit.json` 与 `stdout.log`；`text`/`latents`/`catalog` 另有 native 子进程报告与日志。

**PID 说明**：`launch.json.pid` 是启动器记录的进程号，`result.json.pid` 是解释器内自报的进程号，二者不同属正常（Windows venv redirector 转发），同一作业。下表同时列出。

### 3.1 主测量表

| # | stage | worker PID（launch / result） | worker 秒 | launcher 墙钟秒 | stage 进程 host peak_wset（字节 / MiB） | CUDA 峰值 allocated / reserved（字节） | CUDA 隐藏 | CUDA 未初始化断言 | 缓存产物 |
| ---: | --- | --- | ---: | ---: | ---: | --- | :---: | :---: | --- |
| 1 | resolve | 68828 / 64560 | 17.116 | 18.266 | 811,970,560 / 774.4 | 未记录（解析期已初始化 CUDA） | 否 | —（`cuda_initialized=true`） | 无（写 `resolved.json` + 3 个 `resolved-*.json`） |
| 2 | catalog | 84576 / 60848 | 8.515 | 8.958 | 552,251,392 / 526.7 | 无（CPU 阶段） | 是 | 起点 true / 终点 true | `catalog.db` 285 行 / 3,686,400 字节 |
| 3 | text | 62688 / 83144 | 325.338 | 326.328 | 1,122,009,088 / 1,070.0（native 子进程 18,318,016,512 / 17,469.4） | **1,308,635,136 / 1,344,274,432**（native 子进程内） | 否 | — | `text.db` 286 条 / 923,926,528；`text.settings.json` 236 |
| 4 | latents | 71840 / 26704 | 34.711 | 35.668 | 833,695,744 / 795.1（native 子进程 2,051,391,488 / 1,956.4） | **259,986,944 / 337,641,472**（native 子进程内） | 否 | — | `latents.db` 285 条 / 18,841,600；`latents.settings.json` 406 |
| 5 | offline | 86960 / 77096 | 19.211 | 20.199 | 1,140,776,960 / 1,087.9 | 无（CPU 阶段） | 是 | 起点 true / 终点 true | `batch-ids.json`（900 个 id） |
| 6 | prepare | 15424 / 81348 | 24.559 | 25.527 | 1,071,689,728 / 1,022.0 | 无（CPU 阶段） | 是 | 起点 true / 终点 true | `freeze.json`、`prepare-summary.json`、`initial-hashes.json`、`compare-task9.json` |
| — | **合计** | — | **429.450** | **434.946** | — | — | — | — | 缓存 5 件（见 §4） |

**CUDA 峰值的归属**：只有 `text` 与 `latents` 有 CUDA 峰值，且都来自 **native 缓存子进程**（`cache_text.py` / `cache_latents.py` 的进程内 CUDA 内存窗口）。`resolve` 阶段没有隐藏 CUDA，解析期执行了 `torch.device("cuda:0")` 设备检查与显存档位解析，但**没有记录** CUDA 峰值；其余三个阶段（catalog / offline / prepare）设置 `CUDA_VISIBLE_DEVICES=""`，并在阶段开始与结束各断言一次 `torch.cuda.is_initialized() is False`。

### 3.2 Native 子进程（真正做事的那一层）

| stage | 子进程 PID | 秒 | host peak_wset（字节 / MiB） | CUDA allocated / reserved | 日志与报告 |
| --- | ---: | ---: | ---: | --- | --- |
| catalog（`native`） | 76768 | 3.775 | 568,770,560 / 542.4 | 无（CUDA 隐藏，`cuda_initialized=false`） | `catalog_build.log`、`native-catalog.json` |
| text（`native`） | 83124 | 304.006 | 18,318,016,512 / 17,469.4（≈17.06 GiB） | 1,308,635,136 / 1,344,274,432（1,248.0 / 1,282.0 MiB） | `cache_text.log`、`native-text.json` |
| latents（`native`） | 85620 | 16.992 | 2,051,391,488 / 1,956.4 | 259,986,944 / 337,641,472（247.9 / 322.0 MiB） | `cache_latents.log`、`native-latents.json` |

- text 阶段 325.3 秒中的 304.0 秒花在 native 子进程；其余为导入、父子进程周转与阶段后断言。
- catalog 的 native CLI 只跑 3.775 秒，扫描日志记录 `catalog: 285 images (0 from cache, 285 probed) in 3.65 s`。

### 3.3 时序与串行性

| stage | start（UTC） | end（UTC） | 依赖上一个 stage |
| --- | --- | --- | --- |
| resolve | 08:32:42.235 | 08:32:59.352 | —（根） |
| catalog | 08:33:16.677 | 08:33:25.191 | `{resolve}` |
| text | 08:33:42.388 | 08:39:07.726 | `{catalog}` |
| latents | 08:39:30.207 | 08:40:04.918 | `{text}` |
| offline | 08:40:17.054 | 08:40:36.266 | `{latents}` |
| prepare | 08:40:51.685 | 08:41:16.244 | `{offline}` |

`launch.json.dependencies` 逐级记录上一步的 `exit_code` 与墙钟；**text 结束（08:39:07）早于 latents 开始（08:39:30）**，两个 GPU 编码作业严格串行，同一时刻只有一个 GPU 作业在跑，不存在显存叠加。

---

## 4. 离线缓存清单

### 4.1 文件与哈希

| 文件 | 条目/行 | 字节 | MiB | sha256 |
| --- | ---: | ---: | ---: | --- |
| `catalog.db` | 285 行（live 285） | 3,686,400 | 3.5 | `fd4bd5debed5dc470edf0e68c676e9f76d3c31c14df8b2bc68882f22adeff0aa` |
| `text.db` | 286 条 | 923,926,528 | 881.1 | `180f536bc53e8bc481709057357ec6c048909716abffa1d36231908e9ed2890c` |
| `text.settings.json` | — | 236 | 0.0 | `64e1a048c995ccef9f8de2efc47b5d914b771527703ca3665a36c8c94d7b0847` |
| `latents.db` | 285 条 | 18,841,600 | 18.0 | `4cb07744c0d662d7a177e70dab5d728693fa40cc0c4195b4c62efec2ed409fdc` |
| `latents.settings.json` | — | 406 | 0.0 | `3db09a06b57aeb0e60aff1f184175c7bfc7f2027675c50cbac47e4e61e32e906` |
| **合计（5 件）** | — | **946,455,170** | **902.6** | — |

目录里另有 30 字节的 `catalog.db.ready` 就绪标记；`cache/` 实际占用 946,455,200 字节（902.6 MiB）。

### 4.2 text 缓存

| 项 | 值 |
| --- | --- |
| 条目数 | 286 = 285 条唯一 caption + 1 条空提示 |
| 空提示长度 | 10 token |
| 通道 / dtype | 4096 / `bf16` |
| 长度（286 条）| min 10 / max 552 / mean 393.357 / median 392 / sum 112,500 |
| 长度（仅 285 条 caption）| min 268 / max 552 / mean 394.702 / sum 112,490 |
| settings 记录 | `{"caption":{"template":"nl"},"caption_variants":1,"seed":42,"text_encoder_id":"qwen3.5-9b","text_encoder_spec":"bf16/int8/g0/src:int8-convrot256","cache_dtype":"bf16"}` |

### 4.3 latents 缓存

| 项 | 值 |
| --- | --- |
| 条目数 | 285 |
| dtype / ndim / 通道 | `bf16` / 2 / 128 |
| 总 image token | 66,900 |
| 目标尺寸种类 | 27 种（144×384 … 400×160，最集中 176×352 共 70 条、160×352 共 43 条） |
| 存储形状种类 | 16 种（216×128 … 255×128，最集中 242×128 共 70 条、220×128 共 69 条） |
| 形状规则 | `value.shape[0] == (target_h // 16) * (target_w // 16)`（factor 16 = VAE /8 × 2×2 patchify） |
| settings 记录 | resize / target_area 65,536 / size_step 16 / aspect [0.25, 4.0] / upscale `never` / crop `center` / flip false / `flux2-vae` / `bf16` / `mode` 抽样 / white / lanczos |

### 4.4 缓存与训练配置的一致性

两个 `result.json` 都同时记录了 `settings_recorded`（缓存文件里实际写入的设置）与 `settings_of_train_config`（由训练配置推导的设置），二者逐键相等；`offline` 阶段又用 `settings_of(config, kind) in recorded(...)` 复核一次。因此**缓存条目数与 key 布局是按训练配置 `settings_of` 记录匹配的**，不是"能读到就算过"。

---

## 5. 断言与证据清单（按 stage）

### 5.1 resolve —— 解析与跨脚本一致性

| 断言 | 证据 | 结果 |
| --- | --- | --- |
| DataConfig 37 字段逐字段相等 | `data_fields_compared` 共 37 项（`datasets`…`data_wait`）；排除 `token_budget`/`text_encoder_spec`/`latent_encoder_spec` | 通过 |
| `token_budget` 设计差异 | train.py `"exact"`；三个 data.py 均为 `"estimate"`（脚本分别断言） | 通过（设计差异） |
| 填入 encoder spec 后缓存设置一致 | `settings_of(config, "latents"/"text") == settings_of(reference, ...)` 对 3 个 data.py 全部成立 | 通过 |
| 文本编码规格 | `bf16/int8/g0/src:int8-convrot256`，且与 task6 `resolved-cache_text.json` 的 `TEXT_ENCODE` 逐键相等 | 通过 |
| VAE 规格 | `IMAGE_ENCODER_KWARGS`（path/device/mode）、`VAE_ENCODE`、`VAE_DTYPE="bf16"` 与 task6 记录逐项相等 | 通过 |
| 配置自检 | `validate(train) == []`、`config_from_knobs(train).validate() == []`、`data_problems(reference) == []` | 三者均为空 |
| manifest | 285 行；每行 `image` 为绝对路径且文件存在 | 通过 |
| 训练续训/内核 | `RESUME == ""`、`KERNELS == {}` | 通过 |
| 训练契约 | `STEPS=900`、`SAVE_EVERY=285`、`KEEP_LAST=4`、`FINAL_CHECKPOINT=true`、`RUN_NAME="lori-dense"`、`OUTPUT_DIR=<session>/training` | 通过（`training/` 尚未创建） |
| LoRI 形态 | `ALGO="lori"`、`DIM=8`、`ALPHA=16.0`、`ALGO_KWARGS={"phase":"calibration","sparsity":0.0}`（稠密 B，不建掩码） | 通过 |
| 数据/批大小 | `SEED=42`、`MAX_TOKENS=6144`、`BATCH_SIZE=GRAD_ACC=1` | 通过 |
| 路径约束 | OUTPUT_DIR、CACHE_DIR、catalog、WEIGHTS、DATASETS 的 `path/root/catalog` 全部在 workspace 内；`import_to` 不存在；三个权重文件存在 | 通过 |
| 显存档位计划 | `ckpt-ring4-actoff-optstream-gradoff` → `fused_step=true`、`grad_offload=true`、`stream_resident=4`、`optim_state="stream"`（另 checkpoint=block、act_offload=true、act_offload_inflight=2、base_precision=int8、fp8_recipe=row、fp8_dgrad=fp8、stream=ring、pinned=true、compile=off、kernels={}） | 通过（此轮未实跑融合 step） |
| CUDA 状态 | `resolved.json.cuda_initialized = true`（解析期 `cuda:0` 设备检查；与 task6 行为一致） | 记录在案，见 §7 |

关键数据配置（`resolved.json.data`）：`resolution_mode=resize`、`target_area=65536`、`size_step=16`、`aspect_range=[0.25,4]`、`batching=count`、`batch_size=1`、`max_tokens=6144`、`overlong=drop`、`token_budget=exact`、`caption={template:nl}`、`caption_variants=1`、`text_drop=0`、`latent_cache=text_cache=offline`、`cache_dtype=bf16`、`latent_factor=16`、`num_workers=0`、`prefetch=2`、`pin_memory=true`、`seed=42`、`data_rescan=false`、`data_scan_workers=1`、`data_wait=3600`。

### 5.2 catalog —— 行数、locator、caption 逐字符

| 断言 | 结果 |
| --- | --- |
| catalog 行数 / live 行数 | 285 / 285（`vault_info: generation 1, schema 2, rows 285`） |
| row.id 序列 | 0…284 连续 |
| locator 与 manifest 的 `image` 一一对应 | 相等（`replace("\\","/")` 规范化后）；无重复 locator |
| caption 与 manifest `caption_nl` 逐字符相等 | **285/285**；无空 caption |
| 图像尺寸范围 | min 698,368 / max 1,046,784 像素 |
| 生成器 | `KohakuMossTrainer/scripts/data/catalog_build.py`，扫描日志 "285 images (0 from cache, 285 probed) in 3.65 s" |
| CUDA 隐藏断言 | 起点、终点 `cuda_initialized=false` |

### 5.3 text —— 286 条、长度分布、逐条可读

| 断言 | 结果 |
| --- | --- |
| 期望条数 `len(caption_set(dataset)) = 285 caption + 1 空提示 = 286` | 实际 286 |
| 逐条校验 | 每条 `value.ndim == 2`、`dtype == bfloat16`、全部有限、`value.shape[0] == length`（`cache_text.log` 记录写入 286 embeddings） |
| 空提示存在 | 是（10 token） |
| settings 匹配 | `settings_recorded` 与 `settings_of_train_config` 相等 |

长度分布实测（286 条）：`min 10 / max 552 / mean 393.357 / median 392 / sum 112,500`；仅 caption（285 条）为 `min 268 / max 552 / mean 394.702`。

### 5.4 latents —— 285 条、bf16、形状规则

| 断言 | 结果 |
| --- | --- |
| 条目数 | 285（`cache_latents.log` 记录写入 285 latents） |
| dtype / ndim / 通道 | `bfloat16` / 2 / 128 |
| 逐条校验 | 键存在、有限、`shape[0] == (h//16)*(w//16)`（按该行目标尺寸） |
| 总 token | 66,900 |
| 尺寸种类 | 目标尺寸 27 种、存储形状 16 种，各自计数合计 285 |
| settings 匹配 | `settings_recorded` 与 `settings_of_train_config` 相等 |

### 5.5 offline —— 900 batch 完整编排（CPU，CUDA 隐藏）

| 断言 | 结果 |
| --- | --- |
| batch 数 / epoch 长度 | 900 / 285（planner 的 epoch 0–3 长度均为 285） |
| 结构 | `3 × 285 = 855` 个 batch 为 3 个完整 epoch + 尾段 45 个 batch |
| 覆盖 | 285 个 id 全部物化；前 855 个 batch 里每个 id 恰出现 3 次（`set(counts.values()) == {3}`） |
| 尾段 | 后 45 个 batch 与 `planner.epoch(3)[:45]` 逐位相等；实测 tail 前 10 个 id 为 `278, 174, 116, 97, 167, 33, 131, 204, 173, 22` |
| loader 末态 | `state_dict() == {"seed": 42, "start": 900}`（初始 `{"seed":42,"start":0}`） |
| 无裁剪 | `batch.text_limit` 全为 0（0 = 整条 caption）；`batch.text_len.sum() == batch.text.shape[0]` |
| 无图像 token 丢弃 | `batch.img_keep` 全为 `None`；每 batch latent 行数 == 该 batch 的 `hw` 乘积 |
| 无丢弃/无未入批 | `planner.drops` 与 `planner.unbatched` 在 4 个 epoch 全为 0 |
| 每 batch 规模 | text token 268–552（900 batch 合计 355,358）；latent 行 216–255（合计 211,287） |
| batch id 序列 | 900 条落在 `stages/offline/batch-ids.json`（含 `ids`、`tail_ids`、`epoch_4_order`、`loader_state`） |
| CUDA 隐藏断言 | 起点、终点 `cuda_initialized=false` |

交叉校验：`355,358 / 900 ≈ 394.8` token/样本，与 text 缓存 caption 均值 394.702 吻合；`211,287 / 900 ≈ 234.8` latent 行/样本，与 `66,900 / 285 ≈ 234.7` 吻合——说明 900 个 batch 使用的正是这 285 条缓存条目，没有回退到在线编码。

### 5.6 prepare —— 冻结与 LoRI 初始态

| 断言 | 值 |
| --- | --- |
| 参数量 | 714 个参数张量 = 冻结 462 + 可训练 252 |
| 冻结元素数 | 5,082,308,608（`frozen_numel`） |
| 可训练集合 | 恰为 `adapters.trainable_params()`（252 个，`trained == trainable`；无其他参数可训练） |
| 可训练元素数 | 10,665,984 == `dense_b_values` |
| LoRI 模块 | 252 个目标模块（`attn-mlp`）；prepare stdout：`create LyCORIS: 252 modules`、`module type table: {'LoConModule': 252}` |
| 稠密 B | 252 个 B 全零、`ndim == 2`、第二维 == `DIM 8` |
| A / alpha | 252 + 252 个固定张量（`_live()` 中 `.A`/`.alpha` 共 504；`.b_dense` 252） |
| 组合 digest | `3731500a5202c3068d1a8a749204c9955b52f091b0fee6139042243aaa6da108` |
| A digest | `7f8b433fe34e4caa789e9923e44227acdef677e04edc1852a71d6242b884db19` |
| alpha digest | `930ce824c0831386ee6bad65542d768c8b56d2c5a3264511be2d7ae02207c991` |
| digest 算法 | 单张量：`kohakumoss.adapters.train.lori._digest`（`sha256("<dtype>:<shape>:" + 原始字节)`）；组合：按名字排序的 `"name:digest\n"` 串的 sha256（与 task9 同源可比） |
| 与 task9 比较 | `step_00000232/trainable.safetensors` 中 504 张量 **504/504 完全一致、0 差异**；`is_a_failure_condition=false`（仅记录，不作为失败条件） |
| CUDA 隐藏断言 | 起点、终点 `cuda_initialized=false`；`torch.set_num_threads(1)` |

冻结覆盖（`freeze.json`）：

| 冻结对象 | 数量 | 校验方式 |
| --- | ---: | --- |
| 私有脚本 + 训练器入口脚本 | 9 | `common.py`、`data.py`、`train.py`、`pipeline.py`、`worker.py`、`scripts/train/moss.py`、三个 `scripts/data/*.py` 的 sha256 |
| manifest | 1 | `manifest_sha256 = 4030fdf3…`（与 dataset.jsonl 一致） |
| 源文件快照 | 570 | 每个 `(size, mtime_ns)` + 目录 `mtime_ns` |
| 缓存文件 | 5 | sha256 + 字节（见 §4.1，与磁盘现值一致） |
| 权重 | 3 | sha256 + size + mtime_ns，`matches=true` |
| trainer | 1 | HEAD `957ccc3f89c1506db8e41ce4d260f529446e142b`、`clean=true`（与 `resolved.json.revision` 一致） |

| 权重 | 字节 | sha256 | 校验 |
| --- | ---: | --- | --- |
| `moss_image_base_int8_convrot.safetensors` | 5,413,698,958 | `f601c1e1161915fb3fea003da0c1782e374dd960fbb3978bc0e4e3fa1f2d8e6f` | matches=true（sha256+size 源自 task7 freeze，mtime 源自 task9 freeze） |
| `qwen3.5_9b_int8_convrot.safetensors` | 9,748,121,532 | `32e3787472f708fa5f83624acbf5ef23f7fc89c439e05065ea57cfe5a97f025a` | matches=true |
| `flux2-vae_small.safetensors` | 249,521,340 | `d8d52ba036475f5fb07c8b435e176d3d97ebfa82f0d1a1c317f9cc1e25bd013b` | matches=true |
| **合计** | **15,411,341,830** | — | 三项全部 `matches=true` |

---

## 6. 与 task6（12 图）的差异：只由数据量解释

| 对照项 | task6（12 图） | task10（285 图） |
| --- | --- | --- |
| 数据规模 | 12（10 portrait + 2 landscape） | 285（273 portrait + 12 landscape）＝ **23.75 倍** |
| catalog 条目 | 12 | 285 |
| text 条目 | 13（12 caption + 空提示） | 286（285 caption + 空提示） |
| latents 条目 | 12 | 285 |
| text 阶段 worker 秒 | 201.472 | 325.338（native 子进程 304.006） |
| latents 阶段 worker 秒 | 1.242 | 34.711（native 子进程 16.992） |
| catalog 阶段 worker 秒 | 0.321 | 8.515（native CLI 3.775） |
| text CUDA 峰值 allocated/reserved | 1,308,635,136 / 1,344,274,432 | **相同**：1,308,635,136 / 1,344,274,432 |
| latents CUDA 峰值 allocated/reserved | 260,576,768 / 301,989,888 | 259,986,944 / 337,641,472 |
| offline 审计范围 | 12 个 count-batch1、epoch-0 batch | 900 个 batch（3.16 epoch） |
| 缓存体积 | 数十 MB 量级 | 946,455,170 字节（902.6 MiB） |

**配方逐项相同**：`12g_stream` 基础配方、`TARGET_AREA=256²`、`SIZE_STEP=16`、`ALPHA=16`、`DIM=8`、`SEED=42`、`TEXT_DROP=0`、bf16 缓存/计算 + INT8 主干、`ckpt-ring4-actoff-optstream-gradoff`、AdamW `1e-4`、constant 学习率、`GRAD_CLIP=1.0`、`KERNELS={}`、`COMPILE=off`。除 `DATASETS`/`CACHE_DIR` 指向新会话外，`common.py` 与 task6 版本**只有 2 行不同**（第 20、21 行：manifest 路径与 root 的 `round2-caption-12` → `round2-caption-285`）。

因此本轮相对 task6 的**全部体积与时间差异量级，都可由数据量（约 23.75 倍条目）解释**：

- **缓存体积**：`text.db` 承载 286 条 4096×bf16 嵌入，约 882 MB；`latents.db` 承载 285 条 128 通道 bf16 latent，约 18 MB——两者比例主要由条目数与单条 token 数决定。
- **耗时**：`text` 从 201 s 量级升到约 300 s（条目 13→286），`latents` 从 1 s 量级升到约 30 s（条目 12→285），`catalog` 从 0.3 s 量级升到约 8 s（条目 12→285）。
- **但不能说"线性"**：task6 的单条均摊（text 201.5 s / 13 条 ≈ 15.5 s/条）明显高于本轮（304.0 s / 286 条 ≈ 1.06 s/条），说明权重加载与 kernel 首次编译是固定开销。同理，本轮 text 约 300 秒、latents 约 30 秒的**绝对差**来自文本编码器（9B 语言模型）与 VAE 的规模差异，**不是**数据量差异。二者 CUDA 峰值也几乎相同（text 完全一致），因为峰值由常驻权重与单条最长 caption 的编码窗口决定，与条目总数无关。
- **900 batch 是训练轮次设计**：900 次更新 ÷ 285 ≈ **3.16 个数据集轮次**，是本轮"新 run、从零初始化 B 与 AdamW、保存 step 285/900 做同配方进度比较"的配方要求；它**不是**把 task6 的 12 图循环放大，也不能与旧 12 图 232/900 步（约 75 轮重复）视为步数对照。

---

## 7. 风险与注意点

| # | 风险/注意点 | 证据 | 影响与建议 |
| ---: | --- | --- | --- |
| 1 | resolve 阶段初始化了 CUDA | `resolved.json.cuda_initialized=true`；该阶段未隐藏 CUDA，解析时用 `torch.device("cuda:0")` 做设备检查与显存档位解析 | 这是解析期的一次性设备检查，进程退出即释放；但它意味着"训练前预检"并非全程不接触 GPU。后续若要求严格 CPU-only，应把 resolve 也设为 `CUDA_VISIBLE_DEVICES=""`——本轮未改，与 task6 保持一致。 |
| 2 | text 缓存子进程主机内存峰值很高 | `native-text.json.peak_working_set_bytes = 18,318,016,512`（≈17.06 GiB）；同阶段外层进程仅 1,122,009,088（1.07 GiB） | 900 步训练若与 9B 文本编码器同时驻留，需要相应主机内存余量；本轮**没有**测量训练期主机内存，不应假定训练阶段一定低于此值。latents 子进程为 2,051,391,488（1.96 GiB）。 |
| 3 | 缓存只覆盖当前配方 | 两个 `settings.json` 与训练 `settings_of` 完全一致，但只含 285 条 caption + 1 条空提示 | 改动 caption 模板/变体数/seed/分辨率/编码器规格都会让 text/latents 缓存失效并需重建；catalog 只对文件清单敏感。 |
| 4 | 启动前 token 粗估偏高 | 粗估均值约 519、最大约 723；实测 caption 均值 **394.702**、286 条最大 **552** | 粗估公式（`chars/3.5 + 12`）对中文 caption 明显保守（约 1.3 倍）。`MAX_TOKENS=6144` 远大于实测 552，`OVERLONG="drop"` 未触发；粗估值可继续作为上界，但不应当作精确预算。 |
| 5 | 源目录 mtime 有两套记录 | `source-snapshot.json` 记 1793169777864695800（前/后相同）；`freeze.json` 记 1793169777864695900；直接查询目录 `LastWriteTimeUtc` 换算为 **1793169777864695900** | 差 100 ns，属 `st_mtime` 浮点换算的记录精度问题，不是目录被改动；两份文件各自的前后对比都表明"未变"。 |
| 6 | 没有独立留出集 | 285 图全部参与训练 | 本轮只能做工作集内的进度对照（step 285 vs step 900），不能据此声称泛化能力或触发词因果效果。 |
| 7 | 磁盘占用 | `cache/` 946,455,200 字节 + `runtime/` 113,342,848 字节（864 个文件，多为 Triton/Inductor 编译缓存），会话目录合计 1,060,134,662 字节 | 训练前不要清理 `cache/` 与 `runtime/`；它们是本轮可复用的产物。 |
| 8 | 非致命提示 | prepare stdout 有 LyCORIS 提示：`Non-native Linear detected but bypass_mode is not set. Automatically using force-bypass mode…`；text 缓存日志只有 `registry` 后端信息 | 提示未改变参数或配方（`ADAPTER_MODE="bypass"`，252 个 `LoConModule`）；本轮 text 日志未出现 task6 那条 Inductor `Not enough SMs` 提示。 |

---

## 8. 失败与残留

| 项 | 结论 |
| --- | --- |
| 失败阶段 | **本轮无失败**：六个 stage 的 `result.json.exit_code` 与等待式 `exit.json.exit_code` 全部为 0，native 子进程 `native_completed=true`、`exit_code=0` |
| 重试/回退 | 无重试、无参数回退、无跳过阶段；`launch.json.dependencies` 记录的上游退出码全部为 0 |
| 不可复用残留 | **没有**。`cache/` 下 5 个缓存工件 + `catalog.db.ready` 可直接被 900 步训练复用，其 settings 与训练配置一致（§4.4） |
| 训练目录 | 会话下 `training/` **不存在**，证明训练未启动；无图片输出目录 |
| 副作用 | 未修改实验目录既有产物、未改缓存、未改数据、未改 `KohakuMossTrainer` 源码、未 `git commit`、未安装依赖、未运行任何 stage 之外的训练或出图 |

---

## 9. 产物路径全清单

下表所有路径均为绝对路径（Windows 本地路径，未做简写）。

| 类别 | 绝对路径 | 说明 |
| --- | --- | --- |
| 数据工件 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\data\round2-caption-285\dataset.jsonl` | 285 行 manifest（549,396 字节） |
| 数据工件 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\data\round2-caption-285\inventory.json` | 源清单/统计/逐条哈希（223,882 字节） |
| 数据工件 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\data\round2-caption-285\source-snapshot.json` | 570 文件只读快照（160,106 字节） |
| 数据工件 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\data\round2-caption-285\build_manifest.py` | 可重跑生成器（25,986 字节） |
| 会话根目录 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\` | 本轮全部实验产物 |
| 会话脚本 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\common.py`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\data.py`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\train.py`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\pipeline.py`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\worker.py` | 私有配方与阶段执行器 |
| resolve | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\resolve\launch.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\resolve\start.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\resolve\result.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\resolve\exit.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\resolve\stdout.log`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\resolve\resolved.json` | 解析与断言记录（resolved.json 22,716 字节） |
| resolve 附加 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\resolved-catalog_build.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\resolved-cache_text.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\resolved-cache_latents.json` | 三个 data.py 的原生 knobs（3,538 / 3,574 / 3,575 字节） |
| catalog | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\catalog\launch.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\catalog\start.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\catalog\result.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\catalog\exit.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\catalog\stdout.log`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\catalog\native-catalog.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\catalog\catalog_build.log` | 建库与扫描日志 |
| text | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\text\launch.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\text\start.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\text\result.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\text\exit.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\text\stdout.log`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\text\native-text.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\text\cache_text.log` | 文本缓存与每条长度 |
| latents | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\latents\launch.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\latents\start.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\latents\result.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\latents\exit.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\latents\stdout.log`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\latents\native-latents.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\latents\cache_latents.log` | latent 缓存与形状统计 |
| offline | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\offline\launch.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\offline\start.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\offline\result.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\offline\exit.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\offline\stdout.log`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\offline\batch-ids.json` | 900 batch 审计与 id 序列（batch-ids.json 12,172 字节） |
| prepare | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\prepare\launch.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\prepare\start.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\prepare\result.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\prepare\exit.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\prepare\stdout.log`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\prepare\initial-hashes.json`<br>`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\stages\prepare\compare-task9.json` | 冻结 / LoRI 初始态证明（initial-hashes.json 119,441 字节） |
| 冻结 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\freeze.json` | 9 脚本 + manifest + 570 源文件 + 5 缓存 + 3 权重 + trainer HEAD（53,854 字节） |
| 摘要 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\prepare-summary.json` | 参数/哈希/trainer 摘要（1,445 字节） |
| 缓存 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\cache\catalog.db` | 285 行、3,686,400 字节 |
| 缓存 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\cache\catalog.db.ready` | 30 字节就绪标记 |
| 缓存 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\cache\text.db` | 286 条、923,926,528 字节 |
| 缓存 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\cache\text.settings.json` | 236 字节 |
| 缓存 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\cache\latents.db` | 285 条、18,841,600 字节 |
| 缓存 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\cache\latents.settings.json` | 406 字节 |
| 运行时缓存 | `D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\runtime\<stage>\` | 进程级缓存，864 个文件、113,342,848 字节；每个 stage 目录含 triton_cache_dir、triton_dump_dir、triton_override_dir、torchinductor_cache_dir、kohakumoss_kernel_cache、cuda_cache_path、torch_extensions_dir、hf_home、torch_home、temp、tmp、xdg_cache_home |
| 本报告 | `D:\Develop\KohakuMossTrainer-LoRI\docs\execution\dit-lori\task-10-data-preparation-report.md` | 本文件 |

`D:\Develop\KohakuMossTrainer-LoRI\experiment\task10\session-20261118-a\runtime\<stage>\` 中的 `<stage>` 取值依次为 `resolve`、`catalog`、`text`、`latents`、`offline`、`prepare`；上述路径全部位于工作区内，不含工作区外的读写目标。

## 10. 复核方式与不变量

本报告的数字全部取自上述 JSON/日志，并对数据工件做了独立只读复算：

| 复核项 | 方式 | 结果 |
| --- | --- | --- |
| 三个 JSON 的字节数与 sha256 | 直接读文件、`sha256` 重算 | 与 brief/记录一致 |
| dataset.jsonl 行数与顺序 | 逐行解析 285 条、比较 `sorted(names, key=lower)` | 285 行、顺序正确 |
| caption 统计 | 逐条 `[...caption]` 计 codepoints、`UTF-8` 计字节，取 min/max/mean/median/sum | 与 inventory 记录一致（含 median 1809 / 1813） |
| caption 与源 txt | 285 条逐字节比对 | 0 mismatch |
| 前缀与去重 | `startswith("nigenmame")`、`set(captions)` | 285/285、285 种 |
| 阶段与缓存数字 | 逐个读 `stages/*/{result,exit,native-*}.json`、`freeze.json`、`prepare-summary.json` | 与报告表格一致 |
| 目录 mtime | `Get-Item ... LastWriteTimeUtc.Ticks × 100` | 1793169777864695900（见 §7 第 5 条） |
| 未改动声明 | 本轮只创建本报告文件；未写实验/缓存/数据/训练器；未运行 stage、训练或出图；未 `git commit` | ✅ |

**进入 900 步训练前可直接依赖的不变量**：

1. manifest 与缓存 key 的对应关系（285 行 ↔ 285 catalog 行 ↔ 285 latent 条 ↔ 286 文本条）已在 offline 阶段端到端走过一遍；
2. A/alpha 初始 digest 与 task9 的 504 张量一致，训练后可用同一算法判断"固定投影是否被改动"；
3. 冻结快照给出了脚本、manifest、源文件、缓存、权重、trainer HEAD 的完整基线，训练后任何一项漂移都可定位；
4. 训练自身（前向/反向、融合 step、检查点、出图）**尚未执行**，其显存与耗时仍需单独测量，本报告不提供该结论。
