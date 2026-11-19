# LoRI 稠密 B 第 570 步：单列补测与 Base / 285 / 570 / 900 四列轨迹

**结论先行。** 用一个只渲染适配器列、复用 `grid.render_columns` 的单列 driver，在不重生 Base 的前提下补出了 step 570 的 24 张图；它先通过了**逐步位级 canary**（用 step 285 适配器重放 r00/r14，与归档 `r00_c01`／`r14_c01` 解码 RGB **sha256 完全相同、MAE 0.0000、changed_pixels 0**），因此这一列与归档的 285/900 列走的是同一条数值路径。

四列（Base / 285 / 570 / 900）在 24 行上的 Base 相对偏移：**30.36 → 31.44 → 34.15**，相邻位移 285→570 为 **21.93**、570→900 为 **20.78**，两者量级接近，说明漂移不是集中在最后一段。**6 组里有 5 组的组均值为单调递增（第 4 组图书馆在 570 处回落到 285 以下）**，但**逐行只有 13/24 行**满足"570 落在 285 与 900 之间"，严格递增只有 12/24 行。

视觉上 570 处在"已经出现退化、但不是全部行都退化"的状态：r08（右侧人物无五官）、r12（左侧人物无眼、手成手套状）、r16（人物完全背身、看不到脸）已经出现与 900 同类或更极端的退化形态（r08 在 900 是转脸、570 是直接留空；r12 在 900 是闭眼长鼻线、570 是无眼空白）；而 r01、r06、r14、r20 仍保持完整面部，其中 r14 的手部明显好于同行 900 的连片手套，r01/r06/r20 与 900 大致相当。所以 570 不是 285 与 900 的逐行插值，退化是**行相关的、不均匀的推进**。全部 285 张训练图都参加了训练，本轮仍没有独立留出集；像素差只表示同种子偏移，不是质量或泛化判据。

[四列对照图](../../../experiment/task10/comparison-570-20261118-a/quad-grid.png) · [一屏总览](../../../experiment/task10/comparison-570-20261118-a/quad-contact-sheet.png) · [盲评图](../../../experiment/task10/comparison-570-20261118-a/blind-quad-grid.png) · [盲评表](../../../experiment/task10/comparison-570-20261118-a/blind-quad-sheet.md) · [像素审计](../../../experiment/task10/comparison-570-20261118-a/pixel-audit-570.json) · [冻结记录](../../../experiment/task10/comparison-570-20261118-a/freeze-570.json)

## 方案：单列 driver 与 compare_grid 同源

`compare_grid.py` 的 `render_columns` 对每一列执行：`adapter_api.load(backbone, path, mult, mode, row_scales=output_row_scales(backbone))` → `requires_grad_(False)` → `backbone.bind_ops()` → `pipe.generate(...)` → 移除后再 `bind_ops()`。归档的 285/900 列就是这条路径。本轮新增的 [render_single.py](../../../experiment/task10/comparison-570-20261118-a/render_single.py) 按 `compare_grid.py` 的同一组参数构造 `MossPipeline.from_pretrained`（同一 WEIGHTS/VARIANT/TRANSFORMER/TEXT_ENCODER/VAE、`PRECISION=bf16`、`TEXT_ENCODE`/`VAE_ENCODE`、`MEMORY_PROFILE=minimal`、`BASE_PRECISION=int8`、`FP16_FIX=none`、`TUNE=off`、`VAE_DTYPE=bf16`、`forward_tokens(24 行 sizes, batch=1, CFG=3, two_pass)`），然后只传一个列 `[grid.Column(label, (adapter, 1.0))]` 给 `grid.render_columns`（mode=bypass、空负向、512²、Euler 50 步、CFG 3、shift 5、default skip、batch 1、two_pass）。**不传 Base 列**，因此不重生 Base。

### canary：单列 driver 与归档逐位相同（本方案可信的前提）

用 step 285 导出重放 r00（seed 42）与 r14（seed 43），与归档列的 `c01` 图对比：

| 行 | 文件 | seed | 归档文件 | 归档 rgb_sha256 | canary rgb_sha256 | 相同 |
| --- | --- | ---: | --- | --- | --- | --- |
| r00 | r00_c285.png | 42 | r00_c01.png | a77ac81d21a5cf85… | a77ac81d21a5cf85… | 是 |
| r14 | r14_c285.png | 43 | r14_c01.png | a8df90e214130731… | a8df90e214130731… | 是 |

结论：2/2 位级相同。这也同时证明"跳过 Base 列、直接渲染适配器列"在数值上等价 —— `compare_grid` 归档时先渲染 Base 列并没有改变后续适配器列的数值。

## 四列像素差

数值是解码后 8-bit RGB 的通道平均绝对差（0–255）。每行的 570 图来自本轮 `images-570/rXX_c570.png`，Base/285/900 列只读复用 `comparison-20261118-a/images/` 的 `c00`/`c01`/`c02`；两组归档列的 24 张图块哈希在本轮 prepare 中已逐块复核未变。

| 行 | 组 | 场景 | seed | 触发词 | Base→285 | Base→570 | Base→900 | 285→570 | 570→900 | 285→900 | 570 在 285/900 之间 |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r00 | 1 | 粉色室内双人与布偶 | 42 | 有 | 41.61 | 39.16 | 44.50 | 21.64 | 35.12 | 30.02 | 否 |
| r01 | 1 | 粉色室内双人与布偶 | 42 | 无 | 27.08 | 29.49 | 37.07 | 21.40 | 28.97 | 30.09 | 是 |
| r02 | 1 | 粉色室内双人与布偶 | 43 | 有 | 40.82 | 35.68 | 40.00 | 32.10 | 22.64 | 31.01 | 否 |
| r03 | 1 | 粉色室内双人与布偶 | 43 | 无 | 32.40 | 51.61 | 53.57 | 46.33 | 27.66 | 46.55 | 是 |
| r04 | 2 | 紫发单人与靛蓝星空 | 42 | 有 | 25.83 | 24.43 | 27.87 | 25.65 | 18.00 | 29.09 | 否 |
| r05 | 2 | 紫发单人与靛蓝星空 | 42 | 无 | 28.54 | 25.29 | 28.79 | 28.25 | 20.73 | 32.71 | 否 |
| r06 | 2 | 紫发单人与靛蓝星空 | 43 | 有 | 31.82 | 36.95 | 38.31 | 23.14 | 27.65 | 33.74 | 是 |
| r07 | 2 | 紫发单人与靛蓝星空 | 43 | 无 | 32.91 | 34.15 | 37.50 | 20.95 | 28.68 | 32.18 | 是 |
| r08 | 3 | 海边夕阳双人与仙女棒 | 42 | 有 | 20.94 | 21.58 | 22.16 | 10.38 | 10.35 | 11.94 | 是 |
| r09 | 3 | 海边夕阳双人与仙女棒 | 42 | 无 | 23.68 | 23.11 | 25.89 | 14.01 | 15.90 | 19.98 | 否 |
| r10 | 3 | 海边夕阳双人与仙女棒 | 43 | 有 | 19.53 | 24.34 | 23.55 | 19.49 | 17.19 | 17.02 | 否 |
| r11 | 3 | 海边夕阳双人与仙女棒 | 43 | 无 | 17.84 | 23.59 | 23.98 | 21.85 | 13.72 | 20.16 | 是 |
| r12 | 4 | 图书馆阅读双人 | 42 | 有 | 31.20 | 29.93 | 29.50 | 19.95 | 18.75 | 21.19 | 是 |
| r13 | 4 | 图书馆阅读双人 | 42 | 无 | 36.98 | 29.87 | 32.16 | 26.92 | 22.01 | 29.55 | 否 |
| r14 | 4 | 图书馆阅读双人 | 43 | 有 | 45.84 | 38.87 | 48.63 | 22.80 | 28.20 | 27.75 | 否 |
| r15 | 4 | 图书馆阅读双人 | 43 | 无 | 41.57 | 41.92 | 45.24 | 21.86 | 27.41 | 26.73 | 是 |
| r16 | 5 | 城市雨夜街头单人 | 42 | 有 | 35.09 | 34.07 | 37.24 | 18.57 | 18.01 | 23.73 | 否 |
| r17 | 5 | 城市雨夜街头单人 | 42 | 无 | 33.01 | 32.64 | 37.28 | 21.70 | 20.54 | 23.86 | 否 |
| r18 | 5 | 城市雨夜街头单人 | 43 | 有 | 32.29 | 31.20 | 32.67 | 16.27 | 16.51 | 15.78 | 否 |
| r19 | 5 | 城市雨夜街头单人 | 43 | 无 | 22.46 | 27.02 | 27.78 | 18.41 | 11.81 | 18.64 | 是 |
| r20 | 6 | 温室花园双人 | 42 | 有 | 27.64 | 29.07 | 33.17 | 20.12 | 19.74 | 16.60 | 是 |
| r21 | 6 | 温室花园双人 | 42 | 无 | 26.49 | 30.23 | 31.11 | 22.25 | 20.45 | 15.52 | 是 |
| r22 | 6 | 温室花园双人 | 43 | 有 | 25.68 | 28.21 | 28.85 | 15.47 | 14.42 | 18.08 | 是 |
| r23 | 6 | 温室花园双人 | 43 | 无 | 27.48 | 32.16 | 32.88 | 16.75 | 14.15 | 18.91 | 是 |
| 24 行均值 | | | | | 30.36 | 31.44 | 34.15 | 21.93 | 20.78 | 24.62 | 13/24 |

按组汇总：

| 组 | 场景 | Base→285 | Base→570 | Base→900 | 285→570 | 570→900 | 285→900 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 粉色双人与布偶 | 35.48 | 38.99 | 43.78 | 30.37 | 28.60 | 34.42 |
| 2 | 紫发星空 | 29.78 | 30.20 | 33.12 | 24.50 | 23.77 | 31.93 |
| 3 | 海边夕阳 | 20.50 | 23.15 | 23.89 | 16.43 | 14.29 | 17.28 |
| 4 | 图书馆阅读 | 38.90 | 35.15 | 38.88 | 22.88 | 24.09 | 26.31 |
| 5 | 雨夜街头 | 30.71 | 31.23 | 33.74 | 18.74 | 16.72 | 20.50 |
| 6 | 温室花园 | 26.82 | 29.91 | 31.50 | 18.65 | 17.19 | 17.28 |

**单调性。** 逐行满足 `Base→285 ≤ Base→570 ≤ Base→900` 的只有 12/24 行（"570 落在 285 与 900 之间"放宽为区间判断是 13/24 行）。不满足区间判断的 11 行是 **r00, r02, r04, r05, r09, r10, r13, r14, r16, r17, r18**；其中 r12 的 900 位移（29.50）小于 285（31.20），其余 10 行是 570 比 285 更靠近 Base。组均值层面 5/6 组单调递增，唯一例外是第 4 组图书馆：38.90 → 35.15 → 38.88（570 回落）。

**相邻位移。** 285→570 均值 21.93、570→900 均值 20.78，与 285→900 的 24.62 相比，说明 285 之后的漂移是两段近似等量的推进，而不是一次跳变。两组相邻位移的逐行最大值都出现在 r03（46.33 / 27.66），最小值在 r08 与 r19（10.35 / 11.81）。

## 视觉观察（570 列）

以下描述来自实际打开 `quad-contact-sheet.png` 的六组总览与 570 列的 8 张 512² 单图（r00、r01、r06、r08、r12、r14、r16、r20）；其余 16 行只看约 10% 缩放的总览图，未逐张放大。

- **r00（粉色双人，seed 42，有词）。** 两人保留了红白表演服、中央布偶与粉色背景；两张脸都有清晰的虹膜与睫毛，但嘴部缺失或极淡；布偶的配色更接近 900 行（白红裙、棕发）。没有出现 900 行那种双眼大小/位置不对称与面部空白。[单图](../../../experiment/task10/comparison-570-20261118-a/images-570/r00_c570.png)
- **r01（粉色双人，seed 42，无词）。** 两张脸完整（眼、鼻、嘴、对称），暖粉配色与 Base/285 接近，布偶是金发条纹上衣；这一行没有任何退化迹象。[单图](../../../experiment/task10/comparison-570-20261118-a/images-570/r01_c570.png)
- **r06（紫发星空，seed 43，有词）。** 紫发、金瞳、蓝白水手领、大蝴蝶结与金色扣件、环形行星都在；头后出现一条与 900 行相同的细圆环（提示未要求）；一只手抬到耳侧，手指可读但偏软。[单图](../../../experiment/task10/comparison-570-20261118-a/images-570/r06_c570.png)
- **r08（海边夕阳，seed 42，有词）。** 双人、栏杆、夕阳海面与仙女棒都在；**右侧人物没有画出五官**（面部空白），左侧人物面部正常。900 行同一位置是把脸转开，570 是直接留空 —— 这里退化已经出现。[单图](../../../experiment/task10/comparison-570-20261118-a/images-570/r08_c570.png)
- **r12（图书馆，seed 42，有词）。** **左侧人物整张脸没有眼睛**（只有一条鼻线与嘴），**左手摊在书页上成一整片手套状**；右侧人物伏案托腮，手指与脸颊黏连。900 行同一行是眼睛闭合、鼻线拉长，570 的"无眼空白"更极端。[单图](../../../experiment/task10/comparison-570-20261118-a/images-570/r12_c570.png)
- **r14（图书馆，seed 43，有词）。** 两张脸都有眼睛、鼻子与嘴，左侧人物按在书页上的手指可数，比 900 行的"连片手套"好；颈部略长、眼睛略窄。[单图](../../../experiment/task10/comparison-570-20261118-a/images-570/r14_c570.png)
- **r16（雨夜街头，seed 42，有词）。** 人物改成**完全背身**，只看到后脑与侧肩，脸不可见；风衣、腰带、折叠伞、湿地霓虹反光与远处持伞行人都还在。900 行是侧脸半遮，570 的"转脸"更彻底。[单图](../../../experiment/task10/comparison-570-20261118-a/images-570/r16_c570.png)
- **r20（温室花园，seed 42，有词）。** 两人的动作、长凳、蕨类、水壶与陶盆都在；右侧人物面部完整，左侧低头只看到侧后脑；手部偏软但可读，没有多余肢体。[单图](../../../experiment/task10/comparison-570-20261118-a/images-570/r20_c570.png)

**570 在轨迹上的位置。** 按 Base 相对位移，570 整体在 285 与 900 之间（均值 31.44，5/6 组单调）；按退化形态，它已经**部分到达 900 的终点**（r08、r12 的空白脸，r16 的完全背身），但在另外一些行仍是 285 的水平（r01、r14、r20）。也就是说：这一轮看到的是"风格漂移逐行不均匀推进"，而不是"整体按比例变差"。像素差不能用来排序质量 —— r12 的 Base→570 位移（29.93）甚至小于 Base→285（31.20），但肉眼上 r12 的 570 退化更明显。

## 实现发现：sample.py 的适配器路径不等价（未改源码）

在改用单列 driver 之前，先按原计划用 `scripts/sample/sample.py` 做过一次 canary：Base 路径 2/2 位级相同，但**传入 `ADAPTERS` 的路径 2/2 都不相同**：

| canary | 行 | RGB MAE | 最大通道差 | 变化像素 |
|---|---|---:|---:|---:|
| canary-base | r00 | 0.0000 | 0 | 0 |
| canary-base | r14 | 0.0000 | 0 | 0 |
| canary-285 | r00 | 3.2312 | 190 | 214204 |
| canary-285 | r14 | 4.5196 | 206 | 251437 |

原因定位到挂载顺序：`compare_grid` 是"先建管线（不带适配器）→ 渲染 Base → `adapter_api.load` → `backbone.bind_ops()` → 生成"，而 `sample.py` 把 `ADAPTERS` 交给 `from_pretrained`，由 `inference/loading.py` 的 `attach_after_plan` 在已放置的 backbone 上挂载、**挂载后不再 `bind_ops()`**。两者都用了同样的 `row_scales` 与 bypass 模式，因此差异落在挂载/重绑次序上。本轮按约束不改源码，改为单列 driver；建议后续在 trainer 侧单独核实 `attach_after_plan` 是否应在挂载后重绑，再决定 sample.py 的适配器输出是否可用。证据保留在 [canary-mismatch.json](../../../experiment/task10/comparison-570-20261118-a/canary-mismatch.json) 与 `images-canary-base/`、`images-canary-285/`（未被删除）。

## 生成与审计记录

- **prepare（CPU，`CUDA_VISIBLE_DEVICES=-1`）** exit 0，启动器 12.13 s，worker 10.05 s，峰值工作集 523 MiB，CUDA 未初始化。断言：两份单列配置的 24 条 prompt/seed/sizes 与上一轮 `resolved.json`/`metadata.json` 逐字符一致、共享采样配方逐项相同、canary 行号 [0,14] 与 570 行号 0..23、570 导出 sha256 = `0ff41b5d…e884bc`（等于 session `final-audit.json` 的 `checkpoints[step_00000570].files` 记录）、285/900 导出哈希未变、72 张归档图块哈希只读复核、三份基础权重与 trainer 源码变更集。
- **canary（GPU）** exit 0，启动器 183.90 s，driver 渲染 164.56 s；CUDA 峰值 allocated 2,120,034,816 B（1.97 GiB）、reserved 2,181,038,080 B（2.03 GiB）。
- **generate（GPU，570 列）** worker `exit_code 0`，24 张；启动器到 worker 完成共 1,562.74 s（26.0 分钟），其中 driver 计时 1,546.06 s；CUDA 峰值 allocated 2,120,034,816 B（1.97 GiB）、reserved 2,212,495,360 B（2.06 GiB）；管线自报峰值 1.52 GiB allocated。**异常**：这一段的启动器进程在 spawn worker 之后、写 `launch.json` 时因 `CONFIGS['generate']` 键名错误崩溃，因此 `stages/single-generate/` 只有 `start.json`、`stdout.log`、`launcher-failure.json` 与 worker 自己写的 `result.json`，**没有 `exit.json`**。worker 与 driver 未受影响（`result.json` 记录 `exit_code 0`、24 个文件名），镜像文件完整；`run_single.py` 的键名错误已修，`freeze-570.json` 的 `amendments` 记录了 `run_single.py` 的旧/新 sha256（`b08b00c6…` → `7a02efce…`）与原因。
- **audit（CPU）** exit 0，启动器 21.71 s，worker 18.39 s，峰值工作集 964 MiB，CUDA 未初始化。核验：canary 2/2、570 的 24 张 RGB/PNG 双层哈希、24 组六向像素差、归档 Base 24/24 与 285/900 24/24 图块未变、四列合成与盲评材料、`audit-hashes/rgb-sha256.json` 隔离写入。

## 产物

- 配置与脚本：[single-570.py](../../../experiment/task10/comparison-570-20261118-a/single-570.py)、[single-canary-285.py](../../../experiment/task10/comparison-570-20261118-a/single-canary-285.py)、[render_single.py](../../../experiment/task10/comparison-570-20261118-a/render_single.py)、[run_single.py](../../../experiment/task10/comparison-570-20261118-a/run_single.py)
- 解析与冻结：[resolved-570.json](../../../experiment/task10/comparison-570-20261118-a/resolved-570.json)、[freeze-570.json](../../../experiment/task10/comparison-570-20261118-a/freeze-570.json)（含 11 个文件哈希、3 个模型、3 份导出、`amendments`）
- 图像：`images-570/`（24 张，`rXX_c570.png`）、`images-canary-285-single/`（2 张，`r00_c285.png`／`r14_c285.png`）、`images-canary-base/` 与 `images-canary-285/`（失败方案的 canary 证据）
- 四列材料：[quad-grid.png](../../../experiment/task10/comparison-570-20261118-a/quad-grid.png)（2463×12324）、[quad-contact-sheet.png](../../../experiment/task10/comparison-570-20261118-a/quad-contact-sheet.png)（2004×3028）、[pixel-audit-570.json](../../../experiment/task10/comparison-570-20261118-a/pixel-audit-570.json)
- 盲评材料：[blind-quad-grid.png](../../../experiment/task10/comparison-570-20261118-a/blind-quad-grid.png)、[blind-quad-key.json](../../../experiment/task10/comparison-570-20261118-a/blind-quad-key.json)（密封）、[blind-quad-sheet.md](../../../experiment/task10/comparison-570-20261118-a/blind-quad-sheet.md)（24 行 × 16 栏）；逐列哈希单独放在 [audit-hashes/rgb-sha256.json](../../../experiment/task10/comparison-570-20261118-a/audit-hashes/rgb-sha256.json)，**评分完成前不要打开**
- 阶段证据：`stages/single-prepare/`、`stages/single-canary/`、`stages/single-generate/`、`stages/single-audit/`

## 限制

1. 全部 285 张训练图都参加了训练，本轮没有独立图像留出集；新增的三组提示只做定性组合观察，不能据此声称严格泛化或触发词因果效果。
2. RGB MAE 只度量同种子输出的偏移幅度，不影响（也不代表）画面质量；本节已给出一例反证（r12 的 570 位移更小但退化更明显）。
3. 570 步相当于约 2.0 个数据轮次（285 图 × 2 轮），与 900 步的约 3.16 轮不同；轨迹比较是**同一配方下的步数比较**，不是收敛性结论。
4. 视觉结论只覆盖 8 张原尺寸单图 + 六组总览；其余 16 行未逐张放大，"退化行清单"可能不完整。
5. 单列 driver 与 compare_grid 的等价性只由 2 张 canary（step 285、r00/r14）证明；其余行沿用同一条代码路径，未逐行重放验证。
