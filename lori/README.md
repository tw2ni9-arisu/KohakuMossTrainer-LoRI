# LoRI on MOSS-Image DiT · 报告、证据与权重

本目录汇总 MOSS-Image DiT 上 LoRI（固定随机投影 A + 稠密或稀疏 B）实验的归档：阶段报告、审计 JSON、对照图与训练导出。训练器实现即本仓库（KohakuMossTrainer，本副本 HEAD cde1a45a32b266c1dae9e87a2d0eba61c1253c75，分支 lori-development）；实验在单张 12GB 消费级显卡与 64GB 主机内存的 Windows 机器上完成，数据为私有的 285 对图文数据集（不随仓库分发）。

## 目录结构

    lori/
      README.md                 本索引
      .gitignore                忽略 images/ 与 weights/
      reports/                  54 篇报告，按实验主题分组
        implementation/         task-1..7 的过程报告与评审，以及编译器诊断、数据视觉复核、交接
        dense-256/              稠密 B 的 232 步与全量 900 步训练/对照报告
        sparsity-256/           稀疏 30/60/90 的续训、对照与稀疏率比较
        native-resolution/      原生精度（非 INT8）稀疏与对照
        audits/                 终审、冻结审计、导出审计、独立评审
        research/               4 篇调查与论文复核
        plan/                   方案、设计、实施计划与进度
      evidence/                 18 个审计 JSON，同样按主题分组
        dense-256/ dense-256-570/ sparsity-256/ native-resolution/ audits/
      images/                   14 张对照图（本地生成，不入库）
        dense-256/ dense-256-570/ sparsity-256/ native-resolution/
      weights/                  40 个训练导出（5 组 run × 4 步 × lycoris/comfy，本地副本，不入库）
        dense-256-task10/ sparse90-256-task11/ sparse60-256-task11/ sparse30-256-task11/ sparse90-native-task13/
          step_XXXXXX/          每步 2 个文件：*-lycoris.safetensors 与 *-comfy.safetensors

## 入库状态

- reports/（54 篇）与 evidence/（18 个）是小文本文件，可入库，已在提交 3d52f66 中入库。
- images/（约 191 MB）与 weights/（约 2.46 GB）是本地大二进制，由 lori/.gitignore 忽略，待用户决定入库范围与方式；weights 里 4 个 sparse30 的 lycoris 文件为 121.45 MB，超过 GitHub 单文件 100 MiB 上限，不能直接 push（详见 lori/weights/README.md）。
- 提交 3d52f66 里另有 8 个旧平铺命名的稠密权重（lori/weights/lori-dense-stepNNN-*.safetensors）；本次重排后这些路径已删除，内容已由 weights/dense-256-task10/step_*/ 下的同名文件承载（逐字节相同）。

## 报告内链接的说明（重要）

本目录是归档副本。报告正文中的相对链接（例如 ../../../experiment/... 或 ../../../docs/...）是按原来的 docs/execution/dit-lori/ 位置写的，复制到 lori/reports/<分组>/ 之后这些相对路径会失效，仓库内也可能没有对应文件。**报告内容一律未修改**；需要核对原始链接时，请以工作区为准确认下表给出的原始路径。审计 JSON 同理。

## reports/ 分组与原始路径

来源目录：工作区 docs/execution/dit-lori/（除 research/ 与 plan/ 两组另有来源）。

### reports/implementation/（29 篇）

| 文件 | 原始路径 |
| --- | --- |
| compiler-diagnosis-report.md | docs/execution/dit-lori/compiler-diagnosis-report.md |
| data-preparation-report.md | docs/execution/dit-lori/data-preparation-report.md |
| data-visual-review.md | docs/execution/dit-lori/data-visual-review.md |
| task-1-brief.md | docs/execution/dit-lori/task-1-brief.md |
| task-1-report.md | docs/execution/dit-lori/task-1-report.md |
| task-1-review.md | docs/execution/dit-lori/task-1-review.md |
| task-2-brief.md | docs/execution/dit-lori/task-2-brief.md |
| task-2-report.md | docs/execution/dit-lori/task-2-report.md |
| task-2-review.md | docs/execution/dit-lori/task-2-review.md |
| task-3-brief.md | docs/execution/dit-lori/task-3-brief.md |
| task-3-report.md | docs/execution/dit-lori/task-3-report.md |
| task-3-review.md | docs/execution/dit-lori/task-3-review.md |
| task-4-brief.md | docs/execution/dit-lori/task-4-brief.md |
| task-4-fix1-review.md | docs/execution/dit-lori/task-4-fix1-review.md |
| task-4-report.md | docs/execution/dit-lori/task-4-report.md |
| task-4-review.md | docs/execution/dit-lori/task-4-review.md |
| task-5-brief.md | docs/execution/dit-lori/task-5-brief.md |
| task-5-report.md | docs/execution/dit-lori/task-5-report.md |
| task-5-review.md | docs/execution/dit-lori/task-5-review.md |
| task-6-brief.md | docs/execution/dit-lori/task-6-brief.md |
| task-6-cache-report.md | docs/execution/dit-lori/task-6-cache-report.md |
| task-6-calibration-report.md | docs/execution/dit-lori/task-6-calibration-report.md |
| task-6-mask-report.md | docs/execution/dit-lori/task-6-mask-report.md |
| task-6-preparation-report.md | docs/execution/dit-lori/task-6-preparation-report.md |
| task-6-review.md | docs/execution/dit-lori/task-6-review.md |
| task-6-sparse-report.md | docs/execution/dit-lori/task-6-sparse-report.md |
| task-7-brief.md | docs/execution/dit-lori/task-7-brief.md |
| task-7-preparation-report.md | docs/execution/dit-lori/task-7-preparation-report.md |
| training-handoff.md | docs/execution/dit-lori/training-handoff.md |

### reports/dense-256/（4 篇）

| 文件 | 原始路径 |
| --- | --- |
| task-10-data-preparation-report.md | docs/execution/dit-lori/task-10-data-preparation-report.md |
| task-10-dense-training-report.md | docs/execution/dit-lori/task-10-dense-training-report.md |
| task-9-dense-comparison-report.md | docs/execution/dit-lori/task-9-dense-comparison-report.md |
| task-9-dense-training-report.md | docs/execution/dit-lori/task-9-dense-training-report.md |

### reports/sparsity-256/（6 篇）

| 文件 | 原始路径 |
| --- | --- |
| task-10-570-comparison-report.md | docs/execution/dit-lori/task-10-570-comparison-report.md |
| task-10-comparison-report.md | docs/execution/dit-lori/task-10-comparison-report.md |
| task-11-sparse-training-report.md | docs/execution/dit-lori/task-11-sparse-training-report.md |
| task-11-sparsity-comparison-report.md | docs/execution/dit-lori/task-11-sparsity-comparison-report.md |
| task-8-comparison-report.md | docs/execution/dit-lori/task-8-comparison-report.md |
| task-8-resume-report.md | docs/execution/dit-lori/task-8-resume-report.md |

### reports/native-resolution/（2 篇）

| 文件 | 原始路径 |
| --- | --- |
| task-13-native-comparison-report.md | docs/execution/dit-lori/task-13-native-comparison-report.md |
| task-13-native-sparse-report.md | docs/execution/dit-lori/task-13-native-sparse-report.md |

### reports/audits/（4 篇）

| 文件 | 原始路径 |
| --- | --- |
| a-frozen-audit.md | docs/execution/dit-lori/a-frozen-audit.md |
| final-review.md | docs/execution/dit-lori/final-review.md |
| lori-export-audit.md | docs/execution/dit-lori/lori-export-audit.md |
| task-10-independent-review.md | docs/execution/dit-lori/task-10-independent-review.md |

### reports/research/（4 篇）

| 文件 | 原始路径 |
| --- | --- |
| 2026-11-17-local-foundation.md | docs/research/2026-11-17-local-foundation.md |
| 2026-11-17-lori-dit-investigation.md | docs/research/2026-11-17-lori-dit-investigation.md |
| 2026-11-18-knit-edit-and-lori-review.md | docs/research/2026-11-18-knit-edit-and-lori-review.md |
| 2026-11-18-lori-experiment-results.md | docs/research/2026-11-18-lori-experiment-results.md |

### reports/plan/（5 篇）

| 文件 | 原始路径 |
| --- | --- |
| 2026-11-17-dit-lori-design.md | docs/superpowers/specs/2026-11-17-dit-lori-design.md |
| 2026-11-17-dit-lori.md | docs/superpowers/plans/2026-11-17-dit-lori.md |
| LoRI阶段成果与900步验证方案.md | （工作区根目录）LoRI阶段成果与900步验证方案.md |
| docs-progress.md | docs/progress.md |
| progress.md | docs/execution/dit-lori/progress.md |

## evidence/ 分组与原始路径

共 18 个审计 JSON，文件名带来源前缀以避免同名冲突。

| 分组 | 文件 | 原始路径 |
| --- | --- | --- |
| dense-256 | task10-final-audit.json | experiment/task10/session-20261118-a/final-audit.json |
| dense-256 | task10-pixel-audit.json | experiment/task10/comparison-20261118-a/pixel-audit.json |
| dense-256 | task9-final-audit.json | experiment/task9/session-20261118-a/final-audit.json |
| dense-256 | task9-matched-training-audit.json | experiment/task9/session-20261118-a/matched-training-audit.json |
| dense-256-570 | task10-570-pixel-audit-570.json | experiment/task10/comparison-570-20261118-a/pixel-audit-570.json |
| sparsity-256 | task11-pixel-audit.json | experiment/task11/comparison-20261119-a/pixel-audit.json |
| sparsity-256 | task11-sparse30-final-audit.json | experiment/task11/session-20261119-a/sparse30-final-audit.json |
| sparsity-256 | task11-sparse60-final-audit.json | experiment/task11/session-20261119-a/sparse60-final-audit.json |
| sparsity-256 | task11-sparse90-final-audit.json | experiment/task11/session-20261119-a/sparse90-final-audit.json |
| native-resolution | task13-pixel-audit.json | experiment/task13/comparison-20261119-a/pixel-audit.json |
| native-resolution | task13-session-final-audit.json | experiment/task13/session-20261119-a/final-audit.json |
| audits | task10-review-phase-a.json | experiment/task10/review-phase-a.json |
| audits | task10-review-phase-b.json | experiment/task10/review-phase-b.json |
| audits | task10-review-phase-c.json | experiment/task10/review-phase-c.json |
| audits | task11-review-phase-d.json | experiment/task11/review-phase-d.json |
| audits | task12-audit-a-frozen-audit.json | experiment/task12-audit/a-frozen-audit.json |
| audits | task13-audit-lori-export-audit.json | experiment/task13-audit/lori-export-audit.json |
| audits | task13-review-phase-e.json | experiment/task13/review-phase-e.json |

说明：experiment/task11/session-20261119-a/ 下没有汇总的 final-audit.json（该 session 按稀疏率分成三次 run），因此收录 sparse30/sparse60/sparse90 三份 per-run 审计作为等价证据。

## images/ 分组与原始路径（本地生成，不入库）

| 分组 | 文件 | 原始路径 |
| --- | --- | --- |
| dense-256 | matched-grid.png | experiment/task10/comparison-20261118-a/matched-grid.png |
| dense-256 | summary-contact-sheet.png | experiment/task10/summary-contact-sheet.png |
| dense-256-570 | quad-contact-sheet.png | experiment/task10/comparison-570-20261118-a/quad-contact-sheet.png |
| dense-256-570 | quad-grid.png | experiment/task10/comparison-570-20261118-a/quad-grid.png |
| sparsity-256 | grid-13col.png | experiment/task11/comparison-20261119-a/visual/grid-13col.png |
| sparsity-256 | sheet-group1.png | experiment/task11/comparison-20261119-a/visual/sheet-group1.png |
| sparsity-256 | sheet-group2.png | experiment/task11/comparison-20261119-a/visual/sheet-group2.png |
| sparsity-256 | sheet-group3.png | experiment/task11/comparison-20261119-a/visual/sheet-group3.png |
| sparsity-256 | sheet-overview.png | experiment/task11/comparison-20261119-a/visual/sheet-overview.png |
| native-resolution | grid-6col.png | experiment/task13/comparison-20261119-a/visual/grid-6col.png |
| native-resolution | sheet-group1.png | experiment/task13/comparison-20261119-a/visual/sheet-group1.png |
| native-resolution | sheet-group2.png | experiment/task13/comparison-20261119-a/visual/sheet-group2.png |
| native-resolution | sheet-group3.png | experiment/task13/comparison-20261119-a/visual/sheet-group3.png |
| native-resolution | sheet-overview.png | experiment/task13/comparison-20261119-a/visual/sheet-overview.png |

## weights/（40 个导出，本地副本，不入库）

| 运行 | 文件数 | 合计 MB | 训练 |
| --- | --- | --- | --- |
| dense-256-task10 | 8 | 496.0 | task10 全量 285 图稠密 B，900 步 |
| sparse30-256-task11 | 8 | 684.2 | task11 稀疏 30%（保留 70%）B，900 步 |
| sparse60-256-task11 | 8 | 530.6 | task11 稀疏 60%（保留 40%）B，900 步 |
| sparse90-256-task11 | 8 | 377.0 | task11 稀疏 90%（保留 10%）B，900 步 |
| sparse90-native-task13 | 8 | 377.0 | task13 原生精度（非 INT8）稀疏 90% B，900 步 |

- 5 组 run 覆盖同一条实验主线：稠密 B（task10）、稀疏 90/60/30（task11）与原生精度稀疏 90（task13）。
- 每组 4 个步点，用于同配方内的进度比较；task10/task11 为 285/570/855/900，task13 为 250/550/850/900。
- lycoris 是原生 LyCORIS 状态（.A / .alpha / .b_dense），comfy 是 ComfyUI 兼容导出（q/k/v 已融合）。
- 逐文件字节数、sha256 前 16 位、来源路径与超过 100 MiB 的清单见 lori/weights/README.md。

## 如何重新生成

1. 数据与缓存：跑 experiment/task10/session-20261118-a/pipeline.py 的六个 stage（resolve → catalog → text → latents → offline → prepare），产出 catalog.db / text.db / latents.db 与 freeze.json。
2. 训练：experiment/task10/session-20261118-a/train_launch.py 启动 train_worker.py，配方在 train.py（900 步、rank 8 / alpha 16、attn-mlp、INT8 + BF16、AdamW 1e-4 恒定、SAVE_EVERY=285、KEEP_LAST=4、FINAL_CHECKPOINT=True）；稀疏与原生精度的 run 见 experiment/task11/session-20261119-a/ 与 experiment/task13/session-20261119-a/ 的 run.py / train-*.py。
3. 终态审计：各 session 的 worker 结束时写 final-audit.json（指标、检查点、优化器、A/alpha 不变与 B 更新、导出可读性、冻结复核）。
4. 权重导出：SAVE_FORMAT=both 且 CHECKPOINT_EXPORT=True，训练器在每个检查点目录的 export/ 下写 lycoris 与 comfy 两份 safetensors；把它们按 weights/<run>/<step>/ 复制归档即可。
5. 对照图：各对比目录下的脚本（例如 experiment/task10/comparison-20261118-a/compare.py、experiment/task13/comparison-20261119-a/run.py 与 audit.py）生成网格与 pixel-audit.json，再按 images/ 的分组复制。
6. 归档：按上文映射把报告与 JSON 复制进 reports/、evidence/，不要就地修改报告正文。

## 边界说明

- 这些是内部实验记录，具体数字以报告引用的审计 JSON 为准。
- 全量 285 图都参与了训练，没有独立留出集：loss 与像素差异只描述训练与输出变化，不能当作泛化或风格质量证据。
- 权重是实验产物，训练数据与原始图像不在本仓库内。
