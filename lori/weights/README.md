# weights/ 清单

本目录是 5 组 LoRI 训练导出（每组 4 个检查点 × lycoris/comfy），全部为本地副本，按运行和步分组。共 40 个文件、2,464,874,480 字节（2464.9 MB）。

## 入库状态

- 当前 lori/.gitignore 忽略 weights/ 与 images/，因此本目录（含本文件）**尚未入库**；是否提交、用不用 LFS 由用户决定。
- 上一轮提交 3d52f66 曾提交过 8 个稠密权重（旧平铺命名 lori/weights/lori-dense-stepNNN-*.safetensors）。本次重排后这些旧路径已删除，其内容与 dense-256-task10/step_*/lori-dense-*.safetensors 逐字节相同（sha256 一致）；如要恢复旧文件名可参照该提交。
- 仓库当前只提交了 lori/reports/ 与 lori/evidence/（见仓库根 README 与 lori/README.md）。

## 分组汇总

| 运行 | 文件数 | 合计字节 | 合计 MB | 训练说明 |
| --- | --- | --- | --- | --- |
| dense-256-task10 | 8 | 495,993,888 | 496.0 | task10 全量 285 图稠密 B，900 步 |
| sparse30-256-task11 | 8 | 684,211,600 | 684.2 | task11 稀疏 30%（保留 70%）B，900 步 |
| sparse60-256-task11 | 8 | 530,617,312 | 530.6 | task11 稀疏 60%（保留 40%）B，900 步 |
| sparse90-256-task11 | 8 | 377,024,304 | 377.0 | task11 稀疏 90%（保留 10%）B，900 步 |
| sparse90-native-task13 | 8 | 377,027,376 | 377.0 | task13 原生精度（非 INT8）稀疏 90% B，900 步 |

合计 40 个文件、2,464,874,480 字节（2464.9 MB）。

## 超过 GitHub 100 MiB 单文件上限的文件

GitHub 对单个文件的上限是 100 MiB（104,857,600 字节）。以下 4 个文件超过该上限，**无法直接 push**：

| 运行 | 步 | 文件 | 字节 | MB | sha256 |
| --- | --- | --- | --- | --- | --- |
| sparse30-256-task11 | step_00000285 | lori-sparse30-lycoris.safetensors | 121,451,580 | 121.5 | 8ac750ba42ddf0b6 |
| sparse30-256-task11 | step_00000570 | lori-sparse30-lycoris.safetensors | 121,451,580 | 121.5 | 58a9b60945809708 |
| sparse30-256-task11 | step_00000855 | lori-sparse30-lycoris.safetensors | 121,451,580 | 121.5 | 46aee2e5d72a50ac |
| sparse30-256-task11 | step_00000900 | lori-sparse30-lycoris.safetensors | 121,451,580 | 121.5 | 78da1a49eb48c556 |

处理方式只能是三选一：(1) 使用 Git LFS；(2) 这些文件不提交，只保留在本地（当前默认）；(3) 由用户决定是否拆分或重新导出更小的格式。本轮不启用 LFS。

## 逐文件清单

路径形式为 weights/<运行>/<步>/<文件名>；sha256 只列前 16 位（完整值见下方"校验说明"）。字节数为实际文件大小，MB 按 1,000,000 字节换算。

| 运行 | 步 | 格式 | 字节 | MB | >=100 MiB | sha256 前 16 | 来源（相对工作区，export/ 下同名文件） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense-256-task10 | step_00000285 | comfy | 49,601,320 | 49.6 | 否 | 81d4797d5e7e709c | experiment/task10/session-20261118-a/training/lori-dense/checkpoints/step_00000285/export/lori-dense-comfy.safetensors |
| dense-256-task10 | step_00000285 | lycoris | 74,397,152 | 74.4 | 否 | 520ac04443e14918 | experiment/task10/session-20261118-a/training/lori-dense/checkpoints/step_00000285/export/lori-dense-lycoris.safetensors |
| dense-256-task10 | step_00000570 | comfy | 49,601,320 | 49.6 | 否 | d0e9949354da1c9a | experiment/task10/session-20261118-a/training/lori-dense/checkpoints/step_00000570/export/lori-dense-comfy.safetensors |
| dense-256-task10 | step_00000570 | lycoris | 74,397,152 | 74.4 | 否 | 0ff41b5da4a436d7 | experiment/task10/session-20261118-a/training/lori-dense/checkpoints/step_00000570/export/lori-dense-lycoris.safetensors |
| dense-256-task10 | step_00000855 | comfy | 49,601,320 | 49.6 | 否 | 92a9533d8e6c6021 | experiment/task10/session-20261118-a/training/lori-dense/checkpoints/step_00000855/export/lori-dense-comfy.safetensors |
| dense-256-task10 | step_00000855 | lycoris | 74,397,152 | 74.4 | 否 | 922ad3a662c41432 | experiment/task10/session-20261118-a/training/lori-dense/checkpoints/step_00000855/export/lori-dense-lycoris.safetensors |
| dense-256-task10 | step_00000900 | comfy | 49,601,320 | 49.6 | 否 | c8e45d41fee8db1d | experiment/task10/session-20261118-a/training/lori-dense/checkpoints/step_00000900/export/lori-dense-comfy.safetensors |
| dense-256-task10 | step_00000900 | lycoris | 74,397,152 | 74.4 | 否 | c4309ca6a4e57c49 | experiment/task10/session-20261118-a/training/lori-dense/checkpoints/step_00000900/export/lori-dense-lycoris.safetensors |
| sparse30-256-task11 | step_00000285 | comfy | 49,601,320 | 49.6 | 否 | 1bf4eac9c629e2cd | experiment/task11/session-20261119-a/runs/lori-sparse30/checkpoints/step_00000285/export/lori-sparse30-comfy.safetensors |
| sparse30-256-task11 | step_00000285 | lycoris | 121,451,580 | 121.5 | 是 | 8ac750ba42ddf0b6 | experiment/task11/session-20261119-a/runs/lori-sparse30/checkpoints/step_00000285/export/lori-sparse30-lycoris.safetensors |
| sparse30-256-task11 | step_00000570 | comfy | 49,601,320 | 49.6 | 否 | 3f4ac2ac00833a32 | experiment/task11/session-20261119-a/runs/lori-sparse30/checkpoints/step_00000570/export/lori-sparse30-comfy.safetensors |
| sparse30-256-task11 | step_00000570 | lycoris | 121,451,580 | 121.5 | 是 | 58a9b60945809708 | experiment/task11/session-20261119-a/runs/lori-sparse30/checkpoints/step_00000570/export/lori-sparse30-lycoris.safetensors |
| sparse30-256-task11 | step_00000855 | comfy | 49,601,320 | 49.6 | 否 | 57f32b8cecd6f503 | experiment/task11/session-20261119-a/runs/lori-sparse30/checkpoints/step_00000855/export/lori-sparse30-comfy.safetensors |
| sparse30-256-task11 | step_00000855 | lycoris | 121,451,580 | 121.5 | 是 | 46aee2e5d72a50ac | experiment/task11/session-20261119-a/runs/lori-sparse30/checkpoints/step_00000855/export/lori-sparse30-lycoris.safetensors |
| sparse30-256-task11 | step_00000900 | comfy | 49,601,320 | 49.6 | 否 | be86e1f2f4c9736b | experiment/task11/session-20261119-a/runs/lori-sparse30/checkpoints/step_00000900/export/lori-sparse30-comfy.safetensors |
| sparse30-256-task11 | step_00000900 | lycoris | 121,451,580 | 121.5 | 是 | 78da1a49eb48c556 | experiment/task11/session-20261119-a/runs/lori-sparse30/checkpoints/step_00000900/export/lori-sparse30-lycoris.safetensors |
| sparse60-256-task11 | step_00000285 | comfy | 49,601,320 | 49.6 | 否 | 61a12254517c3ecd | experiment/task11/session-20261119-a/runs/lori-sparse60/checkpoints/step_00000285/export/lori-sparse60-comfy.safetensors |
| sparse60-256-task11 | step_00000285 | lycoris | 83,053,008 | 83.1 | 否 | 1ad98288bcc6a3d0 | experiment/task11/session-20261119-a/runs/lori-sparse60/checkpoints/step_00000285/export/lori-sparse60-lycoris.safetensors |
| sparse60-256-task11 | step_00000570 | comfy | 49,601,320 | 49.6 | 否 | 2dd4e5f724900796 | experiment/task11/session-20261119-a/runs/lori-sparse60/checkpoints/step_00000570/export/lori-sparse60-comfy.safetensors |
| sparse60-256-task11 | step_00000570 | lycoris | 83,053,008 | 83.1 | 否 | 627c900cb8e5f544 | experiment/task11/session-20261119-a/runs/lori-sparse60/checkpoints/step_00000570/export/lori-sparse60-lycoris.safetensors |
| sparse60-256-task11 | step_00000855 | comfy | 49,601,320 | 49.6 | 否 | c51fe3a2ea95e511 | experiment/task11/session-20261119-a/runs/lori-sparse60/checkpoints/step_00000855/export/lori-sparse60-comfy.safetensors |
| sparse60-256-task11 | step_00000855 | lycoris | 83,053,008 | 83.1 | 否 | 6a95e8687fe9da7c | experiment/task11/session-20261119-a/runs/lori-sparse60/checkpoints/step_00000855/export/lori-sparse60-lycoris.safetensors |
| sparse60-256-task11 | step_00000900 | comfy | 49,601,320 | 49.6 | 否 | 2e7fc6bee8a8281d | experiment/task11/session-20261119-a/runs/lori-sparse60/checkpoints/step_00000900/export/lori-sparse60-comfy.safetensors |
| sparse60-256-task11 | step_00000900 | lycoris | 83,053,008 | 83.1 | 否 | 7c4f7223082d8a33 | experiment/task11/session-20261119-a/runs/lori-sparse60/checkpoints/step_00000900/export/lori-sparse60-lycoris.safetensors |
| sparse90-256-task11 | step_00000285 | comfy | 49,601,320 | 49.6 | 否 | b6219bd39f9b0dce | experiment/task11/session-20261119-a/runs/lori-sparse90/checkpoints/step_00000285/export/lori-sparse90-comfy.safetensors |
| sparse90-256-task11 | step_00000285 | lycoris | 44,654,756 | 44.7 | 否 | 2318a8557e7e2f4f | experiment/task11/session-20261119-a/runs/lori-sparse90/checkpoints/step_00000285/export/lori-sparse90-lycoris.safetensors |
| sparse90-256-task11 | step_00000570 | comfy | 49,601,320 | 49.6 | 否 | fe5f7cb837728fe1 | experiment/task11/session-20261119-a/runs/lori-sparse90/checkpoints/step_00000570/export/lori-sparse90-comfy.safetensors |
| sparse90-256-task11 | step_00000570 | lycoris | 44,654,756 | 44.7 | 否 | 01a1d40f2187484c | experiment/task11/session-20261119-a/runs/lori-sparse90/checkpoints/step_00000570/export/lori-sparse90-lycoris.safetensors |
| sparse90-256-task11 | step_00000855 | comfy | 49,601,320 | 49.6 | 否 | 24e240923818c504 | experiment/task11/session-20261119-a/runs/lori-sparse90/checkpoints/step_00000855/export/lori-sparse90-comfy.safetensors |
| sparse90-256-task11 | step_00000855 | lycoris | 44,654,756 | 44.7 | 否 | c1f35ffdca3bd0b5 | experiment/task11/session-20261119-a/runs/lori-sparse90/checkpoints/step_00000855/export/lori-sparse90-lycoris.safetensors |
| sparse90-256-task11 | step_00000900 | comfy | 49,601,320 | 49.6 | 否 | ce9ba0ffc798af02 | experiment/task11/session-20261119-a/runs/lori-sparse90/checkpoints/step_00000900/export/lori-sparse90-comfy.safetensors |
| sparse90-256-task11 | step_00000900 | lycoris | 44,654,756 | 44.7 | 否 | fcfef742c16e19db | experiment/task11/session-20261119-a/runs/lori-sparse90/checkpoints/step_00000900/export/lori-sparse90-lycoris.safetensors |
| sparse90-native-task13 | step_00000250 | comfy | 49,601,744 | 49.6 | 否 | f67285d3e487fce6 | experiment/task13/session-20261119-a/runs/lori-sparse90-native/checkpoints/step_00000250/export/lori-sparse90-native-comfy.safetensors |
| sparse90-native-task13 | step_00000250 | lycoris | 44,655,100 | 44.7 | 否 | 900fcad9a14a9ff5 | experiment/task13/session-20261119-a/runs/lori-sparse90-native/checkpoints/step_00000250/export/lori-sparse90-native-lycoris.safetensors |
| sparse90-native-task13 | step_00000550 | comfy | 49,601,744 | 49.6 | 否 | 961d02d82715ef99 | experiment/task13/session-20261119-a/runs/lori-sparse90-native/checkpoints/step_00000550/export/lori-sparse90-native-comfy.safetensors |
| sparse90-native-task13 | step_00000550 | lycoris | 44,655,100 | 44.7 | 否 | ae04510312b23be3 | experiment/task13/session-20261119-a/runs/lori-sparse90-native/checkpoints/step_00000550/export/lori-sparse90-native-lycoris.safetensors |
| sparse90-native-task13 | step_00000850 | comfy | 49,601,744 | 49.6 | 否 | f509acb0c5397b82 | experiment/task13/session-20261119-a/runs/lori-sparse90-native/checkpoints/step_00000850/export/lori-sparse90-native-comfy.safetensors |
| sparse90-native-task13 | step_00000850 | lycoris | 44,655,100 | 44.7 | 否 | 7a4657284c8e7dc3 | experiment/task13/session-20261119-a/runs/lori-sparse90-native/checkpoints/step_00000850/export/lori-sparse90-native-lycoris.safetensors |
| sparse90-native-task13 | step_00000900 | comfy | 49,601,744 | 49.6 | 否 | 307f43baa3a2a877 | experiment/task13/session-20261119-a/runs/lori-sparse90-native/checkpoints/step_00000900/export/lori-sparse90-native-comfy.safetensors |
| sparse90-native-task13 | step_00000900 | lycoris | 44,655,100 | 44.7 | 否 | 30d87ca202c94b32 | experiment/task13/session-20261119-a/runs/lori-sparse90-native/checkpoints/step_00000900/export/lori-sparse90-native-lycoris.safetensors |

## 校验说明

- 每个文件都从对应 run 的 checkpoints/<步>/export/ 复制而来（copy，非移动），复制后逐文件比较 sha256 与源文件一致。
- 需要完整 sha256 时可在工作区运行：Get-FileHash -Algorithm SHA256 <路径>；或对照 experiment/ 下对应源文件。
- lycoris 文件是原生 LyCORIS 状态（.A / .alpha / .b_dense，256² 任务为 756 张量 / 252 层），comfy 文件是 ComfyUI 兼容导出（lora_down.weight / lora_up.weight / alpha，504 张量，q/k/v 已融合）。
