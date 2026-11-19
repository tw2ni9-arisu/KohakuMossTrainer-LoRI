# LoRI step 32 / 232 同种子对照

原生 `compare_grid` 生成了 4 张新图，覆盖上一轮 r04、r05 的 Base 与 step 232 ×1。使用同一套权重和 512×512、seed 42、Euler 50 步、CFG 3、shift 5、two-pass CFG、batch 1。两行分别保留和移除开头的 `nigenmame,`，其余提示相同。新 Base 的解码 RGB 与归档的 Base 逐像素相等，因此三列图复用了归档的 step 32。 [三列对照图](../../../experiment/task8/session-20261118-a/comparison/matched-grid.png) · [原生元数据](../../../experiment/task8/session-20261118-a/comparison/images/metadata.json)

| 提示 | step 32 距 Base 的平均绝对 RGB 差 | step 232 距 Base 的平均绝对 RGB 差 |
| --- | ---: | ---: |
| r04 有触发词 | 16.66 | 41.05 |
| r05 无触发词 | 11.69 | 28.85 |

差值以 8-bit RGB 的 0–255 计，两行都增至约 2.46 倍。step 232 保留粉色背景、红白双人和中央布偶，两行都重绘了头发、眼形和制服细节。有触发词的一行改为无袖制服、条纹领带和红贝雷帽，布偶失去明显的双手支撑。无触发词的一行也改变了刘海、领口、衣襟扣件与布偶脸部。两行结果都呈现更细的发丝和衣缘线条，但同一场景、单一种子的两行不足以判断这些变化是否来自特定训练风格，或能否稳定依赖触发词。 [像素审计](../../../experiment/task8/session-20261118-a/comparison/pixel-audit.json) · [有词 step 232](../../../experiment/task8/session-20261118-a/comparison/images/r00_c01.png) · [无词 step 232](../../../experiment/task8/session-20261118-a/comparison/images/r01_c01.png)

采样日志记录 Base 两张用时 117.18 秒、step 232 ×1 两张用时 136.86 秒，完整启动器用时 311.80 秒。原生 CLI 成功写出 4 张单图、网格和元数据后按惯例触发 `SystemExit(0)`。私有包装器把它误判为失败，原 worker/launcher 的退出 1 证据已保留。独立 CPU 补验用已有文件验证元数据、归档像素哈希及新图，并生成三列图，退出 0。采样执行了一次。 [原生采样日志](../../../experiment/task8/session-20261118-a/comparison/stages/generate/stdout.log) · [保留的包装器结果](../../../experiment/task8/session-20261118-a/comparison/stages/generate/result.json) · [CPU 补验](../../../experiment/task8/session-20261118-a/comparison/stages/posthoc-audit/result.json)
