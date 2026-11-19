# LoRI 数据准备报告（Task 6 子任务）

- 生成时间（UTC）：2026-11-17T14:08:01.571426+00:00
- 数据源：D:\Develop\nigenmame-LoRA\dataset-raw\round2-caption（只读）
- 源清单：285 张图像、285 个同名 TXT；PNG 279、JPG 6。
- 方向分组：portrait 273、landscape 12；没有正方形图像。
- 选择规则：按文件名升序分别分组，包含首尾端点均匀取样；portrait 索引（0 起）0, 30, 60, 91, 121, 151, 181, 212, 242, 272；landscape 索引 0, 11；合计 12 张（10 portrait + 2 landscape）。
- JSONL 使用当前 JsonlSource 支持的 image 字段，值为原图绝对路径；caption_nl 保存对应 UTF-8 TXT 全文，不裁剪、不改写。逐行解析回读、绝对路径和原文 round-trip 校验通过。
- 已输出所选 12 对图像/Caption 的 SHA-256、尺寸和文件大小；完整源目录文件名/大小/最后写入时间快照在 source-snapshot.json。读取前后 570 个源文件元数据一致，所选源文件复算哈希一致。
- 源图和源 TXT 均未复制、移动或写入；所有数据产物位于外层 workspace experiment/data/round2-caption-12/，在内层 KohakuMossTrainer git 根目录之外。
- 训练提示词来源建议：训练时使用 JSONL 中原始 caption_nl；设计段注明 {template:nl}、1 个变体、text drop 0、不开 flip、center resize。这里只记录建议，没有改训练配置。
- 对照提示词建议见 comparison-prompts.json：4 组非显式描述，每组匹配带/不带 nigenmame 的提示词，建议种子 42/43；后续比较保持 sampler、尺寸、CFG、shift 等设置一致。
- 缓存路径未创建或设置；开始缓存前须由 Sol 将 CACHE_DIR 设到 workspace 内目录。未启动训练，也未修改训练器代码、参数、配方或源 venv。
- 选中 Caption 的保守词项筛查有 0 条潜在显式/敏感信号（仅用于提示，不构成内容安全判断）；正文不复述相关内容。后续出图可使用上述独立非显式提示词。

## 产物

- JSONL：D:\Develop\KohakuMossTrainer-LoRI\experiment\data\round2-caption-12\dataset.jsonl
- 清单与哈希：D:\Develop\KohakuMossTrainer-LoRI\experiment\data\round2-caption-12\inventory.json
- 源文件只读快照：D:\Develop\KohakuMossTrainer-LoRI\experiment\data\round2-caption-12\source-snapshot.json
- 对照提示词：D:\Develop\KohakuMossTrainer-LoRI\experiment\data\round2-caption-12\comparison-prompts.json
- 报告：D:\Develop\KohakuMossTrainer-LoRI\docs\execution\dit-lori\data-preparation-report.md
