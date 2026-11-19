# LoRI 论文与 DiT 移植调查

日期：2026-11-17  
范围：核对 LoRI 原论文和作者公开代码，推导迁移到 KohakuMossTrainer 的接口与首轮验证方案。本调查没有修改训练器代码，也没有运行训练。

## 调查结论

这里的 LoRI 指 Zhang 等人的 **LoRA with Reduced Interference**，论文发表于 COLM 2025。它不是泛指 LoRA-FA、固定 A 的 LoRA 或其它缩写相近的方法。作者代码使用 PEFT 和 FSDP，训练 Llama 3 / Mistral 的语言任务；未提供 MOSS-Image、DiT 或图像扩散训练器。因此，这份调查可用来实现论文算法，但不能作为已有 DiT 效果或低显存效果的证据。[论文](https://arxiv.org/pdf/2504.07448)；[作者仓库](https://github.com/juzhengz/LoRI)

LoRI 的主要贡献是固定随机低秩投影并给每个任务分配稀疏的 B 更新，以减少可训练参数和多任务适配器干扰。论文在语言模型上报告最高 90% 的 B 稀疏率；冻结主干、前向计算以及反向传播仍在，因此稀疏适配器不会自动消除 DiT 主干权重和激活占用。

## 论文算法

### 记号与初始化

论文用行向量约定：

\[
y=xW_0+xA(B\odot M),\qquad
A\in\mathbb{R}^{d_{in}\times r},\quad
B,M\in\mathbb{R}^{r\times d_{out}}.
\]

每个目标线性层独立初始化固定投影 A，使用 Kaiming Uniform：每个元素均匀采样自 \([-\sqrt{3/d_{in}},\sqrt{3/d_{in}}]\)。B 初始为零。训练时冻结 A，只更新 B。对 PyTorch `nn.Linear` 的 `[out, in]` 权重，等价实现的形状是 `A_down[r, in] = A.T`（冻结）和 `B_up[out, r] = B.T`（可训练），适配器增量为 `((B_up * M_up) @ A_down)`。

论文公式没有显式写 LoRA 缩放系数。作者实现通过 PEFT 的 `LoraConfig(r, lora_alpha, ...)` 构造模块，默认 `use_rslora=False`，因此按 PEFT 的约定使用 `lora_alpha / r`。论文实验常见配置为 `r=32, α=64` 或 `r=64, α=128`，对应缩放 2；移植时应明确记录这一系数，避免公式实现和实验配置不一致。[论文算法与超参表](https://arxiv.org/pdf/2504.07448)；[作者训练入口](https://github.com/juzhengz/LoRI/blob/main/src/train_lori.py)；[PEFT LoRA 缩放实现](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)

### 校准、评分与掩码

1. 在校准数据上，用目标任务损失训练 **无掩码的 B**；A 固定。校准样本来自该任务的适配数据。论文消融显示增加校准步数通常有益，实验使用完整适配数据作为校准集。
2. 收集所有目标层、所有目标投影的 B 元素，跨模型合并计算一个全局阈值 \(\tau=Quantile_s(|B|)\)。这里 \(s\) 是要置零的稀疏比例。
3. 每个 B 使用同一个阈值生成掩码：\(M=I(|B|\ge\tau)\)。留下全模型 magnitude 最高的约 \((1-s)\) 比例；这是 **全局** 分配预算，不是每层、每个矩阵或每种 projection 分别选相同比例。
4. 掩码校准完成后，将所有 B **重置为零**，然后在适配数据上重新训练，只允许掩码内的 B 参数更新。校准得到的 B 数值本身不作为 LoRI-S 的起点。

LoRI 的重要性分数是校准后 B 的绝对值，不是梯度分数。论文提及 Fisher 或 SNIP 可以替换为局部梯度敏感性分数，但它们不是论文主算法或实验选择规则。论文认为最终 magnitude 汇集了整个微调过程的重要性。

作者 `extract_mask.py` 将所有 `lora_B` 的绝对值拼接后用 `torch.topk` 找全局第 k 大阈值，再对每个矩阵使用 `abs(B) >= threshold`；这对应论文的全局 top-(1-s)%。由于用阈值加 `>=`，若阈值元素有并列，实际保留数可能略高于 k。DiT 移植可保留论文定义的阈值规则，也可用稳定排序精确取 k；如果更改为精确 k，应记录为实现层的 tie-breaking 规则。[作者掩码提取代码](https://github.com/juzhengz/LoRI/blob/main/src/extract_mask.py)

论文把 \(s=0\) 的单任务极限称为 LoRA-FA：固定 A、稠密训练 B。实际被称为 LoRI-S 的实验配置常用 \(s=0.9\)。需要分别报告 LoRI-D（校准阶段的固定 A 稠密训练）和 LoRI-S（重置 B 后按 mask 稀疏训练），不能把二者混为一个 checkpoint。[论文](https://arxiv.org/pdf/2504.07448)

### A 与优化器状态的生命周期

按论文算法，A 在初始化后保持固定，并贯穿校准与稀疏再训练；掩码来自这组 A 对应的校准 B。生成 mask 后重置的是 B，且正式稀疏训练应有新的优化器状态。

作者示例 shell 脚本将 LoRI-D、mask 提取和 LoRI-S 分成独立命令。LoRI-S 不加载 LoRI-D adapter；Llama 3 配置的 `archive` 也是 `null`，所以新阶段会得到零初始化的 B 和新优化器，这与 B / optimizer reset 相符。不过该流程没有复用 LoRI-D checkpoint 中的 A；训练入口在创建 PEFT adapter 后才设置训练种子，未能确认两个进程会得到相同的 A。因此其公开脚本在 A 是否复用上与论文流程可能不完全一致。新实现应在阶段切换时保存并复用 A（或固定且校验初始化种子），不能依赖两个进程“碰巧”抽到相同随机投影。[作者脚本](https://github.com/juzhengz/LoRI/blob/main/scripts/codealpaca_llama3_r_32.sh)；[Llama 3 配置](https://github.com/juzhengz/LoRI/blob/main/config/model/llama3.yaml)；[作者训练入口](https://github.com/juzhengz/LoRI/blob/main/src/train_lori.py)

## 作者代码与物理稀疏的差距

作者实现冻结 PEFT `lora_A`，并通过 mask 将 `lora_B` 的 inactive 梯度置零，再调用标准 AdamW。参数对象仍是完整稠密 B；mask 也是和 B 同形状的张量。AdamW 的 `exp_avg`、`exp_avg_sq` 等状态仍按完整 B 分配，因而此实现证明的是“只有部分 B 有梯度”，不证明训练态优化器状态随 nnz 缩小。checkpoint 的 B 仍为矩阵形状；单有零元素也不会自动让适配器文件缩到 nnz 大小。[作者 mask trainer](https://github.com/juzhengz/LoRI/blob/main/src/mask_trainers.py)；[作者 masked optimizer](https://github.com/juzhengz/LoRI/blob/main/src/apply_mask_in_backward.py)

若目标包含真实的优化器显存节省，训练参数必须是活跃坐标的紧凑向量，而不应将完整 B 注册为 trainable Parameter 后仅乘 mask：

- 每个目标模块保存冻结 A、固定坐标索引 `active_idx`，以及真正可训练的 `active_values`（长度 nnz）。AdamW 只接收 `active_values`，其梯度与优化器状态也只有 nnz 大小。
- calibration 可暂时用 dense B，以便遵循论文的全局 magnitude 选 mask。阶段结束后从 B 取出活跃坐标和阈值，丢弃 dense B 和 calibration optimizer；新建全零 `active_values`，并复用同一个 A。
- 最简正确性路径可以在 forward 临时 scatter/reconstruct 稠密 B，再复用已有低秩路径。这能压缩可训练参数与 optimizer state，却仍有临时 dense B 和稠密乘法，不能据此宣称计算量同比下降。后续若要减少适配器计算，再为稀疏矩阵乘加实现融合 sparse kernel。
- 存储索引本身也有成本：如每个 active element 用 int32 索引，索引字节可能抵消小型 adapter 的一部分节省。应比较 CSR/分组索引、压缩位图和简单 int32 索引的总 bytes；导出时可将紧凑状态重建成 ComfyUI/LyCORIS 需要的稠密 B，或定义带索引的训练 checkpoint 格式。

## KohakuMossTrainer 接口映射

以下针对只读检查过的 KohakuMossTrainer 源码结构给出插入点；不同副本的实际版本应在实现前复核。

| 责任 | 源码接口 | LoRI 接入要求 |
| --- | --- | --- |
| 用户配置与算法记录 | `src/kohakumoss/adapters/build/config.py`、`src/kohakumoss/adapters/kinds/record.py`、`src/kohakumoss/adapters/kinds/__init__.py` | 增加独立的 LoRI 配置和 `ALGO`/kind 记录，明确 rank、alpha、sparsity、calibration 步数或数据轮数；保持 LoRA 与 LoRI 参数语义可区分。 |
| 创建和挂接模块 | `src/kohakumoss/adapters/build/network.py:build_targets`、`src/kohakumoss/adapters/build/apply.py:apply` | 现有构建路径经 LyCORIS `LycorisNetwork` 创建目标模块，再将模块注册为目标 `MossLinear` 的 child。LoRI 需要可装入该挂接与 block swapping 的模块对象；只在 Kind registry 增名不足以创建新的 LyCORIS module。 |
| 前向、差分与导出 | `src/kohakumoss/adapters/kinds/lowrank.py`、`src/kohakumoss/adapters/bind/`、`src/kohakumoss/adapters/export/` | 以 PyTorch `[out,in]` 权重约定实现 `B_up_sparse @ A_down`；保持现有 bypass/merge 约定。第一阶段可临时重建 dense B 复用低秩 forward；导出时转换为 dense B，或另定义 compact state + indices。 |
| 可训练参数和 checkpoint | `src/kohakumoss/adapters/train/adapter_set.py:LycoAdapterSet.named_trainable`、`param_groups`、`state_dict`、`load_state_dict`、`export` | 只枚举 `active_values` 和任何 LoRI 必需的参数；A 必须被冻结但包含在 adapter state / metadata 中；mask、shape、模块路径、阈值和投影 seed 可恢复。checkpoint 恢复不能漏掉 active index。 |
| 优化器与低显存管线 | `src/kohakumoss/training/build/loop.py`、`src/kohakumoss/training/memory/optimizers.py` | compact `active_values` 经 `named_trainable/param_groups` 进入现有 AdamW/host-state/grad-offload 路径。若只有 dense masked B，host optimizer offload 只改变状态放置位置，不改变其全矩阵大小。 |

首轮建议先不扩展自定义 kernel：用 compact 参数确保优化器 state 真正按 nnz 缩小，dense scatter forward 作为参考实现，并测量它的临时峰值。优化器数据结构必须保留每个 module 独立索引，以便将 mask 和 `active_values` 精确映射回 B。不要把论文 mask 解释为对激活或冻结主干权重的稀疏化。

## 首轮验证方案

数据按用户已定的 **round2-caption 标注图像**，不另找或复制数据。首轮是算法可行性实验，不足以证明 DiT 效果与 LoRA 相当。

1. **数值单测/小前向**：选一个 attention projection 和一个 MLP projection，验证 compact LoRI 的 forward 等于用同一 A、B、mask 显式计算 `B⊙M` 的 dense reference；检查 B 为零时输出与 base 相同、A 无梯度。
2. **全局 mask 检查**：用目标数据跑固定 A 的无 mask 校准，合并所有预定 target modules 的 `abs(B)` 后选一次全局阈值。记录活跃参数总数、各层和各投影分布；验证 mask 不会在每层单独达到 10% 保留率。
3. **阶段边界检查**：校准完成后 B 全零、训练 optimizer 新建，A checksum 与校准阶段一致。跑一个 sparse step：inactive 坐标恒为零、active 坐标可更新，AdamW 参数、梯度及一阶/二阶状态只含 nnz，而非完整 B。
4. **视觉小试验**：固定 MOSS checkpoint、round2-caption 数据拆分、分辨率/桶、caption、噪声与 timestep 策略、target modules、训练步数、seed 和采样参数。比较标准 LoRA 与 LoRI：先试 `r=32, α=64`、LoRI `s=0.9`；如明显欠拟合，再加 `s=0.5` 作为容量诊断。LoRI 先用一个小的训练子集或 1 个 epoch 校准，后续验证再改用完整训练集，因为论文报告 mask 质量通常随校准步数增加。
5. **同时报告质量和内存**：固定 prompt/seed 做基座、LoRA、LoRI 的并排采样；记录 train/validation loss、峰值 allocated/reserved VRAM、主机 RAM、trainable bytes、optimizer-state bytes、compact checkpoint bytes、每步用时。一次小试结果仅用于发现接口和明显容量问题；可比质量结论至少要有重复 seed 和更大的 holdout。

如果目标是先证明论文算法能学习图像概念，比较 LoRA 与 LoRI 质量；如果目标是先证明 12 GB 低显存能力，则另测训练峰值和 offload，不应将稀疏比例直接等同于总显存节省。基础 DiT 权重、激活和编码器通常不会随 B 的 nnz 一起缩小。

## 风险与待验证边界

- 论文只在 Llama 3 / Mistral 语言任务评测。作者提到向 diffusion / vision-language 扩展是 future work；MOSS-Image 的图像风格、角色一致性、提示遵循效果没有论文证据。
- 论文的随机子空间正交性分析依赖独立随机 A、低秩相对输入维数较小等条件，讨论的是不同任务 adapter 的干扰，不保证单个 DiT LoRI 的收敛或视觉质量。
- 全局 B magnitude 会在目标模块之间分配预算；不同线性层尺寸、输入激活分布、learning rate 和缩放都可能影响分数。应固定并记录 target set、A 初始化、`alpha/r`、校准损失、学习率和校准数据。
- 非结构化 mask 若仍走稠密 GEMM，预计主要缩小 trainable B 及其 optimizer state；FLOPs 和 step time 不一定下降。显存大头仍可能是主干、激活、缓存、优化器驻留方式。
- 官方 PEFT/FSDP实现可作为算法参考，不可照搬“梯度置零”就宣称压缩了优化器状态。紧凑优化器状态与导出体积需要分别验证。

## 参考资料

- [COLM 2025 论文 PDF / arXiv:2504.07448](https://arxiv.org/pdf/2504.07448)
- [作者官方 LoRI 仓库与 README](https://github.com/juzhengz/LoRI)
- [作者 LoRI 训练入口](https://github.com/juzhengz/LoRI/blob/main/src/train_lori.py)
- [作者全局 mask 提取实现](https://github.com/juzhengz/LoRI/blob/main/src/extract_mask.py)
- [作者训练器 mask 加载逻辑](https://github.com/juzhengz/LoRI/blob/main/src/mask_trainers.py)
- [作者 masked AdamW 实现](https://github.com/juzhengz/LoRI/blob/main/src/apply_mask_in_backward.py)
- [作者 LoRI-D / LoRI-S 示例脚本](https://github.com/juzhengz/LoRI/blob/main/scripts/codealpaca_llama3_r_32.sh)
- [Hugging Face PEFT LoRA scaling 源码](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)
