# Task 7 只读准备报告

日期：2026-11-18。此报告仅做源码/文件静态核实；没有加载模型、运行采样、写推理配置或修改训练参数。未派代理。以下设置是后续执行建议，不是已运行结果。

## 路由与当前缺件

- 复用内层 `scripts/sample/compare_grid.py`，无须新增 web 服务、ComfyUI 进程或采样框架。
- Task 6 的预期 native 权重：`experiment/task6/session-20261118-a/sparse/lori-sparse/export/lori-sparse-lycoris.safetensors`（相对外层工作区）。本次核实时 `Test-Path=False`；不能使用 `sparse-init.safetensors` 或 calibration 导出替代训练结果。
- 必须等 sparse32 完成并审计 final native/portable 文件、非零有限 B、精确 A 复用等证据；由 root 后续授权实际推理并先安排 Luna 监控。root 在运行期间 sleep。
- 所需模型三件均已存在于当前工作区，不需下载模型。源码自带 Qwen config/tokenizer/chat template，不缺独立 HF 模型目录。
- 尚未创建 Task 7 私有配置、launcher、输出/缓存目录；应由后续执行 owner 在获授权后创建并静态审查。当前没有推理资源峰值或时长保证。

## 已核实的输入

工作区根：`D:/Develop/KohakuMossTrainer-LoRI`。内层 HEAD：`285f4bd2b5c29378ca75797dd247690f6a06a21a`，本次 `git status --short` 无输出。运行前重新记录实际 HEAD/diff，避免将此快照当作未来 revision。

| 组件 | 当前工作区相对路径 | 文件字节数 |
| --- | --- | ---: |
| DiT | models/ComfyUI/models/diffusion_models/moss_image_base_int8_convrot.safetensors | 5413698958 |
| Text encoder | models/ComfyUI/models/text_encoders/qwen3.5_9b_int8_convrot.safetensors | 9748121532 |
| VAE | models/ComfyUI/models/vae/flux2-vae_small.safetensors | 249521340 |

读取了 task-7-brief、data-visual-review、training-handoff、compiler-diagnosis-report；未读取原数据图像或 caption。采样选择角色/风格提示依据已有视觉复核，不能用静物提示来判断角色画风迁移。

## Native LoRI 与 portable Comfy 入口

`compare_grid.main()` 将 CHECKPOINTS 转为 `(path, multiplier)`，经 `inference/grid.py::render_columns()` 调用 `adapters.load()`。`adapters/infer/inference.py::add()` 读取 metadata 并验证 LoRI；raw-tag native 状态经过 `_raw_tensors()`，按保存 spec 重建模块、转换到 unfolded native tensors，再由 `_build()` 根据 `A` 和 `b_values` 重建 sparse LoRIModule。`_source_base()` 对 LoRI 不需要物化完整基础权重。

`adapters/train/adapter_set.py::export()` 对 LoRI 的 lycoris 导出有专门分支，保存 native A/indices/b_values 与 metadata，必要时撤销 fold。因此后续优先直接比较最终 `lori-sparse-lycoris.safetensors`，保留原始 sparse 表示和出处。

portable `*-comfy.safetensors` 则由 `adapters/kinds/lori.py::_comfy_tensors()` 转为普通 LoRA 的 up=dense B、down=A、alpha，之后走已有 Comfy key mapping/fused qkv 路径。现有 trainer 的 `compare_grid`/`sample.py` 可加载这种文件；另一个真正 ComfyUI 安装是否有兼容的 MOSS loader 没有在此核实，Task 7 不依赖它。portable 不再保留物理稀疏 B 的存储优势；native sparse forward 也会临时 scatter 成 [out,rank] B，不能声称其推理计算同比例稀疏。

## 最小可解释的对照矩阵

建议一次 grid 使用 8 个 prompt rows（4 个语义场景 × 带/不带 trigger），每个 row seed=42。compare_grid 自带纯 base 列；CHECKPOINTS 的顺序建议为 **同一最终 native 文件 multiplier=1，然后同文件 multiplier=0**，LABELS 对应 `LoRI x1`、`LoRI x0 after x1`。由此得到 base、x1、x0 三列，共 24 张。把零倍率放在 x1 后，可同时检查一次 adapter 撤销后的 baseline。预算充裕再为所有 rows 添加 seed43，合计48张；不应只为某一列改变种子。

`grid.render_columns()` 在任何 adapter 加载前一次编码全部正/负提示；每列共享同一份 embedding、seeds、sizes、steps、CFG、shift、skip_layers、batch 和 cfg_mode。每列 adapter 的生成放在 `try/finally` 内，随后 `stack.remove()` 并 `backbone.bind_ops()`；remove 恢复保存的 forwards/weights，清空 live/earlier 状态。量化/streamed 基础层用 stored bypass，不把 adapter 合入 INT8 权重。此为源码隔离机制，尚不等同本机实际验证。

验收至少检查每行 base 与末列 x0 的 decoded RGB 像素一致性（解码后的像素摘要，不能仅比较 PNG 文件字节），并检查 x1 相对 base 的像素差异、差异范围与视觉图。出现 base/x0 差异时保留证据，先核查数值确定性/撤销/NaN，不能据此直接判定风格效果。x1 非零差异只证明影响输出，不自动证明风格学习。源测试 `tests/inference/test_compare_grid.py` 只覆盖共享编码/布局等，不构成本机真实 adapter 隔离证明；本次未运行测试。

## 分辨率与采样设置建议（未写配置）

官方源码默认在 `configs/sample/_base/base_sample.py` 和 compare_grid 中都是 **1024×1024、50 steps、CFG 3、shift 5**；`diffusion/sampling/sample.py` 是固定 Euler 路线，没有必要新增 sampler knob。

建议本次短跑第一轮使用 **512×512**，保留官方50/3/5。理由：相对1024，每张图的图像 token 数为四分之一；相对256训练输入，512有四倍像素，能更清楚观察脸部/服装/背景细节，同时为8条提示×3列的短跑控制成本。它是本次工程折中，并非模型官方分辨率，也不是从256短训推出的质量保证。所有场景先统一正方形，以免把形状与资源差异混入首轮对照；后续若需要保留横向构图，再明确成对使用相同横幅尺寸。

建议固定：VARIANT=base、PRECISION=bf16、BASE_PRECISION=int8、FP16_FIX=none、ADAPTER_MODE=bypass、BATCH=1、CFG_MODE=two_pass、SKIP_LAYERS=default、NEGATIVE为空、TUNE=off。two_pass 源码说明为条件/无条件两次 forward，单次 token 峰值较 packed 小；其吞吐代价需实测。所有列必须完全相同，不应将 packed base 与 two_pass adapter 比较。50 steps ×24 张约1200个样本采样步；two_pass 会再分为两条 CFG 分支调用，不能把1200当作 DiT forward 总数或时间估计。

## 低显存加载顺序与缓存

`MossPipeline.from_pretrained()` 当前顺序固定为：先 DiT → text encoder → VAE；然后 `grid.render_columns()` 编码提示 → 逐列 DiT 采样/VAE解码。compare_grid 不暴露先独立编码再加载 DiT 的流程，也没有离线 embedding 文件 knob。训练 caption cache 不是这组新提示的有效替代。无需为此新增框架；利用已有 placement 和一次编码即可。

建议 DiT `MEMORY_PROFILE=minimal`，由 inference resolve 清除训练专用 checkpoint/act_offload/optimizer 设置，保留 ring2；`BASE_PRECISION=int8` 保留本地预量化权重。`memory_of()` 会正确将 BASE_PRECISION 放进 MemoryPlan overrides。不要在 compare_grid 使用训练专用的 `ckpt-ring4-actoff-optstream-gradoff`；其 choices 仅接受现有推理 profile 名。脚本注释虽提到 manual，但当前 choice validation 未包含 manual，优先使用源码实际支持的 minimal。

建议 TEXT_DEVICE=cuda，TEXT_ENCODE 使用 memory=ring、stream_resident=2、batch=1、max_tokens=4096、dtype=bf16、precision=auto、free_after_use=True。不建议把 INT8 Qwen 直接放CPU：TextEncoderPlan 的 host compute 路径可能采用 compute 权重，失去量化内存优势。当前 Task6 同 ring2/batch1 文本预处理已成功，但当时只编码器阶段；不能把其峰值等同于 DiT 同时存在时的 Task7 峰值。

`QwenTextEncoder._encode()` 先缓存 empty embedding，再一次处理传入所有提示（内部 batch1 分组），最后 release 权重。`_embeddings()` 会在同一个 encode 调用中合并非空负提示，并在需要时读已缓存 empty。因此上述 free_after_use 与 grid 共享 embedding 路线匹配；后续不得再调用 pipe.encode 新提示。embeddings 只保留在内存并复用到所有列，不会被 compare_grid 自动保存为磁盘缓存。

建议 VAE_ENCODE memory=offload、batch=1、compile=off，VAE_DTYPE=bf16；权重先读到host，逐张解码时临时搬入设备。512首轮无需额外 tile；若发现实测峰值问题，再基于证据调整并保持所有列同设置。禁止默默降到256或改采样步数。

保持 source venv 只读：使用 `D:/Develop/nigenmame-LoRA/KohakuMossTrainer/.venv/Scripts/python.exe`，PYTHONDONTWRITEBYTECODE=1，PYTHONPATH 指向当前副本 src 与工作区 `.tools/python`。进程级 CC 指向 source venv bundled Triton `runtime/tcc/tcc.exe`，不改 CXX。TRITON_CACHE_DIR/TRITON_DUMP_DIR/TRITON_OVERRIDE_DIR/TORCHINDUCTOR_CACHE_DIR/KOHAKUMOSS_KERNEL_CACHE/CUDA_CACHE_PATH/TORCH_EXTENSIONS_DIR/HF_HOME/TORCH_HOME/TEMP/TMP 都放后续 Task7 私有 runtime 目录，并设置 HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1。先创建这些工作区目录，再启动子进程；不复用诊断或Task6运行缓存，不写源venv。无需杀死其他 moss-controlnet 进程；启动前核查空闲RAM/VRAM。

## 后续最小调用形态

在已复制的内层 trainer cwd，以源 Python 调用现有 CLI：

```text
<source-venv-python> -u -m kohakuengine.cli run scripts/sample/compare_grid.py --config <workspace-private-task7-config.py>
```

源 venv 没有 kohakuengine/__main__.py，故不要写成 `-m kohakuengine`；`cli.py` 末尾确有 main 入口。也已确认源 venv Scripts/kogine.exe 存在。这里只记录调用形态，配置文件尚不存在，也未执行命令。

私有配置显式设置 WEIGHTS 为当前 `models/ComfyUI/models`，三个组件用上表精确路径，OUT 使用全新的工作区 Task7目录；CHECKPOINTS 必须最终 native 路径和明确(1.0,0.0)倍率。用单个配置完成全部列，避免多个独立调用带来precision/batch/CFG差异。后续 owner 可用轻量 launcher 调用上述现有CLI并记录环境/资源，不应改采样框架。

## 建议提示全文

每条有两个严格匹配版本：下列原文与只删除开头 `nigenmame, ` 的版本；其余字符不变。四条均来自 data-visual-review，不改变原manifest或comparison-prompts.json。

1. nigenmame, polished anime character illustration of an adult woman with long lavender hair and golden eyes, wearing a modest blue-and-white sailor-inspired dress with a navy bow, one hand near her cheek, framed waist-up against an indigo starfield with small pastel ringed planets; clean linework, layered fabric details, soft highlights, dreamy portrait composition.
2. nigenmame, anime-inspired adult woman with teal twin tails and red hair ribbons in a modest teal-trimmed stage outfit, holding a softly glowing bouquet beneath a deep indigo night sky; full-body portrait, flowing hair, luminous flowers, graceful calm pose, restrained fantasy lighting.
3. nigenmame, two adult anime-style women in modest coordinated red-and-white performance outfits, standing side by side with a small plush doll between them, pastel pink studio backdrop; clean linework, gentle expressions, warm soft light, balanced close two-shot.
4. nigenmame, two adult anime-inspired fantasy women in modest summer dresses and light jackets beside a sparkler on a seaside promenade at sunset, calm ocean and peach-colored sky; expressive faces, warm rim light, detailed hair, wide landscape composition.

## 执行后证据要求

compare_grid 自带 `grid.png`、`r00_c00.png` 等 singles、metadata.json。metadata 含 resolved knobs、columns(path,multiplier)、rows(prompt,seed)；SIZES 等在 knobs 内。它不自动保存文件hash、source revision或完整进程峰值，后续需将这些放外层报告/launcher sidecar。columns 是权重索引、rows 是prompt索引，即使 GRID_LAYOUT=prompt-rows，单图文件的 r/c 含义也不变。

路径直接依据 `compare_grid.py:180–206`：`out=Path(k["OUT"])`，`image.save(out/name)`、`sheet.save(out/"grid.png")`、`(out/"metadata.json").write_text(...)`。这里没有 OUTPUT_DIR、RUN_NAME 或训练任务的嵌套运行目录；OUT 本身就是图像最终目录。`main()` 不返回 artifact 路径对象，需从 resolved OUT 和落盘 metadata 验证。写文件发生在 `render_columns()` 完成全部列之后；中途失败可能尚无任何 singles，因此不能把“尚未出现图片”当成进程退出或重启依据。`mkdir(exist_ok=True)` 允许覆盖同名图片，执行owner必须选全新OUT并预检，避免覆盖旧证据。

监控必须覆盖加载/编码/各列采样/解码的真实进程与退出状态，记录 host peak working set、GPU allocated/reserved 与外部可见VRAM、elapsed/step时长及错误日志。pipeline 日志只覆盖 generate 自己的窗口，不能代替启动加载峰值。生成完成后由执行者看 grid 和相关单图，汇报非零影响、可见风格线索、人物/手部/服装/背景伪影及弱效果；256输入的12图短训不能支持质量对齐或泛化结论。

本次唯一写入文件为此报告，未运行模型、采样、测试、下载、训练或任何参数变更。
