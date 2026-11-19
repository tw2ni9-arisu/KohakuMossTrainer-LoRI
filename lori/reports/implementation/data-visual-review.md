# LoRI 数据视觉复核

## 样本与观察

只读查看了已选 12 张中的 4 张：portrait 组取两端（组内索引 0、9），landscape 组查看全部 2 张（组内索引 0、1）。使用 view_image 查看原图，未生成或修改图像。

| 方向 | 图像 | 尺寸 |
| --- | --- | --- |
| portrait | 110959357_p0.png | 752×1344 |
| portrait | 148404748_p0.png | 784×1328 |
| landscape | 112715321_p1.png | 1040×992 |
| landscape | 143436718_p0.png | 1072×944 |

样本显示数据域以二次元角色插画为主，既有单角色构图，也有双角色场景。角色常见长发、明显发色、精细眼部表现、层叠服饰与装饰配件；画面使用干净线稿、柔和渐变、清晰高光和较强的角色轮廓光。背景会从星空/幻想主题延伸到室内或海边场景，常以靛蓝、紫、青绿或柔和粉彩作为主色。caption 以 nigenmame 触发词开头，整体是自然语言描述。

这是对 4 张的抽样观察，不能据此代表全部 12 张。复核样本中存在可能被后续视为敏感的语义；为避免复述原 caption，建议后续出图提示明确成年角色、保守衣着和普通姿态，并避开亲密或束缚题材。下列提示均按这个边界编写。

## 后续对照提示建议

以下 4 条对齐角色插画、配色和场景构图；至少两条为双角色。可为每条另建不带触发词的 matched control，只移除开头的 nigenmame，其余提示与生成设置保持一致。建议由 Sol 决定是否替换已有的静物提示；本报告不修改原 manifest 或 comparison-prompts 文件，也不新增训练目标。

1. nigenmame, polished anime character illustration of an adult woman with long lavender hair and golden eyes, wearing a modest blue-and-white sailor-inspired dress with a navy bow, one hand near her cheek, framed waist-up against an indigo starfield with small pastel ringed planets; clean linework, layered fabric details, soft highlights, dreamy portrait composition.
2. nigenmame, anime-inspired adult woman with teal twin tails and red hair ribbons in a modest teal-trimmed stage outfit, holding a softly glowing bouquet beneath a deep indigo night sky; full-body portrait, flowing hair, luminous flowers, graceful calm pose, restrained fantasy lighting.
3. nigenmame, two adult anime-style women in modest coordinated red-and-white performance outfits, standing side by side with a small plush doll between them, pastel pink studio backdrop; clean linework, gentle expressions, warm soft light, balanced close two-shot.
4. nigenmame, two adult anime-inspired fantasy women in modest summer dresses and light jackets beside a sparkler on a seaside promenade at sunset, calm ocean and peach-colored sky; expressive faces, warm rim light, detailed hair, wide landscape composition.

建议比较时沿用相同种子与采样设置，优先检查触发词带来的角色插画风格、发丝/服饰细节、柔和高光和背景色彩变化；不要将单次短跑差异解释为泛化或质量结论。
