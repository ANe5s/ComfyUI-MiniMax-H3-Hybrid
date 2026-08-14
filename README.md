# ComfyUI MiniMax H3 Hybrid

Workflow support for ComfyUI MiniMax H3 reference-to-video generation. It uses a stronger FL2VA model together with validated Ref2VA tensor-overlay modes to unlock MiniMax H3 capabilities. The three loading modes are: `none` uses the native FL2VA model; `block_range_adaln` overlays a validated Ref2VA tensor combination to improve detail depiction and clarity, with a small risk of fitting extra objects; `all_adaln` combines Ref2VA-like shot composition with FL2VA's strong image generation and color style in limited tests. It stays on ComfyUI's file-backed DynamicVRAM/AIMDO loading path.

> Full design rationale, function-space analysis, and video-validation evidence are documented in the [Technical Whitepaper (EN)](minimax_h3_hybrid_technical_whitepaper_en.md). 中文说明见下文。

## Node

- Type: `MinimaxH3_HybridLoader`
- Category: `ANe5s Nodes` (shown as `ANe5s节点` in the Chinese UI)
- Inputs: `base_model`, optional `overlay_model`, and `overlay_preset`
- Output: `MODEL`

## Presets

| Preset | Behavior |
|---|---|
| `none` | Pure FL2VA load. The Ref2VA checkpoint is never opened. |
| `block_range_adaln` (default) | Overlay `adaln_proj` for blocks 45–49. |
| `all_adaln` | Overlay all block AdaLN, final AdaLN, and the format-specific time embedding (`adaln_t_table` for pruned, `time_embedder.*` for full). |

Output heads always remain FL2VA. For a lossless baseline with full reference capability and single-model memory behavior, use `overlay_preset = none` with ComfyUI's reference-conditioning node.

## Format detection and validation

The loader detects the pruned (`adaln_t_table`) and full (`time_embedder.*`) checkpoint formats and rejects mixed pairs. It validates key sets, metadata, and the shape/dtype of selected overlay tensors. Quantization siblings (`.comfy_quant`, `weight_scale`, `pre_quant_scale`, etc.) always come from the same checkpoint as their owning weight.

## Memory behavior

`none` loads a single FL2VA checkpoint. Hybrid presets use ComfyUI's file-backed DynamicVRAM/AIMDO path when enabled, and bounded streaming safetensors readers otherwise, with a system RAM guard when AIMDO is off.

## Actual video comparison

The following screenshots are extracted at 3 s, 6 s, 7.5 s, 9 s, 12 s, and 14.5 s from the four local output videos in `ref2va_vs_hybrid/`:

- `Unet loader-ref2va.mp4` — native ComfyUI UNet loader with Ref2VA
- `Hybrid loader-none.mp4` — hybrid loader, `none`
- `Hybrid loader-block_range_adaln.mp4` — hybrid loader, `block_range_adaln`
- `Hybrid loader-all_adaln.mp4` — hybrid loader, `all_adaln`

Each composite is ordered left to right as `Native UNet Loader Using Ref2VA Alone`, `MiniMax H3 Hybrid Loader Using none Mode`, `MiniMax H3 Hybrid Loader Using block_range_adaln Mode`, and `MiniMax H3 Hybrid Loader Using all_adaln Mode`.

![Ref2VA versus hybrid output at 3 seconds](docs/assets/video_frames/ref2va_vs_hybrid/comparison_3s.jpg)

![Ref2VA versus hybrid output at 6 seconds](docs/assets/video_frames/ref2va_vs_hybrid/comparison_6s.jpg)

![Ref2VA versus hybrid output at 7.5 seconds](docs/assets/video_frames/ref2va_vs_hybrid/comparison_7_5s.jpg)

![Ref2VA versus hybrid output at 9 seconds](docs/assets/video_frames/ref2va_vs_hybrid/comparison_9s.jpg)

![Ref2VA versus hybrid output at 12 seconds](docs/assets/video_frames/ref2va_vs_hybrid/comparison_12s.jpg)

![Ref2VA versus hybrid output at 14.5 seconds](docs/assets/video_frames/ref2va_vs_hybrid/comparison_14_5s.jpg)

In this run, the native UNet + Ref2VA output and the three hybrid outputs show different object states and shot progression at multiple checkpoints. The hybrid presets share the same broad scene and action direction, but local hand, bottle, and cap details still differ at some timestamps. These screenshots document the produced videos; they are not a claim of pixel identity and do not replace full temporal inspection.

## Installation

Copy this directory into `ComfyUI/custom_nodes/` and restart ComfyUI. The plugin has no extra Python dependencies beyond the libraries bundled with ComfyUI.

## License

GPL-3.0-or-later. See `LICENSE`.

---

# ComfyUI MiniMax H3 Hybrid（中文说明）

用于 ComfyUI MiniMax H3 参考生视频工作流。通过使用生成能力更强的 FL2VA 模型，并结合经过验证的 Ref2VA 张量叠加模式，解锁 MiniMax H3 的模型能力。三种加载模式分别为：`none` 使用原生 FL2VA 模型；`block_range_adaln` 在 FL2VA 上叠加经过验证的 Ref2VA 张量组合，增强细节刻画与画面清晰度，但存在极少量引入拟合额外物体的风险；`all_adaln` 在少量测试中同时呈现 Ref2VA 的镜头构图特征，并保留 FL2VA 出色的画面生成能力与色彩风格。节点沿用 ComfyUI 的文件后援 DynamicVRAM/AIMDO 加载路径。

> 完整的设计依据、函数空间分析与视频验证证据见[技术白皮书（中文）](minimax_h3_hybrid_technical_whitepaper_zh.md)。English version: see above。

## 节点

- 类型：`MinimaxH3_HybridLoader`
- 分类：`ANe5s节点`（英文界面显示 `ANe5s Nodes`）
- 输入：`base_model`、可选 `overlay_model` 与 `overlay_preset`
- 输出：`MODEL`

## 预设

| 预设 | 行为 |
|---|---|
| `none` | 纯 FL2VA 加载，Ref2VA 检查点永不打开。 |
| `block_range_adaln`（默认） | 覆盖 blocks 45–49 的 `adaln_proj`。 |
| `all_adaln` | 覆盖全部 block AdaLN、final AdaLN 与格式对应的时间嵌入（pruned 为 `adaln_t_table`，full 为 `time_embedder.*`）。 |

输出头始终保留 FL2VA。需要"无损基线 + 完整参考能力 + 单模型内存"时，使用 `overlay_preset = none` 并配合 ComfyUI 的参考条件节点即可。

## 格式识别与校验

加载器自动识别 pruned（`adaln_t_table`）与 full（`time_embedder.*`）两种检查点格式，并拒绝 pruned/full 混配；同时对键集合、metadata 与被覆盖张量的 shape/dtype 做一致性校验。量化伴生键（`.comfy_quant`、`weight_scale`、`pre_quant_scale` 等）始终与所属权重同源。

## 内存行为

`none` 只加载单个 FL2VA 检查点。混合预设优先走 ComfyUI 的文件后援 DynamicVRAM/AIMDO 路径；该路径关闭时退化为有界流式读取，并带有系统 RAM 保护。

## 实际视频截图对比

以下截图直接从本地 `ref2va_vs_hybrid/` 中的四个实际输出视频抽取，时间点为 3 秒、6 秒、7.5 秒、9 秒、12 秒和 14.5 秒：

- `Unet loader-ref2va.mp4`：ComfyUI 原生 UNet 加载器 + Ref2VA
- `Hybrid loader-none.mp4`：插件混合加载器，`none`
- `Hybrid loader-block_range_adaln.mp4`：插件混合加载器，`block_range_adaln`
- `Hybrid loader-all_adaln.mp4`：插件混合加载器，`all_adaln`

每张合成图从左到右依次为：`Native UNet Loader Using Ref2VA Alone`、`MiniMax H3 Hybrid Loader Using none Mode`、`MiniMax H3 Hybrid Loader Using block_range_adaln Mode`、`MiniMax H3 Hybrid Loader Using all_adaln Mode`。

![3 秒：原生 Ref2VA 与混合加载器对比](docs/assets/video_frames/ref2va_vs_hybrid/comparison_3s.jpg)

![6 秒：原生 Ref2VA 与混合加载器对比](docs/assets/video_frames/ref2va_vs_hybrid/comparison_6s.jpg)

![7.5 秒：原生 Ref2VA 与混合加载器对比](docs/assets/video_frames/ref2va_vs_hybrid/comparison_7_5s.jpg)

![9 秒：原生 Ref2VA 与混合加载器对比](docs/assets/video_frames/ref2va_vs_hybrid/comparison_9s.jpg)

![12 秒：原生 Ref2VA 与混合加载器对比](docs/assets/video_frames/ref2va_vs_hybrid/comparison_12s.jpg)

![14.5 秒：原生 Ref2VA 与混合加载器对比](docs/assets/video_frames/ref2va_vs_hybrid/comparison_14_5s.jpg)

本次实测中，原生 UNet + Ref2VA 与三种混合加载器预设在多个时间点的物体状态和镜头推进上存在差异。三种混合预设的整体场景与动作方向较为接近，但部分时间点的手部、瓶体和瓶盖细节仍有差别。这里仅记录实际输出截图，不宣称像素级一致，也不以截图替代完整视频的连续性与稳定性检查。

## 安装

把本目录复制到 `ComfyUI/custom_nodes/` 后重启 ComfyUI。除 ComfyUI 自带库外无额外 Python 依赖。

## 许可证

GPL-3.0-or-later，见 `LICENSE`。
