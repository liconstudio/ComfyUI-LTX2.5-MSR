# ComfyUI-LTX2.5-MSR

Standalone multi-reference image and optional AVref audio conditioning nodes for LTX-2.5 in ComfyUI.

This extension loads MSR LoRA checkpoints with learned reference-slot embeddings and applies up to five independently encoded image references to an LTX-2.5 video latent. It preserves the MSR training convention: stable slot ordering, learned slot identity embeddings, prepended reference tokens, and consecutive negative temporal offsets.

The extension contains the complete MSR loader and guide implementation. It does **not** require `LTX-MSR-Multi-Reference-Guide` or another custom MSR node package.

## Features

- One MSR/AVref LoRA loader and one main Reference Guide.
- Optional native audio LATENT inputs on the main Guide, preserving sparse absolute audio windows.
- One to five references: `pic1`, optional `pic2`–`pic4`, and optional `background`.
- Learned Fourier-MLP slot embeddings extracted directly from the LoRA checkpoint.
- Stable reference ordering with consecutive negative temporal positions.
- Reference-token prepending compatible with the MSR training layout.
- Automatic reference resizing, center cropping, and white-padding behavior.
- Optional tiled VAE encoding for reference images.
- Support for LoRA metadata such as `reference_downscale_factor`.
- Native ComfyUI model, conditioning, latent, VAE, sampler, and AV nodes remain usable.
- Suitable for both single-stage generation and two-stage latent-upscale refinement.

## Nodes

### ComfyUI-LTX2.5-MSR IC-LoRA Loader

Registration ID: `ComfyUILTX25MSRICLoRALoader`

Loads an MSR or AVref LoRA into a native ComfyUI `MODEL` and extracts the learned slot-embedding tensors and metadata. AVref checkpoints are detected from their audio weights/metadata, strictly validated, and enable the absolute audio-slot position patch.

Inputs:

- `model`: an LTX-2.5 model loaded by a native ComfyUI model loader.
- `lora_name`: an MSR LoRA from `ComfyUI/models/loras`.
- `strength_model`: LoRA strength.

Outputs:

- `model`: the LoRA-loaded native ComfyUI model.
- `msr_parameters`: slot-embedding weights and reference metadata for the guide node.

The selected LoRA must contain `reference_slot_embedding` weights. A normal LoRA without those weights is rejected with a clear error.

### ComfyUI-LTX2.5-MSR Multi-Reference Guide

Registration ID: `ComfyUILTX25MSRMultiReferenceGuide`

Encodes each supplied reference independently, adds its learned slot embedding, and appends it to the video latent at the negative temporal positions used during MSR training.

Reference inputs are processed in this stable order:

1. `pic1`
2. `pic2`
3. `pic3`
4. `pic4`
5. `background`

For image-only use, missing optional inputs are skipped and image slot IDs are assigned consecutively. When audio references are connected, numbered pictures must be contiguous (`pic1..picN`), and each `audio_refN` requires the same numbered picture. Background counts toward the total slot count. Missing audio inputs preserve their time windows and do not renumber the remaining audio.

Important options:

- `strength`: reference conditioning strength.
- `reference_frames`: `25` or `33`; default is `33`. Set it to `25` when using an AVref checkpoint, even without reference audio.
- `audio_ref1` / `audio_ref2`: optional LATENT outputs of native `LTXVAudioVAEEncode`, paired with `pic1` / `pic2`. The third audio input is hidden.
- `use_tiled_encode`: enables tiled reference VAE encoding.
- `tile_size` / `tile_overlap`: tiled-encoding settings.

The guide expects a video-only LTX latent shaped `[B, 128, F, H, W]`. Connect it before `LTXVConcatAVLatent`. Current MSR inference requires a batch size of one.

## Installation

1. Place the repository at:

   ```text
   ComfyUI/custom_nodes/ComfyUI-LTX2.5-MSR
   ```

2. Update ComfyUI to a version with native LTX-2.5 AV nodes.
3. Restart ComfyUI completely.

No additional Python packages are required beyond the dependencies already used by ComfyUI's native LTX implementation.

## Model locations

The extension does not distribute model weights. Use the standard ComfyUI folders:

```text
ComfyUI/models/
├── diffusion_models/       # LTX-2.5 transformer
├── text_encoders/          # LTX-2.5 Gemma text encoder
├── vae/                    # LTX-2.5 video and audio VAEs
├── loras/                  # MSR LoRA checkpoints
└── latent_upscale_models/  # optional LTX-2.5 latent upscaler
```

## Basic workflow

```text
Native LTX-2.5 model loader
  → ComfyUI-LTX2.5-MSR IC-LoRA Loader
  → native guider / sampler model input

Positive + negative conditioning
Video VAE
Empty video latent
Reference images
MSR parameters
  → ComfyUI-LTX2.5-MSR Multi-Reference Guide
  → LTXVConcatAVLatent
  → native LTX-2.5 sampler
  → LTXVSeparateAVLatent
  → LTXVCropGuides
  → video/audio decode
```

Always pass the guide's conditioning outputs to `LTXVCropGuides` after sampling so the appended reference slots are removed before video decoding.

## Two-stage latent upscaling

For a two-stage workflow:

1. Run the first MSR-guided sampling pass.
2. Separate the AV latent and crop the first-stage reference slots.
3. Spatially upscale the generated video latent with `LTXVLatentUpsampler`.
4. Apply a second `ComfyUI-LTX2.5-MSR Multi-Reference Guide` using the same references.
5. Rejoin the upscaled video latent with the audio latent.
6. Run a low-noise refinement pass.
7. Crop the second-stage reference slots before decoding.

Do not replace this path with `LTXVImgToVideoInplace`: that node re-encodes pixel-space images or videos and overwrites latent frames. The MSR refiner should retain the upscaled generated latent while rebuilding the high-resolution reference conditions.

## Included workflow

`LTX2.5-MSR-sample-workflow.json` contains a native ComfyUI example using the MSR loader and multi-reference guide. Model filenames and LoRA filenames in the example are placeholders for files installed in your own ComfyUI model directories.

## LoRA compatibility

Compatible MSR checkpoints must contain the learned slot-embedding tensors, including:

```text
reference_slot_embedding.frequencies
reference_slot_embedding.net.0.weight
reference_slot_embedding.net.0.bias
reference_slot_embedding.net.2.weight
reference_slot_embedding.net.2.bias
```

The loader also accepts the `diffusion_model.reference_slot_embedding.*` key prefix.

Supported metadata conventions:

- `reference_token_order=prepend`
- `reference_slot_time_offsets=pic1_based_negative_time`
- `reference_downscale_factor`

For ComfyUI compatibility, the guide uses a temporal scale factor of `1` while preserving the learned slot embeddings and consecutive negative positions.

## Troubleshooting

### `Invalid input: lora_name`

The workflow contains a LoRA path that is not present in ComfyUI's current model list. Confirm the file is under `ComfyUI/models/loras`, then refresh the browser model list or restart ComfyUI. The path stored in the workflow must exactly match the value shown by the loader dropdown.

### `This LoRA does not contain reference_slot_embedding weights`

The selected checkpoint is a normal LoRA rather than an MSR multi-reference checkpoint.

### Guide latent shape error

Connect the guide to the video latent before `LTXVConcatAVLatent`. A combined audio/video latent is not accepted.

### Reference slots appear in the decoded video

Connect the guide conditioning and sampled video latent through `LTXVCropGuides` before VAE decoding.

## Scope

This repository implements only MSR-specific LoRA loading and multi-reference conditioning. Base-model loading, text encoding, prompt enhancement, sigma schedules, sampling, latent upscaling, VAE decoding, audio decoding, and video output are intentionally handled by native ComfyUI nodes.


## AVref 音频参考功能

音频参考已直接接入主 `ComfyUI-LTX2.5-MSR Multi-Reference Guide`。主加载器同时支持 MSR 和 AVref LoRA；使用 AVref 时将 `reference_frames` 设为 `25`。

连接方式：`LoadAudio → LTXV Audio VAE Encode → Guide.audio_ref1 / audio_ref2`。第三路音频保持隐藏，前两路均为可选输入，分别对应 `pic1` / `pic2`。保留 AVref 的独立槽位 embedding、长度截断和稀疏绝对时间窗，缺失音频不会补静音或挤占其他槽位。

插件只注册主加载器和主 Guide 两个节点。原独立 AVref 加载器、音频编码器和 Guide 已移除；旧 AVref 工作流需改用主节点及原生音频编码器。独立 AVref 插件继续停用。

完整连接说明见 [AVref 说明](README-AVref.md)，更新后的两阶段示例见 `LTX2.5-MSR-AVref-sample-workflow.json`。示例仍包含 KJNodes、rgthree、PromptRelay 等辅助插件节点；本插件的加载器和 Guide 本身不依赖这些插件。

## 1.2.0

- 音频参考合入主 Guide，只保留主加载器和主 Guide 两个节点。
- 保留 AVref 稀疏绝对时间窗，第三路音频接口隐藏。
- 更新两阶段示例；旧版三个独立 AVref 节点已移除。

## License

Licensed under the [Apache License 2.0](LICENSE).
