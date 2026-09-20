# ComfyUI-LTX2.5-MSR-AVref

这是合并到 `ComfyUI-LTX2.5-MSR` 中的 AVref 功能。它保留原插件的多参考图逻辑，并增加三个固定编号的参考音频槽位，复现 Stage4 训练使用的稀疏绝对时间窗布局。

原版两个节点保持不变，新增三个 AVref 节点。独立 AVref 插件应停用，避免同名节点重复注册。

## 节点

### MSR-AVref IC-LoRA Loader

节点 ID：`ComfyUILTX25MSRAVrefICLoRALoader`

- 加载普通 LoRA 权重。
- 从 checkpoint 中单独提取 `reference_slot_embedding.*`。
- 单独提取 `reference_audio_slot_embedding.*`。
- 安装仅针对 MSR-AVref conditioning 生效的 LTX audio position patch。
- 输出 `MODEL` 和 `LTX_MSR_AVREF_PARAMETERS`。

加载器要求 checkpoint metadata 明确声明：

- `reference_audio_conditioning=id_lora_clean_negative_rope`
- `reference_audio_token_order=pic1_to_picN_then_target`
- `reference_audio_rope_layout=absolute_image_slot_windows`
- `reference_audio_overflow_mode=truncate`

### Three Audio Reference Encoder

节点 ID：`ComfyUILTX25MSRAVrefAudioEncoder`

输入原生 ComfyUI `AUDIO` 和 LTX Audio VAE。当前界面显示 `audio_ref1`、`audio_ref2`；`audio_ref3` 按原配置保持隐藏，后端保留该参数。槽位映射如下：

- `audio_ref1` 对应 `pic1`
- `audio_ref2` 对应 `pic2`
- `audio_ref3` 对应 `pic3`

可见接口均可留空，但至少连接一个。某张参考图没有参考音频时，直接不连接同编号接口。缺失槽位不会插入静音 token，也不会导致后续音频重新编号。

每段音频经 Audio VAE 编码为 `[B,T,128]` token，再叠加 checkpoint 中同编号的独立 audio slot embedding。超过 5 秒槽容量的 latent token 从尾部截断；短音频不补齐。

编码结果是可复用 payload。同一个 payload 可以同时连接两阶段工作流中的两个 Guide，避免重复运行 Audio VAE。

### Multi-Reference Guide

节点 ID：`ComfyUILTX25MSRAVrefMultiReferenceGuide`

保留原插件的 `pic1`、可选 `pic2`–`pic4`、可选 `background`、居中缩放/裁切、白边适配、图像 slot embedding 和负时间 guide 逻辑。

AVref checkpoint 的参考图固定编码为 25 帧。连接 audio payload 后，Guide 会验证每个 `audio_refN` 都存在同编号 `picN`，并以实际参考图总数计算音频时间窗。

例如存在四张参考图，只连接 `audio_ref1` 和 `audio_ref3`：

```text
audio_ref1 -> pic1 window [-20.04, -15.04]
audio_ref2 -> empty window [-15.04, -10.04]
audio_ref3 -> pic3 window [-10.04,  -5.04]
pic4       -> empty window [ -5.04,  -0.04]
target audio starts at 0
```

各参考音频在自己的 5 秒窗口内向右对齐，末端保留 0.04 秒 margin。所有已连接参考音频按 slot 编号升序放在目标音频 token 之前，使用 timestep 0，并在模型输出时自动移除。

## 基本连接顺序

```text
Native LTX-2.5 model loader
  -> MSR-AVref IC-LoRA Loader
  -> sampler model input

LTX Audio VAE + audio_ref1/2/3 + AVref parameters
  -> Three Audio Reference Encoder
  -> audio_references

positive/negative + Video VAE + video-only latent
+ pic1...pic4/background + AVref parameters + audio_references
  -> MSR-AVref Multi-Reference Guide
  -> LTXVConcatAVLatent
  -> native sampler
  -> LTXVSeparateAVLatent
  -> LTXVCropGuides for the video latent
  -> video/audio decode
```

Guide 必须在 `LTXVConcatAVLatent` 之前接收 video-only latent。音频参考不是 AV latent 的一部分，因此不需要额外 crop；参考图仍需在采样后通过 `LTXVCropGuides` 移除。

不要再串接原生 `LTXVReferenceAudio`。原生节点会覆盖 `ref_audio`，并把多个参考排成连续负时间流，不符合本 checkpoint 的固定稀疏槽位训练布局。

## Sample workflow

插件目录内提供：

```text
LTX2.5-MSR-AVref-sample-workflow.json
```

它沿用原 MSR 的双阶段采样结构，并新增三条固定映射：

```text
LoadAudio 0001.mp4 -> audio_ref1 -> pic1
LoadAudio 0002.mp4 -> audio_ref2 -> pic2
LoadAudio 0003.mp4 -> audio_ref3 -> pic3
```

示例保留三路音频结构；当前界面隐藏第三路输入，请按实际可见接口调整连接。参考音频只经过一次 Audio VAE 编码，输出同时复用到 Stage 1 和 Stage 2 Guide。示例已启用第三张参考图以匹配 `audio_ref3`。请将示例中的 LoRA、图片和音频文件替换为实际文件；某个 `picN` 没有参考音频时，断开同编号 `audio_refN`，不要接静音、不要把后续音频前移补位。

## 安装

目录应为：

```text
ComfyUI/custom_nodes/ComfyUI-LTX2.5-MSR
```

本插件不引入额外依赖，使用 ComfyUI 原生 LTX-2.5、Audio VAE、AUDIO、AV latent 和 sampler 实现。安装后完整重启 ComfyUI。

## checkpoint key

图像和音频 embedding 均兼容带或不带 `diffusion_model.` 的 key：

```text
diffusion_model.reference_slot_embedding.*
reference_slot_embedding.*
diffusion_model.reference_audio_slot_embedding.*
reference_audio_slot_embedding.*
```

两套 embedding 都必须包含：

```text
frequencies
net.0.weight
net.0.bias
net.2.weight
net.2.bias
```

当前 Stage4 checkpoint 的图像和音频输出维度均为 128。
