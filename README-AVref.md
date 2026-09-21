# 主节点中的 AVref 音频参考

AVref 已合并到 `ComfyUI-LTX2.5-MSR` 的两个主节点：

- `ComfyUI-LTX2.5-MSR IC-LoRA Loader`（`ComfyUILTX25MSRICLoRALoader`）
- `ComfyUI-LTX2.5-MSR Multi-Reference Guide`（`ComfyUILTX25MSRMultiReferenceGuide`）

前端采用 VAref 的连接方式：音频由原生 `LTXV Audio VAE Encode` 编码为 LATENT，直接接入主 Guide。后端仍使用 AVref 的稀疏绝对时间窗，并未改成 VAref 的连续音频排列。

## 连接方式

```text
LTX-2.5 model -> MSR IC-LoRA Loader -> sampler model
                         |
                  msr_parameters
                         |
LoadAudio -> LTXV Audio VAE Encode -> Guide.audio_ref1  (对应 pic1)
LoadAudio -> LTXV Audio VAE Encode -> Guide.audio_ref2  (对应 pic2)
                         |
positive/negative + Video VAE + video-only latent + pic1...pic4/background
                         |
             MSR Multi-Reference Guide (reference_frames=25)
                         |
                LTXVConcatAVLatent -> sampler
                         |
              LTXVSeparateAVLatent -> LTXVCropGuides -> decode
```

每个 `LTXV Audio VAE Encode` 需要连接 LTX Audio VAE。同一音频 LATENT 可以同时接入两阶段 Guide，避免重复 VAE 编码。不再需要独立的 AVref Audio Encoder 节点。

Guide 界面显示 `audio_ref1` 和 `audio_ref2`，两路输入都可留空。第三路音频保持隐藏，后端保留该参数。连接音频时，编号图片必须连续，且 `audio_refN` 必须存在同编号 `picN`。

## 空窗槽位逻辑

每段音频按原始编号添加独立 audio slot embedding。缺失音频不会插入静音 token，不会重新编号，也不会让后续音频前移。每段音频在其时间窗内向右对齐，长度超过槽容量时从尾部截断，短音频不补齐。

默认槽宽为 5 秒、末端 margin 为 0.04 秒，具体值读取 checkpoint metadata。图片总数包括 background。例如连接 `pic1`、`pic2`、`pic3`、`background`，只给 `pic2` 音频：

```text
pic1       -> 空窗 [-20.04, -15.04]
audio_ref2 -> 窗口 [-15.04, -10.04]，音频末端对齐 -10.04
pic3       -> 空窗 [-10.04,  -5.04]
background -> 空窗 [ -5.04,  -0.04]
target audio starts at 0
```

目标音频位置不变。参考音频使用 timestep 0，模型输出时自动移除；图像参考仍通过采样后的 `LTXVCropGuides` 移除。不要再串接会覆盖 `ref_audio` 的原生 `LTXVReferenceAudio`。

## 模型要求

主加载器保留普通 MSR 的图像路径；识别到 AVref 权重/声明时，校验两套 slot embedding 和训练 metadata，并通过 ModelPatcher 安装音频时间坐标补丁。

AVref 必须同时包含 `reference_slot_embedding.*` 和 `reference_audio_slot_embedding.*`（均支持 `diffusion_model.` 前缀）。两套 embedding 都包含 `frequencies`、`net.0.weight`、`net.0.bias`、`net.2.weight`、`net.2.bias`。

要求 metadata 包含：

- `reference_token_order=prepend`
- `reference_slot_time_offsets=pic1_based_negative_time`
- `reference_audio_conditioning=id_lora_clean_negative_rope`
- `reference_audio_token_order=pic1_to_picN_then_target`
- `reference_audio_rope_layout=absolute_image_slot_windows`
- `reference_audio_overflow_mode=truncate`
- `reference_audio_sparse_slots=true`
- `reference_slot_embedding_enabled=true`
- `reference_audio_slot_embedding_enabled=true`

AVref 图像参考必须设为 25 帧，即使没有连接参考音频。普通 MSR 仍支持 25/33 帧；未连接 MSR 参数时，原标准图像引导路径保留。

## 旧工作流

本插件只注册主加载器和主 Guide 两个节点。旧版三个 `ComfyUILTX25MSRAVref...` 节点已删除，不保留兼容注册。旧 AVref 工作流需替换为上面的主节点和原生音频编码器连接方式，参数统一为 `LTX_MSR_REFERENCE_PARAMETERS`。独立 `ComfyUI-LTX2.5-MSR-AVref.disabled` 目录继续停用。

安装后完整重启 ComfyUI。
