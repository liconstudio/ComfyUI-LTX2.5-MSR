import logging

import comfy.sd
import comfy.utils
import comfy_extras.nodes_lt as nodes_lt
import folder_paths
import node_helpers
import torch
from comfy_api.latest import io


MSRReferenceParameters = io.Custom("LTX_MSR_REFERENCE_PARAMETERS")
_SLOT_PREFIXES = (
    "diffusion_model.reference_slot_embedding.",
    "reference_slot_embedding.",
)


def _metadata_bool(metadata, key, default=False):
    value = metadata.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _extract_slot_state(lora):
    state = {}
    normal_lora = {}
    for key, value in lora.items():
        matched = False
        for prefix in _SLOT_PREFIXES:
            if key.startswith(prefix):
                state[key[len(prefix) :]] = value.detach().cpu()
                matched = True
                break
        if not matched:
            normal_lora[key] = value
    return normal_lora, state


def _validate_slot_state(state, metadata):
    required = {
        "frequencies",
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
    }
    missing = sorted(required.difference(state))
    enabled = _metadata_bool(metadata, "reference_slot_embedding_enabled", bool(state))
    if enabled and missing:
        raise ValueError(
            "MSR LoRA declares reference slot embeddings, but these tensors are missing: "
            + ", ".join(missing)
        )
    if not state:
        raise ValueError(
            "This LoRA does not contain reference_slot_embedding weights and is not an "
            "MSR multi-reference checkpoint."
        )


def _slot_embedding(slot_id, state, device, dtype):
    frequencies = state["frequencies"].to(device=device, dtype=torch.float32)
    slot_value = torch.tensor(float(slot_id), device=device, dtype=torch.float32)
    scaled = slot_value / 16.0
    phases = scaled * frequencies
    features = torch.cat((scaled.reshape(1), torch.sin(phases), torch.cos(phases)))

    weight0 = state["net.0.weight"].to(device=device, dtype=torch.float32)
    bias0 = state["net.0.bias"].to(device=device, dtype=torch.float32)
    hidden = torch.nn.functional.silu(torch.nn.functional.linear(features, weight0, bias0))
    weight2 = state["net.2.weight"].to(device=device, dtype=torch.float32)
    bias2 = state["net.2.bias"].to(device=device, dtype=torch.float32)
    embedding = torch.nn.functional.linear(hidden, weight2, bias2)
    return embedding.to(dtype=dtype)


def _conditioning_get(conditioning, key, default=None):
    for _, values in conditioning:
        if key in values:
            return values[key]
    return default


def _append_attention_entry(conditioning, pre_filter_count, latent_shape, strength):
    existing = _conditioning_get(conditioning, "guide_attention_entries", [])
    entry = {
        "pre_filter_count": pre_filter_count,
        "strength": strength,
        "pixel_mask": None,
        "latent_shape": latent_shape,
    }
    return node_helpers.conditioning_set_values(
        conditioning, {"guide_attention_entries": [*existing, entry]}
    )


class ComfyUILTX25MSRICLoRALoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUILTX25MSRICLoRALoader",
            display_name="ComfyUI-LTX2.5-MSR IC-LoRA Loader",
            category="ComfyUI-LTX2.5-MSR",
            description=(
                "Loads an LTX MSR LoRA and extracts its learned Fourier-MLP reference "
                "slot embedding for the Multi-Reference Guide node."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input(
                    "strength_model", default=1.0, min=-100.0, max=100.0, step=0.01
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                MSRReferenceParameters.Output("msr_parameters"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(
            lora_path, safe_load=True, return_metadata=True
        )
        metadata = metadata or {}
        normal_lora, slot_state = _extract_slot_state(lora)
        _validate_slot_state(slot_state, metadata)

        if strength_model != 0:
            loaded_model, _ = comfy.sd.load_lora_for_models(
                model,
                None,
                normal_lora,
                strength_model,
                0,
                lora_metadata=metadata,
            )
        else:
            loaded_model = model

        params = {
            "slot_state": slot_state,
            "metadata": dict(metadata),
            "lora_name": lora_name,
            "reference_downscale_factor": max(
                1, round(float(metadata.get("reference_downscale_factor", 1)))
            ),
            # ComfyUI compatibility mode intentionally uses its established
            # guide coordinates for every checkpoint, including LoRAs whose
            # training metadata records another temporal scale.
            "reference_temporal_scale_factor": 1,
        }
        logging.info(
            "[LTX MSR] Loaded %s with learned reference slot embedding (%d tensors)",
            lora_name,
            len(slot_state),
        )
        logging.info(
            "[LTX MSR] Parameters: slot_output_dim=%d, reference_downscale_factor=%d, "
            "reference_temporal_scale_factor=%d (compatibility default), "
            "token_order=%s, time_offsets=%s",
            int(slot_state["net.2.bias"].numel()),
            params["reference_downscale_factor"],
            params["reference_temporal_scale_factor"],
            metadata.get("reference_token_order", "prepend"),
            metadata.get("reference_slot_time_offsets", "pic1_based_negative_time"),
        )
        return io.NodeOutput(loaded_model, params)


class ComfyUILTX25MSRMultiReferenceGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUILTX25MSRMultiReferenceGuide",
            display_name="ComfyUI-LTX2.5-MSR Multi-Reference Guide",
            category="ComfyUI-LTX2.5-MSR",
            description=(
                "Encodes pic1...pic4 and an optional background independently, applies the learned "
                "slot embeddings, and appends clean guide tokens at the negative temporal "
                "positions used during MSR training."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Latent.Input("latent"),
                io.Image.Input("pic1"),
                io.Image.Input("pic2", optional=True),
                io.Image.Input("pic3", optional=True),
                io.Image.Input("pic4", optional=True),
                io.Image.Input("background", optional=True),
                MSRReferenceParameters.Input(
                    "msr_parameters",
                    optional=True,
                    tooltip=(
                        "Optional MSR parameters from the LTX MSR IC-LoRA Loader. "
                        "When omitted, references are appended as standard LTX guides "
                        "without MSR slot embeddings or negative temporal offsets."
                    ),
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("reference_frames", options=["25", "33"], default="33"),
                io.Boolean.Input("use_tiled_encode", default=False),
                # min=0 keeps older workflows loadable after the crop widget was
                # removed; execute() migrates the shifted legacy values below.
                io.Int.Input("tile_size", default=256, min=0, max=512, step=32),
                io.Int.Input("tile_overlap", default=64, min=0, max=256, step=16),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("latent"),
            ],
        )

    @staticmethod
    def _collect_references(pic1, pic2, pic3, pic4, background):
        refs = [
            ("pic1", pic1, False),
            ("pic2", pic2, False),
            ("pic3", pic3, False),
            ("pic4", pic4, False),
            ("background", background, True),
        ]
        selected = []
        for label, image, is_background in refs:
            if image is None:
                continue
            if image.shape[0] != 1:
                raise ValueError(
                    f"{label} must be one still image, but received {image.shape[0]} frames."
                )
            selected.append((label, image, is_background))
        return selected

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent,
        pic1,
        strength,
        reference_frames,
        use_tiled_encode,
        tile_size,
        tile_overlap,
        msr_parameters=None,
        pic2=None,
        pic3=None,
        pic4=None,
        background=None,
        **legacy_inputs,
    ):
        # Keep old workflows executable without exposing pic5 in the new UI.
        legacy_pic5 = legacy_inputs.pop("pic5", None)
        if legacy_inputs:
            unexpected = ", ".join(sorted(legacy_inputs))
            raise TypeError(f"Unexpected input(s): {unexpected}")
        if legacy_pic5 is not None:
            if background is None:
                background = legacy_pic5
                logging.warning(
                    "[LTX MSR] Migrated legacy input pic5 -> background"
                )
            else:
                logging.warning(
                    "[LTX MSR] Both legacy pic5 and background were provided; "
                    "using background"
                )
        references = cls._collect_references(pic1, pic2, pic3, pic4, background)
        num_slots = len(references)
        if not 1 <= num_slots <= 5:
            raise ValueError(f"MSR requires 1-5 references, got {num_slots}.")
        reference_frames = int(reference_frames)
        if reference_frames not in (25, 33):
            raise ValueError(
                f"reference_frames must be 25 or 33, got {reference_frames}."
            )

        # Workflows saved before the crop widget was removed still contain its
        # value in the positional widget list. In those workflows the old crop
        # string lands in use_tiled_encode and the old boolean lands in tile_size.
        if (
            not isinstance(use_tiled_encode, bool)
            and tile_size == 0
            and isinstance(tile_overlap, (int, float))
            and tile_overlap >= 64
        ):
            logging.warning(
                "[LTX MSR] Migrating legacy widget layout: crop=%s; "
                "resetting use_tiled_encode=false, tile_size=256, tile_overlap=64",
                use_tiled_encode,
            )
            use_tiled_encode = False
            tile_size = 256
            tile_overlap = 64
        elif tile_size < 64:
            logging.warning(
                "[LTX MSR] Invalid tile_size=%s; using default tile_size=256",
                tile_size,
            )
            tile_size = 256
        elif tile_overlap < 16:
            logging.warning(
                "[LTX MSR] Invalid tile_overlap=%s; using default tile_overlap=64",
                tile_overlap,
            )
            tile_overlap = 64

        msr_enabled = msr_parameters is not None
        metadata = msr_parameters["metadata"] if msr_enabled else {}
        if msr_enabled:
            if metadata.get("reference_token_order", "prepend") != "prepend":
                raise ValueError("Unsupported reference_token_order; expected 'prepend'.")
            offsets = metadata.get("reference_slot_time_offsets", "pic1_based_negative_time")
            if offsets != "pic1_based_negative_time":
                raise ValueError(
                    "Unsupported reference_slot_time_offsets; expected 'pic1_based_negative_time'."
                )

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = nodes_lt.get_noise_mask(latent)
        if latent_image.ndim != 5 or latent_image.shape[1] != 128:
            raise ValueError(
                "The guide node needs a video-only LTX latent shaped [B, 128, F, H, W]. "
                "Connect it before LTXVConcatAVLatent."
            )
        if latent_image.shape[0] != 1:
            raise ValueError("MSR multi-reference inference currently requires batch_size=1.")

        _, _, _, latent_height, latent_width = latent_image.shape
        downscale = msr_parameters["reference_downscale_factor"] if msr_enabled else 1
        temporal_scale = 1
        if latent_height % downscale or latent_width % downscale:
            raise ValueError(
                f"Target latent grid {latent_width}x{latent_height} is not divisible by "
                f"reference_downscale_factor={downscale}."
            )

        slot_state = msr_parameters["slot_state"] if msr_enabled else None
        logging.info(
            "[LTX MSR] Guide start: mode=%s, references=%d, reference_frames=%d, "
            "downscale=%d, temporal_scale=%s, target_latent=%s",
            "MSR" if msr_enabled else "standard fallback",
            num_slots,
            reference_frames,
            downscale,
            temporal_scale,
            tuple(latent_image.shape),
        )
        for slot_index, (label, image, is_background) in enumerate(references):
            slot_id = slot_index + 1
            repeated = image.repeat(reference_frames, 1, 1, 1)
            _, guide_latent = cls._encode_reference(
                vae,
                latent_width,
                latent_height,
                repeated,
                scale_factors,
                downscale,
                is_background,
                use_tiled_encode,
                tile_size,
                tile_overlap,
            )
            embedding_dim = 0
            embedding_norm = 0.0
            if msr_enabled:
                embedding = _slot_embedding(
                    slot_id, slot_state, guide_latent.device, guide_latent.dtype
                )
                if embedding.numel() != guide_latent.shape[1]:
                    raise ValueError(
                        f"Slot embedding dimension {embedding.numel()} does not match "
                        f"LTX latent channels {guide_latent.shape[1]}."
                    )
                embedding_dim = embedding.numel()
                embedding_norm = embedding.detach().float().norm().item()
                embedding_preview = [
                    round(value, 6)
                    for value in embedding.detach().float().flatten()[:8].tolist()
                ]
                guide_latent = guide_latent + embedding.view(1, -1, 1, 1, 1)
                logging.warning(
                    "[LTX MSR][VERIFY] slot_embedding=APPLIED label=%s slot_id=%d "
                    "operation=guide_latent_plus_broadcast_embedding embedding_dim=%d "
                    "embedding_norm=%.6f embedding_preview=%s",
                    label,
                    slot_id,
                    embedding_dim,
                    embedding_norm,
                    embedding_preview,
                )
            else:
                logging.warning(
                    "[LTX MSR][VERIFY] slot_embedding=SKIPPED label=%s "
                    "reason=msr_parameters_not_connected",
                    label,
                )
            original_shape = list(guide_latent.shape[2:])

            guide_mask = None
            if downscale > 1:
                guide_latent, guide_mask = nodes_lt.LTXVAddGuide.dilate_latent(
                    guide_latent, downscale
                )

            # MSR uses learned pic slots at distinct negative temporal positions.
            # Fallback mode follows the standard LTX guide convention: every
            # independent reference starts at frame 0, with no MSR time offset.
            frame_offset = -(num_slots - slot_index) if msr_enabled else 0
            logging.info(
                "[LTX MSR] %s: slot_embedding=%s, slot_id=%s, embedding_dim=%d, "
                "embedding_norm=%.6f, time_offset=%d, guide_latent=%s",
                label,
                "applied" if msr_enabled else "skipped",
                slot_id if msr_enabled else "none",
                embedding_dim,
                embedding_norm,
                frame_offset,
                tuple(guide_latent.shape),
            )
            positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                positive,
                negative,
                frame_offset,
                latent_image,
                noise_mask,
                guide_latent,
                strength,
                scale_factors,
                guide_mask=guide_mask,
                latent_downscale_factor=downscale,
                causal_fix=True,
            )
            logging.warning(
                "[LTX MSR][VERIFY] negative_time_offset=%s label=%s slot_id=%d "
                "operation=append_keyframe frame_offset=%d "
                "temporal_scale=%g compatibility_default=APPLIED "
                "temporal_position=negative_%d "
                "append_keyframe=SUCCESS",
                "APPLIED" if msr_enabled else "SKIPPED",
                label,
                slot_id,
                frame_offset,
                temporal_scale,
                abs(frame_offset),
            )
            token_count = (
                guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
            )
            positive = _append_attention_entry(
                positive, token_count, original_shape, strength
            )
            negative = _append_attention_entry(
                negative, token_count, original_shape, strength
            )

        logging.info(
            "[LTX MSR] Guide complete: added=%d, order=pic1..pic%d, frames_each=%d, "
            "mode=%s, output_latent=%s",
            num_slots,
            num_slots,
            reference_frames,
            "MSR" if msr_enabled else "standard fallback",
            tuple(latent_image.shape),
        )
        return io.NodeOutput(
            positive,
            negative,
            {"samples": latent_image, "noise_mask": noise_mask},
        )

    @staticmethod
    def _encode_reference(
        vae,
        latent_width,
        latent_height,
        images,
        scale_factors,
        downscale,
        is_background,
        use_tiled_encode,
        tile_size,
        tile_overlap,
    ):
        time_scale, width_scale, height_scale = scale_factors
        keep = ((images.shape[0] - 1) // time_scale) * time_scale + 1
        images = images[:keep]
        target_width = int(latent_width * width_scale / downscale)
        target_height = int(latent_height * height_scale / downscale)
        pixels = ComfyUILTX25MSRMultiReferenceGuide._resize_reference(
            images, target_width, target_height, is_background
        )
        pixels = pixels[..., :3]
        if use_tiled_encode:
            guide_latent = vae.encode_tiled(
                pixels, tile_x=tile_size, tile_y=tile_size, overlap=tile_overlap
            )
        else:
            guide_latent = vae.encode(pixels)
        return pixels, guide_latent

    @staticmethod
    def _resize_reference(images, target_width, target_height, is_background):
        if is_background:
            return comfy.utils.common_upscale(
                images.movedim(-1, 1),
                target_width,
                target_height,
                "bilinear",
                crop="center",
            ).movedim(1, -1)

        source_height, source_width = images.shape[1:3]
        if source_width == target_width and source_height == target_height:
            return images

        def aspect_family(width, height):
            ratio = width / height
            if ratio >= 1.25:
                return "landscape"
            if ratio <= 0.8:
                return "portrait"
            return "square"

        same_family = aspect_family(source_width, source_height) == aspect_family(
            target_width, target_height
        )
        source_is_smaller = source_width <= target_width and source_height <= target_height

        # Similar aspect ratios: enlarge smaller references with white padding;
        # reduce larger references to cover the target and center-crop overflow.
        # Dissimilar aspect ratios: always preserve the complete image with white
        # padding, regardless of relative size.
        if same_family and not source_is_smaller:
            return comfy.utils.common_upscale(
                images.movedim(-1, 1),
                target_width,
                target_height,
                "bilinear",
                crop="center",
            ).movedim(1, -1)

        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, min(target_width, round(source_width * scale)))
        resized_height = max(1, min(target_height, round(source_height * scale)))
        resized = comfy.utils.common_upscale(
            images.movedim(-1, 1),
            resized_width,
            resized_height,
            "bilinear",
            crop="disabled",
        ).movedim(1, -1)
        canvas = torch.ones(
            (images.shape[0], target_height, target_width, images.shape[-1]),
            dtype=images.dtype,
            device=images.device,
        )
        left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2
        canvas[:, top : top + resized_height, left : left + resized_width] = resized
        return canvas


NODE_CLASS_MAPPINGS = {
    "ComfyUILTX25MSRICLoRALoader": ComfyUILTX25MSRICLoRALoader,
    "ComfyUILTX25MSRMultiReferenceGuide": ComfyUILTX25MSRMultiReferenceGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUILTX25MSRICLoRALoader": "ComfyUI-LTX2.5-MSR IC-LoRA Loader",
    "ComfyUILTX25MSRMultiReferenceGuide": "ComfyUI-LTX2.5-MSR Multi-Reference Guide",
}
