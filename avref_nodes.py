import logging
import types

import comfy.sd
import comfy.utils
import comfy_extras.nodes_lt as nodes_lt
import folder_paths
import node_helpers
import torch
import torchaudio
from comfy_api.latest import io
from comfy.patcher_extension import CallbacksMP


MSRAVRefParameters = io.Custom("LTX_MSR_AVREF_PARAMETERS")
MSRAVRefAudioReferences = io.Custom("LTX_MSR_AVREF_AUDIO_REFERENCES")
_IMAGE_SLOT_PREFIXES = (
    "diffusion_model.reference_slot_embedding.",
    "reference_slot_embedding.",
)
_AUDIO_SLOT_PREFIXES = (
    "diffusion_model.reference_audio_slot_embedding.",
    "reference_audio_slot_embedding.",
)
_AVREF_LAYOUT_MARKER = "ltx_msr_avref_absolute_slots_v1"
_PROCESS_INPUT_PATCH_MARKER = "_ltx_msr_avref_process_input_patch"
_PROCESS_INPUT_PATCH_PATH = "diffusion_model._process_input"
_REBIND_CALLBACK_KEY = "ltx_msr_avref_rebind_process_input"


def _metadata_bool(metadata, key, default=False):
    value = metadata.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _extract_slot_states(lora):
    image_state = {}
    audio_state = {}
    normal_lora = {}
    for key, value in lora.items():
        matched = False
        for prefix in _IMAGE_SLOT_PREFIXES:
            if key.startswith(prefix):
                image_state[key[len(prefix) :]] = value.detach().cpu()
                matched = True
                break
        if not matched:
            for prefix in _AUDIO_SLOT_PREFIXES:
                if key.startswith(prefix):
                    audio_state[key[len(prefix) :]] = value.detach().cpu()
                    matched = True
                    break
        if not matched:
            normal_lora[key] = value
    return normal_lora, image_state, audio_state


def _validate_slot_state(state, metadata, *, audio=False):
    required = {
        "frequencies",
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
    }
    missing = sorted(required.difference(state))
    prefix = "reference_audio" if audio else "reference"
    label = "audio reference" if audio else "image reference"
    enabled = _metadata_bool(metadata, f"{prefix}_slot_embedding_enabled", bool(state))
    if enabled and missing:
        raise ValueError(
            f"The MSR-AVref LoRA declares {label} slot embeddings, but these tensors "
            "are missing: "
            + ", ".join(missing)
        )
    if not state:
        raise ValueError(
            f"This LoRA does not contain {prefix}_slot_embedding weights and is not a "
            "compatible MSR-AVref checkpoint."
        )
    if missing:
        raise ValueError(f"Incomplete {label} slot embedding: " + ", ".join(missing))

    frequencies = state["frequencies"]
    weight0 = state["net.0.weight"]
    bias0 = state["net.0.bias"]
    weight2 = state["net.2.weight"]
    bias2 = state["net.2.bias"]
    expected_features = 1 + 2 * frequencies.numel()
    if frequencies.ndim != 1 or weight0.ndim != 2 or weight0.shape[1] != expected_features:
        raise ValueError(f"Invalid {label} Fourier-MLP input shape.")
    if bias0.shape != weight0.shape[:1] or weight2.ndim != 2 or weight2.shape[1] != weight0.shape[0]:
        raise ValueError(f"Invalid {label} Fourier-MLP hidden shape.")
    if bias2.shape != weight2.shape[:1]:
        raise ValueError(f"Invalid {label} Fourier-MLP output shape.")

    metadata_dim = metadata.get(f"{prefix}_slot_embedding_dim")
    if metadata_dim is not None and int(float(metadata_dim)) != bias2.numel():
        raise ValueError(
            f"{label.title()} slot dimension in metadata ({metadata_dim}) does not match "
            f"the checkpoint ({bias2.numel()})."
        )
    metadata_type = metadata.get(f"{prefix}_slot_embedding_type")
    if metadata_type is not None and str(metadata_type).strip() != "fourier_mlp":
        raise ValueError(
            f"Unsupported {label} slot embedding type {metadata_type!r}; "
            "expected 'fourier_mlp'."
        )


def _metadata_float(metadata, key, default):
    try:
        return float(metadata.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric LoRA metadata {key}={metadata.get(key)!r}.") from exc


def _validate_avref_metadata(metadata):
    expected = {
        "reference_token_order": "prepend",
        "reference_slot_time_offsets": "pic1_based_negative_time",
        "reference_audio_conditioning": "id_lora_clean_negative_rope",
        "reference_audio_token_order": "pic1_to_picN_then_target",
        "reference_audio_rope_layout": "absolute_image_slot_windows",
        "reference_audio_overflow_mode": "truncate",
    }
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual is None:
            raise ValueError(
                f"The selected LoRA is missing required MSR-AVref metadata: {key}."
            )
        if str(actual).strip() != expected_value:
            raise ValueError(
                f"Unsupported {key}={actual!r}; expected {expected_value!r}."
            )
    if not _metadata_bool(metadata, "reference_audio_sparse_slots", False):
        raise ValueError("The selected LoRA does not declare sparse absolute audio slots.")
    if not _metadata_bool(metadata, "reference_slot_embedding_enabled", False):
        raise ValueError("The selected LoRA does not enable image reference slot embeddings.")
    if not _metadata_bool(metadata, "reference_audio_slot_embedding_enabled", False):
        raise ValueError("The selected LoRA does not enable audio reference slot embeddings.")

    slot_duration = _metadata_float(
        metadata, "reference_audio_slot_duration_seconds", 5.0
    )
    end_margin = _metadata_float(
        metadata, "reference_audio_end_margin_seconds", 0.04
    )
    if slot_duration <= 0 or end_margin <= 0:
        raise ValueError("Audio slot duration and end margin must both be positive.")
    return slot_duration, end_margin


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


def _maximum_audio_tokens(patchifier, slot_duration):
    if not getattr(patchifier, "start_end", False):
        raise ValueError("LTX MSR-AVref requires an AudioPatchifier with start/end coordinates.")
    token_seconds = (
        patchifier.hop_length
        * patchifier.audio_latent_downsample_factor
        / patchifier.sample_rate
    )
    probe_count = max(8, int(slot_duration / token_seconds) + 8)
    ends = patchifier._get_audio_latent_time_in_sec(
        1, probe_count + 1, torch.float32, torch.device("cpu")
    )
    return max(1, int((ends <= slot_duration + 1e-6).sum().item()))


def _is_avref_process_input_patch(value):
    function = getattr(value, "__func__", value)
    return bool(getattr(function, _PROCESS_INPUT_PATCH_MARKER, False))


def _make_avref_process_input_patch(original_process_input):
    if isinstance(original_process_input, types.MethodType):
        original_callable = original_process_input.__func__
        original_needs_model = True
    else:
        original_callable = original_process_input
        original_needs_model = False

    def process_input(diffusion_model, x, keyframe_idxs, denoise_mask, **kwargs):
        ref_audio = kwargs.get("ref_audio")
        if not isinstance(ref_audio, dict) or ref_audio.get("layout_marker") != _AVREF_LAYOUT_MARKER:
            if original_needs_model:
                return original_callable(
                    diffusion_model, x, keyframe_idxs, denoise_mask, **kwargs
                )
            return original_callable(x, keyframe_idxs, denoise_mask, **kwargs)

        if original_needs_model:
            result = original_callable(
                diffusion_model, x, keyframe_idxs, denoise_mask, **kwargs
            )
        else:
            result = original_callable(x, keyframe_idxs, denoise_mask, **kwargs)
        processed_latents, processed_positions, additional_args = result
        vx, ax = processed_latents
        video_positions, audio_positions = processed_positions

        slot_ids = tuple(int(value) for value in ref_audio.get("slot_ids", ()))
        slot_lengths = tuple(int(value) for value in ref_audio.get("slot_lengths", ()))
        image_slot_count = int(ref_audio.get("image_slot_count", 0))
        slot_duration = float(ref_audio.get("slot_duration_seconds", 0.0))
        end_margin = float(ref_audio.get("end_margin_seconds", 0.0))
        if not slot_ids or len(slot_ids) != len(slot_lengths):
            raise ValueError("Invalid MSR-AVref audio slot metadata.")
        if tuple(sorted(set(slot_ids))) != slot_ids:
            raise ValueError("MSR-AVref audio slots must be unique and in ascending order.")
        if any(length <= 0 for length in slot_lengths):
            raise ValueError("MSR-AVref audio slot lengths must be positive.")
        if image_slot_count < 1 or any(slot_id < 1 or slot_id > image_slot_count for slot_id in slot_ids):
            raise ValueError("A connected audio reference has no matching image slot.")
        if slot_duration <= 0 or end_margin <= 0:
            raise ValueError("Invalid MSR-AVref audio slot timing metadata.")

        native_ref_length = int(additional_args.get("ref_audio_seq_len", 0))
        if native_ref_length != sum(slot_lengths):
            raise ValueError(
                "The native LTX reference-audio length does not match the MSR-AVref payload."
            )
        if audio_positions.shape[2] < native_ref_length or ax.shape[1] < native_ref_length:
            raise ValueError("The native LTX model returned an invalid reference-audio prefix.")

        batch_size = ax.shape[0]
        cursor = 0
        reference_latents = []
        reference_positions = []
        for slot_id, slot_length in zip(slot_ids, slot_lengths):
            block = ax[:, cursor : cursor + slot_length]
            dummy = torch.empty(
                (batch_size, 1, slot_length, 1),
                device=ax.device,
                dtype=torch.float32,
            )
            _, local_positions = diffusion_model.a_patchifier.patchify(dummy)
            local_end = local_positions[:, :, -1, 1]
            slot_end = -((image_slot_count - slot_id) * slot_duration) - end_margin
            shift = slot_end - local_end
            local_positions = local_positions + shift[:, :, None, None]
            reference_latents.append(block)
            reference_positions.append(local_positions.to(audio_positions))
            cursor += slot_length

        target_ax = ax[:, native_ref_length:]
        target_positions = audio_positions[:, :, native_ref_length:]
        ax = torch.cat([*reference_latents, target_ax], dim=1)
        audio_positions = torch.cat([*reference_positions, target_positions], dim=2)
        additional_args = dict(additional_args)
        additional_args["ref_audio_seq_len"] = sum(slot_lengths)
        additional_args["target_audio_seq_len"] = target_ax.shape[1]
        return [vx, ax], [video_positions, audio_positions], additional_args

    setattr(process_input, _PROCESS_INPUT_PATCH_MARKER, True)
    return process_input


def _rebind_avref_process_input_on_clone(source_model, cloned_model):
    if source_model.model is cloned_model.model:
        return
    existing = cloned_model.object_patches.get(_PROCESS_INPUT_PATCH_PATH)
    if not _is_avref_process_input_patch(existing):
        return
    diffusion_model = cloned_model.get_model_object("diffusion_model")
    function = getattr(existing, "__func__", existing)
    cloned_model.object_patches[_PROCESS_INPUT_PATCH_PATH] = types.MethodType(
        function, diffusion_model
    )


def _install_avref_process_input_patch(model):
    patched_model = model.clone()
    diffusion_model = patched_model.get_model_object("diffusion_model")
    patchifier = getattr(diffusion_model, "a_patchifier", None)
    if patchifier is None:
        raise ValueError("The loaded model is not an LTX audio/video diffusion model.")

    current = patched_model.get_model_object(_PROCESS_INPUT_PATCH_PATH)
    if not _is_avref_process_input_patch(current):
        function = _make_avref_process_input_patch(current)
        patched_model.add_object_patch(
            _PROCESS_INPUT_PATCH_PATH, types.MethodType(function, diffusion_model)
        )
    if not patched_model.get_callbacks(CallbacksMP.ON_CLONE, _REBIND_CALLBACK_KEY):
        patched_model.add_callback_with_key(
            CallbacksMP.ON_CLONE,
            _REBIND_CALLBACK_KEY,
            _rebind_avref_process_input_on_clone,
        )
    return patched_model, patchifier


class ComfyUILTX25MSRAVrefICLoRALoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUILTX25MSRAVrefICLoRALoader",
            display_name="ComfyUI-LTX2.5-MSR-AVref IC-LoRA Loader",
            category="ComfyUI-LTX2.5-MSR-AVref",
            description=(
                "Loads an LTX MSR-AVref LoRA, extracts its independent image/audio "
                "slot embeddings, and installs the trained absolute audio-slot layout."
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
                MSRAVRefParameters.Output("avref_parameters"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name, strength_model):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(
            lora_path, safe_load=True, return_metadata=True
        )
        metadata = metadata or {}
        normal_lora, image_slot_state, audio_slot_state = _extract_slot_states(lora)
        _validate_slot_state(image_slot_state, metadata, audio=False)
        _validate_slot_state(audio_slot_state, metadata, audio=True)
        slot_duration, end_margin = _validate_avref_metadata(metadata)

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

        loaded_model, patchifier = _install_avref_process_input_patch(loaded_model)
        timing = {
            "sample_rate": int(patchifier.sample_rate),
            "hop_length": int(patchifier.hop_length),
            "audio_latent_downsample_factor": int(
                patchifier.audio_latent_downsample_factor
            ),
            "is_causal": bool(patchifier.is_causal),
            "start_end": bool(patchifier.start_end),
            "shift": int(patchifier.shift),
            "patch_size": tuple(int(value) for value in patchifier.patch_size),
        }
        expected_timing = {
            "sample_rate": 16000,
            "hop_length": 160,
            "audio_latent_downsample_factor": 4,
            "is_causal": True,
            "start_end": True,
            "shift": 0,
            "patch_size": (1, 1, 1),
        }
        if timing != expected_timing:
            raise ValueError(
                f"Unsupported LTX audio patchifier timing {timing}; expected {expected_timing}."
            )
        max_audio_tokens = _maximum_audio_tokens(patchifier, slot_duration)

        params = {
            "image_slot_state": image_slot_state,
            "audio_slot_state": audio_slot_state,
            "metadata": dict(metadata),
            "lora_name": lora_name,
            "reference_downscale_factor": max(
                1, round(float(metadata.get("reference_downscale_factor", 1)))
            ),
            # ComfyUI compatibility mode intentionally uses its established
            # guide coordinates for every checkpoint, including LoRAs whose
            # training metadata records another temporal scale.
            "reference_temporal_scale_factor": 1,
            "reference_audio_slot_duration_seconds": slot_duration,
            "reference_audio_end_margin_seconds": end_margin,
            "reference_audio_max_tokens": max_audio_tokens,
            "audio_patchifier_timing": timing,
        }
        logging.info(
            "[LTX MSR-AVref] Loaded %s with image/audio slot embeddings (%d/%d tensors)",
            lora_name,
            len(image_slot_state),
            len(audio_slot_state),
        )
        logging.info(
            "[LTX MSR-AVref] image_dim=%d audio_dim=%d downscale=%d "
            "audio_layout=absolute_image_slot_windows duration=%.3fs margin=%.3fs "
            "max_audio_tokens=%d",
            int(image_slot_state["net.2.bias"].numel()),
            int(audio_slot_state["net.2.bias"].numel()),
            params["reference_downscale_factor"],
            slot_duration,
            end_margin,
            max_audio_tokens,
        )
        return io.NodeOutput(loaded_model, params)


class ComfyUILTX25MSRAVrefAudioEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUILTX25MSRAVrefAudioEncoder",
            display_name="ComfyUI-LTX2.5-MSR-AVref Three Audio Reference Encoder",
            category="ComfyUI-LTX2.5-MSR-AVref",
            description=(
                "Encodes up to three sparse reference-audio clips. audio_ref1/2/3 "
                "map to pic1/2/3 without compacting missing slots."
            ),
            inputs=[
                io.Vae.Input(
                    id="audio_vae",
                    display_name="Audio VAE",
                    tooltip="Native LTX-2.5 Audio VAE.",
                ),
                MSRAVRefParameters.Input("avref_parameters"),
                io.Audio.Input(
                    "audio_ref1",
                    optional=True,
                    tooltip="Reference audio for pic1. Leave disconnected when pic1 has no audio.",
                ),
                io.Audio.Input(
                    "audio_ref2",
                    optional=True,
                    tooltip="Reference audio for pic2. Leave disconnected when pic2 has no audio.",
                ),
            ],
            outputs=[MSRAVRefAudioReferences.Output("audio_references")],
        )

    @classmethod
    def execute(
        cls,
        audio_vae,
        avref_parameters,
        audio_ref1=None,
        audio_ref2=None,
        audio_ref3=None,
    ):
        selected = [
            (slot_id, audio)
            for slot_id, audio in enumerate(
                (audio_ref1, audio_ref2, audio_ref3), start=1
            )
            if audio is not None
        ]
        if not selected:
            raise ValueError("Connect at least one of audio_ref1, audio_ref2, or audio_ref3.")

        max_tokens = int(avref_parameters["reference_audio_max_tokens"])
        audio_slot_state = avref_parameters["audio_slot_state"]
        blocks = []
        for slot_id, audio in selected:
            tokens, original_tokens = cls._encode_audio(audio_vae, audio)
            if tokens.shape[1] > max_tokens:
                tokens = tokens[:, :max_tokens]
                logging.warning(
                    "[LTX MSR-AVref] audio_ref%d truncated from %d to %d tokens "
                    "to fit its %.3fs slot.",
                    slot_id,
                    original_tokens,
                    max_tokens,
                    avref_parameters["reference_audio_slot_duration_seconds"],
                )
            embedding = _slot_embedding(
                slot_id, audio_slot_state, tokens.device, tokens.dtype
            )
            if embedding.numel() != tokens.shape[-1]:
                raise ValueError(
                    f"Audio slot embedding dimension {embedding.numel()} does not match "
                    f"audio_ref{slot_id} token dimension {tokens.shape[-1]}."
                )
            tokens = tokens + embedding.view(1, 1, -1)
            blocks.append({"slot_id": slot_id, "tokens": tokens})
            logging.info(
                "[LTX MSR-AVref] audio_ref%d encoded: tokens=%d slot_embedding=applied",
                slot_id,
                tokens.shape[1],
            )

        payload = {
            "layout_marker": _AVREF_LAYOUT_MARKER,
            "blocks": tuple(blocks),
            "lora_name": avref_parameters["lora_name"],
            "slot_duration_seconds": avref_parameters[
                "reference_audio_slot_duration_seconds"
            ],
            "end_margin_seconds": avref_parameters[
                "reference_audio_end_margin_seconds"
            ],
        }
        return io.NodeOutput(payload)

    @staticmethod
    def _encode_audio(audio_vae, audio):
        waveform = audio.get("waveform")
        sample_rate = int(audio.get("sample_rate", 0))
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
            raise ValueError("Reference AUDIO must contain waveform shaped [B, C, samples].")
        if waveform.shape[0] != 1:
            raise ValueError(
                f"MSR-AVref inference requires batch_size=1; got {waveform.shape[0]}."
            )
        if waveform.shape[-1] < 1 or sample_rate <= 0:
            raise ValueError("Reference audio is empty or has an invalid sample rate.")

        vae_sample_rate = int(getattr(audio_vae, "audio_sample_rate", 44100))
        if sample_rate != vae_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, vae_sample_rate
            )
        audio_latents = audio_vae.encode(waveform.movedim(1, -1))
        if not isinstance(audio_latents, torch.Tensor) or audio_latents.ndim != 4:
            raise ValueError("The LTX Audio VAE must return [B, C, T, F] latents.")
        batch, channels, time_steps, frequency_bins = audio_latents.shape
        if batch != 1 or time_steps < 1:
            raise ValueError("The LTX Audio VAE returned an empty or batched reference.")
        tokens = audio_latents.permute(0, 2, 1, 3).reshape(
            batch, time_steps, channels * frequency_bins
        )
        return tokens, time_steps


class ComfyUILTX25MSRAVrefMultiReferenceGuide(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyUILTX25MSRAVrefMultiReferenceGuide",
            display_name="ComfyUI-LTX2.5-MSR-AVref Multi-Reference Guide",
            category="ComfyUI-LTX2.5-MSR-AVref",
            description=(
                "Encodes 25-frame image references and attaches sparse audio_ref1/2/3 "
                "to the same numbered pic slots used during MSR-AVref training."
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
                MSRAVRefParameters.Input(
                    "avref_parameters",
                    tooltip="Parameters from the MSR-AVref IC-LoRA Loader.",
                ),
                MSRAVRefAudioReferences.Input(
                    "audio_references",
                    optional=True,
                    tooltip=(
                        "Optional payload from the Three Audio Reference Encoder. "
                        "Each connected audio_refN maps to picN."
                    ),
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("reference_frames", options=["25"], default="25"),
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
        avref_parameters,
        audio_references=None,
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
        if reference_frames != 25:
            raise ValueError(f"MSR-AVref reference_frames must be 25, got {reference_frames}.")

        if audio_references is not None:
            cls._validate_audio_image_mapping(
                audio_references,
                avref_parameters,
                pic1,
                pic2,
                pic3,
                pic4,
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

        metadata = avref_parameters["metadata"]
        if metadata.get("reference_token_order") != "prepend":
            raise ValueError("Unsupported reference_token_order; expected 'prepend'.")
        if metadata.get("reference_slot_time_offsets") != "pic1_based_negative_time":
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
        downscale = avref_parameters["reference_downscale_factor"]
        temporal_scale = 1
        if latent_height % downscale or latent_width % downscale:
            raise ValueError(
                f"Target latent grid {latent_width}x{latent_height} is not divisible by "
                f"reference_downscale_factor={downscale}."
            )

        slot_state = avref_parameters["image_slot_state"]
        logging.info(
            "[LTX MSR-AVref] Guide start: references=%d, reference_frames=%d, "
            "downscale=%d, temporal_scale=%s, target_latent=%s",
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
            guide_latent = guide_latent + embedding.view(1, -1, 1, 1, 1)
            logging.info(
                "[LTX MSR-AVref] image slot applied: label=%s slot_id=%d "
                "embedding_dim=%d embedding_norm=%.6f",
                label,
                slot_id,
                embedding_dim,
                embedding_norm,
            )
            original_shape = list(guide_latent.shape[2:])

            guide_mask = None
            if downscale > 1:
                guide_latent, guide_mask = nodes_lt.LTXVAddGuide.dilate_latent(
                    guide_latent, downscale
                )

            frame_offset = -(num_slots - slot_index)
            logging.info(
                "[LTX MSR-AVref] %s: slot_embedding=applied, slot_id=%d, embedding_dim=%d, "
                "embedding_norm=%.6f, time_offset=%d, guide_latent=%s",
                label,
                slot_id,
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
            token_count = (
                guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
            )
            positive = _append_attention_entry(
                positive, token_count, original_shape, strength
            )
            negative = _append_attention_entry(
                negative, token_count, original_shape, strength
            )

        if audio_references is not None:
            positive, negative = cls._attach_audio_references(
                positive,
                negative,
                audio_references,
                avref_parameters,
                num_slots,
            )

        logging.info(
            "[LTX MSR-AVref] Guide complete: image_slots=%d frames_each=%d "
            "audio_slots=%s output_latent=%s",
            num_slots,
            reference_frames,
            [block["slot_id"] for block in audio_references["blocks"]]
            if audio_references is not None
            else [],
            tuple(latent_image.shape),
        )
        return io.NodeOutput(
            positive,
            negative,
            {"samples": latent_image, "noise_mask": noise_mask},
        )

    @staticmethod
    def _validate_audio_image_mapping(
        audio_references,
        avref_parameters,
        pic1,
        pic2,
        pic3,
        pic4,
    ):
        if not isinstance(audio_references, dict):
            raise ValueError("Invalid MSR-AVref audio-reference payload.")
        if audio_references.get("layout_marker") != _AVREF_LAYOUT_MARKER:
            raise ValueError("Invalid MSR-AVref audio-reference payload.")
        if audio_references.get("lora_name") != avref_parameters.get("lora_name"):
            raise ValueError(
                "The audio references were encoded with a different MSR-AVref LoRA."
            )

        pictures = {1: pic1, 2: pic2, 3: pic3, 4: pic4}
        present_picture_slots = [slot for slot, image in pictures.items() if image is not None]
        if not present_picture_slots or present_picture_slots != list(
            range(1, present_picture_slots[-1] + 1)
        ):
            raise ValueError(
                "When audio references are used, numbered image inputs must be contiguous "
                f"pic1..picN; found {present_picture_slots}."
            )

        audio_slots = [int(block["slot_id"]) for block in audio_references.get("blocks", ())]
        if not audio_slots or audio_slots != sorted(set(audio_slots)):
            raise ValueError("Audio reference slots must be unique and in ascending order.")
        for slot_id in audio_slots:
            if slot_id not in (1, 2, 3) or pictures[slot_id] is None:
                raise ValueError(
                    f"audio_ref{slot_id} is connected, but matching pic{slot_id} is not."
                )

    @staticmethod
    def _attach_audio_references(
        positive,
        negative,
        audio_references,
        avref_parameters,
        image_slot_count,
    ):
        blocks = audio_references["blocks"]
        slot_ids = tuple(int(block["slot_id"]) for block in blocks)
        tokens_by_slot = []
        slot_lengths = []
        for slot_id, block in zip(slot_ids, blocks):
            tokens = block["tokens"]
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                raise ValueError(f"audio_ref{slot_id} tokens must be shaped [B, T, C].")
            if tokens.shape[0] != 1 or tokens.shape[1] < 1:
                raise ValueError(f"audio_ref{slot_id} must contain one non-empty clip.")
            if slot_id > image_slot_count:
                raise ValueError(f"audio_ref{slot_id} has no matching image slot.")
            tokens_by_slot.append(tokens)
            slot_lengths.append(int(tokens.shape[1]))

        ref_tokens = torch.cat(tokens_by_slot, dim=1)
        ref_audio = {
            "layout_marker": _AVREF_LAYOUT_MARKER,
            "tokens": ref_tokens,
            "slot_ids": slot_ids,
            "slot_lengths": tuple(slot_lengths),
            "image_slot_count": int(image_slot_count),
            "slot_duration_seconds": float(
                avref_parameters["reference_audio_slot_duration_seconds"]
            ),
            "end_margin_seconds": float(
                avref_parameters["reference_audio_end_margin_seconds"]
            ),
        }
        positive = node_helpers.conditioning_set_values(
            positive, {"ref_audio": ref_audio}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"ref_audio": ref_audio}
        )
        logging.info(
            "[LTX MSR-AVref] Attached audio slots=%s lengths=%s before target; "
            "image_slot_count=%d",
            slot_ids,
            tuple(slot_lengths),
            image_slot_count,
        )
        return positive, negative

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
        pixels = ComfyUILTX25MSRAVrefMultiReferenceGuide._resize_reference(
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
    "ComfyUILTX25MSRAVrefICLoRALoader": ComfyUILTX25MSRAVrefICLoRALoader,
    "ComfyUILTX25MSRAVrefAudioEncoder": ComfyUILTX25MSRAVrefAudioEncoder,
    "ComfyUILTX25MSRAVrefMultiReferenceGuide": ComfyUILTX25MSRAVrefMultiReferenceGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUILTX25MSRAVrefICLoRALoader": "ComfyUI-LTX2.5-MSR-AVref IC-LoRA Loader",
    "ComfyUILTX25MSRAVrefAudioEncoder": "ComfyUI-LTX2.5-MSR-AVref Three Audio Reference Encoder",
    "ComfyUILTX25MSRAVrefMultiReferenceGuide": "ComfyUI-LTX2.5-MSR-AVref Multi-Reference Guide",
}
