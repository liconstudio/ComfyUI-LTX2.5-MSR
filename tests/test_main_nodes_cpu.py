"""CPU integration regressions for the unified MSR/AVref entry points."""

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch


PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
original_argv = sys.argv
sys.argv = [original_argv[0], "--cpu"]
import comfy.options

comfy.options.enable_args_parsing()
SPEC = importlib.util.spec_from_file_location(
    "ltx_msr_main_test_package",
    PLUGIN_ROOT / "__init__.py",
    submodule_search_locations=[str(PLUGIN_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
sys.argv = original_argv

MAIN = PACKAGE.nodes
GUIDE = MAIN.ComfyUILTX25MSRMultiReferenceGuide
LOADER = MAIN.ComfyUILTX25MSRICLoRALoader

from comfy.ldm.lightricks.symmetric_patchifier import AudioPatchifier


def slot_state():
    return {
        "frequencies": torch.tensor([1.0, 2.0]),
        "net.0.weight": torch.full((4, 5), 0.1),
        "net.0.bias": torch.zeros(4),
        "net.2.weight": torch.full((128, 4), 0.1),
        "net.2.bias": torch.full((128,), 0.25),
    }


def avref_metadata():
    return {
        "reference_token_order": "prepend",
        "reference_slot_time_offsets": "pic1_based_negative_time",
        "reference_audio_conditioning": "id_lora_clean_negative_rope",
        "reference_audio_token_order": "pic1_to_picN_then_target",
        "reference_audio_rope_layout": "absolute_image_slot_windows",
        "reference_audio_overflow_mode": "truncate",
        "reference_audio_sparse_slots": "true",
        "reference_slot_embedding_enabled": "true",
        "reference_audio_slot_embedding_enabled": "true",
    }


def reference_parameters(audio=False):
    state = slot_state()
    params = {
        "slot_state": state,
        "metadata": {},
        "lora_name": "test.safetensors",
        "reference_downscale_factor": 1,
        "reference_temporal_scale_factor": 1,
    }
    if audio:
        params.update(
            audio_slot_state=slot_state(),
            metadata=avref_metadata(),
            reference_audio_slot_duration_seconds=5.0,
            reference_audio_end_margin_seconds=0.04,
            reference_audio_max_tokens=125,
        )
    return params


class FakeVideoVAE:
    downscale_index_formula = (8, 2, 2)

    def __init__(self):
        self.encoded_frames = []

    def encode(self, pixels):
        self.encoded_frames.append(pixels.shape[0])
        frames = (pixels.shape[0] - 1) // 8 + 1
        return torch.zeros(1, 128, frames, pixels.shape[1] // 2, pixels.shape[2] // 2)


def guide_inputs(params=None, frames="25"):
    return {
        "positive": [[torch.zeros(1, 1, 1), {}]],
        "negative": [[torch.zeros(1, 1, 1), {}]],
        "vae": FakeVideoVAE(),
        "latent": {"samples": torch.zeros(1, 128, 2, 2, 2)},
        "pic1": torch.zeros(1, 4, 4, 3),
        "strength": 1.0,
        "reference_frames": frames,
        "use_tiled_encode": False,
        "tile_size": 256,
        "tile_overlap": 64,
        "msr_parameters": params,
    }


class FakeDiffusionModel:
    def __init__(self):
        self.a_patchifier = AudioPatchifier(1, start_end=True)

    def _process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
        refs = kwargs["ref_audio"]["tokens"]
        target = torch.full((1, 2, 128), 9.0)
        _, target_positions = self.a_patchifier.patchify(torch.zeros(1, 1, 2, 1))
        positions = torch.cat(
            [torch.zeros(1, 1, refs.shape[1], 2), target_positions], dim=2
        )
        return (
            [torch.zeros(1), torch.cat([refs, target], dim=1)],
            [torch.zeros(1), positions],
            {"ref_audio_seq_len": refs.shape[1], "target_audio_seq_len": 2},
        )


class FakePatcher:
    def __init__(self, model=None):
        self.model = model or types.SimpleNamespace(diffusion_model=FakeDiffusionModel())
        self.object_patches = {}
        self.callbacks = {}

    def clone(self):
        result = FakePatcher(self.model)
        result.object_patches = dict(self.object_patches)
        result.callbacks = dict(self.callbacks)
        return result

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        result = self.model
        for part in name.split("."):
            result = getattr(result, part)
        return result

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def get_callbacks(self, callback, key):
        return self.callbacks.get((callback, key), [])

    def add_callback_with_key(self, callback, key, value):
        self.callbacks[(callback, key)] = [value]


class UnifiedMSRCPUTests(unittest.TestCase):
    def test_only_two_main_nodes_with_audio_latent_inputs(self):
        schema = GUIDE.define_schema()
        ids = [item.id for item in schema.inputs]
        self.assertEqual(schema.node_id, "ComfyUILTX25MSRMultiReferenceGuide")
        self.assertEqual(ids[:10], [
            "positive", "negative", "vae", "latent", "pic1", "pic2", "pic3",
            "pic4", "background", "msr_parameters",
        ])
        self.assertEqual(ids[10:12], ["audio_ref1", "audio_ref2"])
        self.assertEqual(ids[12:], [
            "strength", "reference_frames", "use_tiled_encode", "tile_size", "tile_overlap",
        ])
        for item in schema.inputs[10:12]:
            self.assertEqual(item.io_type, "LATENT")
            self.assertTrue(item.optional)
        self.assertNotIn("audio_ref3", ids)
        self.assertFalse(schema.is_deprecated)
        self.assertFalse(LOADER.define_schema().is_deprecated)
        self.assertEqual(set(PACKAGE.NODE_CLASS_MAPPINGS), {
            "ComfyUILTX25MSRICLoRALoader", "ComfyUILTX25MSRMultiReferenceGuide",
        })

    def test_main_guide_keeps_sparse_slot_and_background_time_window(self):
        for frames, expected_latent_frames in (("25", 18), ("33", 22)):
            with self.subTest(frames=frames):
                args = guide_inputs(reference_parameters(audio=True), frames)
                args.update(pic2=args["pic1"], pic3=args["pic1"], background=args["pic1"])
                args["audio_ref2"] = {"samples": torch.zeros(1, 8, 2, 16)}
                positive, negative, latent = GUIDE.execute(**args).result
                audio = positive[0][1]["ref_audio"]
                self.assertIs(negative[0][1]["ref_audio"], audio)
                self.assertEqual(audio["slot_ids"], (2,))
                self.assertEqual(audio["slot_lengths"], (2,))
                self.assertEqual(audio["image_slot_count"], 4)
                self.assertEqual(args["vae"].encoded_frames, [int(frames)] * 4)
                self.assertEqual(tuple(latent["samples"].shape), (1, 128, expected_latent_frames, 2, 2))
                self.assertEqual(len(positive[0][1]["guide_attention_entries"]), 4)

                model = FakeDiffusionModel()
                patched = types.MethodType(MAIN._make_avref_process_input_patch(model._process_input), model)
                (_, tokens), (_, positions), _ = patched(None, None, None, ref_audio=audio)
                torch.testing.assert_close(
                    positions[0, 0, 1, 1], torch.tensor(-10.04), atol=1e-6, rtol=0
                )
                torch.testing.assert_close(tokens[:, :2], audio["tokens"])
                torch.testing.assert_close(tokens[:, 2:], torch.full((1, 2, 128), 9.0))

    def test_image_only_modes_keep_original_frame_choices(self):
        for params in (None, reference_parameters(), reference_parameters(audio=True)):
            for frames, expected_latent_frames in (("25", 6), ("33", 7)):
                with self.subTest(msr=params is not None, frames=frames):
                    args = guide_inputs(params, frames)
                    positive, _, latent = GUIDE.execute(**args).result
                    self.assertEqual(args["vae"].encoded_frames, [int(frames)])
                    self.assertEqual(latent["samples"].shape[2], expected_latent_frames)
                    self.assertNotIn("ref_audio", positive[0][1])

    def test_audio_requires_avref(self):
        for params in (None, reference_parameters()):
            args = guide_inputs(params)
            args["audio_ref1"] = {"samples": torch.zeros(1, 8, 2, 16)}
            with self.assertRaisesRegex(ValueError, "require an AVref LoRA"):
                GUIDE.execute(**args)
            self.assertEqual(args["vae"].encoded_frames, [])

    def test_audio_rejects_missing_picture_and_invalid_native_latents(self):
        args = guide_inputs(reference_parameters(audio=True))
        args["audio_ref2"] = {"samples": torch.zeros(1, 8, 2, 16)}
        with self.assertRaisesRegex(ValueError, "matching pic2"):
            GUIDE.execute(**args)
        for invalid in (torch.zeros(1, 128, 2), torch.zeros(2, 8, 2, 16), torch.zeros(1, 8, 0, 16)):
            with self.subTest(shape=tuple(invalid.shape)):
                args = guide_inputs(reference_parameters(audio=True))
                args["audio_ref1"] = {"samples": invalid}
                with self.assertRaises(ValueError):
                    GUIDE.execute(**args)
                self.assertEqual(args["vae"].encoded_frames, [])

    def test_native_audio_flattens_and_embeds_sparse_truncated_slots(self):
        raw = torch.arange(8 * 126 * 16, dtype=torch.float32).reshape(1, 8, 126, 16)
        params = reference_parameters(audio=True)
        # Constant learned output makes the expected embedding independent of
        # the production Fourier-MLP implementation.
        params["audio_slot_state"]["net.2.weight"].zero_()
        native = MAIN._audio_latents_to_references(
            ({"samples": raw}, None, {"samples": raw}), params
        )
        self.assertEqual([block["slot_id"] for block in native["blocks"]], [1, 3])
        expected = torch.stack([raw[0, :, time, :].flatten() for time in range(125)]).unsqueeze(0) + 0.25
        for current in native["blocks"]:
            self.assertEqual(tuple(current["tokens"].shape), (1, 125, 128))
            torch.testing.assert_close(current["tokens"], expected)
        self.assertEqual(native["layout_marker"], MAIN._AVREF_LAYOUT_MARKER)
        self.assertEqual(native["slot_duration_seconds"], 5.0)
        self.assertEqual(native["end_margin_seconds"], 0.04)

    def test_loader_selects_original_or_avref_and_strips_embedding_weights(self):
        for audio in (False, True):
            with self.subTest(audio=audio):
                lora = {"diffusion_model.reference_slot_embedding." + key: value
                        for key, value in slot_state().items()}
                lora["ordinary.lora.weight"] = torch.ones(1)
                metadata = avref_metadata() if audio else {}
                if audio:
                    lora.update({"reference_audio_slot_embedding." + key: value
                                 for key, value in slot_state().items()})
                model = FakePatcher()
                with mock.patch.object(MAIN.folder_paths, "get_full_path_or_raise", return_value="test.safetensors"), \
                     mock.patch.object(MAIN.comfy.utils, "load_torch_file", return_value=(lora, metadata)), \
                     mock.patch.object(MAIN.comfy.sd, "load_lora_for_models", return_value=(model, None)) as load:
                    output, params = LOADER.execute(model, "test.safetensors", 1.0).result
                self.assertEqual(set(load.call_args.args[2]), {"ordinary.lora.weight"})
                self.assertIn("slot_state", params)
                self.assertEqual("audio_slot_state" in params, audio)
                self.assertEqual(bool(output.object_patches), audio)
                self.assertFalse(model.object_patches)
                if audio:
                    self.assertEqual(params["slot_state"]["net.2.bias"].numel(), 128)
                    self.assertEqual(params["reference_audio_max_tokens"], 125)
                    patch = output.object_patches[MAIN._PROCESS_INPUT_PATCH_PATH]
                    self.assertTrue(MAIN._is_avref_process_input_patch(patch))
                    self.assertTrue(output.callbacks)

    def test_loader_does_not_fall_back_when_avref_metadata_is_incomplete(self):
        lora = {"reference_slot_embedding." + key: value for key, value in slot_state().items()}
        lora.update({"reference_audio_slot_embedding." + key: value for key, value in slot_state().items()})
        with mock.patch.object(MAIN.folder_paths, "get_full_path_or_raise", return_value="test.safetensors"), \
             mock.patch.object(MAIN.comfy.utils, "load_torch_file", return_value=(lora, {})), \
             mock.patch.object(MAIN.comfy.sd, "load_lora_for_models") as load:
            with self.assertRaisesRegex(ValueError, "missing required MSR-AVref metadata"):
                LOADER.execute(FakePatcher(), "test.safetensors", 1.0)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
