import importlib.util
import pathlib
import sys
import types
import unittest

import torch


COMFY_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMFY_ROOT))

original_argv = sys.argv
sys.argv = [original_argv[0], "--cpu"]
import comfy.options

comfy.options.enable_args_parsing()
PLUGIN_FILE = pathlib.Path(__file__).resolve().parents[1] / "avref_nodes.py"
SPEC = importlib.util.spec_from_file_location("ltx_msr_avref_nodes", PLUGIN_FILE)
PLUGIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLUGIN)
sys.argv = original_argv

from comfy.ldm.lightricks.symmetric_patchifier import AudioPatchifier


class MSRAVRefCPUTests(unittest.TestCase):
    def test_three_unique_nodes_are_registered(self):
        self.assertEqual(
            set(PLUGIN.NODE_CLASS_MAPPINGS),
            {
                "ComfyUILTX25MSRAVrefICLoRALoader",
                "ComfyUILTX25MSRAVrefAudioEncoder",
                "ComfyUILTX25MSRAVrefMultiReferenceGuide",
            },
        )
        encoder_inputs = {
            item.id
            for item in PLUGIN.ComfyUILTX25MSRAVrefAudioEncoder.define_schema().inputs
        }
        self.assertTrue({"audio_ref1", "audio_ref2"} <= encoder_inputs)
        self.assertNotIn("audio_ref3", encoder_inputs)

    def test_image_and_audio_slot_tensors_are_removed_from_normal_lora(self):
        lora = {
            "diffusion_model.reference_slot_embedding.frequencies": torch.ones(16),
            "reference_audio_slot_embedding.frequencies": torch.full((16,), 2.0),
            "diffusion_model.transformer_blocks.0.fake": torch.ones(1),
        }
        normal, image, audio = PLUGIN._extract_slot_states(lora)
        self.assertEqual(set(normal), {"diffusion_model.transformer_blocks.0.fake"})
        self.assertEqual(set(image), {"frequencies"})
        self.assertEqual(set(audio), {"frequencies"})

    def test_five_second_slot_keeps_125_causal_tokens(self):
        patchifier = AudioPatchifier(1, start_end=True)
        self.assertEqual(PLUGIN._maximum_audio_tokens(patchifier, 5.0), 125)
        _, positions = patchifier.patchify(torch.empty(1, 1, 125, 1))
        torch.testing.assert_close(
            positions[0, 0, 0], torch.tensor([0.0, 0.01]), atol=1e-7, rtol=0
        )
        torch.testing.assert_close(
            positions[0, 0, -1], torch.tensor([4.93, 4.97]), atol=1e-6, rtol=0
        )

    def test_sparse_slots_keep_absolute_windows_and_target_positions(self):
        class FakeDiffusionModel:
            def __init__(self):
                self.a_patchifier = AudioPatchifier(1, start_end=True)

        model = FakeDiffusionModel()
        target_tokens = torch.full((1, 2, 128), 9.0)
        _, target_positions = model.a_patchifier.patchify(
            torch.empty(1, 1, target_tokens.shape[1], 1)
        )

        def native_process_input(x, keyframe_idxs, denoise_mask, **kwargs):
            ref_tokens = kwargs["ref_audio"]["tokens"]
            ref_count = ref_tokens.shape[1]
            audio = torch.cat([ref_tokens, target_tokens], dim=1)
            native_ref_positions = torch.zeros(1, 1, ref_count, 2)
            positions = torch.cat([native_ref_positions, target_positions], dim=2)
            return (
                [torch.zeros(1), audio],
                [torch.zeros(1), positions],
                {"ref_audio_seq_len": ref_count, "target_audio_seq_len": 2},
            )

        patched = types.MethodType(
            PLUGIN._make_avref_process_input_patch(native_process_input), model
        )
        ref_tokens = torch.arange(3 * 128, dtype=torch.float32).reshape(1, 3, 128)
        ref_audio = {
            "layout_marker": PLUGIN._AVREF_LAYOUT_MARKER,
            "tokens": ref_tokens,
            "slot_ids": (1, 3),
            "slot_lengths": (2, 1),
            "image_slot_count": 3,
            "slot_duration_seconds": 5.0,
            "end_margin_seconds": 0.04,
        }
        (_, audio), (_, positions), extra = patched(
            None, None, None, ref_audio=ref_audio
        )

        torch.testing.assert_close(audio[:, :3], ref_tokens)
        torch.testing.assert_close(audio[:, 3:], target_tokens)
        torch.testing.assert_close(
            positions[0, 0, :2],
            torch.tensor([[-10.09, -10.08], [-10.08, -10.04]]),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            positions[0, 0, 2], torch.tensor([-0.05, -0.04]), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(positions[:, :, 3:], target_positions)
        self.assertEqual(extra["ref_audio_seq_len"], 3)
        self.assertEqual(extra["target_audio_seq_len"], 2)

    def test_process_input_patch_rebinds_to_a_deep_cloned_model(self):
        class FakeDiffusionModel:
            def __init__(self, tag):
                self.tag = tag

            def _process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
                return self.tag

        class FakePatcher:
            def __init__(self, model, diffusion_model, object_patch):
                self.model = model
                self.diffusion_model = diffusion_model
                self.object_patches = {
                    PLUGIN._PROCESS_INPUT_PATCH_PATH: object_patch
                }

            def get_model_object(self, name):
                if name == "diffusion_model":
                    return self.diffusion_model
                raise KeyError(name)

        first_dm = FakeDiffusionModel("first")
        second_dm = FakeDiffusionModel("second")
        function = PLUGIN._make_avref_process_input_patch(first_dm._process_input)
        stale_patch = types.MethodType(function, first_dm)
        source = FakePatcher(object(), first_dm, stale_patch)
        cloned = FakePatcher(object(), second_dm, stale_patch)

        PLUGIN._rebind_avref_process_input_on_clone(source, cloned)
        rebound = cloned.object_patches[PLUGIN._PROCESS_INPUT_PATCH_PATH]
        self.assertIs(rebound.__self__, second_dm)
        self.assertEqual(rebound(None, None, None), "second")

    def test_audio_slot_requires_same_numbered_picture(self):
        payload = {
            "layout_marker": PLUGIN._AVREF_LAYOUT_MARKER,
            "lora_name": "stage4.safetensors",
            "blocks": ({"slot_id": 3, "tokens": torch.zeros(1, 1, 128)},),
        }
        params = {"lora_name": "stage4.safetensors"}
        image = torch.zeros(1, 8, 8, 3)
        with self.assertRaisesRegex(ValueError, "pic3"):
            PLUGIN.ComfyUILTX25MSRAVrefMultiReferenceGuide._validate_audio_image_mapping(
                payload, params, image, image, None, None
            )

        PLUGIN.ComfyUILTX25MSRAVrefMultiReferenceGuide._validate_audio_image_mapping(
            payload, params, image, image, image, None
        )

    def test_encoder_preserves_sparse_slot_ids_and_truncates(self):
        class FakeAudioVAE:
            audio_sample_rate = 16000

            @staticmethod
            def encode(waveform):
                self_shape = waveform.shape
                if self_shape[0] != 1:
                    raise AssertionError("unexpected batch")
                return torch.zeros(1, 8, 126, 16)

        state = {
            "frequencies": torch.ones(16),
            "net.0.weight": torch.zeros(256, 33),
            "net.0.bias": torch.zeros(256),
            "net.2.weight": torch.zeros(128, 256),
            "net.2.bias": torch.zeros(128),
        }
        params = {
            "reference_audio_max_tokens": 125,
            "reference_audio_slot_duration_seconds": 5.0,
            "reference_audio_end_margin_seconds": 0.04,
            "audio_slot_state": state,
            "lora_name": "stage4.safetensors",
        }
        audio = {"waveform": torch.zeros(1, 1, 16000), "sample_rate": 16000}
        output = PLUGIN.ComfyUILTX25MSRAVrefAudioEncoder.execute(
            FakeAudioVAE(), params, audio_ref1=audio, audio_ref3=audio
        )
        payload = output.result[0]
        self.assertEqual([block["slot_id"] for block in payload["blocks"]], [1, 3])
        self.assertEqual([block["tokens"].shape[1] for block in payload["blocks"]], [125, 125])


if __name__ == "__main__":
    unittest.main()
