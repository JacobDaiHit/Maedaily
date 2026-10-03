"""Capacity/protocol regressions for configurable layers and legacy checkpoints."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "post_train" / "experiments" / "model_training_lab.py"
if TORCH_AVAILABLE:
    import torch
    spec = importlib.util.spec_from_file_location("model_capacity_lab", SCRIPT)
    lab = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lab
    spec.loader.exec_module(lab)


@unittest.skipUnless(TORCH_AVAILABLE, "capacity tests require optional CPU PyTorch")
class ModelCapacityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        self.tokenizer = lab.CharacterTokenizer()
        self.records = [{"id": "short", "question": "2+3=", "answer": "5"},
                        {"id": "signed", "question": "1-4=", "answer": "-3"}]
        self.batch = lab.build_batch(self.records, self.tokenizer)
        self.model = lab.TinyTextLM(len(self.tokenizer.pieces), width=24, heads=3, layers=2)
        self.temporary = tempfile.TemporaryDirectory(prefix="maedaily_capacity_test_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def run_cli(self, arguments):
        return subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT), *map(str, arguments)],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=60)

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def tree_hashes(cls, directory):
        return {path.relative_to(directory).as_posix(): cls.digest(path)
                for path in directory.rglob("*") if path.is_file()}

    def test_two_layers_receive_real_gradients_and_parameter_updates(self):
        parameters = dict(self.model.named_parameters())
        names = ("attention.qkv.weight", "mlp.0.weight",
                 "extra_blocks.0.attention.qkv.weight", "extra_blocks.0.mlp.0.weight")
        before = {name: parameters[name].detach().clone() for name in names}
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002)
        optimizer.zero_grad(set_to_none=True)
        loss = lab.sft_loss(self.model, self.batch)
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(reachable_parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
        for name in names:
            with self.subTest(parameter=name):
                gradient = parameters[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertGreater(gradient.abs().sum().item(), 0)
        optimizer.step()
        for name in names:
            with self.subTest(updated_parameter=name):
                self.assertGreater((parameters[name].detach() - before[name]).abs().max().item(), 0)

    def test_two_layer_causal_mask_and_padding_preserve_prefix_logits(self):
        ids = self.batch["input_ids"]
        altered = ids.clone()
        position = self.batch["prefix_lengths"][0]
        altered[0, position] = self.tokenizer.encode("7")[0]
        self.model.eval()
        with torch.no_grad():
            before = self.model(ids)
            after = self.model(altered)
            valid_end = int(self.batch["attention_mask"][0].sum())
            without_padding = self.model(ids[:1, :valid_end])
        torch.testing.assert_close(before[0, :position], after[0, :position], atol=1e-7, rtol=0)
        self.assertGreater((before[0, position:] - after[0, position:]).abs().max().item(), 1e-4)
        torch.testing.assert_close(before[0, :valid_end], without_padding[0], atol=1e-6, rtol=0)

    def test_nondefault_capacity_and_profile_survive_exact_checkpoint_reload(self):
        # A real update avoids validating only a freshly initialized state.
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002)
        lab.checked_step(lab.sft_loss(self.model, self.batch), self.model, optimizer)
        self.model.data_profile = "quality-v2"
        path = self.directory / "two_layer.pt"
        lab.save_checkpoint(path, self.model, self.tokenizer, "capacity-test", optimizer, cursor=1)
        original_file_hash = self.digest(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["model_config"]["layers"], 2)
        self.assertEqual(payload["model_config"]["width"], 24)
        self.assertEqual(payload["model_config"]["heads"], 3)
        self.assertEqual(payload["data_profile"], "quality-v2")
        restored, tokenizer = lab.load_model_and_tokenizer(path)
        self.assertEqual(restored.config, self.model.config)
        self.assertEqual(restored.data_profile, "quality-v2")
        self.assertEqual(len(restored.extra_blocks), 1)
        self.assertEqual(tokenizer.to_dict(), self.tokenizer.to_dict())
        self.assertEqual(lab.weight_hash(restored), lab.weight_hash(self.model))
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.batch["input_ids"])
            actual = restored(self.batch["input_ids"])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(self.digest(path), original_file_hash)

    def test_checkpoint_without_layers_or_profile_loads_as_original_single_layer(self):
        single = lab.TinyTextLM(len(self.tokenizer.pieces), width=16, heads=2)
        current = self.directory / "current.pt"
        lab.save_checkpoint(current, single, self.tokenizer, "legacy-fixture")
        current_hash = self.digest(current)
        payload = torch.load(current, map_location="cpu", weights_only=True)
        self.assertEqual(payload["data_profile"], "legacy-v1")
        # The former model had the same single-layer parameter names; removing
        # only the two new metadata fields recreates its serialized format.
        del payload["model_config"]["layers"]
        del payload["data_profile"]
        legacy = self.directory / "legacy.pt"
        with legacy.open("xb") as stream:
            torch.save(payload, stream)
        legacy_hash = self.digest(legacy)
        restored, tokenizer = lab.load_model_and_tokenizer(legacy)
        self.assertEqual(restored.config["layers"], 1)
        self.assertEqual(len(restored.extra_blocks), 0)
        self.assertEqual(restored.data_profile, "legacy-v1")
        self.assertEqual(tokenizer.to_dict(), self.tokenizer.to_dict())
        self.assertEqual(lab.weight_hash(restored), lab.weight_hash(single))
        single.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(self.batch["input_ids"]), single(self.batch["input_ids"]), atol=0, rtol=0)
        self.assertEqual(self.digest(current), current_hash)
        self.assertEqual(self.digest(legacy), legacy_hash)

    def test_layer_range_and_cli_capacity_options_are_enforced(self):
        for layers in (1, 2, 3, 4):
            with self.subTest(valid_layers=layers):
                model = lab.TinyTextLM(len(self.tokenizer.pieces), width=8, heads=2, layers=layers)
                self.assertEqual(model.config["layers"], layers)
                self.assertEqual(len(model.extra_blocks), layers - 1)
                with torch.no_grad():
                    output = model(self.batch["input_ids"])
                self.assertEqual(output.shape, (*self.batch["input_ids"].shape, len(self.tokenizer.pieces)))
                self.assertTrue(torch.isfinite(output).all().item())
        for layers in (0, 5):
            with self.subTest(invalid_layers=layers), self.assertRaises(ValueError):
                lab.TinyTextLM(len(self.tokenizer.pieces), width=8, heads=2, layers=layers)
        with patch.object(sys, "argv", [str(SCRIPT), "--width", "24", "--heads", "3", "--layers", "2", "--data-profile", "quality-v2"]):
            arguments = lab.parse_args()
        self.assertEqual((arguments.width, arguments.heads, arguments.layers, arguments.data_profile), (24, 3, 2, "quality-v2"))
        for invalid in (("--layers", "0"), ("--layers", "5"), ("--width", "18", "--heads", "4"), ("--heads", "0"), ("--width", "257")):
            with self.subTest(invalid_cli=invalid), patch.object(sys, "argv", [str(SCRIPT), *invalid]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    lab.parse_args()
                self.assertEqual(raised.exception.code, 2)

    def test_real_cli_uses_nondefault_capacity_in_persisted_model(self):
        output = self.directory / "tiny_nondefault_run"
        completed = self.run_cli(["--mode", "sft", "--width", "24", "--heads", "3", "--layers", "2",
                                  "--data-profile", "legacy-v1", "--base-steps", "1", "--sft-steps", "1",
                                  "--max-new-tokens", "1", "--threads", "1", "--output-dir", output])
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        resolved = json.loads((output / "config.resolved.json").read_text(encoding="utf-8"))
        self.assertEqual((resolved["width"], resolved["heads"], resolved["layers"], resolved["data_profile"]),
                         (24, 3, 2, "legacy-v1"))
        restored, tokenizer = lab.load_model_and_tokenizer(output / "final_checkpoint.pt")
        self.assertEqual((restored.config["width"], restored.config["heads"], restored.config["layers"]), (24, 3, 2))
        self.assertEqual(restored.data_profile, "legacy-v1")
        self.assertEqual(tokenizer.to_dict(), self.tokenizer.to_dict())
        self.assertEqual(len(restored.extra_blocks), 1)
        for filename in ("base_train.json", "sft_train.json"):
            logs = json.loads((output / filename).read_text(encoding="utf-8"))
            self.assertEqual(len(logs), 1)
            self.assertGreater(logs[0]["parameter_delta_norm"], 0)

    def test_quality_protocol_cannot_bypass_sealed_selection_workflow(self):
        output = self.directory / "sealed_quality_run"
        completed = self.run_cli(["--mode", "all", "--data-profile", "quality-v2",
                                  "--base-steps", "1", "--sft-steps", "1", "--output-dir", output])
        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        self.assertIn("model_quality_lab.py", completed.stderr)
        self.assertFalse(output.exists())
        with patch.object(lab, "evaluate") as evaluate:
            with self.assertRaisesRegex(ValueError, "sealed"):
                lab.run({"data_profile": "quality-v2"}, output)
            evaluate.assert_not_called()

    def test_profile_conflict_fails_real_cli_and_preserves_checkpoint_and_old_output(self):
        self.model.data_profile = "quality-v2"
        checkpoint = self.directory / "quality_model.pt"
        lab.save_checkpoint(checkpoint, self.model, self.tokenizer, "conflict-fixture")
        checkpoint_hash = self.digest(checkpoint)
        output = self.directory / "conflicting_new_run"
        arguments = ["--mode", "sft", "--model-path", checkpoint, "--data-profile", "legacy-v1",
                     "--base-steps", "1", "--sft-steps", "1", "--output-dir", output]
        completed = self.run_cli(arguments)
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["error_type"], "ValueError")
        self.assertIn("data profile", failure["reason"])
        self.assertEqual(failure["config"]["data_profile"], "legacy-v1")
        self.assertFalse(any(output.glob("*.pt")))
        self.assertEqual(self.digest(checkpoint), checkpoint_hash)
        # A failed run remains evidence; repeating it cannot overwrite its log.
        failed_run_hashes = self.tree_hashes(output)
        repeated = self.run_cli(arguments)
        self.assertEqual(repeated.returncode, 2)
        self.assertEqual(self.tree_hashes(output), failed_run_hashes)
        self.assertEqual(self.digest(checkpoint), checkpoint_hash)
        existing = self.directory / "previous_results"
        existing.mkdir()
        (existing / "summary.json").write_text('{"previous_result":true}\n', encoding="utf-8")
        (existing / "saved.pt").write_bytes(b"existing unrelated checkpoint bytes")
        existing_hashes = self.tree_hashes(existing)
        rejected = self.run_cli([*arguments[:-1], existing])
        self.assertEqual(rejected.returncode, 2)
        self.assertEqual(self.tree_hashes(existing), existing_hashes)
        self.assertEqual(self.digest(checkpoint), checkpoint_hash)


if __name__ == "__main__":
    unittest.main()
