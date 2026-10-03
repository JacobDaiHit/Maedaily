"""Real fresh-process resume, causal error injection and protected evidence export."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "transformer" / "experiments" / "project_a_reference.py"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    spec = importlib.util.spec_from_file_location("project_a_reference_test", SCRIPT)
    lab = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lab
    spec.loader.exec_module(lab)


@unittest.skipUnless(TORCH_AVAILABLE, "Project A reference requires existing optional PyTorch")
class ProjectAReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="maedaily_project_a_test_")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.output = cls.directory / "run"
        result = cls.cli("--steps", "4", "--split-step", "2", "--eval-every", "2",
                         "--output-dir", cls.output)
        if result.returncode:
            raise AssertionError(f"real Project A run failed: {result.stdout}\n{result.stderr}")
        cls.summary = json.loads((cls.output / "summary.json").read_text(encoding="utf-8"))

    @classmethod
    def cli(cls, *arguments):
        return subprocess.run([sys.executable, "-X", "utf8", str(SCRIPT), *map(str, arguments)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)

    @staticmethod
    def tree_hashes(directory):
        return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob("*") if path.is_file()}

    def test_fresh_process_resume_every_step_and_complete_final_state(self):
        self.assertEqual(self.summary["status"], "mechanism_passed")
        self.assertEqual(self.summary["mechanism_acceptance"]["fresh_process_count"], 6)
        processes = self.summary["processes"]
        self.assertEqual(len({row["pid"] for row in processes.values()}), 6)
        for branch in ("prefix", "resumed"):
            comparison = self.summary["comparisons"][branch]
            self.assertEqual(comparison["steps_compared"], 2)
            for key in ("all_batch_ids_equal", "all_losses_exact", "all_parameter_states_exact",
                        "all_optimizer_states_exact", "all_rng_states_exact", "all_rng_witnesses_exact"):
                self.assertTrue(comparison[key], (branch, key))
        continuous = torch.load(self.output / "continuous" / "final.pt", weights_only=True)
        restored = torch.load(self.output / "resumed" / "final.pt", weights_only=True)
        for key in ("model_state", "optimizer_state", "rng_state", "global_step", "processed_tokens"):
            self.assertTrue(lab.resume.states_equal(continuous[key], restored[key]), key)
        self.assertEqual(continuous["global_step"], 4)
        self.assertGreater(continuous["processed_tokens"], 0)

    def test_missing_optimizer_and_sampler_fail_at_the_expected_boundary(self):
        optimizer = self.summary["comparisons"]["missing_optimizer"]
        self.assertTrue(optimizer["all_batch_ids_equal"])
        self.assertTrue(optimizer["first_loss_exact"])
        self.assertFalse(optimizer["first_parameter_state_exact"])
        self.assertFalse(optimizer["all_optimizer_states_exact"])
        sampler = self.summary["comparisons"]["missing_batch_generator"]
        self.assertFalse(sampler["first_batch_equal"])
        self.assertFalse(sampler["first_parameter_state_exact"])
        self.assertFalse(sampler["all_rng_states_exact"])

    def test_single_lr_change_has_identical_initialization_data_and_budget(self):
        requests = {name: json.loads((self.output / f"request_{name}.json").read_text(encoding="utf-8"))
                    for name in ("continuous", "alternate_lr")}
        alternate = requests["alternate_lr"]["config"]
        self.assertEqual(alternate["optimizer_parameters"]["lr"], lab.OTHER_LR)
        alternate["optimizer_parameters"]["lr"] = lab.BASE_LR
        self.assertEqual(alternate, requests["continuous"]["config"])
        quality = self.summary["quality"]
        self.assertEqual(quality["status"], "measured_without_general_language_acceptance")
        self.assertEqual(quality["base_lr"]["learning_rate"], lab.BASE_LR)
        self.assertEqual(quality["alternate_lr"]["learning_rate"], lab.OTHER_LR)
        for branch in ("continuous", "alternate_lr"):
            budget = self.summary["budgets"][branch]
            self.assertEqual(budget["updates"], 4)
            self.assertGreater(budget["updated_tokens"], 0)
            self.assertGreater(budget["wall_seconds"], 0)
            self.assertIn(budget["peak_memory"]["status"], ("measured", "not_measured"))
        self.assertEqual(self.summary["budgets"]["continuous"]["updated_tokens"],
                         self.summary["budgets"]["alternate_lr"]["updated_tokens"])
        with (self.output / "learning_curves.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for branch in ("base_lr", "alternate_lr"):
            selected = [row for row in rows if row["branch"] == branch]
            self.assertEqual([int(row["step"]) for row in selected], [0, 2, 4])
            self.assertTrue(all(float(row["train_loss"]) > 0 and float(row["validation_loss"]) > 0
                                for row in selected))

    def test_future_leakage_and_double_shift_injections_are_detected(self):
        torch.set_num_threads(1)
        model, _, _ = lab.resume.new_objects(29)
        train = lab.lm.make_sequences(lab.lm.TRAIN_LENGTHS)
        self.assertLessEqual(lab.causal_check(model, train), 1e-12)
        model.attention.causal_allowed.fill_(True)
        with self.assertRaisesRegex(AssertionError, "future-token leakage"):
            lab.causal_check(model, train)
        sequence = torch.tensor([[lab.lm.BOS, 3, 4, lab.lm.EOS, lab.lm.PAD]])
        inputs, targets = lab.lm.inputs_and_targets(sequence)
        lab.shift_check(sequence, inputs, targets)
        with self.assertRaisesRegex(AssertionError, "exactly once"):
            lab.shift_check(sequence, inputs, torch.tensor([[4, lab.lm.EOS, -100, -100]]))

    def test_state_hash_includes_tensor_dtype_shape_and_raw_bits(self):
        self.assertNotEqual(lab.state_hash(torch.zeros(2, dtype=torch.float64)),
                            lab.state_hash(torch.zeros(2, dtype=torch.float32)))
        self.assertNotEqual(lab.state_hash(torch.zeros(2, dtype=torch.float64)),
                            lab.state_hash(torch.zeros(1, 2, dtype=torch.float64)))
        self.assertNotEqual(lab.state_hash(torch.tensor([0.0], dtype=torch.float64)),
                            lab.state_hash(torch.tensor([-0.0], dtype=torch.float64)))
        self.assertEqual(lab.state_hash({"b": [2], "a": (1,)}),
                         lab.state_hash({"a": (1,), "b": [2]}))

    def test_compact_export_preserves_evidence_and_checks_integrity(self):
        before = self.tree_hashes(self.output)
        destination = self.directory / "export"
        lab.export_evidence(self.output, destination)
        for name in ("evidence.json", "step_comparison.csv", "learning_curves.csv"):
            self.assertEqual((destination / name).read_bytes(), (self.output / name).read_bytes())
        self.assertFalse(any(destination.glob("*.pt")))
        self.assertEqual(self.tree_hashes(self.output), before)
        receipt = json.loads((destination / "export_receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["verified_artifacts"], len(self.summary["artifact_manifest"]))
        sentinel = self.directory / "not_a_valid_run"
        sentinel.mkdir()
        (sentinel / "summary.json").write_text(json.dumps(self.summary), encoding="utf-8")
        missing_destination = self.directory / "invalid_export"
        with self.assertRaisesRegex(ValueError, "integrity"):
            lab.export_evidence(sentinel, missing_destination)
        self.assertFalse(missing_destination.exists())

    def test_nonempty_output_and_export_never_overwrite_existing_evidence(self):
        before = self.tree_hashes(self.output)
        rejected = self.cli("--steps", "4", "--split-step", "2", "--output-dir", self.output)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(self.tree_hashes(self.output), before)
        destination = self.directory / "protected"
        destination.mkdir()
        (destination / "sentinel.txt").write_text("keep this evidence", encoding="utf-8")
        protected = self.tree_hashes(destination)
        with self.assertRaises(ValueError):
            lab.export_evidence(self.output, destination)
        self.assertEqual(self.tree_hashes(destination), protected)


if __name__ == "__main__":
    unittest.main()
