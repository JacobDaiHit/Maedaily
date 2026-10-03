"""Quality-workflow isolation, failure-state, repetition and output checks."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    SPEC = importlib.util.spec_from_file_location("model_quality_workflow", ROOT / "post_train/experiments/model_quality_lab.py")
    workflow = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(workflow)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@unittest.skipUnless(TORCH_AVAILABLE, "Quality training workflow requires optional CPU PyTorch")
class ModelQualityWorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "run"
        self.config = {"checkpoint_steps": [2], "seeds": [17], "threads": 1}

    @staticmethod
    def fake_evaluation(model, tokenizer, panels, max_new_tokens, *, fail_final=False):
        final = set(panels) == set(workflow.FINAL_PANELS)
        metrics = {name: {"correct": 0 if final and fail_final else len(rows), "total": len(rows),
                          "format_valid": len(rows), "generated_tokens": len(rows) * 2, "accuracy": 0.0
                          if final and fail_final else 1.0} for name, rows in panels.items()}
        return {"panels": metrics, "predictions": [], "response_nll": None if final else 1.0,
                "response_tokens": 0 if final else 100, "decoding": "greedy", "max_new_tokens": max_new_tokens}

    def test_actual_short_training_keeps_final_sealed_and_skips_planned_repetitions(self):
        actual = workflow.evaluate_panels
        seen = []
        def guarded(model, tokenizer, panels, budget):
            seen.append(set(panels))
            self.assertEqual(set(panels), set(workflow.DEV_PANELS), "final panels reached before development qualification")
            return actual(model, tokenizer, panels, budget)
        config = {**self.config, "seeds": [17, 23, 41]}
        with mock.patch.object(workflow, "evaluate_panels", side_effect=guarded):
            summary = workflow.run(config, self.output)
        self.assertEqual(summary["status"], "development_unqualified")
        self.assertEqual(workflow.exit_code(summary), 4)
        self.assertIsNone(summary["final_metrics"])
        self.assertEqual(summary["planned_seeds"], [17, 23, 41])
        self.assertEqual(summary["executed_seeds"], [17])
        self.assertEqual(len(seen), len(workflow.CANDIDATES))
        self.assertFalse((self.output / "replications").exists())
        self.assertFalse(list(self.output.glob("seed_*_final.json")))
        events = read_json(self.output / "events.json")
        self.assertNotIn("final_opened", [event["event"] for event in events])
        selection = read_json(self.output / "selection.json")
        self.assertEqual(selection["allowed_panels"], ["dev", "dev_copy"])
        self.assertFalse(selection["selected"]["development_gate"]["qualified"])

    def test_selection_is_on_disk_before_final_and_final_failure_never_reselects(self):
        final_calls = []
        def evaluate(model, tokenizer, panels, budget):
            if set(panels) == set(workflow.FINAL_PANELS):
                selection_path = self.output / "selection.json"
                self.assertTrue(selection_path.is_file(), "selection must precede final evaluation")
                selected = read_json(selection_path)["selected"]
                self.assertTrue(selected["development_gate"]["qualified"])
                self.assertEqual(workflow.model_lab.file_hash(Path(selected["checkpoint"])), selected["checkpoint_sha256"])
                final_calls.append(set(panels))
            return self.fake_evaluation(model, tokenizer, panels, budget, fail_final=True)
        with mock.patch.object(workflow, "evaluate_panels", side_effect=evaluate), \
             mock.patch.object(workflow, "train_candidate", wraps=workflow.train_candidate) as train:
            summary = workflow.run(self.config, self.output)
        self.assertEqual(summary["status"], "quality_unqualified")
        self.assertEqual(summary["exit_code"], 4)
        self.assertEqual(len(final_calls), 1)
        self.assertEqual(train.call_count, len(workflow.CANDIDATES))
        self.assertEqual(summary["executed_seeds"], [17])
        self.assertIsNotNone(summary["final_metrics"])
        events = read_json(self.output / "events.json")
        names = [event["event"] for event in events]
        self.assertLess(names.index("selection_frozen"), names.index("final_opened"))
        self.assertEqual(summary["selection_sha256"], workflow.model_lab.file_hash(self.output / "selection.json"))
        self.assertFalse(summary["final_used_for_selection"])

    def test_repetitions_use_one_frozen_architecture_and_budget_without_seed_selection(self):
        final_seeds = []
        def evaluate(model, tokenizer, panels, budget):
            final = set(panels) == set(workflow.FINAL_PANELS)
            if final:
                final_seeds.append(len(final_seeds))
            # The third registered seed deliberately fails its final gate.
            return self.fake_evaluation(model, tokenizer, panels, budget,
                                        fail_final=final and len(final_seeds) == 3)
        config = {**self.config, "seeds": [17, 23, 41]}
        with mock.patch.object(workflow, "evaluate_panels", side_effect=evaluate), \
             mock.patch.object(workflow, "train_candidate", wraps=workflow.train_candidate) as train:
            summary = workflow.run(config, self.output)
        self.assertEqual(summary["status"], "quality_unqualified")
        self.assertEqual(summary["planned_seeds"], [17, 23, 41])
        self.assertEqual(summary["executed_seeds"], [17, 23, 41])
        self.assertEqual([row["seed"] for row in summary["seed_reports"]], [17, 23, 41])
        self.assertEqual(summary["failed_seeds"], [41])
        self.assertEqual(set(summary["final_metrics"]), {"17", "23", "41"})
        self.assertEqual(train.call_count, len(workflow.CANDIDATES) + 2)
        chosen = summary["selection"]
        for call in train.call_args_list[len(workflow.CANDIDATES):]:
            self.assertEqual(call.args[0], chosen["candidate"])
            self.assertEqual(call.args[2], [chosen["step"]])
        self.assertEqual(len(final_seeds), 3)

    def test_output_directory_cannot_be_reused_or_original_files_changed(self):
        with mock.patch.object(workflow, "CANDIDATES", workflow.CANDIDATES[:1]):
            workflow.run(self.config, self.output)
            hashes = {path.relative_to(self.output): workflow.model_lab.file_hash(path)
                      for path in self.output.rglob("*") if path.is_file()}
            with self.assertRaisesRegex(ValueError, "新目录或空目录"):
                workflow.run(self.config, self.output)
        after = {path.relative_to(self.output): workflow.model_lab.file_hash(path)
                 for path in self.output.rglob("*") if path.is_file()}
        self.assertEqual(after, hashes)

    def test_frozen_selection_rejects_unqualified_or_tampered_weights(self):
        with mock.patch.object(workflow, "CANDIDATES", workflow.CANDIDATES[:1]):
            summary = workflow.run(self.config, self.output)
        selection_path = self.output / "selection.json"
        with self.assertRaisesRegex(ValueError, "unqualified development"):
            workflow.read_frozen_selection(selection_path, summary["protocol_sha256"], summary["selection_sha256"])
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            workflow.read_frozen_selection(selection_path, summary["protocol_sha256"], "0" * 64)
        selection = read_json(selection_path)
        chosen = selection["selected"]
        chosen["dev_metrics"] = {name: {"correct": 20, "total": 20, "format_valid": 20} for name in workflow.DEV_PANELS}
        chosen["development_gate"] = workflow.quality_data.development_gate(chosen["dev_metrics"])
        qualified_path = self.output / "test_qualified_selection.json"
        workflow.write_json(qualified_path, selection)
        with Path(chosen["checkpoint"]).open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            workflow.read_frozen_selection(qualified_path, summary["protocol_sha256"],
                                           workflow.model_lab.file_hash(qualified_path))

    def test_real_receipts_record_balanced_tasks_tokens_and_parameter_changes(self):
        with mock.patch.object(workflow, "CANDIDATES", workflow.CANDIDATES[:1]):
            summary = workflow.run(self.config, self.output)
        chosen = summary["selection"]
        logs = read_json(Path(chosen["checkpoint"]).parent / "training.json")
        self.assertEqual(len(logs["steps"]), 2)
        data = workflow.quality_data.fixed_quality_data()
        by_id = {row["id"]: row for row in data["train"]}
        tokenizer = workflow.model_lab.CharacterTokenizer()
        total_tokens = 0
        for update in logs["steps"]:
            self.assertEqual(update["sample_tasks"].count("arithmetic"), 16)
            self.assertEqual(update["sample_tasks"].count("copy"), 16)
            self.assertEqual(len(update["sample_ids"]), 32)
            records = [by_id[sample_id] for sample_id in update["sample_ids"]]
            expected = sum(len(tokenizer.encode(row["answer"])) + 1 for row in records)
            self.assertEqual(update["effective_tokens"], expected)
            total_tokens += expected
            self.assertGreater(update["gradient_norm_before_clip"], 0)
            self.assertGreater(update["parameter_delta_norm"], 0)
            self.assertTrue(__import__("math").isfinite(update["loss"]))
        self.assertEqual(chosen["effective_update_tokens"], total_tokens)
        self.assertEqual(chosen["training_samples"], 64)
        self.assertNotEqual(chosen["initial_weight_hash"], chosen["weight_hash"])
        restored, _ = workflow.model_lab.load_model_and_tokenizer(Path(chosen["checkpoint"]))
        self.assertEqual(workflow.model_lab.weight_hash(restored), chosen["weight_hash"])
        self.assertEqual(restored.data_profile, "quality-v2")
        self.assertEqual(chosen["data_profile"], "quality-v2")
        self.assertEqual(restored.config, chosen["model_config"])
        self.assertTrue(summary["training_diagnostic"]["scope"].startswith("read-only"))

    def test_final_commitments_and_checkpoint_profile_cannot_be_substituted(self):
        with mock.patch.object(workflow, "CANDIDATES", workflow.CANDIDATES[:1]), \
             mock.patch.object(workflow, "evaluate_panels", side_effect=self.fake_evaluation):
            summary = workflow.run(self.config, self.output)
        selected = summary["selection"]
        data = workflow.quality_data.fixed_quality_data()
        final = {name: data[name] for name in workflow.FINAL_PANELS}
        final["eval_main"][0]["id"] = "substituted-final-sample"
        with self.assertRaisesRegex(ValueError, "sample IDs or records"):
            workflow.open_final_evaluation(self.output / "selection.json", summary["protocol_sha256"],
                                           summary["selection_sha256"], final, selected,
                                           workflow.normalized_config(self.config))
        wrong_profile = dict(selected, data_profile="legacy-v1")
        with self.assertRaisesRegex(ValueError, "data profile"):
            workflow.load_checked_checkpoint(wrong_profile)
        wrong_architecture = dict(selected, candidate={**selected["candidate"], "width": 64})
        with self.assertRaisesRegex(ValueError, "model config"):
            workflow.load_checked_checkpoint(wrong_architecture)

    def test_strict_config_and_dev_only_input_validation(self):
        for config in ({"seeds": [23]}, {"seeds": [17, 17]}, {"checkpoint_steps": [2, 1]},
                       {"learning_rate": float("nan")}, {"feature_mode": "ordered-digit"}):
            with self.assertRaises(ValueError):
                workflow.normalized_config(config)
        data = workflow.quality_data.fixed_quality_data()
        with self.assertRaisesRegex(ValueError, "forbids final"):
            workflow.select_development(data, workflow.normalized_config(self.config), self.output)
        with self.assertRaisesRegex(ValueError, "must not receive final"):
            workflow.train_candidate(workflow.CANDIDATES[0], 17, [2], data,
                                     workflow.normalized_config(self.config), self.output)

    def test_cli_returns_four_for_unqualified_quality_and_refuses_directory_reuse(self):
        command = [sys.executable, "-X", "utf8", str(ROOT / "post_train/experiments/model_quality_lab.py"),
                   "--checkpoint-steps", "2", "--seeds", "17", "--output-dir", str(self.output)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(result.returncode, 4, result.stderr)
        summary = read_json(self.output / "summary.json")
        self.assertEqual(summary["status"], "development_unqualified")
        self.assertIsNone(summary["final_metrics"])
        original = workflow.model_lab.file_hash(self.output / "summary.json")
        repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(repeated.returncode, 1)
        self.assertEqual(workflow.model_lab.file_hash(self.output / "summary.json"), original)


if __name__ == "__main__":
    unittest.main()
