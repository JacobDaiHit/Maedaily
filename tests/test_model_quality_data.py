"""Independent semantic, leakage and boundary checks for the frozen quality data."""
import copy
import importlib.util
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("quality_data", ROOT / "post_train/experiments/model_quality_data.py")
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


def panel(correct=20, total=20, valid=20):
    return {"correct": correct, "total": total, "format_valid": valid}


class QualityDataTests(unittest.TestCase):
    def test_frozen_data_is_deterministic_and_returns_independent_objects(self):
        first, second = quality.fixed_quality_data(), quality.fixed_quality_data()
        self.assertEqual(first, second)
        self.assertTrue(quality.validate_quality_data(first)["passed"])
        first["train"][0]["answer"] = "999"
        self.assertNotEqual(first, second)
        self.assertEqual(second, quality.fixed_quality_data())
        self.assertEqual(set(second), {"train", "dev", "dev_copy", "eval_main", "eval_transfer", "eval_regression"})
        for name in ("dev", "dev_copy", "eval_main", "eval_regression"):
            self.assertGreaterEqual(len(second[name]), 20)

    def test_independent_prompt_semantics_and_training_answer_coverage(self):
        data = quality.fixed_quality_data()
        owners, training_labels, questions = {}, set(), set()
        for split, records in data.items():
            for row in records:
                self.assertNotIn(row["question"], questions)
                questions.add(row["question"])
                match = re.fullmatch(r"(\d+)([+-])(\d+)=", row["question"])
                if match:
                    left, operation, right = match.groups()
                    left, right = int(left), int(right)
                    canonical = ("arithmetic", min(left, right), max(left, right))
                    expected = left + right if operation == "+" else left - right
                    if split == "train":
                        training_labels.add(expected)
                else:
                    value = int(row["question"].removeprefix("copy "))
                    canonical, expected = ("copy", value), value
                self.assertEqual(row["answer"], str(expected))
                if canonical in owners:
                    self.assertEqual(owners[canonical], split)
                owners[canonical] = split
        self.assertEqual(training_labels, set(range(-9, 19)))

    def test_all_copy_spellings_share_value_family_and_split(self):
        data = quality.fixed_quality_data()
        found = {}
        for split, records in data.items():
            for row in records:
                if row["question"].startswith("copy "):
                    value = int(row["question"][5:])
                    found.setdefault(value, []).append((split, row))
        self.assertEqual(set(found), set(range(100)))
        for value, rows in found.items():
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({split for split, _ in rows}), 1)
            self.assertEqual({row["question"] for _, row in rows},
                             {f"copy {value}", f"copy 0{value}", f"copy 00{value}"})
            self.assertEqual({row["family"] for _, row in rows}, {"copy"})
            self.assertEqual({row["answer"] for _, row in rows}, {str(value)})

    def test_semantic_family_tampering_and_cross_split_alias_are_rejected(self):
        data = quality.fixed_quality_data()
        copied = copy.deepcopy(data)
        copied["train"][0]["family_id"] = "invented-family"
        with self.assertRaisesRegex(ValueError, "independent prompt semantics"):
            quality.validate_quality_data(copied)
        row = next(record for record in data["train"] if record["task"] == "copy")
        data["train"].remove(row)
        row["split"], row["id"] = "dev_copy", "moved-copy-alias"
        data["dev_copy"].append(row)
        with self.assertRaisesRegex(ValueError, "family crosses splits"):
            quality.validate_quality_data(data)

    def test_missing_training_label_and_small_panel_are_rejected(self):
        data = quality.fixed_quality_data()
        data["train"] = [row for row in data["train"] if row["answer"] != "-9"]
        with self.assertRaisesRegex(ValueError, "arithmetic labels"):
            quality.validate_quality_data(data)
        data = quality.fixed_quality_data()
        data["eval_main"] = data["eval_main"][:19]
        with self.assertRaisesRegex(ValueError, "at least 20"):
            quality.validate_quality_data(data)

    def test_quality_gate_uses_exact_boundaries_counts_and_format(self):
        metrics = {"eval_main": panel(16), "eval_regression": panel(18), "eval_transfer": panel(0)}
        self.assertTrue(quality.quality_gate(metrics)["qualified"])
        for name, replacement in (("eval_main", panel(15)), ("eval_regression", panel(17)),
                                  ("eval_regression", panel(18, valid=19)), ("eval_main", panel(8, 10, 10))):
            changed = copy.deepcopy(metrics)
            changed[name] = replacement
            self.assertFalse(quality.quality_gate({"panels": changed})["qualified"])
        # A forged accuracy field cannot override counts.
        metrics["eval_main"].update(correct=15, accuracy=1.0)
        self.assertFalse(quality.quality_gate(metrics)["qualified"])
        for invalid in (panel(True), panel(21), panel(19, valid=18), panel(total=0, correct=0, valid=0)):
            with self.assertRaisesRegex(ValueError, "counts"):
                quality.quality_gate({"eval_main": invalid, "eval_regression": panel(18)})

    def test_selection_is_dev_only_and_prefers_copy_format_then_arithmetic(self):
        valid = {"dev": panel(16), "dev_copy": panel(18)}
        losing_copy = {"dev": panel(20), "dev_copy": panel(17)}
        self.assertGreater(quality.selection_key(valid, nll=1.0, step=20),
                           quality.selection_key(losing_copy, nll=0.1, step=10))
        better_arithmetic = {"dev": panel(17), "dev_copy": panel(18)}
        self.assertGreater(quality.selection_key(better_arithmetic, nll=2.0, step=30),
                           quality.selection_key(valid, nll=1.0, step=20))
        self.assertGreater(quality.selection_key(valid, nll=0.9, step=30),
                           quality.selection_key(valid, nll=1.0, step=20))
        self.assertGreater(quality.selection_key(valid, nll=1.0, step=10),
                           quality.selection_key(valid, nll=1.0, step=20))
        for illegal in ({"eval_main": panel(20), "eval_regression": panel(20)},
                        {**valid, "eval_transfer": panel(20)}, {"dev": panel(20)}):
            with self.assertRaisesRegex(ValueError, "eval metrics are forbidden"):
                quality.selection_key(illegal, nll=1.0, step=10)
        for invalid_nll in (float("nan"), float("inf"), -0.1):
            with self.assertRaisesRegex(ValueError, "NLL"):
                quality.selection_key(valid, nll=invalid_nll, step=10)

    def test_development_qualification_does_not_open_final_panels(self):
        metrics = {"dev": panel(16), "dev_copy": panel(18)}
        result = quality.development_gate(metrics)
        self.assertTrue(result["qualified"])
        self.assertEqual(set(result["panels"]), {"dev", "dev_copy"})
        metrics["dev_copy"] = panel(17)
        self.assertFalse(quality.development_gate(metrics)["qualified"])
        with self.assertRaisesRegex(ValueError, "exactly dev"):
            quality.development_gate({**metrics, "eval_main": panel(20)})


if __name__ == "__main__":
    unittest.main()
