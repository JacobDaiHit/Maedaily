"""Independent interval, clustered pairing, provenance, blinding and package tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "post_train/experiments/reference_reporting.py"
SPEC = importlib.util.spec_from_file_location("reference_reporting", SCRIPT)
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def records(arm="no_update", seed=17, repeats=1):
    result = []
    for index in range(4):
        for generation in range(repeats):
            first_family = index < 2
            correct = not first_family if arm == "no_update" else first_family
            result.append({"sample_id": f"sample-{index}", "family_id": "family-a" if first_family else "family-b",
                           "panel": "main", "bucket": "small", "arm": arm, "training_seed": seed,
                           "training_run_id": f"{arm}-{seed}", "checkpoint_id": f"checkpoint-{arm}-{seed}",
                           "correct": correct, "format_valid": True, "input_tokens": 3, "output_tokens": 2,
                           "elapsed_seconds": 0.5, "output_text": "2", "termination_reason": "eos",
                           "prompt_text": f"question {index}", "generation_index": generation,
                           "eval_protocol_hash": "frozen-example", "max_new_tokens": 4, "example_only": True})
    return result


def receipts(rows):
    result = []
    for arm, seed, run, checkpoint in sorted({(r["arm"], r["training_seed"], r["training_run_id"], r["checkpoint_id"]) for r in rows}):
        initial = hashlib.sha256(f"initial-{seed}".encode()).hexdigest()
        result.append({"arm": arm, "training_seed": seed, "training_run_id": run, "checkpoint_id": checkpoint,
                       "optimizer_steps": 0 if arm == "no_update" else 2, "initial_weight_hash": initial,
                       "final_weight_hash": initial if arm == "no_update" else hashlib.sha256(run.encode()).hexdigest()})
    return result


class ReportingTests(unittest.TestCase):
    def test_wilson_boundaries_and_independent_score_equation(self):
        zero, full = reporting.wilson_interval(0, 10), reporting.wilson_interval(10, 10)
        self.assertAlmostEqual(zero["low"], 0)
        self.assertAlmostEqual(zero["high"], 0.2775327998628892)
        self.assertAlmostEqual(full["low"], 1 - zero["high"])
        self.assertAlmostEqual(full["high"], 1)
        half = reporting.wilson_interval(5, 10)
        z = statistics.NormalDist().inv_cdf(0.975)
        for boundary in (half["low"], half["high"]):
            self.assertAlmostEqual((0.5-boundary)**2 / (boundary*(1-boundary)/10), z*z)
        self.assertIsNone(reporting.wilson_interval(0, 0)["estimate"])
        for correct, total in ((True, 10), (11, 10), (1, -1)):
            with self.assertRaises(ValueError):
                reporting.wilson_interval(correct, total)

    def test_bootstrap_resamples_whole_families_and_is_order_invariant(self):
        baseline, candidate = records(repeats=3), records("rlvr", repeats=3)
        result = reporting.paired_family_bootstrap(baseline, candidate, seed=23, replicates=1000)
        self.assertEqual(result["difference"], 0)
        self.assertEqual(result["family_counts"], {"main": 2})
        self.assertEqual(result["low"], -1)
        self.assertEqual(result["high"], 1)
        random.Random(5).shuffle(baseline)
        random.Random(6).shuffle(candidate)
        self.assertEqual(result, reporting.paired_family_bootstrap(baseline, candidate, seed=23, replicates=1000))

    def test_pairing_rejects_duplicates_missing_and_semantic_changes(self):
        baseline, original = records(), records("rlvr")
        cases = [original + [copy.deepcopy(original[0])], original[:-1]]
        for key, value in (("family_id", "other"), ("bucket", "other"), ("prompt_text", "changed"),
                           ("eval_protocol_hash", "changed"), ("max_new_tokens", 6)):
            changed = copy.deepcopy(original)
            changed[0][key] = value
            cases.append(changed)
        cases.append(records("rlvr", seed=23))
        for candidate in cases:
            with self.subTest(candidate=candidate[0].get("family_id")), self.assertRaises(ValueError):
                reporting.paired_family_bootstrap(baseline, candidate, replicates=20)

    def test_training_repeats_are_not_independent_training_seeds(self):
        rows = [row for seed in (17, 23, 41) for arm in ("no_update", "rlvr") for row in records(arm, seed, repeats=4)]
        report = reporting.build_report(rows, training_receipts=receipts(rows), bootstrap_replicates=30)
        self.assertEqual(report["training_provenance"]["verified_training_seed_count"], 3)
        self.assertEqual(report["source_scope"], "synthetic_example")
        self.assertFalse(report["training_provenance"]["independent_training_seeds_verified"])
        self.assertFalse(report["training_provenance"]["generation_repeats_are_training_seeds"])
        self.assertEqual(report["multi_seed"][0]["panels"]["main"]["training_seed_count"], 3)
        unknown = reporting.build_report(rows, bootstrap_replicates=10)
        self.assertFalse(unknown["training_provenance"]["independent_training_seeds_verified"])
        wrong = copy.deepcopy(rows)
        wrong[-1]["training_run_id"] = wrong[0]["training_run_id"]
        with self.assertRaisesRegex(ValueError, "relabel"):
            reporting.build_report(wrong, bootstrap_replicates=10)

    def test_missing_seeds_and_changed_cross_seed_eval_are_rejected(self):
        rows = records() + records(seed=23) + records("rlvr")
        with self.assertRaisesRegex(ValueError, "seed sets"):
            reporting.build_report(rows, bootstrap_replicates=10)
        rows = records() + records(seed=23) + records("rlvr") + records("rlvr", seed=23)
        rows[4]["sample_id"] = "changed-id"
        with self.assertRaisesRegex(ValueError, "frozen identities"):
            reporting.build_report(rows, bootstrap_replicates=10)

    def test_metrics_preserve_bucket_format_length_and_actual_cost(self):
        rows = records()
        rows[0].update(correct=False, format_valid=False, termination_reason="new_token_budget", output_text="")
        report = reporting.build_report(rows, bootstrap_replicates=10)
        values = report["runs"][0]["buckets"]["main/small"]
        self.assertEqual((values["correct"], values["total"], values["format_valid"]), (2, 4, 3))
        self.assertEqual(values["termination_counts"]["new_token_budget"], 1)
        self.assertEqual(values["cost"]["input_tokens"], 12)
        self.assertEqual(values["cost"]["output_tokens"], 8)
        self.assertEqual(values["cost"]["summed_generation_seconds"], 2)
        self.assertEqual(values["length"]["mean_output_tokens"], 2)

    def test_sampling_cost_is_distinct_from_chosen_response_length(self):
        rows = records()
        rows[0].update(evaluation_cost_tokens=12, evaluation_cost_input_tokens=30)
        report = reporting.build_report(rows, bootstrap_replicates=10)
        values = report["runs"][0]["panels"]["main"]
        self.assertEqual(values["length"]["mean_output_tokens"], 2)
        self.assertEqual(values["result_length"]["maximum"], 2)
        self.assertEqual(values["cost"]["output_tokens"], 18)
        self.assertEqual(values["cost"]["input_tokens"], 39)
        self.assertEqual(values["cost"]["selected_response_output_tokens"], 8)
        for field in ("evaluation_cost_tokens", "evaluation_cost_input_tokens"):
            changed = copy.deepcopy(rows)
            changed[0][field] = 1
            with self.assertRaises(ValueError):
                reporting.build_report(changed, bootstrap_replicates=10)

    def test_update_receipts_cannot_fake_independent_actual_runs(self):
        rows = records() + records("rlvr")
        actual = receipts(rows)
        for changed in (actual[:-1], actual + [actual[0]], [dict(actual[0], optimizer_steps=1), actual[1]]):
            with self.assertRaises(ValueError):
                reporting.build_report(rows, training_receipts=changed, bootstrap_replicates=10)
        changed = copy.deepcopy(actual)
        changed[1]["final_weight_hash"] = changed[1]["initial_weight_hash"]
        with self.assertRaisesRegex(ValueError, "weights did not change"):
            reporting.build_report(rows, training_receipts=changed, bootstrap_replicates=10)

    def test_exact_single_factor_change_has_no_ignored_runtime_fields(self):
        base = {"seeds": [17, 23, 41], "optimizer": {"lr": 0.01}, "loss": {"beta": 0.2}}
        variant = copy.deepcopy(base)
        variant["loss"]["beta"] = 0.1
        result = reporting.single_factor_difference(base, variant, expected_path="loss.beta")
        self.assertEqual(result["changed_factor"]["before"], 0.2)
        with self.assertRaises(ValueError):
            reporting.single_factor_difference(base, base)
        variant["optimizer"]["lr"] = 0.02
        with self.assertRaises(ValueError):
            reporting.single_factor_difference(base, variant)
        with self.assertRaises(ValueError):
            reporting.single_factor_difference(base, {**base, "output_dir": "other"}, expected_path="loss.beta")

    def test_blind_review_removes_identity_and_blank_sheet_is_unreviewed(self):
        rows = records() + records("rlvr")
        packet = reporting.blind_review_packet(rows, seed=19)
        self.assertEqual(packet, reporting.blind_review_packet(list(reversed(rows)), seed=19))
        for row in packet["records"]:
            self.assertFalse({"arm", "training_seed", "training_run_id", "checkpoint_id", "correct", "format_valid"} & row.keys())
        result = reporting.review_summary(packet, packet["blank_reviews"])
        self.assertEqual(result["status"], "not_reviewed")
        self.assertEqual(result["reviewed"], 0)
        self.assertFalse(result["third_party_human_review_verified"])
        review = dict(packet["blank_reviews"][0], status="reviewed", reviewer_id="test-ai", reviewer_type="ai-assisted",
                      correct=False, format_valid=True, reason="synthetic test review")
        result = reporting.review_summary(packet, [review])
        self.assertEqual(result["by_type"], {"ai-assisted": 1})
        self.assertFalse(result["third_party_human_review_verified"])

    def test_evidence_is_self_contained_detects_tampering_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text('{"real_file": true}\n', encoding="utf-8")
            package = root / "evidence"
            reporting.create_evidence_package({"records/data.json": source}, package)
            self.assertTrue(reporting.verify_evidence_package(package)["passed"])
            with self.assertRaisesRegex(ValueError, "immutable"):
                reporting.create_evidence_package({"records/data.json": source}, package)
            with self.assertRaises(ValueError):
                reporting.verify_evidence_package(package, required_files=["ignored/study_runs/missing.pt"])
            (package / "records/data.json").write_text('{"real_file": false}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                reporting.verify_evidence_package(package)

    def test_evidence_rejects_traversal_alias_external_and_missing_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.txt"
            source.write_text("real", encoding="utf-8")
            for name in ("../escaped.txt", "/absolute.txt", "C:/outside.txt", "a//alias.txt", "a\\windows.txt"):
                with self.assertRaises(ValueError):
                    reporting.create_evidence_package({name: source}, root / "invalid")
            with self.assertRaises(ValueError):
                reporting.create_evidence_package({"missing.pt": root / "ignored/missing.pt"}, root / "invalid")
            with self.assertRaises(ValueError):
                reporting.create_evidence_package({"data.txt": source}, root / "invalid", required_files=["missing.pt"])
            with self.assertRaises(ValueError):
                reporting.create_evidence_package({"data.txt": source}, root / "invalid", metadata={"external_required_artifacts": ["study_runs/x"]})

    def test_strict_json_rejects_duplicate_nonfinite_and_overflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.jsonl"
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[1]'):
                path.write_text(text + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    reporting.read_jsonl(path)

    def test_synthetic_records_cannot_be_mixed_with_actual_predictions(self):
        rows = records()
        del rows[0]["example_only"]
        with self.assertRaisesRegex(ValueError, "synthetic examples"):
            reporting.build_report(rows, bootstrap_replicates=10)

    def test_cli_generates_portable_recomputable_package_and_seals_map_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = records() + records("rlvr")
            source = root / "predictions.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "report"
            command = [sys.executable, str(SCRIPT), "--records", str(source), "--output-dir", str(output), "--bootstrap-replicates", "20"]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((root / "report_sealed_identity_map.json").is_file())
            manifest = reporting.verify_evidence_package(output / "evidence")
            self.assertTrue(manifest["passed"])
            self.assertFalse(any("identity_map" in str(path) for path in (output / "evidence").rglob("*")))
            copied = root / "copied_package"
            import shutil
            shutil.copytree(output / "evidence", copied)
            completed = subprocess.run([sys.executable, str(copied / "post_train/experiments/reference_reporting.py"),
                                        "--records", str(copied / "records.jsonl"), "--output-dir", str(root / "recomputed"),
                                        "--bootstrap-replicates", "20"], cwd=copied, capture_output=True, text=True,
                                       encoding="utf-8", timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            before = json.loads((output / "report.json").read_text(encoding="utf-8"))
            after = json.loads((root / "recomputed/report.json").read_text(encoding="utf-8"))
            self.assertEqual(before["paired_comparisons"], after["paired_comparisons"])
            self.assertEqual(before["runs"], after["runs"])


if __name__ == "__main__":
    unittest.main()
