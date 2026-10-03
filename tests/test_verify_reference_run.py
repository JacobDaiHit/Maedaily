"""Independent scoring/package adversarial tests using only standard-library Python."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reference_verifier_test", ROOT / "scripts/verify_reference_run.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class IndependentReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "fixture-source"
        self.run = self.root / "ignored-run"
        self.run.mkdir()
        experiments = self.source / "post_train/experiments"
        experiments.mkdir(parents=True)
        shutil.copy2(ROOT / "post_train/experiments/reference_reporting.py", experiments / "reference_reporting.py")
        scripts = self.source / "scripts"
        scripts.mkdir()
        for name in ("verify_reference_run.py", "prepare_reference_model.py", "lesson_runtime.py"):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        self.module = verify.reporting(self.source)
        data = {name: [] for name in verify.SPLITS}
        def row(question, split):
            answer, family, bucket = verify.oracle(question)
            return {"id": split + ":" + question, "question": question, "answer": answer,
                    "family_id": family, "family": bucket, "split": split,
                    "task": "copy" if bucket == "copy" else "arithmetic"}
        data["train"] = [row("0+0=", "train"), row("copy 0", "train")]
        data["dev"] = [row("0+1=", "dev")]
        data["dev_copy"] = [row("copy 1", "dev_copy")]
        for left in range(2, 7):
            for right in (7, 8):
                for symbol in ("+", "-"):
                    data["eval_main"].append(row(f"{left}{symbol}{right}=", "eval_main"))
        data["eval_transfer"] = [row(f"{left}{symbol}10=", "eval_transfer") for left in range(10) for symbol in ("+", "-")]
        data["eval_regression"] = [row(f"copy {value}", "eval_regression") for value in range(2, 22)]
        self.data = data
        manifest = {"version": "fixture-only-frozen-protocol", "splits": data, "data_sha256": verify.digest(data)}
        config = {"seeds": [17, 23, 41], "max_new_tokens": 8, "bootstrap_replicates": 2,
                  "sampling_baseline_size": 4}
        model = {"repo_id": "fixture-only/model", "revision": "a" * 40,
                 "files": {"model.safetensors": {"sha256": "b" * 64, "bytes": 1}}}
        write(experiments / "reference_model.json", model)
        self.registration = {"config": config, "arms": list(verify.ARMS), "data": manifest,
            "model_file_hashes": {"model.safetensors": "b" * 64},
            "code_hashes": {"reference_reporting.py": verify.file_hash(experiments / "reference_reporting.py")},
            "final_used_for_selection": False, "preregistered_at": "2026-10-02T00:00:00+00:00", "fixture_only": True}
        write(self.run / "preregistration.json", self.registration)
        write(self.run / "config.resolved.json", config)
        write(self.run / "data_manifest.json", {"data": data, "manifest": manifest})
        write(self.run / "source_versions.json", self.registration["code_hashes"])
        self.receipts, self.records = [], []
        for seed in config["seeds"]:
            hashes = {arm: verify.digest({"seed": seed, "arm": arm}) for arm in verify.ARMS}
            hashes["sampling_only"] = hashes["no_update"]
            for arm in verify.ARMS:
                checkpoint = hashes[arm]
                steps = 0 if arm in ("base", "sampling_only") else 1
                receipt = {"training_seed": seed, "arm": arm, "training_run_id": f"{seed}:{arm}",
                    "checkpoint_id": checkpoint, "optimizer_steps": steps,
                    "initial_weight_hash": checkpoint if steps == 0 else hashes["base" if arm == "no_update" else "no_update"],
                    "final_weight_hash": checkpoint,
                    "source_training_run_id": f"{seed}:no_update" if arm == "sampling_only" else None,
                    "source_checkpoint_id": hashes["no_update"] if arm == "sampling_only" else None}
                self.receipts.append(receipt)
                if steps:
                    budget = {key: receipt[key] for key in ("optimizer_steps", "initial_weight_hash", "final_weight_hash")}
                    budget["logs"] = [{"sample_ids": [data["train"][0]["id"]]}]
                    if arm.startswith("rlvr"):
                        budget.update({"policy_gradient_proved": True,
                            "logs": [{"status": "updated", "policy_gradient_norm": 1.0}],
                            "collections": [{"groups": [{"sample_id": data["train"][0]["id"]}]}]})
                    write(self.run / f"seed_{seed}/{'sft' if arm == 'no_update' else arm}_train.json", budget)
                raw_rows = []
                for panel in verify.PANELS:
                    for item in data[panel]:
                        self.records.append({"sample_id": item["id"], "family_id": item["family_id"],
                            "panel": panel, "bucket": item["family"], "arm": arm, "training_seed": seed,
                            "training_run_id": f"{seed}:{arm}", "checkpoint_id": checkpoint,
                            "correct": True, "format_valid": True, "input_tokens": 2, "output_tokens": 2,
                            "evaluation_cost_tokens": 8 if arm == "sampling_only" else 2,
                            "evaluation_cost_input_tokens": 8 if arm == "sampling_only" else 2,
                            "elapsed_seconds": .001, "output_text": item["answer"],
                            "prompt_text": item["question"], "termination_reason": "eos",
                            "question": item["question"], "expected_answer": item["answer"],
                            "max_new_tokens": 8, "eval_protocol_hash": verify.digest({"data": manifest, "config": config, "arms": list(verify.ARMS)})})
                        raw = {"sample_id": item["id"], "question": item["question"], "answer": item["answer"],
                            "text": item["answer"], "prompt_token_ids": [1, 2], "response_token_ids": [3, 4],
                            "termination_reason": "eos", "elapsed_seconds": .001,
                            "reward": {"correct": True, "format_valid": True}, "panel": panel}
                        if arm == "sampling_only":
                            raw["all_samples"] = [copy.deepcopy(raw) for _ in range(4)]
                        raw_rows.append(raw)
                write(self.run / f"seed_{seed}/{arm}_evaluation.json", {"predictions": raw_rows,
                    "panels": {panel: {"correct": 20, "format_valid": 20, "total": 20} for panel in verify.PANELS}})
        write(self.run / "training_receipts.json", self.receipts)
        self.save_records()
        write(self.run / "report.json", self.module.build_report(self.records, training_receipts=self.receipts,
            baseline_arm="no_update", bootstrap_replicates=config["bootstrap_replicates"]))
        write(self.run / "summary.json", {"status": "passed", "mechanisms_passed": True,
            "executed_training_seeds": config["seeds"], "arms": list(verify.ARMS), "contract_hash": verify.digest(self.registration), "fixture_only": True})
        write(self.run / "sft_lock.json", {"contract_hash": verify.digest(self.registration),
            "weights": {str(item["training_seed"]): item["checkpoint_id"] for item in self.receipts if item["arm"] == "no_update"}})
        write(self.run / "final_opening.json", {"protocol_hash": self.records[0]["eval_protocol_hash"],
            "selection_locked": True, "no_retuning_after_final": True, "at": "2026-10-02T00:00:00+00:00"})
        batch = {name: {"passed": True} for name in ("sft", "dpo", "frozen", "verifier")}
        resume = {"passed": True, "comparisons": {"weights": True, "optimizer": True, "RNG": True}}
        write(self.run / "batch_checks.json", batch)
        write(self.run / "resume_check.json", resume)
        reloads = [{"seed": item["training_seed"], "arm": item["arm"], "weight_hash": item["checkpoint_id"], "reload_exact": True}
                   for item in self.receipts]
        write(self.run / "reload_checks.json", {"reloaded_checkpoints": reloads})
        # Synthetic fixtures exercise verification, never claim actual model training.
        weight = self.run / "seed_17/base.pt"
        weight.parent.mkdir(exist_ok=True)
        weight.write_bytes(b"fixture weight bytes only")
        hashes = {path.relative_to(self.run).as_posix(): verify.file_hash(path) for path in self.run.rglob("*") if path.is_file()}
        names = ["00_contract", "01_batch", "exact_resume"] + [f"{seed}:02_sft" for seed in config["seeds"]]
        names += [name for seed in config["seeds"] for name in (f"{seed}:03_dpo", f"{seed}:04_rollout_05_update")]
        names += ["checkpoint_reload", "06_eval", "07_reproduce"]
        stages = []
        for name in names:
            checks = {}
            if name == "00_contract":
                checks = {"contract_hash": verify.digest(self.registration), "data_hash": manifest["data_sha256"]}
            elif name == "01_batch":
                checks = batch
            elif name == "exact_resume":
                checks = resume
            elif name.endswith(":02_sft"):
                checks = {"development_qualified": True, "final_opened": False}
            elif name.endswith(":04_rollout_05_update"):
                checks = {"policy_gradient_proved": {arm: True for arm in ("rlvr", "rlvr_group_size", "rlvr_no_format")}}
            elif name == "checkpoint_reload":
                checks = {"reloaded_checkpoints": reloads}
            elif name == "06_eval":
                checks = {"all_arms_reported": list(verify.ARMS), "all_seeds_reported": config["seeds"]}
            elif name == "07_reproduce":
                checks = {"mechanisms_passed": True, "baseline_qualified": True}
            for status in ("running", "passed"):
                stages.append({"stage": name, "status": status, "started_at": "2026-10-02T00:00:00+00:00",
                    "finished_at": "2026-10-02T00:00:00+00:00",
                    "input_artifact_hashes": hashes, "output_artifact_hashes": {}, "failure_reason": None, "checks": checks})
        (self.run / "stages.jsonl").write_text("".join(verify.canonical(row).decode() + "\n" for row in stages), encoding="utf-8")

    def save_records(self):
        (self.run / "predictions.jsonl").write_text("".join(verify.canonical(row).decode() + "\n" for row in self.records), encoding="utf-8")

    def test_oracle_rejects_wrong_gold_even_with_updated_manifest_hash(self):
        document = verify.read_json(self.run / "data_manifest.json")
        document["data"]["eval_main"][0]["answer"] = "999"
        document["manifest"]["splits"] = document["data"]
        document["manifest"]["data_sha256"] = verify.digest(document["data"])
        registration = {**self.registration, "data": document["manifest"]}
        with self.assertRaisesRegex(ValueError, "wrong frozen gold"):
            verify.validate_data(document, registration)

    def test_format_is_canonical_length_and_eos(self):
        for text in ("", "05", "+5", "5 ", "5\n", "-0", "99999", "５"):
            self.assertFalse(verify.score("2+3=", text, "eos")["format_valid"], text)
        self.assertTrue(verify.score("2-5=", "-3", "eos")["correct"])
        self.assertFalse(verify.score("2+3=", "5", "new_token_budget")["correct"])
        self.assertTrue(verify.score("copy 0005", "5", "eos")["correct"])

    def test_cross_python_float_last_bits_only_derived_cost_tolerated(self):
        report = {"correct": 34, "format_valid": True, "sample_id": "frozen:1",
                  "confidence": .95, "cost": {"summed_generation_seconds": .8660299999755806}}
        changed = copy.deepcopy(report)
        changed["cost"]["summed_generation_seconds"] += 1e-13
        self.assertTrue(verify.equivalent_derived(report, changed))
        changed["cost"]["summed_generation_seconds"] += 1e-4
        self.assertFalse(verify.equivalent_derived(report, changed))
        for key, value in (("correct", 33), ("correct", 34.0), ("format_valid", False), ("sample_id", "frozen:2"), ("confidence", .95 + 1e-13)):
            changed = copy.deepcopy(report)
            changed[key] = value
            self.assertFalse(verify.equivalent_derived(report, changed), key)
        changed = copy.deepcopy(report)
        changed["cost"]["summed_generation_seconds"] = float("inf")
        self.assertFalse(verify.equivalent_derived(report, changed))

    def test_tampered_scores_rejected_before_saved_report(self):
        self.records[0]["correct"] = False
        self.save_records()
        with self.assertRaisesRegex(ValueError, "scores differ"):
            verify.verify_run(self.run, self.source)

    def test_missing_seed_cannot_be_relabelled_as_completed(self):
        self.records = [row for row in self.records if row["training_seed"] != 41]
        self.save_records()
        with self.assertRaisesRegex(ValueError, "missing prediction/seed"):
            verify.verify_run(self.run, self.source)

    def test_wrong_family_and_duplicate_identity_hard_fail(self):
        lookup = verify.validate_data(verify.read_json(self.run / "data_manifest.json"), self.registration)
        changed = copy.deepcopy(self.records)
        changed[0]["family_id"] = "operand-pair:9:9"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            verify.verify_predictions(changed, lookup, self.registration["config"], self.registration, self.receipts)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify.verify_predictions(self.records + [self.records[0]], lookup, self.registration["config"], self.registration, self.receipts)

    def test_saved_report_and_stage_hash_tampering_rejected(self):
        report = verify.read_json(self.run / "report.json")
        report["records"] -= 1
        write(self.run / "report.json", report)
        with self.assertRaisesRegex(ValueError, "report differs"):
            verify.verify_run(self.run, self.source)

    def test_failed_formal_run_cannot_export_success(self):
        write(self.run / "failure.json", {"status": "failed", "reason": "no policy gradient"})
        with self.assertRaisesRegex(ValueError, "failed run"):
            verify.export_run(self.run, self.root / "public", self.source)
        self.assertFalse((self.root / "public").exists())

    def test_all_candidate_cost_cannot_be_replaced_by_selected_length(self):
        row = next(row for row in self.records if row["arm"] == "sampling_only")
        row["evaluation_cost_tokens"] = row["output_tokens"]
        self.save_records()
        with self.assertRaisesRegex(ValueError, "evaluation cost differs"):
            verify.verify_run(self.run, self.source)

    def test_sampling_oracle_selection_is_rejected_by_raw_majority(self):
        path = self.run / "seed_17/sampling_only_evaluation.json"
        evaluation = verify.read_json(path)
        for candidate in evaluation["predictions"][0]["all_samples"][:3]:
            candidate["text"] = "999"
            candidate["reward"]["correct"] = False
        write(path, evaluation)
        with self.assertRaisesRegex(ValueError, "text/EOS-only majority"):
            verify.verify_run(self.run, self.source)

    def test_only_kl_or_zero_gradient_cannot_claim_rl_update(self):
        path = self.run / "seed_17/rlvr_train.json"
        budget = verify.read_json(path)
        budget["logs"][0]["policy_gradient_norm"] = 0
        write(path, budget)
        with self.assertRaisesRegex(ValueError, "positive policy-gradient"):
            verify.verify_run(self.run, self.source)

    def test_registered_script_sources_resolve_safely(self):
        self.assertEqual(verify.source_path(self.source, "scripts/lesson_runtime.py"), self.source / "scripts/lesson_runtime.py")
        self.assertEqual(verify.source_path(self.source, "reference_reporting.py"), self.source / "post_train/experiments/reference_reporting.py")
        for name in ("../outside.py", "D:/outside.py", "scripts/../../outside.py"):
            with self.assertRaises(ValueError):
                verify.source_path(self.source, name)

    def test_sft_numerator_denominator_and_shift_recomputed(self):
        identities = [row["id"] for row in self.data["train"]]
        lookup = {row["id"]: row for row in self.data["train"]}
        tokenized = {"splits": {"train": {"records": [
            {"source_sample_id": identities[0], "prompt_tokens": 4, "target_tokens": 2},
            {"source_sample_id": identities[1], "prompt_tokens": 6, "target_tokens": 3}]}}}
        receipt = {"effective_token_denominator": 5, "negative_log_likelihood_numerator": 8.0, "loss": 1.6,
            "per_sample": [{"sample_id": identities[0], "effective_tokens": 2, "target_positions": [4, 5],
                "token_logprobs": [-1., -1.], "negative_log_likelihood": 2., "sequence_logprob": -2.},
                {"sample_id": identities[1], "effective_tokens": 3, "target_positions": [6, 7, 8],
                 "token_logprobs": [-2., -2., -2.], "negative_log_likelihood": 6., "sequence_logprob": -6.}]}
        self.assertEqual(verify.verify_sft_receipt(receipt, tokenized, lookup)["effective_token_denominator"], 5)
        wrong = copy.deepcopy(receipt)
        wrong["loss"] = 1.5  # averaging per-sequence means gives the wrong objective
        with self.assertRaisesRegex(ValueError, "whole-batch"):
            verify.verify_sft_receipt(wrong, tokenized, lookup)
        wrong = copy.deepcopy(receipt)
        wrong["per_sample"][0]["target_positions"] = [3, 4]
        with self.assertRaisesRegex(ValueError, "shift"):
            verify.verify_sft_receipt(wrong, tokenized, lookup)

    def test_two_factor_arm_change_is_rejected_even_if_named_single_factor(self):
        config = {"seeds": [17, 23, 41], "beta": .2, "beta_ablation": .1,
            "group_size": 4, "group_size_ablation": 8, "format_reward": .1, "learning_rate": .001}
        arms = {arm: dict(config) for arm in verify.ARMS}
        arms["dpo_beta"]["beta"] = .1
        arms["rlvr_group_size"]["group_size"] = 8
        arms["rlvr_no_format"]["format_reward"] = 0.
        arms["rlvr_group_size"]["learning_rate"] = .002
        write(self.run / "arm_configs.json", arms)
        with self.assertRaisesRegex(ValueError, "beyond registered factor"):
            verify.verify_arm_configs(self.run, config, self.module)

    def test_exploration_counts_actual_steps_not_repeated_checkpoint_views(self):
        documents = {"lr0.1_train.json": [{"step": 1, "sample_ids": ["train:a"]}, {"step": 2, "sample_ids": ["train:b"]}],
            "summary.json": {"results": [{"step": 1}, {"step": 2}, {"step": 2}]}}
        budget = verify.summarize_budget_documents(documents, "development_exploration")
        self.assertEqual(budget["optimizer_steps_logged"], 2)
        self.assertIsNone(budget["effective_update_tokens_logged"])
        self.assertFalse(budget["complete_total_runtime_measured"])
        self.assertIn("unknown", budget["unlogged_failed_work"])

    def test_derived_rl_ratios_masks_and_zero_variance(self):
        identity = self.data["train"][0]["id"]
        frozen = self.data["train"][0]
        rollouts = [{"sample_id": identity, "question": frozen["question"], "text": "0", "termination_reason": "eos",
            "response_token_ids": [1, 2], "reward": {"correct": True, "format_valid": True}, "reward_total": 1.1, "group_id": 0} for _ in range(2)]
        collection = {"rollouts": rollouts, "groups": [{"group_id": 0, "rewards": [1.1, 1.1]}],
            "action_mask": [[False, True], [False, True]], "old_token_logprobs": [[0, -1], [0, -1]],
            "current_token_logprobs_before": [[0, -1], [0, -.5]], "ratios_before": [[1, 1], [1, 1.6487212707001282]]}
        for arm in ("rlvr", "rlvr_group_size", "rlvr_no_format"):
            rows = copy.deepcopy(collection)
            if arm == "rlvr_no_format":
                rows["groups"][0]["rewards"] = [1., 1.]
                for row in rows["rollouts"]:
                    row["reward_total"] = 1.
            write(self.run / f"seed_17/{arm}_train.json", {"collections": [rows]})
        result = verify.derived_rl_analysis(self.run, {"seeds": [17], "format_reward": .1, "clip": .2, "max_new_tokens": 8}, {identity: frozen})
        arm = result["arms"]["17:rlvr"]
        self.assertEqual(arm["pre_update_clip_fraction"], .5)
        self.assertEqual(arm["zero_variance_fraction"], 1.)
        self.assertEqual(arm["ratio_tokens"], 2)
        self.assertEqual(arm["generated_tokens"], 4)

    def test_public_package_rescores_without_original_ignored_run(self):
        budget_sources = self.root / "budget-sources.json"
        write(budget_sources, {"fixture_recorded_cost": {"directory": str(self.run), "classification": "unit_test_fixture_only"}})
        result = verify.export_run(self.run, self.root / "public", self.source, budget_sources=budget_sources, runner_exit_code=0)
        self.assertEqual(result["status"], "verified")
        self.assertLessEqual(result["package_bytes"], 32 * 1024 * 1024)
        self.assertEqual(result["stages"]["omitted_weight_files"], 1)
        self.assertFalse(result["stages"]["weight_bytes_independently_verified"])
        self.assertFalse(any(path.name.endswith(".sealed.json") for path in (self.root / "public").rglob("*")))
        self.assertEqual(verify.read_json(self.root / "public/09_budget_audit.json")["experiments"]["fixture_recorded_cost"]["summary"]["optimizer_steps_logged"], 21)
        clone = self.root / "clone"
        shutil.copytree(self.root / "public", clone)
        shutil.rmtree(self.run)
        shutil.rmtree(self.source)
        completed = subprocess.run([sys.executable, str(clone / "scripts/verify_reference_run.py"),
            "--verify-package", str(clone)], capture_output=True, text=True, encoding="utf-8", check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["prediction_count"], len(self.records))
        (clone / "run/predictions.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "size/hash mismatch"):
            verify.verify_package(clone)


if __name__ == "__main__":
    unittest.main()
