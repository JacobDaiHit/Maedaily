"""B/C orchestration gates with an offline CPU adapter double, no model download."""
from contextlib import ExitStack, contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "post_train/experiments/project_bc_reference.py"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(SCRIPT.parent))
        spec = importlib.util.spec_from_file_location("project_bc_test_target", SCRIPT)
        project = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(project)
        import adapter_checks
    finally:
        sys.path[:] = old_path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@unittest.skipUnless(TORCH_AVAILABLE, "B/C gate tests need optional CPU PyTorch, not a pretrained model")
class ReferenceProjectTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @contextmanager
    def offline_run(self, *, development_qualified=True, rl_signal=True):
        """Replace heavy training only; retain real stages, config, data and report."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as patches:
            root = Path(temporary)
            output = root / "run"
            output.mkdir()
            model_dir = root / "local-model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"fixture_only":true}', encoding="utf-8")
            events = []

            class FakeAdapter:
                def __init__(self, path, *, device, seed, rank, alpha):
                    self.device = "cpu"
                    self.config = {"seed": seed, "rank": rank, "alpha": alpha}
                    self.model = torch.nn.Module()
                    self.model.adapter = torch.nn.Module()
                    self.model.adapter.a = torch.nn.Parameter(torch.zeros(1, 1))
                    self.model.adapter.b = torch.nn.Parameter(torch.zeros(1, 1))
                    self.tokenizer = type("Tokenizer", (), {"decode": lambda self, ids: "fixed fixture prompt"})()
                    if not (output / "preregistration.json").is_file():
                        raise AssertionError("model operation began before registration")
                    events.append(("model_loaded", seed))

                def parameters(self):
                    return list(self.model.parameters())

                def snapshot(self):
                    return {name: value.detach().clone() for name, value in self.model.named_parameters()}

                def load_snapshot(self, snapshot):
                    with torch.no_grad():
                        for name, value in self.model.named_parameters():
                            value.copy_(snapshot[name])

                def save(self, path, optimizer=None, **kwargs):
                    with Path(path).open("xb") as stream:
                        torch.save({"snapshot": self.snapshot(), "fixture_only": True, "kwargs": {k:v for k,v in kwargs.items() if k not in ("rng", "generator")}}, stream)

                def restore(self, path, optimizer=None, *, contract_hash=None, **kwargs):
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if saved["kwargs"].get("contract_hash") != contract_hash:
                        raise AssertionError("fixture checkpoint contract differs")
                    self.load_snapshot(saved["snapshot"])
                    events.append(("disk_restore", str(path), self.config["seed"]))
                    return saved["kwargs"].get("step", 0)

                def batch(self, rows, **kwargs):
                    return {"records": rows, "mask": torch.ones(len(rows), 2, dtype=torch.bool)}

                def logprobs(self, batch, **kwargs):
                    value = sum(parameter.sum() for parameter in self.parameters())
                    per_token = value.repeat(len(batch["records"]), 2)
                    return per_token, per_token.sum(-1), torch.full((len(batch["records"]),), 2)

                def generate(self, record, **kwargs):
                    split = record["split"]
                    events.append(("generation", split, self.config["seed"]))
                    correct = development_qualified or record["task"] == "copy" or split.startswith("eval_")
                    return {"sample_id": record["id"], "question": record["question"], "answer": record["answer"],
                            "text": record["answer"], "prompt_token_ids": [1, 2],
                            "response_token_ids": [3, 4], "termination_reason": "eos", "elapsed_seconds": 0.001,
                            "behavior_token_logprobs": [-0.5, -0.5],
                            "reward": {"correct": correct, "format_valid": True, "total": float(correct) + 0.1}}

                def generate_many(self, records, **kwargs):
                    return [self.generate(record, **kwargs) for record in records]

            def update(model, steps, path, *, kind, active=True):
                registration = read(output / "preregistration.json")
                if registration["arms"] != list(project.ARMS):
                    raise AssertionError("matrix was not registered before updates")
                if (output / "final_opening.json").exists():
                    raise AssertionError("training occurred after final panels opened")
                before = project.digest_state(model.snapshot())
                if active:
                    with torch.no_grad():
                        model.model.adapter.b.add_(0.01 * steps)
                after = project.digest_state(model.snapshot())
                events.append(("training", kind, model.config["seed"]))
                model.save(path, step=steps, contract_hash=project.sha(registration))
                return {"optimizer_steps": steps if active else 0, "effective_update_tokens": 2*steps if active else 0,
                        "elapsed_seconds": 0.001, "initial_weight_hash": before, "final_weight_hash": after,
                        "logs": [{"fixture_only": True, "step": 1}], "policy_gradient_proved": active,
                        "collections": [], "generated_tokens": 2, "generated_samples": 1}

            def sft(model, records, steps, lr, seed, size, path, contract):
                return update(model, steps, path, kind="sft")

            def dpo(model, reference, records, config, beta, seed, path, contract):
                return update(model, config["dpo_steps"], path, kind="dpo")

            def rl(model, reference, records, config, seed, path, contract, **kwargs):
                return update(model, config["rl_rounds"], path, kind="rl", active=rl_signal)

            patches.enter_context(mock.patch.object(project, "LocalAdapterLM", FakeAdapter))
            patches.enter_context(mock.patch.object(project, "train_sft", side_effect=sft))
            patches.enter_context(mock.patch.object(project, "train_dpo", side_effect=dpo))
            patches.enter_context(mock.patch.object(project, "train_rl", side_effect=rl))
            for name in ("check_sft", "check_dpo", "check_lora_frozen", "check_verifier_boundaries", "check_exact_resume"):
                patches.enter_context(mock.patch.object(adapter_checks, name, return_value={"passed": True, "fixture_only": True}))
            patches.enter_context(mock.patch.object(project, "environment_receipt", return_value={"fixture_only": True}))
            patches.enter_context(mock.patch.object(project, "resource_probe", return_value={"status": "passed", "state_restored": True, "frozen_base_unchanged": True, "fixture_only": True}))
            patches.enter_context(mock.patch.object(project, "tokenization_manifest", return_value={"fixture_only": True}))
            patches.enter_context(mock.patch.object(project, "token_alignment", return_value=[{"fixture_only": True}]))
            patches.enter_context(mock.patch.object(project, "sft_probability_receipt", return_value={"fixture_only": True, "loss": 0.1}))
            config = {"sft_steps": 1, "dpo_steps": 1, "rl_rounds": 1, "bootstrap_replicates": 10}
            yield output, model_dir, events, config

    def test_stages_refuse_missing_failed_dependencies_and_record_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stages.jsonl"
            stages = project.Stages(path)
            with self.assertRaisesRegex(AssertionError, "dependency"):
                with stages.stage("01", depends=("00",)):
                    pass
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with stages.stage("00"):
                    raise RuntimeError("intentional fixture failure")
            with self.assertRaisesRegex(AssertionError, "dependency"):
                with stages.stage("01", depends=("00",)):
                    pass
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["status"] for row in rows], ["running", "failed"])
            self.assertEqual(rows[-1]["failure_reason"], "intentional fixture failure")

    def test_configs_require_three_distinct_training_seeds_and_real_ablations(self):
        for value in ({"seeds": [17, 23]}, {"seeds": [17, 17, 41]}, {"seeds": [True, 23, 41]},
                      {"unexpected": 1}, {"batch_size": 3}, {"group_size_ablation": 4},
                      {"beta_ablation": 0.2}, {"format_reward": 0}, {"temperature": float("nan")}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                project.normalize_config(value)
        config = project.normalize_config({})
        self.assertEqual(config["seeds"], [17, 23, 41])
        self.assertEqual(len(project.ARMS), 9)
        self.assertNotEqual(config["beta"], config["beta_ablation"])
        self.assertNotEqual(config["group_size"], config["group_size_ablation"])

    def test_development_failure_reports_all_seeds_without_final(self):
        with self.offline_run(development_qualified=False) as (output, model_dir, events, config):
            result = project.run(config, model_dir, "cpu", output)
            self.assertEqual(result["status"], "development_unqualified")
            self.assertEqual(result["executed_training_seeds"], [17, 23, 41])
            self.assertIsNone(result["final_metrics"])
            self.assertFalse((output / "final_opening.json").exists())
            self.assertFalse(any(event[0] == "generation" and event[1].startswith("eval_") for event in events))

    def test_zero_advantage_variant_fails_stage_and_keeps_final_sealed(self):
        with self.offline_run(rl_signal=False) as (output, model_dir, events, config):
            with self.assertRaisesRegex(AssertionError, "policy-gradient"):
                project.run(config, model_dir, "cpu", output)
            self.assertFalse((output / "final_opening.json").exists())
            rows = [json.loads(line) for line in (output / "stages.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[-1]["status"], "failed")
            self.assertTrue((output / "seed_17/rlvr_train.json").is_file())

    def test_matrix_trains_before_final_and_zero_update_arms_have_honest_receipts(self):
        with self.offline_run() as (output, model_dir, events, config):
            result = project.run(config, model_dir, "cpu", output)
            self.assertEqual(result["executed_training_seeds"], [17, 23, 41])
            first_final = next(index for index, event in enumerate(events) if event[0] == "generation" and event[1].startswith("eval_"))
            training = [event for event in events[:first_final] if event[0] == "training"]
            self.assertEqual(len(training), 3 * 7)
            self.assertFalse(any(event[0] == "training" for event in events[first_final:]))
            self.assertEqual(sum(event[0] == "disk_restore" for event in events[:first_final]), 3 * 9)
            report = read(output / "report.json")
            self.assertEqual(len(report["runs"]), 3 * 9)
            for receipt in read(output / "training_receipts.json"):
                if receipt["arm"] in ("base", "sampling_only"):
                    self.assertEqual(receipt["optimizer_steps"], 0)
                    self.assertEqual(receipt["initial_weight_hash"], receipt["final_weight_hash"])
                if receipt["arm"] == "sampling_only":
                    self.assertEqual(receipt["source_training_run_id"], f"{receipt['training_seed']}:no_update")
                    self.assertEqual(receipt["source_checkpoint_id"], receipt["checkpoint_id"])
            registration = read(output / "preregistration.json")
            self.assertFalse(registration["final_used_for_selection"])
            self.assertEqual(registration["arms"], list(project.ARMS))

    def test_corrupted_checkpoint_reload_blocks_final_opening(self):
        with self.offline_run() as (output, model_dir, events, config):
            original = project.train_rl
            def corrupt(*args, **kwargs):
                result = original(*args, **kwargs)
                path = args[5]
                if path.stem == "rlvr":
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    first = next(iter(saved["snapshot"]))
                    saved["snapshot"][first].add_(1)
                    torch.save(saved, path)
                return result
            with mock.patch.object(project, "train_rl", side_effect=corrupt):
                with self.assertRaisesRegex(AssertionError, "saved checkpoint differs"):
                    project.run(config, model_dir, "cpu", output)
            self.assertFalse((output / "final_opening.json").exists())
            rows = [json.loads(line) for line in (output / "stages.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[-1]["stage"], "checkpoint_reload")
            self.assertEqual(rows[-1]["status"], "failed")

    def test_sampling_selector_is_text_only_majority_and_earliest_tie(self):
        class ForbiddenOracle(dict):
            def __getitem__(self, key):
                raise AssertionError("selection read oracle/reward")
            def get(self, key, default=None):
                raise AssertionError("selection read oracle/reward")
        def sample(text, ending="eos"):
            return {"text": text, "termination_reason": ending, "response_token_ids": [1], "reward": ForbiddenOracle()}
        choose = project.select_sampling_answer
        rows = [sample("7"), sample("4"), sample("4")]
        self.assertIs(choose(rows), rows[1])
        tied = [sample("7"), sample("4")]
        self.assertIs(choose(tied), tied[0])
        invalid = [sample("04"), sample("4", "new_token_budget"), sample(""), sample("-3")]
        self.assertIs(choose(invalid), invalid[-1])
        all_invalid = [sample("04"), sample("x")]
        self.assertIs(choose(all_invalid), all_invalid[0])


if __name__ == "__main__":
    unittest.main()
