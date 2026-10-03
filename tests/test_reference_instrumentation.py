"""Real offline adapter probes: evidence counts and caller state restoration."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reference_instrumentation as instrumentation

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    fixture_spec = importlib.util.spec_from_file_location(
        "_instrumentation_adapter_fixture", ROOT / "tests/test_pretrained_adapter.py")
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)


@unittest.skipUnless(TORCH_AVAILABLE, "Optional PyTorch runtime needed for actual adapter probes")
class ReferenceInstrumentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.adapter = fixture_module.fixture()
        tokenizer = self.adapter.tokenizer
        tokenizer.get_vocab = lambda: dict(tokenizer.ids)
        tokenizer.convert_ids_to_tokens = lambda token: tokenizer.pieces[token]
        tokenizer.chat_template = "offline fixture: Q:{question}\\nA:"
        tokenizer.name_or_path = "offline-tiny-tokenizer"
        tokenizer.bos_token_id = 1
        self.records = [
            {"id": "short", "family_id": "sum", "question": "2+3=", "answer": "5"},
            {"id": "negative", "family_id": "sub", "question": "1-4=", "answer": "-3"},
            {"id": "long-copy", "family_id": "copy", "question": "copy 0012345", "answer": "12345"},
        ]

    def remember_state(self):
        return {
            "policy": self.adapter.snapshot(),
            "base": instrumentation._frozen_hash(self.adapter),
            "grads": [None if p.grad is None else p.grad.clone() for p in self.adapter.parameters()],
            "modes": [module.training for module in self.adapter.model.modules()],
            "python": random.getstate(), "cpu": torch.get_rng_state().clone(),
        }

    def assert_restored(self, before):
        for name, value in self.adapter.snapshot().items():
            self.assertTrue(torch.equal(value, before["policy"][name]), name)
        self.assertEqual(instrumentation._frozen_hash(self.adapter), before["base"])
        for parameter, old in zip(self.adapter.parameters(), before["grads"]):
            self.assertEqual(parameter.grad is None, old is None)
            if old is not None:
                self.assertTrue(torch.equal(parameter.grad, old))
        self.assertEqual([module.training for module in self.adapter.model.modules()], before["modes"])
        self.assertEqual(random.getstate(), before["python"])
        self.assertTrue(torch.equal(torch.get_rng_state(), before["cpu"]))

    def test_real_forward_update_generation_counts_and_exact_caller_restore(self):
        for index, parameter in enumerate(self.adapter.parameters()):
            parameter.grad = None if index % 2 else torch.full_like(parameter, 0.125)
        self.adapter.model.q_proj.train()
        batch = self.adapter.batch(self.records)
        old_generate = self.adapter.generate
        generated = []

        def consuming_generate(*args, **kwargs):
            random.random(); torch.rand(3)
            self.adapter.model.train()
            result = old_generate(*args, **kwargs)
            generated.append(result)
            return result

        before = self.remember_state()
        with mock.patch.object(self.adapter, "generate", side_effect=consuming_generate):
            receipt = instrumentation.resource_probe(self.adapter, self.records)
        self.assert_restored(before)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["input_tokens"], int(batch["attention"].sum()))
        self.assertEqual(receipt["target_tokens"], 2 + 3 + 6)
        self.assertEqual(receipt["generated_tokens"], len(generated[0]["response_token_ids"]))
        self.assertEqual(receipt["generation_prompt_tokens"], len(batch["prefixes"][0]))
        self.assertEqual(receipt["source_sample_ids"], [r["id"] for r in self.records])
        self.assertGreater(receipt["update_check"]["gradient_norm"], 0)
        self.assertGreater(receipt["update_check"]["parameter_delta_norm"], 0)
        self.assertNotEqual(receipt["update_check"]["weight_before"], receipt["update_check"]["weight_after"])
        self.assertTrue(receipt["state_restored"] and receipt["frozen_base_unchanged"])
        for key in ("forward_seconds", "backward_optimizer_seconds", "generation_seconds"):
            self.assertGreaterEqual(receipt[key], 0)
        self.assertIsNone(receipt["cuda_peak_allocated_bytes"])
        self.assertEqual(receipt["cpu_after"]["status"], "measured")
        self.assertGreater(receipt["cpu_after"]["peak_resident_bytes"], 0)
        self.assertIn("whole process lifetime", receipt["cpu_after"]["scope"])
        json.dumps(receipt, allow_nan=False)

    def test_generation_failure_after_actual_update_also_restores_state(self):
        before = self.remember_state()

        def fail_after_update(*args, **kwargs):
            self.assertTrue(any(not torch.equal(value, before["policy"][name])
                                for name, value in self.adapter.snapshot().items()))
            random.random(); torch.rand(2)
            self.adapter.model.train()
            raise RuntimeError("injected generation failure")

        with mock.patch.object(self.adapter, "generate", side_effect=fail_after_update):
            with self.assertRaisesRegex(RuntimeError, "injected generation"):
                instrumentation.resource_probe(self.adapter, self.records)
        self.assert_restored(before)

    def test_frozen_base_mutation_is_detected_instead_of_success_receipt(self):
        actual_step = self.adapter.checked_step

        def corrupt_base(loss, optimizer):
            result = actual_step(loss, optimizer)
            with torch.no_grad():
                self.adapter.model.embedding.weight[0, 0].add_(0.01)
            return result

        with mock.patch.object(self.adapter, "checked_step", side_effect=corrupt_base):
            with self.assertRaisesRegex(AssertionError, "changed frozen base"):
                instrumentation.resource_probe(self.adapter, self.records)

    def test_alignment_keeps_eos_and_equal_id_padding_distinct_and_shifts_once(self):
        self.adapter.pad = self.adapter.eos
        batch = self.adapter.batch(self.records)
        rows = instrumentation.token_alignment(self.adapter, self.records)
        width = batch["input_ids"].shape[1]
        self.assertEqual(len(rows), len(self.records) * width)
        self.assertEqual(sum(row["loss_mask"] for row in rows), 11)
        roles = {row["role"] for row in rows}
        self.assertEqual(roles, {"prompt", "answer", "EOS", "PAD"})
        for row in rows:
            source = self.records[row["batch_row"]]
            index, position = row["batch_row"], row["position"]
            self.assertEqual(row["source_sample_id"], source["id"])
            self.assertEqual(row["token_id"], int(batch["input_ids"][index, position]))
            self.assertEqual(row["readable_token"], self.adapter.tokenizer.pieces[row["token_id"]])
            self.assertEqual(row["target_prediction_position"], position - 1)
            if row["role"] in ("prompt", "PAD"):
                self.assertEqual((row["label"], row["loss_mask"]), (-100, 0))
            else:
                self.assertEqual((row["label"], row["loss_mask"]), (row["token_id"], 1))
            if row["role"] in ("EOS", "PAD"):
                self.assertEqual(row["token_id"], self.adapter.eos)
                self.assertEqual(row["attention_mask"], int(row["role"] == "EOS"))

    def test_alignment_rejects_wrong_source_identity(self):
        batch = self.adapter.batch(self.records)
        batch["sample_ids"][1] = "foreign-record"
        with mock.patch.object(self.adapter, "batch", return_value=batch):
            with self.assertRaisesRegex(ValueError, "identities differ"):
                instrumentation.token_alignment(self.adapter, self.records)

    def test_manifest_counts_real_tokens_and_binds_data_template_vocab_and_trainables(self):
        data = {"train": self.records, "dev": [self.records[1]]}
        manifest = instrumentation.tokenization_manifest(self.adapter, data)
        self.assertEqual(manifest, instrumentation.tokenization_manifest(self.adapter, copy.deepcopy(data)))
        self.assertEqual(manifest["splits"]["train"]["target_tokens"], 11)
        for row, source in zip(manifest["splits"]["train"]["records"], self.records):
            self.assertEqual(row["source_sample_id"], source["id"])
            self.assertEqual(row["answer_tokens_excluding_eos"], len(source["answer"]))
            self.assertEqual(row["eos_tokens"], 1)
            self.assertEqual(row["input_tokens"], row["prompt_tokens"] + row["target_tokens"])
            self.assertFalse(row["truncated"])
        self.assertEqual({row["name"] for row in manifest["trainable_parameters"]}, self.adapter.adapter_names)
        self.assertEqual(sum(row["parameters"] for row in manifest["trainable_parameters"]),
                         sum(parameter.numel() for parameter in self.adapter.parameters()))
        changed = copy.deepcopy(data); changed["train"][0]["answer"] = "4"
        changed_manifest = instrumentation.tokenization_manifest(self.adapter, changed)
        self.assertNotEqual(manifest["data_sha256"], changed_manifest["data_sha256"])
        self.assertNotEqual(manifest["stable_content_sha256"], changed_manifest["stable_content_sha256"])
        self.adapter.tokenizer.chat_template += " changed"
        template_changed = instrumentation.tokenization_manifest(self.adapter, data)
        self.assertNotEqual(manifest["tokenizer"]["chat_template_sha256"], template_changed["tokenizer"]["chat_template_sha256"])
        self.adapter.tokenizer.get_vocab = lambda: {**self.adapter.tokenizer.ids, "extra": 999}
        vocab_changed = instrumentation.tokenization_manifest(self.adapter, data)
        self.assertNotEqual(manifest["tokenizer"]["vocab_sha256"], vocab_changed["tokenizer"]["vocab_sha256"])
        json.dumps(manifest, allow_nan=False)

    def test_manifest_overlength_is_rejected_without_claiming_truncation(self):
        long_record = {"id": "overlength", "question": "copy " + "0" * 260, "answer": "0"}
        with self.assertRaisesRegex(ValueError, "no silent truncation"):
            instrumentation.tokenization_manifest(self.adapter, {"train": [long_record]})
        with self.assertRaisesRegex(ValueError, "empty tokenization"):
            instrumentation.tokenization_manifest(self.adapter, {"train": []})
        with self.assertRaises(ValueError):
            instrumentation.resource_probe(self.adapter, [])

    def test_environment_is_current_interpreter_and_validated_load_duration(self):
        receipt = instrumentation.environment_receipt(self.adapter, 0.125)
        self.assertEqual(receipt["python_executable"], sys.executable)
        self.assertEqual(receipt["versions"]["torch"], torch.__version__)
        self.assertEqual(receipt["device"], "cpu")
        self.assertEqual(receipt["parameter_dtypes"], ["torch.float32"])
        self.assertEqual(receipt["load_seconds"], 0.125)
        self.assertFalse(receipt["gpu"]["applicable"])
        self.assertEqual(set(receipt["versions"]), {"torch", "transformers", "huggingface_hub", "safetensors"})
        for value in (True, -1, float("nan"), float("inf"), "unknown"):
            with self.assertRaises(ValueError):
                instrumentation.environment_receipt(self.adapter, value)
        json.dumps(receipt, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
