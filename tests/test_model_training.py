"""Meaningful model-graph, probability-boundary and checkpoint regression checks."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]
if TORCH_AVAILABLE:
    import torch
    import sys
    spec = importlib.util.spec_from_file_location("model_training_lab", ROOT / "post_train/experiments/model_training_lab.py")
    lab = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lab
    spec.loader.exec_module(lab)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch model tests require the optional CPU PyTorch runtime")
class ModelTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.tokenizer = lab.CharacterTokenizer()
        self.model = lab.TinyTextLM(len(self.tokenizer.pieces))
        self.records = [{"id": "one", "question": "2+3=", "answer": "5"},
                        {"id": "two", "question": "1-4=", "answer": "-3"}]
        self.batch = lab.build_batch(self.records, self.tokenizer)

    def test_answer_and_eos_gradients_match_independent_cross_entropy(self):
        result = lab.sft_checks(self.model, self.batch)
        self.assertTrue(result["passed"])
        self.assertEqual(result["per_sample_tokens"], [2, 3])
        self.assertEqual(result["unsupervised_logits_gradient_max"], 0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002)
        updated = lab.checked_step(lab.sft_loss(self.model, self.batch), self.model, optimizer)
        self.assertGreater(updated["gradient_norm_before_clip"], 0)
        self.assertGreater(updated["parameter_delta_norm"], 0)
        self.assertNotEqual(updated["weight_before"], updated["weight_after"])

    def test_label_shift_and_prompt_mask_fail_validation(self):
        shifted = copy.deepcopy(self.batch)
        shifted["labels"] = torch.roll(shifted["labels"], -1, 1)
        with self.assertRaisesRegex(ValueError, "unshifted"):
            lab.validate_batch(shifted)
        prompt_supervised = copy.deepcopy(self.batch)
        prompt_supervised["loss_mask"][0, 1] = True
        prompt_supervised["labels"][0, 1] = prompt_supervised["input_ids"][0, 1]
        with self.assertRaisesRegex(ValueError, "prompt/padding"):
            lab.validate_batch(prompt_supervised)
        eos_removed = copy.deepcopy(self.batch)
        end = int(eos_removed["attention_mask"][0].sum()) - 1
        eos_removed["loss_mask"][0, end] = False
        eos_removed["labels"][0, end] = lab.IGNORE
        with self.assertRaisesRegex(ValueError, "omits answer/EOS"):
            lab.validate_batch(eos_removed)

    def test_attention_is_causal_and_padding_does_not_change_real_logits(self):
        ids = self.batch["input_ids"]
        altered = ids.clone()
        position = self.batch["prefix_lengths"][0]
        altered[0, position] = self.tokenizer.encode("7")[0]
        with torch.no_grad():
            before = self.model(ids)
            after = self.model(altered)
            valid_end = int(self.batch["attention_mask"][0].sum())
            single = self.model(ids[0:1, :valid_end])
        self.assertTrue(torch.allclose(before[0, :position], after[0, :position], atol=1e-6))
        self.assertTrue(torch.allclose(before[0, :valid_end], single[0], atol=1e-6))

    def test_dpo_four_probabilities_gradient_and_frozen_reference(self):
        reference = lab.frozen_copy(self.model)
        before = lab.weight_hash(reference)
        wrong = lab.build_batch([{**row, "answer": str(int(row["answer"]) + 1)} for row in self.records], self.tokenizer)
        checks = lab.dpo_checks(self.model, reference, self.batch, wrong, 0.2)
        self.assertTrue(checks["passed"])
        self.assertAlmostEqual(checks["initial_loss"], __import__("math").log(2), places=5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.002)
        loss, terms = lab.dpo_loss(self.model, reference, self.batch, wrong, 0.2)
        self.assertTrue(terms["policy_chosen"].requires_grad)
        self.assertFalse(terms["reference_chosen"].requires_grad)
        lab.checked_step(loss, self.model, optimizer)
        self.assertEqual(lab.weight_hash(reference), before)
        self.assertTrue(all(parameter.grad is None for parameter in reference.parameters()))
        with self.assertRaisesRegex(AssertionError, "independent frozen"):
            lab.dpo_loss(self.model, self.model, self.batch, wrong, 0.2)
        with torch.no_grad():
            detached_loss, _ = lab.dpo_loss(self.model, reference, self.batch, wrong, 0.2)
        with self.assertRaisesRegex(AssertionError, "graph is detached"):
            lab.checked_step(detached_loss, self.model, optimizer)

    def test_real_sampling_replays_at_nonunit_temperature_and_mismatch_fails(self):
        old = lab.frozen_copy(self.model)
        rows, receipt = lab.collect_rollouts(old, self.tokenizer, self.records, groups=2, group_size=4,
                                             max_new_tokens=4, temperature=0.7, seed=17, round_number=0)
        self.assertEqual(receipt["sampling_optimizer_steps"], 0)
        self.assertFalse(receipt["sampling_changed_weights"])
        self.assertLess(lab.replay_check(old, self.tokenizer, rows, 0.7), 2e-5)
        with self.assertRaisesRegex(AssertionError, "temperature mismatch"):
            lab.replay_check(old, self.tokenizer, rows, 1.0)
        corrupted = copy.deepcopy(rows)
        corrupted[0]["behavior_token_logprobs"][0] += 0.2
        with self.assertRaisesRegex(AssertionError, "probability replay mismatch"):
            lab.replay_check(old, self.tokenizer, corrupted, 0.7)

    def test_greedy_behavior_is_declared_deterministic(self):
        row = lab.generate(self.model, self.tokenizer, self.records[0], 4, sample=False)
        self.assertEqual(row["distribution"]["kind"], "greedy-deterministic")
        self.assertFalse(row["distribution"]["do_sample"])
        self.assertEqual(row["behavior_token_logprobs"], [0.0] * len(row["response_token_ids"]))
        self.assertTrue(all(value < 0 for value in row["model_token_logprobs"]))

    def test_verifier_and_same_reward_groups(self):
        self.assertTrue(lab.verifier_checks()["passed"])
        for rewards in ([0, 0, 0], [1.1, 1.1, 1.1]):
            values, stats = lab.advantages(rewards)
            self.assertEqual(values, [0, 0, 0])
            self.assertFalse(stats["active"])
        values, stats = lab.advantages([0, 1.1])
        self.assertTrue(stats["active"])
        self.assertLess(values[0], 0)
        self.assertGreater(values[1], 0)
        self.assertAlmostEqual(sum(values), 0)

    def test_clip_branches_and_zero_advantage_have_expected_derivatives(self):
        ratios = torch.tensor([[1.5], [0.5], [1.5], [0.5]], requires_grad=True)
        current = ratios.log()
        old = torch.zeros_like(current).detach()
        mask = torch.ones_like(current, dtype=torch.bool)
        advantages = torch.tensor([1., -1., -1., 1.])
        loss, actual_ratios = lab.clipped_policy_loss(current, old, mask, advantages, clip=0.2)
        loss.backward()
        self.assertTrue(torch.allclose(actual_ratios, ratios))
        self.assertAlmostEqual(ratios.grad[0].item(), 0)
        self.assertAlmostEqual(ratios.grad[1].item(), 0)
        self.assertAlmostEqual(ratios.grad[2].item(), 0.25)
        self.assertAlmostEqual(ratios.grad[3].item(), -0.25)
        current = torch.zeros(2, 2, requires_grad=True)
        zero, _ = lab.clipped_policy_loss(current, torch.zeros_like(current).detach(),
                                          torch.ones_like(current, dtype=torch.bool), torch.zeros(2))
        zero.backward()
        self.assertEqual(current.grad.abs().sum().item(), 0)
        with self.assertRaisesRegex(AssertionError, "must be detached"):
            lab.clipped_policy_loss(current, current, torch.ones_like(current, dtype=torch.bool), torch.ones(2))

    def test_saved_checkpoint_is_immutable_and_reload_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            lab.save_checkpoint(path, self.model, self.tokenizer, "test")
            first_hash = lab.file_hash(path)
            self.assertTrue(lab.reload_check(path, self.model, self.tokenizer)["passed"])
            with self.assertRaises(FileExistsError):
                lab.save_checkpoint(path, self.model, self.tokenizer, "overwrite-attempt")
            self.assertEqual(lab.file_hash(path), first_hash)

    def test_canonical_operand_families_do_not_cross_splits(self):
        data = lab.fixed_data()
        by_question = {record["question"]: record["split"] for records in data.values() for record in records}
        for left in range(6):
            for right in range(6):
                self.assertEqual(by_question[f"{left}+{right}="], by_question[f"{right}+{left}="])
                self.assertEqual(by_question[f"{left}+{right}="], by_question[f"{left}-{right}="])
        family_sets = [{record["family_id"] for record in records} for records in data.values()]
        for left, first in enumerate(family_sets):
            for second in family_sets[left + 1:]:
                self.assertFalse(first & second)


if __name__ == "__main__":
    unittest.main()
