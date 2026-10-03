"""Offline tiny-model proofs for the pretrained LoRA adapter implementation."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from torch.nn import functional as F
    sys.path.insert(0, str(ROOT / "post_train/experiments"))
    import pretrained_adapter as implementation
    import adapter_checks as checks

    class TinyTokenizer:
        def __init__(self):
            self.pieces = ["<PAD>", "<BOS>", "<EOS>"] + list("\n +-0123456789:=QcopyA") + ["<im_start>"]
            self.ids = {piece: index for index, piece in enumerate(self.pieces)}
            self.eos_token_id, self.pad_token_id = 2, 0
            self.all_special_ids = [0, 1, 2, self.ids["<im_start>"]]
        def encode(self, text, add_special_tokens=False):
            return [self.ids[character] for character in text]
        def decode(self, tokens, skip_special_tokens=True):
            return "".join(self.pieces[token] for token in tokens if not skip_special_tokens or token not in self.all_special_ids)
        def __len__(self):
            return len(self.pieces)

    class TinyCausalModel(nn.Module):
        def __init__(self, vocabulary):
            super().__init__()
            self.config = SimpleNamespace(vocab_size=vocabulary)
            self.embedding = nn.Embedding(vocabulary, 8)
            self.positions = nn.Embedding(64, 8)
            self.q_proj, self.k_proj, self.v_proj = [nn.Linear(8, 8) for _ in range(3)]
            self.output = nn.Linear(8, vocabulary)
            self.requires_grad_(False)
            self.q_proj = implementation.LoRALinear(self.q_proj, 2, 4)
            self.v_proj = implementation.LoRALinear(self.v_proj, 2, 4)
        def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None,
                    logits_to_keep=0, **kwargs):
            if past_key_values is not None:
                input_ids = torch.cat([past_key_values, input_ids], dim=1)
            length = input_ids.shape[1]
            hidden = self.embedding(input_ids) + self.positions(torch.arange(length))[None]
            q, k, v = self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)
            scores = q @ k.transpose(-2, -1) / (8 ** 0.5)
            allowed = torch.ones(length, length, dtype=torch.bool).tril()[None]
            if attention_mask is not None:
                allowed = allowed & attention_mask[:, None, :].bool()
            values = scores.masked_fill(~allowed, float("-inf")).softmax(-1) @ v
            values = hidden + values
            if isinstance(logits_to_keep, torch.Tensor):
                values = values[:, logits_to_keep]
            elif logits_to_keep:
                values = values[:, -logits_to_keep:]
            logits = self.output(values)
            return SimpleNamespace(logits=logits,
                                   past_key_values=input_ids.detach().clone() if use_cache else None)
        def get_input_embeddings(self):
            return self.embedding

    def fixture():
        torch.manual_seed(17)
        adapter = implementation.LocalAdapterLM.__new__(implementation.LocalAdapterLM)
        adapter.device = torch.device("cpu")
        adapter.tokenizer = TinyTokenizer()
        adapter.model = TinyCausalModel(len(adapter.tokenizer)).eval()
        adapter.eos, adapter.pad = 2, 0
        adapter.adapter_names = {name for name, parameter in adapter.model.named_parameters() if parameter.requires_grad}
        adapter.config = {"local_path": "offline-tiny-unit-model", "device": "cpu", "dtype": "float32",
                          "rank": 2, "alpha": 4, "seed": 17, "eos": 2, "pad": 0,
                          "system": implementation.SYSTEM, "targets": ["q_proj", "v_proj"]}
        adapter.prefix = lambda record: [1] + adapter.tokenizer.encode("Q:" + record["question"] + "\nA:")
        return adapter


@unittest.skipUnless(TORCH_AVAILABLE, "Offline LoRA tests require the optional PyTorch runtime")
class PretrainedAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.adapter = fixture()
        self.records = [{"id": "sum", "question": "2+3=", "answer": "5"},
                        {"id": "negative", "question": "1-4=", "answer": "-3"}]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_lora_forward_and_parameter_gradients_match_handwritten_matrix_math(self):
        torch.manual_seed(23)
        layer = implementation.LoRALinear(nn.Linear(4, 3), 2, 6)
        with torch.no_grad(): layer.b.normal_(0, 0.1)
        x, target = torch.randn(5, 4), torch.randn(5, 3)
        actual = layer(x)
        expected = x @ layer.base.weight.T + layer.base.bias + 3 * (x @ layer.a.T) @ layer.b.T
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        loss = (actual - target).square().mean()
        grad_a, grad_b = torch.autograd.grad(loss, [layer.a, layer.b])
        derivative = 2 * (expected.detach() - target) / actual.numel()
        self.assertTrue(torch.allclose(grad_b, 3 * derivative.T @ (x @ layer.a.detach().T), atol=1e-6))
        self.assertTrue(torch.allclose(grad_a, 3 * (derivative @ layer.b.detach()).T @ x, atol=1e-6))
        self.assertFalse(layer.base.weight.requires_grad)
        for alpha in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError): implementation.LoRALinear(nn.Linear(4, 3), 2, alpha)

    def test_real_adapter_update_keeps_full_base_frozen(self):
        before = checks.check_lora_frozen(self.adapter, full_base_hash=True)
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=0.001)
        receipt = self.adapter.checked_step(self.adapter.sft_loss(self.adapter.batch(self.records)), optimizer)
        after = checks.check_lora_frozen(self.adapter, full_base_hash=True)
        self.assertEqual(before["base_sha256"], after["base_sha256"])
        self.assertGreater(receipt["gradient_norm"], 0)
        self.assertGreater(receipt["parameter_delta_norm"], 0)
        self.assertNotEqual(receipt["weight_before"], receipt["weight_after"])

    def test_compact_sft_matches_full_forward_hand_ce_and_adapter_gradients(self):
        result = checks.check_sft(self.adapter, self.records, atol=1e-6)
        self.assertTrue(result["passed"])
        self.assertEqual(result["batch"]["per_sample_tokens"], [2, 3])
        self.assertLessEqual(result["adapter_gradient_max_error"], 1e-6)
        self.assertEqual(result["ignored_logits_gradient_max"], 0)
        batch = self.adapter.batch(self.records)
        tampered = copy.deepcopy(batch)
        tampered["mask"][0, 1] = True
        with self.assertRaisesRegex(AssertionError, "prompt/padding"):
            checks.validate_batch(self.adapter, tampered, self.records)
        tampered = copy.deepcopy(batch)
        last = int(tampered["attention"][0].sum()) - 1
        tampered["mask"][0, last] = False
        with self.assertRaisesRegex(AssertionError, "omits response/EOS"):
            checks.validate_batch(self.adapter, tampered, self.records)
        tampered = copy.deepcopy(batch)
        tampered["input_ids"][0, batch["prefixes"][0].__len__()] = self.adapter.tokenizer.ids["4"]
        with self.assertRaisesRegex(AssertionError, "unshifted actual answer"):
            checks.validate_batch(self.adapter, tampered, self.records)

    def test_functional_reference_is_detached_immutable_and_restores_policy(self):
        before = implementation.digest_state(self.adapter.snapshot())
        result = checks.check_reference(self.adapter, self.records)
        self.assertTrue(result["passed"])
        self.assertFalse(result["reference_requires_grad"])
        self.assertEqual(before, implementation.digest_state(self.adapter.snapshot()))
        snapshot = self.adapter.snapshot()
        next(iter(snapshot.values())).requires_grad_(True)
        batch = self.adapter.batch(self.records)
        with self.assertRaisesRegex(ValueError, "detached"):
            self.adapter.forward(batch["input_ids"], batch["attention"], snapshot=snapshot)

    def test_dpo_four_probabilities_analytic_gradients_and_swap_direction(self):
        rejected = [{**row, "answer": str(int(row["answer"]) + 1)} for row in self.records]
        result = checks.check_dpo(self.adapter, self.records, rejected, beta=0.2, atol=1e-6)
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["initial_loss"], math_log_two := __import__("math").log(2), places=6)
        def gradient(chosen_rows, rejected_rows):
            _, chosen, _ = self.adapter.logprobs(self.adapter.batch(chosen_rows))
            _, wrong, _ = self.adapter.logprobs(self.adapter.batch(rejected_rows))
            # At policy=reference, dL/d(policy chosen-rejected)=-beta/2.
            scalar = -0.1 * (chosen - wrong).mean()
            return torch.autograd.grad(scalar, self.adapter.parameters(), allow_unused=True)
        first, reversed_ = gradient(self.records, rejected), gradient(rejected, self.records)
        for a, b in zip(first, reversed_):
            if a is not None: self.assertTrue(torch.allclose(a, -b, atol=1e-6))

    def test_online_cached_probabilities_replay_and_detect_temperature_or_tamper(self):
        snapshot = self.adapter.snapshot()
        generator = torch.Generator().manual_seed(41)
        rollouts = [self.adapter.generate(row, max_new_tokens=3, temperature=0.7,
                                         sample=True, generator=generator, snapshot=snapshot) for row in self.records]
        result = checks.check_rollout_replay(self.adapter, self.records, rollouts, snapshot=snapshot, atol=1e-6)
        self.assertTrue(result["passed"])
        tampered = copy.deepcopy(rollouts)
        tampered[0]["behavior_token_logprobs"][0] += 0.2
        with self.assertRaisesRegex(AssertionError, "do not replay"):
            checks.check_rollout_replay(self.adapter, self.records, tampered, snapshot=snapshot)
        tampered = copy.deepcopy(rollouts)
        tampered[0]["reward"]["total"] = 100.0
        with self.assertRaisesRegex(AssertionError, "reward differs"):
            checks.check_rollout_replay(self.adapter, self.records, tampered, snapshot=snapshot)
        tampered = copy.deepcopy(rollouts)
        tampered[0]["text"] = "different generated text"
        with self.assertRaisesRegex(AssertionError, "text/termination"):
            checks.check_rollout_replay(self.adapter, self.records, tampered, snapshot=snapshot)
        tampered = copy.deepcopy(rollouts)
        tampered[0]["distribution"]["temperature"] = 1.0
        with self.assertRaisesRegex(AssertionError, "do not replay"):
            checks.check_rollout_replay(self.adapter, self.records, tampered, snapshot=snapshot)

    def _forced_generate(self, actions, *, sample=False):
        outputs = []
        for action in actions:
            logits = torch.full((1, 1, len(self.adapter.tokenizer)), -1000.0)
            logits[0, 0, action] = 0.0
            outputs.append(SimpleNamespace(logits=logits, past_key_values=None))
        with mock.patch.object(self.adapter, "forward", side_effect=outputs):
            return self.adapter.generate(self.records[0], max_new_tokens=len(actions),
                                         sample=sample, generator=torch.Generator().manual_seed(17))

    def test_single_generation_cannot_hide_nonterminal_special_actions(self):
        digit = self.adapter.tokenizer.ids["5"]
        for special in (self.adapter.pad, 1, self.adapter.tokenizer.ids["<im_start>"]):
            for sample in (False, True):
                with self.subTest(special=special, sample=sample):
                    actions = [special, digit, self.adapter.eos]
                    rollout = self._forced_generate(actions, sample=sample)
                    self.assertEqual(rollout["response_token_ids"], actions)
                    self.assertEqual(rollout["text"], self.adapter.tokenizer.decode(actions[:-1], skip_special_tokens=False))
                    self.assertFalse(rollout["reward"]["correct"])
                    self.assertFalse(rollout["reward"]["format_valid"])
                    result = checks.check_action_text(self.adapter, self.records[0], rollout)
                    self.assertEqual(result["nonterminal_special_actions"], [special])
        clean = self._forced_generate([digit, self.adapter.eos])
        self.assertEqual(clean["text"], "5")
        self.assertTrue(clean["reward"]["correct"])
        self.assertTrue(checks.check_action_text(self.adapter, self.records[0], clean)["passed"])

    def test_batch_generation_distinguishes_illegal_actions_from_post_eos_padding(self):
        digit = self.adapter.tokenizer.ids["5"]
        records = [self.records[0], {"id": "copy", "question": "copy5", "answer": "5"}]
        prefixes = [self.adapter.prefix(row) for row in records]
        width = max(map(len, prefixes))
        inputs = [[self.adapter.pad] * (width-len(prefix)) + prefix for prefix in prefixes]
        for special in (self.adapter.pad, 1, self.adapter.tokenizer.ids["<im_start>"]):
            actions = [[special, digit, self.adapter.eos, self.adapter.pad],
                       [digit, self.adapter.eos, self.adapter.pad, self.adapter.pad]]
            output = torch.tensor([prefix + response for prefix, response in zip(inputs, actions)])
            with mock.patch.object(self.adapter.model, "generate", create=True, return_value=output):
                rows = self.adapter.generate_many(records, max_new_tokens=4, batch_size=2)
            with self.subTest(special=special):
                self.assertEqual(rows[0]["response_token_ids"], actions[0][:3])
                self.assertEqual(rows[0]["text"], self.adapter.tokenizer.decode(actions[0][:2], skip_special_tokens=False))
                self.assertFalse(rows[0]["reward"]["correct"])
                self.assertFalse(rows[0]["reward"]["format_valid"])
                self.assertEqual(rows[1]["response_token_ids"], [digit, self.adapter.eos])
                self.assertEqual(rows[1]["text"], "5")
                self.assertTrue(rows[1]["reward"]["correct"])
                for record, row in zip(records, rows):
                    self.assertTrue(checks.check_action_text(self.adapter, record, row)["passed"])

    def test_action_text_oracle_rejects_hidden_specials_and_preserves_truncated_actions(self):
        from model_training_lab import verify_answer
        digit, special = self.adapter.tokenizer.ids["5"], self.adapter.tokenizer.ids["<im_start>"]
        forged = {"response_token_ids": [special, digit, self.adapter.eos], "text": "5",
                  "termination_reason": "eos", "reward": verify_answer("5", "5", "eos")}
        with self.assertRaisesRegex(AssertionError, "hides or changes"):
            checks.check_action_text(self.adapter, self.records[0], forged)
        truncated = self._forced_generate([special, digit])
        self.assertEqual(truncated["termination_reason"], "new_token_budget")
        self.assertEqual(truncated["text"], "<im_start>5")
        self.assertFalse(truncated["reward"]["correct"])
        self.assertFalse(checks.check_action_text(self.adapter, self.records[0], truncated)["registered_eos_removed"])
        after_eos = {**forged, "response_token_ids": [digit, self.adapter.eos, special]}
        with self.assertRaisesRegex(AssertionError, "continue after"):
            checks.check_action_text(self.adapter, self.records[0], after_eos)

    def test_verifier_extremes_format_and_truncation_are_separate(self):
        result = checks.check_verifier_boundaries()
        self.assertTrue(result["passed"])
        statuses = {row["status"] for row in result["cases"]}
        self.assertTrue({"correct", "wrong", "empty", "malformed", "too_long", "truncated"} <= statuses)

    def test_exact_resume_restores_adapters_optimizer_all_rng_and_sampler(self):
        original = implementation.digest_state(self.adapter.snapshot())
        rng = torch.get_rng_state().clone()
        result = checks.check_exact_resume(self.adapter, self.records, Path(self.temp.name) / "resume", contract_hash="frozen-test-data")
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["comparisons"].values()))
        self.assertEqual(original, implementation.digest_state(self.adapter.snapshot()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue((Path(self.temp.name) / "resume/step_000001.pt").is_file())
        with self.assertRaises(FileExistsError):
            checks.check_exact_resume(self.adapter, self.records, Path(self.temp.name) / "resume", contract_hash="frozen-test-data")

    def test_nonfinite_loss_detached_graph_and_corrupt_snapshots_are_rejected(self):
        original = implementation.digest_state(self.adapter.snapshot())
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=0.001)
        batch = self.adapter.batch(self.records)
        for bad in (float("inf"), float("nan")):
            loss = self.adapter.sft_loss(batch) + bad
            with self.assertRaises((ValueError, AssertionError)):
                self.adapter.checked_step(loss, optimizer)
            self.assertEqual(original, implementation.digest_state(self.adapter.snapshot()))
        with self.assertRaises((ValueError, AssertionError)):
            self.adapter.checked_step(self.adapter.sft_loss(batch).detach(), optimizer)
        name = next(name for name in self.adapter.snapshot() if name.endswith(".a"))
        variants = []
        state = self.adapter.snapshot(); state[name] = state[name][:1]; variants.append(state)
        state = self.adapter.snapshot(); state[name] = state[name].double(); variants.append(state)
        state = self.adapter.snapshot(); state[name].flatten()[0] = float("nan"); variants.append(state)
        state = self.adapter.snapshot(); del state[name]; variants.append(state)
        for state in variants:
            with self.assertRaises(ValueError):
                self.adapter.load_snapshot(state)
            with self.assertRaises(ValueError):
                self.adapter.forward(batch["input_ids"], batch["attention"], snapshot=state)
            self.assertEqual(original, implementation.digest_state(self.adapter.snapshot()))

    def test_batch_temperature_checkpoint_contract_and_immutable_paths(self):
        batch = self.adapter.batch(self.records)
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): self.adapter.logprobs(batch, temperature=temperature)
        with self.assertRaises(ValueError): self.adapter.batch(self.records, answers=[[2]])
        with self.assertRaises(ValueError): self.adapter.batch(self.records[:1], answers=[[2, 4]])
        with self.assertRaises(ValueError): self.adapter.batch(self.records[:1], answers=[[-1]])
        path = Path(self.temp.name) / "immutable.pt"
        self.adapter.save(path, contract_hash="first")
        with self.assertRaises(FileExistsError): self.adapter.save(path, contract_hash="first")
        with self.assertRaisesRegex(ValueError, "contract differs"):
            self.adapter.restore(path, contract_hash="second")


if __name__ == "__main__":
    unittest.main()
