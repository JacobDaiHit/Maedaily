"""Independent numerical oracles for teaching code, including sign/mask bugs."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


post = load("post_train/experiments/posttraining_math_lab.py", "post_math")
trajectory = load("agentic_rl/experiments/trajectory_accounting.py", "trajectory_math")
embedding = load("transformer/experiments/embedding_math_lab.py", "embedding_math")
rl = load("RL/experiments/tabular_mdp.py", "rl_math")


class MathematicalOracles(unittest.TestCase):
    def test_completion_logprob_shifts_once_and_masks_prompt_padding(self):
        probabilities = [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.6, 0.3, 0.1], [0.8, 0.1, 0.1], [0.4, 0.4, 0.2]]
        logits = [[math.log(p) for p in row] for row in probabilities]
        tokens = [0, 2, 1, 0, 2]
        mask = [0, 1, 1, 0]
        expected = math.log(0.7 * 0.6)
        self.assertAlmostEqual(post.completion_logprob(logits, tokens, mask), expected, places=12)
        for index in (0, 3, 4):
            logits[index] = [1000, -1000, 500]
        self.assertAlmostEqual(post.completion_logprob(logits, tokens, mask), expected, places=12)
        for invalid in ([0, 0, 0, 0], [1, 0, 1], [1, 2, 0, 0]):
            with self.subTest(mask=invalid), self.assertRaises(ValueError):
                post.completion_logprob(logits, tokens, invalid)

    def test_dpo_four_probabilities_and_derivative_sign(self):
        chosen, rejected, ref_chosen, ref_rejected, beta = -1.7, -2.4, -2.2, -2.1, 0.35
        margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
        expected = -math.log(1.0 / (1.0 + math.exp(-margin)))
        actual = post.dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta)
        self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(post.dpo_loss(ref_chosen, ref_rejected, ref_chosen, ref_rejected, beta), math.log(2), places=12)
        step = 1e-5
        derivative = (post.dpo_loss(chosen + step, rejected, ref_chosen, ref_rejected, beta) - post.dpo_loss(chosen - step, rejected, ref_chosen, ref_rejected, beta)) / (2 * step)
        self.assertAlmostEqual(derivative, -beta / (1.0 + math.exp(margin)), places=9)
        self.assertLess(post.dpo_loss(chosen + 0.1, rejected, ref_chosen, ref_rejected, beta), actual)
        self.assertGreater(post.dpo_loss(chosen, rejected + 0.1, ref_chosen, ref_rejected, beta), actual)

    def test_group_normalization_clipping_and_pass_at_k(self):
        rewards = [-3, 1, 2, 6]
        advantages = post.group_advantages(rewards)
        self.assertAlmostEqual(sum(advantages), 0, places=12)
        self.assertAlmostEqual(sum(a * a for a in advantages) / len(advantages), 1, places=7)
        self.assertEqual(post.group_advantages([5, 5, 5]), [0, 0, 0])
        self.assertAlmostEqual(post.ppo_surrogate(1.4, 2), 2.4)
        self.assertAlmostEqual(post.ppo_surrogate(1.4, -2), -2.8)
        self.assertAlmostEqual(post.ppo_surrogate(0.6, -2), -1.6)
        self.assertAlmostEqual(post.pass_at_k(8, 3, 2), 1 - (5 * 4) / (8 * 7))
        for args in ((4, 5, 1), (4, 1, 0), (4, 1, 5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                post.pass_at_k(*args)

    def test_trajectory_mask_and_termination_change_bootstrap(self):
        self.assertAlmostEqual(trajectory.masked_score_loss([-0.2, -900, -0.8], [1, 0, 1]), 0.5)
        self.assertEqual(trajectory.discounted_returns([2, -1, 4], 0.5), [2.5, 1, 4])
        self.assertAlmostEqual(trajectory.td_target(-0.2, 0.9, 2, True), -0.2)
        self.assertAlmostEqual(trajectory.td_target(-0.2, 0.9, 2, False), 1.6)
        with self.assertRaises(ValueError):
            trajectory.masked_score_loss([-1, -2], [0, 0])

    def test_bellman_solver_against_closed_form_at_several_discounts(self):
        for gamma in (0, 0.2, 0.6, 0.9, 0.99):
            with self.subTest(gamma=gamma):
                values, q, _, residual = rl.value_iteration(gamma)
                self.assertAlmostEqual(values["start"], max(1, -0.2 + 2 * gamma), places=12)
                self.assertAlmostEqual(values["work"], 2, places=12)
                self.assertAlmostEqual(q["work", "wait"], -0.1 + 2 * gamma, places=12)
                self.assertLess(residual, 1e-12)
        self.assertEqual(rl.greedy_policy(rl.value_iteration(0.2)[1])["start"], "quit")
        self.assertEqual(rl.greedy_policy(rl.value_iteration(0.9)[1])["start"], "invest")

    def test_q_learning_bootstraps_external_truncations(self):
        q, visits, _, truncations = rl.q_learning(gamma=0.9, seed=19, episodes=20000, epsilon=0.3, max_steps=2, record_every=1000)
        self.assertGreater(truncations, 0)
        self.assertGreater(min(visits.values()), 0)
        expected = {("start", "quit"): 1, ("start", "invest"): 1.6, ("work", "finish"): 2, ("work", "wait"): 1.7}
        for pair, value in expected.items():
            self.assertAlmostEqual(q[pair], value, places=6)

    def test_sgns_gradients_against_independent_finite_differences(self):
        rows = [[0.3, -0.7], [1.2, 0.4], [-0.2, 0.9]]

        def oracle(values):
            pos = sum(a * b for a, b in zip(values[0], values[1]))
            neg = sum(a * b for a, b in zip(values[0], values[2]))
            return -math.log(1 / (1 + math.exp(-pos))) - math.log(1 / (1 + math.exp(neg)))

        self.assertAlmostEqual(embedding.sgns_loss(rows), oracle(rows), places=12)
        gradient = embedding.sgns_gradients(rows)
        step = 1e-6
        for i in range(3):
            for j in range(2):
                plus, minus = [row[:] for row in rows], [row[:] for row in rows]
                plus[i][j] += step
                minus[i][j] -= step
                expected = (oracle(plus) - oracle(minus)) / (2 * step)
                self.assertAlmostEqual(gradient[i][j], expected, places=8)

    def test_contrastive_temperature_gradient_and_masked_pooling(self):
        similarities, temperature = [0.2, 0.7, -0.1], 0.3
        loss, probabilities, gradients = embedding.contrastive_row(similarities, temperature)
        weights = [math.exp(value / temperature) for value in similarities]
        expected_probability = weights[0] / sum(weights)
        self.assertAlmostEqual(loss, -math.log(expected_probability), places=12)
        self.assertAlmostEqual(sum(probabilities), 1, places=12)
        for index in range(len(similarities)):
            plus, minus = similarities[:], similarities[:]
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numeric = (embedding.contrastive_row(plus, temperature)[0] - embedding.contrastive_row(minus, temperature)[0]) / 2e-6
            self.assertAlmostEqual(gradients[index], numeric, places=8)
        self.assertEqual(embedding.masked_mean([[1, 3], [900, 900], [5, 7]], [1, 0, 1]), [3, 5])
        with self.assertRaises(ValueError):
            embedding.masked_mean([[1, 3]], [0])


if __name__ == "__main__":
    unittest.main()
