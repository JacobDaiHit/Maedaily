"""CPU float64 attention oracles; skipped explicitly without PyTorch."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
lab = None
if torch is not None:
    spec = importlib.util.spec_from_file_location("attention_oracle_lab", ROOT / "transformer" / "experiments" / "attention_math_lab.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)


@unittest.skipIf(torch is None, "PyTorch CPU required for attention tests")
class AttentionOracles(unittest.TestCase):
    def test_scalar_attention_output_weights_and_gradients(self):
        q = torch.tensor([[[[1.0]]]], dtype=torch.float64, requires_grad=True)
        k = torch.tensor([[[[0.0], [math.log(3)]]]], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([[[[2.0], [6.0]]]], dtype=torch.float64, requires_grad=True)
        output, weights = lab.attention(q, k, v, torch.tensor([[True, True]]))
        output.sum().backward()
        self.assertAlmostEqual(float(output.item()), 5.0, places=12)
        self.assertEqual(weights.tolist(), [[[[0.25, 0.75]]]])
        self.assertAlmostEqual(float(q.grad.item()), 0.75 * math.log(3), places=12)
        torch.testing.assert_close(k.grad, torch.tensor([[[[-0.75], [0.75]]]], dtype=torch.float64))
        torch.testing.assert_close(v.grad, torch.tensor([[[[0.25], [0.75]]]], dtype=torch.float64))

    def test_masked_future_key_has_zero_probability_and_gradient(self):
        q = torch.tensor([[[[1., 2.]]]], dtype=torch.float64, requires_grad=True)
        k = torch.tensor([[[[1., 0.], [999., 999.]]]], dtype=torch.float64, requires_grad=True)
        v = torch.tensor([[[[3., 4.], [999., 999.]]]], dtype=torch.float64, requires_grad=True)
        output, weights = lab.attention(q, k, v, torch.tensor([[True, False]]))
        output.sum().backward()
        torch.testing.assert_close(output, torch.tensor([[[[3., 4.]]]], dtype=torch.float64))
        self.assertEqual(float(weights[0, 0, 0, 1]), 0)
        self.assertEqual(k.grad[0, 0, 1].abs().sum().item(), 0)
        self.assertEqual(v.grad[0, 0, 1].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            lab.attention(q, k, v, torch.tensor([[False, False]]))

    def test_rope_angles_and_relative_position_dot_product(self):
        x = torch.tensor([[[[1., 0., 0., 1.]]]], dtype=torch.float64)
        rotated = lab.rope(x, torch.tensor([3]))
        expected = torch.tensor([[[[math.cos(3), math.sin(3), -math.sin(0.03), math.cos(0.03)]]]], dtype=torch.float64)
        torch.testing.assert_close(rotated, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1), atol=1e-12, rtol=1e-12)
        q, k = x, torch.tensor([[[[0.3, -0.2, 0.7, 0.4]]]], dtype=torch.float64)
        original = (lab.rope(q, torch.tensor([3])) * lab.rope(k, torch.tensor([7]))).sum()
        shifted = (lab.rope(q, torch.tensor([13])) * lab.rope(k, torch.tensor([17]))).sum()
        torch.testing.assert_close(original, shifted, atol=1e-12, rtol=1e-12)

    def test_cached_chunk_requires_global_causal_offset(self):
        generator = torch.Generator().manual_seed(41)
        q, k, v = [torch.randn(1, 1, 5, 4, generator=generator, dtype=torch.float64) for _ in range(3)]
        positions = torch.arange(5)
        q, k = lab.rope(q, positions), lab.rope(k, positions)
        full = lab.attention(q, k, v, positions[None, :] <= positions[:, None])[0]
        incremental = lab.attention(q[:, :, 3:], k, v, positions[None, :] <= positions[3:, None])[0]
        torch.testing.assert_close(incremental, full[:, :, 3:], atol=1e-12, rtol=1e-12)
        wrong = lab.attention(q[:, :, 3:], k, v, torch.ones(2, 5, dtype=torch.bool).tril())[0]
        self.assertGreater(float((wrong - full[:, :, 3:]).abs().max()), 0.01)

    def test_online_softmax_blocks_are_stable_after_large_score_offsets(self):
        scores = torch.tensor([-2., 0.5, 1., -0.1, 3.], dtype=torch.float64)
        values = torch.tensor([[1., 3.], [7., 1.], [-2., 2.], [8., 5.], [4., -1.]], dtype=torch.float64)
        expected = torch.softmax(scores, 0) @ values
        for offset in (-10000, 0, 10000):
            for block in (1, 2, 4, 9):
                with self.subTest(offset=offset, block=block):
                    actual = lab.online_weighted_sum(scores + offset, values, block)
                    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        with self.assertRaises(ValueError):
            lab.online_weighted_sum(torch.tensor([float("inf")]), torch.ones(1, 1), 1)


if __name__ == "__main__":
    unittest.main()
