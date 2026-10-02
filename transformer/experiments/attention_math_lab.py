"""CPU 上核对 Attention、RoPE、增量缓存、在线 softmax 与 MLA 线性等价。

这是数学与实现练习，不是 FlashAttention GPU 内核或大规模模型训练。
运行：python -X utf8 transformer/experiments/attention_math_lab.py
"""
from __future__ import annotations

import json
import math

import torch


def attention(q, k, v, allowed):
    """q:[B,H,Q,D], k:[B,H,K,D], v:[B,H,K,Dv]; True 表示可见。"""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q/K/V 必须有四个轴")
    if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]:
        raise ValueError("批、头或 KV 位置轴不一致")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] == 0:
        raise ValueError("Q/K 特征维必须相同且非空")
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    visible = torch.broadcast_to(allowed.to(device=q.device, dtype=torch.bool), scores.shape)
    if not visible.any(-1).all():
        raise ValueError("每个 query 必须至少有一个可见 key")
    weights = scores.masked_fill(~visible, -torch.inf).softmax(-1)
    return weights @ v, weights


def rope(x, positions, base=10000.0):
    """相邻配对 RoPE；x:[B,H,T,D]，positions:[T]，全部 D 为旋转维。"""
    dimension = x.shape[-1]
    if x.ndim != 4 or dimension <= 0 or dimension % 2:
        raise ValueError("旋转维必须为正偶数，输入有四个轴")
    if positions.ndim != 1 or positions.numel() != x.shape[-2] or base <= 0:
        raise ValueError("位置数量或频率基数无效")
    frequency = base ** (-torch.arange(0, dimension, 2, device=x.device, dtype=x.dtype) / dimension)
    angle = positions.to(device=x.device, dtype=x.dtype)[:, None] * frequency
    cosine, sine = angle.cos(), angle.sin()
    pairs = x.reshape(*x.shape[:-1], dimension // 2, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    return torch.stack((first * cosine - second * sine,
                        first * sine + second * cosine), -1).flatten(-2)


def online_weighted_sum(scores, values, block_size):
    """单 query，有限且全部可见的 scores:[K]、values:[K,Dv]。"""
    if scores.ndim != 1 or values.ndim != 2 or values.shape[0] != scores.numel():
        raise ValueError("分数和值的形状不匹配")
    if not scores.numel() or block_size < 1 or not torch.isfinite(scores).all():
        raise ValueError("要求非空有限分数和正块大小")
    if not torch.isfinite(values).all():
        raise ValueError("values 必须有限")
    maximum = scores.new_tensor(-torch.inf)
    denominator = scores.new_zeros(())
    numerator = values.new_zeros(values.shape[1])
    for start in range(0, scores.numel(), block_size):
        block = scores[start:start + block_size]
        block_values = values[start:start + block_size]
        new_maximum = torch.maximum(maximum, block.max())
        rescale = (maximum - new_maximum).exp()
        weights = (block - new_maximum).exp()
        numerator = rescale * numerator + weights @ block_values
        denominator = rescale * denominator + weights.sum()
        maximum = new_maximum
    return numerator / denominator


def close(actual, expected, tolerance=1e-11):
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def check_attention_gradient():
    q = torch.tensor([[[[1.0]]]], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([[[[0.0], [math.log(3)]]]], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([[[[2.0], [6.0]]]], dtype=torch.float64, requires_grad=True)
    output, weights = attention(q, k, v, torch.ones(1, 2, dtype=torch.bool))
    output.sum().backward()
    close(output, torch.full_like(output, 5.0))
    close(weights, torch.tensor([[[[0.25, 0.75]]]], dtype=torch.float64))
    close(q.grad, torch.full_like(q, 0.75 * math.log(3)))
    close(k.grad, torch.tensor([[[[-0.75], [0.75]]]], dtype=torch.float64))
    close(v.grad, torch.tensor([[[[0.25], [0.75]]]], dtype=torch.float64))
    # 非标量随机输入，用解析 VJP 独立核对 autograd。
    generator = torch.Generator().manual_seed(7)
    q, k, v = [torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
               for shape in ((2, 2, 3, 4), (2, 2, 5, 4), (2, 2, 5, 3))]
    allowed = torch.arange(5)[None, :] <= torch.arange(3)[:, None] + 2
    output, a = attention(q, k, v, allowed)
    g = torch.randn(output.shape, generator=generator, dtype=torch.float64)
    gradients = torch.autograd.grad((output * g).sum(), (q, k, v))
    with torch.no_grad():
        da = g @ v.transpose(-2, -1)
        ds = a * (da - (a * da).sum(-1, keepdim=True))
        expected = (ds @ k / 2, ds.transpose(-2, -1) @ q / 2,
                    a.transpose(-2, -1) @ g)
        errors = [float((actual - reference).abs().max()) for actual, reference in zip(gradients, expected)]
        for actual, reference in zip(gradients, expected):
            close(actual, reference)
    try:
        attention(q, k, v, torch.zeros(3, 5, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError("未拒绝全屏蔽行")
    return max(errors)


def check_cache():
    generator = torch.Generator().manual_seed(11)
    q, k, v = [torch.randn(2, 3, 7, 4, generator=generator, dtype=torch.float64) for _ in range(3)]
    positions = torch.arange(7)
    rotated_q, rotated_k = rope(q, positions), rope(k, positions)
    causal = positions[:, None] >= positions[None, :]
    full, _ = attention(rotated_q, rotated_k, v, causal)
    close(rotated_q.square().sum(-1), q.square().sum(-1))
    outputs, key_cache, value_cache = [], None, None
    start = 0
    for size in (3, 2, 2):
        end = start + size
        current_q = rope(q[:, :, start:end], positions[start:end])
        current_k = rope(k[:, :, start:end], positions[start:end])
        key_cache = current_k if key_cache is None else torch.cat((key_cache, current_k), -2)
        value_cache = v[:, :, start:end] if value_cache is None else torch.cat((value_cache, v[:, :, start:end]), -2)
        allowed = torch.arange(end)[None, :] <= torch.arange(start, end)[:, None]
        output, _ = attention(current_q, key_cache, value_cache, allowed)
        outputs.append(output)
        start = end
    cached = torch.cat(outputs, -2)
    close(cached, full)
    wrong, _ = attention(rotated_q[:, :, 3:5], rotated_k[:, :, :5], v[:, :, :5],
                         torch.ones(2, 5, dtype=torch.bool).tril())
    wrong_error = float((wrong - full[:, :, 3:5]).abs().max())
    if wrong_error < 1e-3:
        raise AssertionError("朴素矩形 tril 的错误对照未产生差异")
    example = rope(torch.tensor([[[[1., 0., 0., 1.]]]], dtype=torch.float64), torch.tensor([1]))
    close(example, torch.tensor([[[[math.cos(1), math.sin(1), -math.sin(.01), math.cos(.01)]]]], dtype=torch.float64))
    return float((cached - full).abs().max()), wrong_error


def check_online_and_mla():
    scores = torch.tensor([0., math.log(2), math.log(4)], dtype=torch.float64)
    values = torch.tensor([[1.], [3.], [5.]], dtype=torch.float64)
    expected = torch.tensor([27 / 7], dtype=torch.float64)
    errors = []
    for block in (1, 2, 3):
        for offset in (0, 1000):
            actual = online_weighted_sum(scores + offset, values, block)
            close(actual, expected)
            errors.append(float((actual - expected).abs().max()))
    generator = torch.Generator().manual_seed(13)
    c = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    uk = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    uv = torch.randn(6, 3, generator=generator, dtype=torch.float64)
    query = torch.randn(4, generator=generator, dtype=torch.float64)
    explicit = ((c @ uk.T @ query / 2).softmax(0) @ (c @ uv.T))
    absorbed = ((c @ (uk.T @ query) / 2).softmax(0) @ c) @ uv.T
    close(explicit, absorbed)
    return max(errors), float((explicit - absorbed).abs().max())


def main():
    gradient_error = check_attention_gradient()
    cache_error, wrong_mask_error = check_cache()
    online_error, mla_error = check_online_and_mla()
    z = torch.tensor([math.log(3), 0., -10.], dtype=torch.float64, requires_grad=True)
    experts = torch.tensor([2., 6., 100.], dtype=torch.float64)
    selected = z.topk(2).indices
    output = (z[selected].softmax(0) * experts[selected]).sum()
    output.backward()
    close(output, torch.tensor(3., dtype=torch.float64))
    close(z.grad, torch.tensor([-.75, .75, 0.], dtype=torch.float64))
    report = {
        "attention_vjp_max_error": gradient_error,
        "rope_cached_attention_max_error": cache_error,
        "wrong_rectangular_mask_error": wrong_mask_error,
        "online_softmax_max_error": online_error,
        "mla_projection_absorption_max_error": mla_error,
        "topk_router_gradient": z.grad.tolist(),
        "scope": "CPU float64 数学与核心函数；未实现 GPU FlashAttention 或完整训练模型",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
