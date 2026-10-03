"""第一课：不安装第三方包，手算并核对一次学习、概率与回报。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lesson_runtime import reserve_output_dir


def calculate(learning_rate: float) -> dict:
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("学习率必须是有限的正数")
    x, target, weight, bias = 2.0, 5.0, 1.0, 0.0
    prediction = weight * x + bias
    error = prediction - target
    loss = 0.5 * error**2
    grad_weight, grad_bias = error * x, error
    next_weight = weight - learning_rate * grad_weight
    next_bias = bias - learning_rate * grad_bias
    next_prediction = next_weight * x + next_bias
    try:
        next_loss = 0.5 * (next_prediction - target)**2
    except OverflowError as error:
        raise ValueError("学习率使更新结果溢出，请减小学习率") from error
    if not all(math.isfinite(value) for value in (next_weight, next_bias, next_prediction, next_loss)):
        raise ValueError("更新后的参数、预测或损失不是有限数，请减小学习率")
    logits = [0.0, math.log(3.0)]
    peak = max(logits)
    exponentials = [math.exp(value - peak) for value in logits]
    probabilities = [value / sum(exponentials) for value in exponentials]
    supervised_probabilities = [0.8, 0.5, 0.25]
    sequence_logprob = sum(math.log(p) for p in supervised_probabilities)
    delayed_return = -0.2 + 0.9 * 2.0
    checks = [
        math.isclose(prediction, 2.0), math.isclose(loss, 4.5),
        math.isclose(grad_weight, -6.0), math.isclose(grad_bias, -3.0),
        math.isclose(sum(probabilities), 1.0), math.isclose(probabilities[1], 0.75),
        math.isclose(sequence_logprob, math.log(0.1)), math.isclose(delayed_return, 1.6),
    ]
    if not all(checks):
        raise AssertionError("手算参照不一致，请检查公式实现")
    if math.isclose(learning_rate, 0.1):
        if not math.isclose(next_loss, 1.125):
            raise AssertionError("默认学习率的一步更新应得到损失 1.125")
    return {
        "learning_rate": learning_rate,
        "before": {"weight": weight, "bias": bias, "prediction": prediction, "loss": loss},
        "gradients": {"weight": grad_weight, "bias": grad_bias},
        "after": {"weight": next_weight, "bias": next_bias, "prediction": next_prediction, "loss": next_loss},
        "classification_probabilities": probabilities,
        "correct_class_nll": -math.log(probabilities[1]),
        "response_logprob": sequence_logprob,
        "response_mean_nll": -sequence_logprob / len(supervised_probabilities),
        "immediate_return": 1.0, "delayed_return": delayed_return,
        "checks_passed": len(checks),
        "scope": "手写解析梯度和数值运算；无自动微分、语言模型或第三方包",
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--learning-rate", type=float, default=0.1, help="梯度更新步长，默认 0.1")
    parser.add_argument("--output-dir", type=Path, help="可选：保存 first_steps.json 的目录")
    args = parser.parse_args()
    try:
        result = calculate(args.learning_rate)
    except ValueError as error:
        parser.error(str(error))
    try:
        args.output_dir = reserve_output_dir(args.output_dir, "first")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    before, after = result["before"], result["after"]
    print("第一课：模型先预测，再根据误差调整参数。")
    print(f"1. 更新前：预测={before['prediction']:.6f}，损失={before['loss']:.6f}")
    print("2. 手算梯度：权重梯度=-6，偏置梯度=-3")
    print(f"3. 学习率={args.learning_rate:g}；更新后：权重={after['weight']:.6f}，偏置={after['bias']:.6f}")
    print(f"4. 更新后：预测={after['prediction']:.6f}，损失={after['loss']:.6f}")
    print("5. 两个候选的 softmax 概率：0.25、0.75；正确候选的负对数损失=0.287682")
    print(f"6. 三个回答 token 的概率相乘=0.1；log-prob={result['response_logprob']:.6f}")
    print("7. 立即结束得 1；先付出 0.2 再得 2，折扣 0.9 后的回报=1.6")
    print(f"数值参照检查通过：{result['checks_passed']} 项。请解释这些数字，运行成功只是第一步。")
    if args.output_dir:
        destination = args.output_dir / "first_steps.json"
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"详细结果：{destination.resolve()}")


if __name__ == "__main__":
    main()
