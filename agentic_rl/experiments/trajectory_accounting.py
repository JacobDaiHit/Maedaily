"""Standard-library arithmetic lab; no language model, API, or RL training."""

from __future__ import annotations

import json
from math import isclose, isfinite
from typing import Sequence


def discounted_returns(rewards: Sequence[float], gamma: float) -> list[float]:
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    returns = [0.0] * len(rewards)
    tail = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        if not isfinite(rewards[index]):
            raise ValueError("rewards must be finite")
        tail = rewards[index] + gamma * tail
        returns[index] = tail
    return returns


def masked_score_loss(log_probs: Sequence[float], masks: Sequence[int]) -> float:
    """Negative mean log-prob with fixed advantage 1; not a PPO implementation."""
    if len(log_probs) != len(masks):
        raise ValueError("log_probs and masks must have equal length")
    if any(mask not in (0, 1) for mask in masks):
        raise ValueError("mask must be binary")
    if not sum(masks):
        raise ValueError("at least one policy token is required")
    if any(not isfinite(value) for value in log_probs):
        raise ValueError("log_probs must be finite")
    return -sum(value * mask for value, mask in zip(log_probs, masks)) / sum(masks)


def td_target(reward: float, gamma: float, next_value: float, terminated: bool) -> float:
    return reward if terminated else reward + gamma * next_value


def run_checks() -> dict[str, object]:
    log_probs = [-0.2, -9.0, -0.3]
    correct_mask_loss = masked_score_loss(log_probs, [1, 0, 1])
    wrong_mask_loss = masked_score_loss(log_probs, [1, 1, 1])
    assert isclose(correct_mask_loss, 0.25)
    assert isclose(wrong_mask_loss, 9.5 / 3)
    try:
        masked_score_loss(log_probs, [0, 0, 0])
    except ValueError:
        zero_mask_rejected = True
    else:
        raise AssertionError("zero mask must not silently produce a valid loss")

    rewards = [-0.05, -0.05, 1.0]
    returns_1 = discounted_returns(rewards, 1.0)
    returns_09 = discounted_returns(rewards, 0.9)
    assert all(isclose(a, b) for a, b in zip(returns_1, [0.90, 0.95, 1.0]))
    assert all(isclose(a, b) for a, b in zip(returns_09, [0.715, 0.85, 1.0]))
    terminal = td_target(-0.05, 0.9, 0.8, terminated=True)
    truncated = td_target(-0.05, 0.9, 0.8, terminated=False)
    assert isclose(terminal, -0.05)
    assert isclose(truncated, 0.67)

    segments = [("prompt", 12), ("policy", 4), ("tool", 30), ("policy", 6), ("pad", 8)]
    policy_tokens = sum(count for role, count in segments if role == "policy")
    context_tokens = sum(count for role, count in segments if role != "pad")
    assert policy_tokens == 10 and context_tokens == 52
    cost = 0.02
    policy_a = {"tasks": 100, "correct": 70, "tool_calls": 200}
    policy_b = {"tasks": 100, "correct": 69, "tool_calls": 100}
    for policy in (policy_a, policy_b):
        policy["success_rate"] = policy["correct"] / policy["tasks"]
        policy["mean_tool_calls"] = policy["tool_calls"] / policy["tasks"]
        policy["utility"] = policy["success_rate"] - cost * policy["mean_tool_calls"]
    assert policy_a["success_rate"] > policy_b["success_rate"]
    assert policy_a["utility"] < policy_b["utility"]

    return {
        "experiment_type": "synthetic_arithmetic_only_no_llm_training",
        "mask": {"correct_loss": correct_mask_loss, "incorrect_loss": wrong_mask_loss,
                 "zero_mask_rejected": zero_mask_rejected},
        "returns": {"gamma_1": returns_1, "gamma_0.9": returns_09},
        "td_target": {"terminated": terminal, "truncated": truncated},
        "tokens": {"policy": policy_tokens, "context_nonpadding": context_tokens},
        "budget": {"cost_per_call": cost, "A": policy_a, "B": policy_b},
        "checks_passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), ensure_ascii=False, indent=2))
