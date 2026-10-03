"""CPU-only numerical checks for the post-training chapters.

Python standard library only; no LLM weights, autograd, GPU, or network.
These checks verify formulas and bookkeeping, not language-model training.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence


def softplus(value: float) -> float:
    """Stable log(1 + exp(value)), including large positive arguments."""
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def log_softmax(row: Sequence[float]) -> list[float]:
    if not row or not all(math.isfinite(value) for value in row):
        raise ValueError("logits must be a nonempty sequence of finite values")
    peak = max(row)
    log_z = peak + math.log(sum(math.exp(value - peak) for value in row))
    return [value - log_z for value in row]


def completion_logprob(
    logits: Sequence[Sequence[float]],
    token_ids: Sequence[int],
    target_mask: Sequence[int],
) -> float:
    """Sum response log-probs after shifting exactly once.

    logits: [L, V]; token_ids: [L]; target_mask: [L-1].
    logits[t] predicts token_ids[t+1]. target_mask is already shifted.
    This mask is a loss mask, not an attention or causal mask.
    """
    if len(token_ids) < 2 or len(logits) != len(token_ids):
        raise ValueError("expected L logits rows and L token IDs, with L >= 2")
    if len(target_mask) != len(token_ids) - 1:
        raise ValueError("target_mask must have length L-1")
    if any(value not in (0, 1) for value in target_mask) or not any(target_mask):
        raise ValueError("target_mask must be binary and contain a supervised token")
    vocab_size = len(logits[0])
    if vocab_size == 0 or any(len(row) != vocab_size for row in logits):
        raise ValueError("all logits rows must have the same nonzero vocab size")
    if any(not isinstance(token, int) or not 0 <= token < vocab_size for token in token_ids):
        raise ValueError("token ID outside vocabulary")
    return sum(
        log_softmax(logits[t])[token_ids[t + 1]]
        for t, supervised in enumerate(target_mask)
        if supervised
    )


def bt_loss(winner_reward: float, loser_reward: float) -> float:
    return softplus(-(winner_reward - loser_reward))


def dpo_loss(
    policy_chosen: float,
    policy_rejected: float,
    reference_chosen: float,
    reference_rejected: float,
    beta: float,
) -> float:
    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    margin = (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)
    return softplus(-beta * margin)


def kl_optimal_policy(
    reference: Sequence[float], rewards: Sequence[float], beta: float
) -> list[float]:
    """Finite-support optimizer of E[r] - beta * KL(policy || reference)."""
    if len(reference) != len(rewards) or not reference:
        raise ValueError("reference and rewards need the same positive length")
    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    if any(prob <= 0 or not math.isfinite(prob) for prob in reference):
        raise ValueError("this example requires full support on the finite set")
    if not math.isclose(sum(reference), 1.0):
        raise ValueError("reference probabilities must sum to one")
    log_weights = [math.log(prob) + reward / beta for prob, reward in zip(reference, rewards)]
    return [math.exp(value) for value in log_softmax(log_weights)]


def kl_divergence(policy: Sequence[float], reference: Sequence[float]) -> float:
    # Only used below with valid finite categorical distributions.
    return sum(prob * math.log(prob / ref) for prob, ref in zip(policy, reference) if prob > 0)


def group_advantages(rewards: Sequence[float], epsilon: float = 1e-8) -> list[float]:
    """Outcome-level GRPO example: population std (ddof=0), epsilon outside sqrt."""
    if len(rewards) < 2:
        raise ValueError("a comparison group must contain at least two rewards")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if not all(math.isfinite(value) for value in rewards):
        raise ValueError("rewards must be finite")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
    return [(value - mean) / (std + epsilon) for value in rewards]


def ppo_surrogate(ratio: float, advantage: float, clip_epsilon: float = 0.2) -> float:
    clipped = min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
    return min(ratio * advantage, clipped * advantage)


def pass_at_k(n: int, correct: int, k: int) -> float:
    if not (0 <= correct <= n and 1 <= k <= n):
        raise ValueError("require 0 <= correct <= n and 1 <= k <= n")
    if n - correct < k:
        return 1.0
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def verify_integer_answer(text: str, expected: int) -> bool:
    """A deliberately narrow toy verifier; never executes generated text.

    Contract: one line 'Answer: <signed integer>', with <= 32 digits.
    Surrounding whitespace is allowed; extra prose/answers are rejected.
    This is NOT a general mathematical equivalence checker.
    """
    if len(text) > 80:
        return False
    match = re.fullmatch(r"Answer:[ \t]*([+-]?(?:0|[1-9][0-9]{0,31}))", text.strip(), re.ASCII)
    return match is not None and int(match.group(1)) == expected


def main() -> None:
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        # Explicit AssertionError remains active even when Python uses -O.
        if not condition:
            raise AssertionError(message)
        checks += 1

    def near(actual: float, expected: float, message: str, tolerance: float = 1e-9) -> None:
        check(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), message)

    # [BOS/PAD, prompt, answer-A, answer-B, EOS, PAD]; vocabulary size 5.
    token_ids = [0, 1, 2, 3, 4, 0]
    probabilities = [[0.2] * 5 for _ in token_ids]
    for row_index, target_id, target_probability in [(1, 2, 0.8), (2, 3, 0.5), (3, 4, 0.25)]:
        probabilities[row_index] = [(1.0 - target_probability) / 4.0] * 5
        probabilities[row_index][target_id] = target_probability
    logits = [[math.log(value) for value in row] for row in probabilities]
    target_mask = [0, 1, 1, 1, 0]
    sequence_logp = completion_logprob(logits, token_ids, target_mask)
    near(sequence_logp, math.log(0.8 * 0.5 * 0.25), "shifted completion log-prob")
    masked_changes = [row[:] for row in logits]
    masked_changes[0] = [1000, -1000, 0, 0, 0]
    masked_changes[4] = [-1000, 1000, 0, 0, 0]
    near(completion_logprob(masked_changes, token_ids, target_mask), sequence_logp, "prompt/pad masking")
    try:
        completion_logprob(logits, token_ids, [0] * 5)
    except ValueError:
        check(True, "empty supervision rejected")
    else:
        check(False, "empty supervision must fail")

    bt = bt_loss(1.2, 0.2)
    near(bt, 0.31326168751822286, "BT loss")
    near(bt_loss(11.2, 10.2), bt, "BT invariant to prompt-wise reward shift")
    near(sigmoid(1.0), 0.7310585786300049, "BT preference probability")
    check(math.isfinite(softplus(1000)), "stable softplus")

    beta = 0.2
    loss = dpo_loss(-4.0, -6.0, -5.0, -5.5, beta)
    near(loss, 0.5543552444685271, "DPO four-logprob loss")
    near(dpo_loss(-5, -5.5, -5, -5.5, beta), math.log(2), "initial DPO loss")
    check(dpo_loss(-6, -4, -5.5, -5, beta) > loss, "reversing pair raises loss")
    step = 1e-5
    finite_difference = (
        dpo_loss(-4 + step, -6, -5, -5.5, beta)
        - dpo_loss(-4 - step, -6, -5, -5.5, beta)
    ) / (2 * step)
    analytic_derivative = -beta * sigmoid(-0.3)
    near(finite_difference, analytic_derivative, "DPO derivative wrt chosen log-prob")

    reference, rewards = [0.8, 0.2], [0.0, 1.0]
    optimal = kl_optimal_policy(reference, rewards, 1.0)
    objective = lambda policy: sum(p * r for p, r in zip(policy, rewards)) - kl_divergence(policy, reference)
    optimum = objective(optimal)
    log_z = math.log(sum(p * math.exp(r) for p, r in zip(reference, rewards)))
    near(optimum, log_z, "KL optimum equals log partition")
    for probability in [0.0, 0.2, 0.5, 0.8, 1.0]:
        candidate = [probability, 1.0 - probability]
        check(objective(candidate) <= optimum + 1e-12, "analytic optimum dominates candidate")
        near(log_z - objective(candidate), kl_divergence(candidate, optimal), "KL gap identity")
    shifted_optimal = kl_optimal_policy(reference, [r + 10 for r in rewards], 1.0)
    near(shifted_optimal[0], optimal[0], "optimal policy invariant to reward shift")

    advantages = group_advantages([0, 0, 1, 1])
    for actual, expected in zip(advantages, [-1, -1, 1, 1]):
        near(actual, expected, "mixed GRPO group", tolerance=1e-7)
    near(sum(advantages), 0.0, "group advantages sum to zero")
    check(group_advantages([1, 1, 1, 1]) == [0, 0, 0, 0], "all-correct group")
    check(group_advantages([0, 0, 0, 0]) == [0, 0, 0, 0], "all-wrong group")
    near(ppo_surrogate(1.4, 2), 2.4, "PPO clips rewarded direction")
    near(ppo_surrogate(1.4, -2), -2.8, "PPO retains adverse direction")
    near(ppo_surrogate(0.6, -2), -1.6, "PPO negative-advantage lower clip")
    near(pass_at_k(5, 2, 2), 0.7, "pass@k combinatorial example")
    near(pass_at_k(5, 0, 2), 0.0, "pass@k no correct answers")
    near(pass_at_k(5, 5, 2), 1.0, "pass@k all correct answers")

    verifier_cases = {
        "Answer: 42": True,
        " Answer: +42 ": True,
        "Answer: 41": False,
        "": False,
        "The answer might be 42, or 41.": False,
        "Answer: 42\nAnswer: 41": False,
        "Answer: 42 and 43": False,
        "Answer: 042": False,
        "Answer: " + "9" * 100: False,
    }
    for output, expected_result in verifier_cases.items():
        check(verify_integer_answer(output, 42) == expected_result, "integer-verifier contract")

    print(json.dumps({
        "checks_passed": checks,
        "completion_logprob": round(sequence_logp, 9),
        "bt_loss": round(bt, 9),
        "dpo_loss": round(loss, 9),
        "dpo_chosen_logprob_derivative": round(finite_difference, 9),
        "kl_optimal_policy": [round(value, 9) for value in optimal],
        "grpo_advantages": [round(value, 7) for value in advantages],
        "pass_at_2": pass_at_k(5, 2, 2),
        "scope": "numerical checks only; no LLM training",
    }, indent=2))


if __name__ == "__main__":
    main()
