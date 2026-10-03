"""A reproducible, standard-library-only MDP laboratory for RL chapters 1-3.

Run from the repository root:
    python scripts/run_lesson.py rl -- --seed 7

The environment is deterministic; the behavior policy is stochastic.  This is
a correctness/learning exercise, not a benchmark of deep RL performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Mapping
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from lesson_runtime import reserve_output_dir


STATES = ("start", "work", "terminal")
ACTIONS = {"start": ("quit", "invest"), "work": ("finish", "wait"), "terminal": ()}
# (next_state, reward, terminated): rewards belong to transitions.
TRANSITIONS = {
    ("start", "quit"): ("terminal", 1.0, True),
    ("start", "invest"): ("work", -0.2, False),
    ("work", "finish"): ("terminal", 2.0, True),
    ("work", "wait"): ("work", -0.1, False),
}

QTable = dict[tuple[str, str], float]


def td_target(reward: float, gamma: float, next_value: float, terminated: bool) -> float:
    """External truncation intentionally does not occur in this mask."""
    return reward + gamma * (0.0 if terminated else next_value)


def analytic_solution(gamma: float) -> tuple[dict[str, float], QTable]:
    # At work, finish is optimal for every 0 <= gamma < 1:
    # -0.1 + gamma * 2 < 2.  Investing depends on the discount.
    values = {"start": max(1.0, -0.2 + 2.0 * gamma), "work": 2.0, "terminal": 0.0}
    q = {
        ("start", "quit"): 1.0,
        ("start", "invest"): -0.2 + 2.0 * gamma,
        ("work", "finish"): 2.0,
        ("work", "wait"): -0.1 + 2.0 * gamma,
    }
    return values, q


def action_values(values: Mapping[str, float], gamma: float) -> QTable:
    return {
        pair: td_target(reward, gamma, values[next_state], terminated)
        for pair, (next_state, reward, terminated) in TRANSITIONS.items()
    }


def value_iteration(gamma: float, tolerance: float = 1e-12) -> tuple[dict[str, float], QTable, int, float]:
    values = dict.fromkeys(STATES, 0.0)
    for iteration in range(1, 100_001):
        q = action_values(values, gamma)
        updated = {
            state: max(q[state, action] for action in ACTIONS[state]) if ACTIONS[state] else 0.0
            for state in STATES
        }
        difference = max(abs(updated[state] - values[state]) for state in STATES)
        values = updated
        if difference < tolerance:
            final_q = action_values(values, gamma)
            residual = max(
                abs(max(final_q[state, action] for action in ACTIONS[state]) - values[state])
                for state in STATES if ACTIONS[state]
            )
            return values, final_q, iteration, residual
    raise RuntimeError("Value iteration did not reach the requested tolerance.")


def greedy_policy(q: QTable) -> dict[str, str]:
    # Deterministic tie-breaking is used only for reporting/evaluation.
    return {state: max(ACTIONS[state], key=lambda action: q[state, action])
            for state in STATES if ACTIONS[state]}


def exact_start_value(policy: Mapping[str, str], gamma: float) -> float:
    """Closed-form evaluation, including a policy that waits forever."""
    if policy["start"] == "quit":
        return 1.0
    work_value = 2.0 if policy["work"] == "finish" else -0.1 / (1.0 - gamma)
    return -0.2 + gamma * work_value


def max_q_error(q: QTable, reference: QTable) -> float:
    return max(abs(q[pair] - reference[pair]) for pair in reference)


def choose_action(q: QTable, state: str, epsilon: float, rng: random.Random) -> str:
    actions = ACTIONS[state]
    if rng.random() < epsilon:
        return rng.choice(actions)
    best = max(q[state, action] for action in actions)
    # Random tie-breaking avoids a hidden preference from dictionary order.
    return rng.choice([action for action in actions if q[state, action] == best])


def q_learning(
    *, gamma: float, seed: int, episodes: int, epsilon: float,
    max_steps: int, record_every: int,
) -> tuple[QTable, dict[tuple[str, str], int], list[dict[str, float | int]], int]:
    rng = random.Random(seed)
    q = dict.fromkeys(TRANSITIONS, 0.0)
    visits = dict.fromkeys(TRANSITIONS, 0)
    _, reference_q = analytic_solution(gamma)
    records: list[dict[str, float | int]] = []
    truncation_count = 0
    for episode in range(1, episodes + 1):
        state = "start"
        for step in range(1, max_steps + 1):
            action = choose_action(q, state, epsilon, rng)
            next_state, reward, terminated = TRANSITIONS[state, action]
            truncated = (step == max_steps) and not terminated
            visits[state, action] += 1
            alpha = visits[state, action] ** -0.6
            next_value = max((q[next_state, a] for a in ACTIONS[next_state]), default=0.0)
            target = td_target(reward, gamma, next_value, terminated)
            q[state, action] += alpha * (target - q[state, action])
            if terminated or truncated:
                truncation_count += int(truncated)
                break
            state = next_state
        if episode == 1 or episode % record_every == 0 or episode == episodes:
            records.append({
                "episode": episode,
                "max_q_error": max_q_error(q, reference_q),
                "greedy_start_value": exact_start_value(greedy_policy(q), gamma),
                "q_start_quit": q["start", "quit"],
                "q_start_invest": q["start", "invest"],
                "q_work_finish": q["work", "finish"],
                "q_work_wait": q["work", "wait"],
                "truncations": truncation_count,
            })
    return q, visits, records, truncation_count


def boundary_checks() -> dict[str, float | str]:
    """Independent numerical oracles for a frequent RL implementation bug."""
    # reward=0, V(s')=2: terminating gives 0, sampling truncation gives 1.8.
    terminated_target = td_target(0.0, 0.9, 2.0, terminated=True)
    truncated_target = td_target(0.0, 0.9, 2.0, terminated=False)
    if not math.isclose(terminated_target, 0.0, abs_tol=1e-12):
        raise AssertionError("A true terminal transition must not bootstrap.")
    if not math.isclose(truncated_target, 1.8, abs_tol=1e-12):
        raise AssertionError("An external truncation must preserve the bootstrap.")
    # The MC example in chapter 3 has rewards [-0.2, 2] and gamma=0.9.
    first_return = -0.2 + 0.9 * 2.0
    if not math.isclose(first_return, 1.6, abs_tol=1e-12):
        raise AssertionError("The complete discounted return is wrong.")
    return {"status": "passed", "terminated_target": terminated_target,
            "truncated_target": truncated_target, "two_step_return": first_return}


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=positive_int, default=20_000)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--max-steps", type=positive_int, default=3,
                        help="external collection limit, not a terminal task rule; must be >=2")
    parser.add_argument("--record-every", type=positive_int, default=500)
    parser.add_argument("--output-dir", type=Path, help="optionally write summary.json and learning_curve.csv")
    parser.add_argument("--require-q-error", type=float,
                        help="fail if the final sup-norm Q error exceeds this threshold")
    args = parser.parse_args()
    if not 0.0 <= args.gamma < 1.0:
        parser.error("--gamma must satisfy 0 <= gamma < 1")
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("--epsilon must satisfy 0 <= epsilon <= 1")
    if args.max_steps < 2:
        parser.error("--max-steps must be >=2 so the work state can be sampled")
    if args.require_q_error is not None and (not math.isfinite(args.require_q_error) or args.require_q_error < 0):
        parser.error("--require-q-error must be finite and nonnegative")

    try:
        args.output_dir = reserve_output_dir(args.output_dir, "rl")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    checks = boundary_checks()
    reference_values, reference_q = analytic_solution(args.gamma)
    vi_values, vi_q, iterations, residual = value_iteration(args.gamma)
    vi_error = max(abs(vi_values[s] - reference_values[s]) for s in STATES)
    if vi_error > 1e-10 or max_q_error(vi_q, reference_q) > 1e-10:
        raise AssertionError("Value iteration disagrees with the independent analytic solution.")

    learned_q, visits, records, truncations = q_learning(
        gamma=args.gamma, seed=args.seed, episodes=args.episodes,
        epsilon=args.epsilon, max_steps=args.max_steps, record_every=args.record_every,
    )
    q_error = max_q_error(learned_q, reference_q)
    learned_policy = greedy_policy(learned_q)
    summary = {
        "environment": "three_state_deterministic_mdp",
        "config": {"seed": args.seed, "episodes": args.episodes, "gamma": args.gamma,
                   "epsilon": args.epsilon, "max_steps": args.max_steps,
                   "learning_rate": "N(s,a)^(-0.6)", "record_every": args.record_every},
        "boundary_checks": checks,
        "analytic_values": reference_values,
        "value_iteration": {"iterations": iterations, "values": vi_values,
                            "max_value_error": vi_error, "bellman_residual": residual},
        "q_learning": {
            "q": {f"{state}/{action}": value for (state, action), value in learned_q.items()},
            "visits": {f"{state}/{action}": count for (state, action), count in visits.items()},
            "max_q_error": q_error, "policy": learned_policy,
            "greedy_start_value": exact_start_value(learned_policy, args.gamma),
            "truncations": truncations,
            "threshold_passed": (q_error <= args.require_q_error) if args.require_q_error is not None else None,
        },
    }
    if args.output_dir:
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with (args.output_dir / "learning_curve.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    print(f"seed={args.seed} episodes={args.episodes} gamma={args.gamma} epsilon={args.epsilon}")
    print(f"Boundary checks: {checks['status']} (terminal=0, truncated=1.8)")
    print(f"Value iteration: {iterations} iterations, residual={residual:.3g}, error={vi_error:.3g}")
    print("State/action       analytic       learned       visits")
    for (state, action), expected in reference_q.items():
        print(f"{state + '/' + action:<18} {expected:>10.6f} {learned_q[state, action]:>13.6f} {visits[state, action]:>8}")
    print(f"Q-learning max error: {q_error:.6g}")
    print(f"Greedy policy: {learned_policy}; exact start value={exact_start_value(learned_policy, args.gamma):.6f}")
    print(f"External truncations: {truncations} (bootstrapping preserved)")
    if args.output_dir:
        print(f"Outputs: {args.output_dir.resolve()}")
    if args.require_q_error is not None and q_error > args.require_q_error:
        raise SystemExit(f"Q error {q_error:.6g} exceeds required threshold {args.require_q_error:.6g}")


if __name__ == "__main__":
    main()
