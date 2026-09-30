"""A deterministic, CPU-only tool-control laboratory.

No LLM, network, shell, eval, or generated-code execution is used.
The scripted policy computes (7 + 5) / denominator with two allowlisted tools.
The experiment checks observable state transitions, not LLM intelligence.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Request:
    request_id: str
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Observation:
    ok: bool
    value: float | None = None
    error: str | None = None
    retryable: bool = False


SCHEMAS = {
    "add": {"left", "right"},
    "divide": {"numerator", "denominator"},
}


def validate(request: Request) -> str | None:
    if request.tool not in SCHEMAS:
        return "unknown_tool"
    if set(request.arguments) != SCHEMAS[request.tool]:
        return "argument_fields_mismatch"
    for value in request.arguments.values():
        if type(value) not in (int, float):
            return "argument_must_be_numeric"
        # A deliberately bounded numeric interface, avoiding enormous integers.
        if not -1e6 <= value <= 1e6 or not math.isfinite(value):
            return "argument_out_of_range"
    if request.tool == "divide" and request.arguments["denominator"] == 0:
        return "division_by_zero"
    return None


class LocalExecutor:
    def __init__(self, temporary_failures: int = 0):
        self.temporary_failures = temporary_failures
        self.calls = 0

    def execute(self, request: Request) -> Observation:
        # Validation also lives at the executor boundary, even if called directly.
        invalid = validate(request)
        if invalid is not None:
            return Observation(False, error=invalid)
        self.calls += 1
        if self.temporary_failures > 0:
            self.temporary_failures -= 1
            return Observation(False, error="temporary_unavailable", retryable=True)
        args = request.arguments
        if request.tool == "add":
            value = float(args["left"] + args["right"])
        else:
            value = float(args["numerator"] / args["denominator"])
        if not math.isfinite(value):
            return Observation(False, error="nonfinite_result")
        return Observation(True, value=value)


ALLOWED_TRANSITIONS = {
    "ready": {"deciding"},
    "deciding": {"validating", "done"},
    "validating": {"calling", "failed", "budget_exhausted"},
    "calling": {"observing"},
    "observing": {"deciding", "failed"},
    "done": set(),
    "failed": set(),
    "budget_exhausted": set(),
}


def run(
    *, max_calls: int = 3, temporary_failures: int = 0,
    denominator: Any = 3, first_tool: str = "add",
) -> dict[str, Any]:
    if type(max_calls) is not int or max_calls < 0:
        raise ValueError("max_calls must be a nonnegative integer")
    if type(temporary_failures) is not int or temporary_failures < 0:
        raise ValueError("temporary_failures must be a nonnegative integer")
    executor = LocalExecutor(temporary_failures)
    state = "ready"
    trace: list[dict[str, Any]] = [{"event": 0, "state": state}]
    values: list[float] = []

    def transition(next_state: str, **fields: Any) -> None:
        nonlocal state
        if next_state not in ALLOWED_TRANSITIONS[state]:
            raise AssertionError(f"illegal transition: {state} -> {next_state}")
        state = next_state
        trace.append({"event": len(trace), "state": state, **fields})

    transition("deciding")
    while state not in {"done", "failed", "budget_exhausted"}:
        # This is a fixed reactive policy, not a language-model completion.
        if len(values) == 2:
            transition("done", final_value=values[-1], evidence_steps=["step-0", "step-1"])
            break
        if not values:
            request = Request("step-0", first_tool, {"left": 7, "right": 5})
        else:
            request = Request("step-1", "divide", {"numerator": values[0], "denominator": denominator})
        transition("validating", request=asdict(request))
        invalid = validate(request)
        if invalid:
            transition("failed", error=invalid)
            break
        if executor.calls >= max_calls:
            transition("budget_exhausted", calls=executor.calls)
            break
        transition("calling", request_id=request.request_id, attempt=executor.calls + 1)
        observation = executor.execute(request)
        transition("observing", request_id=request.request_id, observation=asdict(observation))
        if observation.ok:
            if observation.value is None:
                raise AssertionError("a successful numeric tool must return a value")
            values.append(observation.value)
            transition("deciding")
        elif observation.retryable:
            # No progress is recorded; the next request keeps the same logical ID.
            transition("deciding", retry_of=request.request_id)
        else:
            transition("failed", error=observation.error)
    return {
        "status": state,
        "calls": executor.calls,
        "final_value": values[-1] if state == "done" else None,
        "trace": trace,
    }


def main() -> None:
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1

    normal = run()
    recovered = run(temporary_failures=1)
    exhausted = run(max_calls=1)
    repeated_failure = run(max_calls=3, temporary_failures=10)
    invalid = run(denominator=0)
    unknown = run(first_tool="execute_arbitrary_code")
    wrong_type = run(denominator="3")
    zero_budget = run(max_calls=0)
    overflow = run(denominator=1e-320)

    check(normal["status"] == "done" and normal["final_value"] == 4.0, "correct final result")
    check(normal["calls"] == 2, "two dependent arithmetic calls")
    check(recovered["status"] == "done" and recovered["final_value"] == 4.0, "recoverable failure")
    check(recovered["calls"] == 3, "retry consumes call budget")
    retried_ids = [item["request_id"] for item in recovered["trace"] if item["state"] == "calling"]
    check(retried_ids == ["step-0", "step-0", "step-1"], "retry preserves logical request identity")
    check(exhausted["status"] == "budget_exhausted" and exhausted["calls"] == 1, "budget stop")
    check(exhausted["final_value"] is None, "partial sum cannot be reported as final answer")
    check(repeated_failure["status"] == "budget_exhausted" and repeated_failure["calls"] == 3, "bounded retry")
    check(invalid["status"] == "failed" and invalid["trace"][-1]["error"] == "division_by_zero", "argument validation")
    check(invalid["calls"] == 1, "invalid divide never executed")
    check(unknown["status"] == "failed" and unknown["calls"] == 0, "tool allowlist")
    check(wrong_type["trace"][-1]["error"] == "argument_must_be_numeric", "no string evaluation")
    check(zero_budget["status"] == "budget_exhausted" and zero_budget["calls"] == 0, "zero call budget")
    check(overflow["status"] == "failed" and overflow["trace"][-1]["error"] == "nonfinite_result", "nonfinite tool result rejected")
    cases = [normal, recovered, exhausted, repeated_failure, invalid, unknown, wrong_type, zero_budget, overflow]
    for result in cases:
        trace = result["trace"]
        check(all(event["event"] == index for index, event in enumerate(trace)), "ordered trace IDs")
        check(all(right["state"] in ALLOWED_TRANSITIONS[left["state"]] for left, right in zip(trace, trace[1:])), "legal full trace")
        check(sum(event["state"] == "calling" for event in trace) == result["calls"], "call accounting")

    print(json.dumps({
        "checks_passed": checks,
        "cases": [{key: result[key] for key in ("status", "calls", "final_value")} for result in cases],
        "recovered_trace": recovered["trace"],
        "scope": "scripted controller only; no LLM, RL training, or external actions",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
