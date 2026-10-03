"""Frozen, standard-library data and acceptance protocol for a local quality baseline.

This protocol is separate from the earlier mechanism lab. All seeds use the same
families, panels and thresholds. Development metrics alone may select weights;
the final panels are opened only after that selection is committed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

PROTOCOL_VERSION = "maedaily-character-quality-v2"
SPLITS = ("train", "dev", "dev_copy", "eval_main", "eval_transfer", "eval_regression")
MAIN_THRESHOLD = 0.8
COPY_THRESHOLD = 0.9
MIN_PANEL_SIZE = 20
ARITHMETIC_LABELS = set(map(str, range(-9, 19)))


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rank(kind: str, family: str) -> str:
    return hashlib.sha256(f"{PROTOCOL_VERSION}|{kind}|{family}".encode("utf-8")).hexdigest()


def _arithmetic_family(left: int, right: int) -> str:
    return f"operand-pair:{min(left, right)}:{max(left, right)}"


def _arithmetic_records(left: int, right: int, split: str) -> list[dict[str, Any]]:
    records = []
    orders = [(left, right)] if left == right else [(left, right), (right, left)]
    for first, second in orders:
        for operation in ("+", "-"):
            answer = first + second if operation == "+" else first - second
            question = f"{first}{operation}{second}="
            records.append({"id": f"{split}:arith:{question}", "question": question, "answer": str(answer),
                            "family": "small-add" if operation == "+" else "small-subtract",
                            "family_id": _arithmetic_family(first, second), "split": split, "task": "arithmetic"})
    return records


def fixed_quality_data() -> dict[str, list[dict[str, Any]]]:
    """Return a fresh copy of immutable protocol data; no seed/config influences it.

    Nineteen boundary families (0,k) or (k,9) ensure train covers all domain
    answer labels. The other 36 families have a frozen SHA ordering: ten dev,
    ten main, sixteen train. Copy values have a separate frozen 70/15/15 split;
    all three input spellings of each value remain together.
    """
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    families = [(left, right) for left in range(10) for right in range(left, 10)]
    nonanchors = [(left, right) for left, right in families if left != 0 and right != 9]
    ranked = sorted(nonanchors, key=lambda pair: _rank("arithmetic", _arithmetic_family(*pair)))
    dev, main = set(ranked[:10]), set(ranked[10:20])
    for left, right in families:
        split = "dev" if (left, right) in dev else "eval_main" if (left, right) in main else "train"
        result[split].extend(_arithmetic_records(left, right, split))
    # Extrapolation intentionally contains both unseen operands and unseen labels.
    # It is reported separately and is not a domain-baseline acceptance condition.
    for left in range(10):
        for right in (10, 11, 12):
            result["eval_transfer"].extend(_arithmetic_records(left, right, "eval_transfer"))
    copy_values = sorted(range(100), key=lambda value: _rank("copy", f"copy-value:{value}"))
    copy_splits = {value: "train" if index < 70 else "dev_copy" if index < 85 else "eval_regression"
                   for index, value in enumerate(copy_values)}
    for value in range(100):
        split = copy_splits[value]
        for variant, prefix in (("canonical", ""), ("one-leading-zero", "0"), ("two-leading-zeros", "00")):
            question = f"copy {prefix}{value}"
            result[split].append({"id": f"{split}:copy:{value}:{variant}", "question": question,
                                  "answer": str(value), "family": "copy", "family_id": f"copy-value:{value}",
                                  "split": split, "task": "copy", "variant": variant})
    validate_quality_data(result)
    return result


def validate_quality_data(data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Reject overlap, semantic aliases, incorrect targets and coverage holes."""
    if set(data) != set(SPLITS):
        raise ValueError("quality protocol requires exactly the six frozen splits")
    ids, questions, owner = set(), set(), {}
    labels: dict[str, set[str]] = {"arithmetic": set(), "copy": set()}
    families: dict[str, set[str]] = {name: set() for name in SPLITS}
    for split in SPLITS:
        if not data[split]:
            raise ValueError(f"empty quality split: {split}")
        for record in data[split]:
            if record.get("split") != split:
                raise ValueError("record split does not match containing split")
            if record.get("id") in ids or record.get("question") in questions:
                raise ValueError("duplicate quality sample ID or prompt")
            ids.add(record.get("id"))
            questions.add(record.get("question"))
            question = record.get("question", "")
            arithmetic = re.fullmatch(r"(0|[1-9][0-9]*)([+-])(0|[1-9][0-9]*)=", question)
            copied = re.fullmatch(r"copy ([0-9]+)", question)
            if arithmetic:
                left, operation, right = arithmetic.groups()
                first, second = int(left), int(right)
                family = _arithmetic_family(first, second)
                answer = str(first + second if operation == "+" else first - second)
                task = "arithmetic"
                if split in ("dev_copy", "eval_regression"):
                    raise ValueError("arithmetic record found in copy-only split")
                if split == "eval_transfer" and not (min(first, second) <= 9 and max(first, second) in (10, 11, 12)):
                    raise ValueError("transfer record is not an unseen-operand family")
                if split != "eval_transfer" and not (0 <= first <= 9 and 0 <= second <= 9):
                    raise ValueError("domain arithmetic operand outside 0..9")
            elif copied:
                value = int(copied.group(1))
                if not 0 <= value < 100:
                    raise ValueError("copy value outside frozen 0..99 domain")
                family, answer, task = f"copy-value:{value}", str(value), "copy"
                if split not in ("train", "dev_copy", "eval_regression"):
                    raise ValueError("copy record found in arithmetic-only split")
            else:
                raise ValueError("unknown quality task prompt")
            if record.get("task") != task or record.get("family_id") != family or record.get("answer") != answer:
                raise ValueError("task/family/answer differs from independent prompt semantics")
            if family in owner and owner[family] != split:
                raise ValueError("canonical numeric family crosses splits")
            owner[family] = split
            families[split].add(family)
            if split == "train":
                labels[task].add(answer)
    if labels["arithmetic"] != ARITHMETIC_LABELS:
        raise ValueError("train arithmetic labels do not cover the domain -9..18")
    # Held-out copy values cannot appear as labels in the copy training task.
    # Digits, minus, EOS and all two-digit output positions are checked separately.
    train_copy_digits = set("".join(labels["copy"]))
    if train_copy_digits != set("0123456789") or not any(len(value) == 2 for value in labels["copy"]):
        raise ValueError("train copy labels do not cover digits and two-digit outputs")
    for split in ("dev", "dev_copy", "eval_main", "eval_regression"):
        if len(data[split]) < MIN_PANEL_SIZE:
            raise ValueError(f"quality panel {split} must have at least {MIN_PANEL_SIZE} samples")
    return {"split_counts": {name: len(data[name]) for name in SPLITS},
            "family_counts": {name: len(families[name]) for name in SPLITS},
            "train_arithmetic_labels": sorted(labels["arithmetic"], key=int),
            "train_copy_label_count": len(labels["copy"]), "train_copy_digits": sorted(train_copy_digits),
            "cross_split_family_overlap": False, "passed": True}


def protocol_manifest(data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {"version": PROTOCOL_VERSION, "data_sha256": _digest(data), "splits": data,
            "validation": validate_quality_data(data),
            "split_rule": {"arithmetic_domain": "all ordered + and - pairs in 0..9",
                           "train_coverage_anchors": "canonical pairs with lower operand 0 or upper operand 9",
                           "remaining_arithmetic": "SHA-256 rank: first 10 dev, next 10 eval_main, remaining train",
                           "copy_domain": "values 0..99; canonical, one-zero, two-zero spellings share value family",
                           "copy": "SHA-256 rank: first 70 train, next 15 dev_copy, last 15 eval_regression",
                           "rank_salt": PROTOCOL_VERSION, "seed_influences_data": False},
            "selection": {"allowed_panels": ["dev", "dev_copy"],
                          "order": ["both panels format 100% and dev_copy accuracy >=90%", "dev arithmetic accuracy",
                                    "dev copy accuracy", "lower response NLL", "earlier step"],
                          "final_panels_not_used_for_selection": True},
            "acceptance": {"scope": "in-domain arithmetic and copy normalization baseline",
                           "eval_main_accuracy_min": MAIN_THRESHOLD,
                           "eval_regression_accuracy_min": COPY_THRESHOLD,
                           "format_valid_fraction_min": 1.0, "minimum_panel_samples": MIN_PANEL_SIZE,
                           "eval_transfer_is_acceptance_condition": False,
                           "copy_retention_is_measured_at_common_sft_origin": True}}


def _counts(panel: dict[str, Any]) -> tuple[int, int, int]:
    values = tuple(panel.get(key) for key in ("correct", "total", "format_valid"))
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("panel counts must be integers")
    correct, total, valid = values
    if total <= 0 or not 0 <= correct <= valid <= total:
        raise ValueError("inconsistent panel counts")
    return correct, total, valid


def selection_key(dev_metrics: dict[str, dict[str, Any]], *, nll: float, step: int) -> tuple[bool, float, float, float, int]:
    """Rank dev checkpoints only; any eval key is a hard error.

    Accuracy is recomputed from integer counts. The caller must only replace the
    incumbent on a strictly greater key, and freeze its checkpoint hash before
    running any final panel. NLL is response-token loss on dev records only.
    """
    if set(dev_metrics) != {"dev", "dev_copy"}:
        raise ValueError("selection accepts exactly dev and dev_copy metrics; eval metrics are forbidden")
    if isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0:
        raise ValueError("dev response NLL must be finite and nonnegative")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("selection step must be a nonnegative integer")
    rates, formats = [], []
    for name in ("dev", "dev_copy"):
        correct, total, valid = _counts(dev_metrics[name])
        if total < MIN_PANEL_SIZE:
            raise ValueError("development panel too small")
        rates.append(correct / total)
        formats.append(valid == total)
    copy_correct, copy_total, _ = _counts(dev_metrics["dev_copy"])
    format_and_copy_qualified = all(formats) and copy_correct * 10 >= copy_total * 9
    return format_and_copy_qualified, rates[0], rates[1], -float(nll), -step


def development_gate(dev_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Same numerical thresholds on development panels, without opening test."""
    if set(dev_metrics) != {"dev", "dev_copy"}:
        raise ValueError("development gate accepts exactly dev and dev_copy metrics")
    result = quality_gate({"eval_main": dev_metrics["dev"], "eval_regression": dev_metrics["dev_copy"]})
    for key in ("conditions", "panels"):
        result[key] = {name.replace("eval_main", "dev").replace("eval_regression", "dev_copy"): value
                       for name, value in result[key].items()}
    result["failed_conditions"] = [name for name, passed in result["conditions"].items() if not passed]
    result["scope"] = "development qualification only; final evaluation remains sealed"
    return result


def quality_gate(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Report a frozen final gate; never use its values as a selection key.

    Accepts either an evaluation with ``panels`` or a panels mapping. All
    comparisons use exact count products, including the 80% and 90% boundaries.
    """
    panels = evaluation.get("panels", evaluation)
    conditions, inspected = {}, {}
    for name, numerator, denominator in (("eval_main", 4, 5), ("eval_regression", 9, 10)):
        if name not in panels:
            raise ValueError(f"missing final acceptance panel: {name}")
        correct, total, valid = _counts(panels[name])
        conditions[f"{name}_sample_count"] = total >= MIN_PANEL_SIZE
        conditions[f"{name}_accuracy"] = correct * denominator >= total * numerator
        conditions[f"{name}_format"] = valid == total
        inspected[name] = {"correct": correct, "total": total, "accuracy": correct / total,
                           "format_valid": valid, "accuracy_threshold": numerator / denominator}
    return {"protocol_version": PROTOCOL_VERSION, "qualified": all(conditions.values()),
            "scope": "in-domain arithmetic and copy normalization baseline", "conditions": conditions,
            "panels": inspected, "transfer_required": False,
            "failed_conditions": [name for name, passed in conditions.items() if not passed]}
