"""Independent standard-library evaluation, blind review and portable evidence.

Recomputes metrics from real prediction records. Family bootstrap treats complete
problem families as clusters; generation repeats never count as training seeds.
The blind worksheet is blank and cannot be described as a completed human review.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
import statistics
import sys
from typing import Any

SCHEMA = "maedaily-reference-report-v1"
PACKAGE_SCHEMA = "maedaily-reference-evidence-v1"
ROOT = Path(__file__).resolve().parents[2]
TEXT_FIELDS = ("sample_id", "family_id", "panel", "bucket", "arm", "training_run_id", "checkpoint_id",
               "output_text", "termination_reason")
IDENTITY_FIELDS = ("family_id", "bucket", "prompt_text", "eval_protocol_hash", "max_new_tokens")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"duplicate JSON key: {name}")
        value[name] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _strict_json(text: str) -> Any:
    value = json.loads(text, object_pairs_hook=_strict_object, parse_constant=_reject_constant)
    def finite(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON number")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = _strict_json(line)
        except (ValueError, RecursionError) as error:
            raise ValueError(f"{path.name}:{line_number}: {error}") from error
        if not isinstance(item, dict):
            raise ValueError(f"{path.name}:{line_number}: record must be an object")
        records.append(item)
    return records


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _confidence(value: float) -> float:
    value = _number(value, "confidence")
    if not 0 < value < 1:
        raise ValueError("confidence must be between zero and one")
    return value


def wilson_interval(correct: int, total: int, confidence: float = 0.95) -> dict[str, Any]:
    """Wilson score interval; no normal approximation p +/- error clipping."""
    correct, total = _integer(correct, "correct"), _integer(total, "total")
    confidence = _confidence(confidence)
    if correct > total:
        raise ValueError("correct cannot exceed total")
    if not total:
        return {"correct": correct, "total": total, "estimate": None, "low": None, "high": None,
                "confidence": confidence, "method": "Wilson score", "status": "not_measured"}
    z = statistics.NormalDist().inv_cdf((1 + confidence) / 2)
    p, z2 = correct / total, z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator
    return {"correct": correct, "total": total, "estimate": p, "low": max(0.0, center - radius),
            "high": min(1.0, center + radius), "confidence": confidence, "method": "Wilson score",
            "status": "computed", "assumption": "independent Bernoulli items; descriptive if families/repeats correlate"}


def _identity(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["panel"], row["sample_id"], row.get("generation_index", 0)


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("prediction records must be nonempty")
    seen, run_seeds, origin = set(), {}, {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("prediction record must be an object")
        for name in TEXT_FIELDS:
            if not isinstance(row.get(name), str) or (name != "output_text" and not row[name]):
                raise ValueError(f"{name} must be a nonempty string")
        _integer(row.get("training_seed"), "training_seed")
        _integer(row.get("generation_index", 0), "generation_index")
        for name in ("input_tokens", "output_tokens"):
            _integer(row.get(name), name)
        for name, selected in (("evaluation_cost_input_tokens", "input_tokens"), ("evaluation_cost_tokens", "output_tokens")):
            if name in row:
                _integer(row[name], name, minimum=row[selected])
        _number(row.get("elapsed_seconds"), "elapsed_seconds")
        for name in ("correct", "format_valid"):
            if not isinstance(row.get(name), bool):
                raise ValueError(f"{name} must be boolean")
        if row["correct"] and not row["format_valid"]:
            raise ValueError("correct answer cannot have invalid registered format")
        if "prompt_text" in row and not isinstance(row["prompt_text"], str):
            raise ValueError("prompt_text must be a string")
        key = (row["arm"], row["training_seed"], *_identity(row))
        if key in seen:
            raise ValueError("duplicate prediction identity; generation repeats need distinct generation_index")
        seen.add(key)
        run_id, seed = row["training_run_id"], row["training_seed"]
        if run_id in run_seeds and run_seeds[run_id] != seed:
            raise ValueError("same actual training run cannot be relabeled as multiple training seeds")
        run_seeds[run_id] = seed
        origin_key = (row["arm"], seed)
        actual = (run_id, row["checkpoint_id"])
        if origin_key in origin and origin[origin_key] != actual:
            raise ValueError("one arm/training seed must identify one frozen run and checkpoint")
        origin[origin_key] = actual


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_family_bootstrap(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], *,
                            seed: int = 19, replicates: int = 2000, confidence: float = 0.95) -> dict[str, Any]:
    """Resample complete paired families, stratified by panel, never split variants."""
    validate_records(baseline)
    validate_records(candidate)
    seed, replicates = _integer(seed, "bootstrap seed"), _integer(replicates, "bootstrap replicates", minimum=1)
    confidence = _confidence(confidence)
    def index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
        result = {}
        origins = {(row["arm"], row["training_seed"], row["training_run_id"], row["checkpoint_id"]) for row in rows}
        if len(origins) != 1:
            raise ValueError("paired bootstrap requires one arm/actual training seed/checkpoint on each side")
        for row in rows:
            identity = _identity(row)
            if identity in result:
                raise ValueError("duplicate paired prediction identity")
            result[identity] = row
        return result
    left, right = index(baseline), index(candidate)
    if left.keys() != right.keys():
        raise ValueError("paired identity mismatch: missing or extra sample/generation/panel")
    if baseline[0]["training_seed"] != candidate[0]["training_seed"]:
        raise ValueError("paired methods must use the same registered training seed")
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    improved, regressed = [], []
    for identity in sorted(left):
        first, second = left[identity], right[identity]
        for name in IDENTITY_FIELDS:
            if first.get(name) != second.get(name):
                raise ValueError(f"paired identity/protocol mismatch: {name}")
        delta = int(second["correct"]) - int(first["correct"])
        grouped[first["panel"]][first["family_id"]].append(delta)
        if delta > 0:
            improved.append(list(identity))
        elif delta < 0:
            regressed.append(list(identity))
    strata = [[(sum(values), len(values)) for _, values in sorted(families.items())]
              for _, families in sorted(grouped.items())]
    rng, draws = random.Random(seed), []
    for _ in range(replicates):
        numerator, denominator = 0, 0
        for groups in strata:
            for _ in groups:
                count, size = groups[rng.randrange(len(groups))]
                numerator += count
                denominator += size
        draws.append(numerator / denominator)
    family_counts = {name: len(families) for name, families in sorted(grouped.items())}
    return {"difference": (len(improved) - len(regressed)) / len(left), "low": _quantile(draws, (1-confidence)/2),
            "high": _quantile(draws, (1+confidence)/2), "confidence": confidence,
            "method": "paired family-cluster percentile bootstrap stratified by panel",
            "bootstrap_seed": seed, "replicates": replicates, "paired_records": len(left),
            "family_counts": family_counts, "improved_ids": improved, "regressed_ids": regressed,
            "unchanged": len(left) - len(improved) - len(regressed),
            "unit": "complete problem family including its generation repeats",
            "limitation": "one-family strata have no between-family uncertainty" if any(n < 2 for n in family_counts.values())
                          else "conditional on these trained models; training-seed variation is reported separately"}


def _metrics(rows: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    total, correct = len(rows), sum(row["correct"] for row in rows)
    valid, lengths = sum(row["format_valid"] for row in rows), [row["output_tokens"] for row in rows]
    elapsed = sum(row["elapsed_seconds"] for row in rows)
    selected_inputs, selected_outputs = sum(row["input_tokens"] for row in rows), sum(lengths)
    inputs = sum(row.get("evaluation_cost_input_tokens", row["input_tokens"]) for row in rows)
    outputs = sum(row.get("evaluation_cost_tokens", row["output_tokens"]) for row in rows)
    length = {"mean_output_tokens": statistics.fmean(lengths), "minimum": min(lengths), "maximum": max(lengths),
              "scope": "selected response length per evaluated sample; candidate sampling cost is separate"}
    return {"correct": correct, "total": total, "accuracy": correct / total,
            "accuracy_interval": wilson_interval(correct, total, confidence),
            "format_valid": valid, "format_rate": valid / total,
            "format_interval": wilson_interval(valid, total, confidence),
            "family_count": len({(row["panel"], row["family_id"]) for row in rows}),
            "sample_count": len({(row["panel"], row["sample_id"]) for row in rows}),
            "generation_records": total, "termination_counts": dict(sorted(_counts(row["termination_reason"] for row in rows).items())),
            "length": length, "result_length": dict(length),
            "evaluation_generation_scope": "all candidate generation costs when declared; one selected result per prediction identity",
            "cost": {"input_tokens": inputs, "output_tokens": outputs, "total_tokens": inputs + outputs,
                     "selected_prompt_input_tokens": selected_inputs, "selected_response_output_tokens": selected_outputs,
                     "summed_generation_seconds": elapsed, "output_tokens_per_summed_second": outputs / elapsed if elapsed else None,
                     "scope": "all declared evaluation candidates and measured generation times; not training wall time, concurrent throughput or billed cost"}}


def _counts(values: Any) -> dict[Any, int]:
    result: dict[Any, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _training_provenance(records: list[dict[str, Any]], receipts: list[dict[str, Any]] | None) -> dict[str, Any]:
    registered = sorted({row["training_seed"] for row in records})
    if receipts is None:
        return {"status": "unverified", "registered_training_seeds": registered, "verified_training_seeds": [],
                "independent_training_seeds_verified": False, "reason": "actual update receipts were not supplied",
                "generation_repeats_are_training_seeds": False}
    by_identity, run_seed = {}, {}
    for receipt in receipts:
        required = ("training_run_id", "arm", "checkpoint_id", "initial_weight_hash", "final_weight_hash")
        if any(not isinstance(receipt.get(name), str) or not receipt[name] for name in required):
            raise ValueError("training receipt is missing an identity or weight hash")
        seed = _integer(receipt.get("training_seed"), "receipt training_seed")
        steps = _integer(receipt.get("optimizer_steps"), "receipt optimizer_steps")
        for name in ("initial_weight_hash", "final_weight_hash"):
            value = receipt[name]
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("training receipt weight hashes must be lowercase SHA-256")
        if steps > 0 and receipt["initial_weight_hash"] == receipt["final_weight_hash"]:
            raise ValueError("training receipt claims updates but weights did not change")
        if steps == 0 and receipt["initial_weight_hash"] != receipt["final_weight_hash"]:
            raise ValueError("zero-update receipt cannot claim a weight change")
        run_id = receipt["training_run_id"]
        if run_id in run_seed and run_seed[run_id] != seed:
            raise ValueError("one actual training run cannot stand for several independent seeds")
        run_seed[run_id] = seed
        identity = (receipt["arm"], seed, run_id, receipt["checkpoint_id"])
        if identity in by_identity:
            raise ValueError("duplicate training receipt identity")
        by_identity[identity] = receipt
    used = {(row["arm"], row["training_seed"], row["training_run_id"], row["checkpoint_id"]) for row in records}
    if used != by_identity.keys():
        raise ValueError("prediction/training receipt identities mismatch; missing or extra actual run")
    trained = sorted({receipt["training_seed"] for receipt in receipts if receipt["optimizer_steps"] > 0})
    return {"status": "receipts_checked", "registered_training_seeds": registered, "verified_training_seeds": trained,
            "independent_training_seeds_verified": bool(trained), "verified_training_seed_count": len(trained),
            "generation_repeats_are_training_seeds": False, "actual_training_run_count": len({r["training_run_id"] for r in receipts}),
            "verification_scope": "identity/update/weight-change receipts; package hashes must separately verify saved artifacts",
            "receipts": receipts}


def build_report(records: list[dict[str, Any]], *, training_receipts: list[dict[str, Any]] | None = None,
                 baseline_arm: str = "no_update", bootstrap_seed: int = 19, bootstrap_replicates: int = 2000,
                 confidence: float = 0.95) -> dict[str, Any]:
    validate_records(records)
    example_count = sum(row.get("example_only") is True for row in records)
    if example_count and example_count != len(records):
        raise ValueError("synthetic examples cannot be mixed with actual prediction records")
    confidence = _confidence(confidence)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["arm"], row["training_seed"])].append(row)
    arms = sorted({arm for arm, _ in grouped})
    seed_sets = {arm: {seed for actual_arm, seed in grouped if actual_arm == arm} for arm in arms}
    if len({tuple(sorted(seeds)) for seeds in seed_sets.values()}) != 1:
        raise ValueError("arms evaluated different training seed sets; missing seeds cannot be dropped")
    for arm in arms:
        reference = None
        for seed in sorted(seed_sets[arm]):
            identities = {_identity(row): tuple(row.get(name) for name in IDENTITY_FIELDS)
                          for row in grouped[(arm, seed)]}
            if reference is not None and identities != reference:
                raise ValueError("training seeds evaluated different frozen identities/protocols")
            reference = identities
    runs, pairs = [], []
    for (arm, seed), rows in sorted(grouped.items()):
        panel_metrics = {name: _metrics([row for row in rows if row["panel"] == name], confidence)
                         for name in sorted({row["panel"] for row in rows})}
        buckets = {f"{panel}/{bucket}": _metrics([row for row in rows if row["panel"] == panel and row["bucket"] == bucket], confidence)
                   for panel, bucket in sorted({(row["panel"], row["bucket"]) for row in rows})}
        runs.append({"arm": arm, "training_seed": seed, "training_run_id": rows[0]["training_run_id"],
                     "checkpoint_id": rows[0]["checkpoint_id"], "panels": panel_metrics, "buckets": buckets})
        if arm != baseline_arm:
            if (baseline_arm, seed) not in grouped:
                raise ValueError("candidate is missing its same-training-seed baseline")
            baseline = grouped[(baseline_arm, seed)]
            # Validate the full identity set before splitting into panel reports.
            full = paired_family_bootstrap(baseline, rows, seed=bootstrap_seed, replicates=bootstrap_replicates, confidence=confidence)
            panel_pairs = {name: paired_family_bootstrap([row for row in baseline if row["panel"] == name],
                                                        [row for row in rows if row["panel"] == name],
                                                        seed=bootstrap_seed, replicates=bootstrap_replicates, confidence=confidence)
                           for name in sorted({row["panel"] for row in rows})}
            pairs.append({"baseline_arm": baseline_arm, "candidate_arm": arm, "training_seed": seed,
                          "all_panels_descriptive": full, "panels": panel_pairs})
    seed_summary = []
    for arm in sorted({row["arm"] for row in records}):
        arm_runs = [run for run in runs if run["arm"] == arm]
        panels = {}
        names = {name for run in arm_runs for name in run["panels"]}
        if any(set(run["panels"]) != names for run in arm_runs):
            raise ValueError("training seeds evaluated different panel sets")
        for name in sorted(names):
            rates = [run["panels"][name]["accuracy"] for run in arm_runs]
            panels[name] = {"mean_accuracy_across_training_seeds": statistics.fmean(rates),
                            "sample_std_across_training_seeds": statistics.stdev(rates) if len(rates) > 1 else None,
                            "minimum": min(rates), "maximum": max(rates), "training_seed_count": len(rates),
                            "per_seed": {str(run["training_seed"]): run["panels"][name]["accuracy"] for run in arm_runs}}
        seed_summary.append({"arm": arm, "training_seeds": [run["training_seed"] for run in arm_runs], "panels": panels,
                             "uncertainty_scope": "descriptive training-seed spread; family bootstrap is conditional per trained pair"})
    provenance = _training_provenance(records, training_receipts)
    if example_count:
        provenance.update(synthetic_example=True, independent_training_seeds_verified=False,
                          verification_scope="synthetic receipt identities only; no real model training is proven")
    return {"schema_version": SCHEMA, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_scope": "synthetic_example" if example_count else "supplied_prediction_records",
            "baseline_arm": baseline_arm, "records": len(records), "training_provenance": provenance,
            "runs": runs, "paired_comparisons": pairs, "multi_seed": seed_summary,
            "review_status": "not_reviewed; a blank worksheet is not completed human/self/AI review",
            "statistical_limits": ["Wilson intervals assume independent items; related families/repeats violate that assumption.",
                                   "Paired bootstrap resamples whole families, including all variants and generations.",
                                   "Three training seeds describe repeat variation and do not establish equivalence or significance.",
                                   "Seed and family uncertainty are not combined into one confidence interval here."]}


def single_factor_difference(base: dict[str, Any], variant: dict[str, Any], *,
                             expected_path: str | None = None) -> dict[str, Any]:
    """Require exactly one changed leaf, with no implicit ignored differences.

    Lists (including seeds/checkpoints) are one atomic config value. Runtime
    output paths and timestamps should be kept outside scientific configs, not
    silently removed by this verifier.
    """
    def walk(left: Any, right: Any, path: str) -> list[dict[str, Any]]:
        if isinstance(left, dict) and isinstance(right, dict):
            result = []
            for key in sorted(left.keys() | right.keys()):
                child = f"{path}.{key}" if path else key
                if key not in left or key not in right:
                    result.append({"path": child, "before": left.get(key), "after": right.get(key),
                                   "before_present": key in left, "after_present": key in right})
                else:
                    result.extend(walk(left[key], right[key], child))
            return result
        if type(left) is not type(right) or left != right:
            return [{"path": path, "before": left, "after": right, "before_present": True, "after_present": True}]
        return []
    if not isinstance(base, dict) or not isinstance(variant, dict):
        raise ValueError("scientific configs must be objects")
    # Serialization validates JSON types and rejects nonfinite config values.
    for config in (base, variant):
        _strict_json(json.dumps(config, allow_nan=False))
    differences = walk(base, variant, "")
    if len(differences) != 1:
        raise ValueError(f"single-factor ablation needs exactly one changed leaf, found {len(differences)}")
    difference = differences[0]
    if expected_path is not None and difference["path"] != expected_path:
        raise ValueError(f"unexpected ablation factor: {difference['path']}")
    return {"passed": True, "changed_factor": difference,
            "base_config_sha256": hashlib.sha256(_json_bytes(base)).hexdigest(),
            "variant_config_sha256": hashlib.sha256(_json_bytes(variant)).hexdigest()}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(_json_bytes(record).decode("utf-8") + "\n")


def blind_review_packet(records: list[dict[str, Any]], *, seed: int = 31) -> dict[str, Any]:
    """Randomize anonymous outputs; return the identity map separately.

    This is preparation for human, self or AI-assisted review. It performs no
    review and never labels its blank worksheet as reviewed or third-party.
    """
    validate_records(records)
    seed = _integer(seed, "review seed")
    ordered = sorted(records, key=lambda row: (row["arm"], row["training_seed"], *_identity(row)))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    packet, blank, mapping = [], [], {}
    for index, row in enumerate(ordered):
        if not isinstance(row.get("prompt_text"), str) or not row["prompt_text"]:
            raise ValueError("blind review needs actual prompt_text, not just sample hashes")
        review_id = f"review-{index + 1:06d}"
        packet.append({"review_id": review_id, "panel": row["panel"], "bucket": row["bucket"],
                       "prompt_text": row["prompt_text"], "output_text": row["output_text"],
                       "termination_reason": row["termination_reason"]})
        blank.append({"review_id": review_id, "status": "not_reviewed", "reviewer_id": None,
                      "reviewer_type": None, "correct": None, "format_valid": None, "reason": None})
        mapping[review_id] = {name: row[name] for name in ("arm", "training_seed", "training_run_id", "checkpoint_id",
                                                        "sample_id", "family_id", "correct", "format_valid")}
        mapping[review_id]["generation_index"] = row.get("generation_index", 0)
    return {"protocol": {"status": "not_reviewed", "review_seed": seed, "count": len(packet),
                         "identity_map_storage": "separate sealed file outside the blind packet/evidence directory",
                         "allowed_reviewer_types": ["human", "self", "ai-assisted"],
                         "third_party_human_review_performed": False,
                         "rubric": "check answer and registered format from actual prompt/output; record boundary/disagreement reasons",
                         "scope": "blank preparation; not completed manual, self or AI review"},
            "records": packet, "blank_reviews": blank, "identity_map": mapping}


def review_summary(packet: dict[str, Any], reviews: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {row["review_id"] for row in packet["records"]}
    seen, reviewed, types = set(), 0, defaultdict(int)
    disagreements = []
    for review in reviews:
        identity = review.get("review_id")
        if identity not in expected or identity in seen:
            raise ValueError("unknown or duplicate review identity")
        seen.add(identity)
        if review.get("status") == "not_reviewed":
            if any(review.get(name) is not None for name in ("correct", "format_valid", "reviewer_type", "reviewer_id", "reason")):
                raise ValueError("unreviewed worksheet cannot contain fabricated scores or reviewer")
            continue
        if review.get("status") != "reviewed" or review.get("reviewer_type") not in ("human", "self", "ai-assisted"):
            raise ValueError("review needs an explicit actual reviewer type and status")
        if not isinstance(review.get("reviewer_id"), str) or not review["reviewer_id"] or not isinstance(review.get("reason"), str) or not review["reason"]:
            raise ValueError("completed review needs reviewer identity and reason")
        if any(not isinstance(review.get(name), bool) for name in ("correct", "format_valid")):
            raise ValueError("completed review requires boolean scores")
        reviewed += 1
        types[review["reviewer_type"]] += 1
        automatic = packet["identity_map"][identity]
        if any(review[name] != automatic[name] for name in ("correct", "format_valid")):
            disagreements.append({"review_id": identity, "automatic": {name: automatic[name] for name in ("correct", "format_valid")},
                                  "review": {name: review[name] for name in ("correct", "format_valid")}, "reason": review["reason"]})
    return {"status": "not_reviewed" if reviewed == 0 else "complete" if reviewed == len(expected) else "partial",
            "expected": len(expected), "submitted": len(seen), "reviewed": reviewed, "by_type": dict(types),
            "disagreements": disagreements,
            "third_party_human_review_verified": False,
            "limitation": "reviewer labels are recorded; this function cannot verify human identity or external independence"}


def _file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("evidence member needs a safe relative POSIX path")
    parsed = PurePosixPath(name)
    if not parsed.parts or parsed.is_absolute() or any(part in (".", "..") or ":" in part for part in parsed.parts) or parsed.as_posix() != name:
        raise ValueError("evidence member path cannot be absolute, traversal, alias or external reference")
    if name == "manifest.json":
        raise ValueError("manifest.json is reserved")
    return name


def create_evidence_package(files: dict[str, Path], destination: Path, *, metadata: dict[str, Any] | None = None,
                            required_files: list[str] | None = None, max_file_bytes: int = 8 * 1024 * 1024,
                            max_total_bytes: int = 32 * 1024 * 1024) -> dict[str, Any]:
    """Copy actual files into a self-contained, bounded, immutable new package."""
    if not files:
        raise ValueError("evidence package requires actual files")
    max_file_bytes, max_total_bytes = _integer(max_file_bytes, "max_file_bytes", minimum=1), _integer(max_total_bytes, "max_total_bytes", minimum=1)
    names = [_member_name(name) for name in files]
    required = names if required_files is None else [_member_name(name) for name in required_files]
    if len(required) != len(set(required)) or set(required) - set(names):
        raise ValueError("required evidence is missing; an ignored/original path is not a packaged artifact")
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict):
        raise ValueError("evidence metadata must be an object")
    _strict_json(json.dumps(metadata, allow_nan=False))
    if metadata.get("external_required_artifacts"):
        raise ValueError("portable evidence cannot depend on external or ignored required artifacts")
    sources, total = {}, 0
    for name, source in files.items():
        source = Path(source)
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"missing or symlink evidence source: {name}")
        size = source.stat().st_size
        if size > max_file_bytes:
            raise ValueError(f"evidence source exceeds compact file budget: {name}")
        total += size
        sources[name] = (source, size, _file_hash(source))
    if total > max_total_bytes:
        raise ValueError("evidence package exceeds compact total budget")
    destination = Path(destination)
    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise ValueError("evidence destination must be new or empty; existing evidence is immutable")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": PACKAGE_SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata, "required_files": sorted(required), "members": {}}
    for name, (source, size, expected) in sorted(sources.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as source_stream, target.open("xb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream)
        if _file_hash(target) != expected or _file_hash(source) != expected:
            raise ValueError("evidence source changed during packaging")
        manifest["members"][name] = {"bytes": size, "sha256": expected}
    _write_json(destination / "manifest.json", manifest)
    verify_evidence_package(destination)
    return manifest


def verify_evidence_package(directory: Path, *, required_files: list[str] | None = None) -> dict[str, Any]:
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("evidence directory is missing or a symlink")
    manifest = _strict_json((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PACKAGE_SCHEMA or not isinstance(manifest.get("members"), dict) or not manifest["members"]:
        raise ValueError("unsupported or empty evidence manifest")
    if not isinstance(manifest.get("metadata", {}), dict):
        raise ValueError("evidence metadata must be an object")
    if manifest.get("metadata", {}).get("external_required_artifacts"):
        raise ValueError("evidence refers to external/ignored required artifacts")
    names = {_member_name(name) for name in manifest["members"]}
    required = manifest.get("required_files", [])
    if not isinstance(required, list) or len(required) != len(set(required)):
        raise ValueError("invalid required evidence list")
    required = {_member_name(name) for name in required}
    if required_files is not None:
        required.update(_member_name(name) for name in required_files)
    if required - names:
        raise ValueError("required artifact is absent from portable evidence")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink is not allowed in evidence package")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != names | {"manifest.json"}:
        raise ValueError("evidence files differ from manifest: missing or undeclared file")
    for name in sorted(names):
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("evidence path escapes package")
        member = manifest["members"][name]
        if not isinstance(member, dict):
            raise ValueError("evidence member must be an object")
        _integer(member.get("bytes"), "evidence member bytes")
        expected_hash = member.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
            raise ValueError("evidence member needs a lowercase SHA-256")
        if path.stat().st_size != member["bytes"] or _file_hash(path) != expected_hash:
            raise ValueError(f"evidence size/hash mismatch: {name}")
    return {"passed": True, "schema_version": PACKAGE_SCHEMA, "files": len(names),
            "total_bytes": sum(manifest["members"][name]["bytes"] for name in names), "required_files": sorted(required),
            "scope": "packaged actual files verified; training/scientific claims require their stage evidence"}


def render_report(report: dict[str, Any]) -> str:
    lines = ["# 参考实训独立评测报告", "", f"记录数：{report['records']}；来源范围：`{report['source_scope']}`；训练来源：`{report['training_provenance']['status']}`。",
             "", "| 支路 | 训练种子 | 面板 | 正确 | 格式有效 | 平均输出 token |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    for run in report["runs"]:
        for name, metrics in run["panels"].items():
            lines.append(f"| {run['arm']} | {run['training_seed']} | {name} | {metrics['correct']}/{metrics['total']} | "
                         f"{metrics['format_valid']}/{metrics['total']} | {metrics['length']['mean_output_tokens']:.3f} |")
    lines += ["", "区间、整题族配对、能力桶、长度与实际 token/耗时见 report.json。Wilson 区间的独立题目假设不适用于相关题族，"
              "配对 bootstrap 按完整 family 重采样；训练种子的离散度另行报告，不把生成重复当独立训练。",
              "", "盲评表当前为空，状态 not_reviewed。self、ai-assisted 和 human 必须标明真实来源，空表不能证明人工已复核。",
              "", "证据包只验证实际打包文件；训练完成、效果提升和学习者独立掌握仍需对应的阶段与研究证据。"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--records", nargs="+", type=Path)
    parser.add_argument("--training-receipts", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline-arm", default="no_update")
    parser.add_argument("--bootstrap-seed", type=int, default=19)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--review-seed", type=int, default=31)
    parser.add_argument("--verify-package", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_package is not None:
            if args.records or args.training_receipts or args.output_dir:
                raise ValueError("verify-package cannot be combined with report inputs/output")
            print(json.dumps(verify_evidence_package(args.verify_package), ensure_ascii=False, indent=2))
            return 0
        if not args.records:
            raise ValueError("--records is required when not verifying a package")
        records = [row for path in args.records for row in read_jsonl(path)]
        receipts = None if args.training_receipts is None else _strict_json(args.training_receipts.read_text(encoding="utf-8"))
        if isinstance(receipts, dict):
            receipts = receipts.get("training_receipts")
        if receipts is not None and not isinstance(receipts, list):
            raise ValueError("training receipts must be a list or training_receipts object")
        report = build_report(records, training_receipts=receipts, baseline_arm=args.baseline_arm,
                              bootstrap_seed=args.bootstrap_seed, bootstrap_replicates=args.bootstrap_replicates)
        packet = blind_review_packet(records, seed=args.review_seed)
        sys.path.insert(0, str(ROOT / "scripts"))
        from lesson_runtime import reserve_output_dir
        output = reserve_output_dir(args.output_dir, "reference-report")
        _write_jsonl(output / "records.jsonl", records)
        _write_json(output / "report.json", report)
        with (output / "report.md").open("x", encoding="utf-8") as stream:
            stream.write(render_report(report))
        review_dir = output / "blind_review"
        review_dir.mkdir()
        _write_json(review_dir / "protocol.json", packet["protocol"])
        _write_jsonl(review_dir / "records.jsonl", packet["records"])
        _write_jsonl(review_dir / "blank_reviews.jsonl", packet["blank_reviews"])
        sealed = output.parent / f"{output.name}_sealed_identity_map.json"
        _write_json(sealed, packet["identity_map"])
        members = {path.relative_to(output).as_posix(): path for path in output.rglob("*") if path.is_file()
                   and path.name != ".maedaily-output.json"}
        members["post_train/experiments/reference_reporting.py"] = Path(__file__)
        members["scripts/lesson_runtime.py"] = ROOT / "scripts/lesson_runtime.py"
        if receipts is not None:
            _write_json(output / "training_receipts.json", receipts)
            members["training_receipts.json"] = output / "training_receipts.json"
        command = ["python", "post_train/experiments/reference_reporting.py", "--records", "records.jsonl",
                   "--output-dir", "../recomputed_report", "--baseline-arm", args.baseline_arm,
                   "--bootstrap-seed", str(args.bootstrap_seed), "--bootstrap-replicates", str(args.bootstrap_replicates),
                   "--review-seed", str(args.review_seed)]
        if receipts is not None:
            command.extend(["--training-receipts", "training_receipts.json"])
        create_evidence_package(members, output / "evidence", metadata={"reproduction_scope": "evaluation metrics and blank review preparation",
                                "recompute_command_argv": command, "identity_map_included": False,
                                "source_scope": report["source_scope"], "does_not_claim_training_complete": True})
        print(json.dumps({"status": "report_generated", "output_dir": str(output), "evidence": str(output / "evidence"),
                          "sealed_identity_map": str(sealed), "review_status": "not_reviewed"}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, TypeError, RecursionError) as error:
        print(f"reference reporting failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
