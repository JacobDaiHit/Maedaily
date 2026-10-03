"""Independently score, verify and export a completed frozen B/C reference run.

The arithmetic/format oracle below does not import the training verifier. Public
packages can be rescored using Python's standard library without ignored runs or
model weights. Repeating training additionally requires the pinned public model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import gzip
import importlib.util
import json
import math
import operator
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PANELS = ("eval_main", "eval_transfer", "eval_regression")
SPLITS = ("train", "dev", "dev_copy", *PANELS)
ARMS = ("base", "no_update", "dpo", "dpo_beta", "continue_sft", "rlvr",
        "rlvr_group_size", "rlvr_no_format", "sampling_only")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError("nonfinite JSON number: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=bad_constant)


def reporting(source_root):
    path = Path(source_root) / "post_train/experiments/reference_reporting.py"
    spec = importlib.util.spec_from_file_location("independent_reference_reporting", path)
    module = importlib.util.module_from_spec(spec)
    # Reading an immutable package must not add an undeclared __pycache__ file.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def oracle(question):
    """Return mathematical target, semantic family and bucket from raw question."""
    if not isinstance(question, str):
        raise ValueError("question must be text")
    match = re.fullmatch(r"(0|[1-9][0-9]*)([+-])(0|[1-9][0-9]*)=", question)
    if match:
        left, symbol, right = match.groups()
        left, right = int(left), int(right)
        operation = {"+": operator.add, "-": operator.sub}[symbol]
        return (str(operation(left, right)), f"operand-pair:{min(left,right)}:{max(left,right)}",
                "small-add" if symbol == "+" else "small-subtract")
    match = re.fullmatch(r"copy ([0-9]+)", question)
    if match:
        value = int(match.group(1))
        return str(value), f"copy-value:{value}", "copy"
    raise ValueError("unrecognized raw question: " + question)


def score(question, text, termination):
    """Canonical integer <=4 characters and EOS are necessary for correctness."""
    expected, _, _ = oracle(question)
    valid = False
    if isinstance(text, str) and 0 < len(text) <= 4 and termination == "eos":
        try:
            valid = str(int(text)) == text
        except ValueError:
            pass
    return {"expected_answer": expected, "format_valid": valid,
            "correct": valid and text == expected}


def validate_data(document, registration):
    data, manifest = document["data"], document["manifest"]
    if set(data) != set(SPLITS) or any(not data[name] for name in SPLITS):
        raise ValueError("exactly six nonempty frozen splits required")
    if manifest != registration["data"] or manifest.get("splits") != data:
        raise ValueError("registration/data manifest mismatch")
    if digest(data) != manifest.get("data_sha256"):
        raise ValueError("data content hash differs")
    lookup, questions, owners = {}, set(), {}
    for split, rows in data.items():
        for row in rows:
            answer, family, bucket = oracle(row["question"])
            sample = row["id"]
            if sample in lookup or row["question"] in questions:
                raise ValueError("duplicate frozen sample/question")
            if row.get("answer") != answer:
                raise ValueError("wrong frozen gold from independent raw-question oracle: " + sample)
            if row.get("split") != split or row.get("family_id") != family or row.get("family") != bucket:
                raise ValueError("frozen sample identity/family mismatch")
            if row.get("task") != ("copy" if bucket == "copy" else "arithmetic"):
                raise ValueError("frozen task mismatch")
            if family in owners and owners[family] != split:
                raise ValueError("semantic family leaks across splits")
            owners[family] = split
            lookup[sample] = row
            questions.add(row["question"])
    return lookup


def verify_predictions(records, lookup, config, registration, receipts):
    seeds = config["seeds"]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(type(x) is not int for x in seeds):
        raise ValueError("at least three unique actual training seeds required")
    if registration["arms"] != list(ARMS):
        raise ValueError("registered nine-arm reference matrix differs")
    identity = {}
    for item in receipts:
        key = item["training_seed"], item["arm"]
        if key in identity:
            raise ValueError("duplicate training receipt")
        identity[key] = item
    if set(identity) != {(seed, arm) for seed in seeds for arm in ARMS}:
        raise ValueError("missing or extra registered training receipt/seed/arm")
    protocol = digest({"data": registration["data"], "config": config, "arms": list(ARMS)})
    expected_ids = {key for key, value in lookup.items() if value["split"] in PANELS}
    required = {(seed, arm, sample) for seed in seeds for arm in ARMS for sample in expected_ids}
    seen = set()
    for row in records:
        key = row["training_seed"], row["arm"], row["sample_id"]
        if key not in required or key in seen or row.get("generation_index", 0) != 0:
            raise ValueError("duplicate, extra or unregistered prediction identity")
        seen.add(key)
        frozen = lookup[row["sample_id"]]
        receipt = identity[key[:2]]
        for field, value in (("question", frozen["question"]), ("expected_answer", frozen["answer"]),
                             ("family_id", frozen["family_id"]), ("bucket", frozen["family"]),
                             ("panel", frozen["split"]), ("eval_protocol_hash", protocol),
                             ("max_new_tokens", config["max_new_tokens"]),
                             ("training_run_id", f"{key[0]}:{key[1]}"),
                             ("checkpoint_id", receipt["checkpoint_id"])):
            if row.get(field) != value:
                raise ValueError("prediction frozen identity differs: " + field)
        result = score(row["question"], row["output_text"], row["termination_reason"])
        if any(type(row.get(field)) is not bool or row[field] != result[field]
               for field in ("correct", "format_valid")):
            raise ValueError("prediction scores differ from independent oracle")
    if seen != required:
        raise ValueError("missing prediction/seed/arm/frozen sample")
    for seed in seeds:
        base, sft, sampling = (identity[seed, arm] for arm in ("base", "no_update", "sampling_only"))
        for receipt in (base, sampling):
            if receipt["optimizer_steps"] != 0 or receipt["initial_weight_hash"] != receipt["final_weight_hash"]:
                raise ValueError("inference-only arm falsely claims real weight updates")
        if sampling.get("source_training_run_id") != f"{seed}:no_update" or sampling.get("source_checkpoint_id") != sft["checkpoint_id"] or sampling["checkpoint_id"] != sft["checkpoint_id"]:
            raise ValueError("sampling-only SFT origin differs")
    return {"prediction_count": len(records), "training_seeds": seeds,
            "arms": list(ARMS), "final_eval_protocol_hash": protocol}


def without_time(report):
    return {key: value for key, value in report.items() if key != "generated_at_utc"}


DERIVED_FLOAT_FIELDS = {"summed_generation_seconds", "output_tokens_per_summed_second",
                        "training_elapsed_seconds_logged", "elapsed_seconds",
                        "pre_update_ratio_min", "pre_update_ratio_max"}


def equivalent_derived(left, right, path=()):
    """Only named derived float fields tolerate Python summation/libm last bits.

    Integers, booleans, identities, metadata, configuration and source bytes stay
    exact. Python 3.12+ sums floats more accurately than 3.11; this is not a reason
    to replace frozen observed records or to accept changed scores.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(equivalent_derived(left[key], right[key], (*path, key)) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(equivalent_derived(a, b, (*path, index)) for index, (a, b) in enumerate(zip(left, right)))
    if isinstance(left, float):
        if not math.isfinite(left) or not math.isfinite(right):
            return False
        if path and path[-1] in DERIVED_FLOAT_FIELDS:
            return math.isclose(left, right, abs_tol=1e-12, rel_tol=1e-12)
    return left == right


def safe_relative(name):
    path = PurePosixPath(name)
    if not isinstance(name, str) or path.is_absolute() or ":" in name or "\\" in name or any(x in (".", "..") for x in path.parts) or str(path) != name:
        raise ValueError("unsafe artifact path")
    return path


def source_path(root, name):
    path = safe_relative(name)
    if not name.endswith(".py"):
        raise ValueError("registered source must be Python")
    return Path(root) / (name if len(path.parts) > 1 else "post_train/experiments/" + name)


def verify_raw_evaluations(directory, records, lookup, config):
    """Bind reported selected length and candidate cost to raw sampled outputs."""
    by_arm = {}
    for row in records:
        by_arm.setdefault((row["training_seed"], row["arm"]), {})[row["sample_id"]] = row
    for (seed, arm), expected in by_arm.items():
        evaluation = read_json(Path(directory) / f"seed_{seed}/{arm}_evaluation.json")
        rows = evaluation["predictions"]
        seen = set()
        metrics = {panel: {"correct": 0, "format_valid": 0, "total": 0} for panel in PANELS}
        for raw in rows:
            identity = raw["sample_id"]
            if identity in seen or identity not in expected:
                raise ValueError("raw evaluation has duplicate/unregistered sample")
            seen.add(identity)
            frozen, row = lookup[identity], expected[identity]
            candidates = raw.get("all_samples") if arm == "sampling_only" else [raw]
            if not isinstance(candidates, list) or len(candidates) != (config["sampling_baseline_size"] if arm == "sampling_only" else 1):
                raise ValueError("raw candidate count differs from registered sampling budget")
            valid = []
            for candidate in candidates:
                if candidate.get("sample_id") != identity or candidate.get("question") != frozen["question"] or candidate.get("answer") != frozen["answer"]:
                    raise ValueError("raw candidate frozen question/gold identity differs")
                result = score(candidate["question"], candidate["text"], candidate["termination_reason"])
                if any(type(candidate["reward"].get(key)) is not bool or candidate["reward"][key] != result[key] for key in ("correct", "format_valid")):
                    raise ValueError("raw candidate scores differ from independent oracle")
                if result["format_valid"]:
                    valid.append(candidate)
            if valid:
                counts = Counter(candidate["text"] for candidate in valid)
                selected = max(valid, key=lambda candidate: counts[candidate["text"]])
            else:
                selected = candidates[0]
            if arm != "sampling_only":
                selected = raw
            for key in ("text", "termination_reason", "response_token_ids", "prompt_token_ids"):
                if raw[key] != selected[key]:
                    raise ValueError("sampling selection differs from text/EOS-only majority")
            if raw.get("panel") != row["panel"] or row["output_text"] != raw["text"] or row["termination_reason"] != raw["termination_reason"]:
                raise ValueError("reported output differs from raw evaluation")
            lengths = {"input_tokens": len(raw["prompt_token_ids"]), "output_tokens": len(raw["response_token_ids"]),
                       "evaluation_cost_tokens": sum(len(candidate["response_token_ids"]) for candidate in candidates),
                       "evaluation_cost_input_tokens": sum(len(candidate["prompt_token_ids"]) for candidate in candidates)}
            if any(row.get(key) != value for key, value in lengths.items()):
                raise ValueError("selected length/all-candidate evaluation cost differs from raw tokens")
            if row["elapsed_seconds"] != raw["elapsed_seconds"]:
                raise ValueError("reported generation time differs from raw evaluation")
            metrics[row["panel"]]["correct"] += int(row["correct"])
            metrics[row["panel"]]["format_valid"] += int(row["format_valid"])
            metrics[row["panel"]]["total"] += 1
        if seen != set(expected) or evaluation.get("panels") != metrics:
            raise ValueError("raw evaluation omits samples or reports wrong panel totals")


def verify_training_receipts(directory, receipts, lookup, config):
    """Cross-check update counts, origins and train-only sampled identities."""
    sft = {item["training_seed"]: item for item in receipts if item["arm"] == "no_update"}
    for receipt in receipts:
        seed, arm = receipt["training_seed"], receipt["arm"]
        if arm in ("base", "sampling_only"):
            continue
        stem = "sft" if arm == "no_update" else arm
        budget = read_json(Path(directory) / f"seed_{seed}/{stem}_train.json")
        for key in ("optimizer_steps", "initial_weight_hash", "final_weight_hash"):
            if budget.get(key) != receipt[key]:
                raise ValueError("training receipt differs from recorded optimizer/weights")
        if arm != "no_update" and receipt["initial_weight_hash"] != sft[seed]["checkpoint_id"]:
            raise ValueError("post-training arm does not start at frozen SFT checkpoint")
        for log in budget.get("logs", []):
            for identity in log.get("sample_ids", []):
                if identity not in lookup or lookup[identity]["split"] != "train":
                    raise ValueError("training log reads non-training sample")
        if arm.startswith("rlvr"):
            active = [log for log in budget["logs"] if log.get("status") == "updated"]
            if budget.get("policy_gradient_proved") is not True or not active or len(active) != receipt["optimizer_steps"] or any(not isinstance(log.get("policy_gradient_norm"), (int, float)) or not math.isfinite(log["policy_gradient_norm"]) or log["policy_gradient_norm"] <= 0 for log in active):
                raise ValueError("RL update lacks independent positive policy-gradient receipt")
            for collection in budget["collections"]:
                for group in collection["groups"]:
                    if group["sample_id"] not in lookup or lookup[group["sample_id"]]["split"] != "train":
                        raise ValueError("rollout reads non-training frozen sample")


def verify_arm_configs(directory, config, module):
    actual = read_json(Path(directory) / "arm_configs.json")
    expected = {arm: dict(config) for arm in ARMS}
    expected["dpo_beta"]["beta"] = config["beta_ablation"]
    expected["rlvr_group_size"]["group_size"] = config["group_size_ablation"]
    expected["rlvr_no_format"]["format_reward"] = 0.0
    if actual != expected:
        raise ValueError("full resolved arm configuration differs beyond registered factor")
    comparisons = {"beta_single_factor": module.single_factor_difference(actual["dpo"], actual["dpo_beta"], expected_path="beta"),
        "group_size_single_factor": module.single_factor_difference(actual["rlvr"], actual["rlvr_group_size"], expected_path="group_size"),
        "reward_single_factor": module.single_factor_difference(actual["rlvr"], actual["rlvr_no_format"], expected_path="format_reward")}
    rows = module.read_jsonl(Path(directory) / "stages.jsonl")
    for seed in config["seeds"]:
        for stage, names in ((f"{seed}:03_dpo", ("beta_single_factor",)),
                             (f"{seed}:04_rollout_05_update", ("group_size_single_factor", "reward_single_factor"))):
            checks = next(row["checks"] for row in rows if row["stage"] == stage and row["status"] == "passed")
            if any(checks.get(name) != comparisons[name] for name in names):
                raise ValueError("stage ablation receipt differs from full resolved configs")
    return comparisons


def verify_sft_receipt(receipt, tokenization, lookup, batch_checks=None, tolerance=3e-5):
    tokenized = {row["source_sample_id"]: row for split in tokenization["splits"].values() for row in split["records"]}
    numerator, denominator, seen = 0.0, 0, set()
    rows = receipt["per_sample"]
    if not rows:
        raise ValueError("empty SFT probability receipt")
    close = lambda left, right: isinstance(right, (int, float)) and math.isfinite(right) and math.isclose(left, right, abs_tol=tolerance, rel_tol=1e-5)
    for row in rows:
        identity = row["sample_id"]
        if identity in seen or identity not in lookup or lookup[identity]["split"] != "train":
            raise ValueError("SFT receipt duplicate or non-training sample")
        seen.add(identity)
        frozen = tokenized[identity]
        values = row["token_logprobs"]
        count = row["effective_tokens"]
        if type(count) is not int or count <= 0 or len(values) != count or count != frozen["target_tokens"]:
            raise ValueError("SFT effective-token denominator differs from answer plus EOS")
        if row["target_positions"] != list(range(frozen["prompt_tokens"], frozen["prompt_tokens"] + count)):
            raise ValueError("SFT targets shift or supervise prompt/padding")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value > 1e-6 for value in values):
            raise ValueError("invalid selected token log probability")
        sequence = math.fsum(values)
        if not close(sequence, row["sequence_logprob"]) or not close(-sequence, row["negative_log_likelihood"]):
            raise ValueError("SFT sequence logprob/NLL differs from token sum")
        numerator -= sequence
        denominator += count
    if receipt["effective_token_denominator"] != denominator or not close(numerator, receipt["negative_log_likelihood_numerator"]) or not close(numerator / denominator, receipt["loss"]):
        raise ValueError("SFT whole-batch numerator/denominator/loss differs")
    if batch_checks is not None and not close(receipt["loss"], batch_checks["sft"]["loss"]):
        raise ValueError("SFT probability receipt differs from batch cross-entropy check")
    return {"samples": len(rows), "effective_token_denominator": denominator,
            "negative_log_likelihood_numerator": numerator, "float_tolerance": tolerance}


def derived_rl_analysis(directory, config, lookup):
    """Recompute diagnostics from recorded train actions, never claim measurements."""
    result = {"scope": "derived from stored pre-update token logprobs/action masks and independent raw-question oracle; not new runtime measurements",
              "arms": {}}
    for seed in config["seeds"]:
        for arm in ("rlvr", "rlvr_group_size", "rlvr_no_format"):
            budget = read_json(Path(directory) / f"seed_{seed}/{arm}_train.json")
            ratios, lengths, zero_groups, groups, truncated, correct, valid, samples = [], [], 0, 0, 0, 0, 0, 0
            format_reward = 0.0 if arm == "rlvr_no_format" else config["format_reward"]
            for collection in budget["collections"]:
                grouped = {}
                for row in collection["rollouts"]:
                    identity = row["sample_id"]
                    if identity not in lookup or lookup[identity]["split"] != "train" or row["question"] != lookup[identity]["question"]:
                        raise ValueError("derived rollout analysis encounters non-training identity")
                    scored = score(row["question"], row["text"], row["termination_reason"])
                    if any(row["reward"][field] != scored[field] for field in ("correct", "format_valid")):
                        raise ValueError("training rollout oracle score differs")
                    reward = float(scored["correct"]) + format_reward * float(scored["format_valid"])
                    if not math.isclose(reward, row["reward_total"], abs_tol=1e-12):
                        raise ValueError("rollout shaped reward differs from registered oracle rule")
                    grouped.setdefault(row["group_id"], []).append(reward)
                    count = len(row["response_token_ids"])
                    if count > config["max_new_tokens"]:
                        raise ValueError("rollout exceeds registered output token budget")
                    lengths.append(count)
                    samples += 1
                    truncated += row["termination_reason"] != "eos"
                    correct += scored["correct"]
                    valid += scored["format_valid"]
                for group in collection["groups"]:
                    rewards = grouped[group["group_id"]]
                    if rewards != group["rewards"]:
                        raise ValueError("group reward list differs from independently scored actions")
                    groups += 1
                    zero_groups += len(set(rewards)) == 1
                masks, old, current, saved = (collection[key] for key in ("action_mask", "old_token_logprobs", "current_token_logprobs_before", "ratios_before"))
                if not (len(masks) == len(old) == len(current) == len(saved) == len(collection["rollouts"])):
                    raise ValueError("rollout probability/mask row count differs")
                for mask, old_row, current_row, saved_row in zip(masks, old, current, saved):
                    if not len(mask) == len(old_row) == len(current_row) == len(saved_row):
                        raise ValueError("rollout probability/mask token count differs")
                    for present, old_lp, current_lp, stored_ratio in zip(mask, old_row, current_row, saved_row):
                        if present:
                            ratio = math.exp(current_lp - old_lp)
                            if not math.isfinite(ratio) or not math.isclose(ratio, stored_ratio, abs_tol=1e-5, rel_tol=1e-5):
                                raise ValueError("stored ratio differs from exp(current-old)")
                            ratios.append(ratio)
            if not samples or not groups or not ratios:
                raise ValueError("empty RL derived diagnostic")
            result["arms"][f"{seed}:{arm}"] = {"generated_samples": samples, "generated_tokens": sum(lengths),
                "correct": correct, "format_valid": valid, "non_eos_samples": truncated,
                "non_eos_fraction": truncated / samples, "length_including_eos": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / samples},
                "zero_variance_groups": zero_groups, "groups": groups, "zero_variance_fraction": zero_groups / groups,
                "pre_update_ratio_min": min(ratios), "pre_update_ratio_max": max(ratios),
                "pre_update_clip_fraction": sum(ratio < 1 - config["clip"] or ratio > 1 + config["clip"] for ratio in ratios) / len(ratios),
                "ratio_tokens": len(ratios)}
    return result


def summarize_budget_documents(documents, classification):
    """Sum actual logged work; repeated checkpoints do not create extra updates."""
    training, temperature = [], None
    for name, value in documents.items():
        if name.endswith("_train.json"):
            if isinstance(value, list):
                if [row["step"] for row in value] != list(range(1, len(value) + 1)):
                    raise ValueError("calibration log has missing/repeated optimizer steps")
                training.append({"artifact": name, "optimizer_steps": len(value),
                    "sample_exposures": sum(len(row.get("sample_ids", [])) for row in value),
                    "effective_update_tokens": None, "elapsed_seconds": None, "generated_samples": 0, "generated_tokens": 0})
            else:
                training.append({"artifact": name, "optimizer_steps": value["optimizer_steps"],
                    "sample_exposures": sum(len(row.get("sample_ids", [])) for row in value.get("logs", [])),
                    "effective_update_tokens": value.get("effective_update_tokens"), "elapsed_seconds": value.get("elapsed_seconds"),
                    "generated_samples": value.get("generated_samples", 0), "generated_tokens": value.get("generated_tokens", 0)})
        elif name == "summary.json" and isinstance(value, dict) and "plan" in value and "temperatures" in value["plan"]:
            raw = [row for item in value["results"] for group in item["groups"] for row in group["rows"]]
            if len(raw) != sum(item["samples"] for item in value["results"]):
                raise ValueError("temperature exploration sample count differs from raw rows")
            temperature = {"temperatures": value["plan"]["temperatures"], "generated_samples": len(raw),
                "generated_tokens": sum(len(row["response_token_ids"]) for row in raw),
                "elapsed_seconds": sum(row["elapsed_seconds"] for row in raw), "final_opened": value["plan"]["final_opened"]}
    def complete_sum(field):
        values = [row[field] for row in training]
        return sum(values) if values and all(value is not None for value in values) else None
    resume = documents.get("resume_check.json")
    probe = documents.get("resource_probe.json")
    return {"classification": classification, "scope": "actual persisted training logs; interrupted/unlogged work is not counted as zero",
        "recorded_training_files": training, "optimizer_steps_logged": sum(row["optimizer_steps"] for row in training),
        "sample_exposures_logged": sum(row["sample_exposures"] for row in training),
        "effective_update_tokens_logged": complete_sum("effective_update_tokens"), "training_elapsed_seconds_logged": complete_sum("elapsed_seconds"),
        "rollout_samples_logged": sum(row["generated_samples"] for row in training),
        "rollout_tokens_logged": sum(row["generated_tokens"] for row in training),
        "temporary_exact_resume_updates": 3 if resume and all(key in resume for key in ("first_step", "continuous_second_step", "resumed_second_step")) else None,
        "temporary_resource_probe_updates": probe.get("temporary_optimizer_steps") if probe else None,
        "engineering_probe_updates": 1 if isinstance(documents.get("summary.json"), dict) and documents["summary.json"].get("step", {}).get("weight_before") != documents["summary.json"].get("step", {}).get("weight_after") else None,
        "temperature_exploration": temperature,
        "complete_total_runtime_measured": False, "unlogged_failed_work": "unknown; registration/absence of logs is not proof of zero work"}


def run_metadata(directory, source_root, registration, stages, runner_exit_code):
    def git_bytes(arguments):
        try:
            result = subprocess.run(["git", "-C", str(source_root), *arguments], capture_output=True, check=False)
            return result.stdout if result.returncode == 0 else None
        except OSError:
            return None
    commit, diff = git_bytes(["rev-parse", "HEAD"]), git_bytes(["diff", "--binary", "HEAD", "--"])
    environment = read_json(Path(directory) / "environment.json") if (Path(directory) / "environment.json").is_file() else {}
    return {"schema_version": "maedaily-reference-run-metadata-v1", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": [environment.get("python_executable", sys.executable), "post_train/experiments/project_bc_reference.py", "--model-dir", "study_runs/models/Qwen2.5-0.5B-Instruct",
            "--config", "post_train/experiments/reference_config.json", "--device", "cuda", "--output-dir", "study_runs/" + Path(directory).name],
        "command_source": "coordinator-provided actual formal runner arguments; executable from actual environment receipt when available",
        "cwd": str(Path(source_root).resolve()), "runner_exit_code": runner_exit_code,
        "exit_code_source": "coordinator-observed process result" if runner_exit_code is not None else "not recorded; not inferred from summary",
        "started_at": stages[0]["started_at"], "finished_at": stages[-1]["finished_at"],
        "git_commit_at_export": commit.decode().strip() if commit is not None else None,
        "tracked_diff_sha256_against_HEAD_at_export": hashlib.sha256(diff).hexdigest() if diff is not None else None,
        "git_scope": "commit/tracked diff observed at export; untracked registered sources are covered by source_versions hashes",
        "contract_hash": digest(registration), "optimizer_scheduler": "not used", "AMP": "not used", "gradient_accumulation": "not used",
        "per_optimizer_step_elapsed_seconds": None, "gradient_norm_after_clip": None,
        "gradient_clipping": {"registered_frozen_source_rule": "clip_grad_norm_ max_norm=1.0", "reported_gradient_norm_scope": "before clipping; post-clipping norm not measured"}}


def checkpoint_manifest(weights, directory, registration, receipts):
    result = {"schema_version": "maedaily-reference-checkpoint-manifest-v1", "weights_published": False,
        "scope": "SHA/bytes measured from actual checkpoint files; boundary metadata derived from frozen save sites and actual training logs; no stdlib deserialization of weights",
        "base_model_file_hashes": registration["model_file_hashes"], "contract_hash": digest(registration),
        "data_sha256": registration["data"]["data_sha256"], "checkpoints": {}}
    config = registration["config"]
    for name, info in weights.items():
        entry = dict(info)
        match = re.fullmatch(r"seed_([0-9]+)/([a-z_]+?)(?:_round([0-9]+))?\.pt", name)
        if match:
            seed, stem, round_id = match.groups()
            seed, arm = int(seed), "no_update" if stem == "sft" else stem
            entry.update(training_seed=seed, arm=arm, inference_consumers=["no_update", "sampling_only"] if arm == "no_update" else [arm])
            if arm == "base":
                boundary, updates = 0, 0
            else:
                budget = read_json(Path(directory) / f"seed_{seed}/{stem}_train.json")
                if arm.startswith("rlvr"):
                    boundary = int(round_id) if round_id else config["rl_rounds"]
                    updates = sum(log.get("status") == "updated" for log in budget["logs"][:boundary])
                else:
                    boundary = updates = budget["optimizer_steps"]
            entry.update(logical_checkpoint_step=boundary, actual_optimizer_updates_before_checkpoint=updates,
                         optimizer_state_expected=arm != "base", boundary_source="frozen save-call semantics and persisted per-round logs")
        else:
            entry.update(boundary_source="isolated exact-resume checkpoint; see run/resume_check.json", training_arm=False)
        result["checkpoints"][name] = entry
    return result


def verify_stages(directory, registration, receipts, omitted_weights=None, source_root=ROOT):
    directory = Path(directory)
    module = reporting(source_root)
    rows = module.read_jsonl(directory / "stages.jsonl")
    seeds = registration["config"]["seeds"]
    expected = ["00_contract", "01_batch", "exact_resume"]
    expected += [f"{seed}:02_sft" for seed in seeds]
    expected += [name for seed in seeds for name in (f"{seed}:03_dpo", f"{seed}:04_rollout_05_update")]
    expected += ["checkpoint_reload", "06_eval", "07_reproduce"]
    if [(row.get("stage"), row.get("status")) for row in rows] != [(name, status) for name in expected for status in ("running", "passed")]:
        raise ValueError("stage failure, missing dependency or wrong registered stage order")
    omitted_weights = omitted_weights or {}
    excluded_seen = set()
    previous_end = datetime.fromisoformat(registration["preregistered_at"])
    for index, row in enumerate(rows):
        if index % 2 and row.get("failure_reason") is not None:
            raise ValueError("passed stage carries a failure")
        if index % 2 and row.get("started_at") != rows[index - 1].get("started_at"):
            raise ValueError("stage start receipt changed")
        started = datetime.fromisoformat(row["started_at"])
        if started < previous_end:
            raise ValueError("stage starts before preregistration/preceding stage completion")
        if index % 2:
            finished = datetime.fromisoformat(row["finished_at"])
            if finished < started:
                raise ValueError("stage completion precedes its start")
            previous_end = finished
        for group in ("input_artifact_hashes", "output_artifact_hashes"):
            for name, declared in row.get(group, {}).items():
                aliases = {"preregistration": "preregistration.json", "sft_lock": "sft_lock.json"}
                name = aliases.get(name, name)
                safe_relative(name)
                path = directory / name
                if path.is_file():
                    if file_hash(path) != declared:
                        raise ValueError("stage artifact hash differs: " + name)
                elif name.endswith(".pt") and name in omitted_weights:
                    if omitted_weights[name].get("sha256") != declared:
                        raise ValueError("omitted weight inventory hash differs")
                    excluded_seen.add(name)
                else:
                    raise ValueError("stage artifact missing; ignored paths cannot reproduce: " + name)
    if set(omitted_weights) != excluded_seen:
        raise ValueError("undeclared/unreferenced omitted weight artifact")
    passed = {row["stage"]: row["checks"] for row in rows if row["status"] == "passed"}
    if passed["00_contract"].get("contract_hash") != digest(registration) or passed["00_contract"].get("data_hash") != registration["data"]["data_sha256"]:
        raise ValueError("stage contract/data hash differs")
    batch = read_json(directory / "batch_checks.json")
    if passed["01_batch"] != batch or any(batch.get(name, {}).get("passed") is not True for name in ("sft", "dpo", "frozen", "verifier")):
        raise ValueError("independent batch/gradient/frozen/verifier gates missing or failed")
    resume = read_json(directory / "resume_check.json")
    if passed["exact_resume"] != resume or resume.get("passed") is not True or not resume.get("comparisons") or not all(value is True for value in resume["comparisons"].values()):
        raise ValueError("exact-resume comparison gate not passed")
    for seed in seeds:
        if passed[f"{seed}:02_sft"].get("development_qualified") is not True or passed[f"{seed}:02_sft"].get("final_opened") is not False:
            raise ValueError("SFT development gate not passed before final")
        if passed[f"{seed}:04_rollout_05_update"].get("policy_gradient_proved") != {arm: True for arm in ("rlvr", "rlvr_group_size", "rlvr_no_format")}:
            raise ValueError("RL arm lacks proved real policy-gradient update")
    reloads = read_json(directory / "reload_checks.json")["reloaded_checkpoints"]
    by_key = {(item["seed"], item["arm"]): item for item in reloads}
    if len(by_key) != len(reloads) or set(by_key) != {(seed, arm) for seed in seeds for arm in ARMS}:
        raise ValueError("missing/duplicate disk reload identity")
    for item in receipts:
        check = by_key[item["training_seed"], item["arm"]]
        if check.get("reload_exact") is not True or check.get("weight_hash") != item["checkpoint_id"]:
            raise ValueError("checkpoint reload differs from evaluated identity")
    if passed["checkpoint_reload"].get("reloaded_checkpoints") != reloads:
        raise ValueError("disk reload stage/file disagree")
    if passed["06_eval"].get("all_arms_reported") != list(ARMS) or passed["06_eval"].get("all_seeds_reported") != seeds:
        raise ValueError("final stage loses registered arm/seed")
    if passed["07_reproduce"].get("mechanisms_passed") is not True or passed["07_reproduce"].get("baseline_qualified") is not True:
        raise ValueError("completion gate failed")
    opening = read_json(directory / "final_opening.json")
    final_start = next(row["started_at"] for row in rows if row["stage"] == "06_eval" and row["status"] == "running")
    final_end = next(row["finished_at"] for row in rows if row["stage"] == "06_eval" and row["status"] == "passed")
    if not datetime.fromisoformat(final_start) <= datetime.fromisoformat(opening["at"]) <= datetime.fromisoformat(final_end):
        raise ValueError("final panels opened outside frozen final stage")
    return {"stage_count": len(expected), "artifact_hashes_verified": True,
            "omitted_weight_files": len(omitted_weights), "weight_bytes_independently_verified": not omitted_weights}


def panel_analysis(records, config, module):
    analyses = {}
    for seed in config["seeds"]:
        for panel in PANELS:
            baseline = [row for row in records if row["training_seed"] == seed and row["arm"] == "no_update" and row["panel"] == panel]
            for arm in ARMS:
                if arm == "no_update":
                    continue
                candidate = [row for row in records if row["training_seed"] == seed and row["arm"] == arm and row["panel"] == panel]
                analyses[f"{seed}:{panel}:{arm}"] = module.paired_family_bootstrap(baseline, candidate,
                    replicates=config["bootstrap_replicates"])
    return analyses


def verify_run(directory, source_root=ROOT, model_manifest=None, omitted_weights=None):
    directory, source_root = Path(directory), Path(source_root)
    if (directory / "failure.json").exists():
        raise ValueError("failed run cannot be exported as completed reference")
    summary = read_json(directory / "summary.json")
    if summary.get("status") != "passed" or summary.get("mechanisms_passed") is not True:
        raise ValueError("formal run did not pass completion gate")
    registration = read_json(directory / "preregistration.json")
    config = read_json(directory / "config.resolved.json")
    if registration["config"] != config or registration.get("final_used_for_selection") is not False:
        raise ValueError("registration/config/final selection disagree")
    if summary.get("contract_hash") != digest(registration) or summary.get("executed_training_seeds") != config["seeds"] or summary.get("arms") != list(ARMS):
        raise ValueError("summary frozen identity/seed matrix differs")
    code = read_json(directory / "source_versions.json")
    if code != registration["code_hashes"]:
        raise ValueError("registered source manifest differs")
    for name, declared in code.items():
        path = source_path(source_root, name)
        if not path.is_file() or file_hash(path) != declared:
            raise ValueError("frozen source missing/changed: " + name)
    manifest = read_json(model_manifest or source_root / "post_train/experiments/reference_model.json")
    if not re.fullmatch(r"[a-f0-9]{40}", manifest.get("revision", "")):
        raise ValueError("model revision must be fixed immutable commit")
    if {name: info["sha256"] for name, info in manifest["files"].items()} != registration["model_file_hashes"]:
        raise ValueError("public pinned model differs from preregistered model files")
    lookup = validate_data(read_json(directory / "data_manifest.json"), registration)
    module = reporting(source_root)
    records = module.read_jsonl(directory / "predictions.jsonl")
    receipts = read_json(directory / "training_receipts.json")
    result = verify_predictions(records, lookup, config, registration, receipts)
    verify_raw_evaluations(directory, records, lookup, config)
    verify_training_receipts(directory, receipts, lookup, config)
    opening = read_json(directory / "final_opening.json")
    if opening.get("selection_locked") is not True or opening.get("no_retuning_after_final") is not True or opening.get("protocol_hash") != result["final_eval_protocol_hash"]:
        raise ValueError("final opening lacks frozen selection/protocol")
    lock = read_json(directory / "sft_lock.json")
    if lock.get("contract_hash") != digest(registration) or lock.get("weights") != {str(item["training_seed"]): item["checkpoint_id"] for item in receipts if item["arm"] == "no_update"}:
        raise ValueError("SFT lock identity differs")
    for seed in config["seeds"]:
        baseline = [row for row in records if row["training_seed"] == seed and row["arm"] == "no_update"]
        for panel, rate in (("eval_main", .8), ("eval_regression", .9)):
            rows = [row for row in baseline if row["panel"] == panel]
            if len(rows) < 20 or sum(row["correct"] for row in rows) < rate * len(rows) or not all(row["format_valid"] for row in rows):
                raise ValueError("independently scored baseline fails frozen quality gate")
    recomputed = module.build_report(records, training_receipts=receipts, baseline_arm="no_update",
                                     bootstrap_replicates=config["bootstrap_replicates"])
    if not equivalent_derived(without_time(recomputed), without_time(read_json(directory / "report.json"))):
        raise ValueError("saved report differs from independent score/report recomputation")
    # Import only the packaged standard-library reporting module, never training code.
    stages = verify_stages(directory, registration, receipts, omitted_weights, source_root)
    if "scripts/reference_instrumentation.py" in code:
        stage_rows = module.read_jsonl(directory / "stages.jsonl")
        for name, stage in (("environment.json", "00_contract"), ("resource_probe.json", "00_contract"),
                            ("token_alignment.jsonl", "01_batch"), ("tokenization_manifest.json", "00_contract"),
                            ("sft_loss_receipt.json", "01_batch")):
            path = directory / name
            hashes = next(row["output_artifact_hashes"] for row in stage_rows if row["stage"] == stage and row["status"] == "passed")
            if not path.is_file() or hashes.get(name) != file_hash(path):
                raise ValueError("required instrumentation lacks bound stage artifact: " + name)
        result["sft_probability_receipt"] = verify_sft_receipt(read_json(directory / "sft_loss_receipt.json"),
            read_json(directory / "tokenization_manifest.json"), lookup, read_json(directory / "batch_checks.json"))
        result["full_config_ablations"] = verify_arm_configs(directory, config, module)
    result.update({"status": "verified", "independent_oracle": "operator-add-sub-copy-int-canonical-eos-v1",
                   "report_recomputed": True, "source_hashes_verified": len(code),
                   "derived_float_comparison": {"absolute_tolerance": 1e-12, "relative_tolerance": 1e-12,
                       "named_fields": sorted(DERIVED_FLOAT_FIELDS), "all_other_values_and_artifact_SHA": "exact"},
                   "model_revision": manifest["revision"], "stages": stages,
                   "review_status": "not_reviewed", "training_repeated": False})
    return result, records, receipts


def export_run(directory, destination, source_root=ROOT, model_manifest=None, *, budget_sources=None, runner_exit_code=None):
    directory, source_root, destination = Path(directory), Path(source_root), Path(destination)
    result, records, receipts = verify_run(directory, source_root, model_manifest)
    registration = read_json(directory / "preregistration.json")
    module = reporting(source_root)
    files, weights = {}, {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("run symlinks cannot be published")
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if path.name.endswith(".sealed.json"):
            continue
        if path.suffix == ".pt":
            weights[relative] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        elif path.suffix in (".json", ".jsonl", ".md"):
            files["run/" + relative] = path
        else:
            raise ValueError("unclassified run artifact: " + relative)
    for name in registration["code_hashes"]:
        member = name if len(safe_relative(name).parts) > 1 else "post_train/experiments/" + name
        files[member] = source_path(source_root, name)
    files["post_train/experiments/reference_model.json"] = Path(model_manifest or source_root / "post_train/experiments/reference_model.json")
    registered_config = source_root / "post_train/experiments/reference_config.json"
    if registered_config.is_file() and all(registration["config"].get(key) == value for key, value in read_json(registered_config).items()):
        files["post_train/experiments/reference_config.json"] = registered_config
    for name in ("verify_reference_run.py", "prepare_reference_model.py", "lesson_runtime.py"):
        files["scripts/" + name] = source_root / "scripts" / name
    packet = module.blind_review_packet(records)
    analyses = panel_analysis(records, registration["config"], module)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        extras = {"omitted_weights.json": weights, "independent_verification.json": result,
                  "07_panel_family_bootstrap.json": analyses,
                  "blind_review/packet.json": {"protocol": packet["protocol"], "records": packet["records"]},
                  "blind_review/blank_worksheet.json": packet["blank_reviews"]}
        stages = module.read_jsonl(directory / "stages.jsonl")
        extras["run_metadata.json"] = run_metadata(directory, source_root, registration, stages, runner_exit_code)
        extras["checkpoint_manifest.json"] = checkpoint_manifest(weights, directory, registration, receipts)
        if "scripts/reference_instrumentation.py" in registration["code_hashes"]:
            lookup = validate_data(read_json(directory / "data_manifest.json"), registration)
            extras["08_derived_rl_analysis.json"] = derived_rl_analysis(directory, registration["config"], lookup)
        if budget_sources:
            sources = read_json(budget_sources)
            audit = {"scope": "historical exploration/voided runs reported separately; never used as final selection evidence", "experiments": {}}
            for label, source in sources.items():
                safe_relative(label)
                if "/" in label:
                    raise ValueError("budget source label must be one safe component")
                origin = Path(source["directory"])
                if not origin.is_absolute():
                    origin = source_root / origin
                paths = list(origin.rglob("*_train.json"))
                paths += [origin / name for name in ("summary.json", "failure.json", "resume_check.json", "resource_probe.json", "preregistration.json") if (origin / name).is_file()]
                if not paths:
                    raise ValueError("budget source needs actual persisted records")
                documents, members = {}, {}
                for path in paths:
                    relative = path.relative_to(origin).as_posix()
                    documents[relative] = read_json(path)
                    member = "budget_sources/" + label + "/" + relative + ".gz"
                    target = temporary / member
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
                    files[member] = target
                    members[relative] = {"member": member, "uncompressed_sha256": file_hash(path)}
                audit["experiments"][label] = {"classification": source["classification"], "members": members,
                    "summary": summarize_budget_documents(documents, source["classification"])}
            extras["09_budget_audit.json"] = audit
        for name, value in extras.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            files[name] = path
        instructions = temporary / "REPRODUCE.md"
        instructions.write_text("# Frozen B/C reference evidence\n\n"
            "From the repository root, verify the canonical public package directly:\n\n"
            "```powershell\npython references/evidence/project_bc_20261002/scripts/verify_reference_run.py "
            "--verify-package references/evidence/project_bc_20261002\n```\n\n"
            "Run from this package directory with Python 3.9 or later (standard library only):\n\n"
            "```powershell\npython scripts/verify_reference_run.py --verify-package .\n```\n\n"
            "This verifies all raw-question scores, frozen identities, stage/source hashes and recomputes the report. "
            "The blind worksheet is blank and has not been reviewed; the identity mapping is sealed separately.\n\n"
            "Python versions can differ in the last bits of floating-point summation. Only named derived "
            "cost/time/ratio floats use finite absolute and relative tolerances of 1e-12. All original artifact "
            "SHA-256 hashes, identities, booleans, integer counts, scores and configuration remain exact. "
            "Actual AI-assisted subset judgments and their crosscheck are preserved in run/ai_assisted_review.json "
            "and run/ai_review_crosscheck.json; human review remains not_reviewed.\n\n"
            "Model and adapter weights are excluded. `omitted_weights.json` records observed adapter file hashes, "
            "which are historical receipts, not an independent verification of omitted bytes. To repeat training, "
            "install the recorded PyTorch/Transformers runtime, run `scripts/prepare_reference_model.py`, then "
            "`python post_train/experiments/project_bc_reference.py --model-dir study_runs/models/Qwen2.5-0.5B-Instruct "
            "--config run/config.resolved.json --device cuda --output-dir study_runs/repeated_bc_reference`. "
            "The preparation script downloads the exact revision and verifies all model file hashes in "
            "`post_train/experiments/reference_model.json`. No original ignored run directory is needed for rescoring.\n",
            encoding="utf-8")
        files["REPRODUCE.md"] = instructions
        files["README.md"] = instructions
        module.create_evidence_package(files, destination, required_files=list(files),
            metadata={"schema": "maedaily-independent-bc-evidence-v1", "formal_status": "passed",
                      "statistics_reproducible_without_weights": True, "training_repeated": False,
                      "review_status": "not_reviewed", "weights_scope": "excluded; fixed-revision download and retraining required",
                      "identity_map_scope": "sealed separately; not a public evidence member"},
            max_file_bytes=32 * 1024 * 1024, max_total_bytes=32 * 1024 * 1024)
    sealed = directory / "blind_review_identity_map.sealed.json"
    if sealed.exists():
        if read_json(sealed) != packet["identity_map"]:
            raise ValueError("sealed review identity map differs")
    else:
        sealed.write_text(json.dumps(packet["identity_map"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verified = verify_package(destination)
    return {**verified, "sealed_identity_map": str(sealed)}


def verify_package(directory):
    directory = Path(directory)
    module = reporting(directory)
    module.verify_evidence_package(directory)
    result, records, _ = verify_run(directory / "run", directory,
                              omitted_weights=read_json(directory / "omitted_weights.json"))
    packet = read_json(directory / "blind_review/packet.json")
    worksheet = read_json(directory / "blind_review/blank_worksheet.json")
    expected = module.blind_review_packet(records)
    if packet != {"protocol": expected["protocol"], "records": expected["records"]} or worksheet != expected["blank_reviews"]:
        raise ValueError("published blind packet/worksheet differs from frozen anonymous records")
    config = read_json(directory / "run/config.resolved.json")
    if not equivalent_derived(read_json(directory / "07_panel_family_bootstrap.json"), panel_analysis(records, config, module)):
        raise ValueError("panel family-bootstrap attachment differs from independent recomputation")
    registration = read_json(directory / "run/preregistration.json")
    weights = read_json(directory / "omitted_weights.json")
    receipts = read_json(directory / "run/training_receipts.json")
    if read_json(directory / "checkpoint_manifest.json") != checkpoint_manifest(weights, directory / "run", registration, receipts):
        raise ValueError("checkpoint boundary/dependency manifest differs from actual recorded provenance")
    metadata = read_json(directory / "run_metadata.json")
    stages = module.read_jsonl(directory / "run/stages.jsonl")
    if metadata.get("contract_hash") != digest(registration) or metadata.get("started_at") != stages[0]["started_at"] or metadata.get("finished_at") != stages[-1]["finished_at"] or metadata.get("runner_exit_code") not in (None, 0):
        raise ValueError("outer run receipt contradicts frozen run/stages")
    if "scripts/reference_instrumentation.py" in registration["code_hashes"]:
        lookup = validate_data(read_json(directory / "run/data_manifest.json"), registration)
        if not equivalent_derived(read_json(directory / "08_derived_rl_analysis.json"), derived_rl_analysis(directory / "run", config, lookup)):
            raise ValueError("derived RL analysis differs from stored raw probabilities/oracle")
    if (directory / "09_budget_audit.json").is_file():
        audit = read_json(directory / "09_budget_audit.json")
        for source in audit["experiments"].values():
            documents = {}
            for name, info in source["members"].items():
                safe_relative(info["member"])
                with gzip.open(directory / info["member"], "rb") as stream:
                    raw = stream.read(32 * 1024 * 1024 + 1)
                if len(raw) > 32 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != info["uncompressed_sha256"]:
                    raise ValueError("budget source uncompressed size/hash differs")
                documents[name] = module._strict_json(raw.decode("utf-8"))
            if not equivalent_derived(source["summary"], summarize_budget_documents(documents, source["classification"])):
                raise ValueError("public historical budget summary differs from actual compressed logs")
    if len(packet["records"]) != result["prediction_count"] or packet["protocol"].get("status") != "not_reviewed" or len(worksheet) != len(packet["records"]):
        raise ValueError("blind packet identity/count differs")
    if any(row.get("status") != "not_reviewed" or any(row.get(key) is not None for key in ("reviewer_id", "reviewer_type", "correct", "format_valid", "reason")) for row in worksheet):
        raise ValueError("published blank blind worksheet claims review")
    return {**result, "package_directory": str(directory), "package_bytes": sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
            "blind_items": len(packet["records"])}


def main(argv=None):
    if sys.version_info < (3, 9):
        print("Independent evidence verification requires Python 3.9 or later (standard library only).", file=sys.stderr)
        return 1
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--budget-sources", type=Path, help="JSON mapping labels to directory/classification; actual logs are compressed into exported package")
    parser.add_argument("--runner-exit-code", type=int, choices=(0,), help="actual coordinator-observed successful runner exit code; never inferred")
    parser.add_argument("--verify-package", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_package:
            if args.run_dir or args.export:
                parser.error("--verify-package cannot be combined with run/export")
            result = verify_package(args.verify_package)
        elif args.run_dir:
            if args.export:
                result = export_run(args.run_dir, args.export, args.source_root, args.model_manifest,
                    budget_sources=args.budget_sources, runner_exit_code=args.runner_exit_code)
            else:
                result, _, _ = verify_run(args.run_dir, args.source_root, args.model_manifest)
        else:
            parser.error("--run-dir or --verify-package is required")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
