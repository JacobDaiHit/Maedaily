"""Verify the published Project A JSON/CSV with Python's standard library only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def number(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite JSON number")
        return value

    def constant(text):
        raise ValueError(f"nonfinite JSON constant: {text}")

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_float=number, parse_constant=constant)


def verify(source_root: Path | None = None) -> dict:
    manifest = strict_json(PACKAGE / "package_manifest.json")
    require(manifest["schema_version"] == "maedaily-public-project-a-package-v1", "package schema mismatch")
    for name, receipt in manifest["files"].items():
        path = (PACKAGE / name).resolve()
        require(path.is_relative_to(PACKAGE) and path.is_file(), f"missing/unsafe file: {name}")
        require(path.stat().st_size == receipt["bytes"] and digest(path) == receipt["sha256"], f"SHA256/size mismatch: {name}")
    evidence = strict_json(PACKAGE / "evidence.json")
    require(evidence["schema_version"] == "maedaily-project-a-reference-v1", "evidence schema mismatch")
    config, preregistration = evidence["config"], evidence["preregistration"]
    identity = preregistration["data_identity"]
    train_lengths, valid_lengths = identity["train_digit_lengths"], identity["validation_digit_lengths"]

    def data(lengths):
        return [[1] + [3 + ((start + direction * index) % 10) for index in range(length)] + [2]
                for length in lengths for direction in (1, -1) for start in range(10)]

    train, valid = data(train_lengths), data(valid_lengths)
    ordered = {"tokens": identity["token_schema"], "train": train, "validation": valid}
    rebuilt_hash = hashlib.sha256(json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    require(rebuilt_hash == identity["sha256"], "independent grammar data identity mismatch")
    require(not set(map(tuple, train)) & set(map(tuple, valid)), "full training/validation sequence overlap")
    with (PACKAGE / "step_comparison.csv").open(encoding="utf-8", newline="") as stream:
        steps = list(csv.DictReader(stream))
    branches = {name: [] for name in ("continuous", "prefix", "resumed", "missing_optimizer", "missing_batch_generator", "alternate_lr")}
    for row in steps:
        require(row["branch"] in branches, "unexpected branch")
        branches[row["branch"]].append(row)
    expected_ranges = {name: list(range(1, config["steps"] + 1)) for name in ("continuous", "alternate_lr")}
    expected_ranges["prefix"] = list(range(1, config["split_step"] + 1))
    for name in ("resumed", "missing_optimizer", "missing_batch_generator"):
        expected_ranges[name] = list(range(config["split_step"] + 1, config["steps"] + 1))
    reference = {int(row["step"]): row for row in branches["continuous"]}
    prefix_tokens = sum(int(row["valid_tokens"]) for row in branches["prefix"])
    for name, rows in branches.items():
        require([int(row["step"]) for row in rows] == expected_ranges[name], f"step range mismatch: {name}")
        tokens = 0 if name in ("continuous", "prefix", "alternate_lr") else prefix_tokens
        for row in rows:
            indices = json.loads(row["batch_ids"])
            require(len(indices) == config["batch_size"] and all(type(i) is int and 0 <= i < len(train) for i in indices), "invalid batch identities")
            count = sum(len(train[i]) - 1 for i in indices)
            require(count == int(row["valid_tokens"]), "independent valid-token count mismatch")
            tokens += count
            require(tokens == int(row["processed_tokens"]), "processed-token cursor mismatch")
            loss, norm = float(row["loss"]), float(row["gradient_norm"])
            require(math.isfinite(loss) and loss > 0 and math.isfinite(norm) and norm > 0, "invalid real-update loss/gradient")
            left = reference[int(row["step"])]
            batch_equal = indices == json.loads(left["batch_ids"])
            require((row["batch_equal"] == "True") == batch_equal, "batch equality declaration mismatch")
            require(float(row["loss_difference"]) == abs(loss - float(left["loss"])), "loss difference recomputation mismatch")
            for key in ("parameters", "optimizer", "rng"):
                value = row[f"{key}_sha256"]
                require(len(value) == 64 and all(character in "0123456789abcdef" for character in value), "invalid state SHA256")
                equal = value == left[f"{key}_sha256"]
                require((row[f"{key}_equal"] == "True") == equal, "state equality declaration mismatch")
                if name in ("prefix", "resumed"):
                    require(equal, f"exact recovery {key} mismatch")
            if name in ("prefix", "resumed"):
                require(batch_equal and loss == float(left["loss"]), "exact recovery batch/loss mismatch")
            if name == "alternate_lr":
                require(batch_equal and count == int(left["valid_tokens"]), "learning-rate branch data/token budget differs")
    optimizer = branches["missing_optimizer"][0]
    sampler = branches["missing_batch_generator"][0]
    require(optimizer["batch_equal"] == "True" and float(optimizer["loss_difference"]) == 0 and optimizer["parameters_equal"] == "False", "optimizer negative control invalid")
    require(sampler["batch_equal"] == "False" and sampler["parameters_equal"] == "False", "sampler negative control invalid")
    with (PACKAGE / "learning_curves.csv").open(encoding="utf-8", newline="") as stream:
        curves = list(csv.DictReader(stream))
    expected_curve_steps = sorted({0, config["split_step"], config["steps"]} |
                                 set(range(config["eval_every"], config["steps"] + 1, config["eval_every"])))
    for name in ("base_lr", "alternate_lr"):
        selected = [row for row in curves if row["branch"] == name]
        quality = evidence["quality"][name]
        require([int(row["step"]) for row in selected] == expected_curve_steps, "curve checkpoints differ")
        require(all(float(row["learning_rate"]) == quality["learning_rate"] and int(row["validation_tokens"]) == sum(len(item) - 1 for item in valid) for row in selected), "curve rate/token protocol differs")
        for position, label in ((0, "initial"), (-1, "final")):
            for panel in ("train", "validation"):
                require(float(selected[position][f"{panel}_loss"]) == quality[f"{label}_{panel}"]["loss"], "curve/JSON metric disagreement")
    checks = evidence["mechanism_acceptance"]
    require(checks["status"] == "passed" and all(checks["final_state_exact"].values()), "original mechanism not passed")
    require(len({row["pid"] for row in evidence["processes"].values()}) == 6, "worker processes not distinct")
    receipt = strict_json(PACKAGE / "export_receipt.json")
    require(receipt["source_summary_sha256"] == manifest["original_run"]["summary_sha256"], "original summary receipt differs")
    if source_root is not None:
        for relative, expected in preregistration["source_hashes"].items():
            path = (source_root.resolve() / relative).resolve()
            require(path.is_relative_to(source_root.resolve()) and path.is_file() and digest(path) == expected, f"source revision SHA256 mismatch: {relative}")
    return {"status": "passed", "verified_public_files": len(manifest["files"]), "csv_rows": len(steps),
            "resumed_steps": len(branches["resumed"]), "updated_tokens_all_branches": sum(int(row["valid_tokens"]) for row in steps),
            "data_sha256": rebuilt_hash, "source_sha256_checked": source_root is not None,
            "scope": "public file integrity and independently recomputed CSV/data counts; original tensors require a reproduced run"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, help="optional strict comparison with the original four training source SHAs")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.source_root), ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Project A public evidence verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
