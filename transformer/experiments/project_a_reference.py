"""Project A reference: fresh-process exact CPU resume on a synthetic grammar.

Existing PyTorch only. This is a from-scratch digit-language training reference,
not a pretrained general LM or evidence of a learner's independent mastery.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts"))
from lesson_runtime import reserve_output_dir

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("请选择已有 PyTorch 解释器；本实训不安装依赖。") from error

import lm_resume_lab as resume
import tiny_causal_lm as lm

SCHEMA = "maedaily-project-a-reference-v1"
BASE_LR = 0.003
OTHER_LR = 0.001


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(value: Any) -> str:
    """Canonical type/shape/dtype/byte hash, including optimizer and RNG tensors."""
    digest = hashlib.sha256()

    def field(tag: bytes, content: bytes) -> None:
        digest.update(tag + struct.pack("!Q", len(content)) + content)

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            field(b"T", str(tensor.dtype).encode())
            visit(list(tensor.shape))
            field(b"B", bytes(tensor.reshape(-1).view(torch.uint8).tolist()))
        elif isinstance(item, dict):
            field(b"D", str(len(item)).encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            field(b"L" if isinstance(item, list) else b"U", str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None:
            field(b"N", b"")
        elif type(item) is bool:
            field(b"Q", bytes([item]))
        elif type(item) is int:
            field(b"I", str(item).encode())
        elif type(item) is float:
            field(b"F", struct.pack("!d", item))
        elif isinstance(item, str):
            field(b"S", item.encode("utf-8"))
        else:
            raise TypeError(f"unsupported state type: {type(item)}")

    visit(value)
    return digest.hexdigest()


def shift_check(sequences: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor) -> None:
    expected = sequences[:, 1:].clone()
    expected[expected == lm.PAD] = lm.IGNORE_INDEX
    if not torch.equal(inputs, sequences[:, :-1]) or not torch.equal(targets, expected):
        raise AssertionError("next-token shift must occur exactly once, with PAD ignored")


@torch.no_grad()
def causal_check(model: lm.TinyCausalLM, train: list[list[int]]) -> float:
    model.eval()
    inputs, _ = lm.inputs_and_targets(lm.collate([train[-1]]))
    changed = inputs.clone()
    changed[:, 5:] = 3 + (changed[:, 5:] - 3 + 4) % 10
    difference = float((model(inputs)[:, :5] - model(changed)[:, :5]).abs().max())
    if difference > 1e-12:
        raise AssertionError("future-token leakage into prefix logits")
    return difference


def numerical_checks(model: lm.TinyCausalLM, train: list[list[int]]) -> dict[str, Any]:
    example = torch.tensor([[lm.BOS, 3, 4, lm.EOS, lm.PAD]])
    inputs, targets = lm.inputs_and_targets(example)
    shift_check(example, inputs, targets)
    wrong = torch.tensor([[4, lm.EOS, lm.IGNORE_INDEX, lm.IGNORE_INDEX]])
    try:
        shift_check(example, inputs, wrong)
    except AssertionError:
        shift_negative_detected = True
    else:
        raise AssertionError("double-shift negative control escaped detection")
    logits = torch.zeros(1, 4, lm.VOCAB_SIZE, dtype=resume.DTYPE)
    for position, target in enumerate((3, 4, lm.EOS)):
        logits[0, position, target] = 6
    logits.requires_grad_()
    loss = lm.summed_loss(logits, targets) / 3
    expected = math.log1p((lm.VOCAB_SIZE - 1) * math.exp(-6))
    loss.backward()
    if abs(float(loss.detach()) - expected) > 1e-12 or float(logits.grad[0, 3].abs().max()) != 0:
        raise AssertionError("analytic CE or ignored-token gradient failed")
    causal_difference = causal_check(model, train)
    allowed = model.attention.causal_allowed.clone()
    try:
        model.attention.causal_allowed.fill_(True)
        try:
            causal_check(model, train)
        except AssertionError:
            causal_negative_detected = True
        else:
            raise AssertionError("unmasked-attention negative control escaped detection")
    finally:
        model.attention.causal_allowed.copy_(allowed)
    with torch.no_grad():
        short = lm.collate([train[0]])
        padded = torch.nn.functional.pad(short, (0, 4), value=lm.PAD)
        short_x, short_y = lm.inputs_and_targets(short)
        padded_x, padded_y = lm.inputs_and_targets(padded)
        padding_difference = abs(float(lm.summed_loss(model(short_x), short_y)) -
                                 float(lm.summed_loss(model(padded_x), padded_y)))
    one, many = lm.evaluate(model, train, 1), lm.evaluate(model, train, 17)
    reduction_difference = abs(one["loss"] - many["loss"])
    if padding_difference > 1e-10 or reduction_difference > 1e-10 or one["tokens"] != many["tokens"]:
        raise AssertionError("padding or token-weighted evaluation is incorrect")
    return {"shift_exactly_once": True, "double_shift_negative_detected": shift_negative_detected,
            "analytic_ce_error": abs(float(loss.detach()) - expected), "ignored_gradient_max": 0.0,
            "causal_prefix_max_difference": causal_difference,
            "unmasked_attention_negative_detected": causal_negative_detected,
            "padding_nll_difference": padding_difference, "partition_nll_difference": reduction_difference}


def peak_rss() -> dict[str, Any]:
    """OS native peak resident/working-set bytes, including native tensor memory."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            probe = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
            probe.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            probe.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not probe(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise OSError(ctypes.get_last_error())
            value, method = counters.PeakWorkingSetSize, "Windows GetProcessMemoryInfo.PeakWorkingSetSize"
        else:
            import resource
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            value *= 1 if sys.platform == "darwin" else 1024
            method = "getrusage RUSAGE_SELF.ru_maxrss"
        return {"status": "measured", "peak_resident_bytes": int(value), "method": method,
                "scope": "whole worker lifetime; CPU resident memory, not GPU VRAM"}
    except (ImportError, OSError, AttributeError) as error:
        return {"status": "not_measured", "peak_resident_bytes": None, "reason": str(error)}


def worker(request_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output = reserve_output_dir(Path(request["output_dir"]), "project-a-worker")
    config, branch = request["config"], request["branch"]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started, cpu_started = time.perf_counter(), time.process_time()
    train, valid = lm.make_sequences(lm.TRAIN_LENGTHS), lm.make_sequences(lm.VALID_LENGTHS)
    identity = resume.data_identity(train, valid)
    if identity != request["data_identity"]:
        raise ValueError("worker data identity differs from registered data")
    restore_checks = None
    if request.get("checkpoint"):
        checkpoint = Path(request["checkpoint"])
        if file_hash(checkpoint) != request["checkpoint_sha256"]:
            raise ValueError("checkpoint bytes differ from registered input")
        model, optimizer, generator, payload, restore_checks = resume.restore_checkpoint(
            checkpoint, config, identity, omit_optimizer=branch == "missing_optimizer",
            omit_batch_generator=branch == "missing_batch_generator")
        start, processed_tokens = payload["global_step"], payload["processed_tokens"]
    else:
        model, optimizer, generator = resume.new_objects(config["seed"])
        for group in optimizer.param_groups:
            group["lr"] = config["optimizer_parameters"]["lr"]
        start, processed_tokens = 0, 0
    initial_hash = state_hash(model.state_dict())
    checks = numerical_checks(model, train) if branch == "continuous" else None
    curve = [{"step": start, "processed_tokens": processed_tokens,
              "train": lm.evaluate(model, train), "validation": lm.evaluate(model, valid)}]
    rows = []
    for step in range(start + 1, request["stop"] + 1):
        before = state_hash(model.state_dict())
        segment, _ = resume.train_segment(model, optimizer, generator, train, step - 1, step)
        row = segment[0]
        after = state_hash(model.state_dict())
        if before == after:
            raise AssertionError(f"step {step}: actual parameter update missing")
        processed_tokens += row["valid_tokens"]
        row.update({"parameters_sha256": after, "optimizer_sha256": state_hash(optimizer.state_dict()),
                    "rng_sha256": state_hash(resume.rng_state(generator)), "processed_tokens": processed_tokens,
                    "global_rng_witness": resume.rng_witness(resume.rng_state(generator))})
        rows.append(row)
        if step % config["eval_every"] == 0 or step in (config["split_step"], request["stop"]):
            curve.append({"step": step, "processed_tokens": processed_tokens,
                          "train": lm.evaluate(model, train), "validation": lm.evaluate(model, valid)})
    generations = [lm.generate(model, prefix) for prefix in ([7, 8], [2, 1], [9, 0], [0, 9])]
    resume.save_checkpoint(output / "final.pt", model, optimizer, generator, config, identity,
                           request["stop"], processed_tokens)
    elapsed = time.perf_counter() - started
    report = {"schema_version": SCHEMA, "branch": branch, "pid": os.getpid(), "parent_pid": os.getppid(),
              "command": [sys.executable, str(Path(__file__).resolve()), "--worker-request", str(request_path)],
              "initial_parameters_sha256": initial_hash, "start_step": start, "stop_step": request["stop"],
              "rows": rows, "learning_curve": curve, "restore_checks": restore_checks,
              "numerical_checks": checks, "generations": generations,
              "resources": {"wall_seconds": elapsed, "cpu_seconds": time.process_time() - cpu_started,
                  "updated_tokens": sum(row["valid_tokens"] for row in rows),
                  "updates": len(rows), "update_tokens_per_worker_second": sum(row["valid_tokens"] for row in rows) / elapsed,
                  "peak_memory": peak_rss(), "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                  "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
                  "generated_tokens": sum(len(row["continuation"]) for row in generations)},
              "final_checkpoint_sha256": file_hash(output / "final.pt")}
    write_json(output / "trace.json", report)
    return 0


def compare_trace(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    if not reference or len(reference) != len(candidate) or [row["step"] for row in reference] != [row["step"] for row in candidate]:
        raise AssertionError("trace step identities differ")
    return {"steps_compared": len(reference),
            "all_batch_ids_equal": all(a["batch_ids"] == b["batch_ids"] for a, b in zip(reference, candidate)),
            "all_losses_exact": all(a["loss"] == b["loss"] for a, b in zip(reference, candidate)),
            "max_loss_abs_difference": max(abs(a["loss"] - b["loss"]) for a, b in zip(reference, candidate)),
            "all_parameter_states_exact": all(a["parameters_sha256"] == b["parameters_sha256"] for a, b in zip(reference, candidate)),
            "all_optimizer_states_exact": all(a["optimizer_sha256"] == b["optimizer_sha256"] for a, b in zip(reference, candidate)),
            "all_rng_states_exact": all(a["rng_sha256"] == b["rng_sha256"] for a, b in zip(reference, candidate)),
            "all_rng_witnesses_exact": all(a["global_rng_witness"] == b["global_rng_witness"] for a, b in zip(reference, candidate)),
            "first_batch_equal": reference[0]["batch_ids"] == candidate[0]["batch_ids"],
            "first_loss_exact": reference[0]["loss"] == candidate[0]["loss"],
            "first_parameter_state_exact": reference[0]["parameters_sha256"] == candidate[0]["parameters_sha256"]}


def exact_pass(comparison: dict[str, Any]) -> bool:
    return all(comparison[key] for key in ("all_batch_ids_equal", "all_losses_exact", "all_parameter_states_exact",
        "all_optimizer_states_exact", "all_rng_states_exact", "all_rng_witnesses_exact"))


def source_receipts() -> dict[str, str]:
    files = (Path(__file__).resolve(), Path(resume.__file__).resolve(), Path(lm.__file__).resolve(),
             ROOT / "scripts" / "lesson_runtime.py")
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in files}


def export_evidence(source: Path, output: Path) -> None:
    source = source.expanduser().resolve()
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema_version") != SCHEMA:
        raise ValueError("not a Project A reference run")
    for relative, receipt in summary["artifact_manifest"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or not path.is_file() or file_hash(path) != receipt["sha256"]:
            raise ValueError(f"artifact integrity check failed: {relative}")
    output = reserve_output_dir(output, "project-a-evidence")
    for name in ("evidence.json", "step_comparison.csv", "learning_curves.csv"):
        with (source / name).open("rb") as reader, (output / name).open("xb") as writer:
            shutil.copyfileobj(reader, writer)
    write_json(output / "export_receipt.json", {"schema_version": SCHEMA,
        "source_summary_sha256": file_hash(source / "summary.json"),
        "source_run": str(source), "verified_artifacts": len(summary["artifact_manifest"]),
        "scope": "compact JSON/CSV; original worker checkpoints are retained in the source run"})


def run(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    initial_sources = source_receipts()
    train, valid = lm.make_sequences(lm.TRAIN_LENGTHS), lm.make_sequences(lm.VALID_LENGTHS)
    if set(map(tuple, train)) & set(map(tuple, valid)):
        raise AssertionError("full training/validation sequences overlap")
    identity = resume.data_identity(train, valid)
    config = resume.configuration(args.seed, args.steps, args.split_step)
    config["eval_every"] = args.eval_every
    config["optimizer_parameters"]["lr"] = BASE_LR
    config = json.loads(json.dumps(config))  # register JSON container types before workers read them
    write_json(output / "config.resolved.json", config)
    write_json(output / "source_versions.json", initial_sources)
    registration = {"schema_version": SCHEMA, "learning_rates": [OTHER_LR, BASE_LR], "changed_variable": "optimizer.lr",
                    "seed": args.seed, "steps": args.steps, "split_step": args.split_step,
                    "quality_selection": "none; both fixed learning rates reported",
                    "data_identity": identity, "config_sha256": state_hash(config), "source_hashes": initial_sources}
    write_json(output / "preregistration.json", registration)
    reports = {}

    def launch(branch: str, stop: int, *, checkpoint: Path | None = None,
               alternate: bool = False) -> dict[str, Any]:
        branch_config = copy.deepcopy(config)
        if alternate:
            branch_config["optimizer_parameters"]["lr"] = OTHER_LR
        request = {"branch": branch, "config": branch_config, "data_identity": identity, "stop": stop,
                   "output_dir": str(output / branch), "checkpoint": str(checkpoint) if checkpoint else None,
                   "checkpoint_sha256": file_hash(checkpoint) if checkpoint else None}
        request_path = output / f"request_{branch}.json"
        write_json(request_path, request)
        result = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
            "--worker-request", str(request_path)], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=args.worker_timeout)
        write_json(output / f"process_{branch}.json", {"command": result.args, "returncode": result.returncode,
                                                    "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError(f"{branch} worker failed ({result.returncode}): {result.stderr[-2000:]}")
        report = json.loads((output / branch / "trace.json").read_text(encoding="utf-8"))
        reports[branch] = report
        return report

    continuous = launch("continuous", args.steps)
    prefix = launch("prefix", args.split_step)
    split_checkpoint = output / "prefix" / "final.pt"
    split_hash = file_hash(split_checkpoint)
    for name in ("resumed", "missing_optimizer", "missing_batch_generator"):
        launch(name, args.steps, checkpoint=split_checkpoint)
    alternate = launch("alternate_lr", args.steps, alternate=True)
    if file_hash(split_checkpoint) != split_hash:
        raise AssertionError("saved boundary checkpoint was modified")
    first = continuous["rows"][:args.split_step]
    tail = continuous["rows"][args.split_step:]
    comparisons = {"prefix": compare_trace(first, prefix["rows"]),
                   **{name: compare_trace(tail, reports[name]["rows"])
                      for name in ("resumed", "missing_optimizer", "missing_batch_generator")}}
    if not exact_pass(comparisons["prefix"]) or not exact_pass(comparisons["resumed"]):
        raise AssertionError("fresh-process exact resume differs from continuous training")
    wrong_optimizer, wrong_sampler = comparisons["missing_optimizer"], comparisons["missing_batch_generator"]
    if not wrong_optimizer["all_batch_ids_equal"] or not wrong_optimizer["first_loss_exact"] or wrong_optimizer["first_parameter_state_exact"]:
        raise AssertionError("missing-optimizer control did not isolate next-update divergence")
    if wrong_sampler["first_batch_equal"] or wrong_sampler["first_parameter_state_exact"]:
        raise AssertionError("missing-sampler control did not isolate next-batch divergence")
    final_reference = torch.load(output / "continuous" / "final.pt", weights_only=True, map_location="cpu")
    final_resume = torch.load(output / "resumed" / "final.pt", weights_only=True, map_location="cpu")
    final_checks = {key: resume.states_equal(final_reference[key], final_resume[key])
                    for key in ("model_state", "optimizer_state", "rng_state", "processed_tokens", "global_step")}
    if not all(final_checks.values()) or not all(reports["resumed"]["restore_checks"].values()):
        raise AssertionError("fresh-process final optimizer/RNG/weights/cursor are not exact")
    if continuous["initial_parameters_sha256"] != alternate["initial_parameters_sha256"]:
        raise AssertionError("learning-rate branches differ in initialization")
    if any(a["batch_ids"] != b["batch_ids"] or a["valid_tokens"] != b["valid_tokens"]
           for a, b in zip(continuous["rows"], alternate["rows"])):
        raise AssertionError("learning-rate branches differ in batches or token budget")
    alternate_request = json.loads((output / "request_alternate_lr.json").read_text(encoding="utf-8"))
    alternate_config = alternate_request["config"]
    alternate_config["optimizer_parameters"]["lr"] = BASE_LR
    if alternate_config != config:
        raise AssertionError("learning-rate branches differ in another configuration field")
    pids = {report["pid"] for report in reports.values()}
    if os.getpid() in pids or len(pids) != len(reports):
        raise AssertionError("worker processes are not distinct")
    if source_receipts() != initial_sources:
        raise AssertionError("training source changed during the run")
    checks = {"status": "passed", "fresh_process_count": len(pids), "prefix_replay_exact": True,
              "every_resumed_step_exact": True, "final_state_exact": final_checks,
              "missing_optimizer_detected": True, "missing_sampler_detected": True,
              "learning_rate_only_configuration_change": True, "same_initialization_batches_and_tokens": True,
              "source_unchanged": True, "input_checkpoint_unchanged": True,
              "numerical": continuous["numerical_checks"]}
    quality = {name: {"learning_rate": rate, "initial_train": report["learning_curve"][0]["train"],
        "final_train": report["learning_curve"][-1]["train"],
        "initial_validation": report["learning_curve"][0]["validation"],
        "final_validation": report["learning_curve"][-1]["validation"], "generations": report["generations"]}
        for name, rate, report in (("base_lr", BASE_LR, continuous), ("alternate_lr", OTHER_LR, alternate))}
    quality.update({"status": "measured_without_general_language_acceptance",
                    "selection": "both preregistered learning rates retained, no best-result filtering"})
    step_rows = []
    reference_by_step = {row["step"]: row for row in continuous["rows"]}
    for name, report in reports.items():
        for row in report["rows"]:
            reference = reference_by_step[row["step"]]
            step_rows.append({"branch": name, "step": row["step"], "batch_ids": json.dumps(row["batch_ids"]),
                "loss": row["loss"], "valid_tokens": row["valid_tokens"], "processed_tokens": row["processed_tokens"],
                "gradient_norm": row["preclip_gradient_norm"],
                "batch_equal": row["batch_ids"] == reference["batch_ids"],
                "loss_difference": abs(row["loss"] - reference["loss"]),
                **{key: row[key] for key in ("parameters_sha256", "optimizer_sha256", "rng_sha256")},
                **{f"{key}_equal": row[f"{key}_sha256"] == reference[f"{key}_sha256"]
                   for key in ("parameters", "optimizer", "rng")}})
    with (output / "step_comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(step_rows[0]))
        writer.writeheader()
        writer.writerows(step_rows)
    curve_rows = []
    for name, rate, report in (("base_lr", BASE_LR, continuous), ("alternate_lr", OTHER_LR, alternate)):
        for row in report["learning_curve"]:
            curve_rows.append({"branch": name, "learning_rate": rate, "step": row["step"],
                "processed_tokens": row["processed_tokens"], "train_loss": row["train"]["loss"],
                "validation_loss": row["validation"]["loss"], "validation_tokens": row["validation"]["tokens"],
                "validation_token_accuracy": row["validation"]["token_accuracy"]})
    with (output / "learning_curves.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    budgets = {name: report["resources"] for name, report in reports.items()}
    evidence = {"schema_version": SCHEMA, "config": config, "preregistration": registration,
        "environment": {"python": platform.python_version(), "pytorch": str(torch.__version__),
                        "platform": platform.platform(), "device": "cpu", "dtype": "float64",
                        "python_executable": sys.executable, "deterministic_algorithms": True, "threads": 1},
        "mechanism_acceptance": checks, "quality": quality, "comparisons": comparisons,
        "budgets": budgets, "processes": {name: {"pid": report["pid"], "parent_pid": report["parent_pid"]}
                                           for name, report in reports.items()},
        "reproduction": {"script": "transformer/experiments/project_a_reference.py",
                         "arguments": ["--seed", str(args.seed), "--steps", str(args.steps),
                                       "--split-step", str(args.split_step), "--eval-every", str(args.eval_every)]},
        "limitations": ["Synthetic modulo-10 grammar; validation holds out lengths with shared grammar/prefixes.",
            "One seed and CPU float64 software scope; no cross-device bitwise guarantee or GPU measurement.",
            "No dropout, scheduler, AMP or accumulation; saves only at optimizer-step boundaries.",
            "Global RNG is restored and checked; actual minibatches depend on the independent batch Generator.",
            "Natural-text/tokenizer/pretrained-model work is an extension, not completed by this reference.",
            "Learner independent rewrite, explanation, error diagnosis and review remain separate evidence."]}
    write_json(output / "evidence.json", evidence)
    manifest = {path.relative_to(output).as_posix(): {"sha256": file_hash(path), "bytes": path.stat().st_size}
                for path in sorted(output.rglob("*")) if path.is_file()}
    summary = {**evidence, "status": "mechanism_passed", "artifact_manifest": manifest,
               "elapsed_seconds": time.perf_counter() - started,
               "total_executed_updates": sum(report["resources"]["updates"] for report in reports.values()),
               "total_updated_tokens": sum(report["resources"]["updated_tokens"] for report in reports.values()),
               "exit_code": 0}
    write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--split-step", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--worker-timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--export-from", type=Path, help="验证已有运行后导出紧凑 JSON/CSV")
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_request:
        return worker(args.worker_request)
    if not 0 <= args.seed <= 2**63 - 102 or not 2 <= args.steps <= 10000 or not 1 <= args.split_step < args.steps:
        parser.error("seed/steps/split-step out of range; require 1 <= split-step < steps")
    if args.eval_every < 1 or args.worker_timeout < 1:
        parser.error("eval-every and worker-timeout must be positive")
    try:
        if args.export_from:
            if args.output_dir is None:
                parser.error("--export-from requires --output-dir")
            export_evidence(args.export_from, args.output_dir)
            print(f"Verified compact evidence: {args.output_dir.resolve()}")
            return 0
        output = reserve_output_dir(args.output_dir, "project-a-reference")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    try:
        summary = run(args, output)
    except Exception as error:
        write_json(output / "failure.json", {"schema_version": SCHEMA, "status": "failed", "exit_code": 1,
                                            "error_type": type(error).__name__, "error": str(error)})
        print(f"Project A failed; evidence retained: {output}\n{error}", file=sys.stderr)
        return 1
    print(f"Project A mechanism passed: {summary['mechanism_acceptance']['fresh_process_count']} fresh workers; "
          f"every resumed batch/loss/model/optimizer/RNG state exact.")
    print(f"Synthetic grammar quality is separately reported; results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
