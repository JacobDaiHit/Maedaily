"""Dev-only selection and sealed final acceptance for a local character LM.

This quality experiment is independent of the mechanism-only SFT/DPO/RLVR lab.
The pre-registered search uses seed 17 and never generates on final panels until
an immutable selection artifact passes the development gate. Repetition seeds
use exactly the selected architecture and number of updates; seeds are not ranked.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import model_training_lab as model_lab
import model_quality_data as quality_data
from lesson_runtime import reserve_output_dir

SCHEMA = "maedaily-character-quality-workflow-v1"
SELECTION_SEED = 17
CANDIDATES = ({"name": "width32_layers2", "width": 32, "layers": 2},
              {"name": "width64_layers2", "width": 64, "layers": 2},
              {"name": "width64_layers3", "width": 64, "layers": 3})
DEFAULT_STEPS = (200, 500, 1000, 2000, 4000)
DEV_PANELS = ("dev", "dev_copy")
FINAL_PANELS = ("eval_main", "eval_transfer", "eval_regression")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def normalized_config(value: dict[str, Any]) -> dict[str, Any]:
    config = {"seeds": [17], "checkpoint_steps": list(DEFAULT_STEPS), "learning_rate": 0.003,
              "max_new_tokens": 4, "threads": 1, "feature_mode": "plain", **value}
    seeds, steps = config["seeds"], config["checkpoint_steps"]
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63 for seed in seeds):
        raise ValueError("seeds must be nonempty integers between zero and 2**63-1")
    if len(set(seeds)) != len(seeds) or seeds[0] != SELECTION_SEED:
        raise ValueError("seeds must be unique and begin with selection seed 17; seeds are never ranked")
    if not steps or any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps):
        raise ValueError("checkpoint steps must be positive integers")
    if steps != sorted(set(steps)):
        raise ValueError("checkpoint steps must be strictly increasing")
    rate = config["learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    for name in ("max_new_tokens", "threads"):
        if isinstance(config[name], bool) or not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config["max_new_tokens"] > 12:
        raise ValueError("max_new_tokens cannot exceed the short-answer budget 12")
    if config["feature_mode"] != "plain":
        raise ValueError("this workflow implements only the registered plain character architecture")
    return config


def balanced_sample(training: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    pools = {task: [row for row in training if row["task"] == task] for task in ("arithmetic", "copy")}
    if any(not records for records in pools.values()):
        raise ValueError("balanced training needs both arithmetic and copy records")
    rows = [pools[task][rng.randrange(len(pools[task]))] for task in ("arithmetic", "copy") for _ in range(16)]
    rng.shuffle(rows)
    return rows


def optimizer_step(loss: torch.Tensor, model: model_lab.TinyTextLM,
                   optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """A real finite-gradient update; state hashes are recorded at checkpoints."""
    if not loss.requires_grad or not bool(torch.isfinite(loss)):
        raise AssertionError("loss is nonfinite or policy graph is detached")
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if expected != actual:
        raise AssertionError("optimizer parameter set differs from model")
    before = model_lab.parameter_vector(model)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise AssertionError("gradients are absent or nonfinite")
    norm = float(torch.linalg.vector_norm(torch.cat([gradient.flatten() for gradient in gradients])))
    if norm <= 0:
        raise AssertionError("zero model gradient is not an update")
    clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
    optimizer.step()
    after = model_lab.parameter_vector(model)
    if not bool(torch.isfinite(after).all()):
        raise AssertionError("parameters became nonfinite")
    delta = float(torch.linalg.vector_norm(after - before))
    if not delta > 0:
        raise AssertionError("optimizer did not change weights")
    return {"loss": float(loss.detach()), "gradient_norm_before_clip": norm,
            "clip_pre_norm": clip_norm, "parameter_delta_norm": delta}


@torch.no_grad()
def evaluate_panels(model: model_lab.TinyTextLM, tokenizer: model_lab.CharacterTokenizer,
                    panels: dict[str, list[dict[str, Any]]], max_new_tokens: int) -> dict[str, Any]:
    """Evaluate only supplied panels; development callers pass no final data."""
    names = set(panels)
    if names not in (set(DEV_PANELS), set(FINAL_PANELS)):
        raise ValueError("evaluation needs exactly the two dev or three final panels")
    model.eval()
    rows, metrics, nll_sum, token_count = [], {}, 0.0, 0
    for name, records in panels.items():
        if not records:
            raise ValueError("empty evaluation panel")
        outputs = [model_lab.generate(model, tokenizer, record, max_new_tokens) for record in records]
        for output in outputs:
            output["panel"] = name
        metrics[name] = {"correct": sum(row["reward"]["correct"] for row in outputs), "total": len(outputs),
                         "format_valid": sum(row["reward"]["format_valid"] for row in outputs),
                         "generated_tokens": sum(len(row["response_token_ids"]) for row in outputs)}
        metrics[name]["accuracy"] = metrics[name]["correct"] / metrics[name]["total"]
        rows.extend(outputs)
        # Teacher-forced NLL is needed only for development selection. It never
        # enters the final gate or a new selection decision.
        if names == set(DEV_PANELS):
            for start in range(0, len(records), 64):
                batch = model_lab.build_batch(records[start:start + 64], tokenizer, model.config["context"])
                _, logprob, counts = model_lab.response_logprobs(model, batch)
                nll_sum -= float(logprob.sum())
                token_count += int(counts.sum())
    return {"panels": metrics, "predictions": rows,
            "response_nll": nll_sum / token_count if token_count else None,
            "response_tokens": token_count, "decoding": "greedy", "max_new_tokens": max_new_tokens}


def train_candidate(candidate: dict[str, Any], seed: int, steps: list[int],
                    development: dict[str, list[dict[str, Any]]], config: dict[str, Any],
                    output: Path) -> list[dict[str, Any]]:
    """Receive train/dev/dev_copy only, not a full data manifest."""
    if set(development) != {"train", *DEV_PANELS}:
        raise ValueError("training/selection must not receive final panels")
    output.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tokenizer = model_lab.CharacterTokenizer()
    model = model_lab.TinyTextLM(len(tokenizer.pieces), width=candidate["width"], heads=2,
                                layers=candidate["layers"], context=32)
    model.data_profile = "quality-v2"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    initial_hash = model_lab.weight_hash(model)
    records, logs = [], []
    effective_tokens, wall_start = 0, time.perf_counter()
    for step in range(1, max(steps) + 1):
        model.train()
        sample = balanced_sample(development["train"], rng)
        batch = model_lab.build_batch(sample, tokenizer, model.config["context"])
        update = optimizer_step(model_lab.sft_loss(model, batch), model, optimizer)
        tokens = int(batch["loss_mask"].sum())
        effective_tokens += tokens
        logs.append({"step": step, "sample_ids": batch["sample_ids"], "sample_tasks": [row["task"] for row in sample],
                     "samples": 32, "effective_tokens": tokens, "learning_rate": config["learning_rate"], **update})
        if step not in steps:
            continue
        checkpoint = output / f"step_{step:06d}.pt"
        checkpoint_sha = model_lab.save_checkpoint(checkpoint, model, tokenizer, "quality-baseline", optimizer, step)
        evaluation = evaluate_panels(model, tokenizer, {name: development[name] for name in DEV_PANELS},
                                     config["max_new_tokens"])
        dev_metrics = evaluation["panels"]
        key = quality_data.selection_key(dev_metrics, nll=evaluation["response_nll"], step=step)
        receipt = {"candidate": copy.deepcopy(candidate), "seed": seed, "step": step,
                   "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
                   "weight_hash": model_lab.weight_hash(model), "initial_weight_hash": initial_hash,
                   "model_config": copy.deepcopy(model.config), "data_profile": model.data_profile,
                   "dev_metrics": dev_metrics, "dev_nll": evaluation["response_nll"], "selection_key": list(key),
                   "development_gate": quality_data.development_gate(dev_metrics), "last_update": update,
                   "optimizer_steps": step, "training_samples": step * 32, "effective_update_tokens": effective_tokens,
                   "elapsed_seconds": time.perf_counter() - wall_start, "recorded_at_utc": utc_now()}
        write_json(output / f"step_{step:06d}_dev.json", {**receipt, "evaluation": evaluation})
        records.append(receipt)
    write_json(output / "training.json", {"candidate": candidate, "seed": seed, "steps": logs,
               "initial_weight_hash": initial_hash, "final_weight_hash": model_lab.weight_hash(model)})
    return records


def select_development(development: dict[str, list[dict[str, Any]]], config: dict[str, Any],
                       output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(development) != {"train", *DEV_PANELS}:
        raise ValueError("development selection forbids final panels")
    all_records, incumbent = [], None
    for candidate in CANDIDATES:
        records = train_candidate(candidate, SELECTION_SEED, config["checkpoint_steps"], development, config,
                                  output / candidate["name"])
        for record in records:
            all_records.append(record)
            # Exact ties keep the earlier registered candidate. Final data never
            # appears in this key, and changing the seed is not a selection option.
            if incumbent is None or tuple(record["selection_key"]) > tuple(incumbent["selection_key"]):
                incumbent = record
    if incumbent is None:
        raise AssertionError("development search produced no checkpoints")
    return copy.deepcopy(incumbent), all_records


def panel_commitments(panels: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if set(panels) != set(FINAL_PANELS):
        raise ValueError("final commitment needs exactly the three frozen panels")
    return {name: {"sample_count": len(panels[name]),
                   "sample_ids_sha256": digest([row["id"] for row in panels[name]]),
                   "records_sha256": digest(panels[name])} for name in FINAL_PANELS}


def load_checked_checkpoint(receipt: dict[str, Any]) -> tuple[model_lab.TinyTextLM, model_lab.CharacterTokenizer]:
    checkpoint = Path(receipt["checkpoint"])
    if model_lab.file_hash(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("checkpoint changed after development selection")
    model, tokenizer = model_lab.load_model_and_tokenizer(checkpoint)
    if model.data_profile != "quality-v2" or receipt.get("data_profile") != "quality-v2":
        raise ValueError("checkpoint data profile differs from frozen quality-v2 protocol")
    candidate = receipt["candidate"]
    expected = {"vocab_size": len(model_lab.CharacterTokenizer().pieces), "width": candidate["width"],
                "heads": 2, "context": 32, "layers": candidate["layers"]}
    if model.config != expected or receipt.get("model_config") != expected:
        raise ValueError("checkpoint model config differs from selected architecture")
    if tokenizer.to_dict() != model_lab.CharacterTokenizer().to_dict():
        raise ValueError("checkpoint tokenizer differs from frozen character vocabulary")
    if model_lab.weight_hash(model) != receipt["weight_hash"]:
        raise ValueError("loaded weights differ from frozen development receipt")
    return model, tokenizer


def read_frozen_selection(path: Path, protocol_hash: str, expected_file_hash: str) -> dict[str, Any]:
    if model_lab.file_hash(path) != expected_file_hash:
        raise ValueError("selection artifact changed after freezing")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("schema") != SCHEMA or selection.get("protocol_sha256") != protocol_hash:
        raise ValueError("selection protocol mismatch")
    chosen = selection["selected"]
    recomputed = quality_data.development_gate(chosen["dev_metrics"])
    if not recomputed["qualified"] or not chosen["development_gate"]["qualified"]:
        raise ValueError("unqualified development selection cannot open final evaluation")
    load_checked_checkpoint(chosen)
    return selection


def open_final_evaluation(selection_path: Path, protocol_hash: str, selection_hash: str,
                          final_panels: dict[str, list[dict[str, Any]]], checkpoint: dict[str, Any],
                          config: dict[str, Any]) -> dict[str, Any]:
    selection = read_frozen_selection(selection_path, protocol_hash, selection_hash)
    selected = selection["selected"]
    if checkpoint["candidate"] != selected["candidate"] or checkpoint["step"] != selected["step"]:
        raise ValueError("replication architecture or budget differs from frozen selection")
    if not quality_data.development_gate(checkpoint["dev_metrics"])["qualified"]:
        raise ValueError("unqualified replication development cannot open final")
    model, tokenizer = load_checked_checkpoint(checkpoint)
    if panel_commitments(final_panels) != selection["final_panel_commitments"]:
        raise ValueError("final sample IDs or records differ from pre-registered frozen panels")
    evaluation = evaluate_panels(model, tokenizer, final_panels, config["max_new_tokens"])
    return {"evaluation": evaluation, "quality_gate": quality_data.quality_gate(evaluation),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"], "selection_sha256": selection_hash,
            "evaluated_at_utc": utc_now()}


def exit_code(summary: dict[str, Any]) -> int:
    return 0 if summary["status"] == "qualified" else 4


def run(value: dict[str, Any], requested_output: Path | None = None) -> dict[str, Any]:
    config = normalized_config(value)
    output = reserve_output_dir(requested_output, "model-quality")
    torch.set_num_threads(config["threads"])
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    def event(name: str, **details: Any) -> None:
        events.append({"sequence": len(events), "event": name, "at_utc": utc_now(), **details})
    data = quality_data.fixed_quality_data()
    protocol = quality_data.protocol_manifest(data)
    protocol_hash = digest(protocol)
    final_commitments = panel_commitments({name: data[name] for name in FINAL_PANELS})
    # A protocol audit is fixed before training. Development functions receive a
    # separate dictionary with no final keys or final examples.
    development = {name: data[name] for name in ("train", *DEV_PANELS)}
    registration = {"schema": SCHEMA, "created_at_utc": utc_now(), "config": config,
                    "protocol_sha256": protocol_hash, "selection_seed": SELECTION_SEED,
                    "final_panel_commitments": final_commitments,
                    "candidates": list(CANDIDATES), "checkpoint_steps": config["checkpoint_steps"],
                    "batch": {"arithmetic": 16, "copy": 16}, "loss": "response-token CE including EOS",
                    "optimizer": "AdamW", "gradient_clip_norm": 1.0,
                    "seed_selection": "forbidden; every registered seed is reported",
                    "final_policy": "one evaluation per seed after qualified immutable dev selection; no retuning",
                    "source_sha256": {path.name: model_lab.file_hash(path) for path in
                                      (Path(__file__), HERE / "model_quality_data.py", HERE / "model_training_lab.py")}}
    write_json(output / "preregistration.json", registration)
    write_json(output / "protocol.json", protocol)
    event("preregistered", protocol_sha256=protocol_hash)
    selected, development_records = select_development(development, config, output / "development")
    selection = {"schema": SCHEMA, "frozen_at_utc": utc_now(), "protocol_sha256": protocol_hash,
                 "selection_seed": SELECTION_SEED, "selected": selected,
                 "allowed_panels": list(DEV_PANELS), "development_records": development_records,
                 "final_panel_commitments": final_commitments}
    selection_path = output / "selection.json"
    write_json(selection_path, selection)
    selection_hash = model_lab.file_hash(selection_path)
    event("selection_frozen", selection_sha256=selection_hash, checkpoint_sha256=selected["checkpoint_sha256"])
    selected_model, selected_tokenizer = load_checked_checkpoint(selected)
    training_outputs = [model_lab.generate(selected_model, selected_tokenizer, row, config["max_new_tokens"])
                        for row in development["train"]]
    training_diagnostic = {"scope": "read-only selected-checkpoint train diagnosis; excluded from selection",
                           "checkpoint_sha256": selected["checkpoint_sha256"],
                           "tasks": {task: {"correct": sum(result["reward"]["correct"] for row, result in
                                      zip(development["train"], training_outputs) if row["task"] == task),
                                      "total": sum(row["task"] == task for row in development["train"])}
                                     for task in ("arithmetic", "copy")}, "predictions": training_outputs}
    write_json(output / "training_diagnostic.json", training_diagnostic)
    event("selected_training_diagnosed", excluded_from_selection=True)
    seed_reports, final_metrics, executed_seeds = [], None, [SELECTION_SEED]
    if selected["development_gate"]["qualified"]:
        read_frozen_selection(selection_path, protocol_hash, selection_hash)
        final_metrics = {}
        for seed in config["seeds"]:
            if seed == SELECTION_SEED:
                receipt = selected
            else:
                # Only the already frozen architecture/step is trained. The new
                # seed has a dev qualification check, not a new checkpoint search.
                executed_seeds.append(seed)
                receipt = train_candidate(selected["candidate"], seed, [selected["step"]], development, config,
                                          output / "replications" / f"seed_{seed}")[0]
            final_result = None
            if receipt["development_gate"]["qualified"]:
                event("final_opened", seed=seed, selection_sha256=selection_hash)
                final_result = open_final_evaluation(selection_path, protocol_hash, selection_hash,
                                                    {name: data[name] for name in FINAL_PANELS}, receipt, config)
                write_json(output / f"seed_{seed}_final.json", final_result)
                final_metrics[str(seed)] = final_result["evaluation"]["panels"]
                event("final_completed", seed=seed, qualified=final_result["quality_gate"]["qualified"])
            else:
                final_metrics[str(seed)] = None
                event("replication_development_unqualified", seed=seed)
            seed_reports.append({"seed": seed, "candidate": receipt["candidate"], "step": receipt["step"],
                                 "checkpoint_sha256": receipt["checkpoint_sha256"], "dev_metrics": receipt["dev_metrics"],
                                 "development_gate": receipt["development_gate"],
                                 "final_gate": None if final_result is None else final_result["quality_gate"],
                                 "qualified": final_result is not None and final_result["quality_gate"]["qualified"],
                                 "optimizer_steps": receipt["optimizer_steps"],
                                 "effective_update_tokens": receipt["effective_update_tokens"]})
        status = "qualified" if all(row["qualified"] for row in seed_reports) else "quality_unqualified"
    else:
        status = "development_unqualified"
        # No final panels are supplied to an evaluator. All registered seeds are
        # explicitly visible, including those not run after the failed search.
        seed_reports = [{"seed": seed, "status": "selection_development_unqualified" if seed == 17
                         else "not_run_selection_development_unqualified", "qualified": False,
                         "final_gate": None} for seed in config["seeds"]]
        event("development_unqualified_final_sealed")
    failed_seeds = [row["seed"] for row in seed_reports if not row["qualified"]]
    environment = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
                   "platform": platform.platform(), "device": "cpu", "threads": torch.get_num_threads()}
    summary = {"schema_version": SCHEMA, "lesson": "model-quality", "status": status,
               "output_dir": str(output), "config": config, "protocol_sha256": protocol_hash,
               "selection_sha256": selection_hash, "selection": selected, "seed_reports": seed_reports,
               "registered_seeds": config["seeds"], "planned_seeds": config["seeds"],
               "executed_seeds": executed_seeds, "failed_seeds": failed_seeds, "final_metrics": final_metrics,
               "training_diagnostic": {key: value for key, value in training_diagnostic.items() if key != "predictions"},
               "environment": environment, "elapsed_seconds": time.perf_counter() - started,
               "final_used_for_selection": False, "all_registered_seeds_qualified": not failed_seeds,
               "limitations": ["Local character model, not a general pretrained LLM.",
                               "Transfer is reported separately and is not an acceptance condition.",
                               "Short checkpoint schedules are workflow checks, not evidence of baseline quality.",
                               "Final failure never triggers a new architecture, budget, checkpoint or seed selection."]}
    summary["exit_code"] = exit_code(summary)
    write_json(output / "events.json", events)
    write_json(output / "summary.json", summary)
    write_json(output / "environment.json", environment)
    lines = ["# 字符模型质量基线结果", "", f"状态：`{status}`；退出码：`{summary['exit_code']}`。",
             f"开发选择种子固定为17；选定结构：`{selected['candidate']['name']}`，更新{selected['step']}步。",
             "", "选择仅使用 dev/dev_copy；selection.json 落盘后才允许打开最终面板。",
             "全部注册种子均出现在报告中，未按最终评测挑选种子。", ""]
    if final_metrics is None:
        lines.append("开发门槛未通过；最终主任务、迁移、回归面板保持封闭，final_metrics=null。")
    else:
        for row in seed_reports:
            lines.append(f"种子 {row['seed']}：开发通过={row['development_gate']['qualified']}；最终通过={row['qualified']}。")
        lines.append("最终失败保持原配置并明确报告，没有回头调参。")
    with (output / "结果说明.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        normalized_config({key: value for key, value in vars(args).items() if key != "output_dir"})
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = {key: value for key, value in vars(args).items() if key != "output_dir"}
    try:
        summary = run(config, args.output_dir)
    except (ValueError, AssertionError) as error:
        print(f"质量实验失败：{error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": summary["status"], "output_dir": summary["output_dir"],
                      "selected": summary["selection"]["candidate"], "steps": summary["selection"]["step"],
                      "failed_seeds": summary["failed_seeds"], "final_metrics": summary["final_metrics"]},
                     ensure_ascii=False, indent=2))
    return exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
