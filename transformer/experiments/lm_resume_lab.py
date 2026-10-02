"""Compare uninterrupted tiny LM training with a full checkpoint resume on CPU.

Uses only an existing PyTorch installation and Python's standard library.
The saved checkpoint contains tensors and primitive containers, so the lab
reloads its own files with weights_only=True. No downloads or GPU are needed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("此解释器没有 PyTorch；请选择现有 PyTorch 环境，本实验不安装依赖。") from error

from tiny_causal_lm import (
    BOS, EOS, IGNORE_INDEX, MODEL_CONFIG, PAD, TRAIN_LENGTHS, VALID_LENGTHS,
    VOCAB_SIZE, TinyCausalLM, collate, evaluate, inputs_and_targets,
    make_sequences, summed_loss,
)


SCHEMA_VERSION = 1
DTYPE = torch.float64
BATCH_SIZE = 16
CLIP_NORM = 1.0
OPTIMIZER_CONFIG = {
    "lr": 0.003, "betas": (0.9, 0.999), "eps": 1e-8,
    "weight_decay": 0.01, "amsgrad": False, "foreach": False, "fused": False,
    "maximize": False, "capturable": False, "differentiable": False,
}


def parameter_vector(model: TinyCausalLM) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).clone()


def new_objects(seed: int) -> tuple[TinyCausalLM, torch.optim.AdamW, torch.Generator]:
    random.seed(seed)
    torch.manual_seed(seed)
    model = TinyCausalLM(**MODEL_CONFIG).to(device="cpu", dtype=DTYPE)
    optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER_CONFIG)
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    return model, optimizer, batch_generator


def rng_state(batch_generator: torch.Generator) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
        "batch_generator": batch_generator.get_state().clone(),
    }


def rng_witness(state: dict[str, object]) -> dict[str, object]:
    """Probe cloned streams without consuming the actual training streams."""
    python_probe = random.Random()
    python_probe.setstate(state["python"])
    torch_probe = torch.Generator(device="cpu")
    torch_probe.set_state(state["torch_cpu"])
    return {
        "python_next_draws": [python_probe.random() for _ in range(4)],
        "torch_next_draws": torch.rand(4, generator=torch_probe, dtype=DTYPE).tolist(),
    }


def states_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(states_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(states_equal(a, b) for a, b in zip(left, right))
    return left == right


def data_identity(train: list[list[int]], valid: list[list[int]]) -> dict[str, object]:
    token_schema = {"pad": PAD, "bos": BOS, "eos": EOS, "digits_start": 3, "vocab_size": VOCAB_SIZE}
    ordered_data = {"tokens": token_schema, "train": train, "validation": valid}
    digest = hashlib.sha256(json.dumps(ordered_data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "sha256": digest, "hash_encoding": "UTF-8 canonical JSON with ordered sequences",
        "token_schema": token_schema, "train_digit_lengths": list(TRAIN_LENGTHS),
        "validation_digit_lengths": list(VALID_LENGTHS),
        "train_sequences": len(train), "validation_sequences": len(valid),
    }


def configuration(seed: int, steps: int, split_step: int) -> dict[str, object]:
    return {
        "seed": seed, "steps": steps, "split_step": split_step,
        "model": dict(MODEL_CONFIG), "vocab_size": VOCAB_SIZE,
        "layers": 1, "normalization": "pre-LN", "activation": "GELU", "dropout": 0.0,
        "dtype": "float64", "device": "cpu", "batch_size": BATCH_SIZE,
        "optimizer": "AdamW", "optimizer_parameters": dict(OPTIMIZER_CONFIG),
        "clip_gradient_norm": CLIP_NORM, "loss": "sum NLL / valid target tokens",
        "sampler": "torch.randint with replacement from independent CPU Generator",
        "batch_generator_initial_seed": seed + 1,
    }


def save_checkpoint(
    path: Path, model: TinyCausalLM, optimizer: torch.optim.AdamW,
    batch_generator: torch.Generator, config: dict[str, object],
    identity: dict[str, object], global_step: int, processed_tokens: int,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION, "experiment": "tiny_causal_lm_resume",
        "config": config, "data_identity": identity, "global_step": global_step,
        "processed_tokens": processed_tokens, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "rng_state": rng_state(batch_generator),
    }
    # Exclusive creation protects existing artifacts even if another run races us.
    with path.open("xb") as stream:
        torch.save(payload, stream)


def validate_metadata(payload: dict[str, object], config: dict[str, object], identity: dict[str, object]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("experiment") != "tiny_causal_lm_resume":
        raise ValueError("checkpoint schema/experiment 不匹配")
    if payload.get("config") != config:
        raise ValueError("checkpoint 训练配置或模型超参数不匹配")
    if payload.get("data_identity") != identity:
        raise ValueError("checkpoint 数据摘要、顺序或 token 定义不匹配")
    step = payload.get("global_step")
    tokens = payload.get("processed_tokens")
    if type(step) is not int or not 0 <= step <= config["steps"]:
        raise ValueError("checkpoint global_step 无效")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("checkpoint processed_tokens 无效")


def restore_checkpoint(
    path: Path, config: dict[str, object], identity: dict[str, object], *,
    omit_optimizer: bool = False, omit_batch_generator: bool = False,
) -> tuple[TinyCausalLM, torch.optim.AdamW, torch.Generator, dict[str, object], dict[str, bool]]:
    # A genuinely new random initialization consumes torch RNG before loading.
    # It is intentionally seeded differently from the original model.
    rebuild_seed = (config["seed"] + 101) % (2**63 - 1)
    model, optimizer, batch_generator = new_objects(rebuild_seed)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_metadata(payload, config, identity)
    new_initialization_differs = not torch.equal(parameter_vector(model), torch.cat([
        payload["model_state"][name].reshape(-1) for name, _ in model.named_parameters()
    ]))
    model.load_state_dict(payload["model_state"], strict=True)
    if not omit_optimizer:
        optimizer.load_state_dict(payload["optimizer_state"])
    # Restore after reconstruction; otherwise initialization advances this RNG.
    random.setstate(payload["rng_state"]["python"])
    torch.set_rng_state(payload["rng_state"]["torch_cpu"])
    if omit_batch_generator:
        # The mistake is restarting the original sampler from its seed.
        batch_generator.manual_seed(config["batch_generator_initial_seed"])
    else:
        batch_generator.set_state(payload["rng_state"]["batch_generator"])
    current_rng = rng_state(batch_generator)
    return model, optimizer, batch_generator, payload, {
        "new_initialization_differs": new_initialization_differs,
        "python_rng_restored": states_equal(current_rng["python"], payload["rng_state"]["python"]),
        "torch_rng_restored": torch.equal(current_rng["torch_cpu"], payload["rng_state"]["torch_cpu"]),
        "batch_generator_restored": torch.equal(current_rng["batch_generator"], payload["rng_state"]["batch_generator"]),
        "python_and_torch_next_draws_equal": rng_witness(current_rng) == rng_witness(payload["rng_state"]),
    }


def train_segment(
    model: TinyCausalLM, optimizer: torch.optim.AdamW, batch_generator: torch.Generator,
    train: list[list[int]], start: int, stop: int,
) -> tuple[list[dict[str, object]], dict[int, torch.Tensor]]:
    rows, parameter_snapshots = [], {}
    for step in range(start + 1, stop + 1):
        model.train()
        indices = torch.randint(len(train), (BATCH_SIZE,), generator=batch_generator).tolist()
        inputs, targets = inputs_and_targets(collate([train[index] for index in indices]))
        valid_tokens = targets.ne(IGNORE_INDEX).sum().item()
        optimizer.zero_grad(set_to_none=True)
        loss = summed_loss(model(inputs), targets) / valid_tokens
        if not bool(torch.isfinite(loss)):
            raise AssertionError(f"step {step}: training loss 非有限")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise AssertionError(f"step {step}: 梯度缺失或非有限 {name}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM, error_if_nonfinite=True).item()
        optimizer.step()
        vector = parameter_vector(model)
        if not bool(torch.isfinite(vector).all()):
            raise AssertionError(f"step {step}: 更新后参数非有限")
        rows.append({"step": step, "batch_ids": indices, "loss": loss.item(), "valid_tokens": valid_tokens, "preclip_gradient_norm": grad_norm})
        parameter_snapshots[step] = vector
    return rows, parameter_snapshots


def compare_branch(
    reference_rows: list[dict[str, object]], reference_parameters: dict[int, torch.Tensor],
    reference_validation: dict[str, object], rows: list[dict[str, object]],
    parameters: dict[int, torch.Tensor], validation: dict[str, object],
) -> dict[str, object]:
    if len(reference_rows) != len(rows) or not rows:
        raise AssertionError("续训对照步数不匹配")
    loss_differences = [abs(left["loss"] - right["loss"]) for left, right in zip(reference_rows, rows)]
    parameter_differences = [
        (parameters[row["step"]] - reference_parameters[row["step"]]).abs().max().item() for row in rows
    ]
    return {
        "first_batch_equal": reference_rows[0]["batch_ids"] == rows[0]["batch_ids"],
        "all_batch_ids_equal": all(left["batch_ids"] == right["batch_ids"] for left, right in zip(reference_rows, rows)),
        "reference_first_batch_ids": reference_rows[0]["batch_ids"], "first_batch_ids": rows[0]["batch_ids"],
        "first_step_loss_abs_difference": loss_differences[0], "max_loss_abs_difference": max(loss_differences),
        "next_step_parameter_max_abs_difference": parameter_differences[0],
        "max_step_parameter_abs_difference": max(parameter_differences),
        "final_parameter_max_abs_difference": parameter_differences[-1],
        "final_validation_loss_abs_difference": abs(reference_validation["loss"] - validation["loss"]),
        "final_validation_equal": reference_validation == validation,
    }


def reserve_output_dir(parser: argparse.ArgumentParser, requested: Path | None) -> Path:
    if requested is None:
        root = Path(__file__).resolve().parents[2] / "study_runs"
        requested = root / ("lm_resume_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output_dir = requested.resolve()
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            parser.error(f"--output-dir 必须是新目录或空目录，拒绝覆盖: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--split-step", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if not 0 <= args.seed <= 2**63 - 2:
        parser.error("--seed 必须在 0 到 2**63-2 之间")
    if args.steps < 2 or not 1 <= args.split_step < args.steps:
        parser.error("要求 --steps >= 2 且 1 <= --split-step < --steps")
    output_dir = reserve_output_dir(parser, args.output_dir)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    train, valid = make_sequences(TRAIN_LENGTHS), make_sequences(VALID_LENGTHS)
    if set(map(tuple, train)) & set(map(tuple, valid)):
        raise AssertionError("训练与验证完整序列重叠")
    identity = data_identity(train, valid)
    config = configuration(args.seed, args.steps, args.split_step)
    split_path = output_dir / f"checkpoint_step{args.split_step}.pt"

    # Reference branch is uninterrupted: no save/rebuild between its segments.
    continuous, continuous_optimizer, continuous_generator = new_objects(args.seed)
    initial_parameters = parameter_vector(continuous)
    initial_validation = evaluate(continuous, valid)
    before_rows, before_parameters = train_segment(continuous, continuous_optimizer, continuous_generator, train, 0, args.split_step)
    reference_split_parameters = parameter_vector(continuous)
    reference_split_rng = rng_state(continuous_generator)
    tail_rows, tail_parameters = train_segment(continuous, continuous_optimizer, continuous_generator, train, args.split_step, args.steps)
    continuous_validation = evaluate(continuous, valid)

    # Independent prefix branch must reach exactly the same save boundary.
    interrupted, interrupted_optimizer, interrupted_generator = new_objects(args.seed)
    prefix_rows, _ = train_segment(interrupted, interrupted_optimizer, interrupted_generator, train, 0, args.split_step)
    prefix_tokens = sum(row["valid_tokens"] for row in prefix_rows)
    if prefix_rows != before_rows or not torch.equal(parameter_vector(interrupted), reference_split_parameters):
        raise AssertionError("独立前缀训练与连续分支不一致")
    if not states_equal(rng_state(interrupted_generator), reference_split_rng):
        raise AssertionError("保存边界的随机状态不一致")
    save_checkpoint(split_path, interrupted, interrupted_optimizer, interrupted_generator, config, identity, args.split_step, prefix_tokens)

    comparisons, branches, restore_checks, csv_rows = {}, {}, {}, []
    reference_parameters = before_parameters | tail_parameters
    continuous_rows = before_rows + tail_rows
    branches["continuous"] = {"initial_validation": initial_validation, "final_validation": continuous_validation, "processed_tokens": sum(row["valid_tokens"] for row in continuous_rows)}
    for row in continuous_rows:
        csv_rows.append({"branch": "continuous", **row, "reference_loss_abs_difference": 0.0, "reference_parameter_max_abs_difference": 0.0, "reference_batch_equal": True})

    for branch, omissions in (
        ("resumed", {}), ("missing_optimizer", {"omit_optimizer": True}),
        ("missing_batch_generator", {"omit_batch_generator": True}),
    ):
        model, optimizer, generator, payload, restored_checks = restore_checkpoint(split_path, config, identity, **omissions)
        if not torch.equal(parameter_vector(model), reference_split_parameters):
            raise AssertionError(f"{branch}: checkpoint 权重未准确恢复")
        start_validation = evaluate(model, valid)
        rows, parameters = train_segment(model, optimizer, generator, train, payload["global_step"], args.steps)
        final_validation = evaluate(model, valid)
        comparisons["complete_resume" if branch == "resumed" else branch] = compare_branch(
            tail_rows, tail_parameters, continuous_validation, rows, parameters, final_validation,
        )
        branches[branch] = {
            "resume_start_step": payload["global_step"], "initial_validation": start_validation,
            "final_validation": final_validation,
            "processed_tokens": payload["processed_tokens"] + sum(row["valid_tokens"] for row in rows),
        }
        restore_checks[branch] = restored_checks
        for reference, row in zip(tail_rows, rows):
            csv_rows.append({
                "branch": branch, **row, "reference_loss_abs_difference": abs(reference["loss"] - row["loss"]),
                "reference_parameter_max_abs_difference": (parameters[row["step"]] - reference_parameters[row["step"]]).abs().max().item(),
                "reference_batch_equal": reference["batch_ids"] == row["batch_ids"],
            })
        if branch == "resumed":
            final_optimizer_equal = states_equal(optimizer.state_dict(), continuous_optimizer.state_dict())
            final_rng_equal = states_equal(rng_state(generator), rng_state(continuous_generator))
            save_checkpoint(output_dir / "checkpoint_final.pt", model, optimizer, generator, config, identity, args.steps, branches[branch]["processed_tokens"])

    complete = comparisons["complete_resume"]
    exact_fields = (
        "first_step_loss_abs_difference", "max_loss_abs_difference", "next_step_parameter_max_abs_difference",
        "max_step_parameter_abs_difference", "final_parameter_max_abs_difference", "final_validation_loss_abs_difference",
    )
    if not all(complete[field] == 0.0 for field in exact_fields) or not complete["all_batch_ids_equal"] or not complete["final_validation_equal"]:
        raise AssertionError("完整恢复未与连续训练逐步严格一致")
    if not all(restore_checks["resumed"].values()) or not final_optimizer_equal or not final_rng_equal:
        raise AssertionError("完整恢复的对象重建、优化器或随机状态检查失败")
    wrong_optimizer = comparisons["missing_optimizer"]
    if not wrong_optimizer["all_batch_ids_equal"] or wrong_optimizer["first_step_loss_abs_difference"] != 0.0 or wrong_optimizer["next_step_parameter_max_abs_difference"] <= 1e-12:
        raise AssertionError("遗漏优化器的错误对照未在下一步产生可辨参数差异")
    wrong_generator = comparisons["missing_batch_generator"]
    if wrong_generator["first_batch_equal"] or wrong_generator["next_step_parameter_max_abs_difference"] <= 1e-12:
        raise AssertionError("遗漏批次 Generator 的错误对照未产生可辨采样和参数差异")
    trained_parameter_change = (parameter_vector(continuous) - initial_parameters).abs().max().item()
    if trained_parameter_change <= 1e-12:
        raise AssertionError("模型参数没有实际更新")

    summary = {
        "experiment": "tiny_causal_lm_resume", "run_timestamp": datetime.now().astimezone().isoformat(),
        "environment": {"python": platform.python_version(), "python_executable": sys.executable, "pytorch": str(torch.__version__), "platform": platform.platform(), "device": "cpu", "dtype": "float64", "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": True},
        "config": config, "data": {**identity, "full_sequence_overlap": 0, "scope": "synthetic shared digit grammar; held-out lengths 7/9 within training length range 6/8/10"},
        "parameter_count": initial_parameters.numel(), "trained_parameter_max_abs_change": trained_parameter_change,
        "branches": branches, "comparisons": comparisons, "restore_checks": restore_checks,
        "checks": {"status": "passed", "complete_resume_exact": True, "prefix_replay_exact": True, "final_optimizer_state_equal": final_optimizer_equal, "final_rng_state_equal": final_rng_equal, "missing_optimizer_detected": True, "missing_batch_generator_detected": True, "all_losses_gradients_parameters_finite": True},
        "artifacts": {"split_checkpoint": split_path.name, "final_checkpoint": "checkpoint_final.pt", "comparison_csv": "resume_comparison.csv", "summary": "summary.json"},
        "experiment_seconds": time.perf_counter() - started,
        "scope": "Same-process disk save/rebuild/load simulation on fixed CPU/software; no cross-device bitwise claim, GPU performance, distributed loader, natural-text benchmark, scheduler, AMP, or dropout training validation.",
        "rng_scope": "TinyCausalLM has random initialization but no dropout. Python/torch global streams are restored and their next draws checked; minibatches actually depend on the independent Generator state.",
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with (output_dir / "resume_comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({**row, "batch_ids": json.dumps(row["batch_ids"])})
    print(f"CPU float64；参数 {initial_parameters.numel()}；连续 {args.steps} 步，对照 {args.split_step}+{args.steps - args.split_step} 步")
    print(f"完整恢复：首批 IDs 相同，每步 loss / 下一步参数 / 最终参数 / 验证 loss 的最大差均为 0")
    print(f"遗漏 AdamW 状态：下一步参数最大差 {wrong_optimizer['next_step_parameter_max_abs_difference']:.9g}")
    print(f"遗漏批次 Generator 状态：首批 IDs 不同，下一步参数最大差 {wrong_generator['next_step_parameter_max_abs_difference']:.9g}")
    print(f"验证 loss：{initial_validation['loss']:.9f} -> {continuous_validation['loss']:.9f}")
    print(f"全部检查通过；持久检查点与结果：{output_dir}")


if __name__ == "__main__":
    main()
