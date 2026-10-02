"""PyTorch training-mechanics lab, tested with PyTorch 2.7.1 on CPU.

Uses an existing PyTorch installation and the Python standard library only.
No datasets, models, dependencies, or other network resources are downloaded.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import sys

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit(
        "PyTorch is not installed in this interpreter. Use an existing environment; "
        "see ../README.md. This script does not install dependencies."
    ) from error


DTYPE = torch.float64
DEVICE = torch.device("cpu")
LEARNING_RATE = 0.05
MOMENTUM = 0.9
SEED = 7
STEPS = 800
SPLIT_STEP = 40


def make_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [(a, b) for a in (-2.0, -1.0, 0.0, 1.0, 2.0)
         for b in (-1.5, -0.5, 0.5, 1.5)], dtype=DTYPE, device=DEVICE)
    x_valid = torch.tensor(
        [[-1.75, -0.25], [0.125, 0.75], [1.25, -1.125], [2.5, 0.25]],
        dtype=DTYPE, device=DEVICE)

    def target(inputs: torch.Tensor) -> torch.Tensor:
        return 2.0 * inputs[:, :1] - 3.0 * inputs[:, 1:2] + 0.5

    return x, target(x), x_valid, target(x_valid)


def gradient_checks(x: torch.Tensor, y: torch.Tensor) -> dict:
    theta = torch.tensor([0.3, -0.2, 0.1], dtype=DTYPE, device=DEVICE, requires_grad=True)
    design = torch.cat([x, torch.ones((len(x), 1), dtype=DTYPE, device=DEVICE)], dim=1)

    def objective(parameters: torch.Tensor) -> torch.Tensor:
        residual = design @ parameters - y[:, 0]
        return residual.square().mean()

    loss = objective(theta)
    auto, = torch.autograd.grad(loss, theta)
    with torch.no_grad():
        residual = design @ theta - y[:, 0]
        manual = (2.0 / len(x)) * design.T @ residual
        h = 1e-6
        finite = torch.empty_like(theta)
        for j in range(theta.numel()):
            offset = torch.zeros_like(theta)
            offset[j] = h
            finite[j] = (objective(theta + offset) - objective(theta - offset)) / (2.0 * h)
    manual_error = (auto - manual).abs().max().item()
    finite_error = (auto - finite).abs().max().item()
    relative_error = finite_error / max(1.0, auto.abs().max().item(), finite.abs().max().item())
    if manual_error > 1e-12 or finite_error > 1e-7:
        raise AssertionError("Gradient check failed against the independent formulas.")

    # A silent broadcast failure: [3,1] minus [3] produces [3,3].
    pred = torch.tensor([[1.0], [2.0], [3.0]], dtype=DTYPE)
    target = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    wrong = (pred - target).square().mean().item()
    right = (pred[:, 0] - target).square().mean().item()
    if abs(wrong - 4.0 / 3.0) > 1e-12 or right != 0.0:
        raise AssertionError("Broadcasting oracle mismatch.")

    # Rebuild the graph each time, intentionally accumulating leaf gradients.
    w = torch.tensor([1.0], dtype=DTYPE, requires_grad=True)
    for _ in range(2):
        (((2.0 * w - 5.0) ** 2).sum() / 2.0).backward()
    accumulated = w.grad.item()
    if accumulated != -12.0:
        raise AssertionError("Expected two separate backward calls to accumulate gradients.")

    logits = torch.zeros((1, 2), dtype=DTYPE, requires_grad=True)
    ce = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
    ce.backward()
    if not torch.equal(logits.grad, torch.tensor([[-0.5, 0.5]], dtype=DTYPE)):
        raise AssertionError("Cross-entropy gradient disagrees with p - one_hot(y).")
    return {
        "loss_at_probe": loss.item(), "autograd": auto.tolist(),
        "analytic": manual.tolist(), "finite_difference": finite.tolist(),
        "finite_difference_step": h, "max_analytic_error": manual_error,
        "max_finite_difference_error": finite_error, "relative_finite_difference_error": relative_error,
        "broadcast_wrong_loss": wrong, "broadcast_correct_loss": right,
        "accumulated_scalar_gradient": accumulated, "binary_uniform_ce": ce.item(),
        "binary_ce_gradient": logits.grad.tolist(), "status": "passed",
    }


def new_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Linear(2, 1, dtype=DTYPE, device=DEVICE)


def new_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return (model(x) - y).square().mean().item()


def train(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    start: int, stop: int, record: bool = True,
) -> list[dict]:
    x, y, x_valid, y_valid = data
    rows = []
    if record:
        rows.append({"step": start, "train_mse": evaluate(model, x, y),
                     "validation_mse": evaluate(model, x_valid, y_valid), "pre_update_grad_norm": None})
    for step in range(start + 1, stop + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        if pred.shape != y.shape:
            raise AssertionError(f"Prediction/label shape mismatch: {pred.shape}, {y.shape}")
        loss = (pred - y).square().mean()
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        grad_square_sum = 0.0
        for parameter in model.parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise AssertionError("Every parameter of this linear model must have a finite gradient.")
            grad_square_sum += parameter.grad.square().sum().item()
        optimizer.step()
        if record and (step == 1 or step % 50 == 0 or step == stop):
            rows.append({"step": step, "train_mse": evaluate(model, x, y),
                         "validation_mse": evaluate(model, x_valid, y_valid),
                         "pre_update_grad_norm": grad_square_sum ** 0.5})
    return rows


def parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([model.weight.detach().reshape(-1), model.bias.detach()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    data = make_data()
    checks = gradient_checks(data[0], data[1])

    full_model = new_model(SEED)
    full_optimizer = new_optimizer(full_model)
    curve = train(full_model, full_optimizer, data, 0, STEPS)
    analytic = torch.tensor([2.0, -3.0, 0.5], dtype=DTYPE)
    parameter_error = (parameter_vector(full_model) - analytic).abs().max().item()
    final_train_mse = evaluate(full_model, data[0], data[1])
    final_valid_mse = evaluate(full_model, data[2], data[3])
    if parameter_error > 1e-6 or final_train_mse > 1e-12 or final_valid_mse > 1e-12:
        raise AssertionError("The learned linear model does not match the analytic target.")

    split_model = new_model(SEED)
    split_optimizer = new_optimizer(split_model)
    train(split_model, split_optimizer, data, 0, SPLIT_STEP, record=False)
    checkpoint_path = args.output_dir / f"checkpoint_step{SPLIT_STEP}.pt"
    torch.save({
        "model_state": split_model.state_dict(), "optimizer_state": split_optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(), "step": SPLIT_STEP,
        "seed": SEED, "learning_rate": LEARNING_RATE, "momentum": MOMENTUM,
    }, checkpoint_path)

    restored = new_model(SEED + 1)
    restored_optimizer = new_optimizer(restored)
    # The file was created by this run; weights_only avoids general object unpickling.
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    restored.load_state_dict(checkpoint["model_state"])
    restored_optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    restored.eval()
    split_model.eval()
    with torch.no_grad():
        reload_error = (restored(data[2]) - split_model(data[2])).abs().max().item()
    # Check the very next update, before eventual convergence could hide a bad resume.
    train(split_model, split_optimizer, data, SPLIT_STEP, SPLIT_STEP + 1, record=False)
    train(restored, restored_optimizer, data, SPLIT_STEP, SPLIT_STEP + 1, record=False)
    next_step_error = (parameter_vector(restored) - parameter_vector(split_model)).abs().max().item()
    weights_only_model = new_model(SEED + 2)
    weights_only_model.load_state_dict(checkpoint["model_state"])
    weights_only_optimizer = new_optimizer(weights_only_model)
    train(weights_only_model, weights_only_optimizer, data, SPLIT_STEP, SPLIT_STEP + 1, record=False)
    missing_optimizer_error = (parameter_vector(weights_only_model) - parameter_vector(split_model)).abs().max().item()
    if next_step_error != 0.0 or missing_optimizer_error < 1e-8:
        raise AssertionError("The immediate resume/optimizer-state contrast was not informative.")
    train(restored, restored_optimizer, data, SPLIT_STEP + 1, STEPS, record=False)
    resume_error = (parameter_vector(restored) - parameter_vector(full_model)).abs().max().item()
    if reload_error != 0.0 or resume_error != 0.0:
        raise AssertionError("CPU deterministic checkpoint recovery differs from the uninterrupted run.")

    summary = {
        "verified_date": datetime.now().astimezone().date().isoformat(),
        "run_timestamp": datetime.now().astimezone().isoformat(), "python": platform.python_version(),
        "python_executable": sys.executable, "torch": torch.__version__,
        "device": str(DEVICE), "dtype": str(DTYPE), "threads": torch.get_num_threads(),
        "seed": SEED, "steps": STEPS, "optimizer": "SGD",
        "learning_rate": LEARNING_RATE, "momentum": MOMENTUM,
        "training_examples": len(data[0]), "validation_examples": len(data[2]),
        "gradient_checks": checks,
        "training": {"initial_train_mse": curve[0]["train_mse"],
                     "final_train_mse": final_train_mse, "validation_mse": final_valid_mse,
                     "learned_weight_and_bias": parameter_vector(full_model).tolist(),
                     "analytic_weight_and_bias": analytic.tolist(), "max_parameter_error": parameter_error},
        "recovery": {"saved_step": SPLIT_STEP, "reload_max_prediction_error": reload_error,
                     "next_step_max_parameter_error": next_step_error,
                     "without_optimizer_next_step_error": missing_optimizer_error,
                     "resume_max_parameter_error": resume_error, "status": "passed"},
        "scope": "CPU training-mechanics check on a noiseless linear problem; not an LLM or GPU benchmark",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "learning_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    print(f"Python {platform.python_version()}, PyTorch {torch.__version__}, CPU float64")
    print(f"Gradient check: AD/manual error={checks['max_analytic_error']:.3g}, AD/FD error={checks['max_finite_difference_error']:.3g}")
    print(f"Broadcasting: wrong loss={checks['broadcast_wrong_loss']:.6f}, corrected={checks['broadcast_correct_loss']:.6f}")
    print(f"Training MSE: {curve[0]['train_mse']:.6g} -> {final_train_mse:.6g}")
    print(f"Validation MSE={final_valid_mse:.6g}; max parameter error={parameter_error:.6g}")
    print(f"Reload prediction error={reload_error}; resume parameter error={resume_error}")
    print(f"Next-step resume error={next_step_error}; omitting optimizer state error={missing_optimizer_error:.6g}")
    print(f"All checks passed. Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
