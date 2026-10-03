"""Train a genuinely differentiable tiny causal LM on synthetic digit grammar.

CPU-only PyTorch, no downloads, no extra dependencies. Two learning rates use
the same initialization and batch stream. This is a teaching experiment, not a
language benchmark, and not proof that the full annual project is complete.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import platform
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from lesson_runtime import reserve_output_dir

import torch
from torch import nn
from torch.nn import functional as F


PAD, BOS, EOS = 0, 1, 2
VOCAB_SIZE = 13
IGNORE_INDEX = -100
TRAIN_LENGTHS = (6, 8, 10)
VALID_LENGTHS = (7, 9)
MAX_CONTEXT = 16
MODEL_CONFIG = {"d_model": 32, "heads": 2, "max_context": MAX_CONTEXT}


def make_sequences(lengths: tuple[int, ...]) -> list[list[int]]:
    """Hold out entire digit lengths; grammar and some prefixes are shared."""
    return [
        [BOS] + [3 + ((start + direction * index) % 10) for index in range(length)] + [EOS]
        for length in lengths for direction in (1, -1) for start in range(10)
    ]


def collate(sequences: list[list[int]]) -> torch.Tensor:
    result = torch.full((len(sequences), max(map(len, sequences))), PAD, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        result[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return result


def inputs_and_targets(sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # logits at t predicts original token t+1; shift exactly once here.
    inputs = sequences[:, :-1]
    targets = sequences[:, 1:].clone()
    targets[targets == PAD] = IGNORE_INDEX
    return inputs, targets


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, max_context: int):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.register_buffer("causal_allowed", torch.tril(torch.ones(max_context, max_context, dtype=torch.bool)))

    def forward(self, hidden: torch.Tensor, key_is_valid: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        # [B,T,3*D] -> [3,B,H,T,D/H]
        q, k, v = self.qkv(hidden).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = self.causal_allowed[:length, :length][None, None] & key_is_valid[:, None, None, :]
        # Every example starts with a valid BOS; even padded queries have a key.
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output(attended)


class TinyCausalLM(nn.Module):
    def __init__(self, d_model: int = 32, heads: int = 2, max_context: int = 16):
        super().__init__()
        self.max_context = max_context
        self.tokens = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.positions = nn.Embedding(max_context, d_model)
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, heads, max_context)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, length = token_ids.shape
        if not 1 <= length <= self.max_context:
            raise ValueError("input length must fit learned position embedding table")
        if batch == 0 or not bool((token_ids[:, 0] == BOS).all()):
            raise ValueError("every nonempty example must start with BOS")
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.tokens(token_ids) + self.positions(positions)[None]
        hidden = hidden + self.attention(self.norm_attention(hidden), token_ids != PAD)
        hidden = hidden + self.mlp(self.norm_mlp(hidden))
        return self.lm_head(self.norm_final(hidden))


def summed_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum")


@torch.no_grad()
def evaluate(model: TinyCausalLM, sequences: list[list[int]], batch_size: int = 17) -> dict[str, float | int]:
    model.eval()
    total_nll, total_tokens, correct = 0.0, 0, 0
    for start in range(0, len(sequences), batch_size):
        inputs, targets = inputs_and_targets(collate(sequences[start : start + batch_size]))
        logits = model(inputs)
        valid = targets != IGNORE_INDEX
        total_nll += summed_loss(logits, targets).item()
        total_tokens += valid.sum().item()
        correct += ((logits.argmax(-1) == targets) & valid).sum().item()
    return {"loss": total_nll / total_tokens, "tokens": total_tokens, "token_accuracy": correct / total_tokens}


@torch.no_grad()
def generate(model: TinyCausalLM, prefix_digits: list[int], max_new_tokens: int = 9) -> dict[str, object]:
    model.eval()
    tokens = [BOS] + [3 + digit for digit in prefix_digits]
    new_tokens = []
    stop_reason = "new_token_budget"
    for _ in range(max_new_tokens):
        if len(tokens) >= model.max_context:
            stop_reason = "context_limit"
            break
        prediction = model(torch.tensor([tokens], dtype=torch.long))[0, -1].argmax().item()
        tokens.append(prediction)
        new_tokens.append(prediction)
        if prediction == EOS:
            stop_reason = "eos"
            break
    def show(token: int) -> str:
        return {PAD: "<PAD>", BOS: "<BOS>", EOS: "<EOS>"}.get(token, str(token - 3))
    return {"prefix_digits": prefix_digits, "continuation": [show(token) for token in new_tokens], "stop_reason": stop_reason}


def numerical_checks(model: TinyCausalLM, train: list[list[int]]) -> dict[str, float | bool]:
    model.eval()
    example = torch.tensor([[BOS, 3, 4, EOS, PAD]], dtype=torch.long)
    inputs, targets = inputs_and_targets(example)
    if inputs.tolist() != [[BOS, 3, 4, EOS]] or targets.tolist() != [[3, 4, EOS, IGNORE_INDEX]]:
        raise AssertionError("next-token shift or padding labels are incorrect")
    # A known analytic CE checks gather/ignore semantics independently of a model.
    logits = torch.zeros(1, 4, VOCAB_SIZE)
    for time_index, target in enumerate([3, 4, EOS]):
        logits[0, time_index, target] = 6.0
    logits.requires_grad_()
    value = summed_loss(logits, targets) / 3
    expected = math.log1p((VOCAB_SIZE - 1) * math.exp(-6.0))
    if not math.isclose(value.item(), expected, rel_tol=1e-5, abs_tol=1e-7):
        raise AssertionError("CE disagrees with independent analytic answer")
    value.backward()
    if logits.grad[0, 3].abs().max().item() != 0 or logits.grad[0, :3].abs().sum().item() == 0:
        raise AssertionError("ignored padding must have zero loss gradient")

    with torch.no_grad():
        # Change future content while holding a five-token prefix exactly fixed.
        full_inputs, _ = inputs_and_targets(collate([train[-1]]))
        changed = full_inputs.clone()
        changed[:, 5:] = 3 + (changed[:, 5:] - 3 + 4) % 10
        causal_difference = (model(full_inputs)[:, :5] - model(changed)[:, :5]).abs().max().item()
        if causal_difference > 1e-6:
            raise AssertionError("future token changes leaked into prefix logits")
        short = collate([train[0]])
        padded = F.pad(short, (0, 4), value=PAD)
        short_x, short_y = inputs_and_targets(short)
        padded_x, padded_y = inputs_and_targets(padded)
        padding_loss_difference = abs(summed_loss(model(short_x), short_y).item() - summed_loss(model(padded_x), padded_y).item())
        if padding_loss_difference > 1e-5:
            raise AssertionError("adding right padding changed valid-token loss")

    # Token-weighted validation should not depend on its minibatch partition.
    eval_one = evaluate(model, train, batch_size=1)
    eval_seventeen = evaluate(model, train, batch_size=17)
    reduction_difference = abs(eval_one["loss"] - eval_seventeen["loss"])
    if reduction_difference > 1e-5 or eval_one["tokens"] != eval_seventeen["tokens"]:
        raise AssertionError("evaluation must aggregate total NLL / total valid tokens")

    probe, _ = inputs_and_targets(collate(train[:4]))
    with torch.no_grad():
        before = model(probe).clone()
    with tempfile.TemporaryDirectory(prefix="maedaily_tiny_lm_") as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        torch.save({"config": MODEL_CONFIG, "state_dict": model.state_dict()}, checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restored = TinyCausalLM(**payload["config"])
        restored.load_state_dict(payload["state_dict"])
        restored.eval()
        with torch.no_grad():
            reload_difference = (restored(probe) - before).abs().max().item()
        if reload_difference != 0.0:
            raise AssertionError("CPU checkpoint roundtrip changed logits")
    return {
        "label_shift_and_ignore_gradient": True,
        "analytic_cross_entropy": value.item(),
        "causal_prefix_max_abs_difference": causal_difference,
        "padding_nll_abs_difference": padding_loss_difference,
        "evaluation_partition_loss_difference": reduction_difference,
        "checkpoint_reload_max_abs_difference": reload_difference,
        "temporary_checkpoint_removed": True,
    }


def train_one(
    learning_rate: float, train: list[list[int]], valid: list[list[int]], *, seed: int, steps: int, batch_size: int
) -> tuple[dict[str, object], list[dict[str, float | int]]]:
    random.seed(seed)
    torch.manual_seed(seed)
    model = TinyCausalLM(**MODEL_CONFIG).cpu()
    batch_generator = torch.Generator().manual_seed(seed + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    initial_train = evaluate(model, train)
    initial_valid = evaluate(model, valid)
    rows = [{"learning_rate": learning_rate, "step": 0, "train_loss": initial_train["loss"], "validation_loss": initial_valid["loss"], "gradient_norm": 0.0, "processed_tokens": 0, "elapsed_seconds": 0.0}]
    processed_tokens = 0
    max_gradient_norm = 0.0
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        indices = torch.randint(len(train), (batch_size,), generator=batch_generator).tolist()
        inputs, targets = inputs_and_targets(collate([train[index] for index in indices]))
        count = targets.ne(IGNORE_INDEX).sum().item()
        optimizer.zero_grad(set_to_none=True)
        loss = summed_loss(model(inputs), targets) / count
        if not torch.isfinite(loss):
            raise AssertionError("non-finite training loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"missing or non-finite gradient: {name}")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True).item()
        max_gradient_norm = max(max_gradient_norm, norm)
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise AssertionError("non-finite parameter after optimizer step")
        processed_tokens += count
        if step % 20 == 0 or step == steps:
            train_metrics, valid_metrics = evaluate(model, train), evaluate(model, valid)
            rows.append({"learning_rate": learning_rate, "step": step, "train_loss": train_metrics["loss"], "validation_loss": valid_metrics["loss"], "gradient_norm": norm, "processed_tokens": processed_tokens, "elapsed_seconds": time.perf_counter() - started})
    duration = time.perf_counter() - started
    final_train, final_valid = evaluate(model, train), evaluate(model, valid)
    if not final_train["loss"] < initial_train["loss"] * 0.7:
        raise AssertionError("training loss did not fall by the required 30 percent")
    if not final_valid["loss"] < initial_valid["loss"]:
        raise AssertionError("held-out-length validation loss did not improve")
    checks = numerical_checks(model, train)
    checks.update({"all_training_gradients_finite": True, "all_updated_parameters_finite": True, "training_loss_fell_at_least_30_percent": True, "validation_loss_decreased": True})
    return {
        "learning_rate": learning_rate,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "steps": steps,
        "batch_size": batch_size,
        "processed_training_tokens": processed_tokens,
        "training_and_periodic_eval_seconds": duration,
        "max_preclip_gradient_norm": max_gradient_norm,
        "initial_train": initial_train,
        "initial_validation": initial_valid,
        "final_train": final_train,
        "final_validation": final_valid,
        "checks": checks,
        "greedy_generations": [generate(model, prefix) for prefix in [[7, 8], [2, 1], [9, 0], [0, 9]]],
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.003])
    parser.add_argument("--output-dir", type=Path, help="新目录或空目录；默认 study_runs 下独立保存")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or any(not math.isfinite(rate) or rate <= 0 for rate in args.learning_rates):
        parser.error("steps/batch-size must be positive; learning rates must be finite and positive")
    try:
        args.output_dir = reserve_output_dir(args.output_dir, "lm")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    train, valid = make_sequences(TRAIN_LENGTHS), make_sequences(VALID_LENGTHS)
    if set(map(tuple, train)) & set(map(tuple, valid)):
        raise AssertionError("full sequences leaked across train/validation split")
    runs, records = [], []
    for rate in args.learning_rates:
        summary, rows = train_one(rate, train, valid, seed=args.seed, steps=args.steps, batch_size=args.batch_size)
        runs.append(summary)
        records.extend(rows)
    output = {
        "experiment": "tiny_causal_digit_language_model",
        "run_timestamp": datetime.now().astimezone().isoformat(),
        "environment": {"python": sys.version, "python_executable": sys.executable, "pytorch": torch.__version__, "platform": platform.platform(), "device": "cpu", "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": True},
        "config": {"seed": args.seed, "model": MODEL_CONFIG, "layers": 1, "normalization": "pre-LN", "activation": "GELU", "dropout": 0.0, "optimizer": "AdamW", "weight_decay": 0.01, "clip_gradient_norm": 1.0, "vocab_size": VOCAB_SIZE, "learning_rates": args.learning_rates},
        "data": {"grammar": "digits increase/decrease modulo 10; BOS and EOS included", "train_digit_lengths": TRAIN_LENGTHS, "validation_digit_lengths": VALID_LENGTHS, "train_sequences": len(train), "validation_sequences": len(valid), "full_sequence_overlap": 0, "split_limit": "held-out lengths within the training length range; shared grammar and overlapping prefixes; no out-of-domain claim"},
        "comparison": "same seed, initialization, minibatch index stream and update count; one seed only",
        "runs": runs,
        "scope": "real CPU LM training; synthetic grammar only; no GPU memory/performance measurement; not complete project A acceptance",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "learning_curve.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for run in runs:
        print(f"lr={run['learning_rate']} params={run['parameter_count']} steps={run['steps']} train={run['initial_train']['loss']:.6f}->{run['final_train']['loss']:.6f} val={run['initial_validation']['loss']:.6f}->{run['final_validation']['loss']:.6f} seconds={run['training_and_periodic_eval_seconds']:.3f}")
        print(json.dumps({"checks": run["checks"], "generations": run["greedy_generations"]}, ensure_ascii=False))
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
