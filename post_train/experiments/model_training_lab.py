"""Real, local CPU text-model SFT, DPO and clipped-surrogate RLVR teaching lab.

The character Transformer is trained here from scratch; no pretrained general
language ability, downloads, GPU, or external service is assumed. Saved actions
and immutable weights make probability and gradient checks independently replayable.
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
import re
import subprocess
import sys
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from lesson_runtime import reserve_output_dir

PAD, BOS, EOS = 0, 1, 2
IGNORE = -100
TEMPLATE = "Q:{question}\nA:"
SCHEMA = "maedaily-character-lm-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class CharacterTokenizer:
    """An explicit, serializable tokenizer; EOS and PAD have different identities."""
    def __init__(self, alphabet: str = "\n +-0123456789:=AQcopy"):
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("alphabet contains duplicates")
        self.pieces = ["<PAD>", "<BOS>", "<EOS>"] + list(alphabet)
        self.ids = {piece: index for index, piece in enumerate(self.pieces)}

    def encode(self, text: str) -> list[int]:
        try:
            return [self.ids[character] for character in text]
        except KeyError as error:
            raise ValueError(f"character outside frozen vocabulary: {error.args[0]!r}") from error

    def decode(self, ids: list[int], *, show_special: bool = False) -> str:
        return "".join(self.pieces[token] for token in ids if token >= 3 or show_special)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "pieces": self.pieces, "pad": PAD, "bos": BOS,
                "eos": EOS, "padding_side": "right", "template": TEMPLATE}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CharacterTokenizer":
        if value.get("schema") != SCHEMA or value.get("pieces", [])[:3] != ["<PAD>", "<BOS>", "<EOS>"]:
            raise ValueError("unsupported tokenizer schema or special tokens")
        if value.get("template") != TEMPLATE or value.get("padding_side") != "right":
            raise ValueError("tokenizer template/padding mismatch")
        result = cls("".join(value["pieces"][3:]))
        if result.to_dict() != value:
            raise ValueError("tokenizer metadata mismatch")
        return result


class CausalAttention(nn.Module):
    def __init__(self, width: int, heads: int, context: int):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.register_buffer("causal", torch.tril(torch.ones(context, context, dtype=torch.bool)))

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        q, k, v = self.qkv(hidden).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = self.causal[:length, :length][None, None] & valid[:, None, None, :]
        weights = (scores.masked_fill(~allowed, float("-inf"))).softmax(-1)
        return self.output((weights @ v).transpose(1, 2).contiguous().reshape(batch, length, width))


class CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int, context: int):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = CausalAttention(width, heads, context)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.norm_attention(hidden), valid)
        return hidden + self.mlp(self.norm_mlp(hidden))


class TinyTextLM(nn.Module):
    def __init__(self, vocab_size: int, width: int = 32, heads: int = 2, context: int = 32, layers: int = 1):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads or context < 16 or vocab_size < 4 or not 1 <= layers <= 4:
            raise ValueError("invalid model dimensions")
        self.config = {"vocab_size": vocab_size, "width": width, "heads": heads, "context": context, "layers": layers}
        self.tokens = nn.Embedding(vocab_size, width, padding_idx=PAD)
        self.positions = nn.Embedding(context, width)
        self.norm_attention = nn.LayerNorm(width)
        self.attention = CausalAttention(width, heads, context)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.extra_blocks = nn.ModuleList(CausalBlock(width, heads, context) for _ in range(layers - 1))
        self.norm_final = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or ids.shape[0] == 0 or not 1 <= ids.shape[1] <= self.config["context"]:
            raise ValueError("batch must be nonempty and fit model context")
        if not bool((ids[:, 0] == BOS).all()):
            raise ValueError("every sequence must start with BOS")
        hidden = self.tokens(ids) + self.positions(torch.arange(ids.shape[1]))[None]
        hidden = hidden + self.attention(self.norm_attention(hidden), ids != PAD)
        hidden = hidden + self.mlp(self.norm_mlp(hidden))
        for block in self.extra_blocks:
            hidden = block(hidden, ids != PAD)
        return self.head(self.norm_final(hidden))


def frozen_copy(model: TinyTextLM) -> TinyTextLM:
    return copy.deepcopy(model).eval().requires_grad_(False)


def weight_hash(model: TinyTextLM) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json_bytes(list(tensor.shape)))
        digest.update(bytes(tensor.detach().contiguous().cpu().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def prompt(record: dict[str, Any]) -> str:
    return TEMPLATE.format(question=record["question"])


def build_batch(records: list[dict[str, Any]], tokenizer: CharacterTokenizer,
                context: int = 32, *, answers: list[list[int]] | None = None) -> dict[str, Any]:
    if not records:
        raise ValueError("empty batch")
    if answers is not None and len(answers) != len(records):
        raise ValueError("response count mismatch")
    sequences, masks, prefix_lengths = [], [], []
    for index, record in enumerate(records):
        prefix = [BOS] + tokenizer.encode(prompt(record))
        response = (tokenizer.encode(record["answer"]) + [EOS]) if answers is None else answers[index]
        if not response or any(token in (PAD, BOS) for token in response):
            raise ValueError("response must contain valid action tokens")
        if EOS in response[:-1]:
            raise ValueError("tokens after EOS are not actions")
        sequence = prefix + response
        if len(sequence) > context:
            raise ValueError("sequence too long: reject instead of losing response/EOS")
        sequences.append(sequence)
        masks.append([False] * len(prefix) + [True] * len(response))
        prefix_lengths.append(len(prefix))
    length = max(map(len, sequences))
    ids = torch.full((len(records), length), PAD, dtype=torch.long)
    mask = torch.zeros((len(records), length), dtype=torch.bool)
    for row, (sequence, loss_mask) in enumerate(zip(sequences, masks)):
        ids[row, :len(sequence)] = torch.tensor(sequence)
        mask[row, :len(sequence)] = torch.tensor(loss_mask)
    labels = ids.clone().masked_fill(~mask, IGNORE)
    result = {"input_ids": ids, "attention_mask": ids != PAD, "labels": labels,
              "loss_mask": mask, "prefix_lengths": prefix_lengths,
              "sample_ids": [record["id"] for record in records]}
    validate_batch(result)
    return result


def validate_batch(batch: dict[str, Any]) -> None:
    ids, mask, labels = batch["input_ids"], batch["loss_mask"], batch["labels"]
    if ids.shape != mask.shape or ids.shape != labels.shape:
        raise ValueError("labels/mask shape differs from actual sequence")
    if not torch.equal(labels, ids.masked_fill(~mask, IGNORE)):
        raise ValueError("labels must be unshifted actual tokens, ignoring exactly the loss mask")
    if not torch.equal(batch["attention_mask"], ids != PAD):
        raise ValueError("padding attention mask mismatch")
    for row, start in enumerate(batch["prefix_lengths"]):
        end = int((ids[row] != PAD).sum())
        expected = torch.zeros_like(mask[row])
        expected[start:end] = True
        if not torch.equal(mask[row], expected) or not 1 <= start < end:
            raise ValueError("answer action mask includes prompt/padding or omits answer/EOS")


def response_logprobs(model: TinyTextLM, batch: dict[str, Any], temperature: float = 1.0,
                      *, behavior: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_batch(batch)
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    logits = model(batch["input_ids"])[:, :-1]
    if behavior:
        logits = logits / temperature
        logits = logits.clone()
        logits[..., [PAD, BOS]] = float("-inf")
    token_lp = logits.log_softmax(-1).gather(-1, batch["input_ids"][:, 1:, None]).squeeze(-1)
    mask = batch["loss_mask"][:, 1:]
    # where avoids 0 * -inf at PAD targets in constrained behavior distributions.
    token_lp = token_lp.masked_fill(~mask, 0.0)
    return token_lp, token_lp.sum(-1), mask.sum(-1)


def parameter_vector(model: TinyTextLM) -> torch.Tensor:
    return torch.cat([parameter.detach().flatten().clone() for parameter in model.parameters()])


def checked_step(loss: torch.Tensor, model: TinyTextLM, optimizer: torch.optim.Optimizer,
                 *, require_signal: bool = True) -> dict[str, Any]:
    if not loss.requires_grad or not bool(torch.isfinite(loss)):
        raise AssertionError("loss is nonfinite or policy graph is detached")
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if expected != actual:
        raise AssertionError("optimizer parameter set differs from trainable policy")
    before, before_hash = parameter_vector(model), weight_hash(model)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise AssertionError("policy gradients missing or nonfinite")
    norm = math.sqrt(sum(float(gradient.square().sum()) for gradient in gradients))
    if require_signal and norm <= 1e-10:
        raise AssertionError("expected a nonzero policy gradient")
    clipped = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    delta = float((parameter_vector(model) - before).norm())
    if require_signal and delta <= 0:
        raise AssertionError("optimizer did not change policy parameters")
    return {"loss": float(loss.detach()), "gradient_norm_before_clip": norm,
            "clip_pre_norm": clipped, "parameter_delta_norm": delta,
            "weight_before": before_hash, "weight_after": weight_hash(model)}


def sft_loss(model: TinyTextLM, batch: dict[str, Any]) -> torch.Tensor:
    _, sums, counts = response_logprobs(model, batch)
    return -sums.sum() / counts.sum()


def sft_checks(model: TinyTextLM, batch: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    logits = model(batch["input_ids"])[:, :-1]
    logits.retain_grad()
    targets, labels = batch["input_ids"][:, 1:], batch["labels"][:, 1:]
    mask = batch["loss_mask"][:, 1:]
    gathered = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    manual = -(gathered * mask).sum() / mask.sum()
    library = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE)
    params = tuple(model.parameters())
    grad_manual = torch.autograd.grad(manual, params, retain_graph=True)
    grad_library = torch.autograd.grad(library, params, retain_graph=True)
    logits.grad = None
    manual.backward()
    analytic = (logits.detach().softmax(-1) - F.one_hot(targets, logits.shape[-1])) * mask[..., None] / mask.sum()
    logit_error = float((logits.grad - analytic).abs().max())
    parameter_error = max(float((left - right).abs().max()) for left, right in zip(grad_manual, grad_library))
    if not torch.allclose(manual, library, atol=1e-6, rtol=1e-5) or logit_error > 1e-5 or parameter_error > 1e-5:
        raise AssertionError("SFT loss/logits/parameter gradients failed independent CE comparison")
    if not any(float(gradient.norm()) > 0 for gradient in grad_manual):
        raise AssertionError("SFT actual model gradient missing")
    model.zero_grad(set_to_none=True)
    return {"manual_loss": float(manual.detach()), "cross_entropy_loss": float(library.detach()),
            "effective_tokens": int(mask.sum()), "per_sample_tokens": mask.sum(-1).tolist(),
            "logits_gradient_max_error": logit_error, "parameter_gradient_max_error": parameter_error,
            "unsupervised_logits_gradient_max": float(logits.grad.masked_select(~mask[..., None].expand_as(logits)).abs().max()),
            "passed": True}


def dpo_loss(policy: TinyTextLM, reference: TinyTextLM, chosen: dict[str, Any],
             rejected: dict[str, Any], beta: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if reference is policy or any(parameter.requires_grad for parameter in reference.parameters()):
        raise AssertionError("reference must be an independent frozen model")
    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    _, pc, _ = response_logprobs(policy, chosen)
    _, pr, _ = response_logprobs(policy, rejected)
    with torch.no_grad():
        _, rc, _ = response_logprobs(reference, chosen)
        _, rr, _ = response_logprobs(reference, rejected)
    u = beta * ((pc - pr) - (rc - rr))
    return F.softplus(-u).mean(), {"policy_chosen": pc, "policy_rejected": pr,
                                   "reference_chosen": rc.detach(), "reference_rejected": rr.detach(), "u": u}


def dpo_checks(policy: TinyTextLM, reference: TinyTextLM, chosen: dict[str, Any],
               rejected: dict[str, Any], beta: float) -> dict[str, Any]:
    loss, terms = dpo_loss(policy, reference, chosen, rejected, beta)
    parameters = tuple(policy.parameters())
    actual = torch.autograd.grad(loss, parameters, retain_graph=True)
    coefficient = (-beta * torch.sigmoid(-terms["u"]) / len(terms["u"])).detach()
    analytic = torch.autograd.grad(((terms["policy_chosen"] - terms["policy_rejected"]) * coefficient).sum(), parameters)
    error = max(float((left - right).abs().max()) for left, right in zip(actual, analytic))
    _, swapped = dpo_loss(policy, reference, rejected, chosen, beta)
    if error > 1e-5 or not torch.allclose(swapped["u"], -terms["u"], atol=1e-6):
        raise AssertionError("DPO preference derivative or swapped sign failed")
    if not any(float(gradient.norm()) > 1e-10 for gradient in actual):
        raise AssertionError("DPO preference gradient is zero")
    if terms["u"].abs().max().item() > 1e-5 or abs(loss.item() - math.log(2)) > 1e-5:
        raise AssertionError("initial policy/reference do not represent identical weights")
    return {**{key: value.detach().tolist() for key, value in terms.items()},
            "initial_loss": float(loss.detach()), "parameter_gradient_max_error": error,
            "reference_requires_grad": False, "reference_gradient_is_none": all(p.grad is None for p in reference.parameters()),
            "passed": True}


def verify_answer(text: str, answer: str, termination: str, *, max_characters: int = 4) -> dict[str, Any]:
    if len(text) > max_characters:
        status = "too_long"
    elif termination != "eos":
        status = "truncated"
    elif not text:
        status = "empty"
    elif text == "-0" or re.fullmatch(r"-?(0|[1-9][0-9]*)", text) is None:
        status = "malformed"
    else:
        status = "correct" if text == answer else "wrong"
    valid = status in ("correct", "wrong")
    # Declared shaping exposes a real format signal when task correctness is rare.
    reward = float(status == "correct") + 0.1 * float(valid)
    return {"version": "strict-canonical-integer-with-eos-v2", "status": status,
            "correct": status == "correct", "format_valid": valid,
            "components": {"exact_answer": float(status == "correct"), "format": 0.1 * float(valid)},
            "total": reward, "expected_answer": answer}


def verifier_checks() -> dict[str, Any]:
    cases = [("5", "eos", "correct"), ("4", "eos", "wrong"), ("", "eos", "empty"),
             ("5 5", "eos", "malformed"), ("05", "eos", "malformed"),
             ("5\n", "eos", "malformed"), ("5", "new_token_budget", "truncated"),
             ("55555", "eos", "too_long"), ("-3", "eos", "wrong")]
    results = [{"text": text, "termination": ending, **verify_answer(text, "5", ending)} for text, ending, _ in cases]
    if any(value["status"] != expected for value, (_, _, expected) in zip(results, cases)):
        raise AssertionError("verifier boundary case failed")
    return {"cases": results, "passed": True}


@torch.no_grad()
def generate(model: TinyTextLM, tokenizer: CharacterTokenizer, record: dict[str, Any],
             max_new_tokens: int, temperature: float = 1.0, *, sample: bool = False,
             generator: torch.Generator | None = None) -> dict[str, Any]:
    model.eval()
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    prefix = [BOS] + tokenizer.encode(prompt(record))
    tokens, lps, ending = [], [], "new_token_budget"
    for _ in range(max_new_tokens):
        if len(prefix) + len(tokens) >= model.config["context"]:
            ending = "context_limit"
            break
        logits = model(torch.tensor([prefix + tokens]))[0, -1] / temperature
        logits[[PAD, BOS]] = float("-inf")
        log_probs = logits.log_softmax(-1)
        token = int(torch.multinomial(log_probs.exp(), 1, generator=generator)) if sample else int(log_probs.argmax())
        tokens.append(token)
        lps.append(float(log_probs[token]))
        if token == EOS:
            ending = "eos"
            break
    text = tokenizer.decode(tokens)
    return {"sample_id": record["id"], "question": record["question"], "answer": record["answer"],
            "prompt_token_ids": prefix, "response_token_ids": tokens, "text": text,
            "termination_reason": ending, "model_token_logprobs": lps,
            "behavior_token_logprobs": lps if sample else [0.0] * len(tokens),
            "distribution": {"temperature": temperature, "top_p": 1.0, "top_k": 0,
                             "forbidden_token_ids": [PAD, BOS], "do_sample": sample,
                             "kind": "temperature-constrained-softmax" if sample else "greedy-deterministic"},
            "reward": verify_answer(text, record["answer"], ending)}


def advantages(rewards: list[float], epsilon: float = 1e-8) -> tuple[list[float], dict[str, Any]]:
    if len(rewards) < 2 or not all(math.isfinite(value) for value in rewards):
        raise ValueError("group needs at least two finite rewards")
    mean = sum(rewards) / len(rewards)
    deviation = math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
    values = [(value - mean) / (deviation + epsilon) for value in rewards] if deviation > epsilon else [0.0] * len(rewards)
    return values, {"mean": mean, "population_std": deviation, "epsilon": epsilon,
                    "active": deviation > epsilon, "method": "population-standardized-group-reward"}


def replay_check(model: TinyTextLM, tokenizer: CharacterTokenizer, rollouts: list[dict[str, Any]],
                 temperature: float, tolerance: float = 2e-5) -> float:
    records = [{"id": row["sample_id"], "question": row["question"], "answer": row["answer"]} for row in rollouts]
    batch = build_batch(records, tokenizer, model.config["context"], answers=[row["response_token_ids"] for row in rollouts])
    with torch.no_grad():
        token_lps, _, _ = response_logprobs(model, batch, temperature, behavior=True)
    errors = []
    for index, row in enumerate(rollouts):
        if row["distribution"]["temperature"] != temperature:
            raise AssertionError("rollout/learner temperature mismatch")
        actual = token_lps[index][batch["loss_mask"][index, 1:]]
        cached = torch.tensor(row["behavior_token_logprobs"])
        errors.append(float((actual - cached).abs().max()))
        if verify_answer(row["text"], row["answer"], row["termination_reason"]) != row["reward"]:
            raise AssertionError("reward is not independently replayable")
    error = max(errors)
    if error > tolerance:
        raise AssertionError(f"behavior probability replay mismatch: {error}")
    return error


def clipped_policy_loss(current: torch.Tensor, old: torch.Tensor, mask: torch.Tensor,
                        advantage: torch.Tensor, clip: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    if old.requires_grad or advantage.requires_grad:
        raise AssertionError("old probabilities/reward advantages must be detached")
    if not 0 < clip < 1:
        raise ValueError("clip must be between zero and one")
    ratio = torch.exp(current - old)
    surrogate = torch.minimum(ratio * advantage[:, None], ratio.clamp(1 - clip, 1 + clip) * advantage[:, None])
    # Every response receives equal weight, independent of its generated length.
    loss = -((surrogate * mask).sum(-1) / mask.sum(-1)).mean()
    return loss, ratio


def mask_table(batch: dict[str, Any], tokenizer: CharacterTokenizer) -> list[dict[str, Any]]:
    return [{"sample_id": batch["sample_ids"][row], "positions": [
        {"position": column, "token_id": int(token), "token": tokenizer.pieces[int(token)],
         "role": "padding" if token == PAD else "answer" if bool(batch["loss_mask"][row, column]) else "prompt",
         "attention_mask": bool(batch["attention_mask"][row, column]),
         "label": int(batch["labels"][row, column]), "loss_mask": bool(batch["loss_mask"][row, column])}
        for column, token in enumerate(sequence)]} for row, sequence in enumerate(batch["input_ids"])]


def fixed_data() -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "eval_main": [], "eval_transfer": [], "eval_regression": []}
    for operation in ("+", "-"):
        for left in range(6):
            for right in range(6):
                key = min(left, right) * 6 + max(left, right)
                split = "eval_main" if key % 7 == 0 else "dev" if key % 7 == 1 else "train"
                answer = left + right if operation == "+" else left - right
                splits[split].append({"id": f"{split}:{left}{operation}{right}", "question": f"{left}{operation}{right}=",
                                      "answer": str(answer), "family": "small-add" if operation == "+" else "small-subtract",
                                      "family_id": f"operand-pair:{min(left,right)}:{max(left,right)}", "split": split})
    for operation in ("+", "-"):
        for left in (7, 8, 9):
            for right in (1, 3):
                answer = left + right if operation == "+" else left - right
                splits["eval_transfer"].append({"id": f"transfer:{left}{operation}{right}", "question": f"{left}{operation}{right}=",
                                                 "answer": str(answer), "family": "unseen-left-operand",
                                                 "family_id": f"operand-pair:{min(left,right)}:{max(left,right)}", "split": "eval_transfer"})
    # A separate copy task gives an explicit task-external panel. Training uses
    # canonical integers; regression uses unseen leading-zero input spellings.
    for digit in range(10):
        splits["train"].append({"id": f"train:copy:{digit}", "question": f"copy {digit}", "answer": str(digit), "family": "copy", "family_id": f"copy-canonical:{digit}", "split": "train"})
        splits["eval_regression"].append({"id": f"regression:copy:{digit}", "question": f"copy 0{digit}", "answer": str(digit), "family": "copy-leading-zero", "family_id": f"copy-leading-zero:{digit}", "split": "eval_regression"})
    questions = [record["question"] for records in splits.values() for record in records]
    if len(questions) != len(set(questions)):
        raise AssertionError("train/dev/eval prompt overlap")
    family_sets = {name: {record["family_id"] for record in records} for name, records in splits.items()}
    for left_name, left_families in family_sets.items():
        for right_name, right_families in family_sets.items():
            if left_name != right_name and left_families & right_families:
                raise AssertionError("canonical operand family crosses data splits")
    return splits


def save_checkpoint(path: Path, model: TinyTextLM, tokenizer: CharacterTokenizer, stage: str,
                    optimizer: torch.optim.Optimizer | None = None, cursor: int = 0) -> str:
    # x mode prevents accidental mutation of an already named snapshot.
    payload = {"schema": SCHEMA, "model_config": model.config, "state_dict": model.state_dict(),
               "tokenizer": tokenizer.to_dict(), "stage": stage, "weight_hash": weight_hash(model),
               "optimizer": None if optimizer is None else optimizer.state_dict(),
               "torch_rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(),
               "data_cursor": cursor, "saved_at_utc": utc_now(), "resume_support": "weights-load; no CLI-exact-resume",
               "data_profile": getattr(model, "data_profile", "legacy-v1")}
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return file_hash(path)


def load_model_and_tokenizer(path: Path) -> tuple[TinyTextLM, CharacterTokenizer]:
    # weights_only unpickles tensors and basic containers; never load arbitrary
    # Python objects from an externally supplied checkpoint.
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema") != SCHEMA:
        raise ValueError("checkpoint is not this character-LM format")
    tokenizer = CharacterTokenizer.from_dict(value["tokenizer"])
    if value["model_config"]["vocab_size"] != len(tokenizer.pieces):
        raise ValueError("checkpoint vocabulary/tokenizer mismatch")
    model = TinyTextLM(**value["model_config"])
    model.load_state_dict(value["state_dict"], strict=True)
    model.data_profile = value.get("data_profile", "legacy-v1")
    model.eval()
    if weight_hash(model) != value["weight_hash"]:
        raise ValueError("checkpoint weight hash mismatch")
    return model, tokenizer


def reload_check(path: Path, model: TinyTextLM, tokenizer: CharacterTokenizer) -> dict[str, Any]:
    restored, restored_tokenizer = load_model_and_tokenizer(path)
    batch = build_batch(fixed_data()["train"][:2], tokenizer, model.config["context"])
    with torch.no_grad():
        error = float((restored(batch["input_ids"]) - model(batch["input_ids"])).abs().max())
    if error > 1e-6 or restored_tokenizer.to_dict() != tokenizer.to_dict():
        raise AssertionError("checkpoint/tokenizer reload differs")
    return {"max_logit_error": error, "weight_hash_equal": weight_hash(restored) == weight_hash(model), "passed": True}


def train_sft(model: TinyTextLM, tokenizer: CharacterTokenizer, records: list[dict[str, Any]],
              steps: int, learning_rate: float, seed: int, *, all_tokens: bool = False) -> tuple[list[dict[str, Any]], torch.optim.Optimizer]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    logs = []
    for step in range(steps):
        sample = [records[rng.randrange(len(records))] for _ in range(16)]
        batch = build_batch(sample, tokenizer, model.config["context"])
        if all_tokens:
            logits = model(batch["input_ids"])[:, :-1]
            labels = batch["input_ids"][:, 1:].clone()
            labels[labels == PAD] = IGNORE
            loss = F.cross_entropy(logits.reshape(-1, len(tokenizer.pieces)), labels.reshape(-1), ignore_index=IGNORE)
            effective = int((labels != IGNORE).sum())
        else:
            loss = sft_loss(model, batch)
            effective = int(batch["loss_mask"].sum())
        started = time.perf_counter()
        logs.append({"step": step + 1, "sample_ids": batch["sample_ids"], "effective_tokens": effective,
                     "samples": len(sample), "learning_rate": learning_rate, "updated_at_utc": utc_now(),
                     **checked_step(loss, model, optimizer), "update_seconds": time.perf_counter() - started})
    model.eval()
    return logs, optimizer


@torch.no_grad()
def evaluate(model: TinyTextLM, tokenizer: CharacterTokenizer, data: dict[str, list[dict[str, Any]]],
             max_new_tokens: int) -> dict[str, Any]:
    rows, panels = [], {}
    for panel in ("eval_main", "eval_transfer", "eval_regression"):
        started = time.perf_counter()
        outputs = [generate(model, tokenizer, record, max_new_tokens) for record in data[panel]]
        for value in outputs:
            value.update({"panel": panel, "checkpoint_weight_hash": weight_hash(model),
                          "blind_review": "not_performed", "generated_tokens": len(value["response_token_ids"])})
        correct = sum(row["reward"]["correct"] for row in outputs)
        valid = sum(row["reward"]["format_valid"] for row in outputs)
        panels[panel] = {"correct": correct, "total": len(outputs), "accuracy": correct / len(outputs),
                         "format_valid": valid, "generated_tokens": sum(row["generated_tokens"] for row in outputs),
                         "elapsed_seconds": time.perf_counter() - started}
        rows.extend(outputs)
    return {"protocol": {"decoding": "greedy", "max_new_tokens": max_new_tokens,
                         "forbidden_token_ids": [PAD, BOS], "sample_ids_hash": content_hash([row["sample_id"] for row in rows]),
                         "selection": "frozen before training; eval not used for tuning"},
            "panels": panels, "predictions": rows}


def paired_comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = {row["sample_id"]: row for row in baseline["predictions"]}
    right = {row["sample_id"]: row for row in candidate["predictions"]}
    if left.keys() != right.keys() or baseline["protocol"] != candidate["protocol"]:
        raise AssertionError("baseline/candidate evaluation budget/protocol differs")
    return {"improved_ids": [key for key in left if not left[key]["reward"]["correct"] and right[key]["reward"]["correct"]],
            "regressed_ids": [key for key in left if left[key]["reward"]["correct"] and not right[key]["reward"]["correct"]],
            "unchanged": sum(left[key]["reward"]["correct"] == right[key]["reward"]["correct"] for key in left),
            "paired_total": len(left)}


def collect_rollouts(old: TinyTextLM, tokenizer: CharacterTokenizer, records: list[dict[str, Any]],
                     *, groups: int, group_size: int, max_new_tokens: int, temperature: float,
                     seed: int, round_number: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    rows, group_receipts = [], []
    before = weight_hash(old)
    rng_before = generator.get_state().tolist()
    for group in range(groups):
        record = records[rng.randrange(len(records))]
        members = [generate(old, tokenizer, record, max_new_tokens, temperature, sample=True, generator=generator)
                   for _ in range(group_size)]
        values, stats = advantages([row["reward"]["total"] for row in members])
        member_ids = [f"round-{round_number}:group-{group}:sample-{index}" for index in range(group_size)]
        for index, row in enumerate(members):
            row.update({"rollout_id": member_ids[index], "group_id": f"round-{round_number}:group-{group}",
                        "split": "train", "old_weight_hash": before, "seed": seed,
                        "advantage": values[index], "group_member_ids": member_ids, "group_statistics": stats})
        rows.extend(members)
        group_receipts.append({"group_id": f"round-{round_number}:group-{group}", "sample_id": record["id"],
                               "member_ids": member_ids, "rewards": [row["reward"]["total"] for row in members],
                               "advantages": values, **stats})
    if weight_hash(old) != before or any(parameter.grad is not None for parameter in old.parameters()):
        raise AssertionError("sampling snapshot changed or acquired gradients")
    error = replay_check(old, tokenizer, rows, temperature)
    return rows, {"groups": group_receipts, "active_groups": sum(group["active"] for group in group_receipts),
                  "probability_replay_max_error": error, "snapshot_weight_hash": before,
                  "generator_state_before": rng_before, "generator_state_after": generator.get_state().tolist(),
                  "sampling_changed_weights": False, "sampling_optimizer_steps": 0}


def train_rlvr(policy: TinyTextLM, tokenizer: CharacterTokenizer, reference: TinyTextLM,
               records: list[dict[str, Any]], output: Path, config: dict[str, Any]) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config["rl_learning_rate"])
    reference_hash = weight_hash(reference)
    updates, collections = [], []
    total_generated_tokens = total_update_tokens = total_samples = 0
    for round_number in range(config["rlvr_steps"]):
        started = time.perf_counter()
        policy.eval()
        old = frozen_copy(policy)
        before_file = output / f"rlvr_{round_number:03d}_before.pt"
        old_file_hash = save_checkpoint(before_file, old, tokenizer, "rlvr-behavior-snapshot", cursor=round_number)
        rows, receipt = collect_rollouts(old, tokenizer, records,
                                        groups=config["groups"], group_size=config["group_size"],
                                        max_new_tokens=config["max_new_tokens"], temperature=config["temperature"],
                                        seed=config["seed"] + 1000 + round_number, round_number=round_number)
        batch_records = [{"id": row["sample_id"], "question": row["question"], "answer": row["answer"]} for row in rows]
        batch = build_batch(batch_records, tokenizer, policy.config["context"], answers=[row["response_token_ids"] for row in rows])
        mask = batch["loss_mask"][:, 1:]
        old_lp = torch.zeros_like(mask, dtype=torch.float32)
        for index, row in enumerate(rows):
            old_lp[index, mask[index]] = torch.tensor(row["behavior_token_logprobs"])
        advantage = torch.tensor([row["advantage"] for row in rows]).detach()
        current, _, _ = response_logprobs(policy, batch, config["temperature"], behavior=True)
        with torch.no_grad():
            reference_lp, _, _ = response_logprobs(reference, batch, config["temperature"], behavior=True)
        reference_lp = reference_lp.detach()
        loss, ratio = clipped_policy_loss(current, old_lp.detach(), mask, advantage, config["clip"])
        ratio_error = float((ratio[mask].detach() - 1).abs().max())
        if ratio_error > 3e-5:
            raise AssertionError("ratio is not one before updating the identical behavior snapshot")
        active = receipt["active_groups"] > 0
        diagnostic: dict[str, Any]
        if active:
            diagnostic = checked_step(loss, policy, optimizer)
            status = "updated"
            total_update_tokens += int(mask.sum())
        else:
            diagnostic = {"loss": float(loss.detach()), "gradient_norm_before_clip": 0.0,
                          "parameter_delta_norm": 0.0, "weight_before": weight_hash(policy), "weight_after": weight_hash(policy)}
            status = "skipped_zero_advantage"
        after_file = output / f"rlvr_{round_number:03d}_after.pt"
        after_file_hash = save_checkpoint(after_file, policy, tokenizer, "rlvr-after-update", optimizer, cursor=round_number + 1)
        if weight_hash(old) != receipt["snapshot_weight_hash"] or weight_hash(reference) != reference_hash:
            raise AssertionError("old/reference weights changed during policy update")
        if any(parameter.grad is not None for model in (old, reference) for parameter in model.parameters()):
            raise AssertionError("old/reference acquired gradients")
        for row in rows:
            row["reference_weight_hash"] = reference_hash
        serialized = {"round": round_number + 1, "status": status, **receipt,
                      "rollouts": rows, "old_token_logprobs": old_lp.tolist(),
                      "current_token_logprobs_before": current.detach().tolist(),
                      "reference_token_logprobs": reference_lp.tolist(), "action_mask": mask.tolist(),
                      "ratio_values_before": ratio.detach().tolist(), "ratio_max_error_before": ratio_error,
                      "old_requires_grad": old_lp.requires_grad, "reference_requires_grad": reference_lp.requires_grad,
                      "advantage_requires_grad": advantage.requires_grad,
                      "loss_definition": "token-clipped surrogate; response mean then batch mean; no KL term",
                      "clip": config["clip"], "clip_fraction_before": float(((ratio.detach() < 1 - config["clip"]) | (ratio.detach() > 1 + config["clip"]))[mask].float().mean()),
                      "before_checkpoint": {"file": before_file.name, "sha256": old_file_hash},
                      "after_checkpoint": {"file": after_file.name, "sha256": after_file_hash},
                      "rollout_reuse_count": 1, "updated_at_utc": utc_now(),
                      "elapsed_seconds": time.perf_counter() - started, **diagnostic}
        collections.append(serialized)
        updates.append({key: value for key, value in serialized.items() if key not in ("rollouts", "generator_state_before", "generator_state_after")})
        total_generated_tokens += sum(len(row["response_token_ids"]) for row in rows)
        total_samples += len(rows)
    write_json(output / "rollouts.json", collections)
    write_json(output / "rlvr_updates.json", updates)
    return {"optimizer_steps": sum(row["status"] == "updated" for row in updates),
            "collection_rounds": len(updates), "generated_tokens": total_generated_tokens,
            "effective_update_tokens": total_update_tokens, "generated_samples": total_samples,
            "policy_gradient_proved": any(row["status"] == "updated" and row["gradient_norm_before_clip"] > 0 for row in updates),
            "reference_unchanged": weight_hash(reference) == reference_hash,
            "zero_advantage_rounds": sum(row["status"] != "updated" for row in updates)}


def git_versions() -> dict[str, Any]:
    result: dict[str, Any] = {"source_file": str(Path(__file__).resolve()), "source_sha256": file_hash(Path(__file__)),
                              "commit": None, "diff_sha256": None}
    try:
        result["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True, timeout=10).stdout
        result["diff_sha256"] = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.SubprocessError):
        result["git_probe"] = "unavailable"
    return result


def budget(logs: list[dict[str, Any]]) -> dict[str, int]:
    return {"optimizer_steps": len(logs), "training_samples": sum(row["samples"] for row in logs),
            "effective_update_tokens": sum(row["effective_tokens"] for row in logs), "generated_tokens": 0}


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if config.get("data_profile", "legacy-v1") == "quality-v2":
        raise ValueError("quality-v2 is sealed: use model_quality_lab.py for development selection and final qualification")
    started_at, started = utc_now(), time.perf_counter()
    torch.set_num_threads(config["threads"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    data_profile = config.get("data_profile", "legacy-v1")
    if data_profile == "quality-v2":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from model_quality_data import fixed_quality_data
        data = fixed_quality_data()
    else:
        data = fixed_data()
    write_json(output / "data_manifest.json", {"splits": data, "sha256": content_hash(data),
               "eval_lock": "fixed in source; never selected using final evaluation"})
    write_json(output / "config.resolved.json", config)
    model_path = config["model_path"]
    if model_path:
        model, tokenizer = load_model_and_tokenizer(Path(model_path))
        if model.data_profile != data_profile:
            raise ValueError("checkpoint data profile differs from the selected experiment protocol")
        base_logs = []
        base_optimizer = None
        origin = {"kind": "existing-local-character-checkpoint", "path": model_path, "sha256": file_hash(Path(model_path))}
    else:
        tokenizer = CharacterTokenizer()
        model = TinyTextLM(len(tokenizer.pieces), width=config.get("width", 32),
                           heads=config.get("heads", 2), layers=config.get("layers", 1))
        model.data_profile = data_profile
        base_logs, base_optimizer = train_sft(model, tokenizer, data["train"], config["base_steps"], config["learning_rate"],
                                            config["seed"], all_tokens=True)
        origin = {"kind": "locally-trained-from-random-initialization", "pretrained_general_llm": False}
    write_json(output / "tokenizer.json", tokenizer.to_dict())
    write_json(output / "base_train.json", base_logs)
    save_checkpoint(output / "base_checkpoint.pt", model, tokenizer, "base", base_optimizer, len(base_logs))
    baseline = evaluate(model, tokenizer, data, config["max_new_tokens"])
    write_json(output / "baseline_predictions.json", baseline)
    short = data["train"][:2] + [next(row for row in data["train"] if row["answer"].startswith("-"))]
    batch = build_batch(short, tokenizer, model.config["context"])
    checks = {"sft": sft_checks(model, batch), "verifier": verifier_checks()}
    write_json(output / "token_masks.json", mask_table(batch, tokenizer))
    sft_logs, sft_optimizer = train_sft(model, tokenizer, data["train"], config["sft_steps"], config["learning_rate"], config["seed"] + 1)
    save_checkpoint(output / "sft_checkpoint.pt", model, tokenizer, "sft", sft_optimizer, len(sft_logs))
    checks["sft_reload"] = reload_check(output / "sft_checkpoint.pt", model, tokenizer)
    sft_snapshot = frozen_copy(model)
    sft_evaluation = evaluate(model, tokenizer, data, config["max_new_tokens"])
    qualification = None
    if data_profile == "quality-v2":
        from model_quality_data import quality_gate
        qualification = quality_gate(sft_evaluation)
        checks["quality_gate"] = qualification
    allow_posttraining = qualification is None or qualification["qualified"]
    evaluations: dict[str, Any] = {"base": baseline, "sft": sft_evaluation}
    budgets: dict[str, Any] = {"base": budget(base_logs), "sft": budget(sft_logs), "no_update": {"optimizer_steps": 0, "effective_update_tokens": 0}}
    write_json(output / "sft_train.json", sft_logs)
    write_json(output / "sft_predictions.json", sft_evaluation)
    comparisons = {"sft_vs_base": paired_comparison(baseline, sft_evaluation)}
    completed = ["local_model", "real_batch", "sft", "reload", "frozen_evaluation"]
    final = model
    if config["mode"] in ("dpo", "all") and allow_posttraining:
        dpo_model = copy.deepcopy(model).requires_grad_(True)
        reference = frozen_copy(sft_snapshot)
        reference_hash = weight_hash(reference)
        optimizer = torch.optim.AdamW(dpo_model.parameters(), lr=config["dpo_learning_rate"])
        arith = [row for row in data["train"] if row["family"] != "copy"]
        chosen = build_batch(arith[:3], tokenizer, model.config["context"])
        rejected_records = [{**row, "answer": str(int(row["answer"]) + 1)} for row in arith[:3]]
        rejected = build_batch(rejected_records, tokenizer, model.config["context"])
        checks["dpo"] = dpo_checks(dpo_model, reference, chosen, rejected, config["beta"])
        logs = []
        rng = random.Random(config["seed"] + 2)
        for step in range(config["dpo_steps"]):
            records = [arith[rng.randrange(len(arith))] for _ in range(12)]
            wrong = [{**row, "answer": str(int(row["answer"]) + 1)} for row in records]
            chosen, rejected = build_batch(records, tokenizer, model.config["context"]), build_batch(wrong, tokenizer, model.config["context"])
            loss, terms = dpo_loss(dpo_model, reference, chosen, rejected, config["beta"])
            logs.append({"step": step + 1, "sample_ids": chosen["sample_ids"], "samples": len(records),
                         "effective_tokens": int(chosen["loss_mask"].sum() + rejected["loss_mask"].sum()),
                         "four_logprobs": {key: value.detach().tolist() for key, value in terms.items()},
                         "learning_rate": config["dpo_learning_rate"], "updated_at_utc": utc_now(),
                         **checked_step(loss, dpo_model, optimizer)})
        checks["dpo_reference_unchanged"] = weight_hash(reference) == reference_hash and all(p.grad is None for p in reference.parameters())
        if not checks["dpo_reference_unchanged"]:
            raise AssertionError("DPO reference updated")
        save_checkpoint(output / "dpo_checkpoint.pt", dpo_model, tokenizer, "dpo", optimizer, len(logs))
        checks["dpo_reload"] = reload_check(output / "dpo_checkpoint.pt", dpo_model, tokenizer)
        evaluations["dpo"] = evaluate(dpo_model, tokenizer, data, config["max_new_tokens"])
        write_json(output / "dpo_train.json", logs)
        write_json(output / "dpo_predictions.json", evaluations["dpo"])
        comparisons["dpo_vs_sft"] = paired_comparison(sft_evaluation, evaluations["dpo"])
        budgets["dpo"] = budget(logs)
        completed.append("dpo")
        final = dpo_model
    if config["mode"] in ("rlvr", "all") and allow_posttraining:
        # All three C branches share this exact immutable SFT starting point.
        rl_model = copy.deepcopy(sft_snapshot).requires_grad_(True)
        continue_model = copy.deepcopy(sft_snapshot).requires_grad_(True)
        reference = frozen_copy(sft_snapshot)
        arith = [row for row in data["train"] if row["family"] != "copy"]
        continuation_logs, continuation_optimizer = train_sft(continue_model, tokenizer, arith, config["rlvr_steps"],
                                                               config["learning_rate"], config["seed"] + 3)
        save_checkpoint(output / "continue_sft_checkpoint.pt", continue_model, tokenizer, "continue-sft", continuation_optimizer, len(continuation_logs))
        write_json(output / "continue_sft_train.json", continuation_logs)
        budgets["continue_sft"] = budget(continuation_logs)
        budgets["rlvr"] = train_rlvr(rl_model, tokenizer, reference, arith, output, config)
        checks["rlvr"] = budgets["rlvr"]
        save_checkpoint(output / "rlvr_checkpoint.pt", rl_model, tokenizer, "rlvr")
        checks["rlvr_reload"] = reload_check(output / "rlvr_checkpoint.pt", rl_model, tokenizer)
        evaluations["continue_sft"] = evaluate(continue_model, tokenizer, data, config["max_new_tokens"])
        evaluations["rlvr"] = evaluate(rl_model, tokenizer, data, config["max_new_tokens"])
        write_json(output / "continue_sft_predictions.json", evaluations["continue_sft"])
        write_json(output / "rlvr_predictions.json", evaluations["rlvr"])
        comparisons["continue_sft_vs_no_update"] = paired_comparison(sft_evaluation, evaluations["continue_sft"])
        comparisons["rlvr_vs_no_update"] = paired_comparison(sft_evaluation, evaluations["rlvr"])
        completed.extend(["online_rollout", "reward_replay"])
        if budgets["rlvr"]["policy_gradient_proved"]:
            completed.append("rlvr_update")
        final = rl_model
    save_checkpoint(output / "final_checkpoint.pt", final, tokenizer, config["mode"])
    checks["final_reload"] = reload_check(output / "final_checkpoint.pt", final, tokenizer)
    versions = git_versions()
    environment = {"python": sys.version, "executable": sys.executable, "torch": str(torch.__version__),
                   "platform": platform.platform(), "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
                   "parameter_count": sum(parameter.numel() for parameter in model.parameters())}
    write_json(output / "environment.json", environment)
    write_json(output / "source_versions.json", versions)
    write_json(output / "checks.json", checks)
    write_json(output / "comparisons.json", comparisons)
    write_json(output / "evaluation_protocol.json", {"splits_hash": content_hash(data), "protocol": baseline["protocol"],
                "template_hash": content_hash(TEMPLATE), "three_panels": list(baseline["panels"]),
                "regression_panel_baseline_qualified": None if qualification is None else
                    qualification["conditions"]["eval_regression_accuracy"] and qualification["conditions"]["eval_regression_format"],
                "qualification_checkpoint": "sft" if qualification is not None else "legacy-diagnostic-only"})
    if not allow_posttraining:
        status = "baseline_unqualified"
    else:
        status = "passed" if config["mode"] not in ("rlvr", "all") or budgets["rlvr"]["policy_gradient_proved"] else "sampling_passed_update_no_signal"
    manifest = {path.name: file_hash(path) for path in output.iterdir() if path.is_file()}
    summary = {"schema_version": SCHEMA, "lesson": f"{config['mode']}-model", "mode": config["mode"],
               "status": status, "started_at_utc": started_at, "finished_at_utc": utc_now(),
               "elapsed_seconds": time.perf_counter() - started, "config": config, "origin": origin,
               "model_config": model.config, "environment": environment, "source_versions": versions,
               "completed_stages": completed, "checks": checks, "baseline_qualification": qualification, "data_profile": data_profile,
               "metrics": {name: value["panels"] for name, value in evaluations.items()},
               "paired_comparisons": comparisons, "budgets": budgets, "artifact_manifest": manifest,
               "limitations": ["Small self-trained character text model; no general pretrained LLM claim.",
                               "One seed in this run; no ablation or independent repeats in default lab.",
                               "RLVR format shaping is declared; correctness improvement is not guaranteed.",
                               "C branches have equal evaluation budgets; training/sample/token budgets differ and are reported.",
                               "Regression baseline qualification is measured; zero baseline performance cannot prove ability retention.",
                               "Token clipped surrogate on saved prefixes, no KL regularizer; not an unbiased trajectory importance estimator.",
                               "Weights/tokenizer and optimizer/RNG saved; CLI supports weights loading, not exact interrupted resume.",
                               "Blind manual review is not performed; full annual project acceptance remains a separate specification."]}
    write_json(output / "summary.json", summary)
    report = ["# 模型后训练实验结果", "", f"实际运行模式：`{config['mode']}`；状态：`{status}`。",
              "", "这是本地从零训练的字符级因果文本模型。下面的生成、梯度和检查点来自本次真实运行。",
              "", "| 检查点 | 主任务正确 | 未见题族正确 | 回归任务正确 |", "| --- | --- | --- | --- |"]
    for name, evaluation in evaluations.items():
        values = [f"{panel['correct']}/{panel['total']}" for panel in evaluation["panels"].values()]
        report.append(f"| {name} | " + " | ".join(values) + " |")
    report += ["", "修改配置后请以 summary.json、逐题输出和预算为准。正收益不是通过条件。",
               "", "本轮只有一个 seed，未做独立重复、单因素消融或盲评；训练预算未匹配，不能据此宣称方法优劣。",
               "", "SFT/DPO 的数值与实际模型梯度检查见 checks.json；RLVR 的真实行为概率、组优势、ratio 和更新证据见 rollouts.json。",
               "", "继续加载可使用 --model-path 指向本次 final_checkpoint.pt。该入口加载权重和 tokenizer，另起新的训练，不承诺无差异断点续训。"]
    with (output / "结果说明.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(report) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("sft", "dpo", "rlvr", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None, help="load this lab's existing local .pt checkpoint (no download)")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data-profile", choices=("legacy-v1", "quality-v2"), default="legacy-v1")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--base-steps", type=int, default=120)
    parser.add_argument("--sft-steps", type=int, default=80)
    parser.add_argument("--dpo-steps", type=int, default=20)
    parser.add_argument("--rlvr-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--dpo-learning-rate", type=float, default=0.001)
    parser.add_argument("--rl-learning-rate", type=float, default=0.0005)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    for name in ("base_steps", "sft_steps", "dpo_steps", "rlvr_steps", "groups", "max_new_tokens", "threads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be between zero and 2**63-1")
    if args.width <= 0 or args.width > 256 or args.heads <= 0 or args.width % args.heads or not 1 <= args.layers <= 4:
        parser.error("model width must be 1..256, divisible by heads; layers must be 1..4")
    if args.group_size < 2:
        parser.error("--group-size must be at least two")
    for name in ("learning_rate", "dpo_learning_rate", "rl_learning_rate", "beta", "temperature"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 0 < args.clip < 1:
        parser.error("--clip must be between zero and one")
    if args.max_new_tokens > 12:
        parser.error("--max-new-tokens must be at most 12 for this short-answer teaching model")
    if args.model_path is not None and not args.model_path.is_file():
        parser.error("--model-path must be an existing local checkpoint file")
    return args


def main() -> int:
    args = parse_args()
    if args.data_profile == "quality-v2":
        print("quality-v2 is sealed: use model_quality_lab.py for development selection and final qualification", file=sys.stderr)
        return 2
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "output_dir"}
    try:
        output = reserve_output_dir(args.output_dir, f"{args.mode}-model")
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    try:
        result = run(config, output)
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed", "failed_at_utc": utc_now(),
                   "error_type": type(error).__name__, "reason": str(error), "config": config})
        raise
    print(f"model training {result['status']}; output: {output}")
    return 0 if result["status"] == "passed" else 4 if result["status"] == "baseline_unqualified" else 3


if __name__ == "__main__":
    raise SystemExit(main())
