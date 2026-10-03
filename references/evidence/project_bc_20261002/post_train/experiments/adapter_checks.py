"""Independent adapter mathematics, replay and exact-resume checks.

Every function receives an already-loaded local adapter. This module never loads
or downloads a model. Temporary updates in the resume check are restored.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
import random
from typing import Any
import torch
from torch.nn import functional as F
from pretrained_adapter import digest_state


def check_lora_frozen(adapter, *, full_base_hash: bool = False) -> dict[str, Any]:
    parameters = dict(adapter.model.named_parameters())
    trainable = {name for name, value in parameters.items() if value.requires_grad}
    if trainable != adapter.adapter_names or not trainable:
        raise AssertionError("trainable set differs from explicit LoRA adapters")
    expected = {name for name in adapter.model.state_dict() if name.endswith((".a", ".b"))}
    if trainable != expected:
        raise AssertionError("trainable set includes a base parameter or misses an adapter")
    frozen = {name: value for name, value in parameters.items() if name not in trainable}
    if not frozen or any(value.requires_grad or value.grad is not None for value in frozen.values()):
        raise AssertionError("base is trainable or acquired gradients")
    result = {"passed": True, "adapter_parameter_tensors": len(trainable),
              "adapter_parameters": sum(parameters[name].numel() for name in trainable),
              "frozen_parameters": sum(value.numel() for value in frozen.values()),
              "base_sha256": None}
    if full_base_hash:
        base = {name: value for name, value in adapter.model.state_dict().items() if name not in trainable}
        result["base_sha256"] = digest_state(base)
    return result


def validate_batch(adapter, batch, records=None, *, require_eos=True) -> dict[str, Any]:
    ids, attention, mask = batch["input_ids"], batch["attention"], batch["mask"]
    if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape != attention.shape or ids.shape != mask.shape:
        raise AssertionError("batch token/attention/response-mask shapes differ")
    if mask.dtype != torch.bool or len(batch["prefixes"]) != ids.shape[0]:
        raise AssertionError("response mask must be bool and prefixes must match rows")
    if len(batch["sample_ids"]) != ids.shape[0] or (records is not None and len(records) != ids.shape[0]):
        raise AssertionError("sample identities do not match batch rows")
    counts = []
    for row, prefix in enumerate(batch["prefixes"]):
        start, end = len(prefix), int(attention[row].sum())
        if not 0 < start < end or not torch.equal(ids[row, :start], torch.tensor(prefix, device=ids.device)):
            raise AssertionError("prompt tokens or response boundary differs")
        expected_attention = torch.arange(ids.shape[1], device=ids.device) < end
        if not torch.equal(attention[row].bool(), expected_attention):
            raise AssertionError("attention must be contiguous right padding")
        expected_mask = (torch.arange(ids.shape[1], device=ids.device) >= start) & expected_attention
        if not torch.equal(mask[row], expected_mask):
            raise AssertionError("loss includes prompt/padding or omits response/EOS")
        if not bool((ids[row, end:] == adapter.pad).all()):
            raise AssertionError("padding IDs differ from tokenizer padding token")
        response = ids[row, start:end].tolist()
        if adapter.eos in response[:-1]:
            raise AssertionError("response contains actions after EOS")
        if require_eos and response[-1] != adapter.eos:
            raise AssertionError("supervised answer must include terminal EOS")
        if records is not None:
            if prefix != adapter.prefix(records[row]) or batch["sample_ids"][row] != records[row]["id"]:
                raise AssertionError("actual record/prompt differs from batch identity")
            if require_eos:
                wanted = adapter.tokenizer.encode(records[row]["answer"], add_special_tokens=False) + [adapter.eos]
                if response != wanted:
                    raise AssertionError("supervised labels differ from unshifted actual answer/EOS")
        counts.append(len(response))
    return {"passed": True, "response_tokens": sum(counts), "per_sample_tokens": counts,
            "prompt_and_padding_supervised": False, "shift_count": 1}


def _max_gradient_error(first, second):
    errors = []
    for left, right in zip(first, second):
        if left is None or right is None:
            if left is not None or right is not None:
                raise AssertionError("independent gradients use different parameters")
        else:
            errors.append(float((left.detach() - right.detach()).abs().max()))
    return max(errors, default=0.0)


def check_sft(adapter, records, *, atol=2e-5):
    batch = adapter.batch(records)
    validation = validate_batch(adapter, batch, records)
    parameters = adapter.parameters()
    actual = adapter.sft_loss(batch)
    actual_value = float(actual.detach())
    actual_grad = torch.autograd.grad(actual, parameters, allow_unused=True)
    logits = adapter.forward(batch["input_ids"], batch["attention"], use_cache=False).logits[:, :-1].float()
    selected = batch["mask"][:, 1:]
    targets = batch["input_ids"][:, 1:]
    ce = F.cross_entropy(logits[selected], targets[selected])
    ce_value = float(ce.detach())
    expected_grad = torch.autograd.grad(ce, parameters, allow_unused=True)
    # Independent stable scalar formula on answer rows, in float64. No prompt
    # loss, pre-shifted labels, padding loss or omitted EOS can pass this oracle.
    chosen_logits = logits.detach()[selected].double()
    chosen_targets = targets[selected]
    maxima = chosen_logits.max(-1).values
    manual = (maxima + (chosen_logits - maxima[:, None]).exp().sum(-1).log()
              - chosen_logits.gather(-1, chosen_targets[:, None]).squeeze(-1)).mean()
    scalar_error = abs(actual_value - ce_value)
    manual_error = abs(actual_value - float(manual))
    gradient_error = _max_gradient_error(actual_grad, expected_grad)
    independent_logits = logits.detach().clone().requires_grad_(True)
    masked_ce = F.cross_entropy(independent_logits[selected], targets[selected])
    logits_gradient = torch.autograd.grad(masked_ce, independent_logits)[0]
    ignored_gradient = float(logits_gradient[~selected].abs().max()) if bool((~selected).any()) else 0.0
    if scalar_error > atol or manual_error > atol or gradient_error > atol or ignored_gradient != 0:
        raise AssertionError("actual SFT differs from independent CE/mask/adapter-gradient oracle")
    check_lora_frozen(adapter)
    return {"passed": True, "loss": actual_value, "cross_entropy": ce_value, "handwritten_loss": float(manual),
            "scalar_max_error": scalar_error, "manual_error": manual_error,
            "adapter_gradient_max_error": gradient_error, "ignored_logits_gradient_max": ignored_gradient,
            "batch": validation}


def check_reference(adapter, records, *, atol=2e-5):
    batch = adapter.batch(records)
    original = adapter.snapshot()
    snapshot = {name: value.detach().clone() for name, value in original.items()}
    snapshot_hash = digest_state(snapshot)
    with torch.no_grad():
        before = adapter.forward(batch["input_ids"], batch["attention"], snapshot=snapshot, use_cache=False).logits.detach().clone()
    try:
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.add_(0.01)
        output = adapter.forward(batch["input_ids"], batch["attention"], snapshot=snapshot, use_cache=False)
        error = float((output.logits.detach() - before).abs().max())
        if output.logits.requires_grad or error > atol or digest_state(snapshot) != snapshot_hash:
            raise AssertionError("functional reference changed, acquired a policy graph, or mutated its snapshot")
        if any(value.requires_grad or value.grad is not None for value in snapshot.values()):
            raise AssertionError("reference adapter snapshot acquired gradient")
        return {"passed": True, "snapshot_sha256": snapshot_hash, "max_logit_error": error,
                "reference_requires_grad": False}
    finally:
        adapter.load_snapshot(original)


def check_dpo(adapter, chosen_records, rejected_records, *, beta=0.2, atol=2e-5):
    if not math.isfinite(beta) or beta <= 0 or len(chosen_records) != len(rejected_records) or not chosen_records:
        raise ValueError("DPO beta or pair counts are invalid")
    for chosen, rejected in zip(chosen_records, rejected_records):
        if chosen["question"] != rejected["question"] or chosen["id"] != rejected["id"] or chosen["answer"] == rejected["answer"]:
            raise ValueError("DPO responses must be different answers to the same identified prompt")
    chosen, rejected = adapter.batch(chosen_records), adapter.batch(rejected_records)
    validate_batch(adapter, chosen, chosen_records); validate_batch(adapter, rejected, rejected_records)
    reference = adapter.snapshot(); reference_hash = digest_state(reference)
    _, policy_chosen, _ = adapter.logprobs(chosen)
    _, policy_rejected, _ = adapter.logprobs(rejected)
    _, reference_chosen, _ = adapter.logprobs(chosen, snapshot=reference)
    _, reference_rejected, _ = adapter.logprobs(rejected, snapshot=reference)
    if reference_chosen.requires_grad or reference_rejected.requires_grad:
        raise AssertionError("reference four-probability terms are not frozen")
    u = beta * (policy_chosen - policy_rejected - reference_chosen + reference_rejected)
    loss = -F.logsigmoid(u).mean()
    derivative = torch.autograd.grad(loss, u, retain_graph=True)[0]
    expected_derivative = (u.detach().sigmoid() - 1) / len(chosen_records)
    derivative_error = float((derivative - expected_derivative).abs().max())
    actual_grad = torch.autograd.grad(loss, adapter.parameters(), retain_graph=True, allow_unused=True)
    expected_grad = torch.autograd.grad(u, adapter.parameters(), grad_outputs=expected_derivative, allow_unused=True)
    gradient_error = _max_gradient_error(actual_grad, expected_grad)
    # Reference equals initialization, so both u=0 and loss=log(2) are exact
    # independent limits; swapped responses reverse the initial direction.
    initial_error = float(u.detach().abs().max())
    if initial_error > atol or abs(float(loss.detach()) - math.log(2)) > atol:
        raise AssertionError("DPO initialized policy/reference equality does not give log(2)")
    if derivative_error > atol or gradient_error > atol or digest_state(reference) != reference_hash:
        raise AssertionError("DPO differs from analytic derivative or mutated frozen reference")
    check_lora_frozen(adapter)
    return {"passed": True, "initial_loss": float(loss.detach()), "initial_u_max": initial_error,
            "analytic_derivative_max_error": derivative_error, "adapter_gradient_max_error": gradient_error,
            "four_logprobs": {"policy_chosen": policy_chosen.detach().cpu().tolist(),
                              "policy_rejected": policy_rejected.detach().cpu().tolist(),
                              "reference_chosen": reference_chosen.detach().cpu().tolist(),
                              "reference_rejected": reference_rejected.detach().cpu().tolist()},
            "reference_sha256": reference_hash, "reference_requires_grad": False}


def check_action_text(adapter, record, rollout):
    """Verify the reward-visible text preserves every nonterminal action.

    Only the final registered EOS is a terminator. A PAD/chat-control action
    before it remains visible and must never earn correctness or format reward.
    This oracle does not reuse the adapter's production decoding helper.
    """
    actions = rollout["response_token_ids"]
    if not actions or any(isinstance(token, bool) or not isinstance(token, int)
                          or not 0 <= token < adapter.model.config.vocab_size for token in actions):
        raise AssertionError("generated action IDs are empty or invalid")
    if adapter.eos in actions[:-1]:
        raise AssertionError("generated actions continue after registered EOS")
    terminal = actions[-1] == adapter.eos
    if terminal != (rollout["termination_reason"] == "eos"):
        raise AssertionError("cached rollout text/termination differs from actual action tokens")
    visible = actions[:-1] if terminal else actions
    decoded = adapter.tokenizer.decode(visible, skip_special_tokens=False)
    if decoded != rollout["text"]:
        raise AssertionError("cached rollout text/termination hides or changes actual action tokens")
    from model_training_lab import verify_answer
    reward = verify_answer(decoded, record["answer"], rollout["termination_reason"])
    special = set(getattr(adapter.tokenizer, "all_special_ids", (adapter.pad, adapter.eos)))
    controls = [token for token in visible if token in special]
    if controls and (reward["correct"] or reward["format_valid"]):
        raise AssertionError("a nonterminal special action acquired canonical answer reward")
    if reward != rollout["reward"]:
        raise AssertionError("cached rollout reward differs from independent decoded-action verification")
    return {"passed": True, "decoded_text": decoded, "action_tokens": len(actions),
            "nonterminal_special_actions": controls, "registered_eos_removed": terminal,
            "reward": reward}


def check_rollout_replay(adapter, records, rollouts, *, snapshot, atol=2e-5):
    if not records or len(records) != len(rollouts):
        raise ValueError("replay record/rollout counts differ")
    snapshot_hash = digest_state(snapshot)
    maximum, tokens = 0.0, 0
    for record, rollout in zip(records, rollouts):
        distribution = rollout["distribution"]
        temperature = distribution["temperature"]
        if not distribution["do_sample"] or distribution["top_k"] != 0 or distribution["top_p"] != 1.0 or distribution["forbidden_token_ids"]:
            raise ValueError("replay supports the recorded unconstrained multinomial distribution only")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("replay temperature is invalid")
        if rollout["sample_id"] != record["id"] or rollout["prompt_token_ids"] != adapter.prefix(record):
            raise AssertionError("rollout identity or exact token prefix differs")
        actions = rollout["response_token_ids"]
        saved = rollout["behavior_token_logprobs"]
        if not actions or len(saved) != len(actions) or any(not math.isfinite(value) for value in saved):
            raise AssertionError("cached behavior probability/action counts are invalid")
        if rollout["question"] != record["question"] or rollout["answer"] != record["answer"]:
            raise AssertionError("cached rollout question or target differs from source record")
        check_action_text(adapter, record, rollout)
        batch = adapter.batch([record], answers=[actions])
        validate_batch(adapter, batch, [record], require_eos=False)
        with torch.no_grad():
            lp, _, _ = adapter.logprobs(batch, snapshot=snapshot, temperature=temperature)
        replayed = lp[0][batch["mask"][0, 1:]].detach().cpu()
        error = float((replayed - torch.tensor(saved)).abs().max())
        maximum = max(maximum, error); tokens += len(actions)
        if error > atol:
            raise AssertionError("actual cached behavior probabilities do not replay under frozen old policy")
    if digest_state(snapshot) != snapshot_hash:
        raise AssertionError("replay mutated its old-policy snapshot")
    return {"passed": True, "rollouts": len(rollouts), "action_tokens": tokens,
            "max_behavior_logprob_error": maximum, "snapshot_sha256": snapshot_hash}


def check_verifier_boundaries():
    from model_training_lab import verify_answer
    cases = [("5", "5", "eos", "correct"), ("4", "5", "eos", "wrong"),
             ("", "5", "eos", "empty"), ("05", "5", "eos", "malformed"),
             ("5 ", "5", "eos", "malformed"), ("5\n", "5", "eos", "malformed"),
             ("+5", "5", "eos", "malformed"), ("-0", "0", "eos", "malformed"),
             ("NaN", "5", "eos", "malformed"),
             ("inf", "5", "eos", "malformed"), ("99999", "5", "eos", "too_long"),
             ("5", "5", "new_token_budget", "truncated"), ("-3", "-3", "eos", "correct")]
    results = []
    for text, answer, ending, expected in cases:
        value = verify_answer(text, answer, ending)
        if value["status"] != expected or not math.isfinite(value["total"]) or not 0 <= value["total"] <= 1.1:
            raise AssertionError("strict verifier format/truncation/extreme-value boundary failed")
        if value["correct"] != (expected == "correct"):
            raise AssertionError("reward correctness differs from independent exact answer")
        results.append({"text": text, "termination": ending, **value})
    return {"passed": True, "cases": results, "maximum_reward": 1.1}


def _same_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(_same_state(a, b) for a, b in zip(left, right))
    return left == right


def check_exact_resume(adapter, records, output_dir, *, learning_rate=0.001, contract_hash):
    if not records:
        raise ValueError("resume check requires nonempty real records")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    original = adapter.snapshot()
    original_gradients = [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in adapter.parameters()]
    global_python, global_torch = random.getstate(), torch.get_rng_state()
    global_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
    rng = random.Random(1729)
    generator = torch.Generator(device=adapter.device).manual_seed(1729)
    def advance():
        indices = [rng.randrange(len(records)) for _ in range(2)]
        torch_draw = torch.rand(3).tolist()
        rollout = adapter.generate(records[indices[0]], max_new_tokens=2, temperature=0.7,
                                   sample=True, generator=generator)
        loss = adapter.sft_loss(adapter.batch([records[index] for index in indices]))
        update = adapter.checked_step(loss, optimizer)
        return {"sample_indices": indices, "torch_draw": torch_draw,
                "sampled_actions": rollout["response_token_ids"], "update": update}
    try:
        first = advance()
        checkpoint = output / "step_000001.pt"
        adapter.save(checkpoint, optimizer, step=1, rng=rng, generator=generator, contract_hash=contract_hash)
        continuous = advance()
        expected_weights, expected_optimizer = adapter.snapshot(), copy.deepcopy(optimizer.state_dict())
        expected_sampler, expected_generator, expected_global = rng.getstate(), generator.get_state(), torch.get_rng_state()
        expected_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        restored_step = adapter.restore(checkpoint, optimizer, rng=rng, generator=generator, contract_hash=contract_hash)
        resumed = advance()
        comparisons = {"step": restored_step == 1, "sample_indices": continuous["sample_indices"] == resumed["sample_indices"],
                       "torch_draw": continuous["torch_draw"] == resumed["torch_draw"],
                       "sampled_actions": continuous["sampled_actions"] == resumed["sampled_actions"],
                       "weights": _same_state(expected_weights, adapter.snapshot()),
                       "optimizer": _same_state(expected_optimizer, optimizer.state_dict()),
                       "python_sampler_rng": _same_state(expected_sampler, rng.getstate()),
                       "independent_generator_rng": _same_state(expected_generator, generator.get_state()),
                       "global_torch_rng": _same_state(expected_global, torch.get_rng_state()),
                       "cuda_rng": _same_state(expected_cuda, torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])}
        if not all(comparisons.values()):
            raise AssertionError(f"exact interrupted resume differs: {comparisons}")
        check_lora_frozen(adapter)
        return {"passed": True, "comparisons": comparisons, "first_step": first,
                "continuous_second_step": continuous, "resumed_second_step": resumed,
                "checkpoint": str(checkpoint), "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
                "caller_policy_restored": True}
    finally:
        adapter.load_snapshot(original)
        for parameter, gradient in zip(adapter.parameters(), original_gradients):
            parameter.grad = gradient
        random.setstate(global_python); torch.set_rng_state(global_torch)
        if global_cuda: torch.cuda.set_rng_state_all(global_cuda)
