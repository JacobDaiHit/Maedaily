"""Runtime receipts for caller-loaded B/C models; never load or download weights.

The resource probe temporarily updates adapters, then restores caller policy,
gradients, module modes and relevant RNG streams. OS CPU peaks cover process
lifetime; CUDA peaks cover a separately reset probe and cannot be restored.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _version(distribution: str, module: str) -> tuple[str | None, str]:
    loaded = sys.modules.get(module)
    if loaded is not None and getattr(loaded, "__version__", None) is not None:
        return str(loaded.__version__), "loaded module __version__"
    try:
        return importlib.metadata.version(distribution), "current interpreter distribution metadata"
    except importlib.metadata.PackageNotFoundError:
        return None, "not installed in current interpreter"


def _cpu_memory() -> dict[str, Any]:
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
            peak, method = counters.PeakWorkingSetSize, "Windows PeakWorkingSetSize"
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak *= 1 if sys.platform == "darwin" else 1024
            method = "getrusage RUSAGE_SELF.ru_maxrss"
        return {"status": "measured", "peak_resident_bytes": int(peak), "method": method,
                "scope": "whole process lifetime including previously loaded native tensors; cannot reset CPU peak"}
    except (ImportError, AttributeError, OSError) as error:
        return {"status": "not_measured", "peak_resident_bytes": None, "reason": str(error)}


def environment_receipt(adapter, load_seconds: float) -> dict[str, Any]:
    import torch
    if isinstance(load_seconds, bool) or not isinstance(load_seconds, (int, float)) or not math.isfinite(load_seconds) or load_seconds < 0:
        raise ValueError("load_seconds must be actual finite nonnegative elapsed time")
    versions, sources = {}, {}
    for distribution, module in (("torch", "torch"), ("transformers", "transformers"),
                                 ("huggingface-hub", "huggingface_hub"), ("safetensors", "safetensors")):
        versions[module], sources[module] = _version(distribution, module)
    parameters = list(adapter.model.parameters())
    device = torch.device(adapter.device)
    gpu: dict[str, Any] = {"applicable": device.type == "cuda"}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu.update(index=index, name=properties.name, total_memory_bytes=properties.total_memory,
                   compute_capability=[properties.major, properties.minor], cuda_runtime=torch.version.cuda)
        executable = shutil.which("nvidia-smi")
        if executable:
            try:
                result = subprocess.run([executable, "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3)
                gpu["driver_probe"] = {"status": "measured" if result.returncode == 0 else "failed",
                    "output": result.stdout.strip()[:4096], "error": result.stderr.strip()[:1024]}
            except (OSError, subprocess.TimeoutExpired) as error:
                gpu["driver_probe"] = {"status": "not_measured", "reason": str(error)}
        else:
            gpu["driver_probe"] = {"status": "not_measured", "reason": "nvidia-smi unavailable"}
    return {"schema_version": "maedaily-reference-environment-v1", "python_executable": sys.executable,
            "python": sys.version, "platform": platform.platform(), "versions": versions,
            "version_sources": sources, "device": str(device),
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
            "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "load_seconds": float(load_seconds), "load_cpu_memory": _cpu_memory(), "gpu": gpu}


def _frozen_hash(adapter) -> str:
    """Hash all frozen parameters/buffers in bounded chunks, without a base clone."""
    import torch
    names = {name for name, parameter in adapter.model.named_parameters() if parameter.requires_grad}
    digest = hashlib.sha256()
    for name, value in sorted(adapter.model.state_dict().items()):
        if name in names:
            continue
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        flat = value.detach().reshape(-1)
        for offset in range(0, flat.numel(), 1024 * 1024):
            raw = flat[offset:offset + 1024 * 1024].contiguous().cpu().view(torch.uint8)
            digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def resource_probe(adapter, records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    if not records:
        raise ValueError("resource probe requires real records")
    device = torch.device(adapter.device)
    parameters = list(adapter.parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("probe requires actual trainable adapter parameters")
    policy = adapter.snapshot()
    gradients = [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]
    modes = [(module, module.training) for module in adapter.model.modules()]
    python_rng, cpu_rng = random.getstate(), torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    frozen_before = _frozen_hash(adapter)
    cpu_before = _cpu_memory()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    try:
        if device.type == "cuda":
            sync(); torch.cuda.reset_peak_memory_stats(device)
        batch = adapter.batch(records)
        input_tokens = int(batch["attention"].sum())
        target_tokens = int(batch["mask"][:, 1:].sum())
        if target_tokens <= 0:
            raise ValueError("probe batch has no actual supervised target tokens")
        optimizer = torch.optim.AdamW(parameters, lr=1e-6)
        sync(); started = time.perf_counter()
        loss = adapter.sft_loss(batch)
        sync(); forward_seconds = time.perf_counter() - started
        sync(); started = time.perf_counter()
        step = adapter.checked_step(loss, optimizer)
        sync(); backward_optimizer_seconds = time.perf_counter() - started
        sync(); started = time.perf_counter()
        generation = adapter.generate(records[0], max_new_tokens=2, temperature=1.0, sample=False)
        sync(); generation_seconds = time.perf_counter() - started
        generated = len(generation["response_token_ids"])
        cuda_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        result = {"schema_version": "maedaily-reference-resource-probe-v1", "status": "passed",
            "load_measurement": "caller-measured environment_receipt.load_seconds; object is already loaded here",
            "device": str(device), "forward_seconds": forward_seconds,
            "backward_optimizer_seconds": backward_optimizer_seconds, "generation_seconds": generation_seconds,
            "input_tokens": input_tokens, "target_tokens": target_tokens, "generated_tokens": generated,
            "generation_prompt_tokens": len(generation["prompt_token_ids"]), "samples": len(records),
            "source_sample_ids": [record["id"] for record in records], "temporary_optimizer_steps": 1,
            "update_check": step, "generation_termination": generation["termination_reason"],
            "cpu_before": cpu_before, "cpu_after": _cpu_memory(), "cuda_peak_allocated_bytes": cuda_peak,
            "cuda_peak_scope": "isolated forward/backward/update/generation after peak reset; loaded model included; counters cannot be restored" if device.type == "cuda" else "not applicable",
            "frozen_base_sha256": frozen_before, "state_restored": True}
    finally:
        adapter.load_snapshot(policy)
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient
        for module, training in modes:
            module.training = training
        random.setstate(python_rng); torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if _frozen_hash(adapter) != frozen_before:
            raise AssertionError("resource probe changed frozen base parameters or buffers")
    result["frozen_base_unchanged"] = True
    return result


def token_alignment(adapter, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("alignment requires real records")
    batch = adapter.batch(records)
    ids, attention, masks = (batch[key].detach().cpu().tolist() for key in ("input_ids", "attention", "mask"))
    output = []
    for index, record in enumerate(records):
        if batch["sample_ids"][index] != record["id"]:
            raise ValueError("batch/source sample identities differ")
        prefix_length = len(batch["prefixes"][index])
        for position, token_id in enumerate(ids[index]):
            present, supervised = bool(attention[index][position]), bool(masks[index][position])
            role = "PAD" if not present else "prompt" if position < prefix_length else "EOS" if token_id == adapter.eos else "answer"
            readable = adapter.tokenizer.convert_ids_to_tokens(token_id) if hasattr(adapter.tokenizer, "convert_ids_to_tokens") else adapter.tokenizer.decode([token_id], skip_special_tokens=False)
            output.append({"source_sample_id": record["id"], "batch_row": index, "position": position,
                "token_id": token_id, "readable_token": readable, "role": role,
                "attention_mask": int(present), "label": token_id if supervised else -100,
                "loss_mask": int(supervised), "target_prediction_position": position - 1,
                "supervision": "unshifted label; logits at position-1 predicts this target"})
    return output


def tokenization_manifest(adapter, data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    vocab = adapter.tokenizer.get_vocab()
    template = getattr(adapter.tokenizer, "chat_template", None)
    splits = {}
    for split, records in sorted(data.items()):
        if not records:
            raise ValueError(f"empty tokenization split: {split}")
        batch = adapter.batch(records)
        all_ids, attentions, masks = (batch[key].detach().cpu().tolist() for key in ("input_ids", "attention", "mask"))
        rows = []
        for index, record in enumerate(records):
            valid = int(sum(attentions[index]))
            prompt = len(batch["prefixes"][index])
            response = int(sum(masks[index]))
            rows.append({"source_sample_id": record["id"], "family_id": record.get("family_id"),
                "record_sha256": _hash(record), "input_tokens": valid, "prompt_tokens": prompt,
                "answer_tokens_excluding_eos": response - 1, "eos_tokens": 1, "target_tokens": response,
                "input_ids_sha256": _hash(all_ids[index][:valid]),
                "loss_mask_sha256": _hash(masks[index][:valid]), "truncated": False})
        splits[split] = {"records": rows, "samples": len(rows),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "target_tokens": sum(row["target_tokens"] for row in rows),
            "tokenized_records_sha256": _hash(rows)}
    trainable = [{"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                  "parameters": parameter.numel()} for name, parameter in adapter.model.named_parameters() if parameter.requires_grad]
    return {"schema_version": "maedaily-reference-tokenization-v1", "tokenizer": {
        "class": f"{type(adapter.tokenizer).__module__}.{type(adapter.tokenizer).__qualname__}",
        "name_or_path": str(getattr(adapter.tokenizer, "name_or_path", "")),
        "vocab_size": len(vocab), "vocab_sha256": _hash(vocab), "chat_template_sha256": _hash(template),
        "serialization_sha256": _hash({"chat_template": template, "system": adapter.config.get("system")})},
        "special_tokens": {"pad": adapter.pad, "eos": adapter.eos,
                           "bos": getattr(adapter.tokenizer, "bos_token_id", None)},
        "padding_side": "right in supervised adapter.batch; PAD and EOS distinguished by attention/position",
        "loss": "answer plus EOS; unshifted labels; logits[:, :-1] predict token[:, 1:]; whole-batch target-token mean",
        "truncation": {"maximum_sequence_tokens": 256, "policy": "reject overlength; no silent truncation"},
        "trainable_parameters": trainable, "splits": splits,
        "data_sha256": _hash(data), "stable_content_sha256": _hash(splits)}
