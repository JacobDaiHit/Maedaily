"""Local frozen-pretrained causal LM with explicit, auditable LoRA and snapshots.

Loading is offline and requires a caller-supplied complete model directory.
Reference/old adapters use functional_call, sharing immutable base weights without
switching the live policy or keeping multiple copies of a large base model.
"""
from __future__ import annotations
import hashlib
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import math
from pathlib import Path
import random
import time
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

SYSTEM = "Answer with only one canonical decimal integer. No explanation, spaces, or leading zeros."


def digest_state(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 or not math.isfinite(alpha) or alpha <= 0: raise ValueError("invalid adapter rank/alpha")
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, inputs):
        return self.base(inputs) + F.linear(F.linear(inputs, self.a), self.b) * self.scale


class LocalAdapterLM:
    def __init__(self, local_path: Path, *, device="cpu", seed=17, rank=4, alpha=8):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        local_path = Path(local_path).resolve()
        required = ("config.json", "tokenizer_config.json", "model.safetensors")
        if not all((local_path / name).is_file() for name in required):
            raise ValueError("local model is incomplete; loading never downloads")
        if device == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA is unavailable")
        random.seed(seed); torch.manual_seed(seed)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True,
                    trust_remote_code=False, dtype=torch.float32, attn_implementation="eager").to(self.device)
        self.model.requires_grad_(False).eval()
        replaced = []
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in ("q_proj", "v_proj"):
                parent_name, leaf = name.rsplit(".", 1)
                setattr(self.model.get_submodule(parent_name), leaf, LoRALinear(module, rank, alpha))
                replaced.append(name)
        if not replaced: raise ValueError("model has no supported q_proj/v_proj linear modules")
        self.eos = self.tokenizer.eos_token_id
        self.pad = self.tokenizer.pad_token_id
        if self.pad is None: self.pad = self.eos
        self.config = {"local_path": str(local_path), "device": device, "dtype": "float32",
                       "rank": rank, "alpha": alpha, "targets": replaced, "seed": seed,
                       "system": SYSTEM, "eos": self.eos, "pad": self.pad}
        self.adapter_names = {name for name, p in self.model.named_parameters() if p.requires_grad}
        if self.adapter_names != {name for name in self.model.state_dict() if name.endswith((".a", ".b"))}:
            raise AssertionError("trainable set differs from registered adapter parameters")

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def snapshot(self):
        return {name: p.detach().clone() for name, p in self.model.named_parameters() if name in self.adapter_names}

    def validate_snapshot(self, state):
        if state.keys() != self.adapter_names: raise ValueError("adapter snapshot names differ")
        current=dict(self.model.named_parameters())
        if any(not isinstance(v,torch.Tensor) or v.shape!=current[k].shape or v.dtype!=current[k].dtype or not bool(torch.isfinite(v).all()) for k,v in state.items()):
            raise ValueError("adapter snapshot shape/dtype/finiteness differs")

    def load_snapshot(self, state):
        self.validate_snapshot(state)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in state: p.copy_(state[name].to(p.device))

    def forward(self, ids, attention=None, *, snapshot=None, **kwargs):
        inputs = {"input_ids": ids, "attention_mask": attention, **kwargs}
        if snapshot is None: return self.model(**inputs)
        self.validate_snapshot(snapshot)
        if any(v.requires_grad for v in snapshot.values()):
            raise ValueError("reference must be a detached exact adapter snapshot")
        return torch.func.functional_call(self.model, snapshot, (), inputs, strict=False)

    def prefix(self, record):
        question = record["question"]
        if question.startswith("copy "):
            user = "Normalize this decimal integer by removing leading zeros: " + question[5:]
        else:
            user = "Compute " + question.removesuffix("=") + "."
        encoded = self.tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},
                {"role":"user","content":user}], tokenize=True, add_generation_prompt=True)
        return list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)

    def batch(self, records, answers=None):
        if not records or (answers is not None and len(answers) != len(records)): raise ValueError("records/actions count differs or batch is empty")
        rows, masks, prefixes = [], [], []
        for index, record in enumerate(records):
            prefix = self.prefix(record)
            response = self.tokenizer.encode(record["answer"], add_special_tokens=False) + [self.eos] if answers is None else list(answers[index])
            if not response: raise ValueError("empty response action sequence")
            if any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < self.model.config.vocab_size for t in response): raise ValueError("invalid action token")
            if self.eos in response[:-1]: raise ValueError("actions continue after EOS")
            if len(prefix)+len(response) > 256: raise ValueError("sequence exceeds frozen limit; no silent truncation")
            rows.append(prefix+response); masks.append([False]*len(prefix)+[True]*len(response)); prefixes.append(prefix)
        width = max(map(len, rows))
        ids = torch.tensor([x+[self.pad]*(width-len(x)) for x in rows],device=self.device)
        attention = torch.tensor([[1]*len(x)+[0]*(width-len(x)) for x in rows],device=self.device)
        mask = torch.tensor([x+[False]*(width-len(x)) for x in masks],device=self.device)
        return {"input_ids":ids,"attention":attention,"mask":mask,"sample_ids":[r["id"] for r in records],"prefixes":prefixes}

    def logprobs(self, batch, *, snapshot=None, temperature=1.0):
        if not math.isfinite(temperature) or temperature <= 0: raise ValueError("invalid temperature")
        mask = batch["mask"][:,1:]
        # Project only positions needed by any response, keeping all causal
        # Transformer context. Other positions are zero placeholders, not logp.
        positions = torch.nonzero(mask.any(0), as_tuple=False).flatten()
        logits = self.forward(batch["input_ids"],batch["attention"],snapshot=snapshot,
                     use_cache=False,logits_to_keep=positions).logits.float()/temperature
        targets = batch["input_ids"][:,1:].index_select(1,positions)
        selected = logits.log_softmax(-1).gather(-1,targets[:,:,None]).squeeze(-1)
        token = torch.zeros_like(mask,dtype=selected.dtype).scatter(1,positions.expand(mask.shape[0],-1),selected)
        return token, (token*mask).sum(-1), mask.sum(-1)

    def sft_loss(self, batch):
        _, sums, counts = self.logprobs(batch)
        return -sums.sum()/counts.sum()

    @torch.no_grad()
    def generate(self, record, *, max_new_tokens=8, temperature=1.0, sample=False, generator=None, snapshot=None):
        from model_training_lab import verify_answer
        if temperature <= 0 or not math.isfinite(temperature): raise ValueError("invalid temperature")
        prefix=self.prefix(record); ids=torch.tensor([prefix],device=self.device)
        tokens=[]; probabilities=[]; ending="new_token_budget"; past=None
        started=time.perf_counter()
        for _ in range(max_new_tokens):
            out=self.forward(ids,snapshot=snapshot,use_cache=True,past_key_values=past,logits_to_keep=1)
            past=out.past_key_values
            lp=(out.logits[0,-1].float()/temperature).log_softmax(-1)
            token=int(torch.multinomial(lp.exp(),1,generator=generator)) if sample else int(lp.argmax())
            tokens.append(token); probabilities.append(float(lp[token])); ids=torch.tensor([[token]],device=self.device)
            if token==self.eos: ending="eos"; break
        text=self.tokenizer.decode(tokens[:-1] if ending=="eos" else tokens,skip_special_tokens=False)
        return {"sample_id":record["id"],"question":record["question"],"answer":record["answer"],
                "prompt_token_ids":prefix,"response_token_ids":tokens,"text":text,
                "termination_reason":ending,"model_token_logprobs":probabilities,
                "behavior_token_logprobs":probabilities if sample else [0.0]*len(tokens),
                "distribution":{"temperature":temperature,"top_k":0,"top_p":1.0,"forbidden_token_ids":[],"do_sample":sample},
                "elapsed_seconds":time.perf_counter()-started,
                "reward":verify_answer(text,record["answer"],ending)}

    @torch.no_grad()
    def generate_many(self, records, *, max_new_tokens=8, batch_size=8):
        from model_training_lab import verify_answer
        result=[]
        for start in range(0,len(records),batch_size):
            group=records[start:start+batch_size]; prefixes=[self.prefix(r) for r in group]
            width=max(map(len,prefixes))
            ids=torch.tensor([[self.pad]*(width-len(p))+p for p in prefixes],device=self.device)
            attention=torch.tensor([[0]*(width-len(p))+[1]*len(p) for p in prefixes],device=self.device)
            began=time.perf_counter()
            output=self.model.generate(input_ids=ids,attention_mask=attention,max_new_tokens=max_new_tokens,
                    do_sample=False,eos_token_id=self.eos,pad_token_id=self.pad,use_cache=True)
            elapsed=time.perf_counter()-began
            for index,record in enumerate(group):
                tokens=output[index,width:].tolist(); ending="new_token_budget"
                if self.eos in tokens:
                    tokens=tokens[:tokens.index(self.eos)+1]; ending="eos"
                text=self.tokenizer.decode(tokens[:-1] if ending=="eos" else tokens,skip_special_tokens=False)
                result.append({"sample_id":record["id"],"question":record["question"],"answer":record["answer"],
                     "prompt_token_ids":prefixes[index],"response_token_ids":tokens,"text":text,
                     "termination_reason":ending,"elapsed_seconds":elapsed/len(group),
                     "distribution":{"do_sample":False,"kind":"greedy-deterministic"},
                     "reward":verify_answer(text,record["answer"],ending)})
        return result

    def checked_step(self, loss, optimizer):
        if not loss.requires_grad or not bool(torch.isfinite(loss)): raise ValueError("loss is detached or nonfinite")
        expected={id(p) for p in self.parameters()}
        actual={id(p) for group in optimizer.param_groups for p in group["params"]}
        if expected!=actual: raise AssertionError("optimizer does not exactly match adapters")
        before=self.snapshot(); optimizer.zero_grad(set_to_none=True); loss.backward()
        grads=[p.grad for p in self.parameters() if p.grad is not None]
        if not grads or any(not bool(torch.isfinite(g).all()) for g in grads): raise AssertionError("invalid gradients")
        norm=float(torch.linalg.vector_norm(torch.cat([g.flatten() for g in grads])))
        if norm<=0: raise AssertionError("zero gradient is not an update")
        torch.nn.utils.clip_grad_norm_(self.parameters(),1.0,error_if_nonfinite=True); optimizer.step()
        after=self.snapshot()
        delta=math.sqrt(sum(float((after[k]-before[k]).square().sum()) for k in after))
        if delta<=0 or any(not bool(torch.isfinite(v).all()) for v in after.values()): raise AssertionError("invalid parameter update")
        if any(p.grad is not None for p in self.model.parameters() if not p.requires_grad): raise AssertionError("frozen base acquired gradient")
        return {"loss":float(loss.detach()),"gradient_norm":norm,"parameter_delta_norm":delta,
                "weight_before":digest_state(before),"weight_after":digest_state(after)}

    def save(self,path,optimizer=None,*,step=0,rng=None,generator=None,contract_hash=None):
        value={"schema":"maedaily-local-adapter-v1","config":self.config,"adapters":self.snapshot(),
               "optimizer":None if optimizer is None else optimizer.state_dict(),"step":step,
               "python_rng":None if rng is None else rng.getstate(),"torch_rng":torch.get_rng_state(),
               "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               "generator_rng":None if generator is None else generator.get_state(),
               "contract_hash":contract_hash}
        with Path(path).open("xb") as stream: torch.save(value,stream)
        return digest_state(value["adapters"])

    def restore(self,path,optimizer=None,*,rng=None,generator=None,contract_hash=None):
        value=torch.load(path,map_location="cpu",weights_only=True)
        if value["schema"]!="maedaily-local-adapter-v1" or value["config"]!=self.config or value["contract_hash"]!=contract_hash:
            raise ValueError("checkpoint schema/config/data contract differs")
        self.load_snapshot(value["adapters"])
        if optimizer is not None:
            if value["optimizer"] is None: raise ValueError("optimizer state absent")
            optimizer.load_state_dict(value["optimizer"])
        if rng is not None:
            if value["python_rng"] is None: raise ValueError("data sampler state absent")
            rng.setstate(value["python_rng"])
        if generator is not None:
            if value.get("generator_rng") is None: raise ValueError("sampling generator state absent")
            generator.set_state(value["generator_rng"].cpu())
        torch.set_rng_state(value["torch_rng"].cpu())
        if value["cuda_rng"]: torch.cuda.set_rng_state_all([x.cpu() for x in value["cuda_rng"]])
        return value["step"]
