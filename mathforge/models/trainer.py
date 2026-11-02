"""Training steps for the self-play policy, run in their own process.

expert_iteration  supervised fine-tuning on verified traces. The data are the model's
                  own attempts, filtered by a verifier, so it is trained only on its
                  *correct* work (rejection-sampling fine-tuning, a.k.a. expert
                  iteration / ReST / STaR) and cannot reinforce its own mistakes.
grpo              group-relative policy optimisation on top of that: attempts at the
                  same prompt are compared with each other, the verifier's verdict is
                  the reward, and each attempt's log-likelihood is pushed up or down by
                  its advantage (reward minus group mean, over the group's spread), with
                  a KL penalty that keeps the policy near the base model.

Runs as ``python -m mathforge.models.trainer job.json`` so GPU memory is fully
released when it finishes, or as a Ray GPU task on a cluster. Produces a LoRA
adapter directory.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys


def build_examples(tok, rows, max_len, key: str | None = None):
    """Tokenised (ids, labels[, row[key]]) with the loss on completion tokens only."""
    from .policy import chat_format

    examples = []
    for r in rows:
        prompt = chat_format(tok, r["prompt"])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(r["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (p_ids + c_ids)[:max_len]
        labels = ([-100] * len(p_ids) + c_ids)[:max_len]
        if any(l != -100 for l in labels):
            examples.append((ids, labels) if key is None else (ids, labels, r[key]))
    return examples


def _load(job: dict):
    """Base model + LoRA (new, or continued from ``init_adapter``), ready to train."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .policy import dtype_kwargs, model_dtype

    t = job["trainer"]
    torch.manual_seed(job.get("seed", 0))
    random.seed(job.get("seed", 0))
    cuda = torch.cuda.is_available()
    dtype = model_dtype(torch)
    tok = AutoTokenizer.from_pretrained(job["model"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        job["model"], **dtype_kwargs(dtype), device_map={"": 0} if cuda else None, trust_remote_code=True)
    model.config.use_cache = False          # incompatible with gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if job.get("init_adapter"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, job["init_adapter"], is_trainable=True)
    else:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=t["lora_r"], lora_alpha=t["lora_alpha"], lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    return torch, tok, model, dtype


def _batch(torch, tok, batch, dev):
    L = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for j, x in enumerate(batch):
        ids[j, :len(x[0])] = torch.tensor(x[0])
        lab[j, :len(x[1])] = torch.tensor(x[1])
        att[j, :len(x[0])] = 1
    return ids.to(dev), att.to(dev), lab.to(dev)


class _Optim:
    """AdamW + cosine schedule, with loss scaling when the model runs in fp16."""

    def __init__(self, torch, model, lr, steps, dtype):
        from transformers import get_cosine_schedule_with_warmup

        self.torch, self.model = torch, model
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.0)
        self.sched = get_cosine_schedule_with_warmup(self.opt, max(1, steps // 20), steps)
        self.scaler = None
        if dtype == torch.float16:
            amp = getattr(torch, "amp", None)
            self.scaler = amp.GradScaler("cuda") if amp and hasattr(amp, "GradScaler") \
                else torch.cuda.amp.GradScaler()

    def backward(self, loss):
        (self.scaler.scale(loss) if self.scaler else loss).backward()

    def step(self):
        if self.scaler:
            self.scaler.unscale_(self.opt)
        self.torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        if self.scaler:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        self.sched.step()
        self.opt.zero_grad()


def _save(tok, model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def train(job: dict) -> str:
    """Expert iteration: supervised fine-tuning on verified traces."""
    torch, tok, model, dtype = _load(job)
    t = job["trainer"]
    with open(job["data"], encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    examples = build_examples(tok, rows, t["max_len"])
    bs, ga = t["batch_size"], t["grad_accum"]
    steps = max(1, math.ceil(len(examples) * t["epochs"] / (bs * ga)))
    opt = _Optim(torch, model, t["lr"], steps, dtype)
    dev = next(model.parameters()).device
    model.train()
    step, micro = 0, 0
    for _ in range(t["epochs"]):
        random.shuffle(examples)
        for i in range(0, len(examples), bs):
            ids, att, lab = _batch(torch, tok, examples[i:i + bs], dev)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            opt.backward(out.loss / ga)
            micro += 1
            if micro % ga == 0:
                opt.step()
                step += 1
                if step % 10 == 0:
                    print(f"[trainer] step {step}/{steps} loss {out.loss.item():.4f}", flush=True)
    return _save(tok, model, job["out_dir"])


def _token_logp(torch, logits, targets, chunk: int = 128):
    """log p(target) per position, chunked so a ~150k-token vocabulary fits in memory."""
    out = []
    for s in range(0, logits.shape[1], chunk):
        lg = logits[:, s:s + chunk].float()
        out.append(lg.gather(-1, targets[:, s:s + chunk].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, -1))
    return torch.cat(out, 1)


def train_grpo(job: dict) -> str:
    """GRPO step. Rows carry a precomputed group-relative ``advantage``.

    Loss per attempt: -advantage * mean token log-prob + beta * mean token KL, with the
    KL estimated against the base model (the adapter switched off) as in the GRPO paper:
    exp(ref - logp) - (ref - logp) - 1. One pass over fresh attempts from the current
    policy, so the importance ratio is 1 and needs no clipping.
    """
    torch, tok, model, dtype = _load(job)
    t = job["trainer"]
    g = t.get("grpo", {})
    beta = g.get("beta", 0.04)
    with open(job["data"], encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    examples = build_examples(tok, rows, t["max_len"], key="advantage")
    bs, ga = t["batch_size"], t["grad_accum"]
    steps = max(1, math.ceil(len(examples) / (bs * ga)))
    opt = _Optim(torch, model, g.get("lr", 1e-5), steps, dtype)
    dev = next(model.parameters()).device
    model.train()
    random.shuffle(examples)
    step, micro = 0, 0
    for i in range(0, len(examples), bs):
        batch = examples[i:i + bs]
        ids, att, lab = _batch(torch, tok, batch, dev)
        adv = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=dev)
        tgt = lab[:, 1:]
        mask = (tgt != -100).float()
        tgt = tgt.clamp(min=0)
        logp = _token_logp(torch, model(input_ids=ids, attention_mask=att).logits[:, :-1], tgt)
        with torch.no_grad(), model.disable_adapter():
            ref = _token_logp(torch, model(input_ids=ids, attention_mask=att).logits[:, :-1], tgt)
        d = ref - logp
        kl = torch.exp(d) - d - 1
        n = mask.sum(1).clamp(min=1)
        loss = (-adv * (logp * mask).sum(1) / n + beta * (kl * mask).sum(1) / n).mean()
        opt.backward(loss / ga)
        micro += 1
        if micro % ga == 0 or i + bs >= len(examples):
            opt.step()
            step += 1
            if step % 10 == 0:
                print(f"[trainer] grpo step {step}/{steps} loss {loss.item():.4f}", flush=True)
    return _save(tok, model, job["out_dir"])


def run(job: dict) -> str:
    return train_grpo(job) if job.get("algorithm") == "grpo" else train(job)


def train_remote(job: dict) -> dict[str, bytes]:
    """Ray GPU task: train on whichever node has a free GPU, ship the adapter back."""
    import tempfile

    local = dict(job)
    local["out_dir"] = tempfile.mkdtemp(prefix="mf_train_")
    data_path = os.path.join(local["out_dir"], "data.jsonl")
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(job["data_text"])
    local["data"] = data_path
    if job.get("init_adapter_files"):           # GRPO continues from an adapter shipped as bytes
        init = os.path.join(local["out_dir"], "init_adapter")
        os.makedirs(init)
        for name, blob in job["init_adapter_files"].items():
            with open(os.path.join(init, name), "wb") as fh:
                fh.write(blob)
        local["init_adapter"] = init
    out = run(local)
    files = {}
    for name in os.listdir(out):
        fp = os.path.join(out, name)
        if os.path.isfile(fp) and name != "data.jsonl":
            with open(fp, "rb") as fh:
                files[name] = fh.read()
    return files


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as fh:
        run(json.load(fh))
