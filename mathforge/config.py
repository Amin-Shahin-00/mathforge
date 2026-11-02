"""Configuration: built-in defaults deep-merged with a YAML file and CLI overrides."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml


class _Loader(yaml.SafeLoader):
    """YAML with the YAML 1.2 float rule: PyYAML follows YAML 1.1, where 1e-10 (no dot)
    is a *string*, so `lr: 1e-4` or `--set success_nmse=1e-10` silently broke every task."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                   |[-+]?(?:[0-9][0-9_]*)[eE][-+]?[0-9]+
                   |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                   |[-+]?\.(?:inf|Inf|INF)
                   |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."))


def yaml_load(text):
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - a SafeLoader subclass

DEFAULTS: dict[str, Any] = {
    "run": {
        "dir": "runs/default",          # where the database, checkpoints and logs live
        "seed": 0,
        "max_rounds": 0,                # 0 = run forever
        "checkpoint_every": 1,          # rounds
        "report_every": 5,              # rounds between writing discoveries.md
        "loggers": [],                  # optional experiment tracking: wandb, mlflow
        "wandb_project": "mathforge",
        "mlflow_experiment": "mathforge",
    },
    "compute": {
        "backend": "local",             # local | ray
        "workers": 0,                   # 0 = all CPU cores
        "ray_address": "auto",          # "auto" joins an existing Ray cluster
        "ray_max_retries": 3,           # re-runs of a task whose machine vanished
    },
    "curriculum": {
        "target_success": 0.5,          # keep problems at the edge of ability
        "band": 0.15,
        "step": 1,
        "min_level": 1,
        "max_level": 50,
    },
    "engines": {
        "constants": {
            "enabled": True,
            "tasks_per_round": 64,
            "candidates_per_task": 400,
            "targets": ["pi", "e", "ln2", "zeta3", "catalan", "sqrt2", "phi",
                        "euler_gamma", "zeta5", "pi2"],
            "depth": 300,                # continued-fraction depth for screening
            "screen_digits": 18,         # convergence needed before identification
            "verify_depth": 2000,
            "verify_digits": 50,         # digits that must match at verification
            "max_relation_coeff": 200,   # PSLQ coefficient bound
            "mitm_coeff": 6,             # hash-table coefficient bound (fast path)
            "pslq_fraction": 0.1,        # share of limits also searched with full PSLQ
            "explore_fraction": 0.5,     # rest is exploitation around known hits
            "structured_fraction": 0.25,  # share of tasks searching the Apery-like family
            "structured_pslq_fraction": 0.3,
            "structured_max_deg": 5,     # deg a = 5, deg b = 10: the zeta(5) analogue of Apery
        },
        "laws": {
            "enabled": True,
            "tasks_per_round": 8,
            "population": 500,
            "generations": 60,
            "samples": 200,
            "noise": 0.0,
            "success_nmse": 1e-10,
            "fit_constants": True,      # free constants fitted by least squares (PySR-style)
            "const_prob": 0.08,         # share of new leaves that are fitted constants (0.15: worse and slower)
            "units": True,              # dimensional analysis when the inputs' units are known
            "units_penalty": 2.0,       # score penalty (log10 NMSE units) for inconsistent formulas
            "max_library": 40,
            "datasets": [],             # extra CSVs: [{path:..., target:...}]
        },
        "selfplay": {
            "enabled": False,           # needs an LLM; turn on in cloud configs
            "verifier": "sympy",        # sympy | lean
            "policy": {
                "backend": "hf",        # vllm | hf | mock (mock = tests only)
                "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
                "max_new_tokens": 512,
                "temperature": 0.8,
                "samples_per_problem": 8,
                "tensor_parallel": 1,       # GPUs per model copy (vllm)
                "num_actors": 0,            # model copies on a Ray cluster; 0 = one per GPU group
                "gpu_memory_utilization": 0.85,
                "max_model_len": 4096,
                "batch_size": 8,            # hf backend only
            },
            "problems_per_round": 128,
            "proposer_fraction": 0.25,   # share of problems invented by the model itself
            "train_every": 5,            # rounds between expert-iteration updates
            "min_new_traces": 64,
            "max_traces": 20000,         # most recent verified traces used per training step
            "eval_every": 5,             # rounds between held-out evaluations (sympy mode)
            "eval_levels": [1, 3, 5, 7, 9],
            "eval_per_level": 4,
            "trainer": {
                "lora_r": 32,
                "lora_alpha": 64,
                "lr": 1e-4,
                "epochs": 1,
                "batch_size": 8,
                "grad_accum": 4,
                "max_len": 1024,
                "keep_policy_loaded": False,  # True only if you have a spare GPU for training
                # expert_iteration: supervised on verified traces, retrained from the base each time.
                # grpo: one expert-iteration step as warm start, then GRPO steps that continue
                # from the current adapter, rewarded by the verifier.
                "algorithm": "expert_iteration",
                "grpo": {
                    "lr": 1.0e-5,
                    "beta": 0.04,             # KL penalty towards the base model
                    "min_rollouts": 256,      # attempts (in groups with a signal) needed for a step
                    "max_rollouts": 8192,
                },
            },
            "lean": {
                "project_dir": "lean_workspace",
                "timeout_s": 120,
                "theorem_file": "",      # optional .jsonl (e.g. miniF2F), .lean file or directory (PutnamBench)
                "theorem_split": "",     # keep only this split of a .jsonl file, e.g. "test" or "valid"
                "header": "",            # file header; empty = Mathlib default in verify/lean.py
                "repair_samples": 2,     # fix attempts per failed theorem, shown Lean's error (0 = off)
                "use_library": True,     # offer proved theorems as lemmas for later proofs
                "library_size": 3,       # lemmas offered per theorem
            },
        },
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(s: str) -> Any:
    try:
        return yaml_load(s)
    except yaml.YAMLError:
        return s


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply dotted overrides like ``engines.constants.depth=500``."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like a.b.c=value, got {item!r}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(val)
    return cfg


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    user: dict = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    cfg = apply_overrides(cfg, overrides)
    Path(cfg["run"]["dir"]).mkdir(parents=True, exist_ok=True)
    return cfg
