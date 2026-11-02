"""Language-model policies that propose problems and write solutions/proofs.

Backends
--------
vllm   fastest; batched sampling with hot-swappable LoRA adapters (Linux + NVIDIA GPU)
hf     plain transformers + peft; works anywhere torch works (slow on CPU)
mock   a scripted stand-in used ONLY by the test-suite to exercise the loop
       without a GPU. It is not a learner and must never be used for research.

``PolicyPool`` puts one policy on every GPU of a Ray cluster and shards each
batch of prompts across them, so adding machines adds throughput linearly.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
import shutil
import tempfile
import traceback


def dtype_kwargs(dtype) -> dict:
    """transformers >= 4.56 calls it ``dtype``; older versions ``torch_dtype``."""
    import transformers
    from packaging.version import Version

    key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return {key: dtype}


def model_dtype(torch):
    """bf16 where the GPU computes it natively (Ampere and newer); fp16 on older GPUs,
    where bf16 is only emulated (Turing: RTX 20xx, T4) and very slow; fp32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    try:
        native = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:                       # older torch: no emulation flag
        native = torch.cuda.get_device_capability(0)[0] >= 8
    return torch.bfloat16 if native else torch.float16


def chat_format(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
    return prompt + "\n"


class Policy:
    def generate(self, prompts: list[str], n: int, temperature: float,
                 max_new_tokens: int) -> list[list[str]]:
        raise NotImplementedError

    def set_adapter(self, path: str | None, version: int) -> None:
        pass

    def unload(self) -> None:
        pass


# ----------------------------------------------------------------------------- vLLM

class VLLMPolicy(Policy):
    def __init__(self, model: str, tensor_parallel: int = 1, lora_r: int = 32,
                 gpu_memory_utilization: float = 0.85, max_model_len: int = 4096):
        from vllm import LLM

        self.llm = LLM(model=model, enable_lora=True, max_lora_rank=max(8, lora_r),
                       tensor_parallel_size=tensor_parallel, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=True)
        self.tok = self.llm.get_tokenizer()
        self.lora = None

    def set_adapter(self, path, version):
        from vllm.lora.request import LoRARequest

        self.lora = LoRARequest(f"mathforge_v{version}", version, path) if path else None

    def generate(self, prompts, n, temperature, max_new_tokens):
        from vllm import SamplingParams

        sp = SamplingParams(n=n, temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                            max_tokens=max_new_tokens)
        texts = [chat_format(self.tok, p) for p in prompts]
        outs = self.llm.generate(texts, sp, lora_request=self.lora, use_tqdm=False)
        return [[o.text for o in out.outputs] for out in outs]

    def unload(self):
        import gc

        import torch

        del self.llm
        gc.collect()
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- transformers

class HFPolicy(Policy):
    def __init__(self, model: str, batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        cuda = torch.cuda.is_available()
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.base = AutoModelForCausalLM.from_pretrained(
            model, **dtype_kwargs(model_dtype(torch)), device_map="auto" if cuda else None,
            trust_remote_code=True)
        self.model = self.base
        self.model.eval()
        self.bs = batch_size
        self.adapters: set[str] = set()

    def set_adapter(self, path, version):
        if not path:
            return
        from peft import PeftModel

        name = f"v{version}"
        if self.model is self.base:
            self.model = PeftModel.from_pretrained(self.base, path, adapter_name=name)
        elif name not in self.adapters:
            self.model.load_adapter(path, adapter_name=name)
        self.model.set_adapter(name)
        self.adapters.add(name)
        self.model.eval()

    def generate(self, prompts, n, temperature, max_new_tokens):
        torch = self.torch
        out: list[list[str]] = []
        dev = next(self.model.parameters()).device
        for i in range(0, len(prompts), self.bs):
            texts = [chat_format(self.tok, p) for p in prompts[i:i + self.bs]]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(dev)
            sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.95} \
                if temperature > 0 else {"do_sample": False}
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, num_return_sequences=n,
                                          pad_token_id=self.tok.pad_token_id, **sampling)
            new = gen[:, enc["input_ids"].shape[1]:]
            dec = self.tok.batch_decode(new, skip_special_tokens=True)
            for j in range(len(texts)):
                out.append(dec[j * n:(j + 1) * n])
        return out

    def unload(self):
        import gc

        del self.model, self.base
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- mock (tests only)

class MockPolicy(Policy):
    """Scripted stand-in for tests. Uses SymPy as an oracle and gets *more accurate
    as the adapter version increases*, so the training/eval plumbing can be
    exercised end-to-end on a laptop. It does not learn anything."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.version = 0

    def set_adapter(self, path, version):
        self.version = version

    def _solve(self, prompt: str) -> str:
        import sympy as sp

        from ..verify.sympy_tasks import time_limit

        x, n, k = sp.symbols("x n k")
        skill = min(0.95, 0.35 + 0.15 * self.version)
        lean = "```lean" in prompt
        if self.rng.random() > skill:     # a wrong attempt (for Lean: one that fails to compile)
            return "```lean4\nby\n  bad_tactic\n```" if lean else "I think the answer is\nANSWER: x**2"
        if lean:
            return "```lean\nby\n  omega\n```"
        try:
            with time_limit(5):
                m = re.search(r"f\(x\) = (.*) with respect to x", prompt)
                if m:
                    return "ANSWER: " + sp.sstr(sp.integrate(sp.sympify(m.group(1)), x))
                m = re.search(r"equation (.*) = 0\.", prompt)
                if m:
                    sols = [s for s in sp.solve(sp.sympify(m.group(1)), x) if s.is_real]
                    return "ANSWER: " + sp.sstr(sols)
                m = re.search(r"k = 1\.\.n of (.*)\.", prompt)
                if m:
                    return "ANSWER: " + sp.sstr(sp.expand(sp.summation(sp.sympify(m.group(1)), (k, 1, n))))
        except Exception:  # noqa: BLE001
            pass
        return "ANSWER: 0"

    def _propose(self, prompt: str) -> str:
        if "Lean" in prompt:
            a = self.rng.randint(1, 9)
            return f"```lean\ntheorem conj_{a} (n : Nat) : n + {a} = {a} + n := by sorry\n```"
        a, b = self.rng.randint(1, 6), self.rng.randint(-6, -1)
        if "polynomial equation" in prompt:
            return f"PROBLEM: x**2 - ({a + b})*x + ({a * b}) = 0"
        if "p(k)" in prompt:
            return f"PROBLEM: {a}*k**2 + {self.rng.randint(1, 5)}*k"
        f = self.rng.choice(["x*exp(x)", "sin(x)*cos(x)", "x**3*log(x)", "1/(x**2+4)", "x*sin(2*x)"])
        return f"PROBLEM: {f} + {self.rng.randint(1, 5)}*x"

    def generate(self, prompts, n, temperature, max_new_tokens):
        return [[(self._propose(p) if "Invent" in p else self._solve(p)) for _ in range(n)]
                for p in prompts]


# ----------------------------------------------------------------------------- factory + pool

def make_policy(pcfg: dict, lora_r: int = 32, seed: int = 0) -> Policy:
    b = pcfg["backend"]
    if b == "vllm":
        return VLLMPolicy(pcfg["model"], pcfg.get("tensor_parallel", 1), lora_r,
                          pcfg.get("gpu_memory_utilization", 0.85), pcfg.get("max_model_len", 4096))
    if b == "hf":
        return HFPolicy(pcfg["model"], pcfg.get("batch_size", 8))
    if b == "mock":
        return MockPolicy(seed)
    raise ValueError(f"unknown policy backend {b!r}")


class ProcessPolicy(Policy):
    """Hosts a policy in its own process. Ending that process is the only fully
    reliable way to give GPU memory back (needed before each training step)."""

    def __init__(self, pcfg: dict, lora_r: int, seed: int):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        # not a daemon: vLLM starts helper processes of its own
        self.proc = ctx.Process(target=_policy_server, args=(child, pcfg, lora_r, seed))
        self.proc.start()
        status, detail = self._recv()
        if status != "ready":
            self.proc.join(5)
            raise RuntimeError(f"model failed to load:\n{detail}")

    def _recv(self):
        # never block forever: notice if the model process dies (e.g. out of memory)
        while not self.conn.poll(1.0):
            if not self.proc.is_alive():
                raise RuntimeError(f"model process exited unexpectedly (code {self.proc.exitcode})")
        return self.conn.recv()

    def _call(self, method, *args):
        self.conn.send((method, args))
        ok, result = self._recv()
        if not ok:
            raise RuntimeError(f"policy.{method} failed:\n{result}")
        return result

    def generate(self, prompts, n, temperature, max_new_tokens):
        return self._call("generate", prompts, n, temperature, max_new_tokens)

    def set_adapter(self, path, version):
        self._call("set_adapter", path, version)

    def unload(self):
        try:
            self.conn.send(("exit", ()))
            self.proc.join(60)
        except (BrokenPipeError, EOFError, OSError):
            pass
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(10)


def _policy_server(conn, pcfg, lora_r, seed):
    try:
        policy = make_policy(pcfg, lora_r, seed)
    except Exception:  # noqa: BLE001
        conn.send(("error", traceback.format_exc()))
        return
    conn.send(("ready", None))
    while True:
        try:
            method, args = conn.recv()
        except EOFError:
            break
        if method == "exit":
            break
        try:
            conn.send((True, getattr(policy, method)(*args)))
        except Exception:  # noqa: BLE001
            conn.send((False, traceback.format_exc()))


def _read_dir(path: str) -> dict[str, bytes]:
    out = {}
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        if os.path.isfile(fp) and not name.endswith((".jsonl", "job.json")):
            with open(fp, "rb") as f:
                out[name] = f.read()
    return out


class PolicyPool:
    """One policy per GPU across a Ray cluster, or a single local policy."""

    def __init__(self, pcfg: dict, lora_r: int, use_ray: bool, seed: int = 0):
        self.pcfg, self.lora_r = pcfg, lora_r
        self.actors = []
        self.local: Policy | None = None
        self.seed = seed
        self.adapter: tuple[str | None, int] = (None, 0)
        n_gpus = 0
        if use_ray and pcfg["backend"] != "mock":
            import ray

            tp = pcfg.get("tensor_parallel", 1)
            n_gpus = int(ray.cluster_resources().get("GPU", 0))
            n_actors = pcfg.get("num_actors") or max(1, n_gpus // tp)
            if n_gpus:
                Actor = ray.remote(num_gpus=tp)(_PolicyActor)
                self.actors = [Actor.remote(pcfg, lora_r, seed + i) for i in range(n_actors)]
        if not self.actors:
            # real models live in their own process; the test-only mock stays in-process
            self.local = MockPolicy(seed) if pcfg["backend"] == "mock" else ProcessPolicy(pcfg, lora_r, seed)

    def generate(self, prompts, n, temperature, max_new_tokens):
        if self.local is not None:
            return self.local.generate(prompts, n, temperature, max_new_tokens)
        import ray

        shards = [prompts[i::len(self.actors)] for i in range(len(self.actors))]
        outs = ray.get([a.generate.remote(s, n, temperature, max_new_tokens)
                        for a, s in zip(self.actors, shards)])
        merged: list = [None] * len(prompts)
        for i, out in enumerate(outs):
            for j, o in enumerate(out):
                merged[i + j * len(self.actors)] = o
        return merged

    def set_adapter(self, path: str | None, version: int):
        self.adapter = (path, version)
        if self.local is not None:
            self.local.set_adapter(path, version)
            return
        import ray

        files = _read_dir(path) if path else None
        ray.get([a.set_adapter.remote(files, version) for a in self.actors])

    def unload(self):
        """Release every GPU this pool holds (Ray actors are terminated)."""
        if self.local is not None:
            self.local.unload()
            self.local = None
        if self.actors:
            import ray

            for a in self.actors:
                ray.kill(a)
            self.actors = []


class _PolicyActor:
    def __init__(self, pcfg, lora_r, seed):
        self.pcfg, self.lora_r, self.seed = pcfg, lora_r, seed
        self.policy = make_policy(pcfg, lora_r, seed)
        self.dir = tempfile.mkdtemp(prefix="mf_adapter_")
        self.adapter = (None, 0)

    def generate(self, prompts, n, temperature, max_new_tokens):
        return self.policy.generate(prompts, n, temperature, max_new_tokens)

    def set_adapter(self, files, version):
        path = None
        if files:
            path = os.path.join(self.dir, f"v{version}")
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path)
            for name, data in files.items():
                with open(os.path.join(path, name), "wb") as f:
                    f.write(data)
        self.adapter = (path, version)
        self.policy.set_adapter(path, version)
