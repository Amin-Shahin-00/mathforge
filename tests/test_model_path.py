"""Optional: the real (non-mock) model path on CPU with a tiny random Llama.

Exercises transformers generation in a separate process, LoRA training in a
subprocess, adapter hot-loading and a full self-play round with an
expert-iteration step. Needs torch, transformers, peft and accelerate; takes
about a minute. Run with:

    MF_SLOW=1 pytest tests/test_model_path.py -q
"""

import os
import random

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("MF_SLOW") != "1", reason="set MF_SLOW=1 to run")


def build_tiny_model(path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from mathforge.engines.selfplay import SOLVER_SYMPY
    from mathforge.verify.sympy_tasks import generate

    corpus = [SOLVER_SYMPY.format(text=generate(f, l, s)["text"]) for f in ("integrate", "solve", "sum")
              for l in (1, 3) for s in range(5)] + ["ANSWER: x**2*exp(x) + [1, -3/2] sqrt log sin cos"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=600, special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>",
                                   eos_token="</s>", pad_token="<pad>")
    fast.save_pretrained(path)
    cfg = LlamaConfig(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                      bos_token_id=1, eos_token_id=2, pad_token_id=3)
    LlamaForCausalLM(cfg).save_pretrained(path)


def test_selfplay_with_a_real_model_trains_and_reloads(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    import sympy as sp

    from mathforge.config import load_config
    from mathforge.distributed import Executor
    from mathforge.engines.selfplay import SOLVER_SYMPY, SelfPlayEngine
    from mathforge.store import Store
    from mathforge.verify.sympy_tasks import generate

    model_dir = tmp_path / "tiny-llama"
    build_tiny_model(model_dir)
    run_dir = tmp_path / "run"
    cfg = load_config(None, [f"run.dir={run_dir}", "compute.backend=serial",
                             "engines.selfplay.enabled=true", "engines.selfplay.policy.backend=hf",
                             f"engines.selfplay.policy.model={model_dir}",
                             "engines.selfplay.policy.max_new_tokens=12",
                             "engines.selfplay.policy.samples_per_problem=2",
                             "engines.selfplay.problems_per_round=6",
                             "engines.selfplay.train_every=1", "engines.selfplay.min_new_traces=8",
                             "engines.selfplay.eval_every=1", "engines.selfplay.eval_per_level=1",
                             "engines.selfplay.eval_levels=[1]",
                             "engines.selfplay.trainer.lora_r=8", "engines.selfplay.trainer.lora_alpha=16",
                             "engines.selfplay.trainer.batch_size=4", "engines.selfplay.trainer.grad_accum=1",
                             "engines.selfplay.trainer.max_len=256"])
    store = Store(run_dir / "mathforge.db")
    eng = SelfPlayEngine(cfg["engines"]["selfplay"], cfg, store, random.Random(0))
    # a random model solves nothing, so seed verified traces as if earlier rounds produced them
    k, n = sp.symbols("k n")
    rows = []
    for s in range(12):
        p = generate("sum", 1, s)
        ans = sp.sstr(sp.expand(sp.summation(sp.sympify(p["summand"]), (k, 1, n))))
        rows.append((SOLVER_SYMPY.format(text=p["text"]), f"ANSWER: {ans}", {"role": "solver"}))
    store.add_traces("selfplay", rows, 0)

    ex = Executor("serial")
    try:
        stats = eng.run_round(ex, 1)
        assert stats["problems"] >= 4
        assert stats.get("trained") == 1 and stats["adapter_version"] == 1
        assert (run_dir / "adapters" / "v1" / "adapter_config.json").exists()
        assert "eval_pass1" in stats
        stats2 = eng.run_round(ex, 2)          # generation now runs with adapter v1 loaded
        assert stats2["adapter_version"] == 1 and stats2["problems"] >= 4
    finally:
        eng.close()

    # A real GRPO step on top of v1: verifier-scored attempts, group-relative advantages.
    import json
    import subprocess
    import sys

    from mathforge.models.trainer import build_examples
    rows = [{"prompt": SOLVER_SYMPY.format(text=generate("sum", 1, s)["text"]),
             "completion": f"ANSWER: n**{2 + s % 2}", "advantage": 1.0 if s % 2 else -1.0} for s in range(8)]
    data = tmp_path / "grpo.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out_dir = tmp_path / "grpo_out"
    job = {"algorithm": "grpo", "model": str(model_dir), "trainer": cfg["engines"]["selfplay"]["trainer"],
           "seed": 0, "out_dir": str(out_dir), "data": str(data), "init_adapter": str(run_dir / "adapters" / "v1")}
    (tmp_path / "grpo_job.json").write_text(json.dumps(job), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "mathforge.models.trainer", str(tmp_path / "grpo_job.json")], check=True)
    assert (out_dir / "adapter_config.json").exists()
    from transformers import AutoTokenizer
    assert len(build_examples(AutoTokenizer.from_pretrained(model_dir), rows, 256, key="advantage")[0]) == 3
