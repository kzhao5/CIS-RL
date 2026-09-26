#!/usr/bin/env python
"""Five-benchmark evaluation of an RL checkpoint with vLLM.

Benchmarks: GSM8K (1,319), MATH-500 (500), SVAMP (300), Minerva Math (272), and
OlympiadBench (674; English, text-only, open-ended). Each problem is wrapped in
the chat template of the model with the instruction to put the final answer in
\\boxed{}, decoded greedily with at most 1,536 new tokens, and judged by
math-verify.

    python eval/eval_suite.py <ckpt_dir> <tag> [tp] [--tokenizer BASE_MODEL] [--out results.tsv]
"""

import argparse
import os
import sys

from datasets import load_dataset

INSTR = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
MAX_NEW_TOKENS = 1536


def bench_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    golds = [a.split("####")[-1].strip().replace(",", "") for a in ds["answer"]]
    return [q + INSTR for q in ds["question"]], golds


def bench_math500():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [q + INSTR for q in ds["problem"]], list(ds["answer"])


def bench_svamp():
    ds = load_dataset("ChilleD/SVAMP", split="test")
    golds = [str(int(a)) if float(a) == int(a) else str(a) for a in ds["Answer"]]
    prompts = [f"{b.strip()} {q.strip()}" + INSTR for b, q in zip(ds["Body"], ds["Question"])]
    return prompts, golds


def bench_minerva():
    ds = load_dataset("math-ai/minervamath", split="test")
    return [q + INSTR for q in ds["question"]], list(ds["answer"])


def bench_olympiad():
    ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    prompts, golds = [], []
    for r in ds:
        ctx = (r.get("context") or "").strip()
        q = (ctx + "\n" + r["question"]) if ctx else r["question"]
        prompts.append(q + INSTR)
        golds.append(r["final_answer"][0] if r["final_answer"] else "")
    return prompts, golds


BENCHES = [
    ("gsm8k", bench_gsm8k),
    ("math500", bench_math500),
    ("svamp", bench_svamp),
    ("minerva", bench_minerva),
    ("olympiad", bench_olympiad),
]


def judge(pred_text, gold):
    from math_verify import parse, verify

    try:
        gp = parse(gold if gold.lstrip().startswith(("$", "\\")) else f"${gold}$")
        if not gp:
            gp = parse(gold)
        pp = parse(pred_text)
        return bool(gp) and bool(pp) and verify(gp, pp)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_path")
    ap.add_argument("tag")
    ap.add_argument("tp", nargs="?", type=int, default=4, help="tensor-parallel size")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer path or id (default: ckpt_path); pass the base model id")
    ap.add_argument("--out", default="outputs/eval_suite.tsv", help="TSV file that results are appended to")
    args = ap.parse_args()

    data = {}
    for name, fn in BENCHES:
        data[name] = fn()
        print(f"loaded {name}: n={len(data[name][0])}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # AReaL checkpoints carry a re-serialized tokenizer that can decode byte-level
    # BPE differently under newer transformers, so the base-model tokenizer is used.
    tok_path = args.tokenizer or args.ckpt_path
    tok = AutoTokenizer.from_pretrained(tok_path)
    llm = LLM(model=args.ckpt_path, tokenizer=tok_path, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=True,
              tensor_parallel_size=args.tp)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    rows = []
    for name, _ in BENCHES:
        prompts, golds = data[name]
        chat = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
                for p in prompts]
        texts = [o.outputs[0].text for o in llm.generate(chat, sp)]
        acc = sum(judge(t, g) for t, g in zip(texts, golds)) / len(golds)
        rows.append((args.tag, name, acc, len(golds)))
        print(f"RESULT {args.tag} {name} acc={acc:.4f} n={len(golds)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        for tag, name, acc, n in rows:
            f.write(f"{tag}\t{name}\t{acc:.4f}\t{n}\n")
    sys.stdout.flush()
    os._exit(0)  # vLLM tensor-parallel workers can hang on interpreter shutdown


if __name__ == "__main__":
    main()
