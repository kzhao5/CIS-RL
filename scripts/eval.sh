#!/bin/bash
# Five-benchmark evaluation (GSM8K, MATH500, SVAMP, Minerva Math, OlympiadBench;
# greedy decoding with vLLM) of one checkpoint. Results are appended to
# $CIS_WORKDIR/eval_suite.tsv.
#
#   EVAL_TOK=Qwen/Qwen1.5-MoE-A2.7B-Chat bash scripts/eval.sh <ckpt_dir> <tag>
#
# EVAL_TOK: tokenizer of the base model (recommended, since AReaL checkpoints carry
#           a re-serialized tokenizer that can decode differently).
# EVAL_TP : tensor-parallel size (default 4).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
CKPT=${1:?usage: eval.sh <ckpt_dir> <tag>}
TAG=${2:?usage: eval.sh <ckpt_dir> <tag>}
TOK_ARGS=()
[ -n "${EVAL_TOK:-}" ] && TOK_ARGS=(--tokenizer "$EVAL_TOK")

cd "$HERE/.."
$CIS_PYTHON eval/eval_suite.py "$CKPT" "$TAG" "${EVAL_TP:-4}" "${TOK_ARGS[@]}" --out "$CIS_WORKDIR/eval_suite.tsv"
