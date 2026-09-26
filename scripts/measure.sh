#!/bin/bash
# Static measurement of the training-inference mismatch for one model.
#
#   ARCH=moe bash scripts/measure.sh            # ARCH in measurement/common.py MODELS
#
#   1. prompts   : 2,500 DAPO-Math-17k prompts                        (measurement/prep_prompts.py)
#   2. generate  : vLLM, T=1, top_p=1, 8 samples, decode-time log-probs (measurement/gen_vllm.py)
#   3. recompute : HF transformers bf16 teacher forcing                (measurement/recompute_train.py)
#   4. join      : per-token table with log k_t = logp_train - logp_infer (measurement/build_dataset.py)
#   5. summarize : spread of log k_t and eps_t across confidence bins  (measurement/summarize.py)
#
# NGPU (default 4) GPUs process NSHARD (default 8) prompt shards in parallel.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
: "${ARCH:?set ARCH=moe|dsv2|q30b|dense}"
NGPU=${NGPU:-4}
NSHARD=${NSHARD:-8}
cd "$HERE/.."

run_sharded() {  # run "$@ --shard s" for every shard, NGPU at a time
  for ((s = 0; s < NSHARD; s++)); do
    CUDA_VISIBLE_DEVICES=$((s % NGPU)) "$@" --shard "$s" &
    (( (s + 1) % NGPU == 0 )) && wait
  done
  wait
}

[ -f "$CIS_WORKDIR/prompts_math.jsonl" ] || $CIS_PYTHON measurement/prep_prompts.py
run_sharded $CIS_PYTHON measurement/gen_vllm.py --arch "$ARCH" --num-shards "$NSHARD"
run_sharded $CIS_PYTHON measurement/recompute_train.py --arch "$ARCH" --precision bf16
$CIS_PYTHON measurement/build_dataset.py "$ARCH"
$CIS_PYTHON measurement/summarize.py "$ARCH"
