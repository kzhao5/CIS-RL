#!/bin/bash
# CIS training of one (MODEL, SEED) run with the patched AReaL on a single 8-GPU
# node, followed by the five-benchmark evaluation of the final checkpoint.
#
#   MODEL=moe  SEED=1 bash scripts/train.sh
#   MODEL=dsv2 SEED=1 bash scripts/train.sh
#   MODEL=q30b SEED=1 bash scripts/train.sh
#
# MODEL : moe  (Qwen/Qwen1.5-MoE-A2.7B-Chat, SGLang rollout)
#         dsv2 (deepseek-ai/DeepSeek-V2-Lite-Chat, vLLM rollout)
#         q30b (Qwen/Qwen3-30B-A3B, vLLM rollout with tensor parallelism 2)
# Optional: LAMBDA (default 2.3), KAPPA (default 5e-3), TAG (trial-name suffix),
#           EXTRA (additional Hydra overrides), RUN_EVAL=0 (skip the evaluation).
# Slurm users can submit this file directly, e.g.
#   sbatch --nodes=1 --gres=gpu:8 --export=ALL,MODEL=moe,SEED=1 scripts/train.sh

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
: "${MODEL:?set MODEL=moe|dsv2|q30b}" "${SEED:?set SEED}"
LAMBDA=${LAMBDA:-2.3}
KAPPA=${KAPPA:-5e-3}
TAG=${TAG:-}

case $MODEL in
  moe)  MODELPATH=Qwen/Qwen1.5-MoE-A2.7B-Chat
        MOVR="" ;;
  dsv2) MODELPATH=deepseek-ai/DeepSeek-V2-Lite-Chat
        MOVR="rollout.backend=vllm:d4p1t1 actor.weight_update_mode=disk ++actor.optimizer_dtype=bfloat16" ;;
  q30b) MODELPATH=Qwen/Qwen3-30B-A3B
        MOVR="rollout.backend=vllm:d2p1t2 actor.weight_update_mode=disk ++actor.optimizer_dtype=bfloat16 actor.mb_spec.max_tokens_per_mb=2048 ref.mb_spec.max_tokens_per_mb=2048" ;;
  *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac

TRIAL=cis${TAG}-s${SEED}
cd "$AREAL_DIR" || { echo "AREAL_DIR=$AREAL_DIR not found (run areal_patch/apply_patch.sh)"; exit 1; }
$AREAL_PYTHON examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_cis.yaml \
    experiment_name=cis-${MODEL} \
    trial_name=${TRIAL} \
    seed=$SEED \
    scheduler.type=local \
    cluster.fileroot="$RUN_ROOT/experiments" \
    cluster.name_resolve.nfs_record_root="$RUN_ROOT/name_resolve" \
    actor.path=$MODELPATH \
    actor.rejection_sampling.cis_lambda=$LAMBDA \
    actor.rejection_sampling.cis_kappa=$KAPPA \
    $MOVR ${EXTRA:-}

CK=$RUN_ROOT/experiments/checkpoints/$USER/cis-${MODEL}/${TRIAL}/default
E=$(ls "$CK" 2>/dev/null | grep "^epoch" | tail -1)
if [ "${RUN_EVAL:-1}" = "1" ]; then
  if [ -n "$E" ]; then
    EVAL_TOK=$MODELPATH bash "$HERE/eval.sh" "$CK/$E" "${MODEL}_cis${TAG}_s${SEED}"
  else
    echo "no epoch checkpoint under $CK; skipping evaluation"
  fi
fi
