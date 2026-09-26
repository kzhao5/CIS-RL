# Shared environment for the CIS scripts. Every variable can be overridden
# from the calling shell, e.g.  AREAL_DIR=/path/to/AReaL bash scripts/train.sh
_CIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CIS_WORKDIR=${CIS_WORKDIR:-$_CIS_ROOT/outputs}          # measurement data and evaluation results
export AREAL_DIR=${AREAL_DIR:-$_CIS_ROOT/third_party/AReaL}    # AReaL checkout with areal_patch applied
export RUN_ROOT=${RUN_ROOT:-$CIS_WORKDIR/rl}                    # AReaL experiments and name_resolve records
export AREAL_PYTHON=${AREAL_PYTHON:-python}                     # interpreter of the training (AReaL) env
export CIS_PYTHON=${CIS_PYTHON:-python}                         # interpreter of the measurement / eval env
mkdir -p "$CIS_WORKDIR" "$RUN_ROOT/experiments" "$RUN_ROOT/name_resolve"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1   # single trusted node; see the AReaL docs before multi-node use
# On compute nodes without internet access, pre-download models and datasets and set:
#   export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
