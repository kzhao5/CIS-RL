"""Shared constants and routing math for the static mismatch measurement.

  D = logp_train - logp_infer   (per-token log importance ratio, log k_t)
  flip = 1{any MoE layer's top-k expert set differs between infer and train}
  margin = per-layer gate prob difference between k-th and (k+1)-th expert
"""

import os

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# All intermediate artefacts (prompts, generations, recomputes, joined tables)
# are written under $CIS_WORKDIR (default ./outputs).
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.abspath(os.environ.get("CIS_WORKDIR", os.path.join(CODE_ROOT, "outputs")))
PROMPTS_JSONL = os.path.join(DATA_ROOT, "prompts_math.jsonl")

# ---------------------------------------------------------------------------
# Models (experiment matrix, architecture axis)
# ---------------------------------------------------------------------------
MODELS = {
    "moe": {
        "path": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        # qwen2_moe: every layer has a sparse MoE block (60 experts, top-4)
        # plus a shared expert; the routed top-4 is what we record.
        "num_layers": 24,
        "num_experts": 60,
        "top_k": 4,
        "norm_topk_prob": False,
        "is_moe": True,
    },
    "dsv2": {
        "path": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "num_layers": 27,
        "is_moe": False,  # routing hooks are qwen2_moe-specific; the other models record log-probs only
    },
    "q30b": {
        "path": "Qwen/Qwen3-30B-A3B",
        "num_layers": 48,
        "is_moe": False,
    },
    "dense": {
        "path": "Qwen/Qwen1.5-14B-Chat",
        "num_layers": 40,
        "is_moe": False,
    },
}

# ---------------------------------------------------------------------------
# Sampling config (mandatory: no truncation)
# temperature=1, top_p=1, top_k disabled, no logit bias -> no truncation,
# delta_hat = 0 identically; nothing to annotate on figures.
# ---------------------------------------------------------------------------
SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": 1024,
}
MAX_MODEL_LEN = 2048
GROUP_SIZE = 8  # G = 8 trajectories per prompt
GLOBAL_SEED = 20260809


def routing_from_router_logits(router_logits: torch.Tensor, top_k: int):
    """Replicate qwen2_moe routing selection from raw router logits.

    qwen2_moe computes: softmax(logits, fp32) -> topk(top_k).
    (norm_topk_prob=False for Qwen1.5-MoE-A2.7B, which only rescales the
    routing *weights*, never the selection -- selection is plain topk.)

    Args:
        router_logits: [n_tokens, n_experts] (any float dtype)
        top_k: number of routed experts

    Returns:
        ids:    [n_tokens, top_k] int64, expert ids sorted by descending prob
        margin: [n_tokens] float32, prob(k-th) - prob(k+1-th)  (selection margin)
    """
    probs = torch.softmax(router_logits.float(), dim=-1)
    top = torch.topk(probs, top_k + 1, dim=-1)
    ids = top.indices[:, :top_k]
    margin = top.values[:, top_k - 1] - top.values[:, top_k]
    return ids, margin


def pack_route_ids(ids: torch.Tensor) -> np.ndarray:
    """[n_tokens, top_k] int64 -> uint8 numpy (num_experts < 256)."""
    return ids.cpu().numpy().astype(np.uint8)


def route_flip(route_a: np.ndarray, route_b: np.ndarray) -> np.ndarray:
    """Per-token flip flag between two [n_tokens, n_layers, top_k] id arrays.

    flip = any layer whose top-k SET differs (order-insensitive: the expert
    set determines which FFNs run; ordering within top-k only reorders the
    weighted sum).
    """
    a = np.sort(route_a, axis=-1)
    b = np.sort(route_b, axis=-1)
    return (a != b).any(axis=(1, 2))
