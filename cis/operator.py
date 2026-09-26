"""Calibrated importance sampling (CIS), Algorithm 1 of the paper.

Rollouts are sampled by an inference engine and gradients are computed by a
training engine. For every sampled token y_t with prefix h_t,

    q_t = pi_infer(y_t | h_t)    probability recorded by the inference engine
    p_t = pi_train(y_t | h_t)    probability recomputed by the training engine
    k_t = p_t / q_t              training-inference importance ratio

CIS replaces k_t by the truncated weight

    f_t   = min(k_t, 1 + lambda * phi_t),
    phi_t = max(1 - p_t, kappa).

The cap truncates the log-odds displacement eps_t = logit p_t - logit q_t at a
single constant threshold, which maps back to a ratio cap that tightens as the
token becomes more confident. The floor kappa keeps the cap of confident tokens
above the storage resolution of the log-probabilities. Only the upper side is
truncated, and the weight carries no gradient.
"""

from __future__ import annotations

import torch

__all__ = ["cis_weight", "cis_policy_loss"]


@torch.no_grad()
def cis_weight(
    logp_train: torch.Tensor,
    logp_infer: torch.Tensor,
    lam: float = 2.3,
    kappa: float = 5e-3,
) -> torch.Tensor:
    """CIS weight f_t = min(k_t, 1 + lam * max(1 - p_t, kappa)).

    Args:
        logp_train: log p_t, training-side log-probability of each sampled token.
        logp_infer: log q_t, inference-side log-probability recorded at sampling
            time, with no temperature rescaling or renormalization afterwards.
        lam: threshold lambda.
        kappa: floor on the confidence term 1 - p_t.

    Returns:
        Float32 tensor with the shape of the inputs. Tokens whose log-ratio is
        not finite receive weight one.
    """
    log_k = logp_train.float() - logp_infer.float()
    log_k = torch.where(torch.isfinite(log_k), log_k, torch.zeros_like(log_k))
    phi = torch.clamp(1.0 - torch.exp(logp_train.float()), min=kappa)
    return torch.minimum(torch.exp(log_k), 1.0 + lam * phi)


def cis_policy_loss(
    logp: torch.Tensor,
    logp_train: torch.Tensor,
    logp_infer: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    eps_clip: float = 0.2,
    eps_clip_higher: float | None = 0.28,
    lam: float = 2.3,
    kappa: float = 5e-3,
) -> torch.Tensor:
    """Token-level decoupled PPO loss with the CIS weight.

        loss_t = f_t * max(-A_t * r_t, -A_t * clip(r_t, 1 - eps_clip, 1 + eps_clip_higher)),

    where r_t = pi_theta / pi_train is the policy ratio against the
    training-side log-probabilities of the rollout policy and f_t is the CIS
    weight of k_t. The loss is averaged over the tokens selected by loss_mask.

    Args:
        logp: log pi_theta of the sampled tokens under the current policy (with gradient).
        logp_train: training-side log-probabilities of the rollout policy.
        logp_infer: inference-side log-probabilities recorded at sampling time.
        advantages: per-token advantages A_t.
        loss_mask: 1 for response tokens that enter the loss.
    """
    mask = loss_mask.bool()
    ratio = torch.exp(logp - logp_train.detach())
    upper = 1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher)
    clipped = torch.clamp(ratio, 1.0 - eps_clip, upper)
    pg = torch.max(-advantages * ratio, -advantages * clipped)
    pg = pg * cis_weight(logp_train, logp_infer, lam, kappa)
    pg = torch.where(mask, pg, torch.zeros_like(pg))
    return pg.sum() / mask.sum().clamp(min=1)
