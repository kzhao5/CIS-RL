"""Tests for the CIS operator.

    python -m pytest tests/ -q

When the patched AReaL is importable, the reference operator is also checked
against the `action: cis` branch of AReaL's apply_rejection_sampling, which is
the code path used for training.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cis import cis_policy_loss, cis_weight  # noqa: E402

LAM, KAPPA = 2.3, 5e-3


def _random_logps(n=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    logp_train = -torch.rand(n, generator=g) ** 4 * 8.0  # confident and uncertain tokens
    noise = torch.randn(n, generator=g) * 0.3
    logp_infer = torch.clamp(logp_train - noise, max=0.0)
    return logp_train, logp_infer


def test_identity_below_cap():
    lp = torch.tensor([-1.0, -0.5, -2.0])
    li = lp + 1e-3  # k_t < 1
    assert torch.allclose(cis_weight(lp, li), torch.exp(lp - li))


def test_cap_tightens_with_confidence():
    lp = torch.log(torch.tensor([0.2, 0.9, 0.999]))
    li = torch.full_like(lp, -20.0)  # very large k_t
    expected = 1.0 + LAM * torch.clamp(1.0 - lp.exp(), min=KAPPA)
    out = cis_weight(lp, li, LAM, KAPPA)
    assert torch.allclose(out, expected)
    assert out[0] > out[1] > out[2]


def test_lower_side_untouched():
    lp = torch.tensor([-3.0])
    li = torch.tensor([-0.01])  # k_t close to zero
    assert torch.allclose(cis_weight(lp, li), torch.exp(lp - li))


def test_floor_for_confident_tokens():
    lp = torch.tensor([0.0])  # p_t = 1
    li = torch.tensor([-0.5])
    assert torch.allclose(cis_weight(lp, li, LAM, KAPPA), torch.tensor([1.0 + LAM * KAPPA]))


def test_lambda_zero_is_tis_with_cap_one():
    lp, li = _random_logps()
    assert torch.allclose(cis_weight(lp, li, lam=0.0), torch.clamp(torch.exp(lp - li), max=1.0))


def test_non_finite_log_ratio_gets_unit_weight():
    lp = torch.tensor([-math.inf])
    li = torch.tensor([-math.inf])
    assert torch.equal(cis_weight(lp, li), torch.tensor([1.0]))


def test_weight_has_no_gradient():
    lp, li = _random_logps(64)
    lp = lp.clone().requires_grad_(True)
    assert not cis_weight(lp, li).requires_grad


def test_loss_runs_and_backprops():
    lp, li = _random_logps(64)
    logp = (lp + 0.01).clone().requires_grad_(True)
    loss = cis_policy_loss(logp, lp, li, torch.randn(64), torch.ones(64, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logp.grad).all()


def test_matches_patched_areal():
    fn = pytest.importorskip("areal.utils.functional.functional")
    from areal.api.cli_args import RejectionSamplingConfig

    if "cis_lambda" not in RejectionSamplingConfig.__dataclass_fields__:
        pytest.skip("the installed AReaL does not include areal_patch/cis_areal.patch")
    lp, li = _random_logps()
    mask = torch.ones_like(lp, dtype=torch.bool)
    mask[:100] = False
    cfg = RejectionSamplingConfig(
        level="token", action="cis", metric="ratio", cis_lambda=LAM, cis_kappa=KAPPA
    )
    res = fn.apply_rejection_sampling(
        proximal_logprobs=lp, old_logprobs=li, loss_mask=mask, cu_seqlens=None, config=cfg
    )
    ref = torch.where(mask, cis_weight(lp, li, LAM, KAPPA), torch.zeros_like(lp))
    assert torch.allclose(res.behave_imp_weight.float(), ref, rtol=1e-6, atol=1e-7)
    assert torch.equal(res.loss_mask, mask)
