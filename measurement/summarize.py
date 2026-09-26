"""Confidence-binned statistics of the training-inference mismatch.

Reads the per-token table written by build_dataset.py and reports, for bins of
the training-side probability p_t, the spread of the log importance ratio
log k_t and of the log-odds displacement eps_t = logit p_t - logit q_t, where
q_t is the inference-side probability of the same token. The spread of log k_t
shrinks by orders of magnitude as p_t -> 1, whereas the spread of eps_t stays
nearly constant (Section 3 and Appendix A of the paper).

    python measurement/summarize.py moe
"""

import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_ROOT

BINS = [
    (0.0, 0.5, "p < 0.5"),
    (0.5, 0.9, "0.5 - 0.9"),
    (0.9, 0.99, "0.9 - 0.99"),
    (0.99, 0.999, "0.99 - 0.999"),
    (0.999, 1.0, "p > 0.999"),
]
MIN_UNCERTAINTY = 1e-7  # tokens with 1 - p_t below this are not resolved in bf16


def logit_from_logp(logp):
    with np.errstate(divide="ignore", invalid="ignore"):
        return logp - np.log(-np.expm1(logp))


def mad(x):
    """Median absolute deviation, scaled by 1.4826 to estimate the standard deviation."""
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def main(arch):
    path = os.path.join(DATA_ROOT, "analysis", f"tokens_{arch}.parquet")
    table = pq.read_table(path, columns=["logp_train", "logp_infer"])
    lp_train = table.column("logp_train").to_numpy().astype(np.float64)
    lp_infer = table.column("logp_infer").to_numpy().astype(np.float64)

    log_k = lp_train - lp_infer
    with np.errstate(invalid="ignore"):
        eps = logit_from_logp(lp_train) - logit_from_logp(lp_infer)
    p = np.exp(lp_train)
    keep = np.isfinite(eps) & (-np.expm1(lp_train) >= MIN_UNCERTAINTY)

    header = f"{'p_t range':<14}{'tokens':>12}{'med eps':>10}{'MAD eps':>10}{'q05 eps':>10}{'q95 eps':>10}{'MAD log k':>12}"
    print(f"{arch}: {int(keep.sum()):,} tokens with 1 - p_t >= {MIN_UNCERTAINTY:g}")
    print(header)
    for lo, hi, name in BINS:
        m = keep & (p >= lo) & (p < hi) if hi < 1.0 else keep & (p >= lo)
        e = eps[m]
        print(
            f"{name:<14}{int(m.sum()):>12,}{np.median(e):>10.4f}{mad(e):>10.3f}"
            f"{np.quantile(e, 0.05):>10.3f}{np.quantile(e, 0.95):>10.3f}{mad(log_k[m]):>12.2e}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python measurement/summarize.py <arch>")
    main(sys.argv[1])
