"""Join all stages into compact per-token analysis tables.

Per arch, shard-wise:
  gen tokens (logp_infer, route_infer, margin_infer)
  + recompute bf16   -> D = logp_train - logp_infer, flip flags
  + recompute fp32   -> D_fp32 (precision axis / C3)
  + c1 / c2a / c2b   -> optional control deltas (merged only if present)

Outputs under DATA_ROOT/analysis/:
  tokens_{arch}.parquet  one row per generated token (compact scalars only;
                         per-layer flips packed into a uint32 bitmask)
  trajs_{arch}.parquet   one row per trajectory (T, sum_D, flip counts)

Run on a CPU node:  python measurement/build_dataset.py moe
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_ROOT, MODELS


def load_pq(path):
    return pq.read_table(path).to_pandas()


def shard_tags(arch):
    paths = sorted(
        glob.glob(os.path.join(DATA_ROOT, "gen", arch, "tokens_shard*.parquet"))
    )
    return [
        os.path.basename(p).removeprefix("tokens_").removesuffix(".parquet")
        for p in paths
    ]


def process_shard(arch, tag):
    spec = MODELS[arch]
    is_moe = spec["is_moe"]
    gen_dir = os.path.join(DATA_ROOT, "gen", arch)
    rc_dir = os.path.join(DATA_ROOT, "recompute", arch)

    tok = load_pq(os.path.join(gen_dir, f"tokens_{tag}.parquet"))
    rec = load_pq(os.path.join(rc_dir, f"recomp_{tag}_bf16.parquet"))
    df = tok.merge(rec, on=["traj_id", "pos"], how="inner", validate="1:1")
    if len(df) != len(tok):
        raise RuntimeError(
            f"{arch}/{tag}: join lost rows ({len(df)} vs {len(tok)})"
        )

    out = pd.DataFrame({
        "traj_id": df["traj_id"].astype(np.int64),
        "prompt_id": df["prompt_id"].astype(np.int32),
        "pos": df["pos"].astype(np.int32),
        "token_id": df["token_id"].astype(np.int32),
        "logp_infer": df["logp_infer"].astype(np.float32),
        "logp_train": df["logp_train"].astype(np.float32),
    })
    out["D"] = (out["logp_train"] - out["logp_infer"]).astype(np.float32)

    if is_moe:
        L, K = spec["num_layers"], spec["top_k"]
        ri = np.stack(df["route_infer"].to_numpy()).reshape(-1, L, K)
        rt = np.stack(df["route_train"].to_numpy()).reshape(-1, L, K)
        layer_flip = (np.sort(ri, -1) != np.sort(rt, -1)).any(-1)  # [N, L]
        bit = np.zeros(len(df), dtype=np.uint32)
        for l in range(L):
            bit |= layer_flip[:, l].astype(np.uint32) << l
        mi = np.stack(df["margin_infer"].to_numpy()).astype(np.float32)
        mt = np.stack(df["margin_train"].to_numpy()).astype(np.float32)
        out["flip"] = layer_flip.any(-1)
        out["n_flip_layers"] = layer_flip.sum(-1).astype(np.uint8)
        out["flip_bits"] = bit
        out["margin_min_infer"] = mi.min(1)
        out["margin_min_train"] = mt.min(1)
        # margin at the flipped layer(s): min margin among flipped layers
        mmf = np.where(layer_flip, mi, np.inf).min(1)
        out["margin_at_flip"] = np.where(
            np.isfinite(mmf), mmf, np.nan
        ).astype(np.float32)
        # bf16 gate probs tie EXACTLY (margin == 0) on several % of
        # (token, layer) pairs, so raw any-layer flips saturate
        # (eps ~ 0.92). A "hard flip" is a layer whose top-k set differs
        # while the inference-side selection margin is clearly positive --
        # a genuinely different routing decision, not an arbitrary
        # tie-break. Threshold 1e-3 in gate-prob units.
        HARD = 1e-3
        out["margin_flip_max"] = np.where(
            layer_flip, mi, 0.0
        ).max(1).astype(np.float32)
        out["n_hard_flip_layers"] = (
            (layer_flip & (mi > HARD)).sum(-1).astype(np.uint8)
        )
        out["hard_flip"] = out["n_hard_flip_layers"] > 0
        out["n_tie_layers"] = (mi <= 1e-6).sum(-1).astype(np.uint8)
        hard_bit = np.zeros(len(df), dtype=np.uint32)
        hard_lf = layer_flip & (mi > HARD)
        for l in range(L):
            hard_bit |= hard_lf[:, l].astype(np.uint32) << l
        out["hard_flip_bits"] = hard_bit

    # fp32 recompute (precision axis / C3)
    p32 = os.path.join(rc_dir, f"recomp_{tag}_fp32.parquet")
    if os.path.exists(p32):
        r32 = load_pq(p32).rename(columns={"logp_train": "logp_train_fp32"})
        cols = ["traj_id", "pos", "logp_train_fp32"]
        out = out.merge(r32[cols], on=["traj_id", "pos"], how="left")

    # C1: same-engine prefill rescore (5%)
    c1p = os.path.join(DATA_ROOT, "c1", arch, f"c1_{tag}.parquet")
    if os.path.exists(c1p):
        c1 = load_pq(c1p)
        out = out.merge(c1, on=["traj_id", "pos"], how="left")

    # C2: determinism floor pair (5%)
    c2a = os.path.join(rc_dir, f"recomp_{tag}_bf16_c2a.parquet")
    c2b = os.path.join(rc_dir, f"recomp_{tag}_bf16_c2b.parquet")
    if os.path.exists(c2a) and os.path.exists(c2b):
        a = load_pq(c2a)[["traj_id", "pos", "logp_train"]].rename(
            columns={"logp_train": "logp_c2a"}
        )
        b = load_pq(c2b)[["traj_id", "pos", "logp_train"]].rename(
            columns={"logp_train": "logp_c2b"}
        )
        out = out.merge(a, on=["traj_id", "pos"], how="left")
        out = out.merge(b, on=["traj_id", "pos"], how="left")

    return out


def main():
    arch = sys.argv[1]
    tags = shard_tags(arch)
    if not tags:
        raise SystemExit(f"no gen shards found for {arch}")
    print(f"[{arch}] shards: {tags}")

    outdir = os.path.join(DATA_ROOT, "analysis")
    os.makedirs(outdir, exist_ok=True)

    parts = []
    for tag in tags:
        part = process_shard(arch, tag)
        parts.append(part)
        print(f"  {tag}: {len(part)} tokens")
    full = pd.concat(parts, ignore_index=True)

    # trajectory-level aggregation
    g = full.groupby("traj_id", sort=False)
    trajs = g.agg(
        prompt_id=("prompt_id", "first"),
        T=("pos", "size"),
        sum_D=("D", "sum"),
        mean_logp_infer=("logp_infer", "mean"),
    ).reset_index()
    if "flip" in full.columns:
        trajs = trajs.merge(
            g.agg(
                any_flip=("flip", "any"),
                n_flips=("flip", "sum"),
                any_hard_flip=("hard_flip", "any"),
                n_hard_flips=("hard_flip", "sum"),
            ).reset_index(),
            on="traj_id",
        )

    pq.write_table(
        pa.Table.from_pandas(full, preserve_index=False),
        os.path.join(outdir, f"tokens_{arch}.parquet"),
    )
    pq.write_table(
        pa.Table.from_pandas(trajs, preserve_index=False),
        os.path.join(outdir, f"trajs_{arch}.parquet"),
    )
    print(f"[{arch}] wrote {len(full)} tokens, {len(trajs)} trajs -> {outdir}")


if __name__ == "__main__":
    main()
