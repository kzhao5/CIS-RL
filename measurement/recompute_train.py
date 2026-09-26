"""Stage B: training-engine teacher-forcing recompute.

Mirrors the AReaL FSDP training engine's numerics as closely as a
single-GPU recompute can: HF transformers model, bf16, attention
implementation flash_attention_2 (AReaL's default `actor.attn_impl`),
with sdpa as a recorded fallback when the flash-attn kernel is not
installable. Right-padded batches: both fa2 (unpad/varlen path) and
masked sdpa keep valid-token numerics independent of padding.

For each trajectory (prompt + generated tokens, exactly as sampled):
  logp_train[k]   = log_softmax(logits[P+k-1].float())[gen_ids[k]]
  route_train[k]  = top-k expert ids at position P+k-1 per MoE layer
  margin_train[k] = gate prob margin (k-th minus k+1-th) at P+k-1

Same position convention as Stage A: row t-1 predicts token t.

Modes:
  --precision bf16          main cell
  --precision fp32          precision axis / C3 (true fp32, TF32 off)
  --tag c2 --subsample 0.05 determinism floor rerun (identical batching)

Usage:
  python measurement/recompute_train.py --arch moe --shard 0 --precision bf16
"""

import argparse
import json
import os
import sys

# fp32 runs sit near the 80GB ceiling; avoid allocator fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_ROOT, MODELS, routing_from_router_logits

TOKEN_BUDGET = 8192       # max padded tokens per forward batch (bf16)
TOKEN_BUDGET_FP32 = 4096  # fp32 doubles weight+activation bytes; halve batch


def build_batches(trajs, subsample, seed, token_budget=TOKEN_BUDGET):
    """Deterministic length-sorted batches under a padded-token budget."""
    order = sorted(range(len(trajs)), key=lambda i: -trajs[i]["total_len"])
    if subsample < 1.0:
        rng = np.random.RandomState(seed)
        keep = set(
            np.nonzero(rng.rand(len(trajs)) < subsample)[0].tolist()
        )
        order = [i for i in order if i in keep]
    batches = []
    cur = []
    cur_max = 0
    for i in order:
        L = trajs[i]["total_len"]
        new_max = max(cur_max, L)
        if cur and new_max * (len(cur) + 1) > token_budget:
            batches.append(cur)
            cur, cur_max = [i], L
        else:
            cur.append(i)
            cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["moe", "dense", "dsv2", "q30b"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--tag", default=None,
                    help="output tag suffix, e.g. c2 for the rerun control")
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--attn", default="flash_attention_2",
                    help="attn impl; auto-falls back to sdpa if unavailable")
    ap.add_argument("--entropy", action="store_true",
                    help="add full-softmax entropy column h_t; skips routing "
                         "capture (use with --tag ent to keep outputs separate)")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    spec = MODELS[args.arch]
    is_moe = spec["is_moe"]
    shard_tag = "smoke" if args.smoke else f"shard{args.shard:03d}"
    gen_dir = os.path.join(DATA_ROOT, "gen", args.arch)
    trajs_pq = pq.read_table(
        os.path.join(gen_dir, f"trajs_{shard_tag}.parquet")
    ).to_pylist()
    for t in trajs_pq:
        t["total_len"] = len(t["prompt_token_ids"]) + len(t["gen_token_ids"])

    # True fp32 reference: keep TF32 off (torch default) and say so.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    attn_impl = args.attn
    try:
        model = AutoModelForCausalLM.from_pretrained(
            spec["path"], dtype=dtype, attn_implementation=attn_impl,
            device_map="cuda:0",
        )
    except (ImportError, ValueError) as e:
        print(f"[recompute] {attn_impl} unavailable ({e}); falling back to sdpa")
        attn_impl = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            spec["path"], dtype=dtype, attn_implementation=attn_impl,
            device_map="cuda:0",
        )
    model.eval()

    budget = TOKEN_BUDGET if args.precision == "bf16" else TOKEN_BUDGET_FP32
    batches = build_batches(
        trajs_pq, args.subsample, seed=1234 + args.shard, token_budget=budget
    )

    want_routing = is_moe and not args.entropy
    out = {"traj_id": [], "pos": [], "logp_train": []}
    if args.entropy:
        out["entropy"] = []
    if want_routing:
        out["route_train"] = []
        out["margin_train"] = []

    n_layers = spec.get("num_layers")
    top_k = spec.get("top_k")

    with torch.no_grad():
        for bi, batch in enumerate(batches):
            seqs = [trajs_pq[i] for i in batch]
            lens = [t["total_len"] for t in seqs]
            maxlen = max(lens)
            input_ids = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            attn_mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for r, t in enumerate(seqs):
                ids = t["prompt_token_ids"] + t["gen_token_ids"]
                input_ids[r, : len(ids)] = torch.tensor(ids)
                attn_mask[r, : len(ids)] = 1
            input_ids = input_ids.cuda()
            attn_mask = attn_mask.cuda()

            kwargs = dict(input_ids=input_ids, attention_mask=attn_mask,
                          use_cache=False)
            if want_routing:
                kwargs["output_router_logits"] = True
            res = model(**kwargs)

            # Keep logits in native dtype; upcast per-sequence row slices
            # only. log_softmax is row-wise, so slicing rows first gives
            # bitwise-identical results to a full-batch log_softmax while
            # avoiding two full [B, S, V] fp32 copies (OOM at fp32).
            logits = res.logits

            router = None
            if want_routing:
                # tuple(num_layers) of [B*S, E] (flattened) or [B, S, E]
                router = []
                for rl in res.router_logits:
                    if rl.dim() == 2:
                        rl = rl.view(len(seqs), maxlen, -1)
                    router.append(rl)

            for r, t in enumerate(seqs):
                P = len(t["prompt_token_ids"])
                T = len(t["gen_token_ids"])
                gen = torch.tensor(t["gen_token_ids"], device=logits.device)
                # rows P-1 .. P+T-2 predict tokens P .. P+T-1
                rows = torch.arange(P - 1, P + T - 1, device=logits.device)
                sub = logits[r, rows].float()  # [T, V], small transient
                ls = torch.log_softmax(sub, dim=-1)
                lp = ls.gather(1, gen.view(-1, 1)).squeeze(1)
                out["traj_id"].extend([t["traj_id"]] * T)
                out["pos"].extend(range(P, P + T))
                out["logp_train"].extend(lp.cpu().tolist())
                if args.entropy:
                    # clamp keeps 0 * -inf out of the sum for zero-prob ids
                    ent = -(ls.exp() * ls.clamp_min(-80.0)).sum(-1)
                    out["entropy"].extend(ent.cpu().tolist())

                if want_routing:
                    ids_arr = np.empty((T, n_layers, top_k), dtype=np.uint8)
                    mg_arr = np.empty((T, n_layers), dtype=np.float32)
                    for li, rl in enumerate(router):
                        ids_l, mg_l = routing_from_router_logits(
                            rl[r, rows], top_k
                        )
                        ids_arr[:, li] = ids_l.cpu().numpy().astype(np.uint8)
                        mg_arr[:, li] = mg_l.cpu().numpy()
                    for k in range(T):
                        out["route_train"].append(
                            ids_arr[k].reshape(-1).tolist()
                        )
                        out["margin_train"].append(mg_arr[k].tolist())
            if bi % 20 == 0:
                print(f"[recompute] batch {bi}/{len(batches)}", flush=True)

    fields = [
        ("traj_id", pa.int64()), ("pos", pa.int32()),
        ("logp_train", pa.float32()),
    ]
    if args.entropy:
        fields.append(("entropy", pa.float32()))
    if want_routing:
        fields += [("route_train", pa.list_(pa.uint8())),
                   ("margin_train", pa.list_(pa.float32()))]
    table = pa.table({k: out[k] for k, _ in fields}, schema=pa.schema(fields))

    meta = {
        "arch": args.arch, "precision": args.precision,
        "attn_impl": attn_impl, "torch": torch.__version__,
        "subsample": str(args.subsample),
    }
    table = table.replace_schema_metadata(
        {k: str(v) for k, v in meta.items()}
    )

    outdir = os.path.join(DATA_ROOT, "recompute", args.arch)
    os.makedirs(outdir, exist_ok=True)
    parts = [shard_tag, args.precision]
    if args.tag:
        parts.append(args.tag)
    path = os.path.join(outdir, "recomp_" + "_".join(parts) + ".parquet")
    pq.write_table(table, path)
    print(f"[recompute] wrote {path} ({table.num_rows} rows), meta={meta}")


if __name__ == "__main__":
    main()
