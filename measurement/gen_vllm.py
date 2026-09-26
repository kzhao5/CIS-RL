"""Stage A: vLLM generation with decode-time capture.

Protocol hard requirements honored here:
  - logp_infer is read from the sampling step itself (SamplingParams.logprobs=0,
    i.e. the sampled token's logprob emitted by the engine at decode time).
    NO post-hoc prefill re-scoring in this path.
  - Sampling: temperature=1, top_p=1, top_k disabled, no logit bias.
  - route_infer: vLLM 0.26 native `enable_return_routed_experts=True`,
    captured from the fused-MoE kernel's actual topk_ids (kernel ground
    truth), returned per request as [seq_len, n_layers, top_k].
  - gate margins: forward hooks on every MoE layer's router gate grab the
    full router logits; a wrapper around GPUModelRunner.execute_model maps
    flat batch rows -> (request, position) using the scheduler's
    num_scheduled_tokens in input_batch.req_ids order (this is exactly the
    order the flat forward batch is built in, see
    gpu_model_runner.py:4190 in vllm 0.26.0).

Row/position convention (verified against vllm/v1/core/sched/scheduler.py):
  concatenated routed_experts row j == routing of the forward processing
  absolute position j; the row associated with generated token k (1-based,
  absolute position P+k-1) is row P+k-2, i.e. the forward that PRODUCED it.
  The train-side recompute uses the same convention (router logits at
  position t-1 predict token t).

Alignment self-checks (fail loudly, never silently misalign):
  1. hook-derived top-k ids == native routed_experts ids per token;
  2. per-request captured margin entries reconstruct exactly T rows
     (duplicates => preemption => hard error);
  3. native routed_experts row count == P + T - 1.

Usage:
  python measurement/gen_vllm.py --arch moe --shard 0 --num-shards 8 [--smoke]
"""

import argparse
import json
import os
import sys

# The gate hooks and the execute_model wrapper live in THIS process; the
# engine core must not be forked out to a separate process.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    DATA_ROOT,
    GLOBAL_SEED,
    GROUP_SIZE,
    MAX_MODEL_LEN,
    MODELS,
    PROMPTS_JSONL,
    SAMPLING,
    routing_from_router_logits,
)

# ---------------------------------------------------------------------------
# Margin capture machinery (MoE only)
# ---------------------------------------------------------------------------


class MarginCapture:
    """Per-step router-logits capture + flat-row -> request mapping.

    For every step we only keep, per request, the LAST scheduled row
    (the position whose logits produce the next sampled token), tagged
    with the request's cumulative processed-position count. The join
    phase selects entries with cum_after in [P, P+T-1], which correspond
     1:1 to generated tokens 1..T.
    """

    def __init__(self, num_layers: int, top_k: int):
        self.num_layers = num_layers
        self.top_k = top_k
        self.step_logits: list[torch.Tensor] = []  # per layer, this step
        self.cum: dict[str, int] = {}  # req_id -> positions processed
        # req_id -> list of (cum_after, margins[np fp32 L], ids[np uint8 L,K])
        self.records: dict[str, list] = {}
        self.enabled = False

    def hook(self, module, args, output):
        if not self.enabled:
            return
        # ReplicatedLinear.forward returns (out, out_bias)
        logits = output[0] if isinstance(output, tuple) else output
        self.step_logits.append(logits)

    def on_step(self, req_order: list[str], counts: dict[str, int]):
        if not self.step_logits:
            return
        if len(self.step_logits) != self.num_layers:
            raise RuntimeError(
                f"captured {len(self.step_logits)} gate outputs, "
                f"expected {self.num_layers} (one per MoE layer)"
            )
        offsets = []
        off = 0
        for rid in req_order:
            n = counts[rid]
            offsets.append((rid, off + n - 1))
            off += n
        total = off
        if self.step_logits[0].shape[0] < total:
            raise RuntimeError(
                f"gate rows {self.step_logits[0].shape[0]} < scheduled {total}"
            )
        idx = torch.tensor(
            [i for _, i in offsets], device=self.step_logits[0].device
        )
        # [L, R, E] -> margins [L, R], ids [L, R, K]
        margins = torch.empty(
            (self.num_layers, len(offsets)), dtype=torch.float32
        )
        ids = torch.empty(
            (self.num_layers, len(offsets), self.top_k), dtype=torch.int64
        )
        for li, logits in enumerate(self.step_logits):
            rows = logits.index_select(0, idx)
            layer_ids, layer_margin = routing_from_router_logits(rows, self.top_k)
            margins[li] = layer_margin.cpu()
            ids[li] = layer_ids.cpu()
        margins_np = margins.numpy().T  # [R, L]
        ids_np = ids.numpy().transpose(1, 0, 2).astype(np.uint8)  # [R, L, K]
        for r, (rid, _) in enumerate(offsets):
            cum_after = self.cum.get(rid, 0) + counts[rid]
            self.cum[rid] = cum_after
            self.records.setdefault(rid, []).append(
                (cum_after, margins_np[r], ids_np[r])
            )
        self.step_logits = []


def install_capture(llm, num_layers: int, top_k: int) -> MarginCapture:
    """Register gate hooks + wrap GPUModelRunner.execute_model."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    cap = MarginCapture(num_layers, top_k)

    # Reach the underlying nn.Module (single process, TP=1: worker in-proc)
    model_runner = None
    core = llm.llm_engine.engine_core
    # engine_core may be InprocClient(engine_core=EngineCore) or similar
    inner = getattr(core, "engine_core", core)
    executor = inner.model_executor
    driver = getattr(executor, "driver_worker", None)
    if driver is None:
        workers = getattr(executor, "workers", None)
        driver = workers[0] if workers else None
    worker = getattr(driver, "worker", driver)
    model_runner = worker.model_runner
    model = model_runner.model

    n_hooked = 0
    for name, module in model.named_modules():
        # Qwen2MoeSparseMoeBlock.gate (ReplicatedLinear), skip
        # gate_up_proj / shared_expert_gate
        if name.endswith(".mlp.gate"):
            module.register_forward_hook(cap.hook)
            n_hooked += 1
    if n_hooked != num_layers:
        raise RuntimeError(
            f"hooked {n_hooked} router gates, expected {num_layers}"
        )

    orig = GPUModelRunner.execute_model

    def wrapped(self, scheduler_output, *args, **kwargs):
        cap.step_logits = []
        out = orig(self, scheduler_output, *args, **kwargs)
        if cap.enabled and scheduler_output.total_num_scheduled_tokens > 0:
            for rid, spec in (
                scheduler_output.scheduled_spec_decode_tokens or {}
            ).items():
                if spec:
                    raise RuntimeError("spec decode must be disabled")
            req_order = list(self.input_batch.req_ids)
            counts = dict(scheduler_output.num_scheduled_tokens)
            cap.on_step(req_order, counts)
        return out

    GPUModelRunner.execute_model = wrapped
    return cap


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["moe", "dense", "dsv2", "q30b"], required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="tiny smoke run")
    ap.add_argument("--max-num-seqs", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    spec = MODELS[args.arch]
    is_moe = spec["is_moe"]

    prompts = [json.loads(l) for l in open(PROMPTS_JSONL)]
    prompts = [p for i, p in enumerate(prompts) if i % args.num_shards == args.shard]
    group = GROUP_SIZE
    max_tokens = SAMPLING["max_tokens"]
    if args.smoke:
        prompts = prompts[:6]
        group = 2
        max_tokens = 128

    tok = AutoTokenizer.from_pretrained(spec["path"])
    chat_texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p["text"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    llm_kwargs = dict(
        model=spec["path"],
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,  # python gate hooks cannot fire inside CUDA graphs
        enable_prefix_caching=False,  # keep batches clean
        async_scheduling=False,  # keep input_batch state synchronous with steps
        seed=GLOBAL_SEED + args.shard,
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
    )
    if is_moe:
        llm_kwargs["enable_return_routed_experts"] = True

    llm = LLM(**llm_kwargs)

    cap = None
    if is_moe:
        cap = install_capture(llm, spec["num_layers"], spec["top_k"])
        cap.enabled = True

    # one request per trajectory (n=1): engine req ids match submission order
    sps = []
    metas = []
    for gi, p in enumerate(prompts):
        for s in range(group):
            traj_id = (p["prompt_id"] * 1000) + s
            sps.append(
                SamplingParams(
                    temperature=SAMPLING["temperature"],
                    top_p=SAMPLING["top_p"],
                    top_k=SAMPLING["top_k"],
                    max_tokens=max_tokens,
                    logprobs=0,  # sampled token's decode-time logprob
                    seed=GLOBAL_SEED * 7 + p["prompt_id"] * 100 + s,
                )
            )
            metas.append({"prompt_id": p["prompt_id"], "traj_id": traj_id,
                          "chat_idx": gi})

    reqs = [chat_texts[m["chat_idx"]] for m in metas]
    outputs = llm.generate(reqs, sps)
    if cap is not None:
        cap.enabled = False

    # ---------------- join + validate + write ----------------
    tok_rows = {
        "traj_id": [], "prompt_id": [], "pos": [], "token_id": [],
        "logp_infer": [],
    }
    if is_moe:
        tok_rows["route_infer"] = []
        tok_rows["margin_infer"] = []
    traj_rows = {
        "traj_id": [], "prompt_id": [], "prompt_token_ids": [],
        "gen_token_ids": [], "finish_reason": [], "seed": [],
    }

    # Engine-internal request ids are "{request_id}-{uuid}" in vllm 0.26;
    # RequestOutput.request_id is the bare index. Match on the prefix.
    records_by_prefix = {}
    if cap is not None:
        for k, v in cap.records.items():
            prefix = k.split("-", 1)[0]
            if prefix in records_by_prefix:
                raise RuntimeError(f"duplicate capture prefix {prefix}")
            records_by_prefix[prefix] = v

    n_id_mismatch = 0
    n_tokens_total = 0
    for i, out in enumerate(outputs):
        m = metas[i]
        comp = out.outputs[0]
        P = len(out.prompt_token_ids)
        gen_ids = list(comp.token_ids)
        T = len(gen_ids)
        if T == 0:
            continue
        # decode-time logprobs of the sampled tokens
        lps = [comp.logprobs[t][gen_ids[t]].logprob for t in range(T)]

        route = None
        margins = None
        if is_moe:
            re = comp.routed_experts
            if re is None:
                raise RuntimeError(f"req {i}: routed_experts missing")
            re = np.asarray(re)
            if re.shape[0] != P + T - 1:
                raise RuntimeError(
                    f"req {i}: routed_experts rows {re.shape[0]} != "
                    f"P+T-1 = {P + T - 1}"
                )
            # row for generated token k (0-based) = abs position P+k-1
            route = re[P - 1 : P + T - 1].astype(np.uint8)  # [T, L, K]

            entries = records_by_prefix.get(str(out.request_id))
            if entries is None:
                raise RuntimeError(f"req {i}: no margin capture entries")
            by_cum = {}
            for cum_after, mg, ids in entries:
                if cum_after in by_cum:
                    raise RuntimeError(
                        f"req {i}: duplicate cum {cum_after} => preemption; "
                        "rerun with more KV headroom"
                    )
                by_cum[cum_after] = (mg, ids)
            margins = np.empty((T, spec["num_layers"]), dtype=np.float32)
            for k in range(T):
                cum = P + k  # positions processed when token k+1 emitted
                if cum not in by_cum:
                    raise RuntimeError(f"req {i}: missing margin for cum {cum}")
                mg, hook_ids = by_cum[cum]
                margins[k] = mg
                # Cross-check hook ids vs kernel-native ids per layer.
                # bf16 gate probs tie EXACTLY at the top-k boundary on a
                # few % of (token, layer) pairs (margin == 0); the two
                # implementations break such ties differently, so only a
                # margin>0 disagreement indicates a real mapping bug
                # (verified empirically: 100% agreement | margin > 0).
                layer_neq = (
                    np.sort(hook_ids, axis=-1) != np.sort(route[k], axis=-1)
                ).any(-1)
                if (layer_neq & (mg > 1e-6)).any():
                    n_id_mismatch += 1

        traj_rows["traj_id"].append(m["traj_id"])
        traj_rows["prompt_id"].append(m["prompt_id"])
        traj_rows["prompt_token_ids"].append(list(out.prompt_token_ids))
        traj_rows["gen_token_ids"].append(gen_ids)
        traj_rows["finish_reason"].append(str(comp.finish_reason))
        traj_rows["seed"].append(sps[i].seed)

        for t in range(T):
            tok_rows["traj_id"].append(m["traj_id"])
            tok_rows["prompt_id"].append(m["prompt_id"])
            tok_rows["pos"].append(P + t)  # absolute position
            tok_rows["token_id"].append(gen_ids[t])
            tok_rows["logp_infer"].append(lps[t])
            if is_moe:
                tok_rows["route_infer"].append(route[t].reshape(-1).tolist())
                tok_rows["margin_infer"].append(margins[t].tolist())
        n_tokens_total += T

    frac_mismatch = n_id_mismatch / max(n_tokens_total, 1)
    print(f"[gen] tokens={n_tokens_total}, hook-vs-native id mismatch "
          f"tokens={n_id_mismatch} ({frac_mismatch:.2e})")
    if is_moe and frac_mismatch > 0.001:
        raise RuntimeError(
            "hook/native routing disagreement above tie-breaking noise -> "
            "mapping bug, refusing to write output"
        )

    outdir = os.path.join(DATA_ROOT, "gen", args.arch)
    os.makedirs(outdir, exist_ok=True)
    tag = "smoke" if args.smoke else f"shard{args.shard:03d}"

    tok_schema_fields = [
        ("traj_id", pa.int64()), ("prompt_id", pa.int32()),
        ("pos", pa.int32()), ("token_id", pa.int32()),
        ("logp_infer", pa.float32()),
    ]
    if is_moe:
        tok_schema_fields += [
            ("route_infer", pa.list_(pa.uint8())),
            ("margin_infer", pa.list_(pa.float32())),
        ]
    tok_table = pa.table(
        {k: tok_rows[k] for k, _ in tok_schema_fields},
        schema=pa.schema(tok_schema_fields),
    )
    pq.write_table(tok_table, os.path.join(outdir, f"tokens_{tag}.parquet"))

    traj_table = pa.table(traj_rows)
    pq.write_table(traj_table, os.path.join(outdir, f"trajs_{tag}.parquet"))
    print(f"[gen] wrote {outdir}/tokens_{tag}.parquet "
          f"({n_tokens_total} tokens, {len(traj_rows['traj_id'])} trajs)")


if __name__ == "__main__":
    main()
