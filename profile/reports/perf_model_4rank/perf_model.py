"""Analytical decode-step model (overlap form): DeepSeek V4.1 Flash, batch 1, hypothetical Trainium.

Inputs, all measured on the 8-device trn3 deployment (rank 0, one decode step, executions 8-11):
  pieces_overlap.csv  per compute piece and physical core: duration T, compute-engine work union W,
                      DMA bytes, and per Tensor Engine class (fp4, fp8, bf16, other) its busy time and
                      the time it runs with no other engine working ("exclusive")
  collectives.csv     per collective: op, ranks, bytes, intrinsic latency, rank 0's wait for late ranks

Compute piece:  T = max(D, W) + S
  D  DMA-bound time: bytes / per-core DMA peak (the piece cannot be shorter than its loads)
  W  compute chain: work union of the compute engines
  S  sync / latency not covered by either = T_measured - max(D, W); held constant
  A faster Tensor Engine class c (speedup s_c) shortens W by (1 - 1/s_c) * t_c, with t_c between the
  class's exclusive time (lower bound on the gain) and its busy time (upper bound). --chain-speedup g
  divides the remaining W by g (all compute engines faster). As W shrinks the piece approaches D: the
  DMA engines stall less and their utilization rises.

Collective:  intrinsic latency(n, bytes) + skew * measured wait for late ranks.

Hypothetical: 4 ranks with today's aggregate HBM bandwidth (rank = 8 of today's ranks = 2 trn3 devices).
Attention, indexer and drafter attention TP4 (today TP8 replicated 4x): their Tensor Engine work and bytes
per rank-equivalent x1/4. MoE and LM head stay sharded over all ranks (x1). Weight prefetch that today
hides inside the 32-rank all-reduces spills into the next piece when the collective gets shorter.
"""
import argparse
import json

import pandas as pd

HOST_GAPS_US = 340.0
TOKENS_PER_STEP = 3.49
B300_STEP_US = 1e6 * TOKENS_PER_STEP / 653.6
TODAY_RANK_PF = {"fp4": 5.0, "fp8": 5.0, "bf16": 2 * 0.67}   # one hypothetical rank = 2 trn3 devices
R_DEREP = {"attn": 0.25, "idx_topk": 0.25, "draft_attn": 0.25}


def piece_times(po, sc, a):
    """Modelled duration of every piece (max over the two cores of max(D, W), plus S)."""
    out = []
    kind_of = {(r.exec, r.seg): r.kind for r in po[po.core == 0].itertuples()}
    spill = {}   # (exec, seg) -> MB per core that the previous collective can no longer hide
    for (e, seg), g in po.groupby(["exec", "seg"], sort=True):
        kind = kind_of[(e, seg)]
        T = g["T"].iloc[0]
        S = T - max(max(r.MB / a.dma_gbps * 1e3, r.W) for r in g.itertuples())
        r_kind = R_DEREP.get(kind, 1.0) if sc["four_ranks"] else 1.0
        new, d_max, w_max = 0.0, 0.0, 0.0
        for r in g.itertuples():
            D = (r.MB * r_kind + spill.get((e, seg), 0.0)) / a.dma_gbps * 1e3
            W = r.W
            for cls in ("fp4", "fp8", "bf16", "pe_other"):
                s_c = (1.0 / r_kind) * sc["k"].get(cls, 1.0)
                W -= (1 - 1 / s_c) * getattr(r, f"{a.bound}_{cls}")
            W /= sc["chain"]
            new, d_max, w_max = max(new, D, W), max(d_max, D), max(w_max, W)
        out.append(dict(exec=e, seg=seg, kind=kind, T=new + S, D=d_max, W=w_max, S=S))
        # the collective after this piece hid next-piece prefetch today; a shorter window hides less
        cc_new = sc["cc_us"].get((e, seg))
        if sc["four_ranks"] and cc_new is not None and (e, seg + 1) in kind_of:
            hid = g.cc_MB.max() * R_DEREP.get(kind_of[(e, seg + 1)], 1.0)
            spill[(e, seg + 1)] = max(0.0, hid - cc_new * a.dma_gbps / 1e3)
    return pd.DataFrame(out)


def collective_us(c, sc, a):
    if not sc["four_ranks"]:
        return c.us_intrinsic, c.wait_rank0
    if sc["link"] == "neuronlink":
        old = c.ranks if c.ranks in (8, 32) else 32
        lat = c.us_intrinsic * a.nccom["4"] / a.nccom[str(old)]
    else:
        lat = a.nvl_alpha if c.op == "BARRIER" else a.nvl_alpha + c.bytes * 1.5 / (a.nvl_gbps_dir * 1e3)
    return lat, a.skew * c.wait_rank0


def run(sc, a, po, coll):
    cc = [(c.exec, c.seg, *collective_us(c, sc, a)) for c in coll.itertuples()]
    sc["cc_us"] = {(e, s): lat + w for e, s, lat, w in cc}
    pt = piece_times(po, sc, a)
    c0 = po[po.core == 0]
    r_of = c0.kind.map(lambda k: R_DEREP.get(k, 1.0)) if sc["four_ranks"] else 1.0
    bytes_core = (c0.MB * r_of).sum() + c0.cc_MB.sum() * (0.25 if sc["four_ranks"] else 1.0)  # prefetch: attention weights
    sc["pieces"] = pt
    res = {k: pt[pt.kind.isin(v)]["T"].sum() for k, v in {
        "attention": ["attn"], "MoE": ["moe"], "indexer top-k": ["idx_topk"],
        "drafter + head": ["draft_attn", "draft_moe", "head"], "other + launch entry": ["other", "launch_entry"]}.items()}
    res["collectives (latency)"] = sum(x[2] for x in cc)
    res["collectives (wait for late ranks)"] = sum(x[3] for x in cc)
    res["host gaps"] = HOST_GAPS_US
    res["step"] = sum(res.values())
    res["DMA utilization (core 0)"] = 100 * bytes_core / a.dma_gbps * 1e3 / res["step"]
    return res


SCENARIOS = {
    "today (32 ranks, NeuronLink)": dict(four_ranks=False, link="neuronlink", k={}, chain=1.0),
    "4 ranks, NVLink 5": dict(four_ranks=True, link="nvlink", k={}, chain=1.0),
    "+ FP4 15 PF": dict(four_ranks=True, link="nvlink", k={"fp4": 15 / TODAY_RANK_PF["fp4"]}, chain=1.0),
    "+ FP4 15 PF, BF16 2.5 PF": dict(four_ranks=True, link="nvlink",
                                     k={"fp4": 15 / TODAY_RANK_PF["fp4"], "bf16": 2.5 / TODAY_RANK_PF["bf16"]}, chain=1.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=".")
    ap.add_argument("--bound", choices=["excl", "busy"], default="excl",
                    help="Tensor Engine speedups shorten the chain by exclusive (lower) or busy (upper) time")
    ap.add_argument("--dma-gbps", type=float, default=560.0, help="per physical core DMA peak")
    ap.add_argument("--skew", type=float, default=0.5)
    ap.add_argument("--nvl-alpha", type=float, default=5.0)
    ap.add_argument("--nvl-gbps-dir", type=float, default=900.0)
    ap.add_argument("--nccom", default='{"4": 10.12, "8": 13.83, "32": 14.60}')
    ap.add_argument("--chain-sweep", default="", help="comma list of extra all-engine speedups for the last scenario")
    a = ap.parse_args()
    a.nccom = json.loads(a.nccom)
    po = pd.read_csv(f"{a.inputs}/pieces_overlap.csv")
    coll = pd.read_csv(f"{a.inputs}/collectives.csv")
    df = pd.DataFrame({name: run(dict(sc), a, po, coll) for name, sc in SCENARIOS.items()})
    us = df.drop(index="DMA utilization (core 0)") / 1e3
    print(f"bound={a.bound} dma={a.dma_gbps}GB/s/core skew={a.skew} nvl_alpha={a.nvl_alpha}us")
    print(us.round(2).to_string())
    print("DMA utilization %  ", df.loc["DMA utilization (core 0)"].round(0).to_dict())
    print("tok/s per user     ", (1e6 * TOKENS_PER_STEP / df.loc["step"]).round(0).to_dict(),
          f"| B300 653.6 tok/s = {B300_STEP_US / 1e3:.2f} ms")
    if a.chain_sweep:
        base = SCENARIOS["+ FP4 15 PF, BF16 2.5 PF"]
        print("\nall compute engines g x faster on top of the last scenario:")
        for g in [float(x) for x in a.chain_sweep.split(",")]:
            r = run(dict(base, chain=g), a, po, coll)
            print(f"  g={g:<4}  step {r['step'] / 1e3:5.2f} ms  MoE {r['MoE'] / 1e3:4.2f}  attention {r['attention'] / 1e3:4.2f}"
                  f"  DMA utilization {r['DMA utilization (core 0)']:3.0f}%  tok/s {1e6 * TOKENS_PER_STEP / r['step']:4.0f}")


if __name__ == "__main__":
    main()
