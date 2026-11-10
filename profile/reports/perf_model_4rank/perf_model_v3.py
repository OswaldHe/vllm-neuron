"""Decode-step model v3: DeepSeek V4.1 Flash, batch 1, on a hypothetical 4-rank Trainium.

Inputs (extract_v3.py, measured on today's 8-device / 32-rank deployment, rank 0, one step):
  pieces_v3.csv       per compute piece and physical core: work union W, bytes, engine busy split by
                      scaling category and engine class, per-instruction floor, stalls by cause
  collectives_v3.csv  per collective: intrinsic latency, rank 0's wait and the mean wait over ranks

Hypothetical rank = 8 of today's ranks merged into one device: 8x engines and DMA bandwidth, no split
between two physical cores, 4 ranks in all. Attention and indexer TP4, MoE TP4 (every rank holds a quarter
of every expert: balanced). Collectives over NVLink 5.

Piece:  T = max(D, W + stall_dma_moving) + stall_core_barrier + stall_engine_barrier + stall_engine
            + stall_dma_latency + stall_other
  W scales by the busy-weighted factor of its work:
      shard  work per rank x8, resources x8          -> x1
      attn   work per rank x2 (TP8 copied 4x -> TP4) -> x1/4
      repl   same work per rank, resources x8        -> x1/8
      dma    DMA issue, follows the piece's bytes    -> x f_D
      fixed  compiler control, collective microcode  -> x1
    Tensor Engine work additionally / (new FLOPs rate / today's rate per rank) for FP4 and BF16.
    --floor keeps each instruction's fixed cost (p10 duration of its engine) instead of scaling it.
  D = bytes x f_D / DMA peak; f_D from the layout: experts split (x1), router and hc weights repeated (x1/8),
      attention heads TP4 (x1/4), shared q/kv projections and KV reads repeated (x1/8).
  Stalls: core barriers vanish (one device per rank); section barriers and engine waits follow W;
      DMA waits while data moves follow D; DMA start-up latency and other stalls stay.
Collective: NVLink latency + transfer; MoE all-reduce waits drop to the mean rank (balanced experts);
  other waits follow the compute factor of the piece before them; launch-barrier waits halve.
"""
import argparse

import pandas as pd

HOST_GAPS_US = 340.0
TOKENS_PER_STEP = 3.49
B300_STEP_US = 1e6 * TOKENS_PER_STEP / 653.6
CATS = ["shard", "attn", "repl", "dma", "fixed"]
CLS = ["fp4", "fp8", "bf16", "pe", "vec"]
S_CAT = {"shard": 1.0, "attn": 1 / 4, "repl": 1 / 8, "fixed": 1.0}            # "dma" -> f_D
F_D = {"attn": 0.5 / 4 + 0.5 / 8, "idx_topk": 0.5 / 4 + 0.5 / 8, "draft_attn": 0.5 / 4 + 0.5 / 8,
       "moe": 0.8 + 0.2 / 8, "draft_moe": 0.8 + 0.2 / 8, "head": 0.5, "other": 0.5, "launch_entry": 1.0}


def factors(sc):
    """Scaling factors for the enabled parts of the hypothesis."""
    at = 1 / 4 if sc.get("derep") else 1.0
    rp = 1 / 8 if sc.get("repl8") else 1.0
    s_cat = {"shard": 1.0, "attn": at, "repl": rp, "fixed": 1.0}
    f_d = {"attn": 0.5 * at + 0.5 * rp, "idx_topk": 0.5 * at + 0.5 * rp, "draft_attn": 0.5 * at + 0.5 * rp,
           "moe": 0.8 + 0.2 * rp, "draft_moe": 0.8 + 0.2 * rp, "head": 0.5 * at + 0.5, "other": 0.5 * rp + 0.5,
           "launch_entry": 1.0}
    return s_cat, f_d


def piece(r, sc, a):
    hyp = True
    s_cat, f_d = factors(sc)
    fD = f_d.get(r.kind, 1.0)
    k = sc.get("k", {})
    num = den = 0.0
    for cat in CATS:
        s = fD if cat == "dma" else s_cat[cat]
        tot = sum(getattr(r, f"busy_{cat}_{c}") for c in CLS)
        fl = getattr(r, f"floor_{cat}") if (a.floor and tot > 0) else 0.0
        for c in CLS:
            b = getattr(r, f"busy_{cat}_{c}")
            fixed = fl * b / tot if tot else 0.0
            num += fixed + (b - fixed) * s / k.get(c, 1.0)
            den += b
    fW = num / den if den else 1.0
    W = r.W * fW
    D = r.MB * fD / a.dma_gbps * 1e3
    st = dict(core_barrier=0.0 if sc.get("onedev") else r.stall_core_barrier,
              engine_barrier=r.stall_engine_barrier * fW, engine=r.stall_engine * fW,
              dma_moving=r.stall_dma_moving * fD, dma_latency=r.stall_dma_latency, other=r.stall_other)
    T = max(D, W + st["dma_moving"]) + sum(v for k_, v in st.items() if k_ != "dma_moving")
    return T, fW, D, W, st


def run(sc, a, pv, coll):
    rows = []
    for (e, seg), g in pv.groupby(["exec", "seg"]):
        best = max((piece(r, sc, a) + (r.kind,) for r in g.itertuples()), key=lambda x: x[0])
        T, fW, D, W, st, kind = best
        rows.append(dict(exec=e, seg=seg, kind=kind, T=T, fW=fW, D=D, W=W, **{f"s_{k}": v for k, v in st.items()}))
    pt = pd.DataFrame(rows)
    fw_of = {(r.exec, r.seg): r.fW for r in pt.itertuples()}
    cc_lat = cc_wait = 0.0
    for c in coll.itertuples():
        fw = fw_of[(c.exec, c.seg)]
        moe_ar = c.after_kind in ("moe", "draft_moe") and c.ranks == 32
        cc_lat += (a.nvl_alpha + (0 if c.op == "BARRIER" else c.bytes * 1.5 / (a.nvl_gbps_dir * 1e3))
                   if sc.get("coll") else c.us_intrinsic)
        if c.op == "BARRIER":
            cc_wait += c.wait_rank0 * (0.5 if sc.get("coll") else 1.0)
        elif moe_ar and sc.get("balance"):
            cc_wait += c.wait_rank0 - c.wait_mean          # everyone arrives at the mean
        else:
            cc_wait += c.wait_rank0 * fw                   # the imbalanced work shrinks with compute
    res = {
        "attention": pt[pt.kind == "attn"]["T"].sum(),
        "MoE": pt[pt.kind == "moe"]["T"].sum(),
        "indexer top-k": pt[pt.kind == "idx_topk"]["T"].sum(),
        "drafter + head": pt[pt.kind.isin(["draft_attn", "draft_moe", "head"])]["T"].sum(),
        "other + launch entry": pt[pt.kind.isin(["other", "launch_entry"])]["T"].sum(),
        "collectives (latency)": cc_lat,
        "collectives (wait for late ranks)": cc_wait,
        "host gaps": HOST_GAPS_US,
    }
    res["step"] = sum(res.values())
    res["-- compute chain W"] = pt.W.sum()
    res["-- DMA time at peak D"] = pt.D.sum()
    res["-- stalls: other core"] = pt.s_core_barrier.sum()
    res["-- stalls: section barriers + engine waits"] = (pt.s_engine_barrier + pt.s_engine).sum()
    res["-- stalls: DMA (moving + start-up)"] = (pt.s_dma_moving + pt.s_dma_latency).sum()
    res["-- stalls: drain / runtime / none"] = pt.s_other.sum()
    c0 = pv[pv.core == 0]
    _, f_d = factors(sc)
    byt = (c0.MB * c0.kind.map(f_d)).sum() + c0.cc_MB.sum() * f_d["attn"]
    res["DMA utilization %"] = 100 * byt / a.dma_gbps * 1e3 / res["step"] / 1e0
    return res


ALL = dict(coll=True, derep=True, repl8=True, onedev=True, balance=True)
FLOPS = {"fp4": 15 / 5.0, "bf16": 2.5 / (2 * 0.67)}
STEPS = [("today", {}),
         ("1. 4 ranks on NVLink 5 (collective latency, launch skew)", dict(coll=True)),
         ("2. + attention/indexer not copied (TP4)", dict(coll=True, derep=True)),
         ("3. + repeated work on 8x resources", dict(coll=True, derep=True, repl8=True)),
         ("4. + one device per rank (no core-to-core sync)", dict(coll=True, derep=True, repl8=True, onedev=True)),
         ("5. + balanced MoE (TP4 experts)", ALL),
         ("6. + FP4 15 PF, BF16 2.5 PF", dict(ALL, k=FLOPS))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=".")
    ap.add_argument("--dma-gbps", type=float, default=560.0)
    ap.add_argument("--nvl-alpha", type=float, default=5.0)
    ap.add_argument("--nvl-gbps-dir", type=float, default=900.0)
    a = ap.parse_args()
    pv = pd.read_csv(f"{a.inputs}/pieces_v3.csv")
    coll = pd.read_csv(f"{a.inputs}/collectives_v3.csv")
    pd.set_option("display.width", 250)
    for floor in (False, True):
        a.floor = floor
        print(f"\n=== {'with per-instruction floor' if floor else 'ideal scaling'}  (NVLink latency {a.nvl_alpha} us)")
        prev = None
        for name, sc in STEPS:
            r = run(sc, a, pv, coll)
            d = "" if prev is None else f"{(r['step'] - prev) / 1e3:+6.2f}"
            print(f"  {name:58s} {r['step'] / 1e3:6.2f} ms {d:>7s}   {1e6 * TOKENS_PER_STEP / r['step']:4.0f} tok/s"
                  f"   DMA util {r['DMA utilization %']:3.0f}%")
            prev = r["step"]
        full = pd.DataFrame({"today": run({}, a, pv, coll), "full hypothesis": run(dict(ALL, k=FLOPS), a, pv, coll)})
        if not floor:
            full_ideal = full
    both = pd.concat([full_ideal["today"], full_ideal["full hypothesis"], full["full hypothesis"]], axis=1,
                     keys=["today", "full, ideal", "full, instr. floor"])
    print("\nbreakdown (ms):")
    print((both.drop(index="DMA utilization %") / 1e3).round(2).to_string())
    print(f"(B300 653.6 tok/s = {B300_STEP_US / 1e3:.2f} ms)")


if __name__ == "__main__":
    main()
