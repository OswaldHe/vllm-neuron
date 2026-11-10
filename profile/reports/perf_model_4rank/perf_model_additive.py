"""Analytical decode-step model: DeepSeek V4.1 Flash, batch 1, on a hypothetical Trainium.

Inputs (from extract_inputs.py, measured on the 8-device trn3 deployment, rank 0, one decode step):
  pieces.csv       every compute piece between two collectives: duration, DMA bytes per core, engine busy
  collectives.csv  every collective: op, ranks, bytes, intrinsic latency, rank 0's wait for later ranks

Model of a compute piece:   T = L + r * X
  X  throughput part: the busiest resource that scales with the work a rank holds
       weights mode: max(bytes / per-core DMA peak, PE busy)       (weights stream through DMA and PE)
       engines mode: max(bytes / per-core DMA peak, PE, DVE, ACT, POOL busy)
  L  latency part = measured T - X (dependency chain, small ops, semaphores, intra-rank sync); unchanged
  r  (work per new rank / work per old rank) / (resources per new rank / resources per old rank)

Model of a collective:      C = intrinsic(n, bytes) + s * wait_measured
  intrinsic  latency once the last rank arrives
  wait       rank 0's measured wait for later ranks (skew), scaled by s

Hypothetical device: 4 ranks with the aggregate compute and HBM bandwidth of today's 32 ranks
(each rank = 8 of today's ranks = 2 trn3 devices: 5 PFLOPS FP8, 9.8 TB/s HBM). Parallelism:
attention, indexer and drafter attention TP4 (today: TP8, replicated 4x), MoE TP4 (today TP8 x EP4).
"""
import argparse
import json
import math

import pandas as pd

HOST_GAPS_US = 340.0          # host time between the 4 launches (system trace)
TOKENS_PER_STEP = 3.49        # 288.3 tok/s at a 12.10 ms step
B300_STEP_US = 1e6 * TOKENS_PER_STEP / 653.6
DMA_PEAK_GBPS_CORE = 560.0    # per physical core, measured peak in 10 us windows

# r per piece kind for the 4-rank layout (rank resources x8)
R_4RANK = {
    "attn": 8 / 4 / 8,        # TP8 x4 replicas -> TP4: work per rank x2, resources x8
    "idx_topk": 8 / 4 / 8,    # replicated with the attention group
    "draft_attn": 8 / 4 / 8,
    "moe": 1.0,               # sharded over all ranks in both layouts
    "draft_moe": 1.0,
    "head": 1.0,              # vocab-sharded over all ranks in both layouts
    "other": 1.0,             # small latency-bound ops (hc, compressor, readout): kept as measured
    "launch_entry": 1.0,
}


def throughput_part(p, mode):
    mb = max(p.MB_core0, p.MB_core1)
    terms = [mb / DMA_PEAK_GBPS_CORE * 1e3, p.pe]   # MB / (GB/s) * 1e3 = us
    if mode == "engines":
        terms += [p.dve, p.act, p.pool]
    if p.kind == "launch_entry":
        return 0.0
    return min(max(terms), p.compute_us)


def collective_intrinsic(c, scenario, a):
    if scenario == "today":
        return c.us_intrinsic
    n_new = 4
    if scenario == "neuronlink4":
        # scale today's in-kernel latency by the standalone ratio t(4) / t(n_old) (nccom-test, 123 KB)
        ratio = a.nccom[str(n_new)] / a.nccom[str(c.ranks if c.ranks in (8, 32) else 32)]
        return c.us_intrinsic * ratio
    if scenario == "nvlink4":
        if c.op == "BARRIER":
            return a.nvl_alpha
        bw_us = (c.bytes * 2 * (n_new - 1) / n_new) / (a.nvl_gbps_dir * 1e3)   # ring volume per rank
        return a.nvl_alpha + bw_us
    raise ValueError(scenario)


def run(scenario, mode, skew, a, pieces, coll):
    r = R_4RANK if scenario != "today" else {k: 1.0 for k in R_4RANK}
    rows = []
    for p in pieces.itertuples():
        x = throughput_part(p, mode)
        lat = p.compute_us - x
        rows.append(dict(part=p.kind, L=lat, X=x * r[p.kind], us=lat + x * r[p.kind]))
    comp = pd.DataFrame(rows)
    s = 1.0 if scenario == "today" else skew
    cc = pd.DataFrame([dict(part=f"cc {c.op} n={c.ranks}", intrinsic=collective_intrinsic(c, scenario, a),
                            wait=s * c.wait_rank0) for c in coll.itertuples()])
    out = {
        "attention": comp[comp.part == "attn"].us.sum(),
        "MoE": comp[comp.part == "moe"].us.sum(),
        "indexer top-k": comp[comp.part == "idx_topk"].us.sum(),
        "drafter + head": comp[comp.part.isin(["draft_attn", "draft_moe", "head"])].us.sum(),
        "other + launch entry": comp[comp.part.isin(["other", "launch_entry"])].us.sum(),
        "collectives (latency)": cc.intrinsic.sum(),
        "collectives (wait for late ranks)": cc.wait.sum(),
        "host gaps": HOST_GAPS_US,
    }
    out["step"] = sum(out.values())
    out["of which latency part L"] = comp.L.sum()
    out["of which throughput part r*X"] = comp.X.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=".")
    ap.add_argument("--mode", choices=["weights", "engines"], default="weights")
    ap.add_argument("--skew", type=float, default=0.5, help="fraction of today's rank skew left with 4 ranks")
    ap.add_argument("--nvl-alpha", type=float, default=5.0, help="NVLink 5 small all-reduce latency, 4 ranks (us)")
    ap.add_argument("--nvl-gbps-dir", type=float, default=900.0, help="NVLink 5 per rank per direction (GB/s)")
    ap.add_argument("--nccom", default='{"4": 1.0, "8": 1.0, "32": 1.0}',
                    help="standalone all-reduce latency (us) at 123 KB by rank count, from nccom-test")
    a = ap.parse_args()
    a.nccom = json.loads(a.nccom)
    pieces = pd.read_csv(f"{a.inputs}/pieces.csv")
    coll = pd.read_csv(f"{a.inputs}/collectives.csv")
    cols = {}
    for sc, label in (("today", "today: 32 ranks, NeuronLink"), ("neuronlink4", "4 ranks, NeuronLink"),
                      ("nvlink4", "4 ranks, NVLink 5")):
        cols[label] = run(sc, a.mode, a.skew, a, pieces, coll)
    df = pd.DataFrame(cols)
    print(f"mode={a.mode} skew={a.skew} nvl_alpha={a.nvl_alpha}us nccom={a.nccom}")
    print((df / 1e3).round(2).to_string(float_format=lambda v: f"{v:.2f}"))
    step = df.loc["step"]
    print("\nper-user tok/s:", (1e6 * TOKENS_PER_STEP / step).round(0).to_dict(),
          f"| B300 653.6 tok/s = {B300_STEP_US / 1e3:.2f} ms step")


if __name__ == "__main__":
    main()
