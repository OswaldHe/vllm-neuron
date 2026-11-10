"""Device idle inside the executions of an ingested device profile, attributed.

    python idle_attr.py <data dir> <first exec> <last exec> [min_gap_us]

For each execution and physical core: the union of every engine's instruction intervals is the
core's busy time; a stretch inside the execution with no instruction running on any engine is an
idle gap. Each gap is attributed, in this order, to what the trace shows during it:
  entry   the launch's start, before its first instruction (the cross-rank entry barrier)
  cc      a collective in flight (CcOp) -- the core waits for the other ranks / the transfer
  corebar a core barrier in progress (CoreBarriers) -- one core waits for the other
  dma     DMA descriptors pending (PendingDma > 0) -- the core waits for a load / store
  wait    none of the above, but the instruction that ends the gap waited on a semaphore
  idle    nothing in the trace explains it
"""
import glob
import sys

import numpy as np
import pandas as pd

D, e0, e1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
MIN_NS = float(sys.argv[4]) * 1e3 if len(sys.argv) > 4 else 1000.0


def table(name, cols=None):
    return pd.read_parquet(glob.glob(f"{D}/**/{name}.parquet", recursive=True)[0], columns=cols)


ins = table("Instruction", ["execution_index", "engine", "pcore_idx", "start_ts", "end_ts", "opcode",
                            "evt_wait_time_ns", "nki_source_location", "kernel_instruction_name"])
ex = table("ExecutionInfo").set_index("execution_index")
cc = table("CcOp", ["start_ts", "end_ts", "operation", "num_elements"])
cb = table("CoreBarriers", ["start_ts", "end_ts", "pcore_idx"])
pend = table("PendingDma", ["timestamp", "value", "pcore_idx"]).sort_values("timestamp")


def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(s, e, df, a="start_ts", b="end_ts"):
    m = (df[a] < e) & (df[b] > s)
    return df[m]


def pending_during(s, e, p):
    x = pend[(pend.pcore_idx == p)]
    before = x[x.timestamp <= s].tail(1)
    inside = x[(x.timestamp > s) & (x.timestamp < e)]
    vals = list(before.value) + list(inside.value)
    return max(vals) if vals else 0


rows = []
for e in range(e0, e1 + 1):
    es, ee = int(ex.loc[e, "execution_start_ts"]), int(ex.loc[e, "execution_end_ts"])
    for p in sorted(ins.pcore_idx.unique()):
        x = ins[(ins.execution_index == e) & (ins.pcore_idx == p)].sort_values("start_ts")
        busy = merge(list(zip(x.start_ts, x.end_ts)))
        gaps = []
        if busy and busy[0][0] - es >= MIN_NS:
            gaps.append((es, busy[0][0], "entry"))
        for (s0, e0_), (s1, _) in zip(busy, busy[1:]):
            if s1 - e0_ >= MIN_NS:
                gaps.append((e0_, s1, None))
        if busy and ee - busy[-1][1] >= MIN_NS:
            gaps.append((busy[-1][1], ee, "exit"))
        nxt = x.set_index("start_ts")
        for s, t, kind in gaps:
            why, detail = kind, ""
            if why is None:
                c = overlap(s, t, cc)
                b = overlap(s, t, cb[cb.pcore_idx == p])
                if len(c):
                    why = "cc"
                    detail = ",".join(sorted(set(c.operation)))
                elif len(b):
                    why = "corebar"
                elif pending_during(s, t, p) > 0:
                    why = "dma"
                else:
                    after = x[x.start_ts >= t].head(1)
                    w = float(after.evt_wait_time_ns.iloc[0]) if len(after) and pd.notna(after.evt_wait_time_ns.iloc[0]) else 0.0
                    why = "wait" if w > 0 else "idle"
                    if len(after):
                        detail = f"{after.engine.iloc[0]} {after.opcode.iloc[0]} {after.nki_source_location.iloc[0]}"
            rows.append(dict(exec=e, neff=ex.loc[e, "neff_name"][:6], core=p, start_us=(s - es) / 1e3, dur_us=(t - s) / 1e3,
                             why=why, detail=detail))
        busy_ns = sum(b - a for a, b in busy)
        rows.append(dict(exec=e, neff=ex.loc[e, "neff_name"][:6], core=p, start_us=-1, dur_us=(ee - es) / 1e3,
                         why="_exec", detail=f"busy {busy_ns / 1e3:.1f} us ({100 * busy_ns / (ee - es):.1f} %)"))

r = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)
summ = r[r.why != "_exec"].groupby(["exec", "neff", "core", "why"]).dur_us.agg(["count", "sum"]).round(1)
print(r[r.why == "_exec"][["exec", "neff", "core", "dur_us", "detail"]].to_string(index=False))
print()
print(summ.to_string())
print()
unexpl = r[r.why.isin(["idle", "wait"])].sort_values("dur_us", ascending=False)
print("largest gaps not covered by a collective / core barrier / pending DMA:")
print(unexpl.head(25).to_string(index=False))
r.to_csv(f"{D}/idle_gaps_{e0}_{e1}.csv", index=False)
