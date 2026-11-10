"""Inputs for perf_model_v3.py: every piece's compute split by how it scales, and every stall by its cause.

Rank 0 of the 8-device profile, step = executions 8-11. Per compute piece and physical core:
  W                 compute-engine work union (waits excluded)
  MB                DMA bytes
  busy_<cat>_<cls>  engine busy time per scaling category and engine class
                    cat: shard (split over all ranks: experts, LM head, Engram), attn (split over the
                    attention group: heads, indexer), repl (repeated on every rank: hc, Sinkhorn, norms,
                    routing, step prologue), dma (DMA issue, follows the bytes), fixed (compiler: collective
                    microcode, control)
                    cls: fp4 / fp8 / bf16 / pe (other Tensor Engine) / vec (Vector, Scalar, GpSimd, Sync)
  floor_<cat>       per-instruction fixed cost in that busy time (n_instr x p10 duration of the engine)
  stall_<why>       time with no compute engine working, by what the core waits for:
                    core_barrier (other physical core), engine_barrier ($S[2] section boundary),
                    engine (another engine's semaphore), dma_moving (a DMA it waits for is moving data),
                    dma_latency (waiting for a DMA that is not moving yet), other (drain, runtime, none)
Collectives get the mean wait over the 8 profiled ranks next to rank 0's wait.
"""
import ast
import functools
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

PROF = "/mnt/nvme/ds41_8d/prof"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
BIN = 20
WAIT = {"EVENT_SEMAPHORE", "EVENT_SEMAPHORE_RANGE_CLEAR", "DRAIN"}
CATS = ["shard", "attn", "repl", "dma", "fixed"]
CLS = ["fp4", "fp8", "bf16", "pe", "vec"]
STALLS = ["core_barrier", "engine_barrier", "engine", "dma_moving", "dma_latency", "other"]


@functools.lru_cache(None)
def spans(p):
    try:
        tree = ast.parse(open(p).read())
    except Exception:
        return []
    out = []

    def walk(n, pre):
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch, (ast.FunctionDef, ast.ClassDef)):
                q = f"{pre}.{ch.name}" if pre else ch.name
                if isinstance(ch, ast.FunctionDef):
                    out.append((ch.lineno, ch.end_lineno, q))
                walk(ch, q)
            else:
                walk(ch, pre)
    walk(tree, "")
    return out


@functools.lru_cache(None)
def fn(loc):
    if not isinstance(loc, str) or ":" not in loc:
        return "(compiler)"
    p, l = loc.rsplit(":", 1)
    l = int(l) if l.isdigit() else -1
    best = None
    for a, b, q in spans(p):
        if a <= l <= b and (best is None or a >= best[0]):
            best = (a, q)
    return f"{os.path.basename(p)}:{best[1] if best else '<module>'}"


@functools.lru_cache(None)
def category(f):
    file, func = f.split(":", 1) if ":" in f else (f, "")
    if f == "(compiler)":
        return "fixed"
    if f in ("b1.py:_block", "mk_mx.py:_wdma", "b1attn.py:load_w") or func.startswith("load_"):
        return "dma"
    if file == "moe_tp.py":
        return "repl" if func in ("route_group", "router_logits", "_onehot") else "shard"
    if file in ("b1moe.py", "head_kernel.py", "b1eng.py"):
        return "shard"
    if file in ("b1attn.py", "b1idx.py", "mk_mx.py", "mk_compress.py", "b1src.py"):
        return "attn"
    return "repl"   # b1.py hc/Sinkhorn/glue, mk_common.py, draft_seq.py, step_kernel.py, blockquant.py


def eng_class(r):
    if r.engine != "Tensor":
        return "vec"
    o = r.operands or ""
    for k, tag in (("fp4", "src=fp4"), ("fp8", "src=fp8"), ("bf16", "src=bfloat16")):
        if tag in o:
            return k
    return "pe"


def stall_target(o):
    o = o or ""
    if re.search(r"\$S\[(24|25|27)\]", o):
        return 0   # core_barrier
    if re.search(r"\$S\[2\]==", o):
        return 1   # engine_barrier
    if re.search(r"S\[\d+\] \(q", o):
        return 3   # dma (moving / latency decided by packets)
    if re.search(r"S\[\d+\] \((Vector|Tensor|Scalar|GpSimd|Sync)\)", o):
        return 2   # engine
    return 5       # other


R0 = [p for p in glob.glob(f"{PROF}/data/ui_in512_bs1/profiles/global/*") if re.search(r"rank00_i-", p)][0]
ex = pd.read_parquet(f"{R0}/ExecutionInfo.parquet").set_index("execution_index")
W0 = int(ex.loc[8, "execution_start_ts"]) - 1000
W1 = int(ex.loc[11, "execution_end_ts"]) + 1000
ins = pd.read_parquet(f"{R0}/Instruction.parquet",
                      columns=["engine", "pcore_idx", "start_ts", "end_ts", "opcode", "operands", "nki_source_location"])
ins = ins[(ins.end_ts > W0) & (ins.start_ts < W1)].copy()
ins["wait"] = ins.opcode.isin(WAIT)
work = ins[~ins.wait].copy()
work["cat"] = work.nki_source_location.map(fn).map(category)
work["cls"] = [eng_class(r) for r in work.itertuples()]
work["dur"] = (work.end_ts - work.start_ts).astype(float)
p10 = work.groupby("engine").dur.quantile(0.10).to_dict()
work["fixed_ns"] = np.minimum(work.dur, work.engine.map(p10))
waits = ins[ins.wait].copy()
waits["tgt"] = waits.operands.map(stall_target)
pk = pd.read_parquet(f"{R0}/DmaPacket.parquet", columns=["start_ts", "end_ts", "pcore_idx", "transfer_bytes", "engine_idx"])
pk = pk[(pk.end_ts > W0) & (pk.start_ts < W1) & (pk.engine_idx < 128)]

pieces = pd.read_csv(f"{OUT}/pieces.csv")
nb = (W1 - W0) // BIN + 1
rows = []
for c in (0, 1):
    wk = work[work.pcore_idx == c]
    busy = np.zeros(nb, bool)
    for s, t in zip((wk.start_ts.values - W0) // BIN, (wk.end_ts.values - W0) // BIN):
        busy[s:t + 1] = True
    tgt = np.full(nb, 9, np.int8)            # lowest target id wins (priority order of STALLS)
    wt = waits[waits.pcore_idx == c]
    for g, s, t in zip(wt.tgt.values, (wt.start_ts.values - W0) // BIN, (wt.end_ts.values - W0) // BIN):
        seg = tgt[s:t + 1]
        np.minimum(seg, g, out=seg)
    dmam = np.zeros(nb, bool)
    xp = pk[pk.pcore_idx == c]
    for s, t in zip((xp.start_ts.values - W0) // BIN, (xp.end_ts.values - W0) // BIN):
        dmam[s:t + 1] = True
    for p in pieces.itertuples():
        s = int(ex.loc[p.exec, "execution_start_ts"]) + int(p.start_us * 1e3)
        t = s + int(p.compute_us * 1e3)
        a, b = (s - W0) // BIN, (t - W0) // BIN
        row = dict(exec=p.exec, seg=p.seg, kind=p.kind, core=c, T=p.compute_us,
                   W=busy[a:b].sum() * BIN / 1e3,
                   MB=xp[(xp.start_ts >= s) & (xp.start_ts < t)].transfer_bytes.sum() / 1e6,
                   cc_MB=xp[(xp.start_ts >= t) & (xp.start_ts < t + int((p.cc_us if p.cc_us == p.cc_us else 0) * 1e3))]
                   .transfer_bytes.sum() / 1e6)
        idle = ~busy[a:b]
        tg = tgt[a:b]
        mv = dmam[a:b]
        row["stall_core_barrier"] = (idle & (tg == 0)).sum() * BIN / 1e3
        row["stall_engine_barrier"] = (idle & (tg == 1)).sum() * BIN / 1e3
        row["stall_engine"] = (idle & (tg == 2)).sum() * BIN / 1e3
        row["stall_dma_moving"] = (idle & (tg == 3) & mv).sum() * BIN / 1e3
        row["stall_dma_latency"] = (idle & (tg == 3) & ~mv).sum() * BIN / 1e3
        row["stall_other"] = (idle & (tg >= 5)).sum() * BIN / 1e3
        w = wk[(wk.end_ts > s) & (wk.start_ts < t)]
        d = (w.end_ts.clip(upper=t) - w.start_ts.clip(lower=s)).astype(float) / 1e3
        for cat in CATS:
            for cls in CLS:
                m = (w.cat == cat) & (w.cls == cls)
                row[f"busy_{cat}_{cls}"] = d[m].sum()
            row[f"floor_{cat}"] = (w.fixed_ns[w.cat == cat] / 1e3).sum()
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/pieces_v3.csv", index=False)

# collectives: add the mean wait over the profiled ranks (balanced load = everyone at the mean)
coll = pd.read_csv(f"{OUT}/collectives.csv")
profs = sorted(p for p in glob.glob(f"{PROF}/data/ui_in512_bs1/profiles/global/*") if re.search(r"rank\d\d_i-", p))
durs = {}
for p in profs:
    r = int(re.search(r"rank(\d\d)", p).group(1))
    e_ = pd.read_parquet(f"{p}/ExecutionInfo.parquet").set_index("execution_index")
    cc = pd.read_parquet(f"{p}/CcOp.parquet", columns=["start_ts", "end_ts", "duration_ns"])
    for e in range(8, 12):
        s, t = int(e_.loc[e, "execution_start_ts"]), int(e_.loc[e, "execution_end_ts"])
        ops = cc[(cc.start_ts >= s) & (cc.end_ts <= t)].sort_values("start_ts").reset_index(drop=True)
        for i, d in enumerate(ops.duration_ns):
            durs.setdefault((e, i), []).append(d / 1e3)
seg_of = {}
for e in range(8, 12):
    segs = pieces[(pieces.exec == e) & pieces.cc_op.notna()].sort_values("seg")
    for i, sg in enumerate(segs.seg):
        seg_of[(e, sg)] = i
coll["wait_mean"] = [np.mean(durs[(c.exec, seg_of[(c.exec, c.seg)])]) - c.us_intrinsic for c in coll.itertuples()]
kind_of = {(p.exec, p.seg): p.kind for p in pieces.itertuples()}
coll["after_kind"] = [kind_of[(c.exec, c.seg)] for c in coll.itertuples()]
coll.to_csv(f"{OUT}/collectives_v3.csv", index=False)

pd.set_option("display.width", 250)
c0 = df[df.core == 0]
print("core 0, ms per step by piece kind:")
cols = ["T", "W"] + [f"stall_{s}" for s in STALLS]
print((c0.groupby("kind")[cols].sum() / 1e3).round(2).to_string())
busy_cols = [f"busy_{cat}_{cls}" for cat in CATS for cls in CLS]
bc = c0.groupby("kind")[busy_cols].sum() / 1e3
bc.columns = pd.MultiIndex.from_tuples([tuple(x.split("_")[1:]) for x in bc.columns])
print("\ncore 0 engine busy (ms) by scaling category (sum over classes):")
print(bc.T.groupby(level=0).sum().T.round(2).to_string())
print("\nper-instruction floor share by category:",
      {cat: round(c0[f"floor_{cat}"].sum() / max(c0[[f"busy_{cat}_{k}" for k in CLS]].sum().sum(), 1e-9), 2) for cat in CATS})
print("\ncollective waits (ms): rank0", round(coll.wait_rank0.sum() / 1e3, 3), " mean over ranks", round(coll.wait_mean.sum() / 1e3, 3))
print(coll.groupby(["op", "ranks", "after_kind"])[["wait_rank0", "wait_mean"]].sum().round(1).to_string())
