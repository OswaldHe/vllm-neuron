"""Where one decode step's device time goes: each launch cut at its collectives, every piece named.

    python step_breakdown.py <rank device dir> <first exec> <last exec> [out prefix]

<rank device dir> is ONE rank's ingested device profile (the *_session_0@latest dir); the exec
range should be one clean steady-state step (all launches of it). Per launch, the timeline is cut
at every collective (CcOp): a compute piece, then the collective, then the next compute piece...
Semaphore waits and drains (EVENT_SEMAPHORE*, DRAIN) are instructions that occupy an engine
while it waits; they are counted as wait, not work. Each compute piece gets
  member    the fused member kernel (file path below fused/<launch>/) its instructions come from
  func      the NKI function owning most of its Tensor-engine work (else of all work),
            resolved from nki_source_location through the source file's AST
  work0/1   union of the work instructions on core 0 / core 1
  pe act dve pool sp   core 0 per-engine work (union): Tensor, Scalar, Vector, GpSimd, Sync queues
and each collective its op, bytes, ranks, duration and trigger delay. Work time is also split over
the NKI functions (each instant shared equally among the work instructions running then, core 0),
so the per-launch function table adds up to core 0's work time. Instructions without a source
line are compiler-inserted: GpSimd UNKNOWN is the collective microcode, the rest are named
by opcode. The step summary groups the pieces by the file of their func.

Writes <out prefix>_segments.csv and <out prefix>_functions.csv (default: next to the data dir).
"""
import ast
import functools
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

D, e0, e1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
OUT = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(D.rstrip("/")), f"step_{e0}_{e1}")
ENGINES = ["Tensor", "Vector", "Scalar", "GpSimd", "Sync"]


def table(name, cols=None):
    return pd.read_parquet(glob.glob(f"{D}/**/{name}.parquet", recursive=True)[0], columns=cols)


@functools.lru_cache(None)
def func_spans(path):
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError):
        return []
    spans = []

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = f"{prefix}.{ch.name}" if prefix else ch.name
                if not isinstance(ch, ast.ClassDef):
                    spans.append((ch.lineno, ch.end_lineno, q))
                walk(ch, q)
            else:
                walk(ch, prefix)

    walk(tree, "")
    return spans


@functools.lru_cache(None)
def resolve(loc):
    """nki_source_location 'path:line' -> (member, 'file.py:function')."""
    if not isinstance(loc, str) or ":" not in loc:
        return ("?", "?")
    path, line = loc.rsplit(":", 1)
    line = int(line) if line.isdigit() else -1
    m = re.search(r"/fused/[^/]+/(.+)/[^/]+$", path)
    member = m.group(1) if m else os.path.basename(os.path.dirname(path))
    best = None
    for a, b, q in func_spans(path):
        if a <= line <= b and (best is None or a >= best[0]):
            best = (a, q)
    return (member, f"{os.path.basename(path)}:{best[1] if best else '<module>'}")


def group_size(g):
    """Ranks in the collective's first replica group (the table holds it as text or a nested list)."""
    if isinstance(g, str):
        m = re.match(r"\s*\[\s*\[([^\]]*)\]", g)
        return len(m.group(1).split(",")) if m else 0
    try:
        return len(g[0])
    except (TypeError, IndexError):
        return 0


def union_ns(s, e):
    if len(s) == 0:
        return 0
    o = np.argsort(s, kind="stable")
    s, e = s[o], e[o]
    tot, cs, ce = 0, s[0], e[0]
    for a, b in zip(s[1:], e[1:]):
        if a > ce:
            tot += ce - cs
            cs, ce = a, b
        elif b > ce:
            ce = b
    return tot + ce - cs


def shared_time(s, e, labels):
    """Split the busy wall time among labels: each instant shared equally by the running instructions."""
    ev = np.concatenate([s, e])
    kind = np.concatenate([np.ones(len(s), int), -np.ones(len(e), int)])
    idx = np.concatenate([np.arange(len(s)), np.arange(len(e))])
    o = np.lexsort((-kind, ev))
    active, out, prev = {}, defaultdict(float), None
    for t, k, i in zip(ev[o], kind[o], idx[o]):
        if prev is not None and active and t > prev:
            share = (t - prev) / len(active)
            for j in active:
                out[labels[j]] += share
        if k > 0:
            active[i] = True
        else:
            active.pop(i, None)
        prev = t
    return out


WAIT_OPS = {"EVENT_SEMAPHORE", "EVENT_SEMAPHORE_RANGE_CLEAR", "DRAIN"}
ENG = {"Tensor": "pe", "Scalar": "act", "Vector": "dve", "GpSimd": "pool", "Sync": "sp"}

ins = table("Instruction", ["execution_index", "engine", "pcore_idx", "start_ts", "end_ts", "opcode",
                            "nki_source_location"])
ex = table("ExecutionInfo").set_index("execution_index")
cc = table("CcOp", ["start_ts", "end_ts", "duration_ns", "operation", "output_size", "cc_trigger_start_delay",
                    "replica_group"]).sort_values("start_ts")


def label(loc, engine, opcode):
    if isinstance(loc, str) and ":" in loc:
        return resolve(loc)
    if engine == "GpSimd" and opcode == "UNKNOWN":
        return ("(compiler)", "(collective microcode)")
    return ("(compiler)", f"({opcode})")


def clipped_union(w, t0, t1):
    return union_ns(np.clip(w.start_ts.values, t0, t1), np.clip(w.end_ts.values, t0, t1)) / 1e3


segs, funcs = [], []
for e in range(e0, e1 + 1):
    es, ee = int(ex.loc[e, "execution_start_ts"]), int(ex.loc[e, "execution_end_ts"])
    neff = ex.loc[e, "neff_name"][:6]
    x = ins[ins.execution_index == e]
    lab = [label(l, g, o) for l, g, o in zip(x.nki_source_location, x.engine, x.opcode)]
    x = x.assign(member=[a for a, _ in lab], func=[b for _, b in lab], wait=x.opcode.isin(WAIT_OPS))
    work = x[~x.wait]
    ops = cc[(cc.start_ts >= es) & (cc.end_ts <= ee)].reset_index(drop=True)
    cuts = [es] + [v for a, b in zip(ops.start_ts, ops.end_ts) for v in (int(a), int(b))] + [ee]
    for k in range(len(cuts) // 2):
        t0, t1 = cuts[2 * k], cuts[2 * k + 1]
        w = work[(work.start_ts < t1) & (work.end_ts > t0)]
        row = dict(exec=e, neff=neff, seg=k, start_us=(t0 - es) / 1e3, compute_us=(t1 - t0) / 1e3)
        for c in (0, 1):
            row[f"work{c}"] = clipped_union(w[w.pcore_idx == c], t0, t1)
        w0 = w[w.pcore_idx == 0]
        for g, n in ENG.items():
            row[n] = clipped_union(w0[w0.engine == g], t0, t1)
        src = w0[w0.engine == "Tensor"] if (w0.engine == "Tensor").any() else w0
        d = src.end_ts.clip(upper=t1) - src.start_ts.clip(lower=t0)
        row["member"] = src.assign(d=d).groupby("member").d.sum().idxmax() if len(src) else ""
        row["func"] = src.assign(d=d).groupby("func").d.sum().idxmax() if len(src) else ""
        if k < len(ops):
            op = ops.loc[k]
            row.update(cc_op=op.operation, cc_bytes=int(op.output_size), cc_ranks=group_size(op.replica_group),
                       cc_us=op.duration_ns / 1e3, cc_trigger_delay_us=(op.cc_trigger_start_delay or 0) / 1e3)
        segs.append(row)
    x0 = work[work.pcore_idx == 0]
    sh = shared_time(x0.start_ts.values, x0.end_ts.values, list(zip(x0.member, x0.func)))
    for (m, f), t in sh.items():
        funcs.append(dict(exec=e, neff=neff, member=m, func=f, work_us=t / 1e3))

S = pd.DataFrame(segs)
F = pd.DataFrame(funcs)
S.to_csv(f"{OUT}_segments.csv", index=False)
F.to_csv(f"{OUT}_functions.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_colwidth", 60)
COLS = ["seg", "start_us", "compute_us", "work0", "work1", "pe", "act", "dve", "pool", "sp", "member", "func",
        "cc_op", "cc_bytes", "cc_ranks", "cc_us"]
step = 0.0
for e in range(e0, e1 + 1):
    s = S[S.exec == e]
    dur = (ex.loc[e, "execution_end_ts"] - ex.loc[e, "execution_start_ts"]) / 1e3
    step += dur
    print(f"\n=== exec {e} {s.neff.iloc[0]}  {dur:.1f} us = compute pieces {s.compute_us.sum():.1f} + collectives "
          f"{s.cc_us.sum():.1f} ({len(s) - 1}); core-0 work {s.work0.sum():.1f}, PE {s.pe.sum():.1f}")
    print(s[COLS].round(1).to_string(index=False))
    f = F[F.exec == e].groupby(["member", "func"]).work_us.sum().sort_values(ascending=False)
    print(f"  core-0 work by NKI function (top 15 of {len(f)}):")
    print(f.head(15).round(1).to_string())

S["kind"] = S.func.str.split(":").str[0].replace("", "(none)")
first = S.seg == 0
print(f"\n=== step summary (execs {e0}-{e1}, core 0): device {step:.1f} us")
print("compute pieces by the file of their main function (entry = piece before each launch's first collective):")
k = S[~first].groupby("kind").agg(pieces=("seg", "count"), compute_us=("compute_us", "sum"), work_us=("work0", "sum"),
                                  pe_us=("pe", "sum"), act_us=("act", "sum"), dve_us=("dve", "sum"),
                                  pool_us=("pool", "sum"), sp_us=("sp", "sum"))
k.loc["entry"] = [first.sum(), S[first].compute_us.sum(), S[first].work0.sum(), S[first].pe.sum(), S[first].act.sum(),
                  S[first].dve.sum(), S[first].pool.sum(), S[first].sp.sum()]
print(k.sort_values("compute_us", ascending=False).round(1).to_string())
print("collectives by op and ranks:")
print(S.dropna(subset=["cc_op"]).groupby(["cc_op", "cc_ranks"]).cc_us.agg(["count", "sum", "median"]).round(1).to_string())
print(f"total: compute pieces {S.compute_us.sum():.1f} us, collectives {S.cc_us.sum():.1f} us")
