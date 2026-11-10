"""Why DMA throughput drops: low-DMA stretches of one rank's device profile, each explained.

    python dma_stalls.py <rank device dir> <first exec> <last exec> [thr_GBps=40] [min_us=4]

Per physical core, the DMA throughput (DmaPacket bytes, spread over each packet's duration) is
binned at 1 us. A low-DMA stretch is a run of bins under thr GB/s lasting >= min_us. Each stretch
is attributed, in this order, to
  between launches   no execution running on the device
  launch entry       inside a launch before its first collective (the entry barrier) ends
  collective         a collective (CcOp) covers most of it (only core 0 drives the CC DMA rings)
  core barrier       this core waits in a core barrier for the other core
  compute            engines busy (work union >= 60 % of the stretch) -- no load needed / issued
  dma latency        engines mostly wait on DMA-queue semaphores while only small DMAs run
                     (a serial chain: tiny load -> compute -> tiny store -> ...)
  sync wait          engines mostly wait on $S semaphores (core barrier / collective / runtime)
  engine wait        engines mostly wait on another engine's semaphore
  idle               none of the above
and named by the NKI function doing most of the work in it. Semaphore waits (EVENT_SEMAPHORE*,
DRAIN) are not work. Also per launch and per collective-delimited piece: bytes and mean
throughput per core, so the two cores of the LNC can be compared.

Writes <dir>/../dma_stalls_<rankNN>_<e0>_<e1>.csv (the stretches) and _pieces.csv.
"""
import ast
import functools
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

D, e0, e1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
THR = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
MIN_US = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
BIN = 1000
_rk = re.search(r"rank\d+", D)
OUT = os.path.join(os.path.dirname(D.rstrip("/")), f"dma_stalls_{_rk.group(0) + '_' if _rk else ''}{e0}_{e1}")
WAIT_OPS = {"EVENT_SEMAPHORE", "EVENT_SEMAPHORE_RANGE_CLEAR", "DRAIN"}


def table(name, cols=None):
    return pd.read_parquet(f"{D}/{name}.parquet", columns=cols)


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
    if not isinstance(loc, str) or ":" not in loc:
        return "(compiler)"
    path, line = loc.rsplit(":", 1)
    line = int(line) if line.isdigit() else -1
    best = None
    for a, b, q in func_spans(path):
        if a <= line <= b and (best is None or a >= best[0]):
            best = (a, q)
    return f"{os.path.basename(path)}:{best[1] if best else '<module>'}"


def wait_targets(o):
    """What a waiting instruction waits for: 'dma' (a DMA queue's completion semaphore), an engine
    name (that engine's semaphore), or 'sync' ($S[..]: runtime start / core barrier / collective
    semaphores)."""
    out = []
    for m in re.finditer(r"(\$?)S\[\d+\](?: \((\w+)\))?", o or ""):
        if m.group(1):
            out.append("sync")
        elif m.group(2):
            out.append("dma" if m.group(2).startswith(("q", "CC")) else m.group(2))
    return out or ["?"]


def merge(s, e):
    if len(s) == 0:
        return []
    o = np.argsort(s, kind="stable")
    out = [[s[o[0]], e[o[0]]]]
    for a, b in zip(s[o[1:]], e[o[1:]]):
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def covered(iv, a, b):
    return sum(max(0, min(y, b) - max(x, a)) for x, y in iv)


ex = table("ExecutionInfo").set_index("execution_index").sort_values("execution_start_ts")
W0 = int(ex.loc[e0, "execution_start_ts"]) - 400_000
W1 = int(ex.loc[e1, "execution_end_ts"]) + 200_000
pk = table("DmaPacket", ["start_ts", "end_ts", "pcore_idx", "transfer_bytes", "queue_type"])
pk = pk[(pk.end_ts > W0) & (pk.start_ts < W1)]
ins = table("Instruction", ["execution_index", "engine", "pcore_idx", "start_ts", "end_ts", "opcode", "operands",
                            "nki_source_location"])
ins = ins[(ins.end_ts > W0) & (ins.start_ts < W1)]
cc = table("CcOp", ["start_ts", "end_ts", "operation", "replica_group"])
cc = cc[(cc.end_ts > W0) & (cc.start_ts < W1)].sort_values("start_ts")
cb = table("CoreBarriers", ["start_ts", "end_ts", "pcore_idx"])
cb = cb[(cb.end_ts > W0) & (cb.start_ts < W1)]
ins = ins.assign(func=[resolve(l) for l in ins.nki_source_location], wait=ins.opcode.isin(WAIT_OPS),
                 dmaop=ins.opcode.str.contains("DMA|TENSOR_LOAD|TENSOR_STORE", regex=True))

nb = (W1 - W0) // BIN + 1
execs = [(e, int(ex.loc[e, "execution_start_ts"]), int(ex.loc[e, "execution_end_ts"]), ex.loc[e, "neff_name"][:6])
         for e in range(e0, e1 + 1)]
entry_end = {}
for e, s, t, _ in execs:
    first = cc[(cc.start_ts >= s) & (cc.end_ts <= t)].head(1)
    entry_end[e] = int(first.end_ts.iloc[0]) if len(first) else s
cc_iv = merge(cc.start_ts.values, cc.end_ts.values)


def throughput(c):
    x = pk[pk.pcore_idx == c]
    s, t, b = x.start_ts.values - W0, np.maximum(x.end_ts.values - W0, x.start_ts.values - W0 + 1), x.transfer_bytes.values
    h = np.zeros(nb)
    bs, bt = s // BIN, (t - 1) // BIN
    one = bs == bt
    np.add.at(h, bs[one], b[one])
    # packets spanning bins: split by overlap (packets are short, <= a few bins)
    for i in np.nonzero(~one)[0]:
        for k in range(bs[i], bt[i] + 1):
            ov = min(t[i], (k + 1) * BIN) - max(s[i], k * BIN)
            h[k] += b[i] * ov / (t[i] - s[i])
    return h / BIN  # bytes / ns = GB/s


rows, pieces = [], []
for c in (0, 1):
    h = throughput(c)
    xc = ins[ins.pcore_idx == c]
    work = xc[~xc.wait]
    work_iv = merge(work.start_ts.values, work.end_ts.values)
    cb_iv = merge(cb[cb.pcore_idx == c].start_ts.values, cb[cb.pcore_idx == c].end_ts.values)
    low = h < THR
    cut = np.zeros(nb, bool)
    for e, st, en, _ in execs:
        for t in (st, en, entry_end[e]):
            cut[(t - W0) // BIN] = True
    a0 = int((execs[0][1] - W0) // BIN) - 150
    a1 = int((execs[-1][2] - W0) // BIN) + 1
    i = a0
    while i < a1:
        if not low[i]:
            i += 1
            continue
        j = i + 1
        while j < a1 and low[j] and not cut[j]:
            j += 1
        if j - i >= MIN_US:
            a, b = W0 + i * BIN, W0 + j * BIN
            L = b - a
            inside = [(e, s, t, n) for e, s, t, n in execs if s < b and t > a]
            ex_cov = sum(max(0, min(t, b) - max(s, a)) for _, s, t, _ in inside)
            w = work[(work.start_ts < b) & (work.end_ts > a)]
            wt = covered(work_iv, a, b)
            top, main = "", ""
            if len(w):
                d = (w.end_ts.clip(upper=b) - w.start_ts.clip(lower=a))
                top = w.assign(d=d).groupby("func").d.sum().sort_values(ascending=False)
                main = top.index[0]
                top = ", ".join(f"{k} {v / 1e3:.1f}" for k, v in top.head(2).items())
            else:
                main = ""
            ws = xc[xc.wait & (xc.start_ts < b) & (xc.end_ts > a)]
            on = Counter()
            for o, s_, t_ in zip(ws.operands, ws.start_ts, ws.end_ts):
                for q in wait_targets(o):
                    on[q] += min(t_, b) - max(s_, a)
            ndma = int(((xc.dmaop) & (xc.start_ts >= a) & (xc.start_ts < b)).sum())
            e = inside[0][0] if inside else None
            if ex_cov < 0.5 * L:
                why = "between launches"
            elif e is not None and a < entry_end[e]:
                why = "launch entry"
            elif covered(cc_iv, a, b) >= 0.5 * L:
                why = "collective"
            elif covered(cb_iv, a, b) >= 0.5 * L:
                why = "core barrier"
            elif wt >= 0.6 * L:
                why = "compute"
            elif on and on.most_common(1)[0][0] == "dma":
                why = "dma latency"
            elif on and on.most_common(1)[0][0] == "sync":
                why = "sync wait"
            elif on:
                why = "engine wait"
            else:
                why = "idle"
            rows.append(dict(core=c, exec=e, neff=inside[0][3] if inside else "", start_ms=a / 1e6, us=L / 1e3, why=why,
                             work_us=wt / 1e3, dmas_issued=ndma, mean_GBps=h[i:j].mean(), waits_on=dict(on.most_common(2)),
                             main=main, top=top))
        i = j
    for e, s, t, n in execs:
        ops = cc[(cc.start_ts >= s) & (cc.end_ts <= t)]
        cuts = [s] + [v for p, q in zip(ops.start_ts, ops.end_ts) for v in (int(p), int(q))] + [t]
        for k in range(len(cuts) // 2):
            p, q = cuts[2 * k], cuts[2 * k + 1]
            x = pk[(pk.pcore_idx == c) & (pk.start_ts >= p) & (pk.start_ts < q)]
            lo = low[(p - W0) // BIN:(q - W0) // BIN].sum()
            pieces.append(dict(core=c, exec=e, neff=n, seg=k, start_us=(p - s) / 1e3, us=(q - p) / 1e3,
                               MB=x.transfer_bytes.sum() / 1e6, GBps=x.transfer_bytes.sum() / max(q - p, 1),
                               low_us=float(lo)))
        # collective k: bytes moved on this core while it runs
        for k, (p, q) in enumerate(zip(ops.start_ts, ops.end_ts)):
            x = pk[(pk.pcore_idx == c) & (pk.start_ts >= p) & (pk.start_ts < q)]
            pieces.append(dict(core=c, exec=e, neff=n, seg=k + 0.5, start_us=(p - s) / 1e3, us=(q - p) / 1e3,
                               MB=x.transfer_bytes.sum() / 1e6, GBps=x.transfer_bytes.sum() / max(q - p, 1),
                               low_us=float(low[(p - W0) // BIN:(q - W0) // BIN].sum())))

R = pd.DataFrame(rows)
P = pd.DataFrame(pieces).sort_values(["core", "exec", "seg"])
R.to_csv(f"{OUT}.csv", index=False)
P.to_csv(f"{OUT}_pieces.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_colwidth", 90)
span = (execs[-1][2] - execs[0][1]) / 1e3
print(f"window: execs {e0}-{e1} = {span:.1f} us; low DMA = under {THR:.0f} GB/s for >= {MIN_US:.0f} us")
for c in (0, 1):
    x = P[(P.core == c) & (P.seg % 1 == 0)]
    tot = pk[(pk.pcore_idx == c) & (pk.start_ts >= execs[0][1]) & (pk.start_ts < execs[-1][2])].transfer_bytes.sum()
    print(f"core {c}: {tot / 1e6:.0f} MB, mean {tot / (span * 1e3):.0f} GB/s over the window")
print("\nlow-DMA time by cause (us) per launch and core:")
print(R.pivot_table(index=["exec", "neff", "core"], columns="why", values="us", aggfunc="sum", fill_value=0).round(1).to_string())
print("\nlow-DMA time by cause, whole window (us):")
print(R.groupby(["core", "why"]).us.agg(["count", "sum"]).round(1).to_string())
print("\nlow-DMA time inside launches by the function doing most of the work then (us, top 25):")
q = R[~R.why.isin(["between launches", "launch entry", "collective"])]
print(q.pivot_table(index="main", columns="core", values="us", aggfunc="sum", fill_value=0)
      .assign(tot=lambda d: d.sum(axis=1)).sort_values("tot", ascending=False).head(25).round(1).to_string())
print("\nlargest low-DMA stretches:")
print(R.sort_values("us", ascending=False).head(40)[["core", "exec", "start_ms", "us", "why", "work_us", "dmas_issued",
                                                      "mean_GBps", "waits_on", "top"]].round(3).to_string(index=False))
