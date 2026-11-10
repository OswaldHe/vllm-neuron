"""Who the collectives wait for: the same collectives of one step on several ranks' device profiles.

    python cc_skew.py <data dir> <first exec> <last exec>

Device clocks differ between devices, so ranks are compared by each collective's duration, not
its timestamps: a rank that reaches a collective early spends the time until the last one
arrives inside it, the last one barely waits. Per collective (the n-th CcOp of an execution on
every profiled rank): the last arriver is the rank with the shortest duration; every other
rank's wait is its duration minus that minimum. Reported per launch: per rank its execution
time, busy share (engine union), time inside collectives and time waiting for a later rank;
and how often each rank was the last to arrive.
"""
import glob
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

D, e0, e1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
profs = sorted(p for p in glob.glob(f"{D}/profiles/global/*") if re.search(r"rank\d\d_i-", p))


def load(p, name, cols=None):
    f = glob.glob(f"{p}/**/{name}.parquet", recursive=True)
    return pd.read_parquet(f[0], columns=cols) if f else None


def busy_ns(ins):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(zip(ins.start_ts, ins.end_ts)):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + ((cur_e - cur_s) if cur_e is not None else 0)


per = {}
for p in profs:
    rank = int(re.search(r"rank(\d\d)", p).group(1))
    ex = load(p, "ExecutionInfo").set_index("execution_index")
    cc = load(p, "CcOp", ["start_ts", "end_ts", "duration_ns", "operation", "num_elements"])
    ins = load(p, "Instruction", ["execution_index", "pcore_idx", "start_ts", "end_ts"])
    d = {}
    for e in range(e0, e1 + 1):
        es, ee = int(ex.loc[e, "execution_start_ts"]), int(ex.loc[e, "execution_end_ts"])
        ops = cc[(cc.start_ts >= es) & (cc.end_ts <= ee)].sort_values("start_ts")
        x = ins[ins.execution_index == e]
        b = [busy_ns(x[x.pcore_idx == c]) for c in sorted(x.pcore_idx.unique())]
        d[e] = dict(neff=ex.loc[e, "neff_name"][:6], dur=ee - es, busy=max(b), ops=ops.reset_index(drop=True))
    per[rank] = d

ranks = sorted(per)
print("ranks:", ranks)
for e in range(e0, e1 + 1):
    n = min(len(per[r][e]["ops"]) for r in ranks)
    last = Counter()
    wait = defaultdict(float)
    worst = []
    for i in range(n):
        durs = {r: int(per[r][e]["ops"].loc[i, "duration_ns"]) for r in ranks}
        lo = min(durs, key=durs.get)
        last[lo] += 1
        for r in ranks:
            wait[r] += durs[r] - durs[lo]
        spread = max(durs.values()) - durs[lo]
        worst.append((spread, i, per[ranks[0]][e]["ops"].loc[i, "operation"], lo))
    print(f"\nexec {e} {per[ranks[0]][e]['neff']}: {n} collectives")
    print("  rank  exec_us  busy%  in_cc_us  waiting_for_later_rank_us  last_to_arrive")
    for r in ranks:
        q = per[r][e]
        incc = q["ops"].duration_ns.sum() / 1e3
        print(f"  {r:4d}  {q['dur'] / 1e3:7.1f}  {100 * q['busy'] / q['dur']:5.1f}  {incc:8.1f}  {wait[r] / 1e3:25.1f}  {last[r]:4d}")
    worst.sort(reverse=True)
    print("  largest arrival spreads (us, collective #, op, last rank):",
          [(round(s / 1e3, 1), i, op, lo) for s, i, op, lo in worst[:5]])
