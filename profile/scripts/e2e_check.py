#!/usr/bin/env python3
"""End-to-end timing of a capture, to compare with the bench (TTFT, decode step).

usage: e2e_check.py <case dir>      (run with a python that has pandas + pyarrow)
Prints, for every profiled rank: the NEFF executions seen by the device profile (ExecutionInfo),
the launches seen by the system trace (nc_exec_running events), and the client-side timing from
capture/windows.json.
"""
import glob, json, statistics, sys
import pandas as pd

case = sys.argv[1].rstrip("/")
g = f"{case}/ingest/profiles/global/"
win = json.load(open(f"{case}/capture/windows.json"))["windows"]
for phase, w in win.items():
    print(f"== client side, {phase} window")
    for r in w["requests"]:
        if r.get("chunks"):
            ts = [c[0] for c in r["chunks"]]
            gaps = [(b - a) * 1e3 for a, b in zip(ts, ts[1:])]
            s, e = w["start"]["wall_ns"] / 1e9, w["stopped"]["wall_ns"] / 1e9
            out = [x for t, x in zip(ts[1:], gaps) if t < s or t > e + 0.05]
            print(f"   streamed chunks {len(ts)}; median gap between chunks outside the profiled window "
                  f"{statistics.median(out):.2f} ms (n={len(out)}); TTFT {1e3 * (ts[0] - r['t_send']):.0f} ms")
        else:
            print(f"   request latency (max_tokens=1, = TTFT) {1e3 * (r['t_done'] - r['t_send']):.1f} ms")
for d in sorted(glob.glob(g + "*_session_*@latest")):
    name = d.split("/")[-1].split("_i-")[0]
    e = pd.read_parquet(d + "/ExecutionInfo.parquet").sort_values("execution_index")
    e["ms"] = (e.execution_end_ts - e.execution_start_ts) / 1e6
    e["gap_before_ms"] = (e.execution_start_ts - e.execution_end_ts.shift()) / 1e6
    print(f"== device profile {name}: {len(e)} NEFF executions, span "
          f"{(e.execution_end_ts.max() - e.execution_start_ts.min()) / 1e6:.2f} ms")
    print(e[["execution_index", "neff_name", "ms", "gap_before_ms"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    first = e.neff_name.iloc[0]
    starts = e[e.neff_name == first].execution_start_ts.values
    if len(starts) > 1:
        print("   start-to-start of the first NEFF (one decode step each):",
              [round((b - a) / 1e6, 2) for a, b in zip(starts, starts[1:])], "ms")
for d in sorted(glob.glob(g + "*@latest")):
    if "_session_" in d:
        continue
    s = pd.read_parquet(d + "/SystemProfileEvents.parquet")
    x = s[s.name == "nc_exec_running"].sort_values("start_ts")
    x = x[x.start_ts.diff().fillna(1e9) > 2e5]      # one event per physical core: keep the first of each pair
    print(f"== system trace {d.split('/')[-1]}: {len(x)} launches ({', '.join(sorted(set(x.name)))}), "
          f"sum {x.duration_ns.sum() / 1e6:.2f} ms, first start to last end {(x.end_ts.max() - x.start_ts.min()) / 1e6:.2f} ms")
    print("   launch ms:", [round(v / 1e6, 2) for v in x.duration_ns.values][:64])
    st = x.start_ts.values
    print("   start-to-start of every 4th launch (decode: one step):", [round((b - a) / 1e6, 2) for a, b in zip(st[::4], st[4::4])][:16])
    print("   gap us:", [round((a - b) / 1e3) for a, b in zip(x.start_ts.values[1:], x.end_ts.values[:-1])][:64])
