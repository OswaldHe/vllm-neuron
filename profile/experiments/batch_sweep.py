#!/usr/bin/env python3
"""Throughput against the number of concurrent users on one 8-device engine (512-token sonnet
prompts, 128 output tokens, greedy). The engine admits ten sequences (DEEPSEEK_V41_MAX_NUM_SEQS=10).
usage: batch_sweep.py [--devices 32-63] [--port 8421] [--batches 1,2,4,8,10]"""
import argparse, json
from engines import *

ap = argparse.ArgumentParser(); ap.add_argument("--devices", default="32-63"); ap.add_argument("--port", type=int, default=8421)
ap.add_argument("--batches", default="2,4,8,10,1"); a = ap.parse_args()
OUT = HERE / "results" / "batch"
P = serve("ms10", a.port, 768, a.devices, max_num_seqs=10, extra_env={"DEEPSEEK_V41_MAX_NUM_SEQS": "10"}, out=OUT)
res = []
try:
    wait_healthy(P)
    for b in [int(x) for x in a.batches.split(",")]:
        n = max(8, 2 * b)
        t = time.time(); w = bench(P, f"warm_bs{b}", batch=b, n=n, warmups=0, out=OUT)
        log(f"bs{b} warm pass {time.time() - t:.0f} s {'ok' if w else 'FAILED'}")
        m = bench(P, f"bs{b}", batch=b, n=n, warmups=0, out=OUT)
        if m is None:
            log(f"bs{b} failed"); continue
        m["batch"] = b; res.append(m)
        log(f"bs{b}: TTFT {m['ttft_p50_ms']:.0f} TPOT {m['tpot_p50_ms']:.2f} per-user {1000 / m['tpot_p50_ms']:.1f} tok/s ITL {m['median_itl_ms']:.2f} "
            f"total output {m['decode_tok_s']:.1f} tok/s acc {m['accept_pct']:.1f}%")
        (OUT / "batch.json").write_text(json.dumps(res, indent=1))
finally:
    stop(P)
