#!/usr/bin/env python3
"""Two 8-device engines on one trn3.48xlarge at the same time (devices 0-7 and 8-15).

Each engine is measured with the bench's batch-1 cell (sonnet, 512-token prompts, 128 output tokens,
greedy, 8 requests): alone (the other engine idle), then while the other engine serves a continuous
stream of the same kind of requests. REPS repetitions of each.
"""
import json, sys, threading, time
from engines import *

REPS = int(os.environ.get("REPS", 3)); ISL = int(os.environ.get("ISL", 512)); CTX = int(os.environ.get("CTX", 768))
OUT = HERE / "results" / f"dual_in{ISL}"
A = serve("A_dev0-7", 8321, CTX, "0-31", out=OUT)
B = serve("B_dev8-15", 8421, CTX, "32-63", out=OUT)
res = {"isl": ISL, "ctx": CTX, "runs": []}
try:
    wait_healthy(A); wait_healthy(B)
    res["healthy_s"] = {p.name: p.t_healthy - p.t_start for p in (A, B)}
    for p in (A, B):
        t = time.time(); bench(p, f"warm_{p.name}", isl=ISL, out=OUT, warmups=0)
        log(f"{p.name} warm pass {time.time() - t:.0f} s")
        bench(p, f"warm2_{p.name}", isl=ISL, out=OUT)
    def show(m, mode):
        m["mode"] = mode; res["runs"].append(m)
        log(f"{mode:28s} {m['engine']:10s} TTFT {m['ttft_p50_ms']:.1f} TPOT {m['tpot_p50_ms']:.3f} ITL {m['median_itl_ms']:.2f} "
            f"acc {m['accept_pct']:.1f}% tok/step {m['tokens_per_step']:.2f}")
        (OUT / "dual.json").write_text(json.dumps(res, indent=1))
    for r in range(REPS):
        for p in (A, B):
            show(bench(p, f"alone_{p.name}_{r}", isl=ISL, out=OUT), "alone")
        for fg, bg in ((A, B), (B, A)):
            # background: a long run of the same requests on the other engine
            box = {}
            th = threading.Thread(target=lambda: box.update(m=bench(bg, f"bg_{bg.name}_{r}", isl=ISL, n=96, warmups=0, out=OUT)))
            n0 = metric(bg.port, "vllm:request_success_total"); th.start()
            while metric(bg.port, "vllm:request_success_total") < n0 + 2:
                time.sleep(0.2)
            m = bench(fg, f"both_{fg.name}_{r}", isl=ISL, out=OUT)
            busy = th.is_alive(); th.join()
            m["bg_still_running_at_end"] = busy
            m["bg_window"] = [box["m"]["t0"], box["m"]["t1"]] if box.get("m") else None
            show(m, "other engine busy")
            if box.get("m"):
                show(box["m"], "background (96 requests)")
        # both measured at the same instant
        box = {}
        ths = [threading.Thread(target=lambda p=p: box.update({p.name: bench(p, f"sim_{p.name}_{r}", isl=ISL, n=32, out=OUT)})) for p in (A, B)]
        [t.start() for t in ths]; [t.join() for t in ths]
        for p in (A, B):
            show(box[p.name], "both, started together (32)")
finally:
    stop(A); stop(B)
    (OUT / "dual.json").write_text(json.dumps(res, indent=1))
    log("done")
