#!/usr/bin/env python3
"""Greedy against sampled decoding (temperature 0.7, top_p default, no seed) on the same prompts, batch 1.
A per-request `seed` is not supported: it stops the engine (no device random generator).
Sampled requests take the port's general path (plugin sampler), not the optimized greedy one."""
import json
from engines import *
from content_sweep import ask, PROMPTS
P = serve("sampled", 8321, 1280, "0-31"); res = []
try:
    wait_healthy(P)
    for kind, p in [PROMPTS[0], PROMPTS[10], PROMPTS[14]]:
        for temp in (0.0, 0.7):
            rs = [ask(8321, p, 256, temperature=temp, seed=None) for i in range(4)]
            for r in rs[1:]:
                r.update(kind=kind, temperature=temp); res.append(r)
                log(f"{kind:9s} T={temp}: {r['ntok']:4d} tok {r['tok_s']:6.1f} tok/s step {r['step_ms']:.2f} ms acc {r['accept_pct'] or 0:.1f}% tok/step {r['tokens_per_step']:.2f} TTFT {r['ttft_ms']:.0f}")
            (HERE / "results" / "sampled.json").write_text(json.dumps(res, indent=1))
finally:
    stop(P)
