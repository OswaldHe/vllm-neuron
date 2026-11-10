#!/usr/bin/env python3
"""Decode speed by content type, batch 1, greedy, chat mode (no thinking), up to 256 output tokens.

Every prompt is sent PASSES times; the last pass is reported (the first pays the one-time
trace of a new prompt length). Per answer: TTFT, tokens, tokens/s = (n-1)/(t_last - t_first),
draft acceptance and tokens per decode step from the server's /metrics counters, and the median
gap between streamed chunks (one decode step).

usage: content_sweep.py <tag> [--nospec] [--devices 0-31] [--port 8321]
"""
import argparse, json, statistics, sys, time, urllib.request
from engines import *

PROMPTS = [
 ("code", "Write a Python function that merges two sorted lists into one sorted list, with a short docstring."),
 ("code", "Write a C function that reverses a singly linked list in place. Include the struct definition."),
 ("code", "Write a SQL query that returns the top 5 customers by total order value in 2024 from tables customers(id, name) and orders(id, customer_id, total, created_at)."),
 ("structured", "Give me a JSON array of the 8 planets of the solar system with fields name, radius_km, and number_of_moons. Output only JSON."),
 ("structured", "Make a markdown table of the first 10 chemical elements with columns: atomic number, symbol, name, standard atomic weight."),
 ("copy", "Repeat the following paragraph three times, exactly, with a blank line between copies:\nThe quick brown fox jumps over the lazy dog near the bank of the quiet river, while the sun sets slowly behind the distant blue hills."),
 ("translate", "Translate into French: 'The committee will meet on Thursday to review the budget proposal, and a final decision is expected by the end of the month. Members who cannot attend should send their comments in writing before the meeting.'"),
 ("translate", "Translate into Chinese: 'The committee will meet on Thursday to review the budget proposal, and a final decision is expected by the end of the month. Members who cannot attend should send their comments in writing before the meeting.'"),
 ("math", "Solve step by step: a train leaves city A at 9:00 at 80 km/h, another leaves city B, 300 km away, at 9:30 at 100 km/h towards A. When do they meet?"),
 ("math", "Compute the derivative of f(x) = x^3 * sin(2x) and then evaluate it at x = pi/4. Show the steps."),
 ("explain", "Explain how a hash table works and why lookups are O(1) on average."),
 ("explain", "What causes the seasons on Earth? Answer in two paragraphs."),
 ("chat", "Hello, who are you?"),
 ("chat", "I am planning a three-day trip to Kyoto in April. What should I not miss?"),
 ("creative", "Write a short story about a lighthouse keeper who finds a message in a bottle."),
 ("creative", "Write a poem of 12 lines about the first snow of winter."),
]


def ask(port, prompt, max_tokens, temperature=0.0, seed=None):
    body = {"model": rb.SERVED_NAME, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, **({"seed": seed} if seed is not None else {}), "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"thinking_mode": "chat"}}
    c0 = rb._scrape_spec_counters(port)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ts = []; n = 0; ptok = 0; text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            u = j.get("usage") or {}
            if j.get("choices") and j["choices"][0]["delta"].get("content"):
                ts.append(time.perf_counter()); text.append(j["choices"][0]["delta"]["content"])
            if u.get("completion_tokens"):
                n = u["completion_tokens"]; ptok = u.get("prompt_tokens", 0)
    time.sleep(0.05)
    c1 = rb._scrape_spec_counters(port)
    d = {k: c1[k] - c0[k] for k in c0}
    gaps = [(b - a) * 1000 for a, b in zip(ts, ts[1:])]
    return {"prompt_tokens": ptok, "ntok": n, "ttft_ms": (ts[0] - t0) * 1000, "total_s": ts[-1] - t0,
            "tok_s": (n - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else None,
            "step_ms": statistics.median(gaps) if gaps else None, "steps": len(ts),
            "accept_pct": 100 * d["accepted_tokens"] / d["draft_tokens"] if d["draft_tokens"] else None,
            "tokens_per_step": (n - 1) / (len(ts) - 1) if len(ts) > 1 else None, "text": "".join(text)}


def run(port, tag, passes=3, max_tokens=256, out=None):
    out = Path(out or HERE / "results"); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, p in PROMPTS:
        rs = [ask(port, p, max_tokens) for _ in range(passes)]
        r = rs[-1]; r["kind"] = kind; r["prompt"] = p; r["first_pass"] = {k: rs[0][k] for k in ("ttft_ms", "tok_s")}
        rows.append(r)
        log(f"{kind:10s} in {r['prompt_tokens']:4d} out {r['ntok']:4d} TTFT {r['ttft_ms']:6.0f} ms  {r['tok_s']:6.1f} tok/s  step {r['step_ms']:5.2f} ms "
            f"acc {r['accept_pct'] if r['accept_pct'] is not None else float('nan'):5.1f}% tok/step {r['tokens_per_step']:.2f} | {p[:50]}")
        (out / f"content_{tag}.json").write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--nospec", action="store_true")
    ap.add_argument("--devices", default="0-31", help="logical cores: 0-31 = devices 0-7, 32-63 = devices 8-15"); ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--ctx", type=int, default=1280); ap.add_argument("--attach", action="store_true")
    ap.add_argument("--bench", action="store_true", help="also run the bench's in512 / in1024 batch-1 cells")
    a = ap.parse_args()
    if a.attach:
        run(a.port, a.tag); sys.exit()
    P = serve(a.tag, a.port, a.ctx, a.devices, extra_env={"DEEPSEEK_V41_NO_SPEC_DECODE": "1"} if a.nospec else None)
    try:
        wait_healthy(P)
        run(a.port, a.tag)
        if a.bench:
            res = []
            for isl in (512, 1024):
                bench(P, f"{a.tag}_warm_in{isl}", isl=isl, warmups=0)
                m = bench(P, f"{a.tag}_in{isl}", isl=isl); res.append(m)
                log(f"bench in{isl}: TTFT {m['ttft_p50_ms']:.1f} TPOT {m['tpot_p50_ms']:.3f} ITL {m['median_itl_ms']:.2f} acc {m['accept_pct']} tok/step {m['tokens_per_step']}")
            (HERE / "results" / f"bench_{a.tag}.json").write_text(json.dumps(res, indent=1))
    finally:
        stop(P)
