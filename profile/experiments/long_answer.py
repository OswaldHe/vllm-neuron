#!/usr/bin/env python3
"""One long answer (batch 1, greedy, chat mode): does the speed hold as the context grows?
Reports tokens/s, the median step and the tokens per step for every 256 output tokens.
usage: long_answer.py [--devices 0-31] [--port 8321] [--ctx 2304] [--tokens 2000]"""
import argparse, json, statistics, urllib.request
from engines import *

ap = argparse.ArgumentParser(); ap.add_argument("--devices", default="0-31"); ap.add_argument("--port", type=int, default=8321)
ap.add_argument("--ctx", type=int, default=2304); ap.add_argument("--tokens", type=int, default=2000); a = ap.parse_args()
PROMPTS = {"essay": "Write a long, detailed essay about the history of computing, from the abacus to modern data centers.",
           "code": "Write a complete Python module that implements a small in-memory key-value store with TTL support, an LRU eviction policy, a command parser, and unit tests. Write long, complete code."}


def ask(port, prompt, max_tokens):
    body = {"model": rb.SERVED_NAME, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"thinking_mode": "chat"}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); pts = []; text = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:]); u = j.get("usage") or {}
            if j.get("choices") and j["choices"][0]["delta"].get("content") is not None and u.get("completion_tokens"):
                pts.append((time.perf_counter() - t0, u["completion_tokens"])); text.append(j["choices"][0]["delta"]["content"])
    return pts, "".join(text)


P = serve("long", a.port, a.ctx, a.devices); res = {}
try:
    wait_healthy(P)
    for name, p in PROMPTS.items():
        for i in range(2):                      # the first pass pays the one-time trace of the prompt length
            pts, text = ask(a.port, p, a.tokens)
        rows = []
        for lo in range(0, a.tokens, 256):
            seg = [(t, n) for t, n in pts if lo < n <= lo + 256]
            if len(seg) < 3:
                continue
            gaps = [(b[0] - a_[0]) * 1e3 for a_, b in zip(seg, seg[1:])]
            rows.append({"tokens": f"{lo + 1}-{lo + 256}", "tok_s": (seg[-1][1] - seg[0][1]) / (seg[-1][0] - seg[0][0]),
                         "step_ms": statistics.median(gaps), "tok_per_step": (seg[-1][1] - seg[0][1]) / (len(seg) - 1)})
            log(f"{name:6s} {rows[-1]['tokens']:>10s}: {rows[-1]['tok_s']:6.1f} tok/s  step {rows[-1]['step_ms']:.2f} ms  {rows[-1]['tok_per_step']:.2f} tok/step")
        res[name] = {"ttft_ms": pts[0][0] * 1e3, "total_s": pts[-1][0], "ntok": pts[-1][1], "overall_tok_s": (pts[-1][1] - 1) / (pts[-1][0] - pts[0][0]),
                     "windows": rows, "text": text}
        log(f"{name}: {res[name]['ntok']} tokens in {res[name]['total_s']:.2f} s, {res[name]['overall_tok_s']:.1f} tok/s overall, TTFT {res[name]['ttft_ms']:.0f} ms")
        (HERE / "results" / "long_answer.json").write_text(json.dumps(res, indent=1))
finally:
    stop(P)
