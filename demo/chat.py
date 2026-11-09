#!/usr/bin/env python3
"""Ollama-style terminal chat against the local vLLM OpenAI server.

Row 1 of the terminal is a fixed header: the model on the left, the live
speed / latency of the current (or last) response in the right corner. The
rows below are a scroll region holding the `>>>` REPL.

Speed is measured the way the benchmark's TPOT is: decode tok/s =
(completion tokens - 1) / (last token time - first token time), with the
completion token count taken from the server's own usage stats
(stream_options.continuous_usage_stats), not estimated from text.

usage: ./chat.py [--wait]
  --wait  wait for the server's /health, send one warmup request (the first
          request after a server start builds the fused decode launches,
          ~3 min), then start the REPL. launch.sh starts it this way.
Commands: /clear (drop the conversation), /bye (exit), /? (help).
"""

import json
import os
import re
import shutil
import sys
import time

import httpx

URL = os.environ.get("CHAT_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("CHAT_MODEL", "deepseek-v4.1-flash")
MAX_MODEL_LEN = int(os.environ.get("CHAT_MAX_MODEL_LEN", "1280"))
MAX_TOKENS = int(os.environ.get("CHAT_MAX_TOKENS", "700"))
MIN_ANSWER = 320  # drop old turns when fewer tokens than this would be left
LOG_DIR = os.environ.get("CHAT_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
STATE = os.path.join(LOG_DIR, "chat_state")  # idle / busy / exit, read by record/drive.py
STATS = os.path.join(LOG_DIR, "chat_stats.jsonl")  # one line per answer
LEFT = os.environ.get(
    "CHAT_HEADER",
    " DeepSeek-V4.1-Flash \u00b7 trn3.48xlarge (8x Trainium3) \u00b7 batch 1 \u00b7 speculative decoding",
)

ESC = "\033["
RESET = ESC + "0m"
DIM = ESC + "2m"
BOLD = ESC + "1m"
HDR_BG = ESC + "48;5;236m"
HDR_FG = ESC + "38;5;250m"
GREEN = ESC + "38;5;114m"
YELLOW = ESC + "38;5;221m"
CYAN = ESC + "38;5;117m"
GRAY = ESC + "38;5;245m"
CODE = ESC + "38;5;151m"
PROMPT_FG = ESC + "38;5;255m"

W, H = shutil.get_terminal_size((160, 27))


def out(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def set_state(s):
    try:
        with open(STATE, "w") as f:
            f.write(s)
    except OSError:
        pass


# ---------------------------------------------------------------- header ----
_last_badge = ["", ""]


def header(rate=None, ttft=None, total=None, ntok=None, live=False):
    """Redraw row 1: model on the left, stats in the right corner."""
    if rate is None and ttft is None:
        badge_plain = " waiting for a prompt "
        badge = GRAY + badge_plain
    else:
        r = f"{rate:6.1f} tok/s" if rate is not None else "    -- tok/s"
        t = f"TTFT {ttft * 1000:4.0f} ms" if ttft is not None else "TTFT   -- ms"
        tot = f"response {total:5.2f} s" if total is not None else "response   -- s"
        n = f"{ntok:4d} tok" if ntok is not None else "   - tok"
        sep = "  |  "
        badge_plain = f" {r}{sep}{t}{sep}{tot}{sep}{n} "
        rate_col = YELLOW if live else GREEN
        badge = (
            " " + BOLD + rate_col + r + RESET + HDR_BG + HDR_FG + sep
            + t + sep + BOLD + ESC + "38;5;255m" + tot + RESET + HDR_BG + HDR_FG + sep + n + " "
        )
    left = LEFT[: max(0, W - len(badge_plain) - 1)]
    pad = W - len(left) - len(badge_plain)
    line = HDR_BG + HDR_FG + left + " " * pad + badge + RESET
    # save cursor, draw row 1, restore cursor
    out("\0337" + ESC + "1;1H" + line + "\0338")


def setup_screen():
    out(ESC + "2J" + ESC + "H")
    header()
    out(ESC + f"3;{H}r")  # scroll region below the header (row 2 stays blank)
    out(ESC + "3;1H")


def restore_screen():
    out(ESC + "r" + ESC + f"{H};1H\n")


# ------------------------------------------------------------- streaming ----
class Printer:
    """Streams text with word wrapping and light markdown colouring."""

    def __init__(self, width):
        self.width = width - 1
        self.col = 0
        self.word = ""
        self.line_start = True
        self.in_code = False
        self.line_buf = ""  # text of the current line so far (for fence detection)

    def _style(self):
        return CODE if self.in_code else PROMPT_FG

    def _emit(self, s):
        out(self._style() + s + RESET)

    def _newline(self):
        out("\n")
        self.col = 0
        self.line_buf = ""

    def _flush_word(self):
        w = self.word
        if not w:
            return
        if self.col + len(w) > self.width and self.col > 0:
            self._newline()
        while len(w) > self.width:
            self._emit(w[: self.width - self.col])
            w = w[self.width - self.col:]
            self._newline()
        self._emit(w)
        self.col += len(w)
        self.line_buf += w
        self.word = ""

    def feed(self, text):
        for ch in text:
            if ch == "\n":
                self._flush_word()
                if self.line_buf.strip().startswith("```"):
                    self.in_code = not self.in_code
                self._newline()
            elif ch == " " or ch == "\t":
                self._flush_word()
                if self.col < self.width:
                    self._emit(" ")
                    self.col += 1
                    self.line_buf += " "
            else:
                self.word += ch

    def finish(self):
        self._flush_word()
        if self.col:
            self._newline()


# ---------------------------------------------------------------- metrics ---
_SPEC = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}


def spec_counters(client):
    try:
        text = client.get(URL + "/metrics", timeout=5).text
    except Exception:
        return None
    vals = {}
    for key, name in _SPEC.items():
        tot = 0.0
        found = False
        for m in re.finditer(r"^" + re.escape(name) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", text, re.M):
            tot += float(m.group(1))
            found = True
        if found:
            vals[key] = tot
    return vals if len(vals) == 3 else None


def count_prompt(client, messages):
    try:
        r = client.post(
            URL + "/tokenize",
            json={"model": MODEL, "messages": messages, "add_generation_prompt": True,
                  "chat_template_kwargs": {"thinking_mode": "chat"}},
            timeout=10,
        )
        return int(r.json()["count"])
    except Exception:
        return None


# ------------------------------------------------------------------- chat ---
def ask(client, messages):
    # fit the history into the context like ollama's num_ctx truncation
    while True:
        n_prompt = count_prompt(client, messages)
        if n_prompt is None or MAX_MODEL_LEN - n_prompt >= MIN_ANSWER or len(messages) <= 1:
            break
        del messages[0:2]
    max_tokens = MAX_TOKENS
    if n_prompt is not None:
        max_tokens = max(16, min(MAX_TOKENS, MAX_MODEL_LEN - n_prompt))

    before = spec_counters(client)
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "chat_template_kwargs": {"thinking_mode": "chat"},
    }
    pr = Printer(W)
    text_parts = []
    ntok = 0
    n_prompt_srv = None
    t_first = t_last = None
    last_hdr = 0.0
    t0 = time.perf_counter()
    header(None, None, 0.0, 0, live=True)
    with client.stream("POST", URL + "/v1/chat/completions", json=body, timeout=None) as r:
        if r.status_code != 200:
            r.read()
            out(ESC + "38;5;203m" + f"error {r.status_code}: {r.text[:300]}" + RESET + "\n")
            return None
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            now = time.perf_counter()
            usage = chunk.get("usage")
            if usage:
                ntok = usage.get("completion_tokens", ntok)
                n_prompt_srv = usage.get("prompt_tokens", n_prompt_srv)
            for ch in chunk.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content") or ""
                if delta:
                    if t_first is None:
                        t_first = now
                    t_last = now
                    text_parts.append(delta)
                    pr.feed(delta)
            if t_first is not None and now - last_hdr > 0.03:
                rate = (ntok - 1) / (t_last - t_first) if t_last > t_first and ntok > 1 else None
                header(rate, t_first - t0, now - t0, ntok, live=True)
                last_hdr = now
    t_end = time.perf_counter()
    pr.finish()
    if t_first is None:
        return ""
    ttft = t_first - t0
    dec = t_last - t_first
    rate = (ntok - 1) / dec if dec > 0 and ntok > 1 else 0.0
    total = t_last - t0
    header(rate, ttft, total, ntok, live=False)

    after = spec_counters(client)
    spec = ""
    if before and after:
        d = {k: after[k] - before[k] for k in after}
        if d["drafts"] > 0 and d["draft_tokens"] > 0:
            acc = d["accepted"] / d["draft_tokens"]
            per_step = 1.0 + d["accepted"] / d["drafts"]
            spec = f"{acc * 100:.1f} % draft acceptance, {per_step:.2f} tokens per decode step"

    rows = [
        ("total duration", f"{total:.3f}s"),
        ("prompt eval count", f"{n_prompt_srv if n_prompt_srv is not None else n_prompt} token(s)"),
        ("prompt eval duration", f"{ttft:.3f}s   (time to first token)"),
        ("eval count", f"{ntok} token(s)"),
        ("eval duration", f"{dec:.3f}s"),
        ("eval rate", f"{rate:.2f} tokens/s"),
    ]
    if spec:
        rows.append(("speculative decoding", spec))
    out("\n")
    for k, v in rows:
        style = BOLD + GREEN if k == "eval rate" else GRAY
        out(GRAY + f"{k + ':':<22}" + RESET + style + v + RESET + "\n")
    try:
        with open(STATS, "a") as f:
            f.write(json.dumps({"t": time.time(), "prompt": messages[-1]["content"], "ttft": ttft,
                                "decode_s": dec, "total": total, "ntok": ntok, "rate": rate,
                                "prompt_tokens": n_prompt_srv, "spec": spec,
                                "end_wait": t_end - t_last}) + "\n")
    except OSError:
        pass
    return "".join(text_parts)


def wait_for_server(client):
    """Block until /health answers, then send one short warmup request."""
    t0 = time.time()
    while True:
        try:
            if client.get(URL + "/health", timeout=3).status_code == 200:
                break
        except Exception:
            pass
        out(f"\r{GRAY}waiting for the server at {URL} ... {time.time() - t0:5.0f} s{RESET}")
        time.sleep(2)
    out(f"\n{GRAY}server is up; warming up (the first request builds the decode launches, ~3 min) ...{RESET}\n")
    t1 = time.time()
    client.post(URL + "/v1/chat/completions", timeout=None, json={
        "model": MODEL, "messages": [{"role": "user", "content": "What is the capital of France? One sentence."}],
        "temperature": 0.0, "max_tokens": 16, "chat_template_kwargs": {"thinking_mode": "chat"}})
    out(f"{GRAY}warm after {time.time() - t1:.0f} s{RESET}\n")
    time.sleep(1)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    if "--wait" in sys.argv[1:]:
        set_state("starting")
        wait_for_server(httpx.Client())
    setup_screen()
    client = httpx.Client()
    messages = []
    try:
        while True:
            set_state("idle")
            try:
                q = input(PROMPT_FG + BOLD + ">>> " + RESET)
            except EOFError:
                break
            q = q.strip()
            if not q:
                continue
            if q in ("/bye", "/exit", "/quit"):
                break
            if q == "/clear":
                messages.clear()
                out(GRAY + "Cleared session context" + RESET + "\n")
                continue
            if q in ("/?", "/help"):
                out(GRAY + "  /clear  clear the session context\n  /bye    exit\n" + RESET)
                continue
            set_state("busy")
            messages.append({"role": "user", "content": q})
            ans = ask(client, messages)
            if ans is None:
                messages.pop()
            else:
                messages.append({"role": "assistant", "content": ans})
            out("\n")
    except KeyboardInterrupt:
        pass
    finally:
        set_state("exit")
        restore_screen()


if __name__ == "__main__":
    main()
