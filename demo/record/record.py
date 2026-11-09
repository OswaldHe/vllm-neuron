#!/usr/bin/env python3
"""Screen-record a tmux window: poll every pane's visible screen (with SGR
attributes), the cursor and the pane geometry, and append a JSON line per
changed frame with its wall-clock time. Stops when the stop file appears.

usage: record.py OUT.jsonl [--socket ds41demo] [--target demo:0] [--fps 25]
"""

import argparse
import json
import os
import subprocess
import time

SEP = "<<<--8<--PANE-->8-->>>"
FMT = "\t".join([
    "#{pane_index}", "#{pane_top}", "#{pane_left}", "#{pane_width}", "#{pane_height}",
    "#{cursor_x}", "#{cursor_y}", "#{cursor_flag}", "#{pane_active}", "#{pane_title}",
])


def snap(sock, target, npanes):
    cmd = ["tmux", "-L", sock, "list-panes", "-t", target, "-F", FMT]
    for i in range(npanes):
        cmd += [";", "display-message", "-p", SEP, ";", "capture-pane", "-p", "-e", "-t", f"{target}.{i}"]
    txt = subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout
    parts = txt.split(SEP + "\n")
    meta = [l.split("\t") for l in parts[0].splitlines() if l]
    panes = []
    for m, body in zip(meta, parts[1:]):
        idx, top, left, w, h, cx, cy, cf, act, title = m
        lines = body.split("\n")[: int(h)]
        panes.append({
            "i": int(idx), "top": int(top), "left": int(left), "w": int(w), "h": int(h),
            "cx": int(cx), "cy": int(cy), "cur": cf == "1", "active": act == "1",
            "title": title, "lines": lines,
        })
    return panes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--socket", default="ds41demo")
    ap.add_argument("--target", default="demo:0")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--stop-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "rec_stop"))
    a = ap.parse_args()
    if os.path.exists(a.stop_file):
        os.remove(a.stop_file)
    npanes = len(subprocess.run(["tmux", "-L", a.socket, "list-panes", "-t", a.target],
                                capture_output=True, text=True).stdout.splitlines())
    period = 1.0 / a.fps
    last = None
    n = 0
    with open(a.out, "w") as f:
        while not os.path.exists(a.stop_file):
            t = time.time()
            panes = snap(a.socket, a.target, npanes)
            key = json.dumps(panes, sort_keys=True)
            if key != last:
                f.write(json.dumps({"t": t, "panes": panes}) + "\n")
                f.flush()
                last = key
                n += 1
            dt = period - (time.time() - t)
            if dt > 0:
                time.sleep(dt)
        f.write(json.dumps({"t": time.time(), "end": True}) + "\n")
    print(f"record: {n} frames -> {a.out}")


if __name__ == "__main__":
    main()
