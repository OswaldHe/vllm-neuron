#!/usr/bin/env python3
"""Type the demo into the tmux chat pane like a person and wait for each reply.

The chat pane must sit at a shell prompt in demo/ with the venv active (what
launch.sh leaves after /bye). Types ./chat.py, the PROMPTS, then /bye.

usage: drive.py [--fast] [--no-launch] [--no-bye]
  --fast       no typing delays or reading pauses (rehearsal / warmup)
  --no-launch  chat.py is already running and idle
  --no-bye     leave the chat open after the last answer (ends a recording on the stats)
"""
import os, random, subprocess, sys, time

SOCK = os.environ.get("DEMO_TMUX_SOCKET", "ds41demo")
PANE = os.environ.get("DEMO_TMUX_SESSION", "demo") + ":0.1"
LOG_DIR = os.environ.get("CHAT_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"))
STATE = os.path.join(LOG_DIR, "chat_state")
PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list, with a short docstring.",
    "Add three pytest unit tests for it. Keep them short.",
    "Now rewrite merge_sorted in Rust.",
]
FAST = "--fast" in sys.argv
rng = random.Random(7)


def tmux(*a):
    subprocess.run(["tmux", "-L", SOCK, *a], check=True)


def state():
    try:
        return open(STATE).read().strip()
    except OSError:
        return ""


def wait_state(want, timeout=600):
    t = time.time()
    while state() != want:
        if time.time() - t > timeout:
            raise SystemExit(f"timeout waiting for state {want!r} (now {state()!r})")
        time.sleep(0.02)


def type_text(s):
    for ch in s:
        tmux("send-keys", "-t", PANE, "-l", ch)
        if not FAST:
            d = rng.uniform(0.022, 0.06)
            if ch in " ,.":
                d += rng.uniform(0.02, 0.08)
            time.sleep(d)


def enter():
    tmux("send-keys", "-t", PANE, "Enter")


def main():
    if "--no-launch" not in sys.argv:
        try:
            os.remove(STATE)
        except OSError:
            pass
        time.sleep(0.2 if FAST else 1.2)
        type_text("./chat.py")
        time.sleep(0.1 if FAST else 0.4)
        enter()
    wait_state("idle", 60)
    time.sleep(0.2 if FAST else 1.0)
    for q in PROMPTS:
        type_text(q)
        time.sleep(0.1 if FAST else 0.5)
        enter()
        wait_state("busy", 30)
        wait_state("idle", 600)
        time.sleep(0.2 if FAST else 4.0)
    if "--no-bye" in sys.argv:
        return
    type_text("/bye")
    time.sleep(0.1 if FAST else 0.4)
    enter()
    wait_state("exit", 30)


if __name__ == "__main__":
    main()
