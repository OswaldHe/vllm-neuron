#!/usr/bin/env bash
# Stop the demo: Ctrl-C the server so its 32 workers shut down cleanly, wait
# for them to exit (frees devices 0-7), then remove the tmux session.
SOCK=${DEMO_TMUX_SOCKET:-ds41demo}
SESSION=${DEMO_TMUX_SESSION:-demo}
T=(tmux -L "$SOCK")

if "${T[@]}" has-session -t "$SESSION" 2>/dev/null; then
  "${T[@]}" send-keys -t "$SESSION:0.1" C-c 2>/dev/null
  "${T[@]}" send-keys -t "$SESSION:0.0" C-c
else
  echo "no tmux session $SESSION on socket $SOCK"
fi
for _ in $(seq 1 90); do
  pgrep -f '^VLLM::|performance\._serve_main' >/dev/null || break
  sleep 2
done
if pgrep -f '^VLLM::|performance\._serve_main' >/dev/null; then
  echo "server processes still running after 180 s:" >&2
  pgrep -fa '^VLLM::|performance\._serve_main' >&2
  exit 1
fi
"${T[@]}" kill-server 2>/dev/null
echo "demo stopped"
