#!/usr/bin/env bash
# Re-record rpt_figs/ds41_trn3_chat_demo.gif from a running demo.
#
# Preconditions: launch.sh is running and the chat has reached >>> once (the
# server is up and warm). This script then
#   1. leaves the chat (/bye) so the pane is at the shell prompt,
#   2. rehearses the demo twice at full speed (every new prompt length is slow
#      the first time; greedy decoding makes the shapes repeat exactly),
#   3. records a human-paced run (typing ./chat.py and the prompts) at 25 fps,
#   4. renders the GIF and prints the per-answer numbers of the recorded run.
#
# usage: demo/record/rec.sh [OUT.gif]
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEMO=$(dirname "$HERE")
OUT=${1:-$(dirname "$DEMO")/rpt_figs/ds41_trn3_chat_demo.gif}
SOCK=${DEMO_TMUX_SOCKET:-ds41demo}
PANE=${DEMO_TMUX_SESSION:-demo}:0.1
VENV=${VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0}
LOGS=$DEMO/logs
source "$VENV/bin/activate"

if [ "$(cat "$LOGS/chat_state" 2>/dev/null)" != "exit" ]; then
  tmux -L "$SOCK" send-keys -t "$PANE" -l "/bye"
  tmux -L "$SOCK" send-keys -t "$PANE" Enter
  sleep 2
fi

echo "rehearsing (2 passes) ..."
python "$HERE/drive.py" --fast
python "$HERE/drive.py" --fast
tmux -L "$SOCK" send-keys -t "$PANE" "clear" Enter
sleep 1

REC=$LOGS/rec_$(date +%Y%m%d_%H%M%S).jsonl
MARK=$(wc -l < "$LOGS/chat_stats.jsonl")
echo "recording -> $REC"
python "$HERE/record.py" "$REC" --fps 25 &
RP=$!
sleep 1.5
python "$HERE/drive.py" --no-bye
sleep 1.5
touch "$LOGS/rec_stop"
wait $RP

python "$HERE/render.py" "$REC" "$OUT" --max-gap 4.5 --hold-end 5
echo "recorded run:"
tail -n +$((MARK + 1)) "$LOGS/chat_stats.jsonl" | python -c '
import json, sys
for l in sys.stdin:
    d = json.loads(l)
    print(f"  {d[\"rate\"]:6.1f} tok/s  TTFT {d[\"ttft\"] * 1000:5.0f} ms  response {d[\"total\"]:5.2f} s  {d[\"ntok\"]:4d} tok  | {d[\"prompt\"][:60]}")'
