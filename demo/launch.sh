#!/usr/bin/env bash
# Start the DeepSeek-V4.1-Flash chat demo in tmux and attach to it:
#   top pane    : the vLLM server (serve.sh) and its log
#   bottom pane : chat.py --wait (waits for the server, warms it up, then >>>)
#
# usage: demo/launch.sh [--detach]
# Env: DEMO_TMUX_SOCKET (ds41demo), DEMO_TMUX_SESSION (demo), DEMO_COLS (160),
#      DEMO_ROWS (48), DEMO_CHAT_ROWS (27), plus serve.sh's VENV / MODEL / MAXLEN / PORT.
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOCK=${DEMO_TMUX_SOCKET:-ds41demo}
SESSION=${DEMO_TMUX_SESSION:-demo}
COLS=${DEMO_COLS:-160}
ROWS=${DEMO_ROWS:-48}
CHAT_ROWS=${DEMO_CHAT_ROWS:-27}
export VENV=${VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0}
export MAXLEN=${MAXLEN:-1280}
export PORT=${PORT:-8000}
export CHAT_MAX_MODEL_LEN=$MAXLEN
export CHAT_URL=http://127.0.0.1:$PORT
T=(tmux -L "$SOCK" -f "$HERE/tmux.conf")

if "${T[@]}" has-session -t "$SESSION" 2>/dev/null; then
  echo "demo already running; attach with: tmux -L $SOCK attach -t $SESSION" >&2
  exit 1
fi
if pgrep -f '^VLLM::|performance\._serve_main' >/dev/null; then
  echo "another vLLM server is running (pgrep -fa VLLM::); stop it first" >&2
  exit 1
fi
mkdir -p "$HERE/logs"

"${T[@]}" new-session -d -s "$SESSION" -x "$COLS" -y "$ROWS" -n demo
"${T[@]}" set-option -t "$SESSION" -w window-size manual
"${T[@]}" split-window -v -t "$SESSION:0" -l "$CHAT_ROWS"
"${T[@]}" select-pane -t "$SESSION:0.0" \
  -T "server log · vLLM 0.29 + vllm-neuron · DeepSeek-V4.1-Flash on trn3.48xlarge (8 Trainium3 devices, TP32)"
"${T[@]}" select-pane -t "$SESSION:0.1" -T "chat"
"${T[@]}" pipe-pane -t "$SESSION:0.0" -o "cat >> '$HERE/logs/server_$(date +%Y%m%d_%H%M%S).log'"
"${T[@]}" send-keys -t "$SESSION:0.0" "clear; '$HERE/serve.sh'" Enter
"${T[@]}" send-keys -t "$SESSION:0.1" "source '$HERE/pane.rc'" Enter
"${T[@]}" send-keys -t "$SESSION:0.1" "./chat.py --wait" Enter
"${T[@]}" select-pane -t "$SESSION:0.1"

if [ "$1" = "--detach" ] || [ ! -t 1 ]; then
  echo "demo started; attach with: tmux -L $SOCK attach -t $SESSION"
else
  exec "${T[@]}" attach -t "$SESSION"
fi
