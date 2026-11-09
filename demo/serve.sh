#!/usr/bin/env bash
# vLLM OpenAI server for DeepSeek-V4.1-Flash on 8 trn3 devices (32 ranks),
# contrib/models/deepseek_v41_8d. launch.sh runs this in the top tmux pane.
#
# Env: VENV, MODEL (checkpoint dir), MAXLEN (max model len, default 1280 =
# the bench's in1024 context, already compiled), PORT (default 8000).
# Extra arguments are passed to the server.
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(dirname "$HERE")
VENV=${VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0}
MODEL=${MODEL:-/mnt/nvme/hf}
MAXLEN=${MAXLEN:-1280}
PORT=${PORT:-8000}

source "$VENV/bin/activate"
cd "$REPO/contrib/models/deepseek_v41_8d/test"
export PYTHONUNBUFFERED=1
# vLLM's periodic "Avg generation throughput" lines average over a window that
# includes idle time; off so they don't contradict the per-response speed.
export VLLM_LOG_STATS_INTERVAL=${VLLM_LOG_STATS_INTERVAL:-3600}

CMD=(python -m performance._serve_main
  --model "$MODEL"
  --served-model-name deepseek-v4.1-flash
  --tensor-parallel-size 32
  --max-model-len "$MAXLEN"
  --max-num-seqs 1
  --port "$PORT"
  --no-enable-prefix-caching
  --enable-log-requests
  --enable-log-outputs
  "$@")
printf '\033[1m$ %s\033[0m\n' "${CMD[*]}"
exec "${CMD[@]}"
