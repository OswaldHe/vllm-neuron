#!/bin/bash
# Prefill case: batch 1, one 1,024-token prompt, max_tokens=1 (prefill + first token).
# Device profile (NTFF) + system profile on rank 0, explicit /start_profile ... /stop_profile window.
# DEEPSEEK_V41_MOE_TP=0 skips the batch-1 decode MoE copy (9.4 GiB a rank; prefill does not use it),
# which leaves HBM for profiler buffers large enough to hold a whole 1,024-token prefill.
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
HERE=$(cd "$(dirname "$0")" && pwd); OUT=$(dirname "$HERE")/prefill_in1024_bs1/capture
DEEPSEEK_V41_MOE_TP=${MOE_TP:-0} NEURON_RT_PROFILE_BUF_INSTRUCTION_MB=${BUF_I:-512} NEURON_RT_PROFILE_BUF_EVENT_MB=${BUF_E:-256} NEURON_RT_PROFILE_BUF_DMA_MB=${BUF_D:-512} \
python "$HERE/capture.py" --out "$OUT" --cell in1024_bs1 --isl 1024 --bs 1 --ms 1 --ctx 1280 \
  --phases prefill --sys-events 4000000 --port 8361 --max-iters 0 --delay-iters 0 --ranks 0 --warm-decode 0
