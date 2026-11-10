#!/bin/bash
# Decode case: batch 1, 512-token prompt, 4 engine iterations (= 4 decode steps) in steady decode.
# Device profile (NTFF) + system profile on rank 0.
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
HERE=$(cd "$(dirname "$0")" && pwd); OUT=$(dirname "$HERE")/decode_in512_bs1/capture
NEURON_RT_PROFILE_BUF_INSTRUCTION_MB=64 NEURON_RT_PROFILE_BUF_EVENT_MB=32 NEURON_RT_PROFILE_BUF_DMA_MB=64 \
python "$HERE/capture.py" --out "$OUT" --cell in512_bs1 --isl 512 --bs 1 --ms 1 --ctx 768 --decode-window 1.0 \
  --phases decode --sys-events 1000000 --port 8361 --max-iters 4 --delay-iters 0 --ranks 0
