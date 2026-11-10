#!/bin/bash
# usage: ingest.sh <case dir, e.g. decode_in512_bs1> <display-name prefix>
# Stages each profiled worker's session (NTFF + system trace + the NEFFs) and ingests it with
# neuron-explorer into <case>/ingest (parquet tables, readable without the device).
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
HERE=$(cd "$(dirname "$0")" && pwd); C=$(dirname "$HERE")/$1; CAP=$C/capture; DATA=$C/ingest
SRC=${SRC_ROOT:-/home/ubuntu/ds41_staging_wt}
mkdir -p "$DATA"
for d in "$CAP"/output/i-*; do
  pid=${d##*_pid_}
  r=$(grep -m1 -o "Worker_TP[0-9]*_EP[0-9]* pid=$pid" "$CAP/server.log" | sed 's/Worker_TP\([0-9]*\)_.*/\1/')
  S=$(ls -d "$d"/*/ | head -1); ST=$CAP/stage_r$r; rm -rf "$ST"; mkdir -p "$ST"
  for f in "$S"/*; do ln -s "$f" "$ST/"; done
  for f in $(find "$CAP/output/neffs" -name "*.neff"); do ln -sf "$f" "$ST/"; done
  name=$(printf "%s_rank%02d" "$2" "$r"); t0=$(date +%s)
  timeout 3600 nice -n 10 neuron-explorer view -d "$ST" --data-path "$DATA" --display-name "$name" --ignore-nc-buf-usage \
    --source-code-root "$SRC" --ingest-only --port 3011 --api-server-port 3012 > "$C/ingest_$name.log" 2>&1
  echo "$name rc $? $(( $(date +%s) - t0 )) s" | tee -a "$C/ingest_all.log"
  rm -rf "$ST"
done
