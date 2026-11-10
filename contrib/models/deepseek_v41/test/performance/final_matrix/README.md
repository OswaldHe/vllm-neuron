# Final bench matrix (DeepSeek-V4.1-Flash on 16 Trainium 3 devices, TP64)

Backup of the last validated bench runs of the final code, copied from the instance's local
disk on 2026-11-10 (`/mnt/nvme/ds41_r4/val/s9`). Method: `run_bench.py` (`vllm bench serve`, sonnet prompts,
128 output tokens, greedy, every cell warmed first); see `../README.md`.

Code: branch `deepseek_v41_trn3_staging`, commit `0a32285`, package `contrib/models/deepseek_v41_trn3`
(the upstream-staging copy of this port; this folder's own `bench_report*.json` belong to this
branch's older code).

Each run folder holds `bench_report.json` (batch 1), `bench_report_ms10.json` (batch 10), the per-cell
`vllm bench serve` results with every request's timings and generated text (`cells/*.json`), the gzipped
server logs, the bench logs, and `chain.log` (integration test result and timestamps).

## s9, 2026-11-05 08:49-10:26 (measured right after the 8-device v9 run; integration test passed)

| Cell | TTFT p50 ms | TPOT p50 ms | per-stream tok/s (1000 / TPOT) | output tok/s (all streams, wall clock incl. prefill) | median ITL ms (one decode step) | mean input tokens | requests |
|---|---|---|---|---|---|---|---|
| in512_bs1 | 134 | 3.74 | 267.3 | 207.9 | 12.67 | 503 | 8 |
| in1024_bs1 | 231 | 3.72 | 268.6 | 182.4 | 12.63 | 1009 | 8 |
| in2048_bs1 | 418 | 3.85 | 259.7 | 141.9 | 12.67 | 2027 | 8 |
| in8192_bs1 | 1659 | 4.00 | 250.3 | 59.0 | 13.21 | 8146 | 8 |
| in512_bs10 | 714 | 16.99 | 58.9 | 432.1 | 39.01 | 502 | 10 |
| in1024_bs10 | 1240 | 19.36 | 51.6 | 335.7 | 39.89 | 1009 | 10 |
| in2048_bs10 | 2236 | 26.64 | 37.5 | 218.4 | 41.18 | 2027 | 10 |
| in8192_bs10 | 9149 | 71.51 | 14.0 | 68.9 | 44.61 | 8151 | 10 |
