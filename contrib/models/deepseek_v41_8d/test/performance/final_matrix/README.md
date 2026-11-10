# Final bench matrix (DeepSeek-V4.1-Flash on 8 Trainium 3 devices, TP32)

Backup of the last validated bench runs of the final code, copied from the instance's local
disk on 2026-11-10 (`/mnt/nvme/ds41_8d/val/v9`, `v8`). Method: `run_bench.py` (`vllm bench serve`, sonnet prompts,
128 output tokens, greedy, every cell warmed first); see `../README.md`.

Code: branch `deepseek_v41_trn3_staging`, commit `0a32285`, package `contrib/models/deepseek_v41_8d_trn3`
(the upstream-staging copy of this port; this folder's own `bench_report*.json` belong to this
branch's older code).

Each run folder holds `bench_report.json` (batch 1), `bench_report_ms10.json` (batch 10), the per-cell
`vllm bench serve` results with every request's timings and generated text (`cells/*.json`), the gzipped
server logs, the bench logs, and `chain.log` (integration test result and timestamps).

## v9, 2026-11-05 07:28-08:49 (the reported matrix: staging README, BLOG.md; integration test passed)

| Cell | TTFT p50 ms | TPOT p50 ms | per-stream tok/s (1000 / TPOT) | output tok/s (all streams, wall clock incl. prefill) | median ITL ms (one decode step) | mean input tokens | requests |
|---|---|---|---|---|---|---|---|
| in512_bs1 | 147 | 3.47 | 288.3 | 214.8 | 12.10 | 503 | 8 |
| in1024_bs1 | 244 | 3.95 | 253.2 | 171.1 | 12.14 | 1009 | 8 |
| in2048_bs1 | 427 | 3.89 | 257.1 | 138.9 | 12.19 | 2027 | 8 |
| in8192_bs1 | 1715 | 4.06 | 246.1 | 57.2 | 12.77 | 8146 | 8 |
| in512_bs10 | 785 | 17.68 | 56.6 | 405.5 | 40.35 | 502 | 10 |
| in1024_bs10 | 1312 | 20.91 | 47.8 | 294.5 | 41.20 | 1009 | 10 |
| in2048_bs10 | 2300 | 27.03 | 37.0 | 214.3 | 43.28 | 2027 | 10 |
| in8192_bs10 | 9403 | 80.55 | 12.4 | 64.6 | 49.74 | 8151 | 10 |

## v8, 2026-11-04 23:52 - 2026-11-05 00:40 (same decode kernels, before the PR-review fixes; integration test passed)

The in512_bs1 cell of this run is the 3.45 ms TPOT.

| Cell | TTFT p50 ms | TPOT p50 ms | per-stream tok/s (1000 / TPOT) | output tok/s (all streams, wall clock incl. prefill) | median ITL ms (one decode step) | mean input tokens | requests |
|---|---|---|---|---|---|---|---|
| in512_bs1 | 149 | 3.45 | 289.6 | 214.4 | 12.09 | 503 | 8 |
| in1024_bs1 | 245 | 3.93 | 254.4 | 170.9 | 12.12 | 1009 | 8 |
| in2048_bs1 | 428 | 3.91 | 255.9 | 138.7 | 12.20 | 2027 | 8 |
| in8192_bs1 | 1714 | 4.07 | 245.7 | 57.3 | 12.76 | 8146 | 8 |
| in512_bs10 | 789 | 17.53 | 57.0 | 406.7 | 40.47 | 502 | 10 |
| in1024_bs10 | 1322 | 21.83 | 45.8 | 296.5 | 41.51 | 1009 | 10 |
| in2048_bs10 | 2298 | 29.01 | 34.5 | 213.2 | 43.13 | 2027 | 10 |
| in8192_bs10 | 9397 | 76.37 | 13.1 | 66.3 | 49.83 | 8151 | 10 |
