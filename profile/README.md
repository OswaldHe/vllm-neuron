# Profiles of DeepSeek-V4.1-Flash on 8 Trainium 3 devices

Backup of one decode profile and one prefill profile of the final 8-device deployment, captured on
2026-11-10 from a live `vllm serve` on a trn3.48xlarge (devices 0-7, 32 ranks, LNC = 2), so that
they can be inspected after the instance is gone. Each case holds the NEFFs, the device trace
(NTFF), the system trace, and the neuron-explorer ingest (parquet tables that any tool can read).

Code: branch `deepseek_v41_trn3_staging`, commit `0a32285`, package
`contrib/models/deepseek_v41_8d_trn3` (the code behind the numbers in `BLOG.md`). Only rank 0 is
profiled (worker `Worker_TP0`, logical core 0 = physical cores 0 and 1 of device 0).

| Case | What is in the window | Folder |
|---|---|---|
| Decode | batch 1, 512-token prompt, 4 consecutive decode steps of a running request (16 NEFF executions) | `decode_in512_bs1/` |
| Prefill | batch 1, one 1,024-token prompt with `max_tokens=1`: the whole prefill and the first token (46 NEFF executions) | `prefill_in1024_bs1/` |

## Do the profiles show the measured latency?

Yes. `<case>/e2e_check.txt` (made by `scripts/e2e_check.py`) lists every NEFF execution with its
duration and the host gap before it.

| | Bench (`final_matrix`, 2026-11-05) | This capture |
|---|---|---|
| Decode step, 512-token prompt | median ITL 12.10 ms | 12.03 ms from the start of one step to the start of the next, in the device profile (executions 4 -> 8) and in the system trace (launches 4 -> 8 and 8 -> 12); 12.06 ms between streamed chunks at the client outside the profiled window |
| Decode launches of the clean step | | 2.24 + 4.08 + 3.89 + 1.49 ms on the device (executions 8-11), host gaps 75-96 us between them |
| TTFT, 1,024-token prompt | median 244 ms | 243.9 ms from the first to the last NEFF execution on the device (226.6 ms executing, 17.3 ms of host gaps, mostly before the first blocks); 257 ms request latency at the client with the profiler on |

Notes for reading them:

- **Decode: use executions 8-11** (the third step). The first step carries the profiler start
  (execution 0 takes 7.5 ms instead of 2.2 ms: its first collective waits for the profiler on the
  other ranks). The device profile holds 12 executions (3 steps); the system trace holds 16
  launches (4 steps), two `nc_exec_running` events per launch (one per physical core).
- Decode NEFFs, in step order: `63f9...` step prologue + embedding + layers 0-7, `bdcc...` layers
  8-23, `83e2...` layers 24-39 + head, `acd4...` drafter + verification.
- **Prefill** runs one NEFF per transformer block (about 5.0-5.5 ms each for 1,024 tokens; a few blocks take
  6-9.5 ms), followed by four short executions (42-45) that produce the first token.
- The prefill server ran with `DEEPSEEK_V41_MOE_TP=0`. This skips the batch-1 decode copy of the
  MoE weights (9.4 GiB per rank), which prefill never uses, and leaves HBM for profiler buffers
  that hold a whole prefill (256 MB instruction, 64 MB event, 256 MB DMA; with the copy loaded
  the buffers cannot be allocated). The prefill kernels and their inputs are the same.
- neuron-explorer's HBM tracks count only static DMA; 98.5 % of this model's DMA bytes use
  descriptors generated on the device. Use `DmaPacket` / `DmaUsage` for traffic.

## Layout of a case

```
<case>/
  capture/run_spec.json          server command and environment
  capture/windows.json           client-side timestamps of the profiled window (and every streamed chunk)
  capture/server.log             full server log of the capture
  capture/output/neffs/          every NEFF the server loaded
  capture/output/i-*_pid_*/<session>/profile_nc_0_session_0.ntff   device trace of rank 0
  capture/output/i-*_pid_*/<session>/ntrace.pb, trace_info.pb      system trace of rank 0
  ingest/                        neuron-explorer data path (parquet)
    profiles/global/<name>_rank00@latest/                     system profile tables (SystemProfileEvents, ...)
    profiles/global/<name>_rank00_i-..._session_0@latest/     device profile tables (Instruction, DmaPacket,
                                                              ExecutionInfo, CcOp, SemaphoreUpdate, Flow, ...)
  e2e_check.txt                  end-to-end timing summary
```

Open the ingest in the UI (any machine with neuron-explorer, no device needed):

```bash
neuron-explorer view --data-path profile/decode_in512_bs1/ingest --port 3001 --api-server-port 3002
```

or read the tables directly (`pandas.read_parquet`, duckdb). Kernel source lines in the tables refer
to the staging commit above.

## Scripts

- `scripts/run_decode.sh`, `scripts/run_prefill.sh`: the two captures (`scripts/capture.py` starts
  the server, warms it up, opens the window with `/start_profile` and closes it with `/stop_profile`).
- `scripts/ingest.sh <case> <display name>`: neuron-explorer ingest of a capture.
- `scripts/e2e_check.py <case>`: the timing summary above.
- `scripts/analysis/`: the tools behind the blog's decode analysis (`step_breakdown.py`,
  `idle_attr.py`, `cc_skew.py`, `dma_stalls.py`, `window.py`, `cc_dma.py`, `idle_why.py`); they take
  the device-profile folder of a rank and the execution range (`8 11`). Run them with a python
  that has pandas and pyarrow.

## Other backups in this folder

- `reports/REPORT_in512_bs1_decode_2026-11-05.md` with `analysis_mr_2026-11-05/` and
  `analysis_dma_2026-11-05/`: the decode analysis that the blog's GPU-comparison section quotes
  (8 profiled ranks, captured 2026-11-05; that raw capture was on the instance's local disk and is
  not kept).
- `reports/perf_model_4rank/`: the analytical model of the "four GPUs" section of the blog.
- `experiments/`: the follow-up experiments of 2026-11-10 quoted in the blog (two engines at once,
  speed by content type, speculative decoding off), scripts and raw results.

## Sizes

About 2.9 GB in total. Seven files are larger than 95 MB (the two NTFFs: 232 MB and 1.1 GB; the
prefill `DmaPacket`, `Instruction`, `Flow` and `DmaUsage` tables; the decode `DmaPacket` table), which
is above GitHub's per-file limit: keep the folder out of a normal git push, or use LFS / a tarball.

Stack: vLLM 0.29.0, torch 2.13.0, neuronx-cc 2.28.7631.0, nki 0.7.0, runtime 2.35.30.0,
collectives 2.35.43.0, driver 2.31.7.0, tools 2.33.38.0.
