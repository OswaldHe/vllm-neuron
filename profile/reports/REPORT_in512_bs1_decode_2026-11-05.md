# DeepSeek-V4.1-Flash 8 devices (deepseek_v41_8d_trn3, staging 0a32285): bs1 in512 decode profile

## What is captured

- Live `vllm serve` (32 ranks, LNC=2), bs1, input 512, decode only, `--profiler-config` 4 iterations, delay 0, device + system profile on ranks 0, 7, 8, 15, 16, 23, 24, 31 (the first and last rank of every second device).
- Capture: `/mnt/nvme/ds41_8d/prof/profiles/in512_bs1_mr`; ingested: `/mnt/nvme/ds41_8d/prof/data/ui_in512_bs1`.
- UI: `neuron-explorer view --data-path /mnt/nvme/ds41_8d/prof/data/ui_in512_bs1 --source-code-root /home/ubuntu/vllm-neuron --port 3001 --api-server-port 3002`, at http://localhost:3001/ (binds to localhost: `ssh -L 3001:localhost:3001 -L 3002:localhost:3002 <host>`). Profiles `ds41_8d_in512_bs1_decode_rankNN`.
- Every decode step is 4 launches: `af65d7` step prologue + embed + L00-L07, `787f53` L08-L23, `484ce3` L24-L39 + head, `ca4943` drafter with in-kernel verify.

## Which executions to look at

- The device profile of each rank holds 12 executions = 3 steps. **Executions 8-11 are the clean steady-state step.**
- Executions 0-7 carry the profiler start: the first launch waits in its first collective for the other ranks to finish starting their profilers (rank 31 execution 4 = 7.9 ms), and the host takes ~2 ms longer between the first launches. Real inference has neither (system-only trace: ~85 us between launches).
- The system trace has 16 launches (4 steps); launches 8-11 (and 12-15) are clean: rank 0 2234 / 4076 / 3895 / 1493 us, host gaps 75-99 us.

## No unexplained device idle

`scripts/idle_attr.py` on executions 8-11 of every profiled rank: every gap >= 1 us with no instruction running on a core is covered by a collective in flight, a core barrier or pending DMA. Nothing is left as idle / unexplained on any of the 8 ranks (`analysis_mr/idle_rankNN.txt`). Per rank and launch, core-0 "busy" is 84-91 % (78-80 % for the drafter).

Caveat for reading busy %: semaphore waits and drains are instructions that hold an engine while it waits (L08_L23 core 0: ~1.6 ms of instructions with no NKI source line, mostly `EVENT_SEMAPHORE` / `DRAIN` / collective microcode). `step_breakdown.py` counts them as wait, not work.

## Step budget (rank 0, step = executions 8-11)

| | us | share |
|---|---|---|
| device, 4 launches | 11,700 | 97 % |
| host gaps between launches (system trace) | ~340 | 3 % |
| step (= ITL 12.10 ms in the bench) | ~12,040 | |

Device time by kind (`step_breakdown.py`, core 0):

| | pieces | us | core-0 work us | PE | ACT | DVE | POOL | SP |
|---|---|---|---|---|---|---|---|---|
| MoE pieces (`b1moe.py`) | 40 | 3,943 | 3,416 | 1,833 | 882 | 2,013 | 1,007 | 959 |
| attention pieces (`b1attn.py`) | 41 | 3,381 | 2,948 | 1,389 | 545 | 648 | 1,229 | 789 |
| indexer top-k (`b1idx.py`) | 8 | 539 | 446 | 217 | 62 | 102 | 156 | 33 |
| drafter (`mk_mx` / `moe_tp` / `mk_compress` / `draft_seq`; one `mk_compress` piece is the L24 launch's head) | 13 | 1,116 | 772 | 228 | 155 | 324 | 209 | 148 |
| other (hc, compressor `b1src`, readout, notify) | 9 | 364 | 275 | 43 | 36 | 94 | 106 | 57 |
| launch entry (before the entry barrier) | 4 | 109 | 26 | | | | | |
| collectives | 111 | 2,248 | | | | | | |

Collectives per step: 51 TP8 all-reduces (attention o-proj, median 14.8 us), 47 TP32 all-reduces (MoE, median 20.0 us), 9 all-gathers (116 us), 4 entry barriers (233 us on rank 0, 106 us on rank 31: a launch's barrier waits for the last rank to start it).

Per layer the pattern is regular: attention piece ~85 us, TP8 all-reduce ~15 us, MoE piece ~96 us, TP32 all-reduce ~20 us = ~216 us; 40 layers = ~8.6 ms of the 11.7 ms. Indexer layers add a top-k piece (~65 us) and one more TP8 all-reduce.

## Where a step can get shorter (measured, ranked by size)

1. **Per-layer pieces are latency-bound, not engine-bound.** Median MoE piece 96 us: core-0 work union 84 us, but the busiest engine is DVE at 50 us (PE 46, POOL 25, SP 24, ACT 21). Median attention piece 85 us: work 74, PE 37, POOL 29, SP 17. No engine is near the piece time, so the time goes to dependency chains between engines. On top of that, ~1.6 ms per step of the pieces is wait on DMA or semaphores (piece time minus work). The levers are more overlap inside a layer and across layers (prefetching the next layer's weights and loads under the current compute), and less DVE work in MoE (2.0 ms per step, the largest single-engine total).
2. **Collectives: 2.25 ms per step (19 %).** 98 small all-reduces (120 KB) at 15-20 us each are latency-dominated. Skew is a small part of that (next item). Fewer collectives per layer (for example the attention TP8 reduction folded into the MoE one) or hiding them under compute is worth up to ~0.8 ms.
3. **Cross-rank imbalance: 0.19-0.55 ms per step.** `cc_skew.py` time spent waiting for a later rank in this step: 553 us for rank 0, 186 us for rank 31 (the least). Part of it is data-dependent: in this step rank 31 does a vocab-sharded embedding gather at the drafter start (`mk_common.engram_gather`, done only by the rank that owns the token) and a 32 us longer MoE piece in L00-L07 (expert-parallel load). Every other rank waits for it at the next all-reduce.
4. **Launch boundaries: ~0.67 ms per step.** Per launch: ~58 us entry barrier + ~25 us before it + ~85 us host gap.

## Tools (all in `/mnt/nvme/ds41_8d/prof/scripts`, run with `/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python`)

- `step_breakdown.py <rank device dir> 8 11 [out prefix]` cuts each launch at its collectives and names every piece (fused member, NKI function), with per-engine work and the following collective. It writes `_segments.csv` and `_functions.csv`; the outputs for ranks 0 and 31 are in `analysis_mr/`. The rank device dir is `data/ui_in512_bs1/profiles/global/ds41_8d_in512_bs1_decode_rankNN_i-*_session_0@latest`.
- `idle_attr.py <rank device dir> 8 11 [min_us]` explains device gaps (entry / cc / corebar / dma / wait / idle).
- `cc_skew.py <data dir> 8 11` runs across all profiled ranks: per collective, who arrives last and how long the others wait.

## Re-capture recipe (after a kernel change)

```
cd /mnt/nvme/ds41_8d/prof && NEURON_RT_PROFILE_BUF_INSTRUCTION_MB=64 NEURON_RT_PROFILE_BUF_EVENT_MB=32 NEURON_RT_PROFILE_BUF_DMA_MB=64 \
  python scripts/profile_cell_trn3.py --cell in512_bs1 --tag _X --isl 512 --bs 1 --ms 1 --ctx 768 --decode-window 1.0 \
  --phases decode --sys-events 1000000 --port 8361 --max-iters 4 --delay-iters 0 --ranks 0,7,8,15,16,23,24,31
```

- Buffers must be powers of 2 and small: HBM is nearly full on 8 devices (256 MB failed to allocate).
- Ingest each rank serially with `neuron-explorer view -d ... --data-path <dir> --display-name ... --ingest-only`.
- Analyse execution indices 8-11 (the 3rd step). With fewer profiled ranks (rank 0 alone) the start disturbance is shorter, but keep using the 3rd step.

## DMA valleys: why DMA throughput drops (rank 0, step = executions 8-11)

Tool: `scripts/dma_stalls.py <rank device dir> 8 11 [thr_GBps] [min_us]` (per-core DMA throughput from DmaPacket in 1 us bins; every stretch under 40 GB/s for >= 4 us attributed). Outputs: `analysis_dma/rank00_stalls.txt`, `rank31_stalls.txt` (same pattern), `analysis_dma/dma_stalls_rank00_8_11*.csv`.

- Per step, core 0 moves 1.74 GB and core 1 1.50 GB (mean 146 / 125 GB/s; up to ~560 GB/s in 10 us windows). At that peak the bytes would take ~3.1 ms of the 11.94 ms step: the weight stream is not the limiter; the valleys are the serial chain not needing (or not being allowed to issue) loads.
- Core 0 under 40 GB/s for ~6.0 ms of the step: compute with no load in flight 2.55 ms, collectives 1.39 ms, waits on core-barrier / collective / runtime semaphores 1.10 ms, between launches 0.39 ms, core barriers 0.23 ms, launch entry 0.20 ms.

### Launch boundaries (40.67-40.87, 44.82-45.02, 48.80-48.95 ms): ~150-200 us each, < 1.3 MB moved

| exec 8 -> 9, core 0 | ms | us | what |
|---|---|---|---|
| last weight burst of L00-L07 ends | 40.672 | | the launch's last layer has nothing left to prefetch (the next weights belong to the next NEFF) |
| last MoE compute | 40.672-40.690 | 18 | |
| last MoE all-reduce | 40.690-40.708 | 18 | |
| tail | 40.708-40.727 | 19 | output store, core barriers |
| no launch | 40.727-40.802 | 75 | runtime: ~25 us to notice the end (GpSimd stdio, CC-core notifications), then `nc_model_switch` to the next NEFF (50-73 us) |
| preamble | 40.802-40.834 | 32 | every engine waits on runtime start semaphores ($S[2] 8 us, then 21 us), register loads |
| kernel start | 40.834-40.870 | 36 | ~100 small loads (row positions, pre_mix rows, the stream: 8-24 element DMAs, ~0.65 us issue each) on the Sync queue ahead of the first layer's weights; the entry BARRIER (40.838-40.901, 63 us) runs meanwhile |
| first weight burst | 40.870 | | |

Exec 9 -> 10 is the same (194-205 us); exec 10 -> 11 (drafter) is 151 us because the drafter issues its weights right at kernel start (during its barrier).

### Drafter (48.94-50.43 ms): latency-bound, not DMA-bound

- 110-117 MB per core in 1.49 ms (~0.2 ms at peak).
- Start (49.00-49.11): entry barrier, verify-meta all-gather (30 us), embedding all-reduce over 32 ranks (63 us). Core 0 drives the collectives' DMA (the CC DMA queues exist on core 0 only) and prefetches 10 MB; core 1 moves 0.4 MB -> the two cores' lanes differ.
- 3 x [attention block `interleave_rows` 143-168 us, AR-8 24-40 us, MoE `moe_tp._up` 123-125 us, AR-32 33-43 us]: each piece is under 40 GB/s for 83-139 us; it loads only 4-15 MB in short bursts. The time is the hc helpers (`mk_common.hc_in_norm_spread`, `hc_out_spread`, `sum_squares`, sinkhorn: ~400 us of core-0 work), which bounce rows through HBM (`[T, dim]` bf16 and `[T, 1]` fp32 round trips) to change layout: serial chains of small DMAs and vector ops. The MoE expert loads then wait for the routing.
- Tail (50.22-50.43): 4 sequential draft tokens (`draft_body` ~25 us + all-gather ~9 us each) and the notify: no weights.
- Core split: core 0 alone runs the verify (`_tail_meta`) and the notify; core barriers 96 us (core 0) / 158 us (core 1).

### Main layers (same mechanisms, most of the step)

A normal layer (~222 us, e.g. exec 9 at 41.425 ms) has near-zero DMA for ~90 us on core 0:
- AR-8 after attention (~20 us): the next loads are ordered after the all-reduce result on the Sync queue (deliberate: issued earlier on the shared rings they delayed the all-reduce's own store, `b1attn.py` notes).
- x_norm / hc_mix / `route_group` before the experts (~28 us): the expert weights depend on routing; the expert burst then runs at ~340 GB/s (17 MB per core in ~50 us).
- MoE tail (`down_term`, overflow check, ~20 us).
- The two cores split attention by function (q path on core 0, kv path on core 1): in this layer core 0 prefetches the next layer's 6.5 MB q_a during the MoE all-reduce and core 1 loads its 2.6 MB kv at the block start, so the two lanes differ.
- Across the step, the AR-8 windows (51 a step, 0.78 ms) move almost nothing on either core (6-50 GB/s). The AR-32 windows carry the next layer's prefetch (exec 9: ~75 MB per core in 365 us), but not evenly every launch (exec 10: 15 MB core 0, 54 MB core 1).

### Ways to shorten the step (largest first)

1. Fewer NEFFs per step. Each boundary costs ~150-200 us of device time: runtime model switch, preamble, entry barrier, cold weight start, a tail with nothing to prefetch. Merging L00-L07 into L08-L23 and the drafter into L24-L39+head (ultimately one NEFF per step, so consecutive steps need no model switch either) is worth up to ~0.45-0.6 ms per step.
2. Drafter: port the main layers' on-chip hc path (stream kept in SBUF, b1 style) to the drafter blocks. Prefetch the next drafter block's weights during its all-reduces. Fold the meta all-gather and embedding all-reduce into one collective (or replicate the drafter embedding rows). The drafter has ~0.9 ms of low-DMA latency phases.
3. At the start of L08-L23 and L24-L39+head, issue the first layer's weights before the ~100 small loads, or move those loads to another queue, as the drafter does. Weights would then start ~35 us earlier, under the entry barrier.
4. Main layers: fill the AR-8, routing and MoE-tail windows with loads that do not depend on them (next layer's attention parts, shared expert), SBUF permitting. The AR-8 windows are the emptiest (0.78 ms a step with almost no DMA on either core). Rebalance the first-GEMM prefetch (6.5 vs 2.6 MB). Up to ~1.2 ms per step of low-DMA time exists there; the gain depends on SBUF room and needs tuning.
