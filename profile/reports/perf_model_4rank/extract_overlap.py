"""Per-piece overlap inputs for perf_model.py: T = max(D, W) + S.

For every compute piece of the step (rank 0, executions 8-11) and each physical core:
  W            compute-engine work union (PE, DVE, ACT, POOL, SP; waits excluded)
  MB           DMA bytes of the piece on this core
  busy_<cls>   union time of one class of Tensor Engine work
  excl_<cls>   time that class runs while no other engine works (it alone is on the critical path)
Classes: fp4 (MATMUL_MX on FP4 weights), fp8 (MX on FP8), bf16, pe_other (fp32 etc.).
Writes pieces_overlap.csv next to pieces.csv.
"""
import glob
import re
import sys

import numpy as np
import pandas as pd

PROF = "/mnt/nvme/ds41_8d/prof"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
BIN = 20  # ns
WAIT = {"EVENT_SEMAPHORE", "EVENT_SEMAPHORE_RANGE_CLEAR", "DRAIN"}

R0 = [p for p in glob.glob(f"{PROF}/data/ui_in512_bs1/profiles/global/*") if re.search(r"rank00_i-", p)][0]
ex = pd.read_parquet(f"{R0}/ExecutionInfo.parquet").set_index("execution_index")
ins = pd.read_parquet(f"{R0}/Instruction.parquet", columns=["engine", "pcore_idx", "start_ts", "end_ts", "opcode", "operands"])
pk = pd.read_parquet(f"{R0}/DmaPacket.parquet", columns=["start_ts", "end_ts", "pcore_idx", "transfer_bytes", "engine_idx"])
pk = pk[pk.engine_idx < 128]
W0 = int(ex.loc[8, "execution_start_ts"]) - 1000
W1 = int(ex.loc[11, "execution_end_ts"]) + 1000
ins = ins[(ins.end_ts > W0) & (ins.start_ts < W1) & ~ins.opcode.isin(WAIT)].copy()
pk = pk[(pk.end_ts > W0) & (pk.start_ts < W1)]


def pe_class(r):
    if r.engine != "Tensor":
        return r.engine
    o = r.operands or ""
    if "src=fp4" in o:
        return "fp4"
    if "src=fp8" in o:
        return "fp8"
    if "src=bfloat16" in o:
        return "bf16"
    return "pe_other"


ins["cls"] = [pe_class(r) for r in ins.itertuples()]
pieces = pd.read_csv(f"{OUT}/pieces.csv")
nb = (W1 - W0) // BIN + 1
rows = []
for c in (0, 1):
    xi = ins[ins.pcore_idx == c]
    act = {}
    for cls, g in xi.groupby("cls"):
        m = np.zeros(nb, bool)
        for s, t in zip((g.start_ts.values - W0) // BIN, (g.end_ts.values - W0) // BIN):
            m[s:t + 1] = True
        act[cls] = m
    pe_cls = [k for k in ("fp4", "fp8", "bf16", "pe_other") if k in act]
    non_pe = [k for k in act if k not in pe_cls]
    any_work = np.any([act[k] for k in act], axis=0)
    other_eng = np.any([act[k] for k in non_pe], axis=0)
    xp = pk[pk.pcore_idx == c]
    for p in pieces.itertuples():
        s = int(ex.loc[p.exec, "execution_start_ts"]) + int(p.start_us * 1e3)
        t = s + int(p.compute_us * 1e3)
        a, b = (s - W0) // BIN, (t - W0) // BIN
        row = dict(exec=p.exec, seg=p.seg, kind=p.kind, core=c, T=p.compute_us,
                   W=any_work[a:b].sum() * BIN / 1e3,
                   MB=xp[(xp.start_ts >= s) & (xp.start_ts < t)].transfer_bytes.sum() / 1e6,
                   cc_MB=xp[(xp.start_ts >= t) & (xp.start_ts < t + int((p.cc_us if p.cc_us == p.cc_us else 0) * 1e3))]
                   .transfer_bytes.sum() / 1e6)
        for k in ("fp4", "fp8", "bf16", "pe_other"):
            m = act.get(k, np.zeros(nb, bool))[a:b]
            row[f"busy_{k}"] = m.sum() * BIN / 1e3
            row[f"excl_{k}"] = (m & ~other_eng[a:b]).sum() * BIN / 1e3
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/pieces_overlap.csv", index=False)
pd.set_option("display.width", 250)
print(df.groupby(["kind", "core"])[["T", "W", "MB", "busy_fp4", "excl_fp4", "busy_bf16", "excl_bf16", "busy_fp8", "excl_fp8"]]
      .median().round(1).to_string())
