"""What every engine of one core does in a time window: python window.py <rank dev dir> <core> <t0 ms> <t1 ms> [dma]"""
import glob, re, sys
import pandas as pd
D, C, A, B = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) * 1e6, float(sys.argv[4]) * 1e6
x = pd.read_parquet(f"{D}/Instruction.parquet", columns=["engine", "pcore_idx", "start_ts", "end_ts", "opcode", "nki_source_location", "operands", "hbm_read_bytes", "hbm_write_bytes"])
x = x[(x.pcore_idx == C) & (x.end_ts > A) & (x.start_ts < B)].sort_values("start_ts")
def src(l):
    return re.sub(r".*/(fused/[^/]+/|kernels/tp32/)", "", l) if isinstance(l, str) else ""
def cond(o):
    m = re.findall(r"\$?S\[\d+\][^ ]*(?: \(\w+\))?(?:>=|==|>|<=)\d+", o or "")
    return " ".join(m)
only_dma = len(sys.argv) > 5
for r in x.itertuples():
    if only_dma and not re.search("DMA|TENSOR_LOAD|TENSOR_STORE", r.opcode):
        continue
    print(f"{(r.start_ts - A) / 1e3:8.2f} {(r.end_ts - r.start_ts) / 1e3:7.2f} {r.engine:7s} {r.opcode:22s} {cond(r.operands):40s} {src(r.nki_source_location)}")
