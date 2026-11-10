"""Extract the decode-step breakdown from the 8-device profile as inputs for perf_model.py.

Rank 0, step = executions 8-11 (profile in /mnt/nvme/ds41_8d/prof). Writes pieces.csv (one row per
compute piece) and collectives.csv (one row per collective, with the intrinsic latency taken as
the duration on the last-arriving of the 8 profiled ranks).
"""
import glob, re, sys
import pandas as pd

PROF = "/mnt/nvme/ds41_8d/prof"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
E0, E1 = 8, 11

S = pd.read_csv(f"{PROF}/analysis_mr/rank00_step_segments.csv")
P = pd.read_csv(f"{PROF}/analysis_dma/dma_stalls_rank00_8_11_pieces.csv")
P = P[P.seg % 1 == 0].assign(seg=lambda d: d.seg.astype(int))
mb = P.pivot_table(index=["exec", "seg"], columns="core", values="MB").rename(columns={0: "MB_core0", 1: "MB_core1"})
cc_mb = (pd.read_csv(f"{PROF}/analysis_dma/dma_stalls_rank00_8_11_pieces.csv").query("seg % 1 != 0")
         .assign(seg=lambda d: (d.seg - 0.5).astype(int))
         .pivot_table(index=["exec", "seg"], columns="core", values="MB")
         .rename(columns={0: "cc_MB_core0", 1: "cc_MB_core1"}))


def kind(func):
    f = str(func)
    if "b1moe" in f: return "moe"
    if "b1attn" in f: return "attn"
    if "b1idx" in f: return "idx_topk"
    if "moe_tp" in f: return "draft_moe"
    if "mk_compress" in f: return "head"
    if any(k in f for k in ("mk_mx", "draft_seq")): return "draft_attn"
    if f.startswith("(TENSOR_LOAD)") or f.startswith("(compiler)"): return "launch_entry"
    return "other"


pieces = S.join(mb, on=["exec", "seg"]).join(cc_mb, on=["exec", "seg"])
pieces["kind"] = pieces.func.map(kind)
pieces.loc[pieces.seg == 0, "kind"] = "launch_entry"   # the piece before each launch's entry barrier

# collectives on all profiled ranks: intrinsic latency = duration on the last arriver
profs = sorted(p for p in glob.glob(f"{PROF}/data/ui_in512_bs1/profiles/global/*") if re.search(r"rank\d\d_i-", p))
durs = {}
for p in profs:
    r = int(re.search(r"rank(\d\d)", p).group(1))
    ex = pd.read_parquet(f"{p}/ExecutionInfo.parquet").set_index("execution_index")
    cc = pd.read_parquet(f"{p}/CcOp.parquet", columns=["start_ts", "end_ts", "duration_ns"])
    for e in range(E0, E1 + 1):
        s, t = int(ex.loc[e, "execution_start_ts"]), int(ex.loc[e, "execution_end_ts"])
        ops = cc[(cc.start_ts >= s) & (cc.end_ts <= t)].sort_values("start_ts").reset_index(drop=True)
        for i, d in enumerate(ops.duration_ns):
            durs.setdefault((e, i), {})[r] = d / 1e3
rows = []
for (e, i), d in sorted(durs.items()):
    seg = S[(S.exec == e)].reset_index(drop=True)
    if i >= len(seg) or pd.isna(seg.loc[i, "cc_op"]):
        continue
    rows.append(dict(exec=e, seg=int(seg.loc[i, "seg"]), op=seg.loc[i, "cc_op"], ranks=int(seg.loc[i, "cc_ranks"]),
                     bytes=seg.loc[i, "cc_bytes"], us_rank0=d.get(0), us_intrinsic=min(d.values()),
                     wait_rank0=d.get(0) - min(d.values()), n_profiled=len(d)))
coll = pd.DataFrame(rows)
pieces.to_csv(f"{OUT}/pieces.csv", index=False)
coll.to_csv(f"{OUT}/collectives.csv", index=False)
print(pieces.groupby("kind").agg(n=("seg", "size"), us=("compute_us", "sum"), MB0=("MB_core0", "sum"), MB1=("MB_core1", "sum")).round(1))
print(coll.groupby(["op", "ranks"]).agg(n=("seg", "size"), rank0_us=("us_rank0", "sum"), median=("us_rank0", "median"),
                                         intrinsic_med=("us_intrinsic", "median"), intrinsic_sum=("us_intrinsic", "sum"),
                                         wait_sum=("wait_rank0", "sum")).round(1))
print("cc-window MB core0/1:", round(pieces.cc_MB_core0.sum(), 1), round(pieces.cc_MB_core1.sum(), 1))

# --- FP4 share of Tensor Engine time per piece (core 0 and 1): union of MATMUL_MX fp4 intervals
def union_us(iv):
    tot, cs, ce = 0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None: tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return (tot + ((ce - cs) if ce is not None else 0)) / 1e3

R0 = [p for p in profs if "rank00_" in p][0]
ex0 = pd.read_parquet(f"{R0}/ExecutionInfo.parquet").set_index("execution_index")
ins = pd.read_parquet(f"{R0}/Instruction.parquet", columns=["engine", "pcore_idx", "start_ts", "end_ts", "opcode", "operands"])
pe = ins[(ins.engine == "Tensor") & ins.opcode.str.match(r"^(MATMUL|LDWEIGHTS)")]
fp4 = pe[(pe.opcode == "MATMUL_MX") & pe.operands.str.contains("src=fp4", regex=False)]
bf16 = pe[pe.operands.str.contains("src=bfloat16", regex=False)]
out = []
for p in pieces.itertuples():
    s = int(ex0.loc[p.exec, "execution_start_ts"]) + int(p.start_us * 1e3)
    t = s + int(p.compute_us * 1e3)
    row = {}
    for c in (0, 1):
        f = fp4[(fp4.pcore_idx == c) & (fp4.end_ts > s) & (fp4.start_ts < t)]
        a = pe[(pe.pcore_idx == c) & (pe.end_ts > s) & (pe.start_ts < t)]
        row[f"pe_fp4_c{c}"] = union_us(zip(f.start_ts.clip(lower=s), f.end_ts.clip(upper=t)))
        b = bf16[(bf16.pcore_idx == c) & (bf16.end_ts > s) & (bf16.start_ts < t)]
        row[f"pe_bf16_c{c}"] = union_us(zip(b.start_ts.clip(lower=s), b.end_ts.clip(upper=t)))
        row[f"pe_all_c{c}"] = union_us(zip(a.start_ts.clip(lower=s), a.end_ts.clip(upper=t)))
    out.append(row)
pieces = pd.concat([pieces.reset_index(drop=True), pd.DataFrame(out)], axis=1)
pieces.to_csv(f"{OUT}/pieces.csv", index=False)
print(pieces.groupby("kind")[["pe", "pe_all_c0", "pe_fp4_c0", "pe_bf16_c0", "pe_all_c1", "pe_fp4_c1", "pe_bf16_c1"]].median().round(1))
