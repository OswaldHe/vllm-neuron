import sys, re, pandas as pd, numpy as np
D=sys.argv[1]; e0,e1=int(sys.argv[2]),int(sys.argv[3])
ex=pd.read_parquet(f"{D}/ExecutionInfo.parquet").set_index("execution_index")
W0=int(ex.loc[e0,"execution_start_ts"]); W1=int(ex.loc[e1,"execution_end_ts"]); T=W1-W0
cc=pd.read_parquet(f"{D}/CcOp.parquet"); cc=cc[(cc.start_ts>=W0)&(cc.end_ts<=W1)].copy()
def gsize(g):
    if not isinstance(g,str): return 0
    m=re.match(r"\s*\[\s*\[([^\]]*)\]",g); return len(m.group(1).split(",")) if m else 0
cc["n"]=cc.replica_group.map(gsize)
cc["kind"]=cc.operation+cc.n.map(lambda n:f"-{n}" if n else "")
g=cc.groupby("kind").agg(count=("duration_ns","size"),median_us=("duration_ns",lambda s:s.median()/1e3),sum_ms=("duration_ns",lambda s:s.sum()/1e6),in_KB=("input_size",lambda s:s.median()/1e3))
print(f"step = {T/1e6:.2f} ms (execs {e0}-{e1})\n", g.round(2).sort_values("sum_ms",ascending=False).to_string())
BIN=50; nb=T//BIN+1
def mask(x):
    m=np.zeros(nb,bool)
    for s,t in zip((x.start_ts.values-W0)//BIN,(x.end_ts.values-W0)//BIN): m[s:t+1]=True
    return m
ccm=mask(cc); kindm={k:mask(x) for k,x in cc.groupby("kind")}
print(f"collective union time: {ccm.sum()*BIN/1e6:.2f} ms = {ccm.mean():.0%} of step; sum of durations {cc.duration_ns.sum()/1e6:.2f} ms")
pk=pd.read_parquet(f"{D}/DmaPacket.parquet",columns=["start_ts","end_ts","pcore_idx","engine_idx","transfer_bytes","queue_name"])
pk=pk[(pk.start_ts>=W0)&(pk.end_ts<=W1)&(pk.engine_idx<128)]
for c in (0,1):
    x=pk[pk.pcore_idx==c]; engs=sorted(x.engine_idx.unique()); ix={e:i for i,e in enumerate(engs)}
    act=np.zeros((len(engs),nb),bool)
    for e,s,t in zip(x.engine_idx.values,(x.start_ts.values-W0)//BIN,(x.end_ts.values-W0)//BIN): act[ix[e],s:t+1]=True
    busy=act.mean(0); idle=1-busy
    byt=np.zeros(nb); mid=((x.start_ts.values+x.end_ts.values)//2-W0)//BIN; np.add.at(byt,mid,x.transfer_bytes.values)
    ccb=x.queue_name.str.startswith("CC").values; byc=np.zeros(nb); np.add.at(byc,mid[ccb],x.transfer_bytes.values[ccb])
    print(f"\ncore{c}: {len(engs)} engines, engine utilization {busy.mean():.0%} | inside collectives {busy[ccm].mean():.0%}, outside {busy[~ccm].mean():.0%} | "
          f"idle engine-time inside collectives {idle[ccm].sum()/idle.sum():.0%} | bytes moved inside collectives {byt[ccm].sum()/byt.sum():.0%} (CC queues' own bytes {byc.sum()/1e6:.1f} MB)")
    for k,m in sorted(kindm.items(),key=lambda kv:-kv[1].sum()):
        print(f"   {k:14s} {m.sum()*BIN/1e6:5.2f} ms  util {busy[m].mean():4.0%}  idle-share {idle[m].sum()/idle.sum():4.0%}  rate {byt[m].sum()/(m.sum()*BIN):6.1f} GB/s")
    m=~ccm; print(f"   {'(outside)':14s} {m.sum()*BIN/1e6:5.2f} ms  util {busy[m].mean():4.0%}  idle-share {idle[m].sum()/idle.sum():4.0%}  rate {byt[m].sum()/(m.sum()*BIN):6.1f} GB/s")
