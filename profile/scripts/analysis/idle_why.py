"""Why DMA engines are idle: classify every 50 ns bin of a step on each physical core."""
import sys, re, ast, os, functools, pandas as pd, numpy as np
from collections import Counter
D=sys.argv[1]; e0,e1=int(sys.argv[2]),int(sys.argv[3]); BIN=50
T_=lambda f,c=None: pd.read_parquet(f"{D}/{f}.parquet",columns=c)
ex=T_("ExecutionInfo").set_index("execution_index")
W0=int(ex.loc[e0,"execution_start_ts"]); W1=int(ex.loc[e1,"execution_end_ts"]); nb=(W1-W0)//BIN+1
def mask(s,t):
    m=np.zeros(nb,bool)
    for a,b in zip((np.asarray(s)-W0)//BIN,(np.asarray(t)-W0)//BIN): m[max(a,0):max(b+1,0)]=True
    return m
cc=T_("CcOp"); cc=cc[(cc.start_ts>=W0)&(cc.end_ts<=W1)]; inCC=mask(cc.start_ts,cc.end_ts)
inEX=mask([int(ex.loc[e,"execution_start_ts"]) for e in range(e0,e1+1)],[int(ex.loc[e,"execution_end_ts"]) for e in range(e0,e1+1)])
pk=T_("DmaPacket",["start_ts","end_ts","pcore_idx","engine_idx"]); pk=pk[(pk.start_ts>=W0)&(pk.end_ts<=W1)&(pk.engine_idx<128)]
ins=T_("Instruction",["engine","pcore_idx","start_ts","end_ts","opcode","operands","nki_source_location"])
ins=ins[(ins.end_ts>W0)&(ins.start_ts<W1)]
WAIT={"EVENT_SEMAPHORE","EVENT_SEMAPHORE_RANGE_CLEAR","DRAIN"}
def target(o):
    o=o or ""
    if re.search(r"\$S\[(24|25|27)\]",o): return "other physical core (core barrier)"
    if re.search(r"\$S\[2\]==",o): return "engine barrier ($S[2])"
    if re.search(r"\$S\[1\]",o): return "launch start (runtime)"
    if re.search(r"S\[\d+\] \(q",o): return "a DMA to complete"
    m=re.search(r"S\[\d+\] \((Vector|Tensor|Scalar|GpSimd|Sync)\)",o)
    if m: return "another engine"
    if "$S" in o: return "other global semaphore"
    return "drain / other"
@functools.lru_cache(None)
def spans(p):
    try: tree=ast.parse(open(p).read())
    except Exception: return []
    out=[]
    def walk(n,pre):
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch,(ast.FunctionDef,ast.ClassDef)):
                q=f"{pre}.{ch.name}" if pre else ch.name
                if isinstance(ch,ast.FunctionDef): out.append((ch.lineno,ch.end_lineno,q))
                walk(ch,q)
            else: walk(ch,pre)
    walk(tree,""); return out
@functools.lru_cache(None)
def fn(loc):
    if not isinstance(loc,str) or ":" not in loc: return "(compiler)"
    p,l=loc.rsplit(":",1); l=int(l) if l.isdigit() else -1; best=None
    for a,b,q in spans(p):
        if a<=l<=b and (best is None or a>=best[0]): best=(a,q)
    return f"{os.path.basename(p)}:{best[1] if best else '<module>'}"
res={}
for c in (0,1):
    x=pk[pk.pcore_idx==c]; engs=sorted(x.engine_idx.unique()); ix={e:i for i,e in enumerate(engs)}; NE=len(engs)
    act=np.zeros((NE,nb),bool)
    for e,s,t in zip(x.engine_idx.values,(x.start_ts.values-W0)//BIN,(x.end_ts.values-W0)//BIN): act[ix[e],s:t+1]=True
    nbusy=act.sum(0)
    xi=ins[ins.pcore_idx==c]; wk=xi[~xi.opcode.isin(WAIT)]; wt=xi[xi.opcode.isin(WAIT)]
    work=mask(wk.start_ts,wk.end_ts)
    # function doing work per bin (longest-running instr wins: approximate by last written)
    fbin=np.full(nb,-1); names=[]; nid={}
    for f,s,t in zip(wk.nki_source_location.map(fn),(wk.start_ts.values-W0)//BIN,(wk.end_ts.values-W0)//BIN):
        if f not in nid: nid[f]=len(names); names.append(f)
        fbin[max(s,0):t+1]=nid[f]
    # wait target per bin
    tnames=[]; tid={}; tbin=np.full(nb,-1)
    for g,s,t in zip(wt.operands.map(target),(wt.start_ts.values-W0)//BIN,(wt.end_ts.values-W0)//BIN):
        if g not in tid: tid[g]=len(tnames); tnames.append(g)
        seg=tbin[max(s,0):t+1]; seg[seg<0]=tid[g]   # first target seen keeps the bin
    cat=np.empty(nb,object)
    cat[:]="none busy: nothing running"
    nz=nbusy==0
    for i in np.nonzero(nz & (tbin>=0))[0]: cat[i]="none busy: core stalled on "+tnames[tbin[i]]
    cat[nz & work]="none busy: engines computing, no load in flight"
    cat[(nbusy>0)&(nbusy<NE)]="some engines busy (1-15)"
    cat[nbusy==NE]="all 16 busy"
    cat[~inEX]="between launches"
    idle=(NE-nbusy)/NE
    res[c]=dict(cat=cat,idle=idle,nbusy=nbusy,fbin=fbin,names=names,NE=NE)
for scope,sel in (("OUTSIDE collectives",~inCC),("INSIDE collectives",inCC)):
    print(f"\n=== {scope}: {sel.sum()*BIN/1e6:.2f} ms of the {(W1-W0)/1e6:.2f} ms step")
    rows=[]
    for c in (0,1):
        r=res[c]; df=pd.DataFrame(dict(cat=r["cat"][sel],idle=r["idle"][sel]))
        g=df.groupby("cat").agg(ms=("idle","size"),idle=("idle","sum")); g["ms"]=g.ms*BIN/1e6
        g["idle_share"]=g.idle/g.idle.sum(); g=g.drop(columns="idle"); g.columns=pd.MultiIndex.from_product([[f"core{c}"],g.columns]); rows.append(g)
    print(pd.concat(rows,axis=1).fillna(0).sort_values(("core0","idle_share"),ascending=False).round(3).to_string())
sel=~inCC
for c in (0,1):
    r=res[c]
    for label,m in (("engines computing, no load in flight", sel&(r["cat"]=="none busy: engines computing, no load in flight")),
                    ("some engines busy (1-15)", sel&(r["cat"]=="some engines busy (1-15)"))):
        f=pd.Series(r["fbin"][m]).map(lambda i: r["names"][i] if i>=0 else "(no instruction)")
        w=pd.Series(r["idle"][m])
        top=(w.groupby(f.values).sum()*BIN/1e6/ (1)).sort_values(ascending=False)
        tot=(r["idle"][sel]).sum()*BIN/1e6
        print(f"\ncore{c} | {label}: top functions by idle engine-time (ms of all-16-idle equivalent; share of all idle outside CC)")
        print((pd.DataFrame(dict(ms=top,share=top/tot)).head(10)).round(3).to_string())
    m=sel&(r["cat"]=="some engines busy (1-15)")
    print(f"core{c} | when some engines busy: mean {r['nbusy'][m].mean():.1f} of {r['NE']} engines busy")
