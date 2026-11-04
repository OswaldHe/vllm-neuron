# SPDX-License-Identifier: Apache-2.0
"""The megakernel's index path (`kernels/mk_compress.py`) at long read lengths, on Trn2.

    python test_index_long.py [read_len ...]

`build_ktT` + `index_scores_tokens` + `index_topk_rows` on both physical cores, the way
`layer_kernel` runs them for one decode span (minus the all-reduce over the attention
group), against torch: the fp32 scores, and the selected set of the masked, bf16-rounded
scores. 4224 keys run the single-pass code; 16384 the column-chunked scores and masking;
65536 and 131072 also the two-level top-k.
"""

import sys
import traceback

from kharness import compare, run_nki  # noqa: E402 - sets the environment first
import torch

import nki
import nki.language as nl
import nki.isa as nisa

from deepseek_v41.kernels.mk_common import NEG_BOUND
from deepseek_v41.kernels.mk_compress import (
    CAND_BLOCK,
    CAND_TOPK,
    build_ktT,
    index_candidates,
    index_scores_tokens,
    index_topk_rows,
)

import os
T, NH, KD, TOPK = int(os.environ.get("T_ROWS", 16)), int(os.environ.get("NH", 8)), 128, 512


@nki.jit
def k_index(qi, wi, ktab, clen, flags):
    (topk,) = flags
    read_len, kd = ktab.shape
    nh = wi.shape[1]
    pid = nl.program_id(0)
    sc = nl.ndarray((T, read_len), dtype=nl.float32, buffer=nl.shared_hbm)
    sc16 = nl.ndarray((T, read_len), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    out = nl.ndarray((T, topk), dtype=nl.int32, buffer=nl.shared_hbm)
    kT = build_ktT(ktab, read_len, kd, pid=pid, name="kT_test")
    half = T // 2
    tok0 = 0 if pid == 0 else T - half
    index_scores_tokens(qi, wi, kT, sc, tok0, half, tok0, nh, read_len, kd)
    nisa.core_barrier(data=sc, cores=(0, 1))
    index_topk_rows(sc, clen, sc16, out, 0, T, read_len, topk, pid)
    return sc, out


@nki.jit
def k_cand_src(qi, wi, ktab, clen):
    """The candidate source layer's mask (its own launch, as in the model)."""
    read_len, kd = ktab.shape
    nh = wi.shape[1]
    nb = read_len // CAND_BLOCK
    pid = nl.program_id(0)
    sc = nl.ndarray((T, read_len), dtype=nl.float32, buffer=nl.shared_hbm, name="sc")
    bs16 = nl.ndarray((T, nb), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="cand_bs16")
    keep = nl.ndarray((T * nb, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="cand_keep")
    kT = build_ktT(ktab, read_len, kd, pid=pid, name="kT_test")
    half = T // 2
    tok0 = 0 if pid == 0 else T - half
    index_scores_tokens(qi, wi, kT, sc, tok0, half, tok0, nh, read_len, kd)
    nisa.core_barrier(data=sc, cores=(0, 1))
    index_candidates(sc, clen, bs16, keep, 0, T, read_len, pid)
    return keep


@nki.jit
def k_cand_use(qi, wi, ktab, clen, keep, flags):
    """A consumer layer's top-k under the source's mask."""
    (topk,) = flags
    read_len, kd = ktab.shape
    nh = wi.shape[1]
    pid = nl.program_id(0)
    sc = nl.ndarray((T, read_len), dtype=nl.float32, buffer=nl.shared_hbm, name="sc")
    sc16 = nl.ndarray((T, read_len), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="sc16")
    out = nl.ndarray((T, topk), dtype=nl.int32, buffer=nl.shared_hbm, name="out")
    kT = build_ktT(ktab, read_len, kd, pid=pid, name="kT_test")
    half = T // 2
    tok0 = 0 if pid == 0 else T - half
    index_scores_tokens(qi, wi, kT, sc, tok0, half, tok0, nh, read_len, kd)
    nisa.core_barrier(data=sc, cores=(0, 1))
    index_topk_rows(sc, clen, sc16, out, 0, T, read_len, topk, pid, keep_hbm=keep)
    return out


def reference(qi, wi, ktab, clen):
    q = qi.float().reshape(T, NH, KD)
    s = torch.relu(torch.einsum("thd,jd->thj", q, ktab.float()))
    sc = (s * wi.float()[:, :, None]).sum(1)
    j = torch.arange(sc.shape[1])[None, :]
    masked = torch.where(j < clen.reshape(T, 1), sc, torch.full_like(sc, NEG_BOUND))
    return sc, masked.to(torch.bfloat16).float()


def check(read_len: int) -> bool:
    torch.manual_seed(read_len)
    qi = torch.randn(T, NH * KD).to(torch.bfloat16)
    wi = torch.rand(T, NH)
    ktab = torch.randn(read_len, KD).to(torch.bfloat16)
    # live lengths: one row full, one shorter than top-k, the rest spread out
    clen = torch.randint(TOPK, read_len + 1, (T, 1), dtype=torch.int32)
    clen[0, 0], clen[1, 0] = read_len, TOPK // 2
    sc, idx = run_nki(k_index, qi, wi, ktab, clen, (TOPK,), grid=2)
    sc_ref, m16 = reference(qi, wi, ktab, clen)
    ok = compare(f"scores n={read_len}", sc, sc_ref, tol_rel=1e-2)
    for r in range(T):
        live = int(clen[r, 0])
        k = min(TOPK, live)
        vals, top = torch.topk(m16[r], k)
        thr = vals.min()
        got = [int(x) for x in idx[r].tolist()]
        picks = [x for x in got if x >= 0]
        bad = [x for x in picks if not (0 <= x < live) or m16[r, x] < thr]
        good = len(picks) == k and len(set(picks)) == k and not bad
        if live < TOPK:
            good &= got.count(-1) == TOPK - live
        if not good:
            print(f"  row {r} (clen {live}): {len(picks)} picks, {len(set(picks))} distinct, "
                  f"{len(bad)} below the threshold, e.g. {bad[:5]}")
        ok &= good
    print(f"read_len={read_len}: {'OK' if ok else 'WRONG'}", flush=True)
    return ok


def check_cand(read_len: int) -> bool:
    torch.manual_seed(read_len + 1)
    qi = torch.randn(T, NH * KD).to(torch.bfloat16)
    wi = torch.rand(T, NH)
    ktab = torch.randn(read_len, KD).to(torch.bfloat16)
    clen = torch.randint(CAND_TOPK * CAND_BLOCK, read_len + 1, (T, 1), dtype=torch.int32)
    clen[0, 0], clen[1, 0], clen[2, 0] = read_len, 1000, read_len - 3
    keep = run_nki(k_cand_src, qi, wi, ktab, clen, grid=2)
    idx = run_nki(k_cand_use, qi, wi, ktab, clen, keep, (TOPK,), grid=2)
    nb = read_len // CAND_BLOCK
    keep = keep.reshape(T, nb)
    sc_ref, m16 = reference(qi, wi, ktab, clen)
    j = torch.arange(read_len)[None, :]
    masked = torch.where(j < clen.reshape(T, 1), sc_ref, torch.full_like(sc_ref, NEG_BOUND))
    blocks = masked.reshape(T, nb, CAND_BLOCK).amax(-1)
    last = (clen.reshape(T) - 1) // CAND_BLOCK
    blocks[torch.arange(T), last] = 1e30
    b16 = blocks.to(torch.bfloat16).float()
    ok = True
    for r in range(T):
        k = min(CAND_TOPK, nb)
        vals = torch.topk(b16[r], k).values
        thr = vals.min()
        want_n = int((vals > NEG_BOUND).sum())
        kept = keep[r] > 0.5
        bad = int((kept & (b16[r] < thr)).sum()) + int((kept & (b16[r] <= NEG_BOUND)).sum())
        good = int(kept.sum()) == want_n and bad == 0 and bool(kept[last[r]])
        # the consumer's picks: live, inside a kept block, at or above its own threshold
        live = (j[0] < int(clen[r, 0])) & kept.repeat_interleave(CAND_BLOCK)
        cand = torch.where(live, m16[r], torch.full_like(m16[r], NEG_BOUND))
        kk = min(TOPK, int(live.sum()))
        cthr = torch.topk(cand, kk).values.min()
        picks = [x for x in idx[r].tolist() if x >= 0]
        good &= len(picks) == kk and len(set(picks)) == kk and all(bool(live[x]) and cand[x] >= cthr for x in picks)
        if not good:
            dead = sum(1 for x in picks if not bool(live[x]))
            low = sum(1 for x in picks if bool(live[x]) and cand[x] < cthr)
            outk = sum(1 for x in picks if not bool(kept[x // CAND_BLOCK]))
            print(f"  row {r} (clen {int(clen[r, 0])}): kept {int(kept.sum())} want {want_n}, {bad} wrongly kept, "
                  f"last kept {bool(kept[last[r]])}, {len(picks)} picks of {kk} ({len(set(picks))} distinct, "
                  f"{dead} not live, {outk} outside kept blocks, {low} below the threshold)")
        ok &= good
    print(f"candidates read_len={read_len}: {'OK' if ok else 'WRONG'}", flush=True)
    return ok


def main():
    args = sys.argv[1:]
    cand = bool(args) and args[0] == "cand"
    lens = [int(a) for a in args[cand:]] or ([32768, 131072] if cand else [4224, 16384, 65536, 131072])
    ok = True
    for n in lens:
        try:
            ok &= check_cand(n) if cand else check(n)
        except Exception:
            traceback.print_exc()
            ok = False
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
