# SPDX-License-Identifier: Apache-2.0
"""Time the prefill index path of one layer (one 1024-row segment, no all-reduce) on one core pair.

    python bench_index.py [read_len] [mode ...]

modes: kt (build_ktT only), score (+ index_scores_tokens), topk (+ index_topk_rows),
cand (+ index_candidates). Each is compiled, run once, then timed over 3 runs.
"""

import sys
import time

from kharness import run_nki  # noqa: E402 - sets the environment first
import torch

import nki
import nki.isa as nisa
import nki.language as nl

from deepseek_v41.kernels.mk_compress import (
    CAND_BLOCK,
    build_ktT,
    index_candidates,
    index_scores_tokens,
    index_topk_rows,
    _mask_cols,
    MASK_COLS,
)
from deepseek_v41.kernels.mk_common import cdiv
from deepseek_v41.kernels.layer_kernel import idx_chunk

T, NH, KD, TOPK = 1024, 4, 128, 512
CORES = 2


def make(mode):
    @nki.jit
    def k_bench(qi, wi, ktab, clen):
        read_len, kd = ktab.shape
        nh = wi.shape[1]
        pid = nl.program_id(0)
        chunk = min(idx_chunk(read_len), T)
        sc = nl.ndarray((chunk, read_len), dtype=nl.float32, buffer=nl.shared_hbm, name="sc")
        sc16 = nl.ndarray((chunk, read_len), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="sc16")
        out = nl.ndarray((T, TOPK), dtype=nl.int32, buffer=nl.shared_hbm, name="out")
        bs16 = nl.ndarray((chunk, read_len // CAND_BLOCK), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="bs16")
        keep = nl.ndarray((T * (read_len // CAND_BLOCK), 1), dtype=nl.float32, buffer=nl.shared_hbm, name="keep")
        kT = build_ktT(ktab, read_len, kd, pid=pid, name="kT")
        if mode != "kt":
            for c0 in range(0, T, chunk):
                n = min(chunk, T - c0)
                half = -(-n // CORES)
                tok0 = c0 + (0 if pid == 0 else n - half)
                index_scores_tokens(qi, wi, kT, sc, tok0, half, tok0 - c0, nh, read_len, kd)
                nisa.core_barrier(data=sc, cores=(0, 1))
                if mode == "cand":
                    index_candidates(sc, clen, bs16, keep, c0, n, read_len, pid)
                if mode in ("topk", "cand"):
                    index_topk_rows(sc, clen, sc16, out, c0, n, read_len, TOPK, pid)
                if mode == "mask":
                    tiles = [(i * 128, min(128, n - i * 128)) for i in range(cdiv(n, 128)) if i % 2 == pid]
                    for r0, nr in tiles:
                        ci = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
                        nisa.dma_copy(dst=ci[0:nr, :], src=clen[c0 + r0 : c0 + r0 + nr, 0:1])
                        cf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=cf[0:nr, :], src=ci[0:nr, :])
                        for m0 in range(0, read_len, MASK_COLS):
                            _mask_cols(sc, sc16, r0, nr, cf, m0, MASK_COLS)
        return out
    return k_bench


def main():
    read_len = int(sys.argv[1]) if len(sys.argv) > 1 else 131072
    modes = sys.argv[2:] or ["kt", "score", "mask", "topk", "cand"]
    torch.manual_seed(0)
    qi = torch.randn(T, NH * KD).to(torch.bfloat16)
    wi = torch.rand(T, NH)
    ktab = torch.randn(read_len, KD).to(torch.bfloat16)
    clen = torch.full((T, 1), read_len, dtype=torch.int32)
    for mode in modes:
        k = make(mode)
        run_nki(k, qi, wi, ktab, clen, grid=2)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            run_nki(k, qi, wi, ktab, clen, grid=2)
            ts.append(time.perf_counter() - t0)
        print(f"read_len={read_len} mode={mode}: {min(ts) * 1e3:.1f} ms (runs {[round(t * 1e3, 1) for t in ts]})", flush=True)


if __name__ == "__main__":
    main()
