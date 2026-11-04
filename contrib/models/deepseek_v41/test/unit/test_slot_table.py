# SPDX-License-Identifier: Apache-2.0
"""`attn_slot_table` (tuned c1_decode) at long read lengths: the compressed columns of
each row must be `cmp_slots[topk[r, j]]`, -1 where `topk[r, j] < 0`.

    python test_slot_table.py [read_len ...]
"""

import sys
import traceback

from kharness import run_nki  # noqa: E402 - sets the environment first
import torch

import nki
import nki.language as nl

from deepseek_v41.kernels.tuned.c1_decode.mk_common import attn_slot_table

NR, NWIN, WIN, TOPK = 8, 256, 128, 512
KPAD = -(-(WIN + TOPK) // 128) * 128


@nki.jit
def k_slots(twin, wv, cbase, topk, win_slots, cmp_slots):
    read_len = cmp_slots.shape[0]
    sid = nl.ndarray((NR, KPAD), dtype=nl.int32, buffer=nl.shared_hbm, name="sid")
    attn_slot_table(sid, NR, twin, wv, cbase, topk, win_slots, cmp_slots, 1, NWIN, read_len, WIN, TOPK, KPAD)
    return sid


def check(read_len: int) -> bool:
    torch.manual_seed(read_len)
    twin = torch.full((NR, 1), NWIN - 1, dtype=torch.int32)
    wv = torch.zeros((NR, 1), dtype=torch.int32)
    cbase = torch.full((NR, 1), NWIN, dtype=torch.int32)
    topk = torch.stack([torch.randperm(read_len)[:TOPK] for _ in range(NR)]).to(torch.int32)
    topk[0, :100] = -1
    topk[1, :] = torch.arange(read_len - TOPK, read_len, dtype=torch.int32)  # the last columns
    win_slots = torch.arange(NWIN, dtype=torch.int32).reshape(-1, 1)
    cmp_slots = (torch.randperm(read_len * 2)[:read_len] + 7).to(torch.int32).reshape(-1, 1)
    sid = run_nki(k_slots, twin, wv, cbase, topk, win_slots, cmp_slots, grid=2)
    want = torch.where(topk >= 0, cmp_slots.reshape(-1)[topk.clamp_min(0).long()], torch.full_like(topk, -1))
    got = sid[:, WIN : WIN + TOPK]
    bad = (got != want).sum().item()
    print(f"read_len={read_len}: {bad} of {got.numel()} compressed entries wrong "
          f"(e.g. row/col {[(r, c, int(got[r, c]), int(want[r, c])) for r, c in (got != want).nonzero()[:3].tolist()]})",
          flush=True)
    return bad == 0


def main():
    lens = [int(a) for a in sys.argv[1:]] or [4224, 32768, 131072]
    ok = True
    for n in lens:
        try:
            ok &= check(n)
        except Exception:
            traceback.print_exc()
            ok = False
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
