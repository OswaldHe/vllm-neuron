"""Animation: from per-op kernels to tuned multi-layer megakernels (one decode step on Trainium3)."""
from pathlib import Path
from anim import *

OUT = Path(__file__).parent / 'out'
PREV = OUT / 'fusion_states'
PREV.mkdir(parents=True, exist_ok=True)

W, H = 760, 346
TX0, TX1, TY, TH = 112, 736, 100, 22          # panel A track
OY, OH = 176, 30                               # op lane
HB0, HB1 = 266, 286                            # HBM band
DIP_TOP = OY + OH

STEPS = [
    ('Per-op', '① Per-op kernels', 'Every op is its own launch, and every intermediate goes back to HBM.', '~500'),
    ('Layer kernel', '② One megakernel per layer',
     'A layer is one NKI kernel: activations stay in SBUF, the all-reduces run inside the kernel.', '40+'),
    ('Multi-layer', '③ Multi-layer megakernels',
     'Eight layers per launch; the embedding and the greedy head fold into the first and last launch.', '8'),
    ('On-chip stream', '④ Merged launches, on-chip residual stream',
     'Three launches plus the drafter. The residual stream stays in SBUF from one layer to the next.', '4'),
    ('Spills & stalls', '⑤ Fusion alone is not enough',
     'The merged kernel spilled tiles to HBM every layer, and hoisted weight loads slowed the all-reduce.', '4'),
    ('Tuned', '⑥ Tuning the fused kernel',
     'Double-buffered expert weights end the spills; loads wait for the all-reduce, then prefetch the next layer.',
     '4'),
]

OPS = [('mHC in', 'mhc', 38), ('QKV proj', 'attn', 48), ('Indexer', 'attn', 44), ('Sparse attn', 'attn', 52),
       ('O proj', 'attn', 38), ('AR', 'cc', 24), ('mHC', 'mhc', 34), ('Router', 'moe', 40),
       ('Experts', 'moe', 50), ('AR', 'cc', 24)]
KC = {'mhc': (GREEN_L, GREEN_M), 'attn': (BLUE_L, BLUE_M), 'moe': (ORANGE_L, ORANGE_M), 'cc': (GREY_L, GREY_M)}
FRAME = ORANGE


def header(s, i):
    st = {}
    _, title, sub, cnt = STEPS[i]
    st[f'title{i}'] = T(24, 30, title, size=15.5, weight='bold')
    st[f'sub{i}'] = T(24, 49, sub, size=10.5, fill=MUTED)
    st[f'cnt{i}'] = T(736, 33, cnt, size=24, fill=ORANGE, anchor='end', weight='bold')
    st['cntlab'] = T(736, 49, 'launches per decode step', size=9.5, fill=MUTED, anchor='end')
    x = 24
    for k, (pill, *_r) in enumerate(STEPS):
        w = measure(pill, 9, bold=True) + 14
        on = k == i
        st[f'pill{k}'] = R(x, 62, w, 16, fill=INK if on else GREY_L, rx=8, z=1)
        st[f'pilltx{k}'] = T(x + w / 2, 73.3, pill, size=9, fill='#ffffff' if on else MUTED, anchor='middle',
                             weight='bold')
        x += w + 6
    st['divA'] = P([(24, 88), (736, 88)], stroke=GRID, sw=0.8, z=0)
    st['lblA'] = T(24, TY + 15, 'Decode step', size=10.5, weight='bold')
    st['divB'] = P([(24, 136), (736, 136)], stroke=GRID, sw=0.8, z=0)
    st['lblB1'] = T(24, OY + 13, 'On chip', size=10.5, weight='bold')
    st['lblB2'] = T(24, OY + 26, '(SBUF)', size=9.5, fill=MUTED)
    st['hbm'] = R(TX0 - 8, HB0, TX1 - TX0 + 16, HB1 - HB0, fill=GREY_L, rx=3, z=0)
    st['lblH'] = T(24, HB0 + 14, 'HBM', size=10.5, weight='bold')
    # legend
    ly = 322
    st['lg1'] = P([(112, ly - 9), (117, ly - 1), (121, ly - 1), (126, ly - 9)], stroke=RED, sw=1.2, z=4)
    st['lg1t'] = T(132, ly, 'activations through HBM', size=9.5, fill=MUTED)
    x2 = 132 + measure('activations through HBM', 9.5) + 22
    st['lg2'] = R(x2, ly - 10, 14, 11, stroke=FRAME, sw=1.4, rx=2.5)
    st['lg2t'] = T(x2 + 20, ly, 'one launch', size=9.5, fill=MUTED)
    x3 = x2 + 20 + measure('one launch', 9.5) + 22
    st['lg3'] = P([(x3, ly - 4), (x3 + 16, ly - 4)], stroke=BLUE, sw=1.2, dash='3 2', marker='arrB')
    st['lg3t'] = T(x3 + 22, ly, 'weight prefetch', size=9.5, fill=MUTED)
    return st


# ---------------- panel A: the launches of one decode step ----------------
def boxA(st, bid, x, w, kind, label=None, op=1.0):
    fill, stroke = {'layer': (ORANGE_L, ORANGE), 'draft': (PURPLE_L, PURPLE), 'small': (GREY_L, GREY_D)}[kind]
    st[bid] = R(x, TY, w, TH, fill=fill, stroke=stroke, sw=1.0, rx=2.5, op=op, z=2)
    if label:
        st[bid + '_t'] = T(x + w / 2, TY + 14.5, label, size=8.8, fill=INK, anchor='middle', op=op,
                           weight='bold')


def layout_s1():
    gap = 2.4
    wl = (TX1 - TX0 - 6 - 8 - 34 - 43 * gap) / 40
    x = TX0
    pos = {'emb': (x, 6)}; x += 6 + gap
    for l in range(40):
        pos[f'L{l}'] = (x, wl); x += wl + gap
    pos['head'] = (x, 8); x += 8 + gap
    pos['drafter'] = (x, TX1 - x)
    return pos


def layout_s2():
    gap = 5
    big = (TX1 - TX0 - 6 - 6 - 44 - 7 * gap) / 5
    x = TX0
    pos = {'pro': (x, 6)}; x += 6 + gap
    for g in range(5):
        pos[f'G{g}'] = (x, big); x += big + gap
    pos['tail'] = (x, 6); x += 6 + gap
    pos['drafter'] = (x, TX1 - x)
    return pos


def layout_s3():
    gap = 6
    tms = [('F0', 2.15), ('F1', 4.25), ('F2', 4.21), ('drafter', 1.6)]
    avail = TX1 - TX0 - 3 * gap
    tot = sum(t for _, t in tms)
    x = TX0
    pos = {}
    for k, t in tms:
        w = avail * t / tot
        pos[k] = (x, w); x += w + gap
    return pos


G_LABEL = ['L0–7 + embed', 'L8–15', 'L16–23', 'L24–31', 'L32–39 + head']
F_LABEL = {'F0': 'embed + L0–7', 'F1': 'L8–23', 'F2': 'L24–39 + head', 'drafter': 'drafter'}


def panelA(i):
    st = {}
    if i == 0:
        gw = (TX1 - TX0) / 41
        for l in range(41):
            gx = TX0 + l * gw
            for j in range(6):
                st[f'tk{l}_{j}'] = R(gx + 1 + j * (gw - 3.5) / 5, TY, 1.5, TH,
                                     fill=PURPLE if l == 40 else GREY_D, rx=0.3, z=2)
        st['a0_l'] = T(TX0, TY + TH + 12, 'layer 0', size=8.5, fill=MUTED)
        st['a0_r'] = T(TX1, TY + TH + 12, 'drafter', size=8.5, fill=MUTED, anchor='end')
        return st
    if i == 1:
        pos = layout_s1()
        boxA(st, 'emb', *pos['emb'], 'small')
        for l in range(40):
            boxA(st, f'L{l}', *pos[f'L{l}'], 'layer')
        boxA(st, 'head', *pos['head'], 'small')
        boxA(st, 'drafter', *pos['drafter'], 'draft', 'drafter')
        # ghost ticks grow into their layer box
        gw = (TX1 - TX0) / 41
        for l in range(41):
            bx, bw = pos['drafter'] if l == 40 else pos[f'L{l}']
            for j in range(6):
                st[f'tk{l}_{j}'] = R(bx, TY, bw, TH, fill=ORANGE_L, rx=0.3, z=2, op=0)
        st['a1_l'] = T(pos['L0'][0], TY + TH + 12, 'L0', size=8.5, fill=MUTED)
        st['a1_r'] = T(pos['L39'][0] + pos['L39'][1], TY + TH + 12, 'L39', size=8.5, fill=MUTED, anchor='end')
        return st
    if i == 2:
        pos = layout_s2()
        boxA(st, 'pro', *pos['pro'], 'small')
        for g in range(5):
            boxA(st, f'G{g}', *pos[f'G{g}'], 'layer', G_LABEL[g])
        boxA(st, 'tail', *pos['tail'], 'small')
        boxA(st, 'drafter', *pos['drafter'], 'draft', 'drafter')
        for l in range(40):
            gx, gwid = pos[f'G{l // 8}']
            st[f'L{l}'] = R(gx, TY, gwid, TH, fill=ORANGE_L, stroke=ORANGE, sw=1.0, rx=2.5, z=2, op=0)
        st['emb'] = R(*[pos['G0'][0], TY, pos['G0'][1], TH], fill=ORANGE_L, stroke=ORANGE, z=2, op=0)
        st['head'] = R(*[pos['G4'][0], TY, pos['G4'][1], TH], fill=ORANGE_L, stroke=ORANGE, z=2, op=0)
        return st
    pos = layout_s3()
    for k in ('F0', 'F1', 'F2'):
        boxA(st, k, *pos[k], 'layer', F_LABEL[k])
    boxA(st, 'drafter', *pos['drafter'], 'draft', 'drafter + verify')
    tgt = {'pro': 'F0', 'G0': 'F0', 'G1': 'F1', 'G2': 'F1', 'G3': 'F2', 'G4': 'F2', 'tail': 'drafter'}
    for k, v in tgt.items():
        x, w = pos[v]
        st[k] = R(x, TY, w, TH, fill=ORANGE_L if v != 'drafter' else PURPLE_L, stroke=ORANGE, sw=1.0, rx=2.5,
                  z=2, op=0)
    return st


# ---------------- panel B: one layer, zoomed ----------------
def ops_layout(gap_in, gap_layer, ar_w=None):
    out = []
    x = TX0
    for k, (nm, kind, w) in enumerate(OPS):
        if ar_w and k == 5:
            w = ar_w
        out.append(('i', k, x, w)); x += w + gap_in
    x += gap_layer - gap_in
    for k, (nm, kind, w) in enumerate(OPS):
        if x + w > TX1:
            break
        out.append(('n', k, x, w)); x += w + gap_in
    return out


def dip(x0, x1, deep=True):
    mid = (x0 + x1) / 2
    y = HB0 + 2 if deep else DIP_TOP
    return [(x0 - 6, DIP_TOP), (mid - 3, y), (mid + 3, y), (x1 + 6, DIP_TOP)]


def panelB(i):
    st = {}
    gaps = {0: (12, 18), 1: (4, 22), 2: (4, 14), 3: (4, 8), 4: (4, 8), 5: (4, 8)}[i]
    lay = ops_layout(*gaps, ar_w=46 if i == 4 else None)
    for L, k, x, w in lay:
        nm, kind, _ = OPS[k]
        fill, stroke = KC[kind]
        if i == 4 and L == 'i' and k == 5:
            fill, stroke = RED_L, RED
        op = 1.0 if L == 'i' else 0.55
        st[f'op{L}{k}'] = R(x, OY, w, OH, fill=fill, stroke=stroke, sw=0.9, rx=3, z=3, op=op)
        st[f'ot{L}{k}'] = T(x + w / 2, OY + 19, nm, size=8.6, anchor='middle', z=5, op=op)
    # layer captions
    li = [e for e in lay if e[0] == 'i']
    ln = [e for e in lay if e[0] == 'n']
    st['cap_i'] = T(li[0][2], OY - 14, 'layer i', size=9.5, fill=MUTED, weight='bold')
    st['cap_n'] = T(ln[0][2], OY - 14, 'layer i+1', size=9.5, fill=MUTED, weight='bold')
    # launch frames
    fy, fh = OY - 5, OH + 10
    if i == 0:
        for L, k, x, w in lay:
            st[f'fr{L}{k}'] = R(x - 4, fy, w + 8, fh, stroke=FRAME, sw=1.3, rx=4, z=4)
    else:
        def span(es, right=None):
            return es[0][2] - 5, (right if right else es[-1][2] + es[-1][3] + 5)
        if i == 1:
            a0, a1 = span(li); b0, b1 = span(ln, right=TX1 + 30)
            st['frLi'] = R(a0, fy, a1 - a0, fh, stroke=FRAME, sw=1.6, rx=5, z=4)
            st['frLn'] = R(b0, fy, b1 - b0, fh, stroke=FRAME, sw=1.6, rx=5, z=4)
            for L, k, x, w in lay:      # per-op frames merge into the layer frame
                g0, g1 = (a0, a1) if L == 'i' else (b0, b1)
                st[f'fr{L}{k}'] = R(g0, fy, g1 - g0, fh, stroke=FRAME, sw=1.6, rx=5, z=4, op=0)
            st['frlab1'] = T(TX1, OY - 14, '1 launch = 1 layer', size=9.5, fill=ORANGE, anchor='end',
                             weight='bold')
        else:
            a0 = li[0][2] - 5
            st['frAll'] = R(a0, fy, TX1 + 40 - a0, fh, stroke=FRAME, sw=1.6, rx=5, z=4)
            st['frLi'] = R(a0, fy, TX1 + 40 - a0, fh, stroke=FRAME, sw=1.6, rx=5, z=4, op=0)
            st['frLn'] = R(a0, fy, TX1 + 40 - a0, fh, stroke=FRAME, sw=1.6, rx=5, z=4, op=0)
            lab = '1 launch = 8 layers' if i == 2 else '1 launch = 16 layers'
            st[f'frlab{2 if i == 2 else 3}'] = T(TX1, OY - 14, lab, size=9.5, fill=ORANGE, anchor='end',
                                                 weight='bold')
    # activation round trips through HBM
    seq = li + ln
    for a, b in zip(seq, seq[1:]):
        x0 = a[2] + a[3]; x1 = b[2]
        boundary = a[0] != b[0]
        deep = (i == 0) or (boundary and i in (1, 2))
        st[f'dip{a[0]}{a[1]}'] = P(dip(x0, x1, deep), stroke=RED, sw=1.15, op=1.0 if deep else 0, marker='arrR', z=2)
    xb = (li[-1][2] + li[-1][3] + ln[0][2]) / 2
    if i in (1, 2):
        st['bnote'] = T(xb + 8, HB0 - 8, 'residual stream to HBM and back', size=9, fill=RED, anchor='start')
    if i >= 3:
        st['snote'] = T(xb, OY + OH + 26, 'stream stays in SBUF', size=9, fill=GREEN, anchor='middle',
                        weight='bold')
        st['sarrow'] = P([(li[-1][2] + li[-1][3] - 8, OY + OH + 3), (xb - 4, OY + OH + 12), (xb + 4, OY + OH + 12),
                          (ln[0][2] + 8, OY + OH + 3)], stroke=GREEN, sw=1.4, marker='arrG', z=4)
    # spills (state 4) / prefetch after the all-reduce (state 5)
    byk = {k: (x, w) for L, k, x, w in li}
    for sid, (ka, kb) in (('sp1', (1, 2)), ('sp2', (7, 8))):
        x0 = byk[ka][0] + byk[ka][1]; x1 = byk[kb][0]
        on = i == 4
        st[sid] = P(dip(x0, x1, on), stroke=RED, sw=1.3, dash='3 2', op=1.0 if on else 0, marker='arrR', z=2)
    if i == 4:
        st['spn'] = T(byk[1][0] + byk[1][1] + 12, HB0 - 8, 'spill / reload', size=9, fill=RED)
        st['spn2'] = T(byk[7][0] + byk[7][1] - 6, HB0 - 8, 'spill / reload', size=9, fill=RED, anchor='end')
        xa = byk[5][0] + byk[5][1] / 2
        st['hoist'] = P([(xa + 10, HB0), (xa + 10, OY + OH + 8)], stroke=BLUE, sw=1.2, dash='3 2',
                        marker='arrB', z=4)
        st['hoistt'] = T(xa + 4, OY + OH + 30, 'next loads hoisted', size=9, fill=BLUE, anchor='end')
        st['hoistt2'] = T(xa + 4, OY + OH + 42, 'into the AR window', size=9, fill=BLUE, anchor='end')
    if i == 5:
        xr = byk[7][0] + byk[7][1] / 2
        nq = [e for e in ln if e[1] == 1][0]
        xq = nq[2] + nq[3] / 2
        st['pref'] = P([(xr, HB0), (xr, OY + OH + 42), (xq, OY + OH + 42), (xq, OY + OH + 3)], stroke=BLUE, sw=1.3,
                       dash='3 2', marker='arrB', z=4)
        st['preft'] = T(xr - 6, HB0 - 6, "next layer's weights, right after the AR", size=9, fill=BLUE,
                        anchor='end')
        st['dbuf'] = T(byk[8][0] + byk[8][1] / 2, OY - 14, 'double-buffered', size=9, fill=ORANGE,
                       anchor='middle', weight='bold')
    return st


def state(i):
    st = header({}, i)
    st.update(panelA(i))
    st.update(panelB(i))
    return st


DEFS = arrow_defs({'arrR': RED, 'arrB': BLUE, 'arrG': GREEN})

PB_Y, PB_H, PB_GAP = 335, 4, 6        # progress bar: one segment per step, the current one fills during its hold
HOLD_MS = 8000


def progress(i, frac):
    st = {}
    n = len(STEPS)
    seg = (TX1 - 24 - (n - 1) * PB_GAP) / n
    for k in range(n):
        x = 24 + k * (seg + PB_GAP)
        f = 1.0 if k < i else (frac if k == i else 0.0)
        st[f'pgb{k}'] = R(x, PB_Y, seg, PB_H, fill=GREY_L, rx=2, z=1)
        st[f'pgf{k}'] = R(x, PB_Y, seg * f, PB_H, fill=ORANGE, rx=2, z=2, op=1.0 if seg * f >= 0.5 else 0.0)
    return st

if __name__ == '__main__':
    states = [state(i) for i in range(6)]
    n = render_sequence(states, [HOLD_MS] * len(states), W, H, OUT / "fusion.gif", defs=DEFS, scale=2.5,
                        preview_dir=PREV, progress=progress, progress_frames=32)
    print('frames', n)
