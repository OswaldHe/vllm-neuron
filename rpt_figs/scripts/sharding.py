"""Figure 8: one transformer block as a left-to-right graph, prefill (top) vs batch-1 decode (bottom),
each op coloured by how its weight is split over the 64 ranks. Run: python3 sharding.py [scale]"""
import sys
from pathlib import Path
from blogkit import *

OUT = Path(__file__).parent / 'out'
OUT.mkdir(exist_ok=True)

SN, ST, SH, SS = 9.0, 8.0, 10.5, 8.5     # op name, op tag, row title, row subtitle
BH = 46                                   # op box height
PH = 15                                   # collective pill height
GW, GH = 40, 8                            # weight glyph
MW, MG = 7.0, 1.6                         # expert mini-matrix width / gap (8 of them)
INK2 = '#3b4552'
HW = GREY_D                               # residual highway colour

CAT = {   # how the op's weight is split over the 64 ranks
    'rep': dict(fill='#f1f3f6', stroke=GREY_D, ink='#566170'),
    'grp': dict(fill=BLUE_L, stroke=BLUE, ink=BLUE),
    'all': dict(fill=GREEN_L, stroke=GREEN, ink='#23794a'),
}

# column -> (name, {phase: (tag, category, glyph, changed)})
OPS = {
    'lookup': ('n-gram lookup', {'P': ('split by row', 'all', 'rows64', False),
                                 'D': ('split by row', 'all', 'rows64', False)}),
    'eproj': ('Engram proj', {'P': ('replicated', 'rep', 'full', False),
                              'D': ('TP8, 8 copies', 'grp', 'tp8', True)}),
    'hc1': ('mHC + norm', {'P': ('replicated', 'rep', 'full', False),
                           'D': ('replicated', 'rep', 'full', False)}),
    'attn': ('Attention', {'P': ('TP8, 8 copies', 'grp', 'tp8', False),
                           'D': ('TP8, 8 copies', 'grp', 'tp8', False)}),
    'hc2': ('mHC + norm', {'P': ('replicated', 'rep', 'full', False),
                           'D': ('replicated', 'rep', 'full', False)}),
    'router': ('Router', {'P': ('replicated', 'rep', 'full', False),
                          'D': ('replicated', 'rep', 'full', False)}),
    'routed': ('Routed experts', {'P': ('EP64', 'all', 'ep_whole', False),
                                  'D': ('TP8 × EP8', 'all', 'ep_slice', True)}),
    'shared': ('Shared expert', {'P': ('replicated', 'rep', 'full', False),
                                 'D': ('split by column', 'all', 'cols64', True)}),
}
PILLS = {'ar_e': 'AR 64', 'ag_e': 'AG 8', 'ar_a': 'AR 8', 'ar_m': 'AR 64'}
ROWS = [
    ('P', 'Prefill and batched decode', 'many rows per step: whole experts on each rank stay busy'),
    ('D', 'Batch-1 decode', '6 rows per step: every expert is split by column so all 64 ranks share the work'),
]


def glyph_w(mode):
    return 8 * MW + 7 * MG if mode.startswith('ep_') else GW


def box_w(col):
    name, ph = OPS[col]
    w = measure(name, SN, bold=True)
    for tag, cat, gm, ch in ph.values():
        w = max(w, measure(tag, ST), glyph_w(gm))
    return w + 12


def pill_w(s):
    return measure(s, 7.5, bold=True) + 12


def glyph(cx, y, mode, c):
    """The op's weight; the filled part is what one rank holds."""
    b = []
    if mode.startswith('ep_'):
        x0 = cx - glyph_w(mode) / 2
        for k in range(8):
            x = x0 + k * (MW + MG)
            b.append(rect(x, y, MW, GH, fill='white', stroke=c, sw=0.6, rx=0.6))
            if mode == 'ep_whole' and k == 0:
                b.append(rect(x, y, MW, GH, fill=c, rx=0.6))
            if mode == 'ep_slice':
                b.append(rect(x, y, 1.7, GH, fill=c, rx=0.2))
        return b
    x0 = cx - GW / 2
    b.append(rect(x0, y, GW, GH, fill='white', stroke=c, sw=0.6, rx=0.8))
    if mode == 'full':
        b.append(rect(x0, y, GW, GH, fill=c, rx=0.8))
    elif mode == 'tp8':
        for k in range(1, 8):
            b.append(line(x0 + k * GW / 8, y + 0.6, x0 + k * GW / 8, y + GH - 0.6, stroke=c, sw=0.35, op=0.45))
        b.append(rect(x0, y, GW / 8, GH, fill=c, rx=0.5))
    elif mode == 'cols64':
        b.append(rect(x0, y, 1.3, GH, fill=c, rx=0.2))
    elif mode == 'rows64':
        b.append(rect(x0, y, GW, 1.6, fill=c, rx=0.2))
    return b


def op_box(x, ym, col, phase, w=None):
    name, ph = OPS[col]
    tag, cat, gm, changed = ph[phase]
    c = CAT[cat]
    w = w or box_w(col)
    y = ym - BH / 2
    b = []
    if changed:
        b.append(rect(x - 2.6, y - 2.6, w + 5.2, BH + 5.2, stroke=ORANGE, sw=1.5, rx=5))
    b.append(rect(x, y, w, BH, fill=c['fill'], stroke=c['stroke'], sw=0.9, rx=3))
    b.append(text(x + w / 2, y + 14, name, size=SN, weight='bold', anchor='middle', fill=INK))
    b += glyph(x + w / 2, y + 20, gm, c['stroke'])
    b.append(text(x + w / 2, y + 39.5, tag, size=ST, anchor='middle', fill=c['ink']))
    return b, w


def pill(x, ym, s, changed=False):
    w = pill_w(s)
    b = []
    if changed:
        b.append(rect(x - 2.4, ym - PH / 2 - 2.4, w + 4.8, PH + 4.8, stroke=ORANGE, sw=1.5, rx=PH / 2 + 2.4))
    b.append(rect(x, ym - PH / 2, w, PH, fill='white', stroke=INK2, sw=0.9, rx=PH / 2))
    b.append(text(x + w / 2, ym + 2.7, s, size=7.5, weight='bold', anchor='middle', fill=INK2))
    return b, w


def plus(cx, cy):
    return [circle(cx, cy, 5.5, fill='white', stroke=INK2, sw=0.9),
            line(cx - 3, cy, cx + 3, cy, stroke=INK2, sw=0.9), line(cx, cy - 3, cx, cy + 3, stroke=INK2, sw=0.9)]


def arr(pts, sw=0.9, c=INK2, marker='ah'):
    d = 'M' + ' L'.join(f'{f(x)},{f(y)}' for x, y in pts)
    return path(d, stroke=c, sw=sw, marker=marker, cap='butt')


def row(phase, top, title, sub, BL, BR):
    b = [rect(BL, top, BR - BL, ROWH, fill=PANEL, rx=6)]
    b.append(text(BL + 12, top + 17, title, size=SH, weight='bold', fill=INK))
    yh = top + 31                       # residual highway
    ym = yh + 62                        # main line of the sublayers
    yu, yd = ym - 28, ym + 28           # routed / shared rows
    nodes, edges = [], []
    # ---- Engram (layers 1 and 14)
    x = BL + 26
    lx = x
    bb, w = op_box(x, ym, 'lookup', phase); nodes += bb
    edges.append(arr([(BL + 8, ym), (x - 1, ym)]))
    x += w + 10
    bb, w = pill(x, ym, PILLS['ar_e']); nodes += bb
    edges.append(arr([(x - 10, ym), (x - 1, ym)]))
    x += w + 10
    bb, w = op_box(x, ym, 'eproj', phase); nodes += bb
    edges.append(arr([(x - 10, ym), (x - 1, ym)]))
    x += w
    agw = pill_w(PILLS['ag_e'])
    if phase == 'D':
        bb, _ = pill(x + 12, ym, PILLS['ag_e'], changed=True); nodes += bb
        edges.append(arr([(x, ym), (x + 11, ym)]))
        edges.append(line(x + 12 + agw, ym, x + 24 + agw, ym, stroke=INK2, sw=0.9))
    else:
        edges.append(line(x, ym, x + 24 + agw, ym, stroke=INK2, sw=0.9))
    x += 24 + agw
    cont = (lx - 9, ym - BH / 2 - 17, x + 2 - (lx - 9), BH + 26)
    edges.insert(0, rect(*cont, stroke=GREY_D, sw=0.8, dash='3 2', rx=5, fill='white', op=1.0))
    edges.insert(1, text(lx - 3, ym - BH / 2 - 6, 'Engram (layers 1 and 14)', size=7.5, fill=MUTED, italic=True))
    pe = x + 12
    edges.append(arr([(x, ym), (pe, ym), (pe, yh + 6.5)]))
    nodes += plus(pe, yh)
    # ---- attention sublayer
    xd = pe + 16
    x = xd + 14
    edges.append(arr([(xd, yh), (xd, ym), (x - 1, ym)]))
    bb, w = op_box(x, ym, 'hc1', phase); nodes += bb
    x += w + 10
    edges.append(arr([(x - 10, ym), (x - 1, ym)]))
    bb, w = op_box(x, ym, 'attn', phase); nodes += bb
    x += w + 10
    edges.append(arr([(x - 10, ym), (x - 1, ym)]))
    bb, w = pill(x, ym, PILLS['ar_a']); nodes += bb
    x += w
    pa = x + 12
    edges.append(arr([(x, ym), (pa, ym), (pa, yh + 6.5)]))
    nodes += plus(pa, yh)
    # ---- MoE sublayer
    xd = pa + 16
    x = xd + 14
    edges.append(arr([(xd, yh), (xd, ym), (x - 1, ym)]))
    bb, w = op_box(x, ym, 'hc2', phase); nodes += bb
    x += w
    xf = x + 10
    xr = xf + 10
    bb, wr = op_box(xr, yu, 'router', phase); nodes += bb
    xe = xr + wr + 12
    we = ws = max(box_w('routed'), box_w('shared'))
    bb, _ = op_box(xe, yu, 'routed', phase, we); nodes += bb
    bb, _ = op_box(xe, yd, 'shared', phase, ws); nodes += bb
    edges.append(line(x, ym, xf, ym, stroke=INK2, sw=0.9))
    edges.append(arr([(xf, ym), (xf, yu), (xr - 1, yu)]))
    edges.append(arr([(xf, ym), (xf, yd), (xe - 1, yd)]))
    edges.append(arr([(xr + wr, yu), (xe - 1, yu)]))
    edges.append(circle(xf, ym, 1.6, fill=INK2))
    xj = xe + we + 10
    edges.append(path(f'M{f(xe + we)},{f(yu)} L{f(xj)},{f(yu)} L{f(xj)},{f(yd)} L{f(xe + ws)},{f(yd)}',
                      stroke=INK2, sw=0.9))
    x = xj + 12
    edges.append(arr([(xj, ym), (x - 1, ym)]))
    bb, w = pill(x, ym, PILLS['ar_m']); nodes += bb
    x += w
    pm = x + 12
    edges.append(arr([(x, ym), (pm, ym), (pm, yh + 6.5)]))
    nodes += plus(pm, yh)
    # ---- residual highway (drawn first, under the nodes)
    hw = [line(BL + 8, yh, pe - 5.5, yh, stroke=HW, sw=2.4),
          line(pe + 5.5, yh, pa - 5.5, yh, stroke=HW, sw=2.4),
          line(pa + 5.5, yh, pm - 5.5, yh, stroke=HW, sw=2.4),
          arr([(pm + 5.5, yh), (BR - 10, yh)], sw=2.4, c=HW, marker='ahg')]
    hw.append(text(BR - 14, yh - 5, 'next block', size=7.5, fill=MUTED, anchor='end'))
    return b + hw + edges + nodes, pm + 5.5


ROWH = 156


def legend(y, BL):
    b = []
    x = BL + 4
    items = [('rep', 'every rank holds all of it'), ('grp', 'split over an 8-rank attention group, 8 copies'),
             ('all', 'split over all 64 ranks')]
    for cat, s in items:
        c = CAT[cat]
        b.append(rect(x, y - 8, 14, 10, fill=c['fill'], stroke=c['stroke'], sw=0.9, rx=2))
        b.append(text(x + 19, y, s, size=8, fill=INK))
        x += 19 + measure(s, 8) + 18
    y2 = y + 17
    x = BL + 4
    b += glyph(x + GW / 2, y2 - 7.5, 'tp8', INK2)
    b.append(text(x + GW + 5, y2, 'weight; the filled part is what one rank holds', size=8, fill=INK))
    x += GW + 5 + measure('weight; the filled part is what one rank holds', 8) + 18
    b.append(rect(x, y2 - 9, 15, 11, stroke=ORANGE, sw=1.5, rx=3))
    b.append(text(x + 20, y2, 'layout changed for decode', size=8, fill=INK))
    x += 20 + measure('layout changed for decode', 8) + 18
    bb, w = pill(x, y2 - 3.5, 'AR n')
    b += bb
    b.append(text(x + w + 5, y2, 'all-reduce / all-gather over n ranks', size=8, fill=INK))
    x += w + 5 + measure('all-reduce / all-gather over n ranks', 8)
    return b, x


def build():
    BL = 8
    # width: lay out once to find the right end
    _, right = row('D', 0, '', '', BL, 2000)
    W = right + 16 + measure('next block', 7.5) + 22
    BR = W - BL
    body = [arrow_defs({'ah': INK2, 'ahg': HW})]
    y = 8
    for phase, title, sub in ROWS:
        bb, _ = row(phase, y, title, sub, BL, BR)
        body += bb
        y += ROWH + 10
    lg, lx = legend(y + 12, BL)
    body += lg
    H = y + 38
    assert lx < W, (lx, W)
    return svg_doc(W, H, body), W


if __name__ == '__main__':
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    svg, W = build()
    (OUT / 'sharding.svg').write_text(svg)
    r = Renderer(svg, scale=scale)
    r.png(svg, OUT / 'sharding.png')
    r.close()
    print('ok W =', round(W, 1))
