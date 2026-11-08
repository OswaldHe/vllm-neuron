"""Figure: optimization trajectory (bs1 decode speed per user on Trainium3, Oct 22 -> Nov 5)."""
from datetime import datetime
from pathlib import Path
from blogkit import *

OUT = Path(__file__).parent / 'out'
OUT.mkdir(exist_ok=True)

# in512 bs1 TPOT p50 (ms) from test/performance/history + round-5 / 8-device commits and the staging README
# (date, tpot_ms, category, label, (label dx, dy) in pt from the point, anchor)
PTS = [
    ('2026-10-22 19:40', 99.4, 'start', None, None, None),
    ('2026-10-23 12:20', 73.1, 'kernel', 'NKI megakernels', (150, 272), 'end'),
    ('2026-10-23 20:27', 31.6, 'kernel', 'Native MX FP8/FP4', (160, 250), 'start'),
    ('2026-10-24 18:59', 31.9, None, None, None, None),
    ('2026-10-25 21:31', 31.8, None, None, None, None),
    ('2026-10-27 03:29', 28.2, 'agentic', 'Agentic Opt kernels', (275, 322), 'end'),
    ('2026-10-27 06:02', 28.3, None, None, None, None),
    ('2026-10-27 09:11', 17.7, 'kernel', 'Multi-layer megakernels', (250, 226), 'middle'),
    ('2026-10-27 13:29', 14.8, 'host', 'One RPC per step', (300, 316), 'start'),
    ('2026-10-28 01:03', 14.0, 'host', 'Direct NEFF launch', (326, 288), 'start'),
    ('2026-10-30 02:30', 11.5, 'agentic', 'Agentic Opt fused launches', (420, 176), 'end'),
    ('2026-10-30 13:34', 9.1, 'host', 'Async scheduler', (436, 248), 'start'),
    ('2026-10-30 17:40', 8.8, None, None, None, None),
    ('2026-10-31 18:23', 7.8, 'agentic', 'Agentic Opt drafter', (510, 150), 'end'),
    ('2026-11-01 18:06', 4.90, 'parallel', 'TP8×EP8 MoE', (508, 120), 'end'),
    ('2026-11-02 19:18', 4.22, 'kernel', 'On-chip blocks', (552, 88), 'end'),
    ('2026-11-02 21:20', 4.17, None, None, None, None),
    ('2026-11-02 23:22', 3.77, 'kernel', 'Fast rsqrt', (558, 58), 'end'),
    ('2026-11-03 05:35', 3.67, 'parallel', None, None, None),
    ('2026-11-03 16:35', 3.645, None, None, None, None),
    ('2026-11-05 10:29', 3.47, 'agentic', 'Agentic Opt launch', (664, 34), 'end'),
]
CAT = {
    'start': (GREY_D, None),
    'kernel': (ORANGE, 'Kernels & fusion'),
    'agentic': (PURPLE, 'Agentic Opt'),
    'host': (BLUE, 'vLLM backend'),
    'parallel': (GREEN, 'Parallelism'),
}
T0 = datetime(2026, 10, 22, 0, 0)
T1 = datetime(2026, 11, 5, 23, 59)
EIGHT_DEV = datetime(2026, 11, 3, 5, 35)

W, H = 720, 372
X0, X1, Y0, Y1 = 62, 700, 20, 330
YMAX = 320


def X(dt):
    return X0 + (dt - T0).total_seconds() / (T1 - T0).total_seconds() * (X1 - X0)


def Y(v):
    return Y1 - v / YMAX * (Y1 - Y0)


def build():
    b = []
    # 8-device region
    xe = X(EIGHT_DEV)
    b.append(rect(xe, Y0, X1 - xe, Y1 - Y0, fill=GREEN_L, rx=0, op=0.55))
    b.append(text(xe + (X1 - xe) / 2, Y1 - 10, '8 devices', size=10.5, fill=GREEN, anchor='middle', weight='bold'))
    # grid + y ticks
    for v in range(0, 301, 50):
        b.append(line(X0, Y(v), X1, Y(v), stroke=GRID, sw=0.7))
        b.append(text(X0 - 8, Y(v) + 3.5, str(v), size=10, fill=MUTED, anchor='end'))
    b.append(line(X0, Y1, X1, Y1, stroke=GREY_M, sw=0.9))
    # x ticks
    for d in [datetime(2026, 10, 22), datetime(2026, 10, 24), datetime(2026, 10, 26), datetime(2026, 10, 28),
              datetime(2026, 10, 30), datetime(2026, 11, 1), datetime(2026, 11, 3), datetime(2026, 11, 5)]:
        x = X(d)
        b.append(line(x, Y1, x, Y1 + 4, stroke=GREY_M, sw=0.9))
        b.append(text(x, Y1 + 16, d.strftime('%b %-d'), size=10, fill=MUTED, anchor='middle'))
    b.append(text(18, (Y0 + Y1) / 2, 'Output tokens/s per user', size=11, fill=INK, anchor='middle', weight='bold',
                  extra=f'transform="rotate(-90 18 {(Y0 + Y1) / 2})"'))
    # step line + area
    pts = [(X(datetime.strptime(d, '%Y-%m-%d %H:%M')), Y(1000 / t)) for d, t, *_ in PTS]
    xend = X(datetime(2026, 11, 5, 20, 0))
    dpath = f'M{f(pts[0][0])},{f(pts[0][1])}'
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        dpath += f' H{f(xb)} V{f(yb)}'
    area = dpath + f' V{f(Y1)} H{f(pts[0][0])} Z'
    b.append(path(dpath, stroke='#3b4552', sw=2.0))
    # leaders + labels first (under markers)
    for (d, t, cat, lab, off, anc), (x, y) in zip(PTS, pts):
        if not lab:
            continue
        c = CAT[cat][0]
        lx, ly = off
        w = measure(lab, 10.5, bold=True)
        x0l = {'start': lx, 'middle': lx - w / 2, 'end': lx - w}[anc]
        ex = min(max(x, x0l + 4), x0l + w - 4)
        ey = ly + 3 if y > ly else ly - 10.5
        b.append(line(x, y, ex, ey, stroke=c, sw=0.9, op=0.85))
        b.append(text(lx, ly, lab, size=10.5, fill=c, anchor=anc, weight='bold'))
    # markers
    for (d, t, cat, lab, off, anc), (x, y) in zip(PTS, pts):
        if cat is None:
            continue
        else:
            b.append(circle(x, y, 4.6, fill=CAT[cat][0], stroke='white', sw=1.4))
    # end values
    x0, y0 = pts[0]
    b.append(text(x0 + 2, y0 - 10, '10', size=13, fill=INK, anchor='middle', weight='bold'))
    xl, yl = pts[-1]
    b.append(text(xl + 9, yl + 4.5, '288', size=13, fill=INK, anchor='start', weight='bold'))
    # legend
    lx, ly = X0 + 14, Y0 + 14
    for key in ('kernel', 'agentic', 'host', 'parallel'):
        c, name = CAT[key]
        b.append(circle(lx + 4, ly - 3.5, 4.6, fill=c, stroke='white', sw=1.4))
        b.append(text(lx + 13, ly, name, size=10.5, fill=INK))
        lx += 22 + measure(name, 10.5)
    return svg_doc(W, H, b)


if __name__ == '__main__':
    svg = build()
    p = OUT / 'trajectory.svg'
    p.write_text(svg)
    r = Renderer(svg, scale=4.0)
    r.png(svg, OUT / 'trajectory.png')
    r.close()
    print('ok')
