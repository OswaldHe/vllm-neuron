"""Figure: TTFT trajectory (in1024 bs1 TTFT p50, same runs and dates as traj.py).
Sources: test/performance/bench_report.{perop,mk1,mk2,json,r2opt,opt3..opt12}.json (in1024_bs1 ttft_p50_ms),
/mnt/nvme/ds41_r4/val/{r5m,r5n,r5r}/bench_bs1.json, deepseek_v41_8d bench_report.json at 83d1d76 / e8c6c8b,
deepseek_v41_8d_trn3/README.md at 0a32285 (staging, 244 ms)."""
from datetime import datetime
from pathlib import Path
from blogkit import *
from traj import CAT, T0, T1, EIGHT_DEV

OUT = Path(__file__).parent / 'out'
OUT.mkdir(exist_ok=True)

# (date, ttft_ms, category, label, (label x, y) in pt, anchor)
PTS = [
    ('2026-10-22 19:40', 1701, 'start', None, None, None),
    ('2026-10-23 12:20', 1157, 'kernel', 'NKI megakernels', (148, 96), 'start'),
    ('2026-10-23 20:27', 1064, 'kernel', 'Native MX FP8/FP4', (168, 128), 'start'),
    ('2026-10-24 18:59', 1061, None, None, None, None),
    ('2026-10-25 21:31', 760, 'agentic', 'Agentic Opt prefill kernel', (258, 180), 'start'),
    ('2026-10-27 03:29', 535, 'agentic', 'Agentic Opt kernels', (312, 222), 'start'),
    ('2026-10-27 06:02', 300, 'kernel', 'Prefill-specialized kernels', (318, 314), 'start'),
    ('2026-10-27 09:11', 299, None, None, None, None),
    ('2026-10-27 13:29', 300, None, None, None, None),
    ('2026-10-28 01:03', 297, None, None, None, None),
    ('2026-10-30 02:30', 232, 'kernel', 'Batched prefill indexer', (440, 262), 'start'),
    ('2026-10-30 13:34', 231, None, None, None, None),
    ('2026-10-30 17:40', 232, None, None, None, None),
    ('2026-10-31 18:23', 232, None, None, None, None),
    ('2026-11-01 18:06', 232, None, None, None, None),
    ('2026-11-02 19:18', 231, None, None, None, None),
    ('2026-11-02 21:20', 231, None, None, None, None),
    ('2026-11-02 23:22', 230, None, None, None, None),
    ('2026-11-03 05:35', 244, None, None, None, None),
    ('2026-11-03 16:35', 245, None, None, None, None),
    ('2026-11-05 10:29', 244, None, None, None, None),
]

W, H = 720, 372
X0, X1, Y0, Y1 = 62, 700, 20, 330
YMAX = 1800


def X(dt):
    return X0 + (dt - T0).total_seconds() / (T1 - T0).total_seconds() * (X1 - X0)


def Y(v):
    return Y1 - v / YMAX * (Y1 - Y0)


def build():
    b = []
    xe = X(EIGHT_DEV)
    b.append(rect(xe, Y0, X1 - xe, Y1 - Y0, fill=GREEN_L, rx=0, op=0.55))
    b.append(text(xe + (X1 - xe) / 2, Y1 - 10, '8 devices', size=10.5, fill=GREEN, anchor='middle', weight='bold'))
    for v in range(0, 1601, 400):
        b.append(line(X0, Y(v), X1, Y(v), stroke=GRID, sw=0.7))
        b.append(text(X0 - 8, Y(v) + 3.5, f'{v:,}', size=10, fill=MUTED, anchor='end'))
    b.append(line(X0, Y1, X1, Y1, stroke=GREY_M, sw=0.9))
    for d in [datetime(2026, 10, 22), datetime(2026, 10, 24), datetime(2026, 10, 26), datetime(2026, 10, 28),
              datetime(2026, 10, 30), datetime(2026, 11, 1), datetime(2026, 11, 3), datetime(2026, 11, 5)]:
        x = X(d)
        b.append(line(x, Y1, x, Y1 + 4, stroke=GREY_M, sw=0.9))
        b.append(text(x, Y1 + 16, d.strftime('%b %-d'), size=10, fill=MUTED, anchor='middle'))
    b.append(text(14, (Y0 + Y1) / 2, 'Time to first token (ms)', size=11, fill=INK, anchor='middle', weight='bold',
                  extra=f'transform="rotate(-90 14 {(Y0 + Y1) / 2})"'))
    pts = [(X(datetime.strptime(d, '%Y-%m-%d %H:%M')), Y(t)) for d, t, *_ in PTS]
    dpath = f'M{f(pts[0][0])},{f(pts[0][1])}'
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        dpath += f' H{f(xb)} V{f(yb)}'
    b.append(path(dpath, stroke='#3b4552', sw=2.0))
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
    for (d, t, cat, lab, off, anc), (x, y) in zip(PTS, pts):
        if cat is not None:
            b.append(circle(x, y, 4.6, fill=CAT[cat][0], stroke='white', sw=1.4))
    x0, y0 = pts[0]
    b.append(text(x0 + 2, y0 - 10, '1,701', size=13, fill=INK, anchor='middle', weight='bold'))
    xl, yl = pts[-1]
    b.append(text(xl + 9, yl - 7, '244', size=13, fill=INK, anchor='start', weight='bold'))
    # legend (top right: the curve starts top left)
    keys = ('kernel', 'agentic')
    tot = sum(22 + measure(CAT[k][1], 10.5) for k in keys) - 22 + 13
    lx, ly = X1 - 10 - tot, Y0 + 14
    for key in keys:
        c, name = CAT[key]
        b.append(circle(lx + 4, ly - 3.5, 4.6, fill=c, stroke='white', sw=1.4))
        b.append(text(lx + 13, ly, name, size=10.5, fill=INK))
        lx += 22 + measure(name, 10.5)
    return svg_doc(W, H, b)


if __name__ == '__main__':
    import sys
    svg = build()
    (OUT / 'ttft_trajectory.svg').write_text(svg)
    r = Renderer(svg, scale=float(sys.argv[1]) if len(sys.argv) > 1 else 4.0)
    r.png(svg, OUT / 'ttft_trajectory.png')
    r.close()
    print('ok')
