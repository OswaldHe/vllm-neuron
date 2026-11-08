"""Figure 11: the whole-model AutoHelix loop on Trainium 2 (trn2.48xlarge, 64 ranks, Oct 22-23): batch-1 TPOT and
TTFT at 512 tokens per iteration. Kept iterations from the gate lines in the commit messages of origin/deepseek_v41
(290b794 it1, ad54902 it2, eaadc48 it5); iteration 8's numbers from its recovery commit bbd1f76 (rejected on
ttft_p50_ms_in8192_bs4, as were 3, 6 and 7); iteration 4 rejected (no record). Run: python3 ah_trn2.py [scale]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blogkit import *  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)

W, H = 720, 304
INK2 = '#3b4552'
LS, TS, VS = 10, 11, 12.5            # axis / label, panel title, value
STATUS = {0: 'start', 1: 'kept', 2: 'kept', 3: 'rej', 4: 'rej', 5: 'kept', 6: 'rej', 7: 'rej', 8: 'rej8'}
PANELS = [
    # id, title, x0, x1, ymax, grid step, kept values {iter: v}, iteration 8 (measured, rejected), fmt
    ('tpot', 'Decode TPOT (ms)', 58, 338, 120, 40, {0: 103.474, 1: 102.489, 2: 99.344, 5: 90.693}, 86.3, '{:.1f}'),
    ('ttft', 'Time to first token (ms)', 428, 708, 1000, 250, {0: 859.615, 1: 837.225, 2: 817.747, 5: 729.369},
     598.3, '{:,.0f}'),
]
Y0, Y1 = 40, 222


def g(id_, items, kind=None):
    k = f' data-kind="{kind}"' if kind else ''
    return f'<g id="{id_}"{k}>' + ''.join(items) + '</g>'


def cross(x, y, r=3.4, c=GREY_D, sw=1.4):
    return (line(x - r, y - r, x + r, y + r, stroke=c, sw=sw, cap='round') +
            line(x - r, y + r, x + r, y - r, stroke=c, sw=sw, cap='round'))


def build():
    b = []
    for pid, title, x0, x1, ymax, step, kept, v8, fmt in PANELS:
        def X(i):
            return x0 + 14 + i * (x1 - x0 - 28) / 8

        def Y(v):
            return Y1 - v / ymax * (Y1 - Y0)

        it = [text(x0, Y0 - 16, title, size=TS, weight='bold')]
        for v in range(0, ymax + 1, step):
            it.append(line(x0, Y(v), x1, Y(v), stroke=GRID, sw=0.7))
            it.append(text(x0 - 7, Y(v) + 3.5, f'{v:,}', size=LS, fill=MUTED, anchor='end'))
        it.append(line(x0, Y1, x1, Y1, stroke=GREY_M, sw=0.9))
        for i in range(9):
            it.append(line(X(i), Y1, X(i), Y1 + 4, stroke=GREY_M, sw=0.9))
            it.append(text(X(i), Y1 + 16, 'start' if i == 0 else str(i), size=LS, fill=MUTED, anchor='middle'))
            # status row: kept = purple dot, rejected = grey cross
            st = STATUS[i]
            if st == 'kept':
                it.append(circle(X(i), Y1 + 29, 3.6, fill=PURPLE))
            elif st in ('rej', 'rej8'):
                it.append(cross(X(i), Y1 + 29, r=2.8, sw=1.2))
        it.append(text((x0 + x1) / 2, Y1 + 47, 'AutoHelix iteration', size=LS, fill=INK2, anchor='middle'))
        b.append(g(f'a:{pid}_axes', it))

        # the model the loop kept: a step line through the kept iterations, held through the rejected ones
        ks = sorted(kept)
        d = f'M{f(X(0))},{f(Y(kept[0]))}'
        for i in ks[1:]:
            d += f' H{f(X(i))} V{f(Y(kept[i]))}'
        d += f' H{f(X(8))}'
        b.append(g(f'e:{pid}_best', [path(d, stroke=INK2, sw=2.0)]))
        pts = [circle(X(i), Y(kept[i]), 4.6, fill=PURPLE if i else GREY_D, stroke='white', sw=1.4) for i in ks]
        b.append(g(f'n:{pid}_kept', pts, 'node'))
        b.append(g(f'a:{pid}_v0', [text(X(0) + 2, Y(kept[0]) - 10, fmt.format(kept[0]), size=VS, weight='bold',
                                        anchor='start')]))
        b.append(g(f'a:{pid}_v5', [text(X(5) - 8, Y(kept[5]) + 17, fmt.format(kept[5]), size=VS, weight='bold',
                                        anchor='end')]))
        pct = (1 - kept[5] / kept[0]) * 100
        b.append(g(f'a:{pid}_pct', [text((x0 + x1) / 2, Y1 - 24, f'−{pct:.0f}% in 8 iterations, about 34 hours',
                                         size=LS, fill=INK2, anchor='middle')]))

    # legend, centred under the panels
    lx, ly = 0, H - 9
    lg = [circle(lx, ly - 3.5, 3.6, fill=PURPLE), text(lx + 8, ly, 'kept', size=LS, fill=INK2)]
    lx += 8 + measure('kept', LS) + 16
    lg += [cross(lx, ly - 3.5, r=2.8, sw=1.2), text(lx + 8, ly, 'rejected', size=LS, fill=INK2)]
    lx += 8 + measure('rejected', LS) + 16
    lg += [line(lx - 4, ly - 3.5, lx + 14, ly - 3.5, stroke=INK2, sw=2.0),
           text(lx + 20, ly, 'model the loop kept', size=LS, fill=INK2)]
    tot = (8 + measure('kept', LS) + 16 + 8 + measure('rejected', LS) + 16 + 20 + measure('model the loop kept', LS)
           + 4 + 4)
    dx = (W - tot) / 2 + 4
    b.append(f'<g id="legend" transform="translate({f(dx)},0)">' + ''.join(lg) + '</g>')
    return svg_doc(W, H, b)


if __name__ == '__main__':
    svg = build()
    (OUT / 'autohelix_trn2.svg').write_text(svg)
    r = Renderer(svg, scale=float(sys.argv[1]) if len(sys.argv) > 1 else 4.0)
    r.png(svg, OUT / 'autohelix_trn2.png')
    r.close()
    print('ok')
