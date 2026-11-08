"""Figure 9: serving two high-interactivity sessions on one trn3.48xlarge: the instance split into two
independent 8-device engines (the HBM panel was removed 2026-11-08 at the user's request; see .bak_with_hbm).
See brief.md / plan.md. Run: python3 two_sessions.py [scale]  -> out/two_sessions.{svg,png}"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blogkit import *  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)

W, H = 680, 188
TS, LS, SS = 9.5, 7.5, 6.8           # title, label, small
INK2 = '#3b4552'
LINK = PURPLE
GREEN_D = '#23784a'
RED_D = '#b02a2a'

# panel (a) geometry
CX0, CX1, CY0, CY1 = 52, 628, 8, 180          # server container
HALF = [(64, 332), (348, 616)]                # engine halves
EY0, EY1 = 20, 160                            # engine box
HDR = 26                                      # engine header height
TY0, TY1 = 56, 142                            # green pair tint
EV = (60, 26)                                 # even device row y, h
OD = (112, 26)                                # odd device row y, h
DW = 50


def g(id_, items, kind=None):
    k = f' data-kind="{kind}"' if kind else ''
    return f'<g id="{id_}"{k}>' + ''.join(items) + '</g>'


def person(cx, cy, color=INK2, s=1.0):
    return (circle(cx, cy - 3.6 * s, 2.5 * s, fill=color) +
            f'<path d="M{f(cx - 4.4 * s)},{f(cy + 4.6 * s)} Q{f(cx - 4.4 * s)},{f(cy - 0.4 * s)} {f(cx)},'
            f'{f(cy - 0.4 * s)} Q{f(cx + 4.4 * s)},{f(cy - 0.4 * s)} {f(cx + 4.4 * s)},{f(cy + 4.6 * s)} Z" '
            f'fill="{color}"/>')


def build():
    b = [arrow_defs({'ah': INK2, 'ahr': RED_D})]

    # ---------------- (a) the instance and its two engines ----------------
    b.append(g('c:server', [
        rect(CX0, CY0, CX1 - CX0, CY1 - CY0, fill=PANEL, stroke=GREY_D, sw=0.9, rx=7, dash='4 3')]))

    for e, (x0, x1) in enumerate(HALF):
        n = e + 1
        dev0 = 8 * e
        dash = None
        pitch = (x1 - x0 - 16) / 4
        items = [rect(x0, EY0, x1 - x0, EY1 - EY0, fill=BG, stroke=BLUE, sw=0.9, rx=5, dash=dash),
                 f'<path d="M{f(x0 + 0.45)},{f(EY0 + HDR)} L{f(x0 + 0.45)},{f(EY0 + 5)} '
                 f'Q{f(x0 + 0.45)},{f(EY0 + 0.45)} {f(x0 + 5)},{f(EY0 + 0.45)} L{f(x1 - 5)},{f(EY0 + 0.45)} '
                 f'Q{f(x1 - 0.45)},{f(EY0 + 0.45)} {f(x1 - 0.45)},{f(EY0 + 5)} L{f(x1 - 0.45)},{f(EY0 + HDR)} Z" '
                 f'fill="{BLUE_L}"/>',
                 line(x0, EY0 + HDR, x1, EY0 + HDR, stroke=BLUE, sw=0.6, dash=dash),
                 text(x0 + 8, EY0 + 12, f'vLLM engine {n}', size=TS, weight='bold', fill=BLUE),
                 text(x0 + 8, EY0 + 21.5, f'NEURON_VISIBLE_DEVICES = {dev0}–{dev0 + 7} · TP = 32', size=SS, fill=INK2)]
        b.append(g(f'c:engine{n}', items))
        for k in range(4):
            cxk = x0 + 8 + pitch * k + pitch / 2
            de, do = dev0 + 2 * k, dev0 + 2 * k + 1
            b.append(g(f'a:pair{n}_{k}', [rect(cxk - pitch / 2 + 2, TY0, pitch - 4, TY1 - TY0, fill=GREEN_L, rx=4)]))
            b.append(g(f'e:d{de}-d{do}', [line(cxk, EV[0] + EV[1], cxk, OD[0], stroke=LINK, sw=1.2)]))
            for d, (y, h) in [(de, EV), (do, OD)]:
                b.append(g(f'n:d{d}', [rect(cxk - DW / 2, y, DW, h, fill=BG, stroke=INK, sw=0.8, rx=2.5),
                                       text(cxk, y + 11, f'Trn3 [{d}]', size=LS, anchor='middle', weight='bold'),
                                       text(cxk, y + 21, f'cores {4 * d}–{4 * d + 3}', size=SS, anchor='middle',
                                            fill=MUTED)], 'node'))
            b.append(g(f'a:grp{n}_{k}', [text(cxk, TY1 + 10.5, f'expert group {k + 1}', size=SS, anchor='middle',
                                              fill=GREEN_D)]))

        # the user of this engine, outside the server, on the outer side
        hy = EY0 + HDR / 2
        ux = 26 if e == 0 else W - 26
        ex = x0 if e == 0 else x1
        b.append(g(f'n:user{n}', [person(ux, hy - 1, s=1.25)], 'node'))
        b.append(g(f'a:user{n}_lab', [text(ux, hy + 15, f'user {n}', size=LS, anchor='middle', weight='bold'),
                                      text(ux, hy + 24.5, '~288 tok/s', size=SS, anchor='middle', fill=INK2)]))
        a0, a1 = (ux + 6, ex - 1.5) if e == 0 else (ux - 6, ex + 1.5)
        b.append(f'<g id="e:user{n}-engine{n}" data-from="user{n}" data-to="engine{n}">'
                 f'<path d="M{f(a0)},{f(hy)} L{f(a1)},{f(hy)}" stroke="{INK2}" stroke-width="1" fill="none" '
                 f'marker-start="url(#ah)" marker-end="url(#ah)"/></g>')

    # key for the pair tint and the link, inside the server under the engines
    ky = 172.5
    kx = HALF[0][0]
    key = [rect(kx, ky - 7, 16, 9, fill=GREEN_L, rx=2),
           text(kx + 21, ky, 'linked device pair = one attention TP8 copy + one group of 96 routed experts '
                '(decode MoE, TP8 × EP4)', size=SS, fill=INK2)]
    kx2 = kx + 21 + measure('linked device pair = one attention TP8 copy + one group of 96 routed experts '
                            '(decode MoE, TP8 × EP4)', SS) + 18
    key += [line(kx2, ky - 2.4, kx2 + 18, ky - 2.4, stroke=LINK, sw=1.2),
            text(kx2 + 23, ky, 'NeuronLink-v4', size=SS, fill=INK2)]
    b.append(g('legend', key))

    return svg_doc(W, H, b)


if __name__ == '__main__':
    svg = build()
    (OUT / 'two_sessions.svg').write_text(svg)
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    r = Renderer(svg, scale=scale)
    r.png(svg, OUT / 'two_sessions.png')
    r.close()
    print('ok')
