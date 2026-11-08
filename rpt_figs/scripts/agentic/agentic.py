"""Figure 10: the agentic optimization flow (outer loop with a human; every stage its own small loop).
See brief.md / plan.md. Run: python3 agentic.py [scale]  -> out/agentic_flow.{svg,png}"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blogkit import *  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)

W, H = 700, 274
HUM = (36, 110)     # human lane y0, y1
AGT = (116, 236)    # agent lane y0, y1
HB = (44, 102)      # human boxes y0, y1
AB = (128, 224)     # agent boxes y0, y1
C = {1: (32, 122), 2: (136, 224), 3: (238, 330), 4: (344, 434), 5: (448, 588), 6: (602, 692)}
TOP = 18            # outer-loop line
INK2 = '#3b4552'
AGENT_BAND = '#f7f3fd'
TS, BS, ES = 8.5, 6.5, 7.0   # title, badge, example sizes


def g(id_, items, kind=None):
    k = f' data-kind="{kind}"' if kind else ''
    return f'<g id="{id_}"{k}>' + ''.join(items) + '</g>'


def defs():
    return ('<defs>'
            f'<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5.5" markerHeight="5.5" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker>'
            f'<marker id="ahb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="3.6" markerHeight="3.6" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker>'
            f'<marker id="ahp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{PURPLE}"/></marker>'
            f'<marker id="ahg" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="3.4" markerHeight="3.4" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{PURPLE}"/></marker>'
            '</defs>')


def badge(cx, cy, n, color):
    return circle(cx, cy, 5.6, fill=color) + text(cx, cy + 2.3, str(n), size=BS, fill='white', anchor='middle',
                                                  weight='bold')


def loop_glyph(cx, cy, r=4.8, color=PURPLE):
    a0, a1 = math.radians(-50), math.radians(215)
    x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
    x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
    return (f'<path d="M{f(x0)},{f(y0)} A{f(r)},{f(r)} 0 1 1 {f(x1)},{f(y1)}" fill="none" stroke="{color}" '
            f'stroke-width="1.2" stroke-linecap="round" marker-end="url(#ahg)"/>')


def person(cx, cy, color=INK2, s=1.0):
    return (circle(cx, cy - 3.6 * s, 2.5 * s, fill=color) +
            f'<path d="M{f(cx - 4.4 * s)},{f(cy + 4.6 * s)} Q{f(cx - 4.4 * s)},{f(cy - 0.4 * s)} {f(cx)},'
            f'{f(cy - 0.4 * s)} Q{f(cx + 4.4 * s)},{f(cy - 0.4 * s)} {f(cx + 4.4 * s)},{f(cy + 4.6 * s)} Z" '
            f'fill="{color}"/>')


def titled_box(id_, col, ys, n, title_lines, examples, loop=None, human=False):
    x0, x1 = C[col]
    y0, y1 = ys
    stroke = GREY_D if human else PURPLE
    items = [rect(x0, y0, x1 - x0, y1 - y0, fill=BG, stroke=stroke, sw=0.9, rx=4)]
    tx = x0 + 8
    ty = y0 + 14
    if n is not None:
        items.append(badge(x0 + 11, y0 + 11, n, GREY_D if human else PURPLE))
        tx = x0 + 21
    for i, t in enumerate(title_lines):
        items.append(text(tx, ty + i * 10.5, t, size=TS, weight='bold'))
    if human:
        items.append(person(x1 - 11, y0 + 12))
    ey = ty + len(title_lines) * 10.5 + 5
    for i, e in enumerate(examples):
        items.append(text(x0 + 8, ey + i * 9.2, e, size=ES, fill=MUTED))
    if loop:
        ly = y1 - 11
        items.append(loop_glyph(x0 + 13, ly - 2.4))
        items.append(text(x0 + 22, ly, loop, size=ES, fill=PURPLE, weight='bold'))
    return g(f'n:{id_}', items, 'node')


def arrow(id_, pts, sw=1.1, marker='ah', color=INK2, dash=None, frm='', to=''):
    d = 'M' + ' L'.join(f'{f(x)},{f(y)}' for x, y in pts)
    return (f'<g id="e:{id_}" data-from="{frm}" data-to="{to}">' +
            path(d, stroke=color, sw=sw, marker=marker, dash=dash, cap='butt', join='round') + '</g>')


def build():
    b = [defs()]
    # lanes
    b.append(g('l:human', [rect(6, HUM[0], W - 12, HUM[1] - HUM[0], fill=PANEL, rx=5),
                           text(21.6, (HUM[0] + HUM[1]) / 2, 'Human', size=TS, weight='bold', fill=GREY_D,
                                anchor='middle',
                                extra=f'transform="rotate(-90 21.6 {(HUM[0] + HUM[1]) / 2})"')]))
    b.append(g('l:agent', [rect(6, AGT[0], W - 12, AGT[1] - AGT[0], fill=AGENT_BAND, rx=5),
                           text(21.6, (AGT[0] + AGT[1]) / 2, 'Agent', size=TS, weight='bold', fill=PURPLE,
                                anchor='middle',
                                extra=f'transform="rotate(-90 21.6 {(AGT[0] + AGT[1]) / 2})"')]))

    # inputs (human lane, c1)
    x0, x1 = C[1]
    b.append(g('n:inputs', [rect(x0, HB[0], x1 - x0, HB[1] - HB[0], fill=GREY_L, stroke=GREY_M, sw=0.8, rx=4),
                            text(x0 + 8, HB[0] + 14, 'Model spec +', size=TS, weight='bold'),
                            text(x0 + 8, HB[0] + 24.5, 'correctness', size=TS, weight='bold'),
                            text(x0 + 8, HB[0] + 35, 'constraint', size=TS, weight='bold'),
                            text(x0 + 8, HB[0] + 49, 'needle, trivia, tests', size=ES, fill=MUTED)], 'node'))
    # agent lane stages
    b.append(titled_box('correctness', 1, AB, 1, ['Correctness', 'loop'],
                        ['per-op port passes', 'the verifier'], loop='implement, verify'))
    b.append(titled_box('profile', 2, AB, 2, ['Profile'],
                        ['e.g. 40 launches', 'take 66 ms of a', '97 ms decode step'], loop='capture, analyze'))
    b.append(titled_box('workspaces', 4, AB, 5, ['Subkernel', 'workspaces'],
                        ['golden I/O traced', 'on all 64 ranks'], loop='trace, validate'))
    b.append(titled_box('integrate', 6, AB, 7, ['Integrate', '& tune'],
                        ['e.g. TPOT', '14.0 → 11.5 ms'], loop='integrate, bench'))
    # human lane stages
    b.append(titled_box('constraints', 4, HB, 4, ['New', 'constraints'],
                        ['e.g. TPOT < 5 ms,', 'pure NKI, bit-exact'], human=True))
    b.append(titled_box('review', 6, HB, 8, ['Review'],
                        ['accept the round', 'or send it back'], human=True))

    # (3) grilling loop, spanning both lanes
    x0, x1 = C[3]
    cx = (x0 + x1) / 2
    it = [rect(x0, HB[0], x1 - x0, AB[1] - HB[0], fill=BG, stroke=GREY_D, sw=0.9, rx=4),
          badge(x0 + 11, HB[0] + 11, 3, INK2),
          text(x0 + 21, HB[0] + 14, 'Grilling loop', size=TS, weight='bold'),
          person(cx, HB[0] + 36, s=1.15),
          text(cx, HB[0] + 52, 'human', size=ES, anchor='middle', fill=INK2, weight='bold')]
    ya, yb = HB[0] + 58, AB[0] + 8
    it.append(path(f'M{f(cx - 7)},{f(ya)} L{f(cx - 7)},{f(yb)}', stroke=INK2, sw=1.0, marker='ah', cap='butt'))
    it.append(path(f'M{f(cx + 7)},{f(yb)} L{f(cx + 7)},{f(ya)}', stroke=PURPLE, sw=1.0, marker='ahp', cap='butt'))
    it.append(text(cx - 11, (ya + yb) / 2 + 2.5, 'questions', size=ES, anchor='end', fill=MUTED))
    it.append(text(cx + 11, (ya + yb) / 2 + 2.5, 'answers', size=ES, anchor='start', fill=MUTED))
    it.append(text(cx, AB[0] + 18, 'agent', size=ES, anchor='middle', fill=PURPLE, weight='bold'))
    for i, e in enumerate(['until the bottleneck', 'is understood']):
        it.append(text(cx, AB[0] + 33 + i * 9.2, e, size=ES, anchor='middle', fill=INK2))
    for i, e in enumerate(['e.g. why does the', 'all-reduce wait?', '→ skewed expert picks']):
        it.append(text(x0 + 8, AB[0] + 58 + i * 9.2, e, size=ES, fill=MUTED))
    b.append(g('n:grilling', it, 'node'))

    # (6) RSI loops, one subkernel at a time
    x0, x1 = C[5]
    y0, y1 = AB
    it = [rect(x0, y0, x1 - x0, y1 - y0, fill=BG, stroke=PURPLE, sw=1.3, rx=4),
          badge(x0 + 11, y0 + 11, 6, PURPLE),
          text(x0 + 21, y0 + 14, 'RSI loops, one subkernel', size=TS, weight='bold'),
          text(x0 + 21, y0 + 24.5, 'at a time', size=TS, weight='bold')]
    kw, kh, gap = 29, 24, 11
    ky = y0 + 33
    kx = x0 + 9
    for i, lab in enumerate(['K1', 'K2', 'K3']):
        xx = kx + i * (kw + gap)
        it.append(rect(xx, ky, kw, kh, fill=PURPLE_L, stroke=PURPLE, sw=0.8, rx=3))
        it.append(text(xx + 8.5, ky + 15.5, lab, size=ES, weight='bold', anchor='middle'))
        it.append(loop_glyph(xx + 21.5, ky + kh / 2, r=4.0))
        if i < 2:
            it.append(path(f'M{f(xx + kw + 1)},{f(ky + kh / 2)} L{f(xx + kw + gap - 1)},{f(ky + kh / 2)}',
                           stroke=PURPLE, sw=1.0, dash='2.4 1.8', marker='ahp', cap='butt'))
    xe = kx + 3 * (kw + gap) - gap
    it.append(text(xe + 4, ky + 15.5, '…', size=TS, weight='bold', fill=PURPLE))
    it.append(text(x0 + 8, y0 + 70, 'e.g. L24–L31: 6.5 → 4.6 ms', size=ES, fill=MUTED))
    ly = y1 - 11
    it.append(loop_glyph(x0 + 13, ly - 2.4))
    it.append(text(x0 + 22, ly, 'edit, verify, measure, review', size=ES, fill=PURPLE, weight='bold'))
    b.append(g('n:rsi', it, 'node'))

    # flows
    am = (AB[0] + AB[1]) / 2
    hm = (HB[0] + HB[1]) / 2
    b.append(arrow('inputs-correctness', [(77, HB[1]), (77, AB[0])], frm='inputs', to='correctness'))
    b.append(arrow('correctness-profile', [(C[1][1], am), (C[2][0], am)], frm='correctness', to='profile'))
    b.append(arrow('profile-grilling', [(C[2][1], am), (C[3][0], am)], frm='profile', to='grilling'))
    b.append(arrow('grilling-constraints', [(C[3][1], hm), (C[4][0], hm)], frm='grilling', to='constraints'))
    b.append(arrow('constraints-workspaces', [(389, HB[1]), (389, AB[0])], frm='constraints', to='workspaces'))
    b.append(arrow('workspaces-rsi', [(C[4][1], am), (C[5][0], am)], frm='workspaces', to='rsi'))
    b.append(arrow('rsi-integrate', [(C[5][1], am), (C[6][0], am)], frm='rsi', to='integrate'))
    b.append(arrow('integrate-review', [(647, AB[0]), (647, HB[1])], frm='integrate', to='review'))
    # outer loop: review -> top -> down into profile
    px = (C[2][0] + C[2][1]) / 2
    b.append(arrow('review-profile', [(647, HB[0]), (647, TOP), (px, TOP), (px, AB[0])], sw=2.0, marker='ahb',
                   frm='review', to='profile'))
    b.append(g('a:resume', [rect((px + 647) / 2 - 42, TOP - 6.5, 84, 13, fill=BG, rx=2),
                            text((px + 647) / 2, TOP + 3, 'resume the loop', size=TS, anchor='middle',
                                 weight='bold', fill=INK2)]))

    # legend
    ly = 262
    lg = [loop_glyph(30, ly - 2.6), text(39, ly, 'agent loop', size=ES)]
    lx = 39 + measure('agent loop', ES) + 22
    lg += [person(lx + 4, ly - 2.6, color=INK2, s=0.9), text(lx + 13, ly, 'human step', size=ES)]
    lx += 13 + measure('human step', ES) + 22
    lg += [path(f'M{f(lx)},{f(ly - 2.6)} L{f(lx + 22)},{f(ly - 2.6)}', stroke=PURPLE, sw=1.0, dash='2.4 1.8',
                marker='ahp', cap='butt'),
           text(lx + 28, ly, 'data flywheel: each subkernel starts from the previous one\'s optimizations', size=ES)]
    b.append(g('legend', lg))
    return svg_doc(W, H, b)


if __name__ == '__main__':
    svg = build()
    (OUT / 'agentic_flow.svg').write_text(svg)
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    r = Renderer(svg, scale=scale)
    r.png(svg, OUT / 'agentic_flow.png')
    r.close()
    print('ok')
