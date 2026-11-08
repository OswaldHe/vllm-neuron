"""The agent-expert heterogeneous system: a human expert and a self-improving agent joined by three links
(① grilling both ways, ② constraints to the agent, ③ feedback to the human). See brief.md.
Run: python3 agent_expert.py [scale]  -> out/agent_expert.{svg,png}; the blog copy is rpt_figs/agent_expert_system.png"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from blogkit import *  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)

W, H = 500, 228
INK2 = '#3b4552'
TS, LS, SS = 9.5, 8.0, 6.8            # titles, labels, small
HX0, HX1 = 20, 190                    # human expert box
AX0, AX1 = 310, 480                   # agent box
BY0, BY1 = 20, 216                    # both boxes
LINKS = [86, 136, 186]                # link y (centre)


def g(id_, items, kind=None, extra=''):
    k = f' data-kind="{kind}"' if kind else ''
    return f'<g id="{id_}"{k}{extra}>' + ''.join(items) + '</g>'


def person(cx, cy, color=INK2, s=1.0):
    return (circle(cx, cy - 3.6 * s, 2.5 * s, fill=color) +
            f'<path d="M{f(cx - 4.4 * s)},{f(cy + 4.6 * s)} Q{f(cx - 4.4 * s)},{f(cy - 0.4 * s)} {f(cx)},'
            f'{f(cy - 0.4 * s)} Q{f(cx + 4.4 * s)},{f(cy - 0.4 * s)} {f(cx + 4.4 * s)},{f(cy + 4.6 * s)} Z" '
            f'fill="{color}"/>')


def loop_glyph(cx, cy, r=5.2, color=PURPLE):
    a0, a1 = math.radians(-50), math.radians(215)
    x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
    x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
    return (f'<path d="M{f(x0)},{f(y0)} A{f(r)},{f(r)} 0 1 1 {f(x1)},{f(y1)}" fill="none" stroke="{color}" '
            f'stroke-width="1.3" stroke-linecap="round" marker-end="url(#ahloop)"/>')


def badge(cx, cy, n, color=INK2):
    return circle(cx, cy, 6.0, fill=color) + text(cx, cy + 2.5, str(n), size=7, fill='white', anchor='middle',
                                                  weight='bold')


def cross(x, y, r=2.4, c=GREY_D, sw=1.1):
    return (line(x - r, y - r, x + r, y + r, stroke=c, sw=sw, cap='round') +
            line(x - r, y + r, x + r, y - r, stroke=c, sw=sw, cap='round'))


def arrow(id_, x0, x1, y, color, frm, to, marker):
    return (f'<g id="{id_}" data-from="{frm}" data-to="{to}">'
            f'<path d="M{f(x0)},{f(y)} L{f(x1)},{f(y)}" stroke="{color}" stroke-width="1.4" fill="none" '
            f'marker-end="url(#{marker})"/></g>')


def build():
    b = [arrow_defs({'ahg': INK2, 'ahp': PURPLE}),
         '<defs><marker id="ahloop" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="3.4" markerHeight="3.4" '
         f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{PURPLE}"/></marker></defs>']
    # the system
    b.append(g('c:system', [rect(8, 8, W - 16, H - 16, fill=PANEL, stroke=GREY_D, sw=0.9, rx=7, dash='4 3')]))

    # ---- human expert ----
    hb = [rect(HX0, BY0, HX1 - HX0, BY1 - BY0, fill=BG, stroke=INK2, sw=1.0, rx=6),
          person(HX0 + 16, BY0 + 15, s=1.15),
          text(HX0 + 28, BY0 + 18.5, 'Human expert', size=TS, weight='bold', fill=INK2)]
    caps = [('Holistic view', 'the whole system, across layers'),
            ('Expert knowledge', 'shortcuts the exploration'),
            ('Decision making', 'trade-offs, what to optimize next'),
            ('Analysis of complex tasks', 'profiles, root causes, open problems')]
    cy0, ch, gap = BY0 + 32, 34, 6
    for i, (t, ex) in enumerate(caps):
        y = cy0 + i * (ch + gap)
        hb += [rect(HX0 + 8, y, HX1 - HX0 - 16, ch, fill=GREY_L, stroke=GREY_M, sw=0.7, rx=3),
               text(HX0 + 16, y + 14, t, size=LS, weight='bold', fill=INK),
               text(HX0 + 16, y + 26.5, ex, size=SS, fill=MUTED)]
    b.append(g('n:human', hb, 'node'))

    # ---- self-improving agent ----
    ab = [rect(AX0, BY0, AX1 - AX0, BY1 - BY0, fill=BG, stroke=PURPLE, sw=1.0, rx=6),
          loop_glyph(AX0 + 16, BY0 + 15),
          text(AX0 + 28, BY0 + 18.5, 'Self-improving agent', size=TS, weight='bold', fill=PURPLE)]
    # the capability: hill-climbing on a clear objective (schematic)
    px0, px1, py0, py1 = AX0 + 8, AX1 - 8, BY0 + 32, BY1 - 10
    ab += [rect(px0, py0, px1 - px0, py1 - py0, fill=PURPLE_L, stroke='none', rx=3, op=0.55),
           text(px0 + 8, py0 + 15.5, 'Hill-climbing on a clear objective', size=LS, weight='bold', fill=INK)]
    cx0, cx1, cy_top, cy_bot = px0 + 22, px1 - 12, py0 + 26, py1 - 30
    ab += [line(cx0, cy_bot, cx1, cy_bot, stroke=GREY_D, sw=0.7),
           line(cx0, cy_bot, cx0, cy_top, stroke=GREY_D, sw=0.7),
           text(cx0 - 5, (cy_top + cy_bot) / 2, 'objective', size=SS, fill=MUTED, anchor='middle',
                extra=f'transform="rotate(-90 {f(cx0 - 5)} {f((cy_top + cy_bot) / 2)})"'),
           text(cx1, cy_bot + 9.5, 'iterations', size=SS, fill=MUTED, anchor='end')]
    # kept edits climb; rejected tries sit below the line
    steps = [(0, 0.06, 'k'), (1, 0.16, 'k'), (2, 0.10, 'r'), (3, 0.30, 'k'), (4, 0.42, 'k'), (5, 0.33, 'r'),
             (6, 0.55, 'k'), (7, 0.50, 'r'), (8, 0.66, 'k'), (9, 0.75, 'k'), (10, 0.71, 'r'), (11, 0.86, 'k')]
    n = len(steps)

    def SX(i):
        return cx0 + 8 + i * (cx1 - cx0 - 14) / (n - 1)

    def SY(v):
        return cy_bot - v * (cy_bot - cy_top - 4)

    best = None
    d = ''
    for i, v, k in steps:
        if k == 'k':
            if best is None:
                d = f'M{f(SX(i))},{f(SY(v))}'
            else:
                d += f' H{f(SX(i))} V{f(SY(v))}'
            best = v
    ab.append(path(d, stroke=INK2, sw=1.4))
    for i, v, k in steps:
        ab.append(circle(SX(i), SY(v), 2.6, fill=PURPLE, stroke='white', sw=0.8) if k == 'k' else cross(SX(i), SY(v)))
    ab.append(text((px0 + px1) / 2, py1 - 8, 'edit → verify → measure → keep or revert', size=SS, fill=INK2,
                   anchor='middle'))
    b.append(g('n:agent', ab, 'node'))

    # ---- interconnect ----
    lx0, lx1 = HX1 + 2, AX0 - 2
    mid = (HX1 + AX0) / 2
    b.append(g('a:bus', [text(mid, BY0 + 17, 'Interconnect', size=SS, fill=MUTED, anchor='middle')]))
    links = [
        (1, 'Grilling', 'sync the idea in both directions', 'both'),
        (2, 'Constraints', 'set up a constrained optimization', 'h2a'),
        (3, 'Feedback', 'results of the optimization', 'a2h'),
    ]
    for (n_, title, sub, dirn), y in zip(links, LINKS):
        tw = measure(title, LS, True)
        tx = mid - (tw + 16) / 2
        it = [badge(tx + 6, y - 12.5, n_, color=INK2),
              text(tx + 16, y - 9.6, title, size=LS, weight='bold', fill=INK)]
        it.append(text(mid, y + 14.5, sub, size=SS, fill=MUTED, anchor='middle'))
        b.append(g(f'a:link{n_}', it))
        if dirn == 'both':
            b.append(arrow('e:grill_h2a', lx0, lx1, y - 2.2, INK2, 'human', 'agent', 'ahg'))
            b.append(arrow('e:grill_a2h', lx1, lx0, y + 2.2, PURPLE, 'agent', 'human', 'ahp'))
        elif dirn == 'h2a':
            b.append(arrow('e:constraints', lx0, lx1, y, INK2, 'human', 'agent', 'ahg'))
        else:
            b.append(arrow('e:feedback', lx1, lx0, y, PURPLE, 'agent', 'human', 'ahp'))
    return svg_doc(W, H, b)


if __name__ == '__main__':
    svg = build()
    (OUT / 'agent_expert.svg').write_text(svg)
    r = Renderer(svg, scale=float(sys.argv[1]) if len(sys.argv) > 1 else 6.0)
    r.png(svg, OUT / 'agent_expert.png')
    r.close()
    print('ok')
