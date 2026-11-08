"""trn3.48xlarge interconnect, from `neuron-ls --topology` on the instance: device 2k <-> 2k+1 direct link, the 8 even
devices all-to-all through one switch, the 8 odd devices through another. LNC=2: 4 logical cores per device.
Attention TP8 group = 8 consecutive ranks (the EP-TP group) = one linked pair; the 8-device runs use devices 0-7."""
import sys
from pathlib import Path
from blogkit import *

HERE = Path(__file__).parent
W, H = 636, 290
X0, PITCH = 22, 74
SW_T = (42, 20)          # top switch y, h
EV = (82, 36)            # even device row y, h
OD = (146, 36)           # odd device row y, h
SW_B = (202, 20)
DW = 62
LINK = PURPLE


def cx(i):
    return X0 + PITCH * i + PITCH / 2


def g(id_, items, kind=None):
    k = f' data-kind="{kind}"' if kind else ''
    return f'<g id="{id_}"{k}>' + ''.join(items) + '</g>'


b = []
# server container
b.append(g('c:server', [rect(8, 8, W - 16, 250, fill=PANEL, stroke=GREY_D, sw=0.9, rx=8, dash='4 3'),
                        text(22, 28, 'trn3.48xlarge', size=11, weight='bold'),
                        text(22 + measure('trn3.48xlarge', 11, True) + 10, 28,
                             '16 Trainium3 devices · 64 logical cores (LNC = 2)', size=7.5, fill=MUTED)]))
# attention TP8 groups: one pale band per linked pair
for i in range(8):
    b.append(g(f'a:tp8_{i}', [rect(X0 + PITCH * i + 2, EV[0] - 6, PITCH - 4, OD[0] + OD[1] - EV[0] + 12,
                                   fill=GREEN_L, rx=5)]))
# switches
sx0, sx1 = X0 + 2, X0 + PITCH * 8 - 2
for sid, (y, h), lab in [('sw_even', SW_T, 'Switch: all-to-all among the 8 even devices (0, 2, 4, … 14)'),
                         ('sw_odd', SW_B, 'Switch: all-to-all among the 8 odd devices (1, 3, 5, … 15)')]:
    b.append(g(f'n:{sid}', [rect(sx0, y, sx1 - sx0, h, fill=ORANGE_L, stroke=ORANGE_M, sw=0.8, rx=3),
                            text((sx0 + sx1) / 2, y + h / 2 + 2.7, lab, size=7.5, anchor='middle')], 'node'))
# links (drawn before devices so the device boxes cover the line ends)
for i in range(8):
    c = cx(i)
    b.append(g(f'e:sw_even-d{2*i}', [line(c, SW_T[0] + SW_T[1], c, EV[0], stroke=LINK, sw=1.2)]))
    b.append(g(f'e:d{2*i}-d{2*i+1}', [line(c, EV[0] + EV[1], c, OD[0], stroke=LINK, sw=1.2)]))
    b.append(g(f'e:d{2*i+1}-sw_odd', [line(c, OD[0] + OD[1], c, SW_B[0], stroke=LINK, sw=1.2)]))
# devices
for i in range(8):
    for d, (y, h) in [(2 * i, EV), (2 * i + 1, OD)]:
        c = cx(i)
        b.append(g(f'n:d{d}', [rect(c - DW / 2, y, DW, h, fill=BG, stroke=INK, sw=0.8, rx=3),
                               text(c, y + 15, f'Trainium3 [{d}]', size=7.5, anchor='middle', weight='bold'),
                               text(c, y + 27, f'cores {4*d}–{4*d+3}', size=6.8, anchor='middle', fill=MUTED)],
                   'node'))
# 8-device configuration bracket
bx0, bx1, by = X0 + 2, X0 + PITCH * 4 - 2, 233
b.append(g('a:8dev', [path(f'M{f(bx0)},{by-5} L{f(bx0)},{by} L{f(bx1)},{by} L{f(bx1)},{by-5}', stroke=INK, sw=0.8,
                           cap='butt', join='miter'),
                      text((bx0 + bx1) / 2, by + 13, 'devices 0–7 (cores 0–31): the 8-device configuration',
                           size=7.5, anchor='middle')]))
# legend
ly = 276
lg = [line(22, ly - 2.6, 44, ly - 2.6, stroke=LINK, sw=1.2), text(50, ly, 'NeuronLink-v4', size=7.5)]
lx = 50 + measure('NeuronLink-v4', 7.5) + 24
lg += [rect(lx, ly - 8, 18, 10, fill=GREEN_L, rx=2),
       text(lx + 24, ly, 'one attention TP8 group: 8 consecutive ranks on a linked device pair', size=7.5)]
b.append(g('legend', lg))

svg = svg_doc(W, H, b)
(HERE / 'out').mkdir(exist_ok=True)
(HERE / 'out/trn3_48xlarge_topology.svg').write_text(svg)
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
r = Renderer(svg, scale=scale)
r.png(svg, HERE / 'out/trn3_48xlarge_topology.png')
r.close()
