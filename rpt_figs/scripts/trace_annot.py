"""Figure: the decode-step device trace (rpt_figs/trace.png) annotated with the four causes of the GPU gap."""
import shutil
from pathlib import Path
from blogkit import *

HERE = Path(__file__).parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
SRC = HERE.parent / 'trace.png'  # rpt_figs/trace.png, the Neuron Explorer screenshot
shutil.copy(SRC, OUT / 'trace_src.png')

IW, IH = 2180, 1145
LEG = 170
W, H = IW, IH + LEG
C1, C2, C3, C4 = '#ffcf3a', '#ff5c8a', '#45d0ff', '#9bf06a'


def badge(cx, cy, n, c):
    return (circle(cx, cy, 25, fill=c, stroke='#111', sw=3) +
            text(cx, cy + 10.5, str(n), size=30, fill='#111', anchor='middle', weight='bold'))


def build():
    b = [f'<image href="trace_src.png" x="0" y="0" width="{IW}" height="{IH}"/>']
    sw = 6
    # 1: DMA queues (top) and DMA throughput (bottom)
    b.append(rect(140, 27, 2034, 96, stroke=C1, sw=sw, rx=8))
    b.append(rect(140, 1062, 2034, 72, stroke=C1, sw=sw, rx=8))
    # 2: collective cores
    b.append(rect(140, 570, 2034, 131, stroke=C2, sw=sw, rx=8))
    # 3: throttling
    b.append(rect(140, 705, 2034, 60, stroke=C3, sw=sw, rx=8))
    # 4: host gaps between launches
    for x0, x1 in [(521, 541), (1227, 1247), (1901, 1924)]:
        b.append(rect(x0 - 4, 24, x1 - x0 + 8, 1112, stroke=C4, sw=5, rx=6, dash='18 10'))
    # badges (on the left edge of each band, over the row-name column's margin)
    b.append(badge(140, 75, 1, C1))
    b.append(badge(140, 1098, 1, C1))
    b.append(badge(140, 635, 2, C2))
    b.append(badge(140, 735, 3, C3))
    for xc in (531, 1237, 1912):
        b.append(badge(xc, 160, 4, C4))
    # legend strip
    y0 = IH
    b.append(rect(0, y0, W, LEG, fill='#ffffff', rx=0))
    items = [(1, C1, 'DMA stays busy through the decode step'),
             (2, C2, 'Many collectives and barriers: stall points'),
             (3, C3, 'Throttling: intermittent low utilization'),
             (4, C4, 'Host gaps between launches')]
    for i, (n, c, s) in enumerate(items):
        col, row = i % 2, i // 2
        x = 60 + col * 1080
        y = y0 + 58 + row * 70
        b.append(circle(x, y - 11, 25, fill=c, stroke='#111', sw=3))
        b.append(text(x, y - 0.5, str(n), size=30, fill='#111', anchor='middle', weight='bold'))
        b.append(text(x + 44, y, s, size=34, fill=INK))
    return svg_doc(W, H, b, bg='#1e1e1e')


if __name__ == '__main__':
    svg = build()
    p = OUT / 'trace_annotated.svg'
    p.write_text(svg)
    r = Renderer(svg, scale=2.0, base_dir=OUT)
    r.png(svg, OUT / 'trace_annotated.png')
    r.close()
    print('ok')
