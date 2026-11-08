"""Trainium3 device (AWS Neuron docs, 8 NeuronCore-v4) + NKI NeuronCore programming model (engines, SBUF/PSUM,
HBM; NKI Trainium3 architecture guide), side by side with a zoom from one core. Right diagram scaled to the chip's height."""
import base64
from pathlib import Path
from PIL import Image
from blogkit import *

HERE = Path(__file__).parent
L = HERE / 'src_imgs/trainium3_chip.png'          # 862 x 871
Rimg = HERE / 'src_imgs/nki-trn3-arch-2.png'      # 732 x 597


SCALE = 2.0      # output px per SVG unit; bitmaps are pre-resized (LANCZOS) to their exact output pixels


def uri(p, w, h):
    import io
    im = Image.open(p).convert('RGB').resize((round(w * SCALE), round(h * SCALE)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


lw, lh = Image.open(L).size
rw0, rh0 = Image.open(Rimg).size
k = lh / rh0                                      # right diagram scaled to the chip's height: no empty bands
rw, rh = rw0 * k, lh
GAP = 70
W, H = lw + GAP + rw, lh
rx0, ry0 = lw + GAP, 0
core = (446, 253, 754, 386)                       # middle-right NeuronCore-v4 box in the chip image
nc = (rx0 + 26 * k, ry0 + 16 * k, rx0 + 703 * k, ry0 + 419 * k)   # NeuronCore box in the programming-model image

body = [
    f'<image x="0" y="0" width="{lw}" height="{lh}" href="{uri(L, lw, lh)}"/>',
    f'<image x="{f(rx0)}" y="{f(ry0)}" width="{f(rw)}" height="{f(rh)}" href="{uri(Rimg, rw, rh)}"/>',
    line(core[2], core[1], nc[0], nc[1], stroke=GREY_D, sw=2, dash='7 5'),
    line(core[2], core[3], nc[0], nc[3], stroke=GREY_D, sw=2, dash='7 5'),
    rect(core[0] - 2, core[1] - 2, core[2] - core[0] + 4, core[3] - core[1] + 4, stroke=ORANGE, sw=4, rx=8),
    rect(nc[0] - 2, nc[1] - 2, nc[2] - nc[0] + 4, nc[3] - nc[1] + 4, stroke=ORANGE, sw=4, rx=3),
]
svg = svg_doc(W, H, body)
(HERE / 'out').mkdir(exist_ok=True)
r = Renderer(svg, scale=SCALE)
r.png(svg, HERE / 'out/trainium3_neuroncore.png')
r.close()
print(W, H)
