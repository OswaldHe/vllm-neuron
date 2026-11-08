"""Shared helpers for the blog figures: palette, SVG primitives (pt units), batch rendering through the
FigGenie renderer's font injection, and GIF assembly."""
import sys, io, math
from pathlib import Path
from xml.sax.saxutils import escape

SKILL = Path.home() / '.claude/plugins/cache/figgenie/figgenie/0.1.0/skills/figgenie-paper-diagram'
sys.path.insert(0, str(SKILL / 'scripts'))
import _svgkit as K  # noqa: E402

FONT = "Inter, Roboto, Arimo, sans-serif"

# palette (one accent family per meaning, pale tints for fills)
INK = '#1f2933'
MUTED = '#5f6b7a'
FAINT = '#9aa5b1'
GRID = '#e4e7eb'
BG = '#ffffff'
PANEL = '#f5f6f8'

ORANGE = '#e8711a'      # kernels / megakernels (Trainium accent)
ORANGE_L = '#fde3cc'
ORANGE_M = '#f6b27a'
PURPLE = '#7b4fc9'      # agentic optimization
PURPLE_L = '#e9e0fa'
BLUE = '#2f6fd0'        # vLLM backend / host
BLUE_L = '#dbe8fb'
BLUE_M = '#9dbdf0'
GREEN = '#2e9a5c'       # parallelism
GREEN_L = '#d7f0e1'
GREEN_M = '#93d1ad'
RED = '#d23c3c'         # bottleneck / waste
RED_L = '#f9dada'
GREY_L = '#e9ecf0'
GREY_M = '#c3cad3'
GREY_D = '#8a94a3'


def f(x):
    return f'{x:.2f}'.rstrip('0').rstrip('.')


def rect(x, y, w, h, fill='none', stroke='none', sw=0.8, rx=2, op=1.0, dash=None, extra=''):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    o = f' opacity="{f(op)}"' if op < 1 else ''
    return (f'<rect x="{f(x)}" y="{f(y)}" width="{f(max(w, 0))}" height="{f(max(h, 0))}" rx="{f(rx)}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}"{d}{o} {extra}/>')


def text(x, y, s, size=9, fill=INK, anchor='start', weight='normal', op=1.0, italic=False, extra=''):
    o = f' opacity="{f(op)}"' if op < 1 else ''
    st = ' font-style="italic"' if italic else ''
    return (f'<text x="{f(x)}" y="{f(y)}" font-size="{f(size)}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{st}{o} {extra}>{escape(s)}</text>')


def line(x1, y1, x2, y2, stroke=INK, sw=0.8, dash=None, op=1.0, cap='butt', marker=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    o = f' opacity="{f(op)}"' if op < 1 else ''
    m = f' marker-end="url(#{marker})"' if marker else ''
    return (f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{stroke}" '
            f'stroke-width="{f(sw)}" stroke-linecap="{cap}"{d}{o}{m}/>')


def path(d, stroke=INK, sw=0.8, fill='none', dash=None, op=1.0, marker=None, cap='round', join='round'):
    dd = f' stroke-dasharray="{dash}"' if dash else ''
    o = f' opacity="{f(op)}"' if op < 1 else ''
    m = f' marker-end="url(#{marker})"' if marker else ''
    return (f'<path d="{d}" stroke="{stroke}" stroke-width="{f(sw)}" fill="{fill}" stroke-linecap="{cap}" '
            f'stroke-linejoin="{join}"{dd}{o}{m}/>')


def circle(cx, cy, r, fill=INK, stroke='none', sw=0.8, op=1.0):
    o = f' opacity="{f(op)}"' if op < 1 else ''
    return f'<circle cx="{f(cx)}" cy="{f(cy)}" r="{f(r)}" fill="{fill}" stroke="{stroke}" stroke-width="{f(sw)}"{o}/>'


def arrow_defs(colors):
    out = ['<defs>']
    for name, c in colors.items():
        out.append(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" '
                   f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    out.append('</defs>')
    return ''.join(out)


def svg_doc(W, H, body, bg=BG):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'viewBox="0 0 {f(W)} {f(H)}" width="{f(W)}" height="{f(H)}" font-family="{FONT}">'
            f'<rect x="0" y="0" width="{f(W)}" height="{f(H)}" fill="{bg}"/>' + ''.join(body) + '</svg>')


def measure(s, size=9, bold=False):
    """Text width in pt (Inter metrics via the skill's measurer)."""
    import measure_text as M
    r = M.width_of(s, 'inter', size, bold=bold)
    return r[0] if isinstance(r, tuple) else r


class Renderer:
    """One headless Chromium page reused for many frames (fonts injected once)."""

    def __init__(self, sample_svg, scale=2.0, base_dir=None):
        self.css = K._embed_css(sample_svg)
        self.scale = scale
        self.base_dir = base_dir
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch()
        vb = K.parse_viewbox(sample_svg)
        self.page = self.browser.new_page(viewport={'width': int(vb[2]) + 1, 'height': int(vb[3]) + 1},
                                          device_scale_factor=scale)

    def png(self, svg_text, out=None):
        html, W, H = K._html_for(K.svg_for_chromium(svg_text, self.base_dir), self.css)
        self.page.set_content(html)
        self.page.evaluate('document.fonts.ready.then(()=>1)')
        data = self.page.locator('svg').first.screenshot(path=str(out) if out else None)
        return data

    def close(self):
        self.browser.close()
        self._pw.stop()


def make_gif(frames_png_bytes, durations_ms, out, colors=256, width=None):
    from PIL import Image
    ims = []
    for b in frames_png_bytes:
        im = Image.open(io.BytesIO(b)).convert('RGB')
        if width and im.width != width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        ims.append(im)
    # one shared palette from a mosaic of key frames keeps colours stable across frames
    step = max(1, len(ims) // 12)
    keys = ims[::step]
    mos = Image.new('RGB', (ims[0].width, ims[0].height * len(keys)))
    for i, k in enumerate(keys):
        mos.paste(k, (0, i * ims[0].height))
    pal = mos.quantize(colors=colors - 1, method=Image.MEDIANCUT)
    p = pal.getpalette()[:3 * (colors - 1)]
    p = p + [0, 0, 0] * (colors - 1 - len(p) // 3) + [255, 255, 255]   # pure white always in the palette
    pal = Image.new('P', (1, 1))
    pal.putpalette(p)
    import numpy as np
    q = []
    for im in ims:
        qi = im.quantize(palette=pal, dither=Image.NONE)
        a = np.array(qi)
        a[(np.array(im) == 255).all(-1)] = colors - 1      # exact white (PIL's palette lookup is approximate)
        o = Image.fromarray(a, 'P')
        o.putpalette(p)
        q.append(o)
    q[0].save(out, save_all=True, append_images=q[1:], duration=durations_ms, loop=0, optimize=True, disposal=1)
    return out


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(lerp(x, y, t)):02x}' for x, y in zip(a, b))
