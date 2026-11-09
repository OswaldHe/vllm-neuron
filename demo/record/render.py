#!/usr/bin/env python3
"""Render a record.py capture (JSON lines of tmux pane screens) to a GIF.

usage: render.py REC.jsonl OUT.gif [--font-size 15] [--max-gap 3.0] [--start T] [--end T]
                 [--still OUT.png --still-at T]
"""

import argparse
import json
import re
import unicodedata

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_UI = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_UI_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BG = (24, 26, 33)
FG = (204, 208, 216)
CHROME = (40, 43, 52)
TAB_BAR = (32, 35, 43)
TAB_ACTIVE = (24, 26, 33)
TAB_TEXT = (150, 156, 170)
TAB_TEXT_ACTIVE = (225, 228, 235)
ACCENT = (97, 175, 239)
CURSOR = (220, 223, 228)

ANSI16 = [
    (40, 44, 52), (224, 108, 117), (152, 195, 121), (229, 192, 123),
    (97, 175, 239), (198, 120, 221), (86, 182, 194), (171, 178, 191),
    (92, 99, 112), (240, 128, 138), (170, 215, 140), (240, 205, 140),
    (125, 195, 250), (215, 145, 235), (110, 200, 212), (230, 233, 238),
]


def xterm256(n):
    if n < 16:
        return ANSI16[n]
    if n < 232:
        n -= 16
        steps = [0, 95, 135, 175, 215, 255]
        return (steps[n // 36], steps[(n // 6) % 6], steps[n % 6])
    v = 8 + (n - 232) * 10
    return (v, v, v)


SGR_RE = re.compile(r"\x1b\[([0-9;:]*)m")
OTHER_ESC = re.compile(r"\x1b(?:\[[0-9;?]*[A-Za-ln-z]|\][^\x07]*\x07|[()][0-9A-Za-z]|[=>78])")


class Attr:
    __slots__ = ("fg", "bg", "bold", "dim", "rev", "ul")

    def __init__(self):
        self.fg = None
        self.bg = None
        self.bold = self.dim = self.rev = self.ul = False

    def copy(self):
        a = Attr()
        a.fg, a.bg, a.bold, a.dim, a.rev, a.ul = self.fg, self.bg, self.bold, self.dim, self.rev, self.ul
        return a

    def key(self):
        return (self.fg, self.bg, self.bold, self.dim, self.rev, self.ul)

    def apply(self, params):
        ps = [p for p in re.split("[;:]", params)] if params else ["0"]
        i = 0
        while i < len(ps):
            p = int(ps[i]) if ps[i] else 0
            if p == 0:
                self.__init__()
            elif p == 1:
                self.bold = True
            elif p == 2:
                self.dim = True
            elif p == 22:
                self.bold = self.dim = False
            elif p == 4:
                self.ul = True
            elif p == 24:
                self.ul = False
            elif p == 7:
                self.rev = True
            elif p == 27:
                self.rev = False
            elif 30 <= p <= 37:
                self.fg = ANSI16[p - 30]
            elif 90 <= p <= 97:
                self.fg = ANSI16[p - 90 + 8]
            elif 40 <= p <= 47:
                self.bg = ANSI16[p - 40]
            elif 100 <= p <= 107:
                self.bg = ANSI16[p - 100 + 8]
            elif p == 39:
                self.fg = None
            elif p == 49:
                self.bg = None
            elif p in (38, 48):
                col = None
                if i + 1 < len(ps) and ps[i + 1] == "5" and i + 2 < len(ps):
                    col = xterm256(int(ps[i + 2]))
                    i += 2
                elif i + 1 < len(ps) and ps[i + 1] == "2" and i + 4 < len(ps):
                    col = (int(ps[i + 2]), int(ps[i + 3]), int(ps[i + 4]))
                    i += 4
                if p == 38:
                    self.fg = col
                else:
                    self.bg = col
            i += 1


def parse_line(line, attr):
    """-> list of (char, Attr, width) cells; attr carries over (tmux -e semantics)."""
    cells = []
    pos = 0
    line = OTHER_ESC.sub("", line)
    for m in SGR_RE.finditer(line):
        for ch in line[pos:m.start()]:
            w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            cells.append((ch, attr.copy(), w))
        attr.apply(m.group(1))
        pos = m.end()
    for ch in line[pos:]:
        w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        cells.append((ch, attr.copy(), w))
    return cells


def blend(a, b, t):
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


class Renderer:
    def __init__(self, cols, rows, size):
        self.cols, self.rows = cols, rows
        self.f = ImageFont.truetype(FONT, size)
        self.fb = ImageFont.truetype(FONT_B, size)
        self.cw = round(self.f.getlength("M"))
        self.ch = round(size * 1.32)
        asc, desc = self.f.getmetrics()
        self.base = (self.ch - (asc + desc)) // 2
        self.ui = ImageFont.truetype(FONT_UI, round(size * 0.92))
        self.uib = ImageFont.truetype(FONT_UI_B, round(size * 0.92))
        self.pad = 14
        self.title_h = round(size * 2.4)
        self.W = cols * self.cw + 2 * self.pad
        self.H = rows * self.ch + self.title_h + self.pad + 6
        self.row_cache = {}

    def row_image(self, width, cells_key, cells):
        img = self.row_cache.get(cells_key)
        if img is not None:
            return img
        img = Image.new("RGB", (width * self.cw, self.ch), BG)
        d = ImageDraw.Draw(img)
        x = 0
        for ch, a, w in cells:
            if x >= width:
                break
            fg = a.fg or FG
            bg = a.bg or BG
            if a.rev:
                fg, bg = bg, fg
            if a.dim:
                fg = blend(fg, bg, 0.45)
            if bg != BG:
                d.rectangle([x * self.cw, 0, (x + w) * self.cw - 1, self.ch - 1], fill=bg)
            if ch != " ":
                d.text((x * self.cw, self.base), ch, font=self.fb if a.bold else self.f, fill=fg)
            if a.ul:
                d.line([x * self.cw, self.ch - 3, (x + w) * self.cw - 1, self.ch - 3], fill=fg)
            x += w
        if len(self.row_cache) > 20000:
            self.row_cache.clear()
        self.row_cache[cells_key] = img
        return img

    def frame(self, panes, window_title):
        im = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(im)
        # window chrome
        d.rectangle([0, 0, self.W, self.title_h], fill=CHROME)
        r = self.title_h // 6
        cy = self.title_h // 2
        for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
            cx = self.pad + r + i * (3 * r)
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
        tw = d.textlength(window_title, font=self.ui)
        d.text(((self.W - tw) / 2, cy), window_title, font=self.ui, fill=(190, 194, 204), anchor="lm")
        ox, oy = self.pad, self.title_h + 6
        for p in panes:
            # tab bar on the border row above the pane
            ty = oy + (p["top"] - 1) * self.ch
            d.rectangle([ox, ty, ox + p["w"] * self.cw, ty + self.ch - 1], fill=TAB_BAR)
            label = p["title"]
            lw = d.textlength(label, font=self.uib)
            tab_x1 = ox + p["left"] * self.cw
            tab_x2 = tab_x1 + lw + 2 * self.cw
            d.rectangle([tab_x1, ty, tab_x2, ty + self.ch - 1],
                        fill=TAB_ACTIVE if p["active"] else (28, 30, 38))
            d.rectangle([tab_x1, ty, tab_x2, ty + 1], fill=ACCENT if p["active"] else (70, 75, 90))
            d.text((tab_x1 + self.cw, ty + self.ch / 2 + 1), label, font=self.uib,
                   fill=TAB_TEXT_ACTIVE if p["active"] else TAB_TEXT, anchor="lm")
            attr = Attr()
            for y, line in enumerate(p["lines"][: p["h"]]):
                start_key = attr.key()
                cells = parse_line(line, attr)
                img = self.row_image(p["w"], (p["w"], start_key, line), cells)
                im.paste(img, (ox + p["left"] * self.cw, oy + (p["top"] + y) * self.ch))
            if p["active"] and p["cur"]:
                x = ox + (p["left"] + p["cx"]) * self.cw
                y = oy + (p["top"] + p["cy"]) * self.ch
                d.rectangle([x, y + 1, x + self.cw - 1, y + self.ch - 2], fill=CURSOR)
        return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("out")
    ap.add_argument("--font-size", type=int, default=15)
    ap.add_argument("--max-gap", type=float, default=3.0)
    ap.add_argument("--start", type=float, default=0.0, help="seconds from the first frame")
    ap.add_argument("--end", type=float, default=1e9)
    ap.add_argument("--hold-end", type=float, default=4.0)
    ap.add_argument("--title", default="deepseek-v4.1-flash  —  trn3.48xlarge")
    ap.add_argument("--still", default=None)
    ap.add_argument("--still-at", type=float, default=None)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.rec)]
    end_t = recs[-1]["t"] if recs[-1].get("end") else None
    frames = [r for r in recs if "panes" in r]
    t0 = frames[0]["t"]
    sel = [f for f in frames if a.start <= f["t"] - t0 <= a.end]
    p0 = sel[0]["panes"]
    cols = max(p["left"] + p["w"] for p in p0)
    rows = max(p["top"] + p["h"] for p in p0)
    R = Renderer(cols, rows, a.font_size)

    if a.still:
        target = t0 + a.still_at
        f = min(frames, key=lambda f: abs(f["t"] - target))
        R.frame(f["panes"], a.title).save(a.still)
        print("still ->", a.still)
        return

    imgs, durs = [], []
    for i, f in enumerate(sel):
        nxt = sel[i + 1]["t"] if i + 1 < len(sel) else (end_t or f["t"])
        dur = min(nxt - f["t"], a.max_gap)
        if i + 1 == len(sel):
            dur = a.hold_end
        imgs.append(R.frame(f["panes"], a.title))
        durs.append(dur)
    # GIF delays are in 10 ms units; carry rounding error forward
    out_d, carry = [], 0.0
    for d in durs:
        ms = d * 1000 + carry
        q = max(20, int(round(ms / 10.0)) * 10)
        carry = ms - q
        out_d.append(q)
    # one shared palette from a few representative frames
    probe_idx = sorted(set([0, len(imgs) // 3, 2 * len(imgs) // 3, len(imgs) - 1]))
    probe = Image.new("RGB", (R.W, R.H * len(probe_idx)))
    for k, j in enumerate(probe_idx):
        probe.paste(imgs[j], (0, R.H * k))
    pal = probe.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(a.out, save_all=True, append_images=q[1:], duration=out_d, loop=0, optimize=False,
              disposal=1)
    print(f"gif: {len(q)} frames, {sum(out_d) / 1000:.1f} s, {R.W}x{R.H} -> {a.out}")


if __name__ == "__main__":
    main()
