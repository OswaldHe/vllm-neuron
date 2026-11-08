"""Tiny keyframe engine: a state is {id: element}; transitions interpolate shared ids, fade the rest.
Elements: {'t': 'rect'|'text'|'poly', ...}. Ghost elements (op 0) placed at merge targets make boxes slide
into the box that replaces them."""
from blogkit import *


def R(x, y, w, h, fill='none', stroke='none', sw=0.8, rx=2, op=1.0, dash=None, z=0):
    return dict(t='rect', x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, sw=sw, rx=rx, op=op, dash=dash, z=z)


def T(x, y, s, size=9, fill=INK, anchor='start', weight='normal', op=1.0, z=5, italic=False):
    return dict(t='text', x=x, y=y, s=s, size=size, fill=fill, anchor=anchor, weight=weight, op=op, z=z,
                italic=italic)


def P(pts, stroke=INK, sw=0.8, op=1.0, dash=None, marker=None, z=3, fill='none'):
    return dict(t='poly', pts=[tuple(p) for p in pts], stroke=stroke, sw=sw, op=op, dash=dash, marker=marker,
                z=z, fill=fill)


def _col(a, b, t):
    if a == b:
        return a
    if isinstance(a, str) and isinstance(b, str) and a.startswith('#') and b.startswith('#'):
        return mix(a, b, t)
    return b if t > 0.5 else a


def interp(e1, e2, t):
    if e1 is None:
        e = dict(e2); e['op'] = e2['op'] * max(0.0, 2 * t - 1); return e
    if e2 is None:
        e = dict(e1); e['op'] = e1['op'] * max(0.0, 1 - 2 * t); return e
    e = dict(e2)
    for k in ('x', 'y', 'w', 'h', 'rx', 'sw', 'op', 'size'):
        if k in e1 and k in e2:
            e[k] = lerp(e1[k], e2[k], t)
    for k in ('fill', 'stroke'):
        if k in e1 and k in e2:
            e[k] = _col(e1[k], e2[k], t)
    if e1['t'] == 'poly' and len(e1['pts']) == len(e2['pts']):
        e['pts'] = [(lerp(a[0], b[0], t), lerp(a[1], b[1], t)) for a, b in zip(e1['pts'], e2['pts'])]
    if e1['t'] == 'text' and e1['s'] != e2['s']:
        e['s'] = e2['s'] if t > 0.5 else e1['s']
    return e


def blend(s1, s2, t):
    ids = list(dict.fromkeys(list(s1) + list(s2)))
    return {i: interp(s1.get(i), s2.get(i), t) for i in ids}


def to_svg(state, W, H, defs=''):
    body = [defs]
    items = sorted(state.items(), key=lambda kv: kv[1]['z'])
    for _, e in items:
        if e['op'] <= 0.01:
            continue
        if e['t'] == 'rect':
            body.append(rect(e['x'], e['y'], e['w'], e['h'], fill=e['fill'], stroke=e['stroke'], sw=e['sw'],
                             rx=e['rx'], op=min(1, e['op']), dash=e['dash']))
        elif e['t'] == 'text':
            body.append(text(e['x'], e['y'], e['s'], size=e['size'], fill=e['fill'], anchor=e['anchor'],
                             weight=e['weight'], op=min(1, e['op']), italic=e.get('italic', False)))
        elif e['t'] == 'poly':
            d = 'M' + ' L'.join(f'{f(x)},{f(y)}' for x, y in e['pts'])
            body.append(path(d, stroke=e['stroke'], sw=e['sw'], op=min(1, e['op']), dash=e['dash'],
                             marker=e['marker'], fill=e['fill']))
    return svg_doc(W, H, body)


def render_sequence(states, holds, W, H, out_gif, defs='', trans_frames=12, trans_ms=55, scale=1.45,
                    preview_dir=None, loop_back=True, progress=None, progress_frames=1):
    """states: list of dicts; holds: ms per state. Writes the GIF and per-state PNG previews.

    progress(i, frac) -> {id: element}: optional overlay drawn on state i at fraction frac of its hold
    (e.g. a progress bar). With it, each hold is split into progress_frames frames that advance frac
    from 1/progress_frames to 1, and the transition to the next state starts from frac 0 of that state.
    """
    def at(i, frac):
        if progress is None:
            return states[i]
        s = dict(states[i])
        s.update(progress(i, frac))
        return s

    seq, durs, hold_last = [], [], []
    n = len(states)
    pairs = list(range(n)) if loop_back else list(range(n - 1))
    kf = progress_frames if progress else 1
    for i in range(n):
        for k in range(kf):
            seq.append(to_svg(at(i, (k + 1) / kf), W, H, defs)); durs.append(holds[i] / kf)
        hold_last.append(len(seq) - 1)
        if i in pairs:
            j = (i + 1) % n
            if j == 0 and not loop_back:
                continue
            a, b = at(i, 1.0), at(j, 0.0)
            for k in range(1, trans_frames + 1):
                t = ease(k / (trans_frames + 1))
                seq.append(to_svg(blend(a, b, t), W, H, defs)); durs.append(trans_ms)
    r = Renderer(seq[0], scale=scale)
    pngs = []
    for k, s in enumerate(seq):
        pngs.append(r.png(s))
    if preview_dir:
        for i, k in enumerate(hold_last):
            (preview_dir / f'state{i}.png').write_bytes(pngs[k])
            (preview_dir / f'state{i}.svg').write_text(seq[k])
    r.close()
    make_gif(pngs, durs, out_gif)
    return len(seq)


def progress_bar(n_steps, x0, x1, y, h=4, gap=6, fill=ORANGE, track=GREY_L):
    """A segmented progress bar for render_sequence(progress=...): one segment per step, finished
    steps filled, the current one filling over its hold."""
    seg = (x1 - x0 - (n_steps - 1) * gap) / n_steps

    def overlay(i, frac):
        st = {}
        for k in range(n_steps):
            x = x0 + k * (seg + gap)
            f = 1.0 if k < i else (frac if k == i else 0.0)
            st[f'pgb{k}'] = R(x, y, seg, h, fill=track, rx=h / 2, z=1)
            st[f'pgf{k}'] = R(x, y, seg * f, h, fill=fill, rx=h / 2, z=2, op=1.0 if seg * f >= 0.5 else 0.0)
        return st
    return overlay
