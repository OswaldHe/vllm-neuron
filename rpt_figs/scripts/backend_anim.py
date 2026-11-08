"""Animation: what the vLLM backend work removed from a batch-1 decode step (two steps shown, schematic timing)."""
from pathlib import Path
from anim import *

OUT = Path(__file__).parent / 'out'
PREV = OUT / 'backend_states'
PREV.mkdir(parents=True, exist_ok=True)

W, H = 760, 290
TX0, TX1 = 112, 736
PX = 7.4                      # pt per ms (schematic)
YB = 100                      # step brackets
YE, YW, YD = 116, 146, 182    # engine, worker, device lane tops
LH, DH = 22, 30

STEPS = [
    ('3 RPCs', '① Synchronous step, three RPCs',
     'The engine schedules and then waits; the device idles while the host prepares inputs, samples and fetches drafts.'),
    ('1 RPC', '② One RPC per step',
     'Sampling runs inside execute_model and the drafts ride back in its output: two RPCs per step are gone.'),
    ('Direct launch', '③ Direct launch of cached NEFFs',
     'Same IO tensors every step, so the runtime reuses each launch\'s DMA ring: first launch ~4 ms → 0.7 ms to submit.'),
    ('Async', '④ Async scheduling, step inputs built on device',
     'An NKI prologue makes positions and slots on the device; step N+1 is scheduled while step N runs (idle ~5 → 0.7 ms).'),
    ('Overlap', '⑤ Overlap collectives with the next layer',
     'Inside each launch, the next layer\'s weights load after the all-reduce, overlapping compute: shorter launches.'),
]

KIND = {
    'sched': (GREY_M, GREY_D, INK),
    'rpc': (BLUE, BLUE, '#ffffff'),
    'host': (BLUE_L, BLUE_M, INK),
    'launch': (ORANGE_L, ORANGE, INK),
    'draft': (PURPLE_L, PURPLE, INK),
    'pro': (GREEN_L, GREEN, INK),
}


def sync_step(i):
    """One synchronous step from t=0: returns segments [(lane, name, t0, t1, kind, label)] and its length."""
    sub = [4.0, 3.6, 3.6, 3.6, 3.6] if i < 2 else [0.7] * 5
    dl = 5.0
    s = [('eng', 'sched', 0, 0.6, 'sched', ''), ('eng', 'rpc1', 0.6, 1.6, 'rpc', 'RPC'),
         ('wrk', 'prep', 1.6, 4.0, 'host', 'prepare')]
    t = 4.0
    dev_t = None
    ends = []
    for k in range(5):
        s.append(('wrk', f'sub{k}', t, t + sub[k], 'host', 'submit' if k == 0 and sub[k] > 2 else ''))
        t += sub[k]
        start = t if dev_t is None else max(t, dev_t)
        s.append(('dev', f'L{k}', start, start + dl, 'launch', ['L0–7', 'L8–15', 'L16–23', 'L24–31', 'L32–39'][k]))
        dev_t = start + dl
    t = dev_t
    if i == 0:
        s += [('eng', 'rpc2', t, t + 1.0, 'rpc', 'RPC'), ('wrk', 'sample', t + 1.0, t + 2.0, 'host', 'sample'),
              ('eng', 'rpc3', t + 2.0, t + 3.0, 'rpc', 'RPC')]
        t += 3.0
    else:
        s += [('eng', 'rpc2', t, t, 'rpc', ''), ('wrk', 'sample', t, t + 1.0, 'host', 'sample'),
              ('eng', 'rpc3', t + 1.0, t + 1.0, 'rpc', '')]
        t += 1.0
    dsub = 2.0 if i < 2 else 0.7
    s.append(('wrk', 'dsub', t, t + dsub, 'host', 'submit' if dsub > 1 else ''))
    t += dsub
    s.append(('dev', 'draft', t, t + 2.5, 'draft', 'draft'))
    t += 2.5
    s.append(('wrk', 'read', t, t + 0.5, 'host', ''))
    t += 0.5
    # hidden in sync mode: the device prologue
    s.append(('dev', 'pro', 4.0, 4.0, 'pro', ''))
    return s, t


def async_step(i):
    """Steady-state async step: device work back to back; the host work of the NEXT step overlaps it."""
    dl = 5.0 if i == 3 else 4.2
    s = [('dev', 'pro', 0.0, 0.25, 'pro', '')]
    t = 0.35
    for k in range(5):
        s.append(('dev', f'L{k}', t, t + dl, 'launch', ['L0–7', 'L8–15', 'L16–23', 'L24–31', 'L32–39'][k]))
        t += dl + 0.06
    s.append(('dev', 'draft', t + 0.05, t + 2.55, 'draft', 'draft'))
    length = t + 2.55 + 0.25
    host = [('eng', 'sched', 1.0, 1.6, 'sched', ''), ('eng', 'rpc1', 1.6, 2.6, 'rpc', 'RPC'),
            ('wrk', 'prep', 2.6, 3.4, 'host', 'prep'),
            ('eng', 'rpc2', 2.6, 2.6, 'rpc', ''), ('eng', 'rpc3', 2.6, 2.6, 'rpc', '')]
    t = 3.4
    for k in range(5):
        host.append(('wrk', f'sub{k}', t, t + 0.7, 'host', ''))
        t += 0.7
    host += [('wrk', 'sample', t, t, 'host', ''), ('wrk', 'dsub', t, t + 0.7, 'host', ''),
             ('wrk', 'read', t + 0.7, t + 0.7, 'host', '')]
    return s, host, length


def X(t):
    return TX0 + t * PX


def seg_el(st, sid, lane, t0, t1, kind, label, op=1.0):
    fill, stroke, tc = KIND[kind]
    y, h = {'eng': (YE, LH), 'wrk': (YW, LH), 'dev': (YD, DH)}[lane]
    x0, x1 = X(t0), X(t1)
    st[sid] = R(x0, y, max(0.0, x1 - x0), h, fill=fill, stroke=stroke, sw=0.8, rx=2.5, op=op, z=3)
    fs = 8.0 if kind in ('launch', 'draft') else 8.5
    wlab = measure(label, fs, bold=kind in ('rpc', 'launch', 'draft')) + 2 if label else 0
    if label and x1 - x0 < wlab and kind == 'launch':
        fs = 7.0
        wlab = measure(label, fs, bold=True) + 1
    if label and x1 - x0 >= wlab:
        st[sid + '_t'] = T((x0 + x1) / 2, y + h / 2 + 3.0, label, size=fs, fill=tc, anchor='middle', op=op,
                           weight='bold' if kind in ('rpc', 'launch', 'draft') else 'normal', z=6)
    else:
        st[sid + '_t'] = T((x0 + x1) / 2, y + h / 2 + 3.1, label or ' ', size=8.5, fill=tc, anchor='middle', op=0,
                           z=6)


def idle_boxes(st, step, intervals, t_end_view):
    k = 0
    for a, b in intervals:
        if b - a < 0.05:
            continue
        x0, x1 = X(a), min(X(b), TX1)
        if x1 <= x0:
            continue
        st[f'idle{step}_{k}'] = R(x0, YD + 3, x1 - x0, DH - 6, fill=RED_L, stroke=RED, sw=0.7, rx=2, dash='2 1.5',
                                  z=2)
        if x1 - x0 > 26:
            st[f'idle{step}_{k}_t'] = T((x0 + x1) / 2, YD + DH / 2 + 3, 'idle', size=8.5, fill=RED,
                                        anchor='middle', z=6)
        k += 1


def ar_stripes(st, sid, t0, t1, i):
    """all-reduce slivers inside a launch; thinner once loads stop contending (state 4)."""
    x0, x1 = X(t0), X(t1)
    n = 4
    w = 2.6 if i < 4 else 1.2
    for j in range(n):
        cx = x0 + (j + 0.5) * (x1 - x0) / n + 6
        st[f'{sid}_ar{j}'] = R(cx, YD + 1, w, DH - 2, fill=GREY_D, rx=0.4, z=4, op=0.8)


def header(i):
    st = {}
    _, title, sub = STEPS[i]
    st[f'title{i}'] = T(24, 30, title, size=15.5, weight='bold')
    st[f'sub{i}'] = T(24, 49, sub, size=10.2, fill=MUTED)
    x = 24
    for k, (pill, *_r) in enumerate(STEPS):
        w = measure(pill, 9, bold=True) + 14
        on = k == i
        st[f'pill{k}'] = R(x, 62, w, 16, fill=INK if on else GREY_L, rx=8, z=1)
        st[f'pilltx{k}'] = T(x + w / 2, 73.3, pill, size=9, fill='#ffffff' if on else MUTED, anchor='middle',
                             weight='bold')
        x += w + 6
    st['lblE'] = T(24, YE + 15, 'Engine', size=10.5, weight='bold')
    st['lblW'] = T(24, YW + 15, 'Worker host', size=10.5, weight='bold')
    st['lblD'] = T(24, YD + 19, 'Device', size=10.5, weight='bold')
    for y in (YE + LH / 2, YW + LH / 2, YD + DH / 2):
        st[f'lane{y}'] = P([(TX0, y), (TX1, y)], stroke=GRID, sw=0.8, z=0)
    st['axis'] = P([(TX0, YD + DH + 12), (TX1, YD + DH + 12)], stroke=GREY_M, sw=0.8, z=0)
    # legend
    items = [('sched', 'schedule'), ('rpc', 'RPC'), ('host', 'host work'), ('launch', 'layer launch'),
             ('draft', 'drafter'), ('pro', 'on-device prologue')]
    x, ly = 112, 268
    for kind, lab in items:
        fill, stroke, _ = KIND[kind]
        st[f'lg_{kind}'] = R(x, ly - 9, 12, 11, fill=fill, stroke=stroke, sw=0.8, rx=2)
        st[f'lg_{kind}_t'] = T(x + 17, ly, lab, size=9.5, fill=MUTED)
        x += 17 + measure(lab, 9.5) + 16
    st['lg_idle'] = R(x, ly - 9, 12, 11, fill=RED_L, stroke=RED, sw=0.7, rx=2, dash='2 1.5')
    st['lg_idle_t'] = T(x + 17, ly, 'device idle', size=9.5, fill=MUTED)
    return st


BASE_LEN = None
STEP_NAMES = ['N', 'N+1', 'N+2']


def state(i):
    global BASE_LEN
    st = header(i)
    if i < 3:
        segs, L = sync_step(i)
        for step in (0, 1):
            off = step * L
            for lane, nm, a, b, kind, lab in segs:
                seg_el(st, f's{step}_{nm}', lane, a + off, b + off, kind, lab)
            dev = sorted((a + off, b + off) for lane, nm, a, b, k, lab in segs if lane == 'dev' and b > a)
            gaps, cur = [], off
            for a, b in dev:
                if a > cur:
                    gaps.append((cur, a))
                cur = max(cur, b)
            if cur < off + L:
                gaps.append((cur, off + L))
            idle_boxes(st, step, gaps, None)
            for lane, nm, a, b, kind, lab in segs:
                if lane == 'dev' and kind == 'launch':
                    ar_stripes(st, f's{step}_{nm}', a + off, b + off, i)
            st[f'br{step}'] = P([(X(off) + 1, YB + 5), (X(off) + 1, YB), (X(off + L) - 1, YB), (X(off + L) - 1, YB + 5)],
                                stroke=GREY_D, sw=0.9, z=1)
            st[f'brt{step}'] = T(X(off + L / 2), YB - 4, f'step {"N" if step == 0 else "N+1"}', size=9,
                                 fill=MUTED, anchor='middle', weight='bold')
            cl_end = max(b for lane, nm, a, b, k, lab in segs if lane == 'wrk' and nm.startswith('sub'))
            st[f'hl{step}'] = T(X(off + cl_end) + 5, YW + 15, f'for step {STEP_NAMES[step]}', size=9, fill=BLUE,
                                weight='bold')
        if i == 0:
            BASE_LEN = 2 * L
        note = {0: 'prepare inputs, then submit 5 launches (~4 ms each)', 1: 'prepare inputs, then submit 5 launches (~4 ms each)',
                2: 'submits now ~0.7 ms each'}[i]
        st[f'wnote{i}'] = T(X(1.6), YW + LH + 10.5, note, size=8.5, fill=BLUE, anchor='start')
        # third step's host work (async only) parked invisibly at the right edge
        for nm in ('sched', 'rpc1', 'prep', 'sub0', 'sub1', 'sub2', 'sub3', 'sub4', 'dsub'):
            lane = 'eng' if nm in ('sched', 'rpc1') else 'wrk'
            seg_el(st, f's2_{nm}', lane, 2 * L, 2 * L, 'host' if lane == 'wrk' else ('rpc' if nm == 'rpc1' else 'sched'),
                   '', op=0)
    else:
        dsegs, hsegs, L = async_step(i)
        for step in (0, 1):
            off = step * L
            for lane, nm, a, b, kind, lab in dsegs:
                seg_el(st, f's{step}_{nm}', lane, a + off, b + off, kind, lab)
                if kind == 'launch':
                    ar_stripes(st, f's{step}_{nm}', a + off, b + off, i)
            dev = sorted((a + off, b + off) for lane, nm, a, b, k, lab in dsegs)
            gaps, cur = [], off
            for a, b in dev:
                if a > cur:
                    gaps.append((cur, a))
                cur = max(cur, b)
            if cur < off + L:
                gaps.append((cur, off + L))
            idle_boxes(st, step, gaps, None)
            st[f'br{step}'] = P([(X(off) + 1, YB + 5), (X(off) + 1, YB), (X(off + L) - 1, YB), (X(off + L) - 1, YB + 5)],
                                stroke=GREY_D, sw=0.9, z=1)
            st[f'brt{step}'] = T(X(off + L / 2), YB - 4, f'step {"N" if step == 0 else "N+1"}', size=9,
                                 fill=MUTED, anchor='middle', weight='bold')
        # host work of step k runs during step k-1: step 0's host work is off-screen (fades), 1 and 2 overlap
        for k in (0, 1, 2):
            off = (k - 1) * L
            for lane, nm, a, b, kind, lab in hsegs:
                seg_el(st, f's{k}_{nm}', lane, a + off, b + off, kind, lab, op=0 if k == 0 else 1.0)
            if k == 0:
                for lane, nm, a, b, kind, lab in hsegs:
                    st[f's0_{nm}']['x'] = TX0
                    st[f's0_{nm}']['w'] = 0
        cl_end = max(b for lane, nm, a, b, k, lab in hsegs if lane == 'wrk' and b > a)
        for k in (1, 2):
            st[f'hl{k}'] = T(X((k - 1) * L + cl_end) + 5, YW + 15, f'for step {STEP_NAMES[k]}', size=9, fill=BLUE,
                             weight='bold')
        st['ovl'] = T(X(L * 0.62), YW + LH + 11.5, 'step N+1 prepared during step N', size=9, fill=BLUE,
                      anchor='middle', weight='bold')
    # ghost of the baseline length
    if i > 0 and BASE_LEN:
        y = YD + DH + 12
        st['ghost'] = R(TX0, y - 2, min(X(BASE_LEN), TX1) - TX0, 4, fill=GREY_M, rx=2, z=1, op=0.9)
        st['ghostt'] = T(min(X(BASE_LEN), TX1), y + 14, 'length of two steps at ①', size=9, fill=FAINT, anchor='end')
    return st


DEFS = arrow_defs({'arrB': BLUE})

if __name__ == '__main__':
    states = [state(i) for i in range(5)]
    n = render_sequence(states, [8000] * len(states), W, H, OUT / "backend.gif", defs=DEFS, scale=2.5,
                        preview_dir=PREV, progress=progress_bar(len(states), 24, TX1, 279), progress_frames=32)
    print('frames', n)
