import sys, importlib
from pathlib import Path
from anim import to_svg
from blogkit import Renderer
mod = importlib.import_module(sys.argv[1])
idx = [int(a) for a in sys.argv[2:]] or range(len(mod.STEPS))
svgs = {i: to_svg(mod.state(i), mod.W, mod.H, mod.DEFS) for i in idx}
r = Renderer(next(iter(svgs.values())), scale=2.0)
for i, s in svgs.items():
    r.png(s, mod.PREV / f'state{i}.png')
r.close()
print('ok')
