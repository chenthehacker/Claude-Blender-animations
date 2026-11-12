"""Render the town river scene at 11 pixelation levels (0 to 100% in steps of 10).

0% is the clean anti aliased 1200x720 render. 100% is the native 3DS top screen look (400x240, one sample, aliased).
Each step in between lowers the render width linearly (1200, 1120, ... 400; height keeps 5:3), renders aliased like
the native mode, and the images are then upscaled with NEAREST to 1200x720 so every level compares at the same size.

    /root/bpyenv/bin/python pixelation_series.py renders/pixelation
"""
import os
import sys

import bpy  # noqa: F401  (must load before the scene module)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import town_river_test as T  # noqa: E402

out_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else sys.argv[1])
os.makedirs(out_dir, exist_ok=True)
for level in range(0, 101, 10):
    w = round(1200 - 8 * level)
    h = round(w * 3 / 5)
    if level == 0:
        T.build(native=False)                       # clean, anti aliased
    else:
        T.NATIVE = (w, h)
        T.build(native=True)                        # aliased, one sample, filter 0.5
    sc = bpy.context.scene
    path = os.path.join(out_dir, f"raw_{level:03d}.png")
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print(f"level {level:3d}%: rendered {w}x{h}", flush=True)
