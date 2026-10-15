"""Render chosen frames of a saved .blend (test frames, thumbnails).

    /root/bpyenv/bin/python tools/render_frames.py SCENE.blend OUTDIR 1,200,450 [--scale 100] [--samples N]
        [--no-blur] [--no-hud] [--prefix frame_]

--no-hud hides the overlay rig (the "HUD" empty and its children, plus "Hi*" highlight shells) for clean
thumbnail plates. Prints seconds per frame (use these timings for render estimates).
"""
import argparse
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
p = argparse.ArgumentParser()
p.add_argument("blend")
p.add_argument("outdir")
p.add_argument("frames")
p.add_argument("--scale", type=int, default=100)
p.add_argument("--samples", type=int, default=None)
p.add_argument("--no-blur", action="store_true")
p.add_argument("--no-hud", action="store_true")
p.add_argument("--prefix", default="frame_")
a = p.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
sc = bpy.context.scene
if a.no_hud and "HUD" in bpy.data.objects:
    hud = bpy.data.objects["HUD"]
    for o in [hud, *hud.children_recursive, *[o for o in bpy.data.objects if o.name.startswith("Hi")]]:
        if o.animation_data:
            o.animation_data_clear()          # drops keyed hide_render/alpha so it stays hidden
        o.hide_render = True
sc.render.resolution_percentage = a.scale
if a.samples:
    sc.eevee.taa_render_samples = a.samples
if a.no_blur:
    sc.render.use_motion_blur = False
os.makedirs(a.outdir, exist_ok=True)
for f in [int(x) for x in a.frames.split(",") if x.strip()]:
    t = time.time()
    sc.frame_set(f)
    sc.render.filepath = os.path.join(os.path.abspath(a.outdir), f"{a.prefix}{f:04d}.png")
    bpy.ops.render.render(write_still=True)
    print(f"rendered {f} in {time.time() - t:.1f}s", flush=True)
