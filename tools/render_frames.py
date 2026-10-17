"""Render chosen frames of a saved .blend (test frames, thumbnails).

    /root/bpyenv/bin/python tools/render_frames.py SCENE.blend OUTDIR 1,200,450 [--scale 100] [--samples N]
        [--no-blur] [--no-hud] [--prefix frame_] [--skip-existing]

Frames can also be ranges with an optional step: 1-900 or 1-1800:2. --skip-existing leaves frames that
are already on disk, so a long batch can be stopped and resumed.

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
p.add_argument("--skip-existing", action="store_true")
a = p.parse_args(argv)


def frame_list(spec):
    out = []
    for part in [x.strip() for x in spec.split(",") if x.strip()]:
        if "-" in part:
            rng, _, step = part.partition(":")
            lo, hi = (int(v) for v in rng.split("-"))
            out += list(range(lo, hi + 1, int(step or 1)))
        else:
            out.append(int(part))
    return out


bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
sc = bpy.data.scenes.get("Main") or bpy.context.scene   # overlay builds keep their text in a second scene
others = [s for s in bpy.data.scenes if s != sc]
if a.no_hud and "HUD" in bpy.data.objects:
    hud = bpy.data.objects["HUD"]
    for o in [hud, *hud.children_recursive, *[o for o in bpy.data.objects if o.name.startswith("Hi")]]:
        if o.animation_data:
            o.animation_data_clear()          # drops keyed hide_render/alpha so it stays hidden
        o.hide_render = True
for s in [sc, *others]:
    s.render.resolution_percentage = a.scale
if a.samples:
    sc.eevee.taa_render_samples = a.samples
if a.no_blur:
    sc.render.use_motion_blur = False
os.makedirs(a.outdir, exist_ok=True)
for f in frame_list(a.frames):
    path = os.path.join(os.path.abspath(a.outdir), f"{a.prefix}{f:04d}.png")
    if a.skip_existing and os.path.exists(path) and os.path.getsize(path) > 0:
        continue
    t = time.time()
    for s in others:
        s.frame_current = f
    sc.frame_set(f)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True, scene=sc.name)
    print(f"rendered {f} in {time.time() - t:.1f}s", flush=True)
