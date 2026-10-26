"""Thumbnail plate: the strike on the glowing egi, with a small school of glowing baitfish scattering below the squid
and a few drifting plankton specks. Built on top of the saved scene (same camera, squid, jig and materials).

    /root/bpyenv/bin/python thumbnail_scene.py night_squid_masterclass.blend OUT.blend [--frame F]
    /root/bpyenv/bin/python ../tools/render_frames.py OUT.blend PLATE_DIR F --no-hud --no-blur --samples 64
    python3 ../tools/thumbnail.py PLATE_DIR/frame_F.png fonts thumbnail "NIGHT SQUID FISHING" ...

The glowing fish and the specks are a stylised illustration for the thumbnail (baitfish catching the lamp light,
not bioluminescence); they are not in the film.
"""
import argparse
import math
import os
import random
import runpy
import sys

import bpy
import bmesh  # noqa: E402  (the bpy module must load first)
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
p = argparse.ArgumentParser()
p.add_argument("blend")
p.add_argument("out")
p.add_argument("--frame", type=int, default=None, help="default: 10 frames after the strike (events.json)")
a = p.parse_args(argv)
if a.frame is None:
    import json
    with open(os.path.join(HERE, "events.json")) as fh:
        a.frame = json.load(fh)["strike"] + 10

g = runpy.run_path(os.path.join(HERE, "night_squid_masterclass.py"), run_name="sqmod")
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
sc = bpy.data.scenes["Main"]
for s in bpy.data.scenes:
    s.frame_current = a.frame
sc.frame_set(a.frame)
srgb, psdf, new_material, heading_quat = g["srgb"], g["psdf"], g["new_material"], g["heading_quat"]
atmos = bpy.data.node_groups.get("Atmosphere")
cam = sc.camera
cm = cam.matrix_world.copy()
FRAME_H = 36.0 / cam.data.lens                     # frame height per metre of depth (vertical sensor fit)
rng = random.Random(7)
col = bpy.data.collections.new("Thumbnail")
sc.collection.children.link(col)


def at(nx, ny, d):
    """World point at normalised screen position (nx, ny in frame heights, centre 0) and depth d metres."""
    return cm @ Vector((nx * FRAME_H * d, ny * FRAME_H * d, -d))


def cam_dir(x, y, z):
    return (cm.to_3x3() @ Vector((x, y, z))).normalized()


# --- glowing baitfish: silver bodies lit green by the lamp, a soft glow of their own -------------------------
glow_fish = new_material("ThumbGlowFish", lambda nt: psdf(nt, srgb(196, 214, 210), rough=0.22, metal=0.55, coat=0.6,
                                                         emission=srgb(110, 255, 196), emission_strength=1.6), atmos)
src = bpy.data.objects["SchoolFish"].data
mesh = src.copy()
mesh.name = "ThumbFish"
mesh.materials[0] = glow_fish
flee = cam_dir(1.0, 0.12, -0.35)                    # the school scatters right and away from the strike
n_fish = 0
placed = []
for i in range(22):
    for _ in range(40):                             # keep each fish clear of the others on screen
        nx, ny = rng.uniform(-0.25, 0.12), rng.uniform(-0.40, -0.15)
        d = rng.uniform(1.3, 2.4)
        if all(math.hypot(nx - ox, ny - oy) > 0.045 for ox, oy in placed):
            break
    placed.append((nx, ny))
    ob = bpy.data.objects.new(f"ThumbFish{i}", mesh)
    col.objects.link(ob)
    ob.location = at(nx, ny, d)
    fwd = (flee + Vector((rng.gauss(0, 0.18), rng.gauss(0, 0.18), rng.gauss(0, 0.1)))).normalized()
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = heading_quat(fwd, rng.uniform(-0.25, 0.25))
    ob.scale = (s := rng.uniform(0.9, 1.15), s, s)
    ob.visible_shadow = False
    n_fish += 1
# two strays behind the jig, farther off (small and soft)
for i, (nx, ny, d) in enumerate(((0.06, 0.06, 2.8), (0.11, -0.02, 3.2))):
    ob = bpy.data.objects.new(f"ThumbStray{i}", mesh)
    col.objects.link(ob)
    ob.location = at(nx, ny, d)
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = heading_quat(cam_dir(-1.0, 0.1, -0.2))
    ob.visible_shadow = False
    n_fish += 1

# --- a few drifting plankton specks (soft bokeh in front, pin points behind) ---------------------------------
specks = {}
for name, color, strength in (("ThumbSpeckCyan", srgb(120, 255, 222), 9.0), ("ThumbSpeckGold", srgb(255, 204, 120), 7.0)):
    specks[name] = new_material(name, lambda nt, c=color, s=strength: g["emission_shader"](nt, c, s))
bm = bmesh.new()
warm = bm.verts.layers.float.new("warm")
n_specks = 0
for _ in range(80):
    nx, ny = rng.uniform(-0.28, 0.14), rng.uniform(-0.42, 0.33)
    if ny > 0.12 and rng.random() < 0.6:            # fewer behind the headline
        continue
    d = rng.uniform(0.45, 2.8)
    r = d * rng.uniform(0.0007, 0.0016)
    res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r, matrix=Matrix.Translation(at(nx, ny, d)))
    w = 1.0 if rng.random() < 0.2 else 0.0
    for v in res["verts"]:
        v[warm] = w
    n_specks += 1
sp = g["finish"]("ThumbSpecks", bm, col, [specks["ThumbSpeckCyan"], specks["ThumbSpeckGold"]])
for poly in sp.data.polygons:                       # gold specks use the second material
    poly.material_index = 1 if sp.data.attributes["warm"].data[poly.vertices[0]].value > 0.5 else 0
sp.visible_shadow = False

bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out), compress=True)
print(f"thumbnail scene: {n_fish} glowing fish, {n_specks} specks; saved {a.out}")
