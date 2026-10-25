"""Thumbnail plate: the hooked giant breaching beside the boat, water bursting off her, the line taut from the
corner of her jaw to the rod. Built on top of the saved scene (same sea, sky, boat, angler and materials).

    /root/bpyenv/bin/python thumbnail_scene.py bluefin_record_masterclass.blend OUT.png [--samples 64] [--scale 100]

Then: python3 ../tools/thumbnail.py OUT.png fonts thumbnail "THE FISHING SERIES" ...
The breach is an illustration for the thumbnail; it is not a shot in the film.
"""
import argparse
import math
import os
import random
import runpy
import sys
from types import SimpleNamespace

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
p = argparse.ArgumentParser()
p.add_argument("blend")
p.add_argument("out")
p.add_argument("--samples", type=int, default=64)
p.add_argument("--scale", type=int, default=100)
p.add_argument("--frame", type=int, default=1010)
a = p.parse_args(argv)

g = runpy.run_path(os.path.join(HERE, "bluefin_record_masterclass.py"), run_name="tnmod")
bpy.ops.wm.open_mainfile(filepath=os.path.abspath(a.blend))
sc = bpy.data.scenes["Main"]
F = a.frame
for s in bpy.data.scenes:
    s.frame_current = F
    s.render.resolution_percentage = a.scale
sc.frame_set(F)
t = g["t_of"](F)
surface_z, sea_point = g["surface_z"], g["sea_point"]
def material(name, color, rough, metal=0.0):
    """The saved scene's material, or (unused ones are not saved) a plain one made here."""
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    return g["new_material"](name, lambda nt: g["psdf"](nt, g["srgb"](*color), rough=rough, metal=metal))


M = SimpleNamespace(line=material("Line", (214, 220, 140), 0.3), hook=material("Hook", (150, 150, 156), 0.25, 1.0),
                    spray=bpy.data.materials["Spray"], wake=bpy.data.materials["WakeFoam"])
# droplets catching the low sun: bright and glossy (the film's spray reads as grey beads this close)
M.spray = g["new_material"]("ThumbSpray", lambda nt: g["psdf"](nt, g["srgb"](246, 248, 246), rough=0.04, transmission=0.92,
                                                                  ior=1.33, emission=g["srgb"](255, 226, 196),
                                                                  emission_strength=0.06))   # clear water: refracts
M.line = g["new_material"]("ThumbLineMat", lambda nt: g["psdf"](nt, g["srgb"](226, 232, 150), rough=0.3,
                                                                    emission=g["srgb"](226, 232, 150), emission_strength=0.8))
col = bpy.data.collections.new("Thumbnail")
sc.collection.children.link(col)


def unkey(ob, hide=False):
    if ob.animation_data:
        ob.animation_data_clear()
    ob.hide_render = hide


# no overlays on the plate, and no fight line (the hooked fish gets its own)
hud = bpy.data.objects.get("HUD")
for o in ([hud, *hud.children_recursive] if hud else []) + [o for o in bpy.data.objects if o.name.startswith("Hi")]:
    unkey(o, hide=True)
for n in ("LineAbove_fight", "LineUnder_fight", "TransomSpray"):
    if n in bpy.data.objects:
        unkey(bpy.data.objects[n], hide=True)

# --- the giant, breaching off the starboard quarter, broadside to the lens ---------------------------------------
boat = bpy.data.objects["Boat"]
bx = boat.matrix_world.translation.x
centre = Vector((bx - 15.0, -4.5, 0.0))
head = Vector((0.29, -0.96, 0.0)).normalized()           # across the frame, so the lens looks past her to the boat
pitch = math.radians(42.0)
fwd = (head * math.cos(pitch) + Vector((0.0, 0.0, math.sin(pitch)))).normalized()
z0 = surface_z(centre.x, centre.y, t)
root_pos = Vector((centre.x, centre.y, z0 + 0.95))          # tail just leaving the water
rig = bpy.data.objects["BluefinRig"]
fish_obs = [rig, *rig.children_recursive]
for o in fish_obs:
    unkey(o)
rig.location = root_pos
rig.rotation_mode = "QUATERNION"
rig.rotation_quaternion = g["heading_quat"](fwd, math.radians(-12))
beat = bpy.data.objects["BluefinBeat"]
unkey(beat)
beat.location = (1.3, 0.06, 0.35)                        # tail kicking, body flexed through the leap
for o in bpy.data.objects:                                # the hook shot's other props stay hidden
    if o.name in ("TowLine", "Glints", "TowBubbles"):
        unkey(o, hide=True)
bpy.context.view_layer.update()

# hook in the corner of the jaw, and the line taut to the rod tip
L = g["TUNA_L"]
q = rig.rotation_quaternion
jaw = root_pos + q @ Vector((L / 2 - 0.09 * L, -0.07, -0.05))
hb = bpy.data.curves.new("ThumbHook", "CURVE")
hb.dimensions = "3D"
hb.bevel_depth = 0.008
hb.bevel_resolution = 3
sp = hb.splines.new("POLY")
pts = [Vector((0.0, 0.0, 0.12)), Vector((0.0, 0.0, 0.0))] + \
      [Vector((0.06 * (1 - math.cos(a_)), 0.0, -0.06 * math.sin(a_))) for a_ in [math.pi * k / 8 for k in range(1, 9)]] + \
      [Vector((0.12, 0.0, 0.045))]
sp.points.add(len(pts) - 1)
for pt, v in zip(sp.points, pts):
    pt.co = (*v, 1.0)
hook = bpy.data.objects.new("ThumbHook", hb)
hook.data.materials.append(M.hook)
col.objects.link(hook)
side = fwd.cross(Vector((0, 0, 1))).normalized()
hook.matrix_world = (q.to_matrix().to_4x4())
hook.location = jaw - side * 0.02
tip = bpy.data.objects["RodTip"].matrix_world.translation.copy()
e0 = bpy.data.objects.new("ThumbLineA", None)
e1 = bpy.data.objects.new("ThumbLineB", None)
for e in (e0, e1):
    col.objects.link(e)
e0.location, e1.location = jaw + Vector((0.0, 0.0, 0.03)), tip
g["line_object"]("ThumbLine", col, M, e0, e1, radius=0.009, count=60)

# --- water: a burst at the waterline, sheets and drops streaming off her body, mist ---------------------------------
rng = random.Random(21)
P = g["Particles"]
burst, shed, mist = P(), P(), P()
entry = root_pos - fwd * (0.95 / math.sin(pitch))          # where her body leaves the water
entry.z = surface_z(entry.x, entry.y, t)
up = Vector((0, 0, 1))
for _ in range(4200):                                      # the crown thrown up as she bursts through
    age = rng.uniform(0.02, 0.55)
    ang = rng.uniform(0, 2 * math.pi)
    r = rng.uniform(0.25, 0.75)
    pos = entry + Vector((math.cos(ang) * r, math.sin(ang) * r, 0.02))
    out = Vector((math.cos(ang), math.sin(ang), 0.0))
    vel = out * rng.uniform(0.8, 3.0) + up * rng.uniform(1.5, 4.5) + fwd * rng.uniform(0.0, 1.8)
    burst.emit(t - age, pos, vel, 1.6, rng.uniform(0.003, 0.014), 9.81)
for _ in range(2400):                                      # water pouring off her flanks and fins
    age = rng.uniform(0.0, 0.45)
    u = rng.uniform(-0.15, 0.5)
    pos = root_pos + fwd * (u * L) + side * rng.uniform(-0.28, 0.28) + up * rng.uniform(-0.15, 0.25)
    vel = fwd * rng.uniform(2.5, 4.5) + side * rng.uniform(-0.8, 0.8) + up * rng.uniform(-0.5, 1.0)
    shed.emit(t - age, pos, vel, 1.2, rng.uniform(0.003, 0.01), 9.81)
for _ in range(2500):                                      # fine mist hanging around the burst
    age = rng.uniform(0.0, 0.8)
    pos = entry + Vector((rng.gauss(0, 0.6), rng.gauss(0, 0.6), rng.uniform(0.0, 0.4)))
    vel = Vector((rng.gauss(0, 0.6), rng.gauss(0, 0.6), rng.uniform(0.5, 2.5)))
    mist.emit(t - age, pos, vel, 1.5, rng.uniform(0.0012, 0.0035), 3.0)
sheet = P()
for _ in range(220):                                       # heavy sheets of water at the foot of the burst
    age = rng.uniform(0.05, 0.4)
    ang = rng.uniform(0, 2 * math.pi)
    pos = entry + Vector((math.cos(ang) * 0.35, math.sin(ang) * 0.35, 0.05))
    vel = Vector((math.cos(ang), math.sin(ang), 0.0)) * rng.uniform(0.6, 1.8) + up * rng.uniform(1.5, 3.5)
    sheet.emit(t - age, pos, vel, 1.2, rng.uniform(0.01, 0.022), 9.81)
for name, ps in (("ThumbBurst", burst), ("ThumbShed", shed), ("ThumbMist", mist), ("ThumbSheet", sheet)):
    ob = ps.build(name, col, M.spray, streak=(name != "ThumbSheet"))
    ng = ob.modifiers[0].node_group                        # round, smooth drops (the film's low poly ones read as hexes)
    for nd in ng.nodes:
        if nd.bl_idname == "GeometryNodeMeshIcoSphere":
            nd.inputs["Subdivisions"].default_value = 3
    out = next(nd for nd in ng.nodes if nd.bl_idname == "NodeGroupOutput")
    src = out.inputs[0].links[0].from_socket
    sm = ng.nodes.new("GeometryNodeSetShadeSmooth")
    ng.links.new(src, sm.inputs["Geometry"])
    ng.links.new(sm.outputs[0], out.inputs[0])

# foam churned up around the breach (the wake material on a disc lying on the swell)
bm_mod = g["bmesh"]
bm = bm_mod.new()
bm_mod.ops.create_circle(bm, cap_ends=True, segments=48, radius=2.2)
bm_mod.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=3, use_grid_fill=True)
me = bpy.data.meshes.new("ThumbFoam")
bm.to_mesh(me)
bm.free()
rest = me.attributes.new("rest", "FLOAT_VECTOR", "POINT")
rest.data.foreach_set("vector", [c for v in me.vertices for c in (v.co.x * 0.5, v.co.y * 1.6, 0.0)])
foam = bpy.data.objects.new("ThumbFoam", me)
foam.data.materials.append(M.wake)
col.objects.link(foam)
foam.location = (entry.x, entry.y, entry.z + 0.03)
foam.visible_shadow = False
foam.hide_render = True                                  # reads as a flat smear at this angle

# --- camera: low on the swell, off her flank, the boat and the angler behind her ----------------------------------
cam = sc.camera
unkey(cam)
if cam.data.animation_data:
    cam.data.animation_data_clear()
look = root_pos + fwd * 0.1 + Vector((0.0, 0.0, 0.9))          # aimed high: she sits under the headline
cpos = look + side * 5.4 + head * 2.6 - Vector((0.0, 0.0, 0.9))      # off her tail: the boat sits left of her
cpos.z = surface_z(cpos.x, cpos.y, t) + 1.0
cam.location = cpos
cam.rotation_mode = "QUATERNION"
cam.rotation_quaternion = (look - cpos).to_track_quat("-Z", "Y")
cam.data.lens = 32.0
cam.data.dof.use_dof = True
cam.data.dof.focus_distance = (look - cpos).length
cam.data.dof.aperture_fstop = 2.8                          # drops near the lens melt into bokeh
print("camera", tuple(round(c, 2) for c in cpos), "fish", tuple(round(c, 2) for c in root_pos),
      "clearance", round(cpos.z - surface_z(cpos.x, cpos.y, t), 2))

fl = bpy.data.lights.new("ThumbFill", "AREA")               # soft fill from the lens side: her silver flank
fl.size, fl.energy, fl.color = 3.0, 750.0, g["srgb"](214, 228, 240)
flo = bpy.data.objects.new("ThumbFill", fl)
col.objects.link(flo)
flo.location = cpos + Vector((0.0, 0.0, 1.5)) + head * 1.0
flo.rotation_mode = "QUATERNION"
flo.rotation_quaternion = (root_pos - flo.location).to_track_quat("-Z", "Y")
sc.render.use_motion_blur = False
sc.eevee.taa_render_samples = a.samples
sc.render.filepath = os.path.abspath(a.out)
bpy.ops.render.render(write_still=True, scene=sc.name)
print("saved", a.out)
