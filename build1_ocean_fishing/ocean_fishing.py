"""Low-poly golden-hour ocean fishing scene (Blender 5.x, EEVEE).

Run with the Blender binary:
    blender --background --python ocean_fishing.py -- [options]
or with the bpy Python module (pip install bpy):
    python ocean_fishing.py [options]

Options:
    --save PATH        save the generated scene as a .blend file
    --check            print the camera clearance / framing report
    --frames 1,60,240  render these frames as PNG stills into --outdir
    --anim             render the whole animation as PNG frames into --outdir
    --outdir DIR       output directory for renders
    --scale PCT        resolution percentage (default 100 -> 1920x1080)
    --samples N        EEVEE render samples (default 16)
    --step N           frame step for --anim (default 1)

World layout (metres, +Z up):
    * The sun sits low on the +Y horizon.
    * The boat is at the origin, bow yawed 40 degrees toward -X/-Y.
    * The fisherman casts toward the camera side; the bobber lands at
      BOBBER_XY and the hook sinks to HOOK_DEPTH over a coral garden.
    * The camera always stays on the -Y side and looks toward the sun.
"""

import argparse
import math
import os
import random
import sys

import bpy  # must come first: the pip bpy module exposes bmesh/mathutils after import
import bmesh
import numpy as np
from mathutils import Euler, Matrix, Vector, noise

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
FPS = 24
FRAME_END = 240

SUN_ELEV = math.radians(2.2)
SUN_DIR = Vector((0.0, math.cos(SUN_ELEV), math.sin(SUN_ELEV)))

BOAT_YAW = math.radians(-40.0)
BOBBER_XY = Vector((1.0, -5.0))
HOOK_DRIFT = Vector((0.1, -0.15))
HOOK_DEPTH = -4.0

CAST_RELEASE = 28      # frame the bobber leaves the rod
CAST_LAND = 50         # frame the bobber hits the water
HOOK_SETTLED = 150     # frame the hook reaches HOOK_DEPTH

UW_FOG_DIST = 14.0     # underwater visibility falloff (m)
AIR_HAZE_DIST = 450.0  # aerial perspective falloff (m)

WAVE_CENTER = (0.0, -6.0)
FINE_HALF = 108.0      # half size of the detailed wave grid
COARSE_HALF = 702.0    # half size of the coarse swell grid
SEABED_CENTER = (1.0, -6.0)
CURTAIN_R = 100.0      # sealed underwater wall (keeps underwater rays off the sky)

# Camera keys: (frame, camera location, look-at target)
CAM_KEYS = [
    (1,   (-2.72, -5.37, 0.55),  (-1.00, -0.30, 0.85)),
    (36,  (-3.00, -8.00, 1.90),  (0.60, -3.30, 0.50)),
    (72,  (-2.50, -11.50, 3.30), (0.60, -4.40, 0.80)),
    (96,  (-1.40, -9.60, 1.50),  (1.00, -5.00, 0.30)),
    (108, (-1.08, -8.78, 0.20),  (1.00, -5.00, -0.35)),   # level at the waterline
    (120, (-0.90, -8.40, -0.90), (1.05, -5.10, -2.70)),   # then tip down the line
    (144, (-0.60, -8.10, -2.60), (1.10, -5.10, -3.70)),
    (168, (-0.60, -8.30, -3.30), (1.20, -5.00, -3.85)),
    (198, (-0.55, -10.60, -2.10), (0.40, -3.00, -0.60)),
    (218, (-0.10, -13.20, 0.35), (0.20, -2.00, 0.90)),
    (240, (0.25, -15.50, 0.70), (0.25, 0.00, 1.40)),
]

# Colours (linear RGB)
HAZE_SUN = (1.00, 0.50, 0.16)
HAZE_AWAY = (0.45, 0.28, 0.32)
RAMP_SUN = [(0.0, HAZE_SUN), (0.03, (1.00, 0.42, 0.12)), (0.10, (0.95, 0.38, 0.13)),
            (0.22, (0.62, 0.26, 0.19)), (0.40, (0.24, 0.14, 0.24)), (0.75, (0.04, 0.06, 0.18))]
RAMP_AWAY = [(0.0, HAZE_AWAY), (0.08, (0.36, 0.22, 0.28)), (0.30, (0.13, 0.12, 0.25)),
             (0.70, (0.03, 0.05, 0.15))]
WORLD_LIGHT = 0.5      # world contribution to object lighting (sky stays full bright)
GLOW_COL = (1.0, 0.55, 0.20)
DISK_COL = (1.0, 0.80, 0.45)
FOG_DEEP = (0.010, 0.10, 0.13)
FOG_SHALLOW = (0.06, 0.38, 0.40)


def _wave_list(specs, speed):
    """(amplitude, wavelength, direction deg, phase) -> (a, kx, ky, omega, phase)."""
    out = []
    for amp, lam, deg, phase in specs:
        k = 2.0 * math.pi / lam
        omega = math.sqrt(9.81 * k) * speed
        out.append((amp, k * math.cos(math.radians(deg)), k * math.sin(math.radians(deg)), omega, phase))
    return out


WAVES = _wave_list([(0.18, 17.0, 250.0, 0.0),
                    (0.11, 10.5, 215.0, 1.3),
                    (0.06, 6.5, 290.0, 2.1)], speed=0.75)
SWELLS = _wave_list([(0.35, 46.0, 255.0, 0.4),
                     (0.22, 29.0, 225.0, 2.2)], speed=0.6)


def smoothstep(e0, e1, x):
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def wave_z(x, y, t):
    """Height of the detailed ocean surface; mirrors the geometry nodes setup."""
    r = math.hypot(x - WAVE_CENTER[0], y - WAVE_CENTER[1])
    fade = 1.0 - smoothstep(60.0, 95.0, r)
    return fade * sum(a * math.sin(kx * x + ky * y - w * t + ph) for a, kx, ky, w, ph in WAVES)


def seabed_z(x, y):
    r = math.hypot(x - SEABED_CENTER[0], y - SEABED_CENTER[1])
    n1 = noise.noise(Vector((x * 0.09, y * 0.09, 0.5)))
    n2 = noise.noise(Vector((x * 0.31, y * 0.31, 3.7)))
    return -5.35 + 0.55 * n1 + 0.18 * n2 - 0.012 * r + 5.0 * smoothstep(52.0, 92.0, r)


def frame_time(f):
    return f / FPS


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--save", default=None)
    p.add_argument("--check", action="store_true")
    p.add_argument("--frames", default="")
    p.add_argument("--anim", action="store_true")
    p.add_argument("--outdir", default=None)
    p.add_argument("--scale", type=int, default=100)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--step", type=int, default=1)
    return p.parse_args(argv)


def collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def mesh_object(name, bm, col, mats=(), color=None, parent=None, recalc=True):
    if recalc:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    if color is not None:
        ob.color = (*color, 1.0)
    if parent is not None:
        ob.parent = parent
    return ob


def empty_object(name, col, loc=(0, 0, 0), parent=None, size=0.1):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_size = size
    ob.location = loc
    col.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def basis(n):
    n = n.normalized()
    ref = Vector((0, 0, 1)) if abs(n.z) < 0.9 else Vector((1, 0, 0))
    u = n.cross(ref).normalized()
    return u, n.cross(u)


def bm_frustum(bm, p0, p1, r0, r1, segs=6, cap0=True, cap1=True, twist=0.0, squash=None):
    """Tapered prism from p0 to p1. squash=(axis Vector, factor) flattens it."""
    p0, p1 = Vector(p0), Vector(p1)
    u, v = basis(p1 - p0)

    def ring(c, r):
        if r <= 0.0:
            return [bm.verts.new(c)]
        pts = []
        for i in range(segs):
            a = twist + 2.0 * math.pi * i / segs
            off = (u * math.cos(a) + v * math.sin(a)) * r
            if squash is not None:
                ax, fac = squash
                off = off - ax * (off.dot(ax) * (1.0 - fac))
            pts.append(bm.verts.new(c + off))
        return pts

    ra, rb = ring(p0, r0), ring(p1, r1)
    for i in range(segs):
        j = (i + 1) % segs
        if len(rb) == 1:
            bm.faces.new((ra[i], ra[j], rb[0]))
        elif len(ra) == 1:
            bm.faces.new((ra[0], rb[j], rb[i]))
        else:
            bm.faces.new((ra[i], ra[j], rb[j], rb[i]))
    if cap0 and len(ra) > 1:
        bm.faces.new(list(reversed(ra)))
    if cap1 and len(rb) > 1:
        bm.faces.new(rb)


def bm_box(bm, center, size, rot=(0, 0, 0)):
    m = Matrix.LocRotScale(Vector(center), Euler(rot), Vector(size))
    bmesh.ops.create_cube(bm, size=1.0, matrix=m)


def bm_rock(bm, center, size, rng, subdiv=1, jitter=0.22, rot_z=0.0):
    m = Matrix.LocRotScale(Vector(center), Euler((0, 0, rot_z)), Vector(size))
    res = bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0, matrix=Matrix())
    for v in res["verts"]:
        v.co = m @ (v.co * (1.0 + rng.uniform(-jitter, jitter)))


def set_face_color(bm, faces, layer, color):
    for f in faces:
        for loop in f.loops:
            loop[layer] = (*color, 1.0)


def keyframe(ob, path, frame):
    ob.keyframe_insert(data_path=path, frame=frame)


# ---- node helpers ---------------------------------------------------------
def _feed(nt, sock, value):
    if value is None:
        return
    if isinstance(value, (int, float)):
        sock.default_value = value
    elif isinstance(value, (tuple, list, Vector)):
        sock.default_value = tuple(value)
    else:
        nt.links.new(value, sock)


def nmath(nt, op, a=None, b=None, c=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for i, val in enumerate((a, b, c)):
        _feed(nt, n.inputs[i], val)
    return n.outputs[0]


def vmath(nt, op, a=None, b=None, scale=None):
    n = nt.nodes.new("ShaderNodeVectorMath")
    n.operation = op
    _feed(nt, n.inputs[0], a)
    _feed(nt, n.inputs[1], b)
    if scale is not None:
        _feed(nt, n.inputs[3], scale)
    return n.outputs["Value"] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs["Vector"]


def mix_rgb(nt, fac, a, b):
    n = nt.nodes.new("ShaderNodeMix")
    n.data_type = "RGBA"
    _feed(nt, n.inputs[0], fac)
    _feed(nt, n.inputs[6], a if not isinstance(a, tuple) else (*a, 1.0))
    _feed(nt, n.inputs[7], b if not isinstance(b, tuple) else (*b, 1.0))
    return n.outputs[2]


def map_range(nt, value, fmin, fmax, tmin=0.0, tmax=1.0, smooth=True):
    n = nt.nodes.new("ShaderNodeMapRange")
    n.interpolation_type = "SMOOTHSTEP" if smooth else "LINEAR"
    _feed(nt, n.inputs["Value"], value)
    n.inputs["From Min"].default_value = fmin
    n.inputs["From Max"].default_value = fmax
    n.inputs["To Min"].default_value = tmin
    n.inputs["To Max"].default_value = tmax
    return n.outputs["Result"]


def separate(nt, vec):
    n = nt.nodes.new("ShaderNodeSeparateXYZ")
    _feed(nt, n.inputs[0], vec)
    return n.outputs


def combine(nt, x=0.0, y=0.0, z=0.0):
    n = nt.nodes.new("ShaderNodeCombineXYZ")
    for i, val in enumerate((x, y, z)):
        _feed(nt, n.inputs[i], val)
    return n.outputs[0]


def color_ramp(nt, value, stops):
    n = nt.nodes.new("ShaderNodeValToRGB")
    _feed(nt, n.inputs[0], value)
    els = n.color_ramp.elements
    els[0].position, els[0].color = stops[0][0], (*stops[0][1], 1.0)
    els[1].position, els[1].color = stops[-1][0], (*stops[-1][1], 1.0)
    for pos, col in stops[1:-1]:
        e = els.new(pos)
        e.color = (*col, 1.0)
    return n.outputs["Color"]


def new_group(name, kind, inputs, outputs):
    ng = bpy.data.node_groups.new(name, kind)
    for nm, st in inputs:
        ng.interface.new_socket(nm, in_out="INPUT", socket_type=st)
    for nm, st in outputs:
        ng.interface.new_socket(nm, in_out="OUTPUT", socket_type=st)
    gi = ng.nodes.new("NodeGroupInput")
    go = ng.nodes.new("NodeGroupOutput")
    return ng, gi, go


# --------------------------------------------------------------------------
# Shared shading: sky colour, atmosphere (air haze + underwater fog)
# --------------------------------------------------------------------------
def build_sky_group():
    ng, gi, go = new_group("SkyColor", "ShaderNodeTree",
                           [("Direction", "NodeSocketVector")],
                           [("Sky", "NodeSocketColor"), ("Haze", "NodeSocketColor")])
    d = vmath(ng, "NORMALIZE", gi.outputs["Direction"])
    z = separate(ng, d)["Z"]
    s = vmath(ng, "DOT_PRODUCT", d, tuple(SUN_DIR))
    sun_side = nmath(ng, "POWER", nmath(ng, "MULTIPLY_ADD", s, 0.5, 0.5, clamp=True), 3.0)
    sky_base = mix_rgb(ng, sun_side, color_ramp(ng, z, RAMP_AWAY), color_ramp(ng, z, RAMP_SUN))
    haze = mix_rgb(ng, sun_side, HAZE_AWAY, HAZE_SUN)
    g = nmath(ng, "MAXIMUM", s, 0.0)
    glow = nmath(ng, "ADD",
                 nmath(ng, "MULTIPLY", nmath(ng, "POWER", g, 5.0), 0.18),
                 nmath(ng, "ADD",
                       nmath(ng, "MULTIPLY", nmath(ng, "POWER", g, 40.0), 0.5),
                       nmath(ng, "MULTIPLY", nmath(ng, "POWER", g, 400.0), 1.4)))
    glow_rgb = vmath(ng, "SCALE", GLOW_COL, scale=glow)
    horizon = map_range(ng, z, -0.004, 0.004)
    disk = nmath(ng, "MULTIPLY",
                 map_range(ng, g, math.cos(math.radians(1.5)), math.cos(math.radians(1.2))),
                 nmath(ng, "MULTIPLY", horizon, 12.0))
    sky = mix_rgb(ng, horizon, haze, sky_base)
    sky = vmath(ng, "ADD", sky, glow_rgb)
    sky = vmath(ng, "ADD", sky, vmath(ng, "SCALE", DISK_COL, scale=disk))
    ng.links.new(sky, go.inputs["Sky"])
    ng.links.new(vmath(ng, "ADD", haze, glow_rgb), go.inputs["Haze"])
    return ng


def build_atmos_group(sky_group):
    """Wraps a surface shader with air haze (camera above water) or
    blue-green fog (camera below water). Whether the camera is under the wave
    surface is a per-frame scene property ("cam_under") keyed by
    key_camera_underwater()."""
    ng, gi, go = new_group("Atmosphere", "ShaderNodeTree",
                           [("Shader", "NodeSocketShader")], [("Shader", "NodeSocketShader")])
    geo = ng.nodes.new("ShaderNodeNewGeometry")
    cam = ng.nodes.new("ShaderNodeCameraData")
    dist = cam.outputs["View Distance"]
    inc = separate(ng, geo.outputs["Incoming"])
    attr = ng.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "VIEW_LAYER"
    attr.attribute_name = "cam_under"
    under = attr.outputs["Factor"]
    above = nmath(ng, "SUBTRACT", 1.0, under)
    fog = nmath(ng, "MULTIPLY", under,
                nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(ng, "MULTIPLY", dist, -1.0 / UW_FOG_DIST))))
    haze = nmath(ng, "MULTIPLY", above,
                 nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(ng, "MULTIPLY", dist, -1.0 / AIR_HAZE_DIST))))
    sky = ng.nodes.new("ShaderNodeGroup")
    sky.node_tree = sky_group
    ng.links.new(vmath(ng, "SCALE", geo.outputs["Incoming"], scale=-1.0), sky.inputs["Direction"])
    em_haze = ng.nodes.new("ShaderNodeEmission")
    ng.links.new(sky.outputs["Haze"], em_haze.inputs["Color"])
    fog_col = mix_rgb(ng, nmath(ng, "MULTIPLY_ADD", inc["Z"], -0.5, 0.5, clamp=True), FOG_DEEP, FOG_SHALLOW)
    em_fog = ng.nodes.new("ShaderNodeEmission")
    ng.links.new(fog_col, em_fog.inputs["Color"])
    m1 = ng.nodes.new("ShaderNodeMixShader")
    ng.links.new(haze, m1.inputs[0])
    ng.links.new(gi.outputs["Shader"], m1.inputs[1])
    ng.links.new(em_haze.outputs[0], m1.inputs[2])
    m2 = ng.nodes.new("ShaderNodeMixShader")
    ng.links.new(fog, m2.inputs[0])
    ng.links.new(m1.outputs[0], m2.inputs[1])
    ng.links.new(em_fog.outputs[0], m2.inputs[2])
    ng.links.new(m2.outputs[0], go.inputs["Shader"])
    return ng


class Materials:
    pass


def new_material(name, atmos, surface_fn):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    shader = surface_fn(nt)
    grp = nt.nodes.new("ShaderNodeGroup")
    grp.node_tree = atmos
    nt.links.new(shader, grp.inputs["Shader"])
    nt.links.new(grp.outputs["Shader"], out.inputs["Surface"])
    m.use_backface_culling = False
    return m


def principled(nt, color, rough=0.75, spec=0.35, emission=None, emit_strength=0.0):
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _feed(nt, b.inputs["Base Color"], (*color, 1.0) if isinstance(color, tuple) else color)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = spec
    if emission is not None:
        b.inputs["Emission Color"].default_value = (*emission, 1.0)
        b.inputs["Emission Strength"].default_value = emit_strength
    return b.outputs["BSDF"]


def build_materials(atmos):
    M = Materials()

    def obj_color(nt):
        return nt.nodes.new("ShaderNodeObjectInfo").outputs["Color"]

    def attr_color(nt):
        a = nt.nodes.new("ShaderNodeAttribute")
        a.attribute_name = "Col"
        return a.outputs["Color"]

    M.painted = new_material("Painted", atmos, lambda nt: principled(nt, obj_color(nt), 0.7))
    M.terrain = new_material("IslandTerrain", atmos, lambda nt: principled(nt, attr_color(nt), 0.9, 0.2))
    M.wood_out = new_material("WoodOuter", atmos, lambda nt: principled(nt, (0.40, 0.21, 0.09), 0.65))
    M.wood_in = new_material("WoodInner", atmos, lambda nt: principled(nt, (0.52, 0.32, 0.16), 0.7))
    M.wood_rim = new_material("WoodRim", atmos, lambda nt: principled(nt, (0.20, 0.10, 0.04), 0.6))
    M.line = new_material("FishingLine", atmos,
                          lambda nt: principled(nt, (0.95, 0.92, 0.82), 0.4, 0.5, (1.0, 0.95, 0.85), 0.8))

    def underwater(nt):
        # Stylised underwater lighting: soft light from above, no lamps needed.
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        ndl = vmath(nt, "DOT_PRODUCT", geo.outputs["Normal"], tuple(Vector((0.1, 0.35, 1.0)).normalized()))
        lit = nmath(nt, "MULTIPLY_ADD", nmath(nt, "MAXIMUM", ndl, 0.0), 0.85, 0.20)
        col = vmath(nt, "SCALE", obj_color(nt), scale=lit)
        col = vmath(nt, "MULTIPLY", col, (0.85, 1.0, 0.97))
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(col, em.inputs["Color"])
        return em.outputs[0]

    M.underwater = new_material("UnderwaterLife", atmos, underwater)

    def water(nt):
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        top = principled(nt, (0.012, 0.085, 0.10), 0.13, 0.6, (0.010, 0.060, 0.070), 1.0)
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.5
        under_col = mix_rgb(nt, lw.outputs["Facing"], (0.30, 0.80, 0.75), (0.05, 0.30, 0.32))
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(under_col, em.inputs["Color"])
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(geo.outputs["Backfacing"], mix.inputs[0])
        nt.links.new(top, mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]

    M.water = new_material("OceanWater", atmos, water)
    return M


def build_world(sky_group):
    world = bpy.data.worlds.new("GoldenHourSky")
    bpy.context.scene.world = world
    try:
        world.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = world.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sky = nt.nodes.new("ShaderNodeGroup")
    sky.node_tree = sky_group
    nt.links.new(tc.outputs["Generated"], sky.inputs["Direction"])
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(sky.outputs["Sky"], bg.inputs["Color"])
    lp = nt.nodes.new("ShaderNodeLightPath")
    nt.links.new(nmath(nt, "MULTIPLY_ADD", lp.outputs["Is Camera Ray"], 1.0 - WORLD_LIGHT, WORLD_LIGHT),
                 bg.inputs["Strength"])
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    world.sun_threshold = 1000.0  # the sun lamp does the lighting; no extraction


def build_sun(col):
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 4.0
    light.color = (1.0, 0.62, 0.36)
    light.angle = math.radians(1.5)
    ob = bpy.data.objects.new("Sun", light)
    col.objects.link(ob)
    ob.rotation_euler = (-(math.pi / 2 - SUN_ELEV), 0.0, 0.0)
    return ob


# --------------------------------------------------------------------------
# Ocean
# --------------------------------------------------------------------------
def gn_wave_sum(nt, waves, x, y, t):
    total = None
    for a, kx, ky, w, ph in waves:
        arg = nmath(nt, "MULTIPLY_ADD", x, kx, nmath(nt, "MULTIPLY_ADD", y, ky, nmath(nt, "MULTIPLY_ADD", t, -w, ph)))
        term = nmath(nt, "MULTIPLY", nmath(nt, "SINE", arg), a)
        total = term if total is None else nmath(nt, "ADD", total, term)
    return total


def build_ocean(col, mat):
    ng, gi, go = new_group("OceanSurface", "GeometryNodeTree",
                           [("Geometry", "NodeSocketGeometry")], [("Geometry", "NodeSocketGeometry")])
    try:
        ng.is_modifier = True
    except AttributeError:
        pass
    nodes, links = ng.nodes, ng.links
    centre = (WAVE_CENTER[0], WAVE_CENTER[1], 0.0)

    t = nmath(ng, "DIVIDE", nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], float(FPS))

    def rel_xy():
        p = nodes.new("GeometryNodeInputPosition").outputs[0]
        return p, separate(ng, vmath(ng, "SUBTRACT", p, centre))

    # Fine faceted waves near the action
    fine = nodes.new("GeometryNodeMeshGrid")
    fine.inputs["Size X"].default_value = fine.inputs["Size Y"].default_value = 2 * FINE_HALF
    fine.inputs["Vertices X"].default_value = fine.inputs["Vertices Y"].default_value = int(2 * FINE_HALF) + 1
    tr1 = nodes.new("GeometryNodeTransform")
    links.new(fine.outputs["Mesh"], tr1.inputs["Geometry"])
    tr1.inputs["Translation"].default_value = centre
    tri1 = nodes.new("GeometryNodeTriangulate")
    links.new(tr1.outputs[0], tri1.inputs["Mesh"])
    p, rel = rel_xy()
    pos = separate(ng, p)
    r = vmath(ng, "LENGTH", combine(ng, rel["X"], rel["Y"], 0.0))
    fade = map_range(ng, r, 60.0, 95.0, 1.0, 0.0)
    h = nmath(ng, "MULTIPLY", gn_wave_sum(ng, WAVES, pos["X"], pos["Y"], t), fade)
    sp1 = nodes.new("GeometryNodeSetPosition")
    links.new(tri1.outputs[0], sp1.inputs["Geometry"])
    links.new(combine(ng, 0.0, 0.0, h), sp1.inputs["Offset"])

    # Coarse swell ring out to the horizon (hole where the fine grid is)
    coarse = nodes.new("GeometryNodeMeshGrid")
    coarse.inputs["Size X"].default_value = coarse.inputs["Size Y"].default_value = 2 * COARSE_HALF
    coarse.inputs["Vertices X"].default_value = coarse.inputs["Vertices Y"].default_value = int(2 * COARSE_HALF / 6) + 1
    tr2 = nodes.new("GeometryNodeTransform")
    links.new(coarse.outputs["Mesh"], tr2.inputs["Geometry"])
    tr2.inputs["Translation"].default_value = centre
    dele = nodes.new("GeometryNodeDeleteGeometry")
    dele.domain = "FACE"
    links.new(tr2.outputs[0], dele.inputs["Geometry"])
    _, rel = rel_xy()
    inside = nmath(ng, "MULTIPLY",
                   nmath(ng, "LESS_THAN", nmath(ng, "ABSOLUTE", rel["X"]), FINE_HALF),
                   nmath(ng, "LESS_THAN", nmath(ng, "ABSOLUTE", rel["Y"]), FINE_HALF))
    links.new(inside, dele.inputs["Selection"])
    tri2 = nodes.new("GeometryNodeTriangulate")
    links.new(dele.outputs[0], tri2.inputs["Mesh"])
    p, rel = rel_xy()
    pos = separate(ng, p)
    cheb = nmath(ng, "MAXIMUM", nmath(ng, "ABSOLUTE", rel["X"]), nmath(ng, "ABSOLUTE", rel["Y"]))
    sfade = nmath(ng, "MULTIPLY", map_range(ng, cheb, FINE_HALF, 190.0), map_range(ng, cheb, 520.0, 690.0, 1.0, 0.0))
    h2 = nmath(ng, "MULTIPLY", gn_wave_sum(ng, SWELLS, pos["X"], pos["Y"], t), sfade)
    sp2 = nodes.new("GeometryNodeSetPosition")
    links.new(tri2.outputs[0], sp2.inputs["Geometry"])
    links.new(combine(ng, 0.0, 0.0, h2), sp2.inputs["Offset"])

    join = nodes.new("GeometryNodeJoinGeometry")
    links.new(sp2.outputs[0], join.inputs[0])
    links.new(sp1.outputs[0], join.inputs[0])
    flat = nodes.new("GeometryNodeSetShadeSmooth")
    links.new(join.outputs[0], flat.inputs[0])
    flat.inputs["Shade Smooth"].default_value = False
    setm = nodes.new("GeometryNodeSetMaterial")
    links.new(flat.outputs[0], setm.inputs["Geometry"])
    setm.inputs["Material"].default_value = mat
    links.new(setm.outputs[0], go.inputs["Geometry"])

    me = bpy.data.meshes.new("Ocean")
    ob = bpy.data.objects.new("Ocean", me)
    col.objects.link(ob)
    mod = ob.modifiers.new("OceanSurface", "NODES")
    mod.node_group = ng
    ob.visible_shadow = False

    # Flat far ring out past the visible horizon
    bm = bmesh.new()
    cx, cy = WAVE_CENTER
    inner = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    vi = [bm.verts.new((cx + sx * COARSE_HALF, cy + sy * COARSE_HALF, 0.0)) for sx, sy in inner]
    vo = [bm.verts.new((cx + sx * 9000.0, cy + sy * 9000.0, 0.0)) for sx, sy in inner]
    for i in range(4):
        j = (i + 1) % 4
        bm.faces.new((vi[i], vo[i], vo[j], vi[j]))
    far = mesh_object("OceanFar", bm, col, [mat])
    for poly in far.data.polygons:
        if poly.normal.z < 0:
            poly.flip()
    far.visible_shadow = False

    # Underwater curtain: a ring wall just below the flat water, so rays from an
    # underwater camera always end in fogged geometry instead of the sunset sky.
    bm = bmesh.new()
    seg = 48
    top = [bm.verts.new((cx + CURTAIN_R * math.cos(2 * math.pi * i / seg),
                         cy + CURTAIN_R * math.sin(2 * math.pi * i / seg), 0.005)) for i in range(seg)]
    bot = [bm.verts.new((v.co.x, v.co.y, -12.0)) for v in top]
    for i in range(seg):
        j = (i + 1) % seg
        bm.faces.new((top[i], top[j], bot[j], bot[i]))
    curtain = mesh_object("UnderwaterCurtain", bm, col, [mat], recalc=False)
    curtain.visible_shadow = False
    return ob


def build_seabed(col, mat):
    bm = bmesh.new()
    half, n = 100.0, 161
    cx, cy = SEABED_CENTER
    grid = []
    for j in range(n):
        row = []
        for i in range(n):
            x = cx - half + 2 * half * i / (n - 1)
            y = cy - half + 2 * half * j / (n - 1)
            row.append(bm.verts.new((x, y, seabed_z(x, y))))
        grid.append(row)
    for j in range(n - 1):
        for i in range(n - 1):
            bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
    bmesh.ops.triangulate(bm, faces=bm.faces)
    ob = mesh_object("SeaFloor", bm, col, [mat], color=(0.55, 0.46, 0.29))
    return ob


# --------------------------------------------------------------------------
# Boat + fisherman
# --------------------------------------------------------------------------
# y, gunwale half-width, gunwale z, chine half-width, chine z, keel z (boat local)
HULL = [
    (2.00, 0.60, 0.46, 0.50, -0.14, -0.24),
    (1.30, 0.74, 0.44, 0.62, -0.20, -0.32),
    (0.40, 0.79, 0.44, 0.66, -0.22, -0.35),
    (-0.50, 0.76, 0.46, 0.61, -0.20, -0.34),
    (-1.30, 0.60, 0.53, 0.43, -0.12, -0.26),
    (-1.85, 0.34, 0.62, 0.20, 0.00, -0.12),
    (-2.15, 0.07, 0.74, 0.05, 0.22, 0.10),
]
FLOOR_Z = 0.14
SEAT_Z = 0.34


def hull_half_width(y, z):
    """Interior-ish half width of the hull at local (y, z)."""
    def at(sec):
        _, w, h, c, zc, k = sec
        if z >= zc:
            return c + (w - c) * (z - zc) / (h - zc)
        return c * (z - k) / max(zc - k, 1e-6)
    for a, b in zip(HULL, HULL[1:]):
        if b[0] <= y <= a[0]:
            f = (y - a[0]) / (b[0] - a[0])
            return at(a) * (1 - f) + at(b) * f
    return 0.0


def build_boat(col, M):
    boat = empty_object("Boat", col, size=0.5)
    boat.rotation_mode = "XYZ"

    # Hull shell (solidified)
    bm = bmesh.new()
    secs = []
    for y, w, h, c, zc, k in HULL:
        secs.append([bm.verts.new(p) for p in ((-w, y, h), (-c, y, zc), (0, y, k), (c, y, zc), (w, y, h))])
    for a, b in zip(secs, secs[1:]):
        for j in range(4):
            bm.faces.new((a[j], a[j + 1], b[j + 1], b[j]))
    bm.faces.new(secs[0])  # transom
    bm.faces.new(secs[-1])  # narrow bow plate (keeps the solidified stem clean)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    keel_face = min(bm.faces, key=lambda f: f.calc_center_median().z)
    if keel_face.normal.z > 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    hull = mesh_object("Hull", bm, col, [M.wood_out, M.wood_in, M.wood_rim], parent=boat, recalc=False)
    sol = hull.modifiers.new("Planks", "SOLIDIFY")
    sol.thickness = 0.05
    sol.offset = -1.0
    sol.use_even_offset = True
    sol.material_offset = 1
    sol.material_offset_rim = 2

    # Floor boards seal the hull above the waterline
    bm = bmesh.new()
    left, right = [], []
    for y, w, h, c, zc, k in HULL[:-1]:
        hw = hull_half_width(y, FLOOR_Z) - 0.03
        left.append(bm.verts.new((-hw, y, FLOOR_Z)))
        right.append(bm.verts.new((hw, y, FLOOR_Z)))
    bm.faces.new(left + list(reversed(right)))
    mesh_object("Floor", bm, col, [M.painted], color=(0.46, 0.28, 0.13), parent=boat)

    # Thwarts (seats) and props
    bm = bmesh.new()
    for sy in (1.55, 0.46, -1.0):
        hw = min(hull_half_width(sy + d, SEAT_Z) for d in (-0.13, 0.0, 0.13)) - 0.06
        bm_box(bm, (0, sy, SEAT_Z - 0.025), (2 * hw, 0.26, 0.05))
    mesh_object("Seats", bm, col, [M.wood_in], parent=boat)

    bm = bmesh.new()
    bm_frustum(bm, (-0.38, 1.25, FLOOR_Z), (-0.38, 1.25, FLOOR_Z + 0.26), 0.12, 0.15, segs=7)
    mesh_object("Bucket", bm, col, [M.painted], color=(0.20, 0.32, 0.45), parent=boat)
    bm = bmesh.new()
    bm_box(bm, (0.36, 1.18, FLOOR_Z + 0.08), (0.22, 0.34, 0.16))
    mesh_object("TackleBox", bm, col, [M.painted], color=(0.16, 0.38, 0.20), parent=boat)
    bm = bmesh.new()
    bm_box(bm, (-0.36, 0.05, FLOOR_Z + 0.03), (0.05, 1.8, 0.035), rot=(0, 0.1, 0))
    bm_box(bm, (-0.36, -0.98, FLOOR_Z + 0.03), (0.12, 0.36, 0.025), rot=(0, 0.1, 0))
    mesh_object("Oar", bm, col, [M.painted], color=(0.55, 0.38, 0.20), parent=boat)
    return boat


def build_fisherman(col, M, boat):
    def part(name, fn, color):
        bm = bmesh.new()
        fn(bm)
        return mesh_object(name, bm, col, [M.painted], color=color, parent=boat)

    shirt, pants, skin = (0.62, 0.11, 0.07), (0.07, 0.11, 0.24), (0.72, 0.45, 0.30)
    hip = Vector((-0.10, 0.46, 0.40))
    part("Torso", lambda bm: bm_frustum(bm, hip, (-0.05, 0.46, 0.86), 0.17, 0.19, segs=6,
                                         squash=(Vector((1, 0, 0)), 0.72)), shirt)
    part("Head", lambda bm: bm_rock(bm, (-0.03, 0.46, 1.01), (0.11, 0.11, 0.12), random.Random(3), jitter=0.04), skin)
    part("Hat", lambda bm: (bm_frustum(bm, (-0.03, 0.46, 1.085), (-0.03, 0.46, 1.11), 0.25, 0.24, segs=10),
                            bm_frustum(bm, (-0.03, 0.46, 1.11), (-0.03, 0.46, 1.23), 0.13, 0.11, segs=8)),
         (0.86, 0.66, 0.30))

    def legs(bm):
        for s in (-1, 1):
            y = 0.46 + 0.10 * s
            bm_frustum(bm, (-0.08, y, 0.40), (0.30, y, 0.43), 0.075, 0.065, segs=6)
            bm_frustum(bm, (0.30, y, 0.43), (0.36, y, 0.19), 0.062, 0.05, segs=6)
    part("Legs", legs, pants)
    part("Boots", lambda bm: [bm_box(bm, (0.40, 0.46 + 0.10 * s, FLOOR_Z + 0.045), (0.20, 0.10, 0.09))
                              for s in (-1, 1)], (0.09, 0.07, 0.05))

    # Rod pivot (in the right hand) and cast animation
    pivot = empty_object("RodPivot", col, parent=boat, size=0.15)
    pivot.rotation_mode = "XYZ"
    target_local = Matrix.Rotation(-BOAT_YAW, 3, "Z") @ Vector((BOBBER_XY.x, BOBBER_XY.y, 0.0))
    yaw = math.atan2(target_local.y - 0.27, target_local.x - 0.30)
    cast = [  # frame, elevation deg, pivot location
        (1, 50, (0.30, 0.27, 0.64)),
        (10, 50, (0.30, 0.27, 0.64)),
        (22, 100, (0.22, 0.26, 0.86)),
        (30, 12, (0.42, 0.25, 0.70)),
        (42, 28, (0.30, 0.27, 0.64)),
        (130, 25, (0.30, 0.27, 0.63)),
        (240, 29, (0.30, 0.27, 0.65)),
    ]
    for f, elev, loc in cast:
        pivot.location = loc
        pivot.rotation_euler = (0.0, -math.radians(elev), yaw)
        keyframe(pivot, "location", f)
        keyframe(pivot, "rotation_euler", f)

    bm = bmesh.new()
    bm_frustum(bm, (-0.30, 0, 0), (0.25, 0, 0), 0.026, 0.022, segs=6)
    bm_frustum(bm, (0.25, 0, 0), (2.30, 0, 0), 0.016, 0.005, segs=5)
    mesh_object("Rod", bm, col, [M.painted], color=(0.12, 0.08, 0.06), parent=pivot)
    bm = bmesh.new()
    bm_frustum(bm, (0.02, -0.03, -0.06), (0.02, 0.03, -0.06), 0.045, 0.045, segs=8)
    mesh_object("Reel", bm, col, [M.painted], color=(0.55, 0.56, 0.58), parent=pivot)
    tip = empty_object("RodTip", col, loc=(2.30, 0, 0), parent=pivot, size=0.05)

    # Arms: unit-length prisms stretched from the shoulder to the hand
    hand_r = empty_object("HandR", col, loc=(0.0, 0.0, 0.0), parent=pivot, size=0.05)
    hand_l = empty_object("HandL", col, loc=(0.18, 0.58, 0.48), parent=boat, size=0.05)
    for name, shoulder, hand in (("ArmR", (-0.04, 0.27, 0.82), hand_r), ("ArmL", (-0.04, 0.65, 0.82), hand_l)):
        bm = bmesh.new()
        bm_frustum(bm, (0, 0, 0), (0, 1, 0), 0.055, 0.045, segs=6)
        bm_rock(bm, (0, 1.0, 0), (0.05, 0.05, 0.05), random.Random(5), subdiv=1, jitter=0.0)
        arm = mesh_object(name, bm, col, [M.painted], color=shirt, parent=boat)
        arm.location = shoulder
        con = arm.constraints.new("STRETCH_TO")
        con.target = hand
        con.rest_length = 1.0
        con.volume = "NO_VOLUME"
    return tip


def build_tackle(col, M, tip, boat):
    """Bobber, hook, splash and the two line segments, all animated."""
    scene = bpy.context.scene
    bobber = empty_object("Bobber", col, size=0.08)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=4, radius=0.09)
    for v in bm.verts:
        if v.co.z < -1e-4:
            v.co.z = 0.0
    mesh_object("BobberTop", bm, col, [M.painted], color=(0.85, 0.05, 0.03), parent=bobber)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=4, radius=0.09)
    for v in bm.verts:
        if v.co.z > 1e-4:
            v.co.z = 0.0
    mesh_object("BobberBottom", bm, col, [M.painted], color=(0.9, 0.9, 0.88), parent=bobber)

    hook = empty_object("Hook", col, size=0.05)
    bm = bmesh.new()
    bm_frustum(bm, (0, 0, 0), (0, 0, -0.07), 0.0, 0.045, segs=4, cap0=False, cap1=False)
    bm_frustum(bm, (0, 0, -0.07), (0, 0, -0.22), 0.045, 0.0, segs=4, cap0=False, cap1=False)
    mesh_object("Lure", bm, col, [M.underwater], color=(1.0, 0.12, 0.04), parent=hook)
    bm = bmesh.new()
    bm_frustum(bm, (0, 0, -0.22), (0, 0, -0.29), 0.008, 0.008, segs=4)
    bm_frustum(bm, (0, 0, -0.29), (0.05, 0, -0.27), 0.008, 0.006, segs=4)
    bm_frustum(bm, (0.05, 0, -0.27), (0.05, 0, -0.22), 0.006, 0.0, segs=4)
    mesh_object("HookBarb", bm, col, [M.underwater], color=(0.85, 0.88, 0.9), parent=hook)
    mid = empty_object("LineSag", col, size=0.05)

    # Splash droplets
    rng = random.Random(11)
    drops = []
    for i in range(10):
        bm = bmesh.new()
        bm_rock(bm, (0, 0, 0), (0.035, 0.035, 0.05), rng, jitter=0.1)
        d = mesh_object(f"Splash{i}", bm, col, [M.painted], color=(0.85, 0.95, 0.95))
        ang = 2 * math.pi * i / 10 + rng.uniform(-0.2, 0.2)
        drops.append((d, Vector((math.cos(ang), math.sin(ang), 0)) * rng.uniform(0.6, 1.1), rng.uniform(1.8, 2.6)))

    # Sample the rod tip over time (the boat bobs, the rod casts)
    tips = {}
    for f in range(1, FRAME_END + 1):
        scene.frame_set(f)
        tips[f] = tip.matrix_world.translation.copy()

    land = Vector((BOBBER_XY.x, BOBBER_XY.y, wave_z(BOBBER_XY.x, BOBBER_XY.y, frame_time(CAST_LAND))))
    hang = Vector((0, 0, -0.55))
    p0 = tips[CAST_RELEASE - 2] + hang
    apex = max(p0.z, land.z) + 1.0
    for f in range(1, FRAME_END + 1):
        t = frame_time(f)
        tp = tips[f]
        if f <= CAST_RELEASE:
            b = tips[max(1, f - 2)] + hang          # hangs and trails the tip
            sag = 0.0
            h = b + Vector((0, 0, -0.35))
        elif f < CAST_LAND:
            s = (f - CAST_RELEASE) / (CAST_LAND - CAST_RELEASE)
            b = p0.lerp(land, s)
            b.z = p0.z + (land.z - p0.z) * s + 4.0 * (apex - max(p0.z, land.z)) * s * (1 - s)
            sag = 0.12
            h = b + Vector((0, 0, -0.35))
        else:
            k = f - CAST_LAND
            dip = -0.10 * math.exp(-k / 5.0) * math.cos(k * 0.6)
            b = Vector((BOBBER_XY.x, BOBBER_XY.y, wave_z(BOBBER_XY.x, BOBBER_XY.y, t) + dip))
            sag = 0.12 + 0.6 * smoothstep(0, 20, k)
            u = min(1.0, k / (HOOK_SETTLED - CAST_LAND))
            e = 1.0 - (1.0 - u) ** 2.5
            xy = BOBBER_XY + HOOK_DRIFT * e
            depth = -0.35 + (HOOK_DEPTH + 0.35) * e + 0.05 * math.sin(t * 1.3) * u
            h = Vector((xy.x, xy.y, depth))
        bobber.location = b
        hook.location = h
        mid.location = (tp + b) * 0.5 + Vector((0, 0, -2.0 * sag))
        for ob in (bobber, hook, mid):
            keyframe(ob, "location", f)

    for d, vel, vz in drops:
        d.hide_render = True
        keyframe(d, "hide_render", 1)
        d.hide_render = False
        keyframe(d, "hide_render", CAST_LAND)
        for k in range(0, 15):
            tt = k / FPS
            d.location = land + vel * tt * 1.4 + Vector((0, 0, vz * tt - 4.9 * tt * tt + 0.05))
            keyframe(d, "location", CAST_LAND + k)
        d.hide_render = True
        keyframe(d, "hide_render", CAST_LAND + 15)

    # Line geometry (geometry nodes reading the empties)
    def line_object(name, start, middle, end, radius):
        ng, gi, go = new_group(name + "GN", "GeometryNodeTree",
                               [("Geometry", "NodeSocketGeometry")], [("Geometry", "NodeSocketGeometry")])
        nodes, links = ng.nodes, ng.links

        def loc(ob):
            n = nodes.new("GeometryNodeObjectInfo")
            n.transform_space = "RELATIVE"
            n.inputs["Object"].default_value = ob
            return n.outputs["Location"]
        if middle is not None:
            c = nodes.new("GeometryNodeCurveQuadraticBezier")
            c.inputs["Resolution"].default_value = 40
            links.new(loc(start), c.inputs["Start"])
            links.new(loc(middle), c.inputs["Middle"])
            links.new(loc(end), c.inputs["End"])
        else:
            c = nodes.new("GeometryNodeCurvePrimitiveLine")
            links.new(loc(start), c.inputs["Start"])
            links.new(loc(end), c.inputs["End"])
        prof = nodes.new("GeometryNodeCurvePrimitiveCircle")
        prof.inputs["Resolution"].default_value = 5
        prof.inputs["Radius"].default_value = radius
        c2m = nodes.new("GeometryNodeCurveToMesh")
        links.new(c.outputs["Curve"], c2m.inputs["Curve"])
        links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
        setm = nodes.new("GeometryNodeSetMaterial")
        setm.inputs["Material"].default_value = M.line
        links.new(c2m.outputs[0], setm.inputs["Geometry"])
        links.new(setm.outputs[0], go.inputs["Geometry"])
        ob = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        col.objects.link(ob)
        ob.modifiers.new(name, "NODES").node_group = ng
        return ob

    line_object("LineAir", tip, mid, bobber, 0.006)
    line_object("LineWater", bobber, None, hook, 0.004)
    return bobber, hook


def animate_boat(boat):
    """Bob, pitch and roll with the analytic wave field."""
    ax_y = Vector((-math.sin(BOAT_YAW), math.cos(BOAT_YAW)))   # boat local +Y in world
    ax_x = Vector((math.cos(BOAT_YAW), math.sin(BOAT_YAW)))    # boat local +X in world
    for f in range(1, FRAME_END + 1):
        t = frame_time(f)
        def slope(ax, e):
            return (wave_z(ax.x * e, ax.y * e, t) - wave_z(-ax.x * e, -ax.y * e, t)) / (2 * e)
        boat.location = (0.0, 0.0, 0.85 * wave_z(0.0, 0.0, t) - 0.02)
        boat.rotation_euler = (math.atan(slope(ax_y, 1.6)) * 0.75, -math.atan(slope(ax_x, 0.6)) * 0.6, BOAT_YAW)
        keyframe(boat, "location", f)
        keyframe(boat, "rotation_euler", f)


# --------------------------------------------------------------------------
# Islands
# --------------------------------------------------------------------------
def add_pine(bm, base, height, layer, rng):
    trunk_col, needle_cols = (0.18, 0.10, 0.05), [(0.04, 0.13, 0.05), (0.05, 0.16, 0.06), (0.03, 0.11, 0.05)]
    n0 = len(bm.faces)
    bm_frustum(bm, base, base + Vector((0, 0, 0.3 * height)), 0.05 * height, 0.035 * height, segs=5)
    set_face_color(bm, list(bm.faces)[n0:], layer, trunk_col)
    for i in range(3):
        n0 = len(bm.faces)
        z0 = (0.2 + 0.24 * i) * height
        bm_frustum(bm, base + Vector((0, 0, z0)), base + Vector((0, 0, z0 + 0.42 * height)),
                   (0.30 - 0.07 * i) * height, 0.0, segs=7, twist=rng.uniform(0, 1))
        set_face_color(bm, list(bm.faces)[n0:], layer, rng.choice(needle_cols))


def build_island(col, M, name, cx, cy, radius, height, seed):
    rng = random.Random(seed)
    bm = bmesh.new()
    layer = bm.loops.layers.float_color.new("Col")
    profile = [(-0.15, 1.10), (0.0, 1.0), (0.22, 0.96), (0.40, 0.86), (0.48, 0.70),
               (0.62, 0.55), (0.78, 0.38), (0.92, 0.18)]
    nseg = 12
    ph1, ph2 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
    rings = []
    for hf, rf in profile:
        ring = []
        for i in range(nseg):
            a = 2 * math.pi * i / nseg + rng.uniform(-0.12, 0.12)
            shape = 1.0 + 0.18 * math.sin(2 * a + ph1) + 0.08 * math.sin(3 * a + ph2)
            r = radius * rf * shape * (1.0 + rng.uniform(-0.12, 0.12))
            z = height * hf + (rng.uniform(-0.03, 0.03) * height if hf > 0.05 else 0.0)
            ring.append(bm.verts.new((cx + r * math.cos(a), cy + r * math.sin(a), z)))
        rings.append(ring)
    peak = bm.verts.new((cx + rng.uniform(-0.1, 0.1) * radius, cy + rng.uniform(-0.1, 0.1) * radius, height))
    for a, b in zip(rings, rings[1:]):
        for i in range(nseg):
            j = (i + 1) % nseg
            bm.faces.new((a[i], a[j], b[j], b[i]))
    for i in range(nseg):
        bm.faces.new((rings[-1][i], rings[-1][(i + 1) % nseg], peak))
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    grass = []
    for f in list(bm.faces):
        c = f.calc_center_median()
        if f.normal.z > 0.62 and c.z > 0.3 * height:
            col_f = rng.choice([(0.10, 0.20, 0.06), (0.13, 0.24, 0.07), (0.09, 0.17, 0.06)])
            grass.append(f)
        elif c.z < 0.08 * height:
            col_f = (0.12, 0.09, 0.07)
        else:
            v = rng.uniform(0.85, 1.15)
            col_f = (0.34 * v, 0.25 * v, 0.18 * v)
        set_face_color(bm, [f], layer, col_f)
    spots = []
    for f in grass:
        area = f.calc_area()
        for _ in range(int(area / 26.0 + rng.random())):
            a, b = rng.random(), rng.random()
            if a + b > 1:
                a, b = 1 - a, 1 - b
            v0, v1, v2 = (v.co for v in f.verts)
            spots.append(v0 + (v1 - v0) * a + (v2 - v0) * b - Vector((0, 0, 0.3)))
    for p in spots:
        add_pine(bm, p, rng.uniform(0.17, 0.27) * height, layer, rng)
    # A couple of sea stacks around the base
    for _ in range(2):
        a = rng.uniform(0, 2 * math.pi)
        d = radius * rng.uniform(1.2, 1.45)
        n0 = len(bm.faces)
        s = radius * rng.uniform(0.06, 0.11)
        bm_rock(bm, (cx + d * math.cos(a), cy + d * math.sin(a), s * 0.6), (s, s, s * 2.2), rng)
        set_face_color(bm, list(bm.faces)[n0:], layer, (0.26, 0.20, 0.15))
    return mesh_object(name, bm, col, [M.terrain])


# --------------------------------------------------------------------------
# Underwater life
# --------------------------------------------------------------------------
def fish_mesh():
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.5,
                              matrix=Matrix.LocRotScale(Vector((0.08, 0, 0)), None, Vector((0.72, 0.22, 0.40))))
    tail = [bm.verts.new(p) for p in ((-0.22, 0, 0), (-0.56, 0, 0.21), (-0.46, 0, 0), (-0.56, 0, -0.21))]
    bm.faces.new((tail[0], tail[1], tail[2]))
    bm.faces.new((tail[0], tail[2], tail[3]))
    fin = [bm.verts.new(p) for p in ((0.16, 0, 0.17), (-0.12, 0, 0.15), (-0.06, 0, 0.32))]
    bm.faces.new(fin)
    me = bpy.data.meshes.new("Fish")
    bm.to_mesh(me)
    bm.free()
    return me


def coral(bm, kind, base, size, rng):
    if kind == "branch":
        for _ in range(rng.randint(4, 7)):
            d = Vector((rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 1.0)).normalized()
            ln = size * rng.uniform(0.6, 1.0)
            tip_ = base + d * ln
            bm_frustum(bm, base, tip_, 0.07 * size, 0.025 * size, segs=5)
            mid_ = base + d * ln * 0.55
            d2 = (d + Vector((rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), 0.2))).normalized()
            bm_frustum(bm, mid_, mid_ + d2 * ln * 0.4, 0.04 * size, 0.018 * size, segs=4)
    elif kind == "brain":
        bm_rock(bm, base, (size * 0.55, size * 0.5, size * 0.38), rng, jitter=0.12)
    elif kind == "fan":
        a = rng.uniform(0, math.pi)
        ax = Vector((math.cos(a), math.sin(a), 0))
        stem_top = base + Vector((0, 0, 0.15 * size))
        bm_frustum(bm, base, stem_top, 0.04 * size, 0.03 * size, segs=4)
        pts = [bm.verts.new(stem_top)]
        for i in range(7):
            t = math.pi * (0.1 + 0.8 * i / 6)
            r = size * rng.uniform(0.75, 1.0)
            pts.append(bm.verts.new(stem_top + ax * math.cos(t) * r + Vector((0, 0, math.sin(t) * r))))
        for i in range(1, 7):
            bm.faces.new((pts[0], pts[i], pts[i + 1]))
    elif kind == "tubes":
        for _ in range(rng.randint(4, 7)):
            off = Vector((rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18), 0)) * size
            top = base + off + Vector((rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.35, 0.8))) * size
            bm_frustum(bm, base + off, top, 0.07 * size, 0.09 * size, segs=6)
    elif kind == "kelp":
        p = base
        d = Vector((0, 0, 1))
        for i in range(4):
            d = (d + Vector((rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), 0))).normalized()
            q = p + d * size * 0.25
            bm_frustum(bm, p, q, 0.05 * size * (1 - 0.2 * i), 0.05 * size * (1 - 0.2 * (i + 1)), segs=3)
            p = q


CORAL_COLORS = {
    "branch": [(1.0, 0.35, 0.25), (1.0, 0.55, 0.20), (0.95, 0.30, 0.55)],
    "brain": [(0.85, 0.30, 0.55), (0.95, 0.70, 0.35), (0.55, 0.75, 0.30)],
    "fan": [(0.60, 0.25, 0.85), (0.90, 0.25, 0.45)],
    "tubes": [(1.0, 0.80, 0.20), (0.95, 0.45, 0.15)],
    "kelp": [(0.25, 0.60, 0.20), (0.35, 0.55, 0.15)],
}


def camera_xy_path():
    """Underwater stretch of the camera path in XY, for keeping props clear."""
    pts = [Vector(k[1]) for k in CAM_KEYS if k[1][2] < 0.5]
    return [Vector((p.x, p.y)) for p in pts]


def dist_to_path(p, path):
    best = 1e9
    for a, b in zip(path, path[1:]):
        ab = b - a
        t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-9)))
        best = min(best, (a + ab * t - p).length)
    return best


def build_reef(col, M):
    rng = random.Random(7)
    path = camera_xy_path()
    hook_xy = BOBBER_XY + HOOK_DRIFT
    garden = Vector((1.5, -3.6))
    placed = []

    def ok(xy, clearance):
        if dist_to_path(xy, path) < clearance:
            return False
        return all((xy - q).length > 0.55 for q in placed)

    def place(kind, xy, size):
        bm = bmesh.new()
        coral(bm, kind, Vector((xy.x, xy.y, seabed_z(xy.x, xy.y) - 0.05)), size, rng)
        ob = mesh_object(f"Coral_{kind}_{len(placed)}", bm, col, [M.underwater],
                         color=rng.choice(CORAL_COLORS[kind]), recalc=kind not in ("fan",))
        placed.append(xy)
        return ob

    kinds = ["branch", "brain", "fan", "tubes", "branch", "brain", "kelp"]
    # Dense garden behind the hook
    tries = 0
    while len(placed) < 38 and tries < 2000:
        tries += 1
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.4, 4.5)
        xy = garden + Vector((math.cos(a), math.sin(a))) * r
        if (xy - hook_xy).length < 0.6 or not ok(xy, 2.0):
            continue
        place(rng.choice(kinds), xy, rng.uniform(0.5, 1.0))
    # Sparse reef over the wider area
    tries = 0
    while len(placed) < 90 and tries < 4000:
        tries += 1
        xy = Vector((rng.uniform(-10, 12), rng.uniform(-22, 8)))
        if not ok(xy, 2.5):
            continue
        place(rng.choice(kinds), xy, rng.uniform(0.5, 1.1))

    # Rocks
    bm = bmesh.new()
    for _ in range(26):
        xy = Vector((rng.uniform(-10, 12), rng.uniform(-22, 8)))
        if dist_to_path(xy, path) < 2.5:
            continue
        s = rng.uniform(0.35, 1.3)
        bm_rock(bm, (xy.x, xy.y, seabed_z(xy.x, xy.y) + 0.1 * s), (s, s * rng.uniform(0.7, 1.0), s * 0.6),
                rng, rot_z=rng.uniform(0, 6.28))
    mesh_object("SeaRocks", bm, col, [M.underwater], color=(0.30, 0.34, 0.36))
    # A coral-topped boulder next to the hook so the descent passes coral
    bm = bmesh.new()
    rock_xy = Vector((2.3, -4.6))
    bm_rock(bm, (rock_xy.x, rock_xy.y, seabed_z(rock_xy.x, rock_xy.y) + 0.1), (0.6, 0.5, 0.55), rng, jitter=0.15)
    mesh_object("CoralBoulder", bm, col, [M.underwater], color=(0.40, 0.36, 0.30))
    top = seabed_z(rock_xy.x, rock_xy.y) + 0.7
    for k, off, s in (("brain", (0.1, 0.1), 0.6), ("branch", (-0.3, 0.2), 0.6), ("fan", (0.2, -0.3), 0.55)):
        bm = bmesh.new()
        coral(bm, k, Vector((rock_xy.x + off[0], rock_xy.y + off[1], top - 0.15)), s, rng)
        mesh_object(f"BoulderCoral_{k}", bm, col, [M.underwater], color=rng.choice(CORAL_COLORS[k]),
                    recalc=k != "fan")


SCHOOLS = [
    # centre, count, radius range, height spread, angular speed, colours, size
    ((1.9, -3.9, -4.55), 11, (0.5, 1.7), 0.35, 0.50,
     [(1.0, 0.78, 0.10), (1.0, 0.48, 0.08), (0.95, 0.90, 0.25)], 0.28),
    ((-0.1, -4.8, -2.8), 8, (0.4, 1.0), 0.25, -0.65,
     [(0.15, 0.45, 1.0), (0.20, 0.80, 0.95), (0.45, 0.65, 1.0)], 0.24),
    ((3.0, -2.0, -4.3), 3, (3.2, 3.8), 0.20, 0.18,
     [(0.95, 0.30, 0.30), (1.0, 0.45, 0.50)], 0.55),
]


def build_fish(col, M):
    rng = random.Random(21)
    me = fish_mesh()
    me.materials.append(M.underwater)
    fishes = []
    for si, (centre, count, (r0, r1), hz, omega, colors, size) in enumerate(SCHOOLS):
        c = Vector(centre)
        for i in range(count):
            ob = bpy.data.objects.new(f"Fish_{si}_{i}", me)
            col.objects.link(ob)
            ob.color = (*rng.choice(colors), 1.0)
            s = size * rng.uniform(0.85, 1.15)
            ob.scale = (s, s, s)
            ob.rotation_mode = "XYZ"
            rad = rng.uniform(r0, r1)
            th0 = rng.uniform(0, 2 * math.pi)
            dz = rng.uniform(-hz, hz)
            ph = rng.uniform(0, 6.28)
            w = omega * rng.uniform(0.85, 1.15)

            def pos(t):
                th = th0 + w * t
                return c + Vector((rad * math.cos(th), 0.8 * rad * math.sin(th),
                                   dz + 0.12 * math.sin(2 * th + ph)))
            for f in range(1, FRAME_END + 1):
                t = frame_time(f)
                p = pos(t)
                v = pos(t + 0.02) - pos(t - 0.02)
                ob.location = p
                yaw = math.atan2(v.y, v.x) + 0.18 * math.sin(t * 2 * math.pi * 2.2 + ph)
                pitch = -math.asin(max(-1.0, min(1.0, v.z / max(v.length, 1e-6))))
                ob.rotation_euler = (0.0, pitch, yaw)
                keyframe(ob, "location", f)
                keyframe(ob, "rotation_euler", f)
            fishes.append(ob)
    return fishes


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
def build_camera(col):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = 28.0
    cd.sensor_width = 36.0
    cd.sensor_fit = "HORIZONTAL"
    cd.clip_start = 0.05
    cd.clip_end = 12000.0
    cam = bpy.data.objects.new("Camera", cd)
    col.objects.link(cam)
    target = empty_object("CameraTarget", col, size=0.3)
    for f, loc, tgt in CAM_KEYS:
        cam.location = loc
        target.location = tgt
        keyframe(cam, "location", f)
        keyframe(target, "location", f)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    bpy.context.scene.camera = cam
    return cam


def key_camera_underwater(cam):
    """Key scene["cam_under"] = 1 on frames where the lens is below the waves."""
    sc = bpy.context.scene
    for f in range(1, FRAME_END + 1):
        sc.frame_set(f)
        p = cam.matrix_world.translation
        sc["cam_under"] = 1.0 if p.z < wave_z(p.x, p.y, frame_time(f)) else 0.0
        sc.keyframe_insert(data_path='["cam_under"]', frame=f)


# --------------------------------------------------------------------------
# Render settings
# --------------------------------------------------------------------------
def setup_render(scale, samples):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.render.resolution_percentage = scale
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, FRAME_END
    ee = sc.eevee
    ee.taa_render_samples = samples
    ee.use_shadows = True
    ee.use_raytracing = False
    ee.use_fast_gi = False
    ee.shadow_resolution_scale = 0.5
    sc.view_settings.view_transform = "AgX"
    for look in ("AgX - Punchy", "AgX - Medium High Contrast"):
        try:
            sc.view_settings.look = look
            break
        except TypeError:
            continue
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def point_triangle_distance(p, A, B, C):
    """Distance from point p to each triangle (A, B, C arrays of shape (n, 3))."""
    ab, ac, ap = B - A, C - A, p - A
    n = np.cross(ab, ac)
    nn = np.linalg.norm(n, axis=1)
    nn[nn == 0] = 1e-12
    plane = np.abs(np.einsum("ij,ij->i", ap, n)) / nn
    # barycentric test for projection inside the triangle
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    den = d00 * d11 - d01 * d01
    den[den == 0] = 1e-12
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    inside = (v >= 0) & (w >= 0) & (v + w <= 1)

    def seg(P, Q):
        pq = Q - P
        t = np.clip(np.einsum("ij,ij->i", p - P, pq) / np.maximum(np.einsum("ij,ij->i", pq, pq), 1e-12), 0, 1)
        return np.linalg.norm(P + pq * t[:, None] - p, axis=1)
    edge = np.minimum(np.minimum(seg(A, B), seg(B, C)), seg(C, A))
    return np.where(inside, plane, edge)


def world_triangles(ob, depsgraph):
    ev = ob.evaluated_get(depsgraph)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    n = len(me.vertices)
    co = np.empty(n * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    mw = np.array(ev.matrix_world)
    co = co @ mw[:3, :3].T + mw[:3, 3]
    ev.to_mesh_clear()
    return co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]


def run_checks(cam):
    sc = bpy.context.scene
    skip = {"Ocean", "OceanFar", "SeaFloor"}
    meshes = [o for o in sc.objects if o.type == "MESH" and o.name not in skip and not o.name.startswith("Island")]
    print("\n=== CAMERA REPORT (frame: pos | pitch | speed m/s | above wave | above seabed | nearest mesh) ===")
    prev = None
    worst = []
    hull_wet = -1e9
    for f in range(1, FRAME_END + 1):
        sc.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        mw = cam.evaluated_get(dg).matrix_world
        p = mw.translation.copy()
        fwd = -(mw.to_3x3() @ Vector((0, 0, 1)))
        pitch = math.degrees(math.asin(max(-1, min(1, fwd.z))))
        speed = 0.0 if prev is None else (p - prev).length * FPS
        prev = p
        t = frame_time(f)
        above_wave = p.z - wave_z(p.x, p.y, t)
        above_bed = p.z - seabed_z(p.x, p.y)
        best = (1e9, "")
        pn = np.array(p)
        for ob in meshes:
            ev = ob.evaluated_get(dg)
            if ev.hide_render:
                continue
            # coarse reject by bounding sphere
            bb = [ev.matrix_world @ Vector(c) for c in ev.bound_box]
            centre = sum(bb, Vector()) / 8.0
            rad = max((c - centre).length for c in bb)
            if (centre - p).length - rad > best[0]:
                continue
            A, B, C = world_triangles(ob, dg)
            if len(A) == 0:
                continue
            d = float(point_triangle_distance(pn, A, B, C).min())
            if d < best[0]:
                best = (d, ob.name)
        worst.append((best[0], f, best[1]))
        # water inside the hull footprint relative to the floor
        boat = sc.objects["Boat"].evaluated_get(dg)
        for ly in (-1.3, -0.5, 0.4, 1.3):
            for lx in (-0.4, 0.0, 0.4):
                wp = boat.matrix_world @ Vector((lx, ly, FLOOR_Z))
                hull_wet = max(hull_wet, wave_z(wp.x, wp.y, t) - wp.z)
        if f % 6 == 1 or f == FRAME_END or abs(above_wave) < 0.3 or best[0] < 0.6:
            print(f"{f:4d}: ({p.x:6.2f},{p.y:7.2f},{p.z:6.2f}) | {pitch:6.1f} | {speed:5.2f} | "
                  f"{above_wave:6.2f} | {above_bed:6.2f} | {best[0]:6.2f} {best[1]}")
    worst.sort()
    print("closest approaches:", [(round(d, 2), f, n) for d, f, n in worst[:6]])
    print(f"max water height above boat floor: {hull_wet:.3f} m (must stay < 0)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, FRAME_END
    sc.render.fps = FPS

    c_env, c_boat = collection("Environment"), collection("Boat")
    c_isl, c_reef, c_cam = collection("Islands"), collection("Underwater"), collection("Camera")

    sky = build_sky_group()
    atmos = build_atmos_group(sky)
    M = build_materials(atmos)
    build_world(sky)
    build_sun(c_env)
    build_ocean(c_env, M.water)
    build_seabed(c_reef, M.underwater)

    boat = build_boat(c_boat, M)
    animate_boat(boat)
    tip = build_fisherman(c_boat, M, boat)
    build_tackle(c_boat, M, tip, boat)

    for i, (x, y, r, h) in enumerate([(-80, 170, 28, 24), (55, 210, 24, 18),
                                       (-175, 330, 55, 42), (115, 195, 26, 22)]):
        build_island(c_isl, M, f"Island{i + 1}", x, y, r, h, seed=100 + i)

    build_reef(c_reef, M)
    build_fish(c_reef, M)
    cam = build_camera(c_cam)
    key_camera_underwater(cam)
    sc.frame_set(1)
    return cam


def main():
    args = parse_args()
    cam = build_scene()
    setup_render(args.scale, args.samples)
    if args.check:
        run_checks(cam)
    if args.save:
        bpy.context.scene.frame_set(1)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save), compress=True)
        print("saved", args.save)
    sc = bpy.context.scene
    if args.frames:
        os.makedirs(args.outdir, exist_ok=True)
        for f in [int(x) for x in args.frames.split(",") if x.strip()]:
            sc.frame_set(f)
            sc.render.filepath = os.path.join(args.outdir, f"frame_{f:03d}.png")
            bpy.ops.render.render(write_still=True)
            print("rendered", sc.render.filepath)
    if args.anim:
        os.makedirs(args.outdir, exist_ok=True)
        sc.frame_step = args.step
        sc.render.filepath = os.path.join(args.outdir, "frame_")
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
