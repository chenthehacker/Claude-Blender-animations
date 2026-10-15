"""Sockeye masterclass: river drift fishing breakdown. Vertical 9:16, 24 fps, 75 s, Blender 5.2 EEVEE.

Run with the Blender binary:
    blender --background --python sockeye_masterclass.py -- [options]
or with the bpy Python module (pip install bpy):
    python sockeye_masterclass.py [options]

Options:
    --save PATH        save the generated scene as a .blend file (textures and fonts packed)
    --check            camera clearances (bed, rocks, trees, angler, fish, rod), camera acceleration and turn
                       rate, water surface crossings and cuts against wipes, overlay overlaps and off-screen boxes
    --frames 1,300     render these frames as PNG stills into --outdir
    --anim             render the animation as PNG frames into --outdir (--start/--end)
    --outdir DIR       output directory for renders
    --scale PCT        resolution percentage (default 100 -> 1080x1920)
    --samples N        EEVEE render samples (default 32)
    --no-blur          switch motion blur off (quick tests)
    --events PATH      write overlay and action timing as JSON (input for sockeye_audio.py)

Facts used in the on-screen text are listed with their sources in SOURCES.md.

World layout (metres, +Z up, water surface at z = 0). The river flows towards -Y. Its centre line meanders
around x = 0 (river_cx) with half width river_w; u = (x - cx) / w runs from -1 (left, rocky bank) to +1
(right bank, a gravel bar). Along the river: lower riffle (y < -30), tailout (-30 to -6), pool (-6 to 34),
pool head and upper riffle (y > 34). The sockeye school holds in the tailout close to the right bank,
where the angler wades on the gravel bar.

Beats (the brief's own timeline; it already sums to 75 s, so nothing is scaled):
    1  F1-121      hook: underwater tracking shot of a sockeye surging into the white water at the pool head
    2  F122-485    location: drone sweep down the river into the tailout and the riffle below it
    3  F486-1095   rig and technique: underwater side view (cross-section) of three drifts through the school
    4  F1096-1560  strike: over-under split shot of the take, then above water for the hook-set and fight
    5  F1561-1800  outro: recap and a wide pull-back over the valley
"""

import argparse
import math
import os
import random
import sys

import bpy  # must come first: the pip bpy module exposes bmesh/mathutils after import
import bmesh
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector, kdtree

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")
ASSET_DIR = os.path.join(HERE, "assets")     # freely licensed downloads, see assets/CREDITS.md


def asset(*parts):
    return os.path.join(ASSET_DIR, *parts)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
FPS = 24
SHUTTER = 0.5          # motion blur exposure, in frames
FRAME_END = 1800
RES_X, RES_Y = 1080, 1920
LENS = 24.0
SENSOR = 36.0          # vertical sensor fit
HUD_DIST = 0.25        # overlay plane distance in front of the lens
HUD_W = 0.5 * RES_X / RES_Y   # overlay half width in frame-height units


def srgb(r, g, b):
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (lin(r), lin(g), lin(b))




# --------------------------------------------------------------------------
# Palette and timeline
# --------------------------------------------------------------------------
GOLD = srgb(246, 188, 70)        # accent words and highlights
BADGE = srgb(250, 172, 52)       # numbered badge discs
WHITE = (1.0, 1.0, 1.0)
INK = srgb(10, 18, 24)
PIN_RED = srgb(240, 96, 80)
GLOW = srgb(255, 200, 96)
TEAL = srgb(12, 86, 92)

CUT1, CUT2, CUT3, CUT4 = 122, 486, 1096, 1262
WIPES = [(CUT1 - 4, CUT1 + 3), (CUT2 - 4, CUT2 + 3), (CUT3 - 4, CUT3 + 3), (CUT4 - 4, CUT4 + 3)]   # opaque a-2..b+2
SEGMENTS = [(1, CUT1 - 1), (CUT1, CUT2 - 1), (CUT2, CUT3 - 1), (CUT3, CUT4 - 1), (CUT4, FRAME_END)]
SPLIT = (CUT3, CUT4 - 1)          # over-under shot: the lens sits exactly on the waterline
STRIKE = 1172                     # the sockeye takes the fly
HOOKSET = 1180                    # the angler sweeps the rod
LEAP = (1318, 1342)               # the hooked fish half clears the surface once
SUN_DIR = Vector((0.40, 0.80, 0.42)).normalized()    # towards the sun: low, upstream and to the right
UW_FOG_DIST = 9.0                 # underwater visibility scale (m): a clear river
DEPTH_FOG = 4.0                   # how fast the bed fades with depth when seen from above (m)
AIR_HAZE_DIST = 700.0
CAM_SMOOTH = 6.0
SPLIT_PITCH = 7.0                 # degrees the over-under lens looks down (puts the waterline above centre)

UW_UP, UW_MID, UW_DOWN = srgb(120, 214, 200), srgb(22, 118, 112), srgb(5, 38, 40)
HAZE = srgb(196, 212, 218)


def smoothstep(e0, e1, x):
    if e0 > e1:
        return 1.0 - smoothstep(e1, e0, x)
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def frame_time(f):
    return f / FPS


def segment_of(f):
    for i, (a, b) in enumerate(SEGMENTS):
        if a <= f <= b:
            return i
    return len(SEGMENTS) - 1


# --------------------------------------------------------------------------
# Generic helpers (shared with the offshore and saltwater builds)
# --------------------------------------------------------------------------
class PyB:
    def add(self, *a):
        return sum(a)

    def sub(self, a, b):
        return a - b

    def mul(self, *a):
        r = 1.0
        for v in a:
            r *= v
        return r

    def div(self, a, b):
        return a / b

    def sin(self, a):
        return math.sin(a)

    def cos(self, a):
        return math.cos(a)

    def exp(self, a):
        return math.exp(a)

    def fract(self, a):
        return a - math.floor(a)

    def abs(self, a):
        return abs(a)

    def pow(self, a, b):
        return max(a, 0.0) ** b

    def ss(self, e0, e1, x):
        return smoothstep(e0, e1, x)

    def clamp(self, a):
        return min(1.0, max(0.0, a))

    def lerp(self, a, b, t):
        return self.add(a, self.mul(self.sub(b, a), t))


def _const(v):
    return isinstance(v, (int, float))


class NodeB(PyB):
    def __init__(self, nt):
        self.nt = nt

    def add(self, *a):
        c = sum(v for v in a if _const(v))
        socks = [v for v in a if not _const(v)]
        if not socks:
            return c
        r = socks[0]
        for s in socks[1:]:
            r = nmath(self.nt, "ADD", r, s)
        return nmath(self.nt, "ADD", r, c) if c else r

    def mul(self, *a):
        c = 1.0
        for v in a:
            if _const(v):
                c *= v
        socks = [v for v in a if not _const(v)]
        if not socks or c == 0.0:
            return c if not socks else 0.0
        r = socks[0]
        for s in socks[1:]:
            r = nmath(self.nt, "MULTIPLY", r, s)
        return nmath(self.nt, "MULTIPLY", r, c) if c != 1.0 else r

    def _op(self, op, fn, *a):
        if all(_const(v) for v in a):
            return fn(*a)
        return nmath(self.nt, op, *a)

    def sub(self, a, b):
        return self._op("SUBTRACT", lambda x, y: x - y, a, b)

    def div(self, a, b):
        return self._op("DIVIDE", lambda x, y: x / y, a, b)

    def sin(self, a):
        return self._op("SINE", math.sin, a)

    def cos(self, a):
        return self._op("COSINE", math.cos, a)

    def exp(self, a):
        return self._op("EXPONENT", math.exp, a)

    def fract(self, a):
        return self._op("FRACT", lambda x: x - math.floor(x), a)

    def abs(self, a):
        return self._op("ABSOLUTE", abs, a)

    def pow(self, a, b):
        if _const(a):
            return max(a, 0.0) ** b
        return nmath(self.nt, "POWER", nmath(self.nt, "MAXIMUM", a, 0.0), b)

    def ss(self, e0, e1, x):
        if _const(x):
            return smoothstep(e0, e1, x)
        if e0 > e1:
            return self.sub(1.0, self.ss(e1, e0, x))
        return map_range(self.nt, x, e0, e1)

    def clamp(self, a):
        return min(1.0, max(0.0, a)) if _const(a) else nmath(self.nt, "ADD", a, 0.0, clamp=True)


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
        ob.color = (*color[:3], 1.0)
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


def bm_frustum(bm, p0, p1, r0, r1, segs=6, cap0=True, cap1=True, twist=0.0):
    p0, p1 = Vector(p0), Vector(p1)
    u, v = basis(p1 - p0)

    def ring(c, r):
        if r <= 0.0:
            return [bm.verts.new(c)]
        return [bm.verts.new(c + (u * math.cos(twist + 2 * math.pi * i / segs) +
                                  v * math.sin(twist + 2 * math.pi * i / segs)) * r) for i in range(segs)]
    ra, rb = ring(p0, r0), ring(p1, r1)
    faces = []
    for i in range(segs):
        j = (i + 1) % segs
        if len(rb) == 1:
            faces.append(bm.faces.new((ra[i], ra[j], rb[0])))
        elif len(ra) == 1:
            faces.append(bm.faces.new((ra[0], rb[j], rb[i])))
        else:
            faces.append(bm.faces.new((ra[i], ra[j], rb[j], rb[i])))
    if cap0 and len(ra) > 1:
        faces.append(bm.faces.new(list(reversed(ra))))
    if cap1 and len(rb) > 1:
        faces.append(bm.faces.new(rb))
    return faces


def bm_box(bm, center, size, rot=(0, 0, 0)):
    m = Matrix.LocRotScale(Vector(center), Euler(rot), Vector(size))
    return bmesh.ops.create_cube(bm, size=1.0, matrix=m)["verts"]


def bm_blob(bm, center, size, rng, subdiv=1, jitter=0.2, rot=(0, 0, 0)):
    m = Matrix.LocRotScale(Vector(center), Euler(rot), Vector(size))
    res = bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0, matrix=Matrix())
    for v in res["verts"]:
        v.co = m @ (v.co * (1.0 + rng.uniform(-jitter, jitter)))
    return list({f for v in res["verts"] for f in v.link_faces})


def color_faces(faces, layer, color):
    for f in faces:
        for loop in f.loops:
            loop[layer] = (*color[:3], 1.0)


def key(ob, path, frame, index=-1):
    ob.keyframe_insert(data_path=path, frame=frame, index=index)


def key_still(ob, path, frame):
    """Key an overlay so it holds still while the shutter is open (frame +-0.25) and moves between
    exposures: it stays sharp under motion blur, which EEVEE cannot switch off per object."""
    prefs = bpy.context.preferences.edit
    old = prefs.keyframe_new_interpolation_type
    prefs.keyframe_new_interpolation_type = "LINEAR"
    for t in (frame - 0.3, frame + 0.3):
        ob.keyframe_insert(data_path=path, frame=t)
    prefs.keyframe_new_interpolation_type = old


def key_alpha(ob, pairs):
    """pairs: [(frame, alpha)] keyed on the object colour alpha (read by the shaders)."""
    for f, a in pairs:
        ob.color[3] = a
        key(ob, "color", f, index=3)


def fade_pairs(f_in, f_out, peak=1.0, dur=8):
    if f_in <= 1:                       # already on screen at the first frame
        return [(1, peak), (f_out, peak), (f_out + dur, 0.0)]
    pts = [(f_in - dur, 0.0), (f_in, peak), (f_out, peak), (f_out + dur, 0.0)]
    return [(max(1, f), a) for f, a in pts]


def add_frame_driver(owner, path, expr, index=-1):
    fc = owner.driver_add(path, index) if index >= 0 else owner.driver_add(path)
    fc.driver.type = "SCRIPTED"
    fc.driver.expression = expr
    return fc


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
    n.clamp_factor = True
    _feed(nt, n.inputs[0], fac)
    _feed(nt, n.inputs[6], (*a, 1.0) if isinstance(a, tuple) else a)
    _feed(nt, n.inputs[7], (*b, 1.0) if isinstance(b, tuple) else b)
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
        els.new(pos).color = (*col, 1.0)
    return n.outputs["Color"]


def new_group(name, kind, inputs, outputs):
    ng = bpy.data.node_groups.new(name, kind)
    for nm, st in inputs:
        ng.interface.new_socket(nm, in_out="INPUT", socket_type=st)
    for nm, st in outputs:
        ng.interface.new_socket(nm, in_out="OUTPUT", socket_type=st)
    return ng, ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")


def attr(nt, name, out="Color"):
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_name = name
    return a.outputs[out]


def load_image(path, non_color=False):
    im = bpy.data.images.load(path, check_existing=True)
    if non_color:
        im.colorspace_settings.name = "Non-Color"
    return im


def image_node(nt, path, non_color=False, vector=None):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = load_image(path, non_color)
    if vector is not None:
        nt.links.new(vector, n.inputs["Vector"])
    return n


def principled(nt, color, rough=0.75, spec=0.3):
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _feed(nt, b.inputs["Base Color"], (*color, 1.0) if isinstance(color, tuple) else color)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = spec
    return b.outputs["BSDF"]


def new_material(name, build, atmos=None, blended=False):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    shader = build(nt)
    if atmos is not None:
        g = nt.nodes.new("ShaderNodeGroup")
        g.node_tree = atmos
        nt.links.new(shader, g.inputs["Shader"])
        shader = g.outputs["Shader"]
    nt.links.new(shader, out.inputs["Surface"])
    m.use_backface_culling = False
    if blended:
        m.surface_render_method = "BLENDED"
    return m


class Materials:
    pass


def set_mod_input(mod, ident, value):
    """Blender 5.2 exposes modifier inputs as mod.properties.inputs.<ident>.value;
    older builds used ID properties (mod[ident])."""
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        getattr(props.inputs, ident).value = value
    else:
        mod[ident] = value


def fish_swim_group():
    ng, gi, go = new_group("FishSwim", "GeometryNodeTree",
                           [("Geometry", "NodeSocketGeometry"), ("Phase", "NodeSocketFloat"),
                            ("Amp", "NodeSocketFloat"), ("Rate", "NodeSocketFloat")],
                           [("Geometry", "NodeSocketGeometry")])
    pos = separate(ng, ng.nodes.new("GeometryNodeInputPosition").outputs[0])
    t = nmath(ng, "DIVIDE", ng.nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], float(FPS))
    w = nmath(ng, "POWER", map_range(ng, pos["X"], 0.25, -0.5, 0.0, 1.0, smooth=False), 1.4)  # tail moves most
    arg = nmath(ng, "MULTIPLY_ADD", t, gi.outputs["Rate"], nmath(ng, "MULTIPLY_ADD", pos["X"], -7.0, gi.outputs["Phase"]))
    dy = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", nmath(ng, "SINE", arg), w), gi.outputs["Amp"])
    sp = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, 0.0, dy, 0.0), sp.inputs["Offset"])
    ng.links.new(sp.outputs[0], go.inputs["Geometry"])
    return ng


def key_fish(ob, frames, positions, yaws, pitches):
    for f, p, yw, pt in zip(frames, positions, yaws, pitches):
        ob.location = p
        ob.rotation_euler = (0.0, pt, yw)
        key(ob, "location", f)
        key(ob, "rotation_euler", f)


def heading_angles(v):
    yaw = math.atan2(v.y, v.x)
    pitch = -math.asin(max(-1.0, min(1.0, v.z / max(v.length, 1e-6))))
    return yaw, pitch


def unwrap(angles):
    out = [angles[0]]
    for a in angles[1:]:
        while a - out[-1] > math.pi:
            a -= 2 * math.pi
        while a - out[-1] < -math.pi:
            a += 2 * math.pi
        out.append(a)
    return out


def smooth_series(vals, k):
    arr = np.array([list(v) for v in vals])
    out = arr.copy()
    for i in range(len(arr)):
        lo, hi = max(0, i - k), min(len(arr), i + k + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    return [Vector(v) for v in out]


def line_object(name, col, M, start, end, middle=None, radius=0.006):
    ng, gi, go = new_group(name + "GN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
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
    ob.visible_shadow = False
    return ob


def hermite_path(keys):
    """Per-frame positions and targets from keys (frame, pos, target, hold)."""
    fr = [k[0] for k in keys]
    P = [Vector(k[1]) for k in keys]
    T = [Vector(k[2]) for k in keys]
    hold = [k[3] for k in keys]

    def tangents(V):
        m = []
        for i in range(len(V)):
            if hold[i] or i == 0 or i == len(V) - 1:
                m.append(Vector((0, 0, 0)))
            else:
                m.append((V[i + 1] - V[i - 1]) / (fr[i + 1] - fr[i - 1]))
        return m
    mP, mT = tangents(P), tangents(T)
    out = {}
    for i in range(len(fr) - 1):
        h = fr[i + 1] - fr[i]
        for f in range(fr[i], fr[i + 1] + 1):
            u = (f - fr[i]) / h
            h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
            h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
            out[f] = (P[i] * h00 + mP[i] * (h10 * h) + P[i + 1] * h01 + mP[i + 1] * (h11 * h),
                      T[i] * h00 + mT[i] * (h10 * h) + T[i + 1] * h01 + mT[i + 1] * (h11 * h))
    return out


def look_rotation(pos, target):
    f = (target - pos).normalized()
    w = smoothstep(0.80, 0.97, -f.z)
    up = (Vector((0, 0, 1)) * (1 - w) + Vector((0, 1, 0)) * w).normalized()
    up = (up - f * up.dot(f)).normalized()
    right = f.cross(up)
    return Matrix((right, up, -f)).transposed().to_quaternion()



def store_rest(me, z_lo=None, z_hi=None):
    """Rest position (for shading after the swim deformation) and 0..1 height (belly..back)."""
    co = [v.co.copy() for v in me.vertices]
    a = me.attributes.new("rest", "FLOAT_VECTOR", "POINT")
    a.data.foreach_set("vector", [c for v in co for c in v])
    zs = [v.z for v in co]
    lo, hi = (min(zs) if z_lo is None else z_lo), (max(zs) if z_hi is None else z_hi)
    b = me.attributes.new("zrel", "FLOAT", "POINT")
    b.data.foreach_set("value", [min(1.0, max(0.0, (z - lo) / (hi - lo))) for z in zs])



def fin_sheet(bm, outline, thickness=0.004, mat=0):
    """A thin double-sided fin from an outline in the XZ plane (y = 0)."""
    top = [bm.verts.new((x, thickness / 2, z)) for x, z in outline]
    bot = [bm.verts.new((x, -thickness / 2, z)) for x, z in outline]
    f1 = bm.faces.new(top)
    f2 = bm.faces.new(list(reversed(bot)))
    for f in (f1, f2):
        f.material_index = mat
    n = len(outline)
    for i in range(n):
        j = (i + 1) % n
        f = bm.faces.new((top[i], top[j], bot[j], bot[i]))
        f.material_index = mat



def highlight_shell(col, M, src, name, thickness=0.006):
    """Glowing outline around a gear object (inverted hull), alpha keyed for pulses."""
    ob = bpy.data.objects.new(name, src.data)
    col.objects.link(ob)
    if src.animation_data is not None and src.animation_data.action is not None:
        ob.parent = src                      # animated gear: ride along with it
    else:
        ob.parent = src.parent
        ob.matrix_parent_inverse = src.matrix_parent_inverse.copy()
        ob.location, ob.rotation_mode = src.location.copy(), src.rotation_mode
        ob.rotation_euler = src.rotation_euler.copy()
        if src.rotation_mode == "QUATERNION":
            ob.rotation_quaternion = src.rotation_quaternion.copy()
        ob.scale = src.scale.copy()
    for m in src.modifiers:
        if m.type == "NODES":
            mm = ob.modifiers.new(m.name, "NODES")
            mm.node_group = m.node_group
    sol = ob.modifiers.new("Outline", "SOLIDIFY")
    sol.thickness = thickness
    sol.offset = 1.0
    sol.use_flip_normals = True
    sol.use_rim = False
    sol.material_offset = 0
    ob.data = src.data.copy()
    ob.data.materials.clear()
    ob.data.materials.append(M.highlight)
    ob.visible_shadow = False
    ob.color = (*GLOW, 0.0)
    return ob


def pulse(ob, f_in, f_out, peak=2.2):
    """Switch the outline on for [f_in, f_out] and pulse its brightness once a second."""
    keys = [(1, 0.0, 0.0), (f_in - 4, peak, 0.0), (f_in, peak, 1.0)]
    f = f_in
    while f < f_out - 6:
        keys.append((f, peak, 1.0))
        if f + 12 < f_out - 6:
            keys.append((f + 12, peak * 0.35, 1.0))
        f += 24
    keys += [(f_out, peak * 0.6, 1.0), (f_out + 4, peak * 0.6, 0.0)]
    if ob.animation_data and ob.animation_data.action:
        keys = [k for k in keys if k[0] != 1]       # second window on the same outline: keep the first one's keys
    for fr, b, a in keys:
        ob.color = (*(c * b for c in GLOW), a)
        ob.keyframe_insert("color", frame=max(1, fr))



def gn_object(name, col, ng):
    ob = bpy.data.objects.new(name, bpy.data.meshes.new(name))
    col.objects.link(ob)
    mod = ob.modifiers.new(name, "NODES")
    mod.node_group = ng
    try:
        ng.is_modifier = True
    except AttributeError:
        pass
    return ob


def gn_time(ng):
    return nmath(ng, "DIVIDE", ng.nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], float(FPS))


def gn_xyz(ng):
    p = separate(ng, ng.nodes.new("GeometryNodeInputPosition").outputs[0])
    return p["X"], p["Y"], p["Z"]


def gn_cam_snap(ng, cam, step):
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    oi.transform_space = "ORIGINAL"
    oi.inputs["Object"].default_value = cam
    v = ng.nodes.new("ShaderNodeVectorMath")
    v.operation = "SNAP"
    ng.links.new(oi.outputs["Location"], v.inputs[0])
    v.inputs[1].default_value = (step, step, 1e6)
    sp = separate(ng, v.outputs["Vector"])
    return combine(ng, sp["X"], sp["Y"], 0.0)



def follow(ob, target, z=True):
    con = ob.constraints.new("COPY_LOCATION")
    con.target = target
    con.use_z = z



# --------------------------------------------------------------------------
# The river: one set of formulas drives Python (placement, checks) and the shaders/geometry nodes
# --------------------------------------------------------------------------
PY = PyB()


def waves(B, x, y, t):
    """The river surface is at z = 0 for placement and particle culling (ripples are a few cm)."""
    return 0.0


def river_cx(y, B=None):
    B = B or PY
    return B.mul(B.sin(B.div(B.add(y, 20.0), 38.0)), 5.0)


def river_w(y, B=None):
    B = B or PY
    return B.add(13.0, B.mul(B.sin(B.div(y, 23.0)), 2.0))


def rx(y, u):
    """World x at distance y along the river and cross position u (-1 left bank, +1 right bank)."""
    return river_cx(y) + u * river_w(y)


def riffle(y, B=None):
    """1 in the riffles (fast, shallow, white water), 0 in the pool."""
    B = B or PY
    return B.clamp(B.add(B.sub(1.0, B.ss(-34.0, -22.0, y)), B.ss(34.0, 44.0, y)))


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def depth_max_np(y):
    """Thalweg depth: lower riffle 0.5 m, tailout rising to 1.9 m, pool 2.8 m, pool head and riffle 0.55 m."""
    return 0.5 + 1.4 * _ss(-30.0, -6.0, y) + 0.9 * _ss(-6.0, 4.0, y) - 2.25 * _ss(34.0, 46.0, y)


def bed_z_np(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    cx = 5.0 * np.sin((y + 20.0) / 38.0)
    w = 13.0 + 2.0 * np.sin(y / 23.0)
    u = (x - cx) / w
    D = depth_max_np(y)
    a = np.clip(u + 0.25, 0.0, None) / 1.25
    s_r = np.clip(1.0 - a ** 2.6, 0.0, None) ** 0.9          # gentle gravel bar on the right
    b = np.clip(-(u + 0.25), 0.0, None) / 0.75
    s_l = np.clip(1.0 - b ** 4, 0.0, None)                   # steep rocky bank on the left
    z = -D * np.where(u >= -0.25, s_r, s_l)
    dr = np.clip(u - 1.0, 0.0, None)
    dl = np.clip(-1.0 - u, 0.0, None)
    z = z + dr * w * 0.07 + 2.6 * _ss(0.35, 1.1, dr) + 22.0 * _ss(2.0, 6.0, dr) + 60.0 * _ss(6.0, 12.0, dr)
    z = z + 4.2 * _ss(0.0, 0.32, dl) + 30.0 * _ss(1.5, 5.5, dl) + 70.0 * _ss(5.5, 12.0, dl)
    n = (np.sin(x * 1.7 + np.sin(y * 0.9)) * np.sin(y * 1.3 + 0.4 * x) * 0.03
         + np.sin(x * 0.45 + 1.3) * np.sin(y * 0.37) * 0.05)
    rough = 1.0 + 3.0 * _ss(0.2, 1.5, dl) + 2.0 * _ss(1.0, 3.0, dr)
    big = np.sin(x * 0.13 + 2.0 * np.sin(y * 0.05)) * np.cos(y * 0.11 + x * 0.03) * (
        _ss(1.0, 4.0, dr) * 6.0 + _ss(1.0, 4.0, dl) * 8.0)
    return z + n * rough + big


def bed_z(x, y):
    return float(bed_z_np(x, y))


def flow_speed(y):
    """Surface current (m/s): about 0.6 in the pool, 1.6 in the riffles (illustrative values)."""
    return 0.6 + 1.0 * riffle(y)


def axis_coords(lo, hi, f_lo, f_hi, fine, coarse):
    out = list(np.arange(lo, f_lo, coarse)) + list(np.arange(f_lo, f_hi, fine)) + list(np.arange(f_hi, hi + 1e-6, coarse))
    return np.array(sorted(set(round(v, 4) for v in out)))


def grid_mesh(name, xs, ys, zfun, col, mats):
    X, Y = np.meshgrid(xs, ys)
    Z = zfun(X, Y)
    nx, ny = len(xs), len(ys)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    quads = np.stack([a, b, c, d], axis=1)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(len(quads) * 4)
    me.loops.foreach_set("vertex_index", quads.astype(np.int32).ravel())
    me.polygons.add(len(quads))
    me.polygons.foreach_set("loop_start", (np.arange(len(quads)) * 4).astype(np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(quads), dtype=bool))
    me.update(calc_edges=True)
    me.validate()
    for m in mats:
        me.materials.append(m)
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    return ob


def far_z_np(x, y):
    """Valley continuing upstream into mountains (the outro looks up the river towards them)."""
    z = bed_z_np(x, y)
    ridge = (np.sin(x * 0.006 + 1.1) * 0.5 + np.sin(x * 0.017 + y * 0.004) * 0.3 + np.sin(x * 0.041 - y * 0.013) * 0.2)
    peaks = 520.0 + 380.0 * ridge
    z = z + _ss(500.0, 1700.0, y) * peaks + _ss(80.0, 380.0, np.abs(x - 5.0 * np.sin((y + 20) / 38))) * _ss(150, 600, y) * (
        90.0 + 60.0 * np.sin(y * 0.01 + x * 0.02))
    return z


def build_terrain(col, M):
    xs = axis_coords(-110.0, 120.0, -22.0, 26.0, 0.3, 2.5)
    ys = axis_coords(-140.0, 160.0, -34.0, 50.0, 0.3, 2.0)
    ob = grid_mesh("RiverValley", xs, ys, bed_z_np, col, [M.terrain])
    print(f"terrain: {len(ob.data.vertices)} verts")
    fx = np.linspace(-1600.0, 1600.0, 181)
    fy = np.concatenate([np.linspace(150.0, 700.0, 56), np.linspace(720.0, 4200.0, 88)])
    far = grid_mesh("FarValley", fx, fy, far_z_np, col, [M.far_terrain])
    far.visible_shadow = False
    return ob, far


# --------------------------------------------------------------------------
# Shading
# --------------------------------------------------------------------------
def scene_attr(nt, name):
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_type = "VIEW_LAYER"
    a.attribute_name = name
    return a.outputs["Factor"]


def uw_color(nt, z_dir):
    """Underwater ambient colour by view direction: bright turquoise towards the surface, deep green below."""
    return color_ramp(nt, nmath(nt, "MULTIPLY_ADD", z_dir, 0.5, 0.5, clamp=True),
                      [(0.0, UW_DOWN), (0.42, srgb(8, 62, 62)), (0.55, UW_MID), (0.78, srgb(52, 160, 150)),
                       (1.0, UW_UP)])


def under_factor(nt, z):
    """1 where the shaded point is seen from under water: the lens is under (cam_under), or in the over-under
    shot (split) the point itself lies below the surface."""
    cu = scene_attr(nt, "cam_under")
    below = nmath(nt, "LESS_THAN", z, 0.0)
    return nmath(nt, "ADD", cu, nmath(nt, "MULTIPLY", scene_attr(nt, "split"), nmath(nt, "SUBTRACT", below, cu)), clamp=True)


def build_atmos_group():
    """Air haze and turquoise depth tint when seen from above; green-blue fog when seen from below."""
    ng, gi, go = new_group("Atmosphere", "ShaderNodeTree", [("Shader", "NodeSocketShader")],
                           [("Shader", "NodeSocketShader")])
    geo = ng.nodes.new("ShaderNodeNewGeometry")
    dist = ng.nodes.new("ShaderNodeCameraData").outputs["View Distance"]
    inc = separate(ng, geo.outputs["Incoming"])
    z = separate(ng, geo.outputs["Position"])["Z"]
    under = under_factor(ng, z)
    above = nmath(ng, "SUBTRACT", 1.0, under)
    haze = nmath(ng, "MULTIPLY", above,
                 nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(ng, "MULTIPLY", dist, -1.0 / AIR_HAZE_DIST))))
    slant = nmath(ng, "DIVIDE", 1.0, nmath(ng, "MAXIMUM", nmath(ng, "ABSOLUTE", inc["Z"]), 0.3))
    depth = nmath(ng, "MULTIPLY", above, nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(
        ng, "MULTIPLY", nmath(ng, "MULTIPLY", nmath(ng, "MINIMUM", z, 0.0), slant), 1.0 / DEPTH_FOG))))
    fog = nmath(ng, "MULTIPLY", under,
                nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(ng, "MULTIPLY", dist, -1.0 / UW_FOG_DIST))))
    em_h = ng.nodes.new("ShaderNodeEmission")
    em_h.inputs["Color"].default_value = (*HAZE, 1.0)
    em_h.inputs["Strength"].default_value = 0.9
    em_d = ng.nodes.new("ShaderNodeEmission")
    em_d.inputs["Color"].default_value = (*srgb(10, 110, 104), 1.0)
    em_f = ng.nodes.new("ShaderNodeEmission")
    ng.links.new(uw_color(ng, nmath(ng, "MULTIPLY", inc["Z"], -1.0)), em_f.inputs["Color"])
    s = gi.outputs["Shader"]
    for fac, em in ((depth, em_d), (haze, em_h), (fog, em_f)):
        m = ng.nodes.new("ShaderNodeMixShader")
        ng.links.new(fac, m.inputs[0])
        ng.links.new(s, m.inputs[1])
        ng.links.new(em.outputs[0], m.inputs[2])
        s = m.outputs[0]
    ng.links.new(s, go.inputs["Shader"])
    return ng


SKY_STRENGTH = 0.11


def build_world():
    w = bpy.data.worlds.new("Sky")
    bpy.context.scene.world = w
    nt = w.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.asin(SUN_DIR.z)
    sky.sun_rotation = -math.atan2(SUN_DIR.x, SUN_DIR.y)
    sky.sun_disc = True
    sky.altitude = 200.0
    sky.air_density = 1.0
    sky.aerosol_density = 2.2
    bg = nt.nodes.new("ShaderNodeBackground")
    lit = vmath(nt, "SCALE", sky.outputs[0], scale=SKY_STRENGTH)
    # Camera rays get a hue-preserving soft shoulder (the Standard view would clip the bright sky to white)
    knee, top = 0.55, 0.97
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(lit, sep.inputs[0])
    peak = nmath(nt, "MAXIMUM", sep.outputs[0], nmath(nt, "MAXIMUM", sep.outputs[1], sep.outputs[2]))
    over = nmath(nt, "MAXIMUM", nmath(nt, "SUBTRACT", peak, knee), 0.0)
    rolled = nmath(nt, "ADD", nmath(nt, "MINIMUM", peak, knee),
                   nmath(nt, "DIVIDE", over, nmath(nt, "ADD", nmath(nt, "MULTIPLY", over, 1.0 / (top - knee)), 1.0)))
    soft = vmath(nt, "SCALE", lit, scale=nmath(nt, "DIVIDE", rolled, nmath(nt, "MAXIMUM", peak, 1e-5)))
    is_cam = nt.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"]
    nt.links.new(vmath(nt, "ADD", lit, vmath(nt, "SCALE", vmath(nt, "SUBTRACT", soft, lit), scale=is_cam)),
                 bg.inputs["Color"])
    nt.links.new(bg.outputs[0], nt.nodes.new("ShaderNodeOutputWorld").inputs["Surface"])
    return sky


def build_sun(col):
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 3.6
    light.color = srgb(255, 238, 214)
    light.angle = math.radians(1.2)
    ob = bpy.data.objects.new("Sun", light)
    col.objects.link(ob)
    ob.rotation_euler = Vector((0, 0, -1)).rotation_difference(-SUN_DIR).to_euler()
    return ob


def emissive_alpha(color=None, alpha_scale=1.0):
    def build(nt):
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        em = nt.nodes.new("ShaderNodeEmission")
        if color is None:
            nt.links.new(oi.outputs["Color"], em.inputs["Color"])
        else:
            em.inputs["Color"].default_value = (*color, 1.0)
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(nmath(nt, "MULTIPLY", oi.outputs["Alpha"], alpha_scale), mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    return build


def box_image(nt, path, scale, non_color=False):
    """Image projected on the object's own coordinates from three sides (no UVs needed)."""
    tc = nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]
    mp = nt.nodes.new("ShaderNodeMapping")
    nt.links.new(tc, mp.inputs["Vector"])
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    n = image_node(nt, path, non_color, mp.outputs[0])
    n.projection = "BOX"
    n.projection_blend = 0.3
    return n


def build_materials(atmos):
    M = Materials()
    M.atmos = atmos

    def pbr(name, color, rough=0.5, metal=0.0, coat=0.0, spec=0.5, sss=0.0, thin=0.0):
        def build(nt):
            b = nt.nodes.new("ShaderNodeBsdfPrincipled")
            _feed(nt, b.inputs["Base Color"], (*color, 1.0) if isinstance(color, tuple) else color)
            b.inputs["Roughness"].default_value = rough
            b.inputs["Metallic"].default_value = metal
            b.inputs["Coat Weight"].default_value = coat
            b.inputs["Specular IOR Level"].default_value = spec
            b.inputs["Subsurface Weight"].default_value = sss
            if thin:
                b.inputs["Thin Film Thickness"].default_value = thin
            return b.outputs[0]
        return new_material(name, build, atmos)

    # ---- terrain -----------------------------------------------------------
    def terrain(nt):
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        pos = geo.outputs["Position"]
        z = separate(nt, pos)["Z"]
        nz = separate(nt, geo.outputs["Normal"])["Z"]
        steep = map_range(nt, nz, 0.86, 0.62)
        v1 = nt.nodes.new("ShaderNodeTexVoronoi")
        v1.inputs["Scale"].default_value = 7.0
        nt.links.new(pos, v1.inputs["Vector"])
        v2 = nt.nodes.new("ShaderNodeTexVoronoi")
        v2.inputs["Scale"].default_value = 23.0
        nt.links.new(pos, v2.inputs["Vector"])
        stones = [(0.0, srgb(64, 64, 62)), (0.22, srgb(104, 98, 88)), (0.45, srgb(146, 138, 120)),
                  (0.62, srgb(92, 90, 96)), (0.76, srgb(136, 96, 70)), (0.88, srgb(172, 164, 146)), (1.0, srgb(80, 84, 78))]
        c1 = color_ramp(nt, separate(nt, v1.outputs["Color"])["X"], stones)
        c2 = color_ramp(nt, separate(nt, v2.outputs["Color"])["X"], stones)
        gravel = mix_rgb(nt, 0.45, c1, c2)
        # pebble gaps: darker sand between the stones
        gap = map_range(nt, v1.outputs["Distance"], 0.42, 0.62)
        gravel = mix_rgb(nt, nmath(nt, "MULTIPLY", gap, 0.6), gravel, srgb(70, 64, 54))
        algae = map_range(nt, z, -0.15, -0.6)
        gravel = mix_rgb(nt, nmath(nt, "MULTIPLY", algae, 0.35), gravel, srgb(70, 84, 46))
        wet = map_range(nt, z, 0.22, 0.0)
        gravel = vmath(nt, "SCALE", gravel, scale=nmath(nt, "MULTIPLY_ADD", wet, -0.42, 1.0))
        nz_ = nt.nodes.new("ShaderNodeTexNoise")
        nz_.inputs["Scale"].default_value = 0.35
        nz_.inputs["Detail"].default_value = 6.0
        nt.links.new(pos, nz_.inputs["Vector"])
        veg_c = color_ramp(nt, nz_.outputs["Fac"], [(0.3, srgb(34, 46, 22)), (0.5, srgb(58, 76, 34)),
                                                    (0.62, srgb(96, 78, 46)), (0.75, srgb(48, 62, 30))])
        veg = nmath(nt, "MULTIPLY", map_range(nt, z, 0.9, 2.0), nmath(nt, "SUBTRACT", 1.0, steep))
        rock_c = color_ramp(nt, nz_.outputs["Fac"], [(0.3, srgb(70, 72, 72)), (0.55, srgb(112, 110, 104)),
                                                     (0.75, srgb(84, 86, 84))])
        rock = nmath(nt, "MULTIPLY", steep, map_range(nt, z, 0.05, 0.6))
        col = mix_rgb(nt, veg, gravel, veg_c)
        col = mix_rgb(nt, rock, col, rock_c)
        h = nmath(nt, "ADD", nmath(nt, "MULTIPLY", map_range(nt, v1.outputs["Distance"], 0.0, 0.6, 1.0, 0.0), 0.7),
                  nmath(nt, "MULTIPLY", map_range(nt, v2.outputs["Distance"], 0.0, 0.6, 1.0, 0.0), 0.3))
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Distance"].default_value = 0.02
        nt.links.new(nmath(nt, "MULTIPLY", nmath(nt, "SUBTRACT", 1.0, veg), 0.6), bump.inputs["Strength"])
        nt.links.new(h, bump.inputs["Height"])
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
        nt.links.new(map_range(nt, wet, 0.0, 1.0, 0.85, 0.32, smooth=False), b.inputs["Roughness"])
        return b.outputs[0]
    M.terrain = new_material("RiverValley", terrain, atmos)

    def far_terrain(nt):
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        z = separate(nt, geo.outputs["Position"])["Z"]
        nz = separate(nt, geo.outputs["Normal"])["Z"]
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 0.02
        n.inputs["Detail"].default_value = 8.0
        nt.links.new(geo.outputs["Position"], n.inputs["Vector"])
        forest = color_ramp(nt, n.outputs["Fac"], [(0.35, srgb(18, 30, 20)), (0.6, srgb(30, 46, 28))])
        rock = color_ramp(nt, n.outputs["Fac"], [(0.35, srgb(80, 84, 90)), (0.65, srgb(120, 122, 126))])
        tree_line = map_range(nt, nmath(nt, "ADD", z, nmath(nt, "MULTIPLY", n.outputs["Fac"], 120.0)), 380.0, 520.0)
        col = mix_rgb(nt, nmath(nt, "MAXIMUM", tree_line, map_range(nt, nz, 0.62, 0.45)), forest, rock)
        snow = nmath(nt, "MULTIPLY", map_range(nt, nmath(nt, "ADD", z, nmath(nt, "MULTIPLY", n.outputs["Fac"], 160.0)), 640.0, 760.0),
                     map_range(nt, nz, 0.45, 0.7))
        col = mix_rgb(nt, snow, col, srgb(236, 240, 246))
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.9
        return b.outputs[0]
    M.far_terrain = new_material("FarValley", far_terrain, atmos)

    def boulder(nt):
        base = box_image(nt, asset("babylon", "rock_baseColor.jpg"), 0.9)
        nrm = box_image(nt, asset("babylon", "rock_normal.jpg"), 0.9, non_color=True)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = 0.8
        nt.links.new(nrm.outputs["Color"], nm.inputs["Color"])
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        z = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"])["Z"]
        tint = mix_rgb(nt, oi.outputs["Random"], srgb(150, 148, 140), srgb(196, 188, 170))
        col = vmath(nt, "MULTIPLY", base.outputs["Color"], tint)
        col = vmath(nt, "SCALE", col, scale=nmath(nt, "MULTIPLY_ADD", map_range(nt, z, 0.25, 0.0), -0.4, 1.0))  # wet
        moss = nmath(nt, "MULTIPLY", map_range(nt, z, 0.3, 0.6), map_range(nt, separate(
            nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Normal"])["Z"], 0.5, 0.9))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", moss, 0.8), col, srgb(62, 80, 36))
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
        nt.links.new(map_range(nt, z, 0.25, 0.0, 0.8, 0.35, smooth=False), b.inputs["Roughness"])
        return b.outputs[0]
    M.boulder = new_material("Boulder", boulder, atmos)

    def cobble(nt):
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        z = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"])["Z"]
        col = color_ramp(nt, oi.outputs["Random"], [(0.0, srgb(78, 78, 76)), (0.2, srgb(120, 112, 98)),
                                                   (0.4, srgb(160, 150, 130)), (0.55, srgb(98, 96, 104)),
                                                   (0.7, srgb(146, 104, 76)), (0.85, srgb(186, 178, 160)),
                                                   (1.0, srgb(92, 98, 88))])
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 40.0
        col = vmath(nt, "SCALE", col, scale=nmath(nt, "MULTIPLY_ADD", n.outputs["Fac"], 0.35, 0.8))
        col = vmath(nt, "SCALE", col, scale=nmath(nt, "MULTIPLY_ADD", map_range(nt, z, 0.15, 0.0), -0.38, 1.0))
        algae = map_range(nt, z, -0.15, -0.6)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", algae, 0.3), col, srgb(74, 88, 50))
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.3
        nt.links.new(n.outputs["Fac"], bump.inputs["Height"])
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
        nt.links.new(map_range(nt, z, 0.15, 0.0, 0.8, 0.55, smooth=False), b.inputs["Roughness"])
        b.inputs["Specular IOR Level"].default_value = 0.3
        return b.outputs[0]
    M.cobble = new_material("Cobble", cobble, atmos)

    # ---- forest -------------------------------------------------------------
    def needles(nt):
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 6.0
        n.inputs["Detail"].default_value = 4.0
        nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], n.inputs["Vector"])
        base = color_ramp(nt, oi.outputs["Random"], [(0.0, srgb(30, 56, 36)), (0.5, srgb(42, 72, 42)),
                                                    (1.0, srgb(58, 82, 46))])
        col = vmath(nt, "SCALE", base, scale=nmath(nt, "MULTIPLY_ADD", n.outputs["Fac"], 0.9, 0.55))
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.6
        nt.links.new(n.outputs["Fac"], bump.inputs["Height"])
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
        b.inputs["Roughness"].default_value = 0.8
        b.inputs["Subsurface Weight"].default_value = 0.15
        b.inputs["Subsurface Radius"].default_value = (0.3, 0.6, 0.2)
        return b.outputs[0]
    M.needles = new_material("PineNeedles", needles, atmos)
    M.bark = pbr("PineBark", srgb(70, 52, 40), rough=0.9)

    # ---- sockeye salmon (spawning colours) -------------------------------------
    def sockeye(nt):
        rest = separate(nt, attr_vec(nt, "rest"))
        zrel = attr_float(nt, "zrel")
        x = rest["X"]
        head = map_range(nt, x, 0.30, 0.36)                     # olive green head
        body = color_ramp(nt, zrel, [(0.0, srgb(206, 92, 80)), (0.25, srgb(198, 30, 26)),
                                     (0.6, srgb(186, 22, 22)), (0.88, srgb(120, 18, 22)), (1.0, srgb(90, 20, 22))])
        headc = color_ramp(nt, zrel, [(0.0, srgb(196, 196, 170)), (0.3, srgb(120, 134, 92)), (0.6, srgb(68, 92, 52)),
                                      (1.0, srgb(44, 64, 40))])
        col = mix_rgb(nt, head, body, headc)
        tail = map_range(nt, x, -0.36, -0.42)                   # green tail fin
        col = mix_rgb(nt, tail, col, srgb(60, 84, 50))
        mot = nt.nodes.new("ShaderNodeTexNoise")                 # mottling on the back
        mot.inputs["Scale"].default_value = 14.0
        mot.inputs["Detail"].default_value = 3.0
        nt.links.new(attr_vec(nt, "rest"), mot.inputs["Vector"])
        col = vmath(nt, "SCALE", col, scale=nmath(nt, "MULTIPLY_ADD", nmath(nt, "MULTIPLY", map_range(
            nt, mot.outputs["Fac"], 0.4, 0.7), map_range(nt, zrel, 0.55, 0.9)), -0.35, 1.0))
        scales = nt.nodes.new("ShaderNodeTexVoronoi")
        scales.inputs["Scale"].default_value = 260.0
        nt.links.new(attr_vec(nt, "rest"), scales.inputs["Vector"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.14
        bump.inputs["Distance"].default_value = 0.0005
        nt.links.new(scales.outputs["Distance"], bump.inputs["Height"])
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
        b.inputs["Roughness"].default_value = 0.3
        b.inputs["Coat Weight"].default_value = 0.55               # wet mucus layer
        b.inputs["Coat Roughness"].default_value = 0.08
        b.inputs["Subsurface Weight"].default_value = 0.25          # light through the fins and thin flesh
        b.inputs["Subsurface Radius"].default_value = (0.9, 0.25, 0.15)
        b.inputs["Subsurface Scale"].default_value = 0.012
        b.inputs["Thin Film Thickness"].default_value = 300
        # the inside of the mouth (back faces) reads dark when the jaw opens
        dark = nt.nodes.new("ShaderNodeBsdfDiffuse")
        dark.inputs["Color"].default_value = (*srgb(40, 8, 10), 1.0)
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(nt.nodes.new("ShaderNodeNewGeometry").outputs["Backfacing"], mix.inputs[0])
        nt.links.new(b.outputs[0], mix.inputs[1])
        nt.links.new(dark.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    M.sockeye = new_material("SockeyeSkin", sockeye, atmos)

    def fin(base, margin):
        def build(nt):
            u = attr_float(nt, "finu")
            col = color_ramp(nt, u, [(0.0, base), (1.0, margin)])
            b = nt.nodes.new("ShaderNodeBsdfPrincipled")
            nt.links.new(col, b.inputs["Base Color"])
            b.inputs["Roughness"].default_value = 0.35
            b.inputs["Subsurface Weight"].default_value = 0.5
            b.inputs["Subsurface Scale"].default_value = 0.01
            return b.outputs[0]
        return build
    M.fin_red = new_material("SockeyeFinRed", fin(srgb(150, 26, 26), srgb(96, 30, 30)), atmos)
    M.fin_green = new_material("SockeyeFinGreen", fin(srgb(66, 88, 52), srgb(40, 56, 36)), atmos)
    M.eye = pbr("SockeyeEye", srgb(12, 10, 8), rough=0.08, coat=1.0)
    M.eye.node_tree.nodes["Principled BSDF"].inputs["Coat Roughness"].default_value = 0.02 \
        if "Principled BSDF" in M.eye.node_tree.nodes else None

    # ---- tackle and angler ------------------------------------------------------
    M.rod = pbr("RodBlank", srgb(40, 30, 26), rough=0.22, coat=1.0)
    M.cork = pbr("CorkGrip", srgb(190, 150, 104), rough=0.8)
    M.reel = pbr("ReelBody", srgb(150, 152, 156), rough=0.25, metal=1.0)
    M.reel_dark = pbr("ReelDark", srgb(22, 22, 24), rough=0.35, metal=0.5)
    M.weight = pbr("LeadWeight", srgb(64, 66, 70), rough=0.5, metal=0.3)
    M.hook = pbr("HookSteel", srgb(176, 180, 186), rough=0.2, metal=1.0)

    def yarn(nt):
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 160.0
        n.inputs["Detail"].default_value = 2.0
        col = color_ramp(nt, attr_float(nt, "yarn"), [(0.0, srgb(255, 92, 150)), (0.45, srgb(255, 96, 140)),
                                                   (0.55, srgb(255, 140, 40)), (1.0, srgb(255, 128, 36))])
        col = vmath(nt, "SCALE", col, scale=nmath(nt, "MULTIPLY_ADD", n.outputs["Fac"], 0.5, 0.75))
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.9
        b.inputs["Sheen Weight"].default_value = 0.8
        b.inputs["Subsurface Weight"].default_value = 0.4
        b.inputs["Subsurface Scale"].default_value = 0.004
        nt.links.new(col, b.inputs["Emission Color"])
        b.inputs["Emission Strength"].default_value = 0.06        # fluorescent yarn reads bright in the green water
        return b.outputs[0]
    M.yarn = new_material("YarnFly", yarn, atmos)

    def line(nt):
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        b.inputs["Base Color"].default_value = (*srgb(232, 236, 220), 1.0)
        b.inputs["Roughness"].default_value = 0.3
        b.inputs["Emission Color"].default_value = (*srgb(236, 240, 226), 1.0)
        b.inputs["Emission Strength"].default_value = 0.35       # monofilament catching the light
        return b.outputs[0]
    M.line = new_material("FishingLine", line, atmos)
    M.waders = pbr("Waders", srgb(66, 70, 58), rough=0.7, coat=0.2)
    M.boots = pbr("WadingBoots", srgb(40, 38, 34), rough=0.8)
    M.jacket = pbr("Jacket", srgb(176, 78, 40), rough=0.75)
    M.skin = pbr("Skin", srgb(200, 146, 112), rough=0.5, sss=0.25)
    M.cap = pbr("Cap", srgb(52, 60, 50), rough=0.8)
    M.shades = pbr("Sunglasses", srgb(8, 10, 12), rough=0.05, spec=1.0, coat=1.0)
    M.hair = pbr("Beard", srgb(70, 50, 36), rough=0.9)

    # ---- water and effects --------------------------------------------------------
    M.water = build_water_material(atmos)

    def backdrop(nt):
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        inc = separate(nt, geo.outputs["Incoming"])
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(uw_color(nt, nmath(nt, "MULTIPLY", inc["Z"], -1.0)), em.inputs["Color"])
        # in the over-under shot only the half below the waterline is under water
        hide = nmath(nt, "MULTIPLY", scene_attr(nt, "split"),
                     nmath(nt, "GREATER_THAN", separate(nt, geo.outputs["Position"])["Z"], 0.0))
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(hide, mix.inputs[0])
        nt.links.new(em.outputs[0], mix.inputs[1])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[2])
        return mix.outputs[0]
    M.backdrop = new_material("UnderwaterBackdrop", backdrop)

    def under_vis(nt):
        """1 when this effect (only ever placed under water) should show: lens under, or the split shot."""
        return nmath(nt, "MAXIMUM", scene_attr(nt, "cam_under"), scene_attr(nt, "split"))

    def beam(nt):
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.5
        core = nmath(nt, "POWER", nmath(nt, "SUBTRACT", 1.0, lw.outputs["Facing"], clamp=True), 1.6)
        z = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"])["Z"]
        near = map_range(nt, nt.nodes.new("ShaderNodeCameraData").outputs["View Distance"], 1.0, 4.0)
        fade = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", map_range(nt, z, -0.05, -0.6),
                                           nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", z, 1.0 / 2.5))), near)
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        t = frame_value(nt, "frame/24")
        flick = nmath(nt, "MULTIPLY_ADD", nmath(nt, "SINE", nmath(nt, "MULTIPLY_ADD", t, 1.7,
                                                                  nmath(nt, "MULTIPLY", oi.outputs["Random"], 31.0))),
                      0.35, 0.65)
        k = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", core, fade), nmath(nt, "MULTIPLY", flick, under_vis(nt)))
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(200, 250, 230), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", k, 0.13), em.inputs["Strength"])
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], add.inputs[1])
        return add.outputs[0]
    M.beam = new_material("SunShaft", beam, blended=True)

    def air_ray(nt):
        """Sun rays through the spray and the trees, seen against the shade (above water only)."""
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.5
        core = nmath(nt, "POWER", nmath(nt, "SUBTRACT", 1.0, lw.outputs["Facing"], clamp=True), 2.0)
        gen = separate(nt, nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"])
        along = nmath(nt, "MULTIPLY", map_range(nt, gen["Z"], 0.0, 0.25), map_range(nt, gen["Z"], 1.0, 0.55))
        near = map_range(nt, nt.nodes.new("ShaderNodeCameraData").outputs["View Distance"], 12.0, 40.0)
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        k = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", core, along), nmath(nt, "MULTIPLY", near, nmath(
            nt, "SUBTRACT", 1.0, scene_attr(nt, "cam_under"))))
        k = nmath(nt, "MULTIPLY", k, nmath(nt, "MULTIPLY_ADD", oi.outputs["Random"], 0.6, 0.4))
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(255, 236, 196), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", k, 0.16), em.inputs["Strength"])
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], add.inputs[1])
        return add.outputs[0]
    M.air_ray = new_material("SunRayAir", air_ray, blended=True)

    def mist(nt):
        gen = separate(nt, nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"])
        edge = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", map_range(nt, gen["X"], 0.0, 0.3), map_range(nt, gen["X"], 1.0, 0.7)),
                     nmath(nt, "MULTIPLY", map_range(nt, gen["Y"], 0.0, 0.3), map_range(nt, gen["Y"], 1.0, 0.7)))
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.noise_dimensions = "4D"
        n.inputs["Scale"].default_value = 0.06
        n.inputs["Detail"].default_value = 4.0
        nt.links.new(nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"], n.inputs["Vector"])
        add_frame_driver(n.inputs["W"], "default_value", "frame*0.002")
        near = map_range(nt, nt.nodes.new("ShaderNodeCameraData").outputs["View Distance"], 4.0, 30.0)
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        a = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", edge, map_range(nt, n.outputs["Fac"], 0.38, 0.7)),
                  nmath(nt, "MULTIPLY", near, oi.outputs["Alpha"]))
        a = nmath(nt, "MULTIPLY", a, nmath(nt, "SUBTRACT", 1.0, scene_attr(nt, "cam_under")))
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(222, 232, 236), 1.0)
        em.inputs["Strength"].default_value = 0.85
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(a, mix.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    M.mist = new_material("Mist", mist, blended=True)

    def silt(nt):
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(214, 236, 214), 1.0)
        em.inputs["Strength"].default_value = 0.7
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(nmath(nt, "MULTIPLY", under_vis(nt), 0.6), mix.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    M.silt = new_material("Silt", silt, blended=True)

    def bubbles(nt):
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        rim = nmath(nt, "POWER", lw.outputs["Facing"], 1.5)
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(236, 252, 248), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY_ADD", rim, 1.6, 0.25), em.inputs["Strength"])
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(nmath(nt, "MULTIPLY_ADD", rim, 0.6, 0.2), mix.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    M.bubbles = new_material("Bubbles", bubbles, blended=True)

    def spray(nt):
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(244, 250, 250), 1.0)
        em.inputs["Strength"].default_value = 1.3
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        core = nmath(nt, "POWER", nmath(nt, "SUBTRACT", 1.0, lw.outputs["Facing"]), 1.2)
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(nmath(nt, "MULTIPLY", core, 0.85), mix.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    M.spray = new_material("Spray", spray, blended=True)

    def zone(nt):
        """Strike zone band: a soft gold glow along the bottom (diagram element inside the water)."""
        gen = separate(nt, nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"])
        # two thin gold edge lines over a faint fill, so the fish inside the band stay readable
        edges = nmath(nt, "MAXIMUM", map_range(nt, gen["Z"], 0.07, 0.0), map_range(nt, gen["Z"], 0.93, 1.0))
        prof = nmath(nt, "MULTIPLY_ADD", edges, 0.9, 0.06)
        ends = nmath(nt, "MULTIPLY", map_range(nt, gen["Y"], 0.0, 0.15), map_range(nt, gen["Y"], 1.0, 0.85))
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        a = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", prof, ends), oi.outputs["Alpha"])
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*GOLD, 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", a, 0.9), em.inputs["Strength"])
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], add.inputs[1])
        return add.outputs[0]
    M.zone = new_material("StrikeZone", zone, blended=True)

    def lane(nt):
        """Migration lane on the water: a glowing ribbon with chevrons streaming upstream."""
        gen = separate(nt, nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"])
        attr_along = attr_float(nt, "along")
        t = frame_value(nt, "frame/24")
        dash = nmath(nt, "FRACT", nmath(nt, "SUBTRACT", nmath(nt, "MULTIPLY", attr_along, 0.25),
                                         nmath(nt, "MULTIPLY", t, 0.9)))
        across = nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", attr_float(nt, "across"), 0.5))
        chev = map_range(nt, nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", nmath(nt, "ADD", dash, nmath(
            nt, "MULTIPLY", across, 0.8)), 0.5)), 0.12, 0.06)
        edge = map_range(nt, across, 0.5, 0.4)
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        a = nmath(nt, "MULTIPLY", nmath(nt, "MAXIMUM", nmath(nt, "MULTIPLY", chev, 0.9), 0.18),
                  nmath(nt, "MULTIPLY", edge, oi.outputs["Alpha"]))
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*GOLD, 1.0)
        em.inputs["Strength"].default_value = 1.4
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(a, mix.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    M.lane = new_material("MigrationLane", lane, blended=True)

    # ---- overlays --------------------------------------------------------------
    def scrim(top_side):
        def build(nt):
            oi = nt.nodes.new("ShaderNodeObjectInfo")
            y = separate(nt, nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"])["Y"]
            prof = nmath(nt, "POWER", y if top_side else nmath(nt, "SUBTRACT", 1.0, y), 1.7)
            tr = nt.nodes.new("ShaderNodeBsdfTransparent")
            em = nt.nodes.new("ShaderNodeEmission")
            em.inputs["Color"].default_value = (0.0, 0.006, 0.008, 1.0)
            mix = nt.nodes.new("ShaderNodeMixShader")
            nt.links.new(nmath(nt, "MULTIPLY", oi.outputs["Alpha"], prof), mix.inputs[0])
            nt.links.new(tr.outputs[0], mix.inputs[1])
            nt.links.new(em.outputs[0], mix.inputs[2])
            return mix.outputs[0]
        return build
    M.scrim_top = new_material("HudScrimTop", scrim(True), blended=True)
    M.scrim_bot = new_material("HudScrimBottom", scrim(False), blended=True)
    M.hud = new_material("HudColor", emissive_alpha(), blended=True)
    M.hud_white = new_material("HudWhite", emissive_alpha(WHITE), blended=True)
    M.hud_accent = new_material("HudAccent", emissive_alpha(GOLD), blended=True)
    M.hud_shadow = new_material("HudShadow", emissive_alpha(tuple(c * 0.2 for c in INK)), blended=True)
    M.highlight = new_material("GearHighlight", emissive_alpha(), blended=True)
    # Overlays use dithered transparency: blended surfaces write no motion vectors, so motion blur would
    # smear them with whatever moves behind them; dithered ones keep their own (still) motion.
    for m in (M.hud, M.hud_white, M.hud_accent, M.hud_shadow, M.highlight):
        m.use_transparency_overlap = True
        m.surface_render_method = "DITHERED"
    for m in (M.beam, M.air_ray, M.mist, M.silt, M.bubbles, M.spray, M.zone, M.lane, M.highlight, M.hud,
              M.hud_white, M.hud_accent, M.hud_shadow, M.scrim_top, M.scrim_bot):
        m.use_transparent_shadow = True
    return M


def frame_value(nt, expr):
    """A shader value driven by the frame number (simple expression, no Python auto-run needed)."""
    v = nt.nodes.new("ShaderNodeValue")
    add_frame_driver(v.outputs[0], "default_value", expr)
    return v.outputs[0]


def attr_vec(nt, name):
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_name = name
    return a.outputs["Vector"]


def attr_float(nt, name):
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_name = name
    return a.outputs["Fac"]


def build_water_material(atmos):
    """River surface: clear turquoise water over the gravel, white water streaming through the riffles,
    and from below a bright Snell's window ringed by total internal reflection."""
    m = bpy.data.materials.new("River")
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    pos = separate(nt, geo.outputs["Position"])
    B = NodeB(nt)
    r = riffle(pos["Y"], B)
    t = frame_value(nt, "frame/24")
    # White water: streaks advected downstream, dense in the riffles, a little in the tailout
    fn = nt.nodes.new("ShaderNodeTexNoise")
    fn.noise_dimensions = "4D"
    fn.inputs["Scale"].default_value = 1.0
    fn.inputs["Detail"].default_value = 8.0
    fn.inputs["Roughness"].default_value = 0.62
    nt.links.new(combine(nt, nmath(nt, "MULTIPLY", pos["X"], 1.3),
                         nmath(nt, "MULTIPLY", nmath(nt, "ADD", pos["Y"], nmath(nt, "MULTIPLY", t, 1.6)), 0.3), 0.0),
                 fn.inputs["Vector"])
    nt.links.new(nmath(nt, "MULTIPLY", t, 0.4), fn.inputs["W"])
    streak = map_range(nt, fn.outputs["Fac"], 0.5, 0.72)
    foam = nmath(nt, "MULTIPLY", streak, nmath(nt, "MULTIPLY_ADD", r, 0.85, 0.04), clamp=True)
    # Ripples: two samples of the downloaded water normal map, flowing downstream
    def ripple(scale, dx, dy):
        mp = nt.nodes.new("ShaderNodeMapping")
        nt.links.new(combine(nt, pos["X"], pos["Y"], 0.0), mp.inputs["Vector"])
        mp.inputs["Scale"].default_value = (1.0 / scale, 1.0 / scale, 1.0)
        add_frame_driver(mp.inputs["Location"], "default_value", f"frame*{dx}", index=0)
        add_frame_driver(mp.inputs["Location"], "default_value", f"frame*{dy}", index=1)
        return image_node(nt, asset("babylon", "waterbump.png"), True, mp.outputs[0]).outputs["Color"]
    rip = separate(nt, vmath(nt, "SUBTRACT", mix_rgb(nt, 0.5, ripple(2.6, 0.0004, 0.012), ripple(0.9, -0.0007, 0.02)),
                             (0.5, 0.5, 0.5)))
    tilt = nmath(nt, "MULTIPLY_ADD", r, 0.5, 0.28)
    normal = vmath(nt, "NORMALIZE", vmath(nt, "ADD", geo.outputs["Normal"],
                                          combine(nt, nmath(nt, "MULTIPLY", rip["X"], tilt),
                                                  nmath(nt, "MULTIPLY", rip["Y"], tilt), 0.0)))
    # Seen from above, the river is a see-through turquoise tint (the bed and fish below carry their own
    # depth tint) plus a reflective skin whose opacity follows Fresnel; foam is opaque white.
    top = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(mix_rgb(nt, foam, srgb(8, 58, 56), srgb(240, 248, 246)), top.inputs["Base Color"])
    nt.links.new(normal, top.inputs["Normal"])
    nt.links.new(map_range(nt, foam, 0.0, 1.0, 0.03, 0.6, smooth=False), top.inputs["Roughness"])
    top.inputs["IOR"].default_value = 1.333
    top.inputs["Specular IOR Level"].default_value = 0.5
    top.inputs["Emission Color"].default_value = (*srgb(20, 120, 112), 1.0)
    top.inputs["Emission Strength"].default_value = 0.04
    fres = nt.nodes.new("ShaderNodeFresnel")
    fres.inputs["IOR"].default_value = 1.333
    nt.links.new(normal, fres.inputs["Normal"])
    alpha = nmath(nt, "ADD", nmath(nt, "MULTIPLY_ADD", fres.outputs["Fac"], 0.9, 0.10), foam, clamp=True)
    tint = nt.nodes.new("ShaderNodeBsdfTransparent")
    tint.inputs["Color"].default_value = (*srgb(170, 232, 222), 1.0)
    skin = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(alpha, skin.inputs[0])
    nt.links.new(tint.outputs[0], skin.inputs[1])
    nt.links.new(top.outputs[0], skin.inputs[2])
    g = nt.nodes.new("ShaderNodeGroup")
    g.node_tree = atmos
    nt.links.new(skin.outputs[0], g.inputs["Shader"])
    # Below: Snell's window (48.6 degrees) and total internal reflection around it
    n_under = vmath(nt, "NORMALIZE", vmath(nt, "ADD", normal, combine(nt, nmath(nt, "MULTIPLY", rip["X"], 0.6),
                                                                   nmath(nt, "MULTIPLY", rip["Y"], 0.6), 0.0)))
    cos_t = nmath(nt, "ABSOLUTE", vmath(nt, "DOT_PRODUCT", geo.outputs["Incoming"], n_under))
    window = map_range(nt, cos_t, 0.56, 0.76)
    ring = nmath(nt, "MULTIPLY", map_range(nt, cos_t, 0.55, 0.64), map_range(nt, cos_t, 0.74, 0.64))
    sky_in = vmath(nt, "SCALE", srgb(170, 226, 220), scale=nmath(nt, "MULTIPLY_ADD", rip["X"], 1.2, 0.8))
    sky_in = mix_rgb(nt, nmath(nt, "MULTIPLY", foam, 0.95), sky_in, srgb(250, 252, 252))
    tir = vmath(nt, "SCALE", srgb(14, 84, 80), scale=nmath(nt, "MULTIPLY_ADD", rip["Y"], 0.5, 0.85))
    tir = mix_rgb(nt, nmath(nt, "MULTIPLY", foam, 0.8), tir, srgb(190, 226, 222))
    below = mix_rgb(nt, window, tir, sky_in)
    below = vmath(nt, "ADD", below, vmath(nt, "SCALE", srgb(210, 250, 240), scale=nmath(nt, "MULTIPLY", ring, 0.35)))
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(below, em.inputs["Color"])
    gu = nt.nodes.new("ShaderNodeGroup")
    gu.node_tree = atmos
    nt.links.new(em.outputs[0], gu.inputs["Shader"])
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(geo.outputs["Backfacing"], mix.inputs[0])
    nt.links.new(g.outputs["Shader"], mix.inputs[1])
    nt.links.new(gu.outputs["Shader"], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.use_backface_culling = False
    m.surface_render_method = "DITHERED"
    m.use_transparent_shadow = True
    return m


# --------------------------------------------------------------------------
# Environment objects
# --------------------------------------------------------------------------
def build_water(col, M):
    """Surface mesh following the channel; ripples and riffle standing waves from geometry nodes."""
    ys = axis_coords(-140.0, 160.0, -34.0, 50.0, 0.35, 1.0)
    nx = 96
    verts, faces = [], []
    for j, y in enumerate(ys):
        cx, w = river_cx(y), river_w(y)
        for i in range(nx):
            u = -1.0 - 4.0 / w + (2.0 + 8.0 / w) * i / (nx - 1)
            verts.append((cx + u * w, y, 0.0))
    for j in range(len(ys) - 1):
        for i in range(nx - 1):
            a = j * nx + i
            faces.append((a, a + 1, a + 1 + nx, a + nx))
    me = bpy.data.meshes.new("RiverSurface")
    me.from_pydata(verts, [], faces)
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(M.water)
    ob = bpy.data.objects.new("RiverSurface", me)
    col.objects.link(ob)
    ng, gi, go = new_group("RiverSurfaceGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    B = NodeB(ng)
    x, y, _z = gn_xyz(ng)
    t = gn_time(ng)
    r = riffle(y, B)
    n = ng.nodes.new("ShaderNodeTexNoise")
    n.noise_dimensions = "4D"
    n.inputs["Scale"].default_value = 0.9
    n.inputs["Detail"].default_value = 3.0
    ng.links.new(combine(ng, x, nmath(ng, "ADD", y, nmath(ng, "MULTIPLY", t, 0.8)), 0.0), n.inputs["Vector"])
    ng.links.new(nmath(ng, "MULTIPLY", t, 0.3), n.inputs["W"])
    amp = nmath(ng, "MULTIPLY_ADD", r, 0.05, 0.008)
    standing = nmath(ng, "MULTIPLY", nmath(ng, "SINE", nmath(ng, "ADD", nmath(ng, "MULTIPLY", y, 2.3),
                                                             nmath(ng, "MULTIPLY", nmath(ng, "SINE", nmath(ng, "MULTIPLY", x, 0.7)), 1.4))),
                     nmath(ng, "MULTIPLY", r, 0.03))
    dz = nmath(ng, "ADD", nmath(ng, "MULTIPLY", nmath(ng, "SUBTRACT", n.outputs["Fac"], 0.5), nmath(ng, "MULTIPLY", amp, 2.0)),
               standing)
    sp = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, 0.0, 0.0, dz), sp.inputs["Offset"])
    ng.links.new(sp.outputs[0], go.inputs["Geometry"])
    ob.modifiers.new("Ripples", "NODES").node_group = ng
    ob.visible_shadow = False
    print(f"river surface: {len(verts)} verts")
    return ob


def build_backdrop(col, M, cam):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=300.0)
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    back = mesh_object("UnderwaterBackdrop", bm, col, [M.backdrop], recalc=False)
    back.visible_shadow = False
    follow(back, cam)
    return back


def boulder_mesh(seed, M):
    rng = random.Random(seed)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
    ph = [rng.uniform(0, 6.28) for _ in range(9)]
    for v in bm.verts:
        p = v.co
        n = (0.18 * math.sin(2.1 * p.x + ph[0]) * math.sin(1.7 * p.y + ph[1]) * math.sin(1.9 * p.z + ph[2])
             + 0.08 * math.sin(5.3 * p.x + ph[3]) * math.sin(4.9 * p.y + ph[4])
             + 0.04 * math.sin(11.0 * p.z + ph[5]) * math.sin(9.0 * p.x + ph[6]))
        flat = 0.72 if p.z > 0 else 0.5
        v.co = Vector((p.x * (1 + n) * rng.uniform(0.95, 1.05), p.y * (1 + n) * 0.85, p.z * (1 + n) * flat))
    me = bpy.data.meshes.new(f"Boulder{seed}")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(M.boulder)
    return me


def build_boulders(col, M, avoid):
    """Boulders: big ones along the steep left bank, a few in the riffles, small ones on the gravel bar.
    avoid: list of (point, radius) the rocks must keep clear of (camera path, angler, school, rig)."""
    meshes = [boulder_mesh(s, M) for s in range(6)]
    rng = random.Random(31)
    placed = []
    kd = kdtree.KDTree(len(avoid))
    for i, (p, _r) in enumerate(avoid):
        kd.insert(p, i)
    kd.balance()

    def ok(c, r):
        for p, idx, d in kd.find_range(c, r + 6.0):
            if d < r + avoid[idx][1]:
                return False
        return all((c - q).length > r + rq + 0.3 for q, rq in placed)
    specs = ([(-1.05, -0.85, 0.7, 1.9, 90)] + [(-0.6, 0.3, 0.35, 0.8, 14)] + [(1.04, 1.4, 0.25, 0.55, 26)]
             + [(-1.3, -1.05, 0.8, 2.2, 50)])
    for u0, u1, r0, r1, n in specs:
        for _ in range(n * 4):
            if n <= 0:
                break
            y = rng.uniform(-110.0, 130.0)
            if u0 > -0.7 and u1 < 0.5 and -34.0 < y < 40.0:
                continue                          # mid-river rocks only in the riffles
            u = rng.uniform(u0, u1)
            r = rng.uniform(r0, r1)
            x = rx(y, u)
            c = Vector((x, y, bed_z(x, y) + 0.15 * r))
            if not ok(c, r):
                continue
            ob = bpy.data.objects.new(f"Boulder{len(placed):03d}", meshes[rng.randrange(6)])
            col.objects.link(ob)
            ob.location = c
            ob.rotation_euler = (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0, 6.28))
            ob.scale = (r, r, r)
            placed.append((c, r))
            n -= 1
    print(f"boulders: {len(placed)}")
    return placed


def cobble_meshes(M):
    out = []
    for k in range(5):
        rng = random.Random(400 + k)
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        sx, sy, sz = rng.uniform(0.8, 1.2), rng.uniform(0.6, 0.95), rng.uniform(0.28, 0.42)
        for v in bm.verts:
            j = 1.0 + 0.08 * math.sin(3.0 * v.co.x + k) * math.sin(2.7 * v.co.y + 2 * k)
            v.co = Vector((v.co.x * sx * j, v.co.y * sy * j, v.co.z * sz * j))
        me = bpy.data.meshes.new(f"Cobble{k}")
        bm.to_mesh(me)
        bm.free()
        for p in me.polygons:
            p.use_smooth = True
        me.materials.append(M.cobble)
        out.append(me)
    return out


def scatter_instances(name, col, lib_objs, pts, attrs, cam=None, cull=None):
    """Instance a library of objects on points (attrs: idx, yaw, scl, tilt), optionally culled by camera distance."""
    me = bpy.data.meshes.new(name + "Pts")
    me.from_pydata([tuple(p) for p in pts], [], [])
    for an, kind, vals in attrs:
        a = me.attributes.new(an, kind, "POINT")
        a.data.foreach_set("value", vals)
    lib = bpy.data.collections.new(name + "Lib")
    for o in lib_objs:
        lib.objects.link(o)
    ng, gi, go = new_group(name + "GN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links
    geo = gi.outputs["Geometry"]
    if cam is not None and cull:
        oi = nodes.new("GeometryNodeObjectInfo")
        oi.inputs["Object"].default_value = cam
        dist = vmath(ng, "DISTANCE", nodes.new("GeometryNodeInputPosition").outputs[0], oi.outputs["Location"])
        dele = nodes.new("GeometryNodeDeleteGeometry")
        dele.domain = "POINT"
        links.new(geo, dele.inputs["Geometry"])
        links.new(nmath(ng, "GREATER_THAN", dist, cull), dele.inputs["Selection"])
        geo = dele.outputs[0]
    ci = nodes.new("GeometryNodeCollectionInfo")
    ci.inputs["Collection"].default_value = lib
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    inst = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(geo, inst.inputs["Points"])
    links.new(ci.outputs[0], inst.inputs["Instance"])
    inst.inputs["Pick Instance"].default_value = True

    def nattr(nm, kind="FLOAT"):
        r = nodes.new("GeometryNodeInputNamedAttribute")
        r.data_type = kind
        r.inputs["Name"].default_value = nm
        return r.outputs[0]
    links.new(nattr("idx", "INT"), inst.inputs["Instance Index"])
    links.new(combine(ng, nattr("tilt"), nmath(ng, "MULTIPLY", nattr("tilt"), 0.7), nattr("yaw")), inst.inputs["Rotation"])
    s = nattr("scl")
    links.new(combine(ng, s, s, s), inst.inputs["Scale"])
    links.new(inst.outputs[0], go.inputs["Geometry"])
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    ob.modifiers.new(name, "NODES").node_group = ng
    return ob


def build_cobbles(col, M, cam, keep_out):
    """Cobbles on the bed where the camera comes close: the tailout and the pool head. keep_out: [(p, r)]."""
    lib = [bpy.data.objects.new(me.name, me) for me in cobble_meshes(M)]
    rng = random.Random(41)
    pts, idx, yaw, scl, tilt = [], [], [], [], []
    regions = [((-24.0, -2.0), (0.25, 1.35), 9000), ((16.0, 42.0), (-0.7, 0.6), 2600), ((-90.0, 120.0), (-0.9, 1.3), 6000)]
    for (y0, y1), (u0, u1), n in regions:
        for _ in range(n):
            y = rng.uniform(y0, y1)
            u = rng.uniform(u0, u1)
            x = rx(y, u)
            p = Vector((x, y, bed_z(x, y) - 0.01))
            if any((p - q).length < r for q, r in keep_out):
                continue
            s = rng.uniform(0.025, 0.08) if rng.random() < 0.85 else rng.uniform(0.08, 0.15)
            pts.append(p)
            idx.append(rng.randrange(5))
            yaw.append(rng.uniform(0, 6.28))
            scl.append(s)
            tilt.append(rng.uniform(-0.15, 0.15))
    ob = scatter_instances("Cobbles", col, lib, pts, [("idx", "INT", idx), ("yaw", "FLOAT", yaw),
                                                     ("scl", "FLOAT", scl), ("tilt", "FLOAT", tilt)], cam, 40.0)
    ob.visible_shadow = False          # 3 to 15 cm stones: their sun shadows barely read and cost a lot to render
    print(f"cobbles: {len(pts)}")
    return ob, list(zip(pts, scl))


def pine_meshes(M):
    out = []
    for k in range(4):
        rng = random.Random(500 + k)
        bm = bmesh.new()
        bm_frustum(bm, (0, 0, -0.2), (0, 0, 0.85), 0.022, 0.004, segs=7)
        for f in bm.faces:
            f.material_index = 1
        n0 = len(bm.faces)
        tiers = rng.randint(9, 12)
        for i in range(tiers):
            u = i / tiers
            h = 0.12 + 0.82 * u
            r = (0.30 - 0.26 * u ** 0.9) * rng.uniform(0.85, 1.15)
            top = h + 0.16 - 0.05 * u
            segs = 11
            ring = []
            for s in range(segs):
                a = 2 * math.pi * s / segs + rng.uniform(-0.15, 0.15)
                rr = r * rng.uniform(0.75, 1.15)
                ring.append(bm.verts.new((rr * math.cos(a), rr * math.sin(a), h - rng.uniform(0.0, 0.05))))
            apex = bm.verts.new((rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), top))
            inner = [bm.verts.new((v.co.x * 0.35, v.co.y * 0.35, h + 0.03)) for v in ring]
            for s in range(segs):
                t2 = (s + 1) % segs
                bm.faces.new((ring[s], ring[t2], apex))
                bm.faces.new((ring[t2], ring[s], inner[s], inner[t2]))
        for f in list(bm.faces)[n0:]:
            f.material_index = 0
        me = bpy.data.meshes.new(f"Pine{k}")
        bm.to_mesh(me)
        bm.free()
        me.materials.append(M.needles)
        me.materials.append(M.bark)
        out.append(me)
    return out


def build_forest(col, M, cam_pts):
    keep_out = kdtree.KDTree(len(cam_pts))
    for i, c in enumerate(cam_pts):
        keep_out.insert(Vector((c.x, c.y, 0.0)), i)
    keep_out.balance()
    keep_z = [c.z for c in cam_pts]
    lib = [bpy.data.objects.new(me.name, me) for me in pine_meshes(M)]
    rng = random.Random(51)
    pts, idx, yaw, scl, tilt, trunks = [], [], [], [], [], []
    for _ in range(9000):
        y = rng.uniform(-140.0, 160.0)
        side = 1 if rng.random() < 0.5 else -1
        u = rng.uniform(1.42, 6.5) if side > 0 else -rng.uniform(1.18, 6.5)
        x = rx(y, u)
        if not -108 < x < 118:
            continue
        z = bed_z(x, y)
        if z < 0.6:
            continue
        p = Vector((x, y, z - 0.15))
        h_try = rng.uniform(9.0, 22.0)
        crown = 0.32 * h_try + 1.5
        hit = False
        for _q, ci, _d in keep_out.find_range(Vector((p.x, p.y, 0.0)), crown):
            if keep_z[ci] < p.z + h_try + 1.5:
                hit = True
                break
        if hit:
            continue
        if any((p.xy - q.xy).length < 1.6 for q, _h in trunks[-40:]):
            continue
        h = h_try * (0.75 if abs(u) < 1.6 else 1.0)
        pts.append(p)
        idx.append(rng.randrange(4))
        yaw.append(rng.uniform(0, 6.28))
        scl.append(h)
        tilt.append(rng.uniform(-0.03, 0.03))
        trunks.append((p, h))
    ob = scatter_instances("Pines", col, lib, pts, [("idx", "INT", idx), ("yaw", "FLOAT", yaw),
                                                   ("scl", "FLOAT", scl), ("tilt", "FLOAT", tilt)])
    print(f"pines: {len(pts)}")
    return ob, trunks


def soft_quad(name, col, mat, center, sx, sy, rot, alpha=1.0):
    bm = bmesh.new()
    bm.faces.new([bm.verts.new(v) for v in ((-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0))])
    ob = mesh_object(name, bm, col, [mat], recalc=False)
    ob.location = center
    ob.rotation_euler = rot
    ob.scale = (sx, sy, 1.0)
    ob.color = (1, 1, 1, alpha)
    ob.visible_shadow = False
    return ob


def build_mist_and_rays(col, M):
    """Mist lying over the water and among the trees; sun rays slanting through the spray above the riffle
    and through the pines (seen against the shade, above water only)."""
    rng = random.Random(61)
    objs = []
    for i in range(16):
        y = -120.0 + 16.0 * i + rng.uniform(-4, 4)
        objs.append(soft_quad(f"MistRiver{i}", col, M.mist, (river_cx(y) + rng.uniform(-4, 4), y, rng.uniform(5.0, 9.0)),
                              river_w(y) * 2.6, 22.0, (0, 0, rng.uniform(-0.2, 0.2)), rng.uniform(0.12, 0.25)))
    for i in range(26):
        y = rng.uniform(-130.0, 150.0)
        u = rng.choice([-1, 1]) * rng.uniform(1.8, 4.5)
        x = rx(y, u)
        z = bed_z(x, y) + rng.uniform(4.0, 12.0)
        objs.append(soft_quad(f"MistTrees{i}", col, M.mist, (x, y, z), rng.uniform(30, 60), rng.uniform(8, 14),
                              (math.pi / 2, 0, rng.uniform(-0.6, 0.6) + math.pi / 2 * (1 if u > 0 else -1)), rng.uniform(0.3, 0.6)))
    bm = bmesh.new()
    bm_frustum(bm, (0, 0, 0), (0, 0, 1), 1.0, 1.0, segs=12, cap0=False, cap1=False)
    unit = bpy.data.meshes.new("RayUnit")
    bm.to_mesh(unit)
    bm.free()
    for p in unit.polygons:
        p.use_smooth = True
    unit.materials.append(M.air_ray)
    rot = Vector((0, 0, 1)).rotation_difference(SUN_DIR).to_euler()
    for i in range(22):
        if i < 12:     # over the upper riffle and the pool head: rays through the spray
            y = rng.uniform(30.0, 70.0)
            x = rx(y, rng.uniform(-0.8, 0.8))
            base = Vector((x, y, -0.2))
        else:          # through the pines on the right bank, upstream of the angler (clear of the close shots)
            y = rng.uniform(45.0, 110.0)
            x = rx(y, rng.uniform(1.1, 1.8))
            base = Vector((x, y, bed_z(x, y) - 0.2))
        ob = bpy.data.objects.new(f"SunRay{i}", unit)
        col.objects.link(ob)
        ob.location = base
        ob.rotation_euler = rot
        r = rng.uniform(0.5, 1.6)
        ob.scale = (r, r, rng.uniform(26.0, 40.0))
        ob.visible_shadow = False
        objs.append(ob)
    return objs


def build_beams(col, M, cam):
    """Underwater sun shafts: soft cylinders along the refracted sun direction, scattered around the lens."""
    horiz = Vector((SUN_DIR.x, SUN_DIR.y, 0.0)).normalized()
    sin_w = math.sqrt(1 - SUN_DIR.z ** 2) / 1.333
    d = (-horiz * sin_w + Vector((0, 0, -math.sqrt(1 - sin_w ** 2)))).normalized()
    bm = bmesh.new()
    bm_frustum(bm, (0, 0, 0), (0, 0, -1), 1.0, 1.0, segs=16, cap0=False, cap1=False)
    unit = bpy.data.meshes.new("BeamUnit")
    bm.to_mesh(unit)
    bm.free()
    for p in unit.polygons:
        p.use_smooth = True
    unit.materials.append(M.beam)
    unit_ob = bpy.data.objects.new("BeamUnit", unit)
    ng, gi, go = new_group("BeamsGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links
    g = nodes.new("GeometryNodeMeshGrid")
    g.inputs["Size X"].default_value = g.inputs["Size Y"].default_value = 24.0
    g.inputs["Vertices X"].default_value = g.inputs["Vertices Y"].default_value = 9
    m2p = nodes.new("GeometryNodeMeshToPoints")
    links.new(g.outputs["Mesh"], m2p.inputs["Mesh"])
    sp = nodes.new("GeometryNodeSetPosition")
    links.new(m2p.outputs[0], sp.inputs["Geometry"])
    rnd = nodes.new("FunctionNodeRandomValue")
    rnd.data_type = "FLOAT_VECTOR"
    rnd.inputs["Min"].default_value = (-0.9, -0.9, 0.0)
    rnd.inputs["Max"].default_value = (0.9, 0.9, 0.0)
    links.new(vmath(ng, "ADD", gn_cam_snap(ng, cam, 2.0), rnd.outputs["Value"]), sp.inputs["Offset"])
    keep = nodes.new("FunctionNodeRandomValue")
    keep.data_type = "FLOAT"
    keep.inputs["Seed"].default_value = 3
    dele = nodes.new("GeometryNodeDeleteGeometry")
    dele.domain = "POINT"
    links.new(sp.outputs[0], dele.inputs["Geometry"])
    links.new(nmath(ng, "LESS_THAN", keep.outputs["Value"], 0.45), dele.inputs["Selection"])
    inst = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(dele.outputs[0], inst.inputs["Points"])
    oi = nodes.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = unit_ob
    oi.inputs["As Instance"].default_value = True
    links.new(oi.outputs["Geometry"], inst.inputs["Instance"])
    align = nodes.new("FunctionNodeAlignEulerToVector")
    align.axis = "Z"
    align.inputs["Vector"].default_value = tuple(-d)
    links.new(align.outputs[0], inst.inputs["Rotation"])
    rs = nodes.new("FunctionNodeRandomValue")
    rs.data_type = "FLOAT"
    rs.inputs["Min"].default_value = 0.08
    rs.inputs["Max"].default_value = 0.35
    rs.inputs["Seed"].default_value = 9
    links.new(combine(ng, rs.outputs["Value"], rs.outputs["Value"], 5.0), inst.inputs["Scale"])
    links.new(inst.outputs[0], go.inputs["Geometry"])
    ob = gn_object("SunShafts", col, ng)
    ob.location = (0, 0, -0.06)
    ob.visible_shadow = False
    return ob


def flowing_points(name, col, mat, box, n, vel, size, seed, life=None, rise=0.0):
    """Specks (silt) or bubbles drifting with the current through a box, wrapping round so the stream never
    runs dry. Each point fades in and out at the ends of its loop, so the wrap never shows (or streaks)."""
    (x0, x1), (y0, y1), (z0, z1) = box
    rng = random.Random(seed)
    pts = [(rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(z0, z1)) for _ in range(n)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(pts, [], [])
    for an, vals in (("ph", [rng.random() for _ in range(n)]), ("sz", [rng.uniform(*size) for _ in range(n)]),
                     ("sp", [rng.uniform(0.75, 1.25) for _ in range(n)])):
        a = me.attributes.new(an, "FLOAT", "POINT")
        a.data.foreach_set("value", vals)
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    ng, gi, go = new_group(name + "GN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links

    def nattr(nm):
        r = nodes.new("GeometryNodeInputNamedAttribute")
        r.data_type = "FLOAT"
        r.inputs["Name"].default_value = nm
        return r.outputs[0]
    t = gn_time(ng)
    period = life if life else (y1 - y0) / max(abs(vel[1]), 1e-3)
    s = nmath(ng, "FRACT", nmath(ng, "ADD", nattr("ph"), nmath(ng, "DIVIDE", nmath(ng, "MULTIPLY", t, nattr("sp")), period)))
    # travel: the loop length is the period at the base velocity, centred on the start point
    disp = vmath(ng, "SCALE", Vector((vel[0], vel[1], vel[2] + rise)), scale=nmath(ng, "MULTIPLY", nmath(ng, "SUBTRACT", s, 0.5), period))
    sp = nodes.new("GeometryNodeSetPosition")
    links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    links.new(disp, sp.inputs["Offset"])
    ico = nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 1.0
    ico.inputs["Subdivisions"].default_value = 1
    _x, _y, z = gn_xyz(ng)
    dele = nodes.new("GeometryNodeDeleteGeometry")
    dele.domain = "POINT"
    links.new(sp.outputs[0], dele.inputs["Geometry"])
    links.new(nmath(ng, "GREATER_THAN", z, -0.03), dele.inputs["Selection"])     # never above the surface
    inst = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(dele.outputs[0], inst.inputs["Points"])
    links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
    fade = nmath(ng, "MULTIPLY", map_range(ng, s, 0.0, 0.12), map_range(ng, s, 1.0, 0.88))
    sz = nmath(ng, "MULTIPLY", nattr("sz"), fade)
    links.new(combine(ng, sz, sz, sz), inst.inputs["Scale"])
    setm = nodes.new("GeometryNodeSetMaterial")
    links.new(inst.outputs[0], setm.inputs["Geometry"])
    setm.inputs["Material"].default_value = mat
    links.new(setm.outputs[0], go.inputs["Geometry"])
    ob.modifiers.new(name, "NODES").node_group = ng
    ob.visible_shadow = False
    return ob


def build_suspended(col, M):
    """Silt in the tailout and at the pool head, and the bubble plume where the riffle pours into the pool."""
    objs = []
    cx_h = river_cx(28.0)
    objs.append(flowing_points("SiltHead", col, M.silt, ((cx_h - 5, cx_h + 5), (16.0, 40.0), (-2.6, -0.1)),
                               6000, (0.0, -1.0, 0.0), (0.0015, 0.004), 71))
    x_a, x_b = rx(-12.0, 0.25), rx(-12.0, 0.95)
    objs.append(flowing_points("SiltTailout", col, M.silt, ((x_a, x_b), (-22.0, -2.0), (-1.5, -0.05)),
                               7000, (0.0, -0.5, 0.0), (0.0012, 0.0035), 72))
    objs.append(flowing_points("WhiteWaterBubbles", col, M.bubbles, ((cx_h - 3.5, cx_h + 3.5), (30.0, 41.0), (-1.9, -0.2)),
                               5200, (0.0, -1.4, 0.0), (0.003, 0.016), 73, life=2.2, rise=0.5))
    return objs


class Particles:
    """Bubbles and spray: every particle is born once with a position, velocity and life;
    a geometry-nodes group moves them with time, so nothing needs per-frame keys."""

    def __init__(self):
        self.items = []     # (birth_frame, pos, vel, life_s, size, gravity)

    def emit(self, f, pos, vel, life, size, gravity=0.0):
        self.items.append((f, Vector(pos), Vector(vel), life, size, gravity))

    def build(self, name, col, mat, above_water, streak=False):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(p) for _f, p, _v, _l, _s, _g in self.items], [], [])
        n = len(self.items)
        for an, kind, vals in (("birth", "FLOAT", [it[0] / FPS for it in self.items]),
                               ("life", "FLOAT", [it[3] for it in self.items]),
                               ("size", "FLOAT", [it[4] for it in self.items]),
                               ("grav", "FLOAT", [it[5] for it in self.items])):
            a = me.attributes.new(an, kind, "POINT")
            a.data.foreach_set("value", vals)
        a = me.attributes.new("vel", "FLOAT_VECTOR", "POINT")
        a.data.foreach_set("vector", [c for it in self.items for c in it[2]])
        ob = bpy.data.objects.new(name, me)
        col.objects.link(ob)
        ng, gi, go = new_group(name + "GN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                               [("Geometry", "NodeSocketGeometry")])
        nodes, links = ng.nodes, ng.links

        def nattr(nm, kind="FLOAT"):
            r = nodes.new("GeometryNodeInputNamedAttribute")
            r.data_type = kind
            r.inputs["Name"].default_value = nm
            return r.outputs[0]
        t = gn_time(ng)
        age = nmath(ng, "SUBTRACT", t, nattr("birth"))
        life = nattr("life")
        alive = nmath(ng, "MULTIPLY", nmath(ng, "GREATER_THAN", age, 0.0), nmath(ng, "LESS_THAN", age, life))
        sp = nodes.new("GeometryNodeSetPosition")
        links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
        disp = vmath(ng, "ADD", vmath(ng, "SCALE", nattr("vel", "FLOAT_VECTOR"), scale=age),
                     combine(ng, 0.0, 0.0, nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", age, age),
                                                nmath(ng, "MULTIPLY", nattr("grav"), -0.5))))
        links.new(disp, sp.inputs["Offset"])
        _x, _y, z = gn_xyz(ng)
        wrong_side = nmath(ng, "GREATER_THAN", z, 0.0) if not above_water else nmath(ng, "LESS_THAN", z, -0.05)
        dele = nodes.new("GeometryNodeDeleteGeometry")
        dele.domain = "POINT"
        links.new(sp.outputs[0], dele.inputs["Geometry"])
        links.new(nmath(ng, "MAXIMUM", nmath(ng, "SUBTRACT", 1.0, alive), wrong_side), dele.inputs["Selection"])
        ico = nodes.new("GeometryNodeMeshIcoSphere")
        ico.inputs["Radius"].default_value = 1.0
        ico.inputs["Subdivisions"].default_value = 1
        inst = nodes.new("GeometryNodeInstanceOnPoints")
        links.new(dele.outputs[0], inst.inputs["Points"])
        links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
        grow = nmath(ng, "MULTIPLY", nattr("size"),
                     nmath(ng, "POWER", nmath(ng, "SUBTRACT", 1.0, nmath(ng, "DIVIDE", age, life), clamp=True), 0.5))
        if streak:
            # Spray streaks along its motion as a shutter would draw it: EEVEE cannot motion blur points
            # that are born and removed from frame to frame. Half length grows by speed x exposure / 2.
            v_now = vmath(ng, "SUBTRACT", nattr("vel", "FLOAT_VECTOR"),
                          combine(ng, 0.0, 0.0, nmath(ng, "MULTIPLY", nattr("grav"), age)))
            align = nodes.new("FunctionNodeAlignRotationToVector")
            align.axis = "Z"
            links.new(v_now, align.inputs["Vector"])
            links.new(align.outputs["Rotation"], inst.inputs["Rotation"])
            half = nmath(ng, "ADD", grow, nmath(ng, "MULTIPLY", vmath(ng, "LENGTH", v_now), SHUTTER / FPS / 2))
            links.new(combine(ng, grow, grow, half), inst.inputs["Scale"])
        else:
            links.new(combine(ng, grow, grow, grow), inst.inputs["Scale"])
        setm = nodes.new("GeometryNodeSetMaterial")
        links.new(inst.outputs[0], setm.inputs["Geometry"])
        setm.inputs["Material"].default_value = mat
        links.new(setm.outputs[0], go.inputs["Geometry"])
        ob.modifiers.new(name, "NODES").node_group = ng
        ob.visible_shadow = False
        print(f"{name}: {n} particles")
        return ob


# --------------------------------------------------------------------------
# Imported models
# --------------------------------------------------------------------------
SOCKEYE_GLB = asset("higgsfield", "sockeye_tripo.glb")
ANGLER_GLB = asset("khronos", "CesiumMan.glb")
FISH_LEN = 0.62          # adult sockeye, metres (USFWS gives 18 to 31 in, about 0.46 to 0.79 m)


def import_glb(path, **kw):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path, **kw)
    return [o for o in bpy.data.objects if o not in before]


def wrap_atmos(mat, atmos):
    """Route an imported material through the scene's air haze / underwater fog group."""
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    link = out.inputs["Surface"].links[0]
    src = link.from_socket
    nt.links.remove(link)
    g = nt.nodes.new("ShaderNodeGroup")
    g.node_tree = atmos
    nt.links.new(src, g.inputs["Shader"])
    nt.links.new(g.outputs["Shader"], out.inputs["Surface"])


def sockeye_mesh(M):
    """The Higgsfield (Tripo) sockeye, turned so the head points along +X, 1 m long, with a rest-position
    attribute for the swimming deformation. Its own baked colour, normal and ORM maps are kept."""
    objs = import_glb(SOCKEYE_GLB)
    src = next(o for o in objs if o.type == "MESH")
    me = src.data
    me.transform(src.matrix_world)
    me.transform(Matrix.Rotation(-math.pi / 2, 4, "Z"))          # head +Y -> +X
    xs = [v.co.x for v in me.vertices]
    zs = [v.co.z for v in me.vertices]
    k = 1.0 / (max(xs) - min(xs))
    off = Vector((-(max(xs) + min(xs)) / 2, 0.0, -(max(zs) + min(zs)) / 2))
    me.transform(Matrix.Scale(k, 4) @ Matrix.Translation(off))
    me.name = "Sockeye"
    for o in objs:
        bpy.data.objects.remove(o)
    mat = me.materials[0]
    mat.name = "SockeyeSkin"
    for n in mat.node_tree.nodes:                      # 4096 px maps -> 2048: a fish never fills more of the frame
        if n.bl_idname == "ShaderNodeTexImage" and n.image and n.image.size[0] > 2048:
            n.image.scale(2048, 2048)
            n.image.pack()                            # keep the scaled pixels in the saved .blend
    b = next(n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    b.inputs["Coat Weight"].default_value = 0.4                 # wet mucus layer
    b.inputs["Coat Roughness"].default_value = 0.08
    b.inputs["Subsurface Weight"].default_value = 0.2           # light through fins and thin flesh
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.3, 0.2)
    b.inputs["Subsurface Scale"].default_value = 0.01
    wrap_atmos(mat, M.atmos)
    for p in me.polygons:
        p.use_smooth = True
    a = me.attributes.new("rest", "FLOAT_VECTOR", "POINT")
    a.data.foreach_set("vector", [c for v in me.vertices for c in v.co])
    return me


def swim_group(name="FishSwim"):
    ng, gi, go = new_group(name, "GeometryNodeTree",
                           [("Geometry", "NodeSocketGeometry"), ("Phase", "NodeSocketFloat"),
                            ("Amp", "NodeSocketFloat"), ("Rate", "NodeSocketFloat")],
                           [("Geometry", "NodeSocketGeometry")])
    pos = separate(ng, ng.nodes.new("GeometryNodeInputPosition").outputs[0])
    t = gn_time(ng)
    w = nmath(ng, "POWER", map_range(ng, pos["X"], 0.2, -0.5, 0.0, 1.0, smooth=False), 1.5)   # the tail moves most
    arg = nmath(ng, "MULTIPLY_ADD", t, gi.outputs["Rate"], nmath(ng, "MULTIPLY_ADD", pos["X"], -6.5, gi.outputs["Phase"]))
    dy = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", nmath(ng, "SINE", arg), w), gi.outputs["Amp"])
    sp = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, 0.0, dy, 0.0), sp.inputs["Offset"])
    ng.links.new(sp.outputs[0], go.inputs["Geometry"])
    return ng


def set_swim(ob, ng, phase, amp, rate):
    mod = ob.modifiers.new("Swim", "NODES")
    mod.node_group = ng
    vals = {"Phase": phase, "Amp": amp, "Rate": rate}
    for item in ng.interface.items_tree:
        if getattr(item, "in_out", None) == "INPUT" and item.name in vals:
            set_mod_input(mod, item.identifier, vals[item.name])


def key_path(ob, path, fwd, frames, roll=None):
    yaws, pitches = [], []
    for f in frames:
        yw, pt = heading_angles(fwd[f])
        yaws.append(yw)
        pitches.append(pt)
    yaws = unwrap(yaws)
    for f, yw, pt in zip(frames, yaws, pitches):
        ob.location = path[f]
        ob.rotation_euler = (roll(f) if roll else 0.0, pt, yw)
        key(ob, "location", f)
        key(ob, "rotation_euler", f)


def forwards(world, k=3):
    keys = sorted(world)
    vel = {}
    for f in keys:
        a, b = world[max(keys[0], f - k)], world[min(keys[-1], f + k)]
        v = b - a
        vel[f] = v.normalized() if v.length > 1e-4 else None
    last = Vector((0, 1, 0))
    for f in keys:                       # holding fish barely move: keep the last good heading
        if vel[f] is None:
            vel[f] = last
        last = vel[f]
    sm = dict(zip(keys, smooth_series([vel[f] for f in keys], 5)))
    return {f: (sm[f].normalized() if sm[f].length > 1e-4 else Vector((0, 1, 0))) for f in keys}


def spline(keys, lo=1, hi=FRAME_END):
    """Catmull-Rom through (frame, (x, y, z)) keys -> {frame: Vector}, held flat beyond the ends."""
    fr = [k[0] for k in keys]
    P = [Vector(k[1]) for k in keys]
    out = {}
    for i in range(len(keys) - 1):
        p0, p1, p2, p3 = P[max(0, i - 1)], P[i], P[i + 1], P[min(len(P) - 1, i + 2)]
        f0, f1 = fr[i], fr[i + 1]
        m1 = (p2 - p0) * ((f1 - f0) / max(1, fr[min(len(P) - 1, i + 1)] - fr[max(0, i - 1)]))
        m2 = (p3 - p1) * ((f1 - f0) / max(1, fr[min(len(P) - 1, i + 2)] - fr[i]))
        for f in range(f0, f1 + 1):
            u = (f - f0) / (f1 - f0)
            h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
            h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
            out[f] = p1 * h00 + m1 * h10 + p2 * h01 + m2 * h11
    for f in range(lo, hi + 1):
        if f < fr[0]:
            out[f] = P[0].copy()
        elif f > fr[-1]:
            out[f] = P[-1].copy()
    return out


def wobble(f, seed, amp=(0.05, 0.08, 0.02), rate=0.05):
    r = random.Random(seed)
    ph = [r.uniform(0, 6.28) for _ in range(6)]
    t = f * rate
    return Vector((amp[0] * (math.sin(t + ph[0]) + 0.5 * math.sin(2.3 * t + ph[1])) / 1.5,
                   amp[1] * (math.sin(0.8 * t + ph[2]) + 0.5 * math.sin(1.9 * t + ph[3])) / 1.5,
                   amp[2] * (math.sin(1.3 * t + ph[4]) + 0.5 * math.sin(2.9 * t + ph[5])) / 1.5))


def on_bed(y, u, h):
    x = rx(y, u)
    return Vector((x, y, bed_z(x, y) + h))


# --------------------------------------------------------------------------
# Choreography: fish, the drift rig, the angler and the rod
# --------------------------------------------------------------------------
HERO_KEYS = [(1, (-0.15, 20.5, -1.15)), (40, (-0.10, 24.6, -1.05)), (80, (-0.18, 28.6, -0.95)), (125, (-0.08, 33.2, -0.85))]
STRIKER_HOLD = (-12.2, 0.605)            # (y, u): the fish that takes the fly
SCHOOL_BOX = ((-16.5, -8.8), (0.62, 0.80))
LEADER = 0.55                            # weight to fly (21.7 in): meets the 18 in minimum (Russian River rule)
RIG_U = 0.60
DRIFTS = [(1, 126, -6.4, -12.0), (128, 328, -6.6, -12.4), (330, 484, -6.9, -12.6),
          (490, 702, -7.6, -12.6), (706, 892, -8.6, -12.9), (896, 1094, -8.8, -14.0), (1098, STRIKE, -9.4, None)]
NO_PICKUP = (896,)                       # this drift is still on the bottom when the shot cuts away
ANGLER_Y, ANGLER_U = -11.0, 0.86
ROD_LEN = 2.75
RIG_SCALE = 1.5                          # weight and fly drawn 1.5x real size so they read on a phone


def hero_path():
    keys = [(f, (rx(y, u), y, z)) for f, (u, y, z) in HERO_KEYS]
    p = spline(keys, 1, CUT1 + 4)
    return p


def mouth_offset():
    return Vector((0.5 * FISH_LEN, 0.0, -0.012 * FISH_LEN / 0.62))


def striker_path():
    """Holds near the bottom until the strike; lunges onto the fly, turns downstream, thrashes at the
    surface, half clears it once, then fights and is led in close to the bank."""
    y0, u0 = STRIKER_HOLD
    base = on_bed(y0, u0, 0.10)
    keys = [(1, tuple(base)), (STRIKE - 8, tuple(base)), (STRIKE, tuple(base + Vector((0.0, 0.16, 0.05)))),
            (STRIKE + 14, tuple(on_bed(y0 - 0.5, u0 - 0.02, 0.2))), (1215, tuple(on_bed(y0 - 1.3, u0 - 0.04, 0.35))),
            (1245, (rx(y0 - 1.6, u0 - 0.03), y0 - 1.6, -0.35)), (1262, (rx(y0 - 1.7, u0 - 0.02), y0 - 1.7, -0.14)),
            (1300, (rx(y0 - 1.8, u0 - 0.03), y0 - 1.8, -0.10)), (LEAP[0], (rx(y0 - 2.0, u0 - 0.04), y0 - 2.0, -0.12)),
            (LEAP[1], (rx(y0 - 2.6, u0 - 0.05), y0 - 2.6, -0.15)), (1400, (rx(y0 - 3.4, u0 - 0.06), y0 - 3.4, -0.35)),
            (1480, (rx(y0 - 4.3, u0 - 0.04), y0 - 4.3, -0.30)), (1560, (rx(y0 - 3.7, u0 + 0.02), y0 - 3.7, -0.22)),
            (1680, (rx(y0 - 3.2, u0 + 0.08), y0 - 3.2, -0.18)), (FRAME_END, (rx(y0 - 2.4, u0 + 0.12), y0 - 2.4, -0.16))]
    p = spline(keys)
    for f in range(1, STRIKE - 8):
        p[f] = base + wobble(f, 9, amp=(0.02, 0.05, 0.01))
    j0, j1 = LEAP
    for f in range(j0, j1 + 1):
        u = (f - j0) / (j1 - j0)
        p[f] = p[f] + Vector((0, 0, 0.62 * 4 * u * (1 - u)))
    for f in range(1262, LEAP[0]):          # thrashing just under and through the surface
        p[f] = p[f] + Vector((0.05 * math.sin(f * 0.9), 0.04 * math.sin(f * 0.7), 0.05 * math.sin(f * 1.3)))
    return p


def striker_forward(path):
    fwd = forwards(path)
    for f in range(1, STRIKE + 2):          # holding: faces upstream (+Y), into the current
        fwd[f] = Vector((0.0, 1.0, 0.0)) if f < STRIKE - 8 else fwd[f].lerp(Vector((0, 1, 0.15)), 0.6).normalized()
    return fwd


def striker_roll(f):
    if f <= HOOKSET:
        return 0.0
    k = 1.0 if f < 1400 else 0.5
    return k * (0.5 * math.sin(f * 0.55) + 0.25 * math.sin(f * 1.3))


def striker_matrix(path, fwd, f):
    yw, pt = heading_angles(fwd[f])
    return Matrix.LocRotScale(path[f], Euler((striker_roll(f), pt, yw)), Vector((FISH_LEN,) * 3))


def angler_root():
    x = rx(ANGLER_Y, ANGLER_U)
    return Vector((x, ANGLER_Y, bed_z(x, ANGLER_Y)))


ANGLER_FACING = Vector((-1.0, -0.18, 0.0)).normalized()     # towards the river, slightly downstream


def hand_point(f):
    """Upper (reel) hand of the angler, world space; sways a little, pulls back during the fight."""
    A = angler_root()
    F = ANGLER_FACING
    R = F.cross(Vector((0, 0, 1))).normalized()
    pull = 0.0
    if f > HOOKSET:
        pull = 0.12 * smoothstep(HOOKSET, HOOKSET + 6, f) + 0.05 * math.sin((f - HOOKSET) * 0.09)
    return A + F * (0.40 - pull) + R * 0.05 + Vector((0, 0, 1.12 + 0.02 * math.sin(f * 0.05)))


def free_rig(d, f):
    """Weight and fly for drift d at frame f, before any strike. Returns (weight, fly)."""
    f0, f1, ye, yx = d
    if yx is None:
        yx = STRIKER_HOLD[0] - 0.9
    L = f1 - f0
    s = f - f0
    pickup = d[3] is not None and f0 not in NO_PICKUP
    p_end = L - 12 if pickup else L + 30
    yy = ye - (ye - yx) * smoothstep(0, 1, (s - 8) / max(1.0, p_end - 8)) ** 0.85 if s > 8 else ye + 0.6 * (1 - s / 8)
    x = rx(yy, RIG_U)
    zb = bed_z(x, yy)
    if s < 8:                                    # in the air: flipped upstream, dropping onto the water
        u = s / 8
        w = Vector((x, yy, 0.9 * (1 - u) + 0.25 * math.sin(math.pi * u)))
        return w, w + Vector((0, 0.32, 0.06))
    sink = smoothstep(8, 30, s)
    bounce = 0.045 * abs(math.sin(math.pi * (s - 8) / 11.0)) ** 0.6 * sink
    zw = (zb + 0.025 + bounce) * sink
    w = Vector((x, yy, zw))
    yf = yy - LEADER * 0.97
    xf = rx(yf, RIG_U)
    zf_drift = bed_z(xf, yf) + 0.12 + 0.015 * math.sin(s * 0.3)
    fly = Vector((xf, yf, zf_drift * sink + (zw + 0.15) * (1 - sink)))
    fly = w + (fly - w) * sink + Vector((0, 0.32, 0.06)) * (1 - sink)
    if pickup and s > L - 12:          # picked up: lifted out and swung back upstream for the next flip
        u = smoothstep(L - 12, L, s)
        lift = Vector((0.0, 0.6 * u, 1.1 * u))
        w, fly = w + lift, fly + lift * 1.15
    return w, fly


def drift_at(f):
    for d in DRIFTS:
        if d[0] <= f <= d[1]:
            return d
    return None


def rod_pose(f, target):
    """Butt, direction, bend for the rod at frame f, pointing at `target` (where the line goes)."""
    H = hand_point(f)
    flat = Vector((target.x - H.x, target.y - H.y, 0.0))
    flat = flat.normalized() if flat.length > 1e-3 else ANGLER_FACING
    el = math.radians(32.0)
    bend = 0.06
    if f > HOOKSET - 1:
        u = smoothstep(HOOKSET - 1, HOOKSET + 5, f)
        el = math.radians(32.0 + 36.0 * u - 8.0 * smoothstep(HOOKSET + 30, HOOKSET + 120, f))
        swing = Matrix.Rotation(math.radians(-28.0) * u, 3, "Z")     # sweep downstream
        flat = (swing @ flat).normalized()
        bend = 0.08 + 0.78 * u * (1 - 0.15 * smoothstep(1500, 1700, f)) + 0.06 * math.sin(f * 0.37) + 0.04 * math.sin(f * 0.91)
    d = (flat * math.cos(el) + Vector((0, 0, math.sin(el)))).normalized()
    return H - d * 0.30, d, bend


def place_rod(r, f, butt, direction, load_dir=None, bend=0.0):
    d = direction.normalized()
    tip_dir = d
    if load_dir is not None and bend > 0:
        tip_dir = d.slerp(load_dir.normalized(), min(0.95, bend * 0.8))
    butt = Vector(butt)
    mid = butt + d * (ROD_LEN * 0.5)
    tip = mid + (d * 0.45 + tip_dir * 0.55).normalized() * (ROD_LEN * 0.5) * (1.0 - 0.08 * bend)
    for k, p in (("Butt", butt), ("Mid", mid), ("Tip", tip)):
        r["ctrl"][k].location = p
        key(r["ctrl"][k], "location", f)
    side = d.cross(Vector((0, 0, 1)))
    side = side.normalized() if side.length > 1e-3 else Vector((1, 0, 0))
    up = side.cross(d).normalized()
    r["mount"].location = butt + d * 0.32 - up * 0.04            # spinning reel hangs under the blank
    r["mount"].rotation_quaternion = Matrix((side, d, up)).transposed().to_quaternion()
    key(r["mount"], "location", f)
    key(r["mount"], "rotation_quaternion", f)
    return tip


def build_rod(col, M):
    ctrl = {k: empty_object(f"Rod{k}", col, size=0.05) for k in ("Butt", "Mid", "Tip")}
    ng, gi, go = new_group("RodGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links

    def loc(ob):
        n = nodes.new("GeometryNodeObjectInfo")
        n.transform_space = "RELATIVE"
        n.inputs["Object"].default_value = ob
        return n.outputs["Location"]

    def blank(radius, mat, flip=False):
        c = nodes.new("GeometryNodeCurveQuadraticBezier")
        c.inputs["Resolution"].default_value = 32
        links.new(loc(ctrl["Butt"]), c.inputs["Start"])
        links.new(loc(ctrl["Mid"]), c.inputs["Middle"])
        links.new(loc(ctrl["Tip"]), c.inputs["End"])
        rad = nodes.new("GeometryNodeSetCurveRadius")
        links.new(c.outputs["Curve"], rad.inputs["Curve"])
        par = nodes.new("GeometryNodeSplineParameter")
        links.new(map_range(ng, par.outputs["Factor"], 0.0, 1.0, 1.0, 0.18, smooth=False), rad.inputs["Radius"])
        prof = nodes.new("GeometryNodeCurvePrimitiveCircle")
        prof.inputs["Resolution"].default_value = 8
        prof.inputs["Radius"].default_value = radius
        c2m = nodes.new("GeometryNodeCurveToMesh")
        links.new(rad.outputs[0], c2m.inputs["Curve"])
        links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
        g = c2m.outputs[0]
        if flip:
            ff = nodes.new("GeometryNodeFlipFaces")
            links.new(g, ff.inputs["Mesh"])
            g = ff.outputs[0]
        setm = nodes.new("GeometryNodeSetMaterial")
        setm.inputs["Material"].default_value = mat
        links.new(g, setm.inputs["Geometry"])
        return setm.outputs[0]
    links.new(blank(0.0085, M.rod), go.inputs["Geometry"])
    ob = bpy.data.objects.new("Rod", bpy.data.meshes.new("Rod"))
    col.objects.link(ob)
    ob.modifiers.new("Rod", "NODES").node_group = ng
    # tension glow: an inverted hull around the same curve, pulsed at the hook-set
    ng2, gi2, go2 = new_group("RodGlowGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                              [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng2.nodes, ng2.links
    ng = ng2
    links.new(blank(0.02, M.highlight, flip=True), go2.inputs["Geometry"])
    glow = bpy.data.objects.new("HiRodGlow", bpy.data.meshes.new("HiRodGlow"))
    col.objects.link(glow)
    glow.modifiers.new("Glow", "NODES").node_group = ng2
    glow.visible_shadow = False
    glow.color = (*GLOW, 0.0)
    # Spinning reel, cork grip
    mount = empty_object("ReelMount", col, size=0.08)
    mount.rotation_mode = "QUATERNION"
    bm = bmesh.new()
    bm_frustum(bm, (0, 0.0, -0.05), (0, 0.0, -0.13), 0.04, 0.045, segs=20)       # spool and body
    bm_frustum(bm, (0, 0.0, -0.13), (0, 0.0, -0.17), 0.03, 0.02, segs=16)
    bm_frustum(bm, (0, 0.0, 0.0), (0, 0.0, -0.06), 0.008, 0.008, segs=6)          # stem
    reel = mesh_object("Reel", bm, col, [M.reel], parent=mount, recalc=False)
    for p in reel.data.polygons:
        p.use_smooth = True
    bm = bmesh.new()
    bm_frustum(bm, (0.05, 0, -0.1), (0.1, 0, -0.1), 0.006, 0.006, segs=6)
    bm_frustum(bm, (0.1, 0, -0.1), (0.1, 0, -0.16), 0.006, 0.006, segs=6)
    bm_frustum(bm, (0.1, 0, -0.16), (0.13, 0, -0.16), 0.012, 0.012, segs=8)
    mesh_object("ReelHandle", bm, col, [M.reel_dark], parent=mount)
    bm = bmesh.new()
    bm_frustum(bm, (0, -0.30, 0.04), (0, 0.10, 0.04), 0.016, 0.014, segs=10)
    grip = mesh_object("CorkGrip", bm, col, [M.cork], parent=mount)
    for p in grip.data.polygons:
        p.use_smooth = True
    return dict(ctrl=ctrl, rod=ob, glow=glow, mount=mount, reel=reel)


def weight_mesh(M):
    bm = bmesh.new()
    bm_frustum(bm, (0.03, 0, 0), (-0.03, 0, 0), 0.0065, 0.0065, segs=10)
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.0068, matrix=Matrix.Translation((0.03, 0, 0)))
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.0068, matrix=Matrix.Translation((-0.03, 0, 0)))
    me = bpy.data.meshes.new("Weight")
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(M.weight)
    return me


def fly_mesh(M):
    """Yarn fly on a single hook (gap 9 mm, inside the 3/8 in limit): pink and orange yarn tied on the shank."""
    bm = bmesh.new()
    ly = bm.verts.layers.float.new("yarn")
    hook = [Vector((0.012 - 0.014 * (1 - math.cos(a)) * 0, 0, 0)) for a in (0,)]
    pts = [Vector((0.016, 0, 0.0))] + [Vector((-0.004 + 0.0055 * math.cos(a), 0, -0.0055 + 0.0055 * math.sin(a)))
                                        for a in np.linspace(math.pi * 0.5, math.pi * 2.0, 9)]
    pts[1] = Vector((-0.004, 0, 0.0))
    for p0, p1 in zip(pts, pts[1:]):
        bm_frustum(bm, p0, p1, 0.0008, 0.0008, segs=5)
    for f in bm.faces:
        f.material_index = 1
    n0 = len(bm.faces)
    rng = random.Random(81)
    for i in range(46):
        c = Vector((rng.uniform(-0.002, 0.016), rng.uniform(-0.007, 0.007), rng.uniform(-0.002, 0.009)))
        res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=rng.uniform(0.0035, 0.0062), matrix=Matrix.Translation(c))
        val = 0.0 if c.x > 0.007 else 1.0                     # pink head, orange tail
        for v in res["verts"]:
            v[ly] = val
            v.co += Vector((rng.uniform(-0.001, 0.001),) * 3)
    for f in list(bm.faces)[n0:]:
        f.material_index = 0
    me = bpy.data.meshes.new("YarnFly")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M.yarn)
    me.materials.append(M.hook)
    for p in me.polygons:
        p.use_smooth = True
    del hook
    return me


def build_angler(col, M):
    """CesiumMan (Khronos sample, CC BY 4.0) re-dressed as a wading angler: the mesh is recoloured into
    chest waders and a jacket by its rest height, the logo head is replaced by a modelled head, cap and
    sunglasses, and IK puts both hands on the rod."""
    objs = import_glb(ANGLER_GLB, bone_heuristic="TEMPERANCE")
    arm = next(o for o in objs if o.type == "ARMATURE")
    body = next(o for o in objs if o.type == "MESH" and o.name.startswith("Cesium"))
    for o in objs:
        if o not in (arm, body) and o.type != "EMPTY":
            bpy.data.objects.remove(o)
    for o in (arm, body):
        for c in o.users_collection:
            c.objects.unlink(o)
        col.objects.link(o)
    for o in objs:
        if o.type == "EMPTY" and o.name in bpy.data.objects:
            if o is not arm.parent:
                pass
    if arm.parent is not None:
        mw = arm.matrix_world.copy()
        arm.parent = None
        arm.matrix_world = mw
    arm.animation_data_clear()
    for b in arm.pose.bones:
        b.rotation_mode = "QUATERNION"
        b.rotation_quaternion = (1, 0, 0, 0)
    # drop the logo head (everything above the neck in the rest pose)
    me = body.data
    bm = bmesh.new()
    bm.from_mesh(me)
    mw = body.matrix_world
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if (mw @ v.co).z > 1.165], context="VERTS")
    bm.to_mesh(me)
    bm.free()
    rest = [mw @ v.co for v in me.vertices]
    a = me.attributes.new("rest", "FLOAT_VECTOR", "POINT")
    a.data.foreach_set("vector", [c for v in rest for c in v])
    for m in me.materials:                         # the original texture (with the logo) is not used
        for n in (m.node_tree.nodes if m and m.node_tree else []):
            if n.bl_idname == "ShaderNodeTexImage" and n.image:
                bpy.data.images.remove(n.image)
    me.materials.clear()

    def clothes(nt):
        r = separate(nt, attr_vec(nt, "rest"))
        ax = nmath(nt, "ABSOLUTE", r["X"])
        arm_m = nmath(nt, "MULTIPLY", map_range(nt, ax, 0.17, 0.2), map_range(nt, r["Z"], 0.80, 0.86))
        jacket_m = nmath(nt, "MAXIMUM", map_range(nt, r["Z"], 0.93, 0.96), arm_m)
        hand_m = map_range(nt, ax, 0.40, 0.43)
        boot_m = map_range(nt, r["Z"], 0.11, 0.08)
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 60.0
        nt.links.new(attr_vec(nt, "rest"), n.inputs["Vector"])
        col_ = mix_rgb(nt, jacket_m, srgb(70, 76, 62), srgb(178, 74, 36))
        col_ = mix_rgb(nt, boot_m, col_, srgb(42, 38, 34))
        col_ = mix_rgb(nt, hand_m, col_, srgb(196, 140, 106))
        col_ = vmath(nt, "SCALE", col_, scale=nmath(nt, "MULTIPLY_ADD", n.outputs["Fac"], 0.25, 0.88))
        wet = map_range(nt, separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"])["Z"], 0.15, -0.05)
        col_ = vmath(nt, "SCALE", col_, scale=nmath(nt, "MULTIPLY_ADD", wet, -0.35, 1.0))
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.25
        nt.links.new(n.outputs["Fac"], bump.inputs["Height"])
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col_, b.inputs["Base Color"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
        nt.links.new(map_range(nt, wet, 0.0, 1.0, 0.72, 0.35, smooth=False), b.inputs["Roughness"])
        b.inputs["Coat Weight"].default_value = 0.15
        return b.outputs[0]
    me.materials.append(new_material("AnglerClothes", clothes, M.atmos))
    for p in me.polygons:
        p.use_smooth = True
    # head, beard, cap and sunglasses on the neck bone
    neck = arm.pose.bones["Skeleton_neck_joint_2"]
    hroot = empty_object("AnglerHead", col, size=0.05)
    hroot.parent = arm
    hroot.parent_type = "BONE"
    hroot.parent_bone = neck.name
    hroot.matrix_world = Matrix.Translation((0.005, -0.01, 1.19))
    rng = random.Random(3)
    parts = [("Head", lambda bm: bm_blob(bm, (0, 0, 0.09), (0.085, 0.095, 0.11), rng, subdiv=2, jitter=0.02), M.skin),
             ("Beard", lambda bm: bm_blob(bm, (0, -0.045, 0.035), (0.075, 0.06, 0.06), rng, subdiv=2, jitter=0.05), M.hair),
             ("Cap", lambda bm: (bm_frustum(bm, (0, 0, 0.13), (0, 0, 0.215), 0.096, 0.08, segs=14),
                                 bm_box(bm, (0, -0.105, 0.14), (0.16, 0.11, 0.014))), M.cap),
             ("Shades", lambda bm: bm_box(bm, (0, -0.088, 0.105), (0.15, 0.02, 0.035)), M.shades)]
    for nm, fn, mat in parts:
        bm = bmesh.new()
        fn(bm)
        ob = mesh_object(f"Angler{nm}", bm, col, [mat], parent=hroot)
        for p in ob.data.polygons:
            p.use_smooth = True
    # IK targets (keyed per frame from the rod) and elbow poles
    tgt = {s: empty_object(f"AnglerHand{s}", col, size=0.03) for s in ("L", "R")}
    pole = {s: empty_object(f"AnglerElbowPole{s}", col, size=0.03) for s in ("L", "R")}
    for s, bn in (("L", "Skeleton_arm_joint_L__2_"), ("R", "Skeleton_arm_joint_R__3_")):
        c = arm.pose.bones[bn].constraints.new("IK")
        c.target = tgt[s]
        c.chain_count = 2
    # stand on the gravel, facing the river: CesiumMan faces -Y once the importer's own Y-up -> Z-up
    # rotation (kept in arm["base_q"]) is applied; facing and lean are composed on top of it
    A = angler_root()
    arm["base_q"] = list(arm.matrix_world.to_quaternion())
    arm.location = A
    key_lean(arm, 1, 0.0, keyframe=False)
    arm.scale = (1.24, 1.24, 1.24)                     # 1.45 m model -> 1.8 m angler
    return dict(arm=arm, body=body, hands=tgt, head=hroot)


def key_lean(arm, f, lean, keyframe=True):
    yaw = math.atan2(ANGLER_FACING.y, ANGLER_FACING.x) - math.atan2(-1.0, 0.0)
    R = ANGLER_FACING.cross(Vector((0, 0, 1))).normalized()
    base = Quaternion(arm["base_q"])
    q = Quaternion(R, lean) @ Euler((0, 0, yaw)).to_quaternion() @ base
    arm.rotation_mode = "QUATERNION"
    arm.rotation_quaternion = q
    if keyframe:
        key(arm, "rotation_quaternion", f)


def build_school(col, M, fish_me, cam_frames):
    """The school holding in the tailout, facing upstream: one point per fish, instanced and realised in
    geometry nodes so each body gets its own tail beat."""
    rng = random.Random(91)
    (y0, y1), (u0, u1) = SCHOOL_BOX
    pts, yaw0, ph, scl, rate = [], [], [], [], []
    rig_cams = [cam_frames[f][0] for f in range(CUT2, CUT4)]
    kd = kdtree.KDTree(len(rig_cams))
    for i, p in enumerate(rig_cams):
        kd.insert(p, i)
    kd.balance()
    sy, su = STRIKER_HOLD
    tries = 0
    while len(pts) < 46 and tries < 4000:
        tries += 1
        y = rng.uniform(y0, y1)
        u = rng.uniform(u0, u1)
        s = rng.uniform(0.9, 1.1) * FISH_LEN
        p = on_bed(y, u, 0.10 * s / FISH_LEN)
        if kd.find(p)[2] < 1.0:
            continue
        if (p - on_bed(sy, su, 0.1)).length < 0.75:
            continue
        if any((p - q).length < 0.42 for q in pts):
            continue
        pts.append(p)
        yaw0.append(math.pi / 2 + rng.uniform(-0.12, 0.12))
        ph.append(rng.random())
        scl.append(s)
        rate.append(rng.uniform(4.5, 6.5))
    me = bpy.data.meshes.new("SchoolPts")
    me.from_pydata([tuple(p) for p in pts], [], [])
    for an, vals in (("yaw0", yaw0), ("ph", ph), ("scl", scl), ("rate", rate)):
        a = me.attributes.new(an, "FLOAT", "POINT")
        a.data.foreach_set("value", vals)
    unit = bpy.data.objects.new("SockeyeUnit", fish_me)
    ng, gi, go = new_group("SchoolGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links

    def nattr(nm, kind="FLOAT"):
        r = nodes.new("GeometryNodeInputNamedAttribute")
        r.data_type = kind
        r.inputs["Name"].default_value = nm
        return r.outputs[0]
    t = gn_time(ng)
    phr = nmath(ng, "MULTIPLY", nattr("ph"), 6.283)
    off = combine(ng, nmath(ng, "MULTIPLY", nmath(ng, "SINE", nmath(ng, "MULTIPLY_ADD", t, 0.37, phr)), 0.05),
                  nmath(ng, "MULTIPLY", nmath(ng, "SINE", nmath(ng, "MULTIPLY_ADD", t, 0.29, nmath(ng, "MULTIPLY", phr, 1.7))), 0.10),
                  nmath(ng, "MULTIPLY", nmath(ng, "SINE", nmath(ng, "MULTIPLY_ADD", t, 0.5, nmath(ng, "MULTIPLY", phr, 1.3))), 0.015))
    sp = nodes.new("GeometryNodeSetPosition")
    links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    links.new(off, sp.inputs["Offset"])
    yaw = nmath(ng, "ADD", nattr("yaw0"), nmath(ng, "MULTIPLY", nmath(ng, "SINE", nmath(ng, "MULTIPLY_ADD", t, 0.33,
                                                                                        nmath(ng, "MULTIPLY", phr, 2.1))), 0.08))
    inst = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(sp.outputs[0], inst.inputs["Points"])
    oi = nodes.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = unit
    oi.inputs["As Instance"].default_value = True
    links.new(oi.outputs["Geometry"], inst.inputs["Instance"])
    links.new(combine(ng, 0.0, 0.0, yaw), inst.inputs["Rotation"])
    s = nattr("scl")
    links.new(combine(ng, s, s, s), inst.inputs["Scale"])
    geo = inst.outputs[0]
    for nm, val in (("i_yaw", yaw), ("i_ph", phr), ("i_scl", s), ("i_rate", nattr("rate"))):
        st = nodes.new("GeometryNodeStoreNamedAttribute")
        st.data_type = "FLOAT"
        st.domain = "INSTANCE"
        st.inputs["Name"].default_value = nm
        links.new(geo, st.inputs["Geometry"])
        links.new(val, st.inputs["Value"])
        geo = st.outputs[0]
    real = nodes.new("GeometryNodeRealizeInstances")
    links.new(geo, real.inputs["Geometry"])
    rest_x = separate(ng, nattr("rest", "FLOAT_VECTOR"))["X"]
    w = nmath(ng, "POWER", map_range(ng, rest_x, 0.2, -0.5, 0.0, 1.0, smooth=False), 1.5)
    arg = nmath(ng, "MULTIPLY_ADD", t, nattr("i_rate"), nmath(ng, "MULTIPLY_ADD", rest_x, -6.5, nattr("i_ph")))
    sway = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", nmath(ng, "SINE", arg), w), nmath(ng, "MULTIPLY", nattr("i_scl"), 0.05))
    iy = nattr("i_yaw")
    lat = combine(ng, nmath(ng, "MULTIPLY", nmath(ng, "SINE", iy), -1.0), nmath(ng, "COSINE", iy), 0.0)
    sp2 = nodes.new("GeometryNodeSetPosition")
    links.new(real.outputs[0], sp2.inputs["Geometry"])
    links.new(vmath(ng, "SCALE", lat, scale=sway), sp2.inputs["Offset"])
    links.new(sp2.outputs[0], go.inputs["Geometry"])
    ob = bpy.data.objects.new("SockeyeSchool", me)
    col.objects.link(ob)
    ob.modifiers.new("School", "NODES").node_group = ng
    print(f"school: {len(pts)} sockeye")
    return ob, pts


def build_action(cols, M, cam_frames):
    c_gear, c_life, c_people = cols
    fish_me = sockeye_mesh(M)
    swim = swim_group()
    # Hook: the hero surging up into the white water, with two more fish behind it
    hp = hero_path()
    fr = list(range(1, CUT1 + 5))
    hero = bpy.data.objects.new("HeroSockeye", fish_me)
    c_life.objects.link(hero)
    hero.scale = (FISH_LEN * 1.2,) * 3
    hero.rotation_mode = "XYZ"
    set_swim(hero, swim, 0.0, 0.085, 15.0)
    key_path(hero, hp, forwards(hp), fr, roll=lambda f: 0.12 * math.sin(f * 0.5))
    extras = []
    for i, (dx, dy, dz, lag) in enumerate(((-0.9, -2.4, -0.3, 6), (0.7, -3.6, 0.15, 12))):
        p = {f: hp[max(1, f - lag)] + Vector((dx, dy, dz)) + wobble(f, 30 + i, amp=(0.1, 0.1, 0.05)) for f in fr}
        ob = bpy.data.objects.new(f"RunSockeye{i}", fish_me)
        c_life.objects.link(ob)
        ob.scale = (FISH_LEN * (0.95 - 0.05 * i),) * 3
        ob.rotation_mode = "XYZ"
        set_swim(ob, swim, 1.7 * (i + 1), 0.08, 14.0)
        key_path(ob, p, forwards(p), fr)
        extras.append(ob)
    for ob in [hero] + extras:
        for f, h in ((1, False), (CUT1 + 4, False), (CUT1 + 5, True)):
            ob.hide_render = h
            key(ob, "hide_render", f)
    # The school and the striker
    school, school_pts = build_school(c_life, M, fish_me, cam_frames)
    sp = striker_path()
    sf = striker_forward(sp)
    striker = bpy.data.objects.new("StrikerSockeye", fish_me)
    c_life.objects.link(striker)
    striker.scale = (FISH_LEN,) * 3
    striker.rotation_mode = "XYZ"
    set_swim(striker, swim, 0.6, 0.06, 6.0)
    key_path(striker, sp, sf, list(range(1, FRAME_END + 1)), roll=striker_roll)
    # Tail beat speeds up once hooked
    for item in swim.interface.items_tree:
        if getattr(item, "in_out", None) == "INPUT" and item.name == "Rate":
            ident = item.identifier
    mod = striker.modifiers["Swim"]
    # Rig: weight, fly, leader and main line
    weight = bpy.data.objects.new("Weight", weight_mesh(M))
    fly = bpy.data.objects.new("YarnFly", fly_mesh(M))
    for o in (weight, fly):
        c_gear.objects.link(o)
        o.rotation_mode = "XYZ"
        o.scale = (RIG_SCALE,) * 3
    rod = build_rod(c_gear, M)
    ang = build_angler(c_people, M)
    mouth = {}
    for f in range(1, FRAME_END + 1):
        mouth[f] = striker_matrix(sp, sf, f) @ (mouth_offset() / FISH_LEN * 1.0)
    rig = {}
    for f in range(1, FRAME_END + 1):
        if f <= STRIKE:
            d = drift_at(f)
            if d is None:                       # between drifts (re-cast in the air)
                d = max((dd for dd in DRIFTS if dd[1] < f), key=lambda dd: dd[1])
                w, fl = free_rig(d, d[1])
            else:
                w, fl = free_rig(d, f)
            if d is DRIFTS[-1] and f > STRIKE - 10:
                u = smoothstep(STRIKE - 10, STRIKE, f)
                fl = fl.lerp(mouth[f], u)
                w = w.lerp(fl + (w - fl).normalized() * LEADER, u)
            rig[f] = (w, fl)
        else:
            rig[f] = (None, mouth[f])
    # Rod and hands, then the weight hangs on the line between the fly and the rod tip
    tips, mids = {}, {}
    for f in range(1, FRAME_END + 1):
        w, fl = rig[f]
        target = (w if w is not None else fl).copy()
        target.z = 0.0
        butt, dvec, bend = rod_pose(f, target)
        load = (fl - butt) if f > STRIKE else (target - butt)
        tips[f] = place_rod(rod, f, butt, dvec, load, bend)
        mids[f] = rod["ctrl"]["Mid"].location.copy()
        for s, off in (("R", 0.30), ("L", 0.04)):
            ang["hands"][s].location = butt + dvec * off
            key(ang["hands"][s], "location", f)
        if w is None:
            w = fl + (tips[f] - fl).normalized() * LEADER
            rig[f] = (w, fl)
        lean = 0.0
        if f > HOOKSET:
            lean = math.radians(-7 - 4 * math.sin((f - HOOKSET) * 0.06)) * smoothstep(HOOKSET, HOOKSET + 8, f)
        if f % 2 == 1 or f == FRAME_END or HOOKSET - 2 <= f <= HOOKSET + 10:
            key_lean(ang["arm"], f, lean)
    for f in range(1, FRAME_END + 1):
        w, fl = rig[f]
        a = (fl - w)
        weight.location = w
        yw, pt = heading_angles(a if a.length > 1e-4 else Vector((0, -1, 0)))
        weight.rotation_euler = (0.0, pt, yw)
        key(weight, "location", f)
        key(weight, "rotation_euler", f)
        fly.location = fl
        dirv = (fl - w) if f <= STRIKE else (fl - w)
        yw, pt = heading_angles(dirv if dirv.length > 1e-4 else Vector((0, -1, 0)))
        fly.rotation_euler = (0.0, pt, yw)
        key(fly, "location", f)
        key(fly, "rotation_euler", f)
    leader = line_object("Leader", c_gear, M, weight, fly, radius=0.0012)
    main = line_object("MainLine", c_gear, M, rod["ctrl"]["Tip"], weight, radius=0.0015)
    for o in (weight, fly, leader, main):
        o.visible_shadow = False
    # Particles: flip splashes, the strike, the thrash and the leap
    bub, spray = Particles(), Particles()
    prng = random.Random(4)
    for d in DRIFTS:
        fe = d[0] + 8
        if fe > FRAME_END:
            continue
        p = rig[fe][0]
        for _ in range(22):
            spray.emit(fe, Vector((p.x, p.y, 0.02)), (prng.uniform(-0.5, 0.5), prng.uniform(-0.5, 0.5), prng.uniform(0.5, 1.4)),
                       prng.uniform(0.25, 0.5), prng.uniform(0.004, 0.01), 9.81)
        for _ in range(40):
            bub.emit(fe + prng.randint(0, 6), Vector((p.x, p.y, -0.05)) + Vector((prng.uniform(-0.03, 0.03), 0, prng.uniform(-0.25, 0))),
                     (prng.uniform(-0.1, 0.1), prng.uniform(-0.4, -0.1), prng.uniform(0.1, 0.4)), prng.uniform(0.5, 1.2),
                     prng.uniform(0.002, 0.007), -0.3)
    p = mouth[STRIKE]
    for _ in range(60):
        bub.emit(STRIKE + prng.randint(0, 4), p + Vector((prng.uniform(-0.1, 0.1), prng.uniform(-0.1, 0.1), prng.uniform(-0.05, 0.05))),
                 (prng.uniform(-0.3, 0.3), prng.uniform(-0.3, 0.3), prng.uniform(0.1, 0.6)), prng.uniform(0.8, 1.6),
                 prng.uniform(0.002, 0.008), -0.5)
    for f in range(1240, 1560, 3):
        p = sp[f]
        if p.z > -0.25:
            n = 26 if f < LEAP[1] + 10 else 8
            for _ in range(n):
                spray.emit(f, Vector((p.x, p.y, 0.03)) + Vector((prng.uniform(-0.2, 0.2), prng.uniform(-0.2, 0.2), 0)),
                           (prng.uniform(-1.0, 1.0), prng.uniform(-1.0, 1.0), prng.uniform(0.6, 2.2)),
                           prng.uniform(0.3, 0.7), prng.uniform(0.006, 0.02), 9.81)
            for _ in range(n // 2):
                bub.emit(f, Vector((p.x, p.y, -0.08)) + Vector((prng.uniform(-0.2, 0.2), prng.uniform(-0.2, 0.2), prng.uniform(-0.2, 0))),
                         (prng.uniform(-0.2, 0.2), prng.uniform(-0.5, 0.1), prng.uniform(0.1, 0.5)), prng.uniform(0.6, 1.4),
                         prng.uniform(0.003, 0.012), -0.4)
    for fj in (LEAP[0] + 2, LEAP[1] - 1):
        p = sp[fj]
        for _ in range(220):
            spray.emit(fj, Vector((p.x, p.y, 0.05)), (prng.uniform(-1.6, 1.6), prng.uniform(-1.6, 1.6), prng.uniform(1.0, 3.4)),
                       prng.uniform(0.5, 1.0), prng.uniform(0.006, 0.022), 9.81)
    bub_ob = bub.build("Bubbles", c_life, M.bubbles, above_water=False)
    spray_ob = spray.build("Spray", c_life, M.spray, above_water=True, streak=True)
    return dict(hero=hero, extras=extras, school=school, school_pts=school_pts, striker=striker, striker_path=sp,
                weight=weight, fly=fly, rig=rig, rod=rod, tips=tips, mids=mids, angler=ang, lines=(leader, main), mouth=mouth,
                bubbles=bub_ob, spray=spray_ob, swim_rate_id=ident, striker_mod=mod)


# --------------------------------------------------------------------------
# Camera: five shots, every cut hidden inside a wipe
# --------------------------------------------------------------------------
def smoothed_rig_target(frames_lo, frames_hi, sigma=11.0):
    """Midpoint of weight and fly for the drift beat, Gaussian-smoothed so the camera glides."""
    raw = {}
    for f in range(frames_lo - 60, frames_hi + 61):
        ff = max(frames_lo, min(frames_hi, f + 6))
        d = drift_at(ff) or max((dd for dd in DRIFTS if dd[1] < ff), key=lambda dd: dd[1])
        w, fl = free_rig(d, min(ff, d[1]))
        m = (w + fl) / 2
        m.z = max(min(m.z, -0.62), bed_z(m.x, m.y) + 0.10)
        raw[f] = m
    rad = int(3 * sigma)
    out = {}
    for f in range(frames_lo, frames_hi + 1):
        acc, ws = Vector(), 0.0
        for k in range(-rad, rad + 1):
            wgt = math.exp(-0.5 * (k / sigma) ** 2)
            acc += raw[f + k] * wgt
            ws += wgt
        out[f] = acc / ws
    return out


def camera_keys():
    """Per segment: (frame, position, target) keys in world space."""
    hp = hero_path()
    segs = []
    # 1 hook: close on the surging hero, ending as it drives into the white water
    def rel(f, d, t=(0, 0, 0)):
        return (f, tuple(hp[f] + Vector(d)), tuple(hp[f] + Vector(t)))
    segs.append([rel(1, (0.56, 0.92, 0.04), (0.12, 0, 0.02)), rel(34, (0.82, 0.40, 0.08), (0.05, 0.1, 0)),
                 rel(78, (0.92, -0.25, 0.14), (0, 0.45, 0)), rel(CUT1 - 1, (0.75, -1.05, 0.22), (0, 1.3, 0.05))])
    # 2 drone: high over the pool, sweeping down the river into the tailout and the riffle below
    S = on_bed(-12.8, 0.70, 0.0)
    S.z = -0.8
    segs.append([(CUT1, (rx(52, -0.55), 52.0, 34.0), (rx(6, 0.15), 6.0, 0.0)),
                 (230, (rx(30, 0.30), 30.0, 25.0), (rx(-9, 0.40), -9.0, 0.0)),
                 (350, (rx(8, 0.95), 8.0, 15.5), (rx(-17, 0.55), -17.0, -0.4)),
                 (CUT2 - 1, (rx(-3.0, 0.52), -3.0, 6.6), tuple(S))])
    # 3 rig: underwater side view tracking the drifts (built from the smoothed rig below)
    segs.append(None)
    # 4 over-under: the lens exactly on the waterline, level, looking across at the angler
    yc0, yc1 = -11.3, -12.9
    dz = -6.0 * math.tan(math.radians(SPLIT_PITCH))
    segs.append([(CUT3, (rx(yc0, 0.455), yc0, 0.0), (rx(yc0, 0.455) + 6.0, yc0 - 0.1, dz)),
                 (CUT4 - 1, (rx(yc1, 0.47), yc1, 0.0), (rx(yc1, 0.47) + 6.0, yc1 + 0.2, dz))])
    # 5 hook-set and fight, then the pull-back over the valley
    A = angler_root()
    A.z = 0.0
    spf = striker_path()

    def fish_pt(f):
        q = spf[f].copy()
        q.z = max(q.z, 0.0)
        return q
    def behind(f, back, side, h):
        fp = fish_pt(f)
        d = Vector((A.x - fp.x, A.y - fp.y, 0.0)).normalized()
        sd = Vector((d.y, -d.x, 0.0))
        return A + d * back + sd * side + Vector((0, 0, h))
    segs.append([(CUT4, tuple(behind(CUT4 + 20, 2.2, -1.1, 1.75)), tuple(fish_pt(CUT4 + 20) * 0.8 + A * 0.2 + Vector((0, 0, 0.3)))),
                 (1340, tuple(behind(1340, 2.1, -0.3, 1.5)), tuple(fish_pt(1340) * 0.75 + A * 0.25 + Vector((0, 0, 0.35)))),
                 (1450, tuple(behind(1450, 3.0, 1.0, 1.55)), tuple(fish_pt(1450) * 0.55 + A * 0.45 + Vector((0, 0, 0.5)))),
                 (1560, tuple(behind(1560, 4.2, -1.8, 2.6)), tuple(fish_pt(1560) * 0.4 + A * 0.6 + Vector((0, 0, 0.7)))),
                 (1660, tuple(A + Vector((3.0, -14.0, 9.0))), tuple(A + Vector((-2.0, 1.5, 0.0)))),
                 (FRAME_END, (rx(-62, 0.35), -62.0, 36.0), (rx(40, -0.15), 40.0, 6.0))])
    return segs


def camera_frames():
    segs = camera_keys()
    local = {}
    for i, (lo, hi) in enumerate(SEGMENTS):
        if segs[i] is None:
            T = smoothed_rig_target(lo, hi)
            for f in range(lo, hi + 1):
                # distance per drift: wide for the first, close on the weight, then the fly at the school
                dist = 2.25 - 0.95 * smoothstep(690, 760, f) + 0.6 * smoothstep(880, 950, f)
                lift = 0.12 + 0.02 * smoothstep(880, 950, f)
                t = T[f]
                pos = Vector((t.x - dist, t.y + 0.12 * dist, t.z + lift))
                local[f] = (pos, t + Vector((0.0, -0.05, -0.06)))
            continue
        keys = segs[i]
        path = hermite_path([(f, p, t, False) for f, p, t in keys])
        for f in range(lo, hi + 1):
            pp, tt = path[max(keys[0][0], min(keys[-1][0], f))]
            local[f] = (Vector(pp), Vector(tt))
    out, q_prev = {}, None
    for i, (lo, hi) in enumerate(SEGMENTS):
        sigma = (CAM_SMOOTH if i != 4 else 10.0) if i != 3 else 0.0
        rad = int(3 * sigma)
        for f in range(lo, hi + 1):
            if sigma:
                p, t, ws = Vector(), Vector(), 0.0
                for k in range(-rad, rad + 1):
                    wgt = math.exp(-0.5 * (k / sigma) ** 2)
                    pp, tt = local[min(hi, max(lo, f + k))]
                    p += pp * wgt
                    t += tt * wgt
                    ws += wgt
                p, t = p / ws, t / ws
            else:
                p, t = local[f]
            if i == 3:                        # lens exactly on the waterline, tilted a fixed few degrees down
                p.z = 0.0
                t.z = -(t - p).xy.length * math.tan(math.radians(SPLIT_PITCH))
            q = look_rotation(p, t)
            if q_prev is not None and q.dot(q_prev) < 0:
                q.negate()
            q_prev = q
            out[f] = (p, t, q)
    return out


def build_camera(col, frames):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = LENS
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = SENSOR
    cd.clip_start = 0.02
    cd.clip_end = 20000.0
    cam = bpy.data.objects.new("Camera", cd)
    col.objects.link(cam)
    cam.rotation_mode = "QUATERNION"
    for f, (p, _t, q) in frames.items():
        cam.location = p
        cam.rotation_quaternion = q
        key(cam, "location", f)
        key(cam, "rotation_quaternion", f)
    bpy.context.scene.camera = cam
    return cam


def under_state(frames):
    return {f: (1.0 if p.z < -0.001 else 0.0) for f, (p, _t, _q) in frames.items()}


def key_scene_props(frames, toggles, water, above_only=(), under_only=()):
    """cam_under and split drive fog, beams and silt; the backdrop and underwater effects swap while a wipe
    is opaque; the surface itself is hidden in the over-under shot and while a wipe hides a crossing."""
    sc = bpy.context.scene
    st = under_state(frames)
    for f in range(1, FRAME_END + 1):
        sc["cam_under"] = st[f]
        sc.keyframe_insert(data_path='["cam_under"]', frame=f)
    for f, v in ((1, 0.0), (SPLIT[0] - 1, 0.0), (SPLIT[0], 1.0), (SPLIT[1], 1.0), (SPLIT[1] + 1, 0.0)):
        sc["split"] = v
        sc.keyframe_insert(data_path='["split"]', frame=f)
    for fc in (sc.animation_data.action.fcurves if hasattr(sc.animation_data.action, "fcurves") else []):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"
    for layer in getattr(sc.animation_data.action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                for fc in bag.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = "CONSTANT"
    for ob, show_when_under in toggles:
        prev = None
        for f in range(1, FRAME_END + 1):
            vis = ((st[f] > 0.5) or (SPLIT[0] <= f <= SPLIT[1])) == show_when_under
            if vis != prev:
                ob.hide_render = not vis
                key(ob, "hide_render", f)
                prev = vis
    # Underwater sun shafts: never in the over-under shot (the lens sits at their tops: full screen overdraw)
    for ob in under_only:
        prev = None
        for f in range(1, FRAME_END + 1):
            vis = st[f] > 0.5 and not SPLIT[0] <= f <= SPLIT[1]
            if vis != prev:
                ob.hide_render = not vis
                key(ob, "hide_render", f)
                prev = vis
    # Seen from under the water the surface is opaque, so nothing above it can show: switch it all off
    for ob in above_only:
        prev = None
        for f in range(1, FRAME_END + 1):
            vis = st[f] < 0.5 or SPLIT[0] <= f <= SPLIT[1]
            if vis != prev:
                ob.hide_render = not vis
                key(ob, "hide_render", f)
                prev = vis
    prev = None
    for f in range(1, FRAME_END + 1):
        hidden = SPLIT[0] <= f <= SPLIT[1] or any(a - 2 <= f <= b + 2 for a, b in WIPES)
        if hidden != prev:
            water.hide_render = hidden
            key(water, "hide_render", f)
            prev = hidden
    return st


def crossings(frames, margin=0.25):
    near = [f for f, (p, _t, _q) in sorted(frames.items()) if abs(p.z) < margin and not SPLIT[0] <= f <= SPLIT[1]]
    spans = []
    for f in near:
        if spans and f == spans[-1][1] + 1:
            spans[-1][1] = f
        else:
            spans.append([f, f])
    return spans


# --------------------------------------------------------------------------
# Overlays: kicker, numbered pills, editorial serif captions, pins, diagram, wipes
# --------------------------------------------------------------------------
class Hud:
    def __init__(self, cam, col, M, frames):
        self.col, self.M, self.frames = col, M, frames
        self.root = empty_object("HUD", col, loc=(0, 0, -HUD_DIST), parent=cam, size=0.02)
        s = HUD_DIST * SENSOR / LENS            # frame height at the HUD distance
        self.root.scale = (s, s, s)
        self.fonts = {k: bpy.data.fonts.load(os.path.join(FONT_DIR, fn), check_existing=True) for k, fn in {
            "serif": "PlayfairDisplay-SemiBold.ttf", "serif_it": "PlayfairDisplay-BoldItalic.ttf",
            "serif_med": "PlayfairDisplay-Medium.ttf", "bold": "Inter-Bold.otf", "semi": "Inter-SemiBold.otf",
            "med": "Inter-Medium.otf"}.items()}
        self.boxes = []      # (name, frame -> (xmin, ymin, xmax, ymax) or None)
        self.events = []     # (frame, kind, label) for the soundtrack
        self.n = 0

    def _name(self, base):
        self.n += 1
        return f"HUD_{self.n:03d}_{base}"

    def _link(self, ob, parent):
        ob.parent = parent or self.root
        ob.visible_shadow = False
        return ob

    def text(self, body, font, size, parent=None, loc=(0, 0, 0.001), align="CENTER", spacing=1.0, mat=None):
        cu = bpy.data.curves.new(self._name("txt"), "FONT")
        cu.body = body
        cu.font = self.fonts[font]
        cu.size = size
        cu.align_x = align
        cu.align_y = "CENTER"
        cu.space_character = spacing
        cu.resolution_u = 6
        cu.materials.append(mat or self.M.hud_white)
        ob = bpy.data.objects.new(cu.name, cu)
        self.col.objects.link(ob)
        ob.location = loc
        return self._link(ob, parent)

    def shape(self, verts2d, color, parent=None, loc=(0, 0, 0.0), mat=None):
        bm = bmesh.new()
        bm.faces.new([bm.verts.new((x, y, 0.0)) for x, y in verts2d])
        ob = mesh_object(self._name("shape"), bm, self.col, [mat or self.M.hud], color=color, recalc=False)
        ob.location = loc
        return self._link(ob, parent)

    @staticmethod
    def rounded(w, h, r, n=8):
        r = min(r, h / 2, w / 2)
        pts = []
        for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90),
                           (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
            for i in range(n + 1):
                a = math.radians(a0 + 90 * i / n)
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return pts

    @staticmethod
    def circle(r, n=32):
        return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]

    @staticmethod
    def bounds(ob):
        ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
        xs = [v[0] for v in ev.bound_box]
        ys = [v[1] for v in ev.bound_box]
        return min(xs), max(xs), min(ys), max(ys)

    def group(self, loc=(0, 0, 0)):
        return empty_object(self._name("grp"), self.col, loc=loc, parent=self.root, size=0.01)

    def fade(self, objs, f_in, f_out, peak=1.0, dur=6):
        for ob in objs:
            k = ob.get("alpha_scale", 1.0)
            pts = [(f_in - dur, 0.0), (f_in, peak), (f_out, peak), (f_out + dur, 0.0)] if f_in > 1 else \
                [(1, peak), (f_out, peak), (f_out + dur, 0.0)]
            key_alpha(ob, [(max(1, f), a * k) for f, a in pts])
            if f_in - dur > 1:
                key_alpha(ob, [(1, 0.0)])

    def slide(self, g, f_in, f_out, base, dx=0.0, dy=0.0, dur=7):
        """Snappy ease-out entrance and ease-in exit, keyed still across each shutter (stays sharp)."""
        for f in range(max(1, f_in - dur), f_in + 1):
            u = smoothstep(f_in - dur, f_in, f)
            g.location = (base[0] + dx * (1 - u) ** 2, base[1] + dy * (1 - u) ** 2, base[2])
            key_still(g, "location", f)
        for f in range(f_out, f_out + dur + 1):
            u = smoothstep(f_out, f_out + dur, f)
            g.location = (base[0] - dx * 0.5 * u * u, base[1] - dy * 0.5 * u * u, base[2])
            key_still(g, "location", f)

    # ---- the series kicker -----------------------------------------------------
    def kicker(self, text, y=0.452):
        g = self.group((0, y, 0))
        sh = self.text(text, "semi", 0.0145, g, loc=(0.0016, -0.0018, 0.0005), spacing=1.65, mat=self.M.hud_shadow)
        tx = self.text(text, "semi", 0.0145, g, spacing=1.65)
        x0, x1, _, _ = self.bounds(tx)
        rule_l = self.shape([(-0.03, -0.0007), (0.0, -0.0007), (0.0, 0.0007), (-0.03, 0.0007)], GOLD, g,
                            loc=(x0 - 0.014, 0, 0.001))
        rule_r = self.shape([(0.0, -0.0007), (0.03, -0.0007), (0.03, 0.0007), (0.0, 0.0007)], GOLD, g,
                            loc=(x1 + 0.014, 0, 0.001))
        self.fade([sh, tx, rule_l, rule_r], 1, FRAME_END + 20)
        self.boxes.append(("kicker", lambda f: (x0 - 0.05, y - 0.012, x1 + 0.05, y + 0.012)))

    # ---- numbered pill badge ------------------------------------------------------
    def badge(self, num, title, sub, f_in, f_out, y=0.392):
        self.events.append((f_in - 6, "badge", title))
        g = self.group((0, y, 0))
        t = self.text(title, "bold", 0.0245, g, align="LEFT", spacing=1.04)
        s = self.text(sub, "med", 0.0148, g, align="LEFT", mat=self.M.hud)
        s.color = (*srgb(196, 206, 210), 0.0)
        nb = self.text(str(num), "bold", 0.0215, g, mat=self.M.hud)
        nb.color = (*INK, 1.0)
        tx0, tx1, _, _ = self.bounds(t)
        sx0, sx1, _, _ = self.bounds(s)
        D, gap, pad, h = 0.043, 0.014, 0.014, 0.066
        w = pad + D + gap + max(tx1 - tx0, sx1 - sx0) + pad + 0.012
        pill = self.shape(self.rounded(w, h, h / 2), INK, g)
        pill["alpha_scale"] = 0.82
        edge = self.shape(self.rounded(w + 0.003, h + 0.003, (h + 0.003) / 2), srgb(60, 70, 74), g, loc=(0, 0, -0.0003))
        edge["alpha_scale"] = 0.82
        cx = -w / 2 + pad + D / 2
        disc = self.shape(self.circle(D / 2), BADGE, g, loc=(cx, 0, 0.0005))
        nb.location = (cx, -0.0008, 0.001)
        t.location = (cx + D / 2 + gap - tx0, 0.0105, 0.001)
        s.location = (cx + D / 2 + gap - sx0, -0.0145, 0.001)
        self.fade([pill, edge, disc, nb, t, s], f_in, f_out)
        self.slide(g, f_in, f_out, (0, y, 0), dx=-0.06)
        self.boxes.append((f"badge {title}", lambda f, a=f_in, b=f_out, w=w: (-w / 2, y - h / 2, w / 2, y + h / 2)
                           if a - 4 <= f <= b + 4 else None))
        return g

    # ---- editorial serif caption ---------------------------------------------------
    def serif(self, text, f_start, f_end, y=-0.315, size=0.046, max_w=0.50, src=None, lines=None, reveal=3.0):
        """Playfair caption revealed word by word. *Starred* words are gold Playfair Bold Italic with a gold
        underline that draws in. lines: force line breaks (list of strings). src: small source line below."""
        self.events.append((f_start - 4, "caption", text))
        if lines is None:
            lines = [text]
        made = []                                   # per line: list of (tx, sh, accent)
        for li, ln in enumerate(lines):
            acc = False
            row = []
            for wd in ln.split():
                start = wd.startswith("*")
                end = wd.rstrip(".,:;!?").endswith("*")
                clean = wd.replace("*", "")
                acc = acc or start
                font = "serif_it" if acc else "serif"
                mat = self.M.hud_accent if acc else self.M.hud_white
                g = self.group()
                sh = self.text(clean, font, size, g, loc=(0.0024, -0.003, 0.0004), align="LEFT", mat=self.M.hud_shadow)
                tx = self.text(clean, font, size, g, align="LEFT", mat=mat)
                row.append((g, tx, sh, acc))
                if end:
                    acc = False
            made.append(row)
        space = size * 0.27
        # automatic wrap when a single given line is too wide
        def widths(row):
            return [self.bounds(tx)[1] - self.bounds(tx)[0] for _g, tx, _s, _a in row]
        if len(made) == 1:
            row = made[0]
            ws = widths(row)
            total = sum(ws) + space * (len(ws) - 1)
            if total > max_w:
                n_lines = int(math.ceil(total / max_w))
                target = total / n_lines
                rows, cur, cw = [], [], 0.0
                for it, wv in zip(row, ws):
                    if cur and cw + space + wv > target * 1.08 and len(rows) < n_lines - 1:
                        rows.append(cur)
                        cur, cw = [], 0.0
                    cur.append(it)
                    cw += (space if len(cur) > 1 else 0.0) + wv
                rows.append(cur)
                made = rows
        lh = size * 1.22
        n_words = sum(len(r) for r in made)
        ytop = y + (len(made) - 1) * lh / 2
        i = 0
        x_min, x_max = 0.0, 0.0
        under = []
        for li, row in enumerate(made):
            ws = widths(row)
            lw = sum(ws) + space * (len(ws) - 1)
            x = -lw / 2
            x_min, x_max = min(x_min, x), max(x_max, x + lw)
            yl = ytop - li * lh
            for (g, tx, sh, acc), wv in zip(row, ws):
                x0 = self.bounds(tx)[0]
                tx.location.x -= x0
                sh.location.x -= x0
                fw = f_start + int(i * reveal)
                base = (x, yl, 0.0)
                if fw <= 1:                                     # already on screen at the first frame
                    g.location = base
                    key_still(g, "location", 1)
                for f in range(max(1, fw - 4), fw + 1) if fw > 1 else ():                # pop: rise and settle
                    u = smoothstep(fw - 4, fw, f)
                    g.location = (base[0], base[1] - 0.014 * (1 - u) ** 2, 0.0)
                    g.scale = (1.0 + 0.10 * (1 - u), 1.0 + 0.10 * (1 - u), 1.0)
                    key_still(g, "location", f)
                    key_still(g, "scale", f)
                for f in range(f_end, f_end + 6):
                    u = smoothstep(f_end, f_end + 5, f)
                    g.location = (base[0], base[1] + 0.008 * u * u, 0.0)
                    key_still(g, "location", f)
                if fw <= 1:
                    key_alpha(tx, [(1, 1.0), (f_end, 1.0), (f_end + 5, 0.0)])
                    key_alpha(sh, [(1, 0.9), (f_end, 0.9), (f_end + 5, 0.0)])
                else:
                    key_alpha(tx, [(1, 0.0), (max(1, fw - 3), 0.0), (fw, 1.0), (f_end, 1.0), (f_end + 5, 0.0)])
                    key_alpha(sh, [(1, 0.0), (max(1, fw - 3), 0.0), (fw, 0.9), (f_end, 0.9), (f_end + 5, 0.0)])
                if acc:
                    under.append((x, yl, wv, fw))
                x += wv + space
                i += 1
        for x, yl, wv, fw in under:                           # gold underline draws in under each accent word
            g = self.group((x, yl - size * 0.52, 0.0005))
            bar = self.shape([(0, -0.0016), (wv, -0.0016), (wv, 0.0016), (0, 0.0016)], GOLD, g)
            fw = max(2, fw)
            for f in range(fw, fw + 9):
                u = smoothstep(fw, fw + 8, f)
                g.scale = (max(1e-3, u), 1.0, 1.0)
                key_still(g, "scale", f)
            key_alpha(bar, [(1, 0.0), (fw, 0.0), (fw + 1, 1.0), (f_end, 1.0), (f_end + 5, 0.0)])
        h = len(made) * lh
        bot = ytop - (len(made) - 1) * lh - size * 0.62
        if src:
            st = self.text(src, "semi", 0.0108, None, loc=(0, bot - 0.014, 0.001), spacing=1.35, mat=self.M.hud)
            st.color = (*srgb(214, 222, 224), 0.0)
            self.fade([st], f_start + 4, f_end, peak=0.92, dur=5)
            bot -= 0.022
        self.boxes.append((f"caption {text[:22]}", lambda f, a=f_start, b=f_end: (x_min - 0.01, bot - 0.004, x_max + 0.01,
                                                                                ytop + size * 0.6)
                           if a - 4 <= f <= b + 4 else None))

    # ---- pins (label pills on leader lines pointing at the world) --------------------------
    def project(self, f, world):
        p, _t, q = self.frames[f]
        m = Matrix.LocRotScale(p, q, None).inverted()
        c = m @ Vector(world)
        if c.z > -0.05:
            return None
        k = LENS / SENSOR
        return Vector((c.x / -c.z * k, c.y / -c.z * k))

    def pin(self, anchor, label, f_lo, f_hi, rise=0.07, dot_color=PIN_RED, y_band=(-0.17, 0.30), side=0.0):
        run, best = [], []
        for f in range(f_lo, f_hi + 1):
            s = self.project(f, anchor(f) if callable(anchor) else anchor)
            ok = s is not None and abs(s.x) < HUD_W - 0.04 and y_band[0] < s.y < y_band[1]
            run = run + [f] if ok else []
            if len(run) > len(best):
                best = run
        if len(best) < 24:
            print(f"WARNING pin '{label}' only on screen for {len(best)} frames in {f_lo}-{f_hi}")
        if not best:
            return None
        f_in, f_out = best[0] + (6 if best[0] > f_lo else 0), best[-1] - (6 if best[-1] < f_hi else 0)
        self.events.append((f_in - 6, "pin", label))
        g = self.group()
        lab = self.group()
        tx = self.text(label.upper(), "bold", 0.0135, lab, spacing=1.12)
        x0, x1, _, _ = self.bounds(tx)
        tx.location = (-(x0 + x1) / 2, -0.0008, 0.001)
        w, h = (x1 - x0) + 0.032, 0.034
        pill = self.shape(self.rounded(w, h, h / 2), INK, lab)
        pill["alpha_scale"] = 0.85
        dot = self.shape(self.circle(0.0068), dot_color, g, loc=(0, 0, 0.0008))
        halo = self.shape(self.circle(0.015), tuple(c * 0.45 for c in dot_color), g, loc=(0, 0, 0.0006))
        halo["alpha_scale"] = 0.7
        lead = self.shape([(-0.0008, 0), (0.0008, 0), (0.0008, 1), (-0.0008, 1)], WHITE, g, loc=(0, 0, 0.0004))
        boxes = {}
        span = list(range(max(1, f_in - 8), min(FRAME_END, f_out + 8) + 1))
        dots, labs = {}, {}
        for f in span:
            s = self.project(f, anchor(f) if callable(anchor) else anchor) or Vector((0, -2))
            dots[f] = s
            labs[f] = Vector((max(-HUD_W + w / 2 + 0.02, min(HUD_W - w / 2 - 0.02, s.x + side)), min(0.31, s.y + rise + h / 2)))
        smooth = {}
        for f in span:
            acc, wsum = Vector((0, 0)), 0.0
            for k in range(-14, 15):
                g2 = min(span[-1], max(span[0], f + k))
                wgt = math.exp(-0.5 * (k / 6.0) ** 2)
                acc += labs[g2] * wgt
                wsum += wgt
            smooth[f] = acc / wsum
        for f in span:
            s = dots[f]
            lx, ly = smooth[f].x, smooth[f].y
            dot.location = (s.x, s.y, 0.0008)
            halo.location = (s.x, s.y, 0.0006)
            sc_ = 1.0 + 0.25 * math.sin(f * 0.25)
            halo.scale = (sc_, sc_, 1)
            lab.location = (lx, ly, 0.0)
            ln = max(0.0, ly - h / 2 - s.y - 0.008)
            lead.location = (s.x, s.y + 0.008, 0.0004)
            lead.scale = (1, max(ln, 1e-4), 1)
            for ob, path in ((dot, "location"), (halo, "location"), (halo, "scale"), (lab, "location"),
                             (lead, "location"), (lead, "scale")):
                key_still(ob, path, f)
            boxes[f] = (min(lx - w / 2, s.x - 0.016), s.y - 0.016, max(lx + w / 2, s.x + 0.016), ly + h / 2)
        self.fade([tx, pill, dot, halo, lead], f_in, f_out)
        self.boxes.append((f"pin {label}", lambda f, a=f_in, b=f_out, bx=boxes: bx.get(f) if a - 4 <= f <= b + 4 else None))
        return g

    # ---- diagram elements for the cross-section ------------------------------------------
    def flow_arrows(self, f_in, f_out, y=0.318):
        """CURRENT label and chevrons streaming left to right (downstream is screen right in the side view)."""
        g = self.group((0, y, 0))
        lab = self.text("CURRENT", "bold", 0.0118, g, loc=(-HUD_W + 0.065, 0.0, 0.001), spacing=1.4)
        objs = [lab]
        row = self.group((0, 0, 0))
        row.parent = g
        step = 0.052
        for i in range(10):
            x = -HUD_W + 0.13 + i * step
            ch = self.shape([(-0.008, 0.011), (0.004, 0.0), (-0.008, -0.011), (-0.004, -0.011), (0.008, 0.0), (-0.004, 0.011)],
                            WHITE, row, loc=(x, 0, 0.001))
            ch["alpha_scale"] = 0.35 + 0.45 * (i % 3 == 0)
            objs.append(ch)
        for f in range(max(1, f_in - 6), f_out + 7):
            row.location = (((f - f_in) * 0.0022) % step, 0, 0)
            key_still(row, "location", f)
        self.fade(objs, f_in, f_out)
        self.boxes.append(("current arrows", lambda f: (-HUD_W + 0.02, y - 0.016, HUD_W - 0.02, y + 0.016)
                           if f_in - 4 <= f <= f_out + 4 else None))

    def dimension(self, a_fn, b_fn, label, f_in, f_out, drop=0.045):
        """Dimension bracket between two world points, drawn below them, with its label."""
        g = self.group()
        line = self.shape([(0, -0.0009), (1, -0.0009), (1, 0.0009), (0, 0.0009)], GOLD, g, loc=(0, 0, 0.001))
        ta = self.shape([(-0.0009, -0.012), (0.0009, -0.012), (0.0009, 0.012), (-0.0009, 0.012)], GOLD, g, loc=(0, 0, 0.001))
        tb = self.shape([(-0.0009, -0.012), (0.0009, -0.012), (0.0009, 0.012), (-0.0009, 0.012)], GOLD, g, loc=(0, 0, 0.001))
        lab = self.group()
        tx = self.text(label, "bold", 0.0148, lab, spacing=1.05, mat=self.M.hud)
        tx.color = (*INK, 0.0)
        x0, x1, _, _ = self.bounds(tx)
        tx.location = (-(x0 + x1) / 2, -0.0008, 0.001)
        w, h = (x1 - x0) + 0.026, 0.03
        plate = self.shape(self.rounded(w, h, h / 2), GOLD, lab)
        boxes = {}
        for f in range(max(1, f_in - 6), min(FRAME_END, f_out + 6) + 1):
            a, b = self.project(f, a_fn(f)), self.project(f, b_fn(f))
            if a is None or b is None:
                a = b = Vector((0, -2))
            if a.x > b.x:
                a, b = b, a
            yy = min(a.y, b.y) - drop
            line.location = (a.x, yy, 0.001)
            line.scale = (max(1e-4, b.x - a.x), 1, 1)
            ta.location = (a.x, yy, 0.001)
            tb.location = (b.x, yy, 0.001)
            lab.location = ((a.x + b.x) / 2, yy - 0.024, 0.0)
            for ob, path in ((line, "location"), (line, "scale"), (ta, "location"), (tb, "location"), (lab, "location")):
                key_still(ob, path, f)
            boxes[f] = (min(a.x, (a.x + b.x) / 2 - w / 2), yy - 0.024 - h / 2, max(b.x, (a.x + b.x) / 2 + w / 2), yy + 0.012)
        self.fade([line, ta, tb, tx, plate], f_in, f_out)
        self.events.append((f_in - 6, "pin", label))
        self.boxes.append((f"dimension {label}", lambda f, bx=boxes: bx.get(f) if f_in - 4 <= f <= f_out + 4 else None))

    def waterline(self, f_in, f_out):
        """The meniscus of the over-under shot: a bright rippled edge with a soft dark band just below it."""
        y0 = math.tan(math.radians(SPLIT_PITCH)) * LENS / SENSOR      # where the water plane meets the frame
        g = self.group((0, y0, 0.0002))
        pts_top, pts_bot = [], []
        for i in range(41):
            x = -0.3 + 0.6 * i / 40
            yv = 0.0025 * math.sin(i * 1.3) + 0.0015 * math.sin(i * 3.1)
            pts_top.append((x, yv + 0.0016))
            pts_bot.append((x, yv - 0.0016))
        edge = self.shape(pts_top + list(reversed(pts_bot)), srgb(236, 250, 246), g)
        edge["alpha_scale"] = 0.75
        band = self.shape([(-0.3, -0.012), (0.3, -0.012), (0.3, -0.002), (-0.3, -0.002)], srgb(8, 40, 40), g)
        band["alpha_scale"] = 0.35
        for f in range(f_in, f_out + 1):
            g.location = (0.006 * math.sin(f * 0.21), y0 + 0.0012 * math.sin(f * 0.37), 0.0002)
            key_still(g, "location", f)
        self.fade([edge, band], f_in, f_out, dur=1)

    def recap(self, items, f0, f_out, y0=0.13, step=0.088, gap=16):
        self.events.append((f0 - 6, "headline", "RECAP"))
        k = self.group((0, y0 + 0.075, 0))
        kt = self.text("THE RECAP", "semi", 0.0135, k, spacing=1.6)
        kt.color = (1, 1, 1, 0)
        self.fade([kt], f0, f_out)
        for i, (num, title) in enumerate(items):
            f_in = f0 + i * gap
            y = y0 - i * step
            self.events.append((f_in - 6, "badge", title))
            g = self.group((0, y, 0))
            t = self.text(title, "bold", 0.027, g, align="LEFT", spacing=1.04)
            nb = self.text(str(num), "bold", 0.024, g, mat=self.M.hud)
            nb.color = (*INK, 1.0)
            tx0, tx1, _, _ = self.bounds(t)
            D, gap_, pad, h = 0.05, 0.016, 0.014, 0.068
            w = pad + D + gap_ + (tx1 - tx0) + pad + 0.014
            pill = self.shape(self.rounded(w, h, h / 2), INK, g)
            pill["alpha_scale"] = 0.84
            cx = -w / 2 + pad + D / 2
            disc = self.shape(self.circle(D / 2), BADGE, g, loc=(cx, 0, 0.0005))
            nb.location = (cx, -0.0008, 0.001)
            t.location = (cx + D / 2 + gap_ - tx0, -0.001, 0.001)
            self.fade([pill, disc, nb, t], f_in, f_out)
            self.slide(g, f_in, f_out, (0, y, 0), dx=0.08 if i % 2 else -0.08)
            self.boxes.append((f"recap {title}", lambda f, a=f_in, b=f_out, w=w, y=y, h=h: (-w / 2, y - h / 2, w / 2, y + h / 2)
                               if a - 4 <= f <= b + 4 else None))
        self.boxes.append(("recap kicker", lambda f: (-0.07, y0 + 0.065, 0.07, y0 + 0.085) if f0 - 4 <= f <= f_out + 4 else None))

    def scrims(self):
        top = self.shape([(-0.3, 0.30), (0.3, 0.30), (0.3, 0.51), (-0.3, 0.51)], WHITE, mat=self.M.scrim_top, loc=(0, 0, -0.0006))
        top.color[3] = 0.45
        bot = self.shape([(-0.3, -0.51), (0.3, -0.51), (0.3, -0.12), (-0.3, -0.12)], WHITE, mat=self.M.scrim_bot,
                         loc=(0, 0, -0.0006))
        for f, a in ((1, 0.62), (SPLIT[0] - 1, 0.62), (SPLIT[0], 0.22), (SPLIT[1], 0.22), (SPLIT[1] + 1, 0.62)):
            bot.color[3] = a
            bot.keyframe_insert("color", index=3, frame=f)

    def wipe(self, f_a, f_b, under=True):
        """Slide wipe: a deep teal panel with a gold leading edge sweeps up over the frame, holds while the
        cut (or the surface crossing) happens behind it, then continues off the top."""
        g = self.group((0, 0, 0.004))
        panel = self.shape([(-0.31, -0.53), (0.31, -0.53), (0.31, 0.53), (-0.31, 0.53)], srgb(10, 64, 66), g)
        glow = self.shape([(-0.31, -0.012), (0.31, -0.012), (0.31, 0.012), (-0.31, 0.012)], srgb(30, 150, 140), g,
                          loc=(0, 0.542, 0.0005))
        edge = self.shape([(-0.31, -0.003), (0.31, -0.003), (0.31, 0.003), (-0.31, 0.003)], GOLD, g, loc=(0, 0.533, 0.001))
        objs = [panel, glow, edge]
        rng = random.Random(f_a)
        if under:
            for i in range(14):
                r = rng.uniform(0.005, 0.016)
                b = self.shape(self.circle(r, 16), srgb(170, 230, 220), g, loc=(rng.uniform(-0.26, 0.26), rng.uniform(-0.45, 0.45), 0.001))
                b["alpha_scale"] = 0.6
                objs.append(b)
        for f in range(f_a - 6, f_b + 9):
            if f <= f_a - 2:
                u = smoothstep(f_a - 6, f_a - 2, f)
                yv = -1.08 * (1 - u)
            elif f <= f_b + 2:
                yv = 0.0
            else:
                u = smoothstep(f_b + 2, f_b + 8, f)
                yv = 1.08 * u
            g.location = (0, yv, 0.004)
            key_still(g, "location", f)
        for ob in objs:
            k = ob.get("alpha_scale", 1.0)
            key_alpha(ob, [(1, 0.0), (f_a - 7, 0.0), (f_a - 6, k), (f_b + 8, k), (f_b + 9, 0.0)])
        self.events.append((f_a - 6, "wipe", "wipe"))


def build_overlays(hud, frames, act):
    sp, rig = act["striker_path"], act["rig"]
    hud.scrims()
    hud.kicker("THE FISHING SERIES")
    # 1 hook
    hud.serif("Stop wasting your day: sockeye pass *behind you.*", -8, CUT1 - 8, y=0.255, size=0.058,
              lines=["Stop wasting your day:", "sockeye pass *behind you.*"], reveal=3.2)
    # 2 location
    lane_pt = lambda f: Vector((rx(-4.0, 0.72), -4.0, 0.05))
    hud.badge(1, "THE MIGRATION HIGHWAY", "sockeye travel very close to shore", 140, 300)
    hud.serif("Sockeye travel very *close to shore.*", 150, 296, src="SOURCE: ADF&G",
              lines=["Sockeye travel very", "*close to shore.*"])
    hud.pin(lane_pt, "migration lane", 152, 296)
    hud.badge(2, "THE TAILOUT ZONE", "slow water before the next riffle", 316, CUT2 - 8)
    hud.serif("They hold in slower water before running the *riffles.*", 322, 394, src="SOURCE: ADF&G",
              lines=["They hold in slower water", "before running the *riffles.*"])
    hud.serif("Wade in too far and they slip *behind you.*", 406, CUT2 - 8, src="SOURCE: ADF&G",
              lines=["Wade in too far and", "they slip *behind you.*"])
    hud.pin(Vector((rx(-38.0, 0.0), -38.0, 0.05)), "fast riffle", 322, 394)
    hud.pin(Vector((rx(-12.8, 0.70), -12.8, -0.5)), "sockeye holding", 406, CUT2 - 10, dot_color=GOLD)
    # 3 rig and technique (underwater side view)
    hud.badge(3, "THE DRIFT RIG", "bounce the bottom, dead drift", 500, CUT3 - 8)
    hud.flow_arrows(498, CUT3 - 8)
    hud.serif("Flip it *10 to 15 feet* upstream, about 45 degrees.", 508, 640, src="SOURCE: ADF&G",
              lines=["Flip it *10 to 15 feet* upstream,", "about 45 degrees."])
    hud.serif("Let it *dead drift* with the current.", 650, 788, src="SOURCE: ADF&G",
              lines=["Let it *dead drift*", "with the current."])
    hud.serif("Add weight so the fly *bounces the bottom.*", 800, 940, src="SOURCE: ADF&G",
              lines=["Add weight so the fly", "*bounces the bottom.*"])
    hud.serif("Weight at least *18 inches* ahead of the fly.", 952, CUT3 - 8, src="RUSSIAN RIVER RULE, ADF&G",
              lines=["Weight at least *18 inches*", "ahead of the fly."])
    w_at = lambda f: rig[f][0]
    f_at = lambda f: rig[f][1]
    hud.pin(lambda f: w_at(f) + Vector((0, 0, 0.03)), "weight", 560, 690)
    hud.pin(lambda f: f_at(f) + Vector((0, 0, 0.03)), "yarn fly", 580, 690, dot_color=srgb(255, 110, 150), side=0.07)
    hud.pin(lambda f: w_at(f) + Vector((0, 0, 0.02)), "ticks the gravel", 800, 935)
    hud.dimension(w_at, f_at, "18 IN +", 955, CUT3 - 14)
    hud.pin(lambda f: f_at(f) + Vector((0, 0, 0.02)), "strike zone", 955, CUT3 - 14, dot_color=GOLD, rise=0.11, side=-0.05)
    # 4 strike: over-under, then above water
    hud.waterline(CUT3, CUT4 - 1)
    hud.badge(4, "THE STRIKE", "they don't feed, but they do strike", CUT3 + 8, CUT4 - 8)
    hud.serif("Sockeye *stop feeding* once they reach fresh water.", CUT3 + 10, 1164, y=-0.035, size=0.042,
              lines=["Sockeye *stop feeding* once", "they reach fresh water."],
              src="SOURCE: NPS, KATMAI")
    hud.serif("But they will sometimes *strike.*", 1176, CUT4 - 8, y=-0.035, size=0.042, src="SOURCE: NPS, KATMAI")
    hud.pin(lambda f: act["mouth"][f], "the take", STRIKE - 14, STRIKE + 30, dot_color=GOLD, y_band=(-0.40, 0.30),
            rise=-0.03, side=0.14)
    hud.badge(5, "THE HOOK SET", "keep only fish hooked in the mouth", CUT4 + 8, 1552)
    hud.serif("Keep only fish *hooked in the mouth.*", CUT4 + 12, 1400, src="SOURCE: ADF&G",
              lines=["Keep only fish", "*hooked in the mouth.*"])
    hud.serif("Hooked anywhere else? *Release it* right away.", 1410, 1552, src="SOURCE: ADF&G",
              lines=["Hooked anywhere else?", "*Release it* right away."])
    # 5 outro: rapid recap, then the clean final card that loops back to the hook
    hud.recap([(1, "FISH CLOSE TO SHORE"), (2, "BOUNCE THE BOTTOM"), (3, "MOUTH HOOKED ONLY")], 1568, 1690)
    hud.serif("They're passing *right behind you.*", 1702, FRAME_END + 10, y=-0.27, size=0.054,
              lines=["They're passing", "*right behind you.*"], src="CHECK CURRENT LOCAL REGULATIONS", reveal=3.5)
    for (a, b), und in zip(WIPES, (True, True, True, False)):
        hud.wipe(a, b, under=und)


def hide_when_invisible(objs):
    """Switch objects off on every frame where their keyed opacity is zero (dithered zero-alpha surfaces can
    still leave marks, and hidden objects cost nothing to render)."""
    n = 0
    for ob in objs:
        ad = ob.animation_data
        fc = None
        if ad and ad.action:
            if hasattr(ad.action, "fcurves"):
                fc = ad.action.fcurves.find("color", index=3)
            for layer in getattr(ad.action, "layers", []):
                for strip in layer.strips:
                    for bag in getattr(strip, "channelbags", []):
                        fc = fc or bag.fcurves.find("color", index=3)
        if fc is None:
            continue
        prev = None
        for f in range(1, FRAME_END + 1):
            hidden = fc.evaluate(f) <= 0.002
            if hidden != prev:
                ob.hide_render = hidden
                ob.keyframe_insert("hide_render", frame=f)
                prev = hidden
        n += 1
    return n


def gear_highlights(col, M, act):
    shells = []
    for src, name, th, wins in ((act["weight"], "HiWeight", 0.004, [(560, 690), (800, 935)]),
                                (act["fly"], "HiFly", 0.0035, [(580, 690), (955, CUT3 - 10)])):
        s = highlight_shell(col, M, src, name, th)
        for a, b in wins:
            pulse(s, a, b)
        shells.append(s)
    glow = act["rod"]["glow"]
    for fr, b, a in ((1, 0.0, 0.0), (HOOKSET - 2, 0.0, 0.0), (HOOKSET + 2, 2.4, 1.0), (1300, 1.8, 1.0), (1330, 0.9, 1.0),
                     (1360, 1.8, 1.0), (1398, 1.2, 1.0), (1404, 1.2, 0.0)):
        glow.color = (*(c * max(b, 0.001) for c in GLOW), a)
        glow.keyframe_insert("color", frame=fr)
    shells.append(glow)
    return shells


def build_zone_and_lane(col, M):
    """Strike zone band along the bottom (rig beat) and the migration lane on the water (drone beat)."""
    y0, y1 = -16.5, -6.0
    verts, faces = [], []
    n = 40
    for i in range(n + 1):
        y = y0 + (y1 - y0) * i / n
        x = rx(y, RIG_U)
        zb = bed_z(x, y)
        verts += [(x, y, zb + 0.06), (x, y, zb + 0.24)]
    for i in range(n):
        a = 2 * i
        faces.append((a, a + 2, a + 3, a + 1))
    me = bpy.data.meshes.new("StrikeZone")
    me.from_pydata(verts, [], faces)
    me.materials.append(M.zone)
    zone = bpy.data.objects.new("StrikeZone", me)
    col.objects.link(zone)
    zone.visible_shadow = False
    for f, a in ((1, 0.0), (950, 0.0), (962, 0.7), (CUT3 - 2, 0.7), (CUT3 - 1, 0.0)):
        zone.color = (1, 1, 1, a)
        zone.keyframe_insert("color", frame=f)
    # lane: from the lower riffle up the right bank, past the tailout, into the pool
    verts, faces, along, across = [], [], [], []
    ys = np.linspace(-70.0, 30.0, 201)
    for y in ys:
        u = 0.70 - 0.08 * math.sin((y + 70.0) / 30.0)
        for k, du in enumerate((-0.07, 0.07)):
            verts.append((rx(y, u + du), y, 0.06))
            along.append(float(y))
            across.append(float(k))
    for i in range(len(ys) - 1):
        a = 2 * i
        faces.append((a, a + 1, a + 3, a + 2))
    me = bpy.data.meshes.new("MigrationLane")
    me.from_pydata(verts, [], faces)
    for nm, vals in (("along", along), ("across", across)):
        at = me.attributes.new(nm, "FLOAT", "POINT")
        at.data.foreach_set("value", vals)
    me.materials.append(M.lane)
    lane = bpy.data.objects.new("MigrationLane", me)
    col.objects.link(lane)
    lane.visible_shadow = False
    for f, a in ((1, 0.0), (142, 0.0), (160, 1.0), (300, 1.0), (318, 0.0)):
        lane.color = (1, 1, 1, a)
        lane.keyframe_insert("color", frame=f)
    return zone, lane


# --------------------------------------------------------------------------
# Render settings
# --------------------------------------------------------------------------
def setup_render(scale, samples, blur=True):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y = RES_X, RES_Y
    sc.render.resolution_percentage = scale
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, FRAME_END
    ee = sc.eevee
    ee.taa_render_samples = samples
    ee.use_shadows = True
    ee.use_raytracing = True
    try:
        ee.ray_tracing_method = "SCREEN"
    except (AttributeError, TypeError):
        pass
    ee.use_fast_gi = False
    sc.render.use_motion_blur = blur
    sc.render.motion_blur_shutter = SHUTTER
    ee.motion_blur_steps = 1
    sc.view_settings.view_transform = "Standard"
    try:
        sc.view_settings.look = "None"
    except TypeError:
        pass
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    # cinematic grade with the view curve: gentle S for contrast, lifted blacks kept teal
    sc.view_settings.use_curve_mapping = True
    cm = sc.view_settings.curve_mapping
    c = cm.curves[3]
    c.points[0].location = (0.0, 0.012)
    c.points[1].location = (1.0, 1.0)
    c.points.new(0.25, 0.215)
    c.points.new(0.75, 0.79)
    cm.update()
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def run_checks(frames, st):
    sc = bpy.context.scene
    act = st["act"]
    print("\n=== CAMERA REPORT (frame: pos | pitch | speed m/s | above bed | above water) ===")
    worst = {k: (1e9, 0, "") for k in ("bed", "boulder", "pine", "angler", "fish", "rod", "rig")}
    prev = None
    boulders = st["boulders"]
    trunks = st["trunks"]
    for f in range(1, FRAME_END + 1):
        p, _t, q = frames[f]
        fwd = q @ Vector((0, 0, -1))
        pitch = math.degrees(math.asin(max(-1, min(1, fwd.z))))
        speed = 0.0 if prev is None or f in [s[0] for s in SEGMENTS] else (p - prev).length * FPS
        prev = p
        hb = p.z - bed_z(p.x, p.y)
        if hb < worst["bed"][0]:
            worst["bed"] = (hb, f, "")
        for c, r in boulders:
            d = (p - c).length - r * 1.15
            if d < worst["boulder"][0]:
                worst["boulder"] = (d, f, "")
        for c, h in trunks:
            if abs(c.x - p.x) < 8 and abs(c.y - p.y) < 8 and c.z - 1 < p.z < c.z + h:
                rr = 0.30 * h * (1 - (p.z - c.z) / h)
                d = (p.xy - c.xy).length - rr
                if d < worst["pine"][0]:
                    worst["pine"] = (d, f, "")
        if f % 24 == 1 or f == FRAME_END or abs(p.z) < 0.25 or hb < 0.25:
            print(f"{f:5d}: ({p.x:6.1f},{p.y:7.1f},{p.z:6.2f}) | {pitch:6.1f} | {speed:5.2f} | {hb:6.2f} | {p.z:6.2f}")
    fish_objs = [act["hero"], act["striker"]] + act["extras"]
    for f in range(2, FRAME_END + 1, 2):
        p = frames[f][0]
        sc.frame_set(f)
        for ob in fish_objs:
            if ob.hide_render:
                continue
            d = (ob.matrix_world.translation - p).length - 0.5 * ob.scale.x
            if d < worst["fish"][0]:
                worst["fish"] = (d, f, ob.name)
        for q_ in act["school_pts"]:
            d = (q_ - p).length - 0.5 * FISH_LEN
            if d < worst["fish"][0]:
                worst["fish"] = (d, f, "school")
        a = act["rod"]["ctrl"]["Butt"].matrix_world.translation
        b = act["rod"]["ctrl"]["Tip"].matrix_world.translation
        ab = b - a
        u = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
        d = (a + ab * u - p).length
        if d < worst["rod"][0]:
            worst["rod"] = (d, f, "")
        root = act["angler"]["arm"].matrix_world.translation
        dd = math.hypot(p.x - root.x, p.y - root.y)
        if -0.2 < p.z - root.z < 2.0:
            dd -= 0.28
        else:
            dd = (p - (root + Vector((0, 0, 1.0)))).length - 1.0
        if dd < worst["angler"][0]:
            worst["angler"] = (dd, f, "")
        w, fl = act["rig"][f]
        d = min((w - p).length, (fl - p).length)
        if d < worst["rig"][0]:
            worst["rig"] = (d, f, "")
    for k, v in worst.items():
        print(f"closest {k}: {tuple(round(x, 2) if isinstance(x, float) else x for x in v)}")
    acc, turn = (0.0, 0), (0.0, 0)
    cuts = [s[0] for s in SEGMENTS[1:]]
    for f in range(3, FRAME_END):
        if any(f - 2 <= c <= f + 1 for c in cuts):
            continue
        p0, p1, p2 = frames[f - 1][0], frames[f][0], frames[f + 1][0]
        a = ((p2 - p1) - (p1 - p0)).length * FPS * FPS
        if a > acc[0]:
            acc = (a, f)
        ang = math.degrees(frames[f - 1][2].rotation_difference(frames[f][2]).angle) * FPS
        if ang > turn[0]:
            turn = (ang, f)
    print(f"max camera acceleration: {acc[0]:.2f} m/s^2 at frame {acc[1]};  max turn rate: {turn[0]:.1f} deg/s at frame {turn[1]}")
    spans = crossings(frames)
    print("surface crossings (|lens z| < 0.25 m, outside the over-under shot):", spans)
    for a, b in spans:
        print(f"   frames {a}-{b} hidden by a wipe: {any(wa - 2 <= a and b <= wb + 2 for wa, wb in WIPES)}")
    for c in cuts:
        print(f"cut at {c} inside a wipe: {any(wa - 2 <= c - 1 and c <= wb + 2 for wa, wb in WIPES)}")
    hud = st["hud"]
    overlaps = {}
    for f in range(1, FRAME_END + 1):
        live = [(n, fn(f)) for n, fn in hud.boxes]
        live = [(n, b) for n, b in live if b is not None]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                (na, a), (nb, b) = live[i], live[j]
                if a[0] < b[2] - 0.004 and b[0] < a[2] - 0.004 and a[1] < b[3] - 0.004 and b[1] < a[3] - 0.004:
                    overlaps.setdefault((na, nb), []).append(f)
    print("overlay overlaps:", "none" if not overlaps else "")
    for (na, nb), fs in overlaps.items():
        print(f"   {na} x {nb}: frames {fs[0]}-{fs[-1]} ({len(fs)} frames)")
    offscreen = {}
    for n, fn in hud.boxes:
        for f in range(1, FRAME_END + 1):
            b = fn(f)
            if b and (b[0] < -HUD_W + 0.01 or b[2] > HUD_W - 0.01 or b[1] < -0.49 or b[3] > 0.49):
                offscreen.setdefault(n, []).append(f)
    print("overlays leaving the frame:", {k: (v[0], v[-1], len(v)) for k, v in offscreen.items()} or "none")


def subject_report(frames, st):
    """Project the key subject of each beat to the screen at the test frames (in frame-height units)."""
    act, hud = st["act"], st["hud"]
    subj = {1: ("hero", lambda f: act["hero"].matrix_world.translation),
            360: ("school centre", lambda f: on_bed(-12.8, 0.70, 0.1)),
            720: ("weight", lambda f: act["rig"][f][0]), 1080: ("fly", lambda f: act["rig"][f][1]),
            1170: ("striker mouth", lambda f: act["mouth"][f]), 1440: ("striker", lambda f: act["striker_path"][f]),
            1800: ("angler", lambda f: angler_root())}
    for f in (1100, 1200, 1262, 1300, 1380, 1440, 1500, 1560):
        subj[f + 0.5] = ("angler chest", lambda f: angler_root() + Vector((0, 0, 1.3)))
    sc = bpy.context.scene
    print("\n=== SUBJECTS ON SCREEN (x in +-0.281, y in +-0.5) ===")
    for f, (nm, fn) in sorted(subj.items()):
        f = int(f)
        sc.frame_set(f)
        s = hud.project(f, fn(f))
        print(f"frame {f}: {nm} ->", None if s is None else (round(s.x, 3), round(s.y, 3)))


def write_events(path, st):
    import json
    fr = st["frames"]
    data = {
        "fps": FPS, "frames": FRAME_END,
        "camera_z": [round(fr[f][0].z, 3) for f in range(1, FRAME_END + 1)],
        "under": [int(st["under"][f]) for f in range(1, FRAME_END + 1)],
        "riffle": [round(riffle(fr[f][0].y), 3) for f in range(1, FRAME_END + 1)],
        "wipes": WIPES, "cuts": [s[0] for s in SEGMENTS[1:]], "split": list(SPLIT),
        "overlays": sorted(st["hud"].events),
        "strike": STRIKE, "hookset": HOOKSET, "leap": list(LEAP),
        "drift_entries": [d[0] + 8 for d in DRIFTS],
        "beats": [1, CUT1, CUT2, CUT3, 1561],
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    print("events written", path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, FRAME_END
    sc.render.fps = FPS
    for prop in ("cam_under", "split"):
        sc[prop] = 0.0
    c_env, c_life, c_gear = collection("Environment"), collection("Sockeye"), collection("Tackle")
    c_people, c_fx, c_cam = collection("Angler"), collection("Effects"), collection("CameraAndOverlays")
    atmos = build_atmos_group()
    M = build_materials(atmos)
    M.highlight.use_backface_culling = True
    build_world()
    build_sun(c_env)
    frames = camera_frames()
    cam = build_camera(c_cam, frames)
    terrain, far = build_terrain(c_env, M)
    water = build_water(c_env, M)
    back = build_backdrop(c_env, M, cam)
    act = build_action((c_gear, c_life, c_people), M, frames)
    A = angler_root()
    avoid = [(frames[f][0], 2.2) for f in range(1, FRAME_END + 1, 3)]
    avoid += [(A, 2.5), (on_bed(-12.8, 0.7, 0.0), 5.0)]
    boulders = build_boulders(c_env, M, avoid)
    keep_cob = [(frames[f][0], 0.35) for f in range(1, FRAME_END + 1, 2)] + [(A, 0.6)]
    cobbles, _cob = build_cobbles(c_env, M, cam, keep_cob)
    forest, trunks = build_forest(c_env, M, [frames[f][0] for f in range(1, FRAME_END + 1)])
    fx = build_mist_and_rays(c_fx, M)
    beams = build_beams(c_fx, M, cam)
    susp = build_suspended(c_fx, M)
    zone, lane = build_zone_and_lane(c_fx, M)
    above_only = [forest, far, act["spray"]] + [o for o in fx if o.name.startswith(("Mist", "SunRay"))]
    under = key_scene_props(frames, [(back, True)] + [(o, True) for o in susp], water, above_only, [beams])
    shells = gear_highlights(c_gear, M, act)
    hud = Hud(cam, c_cam, M, frames)
    build_overlays(hud, frames, act)
    n = hide_when_invisible([o for o in bpy.data.objects["HUD"].children_recursive if o.type in ("MESH", "FONT")] + shells
                            + [zone, lane])
    print("objects switched off while invisible:", n)
    sc.frame_set(1)
    return dict(frames=frames, hud=hud, under=under, act=act, boulders=boulders, trunks=trunks)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--save", default=None)
    p.add_argument("--check", action="store_true")
    p.add_argument("--frames", default="")
    p.add_argument("--anim", action="store_true")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=FRAME_END)
    p.add_argument("--outdir", default=None)
    p.add_argument("--scale", type=int, default=100)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--no-blur", action="store_true")
    p.add_argument("--events", default=None)
    return p.parse_args(argv)


def main():
    args = parse_args()
    st = build_scene()
    setup_render(args.scale, args.samples, blur=not args.no_blur)
    if args.events:
        write_events(args.events, st)
    if args.check:
        run_checks(st["frames"], st)
        subject_report(st["frames"], st)
    sc = bpy.context.scene
    if args.save:
        sc.frame_set(1)
        sc.render.filepath = "//gpu_render/frames/frame_"
        sc.render.use_overwrite = False
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save), compress=True)
        print("saved", args.save)
    if args.frames:
        os.makedirs(args.outdir, exist_ok=True)
        for f in [int(x) for x in args.frames.split(",") if x.strip()]:
            sc.frame_set(f)
            sc.render.filepath = os.path.join(os.path.abspath(args.outdir), f"frame_{f:04d}.png")
            bpy.ops.render.render(write_still=True)
            print("rendered", sc.render.filepath, flush=True)
    if args.anim:
        os.makedirs(args.outdir, exist_ok=True)
        sc.frame_start, sc.frame_end = args.start, args.end
        sc.render.filepath = os.path.join(os.path.abspath(args.outdir), "frame_")
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
