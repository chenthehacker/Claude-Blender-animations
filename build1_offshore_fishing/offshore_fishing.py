"""Offshore trolling breakdown: mahi mahi on a sargassum weed line. Vertical 9:16, Blender 5.x EEVEE.

Run with the Blender binary:
    blender --background --python offshore_fishing.py -- [options]
or with the bpy Python module (pip install bpy):
    python offshore_fishing.py [options]

Options:
    --save PATH        save the generated scene as a .blend file (textures and fonts packed)
    --check            print camera clearance, surface crossing and overlay overlap reports
    --frames 1,300     render these frames as PNG stills into --outdir
    --anim             render the animation as PNG frames into --outdir
    --start N/--end N  frame range for --anim (default 1..1800)
    --outdir DIR       output directory for renders
    --scale PCT        resolution percentage (default 100 -> 1080x1920)
    --samples N        EEVEE render samples (default 32)
    --no-blur          switch motion blur off (quick tests)
    --events PATH      write overlay and action timing as JSON (input for offshore_audio.py)

The final render is meant for a GPU: see gpu_render/README.md.
Facts used in the on-screen text are listed with their sources in SOURCES.md.

World layout (metres, +Z up). The boat trolls along +Y beside a sargassum weed line that meanders
around x = -13. Everything that rides with the boat (lures, camera moves) is laid out in boat space
and moved by the boat's position, heave, pitch and roll, which come from the same wave field as the sea.

Beats (the brief's 0-15/15-40/40-60/60-80 s timeline scaled to 75 s, factor 75/80 = 0.9375):
    A  F1-338      drone over the wake, high-angle dolly along the weed line, dive
    B  F338-900    underwater: the spread skipping overhead, a mahi school rising behind the baits
    C  F900-1350   close-ups: the strike on the ballyhoo, then the rod, reel and drag in the cockpit
    D  F1350-1800  wide pull-back from the transom: the angler fighting the fish, release summary
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

BEAT_A, BEAT_B, BEAT_C, BEAT_D = 1, 338, 900, 1350
STRIKE = 1010          # the bull mahi takes the left outrigger ballyhoo
TROLL_SPEED = 3.1      # m/s, about 6 knots
UW_FOG_DIST = 26.0     # underwater visibility scale (m)
AIR_HAZE_DIST = 2600.0
DEPTH_FOG = 7.0        # how fast things fade with depth when seen from above (m)

SUN_DIR = Vector((0.25, 0.55, 0.80)).normalized()     # towards the sun: high, ahead of the boat


def srgb(r, g, b):
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (lin(r), lin(g), lin(b))


ORANGE = srgb(245, 178, 76)
WHITE = (1.0, 1.0, 1.0)
INK = srgb(16, 26, 38)
PIN_RED = srgb(240, 92, 80)
WIPE_AQUA = srgb(150, 214, 236)
GLOW = srgb(255, 196, 92)
DEEP_BLUE = srgb(6, 34, 78)
UW_UP, UW_MID, UW_DOWN = srgb(64, 160, 204), srgb(10, 70, 130), srgb(2, 14, 40)


def smoothstep(e0, e1, x):
    if e0 > e1:
        return 1.0 - smoothstep(e1, e0, x)
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def frame_time(f):
    return f / FPS


# --------------------------------------------------------------------------
# Generic helpers (shared with the saltwater build)
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


class Hud:
    def __init__(self, cam, col, M, frames):
        self.col, self.M, self.frames = col, M, frames
        self.root = empty_object("HUD", col, loc=(0, 0, -HUD_DIST), parent=cam, size=0.02)
        s = HUD_DIST * SENSOR / LENS            # frame height at the HUD distance
        self.root.scale = (s, s, s)
        self.fonts = {k: bpy.data.fonts.load(os.path.join(FONT_DIR, fn), check_existing=True) for k, fn in {
            "serif": "PlayfairDisplay-BoldItalic.ttf", "bold": "Inter-Bold.otf",
            "semi": "Inter-SemiBold.otf", "med": "Inter-Medium.otf"}.items()}
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

    def text(self, body, font, size, parent=None, loc=(0, 0, 0.001), align="CENTER", accents=(),
             spacing=1.0, mat=None):
        cu = bpy.data.curves.new(self._name("txt"), "FONT")
        cu.body = body
        cu.font = self.fonts[font]
        cu.size = size
        cu.align_x = align
        cu.align_y = "CENTER"
        cu.space_character = spacing
        cu.resolution_u = 5
        cu.materials.append(mat or self.M.hud_white)
        cu.materials.append(self.M.hud_accent)
        for a0, a1 in accents:
            for i in range(a0, a1):
                cu.body_format[i].material_index = 1
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
    def circle(r, n=28):
        return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]

    @staticmethod
    def bounds(ob):
        ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
        xs = [v[0] for v in ev.bound_box]
        ys = [v[1] for v in ev.bound_box]
        return min(xs), max(xs), min(ys), max(ys)

    def group(self, loc=(0, 0, 0)):
        g = empty_object(self._name("grp"), self.col, loc=loc, parent=self.root, size=0.01)
        return g

    def fade(self, objs, f_in, f_out, peak=1.0, dur=8):
        for ob in objs:
            k = ob.get("alpha_scale", 1.0)
            key_alpha(ob, [(f, a * k) for f, a in fade_pairs(f_in, f_out, peak, dur)])
            if f_in - dur > 1:
                key_alpha(ob, [(1, 0.0)])

    def all_objs(self, g):
        return [c for c in g.children_recursive if c.type in ("MESH", "FONT")]

    # ---- elements ---------------------------------------------------------
    def badge(self, num, title, sub, f_in, f_out, y=0.385):
        self.events.append((f_in - 8, "badge", title))
        g = self.group((0, y, 0))
        t = self.text(title, "bold", 0.027, g, align="LEFT")
        s = self.text(sub, "med", 0.0155, g, align="LEFT", mat=self.M.hud)
        s.color = (0.62, 0.66, 0.70, 0.0)
        nb = self.text(str(num), "bold", 0.022, g, mat=self.M.hud)
        nb.color = (*INK, 1.0)
        tx0, tx1, _, _ = self.bounds(t)
        sx0, sx1, _, _ = self.bounds(s)
        D, gap, pad, h = 0.044, 0.014, 0.017, 0.068
        w = pad + D + gap + max(tx1 - tx0, sx1 - sx0) + pad + 0.006
        pill = self.shape(self.rounded(w, h, h / 2), INK, g)
        cx = -w / 2 + pad + D / 2
        disc = self.shape(self.circle(D / 2), ORANGE, g, loc=(cx, 0, 0.0005))
        nb.location = (cx, -0.001, 0.001)
        t.location = (cx + D / 2 + gap - tx0, 0.0105, 0.001)
        s.location = (cx + D / 2 + gap - sx0, -0.0145, 0.001)
        objs = [pill, disc, nb, t, s]
        self.fade(objs, f_in, f_out)
        g.location.y = y + 0.012
        key(g, "location", f_in - 8)
        g.location.y = y
        key(g, "location", f_in + 4)
        self.boxes.append((f"badge {title}", lambda f, a=f_in, b=f_out, w=w: (-w / 2, y - h / 2, w / 2, y + h / 2)
                           if a - 4 <= f <= b + 4 else None))
        return g

    def headline(self, kicker, lines, f_in, f_out, y=0.43):
        self.events.append((max(1, f_in - 10), "headline", kicker))
        g = self.group((0, 0, 0))
        objs = []
        k = self.text(kicker, "semi", 0.0135, g, loc=(0, y, 0.001), spacing=1.5, mat=self.M.hud)
        k.color = (0.75, 0.77, 0.80, 0.0)
        objs.append(k)
        yy = y - 0.05
        x0, x1 = 0, 0
        for body, accent_words in lines:
            acc = []
            for wd in accent_words:
                i = body.find(wd)
                acc.append((i, i + len(wd)))
            sh = self.text(body, "serif", 0.052, g, loc=(0.0028, yy - 0.0028, 0.0005), mat=self.M.hud_shadow)
            tx = self.text(body, "serif", 0.052, g, loc=(0, yy, 0.001), accents=acc)
            bx0, bx1, _, _ = self.bounds(tx)
            x0, x1 = min(x0, bx0), max(x1, bx1)
            objs += [sh, tx]
            yy -= 0.056
        self.fade(objs, f_in, f_out, dur=10)
        top, bot = y + 0.012, yy + 0.025
        self.boxes.append((f"headline {kicker}", lambda f, a=f_in, b=f_out: (x0, bot, x1, top)
                           if a - 4 <= f <= b + 4 else None))
        return g

    def caption(self, text, f_start, f_end, accents=(), y=-0.17, size=0.036, max_w=0.47):
        self.events.append((f_start - 6, "caption", text))
        words = text.split()
        accents = {a.strip(".,!?:;").lower() for a in accents}
        g = self.group((0, y, 0))
        made = []
        for wd in words:
            clean = wd.strip(".,!?:;").lower()
            sh = self.text(wd, "bold", size, g, align="LEFT", mat=self.M.hud_shadow)
            tx = self.text(wd, "bold", size, g, align="LEFT", mat=self.M.hud)
            tx["base_rgb"] = ORANGE if clean in accents else WHITE
            made.append((wd, tx, sh))
        space = size * 0.3
        items = []
        for wd, tx, sh in made:
            x0, x1, _, _ = self.bounds(tx)
            items.append((tx, sh, x0, x1 - x0))

        def wrap(limit):
            lines, cur, cur_w = [], [], 0.0
            for it in items:
                if cur and cur_w + space + it[3] > limit:
                    lines.append((cur, cur_w))
                    cur, cur_w = [], 0.0
                cur.append(it)
                cur_w += (space if len(cur) > 1 else 0.0) + it[3]
            lines.append((cur, cur_w))
            return lines
        lines = wrap(max_w)
        if len(lines) > 1 and len(lines[-1][0]) == 1:
            # A lone last word: narrow the measure to the tightest width that keeps the same line count
            lo, hi = max(it[3] for it in items), max_w
            for _ in range(30):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if len(wrap(mid)) > len(lines) else (lo, mid)
            lines = wrap(hi)
            print(f"caption rebalanced: {text!r} -> {[len(l[0]) for l in lines]} words per line")
        lh = size * 1.25
        ytop = (len(lines) - 1) * lh / 2
        n_words = len(made)
        span = max(1, f_end - f_start - 24)
        i = 0
        max_w_used = max(w for _, w in lines)
        for li, (items, lw) in enumerate(lines):
            x = -lw / 2
            yl = ytop - li * lh
            for tx, sh, x0, w in items:
                tx.location = (x - x0, yl, 0.001)
                sh.location = (x - x0 + 0.0025, yl - 0.0025, 0.0005)
                t_word = f_start + 4 + int(span * i / max(1, n_words - 1)) if n_words > 1 else f_start + 4
                full = tuple(tx["base_rgb"])
                dim = tuple(c * 0.36 for c in full)
                for fr, rgb, a in ((1, dim, 0.0), (f_start - 6, dim, 0.0), (f_start, dim, 1.0), (t_word, dim, 1.0),
                                   (t_word + 3, full, 1.0), (f_end, full, 1.0), (f_end + 6, full, 0.0)):
                    tx.color = (*rgb, a)
                    tx.keyframe_insert("color", frame=fr)
                key_alpha(sh, [(1, 0.0), (f_start - 6, 0.0), (f_start, 1.0), (f_end, 1.0), (f_end + 6, 0.0)])
                x += w + space
                i += 1
        h = len(lines) * lh
        plate = self.shape(self.rounded(max_w_used + 0.045, h + 0.022, 0.024), INK, g, loc=(0, 0, -0.0002))
        key_alpha(plate, [(1, 0.0), (f_start - 6, 0.0), (f_start, 1.0), (f_end, 1.0), (f_end + 6, 0.0)])
        self.boxes.append((f"caption {text[:18]}", lambda f, a=f_start, b=f_end: (-max_w_used / 2 - 0.023, y - h / 2 - 0.011,
                                                                               max_w_used / 2 + 0.023, y + h / 2 + 0.011)
                           if a - 4 <= f <= b + 4 else None))
        return g

    def project(self, f, world):
        p, _t, q = self.frames[f]
        m = Matrix.LocRotScale(p, q, None).inverted()
        c = m @ Vector(world)
        if c.z > -0.05:
            return None
        k = LENS / SENSOR
        return Vector((c.x / -c.z * k, c.y / -c.z * k))

    def pin(self, anchor, label, f_lo, f_hi, rise=0.075, dot_color=PIN_RED, ring=True):
        """Label pill on a leader line, pointing at a world position (or callable(frame)).
        Shown for the longest stretch of [f_lo, f_hi] where the anchor sits safely in frame."""
        run, best = [], []
        for f in range(f_lo, f_hi + 1):
            s = self.project(f, anchor(f) if callable(anchor) else anchor)
            ok = s is not None and abs(s.x) < HUD_W - 0.04 and -0.10 < s.y < 0.22   # clear of captions
            run = run + [f] if ok else []
            if len(run) > len(best):
                best = run
        if len(best) < 30:
            print(f"WARNING pin '{label}' only on screen for {len(best)} frames in {f_lo}-{f_hi}")
        if not best:
            return None
        f_in, f_out = best[0] + (8 if best[0] > f_lo else 0), best[-1] - (8 if best[-1] < f_hi else 0)
        self.events.append((f_in - 8, "pin", label))
        g = self.group()
        lab = self.group()
        tx = self.text(label, "semi", 0.0165, lab)
        x0, x1, _, _ = self.bounds(tx)
        tx.location = (-(x0 + x1) / 2, -0.001, 0.001)
        w, h = (x1 - x0) + 0.03, 0.036
        pill = self.shape(self.rounded(w, h, h / 2), INK, lab)
        dot = self.shape(self.circle(0.0075), dot_color, g, loc=(0, 0, 0.0008))
        halo = self.shape(self.circle(0.016), tuple(c * 0.4 for c in dot_color), g, loc=(0, 0, 0.0006))
        lead = self.shape([(-0.0009, 0), (0.0009, 0), (0.0009, 1), (-0.0009, 1)], WHITE, g, loc=(0, 0, 0.0004))
        boxes = {}
        span = list(range(max(1, f_in - 10), min(FRAME_END, f_out + 10) + 1))
        dots, labs = {}, {}
        for f in span:
            wp = anchor(f) if callable(anchor) else anchor
            s = self.project(f, wp)
            if s is None:
                s = Vector((0, -2))
            dots[f] = s
            labs[f] = Vector((max(-HUD_W + w / 2 + 0.018, min(HUD_W - w / 2 - 0.018, s.x)), min(0.27, s.y + rise + h / 2)))
        # The label glides on a smoothed path so it does not jitter with the bait; the dot tracks exactly
        smooth = {}
        for f in span:
            acc, wsum = Vector((0, 0)), 0.0
            for k in range(-16, 17):
                g2 = min(span[-1], max(span[0], f + k))
                wgt = math.exp(-0.5 * (k / 7.0) ** 2)
                acc += labs[g2] * wgt
                wsum += wgt
            smooth[f] = acc / wsum
        for f in span:
            s = dots[f]
            lx, ly = smooth[f].x, smooth[f].y
            dot.location = (s.x, s.y, 0.0008)
            halo.location = (s.x, s.y, 0.0006)
            sc = 1.0 + 0.25 * math.sin(f * 0.25)
            halo.scale = (sc, sc, 1)
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

    def glow_dots(self, anchors, f_in, f_out, color=srgb(210, 250, 255)):
        for i, a in enumerate(anchors):
            g = self.group()
            dot = self.shape(self.circle(0.006), color, g, loc=(0, 0, 0.0008))
            halo = self.shape(self.circle(0.014), color, g, loc=(0, 0, 0.0006))
            halo["alpha_scale"] = 0.3
            for f in range(max(1, f_in - 10), min(FRAME_END, f_out + 10) + 1):
                s = self.project(f, a) or Vector((0, -2))
                g.location = (s.x, s.y, 0)
                key_still(g, "location", f)
            tw = 0.75 + 0.25 * math.sin(i)
            self.fade([dot, halo], f_in + i * 3, f_out, peak=tw)

    def depth_gauge(self, under_frames):
        """Left-edge depth ruler; the pointer tracks the camera depth."""
        g = self.group((-HUD_W + 0.045, 0.0, 0))
        top, step = 0.25, 0.05
        objs = [self.shape([(-0.0012, top - 5 * step), (0.0012, top - 5 * step), (0.0012, top), (-0.0012, top)], WHITE, g)]
        title = self.text("DEPTH", "semi", 0.0115, g, loc=(0.012, top + 0.02, 0.001), align="LEFT", spacing=1.4)
        objs.append(title)
        for m in range(6):              # 0 to 10 m in 2 m steps
            yy = top - m * step
            objs.append(self.shape([(-0.008, yy - 0.0012), (0.008, yy - 0.0012), (0.008, yy + 0.0012),
                                    (-0.008, yy + 0.0012)], WHITE, g))
            objs.append(self.text(f"{2 * m} m", "med", 0.0125, g, loc=(0.014, yy, 0.001), align="LEFT"))
        ptr = self.shape([(-0.022, 0.0), (-0.006, 0.009), (-0.006, -0.009)], ORANGE, g, loc=(0, top, 0.001))
        objs.append(ptr)
        for ob in objs:
            ob["alpha_scale"] = 1.0
        spans = []
        for f in under_frames:
            if spans and f == spans[-1][1] + 1:
                spans[-1][1] = f
            else:
                spans.append([f, f])
        for ob in objs:
            pairs = [(1, 0.0)]
            for a, b in spans:
                k = ob.get("alpha_scale", 1.0)
                pairs += [(a + 6, 0.0), (a + 16, k), (b - 6, k), (b, 0.0)]
            key_alpha(ob, pairs)
        for f in under_frames:
            d = max(0.0, min(10.0, -self.frames[f][0].z))
            ptr.location = (0, top - d / 2.0 * step, 0.001)
            key_still(ptr, "location", f)
        self.boxes.append(("depth gauge", lambda f, s=set(under_frames): (-HUD_W + 0.02, top - 5 * step - 0.012,
                                                                          -HUD_W + 0.11, top + 0.03)
                           if f in s else None))

    def scrims(self):
        """Soft dark gradients behind the title area and the caption band."""
        top = self.shape([(-0.3, 0.12), (0.3, 0.12), (0.3, 0.51), (-0.3, 0.51)], WHITE, mat=self.M.scrim,
                         loc=(0, 0, -0.0005))
        top.color[3] = 0.42
        band = self.shape([(-0.3, -0.30), (0.3, -0.30), (0.3, -0.04), (-0.3, -0.04)], WHITE, mat=self.M.scrim_band,
                          loc=(0, 0, -0.0005))
        band.color[3] = 0.2

    def wipe(self, f_a, f_b):
        """Dip to aqua with rising bubbles to hide the lens crossing the surface."""
        g = self.group((0, 0, 0.004))
        plane = self.shape([(-0.31, -0.53), (0.31, -0.53), (0.31, 0.53), (-0.31, 0.53)], WIPE_AQUA, g)
        key_alpha(plane, [(1, 0.0), (f_a - 9, 0.0), (f_a - 2, 1.0), (f_b + 2, 1.0), (f_b + 12, 0.0)])
        rng = random.Random(f_a)
        for i in range(16):
            r = rng.uniform(0.006, 0.02)
            b = self.shape(self.circle(r, 16), WHITE, g, loc=(0, 0, 0.001))
            x = rng.uniform(-0.26, 0.26)
            for f, yy in ((f_a - 9, -0.55 + rng.uniform(0, 0.2)), (f_b + 12, 0.45 + rng.uniform(0, 0.15))):
                b.location = (x + rng.uniform(-0.02, 0.02), yy, 0.001)
                key(b, "location", f)
            key_alpha(b, [(1, 0.0), (f_a - 9, 0.0), (f_a - 4, 0.65), (f_b + 6, 0.65), (f_b + 12, 0.0)])


def gfx_object(name, bm, col, M, color):
    ob = mesh_object(name, bm, col, [M.gfx], color=color, recalc=False)
    ob.visible_shadow = False
    return ob


# --------------------------------------------------------------------------
# Sea state: one formula drives Python (placement, boat motion, checks) and geometry nodes (the sea)
# --------------------------------------------------------------------------
WAVES = [  # wavelength (m), amplitude (m), travel direction (degrees from +X), phase
    (64.0, 0.42, 70.0, 0.3),
    (36.0, 0.20, 38.0, 1.9),
    (13.0, 0.070, 112.0, 4.1),
    (7.5, 0.035, 18.0, 2.6),
]


def waves(B, x, y, t):
    """Deep-water swell: sum of travelling sines with the deep-water dispersion relation."""
    h = 0.0
    for lam, amp, deg, ph in WAVES:
        k = 2.0 * math.pi / lam
        w = math.sqrt(9.81 * k)
        kx, ky = k * math.cos(math.radians(deg)), k * math.sin(math.radians(deg))
        arg = B.add(B.mul(x, kx), B.mul(y, ky), B.mul(t, -w), ph)
        h = B.add(h, B.mul(B.sin(arg), amp))
    return h


PY = PyB()


def wave_z(x, y, t):
    return waves(PY, x, y, t)


def weed_x(y, B=None):
    """Centre line of the sargassum weed line (it drifts with the water, so it is fixed in this frame).
    The long-rigger bait runs about 2 m outside its edge, the way a weed line is usually trolled."""
    B = B or PY
    return B.add(-17.0, B.mul(B.sin(B.add(B.div(y, 38.0), 0.6)), 1.8), B.mul(B.sin(B.div(y, 13.0)), 0.8))


def weed_half_width(y, B=None):
    B = B or PY
    return B.add(2.0, B.mul(B.sin(B.add(B.div(y, 21.0), 1.3)), 0.8))


# --------------------------------------------------------------------------
# Boat motion: trolling speed until the strike, then easing off for the fight
# --------------------------------------------------------------------------
def boat_speed(f):
    if f <= STRIKE:
        return TROLL_SPEED
    v = TROLL_SPEED + (0.9 - TROLL_SPEED) * smoothstep(STRIKE, STRIKE + 150, f)
    return v + (0.55 - 0.9) * smoothstep(STRIKE + 150, BEAT_D, f)


BOAT_Y = {}
_y = 0.0
for _f in range(-60, FRAME_END + 61):
    BOAT_Y[_f] = _y
    _y += boat_speed(_f) / FPS


def _boat_attitude():
    """Heave, pitch and roll from the waves under the hull, low-passed because a 13 m hull averages them."""
    raw = {}
    for f in range(-60, FRAME_END + 61):
        t, y = frame_time(f), BOAT_Y[f]
        bow, stern = wave_z(0.0, y + 5.5, t), wave_z(0.0, y - 5.5, t)
        port, stb = wave_z(-2.1, y, t), wave_z(2.1, y, t)
        mid = wave_z(0.0, y, t)
        heave = 0.55 * (bow + stern + port + stb + mid) / 5.0
        pitch = 0.7 * math.atan2(bow - stern, 11.0) + math.radians(1.6) * min(1.0, boat_speed(f) / TROLL_SPEED)
        roll = 0.6 * math.atan2(port - stb, 4.2)
        raw[f] = (heave, pitch, roll)
    out = {}
    k = 7
    for f in range(-60 + k, FRAME_END + 61 - k):
        acc = [0.0, 0.0, 0.0]
        for d in range(-k, k + 1):
            for i in range(3):
                acc[i] += raw[f + d][i]
        out[f] = tuple(a / (2 * k + 1) for a in acc)
    return out


BOAT_ATT = _boat_attitude()


def boat_matrix(f):
    f = max(-50, min(FRAME_END + 50, f))
    heave, pitch, roll = BOAT_ATT[f]
    return Matrix.LocRotScale(Vector((0.0, BOAT_Y[f], heave)), Euler((pitch, roll, 0.0)), None)


def boat_to_world(f, p, rigid=True):
    """Boat-space point to world. rigid=False ignores heave/pitch/roll (things towed behind the boat)."""
    if rigid:
        return boat_matrix(f) @ Vector(p)
    return Vector((p[0], p[1] + BOAT_Y[max(-60, min(FRAME_END + 60, f))], p[2]))


# --------------------------------------------------------------------------
# The trolling spread: two outrigger baits wide, two flat-line baits close
# --------------------------------------------------------------------------
SPREAD = {   # name: (x, y behind the transom in boat space, kind, skip period s, phase)
    "LongRigger": (-9.6, -43.0, "ballyhoo", 2.4, 0.10),     # weed-line side: this one gets bit
    "ShortRigger": (9.0, -36.0, "skirt", 2.1, 0.55),
    "LeftFlat": (-3.2, -30.0, "skirt", 1.9, 0.30),
    "RightFlat": (3.2, -26.0, "ballyhoo", 2.6, 0.80),
}


def lure_skip(t, period, phase):
    """Height offset from the water surface: rides and skips on top, dives under now and then."""
    s = (t / period + phase) % 1.0
    if s < 0.18:                                         # out on top for a skip
        return -0.06 + 0.16 * math.sin(math.pi * s / 0.18)
    u = (s - 0.18) / 0.82                                # swimming just under, dipping deeper mid-cycle
    return -0.06 - 0.28 * math.sin(math.pi * u) ** 1.5


STRIKE_FISH_LEAD = 18   # frames before the strike when the bull locks on to the bait


def lure_world(name, f):
    x, y, _kind, per, ph = SPREAD[name]
    t = frame_time(f)
    wx, wy = x, y + BOAT_Y[f]
    sway = 0.35 * math.sin(t * 0.9 + ph * 7.0)            # lines swing a little in the wake
    return Vector((wx + sway, wy, wave_z(wx + sway, wy, t) + lure_skip(t, per, ph)))


# --------------------------------------------------------------------------
# Choreography helpers
# --------------------------------------------------------------------------
def spline(keys):
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
    for f in range(1, FRAME_END + 1):
        if f < fr[0]:
            out[f] = P[0].copy()
        elif f > fr[-1]:
            out[f] = P[-1].copy()
    return out


def wobble(f, seed, amp=(0.6, 0.5, 0.25), rate=0.035):
    r = random.Random(seed)
    ph = [r.uniform(0, 6.28) for _ in range(6)]
    t = f * rate
    return Vector((amp[0] * (math.sin(t + ph[0]) + 0.5 * math.sin(2.3 * t + ph[1])) / 1.5,
                   amp[1] * (math.sin(0.8 * t + ph[2]) + 0.5 * math.sin(1.9 * t + ph[3])) / 1.5,
                   amp[2] * (math.sin(1.3 * t + ph[4]) + 0.5 * math.sin(2.9 * t + ph[5])) / 1.5))


def weed_local(f, y_local, dx=0.0):
    """Boat-space x of the weed line beside a boat-space y at frame f."""
    return weed_x(y_local + BOAT_Y[f]) + dx


# --------------------------------------------------------------------------
# Mahi school: positions in boat space, turned into world space per frame
# --------------------------------------------------------------------------
BULL_LEN = 1.25
JUMP = (1468, 1496)      # the bull clears the water once near the transom


def bull_local_keys():
    return [
        (300, (None, -38.0, -12.5)), (430, (None, -39.0, -9.0)), (560, (-10.4, -41.0, -5.2)),
        (690, (-7.5, -42.0, -2.2)), (810, (-7.0, -44.0, -1.6)), (900, (-9.2, -50.0, -1.25)),
        (965, (-9.5, -46.6, -0.62)),
        # strike at STRIKE: the mouth meets the bait (fixed up below)
        (STRIKE + 30, (-11.0, -42.5, -0.9)), (STRIKE + 90, (-14.5, -47.0, -1.5)), (1200, (-14.0, -46.0, -1.3)),
        (1300, (-9.0, -33.0, -0.9)), (1350, (-7.0, -26.0, -0.8)), (1420, (-2.6, -11.8, -0.75)),
        # the jump: off the port quarter, leaping away from the boat, in the open water left of the hull
        (JUMP[0], (-2.5, -7.9, -0.35)), (JUMP[1], (-3.4, -7.4, -0.4)), (1560, (-2.9, -8.4, -0.8)),
        (1650, (-1.9, -8.9, -0.6)), (1740, (-1.3, -8.9, -0.42)), (1800, (-1.1, -8.6, -0.38)),
    ]


SCHOOL_KEYS = [
    (300, (None, -36.0, -13.0)), (450, (None, -40.0, -9.0)), (560, (-10.5, -42.0, -5.5)),
    (680, (-7.0, -40.0, -2.6)), (800, (-5.0, -36.0, -1.9)), (900, (-7.5, -42.0, -1.7)),
    (1010, (-9.0, -44.0, -1.6)), (1100, (-13.0, -49.0, -2.1)), (1300, (-8.5, -32.0, -2.4)),
    (1400, (-5.0, -20.0, -3.6)), (1500, (-4.0, -15.0, -6.0)), (1600, (-3.0, -13.0, -9.5)),
    (1800, (-2.0, -10.0, -12.0)),
]


def _resolve(keys):
    """Keys with x = None sit under the weed line."""
    return [(f, (weed_local(f, p[1], 0.8) if p[0] is None else p[0], p[1], p[2])) for f, p in keys]


def fish_paths():
    """{name: (length, {frame: world position}, {frame: forward vector})} for the bull and his school."""
    paths = {}
    bull_loc = spline(_resolve(bull_local_keys()))
    school_loc = spline(_resolve(SCHOOL_KEYS))
    world = {}
    for f in range(1, FRAME_END + 1):
        p = bull_loc[f]
        world[f] = Vector((p.x, p.y + BOAT_Y[f], p.z))
    # The jump: a ballistic arc out of the water near the transom
    j0, j1 = JUMP
    for f in range(j0, j1 + 1):
        u = (f - j0) / (j1 - j0)
        world[f].z += 1.75 * 4 * u * (1 - u)
    # Lock the mouth onto the bait for the strike
    fwd = _forwards(world)
    for f in range(STRIKE - STRIKE_FISH_LEAD, STRIKE + 1):
        w = smoothstep(STRIKE - STRIKE_FISH_LEAD, STRIKE - 2, f)
        target = lure_world("LongRigger", f) - fwd[f] * (0.46 * BULL_LEN) + Vector((0, 0, -0.05))
        world[f] = world[f].lerp(target, w)
    for f in range(STRIKE + 1, STRIKE + 12):            # ease back onto the escape path
        w = smoothstep(STRIKE + 11, STRIKE + 1, f)
        target = lure_world("LongRigger", STRIKE) - fwd[STRIKE] * (0.46 * BULL_LEN) + Vector((0, 0, -0.05))
        world[f] = world[f].lerp(target, w)
    paths["Bull"] = (BULL_LEN, world, _forwards(world))
    rng = random.Random(7)
    for i in range(8):
        length = rng.uniform(0.75, 1.05)
        off0 = Vector((rng.uniform(-3.0, 3.0), rng.uniform(-4.0, 4.0), rng.uniform(-1.4, 1.2)))
        w = {}
        for f in range(1, FRAME_END + 1):
            base = school_loc[f]
            if f > STRIKE:          # after the hook-up the school shadows the hooked fish for a while
                k = smoothstep(STRIKE, STRIKE + 60, f) * (1 - smoothstep(1380, 1520, f))
                base = base.lerp(bull_loc[f] + Vector((0, -2.0, -1.2)), k)
            spread = 0.6 + 0.4 * smoothstep(560, 700, f)
            p = base + off0 * spread + wobble(f, 100 + i)
            p.z = min(p.z, -0.55)                 # school fish stay under the surface
            w[f] = Vector((p.x, p.y + BOAT_Y[f], p.z))
        paths[f"Mahi{i}"] = (length, w, _forwards(w))
    return paths


def _forwards(world):
    keys = sorted(world)
    vel = {}
    for f in keys:
        a, b = world[max(keys[0], f - 3)], world[min(keys[-1], f + 3)]
        v = b - a
        vel[f] = v.normalized() if v.length > 1e-4 else Vector((0, 1, 0))
    sm = dict(zip(keys, smooth_series([vel[f] for f in keys], 5)))
    return {f: (sm[f].normalized() if sm[f].length > 1e-4 else Vector((0, 1, 0))) for f in keys}


def avoid_camera(paths, frames, radius=1.4):
    """Fish swerve around the lens: a smoothed push keeps every body at least `radius` from the camera.
    The bull is left alone around the strike (his mouth is locked onto the bait there)."""
    for name, (length, world, _fwd) in list(paths.items()):
        push = {}
        for f in range(1, FRAME_END + 1):
            if name == "Bull" and STRIKE - 40 <= f <= STRIKE + 15:
                push[f] = Vector()
                continue
            d = world[f] - frames[f][0]
            need = radius + 0.5 * length - d.length
            push[f] = d.normalized() * need if need > 0 and d.length > 1e-4 else Vector()
        keys = list(range(1, FRAME_END + 1))
        sm = {}
        for f in keys:                      # widen each push so the swerve starts early and ends late
            acc, wsum = Vector(), 0.0
            for k in range(-12, 13):
                g = min(FRAME_END, max(1, f + k))
                w = math.exp(-0.5 * (k / 6.0) ** 2)
                acc += push[g] * w
                wsum += w
            sm[f] = acc / wsum * 2.2
        new = {f: world[f] + sm[f] for f in keys}
        paths[name] = (length, new, _forwards(new))
    return paths


def bait_world(name, f, paths):
    """The long-rigger ballyhoo rides in the bull's mouth after the strike."""
    if name == "LongRigger" and f > STRIKE:
        _l, w, fw = paths["Bull"]
        return w[f] + fw[f] * (0.46 * BULL_LEN) + Vector((0, 0, 0.05))
    return lure_world(name, f)


# --------------------------------------------------------------------------
# Shading
# --------------------------------------------------------------------------
HAZE = srgb(206, 222, 236)


def scene_attr(nt, name):
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_type = "VIEW_LAYER"
    a.attribute_name = name
    return a.outputs["Factor"]


def uw_color(nt, z_dir):
    """Underwater ambient colour by view direction: bright towards the surface, inky below."""
    return color_ramp(nt, nmath(nt, "MULTIPLY_ADD", z_dir, 0.5, 0.5, clamp=True),
                      [(0.0, UW_DOWN), (0.42, srgb(5, 40, 90)), (0.55, UW_MID), (0.78, srgb(30, 118, 178)),
                       (1.0, UW_UP)])


def build_atmos_group():
    """Air haze and depth fade when the lens is above water; blue fog when it is below.
    scene["cam_under"] (keyed per frame) says which side of the surface the lens is on."""
    ng, gi, go = new_group("Atmosphere", "ShaderNodeTree", [("Shader", "NodeSocketShader")],
                           [("Shader", "NodeSocketShader")])
    geo = ng.nodes.new("ShaderNodeNewGeometry")
    dist = ng.nodes.new("ShaderNodeCameraData").outputs["View Distance"]
    inc = separate(ng, geo.outputs["Incoming"])
    z = separate(ng, geo.outputs["Position"])["Z"]
    under = scene_attr(ng, "cam_under")
    above = nmath(ng, "SUBTRACT", 1.0, under)
    haze = nmath(ng, "MULTIPLY", above,
                 nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(ng, "MULTIPLY", dist, -1.0 / AIR_HAZE_DIST))))
    depth = nmath(ng, "MULTIPLY", above,
                  nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT",
                                                    nmath(ng, "MULTIPLY", nmath(ng, "MINIMUM", z, 0.0), 1.0 / DEPTH_FOG))))
    fog = nmath(ng, "MULTIPLY", under,
                nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(ng, "MULTIPLY", dist, -1.0 / UW_FOG_DIST))))
    em_h = ng.nodes.new("ShaderNodeEmission")
    em_h.inputs["Color"].default_value = (*HAZE, 1.0)
    em_d = ng.nodes.new("ShaderNodeEmission")
    em_d.inputs["Color"].default_value = (*srgb(8, 52, 112), 1.0)
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


def build_world():
    w = bpy.data.worlds.new("Sky")
    bpy.context.scene.world = w
    nt = w.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.asin(SUN_DIR.z)
    sky.sun_rotation = math.atan2(SUN_DIR.x, SUN_DIR.y) * -1.0 + math.pi * 0.5 * 0.0
    sky.sun_disc = True
    sky.altitude = 20.0
    sky.air_density = 1.0
    sky.aerosol_density = 1.3
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 1.0
    lit = vmath(nt, "SCALE", sky.outputs[0], scale=SKY_STRENGTH)
    # What the camera sees directly gets a soft shoulder: the Standard view would clip the bright horizon
    # haze to flat white. The brightest channel is compressed and the colour scaled with it, so the sky
    # keeps its hue. Lighting and reflections keep the full sky.
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
    # Underwater the world is never seen directly (backdrop sphere), but keep reflections sane
    nt.links.new(bg.outputs[0], nt.nodes.new("ShaderNodeOutputWorld").inputs["Surface"])
    return sky


SKY_STRENGTH = 0.11


def build_sun(col):
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 3.4
    light.color = srgb(255, 247, 234)
    light.angle = math.radians(0.6)
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


def build_materials(atmos):
    M = Materials()
    M.atmos = atmos

    def pbr(name, color, rough=0.4, metal=0.0, coat=0.0, spec=0.5, sss=0.0, thin=0.0, alpha=None):
        def build(nt):
            b = nt.nodes.new("ShaderNodeBsdfPrincipled")
            _feed(nt, b.inputs["Base Color"], (*color, 1.0) if isinstance(color, tuple) else color)
            b.inputs["Roughness"].default_value = rough
            b.inputs["Metallic"].default_value = metal
            b.inputs["Coat Weight"].default_value = coat
            b.inputs["Coat Roughness"].default_value = 0.05
            b.inputs["Specular IOR Level"].default_value = spec
            b.inputs["Subsurface Weight"].default_value = sss
            if thin:
                b.inputs["Thin Film Thickness"].default_value = thin
            return b.outputs[0]
        return new_material(name, build, atmos)

    # ---- boat ------------------------------------------------------------
    M.hull = pbr("HullNavy", srgb(14, 30, 58), rough=0.18, coat=1.0)
    M.gelcoat = pbr("GelcoatWhite", srgb(236, 238, 236), rough=0.22, coat=0.6)
    M.bottom = pbr("Antifouling", srgb(28, 26, 32), rough=0.7)
    M.stainless = pbr("Stainless", srgb(214, 218, 222), rough=0.14, metal=1.0)
    M.rubber = pbr("BlackRubber", srgb(18, 18, 20), rough=0.55)
    M.canvas = pbr("CanvasGrey", srgb(64, 70, 78), rough=0.8)
    M.cushion = pbr("CushionWhite", srgb(226, 224, 216), rough=0.45)
    M.glass = pbr("TintedGlass", srgb(10, 16, 22), rough=0.04, spec=1.0, coat=1.0)

    def teak(nt):
        tc = nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]
        wv = nt.nodes.new("ShaderNodeTexWave")
        wv.wave_type = "BANDS"
        wv.bands_direction = "X"
        wv.inputs["Scale"].default_value = 22.0
        wv.inputs["Distortion"].default_value = 1.2
        wv.inputs["Detail"].default_value = 1.0
        nt.links.new(tc, wv.inputs["Vector"])
        planks = nmath(nt, "FRACT", nmath(nt, "MULTIPLY", separate(nt, tc)["X"], 1.0 / 0.09))
        seam = nmath(nt, "MULTIPLY", map_range(nt, planks, 0.0, 0.05, 0.2, 1.0), map_range(nt, planks, 1.0, 0.95, 0.2, 1.0))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", wv.outputs["Fac"], 0.6), srgb(168, 120, 74), srgb(194, 146, 96))
        col = vmath(nt, "SCALE", col, scale=seam)
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.55
        return b.outputs[0]
    M.teak = new_material("Teak", teak, atmos)

    # ---- tackle ----------------------------------------------------------
    M.rod = pbr("RodBlank", srgb(22, 24, 28), rough=0.2, coat=1.0)
    M.reel = pbr("ReelGold", srgb(206, 172, 112), rough=0.28, metal=1.0)
    M.spool = pbr("SpoolLine", srgb(196, 206, 150), rough=0.5)
    M.reel_dark = pbr("ReelBlack", srgb(20, 20, 22), rough=0.3, metal=0.6)
    M.lever = pbr("DragLever", srgb(200, 40, 36), rough=0.25, coat=0.8)
    M.hook = pbr("HookSteel", srgb(190, 196, 200), rough=0.2, metal=1.0)
    M.skirt_pink = pbr("SkirtPink", srgb(236, 72, 150), rough=0.25, sss=0.3, thin=380)
    M.skirt_blue = pbr("SkirtBlue", srgb(36, 120, 230), rough=0.25, sss=0.3, thin=420)
    M.lure_head = pbr("LureHead", srgb(240, 236, 220), rough=0.08, spec=1.0, coat=1.0, thin=300)

    def ballyhoo(nt):
        z = separate(nt, attr_vec(nt, "rest"))["Z"]
        col = color_ramp(nt, map_range(nt, z, -0.012, 0.012, smooth=False),
                         [(0.0, srgb(226, 230, 236)), (0.55, srgb(176, 196, 214)), (0.75, srgb(70, 104, 150)),
                          (1.0, srgb(34, 56, 96))])
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.2
        b.inputs["Metallic"].default_value = 0.45
        b.inputs["Thin Film Thickness"].default_value = 260
        return b.outputs[0]
    M.ballyhoo = new_material("Ballyhoo", ballyhoo, atmos)

    def line(nt):
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        b.inputs["Base Color"].default_value = (*srgb(214, 226, 170), 1.0)
        b.inputs["Roughness"].default_value = 0.3
        b.inputs["Emission Color"].default_value = (*srgb(214, 226, 170), 1.0)
        b.inputs["Emission Strength"].default_value = 0.12       # monofilament catching the light
        return b.outputs[0]
    M.line = new_material("FishingLine", line, atmos)

    # ---- mahi mahi -------------------------------------------------------
    def line_mask(nt, dist, width):
        return map_range(nt, nmath(nt, "ABSOLUTE", dist), width, 0.0)

    def mahi(nt):
        rest = separate(nt, attr_vec(nt, "rest"))
        zrel = attr_float(nt, "zrel")                         # 0 belly .. 1 back, along each section
        lit = scene_attr(nt, "mahi_lit")                       # excitement: brighter blue and gold
        body = color_ramp(nt, zrel, [(0.0, srgb(242, 238, 214)), (0.16, srgb(244, 226, 120)),
                                     (0.36, srgb(232, 200, 62)), (0.55, srgb(178, 182, 66)),
                                     (0.68, srgb(78, 156, 108)), (0.82, srgb(34, 116, 158)), (1.0, srgb(22, 72, 132))])
        vec = attr_vec(nt, "rest")
        noise = nt.nodes.new("ShaderNodeTexNoise")              # irregular metallic blue and green patches
        noise.inputs["Scale"].default_value = 9.0
        noise.inputs["Detail"].default_value = 3.0
        nt.links.new(vec, noise.inputs["Vector"])
        patch = nmath(nt, "MULTIPLY", map_range(nt, noise.outputs["Fac"], 0.50, 0.62), map_range(nt, zrel, 0.42, 0.72))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", patch, 0.6), body, srgb(36, 150, 168))
        vor = nt.nodes.new("ShaderNodeTexVoronoi")              # small dark spots on the sides
        vor.inputs["Scale"].default_value = 34.0
        nt.links.new(combine(nt, rest["X"], 0.0, rest["Z"]), vor.inputs["Vector"])
        side = nmath(nt, "MULTIPLY", map_range(nt, zrel, 0.14, 0.32), map_range(nt, zrel, 0.86, 0.72))
        spots = nmath(nt, "MULTIPLY", map_range(nt, vor.outputs["Distance"], 0.12, 0.05), side)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", spots, 0.85), col, mix_rgb(nt, lit, srgb(36, 60, 92), srgb(28, 112, 224)))
        # Head: the gill cover edge and the mouth line
        gx = nmath(nt, "SUBTRACT", 0.392, nmath(nt, "MULTIPLY", nmath(nt, "POWER", nmath(
            nt, "DIVIDE", nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", rest["Z"], 0.01)), 0.11), 2.0), 0.03))
        gill = nmath(nt, "MULTIPLY", line_mask(nt, nmath(nt, "SUBTRACT", rest["X"], gx), 0.0035),
                     map_range(nt, zrel, 0.06, 0.14))
        gill = nmath(nt, "MULTIPLY", gill, map_range(nt, zrel, 0.82, 0.70))
        zm = nmath(nt, "SUBTRACT", -0.004, nmath(nt, "MULTIPLY", nmath(nt, "SUBTRACT", 0.5, rest["X"]), 0.40))
        mouth = nmath(nt, "MULTIPLY", line_mask(nt, nmath(nt, "SUBTRACT", rest["Z"], zm), 0.0022),
                      map_range(nt, rest["X"], 0.452, 0.458))
        dark = nmath(nt, "MAXIMUM", nmath(nt, "MULTIPLY", gill, 0.45), nmath(nt, "MULTIPLY", mouth, 0.75))
        col = vmath(nt, "SCALE", col, scale=nmath(nt, "SUBTRACT", 1.0, dark))
        col = vmath(nt, "SCALE", col, scale=nmath(nt, "MULTIPLY_ADD", lit, 0.22, 1.0))
        scales = nt.nodes.new("ShaderNodeTexVoronoi")           # fine scales, fixed to the body as it bends
        scales.inputs["Scale"].default_value = 230.0
        nt.links.new(vec, scales.inputs["Vector"])
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.12
        bump.inputs["Distance"].default_value = 0.0006
        nt.links.new(scales.outputs["Distance"], bump.inputs["Height"])
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
        b.inputs["Roughness"].default_value = 0.24
        b.inputs["Specular IOR Level"].default_value = 0.7
        b.inputs["Coat Weight"].default_value = 0.6                # wet mucus layer
        b.inputs["Coat Roughness"].default_value = 0.08
        b.inputs["Subsurface Weight"].default_value = 0.15
        b.inputs["Subsurface Radius"].default_value = (0.6, 0.45, 0.3)
        b.inputs["Subsurface Scale"].default_value = 0.02
        b.inputs["Thin Film Thickness"].default_value = 520        # gold-to-blue flash at glancing angles
        b.inputs["Thin Film IOR"].default_value = 1.45
        return b.outputs[0]
    M.mahi = new_material("MahiSkin", mahi, atmos)

    def fin(base, margin, ray_col, n_rays):
        """Fin membrane with darker rays (finu: base 0 to margin 1, finv: across the rays)."""
        def build(nt):
            u, v = attr_float(nt, "finu"), attr_float(nt, "finv")
            col = color_ramp(nt, u, [(0.0, base), (0.6, base), (1.0, margin)])
            ray = line_mask(nt, nmath(nt, "SUBTRACT", nmath(nt, "FRACT", nmath(nt, "MULTIPLY", v, float(n_rays))), 0.5), 0.1)
            col = mix_rgb(nt, nmath(nt, "MULTIPLY", ray, 0.55), col, ray_col)
            b = nt.nodes.new("ShaderNodeBsdfPrincipled")
            nt.links.new(col, b.inputs["Base Color"])
            b.inputs["Roughness"].default_value = 0.32
            b.inputs["Coat Weight"].default_value = 0.4
            b.inputs["Subsurface Weight"].default_value = 0.45
            b.inputs["Subsurface Scale"].default_value = 0.01
            b.inputs["Thin Film Thickness"].default_value = 400
            return b.outputs[0]
        return build
    M.fin_dorsal = new_material("MahiDorsalFin", fin(srgb(18, 56, 128), srgb(44, 112, 200), srgb(10, 30, 76), 60), atmos)
    M.fin_tail = new_material("MahiTail", fin(srgb(206, 186, 76), srgb(96, 124, 128), srgb(150, 130, 50), 34), atmos)
    M.fin_small = new_material("MahiFins", fin(srgb(222, 194, 84), srgb(236, 218, 140), srgb(176, 146, 56), 18), atmos)

    def eye(nt):
        rest = separate(nt, attr_vec(nt, "rest"))
        d = vmath(nt, "LENGTH", combine(nt, nmath(nt, "SUBTRACT", rest["X"], MAHI_EYE[0]), 0.0,
                                        nmath(nt, "SUBTRACT", rest["Z"], MAHI_EYE[1])))
        col = color_ramp(nt, d, [(0.0, srgb(6, 6, 8)), (0.0062, srgb(6, 6, 8)), (0.0072, srgb(176, 134, 52)),
                                 (0.0105, srgb(120, 92, 40)), (0.0125, srgb(30, 34, 36)), (1.0, srgb(30, 34, 36))])
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.1
        b.inputs["Coat Weight"].default_value = 1.0
        b.inputs["Coat Roughness"].default_value = 0.02
        return b.outputs[0]
    M.eye = new_material("MahiEye", eye, atmos)
    M.bait = pbr("Baitfish", srgb(190, 204, 214), rough=0.2, metal=0.6, thin=300)

    # ---- sargassum -------------------------------------------------------
    def weed(nt):
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 3.0
        nt.links.new(nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"], n.inputs["Vector"])
        v = nmath(nt, "ADD", nmath(nt, "MULTIPLY", oi.outputs["Random"], 0.6), nmath(nt, "MULTIPLY", n.outputs["Fac"], 0.4))
        col = color_ramp(nt, v, [(0.0, srgb(120, 92, 26)), (0.5, srgb(176, 136, 44)), (1.0, srgb(206, 168, 70))])
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.42
        b.inputs["Subsurface Weight"].default_value = 0.5      # light glows through the fronds from below
        b.inputs["Subsurface Radius"].default_value = (0.8, 0.6, 0.2)
        b.inputs["Subsurface Scale"].default_value = 0.01
        b.inputs["Coat Weight"].default_value = 0.3
        return b.outputs[0]
    M.weed = new_material("Sargassum", weed, atmos)

    def weed_mat(nt):
        """Far weed line: patchy floating mat, cut out with a noise mask."""
        p = nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"]
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 0.55
        n.inputs["Detail"].default_value = 6.0
        n.inputs["Roughness"].default_value = 0.65
        nt.links.new(p, n.inputs["Vector"])
        across = attr_float(nt, "across")                 # 0 at the centre line, 1 at the edge
        cov = nmath(nt, "SUBTRACT", n.outputs["Fac"], nmath(nt, "MULTIPLY_ADD", across, 0.34, 0.4))
        alpha = map_range(nt, cov, 0.0, 0.04)
        col = color_ramp(nt, n.outputs["Fac"], [(0.3, srgb(98, 72, 20)), (0.6, srgb(156, 118, 40)),
                                                (0.8, srgb(186, 150, 68))])
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.5
        b.inputs["Subsurface Weight"].default_value = 0.4
        b.inputs["Subsurface Scale"].default_value = 0.02
        g = nt.nodes.new("ShaderNodeGroup")
        g.node_tree = atmos
        nt.links.new(b.outputs[0], g.inputs["Shader"])
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(alpha, mix.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        nt.links.new(g.outputs["Shader"], mix.inputs[2])
        return mix.outputs[0]
    M.weed_mat = new_material("SargassumMat", weed_mat)
    M.weed_mat.use_transparent_shadow = True

    # ---- water -----------------------------------------------------------
    M.water = build_water_material(atmos)

    def backdrop(nt):
        inc = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Incoming"])
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(uw_color(nt, nmath(nt, "MULTIPLY", inc["Z"], -1.0)), em.inputs["Color"])
        return em.outputs[0]
    M.backdrop = new_material("UnderwaterBackdrop", backdrop)

    def deep_floor(nt):
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(8, 46, 104), 1.0)
        return em.outputs[0]
    M.deep_floor = new_material("DeepWaterBelow", deep_floor)

    def beam(nt):
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.5
        core = nmath(nt, "POWER", nmath(nt, "SUBTRACT", 1.0, lw.outputs["Facing"], clamp=True), 1.6)
        z = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"])["Z"]
        near = map_range(nt, nt.nodes.new("ShaderNodeCameraData").outputs["View Distance"], 2.0, 7.0)
        fade = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", map_range(nt, z, -0.1, -1.2),
                                           nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", z, 1.0 / 6.0))), near)
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        t = frame_value(nt, "frame/24")
        flick = nmath(nt, "MULTIPLY_ADD", nmath(nt, "SINE", nmath(nt, "MULTIPLY_ADD", t, 1.3,
                                                                  nmath(nt, "MULTIPLY", oi.outputs["Random"], 31.0))),
                      0.35, 0.65)
        k = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", core, fade), nmath(nt, "MULTIPLY", flick, scene_attr(nt, "cam_under")))
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(170, 236, 255), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", k, BEAM_GAIN), em.inputs["Strength"])
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], add.inputs[1])
        return add.outputs[0]
    M.beam = new_material("SunShaft", beam, blended=True)

    def snow(nt):
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(200, 236, 240), 1.0)
        em.inputs["Strength"].default_value = 0.8
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(nmath(nt, "MULTIPLY", scene_attr(nt, "cam_under"), 0.55), mix.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    M.snow = new_material("MarineSnow", snow, blended=True)

    def bubbles(nt):
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        rim = nmath(nt, "POWER", lw.outputs["Facing"], 1.5)
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(230, 248, 255), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY_ADD", rim, 1.4, 0.15), em.inputs["Strength"])
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(nmath(nt, "MULTIPLY_ADD", rim, 0.6, 0.15), mix.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    M.bubbles = new_material("Bubbles", bubbles, blended=True)

    def spray(nt):
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(240, 248, 250), 1.0)
        em.inputs["Strength"].default_value = 1.2
        lw = nt.nodes.new("ShaderNodeLayerWeight")              # soft droplet edges
        core = nmath(nt, "POWER", nmath(nt, "SUBTRACT", 1.0, lw.outputs["Facing"]), 1.2)
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(nmath(nt, "MULTIPLY", core, 0.8), mix.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    M.spray = new_material("Spray", spray, blended=True)

    # ---- people ----------------------------------------------------------
    M.shirt = pbr("PerformanceShirt", srgb(150, 196, 226), rough=0.7)
    M.shorts = pbr("Shorts", srgb(168, 150, 116), rough=0.8)
    M.skin = pbr("Skin", srgb(206, 150, 112), rough=0.5, sss=0.25)
    M.cap = pbr("Cap", srgb(28, 40, 60), rough=0.7)
    M.shades = pbr("Sunglasses", srgb(8, 8, 10), rough=0.05, spec=1.0, coat=1.0)

    # ---- overlays --------------------------------------------------------
    def scrim(band):
        def build(nt):
            oi = nt.nodes.new("ShaderNodeObjectInfo")
            y = separate(nt, nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"])["Y"]
            if band:
                prof = nmath(nt, "SUBTRACT", 1.0, nmath(nt, "ABSOLUTE", nmath(nt, "MULTIPLY_ADD", y, 2.0, -1.0)))
            else:
                prof = nmath(nt, "POWER", y, 1.6)
            tr = nt.nodes.new("ShaderNodeBsdfTransparent")
            em = nt.nodes.new("ShaderNodeEmission")
            em.inputs["Color"].default_value = (0.0, 0.01, 0.025, 1.0)
            mix = nt.nodes.new("ShaderNodeMixShader")
            nt.links.new(nmath(nt, "MULTIPLY", oi.outputs["Alpha"], prof), mix.inputs[0])
            nt.links.new(tr.outputs[0], mix.inputs[1])
            nt.links.new(em.outputs[0], mix.inputs[2])
            return mix.outputs[0]
        return build
    M.scrim = new_material("HudScrimTop", scrim(False), blended=True)
    M.scrim_band = new_material("HudScrimBand", scrim(True), blended=True)
    M.gfx = new_material("Graphic", emissive_alpha(), blended=True)
    M.hud = new_material("HudColor", emissive_alpha(), blended=True)
    M.hud_white = new_material("HudWhite", emissive_alpha(WHITE), blended=True)
    M.hud_accent = new_material("HudAccent", emissive_alpha(ORANGE), blended=True)
    M.hud_shadow = new_material("HudShadow", emissive_alpha(tuple(c * 0.25 for c in INK), 1.0), blended=True)
    M.highlight = new_material("GearHighlight", emissive_alpha(), blended=True)
    for m in (M.gfx, M.hud, M.hud_white, M.hud_accent, M.hud_shadow, M.highlight):
        m.use_transparency_overlap = True
    # Overlays and outlines use dithered transparency: blended surfaces write no motion vectors, so motion
    # blur would smear them with the movement of whatever is behind them. Dithered ones keep their own
    # (zero, camera-locked) motion and resolve to smooth transparency over the render samples.
    for m in (M.gfx, M.hud, M.hud_white, M.hud_accent, M.hud_shadow, M.highlight):
        m.surface_render_method = "DITHERED"       # (the soft scrims stay blended: blur cannot hurt a gradient)
    for m in (M.beam, M.snow, M.bubbles, M.spray, M.highlight, M.gfx, M.hud, M.hud_white, M.hud_accent, M.hud_shadow,
              M.scrim, M.scrim_band):
        m.use_transparent_shadow = True
    return M


BEAM_GAIN = 0.06


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
    """Open-ocean surface: deep blue with sun glitter and sky reflection, a turquoise wake and foam
    behind the boat (masked in boat space via the WakeRef empty), and from below a bright Snell's
    window ringed by total internal reflection."""
    m = bpy.data.materials.new("Ocean")
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    pos = separate(nt, geo.outputs["Position"])
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = bpy.data.objects.get("WakeRef")
    bp = separate(nt, tc.outputs["Object"])                 # boat space (the empty rides with the boat, no tilt)
    wake_on = scene_attr(nt, "wake_gain")                   # follows the boat speed
    d = nmath(nt, "SUBTRACT", nmath(nt, "MULTIPLY", bp["Y"], -1.0), 6.3)        # metres behind the transom
    behind = map_range(nt, d, -1.0, 1.5)
    ax = nmath(nt, "ABSOLUTE", bp["X"])
    # Turbulent centre wake from the props, widening and fading with distance
    width = nmath(nt, "MULTIPLY_ADD", nmath(nt, "MAXIMUM", d, 0.0), 0.05, 1.3)
    core = nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", nmath(nt, "POWER", nmath(nt, "DIVIDE", ax, width), 2.0), -1.0))
    core = nmath(nt, "MULTIPLY", core, nmath(nt, "MULTIPLY", behind,
                                             nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", d, -1.0 / 55.0))))
    # Kelvin wake arms at 19.47 degrees
    arm_d = nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", ax, nmath(nt, "MULTIPLY", d, 0.3535)))
    arm_w = nmath(nt, "MULTIPLY_ADD", nmath(nt, "MAXIMUM", d, 0.0), 0.035, 0.7)
    arms = nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", nmath(nt, "POWER", nmath(nt, "DIVIDE", arm_d, arm_w), 2.0), -1.0))
    arms = nmath(nt, "MULTIPLY", arms, nmath(nt, "MULTIPLY", behind,
                                             nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", d, -1.0 / 110.0))))
    chev = nmath(nt, "MULTIPLY_ADD", nmath(nt, "SINE", nmath(nt, "MULTIPLY", d, 1.6)), 0.3, 0.7)
    arms = nmath(nt, "MULTIPLY", arms, chev)
    # Streaky foam that stays where the boat left it (world-space noise, stretched along the track)
    fn = nt.nodes.new("ShaderNodeTexNoise")
    fn.inputs["Scale"].default_value = 1.0
    fn.inputs["Detail"].default_value = 8.0
    fn.inputs["Roughness"].default_value = 0.6
    nt.links.new(combine(nt, nmath(nt, "MULTIPLY", pos["X"], 1.6), nmath(nt, "MULTIPLY", pos["Y"], 0.22), 0.0),
                 fn.inputs["Vector"])
    streak = map_range(nt, fn.outputs["Fac"], 0.52, 0.72)
    wake = nmath(nt, "MULTIPLY", wake_on, nmath(nt, "MAXIMUM", core, nmath(nt, "MULTIPLY", arms, 0.35)))
    near = nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", nmath(nt, "MAXIMUM", d, 0.0), -1.0 / 14.0))   # white water right behind the transom
    foam = nmath(nt, "MULTIPLY", wake, nmath(nt, "MULTIPLY_ADD", streak, 0.55, nmath(nt, "MULTIPLY", near, 0.6)), clamp=True)
    # Sparse whitecaps on the open sea
    wc = nt.nodes.new("ShaderNodeTexNoise")
    wc.noise_dimensions = "4D"
    wc.inputs["Scale"].default_value = 0.08
    wc.inputs["Detail"].default_value = 4.0
    nt.links.new(geo.outputs["Position"], wc.inputs["Vector"])
    add_frame_driver(wc.inputs["W"], "default_value", "frame*0.01")
    caps = map_range(nt, wc.outputs["Fac"], 0.77, 0.80)
    foam = nmath(nt, "MAXIMUM", foam, nmath(nt, "MULTIPLY", caps, 0.35))
    # Ripples: two scrolling samples of the downloaded water normal map
    def ripple(scale, dx, dy):
        mp = nt.nodes.new("ShaderNodeMapping")
        nt.links.new(combine(nt, pos["X"], pos["Y"], 0.0), mp.inputs["Vector"])
        mp.inputs["Scale"].default_value = (1.0 / scale, 1.0 / scale, 1.0)
        add_frame_driver(mp.inputs["Location"], "default_value", f"frame*{dx}", index=0)
        add_frame_driver(mp.inputs["Location"], "default_value", f"frame*{dy}", index=1)
        return image_node(nt, asset("babylon", "waterbump.png"), True, mp.outputs[0]).outputs["Color"]
    rip = separate(nt, vmath(nt, "SUBTRACT", mix_rgb(nt, 0.5, ripple(6.0, 0.0012, 0.0021),
                                                      ripple(2.1, -0.0017, 0.0011)), (0.5, 0.5, 0.5)))
    tilt = nmath(nt, "MULTIPLY", nmath(nt, "SUBTRACT", 1.0, foam), 0.55)
    normal = vmath(nt, "NORMALIZE", vmath(nt, "ADD", geo.outputs["Normal"],
                                          combine(nt, nmath(nt, "MULTIPLY", rip["X"], tilt),
                                                  nmath(nt, "MULTIPLY", rip["Y"], tilt), 0.0)))
    # Above: deep blue that turns turquoise where the wake has stirred air into the water
    deep = srgb(6, 40, 96)
    base = mix_rgb(nt, nmath(nt, "MULTIPLY", wake, 1.1), deep, srgb(30, 150, 176))
    base = mix_rgb(nt, foam, base, srgb(236, 244, 246))
    top = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(base, top.inputs["Base Color"])
    nt.links.new(normal, top.inputs["Normal"])
    nt.links.new(map_range(nt, foam, 0.0, 1.0, 0.03, 0.6, smooth=False), top.inputs["Roughness"])
    top.inputs["IOR"].default_value = 1.333
    top.inputs["Specular IOR Level"].default_value = 0.5
    nt.links.new(nmath(nt, "MULTIPLY", nmath(nt, "SUBTRACT", 1.0, foam), 0.82), top.inputs["Transmission Weight"])
    nt.links.new(mix_rgb(nt, nmath(nt, "MULTIPLY", wake, 1.2), srgb(4, 30, 72), srgb(30, 140, 160)),
                 top.inputs["Emission Color"])
    top.inputs["Emission Strength"].default_value = 0.25          # light scattered back out of the water
    g = nt.nodes.new("ShaderNodeGroup")
    g.node_tree = atmos
    nt.links.new(top.outputs[0], g.inputs["Shader"])
    # Below: Snell's window (48.6 degrees) and total internal reflection around it
    n_under = vmath(nt, "NORMALIZE", vmath(nt, "ADD", normal, combine(nt, nmath(nt, "MULTIPLY", rip["X"], 0.6),
                                                                   nmath(nt, "MULTIPLY", rip["Y"], 0.6), 0.0)))
    cos_t = nmath(nt, "ABSOLUTE", vmath(nt, "DOT_PRODUCT", geo.outputs["Incoming"], n_under))
    window = map_range(nt, cos_t, 0.56, 0.76)
    ring = nmath(nt, "MULTIPLY", map_range(nt, cos_t, 0.55, 0.64), map_range(nt, cos_t, 0.74, 0.64))
    sky_in = vmath(nt, "SCALE", srgb(120, 196, 228), scale=nmath(nt, "MULTIPLY_ADD", rip["X"], 1.2, 0.75))
    sky_in = mix_rgb(nt, nmath(nt, "MULTIPLY", foam, 0.9), sky_in, srgb(250, 252, 252))
    tir = vmath(nt, "SCALE", srgb(8, 54, 104), scale=nmath(nt, "MULTIPLY_ADD", rip["Y"], 0.5, 0.85))
    tir = mix_rgb(nt, nmath(nt, "MULTIPLY", foam, 0.8), tir, srgb(170, 210, 220))
    below = mix_rgb(nt, window, tir, sky_in)
    # The floating weed mat seen from below: the same coverage pattern as SargassumMat, dark gold
    B = NodeB(nt)
    wn = nt.nodes.new("ShaderNodeTexNoise")
    wn.inputs["Scale"].default_value = 0.55
    wn.inputs["Detail"].default_value = 6.0
    wn.inputs["Roughness"].default_value = 0.65
    nt.links.new(geo.outputs["Position"], wn.inputs["Vector"])
    across = nmath(nt, "DIVIDE", nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", pos["X"], weed_x(pos["Y"], B))),
                   nmath(nt, "ADD", weed_half_width(pos["Y"], B), 0.8))
    cov = nmath(nt, "SUBTRACT", wn.outputs["Fac"], nmath(nt, "MULTIPLY_ADD", across, 0.34, 0.4))
    weed_cov = nmath(nt, "MULTIPLY", map_range(nt, cov, 0.0, 0.04), map_range(nt, across, 1.05, 0.95))
    below = mix_rgb(nt, nmath(nt, "MULTIPLY", weed_cov, 0.9), below,
                    vmath(nt, "SCALE", srgb(150, 118, 40), scale=nmath(nt, "MULTIPLY_ADD", window, 0.7, 0.25)))
    below = vmath(nt, "ADD", below, vmath(nt, "SCALE", srgb(200, 240, 255), scale=nmath(nt, "MULTIPLY", ring, 0.35)))
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(below, em.inputs["Color"])
    em.inputs["Strength"].default_value = 1.0
    gu = nt.nodes.new("ShaderNodeGroup")
    gu.node_tree = atmos
    nt.links.new(em.outputs[0], gu.inputs["Shader"])
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(geo.outputs["Backfacing"], mix.inputs[0])
    nt.links.new(g.outputs["Shader"], mix.inputs[1])
    nt.links.new(gu.outputs["Shader"], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.use_backface_culling = False
    m.use_raytrace_refraction = True
    m.thickness_mode = "SLAB"
    return m


# --------------------------------------------------------------------------
# The sportfisher (boat space: origin on the waterline amidships, +Y forward)
# --------------------------------------------------------------------------
TRANSOM_Y, STEM_Y = -6.35, 7.25
GUNWALE_Z, COCKPIT_Z = 1.30, 0.62
RIGGER_BASE = {-1: Vector((-1.62, 0.55, 4.66)), 1: Vector((1.62, 0.55, 4.66))}
RIGGER_DIR = {s: Vector((s * 0.723, -0.507, 0.469)).normalized() for s in (-1, 1)}
RIGGER_LEN = 8.5
HOLDERS = {   # rod holder: (butt position, rod direction) in boat space
    "LongRigger": (Vector((-1.98, -3.15, GUNWALE_Z + 0.02)), Vector((-0.30, -0.42, 0.86)).normalized()),
    "ShortRigger": (Vector((1.98, -3.15, GUNWALE_Z + 0.02)), Vector((0.30, -0.42, 0.86)).normalized()),
    "LeftFlat": (Vector((-1.72, -6.05, GUNWALE_Z + 0.02)), Vector((-0.25, -0.62, 0.74)).normalized()),
    "RightFlat": (Vector((1.72, -6.05, GUNWALE_Z + 0.02)), Vector((0.25, -0.62, 0.74)).normalized()),
}
ROD_LEN = 2.05


def hull_station(y):
    """Half section at y: list of (x, z) from the keel up to the sheer."""
    u = (y - TRANSOM_Y) / (STEM_Y - TRANSOM_Y)
    b = 2.16 + 0.14 * smoothstep(0.0, 0.42, u) - 2.30 * smoothstep(0.52, 1.0, u) ** 1.35
    b = max(b, 0.0)
    keel = -0.86 + 1.25 * smoothstep(0.62, 1.0, u) ** 1.6
    beta = math.radians(16 + 18 * smoothstep(0.35, 0.8, u) + 20 * smoothstep(0.8, 1.0, u))
    sheer = GUNWALE_Z + 0.16 * smoothstep(0.30, 0.45, u) + 0.62 * smoothstep(0.45, 1.0, u) ** 1.3
    xc = 0.80 * b
    zc = keel + math.tan(beta) * xc
    pts = [(0.0, keel), (0.45 * xc, keel + math.tan(beta) * 0.45 * xc), (xc, zc), (xc + 0.05 * b, zc + 0.03),
           (0.96 * b, zc + 0.55 * (sheer - zc)), (b, sheer)]
    return pts


def build_boat(col, M):
    root = empty_object("Boat", col, size=1.0)
    root.rotation_mode = "XYZ"
    parts = []

    def part(name, bm, mats, smooth=False, parent=root):
        ob = mesh_object(name, bm, col, mats, parent=parent)
        if smooth:
            for p in ob.data.polygons:
                p.use_smooth = True
        parts.append(ob)
        return ob

    # ---- hull: loft of the half sections, mirrored -------------------------------------------
    bm = bmesh.new()
    ys = [TRANSOM_Y + (STEM_Y - TRANSOM_Y) * (i / 46) ** 0.9 for i in range(47)]
    rings = []
    for y in ys:
        half = hull_station(y)
        right = [(x, y, z) for x, z in half]
        left = [(-x, y, z) for x, z in reversed(half[1:])]
        rings.append([bm.verts.new(p) for p in left + right])
    for a, b in zip(rings, rings[1:]):
        for i in range(len(a) - 1):
            bm.faces.new((a[i], a[i + 1], b[i + 1], b[i]))
    bm.faces.new(list(reversed(rings[0])))             # transom
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.004)
    for f in bm.faces:
        c = f.calc_center_median()
        f.material_index = 0 if c.z > 0.08 else 1
    hull = part("Hull", bm, [M.hull, M.bottom], smooth=True)
    hull.data.polygons.foreach_set("use_smooth", [True] * len(hull.data.polygons))

    # Rub rail and boot stripe
    bm = bmesh.new()
    for side in (-1, 1):
        pts = [Vector((side * hull_station(y)[-1][0], y, hull_station(y)[-1][1])) for y in ys[:-1]]
        for p0, p1 in zip(pts, pts[1:]):
            bm_frustum(bm, p0 + Vector((side * 0.03, 0, -0.02)), p1 + Vector((side * 0.03, 0, -0.02)), 0.035, 0.035, segs=6)
    part("RubRail", bm, [M.stainless], smooth=True)

    # ---- cockpit -----------------------------------------------------------------------------
    def box(bm, c, s):
        return bm_box(bm, c, s)
    bm = bmesh.new()
    box(bm, (0, (TRANSOM_Y + 0.12 - 1.6) / 2, COCKPIT_Z - 0.04), (3.95, -1.6 - TRANSOM_Y - 0.12, 0.08))
    part("CockpitSole", bm, [M.teak])
    bm = bmesh.new()
    for side in (-1, 1):       # cockpit side liners and covering boards
        box(bm, (side * 2.02, (TRANSOM_Y - 1.6) / 2, (COCKPIT_Z + GUNWALE_Z) / 2), (0.12, -1.6 - TRANSOM_Y, GUNWALE_Z - COCKPIT_Z))
        box(bm, (side * 1.98, (TRANSOM_Y - 1.6) / 2, GUNWALE_Z + 0.03), (0.34, -1.6 - TRANSOM_Y, 0.06))
    box(bm, (0, TRANSOM_Y + 0.1, (COCKPIT_Z + GUNWALE_Z) / 2), (4.1, 0.16, GUNWALE_Z - COCKPIT_Z))
    box(bm, (0, TRANSOM_Y + 0.1, GUNWALE_Z + 0.03), (4.2, 0.3, 0.06))
    part("CockpitLiner", bm, [M.gelcoat])
    bm = bmesh.new()        # transom door seam and bait well lid
    box(bm, (1.0, TRANSOM_Y + 0.19, 0.98), (0.62, 0.012, 0.62))
    box(bm, (0, TRANSOM_Y + 0.55, COCKPIT_Z + 0.32), (0.9, 0.5, 0.64))
    part("TransomDetails", bm, [M.gelcoat])
    bm = bmesh.new()
    box(bm, (1.0, TRANSOM_Y + 0.182, 0.98), (0.66, 0.01, 0.66))
    part("TransomDoorGap", bm, [M.rubber])

    # Fighting chair
    bm = bmesh.new()
    bm_frustum(bm, (0, -4.45, COCKPIT_Z), (0, -4.45, COCKPIT_Z + 0.42), 0.16, 0.11, segs=12)
    bm_frustum(bm, (0, -4.45, COCKPIT_Z + 0.42), (0, -4.45, COCKPIT_Z + 0.52), 0.24, 0.24, segs=12)
    bm_frustum(bm, (-0.42, -4.95, COCKPIT_Z + 0.32), (0.42, -4.95, COCKPIT_Z + 0.32), 0.025, 0.025, segs=8)
    for s in (-1, 1):
        bm_frustum(bm, (s * 0.3, -4.5, COCKPIT_Z + 0.5), (s * 0.4, -4.95, COCKPIT_Z + 0.32), 0.02, 0.02, segs=6)
    part("ChairFrame", bm, [M.stainless], smooth=True)
    bm = bmesh.new()
    box(bm, (0, -4.45, COCKPIT_Z + 0.6), (0.62, 0.56, 0.14))
    box(bm, (0, -4.15, COCKPIT_Z + 0.95), (0.6, 0.12, 0.62))
    part("ChairCushions", bm, [M.cushion])

    # ---- salon, windshield, foredeck -------------------------------------------------------
    bm = bmesh.new()
    verts = []
    for x in (-1.98, 1.98):
        verts.append([bm.verts.new(p) for p in ((x, -1.6, GUNWALE_Z - 0.05), (x, 3.5, GUNWALE_Z - 0.05),
                                                 (x, 3.5, 2.12), (x, 2.6, 2.58), (x, -1.6, 2.58))])
    l, r = verts
    bm.faces.new(l)
    bm.faces.new(list(reversed(r)))
    for i in range(5):
        j = (i + 1) % 5
        bm.faces.new((l[i], r[i], r[j], l[j]))
    salon = part("Salon", bm, [M.gelcoat])
    bm = bmesh.new()        # dark glass: side windows, raked windshield, aft door
    for x in (-1.995, 1.995):
        bm.faces.new([bm.verts.new(p) for p in ((x, -0.9, 1.78), (x, 2.4, 1.78), (x, 2.4, 2.38), (x, -0.9, 2.38))])
    bm.faces.new([bm.verts.new(p) for p in ((-1.7, 3.12, 2.20), (1.7, 3.12, 2.20), (1.5, 2.67, 2.52), (-1.5, 2.67, 2.52))])
    bm.faces.new([bm.verts.new(p) for p in ((-1.5, -1.615, 1.0), (1.5, -1.615, 1.0), (1.5, -1.615, 2.35), (-1.5, -1.615, 2.35))])
    part("SalonGlass", bm, [M.glass])
    bm = bmesh.new()
    rings = []
    for y in [3.5 + (STEM_Y - 0.05 - 3.5) * i / 14 for i in range(15)]:
        hs = hull_station(y)
        b, z = hs[-1][0] * 0.985, hs[-1][1] + 0.01
        crown = 0.12 * (b / 2.0)
        rings.append([bm.verts.new((x, y, z + crown * (1 - (x / max(b, 1e-3)) ** 2))) for x in np.linspace(-b, b, 9)])
    for a, b2 in zip(rings, rings[1:]):
        for i in range(8):
            bm.faces.new((a[i], a[i + 1], b2[i + 1], b2[i]))
    part("Foredeck", bm, [M.gelcoat], smooth=True)
    bm = bmesh.new()        # bow rail
    for side in (-1, 1):
        pts = []
        for y in np.linspace(2.4, STEM_Y - 0.25, 10):
            hs = hull_station(y)
            pts.append(Vector((side * (hs[-1][0] - 0.12), y, hs[-1][1] + 0.62)))
        for p0, p1 in zip(pts, pts[1:]):
            bm_frustum(bm, p0, p1, 0.016, 0.016, segs=6)
        for p in pts[::3]:
            bm_frustum(bm, p, p - Vector((0, 0, 0.6)), 0.014, 0.014, segs=6)
    part("BowRail", bm, [M.stainless], smooth=True)

    # ---- flybridge and hardtop -------------------------------------------------------------
    bm = bmesh.new()
    box(bm, (0, 0.2, 2.62), (3.9, 5.6, 0.1))                  # bridge deck, overhanging the cockpit
    for s in (-1, 1):
        box(bm, (s * 1.92, 0.4, 2.92), (0.08, 4.4, 0.5))       # coaming
    box(bm, (0, 2.62, 2.95), (3.8, 0.1, 0.56))
    box(bm, (0, 1.75, 3.0), (2.6, 0.7, 0.62))                  # helm console
    part("Flybridge", bm, [M.gelcoat])
    bm = bmesh.new()
    box(bm, (0, 1.92, 3.36), (2.4, 0.5, 0.06))
    part("Dash", bm, [M.glass])
    bm = bmesh.new()
    for x in (-0.6, 0.6):
        box(bm, (x, 0.95, 3.12), (0.55, 0.55, 0.14))
        box(bm, (x, 0.7, 3.45), (0.55, 0.1, 0.55))
        bm_frustum(bm, (x, 0.95, 2.67), (x, 0.95, 3.05), 0.06, 0.06, segs=8)
    part("HelmChairs", bm, [M.cushion])
    bm = bmesh.new()
    box(bm, (0, -0.35, 4.62), (3.5, 5.1, 0.09))
    part("Hardtop", bm, [M.gelcoat])
    bm = bmesh.new()
    for x in (-1.6, 1.6):
        for y in (-2.6, 1.9):
            bm_frustum(bm, (x, y, 2.66), (x * 0.97, y, 4.58), 0.032, 0.032, segs=10)
    for x in (-1.6, 1.6):              # ladder from the cockpit to the bridge, port side
        pass
    bm_frustum(bm, (-1.2, -2.3, COCKPIT_Z), (-1.2, -1.75, 2.66), 0.022, 0.022, segs=6)
    bm_frustum(bm, (-0.75, -2.3, COCKPIT_Z), (-0.75, -1.75, 2.66), 0.022, 0.022, segs=6)
    for k in range(1, 7):
        u = k / 7
        bm_frustum(bm, (-1.2, -2.3 + 0.55 * u, COCKPIT_Z + 2.04 * u), (-0.75, -2.3 + 0.55 * u, COCKPIT_Z + 2.04 * u),
                   0.016, 0.016, segs=6)
    part("Stanchions", bm, [M.stainless], smooth=True)

    # ---- outriggers ----------------------------------------------------------------------
    bm = bmesh.new()
    for s in (-1, 1):
        b0, d = RIGGER_BASE[s], RIGGER_DIR[s]
        bm_frustum(bm, b0 - Vector((0, 0, 0.06)), b0 + d * 0.5, 0.05, 0.04, segs=10)
        bm_frustum(bm, b0 + d * 0.5, b0 + d * RIGGER_LEN, 0.032, 0.012, segs=8)
        bm_frustum(bm, b0 + d * 2.8, b0 + d * 2.8 + Vector((0, 0.9, 0.5)), 0.008, 0.008, segs=5)   # spreader
    part("Outriggers", bm, [M.stainless], smooth=True)

    # ---- rod holders -------------------------------------------------------------------------
    bm = bmesh.new()
    for name, (butt, d) in HOLDERS.items():
        bm_frustum(bm, butt - d * 0.25, butt + d * 0.04, 0.03, 0.03, segs=10)
    part("RodHolders", bm, [M.stainless], smooth=True)

    # ---- running gear (props spin) -------------------------------------------------------
    props = []
    bm = bmesh.new()
    for s in (-1, 1):
        bm_frustum(bm, (s * 0.62, -1.5, -0.62), (s * 0.62, -4.75, -0.98), 0.035, 0.035, segs=8)     # shaft
        bm_frustum(bm, (s * 0.62, -4.2, -0.62), (s * 0.62, -4.25, -0.95), 0.02, 0.02, segs=6)       # strut
        box(bm, (s * 0.62, -5.55, -0.9), (0.03, 0.5, 0.5))                                             # rudder
    part("RunningGear", bm, [M.stainless], smooth=True)
    for s in (-1, 1):
        hub = empty_object(f"Prop{'L' if s < 0 else 'R'}", col, loc=(s * 0.62, -4.85, -0.99), parent=root, size=0.2)
        hub.rotation_mode = "XYZ"
        hub.rotation_euler = (math.radians(-6), 0, 0)
        bm = bmesh.new()
        bm_frustum(bm, (0, 0.08, 0), (0, -0.12, 0), 0.07, 0.05, segs=8)
        for k in range(4):
            a = 2 * math.pi * k / 4
            c, sn = math.cos(a), math.sin(a)
            blade = [(0.06 * c - 0.05 * sn, 0.03, 0.06 * sn + 0.05 * c), (0.36 * c - 0.1 * sn, -0.04, 0.36 * sn + 0.1 * c),
                     (0.36 * c + 0.08 * sn, 0.0, 0.36 * sn - 0.08 * c), (0.06 * c + 0.05 * sn, -0.05, 0.06 * sn - 0.05 * c)]
            bm.faces.new([bm.verts.new(p) for p in blade])
        prop = mesh_object(f"PropBlades{s}", bm, col, [M.reel], parent=hub, recalc=False)
        parts.append(prop)
        fc = hub.driver_add("rotation_euler", 1)
        fc.driver.type = "SCRIPTED"
        fc.driver.expression = f"frame*{0.9 * s}"
        props.append(hub)

    # ---- release clips at the outrigger tips and transom (flat lines) ------------------------
    clips = {}
    for name, side in (("LongRigger", -1), ("ShortRigger", 1)):
        clips[name] = RIGGER_BASE[side] + RIGGER_DIR[side] * (RIGGER_LEN * 0.9) + Vector((0, 0, -0.25))
    clips["LeftFlat"] = Vector((-1.1, TRANSOM_Y - 0.02, 0.85))
    clips["RightFlat"] = Vector((1.1, TRANSOM_Y - 0.02, 0.85))
    bm = bmesh.new()
    for name, p in clips.items():
        box(bm, p, (0.05, 0.03, 0.09))
    clip_ob = part("ReleaseClips", bm, [M.rubber])
    return root, parts, clips


def animate_boat(root, wake_ref):
    sc = bpy.context.scene
    for f in range(1, FRAME_END + 1):
        m = boat_matrix(f)
        root.location = m.translation
        root.rotation_euler = m.to_euler("XYZ")
        key(root, "location", f)
        key(root, "rotation_euler", f)
        wake_ref.location = (0.0, BOAT_Y[f], 0.0)
        key(wake_ref, "location", f)
        sc["wake_gain"] = min(1.0, 0.25 + 0.75 * boat_speed(f) / TROLL_SPEED)
        sc.keyframe_insert(data_path='["wake_gain"]', frame=f)


# --------------------------------------------------------------------------
# Creatures and baits (procedural meshes, head along +X)
# --------------------------------------------------------------------------
def store_rest(me, z_lo=None, z_hi=None):
    """Rest position (for shading after the swim deformation) and 0..1 height (belly..back)."""
    co = [v.co.copy() for v in me.vertices]
    a = me.attributes.new("rest", "FLOAT_VECTOR", "POINT")
    a.data.foreach_set("vector", [c for v in co for c in v])
    zs = [v.z for v in co]
    lo, hi = (min(zs) if z_lo is None else z_lo), (max(zs) if z_hi is None else z_hi)
    b = me.attributes.new("zrel", "FLOAT", "POINT")
    b.data.foreach_set("value", [min(1.0, max(0.0, (z - lo) / (hi - lo))) for z in zs])


def loft_body(bm, stations, segs=16):
    """stations: (x, z_bottom, z_top, half_width). Returns rings of verts; nose and tail are closed."""
    rings = []
    for x, zb, zt, w in stations:
        ring = []
        for i in range(segs):
            a = 2 * math.pi * i / segs
            s, c = math.sin(a), math.cos(a)
            y = w * math.copysign(abs(c) ** 0.8, c)
            z = zb + (zt - zb) * (0.5 + 0.5 * math.copysign(abs(s) ** 0.9, s))
            ring.append(bm.verts.new((x, y, z)))
        rings.append(ring)
    for a, b in zip(rings, rings[1:]):
        for i in range(segs):
            j = (i + 1) % segs
            bm.faces.new((a[i], a[j], b[j], b[i]))
    for ring, rev in ((rings[0], True), (rings[-1], False)):
        c = sum((v.co for v in ring), Vector()) / len(ring)
        cv = bm.verts.new(c)
        for i in range(segs):
            j = (i + 1) % segs
            bm.faces.new((ring[j], ring[i], cv) if rev else (ring[i], ring[j], cv))
    return rings


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


# Mahi: side profiles (x, z) for a 1 m fish, head along +X.
MAHI_BULL_TOP = [(0.500, -0.002), (0.4985, 0.040), (0.496, 0.090), (0.492, 0.125), (0.485, 0.148), (0.474, 0.160),
            (0.455, 0.166), (0.420, 0.167), (0.370, 0.163), (0.280, 0.152), (0.150, 0.136), (0.020, 0.117),
            (-0.100, 0.093), (-0.210, 0.067), (-0.300, 0.044), (-0.360, 0.028), (-0.395, 0.020), (-0.415, 0.017)]
MAHI_COW_TOP = [(0.500, 0.002), (0.495, 0.018), (0.485, 0.036), (0.468, 0.058), (0.445, 0.080), (0.410, 0.100),
           (0.360, 0.116), (0.290, 0.126), (0.200, 0.128), (0.100, 0.120), (-0.010, 0.103), (-0.110, 0.085),
           (-0.210, 0.063), (-0.300, 0.042), (-0.360, 0.027), (-0.395, 0.020), (-0.415, 0.017)]
MAHI_BOT = [(0.500, -0.004), (0.497, -0.028), (0.486, -0.053), (0.462, -0.076), (0.420, -0.093), (0.350, -0.103),
       (0.250, -0.106), (0.120, -0.103), (0.000, -0.095), (-0.110, -0.080), (-0.210, -0.060), (-0.300, -0.040),
       (-0.360, -0.027), (-0.395, -0.020), (-0.415, -0.017)]
MAHI_HALF_W = [(0.500, 0.004), (0.494, 0.013), (0.480, 0.023), (0.455, 0.030), (0.420, 0.035), (0.350, 0.039),
          (0.250, 0.041), (0.120, 0.039), (0.000, 0.035), (-0.120, 0.030), (-0.220, 0.023), (-0.300, 0.016),
          (-0.360, 0.011), (-0.395, 0.008), (-0.415, 0.007)]
MAHI_NOSE, MAHI_END = 0.5, -0.415
MAHI_EYE = (0.452, 0.006)            # low on the head, just above the corner of the mouth
MAHI_EYE_R = 0.0145


def monotone_curve(points, scale=1.0):
    """Shape-preserving cubic through (x, y) points (Fritsch and Carlson): no overshoot between them."""
    pts = sorted(points)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] * scale for p in pts])
    h, dl = np.diff(xs), np.diff(ys) / np.diff(xs)
    m = np.zeros_like(ys)
    m[0], m[-1] = dl[0], dl[-1]
    for i in range(1, len(xs) - 1):
        if dl[i - 1] * dl[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / dl[i - 1] + w2 / dl[i])

    def f(x):
        x = min(xs[-1], max(xs[0], x))
        i = min(len(xs) - 2, int(np.searchsorted(xs, x, side="right")) - 1)
        t = (x - xs[i]) / h[i]
        h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
        return float(h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1])
    return f


def mahi_profiles(bull):
    top = monotone_curve(MAHI_BULL_TOP if bull else MAHI_COW_TOP)
    bot = monotone_curve(MAHI_BOT, 1.0 if bull else 0.94)
    w = monotone_curve(MAHI_HALF_W, 1.0 if bull else 0.95)
    return top, bot, w


def mahi_section(zb, zt, w, s, c):
    """Point on a compressed cross-section: narrow dorsal keel, fuller belly."""
    t = 0.5 + 0.5 * math.copysign(abs(s) ** 0.85, s)
    shape = 1.0 - 0.40 * max(s, 0.0) ** 1.5 - 0.10 * max(-s, 0.0) ** 2
    return w * math.copysign(abs(c) ** 0.75, c) * shape, zb + (zt - zb) * t, t


def fin_rays(bm, rays, normal, rows, thick, mat):
    """rays: [(base, tip)] Vectors; builds a two-sided strip and fills the finu/finv layers."""
    lu, lv = bm.verts.layers.float["finu"], bm.verts.layers.float["finv"]
    n = len(rays)
    top, bot = [], []
    for j, (b, t) in enumerate(rays):
        rt, rb = [], []
        for r in range(rows + 1):
            u = r / rows
            p = b.lerp(t, u)
            h = thick * (1.0 - 0.75 * u) / 2
            for lst, sgn in ((rt, 1), (rb, -1)):
                v = bm.verts.new(p + normal * (h * sgn))
                v[lu], v[lv] = u, j / max(1, n - 1)
                lst.append(v)
        top.append(rt)
        bot.append(rb)
    faces = []
    for j in range(n - 1):
        for r in range(rows):
            faces.append(bm.faces.new((top[j][r], top[j + 1][r], top[j + 1][r + 1], top[j][r + 1])))
            faces.append(bm.faces.new((bot[j][r + 1], bot[j + 1][r + 1], bot[j + 1][r], bot[j][r])))
        faces.append(bm.faces.new((top[j][rows], top[j + 1][rows], bot[j + 1][rows], bot[j][rows])))
    for j, rev in ((0, False), (n - 1, True)):
        for r in range(rows):
            q = (top[j][r], top[j][r + 1], bot[j][r + 1], bot[j][r])
            faces.append(bm.faces.new(tuple(reversed(q)) if rev else q))
    for f in faces:
        f.material_index = mat


def mahi_mesh(M, bull=True, name=None):
    """Material slots: 0 skin, 1 dorsal fin, 2 tail, 3 eye, 4 pectoral, pelvic and anal fins."""
    top, bot, wid = mahi_profiles(bull)
    bm = bmesh.new()
    lz = bm.verts.layers.float.new("zrel")              # bmesh layers become mesh attributes
    bm.verts.layers.float.new("finu")
    bm.verts.layers.float.new("finv")

    # Body: dense stations at the head, where the profile turns fastest
    segs, n_st = 32, 52
    xs = [MAHI_NOSE - (MAHI_NOSE - MAHI_END) * (k / (n_st - 1)) ** 1.45 for k in range(n_st)]
    rings = []
    for x in xs:
        zb, zt, w = bot(x), top(x), wid(x)
        ring = []
        for i in range(segs):
            a = 2 * math.pi * i / segs
            y, z, t = mahi_section(zb, zt, w, math.sin(a), math.cos(a))
            v = bm.verts.new((x, y, z))
            v[lz] = t
            ring.append(v)
        rings.append(ring)
    for ra, rb in zip(rings, rings[1:]):
        for i in range(segs):
            j = (i + 1) % segs
            bm.faces.new((ra[i], ra[j], rb[j], rb[i]))
    for ring, rev in ((rings[0], True), (rings[-1], False)):
        cv = bm.verts.new(sum((v.co for v in ring), Vector()) / len(ring))
        cv[lz] = 0.5
        for i in range(segs):
            j = (i + 1) % segs
            bm.faces.new((ring[j], ring[i], cv) if rev else (ring[i], ring[j], cv))
    for f in bm.faces:
        f.material_index = 0
        f.smooth = True

    # Dorsal fin: one fin from above the eye almost to the tail, rays raked back, tallest over the head
    x0 = 0.446 if bull else 0.43
    rays = []
    for k in range(64):
        x = x0 + (MAHI_END + 0.035 - x0) * k / 63
        u = (x0 - x) / (x0 - MAHI_END)
        h = (0.060 + 0.030 * math.sin(min(1.0, u / 0.12) * math.pi / 2)) * (1 - 0.55 * u ** 1.3)
        h *= 1.0 + 0.04 * math.sin(k * math.pi)                     # ray tips
        b = Vector((x, 0.0, top(x) - 0.004))
        rays.append((b, Vector((x - 0.35 * h, 0.0, top(x) + h))))
    fin_rays(bm, rays, Vector((0, 1, 0)), 3, 0.004, 1)

    # Anal fin: from mid body almost to the tail, concave margin (tall at the front)
    rays = []
    for k in range(30):
        x = 0.02 + (MAHI_END + 0.04 - 0.02) * k / 29
        u = k / 29
        h = 0.022 + 0.042 * (1 - u) ** 2.2
        b = Vector((x, 0.0, bot(x) + 0.004))
        rays.append((b, Vector((x - 0.35 * h, 0.0, bot(x) - h))))
    fin_rays(bm, rays, Vector((0, 1, 0)), 3, 0.004, 4)

    # Tail: deeply forked, rays fan from the peduncle to the trailing edge; the outer rays are the lobes' leading edges
    edge = [(-0.610, 0.218), (-0.585, 0.180), (-0.555, 0.130), (-0.525, 0.085), (-0.500, 0.045), (-0.484, 0.012),
            (-0.480, 0.0), (-0.484, -0.012), (-0.500, -0.045), (-0.525, -0.085), (-0.555, -0.130), (-0.585, -0.180),
            (-0.610, -0.218)]
    d = [0.0]
    for p, q in zip(edge, edge[1:]):
        d.append(d[-1] + math.dist(p, q))
    n_r = 36
    ex = np.interp(np.linspace(0, d[-1], n_r), d, [p[0] for p in edge])
    ez = np.interp(np.linspace(0, d[-1], n_r), d, [p[1] for p in edge])
    rays = []
    for k in range(n_r):
        zb = 0.019 - 0.038 * k / (n_r - 1)
        rays.append((Vector((MAHI_END + 0.012, 0.0, zb)), Vector((float(ex[k]), 0.0, float(ez[k])))))
    fin_rays(bm, rays, Vector((0, 1, 0)), 4, 0.005, 2)

    # Pectoral fins: long and sickle shaped, low on the side behind the gill cover; pelvic fins below them
    for side in (-1, 1):
        xa = 0.372
        za = bot(xa) + 0.30 * (top(xa) - bot(xa))
        base = Vector((xa, side * wid(xa) * 0.9, za))
        along = Vector((-1.0, side * 0.22, -0.20)).normalized()
        up = (Vector((0, 0, 1)) - along * along.z).normalized()
        nrm = along.cross(up).normalized()
        rays = []
        for k in range(16):
            v = k / 15
            b = base + up * (0.015 - 0.030 * v)
            reach = 0.150 * (1 - v) ** 0.7 + 0.030 * v
            t = base + along * reach + up * (-0.012 - 0.016 * v)
            rays.append((b, t))
        fin_rays(bm, rays, nrm, 3, 0.003, 4)
        xp = 0.35
        base = Vector((xp, side * 0.012, bot(xp) + 0.006))
        along = Vector((-1.0, side * 0.18, -0.40)).normalized()
        nrm = Vector((1, 0, 0)).cross(along).normalized()
        rays = [(base + Vector((0.008 - 0.016 * k / 9, 0, 0)), base + along * (0.06 * (1 - 0.55 * k / 9)))
                for k in range(10)]
        fin_rays(bm, rays, nrm, 2, 0.003, 4)

    # Eyes
    ex_, ez_ = MAHI_EYE
    for side in (-1, 1):
        cy = side * (wid(ex_) * 0.97 - MAHI_EYE_R * 0.7)         # mostly flush with the head
        res = bmesh.ops.create_icosphere(bm, subdivisions=3, radius=MAHI_EYE_R)
        for v in res["verts"]:
            v.co += Vector((ex_, cy, ez_))
            v[lz] = 0.5
        for f in {f for v in res["verts"] for f in v.link_faces}:
            f.material_index = 3
            f.smooth = True

    me = bpy.data.meshes.new(name or ("MahiBull" if bull else "Mahi"))
    bm.to_mesh(me)
    bm.free()
    for m in (M.mahi, M.fin_dorsal, M.fin_tail, M.eye, M.fin_small):
        me.materials.append(m)
    a = me.attributes.new("rest", "FLOAT_VECTOR", "POINT")
    a.data.foreach_set("vector", [c for v in me.vertices for c in v.co])
    return me


def ballyhoo_mesh(M):
    """Rigged ballyhoo, 0.26 m: slim body, long lower beak, pink skirt over the head, hook point under the belly."""
    prof = [(0.13, -0.002, 0.002, 0.002), (0.11, -0.008, 0.008, 0.007), (0.06, -0.012, 0.013, 0.011),
            (-0.04, -0.011, 0.012, 0.01), (-0.09, -0.007, 0.008, 0.006), (-0.115, -0.004, 0.004, 0.003)]
    bm = bmesh.new()
    loft_body(bm, prof, segs=10)
    bm_frustum(bm, (0.12, 0, -0.004), (0.20, 0, -0.006), 0.0025, 0.0006, segs=5)       # beak
    fin_sheet(bm, [(-0.11, 0.0), (-0.15, 0.03), (-0.135, 0.0), (-0.15, -0.03)], thickness=0.002)
    for f in bm.faces:
        f.material_index = 0
    n0 = len(bm.faces)
    for k in range(14):                                  # skirt strands over the head
        a = 2 * math.pi * k / 14
        y, z = 0.016 * math.cos(a), 0.016 * math.sin(a)
        y2, z2 = 0.03 * math.cos(a), 0.03 * math.sin(a)
        p = [(0.115, y, z), (0.115, y * 1.15, z * 1.15), (0.01, y2 * 1.1, z2 * 1.1), (0.01, y2, z2)]
        bm.faces.new([bm.verts.new(v) for v in p])
    for f in list(bm.faces)[n0:]:
        f.material_index = 1
    n1 = len(bm.faces)
    hook = [Vector((0.02 + 0.03 * math.cos(a), 0.0, -0.022 + 0.03 * math.sin(a)))
            for a in np.linspace(math.pi * 1.05, math.pi * 2.1, 8)]
    for p0, p1 in zip(hook, hook[1:]):
        bm_frustum(bm, p0, p1, 0.0018, 0.0018, segs=5)
    for f in list(bm.faces)[n1:]:
        f.material_index = 2
    me = bpy.data.meshes.new("Ballyhoo")
    bm.to_mesh(me)
    bm.free()
    for m in (M.ballyhoo, M.skirt_pink, M.hook):
        me.materials.append(m)
    for p in me.polygons:
        p.use_smooth = p.material_index == 0
    store_rest(me)
    return me


def skirt_lure_mesh(M):
    """Trolling lure: resin chugger head and a flared two-colour vinyl skirt, hook inside."""
    bm = bmesh.new()
    bm_frustum(bm, (0.09, 0, 0), (0.0, 0, 0), 0.026, 0.022, segs=14)
    for f in bm.faces:
        f.material_index = 0
    n0 = len(bm.faces)
    for k in range(20):
        a = 2 * math.pi * k / 20
        c, s = math.cos(a), math.sin(a)
        p = [(0.005, 0.02 * c, 0.02 * s), (0.005, 0.022 * c, 0.022 * s),
             (-0.22, 0.05 * c + 0.004 * s, 0.05 * s - 0.004 * c), (-0.22, 0.047 * c, 0.047 * s)]
        f = bm.faces.new([bm.verts.new(v) for v in p])
        f.material_index = 1 if k % 2 else 2
    hook = [Vector((-0.17 + 0.035 * math.cos(a), 0.0, -0.01 + 0.035 * math.sin(a)))
            for a in np.linspace(math.pi * 1.05, math.pi * 2.1, 8)]
    for p0, p1 in zip(hook, hook[1:]):
        bm_frustum(bm, p0, p1, 0.002, 0.002, segs=5)
    for f in list(bm.faces)[n0 + 20:]:
        f.material_index = 3
    me = bpy.data.meshes.new("SkirtLure")
    bm.to_mesh(me)
    bm.free()
    for m in (M.lure_head, M.skirt_blue, M.skirt_pink, M.hook):
        me.materials.append(m)
    store_rest(me)
    return me


def baitfish_mesh(M):
    prof = [(0.05, -0.002, 0.002, 0.002), (0.035, -0.009, 0.01, 0.006), (-0.01, -0.009, 0.01, 0.006),
            (-0.04, -0.004, 0.004, 0.003), (-0.05, -0.002, 0.002, 0.002)]
    bm = bmesh.new()
    loft_body(bm, prof, segs=6)
    fin_sheet(bm, [(-0.045, 0.0), (-0.065, 0.014), (-0.058, 0.0), (-0.065, -0.014)], thickness=0.001)
    me = bpy.data.meshes.new("Baitfish")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M.bait)
    return me


# --------------------------------------------------------------------------
# Tackle: rods that bend towards their load, reels with a drag lever, lines
# --------------------------------------------------------------------------
def rod_gn(name):
    """A tapered rod blank along a quadratic bezier between three empties (butt, mid, tip)."""
    ng, gi, go = new_group(name + "GN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    return ng, gi, go


def build_rod(col, M, name, boat_root):
    ctrl = {k: empty_object(f"{name}Rod{k}", col, parent=boat_root, size=0.05) for k in ("Butt", "Mid", "Tip")}
    ng, gi, go = rod_gn(name + "Rod")
    nodes, links = ng.nodes, ng.links

    def loc(ob):
        n = nodes.new("GeometryNodeObjectInfo")
        n.transform_space = "RELATIVE"
        n.inputs["Object"].default_value = ob
        return n.outputs["Location"]
    c = nodes.new("GeometryNodeCurveQuadraticBezier")
    c.inputs["Resolution"].default_value = 24
    links.new(loc(ctrl["Butt"]), c.inputs["Start"])
    links.new(loc(ctrl["Mid"]), c.inputs["Middle"])
    links.new(loc(ctrl["Tip"]), c.inputs["End"])
    rad = nodes.new("GeometryNodeSetCurveRadius")
    links.new(c.outputs["Curve"], rad.inputs["Curve"])
    par = nodes.new("GeometryNodeSplineParameter")
    links.new(map_range(ng, par.outputs["Factor"], 0.0, 1.0, 1.0, 0.25, smooth=False), rad.inputs["Radius"])
    prof = nodes.new("GeometryNodeCurvePrimitiveCircle")
    prof.inputs["Resolution"].default_value = 8
    prof.inputs["Radius"].default_value = 0.013
    c2m = nodes.new("GeometryNodeCurveToMesh")
    links.new(rad.outputs[0], c2m.inputs["Curve"])
    links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
    try:
        c2m.inputs["Fill Caps"].default_value = True
    except KeyError:
        pass
    setm = nodes.new("GeometryNodeSetMaterial")
    setm.inputs["Material"].default_value = M.rod
    links.new(c2m.outputs[0], setm.inputs["Geometry"])
    links.new(setm.outputs[0], go.inputs["Geometry"])
    ob = bpy.data.objects.new(name + "Rod", bpy.data.meshes.new(name + "Rod"))
    col.objects.link(ob)
    ob.modifiers.new("Rod", "NODES").node_group = ng
    # Reel: gold body, spool, handle, drag lever (keyed on the lever's pivot)
    mount = empty_object(f"{name}ReelMount", col, parent=boat_root, size=0.08)
    mount.rotation_mode = "QUATERNION"
    bm = bmesh.new()
    for f in bm_frustum(bm, (-0.052, 0, 0), (0.052, 0, 0), 0.05, 0.05, segs=24):
        f.material_index = 1                                   # line on the spool
    bm_frustum(bm, (-0.068, 0, 0), (-0.052, 0, 0), 0.068, 0.068, segs=24)
    bm_frustum(bm, (0.052, 0, 0), (0.068, 0, 0), 0.068, 0.068, segs=24)
    for z in (-0.06, 0.06):                                    # frame posts between the side plates
        bm_frustum(bm, (-0.055, 0.0, z), (0.055, 0.0, z), 0.007, 0.007, segs=6)
    reel = mesh_object(f"{name}Reel", bm, col, [M.reel, M.spool], parent=mount, recalc=False)
    for p in reel.data.polygons:
        p.use_smooth = True
    bm = bmesh.new()
    bm_frustum(bm, (0.07, 0, 0), (0.12, 0.0, 0.0), 0.012, 0.012, segs=8)
    bm_frustum(bm, (0.12, 0, 0), (0.12, -0.08, -0.03), 0.009, 0.009, segs=6)
    bm_frustum(bm, (0.12, -0.08, -0.03), (0.16, -0.08, -0.03), 0.016, 0.016, segs=10)
    bm_box(bm, (0, 0.05, -0.06), (0.03, 0.06, 0.03))            # reel seat clamp
    mesh_object(f"{name}ReelHandle", bm, col, [M.reel_dark], parent=mount)
    lever = empty_object(f"{name}DragLever", col, loc=(-0.07, 0.0, 0.02), parent=mount, size=0.03)
    lever.rotation_mode = "XYZ"
    bm = bmesh.new()
    bm_frustum(bm, (0, 0, 0), (-0.012, 0.0, 0.085), 0.01, 0.007, segs=8)
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.014, matrix=Matrix.Translation((-0.012, 0.0, 0.09)))
    lev = mesh_object(f"{name}DragLeverArm", bm, col, [M.lever], parent=lever)
    return dict(ctrl=ctrl, rod=ob, mount=mount, reel=reel, lever=lever, lever_arm=lev)


def place_rod(r, f, butt, direction, load_dir=None, bend=0.0):
    """Key the rod at frame f in boat space. bend 0..1 curls the tip towards load_dir."""
    d = direction.normalized()
    tip_dir = d
    if load_dir is not None and bend > 0:
        tip_dir = d.slerp(load_dir.normalized(), min(0.95, bend * 0.75))
    butt = Vector(butt)
    mid = butt + d * (ROD_LEN * 0.5)
    tip = mid + (d * 0.45 + tip_dir * 0.55).normalized() * (ROD_LEN * 0.5) * (1.0 - 0.08 * bend)
    for k, p in (("Butt", butt - d * 0.35), ("Mid", mid), ("Tip", tip)):
        r["ctrl"][k].location = p
        key(r["ctrl"][k], "location", f)
    side = d.cross(Vector((0, 0, 1)))
    if side.length < 1e-3:
        side = Vector((1, 0, 0))
    side.normalize()
    up = side.cross(d).normalized()
    rot = Matrix((side, d, up)).transposed().to_quaternion()
    r["mount"].location = butt + d * 0.24 + up * 0.075      # conventional reel rides on top of the blank
    r["mount"].rotation_quaternion = rot
    key(r["mount"], "location", f)
    key(r["mount"], "rotation_quaternion", f)
    return tip


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


# --------------------------------------------------------------------------
# Environment generators (geometry nodes)
# --------------------------------------------------------------------------
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


def build_ocean(col, M, cam):
    """Fine 160 m patch + coarse ring that follow the camera, snapped to the vertex lattice so the
    waves never swim; heights come from waves(), the same formula the boat rides on."""
    ng, gi, go = new_group("OceanGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links
    B = NodeB(ng)
    t = gn_time(ng)
    centre = gn_cam_snap(ng, cam, 8.0)

    def grid(half, step):
        g = nodes.new("GeometryNodeMeshGrid")
        g.inputs["Size X"].default_value = g.inputs["Size Y"].default_value = 2 * half
        g.inputs["Vertices X"].default_value = g.inputs["Vertices Y"].default_value = int(2 * half / step) + 1
        sp = nodes.new("GeometryNodeSetPosition")
        links.new(g.outputs["Mesh"], sp.inputs["Geometry"])
        links.new(centre, sp.inputs["Offset"])
        return sp.outputs[0]

    fine = grid(80.0, 0.5)
    x, y, _z = gn_xyz(ng)
    sp = nodes.new("GeometryNodeSetPosition")
    links.new(fine, sp.inputs["Geometry"])
    links.new(combine(ng, 0.0, 0.0, waves(B, x, y, t)), sp.inputs["Offset"])
    fine = sp.outputs[0]

    coarse = grid(1200.0, 8.0)
    dele = nodes.new("GeometryNodeDeleteGeometry")
    dele.domain = "FACE"
    links.new(coarse, dele.inputs["Geometry"])
    x, y, _z = gn_xyz(ng)
    cs = separate(ng, centre)
    inside = nmath(ng, "MULTIPLY",
                   nmath(ng, "LESS_THAN", nmath(ng, "ABSOLUTE", nmath(ng, "SUBTRACT", x, cs["X"])), 72.0),
                   nmath(ng, "LESS_THAN", nmath(ng, "ABSOLUTE", nmath(ng, "SUBTRACT", y, cs["Y"])), 72.0))
    links.new(inside, dele.inputs["Selection"])
    x, y, _z = gn_xyz(ng)
    dist = nmath(ng, "MAXIMUM", nmath(ng, "ABSOLUTE", nmath(ng, "SUBTRACT", x, cs["X"])),
                 nmath(ng, "ABSOLUTE", nmath(ng, "SUBTRACT", y, cs["Y"])))
    swell = B.add(*[B.mul(B.sin(B.add(B.mul(x, 2 * math.pi / lam * math.cos(math.radians(deg))),
                                      B.mul(y, 2 * math.pi / lam * math.sin(math.radians(deg))),
                                      B.mul(t, -math.sqrt(9.81 * 2 * math.pi / lam)), ph)), amp)
                    for lam, amp, deg, ph in WAVES[:2]])
    swell = nmath(ng, "MULTIPLY", swell, map_range(ng, dist, 700.0, 1150.0, 1.0, 0.0))
    under = nmath(ng, "MULTIPLY", map_range(ng, dist, 82.0, 72.0), -0.18)    # tucked under the fine patch edge
    sp2 = nodes.new("GeometryNodeSetPosition")
    links.new(dele.outputs[0], sp2.inputs["Geometry"])
    links.new(combine(ng, 0.0, 0.0, nmath(ng, "ADD", swell, under)), sp2.inputs["Offset"])
    join = nodes.new("GeometryNodeJoinGeometry")
    links.new(sp2.outputs[0], join.inputs[0])
    links.new(fine, join.inputs[0])
    smooth = nodes.new("GeometryNodeSetShadeSmooth")
    links.new(join.outputs[0], smooth.inputs[0])
    setm = nodes.new("GeometryNodeSetMaterial")
    links.new(smooth.outputs[0], setm.inputs["Geometry"])
    setm.inputs["Material"].default_value = M.water
    links.new(setm.outputs[0], go.inputs["Geometry"])
    ob = gn_object("Ocean", col, ng)
    ob.visible_shadow = False
    # Flat far sea out to the horizon, following the camera
    bm = bmesh.new()
    inner, outer = 1150.0, 20000.0
    vi = [bm.verts.new((sx * inner, sy * inner, 0.0)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    vo = [bm.verts.new((sx * outer, sy * outer, 0.0)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for i in range(4):
        j = (i + 1) % 4
        bm.faces.new((vi[i], vo[i], vo[j], vi[j]))
    far = mesh_object("OceanFar", bm, col, [M.water], recalc=False)
    far.visible_shadow = False
    follow(far, cam, z=False)
    return ob, far


def follow(ob, target, z=True):
    con = ob.constraints.new("COPY_LOCATION")
    con.target = target
    con.use_z = z


def build_depth_props(col, M, cam):
    """Backdrop sphere (underwater) and a deep-blue floor seen through the surface from above."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=400.0)
    bmesh.ops.reverse_faces(bm, faces=bm.faces)
    back = mesh_object("UnderwaterBackdrop", bm, col, [M.backdrop], recalc=False)
    back.visible_shadow = False
    follow(back, cam)
    bm = bmesh.new()
    bm.faces.new([bm.verts.new(p) for p in ((-600, -600, -30), (600, -600, -30), (600, 600, -30), (-600, 600, -30))])
    floor = mesh_object("DeepWaterBelow", bm, col, [M.deep_floor], recalc=False)
    floor.visible_shadow = False
    follow(floor, cam, z=False)
    return back, floor


def build_beams(col, M, cam):
    """Sun shafts under the surface: soft cylinders slanting along the refracted sun direction,
    thinned out beneath the weed line where the floating mat shades the water."""
    horiz = Vector((SUN_DIR.x, SUN_DIR.y, 0.0)).normalized()
    sin_w = math.sqrt(1 - SUN_DIR.z ** 2) / 1.333
    d = (-horiz * sin_w + Vector((0, 0, -math.sqrt(1 - sin_w ** 2)))).normalized()   # travel direction
    bm = bmesh.new()
    bm_frustum(bm, (0, 0, 0), (0, 0, -1), 1.0, 1.0, segs=16, cap0=False, cap1=False)
    unit = bpy.data.meshes.new("BeamUnit")
    bm.to_mesh(unit)
    bm.free()
    for p in unit.polygons:
        p.use_smooth = True
    unit.materials.append(M.beam)
    unit_ob = bpy.data.objects.new("BeamUnit", unit)        # not linked to the scene
    ng, gi, go = new_group("BeamsGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links
    B = NodeB(ng)
    g = nodes.new("GeometryNodeMeshGrid")
    g.inputs["Size X"].default_value = g.inputs["Size Y"].default_value = 84.0
    g.inputs["Vertices X"].default_value = g.inputs["Vertices Y"].default_value = 15
    m2p = nodes.new("GeometryNodeMeshToPoints")
    links.new(g.outputs["Mesh"], m2p.inputs["Mesh"])
    sp = nodes.new("GeometryNodeSetPosition")
    links.new(m2p.outputs[0], sp.inputs["Geometry"])
    rnd = nodes.new("FunctionNodeRandomValue")
    rnd.data_type = "FLOAT_VECTOR"
    rnd.inputs["Min"].default_value = (-2.7, -2.7, 0.0)
    rnd.inputs["Max"].default_value = (2.7, 2.7, 0.0)
    off = vmath(ng, "ADD", gn_cam_snap(ng, cam, 6.0), rnd.outputs["Value"])
    links.new(off, sp.inputs["Offset"])
    x, y, _z = gn_xyz(ng)
    shade = map_range(ng, nmath(ng, "ABSOLUTE", nmath(ng, "SUBTRACT", x, weed_x(y, B))), 0.0, 3.6, 1.0, 0.0)
    keep = nodes.new("FunctionNodeRandomValue")
    keep.data_type = "FLOAT"
    keep.inputs["Min"].default_value = 0.0
    keep.inputs["Max"].default_value = 1.0
    keep.inputs["Seed"].default_value = 3
    dele = nodes.new("GeometryNodeDeleteGeometry")
    dele.domain = "POINT"
    links.new(sp.outputs[0], dele.inputs["Geometry"])
    links.new(nmath(ng, "LESS_THAN", keep.outputs["Value"], nmath(ng, "MULTIPLY_ADD", shade, 0.8, 0.25)),
              dele.inputs["Selection"])
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
    rs.inputs["Min"].default_value = 0.3
    rs.inputs["Max"].default_value = 1.1
    rs.inputs["Seed"].default_value = 9
    links.new(combine(ng, rs.outputs["Value"], rs.outputs["Value"], 34.0), inst.inputs["Scale"])
    links.new(inst.outputs[0], go.inputs["Geometry"])
    ob = gn_object("SunShafts", col, ng)
    ob.location = (0, 0, -0.35)
    ob.visible_shadow = False
    return ob


def build_snow(col, M):
    """Suspended specks, fixed in the water along the underwater part of the route."""
    rng = random.Random(5)
    bm = bmesh.new()
    for _ in range(4200):
        p = Vector((rng.uniform(-26, 12), rng.uniform(-30, 120), rng.uniform(-13, -0.6)))
        r = rng.uniform(0.006, 0.02)
        bmesh.ops.create_icosphere(bm, subdivisions=0, radius=r, matrix=Matrix.Translation(p))
    ob = mesh_object("MarineSnow", bm, col, [M.snow], recalc=False)
    ob.visible_shadow = False
    return ob


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
        B = NodeB(ng)

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
        x, y, z = gn_xyz(ng)
        surf = waves(B, x, y, t)
        wrong_side = nmath(ng, "GREATER_THAN", z, surf) if not above_water else nmath(ng, "LESS_THAN", z, nmath(ng, "SUBTRACT", surf, 0.05))
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
            half = nmath(ng, "ADD", grow, nmath(ng, "MULTIPLY", vmath(ng, "LENGTH", v_now),
                                                SHUTTER / FPS / 2))
            links.new(combine(ng, grow, grow, half), inst.inputs["Scale"])
        else:
            links.new(combine(ng, grow, grow, grow), inst.inputs["Scale"])
        setm = nodes.new("GeometryNodeSetMaterial")
        links.new(inst.outputs[0], setm.inputs["Geometry"])
        setm.inputs["Material"].default_value = mat
        links.new(setm.outputs[0], go.inputs["Geometry"])
        mod = ob.modifiers.new(name, "NODES")
        mod.node_group = ng
        ob.visible_shadow = False
        print(f"{name}: {n} particles")
        return ob


def build_weed_line(col, M, cam):
    """The weed line: a patchy floating mat (seen from afar and from below) plus detailed clumps near the camera."""
    ng, gi, go = new_group("WeedMatGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links
    B = NodeB(ng)
    t = gn_time(ng)
    g = nodes.new("GeometryNodeMeshGrid")             # u across (-1..1) in X, distance along in Y
    g.inputs["Size X"].default_value = 2.0
    g.inputs["Size Y"].default_value = 520.0
    g.inputs["Vertices X"].default_value = 11
    g.inputs["Vertices Y"].default_value = 1041
    u, v, _z = gn_xyz(ng)
    yy = nmath(ng, "ADD", v, 110.0)                    # world y from -150 to 370
    w = nmath(ng, "ADD", weed_half_width(yy, B), 0.8)
    xx = nmath(ng, "ADD", weed_x(yy, B), nmath(ng, "MULTIPLY", u, w))
    st = nodes.new("GeometryNodeStoreNamedAttribute")
    st.data_type = "FLOAT"
    st.domain = "POINT"
    links.new(g.outputs["Mesh"], st.inputs["Geometry"])
    st.inputs["Name"].default_value = "across"
    links.new(nmath(ng, "ABSOLUTE", u), st.inputs["Value"])
    sp = nodes.new("GeometryNodeSetPosition")
    links.new(st.outputs[0], sp.inputs["Geometry"])
    links.new(combine(ng, xx, yy, nmath(ng, "ADD", waves(B, xx, yy, t), 0.02)), sp.inputs["Position"])
    setm = nodes.new("GeometryNodeSetMaterial")
    links.new(sp.outputs[0], setm.inputs["Geometry"])
    setm.inputs["Material"].default_value = M.weed_mat
    sm = nodes.new("GeometryNodeSetShadeSmooth")
    links.new(setm.outputs[0], sm.inputs[0])
    links.new(sm.outputs[0], go.inputs["Geometry"])
    mat_ob = gn_object("WeedLineMat", col, ng)

    # Detailed clumps: six variants, scattered along the line, culled beyond 70 m of the lens
    rng = random.Random(23)
    lib = bpy.data.collections.new("WeedClumps")
    variants = []
    for k in range(6):
        bm = bmesh.new()
        for s in range(rng.randint(5, 8)):
            a = rng.uniform(0, 2 * math.pi)
            ln = rng.uniform(0.18, 0.38)
            p0 = Vector((0, 0, rng.uniform(-0.05, 0.0)))
            p1 = p0 + Vector((math.cos(a) * ln, math.sin(a) * ln, rng.uniform(-0.12, 0.02)))
            bm_frustum(bm, p0, p1, 0.006, 0.004, segs=4)
            for j in range(rng.randint(7, 11)):
                q = p0.lerp(p1, rng.uniform(0.15, 1.0))
                b = rng.uniform(0, 2 * math.pi)
                L, W = rng.uniform(0.04, 0.075), rng.uniform(0.008, 0.013)
                dirv = Vector((math.cos(b), math.sin(b), rng.uniform(-0.4, 0.3))).normalized()
                side = dirv.cross(Vector((0, 0, 1))).normalized() * W
                bm.faces.new([bm.verts.new(c) for c in (q, q + dirv * L * 0.5 + side, q + dirv * L, q + dirv * L * 0.5 - side)])
            for j in range(rng.randint(5, 9)):
                q = p0.lerp(p1, rng.uniform(0.1, 1.0)) + Vector((rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), -0.01))
                bmesh.ops.create_icosphere(bm, subdivisions=0, radius=rng.uniform(0.006, 0.01), matrix=Matrix.Translation(q))
        me = bpy.data.meshes.new(f"WeedClump{k}")
        bm.to_mesh(me)
        bm.free()
        me.materials.append(M.weed)
        ob = bpy.data.objects.new(f"WeedClump{k}", me)
        lib.objects.link(ob)
        variants.append(ob)
    pts = []
    for i in range(7000):
        y = rng.uniform(-60.0, 240.0)
        u = rng.gauss(0.0, 0.55)
        if abs(u) > 1.3:
            continue
        x = weed_x(y) + u * weed_half_width(y)
        pts.append((x, y, 0.0))
    me = bpy.data.meshes.new("WeedClumpPoints")
    me.from_pydata(pts, [], [])
    a = me.attributes.new("idx", "INT", "POINT")
    a.data.foreach_set("value", [rng.randrange(6) for _ in pts])
    a = me.attributes.new("yaw", "FLOAT", "POINT")
    a.data.foreach_set("value", [rng.uniform(0, 6.28) for _ in pts])
    a = me.attributes.new("scl", "FLOAT", "POINT")
    a.data.foreach_set("value", [rng.uniform(0.8, 1.7) for _ in pts])
    ng, gi, go = new_group("WeedClumpsGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links
    B = NodeB(ng)
    t = gn_time(ng)
    x, y, _z = gn_xyz(ng)
    oi = nodes.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = cam
    dist = vmath(ng, "DISTANCE", nodes.new("GeometryNodeInputPosition").outputs[0], oi.outputs["Location"])
    dele = nodes.new("GeometryNodeDeleteGeometry")
    dele.domain = "POINT"
    links.new(gi.outputs["Geometry"], dele.inputs["Geometry"])
    links.new(nmath(ng, "GREATER_THAN", dist, 70.0), dele.inputs["Selection"])
    sp = nodes.new("GeometryNodeSetPosition")
    links.new(dele.outputs[0], sp.inputs["Geometry"])
    x, y, _z = gn_xyz(ng)
    links.new(combine(ng, 0.0, 0.0, nmath(ng, "SUBTRACT", waves(B, x, y, t), 0.04)), sp.inputs["Offset"])
    ci = nodes.new("GeometryNodeCollectionInfo")
    ci.inputs["Collection"].default_value = lib
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    inst = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(sp.outputs[0], inst.inputs["Points"])
    links.new(ci.outputs[0], inst.inputs["Instance"])
    inst.inputs["Pick Instance"].default_value = True

    def nattr(nm, kind="FLOAT"):
        r = nodes.new("GeometryNodeInputNamedAttribute")
        r.data_type = kind
        r.inputs["Name"].default_value = nm
        return r.outputs[0]
    links.new(nattr("idx", "INT"), inst.inputs["Instance Index"])
    links.new(combine(ng, 0.0, 0.0, nattr("yaw")), inst.inputs["Rotation"])
    s = nattr("scl")
    links.new(combine(ng, s, s, s), inst.inputs["Scale"])
    links.new(inst.outputs[0], go.inputs["Geometry"])
    clumps = bpy.data.objects.new("WeedClumps", me)
    col.objects.link(clumps)
    clumps.modifiers.new("WeedClumps", "NODES").node_group = ng
    return mat_ob, clumps


BAIT_BALLS = [  # world centre, radius, count: small fish milling under the weed edge
    (Vector((-13.4, 6.0, -1.6)), 1.1, 160),
    (Vector((-15.6, 22.0, -2.2)), 1.4, 220),
    (Vector((-14.0, 48.0, -1.8)), 1.2, 180),
]


def build_baitfish(col, M):
    """Milling balls of baitfish: each fish circles the ball centre on its own tilted orbit."""
    fish = bpy.data.objects.new("BaitfishUnit", baitfish_mesh(M))
    objs = []
    for bi, (c, rad, n) in enumerate(BAIT_BALLS):
        rng = random.Random(70 + bi)
        pts, attrs = [], {"r": [], "ph": [], "w": [], "tilt": [], "h": []}
        for i in range(n):
            pts.append((0.0, 0.0, 0.0))
            attrs["r"].append(rad * rng.uniform(0.35, 1.0))
            attrs["ph"].append(rng.uniform(0, 6.28))
            attrs["w"].append(rng.uniform(0.9, 1.4) * (1 if rng.random() < 0.85 else -1))
            attrs["tilt"].append(rng.uniform(-0.35, 0.35))
            attrs["h"].append(rng.uniform(-0.6, 0.6) * rad)
        me = bpy.data.meshes.new(f"BaitBall{bi}")
        me.from_pydata(pts, [], [])
        for k, vals in attrs.items():
            a = me.attributes.new(k, "FLOAT", "POINT")
            a.data.foreach_set("value", vals)
        ng, gi, go = new_group(f"BaitBall{bi}GN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                               [("Geometry", "NodeSocketGeometry")])
        nodes, links = ng.nodes, ng.links

        def nattr(nm):
            r = nodes.new("GeometryNodeInputNamedAttribute")
            r.data_type = "FLOAT"
            r.inputs["Name"].default_value = nm
            return r.outputs[0]
        t = gn_time(ng)
        ang = nmath(ng, "MULTIPLY_ADD", t, nattr("w"), nattr("ph"))
        r = nattr("r")
        px = nmath(ng, "MULTIPLY", nmath(ng, "COSINE", ang), r)
        py = nmath(ng, "MULTIPLY", nmath(ng, "SINE", ang), r)
        pz = nmath(ng, "ADD", nattr("h"), nmath(ng, "MULTIPLY", px, nattr("tilt")))
        sp = nodes.new("GeometryNodeSetPosition")
        links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
        links.new(combine(ng, px, py, pz), sp.inputs["Position"])
        inst = nodes.new("GeometryNodeInstanceOnPoints")
        links.new(sp.outputs[0], inst.inputs["Points"])
        oi = nodes.new("GeometryNodeObjectInfo")
        oi.inputs["Object"].default_value = fish
        oi.inputs["As Instance"].default_value = True
        links.new(oi.outputs["Geometry"], inst.inputs["Instance"])
        heading = nmath(ng, "ADD", ang, nmath(ng, "MULTIPLY", nmath(ng, "SIGN", nattr("w")), math.pi / 2))
        links.new(combine(ng, 0.0, 0.0, heading), inst.inputs["Rotation"])
        links.new(inst.outputs[0], go.inputs["Geometry"])
        ob = bpy.data.objects.new(f"BaitBall{bi}", me)
        col.objects.link(ob)
        ob.location = c
        ob.modifiers.new("Mill", "NODES").node_group = ng
        objs.append(ob)
    return objs


# --------------------------------------------------------------------------
# People
# --------------------------------------------------------------------------
def figure(col, M, name, parent, seated=False):
    """A simple crew figure built from tapered limbs (boat space, feet at the root)."""
    root = empty_object(name, col, parent=parent, size=0.3)
    root.rotation_mode = "XYZ"
    parts = {}

    def part(nm, fn, mat, par=None):
        bm = bmesh.new()
        fn(bm)
        ob = mesh_object(f"{name}{nm}", bm, col, [mat], parent=par or root)
        for p in ob.data.polygons:
            p.use_smooth = True
        parts[nm] = ob
        return ob
    hip_z = 0.5 if seated else 0.92
    if seated:
        part("Legs", lambda bm: [bm_frustum(bm, (s * 0.1, 0.0, hip_z), (s * 0.1, 0.42, hip_z - 0.02), 0.075, 0.065, segs=8)
                                 for s in (-1, 1)] + [bm_frustum(bm, (s * 0.1, 0.42, hip_z), (s * 0.1, 0.46, 0.02), 0.06, 0.05, segs=8)
                                                      for s in (-1, 1)], M.shorts)
    else:
        part("Legs", lambda bm: [bm_frustum(bm, (s * 0.11, 0, 0.02), (s * 0.1, 0, hip_z), 0.06, 0.085, segs=8)
                                 for s in (-1, 1)], M.skin)
        part("Shorts", lambda bm: [bm_frustum(bm, (s * 0.1, 0, 0.5), (s * 0.1, 0, hip_z + 0.06), 0.09, 0.1, segs=8)
                                   for s in (-1, 1)], M.shorts)
    torso = empty_object(f"{name}TorsoPivot", col, loc=(0, 0, hip_z), parent=root, size=0.15)
    torso.rotation_mode = "XYZ"
    part("Torso", lambda bm: bm_frustum(bm, (0, 0, 0.0), (0, 0.0, 0.58), 0.17, 0.2, segs=10), M.shirt, torso)
    part("Head", lambda bm: bm_blob(bm, (0, 0.0, 0.74), (0.105, 0.115, 0.125), random.Random(3), jitter=0.02), M.skin, torso)
    part("Cap", lambda bm: (bm_frustum(bm, (0, 0.0, 0.8), (0, 0.0, 0.89), 0.118, 0.1, segs=10),
                            bm_box(bm, (0, 0.13, 0.81), (0.19, 0.13, 0.018))), M.cap, torso)
    part("Shades", lambda bm: bm_box(bm, (0, 0.1, 0.75), (0.17, 0.02, 0.04)), M.shades, torso)
    arms = {}
    for s, side in ((-1, "L"), (1, "R")):
        bm = bmesh.new()
        bm_frustum(bm, (0, 0, 0), (0, 1, 0), 0.05, 0.042, segs=8)
        arm = mesh_object(f"{name}Arm{side}", bm, col, [M.shirt], parent=torso)
        arm.location = (s * 0.2, 0.0, 0.52)
        arms[side] = arm
    return root, torso, arms


def aim_arm(arm, targets_influences):
    for tgt, infl in targets_influences:
        con = arm.constraints.new("STRETCH_TO")
        con.target = tgt
        con.rest_length = 1.0
        con.volume = "NO_VOLUME"
        con.influence = infl
    return [c for c in arm.constraints if c.type == "STRETCH_TO"]


# --------------------------------------------------------------------------
# Animated assembly: baits, lines, rods, fish, people, particles
# --------------------------------------------------------------------------
LINES_CLEARED = 1180          # the crew winds in the other three baits after the hook-up
ANGLER_TAKES_ROD = (1362, 1395)
FIGHT_SPOT = Vector((-1.1, -5.45, COCKPIT_Z))


def build_action(cols, M, boat_root, clips, paths):
    c_gear, c_life, c_people = cols
    sc = bpy.context.scene
    meshes = {"ballyhoo": ballyhoo_mesh(M), "skirt": skirt_lure_mesh(M)}
    lures, lure_pos = {}, {}
    for name, (_x, _y, kind, _p, _ph) in SPREAD.items():
        ob = bpy.data.objects.new(f"Bait{name}", meshes[kind])
        c_gear.objects.link(ob)
        ob.rotation_mode = "XYZ"
        ob.scale = (1.3, 1.3, 1.3)
        lures[name] = ob
        pos = {f: bait_world(name, f, paths) for f in range(1, FRAME_END + 1)}
        lure_pos[name] = pos
        for f in range(1, FRAME_END + 1):
            if name == "LongRigger" and f > STRIKE:
                fw = paths["Bull"][2][f]
                yaw, pitch = heading_angles(fw)
            else:
                dz = (pos[min(FRAME_END, f + 1)].z - pos[max(1, f - 1)].z) * FPS / 2
                yaw, pitch = math.pi / 2, -math.atan2(dz, boat_speed(f) + 0.5)
            ob.location = pos[f]
            ob.rotation_euler = (0.18 * math.sin(f * 0.7), pitch, yaw)
            key(ob, "location", f)
            key(ob, "rotation_euler", f)
        if name != "LongRigger":
            for f, h in ((1, False), (LINES_CLEARED, False), (LINES_CLEARED + 1, True)):
                ob.hide_render = h
                key(ob, "hide_render", f)

    # Rods: in their holders; the long-rigger rod bends hard at the strike and goes to the angler
    rods = {}
    for name in SPREAD:
        rods[name] = build_rod(c_gear, M, name, boat_root)
    clip_emp = {}
    for name, p in clips.items():
        clip_emp[name] = empty_object(f"Clip{name}", c_gear, loc=p, parent=boat_root, size=0.05)
    tips = {}
    for name, r in rods.items():
        butt, d = HOLDERS[name]
        tip_loc = {}
        for f in range(1, FRAME_END + 1):
            inv = boat_matrix(f).inverted()
            if name == "LongRigger" and f > STRIKE:
                fish_b = inv @ lure_pos[name][f]
                tug = 0.55 + 0.3 * smoothstep(STRIKE, STRIKE + 8, f) * (1 - smoothstep(1150, 1250, f)) \
                    + 0.12 * math.sin(f * 0.21) + 0.06 * math.sin(f * 0.53)
                if f >= ANGLER_TAKES_ROD[0]:
                    u = smoothstep(*ANGLER_TAKES_ROD, f)
                    lift_hand = Vector((-1.5, -3.95, COCKPIT_Z + 0.9))
                    fight_hand = FIGHT_SPOT + Vector((-0.05, 0.22, 0.84))
                    hand = lift_hand.lerp(fight_hand, smoothstep(1395, 1430, f))
                    b2 = butt.lerp(hand, u)
                    pump = 0.12 * math.sin(max(0, f - 1430) * 0.075)
                    to_fish = (fish_b - b2).normalized()
                    d2 = d.lerp((to_fish * 0.8 + Vector((0, 0, 0.62 + pump))).normalized(), u).normalized()
                    tip = place_rod(r, f, b2, d2, fish_b - b2, bend=tug)
                else:
                    tip = place_rod(r, f, butt, d, fish_b - butt, bend=tug)
            elif f in (1, FRAME_END) or f % 6 == 0 or (name == "LongRigger" and STRIKE - 6 <= f <= STRIKE + 1):
                tgt = clips[name]
                tip = place_rod(r, f, butt, d, tgt - butt, bend=0.12)
            else:
                continue
            tip_loc[f] = tip
        tips[name] = r["ctrl"]["Tip"]

    # Lines: rod tip -> release clip -> bait; the long rigger drops out of its clip at the strike
    line_objs = {}
    for name in SPREAD:
        sag = empty_object(f"Sag{name}", c_gear, size=0.05)
        for f in range(1, FRAME_END + 1, 2):
            a = boat_to_world(f, clips[name])
            b = lure_pos[name][f]
            L = (b - a).length
            sag.location = a.lerp(b, 0.55) + Vector((0, 0, -0.035 * L))
            key(sag, "location", f)
        l1 = line_object(f"Line{name}Rod", c_gear, M, tips[name], clip_emp[name], radius=0.0035)
        l2 = line_object(f"Line{name}Out", c_gear, M, clip_emp[name], lures[name], middle=sag, radius=0.0035)
        line_objs[name] = [l1, l2]
        end = STRIKE if name == "LongRigger" else LINES_CLEARED
        for ob in (l1, l2):
            for f, h in ((1, False), (end, False), (end + 1, True)):
                ob.hide_render = h
                key(ob, "hide_render", f)
    sag = empty_object("SagFight", c_gear, size=0.05)
    fight_line = line_object("LineFight", c_gear, M, tips["LongRigger"], lures["LongRigger"], middle=sag, radius=0.0035)
    for f, h in ((1, True), (STRIKE, True), (STRIKE + 1, False)):
        fight_line.hide_render = h
        key(fight_line, "hide_render", f)

    # Mahi: the bull and his school
    global SWIM
    SWIM = fish_swim_group()
    bull_me, mahi_me = mahi_mesh(M, True), mahi_mesh(M, False)
    fish_objs = {}
    rng = random.Random(11)
    for name, (length, world, fwd) in paths.items():
        ob = bpy.data.objects.new(name, bull_me if name == "Bull" else mahi_me)
        c_life.objects.link(ob)
        ob.scale = (length, length, length)
        ob.rotation_mode = "XYZ"
        mod = ob.modifiers.new("Swim", "NODES")
        mod.node_group = SWIM
        vals = {"Phase": rng.uniform(0, 6.28), "Amp": 0.06, "Rate": 9.0 if name == "Bull" else rng.uniform(7.0, 9.0)}
        for item in SWIM.interface.items_tree:
            if getattr(item, "in_out", None) == "INPUT" and item.name in vals:
                set_mod_input(mod, item.identifier, vals[item.name])
        yaws, pitches = [], []
        frames = list(range(1, FRAME_END + 1, 2)) + ([FRAME_END] if FRAME_END % 2 == 0 else [])
        for f in frames:
            yw, pt = heading_angles(fwd[f])
            yaws.append(yw)
            pitches.append(pt)
        yaws = unwrap(yaws)
        for f, yw, pt in zip(frames, yaws, pitches):
            ob.location = world[f]
            roll = 0.0
            if name == "Bull" and f > STRIKE:
                roll = 0.35 * math.sin(f * 0.31)            # head-shaking on the line
            ob.rotation_euler = (roll, pt, yw)
            key(ob, "location", f)
            key(ob, "rotation_euler", f)
        fish_objs[name] = ob
    for f in range(1, FRAME_END + 1, 6):            # mahi light up as they chase
        sc["mahi_lit"] = smoothstep(600, 760, f) * (1 - 0.3 * smoothstep(1300, 1500, f))
        sc.keyframe_insert(data_path='["mahi_lit"]', frame=f)

    # Particles: prop wash, bait bubble trails and splashes, the strike, the jump
    bub, spray = Particles(), Particles()
    prng = random.Random(4)
    for f in range(1, FRAME_END + 1):
        t = frame_time(f)
        k = boat_speed(f) / TROLL_SPEED
        for s in (-1, 1):
            p = boat_matrix(f) @ Vector((s * 0.62, -5.1, -0.95))
            for _ in range(3 if k > 0.5 else 1):
                bub.emit(f, p + Vector((prng.uniform(-0.15, 0.15), prng.uniform(-0.2, 0.2), prng.uniform(-0.1, 0.1))),
                         (prng.uniform(-0.3, 0.3), -1.2 * k + prng.uniform(-0.3, 0.3), prng.uniform(0.25, 0.6)),
                         prng.uniform(2.0, 4.0), prng.uniform(0.008, 0.03), -0.4)
        for name in SPREAD:
            if name != "LongRigger" and f > LINES_CLEARED:
                continue
            p, p0 = lure_pos[name][f], lure_pos[name][max(1, f - 1)]
            surf = wave_z(p.x, p.y, t)
            if p.z < surf - 0.04:
                for _ in range(3):
                    bub.emit(f, p + Vector((prng.uniform(-0.04, 0.04), -0.12 + prng.uniform(-0.05, 0.05), 0)),
                             (prng.uniform(-0.1, 0.1), prng.uniform(-0.2, 0.1), prng.uniform(0.1, 0.4)),
                             prng.uniform(1.2, 2.6), prng.uniform(0.005, 0.016), -0.6)
            popped = p0.z < wave_z(p0.x, p0.y, t) - 0.02 <= p.z
            skipping = p.z >= surf - 0.01 and f % 11 == 0 and (name != "LongRigger" or f <= STRIKE)
            for _ in range(10 if popped else (4 if skipping else 0)):
                if True:
                    spray.emit(f, p + Vector((0, -0.1, 0.02)),
                               (prng.uniform(-0.6, 0.6), prng.uniform(-1.4, 0.2), prng.uniform(0.6, 1.8)),
                               prng.uniform(0.35, 0.7), prng.uniform(0.01, 0.025), 9.81)
    bp = lure_pos["LongRigger"][STRIKE]
    for _ in range(140):
        bub.emit(STRIKE, bp + Vector((prng.uniform(-0.2, 0.2), prng.uniform(-0.3, 0.3), prng.uniform(-0.3, 0.0))),
                 (prng.uniform(-0.6, 0.6), prng.uniform(-0.6, 0.6), prng.uniform(0.0, 0.9)),
                 prng.uniform(1.0, 2.5), prng.uniform(0.006, 0.03), -0.8)
    for _ in range(90):
        spray.emit(STRIKE, bp, (prng.uniform(-1.4, 1.4), prng.uniform(-1.4, 1.4), prng.uniform(1.0, 3.2)),
                   prng.uniform(0.4, 0.9), prng.uniform(0.012, 0.035), 9.81)
    bull = paths["Bull"][1]
    for fj in (JUMP[0] + 2, JUMP[1] - 1):
        bp = bull[fj]
        for _ in range(240):
            spray.emit(fj, bp + Vector((0, 0, 0.1)), (prng.uniform(-1.6, 1.6), prng.uniform(-1.6, 1.6), prng.uniform(1.0, 3.6)),
                       prng.uniform(0.5, 1.0), prng.uniform(0.007, 0.022), 9.81)
    for f in range(1500, FRAME_END + 1, 16):     # thrashing at the transom
        bp = bull[f]
        for _ in range(18):
            spray.emit(f, bp + Vector((0, 0, 0.25)), (prng.uniform(-0.8, 0.8), prng.uniform(-0.8, 0.8), prng.uniform(0.6, 1.6)),
                       prng.uniform(0.3, 0.6), prng.uniform(0.01, 0.025), 9.81)
    bub_ob = bub.build("Bubbles", c_life, M.bubbles, above_water=False)
    spray_ob = spray.build("Spray", c_life, M.spray, above_water=True, streak=True)

    # Crew: the captain at the helm and the angler, who takes the rod for the fight
    cap_root, _t, cap_arms = figure(c_people, M, "Captain", boat_root, seated=True)
    cap_root.location = (-0.6, 0.85, 2.67)
    wheel = empty_object("Wheel", c_people, loc=(-0.6, 1.45, 3.35), parent=boat_root, size=0.05)
    for arm in cap_arms.values():
        aim_arm(arm, [(wheel, 1.0)])
    ang_root, ang_torso, ang_arms = figure(c_people, M, "Angler", boat_root)
    idle = {s: empty_object(f"AnglerIdle{s}", c_people, loc=(sx * 0.32, 0.12, 0.62), parent=ang_root, size=0.03)
            for s, sx in (("L", -1), ("R", 1))}
    mount = rods["LongRigger"]["mount"]
    grip = {"L": empty_object("AnglerGripFore", c_people, loc=(0.0, 0.32, 0.03), parent=mount, size=0.03),
            "R": empty_object("AnglerGripReel", c_people, loc=(0.15, -0.08, -0.03), parent=mount, size=0.03)}
    for side, arm in ang_arms.items():
        cons = aim_arm(arm, [(idle[side], 1.0), (grip[side], 0.0)])
        for f, w in ((1, 0.0), (ANGLER_TAKES_ROD[0] - 6, 0.0), (ANGLER_TAKES_ROD[0] + 8, 1.0)):
            cons[1].influence = w
            cons[1].keyframe_insert("influence", frame=f)
    spots = [(1, (-0.35, -4.9), math.radians(185)), (STRIKE, (-0.35, -4.9), math.radians(185)),
             (STRIKE + 40, (-1.45, -4.65), math.radians(150)), (ANGLER_TAKES_ROD[0], (-1.45, -4.65), math.radians(150)),
             (ANGLER_TAKES_ROD[1], (-1.45, -4.05), math.radians(120)), (1430, (FIGHT_SPOT.x, FIGHT_SPOT.y), math.radians(185)),
             (FRAME_END, (FIGHT_SPOT.x, FIGHT_SPOT.y), math.radians(182))]
    for f, (x, y), yaw in spots:
        ang_root.location = (x, y, COCKPIT_Z)
        ang_root.rotation_euler = (0, 0, yaw)
        key(ang_root, "location", f)
        key(ang_root, "rotation_euler", f)
    for f in range(1, FRAME_END + 1, 4):
        lean = 0.0
        if f > 1430:
            lean = math.radians(-14 + 8 * math.sin((f - 1430) * 0.075))     # pump and wind
        elif f > ANGLER_TAKES_ROD[0]:
            lean = math.radians(-8)
        ang_torso.rotation_euler = (lean, 0, 0)
        key(ang_torso, "rotation_euler", f)
    return dict(lures=lures, lure_pos=lure_pos, rods=rods, fish=fish_objs, lines=line_objs, fight_line=fight_line,
                angler=ang_root, captain=cap_root, bubbles=bub_ob, spray=spray_ob, sag_fight=sag)


def key_fight_sag(sag, rods, lures):
    sc = bpy.context.scene
    for f in range(STRIKE - 2, FRAME_END + 1, 2):
        sc.frame_set(f)
        a = rods["LongRigger"]["ctrl"]["Tip"].matrix_world.translation.copy()
        b = lures["LongRigger"].matrix_world.translation.copy()
        L = (b - a).length
        sag.location = a.lerp(b, 0.5) + Vector((0, 0, -0.02 * L))
        key(sag, "location", f)


# --------------------------------------------------------------------------
# Camera: two continuous moves; the hand-off from the strike to the cockpit is a cut hidden by a wipe
# --------------------------------------------------------------------------
def _school_local(f):
    p = spline(_resolve(SCHOOL_KEYS))[f]
    return (p.x, p.y, p.z)


def cam_seg1():
    """(frame, boat-space position, boat-space target): drone, weed line, dive, the spread, the school, the strike."""
    keys = [
        (1, (2.5, -31.0, 41.0), (0.0, -13.0, 0.0)),
        (110, (0.5, -28.5, 35.0), (-2.5, -9.5, 0.0)),
        (200, (-7.0, -30.5, 21.0), (-15.0, -14.0, 0.0)),
        (270, (-10.6, -36.5, 11.5), (-15.5, -24.0, -0.5)),
        (318, (-11.0, -40.5, 3.2), (-13.6, -31.0, -1.5)),
        (338, (-11.1, -42.0, -0.5), (-13.8, -33.0, -2.0)),
        (356, (-11.0, -42.6, -2.6), (-14.5, -33.0, -1.0)),
        (420, (-7.0, -45.8, -1.9), (-9.6, -43.0, -0.35)),
        (512, (-6.8, -45.4, -1.8), (-9.6, -42.8, -0.3)),
        (585, (-5.9, -33.4, -2.0), (-3.2, -30.2, -0.35)),
        (640, (-5.7, -32.8, -1.9), (-3.2, -29.8, -0.3)),
        (712, (-4.6, -46.4, -2.4), (-7.2, -40.5, -1.8)),      # drop back behind the school, still looking ahead
        (780, (-4.0, -47.6, -2.1), (-7.0, -42.0, -1.5)),
        (850, (-4.9, -50.0, -1.9), (-7.6, -44.5, -1.2)),
        (915, (-6.2, -49.0, -1.55), (-9.0, -45.0, -0.6)),
        (970, (-6.7, -46.2, -1.4), (-9.4, -44.0, -0.4)),
        (1010, (-6.9, -45.4, -1.3), (-9.5, -43.6, -0.35)),
        (1040, (-7.2, -45.1, -1.25), (-10.2, -43.6, -0.45)),
        (1062, (-7.3, -45.0, -1.0), (-10.4, -43.8, -0.3)),
    ]
    return keys


CUT = 1063
CAM_SEG2 = [   # cockpit (rides with the boat), then the crane pull-back
    (CUT, (-1.42, -3.70, 1.93), (-2.05, -3.20, 1.60)),
    (1200, (-1.36, -3.64, 1.98), (-2.05, -3.20, 1.62)),
    (1300, (-1.32, -3.72, 2.04), (-2.05, -3.17, 1.68)),
    (1350, (-1.10, -3.95, 2.30), (-2.00, -3.40, 1.85)),
    (1392, (-0.20, -4.70, 3.30), (-1.40, -5.20, 1.60)),
    (1445, (1.20, -8.20, 4.70), (-1.10, -6.40, 1.30)),
    (1560, (1.90, -13.0, 7.2), (-1.20, -6.90, 0.90)),
    (FRAME_END, (2.1, -21.0, 10.4), (-1.0, -6.0, 0.6)),
]
WIPES = [(330, 346), (CUT - 5, CUT + 4)]      # opaque from a-2 to b+2
CAM_SMOOTH = 7.0


def rigid_weight(f):
    if f < CUT:
        return 0.0
    return 1.0 - smoothstep(1400, 1530, f)


def camera_frames():
    """Hermite through the boat-space keys, Gaussian-smoothed in boat space (each side of the cut on
    its own), then placed in the world: towed with the boat, or riding it rigidly on board."""
    local = {}
    for keys, lo, hi in ((cam_seg1(), 1, CUT - 1), (CAM_SEG2, CUT, FRAME_END)):
        path = hermite_path([(f, p, t, False) for f, p, t in keys])
        for f in range(lo, hi + 1):
            local[f] = path[max(keys[0][0], min(keys[-1][0], f))]
    sigma, rad = CAM_SMOOTH, int(3 * CAM_SMOOTH)
    kern = [math.exp(-0.5 * (k / sigma) ** 2) for k in range(-rad, rad + 1)]
    out, q_prev = {}, None
    for f in range(1, FRAME_END + 1):
        lo, hi = (1, CUT - 1) if f < CUT else (CUT, FRAME_END)
        p, t, ws = Vector(), Vector(), 0.0
        for k, wgt in zip(range(-rad, rad + 1), kern):
            pp, tt = local[min(hi, max(lo, f + k))]
            p += pp * wgt
            t += tt * wgt
            ws += wgt
        p, t = p / ws, t / ws
        w = rigid_weight(f)
        pw = boat_to_world(f, p, rigid=False).lerp(boat_to_world(f, p, rigid=True), w)
        tw = boat_to_world(f, t, rigid=False).lerp(boat_to_world(f, t, rigid=True), w)
        q = look_rotation(pw, tw)
        if q_prev is not None and q.dot(q_prev) < 0:
            q.negate()
        q_prev = q
        out[f] = (pw, tw, q)
    return out


def build_camera(col, frames):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = LENS
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = SENSOR
    cd.clip_start = 0.03
    cd.clip_end = 30000.0
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
    return {f: 1.0 if p.z < wave_z(p.x, p.y, frame_time(f)) else 0.0 for f, (p, _t, _q) in frames.items()}


def key_under(frames, toggles):
    """cam_under drives fog, beams and snow; the backdrop and deep floor swap while a wipe is opaque."""
    sc = bpy.context.scene
    st = under_state(frames)
    for f in range(1, FRAME_END + 1):
        sc["cam_under"] = st[f]
        sc.keyframe_insert(data_path='["cam_under"]', frame=f)
    for ob, show_when_under in toggles:
        prev = None
        for f in range(1, FRAME_END + 1):
            vis = (st[f] > 0.5) == show_when_under
            if vis != prev:
                ob.hide_render = not vis
                key(ob, "hide_render", f)
                prev = vis
    return st


def crossings(frames, margin=0.45):
    near = [f for f, (p, _t, _q) in sorted(frames.items())
            if abs(p.z - wave_z(p.x, p.y, frame_time(f))) < margin]
    spans = []
    for f in near:
        if spans and f == spans[-1][1] + 1:
            spans[-1][1] = f
        else:
            spans.append([f, f])
    return spans


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------
def drag_gauge(hud, f_in, f_out, pct=25, y=0.06):
    """Dial: drag as a percentage of line strength, needle sweeping up to the strike setting."""
    g = hud.group((0.0, y, 0))
    r = 0.085
    objs = []
    plate = hud.shape(hud.rounded(0.30, 0.178, 0.03), INK, g, loc=(0, 0.034, -0.0004))   # dial, labels, title
    objs.append(plate)
    arc = []
    for i in range(41):
        a = math.radians(180 - 180 * i / 40)
        arc.append((r * math.cos(a), r * math.sin(a)))
    inner = [(x * 0.86, yy * 0.86) for x, yy in reversed(arc)]
    band = hud.shape(arc + inner, (0.30, 0.33, 0.36), g, loc=(0, 0.0, 0.0002))
    objs.append(band)
    lo, hi = 20, 30                          # Salt Water Sportsman: 20 to 30% depending on line class
    zone = []
    for i in range(11):
        a = math.radians(180 - 180 * (lo + (hi - lo) * i / 10) / 50)
        zone.append((r * math.cos(a), r * math.sin(a)))
    zone += [(x * 0.86, yy * 0.86) for x, yy in reversed(zone)]
    zo = hud.shape(zone, ORANGE, g, loc=(0, 0.0, 0.0004))
    objs.append(zo)
    for v in (0, 10, 20, 30, 40, 50):
        a = math.radians(180 - 180 * v / 50)
        lab = hud.text(f"{v}", "med", 0.011, g, loc=(1.2 * r * math.cos(a), 1.2 * r * math.sin(a), 0.001))
        objs.append(lab)
    title = hud.text("DRAG  % OF LINE STRENGTH", "semi", 0.0105, g, loc=(0, -0.03, 0.001), spacing=1.3)
    objs.append(title)
    needle_piv = hud.group((0, 0, 0))
    needle_piv.parent = g
    needle = hud.shape([(0, -0.003), (r * 0.92, 0), (0, 0.003)], ORANGE, needle_piv, loc=(0, 0, 0.0008))
    hub = hud.shape(hud.circle(0.008), WHITE, g, loc=(0, 0, 0.001))
    objs += [needle, hub]
    needle_piv.rotation_euler = (0, 0, math.radians(180))
    key(needle_piv, "rotation_euler", f_in + 6)
    needle_piv.rotation_euler = (0, 0, math.radians(180 - 180 * pct / 50))
    key(needle_piv, "rotation_euler", f_in + 30)
    hud.fade(objs, f_in, f_out)
    hud.events.append((f_in + 24, "gauge", f"{pct}%"))
    hud.boxes.append(("drag gauge", lambda f, a=f_in, b=f_out: (-0.15, y - 0.055, 0.15, y + 0.123)
                      if a - 4 <= f <= b + 4 else None))


def build_overlays(hud, frames, act, paths):
    lp, bull = act["lure_pos"], paths["Bull"][1]
    hud.scrims()
    # A: the weed line from the air
    hud.headline("OFFSHORE TROLLING", [("Find the weed line,", ["weed line,"]), ("find the mahi.", ["mahi."])], 1, 112)
    hud.badge(1, "THE WEED LINE", "floating sargassum", 122, 326)
    hud.pin(lambda f: Vector((weed_x(BOAT_Y[f] - 6.0), BOAT_Y[f] - 6.0, 0.1)), "weed line", 124, 206)
    hud.pin(lambda f: Vector((weed_x(BOAT_Y[f] - 15.0), BOAT_Y[f] - 15.0, 0.05)), "shade under the mat", 210, 252)
    hud.pin(BAIT_BALLS[1][0] + Vector((0, 0, 0.6)), "baitfish", 266, 328, dot_color=srgb(150, 230, 255))
    hud.caption("Many mahi gather near floating weed and drifting objects.", 130, 232, accents=("weed",))
    hud.caption("In a Gulf Stream survey, weed held far more small fish than open water.", 244, 330, accents=("small", "fish"))
    # B: the spread and the school, underwater
    hud.badge(2, "THE SPREAD", "skirted lures and rigged ballyhoo", 372, 612)
    hud.pin(lambda f: lp["LongRigger"][f] + Vector((0, 0, 0.16)), "rigged ballyhoo", 388, 512)
    hud.pin(lambda f: lp["LeftFlat"][f] + Vector((0, 0, 0.16)), "skirted lure", 520, 612)
    hud.caption("Two baits ride wide on the outriggers, two close on the flat lines.", 380, 498, accents=("outriggers,",))
    hud.caption("They skip on top and dive, trailing bubbles.", 510, 612, accents=("bubbles.",))
    hud.badge(3, "MAHI RISING", "the school comes up behind the baits", 640, 892)
    hud.pin(lambda f: bull[f] + Vector((0, 0, 0.45)), "bull mahi", 720, 838)
    hud.caption("Mahi feed at the surface on fish and crabs from the weed.", 652, 800, accents=("fish", "crabs"))
    # C: the strike underwater, then the cockpit
    hud.badge(4, "THE STRIKE", "the line pops from the release clip", 904, 1054)
    hud.pin(lambda f: lp["LongRigger"][f] + Vector((0, 0, 0.16)), "ballyhoo", 906, 968)
    hud.pin(lambda f: lp["LongRigger"][f] + Vector((0, 0.03, -0.03)), "hook point", 980, 1008, dot_color=ORANGE)
    hud.caption("When a fish strikes, the line pops free of the release clip.", 912, 1050, accents=("pops", "free"))
    hud.badge(5, "SET YOUR DRAG", "check it with a scale", 1080, 1340)
    hud.caption("That moment of slack lets the fish turn the bait.", 1088, 1202, accents=("slack",))
    hud.caption("Strike drag: about 25% of line strength on 30 to 50 lb mono.", 1214, 1342, accents=("25%",))
    drag_gauge(hud, 1214, 1342)
    # D: the fight and the release
    hud.badge(6, "RELEASE IT RIGHT", "NOAA Fisheries best practice", 1372, 1560)
    hud.caption("Unhook it in the water whenever you can.", 1380, 1466, accents=("water",))
    hud.caption("If it comes out, keep it under 60 seconds.", 1478, 1560, accents=("60", "seconds."))
    hud.caption("Wet hands, a soft net, a dehooker ready.", 1572, 1652, accents=("dehooker",))
    hud.headline("CATCH AND RELEASE", [("Keep it wet. Keep it quick.", ["wet.", "quick."]),
                                       ("Check your local limits.", ["limits."])], 1664, FRAME_END + 20, y=-0.20)
    for a, b in WIPES:
        hud.wipe(a, b)
    hud.depth_gauge([f for f, (p, _t, _q) in frames.items() if p.z < wave_z(p.x, p.y, frame_time(f))])


def hide_when_invisible(objs):
    """Switch objects off on every frame where their keyed opacity is zero. A fully transparent dithered
    surface can still leave marks where it sits at the same depth as another overlay, and hidden objects
    cost nothing to render."""
    n = 0
    for ob in objs:
        ad = ob.animation_data
        fc = ad.action.fcurves.find("color", index=3) if ad and ad.action and hasattr(ad.action, "fcurves") else None
        if fc is None and ad and ad.action:
            for layer in getattr(ad.action, "layers", []):          # Blender 5 layered actions
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


def gear_highlights(col, M, act, boat_parts):
    """Glowing outlines that pulse on the piece of gear being talked about."""
    shells = []
    lures, rods = act["lures"], act["rods"]
    s = highlight_shell(col, M, lures["LongRigger"], "HiBallyhoo", 0.012)
    pulse(s, 390, 512)
    pulse(s, 906, 1000)
    s2 = highlight_shell(col, M, lures["LeftFlat"], "HiSkirt", 0.012)
    pulse(s2, 520, 612)
    clip = next(p for p in boat_parts if p.name == "ReleaseClips")
    s3 = highlight_shell(col, M, clip, "HiClips", 0.02)
    pulse(s3, 150, 320)
    s4 = highlight_shell(col, M, rods["LongRigger"]["reel"], "HiReel", 0.006)
    pulse(s4, 1120, 1210)
    s5 = highlight_shell(col, M, rods["LongRigger"]["lever_arm"], "HiDragLever", 0.006)
    pulse(s5, 1214, 1340)
    shells += [s, s2, s3, s4, s5]
    return shells


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
    ee.shadow_resolution_scale = 1.0
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
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def run_checks(frames, st):
    sc = bpy.context.scene
    print("\n=== CAMERA REPORT (frame: pos | pitch | speed m/s | above wave | boat clearance) ===")
    dg = bpy.context.evaluated_depsgraph_get()
    boat_parts = [o for o in st["boat_parts"] if o.type == "MESH"]
    pts = []
    for ob in boat_parts:
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        mw = st["boat_root"].matrix_world.inverted() @ ob.matrix_world
        pts += [mw @ v.co for v in me.vertices]
        ev.to_mesh_clear()
    kd = kdtree.KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert(p, i)
    kd.balance()
    print("boat obstacle points:", len(pts))
    worst = {"boat": (1e9, 0), "fish": (1e9, 0, ""), "bait": (1e9, 0, ""), "angler": (1e9, 0), "rod": (1e9, 0, ""),
             "weed": (1e9, 0)}
    weed_pts = [Vector(v.co) for v in bpy.data.objects["WeedClumps"].data.vertices]
    kw = kdtree.KDTree(len(weed_pts))
    for i, p in enumerate(weed_pts):
        kw.insert(p, i)
    kw.balance()
    prev = None
    for f in range(1, FRAME_END + 1):
        p, _t, q = frames[f]
        fwd = q @ Vector((0, 0, -1))
        pitch = math.degrees(math.asin(max(-1, min(1, fwd.z))))
        speed = 0.0 if prev is None or f == CUT else (p - prev).length * FPS
        prev = p
        aw = p.z - wave_z(p.x, p.y, frame_time(f))
        local = boat_matrix(f).inverted() @ p
        d_boat = kd.find(local)[2]
        if d_boat < worst["boat"][0]:
            worst["boat"] = (d_boat, f)
        wp = kw.find(Vector((p.x, p.y, 0.0)))
        if wp[2] is not None:
            dz = abs(p.z - wave_z(wp[0].x, wp[0].y, frame_time(f)))
            dw = math.hypot(wp[2], dz)
            if dw < worst["weed"][0]:
                worst["weed"] = (dw, f)
        if f % 24 == 1 or abs(aw) < 0.45 or d_boat < 0.6 or f == FRAME_END:
            print(f"{f:5d}: ({p.x:6.1f},{p.y:7.1f},{p.z:6.2f}) | {pitch:6.1f} | {speed:5.2f} | {aw:6.2f} | {d_boat:5.2f}")
    for f in range(2, FRAME_END + 1, 2):
        p = frames[f][0]
        sc.frame_set(f)
        for name, ob in st["fish"].items():
            d = (ob.matrix_world.translation - p).length - 0.5 * ob.scale.x
            if d < worst["fish"][0]:
                worst["fish"] = (d, f, name)
        for name, ob in st["lures"].items():
            if not ob.hide_render:
                d = (ob.matrix_world.translation - p).length
                if d < worst["bait"][0]:
                    worst["bait"] = (d, f, name)
        if f >= 1000:
            for name, r in st["rods"].items():
                a = r["ctrl"]["Butt"].matrix_world.translation
                b = r["ctrl"]["Tip"].matrix_world.translation
                ab = b - a
                u = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
                d = (a + ab * u - p).length
                if d < worst["rod"][0]:
                    worst["rod"] = (d, f, name)
            for part in st["angler"].children_recursive:
                if part.type == "MESH":
                    d = (part.matrix_world.translation - p).length
                    if d < worst["angler"][0]:
                        worst["angler"] = (d, f)
    for k, v in worst.items():
        print(f"closest {k}: {tuple(round(x, 2) if isinstance(x, float) else x for x in v)}")
    # Motion smoothness: acceleration and turn rate (the cut is a deliberate jump inside a wipe)
    acc, turn = (0.0, 0), (0.0, 0)
    for f in range(3, FRAME_END):
        if f - 2 <= CUT <= f + 1:
            continue
        p0, p1, p2 = frames[f - 1][0], frames[f][0], frames[f + 1][0]
        a = ((p2 - p1) - (p1 - p0)).length * FPS * FPS
        if a > acc[0]:
            acc = (a, f)
        q0, q1 = frames[f - 1][2], frames[f][2]
        ang = math.degrees(q0.rotation_difference(q1).angle) * FPS
        if ang > turn[0]:
            turn = (ang, f)
    print(f"max camera acceleration: {acc[0]:.2f} m/s^2 at frame {acc[1]};  max turn rate: {turn[0]:.1f} deg/s at frame {turn[1]}")
    spans = crossings(frames)
    print("surface crossings (|lens - wave| < 0.45 m):", spans)
    for a, b in spans:
        covered = any(wa - 2 <= a and b <= wb + 2 for wa, wb in WIPES)
        print(f"   frames {a}-{b} hidden by a wipe: {covered}")
    print("cut at", CUT, "inside a wipe:", any(wa - 2 <= CUT - 1 and CUT <= wb + 2 for wa, wb in WIPES))
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
            if b and (b[0] < -HUD_W or b[2] > HUD_W or b[1] < -0.5 or b[3] > 0.5):
                offscreen.setdefault(n, []).append(f)
    print("overlays leaving the frame:", {k: (v[0], v[-1], len(v)) for k, v in offscreen.items()} or "none")


def write_events(path, st):
    import json
    fr = st["frames"]
    lp = st["lure_pos"]
    pops = {}
    for name in SPREAD:
        ev = []
        for f in range(2, FRAME_END + 1):
            p0, p = lp[name][f - 1], lp[name][f]
            if p0.z < wave_z(p0.x, p0.y, frame_time(f)) - 0.02 <= p.z and (name == "LongRigger" and f <= STRIKE or
                                                                              name != "LongRigger" and f <= LINES_CLEARED):
                ev.append(f)
        pops[name] = ev
    data = {
        "fps": FPS, "frames": FRAME_END,
        "camera_z": [round(fr[f][0].z, 3) for f in range(1, FRAME_END + 1)],
        "under": [int(st["under"][f]) for f in range(1, FRAME_END + 1)],
        "boat_speed": [round(boat_speed(f), 3) for f in range(1, FRAME_END + 1)],
        "cam_boat_dist": [round((fr[f][0] - Vector((0, BOAT_Y[f], 0))).length, 2) for f in range(1, FRAME_END + 1)],
        "wipes": WIPES, "cut": CUT,
        "overlays": sorted(st["hud"].events),
        "strike": STRIKE, "jump": list(JUMP), "lines_cleared": LINES_CLEARED,
        "angler_takes_rod": list(ANGLER_TAKES_ROD), "bait_pops": pops,
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
    for prop, val in (("cam_under", 0.0), ("wake_gain", 1.0), ("mahi_lit", 0.0)):
        sc[prop] = val
    c_env, c_boat, c_gear = collection("Environment"), collection("Boat"), collection("Tackle")
    c_life, c_people, c_cam = collection("SeaLife"), collection("Crew"), collection("CameraAndOverlays")
    wake_ref = empty_object("WakeRef", c_env, size=1.0)
    atmos = build_atmos_group()
    M = build_materials(atmos)
    M.highlight.use_backface_culling = True
    build_world()
    build_sun(c_env)
    paths = fish_paths()
    frames = camera_frames()
    paths = avoid_camera(paths, frames)
    cam = build_camera(c_cam, frames)
    ocean, far = build_ocean(c_env, M, cam)
    back, floor = build_depth_props(c_env, M, cam)
    beams = build_beams(c_env, M, cam)
    snow = build_snow(c_env, M)
    weed_mat, clumps = build_weed_line(c_env, M, cam)
    baits = build_baitfish(c_life, M)
    boat_root, boat_parts, clips = build_boat(c_boat, M)
    animate_boat(boat_root, wake_ref)
    act = build_action((c_gear, c_life, c_people), M, boat_root, clips, paths)
    key_fight_sag(act["sag_fight"], act["rods"], act["lures"])
    under = key_under(frames, [(back, True), (beams, True), (snow, True), (floor, False)])
    # While a wipe is opaque the lens may cross the surface: hide the sea then
    for ob in (ocean, far):
        ob.hide_render = False
        key(ob, "hide_render", 1)
        for a, b in WIPES:
            for f, hidden in ((a - 3, False), (a - 2, True), (b + 2, True), (b + 3, False)):
                ob.hide_render = hidden
                key(ob, "hide_render", f)
    shells = gear_highlights(c_gear, M, act, boat_parts)
    hud = Hud(cam, c_cam, M, frames)
    build_overlays(hud, frames, act, paths)
    n = hide_when_invisible([o for o in bpy.data.objects["HUD"].children_recursive if o.type in ("MESH", "FONT")] + shells)
    print("objects switched off while invisible:", n)
    sc.frame_set(1)
    return dict(frames=frames, hud=hud, under=under, boat_parts=boat_parts, boat_root=boat_root, fish=act["fish"],
                lures=act["lures"], lure_pos=act["lure_pos"], rods=act["rods"], angler=act["angler"])


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
    if args.save:
        sc = bpy.context.scene
        sc.frame_set(1)
        sc.render.filepath = "//gpu_render/frames/frame_"
        sc.render.use_overwrite = False
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save), compress=True)
        print("saved", args.save)
    sc = bpy.context.scene
    if args.frames:
        os.makedirs(args.outdir, exist_ok=True)
        for f in [int(x) for x in args.frames.split(",") if x.strip()]:
            sc.frame_set(f)
            sc.render.filepath = os.path.join(args.outdir, f"frame_{f:04d}.png")
            bpy.ops.render.render(write_still=True)
            print("rendered", sc.render.filepath, flush=True)
    if args.anim:
        os.makedirs(args.outdir, exist_ok=True)
        sc.frame_start, sc.frame_end = args.start, args.end
        sc.render.filepath = os.path.join(args.outdir, "frame_")
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
