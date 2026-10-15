"""Educational saltwater fishing breakdown, vertical 9:16 (Blender 5.x, EEVEE).

Run with the Blender binary:
    blender --background --python saltwater_fishing.py -- [options]
or with the bpy Python module (pip install bpy):
    python saltwater_fishing.py [options]

Options:
    --save PATH        save the generated scene as a .blend file
    --check            print camera clearance, crossing and overlay overlap reports
    --frames 1,300     render these frames as PNG stills into --outdir
    --anim             render the animation as PNG frames into --outdir
    --start N/--end N  frame range for --anim (default 1..1560)
    --outdir DIR       output directory for renders
    --scale PCT        resolution percentage (default 100 -> 1080x1920)
    --samples N        EEVEE render samples (default 16)
    --step N           frame step for --anim (default 1)
    --events PATH      write overlay and action timing as JSON (input for saltwater_audio.py)

Downloaded textures and models are read from assets/ (sources and licenses in assets/CREDITS.md).
The final render is meant for a GPU: see gpu_render/README.md.

World layout (metres, +Z up, +Y out to sea):
    * Beach along the X axis, waterline near y = 0, dunes inland (-Y).
    * Rock jetty along x = JETTY_X from the beach out to y = JETTY_Y1.
    * Sandbar near y = 27 with a rip channel cut through it at x = RIP_X;
      the trough (deeper water right behind the first wave) is near y = 14.
    * An angler stands on the beach and casts into the rip channel.

Beats (the brief's 0-15/15-40/40-60/60-70 s timeline scaled to 65 s):
    A  F1-334     aerial top-down, dolly along the jetty, dive to the tip school
    B  F334-891   rise, pan across the surf zone: trough, rip, the cast
    C  F891-1337  underwater close-up: bounce jig, steady retrieve, lift and drop
    D  F1337-1560 rise and pull back to a wide recap
"""

import argparse
import math
import os
import random
import sys

import bpy  # must come first: the pip bpy module exposes bmesh/mathutils after import
import bmesh
import numpy as np
from mathutils import Euler, Matrix, Vector, kdtree, noise

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")
ASSET_DIR = os.path.join(HERE, "assets")     # freely licensed downloads, see assets/CREDITS.md


def asset(*parts):
    return os.path.join(ASSET_DIR, *parts)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
FPS = 24
FRAME_END = 1560
RES_X, RES_Y = 1080, 1920
LENS = 24.0
SENSOR = 36.0          # vertical sensor fit
HUD_DIST = 0.25        # overlay plane distance in front of the lens
HUD_W = 0.5 * RES_X / RES_Y   # overlay half width in frame-height units

JETTY_X, JETTY_Y0, JETTY_Y1, JETTY_CREST = -20.0, -8.0, 62.0, 2.2
RIP_X, RIP_HALF = 16.0, 4.0
ANGLER = Vector((15.5, -4.0))
CAST_LAND = Vector((16.5, 22.0))
WATER_CENTER = (0.0, 30.0)
FINE_HALF, COARSE_HALF = 108.0, 702.0
CURTAIN_R = 100.0
UW_FOG_DIST = 14.0
AIR_HAZE_DIST = 700.0
WAVE_L, WAVE_V = 16.0, 3.2

TIP_SCHOOL = Vector((-19.0, 67.6, -2.9))
POCKETS = {  # world anchors of the structural fish-holding pockets
    "inside pocket": Vector((-15.6, 14.0, 0.3)),
    "bar gap": Vector((-15.8, 28.0, 0.3)),
    "tip eddy": Vector((-17.0, 66.0, 0.3)),
}


def srgb(r, g, b):
    def lin(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (lin(r), lin(g), lin(b))


ORANGE = srgb(245, 178, 76)
WHITE = (1.0, 1.0, 1.0)
INK = srgb(24, 30, 36)
PIN_RED = srgb(240, 92, 80)
WIPE_AQUA = srgb(158, 222, 216)
TROUGH_BLUE = srgb(120, 196, 255)
DASH_YELLOW = srgb(246, 206, 92)
SKY_TOP, SKY_HORIZON = srgb(150, 192, 222), srgb(226, 234, 236)
HAZE = srgb(214, 228, 232)
FOG_DEEP, FOG_SHALLOW = srgb(28, 108, 124), srgb(96, 182, 186)


def smoothstep(e0, e1, x):
    if e0 > e1:
        return 1.0 - smoothstep(e1, e0, x)
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def frame_time(f):
    return f / FPS


# --------------------------------------------------------------------------
# Expression backends: one formula drives both Python (checks, placement)
# and geometry nodes (the animated water), so they cannot drift apart.
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


def rip_mask(B, x):
    return B.ss(RIP_HALF + 3.0, RIP_HALF, B.abs(B.sub(x, RIP_X)))


def bed(B, x, y):
    """Sea floor / beach height."""
    rip = rip_mask(B, x)
    lin = B.mul(y, -0.06)
    trough = B.mul(B.exp(B.mul(B.pow(B.abs(B.div(B.sub(y, 14.0), 6.0)), 2.0), -1.0)), -1.4)
    bar = B.mul(B.exp(B.mul(B.pow(B.abs(B.div(B.sub(y, 27.0), 4.5)), 2.0), -1.0)), B.sub(1.0, rip))
    chan = B.mul(B.ss(8.0, 18.0, y), B.ss(48.0, 36.0, y), rip, -0.8)
    off = B.mul(B.ss(40.0, 110.0, y), -2.0)
    dune = B.mul(B.ss(-30.0, -50.0, y), 1.4)
    shore = B.mul(y, B.exp(B.mul(B.pow(B.abs(B.div(y, 3.0)), 2.0), -1.0)), -0.12)   # steeper swash slope
    return B.add(lin, trough, bar, chan, off, dune, shore)


def waves(B, x, y, t):
    """Returns (height, crest 0..1, rip mask) of the near-shore wave field."""
    yy = B.add(y, B.mul(B.sin(B.add(B.mul(x, 0.06), 0.8)), 1.5))
    s = B.fract(B.div(B.add(yy, B.mul(t, WAVE_V)), WAVE_L))
    p = B.pow(B.sub(0.5, B.mul(B.cos(B.mul(s, 2.0 * math.pi)), 0.5)), 3.0)
    rip = rip_mask(B, x)
    grow = B.ss(70.0, 30.0, y)
    brk = B.ss(22.0, 29.0, y)
    a_norm = B.add(B.mul(B.add(0.40, B.mul(grow, 0.65)), brk),
                   B.mul(0.28, B.sub(1.0, brk), B.ss(0.5, 6.0, y)))
    a_rip = B.mul(B.add(0.40, B.mul(grow, 0.25)), B.ss(4.0, 14.0, y))
    amp = B.lerp(a_norm, a_rip, rip)
    chop = B.add(B.mul(B.sin(B.add(B.mul(x, 0.9), B.mul(y, 0.5), B.mul(t, -1.7))), 0.06),
                 B.mul(B.sin(B.add(B.mul(x, -0.7), B.mul(y, 1.1), B.mul(t, -2.1))), 0.05))
    h = B.add(B.mul(amp, B.sub(p, 0.3125)), B.mul(chop, B.ss(-1.0, 5.0, y)))
    return h, p, rip


def foam(B, x, y, t, p, rip):
    crest = B.pow(p, 1.4)
    inside = B.mul(B.ss(31.0, 26.0, y), B.ss(9.0, 14.0, y), B.add(0.35, B.mul(B.ss(9.0, 26.0, y), 0.65)))
    on_bar = B.mul(B.pow(p, 2.0), B.ss(34.0, 29.0, y), B.ss(24.0, 27.5, y))
    surf = B.mul(B.add(B.mul(crest, inside), on_bar), B.sub(1.0, B.mul(rip, 0.85)))
    swash = B.mul(B.ss(3.5, 0.8, y), B.ss(-1.5, 0.3, y),
                  B.add(0.55, B.mul(B.sin(B.add(B.mul(t, 1.3), B.mul(x, 0.15))), 0.45)))
    dj = B.sub(B.abs(B.sub(x, JETTY_X)), 4.0)
    jet = B.mul(B.ss(1.4, -0.4, dj), B.ss(-2.0, 1.0, y), B.ss(70.0, 64.0, y), B.add(0.3, B.mul(crest, 0.55)))
    return B.clamp(B.add(surf, swash, jet))


def wave_fade(x, y):
    cheb = max(abs(x - WATER_CENTER[0]), abs(y - WATER_CENTER[1]))
    return 1.0 - smoothstep(90.0, 105.0, cheb)


PY = PyB()


def wave_z(x, y, t):
    return waves(PY, x, y, t)[0] * wave_fade(x, y)


def bed_z(x, y):
    return bed(PY, x, y)


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------
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
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--events", default=None, help="write overlay/action timing as JSON (for the soundtrack)")
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


def uv_scaled(nt, scale):
    mp = nt.nodes.new("ShaderNodeMapping")
    nt.links.new(nt.nodes.new("ShaderNodeUVMap").outputs["UV"], mp.inputs["Vector"])
    mp.inputs["Scale"].default_value = (scale, scale, 1.0)
    return mp.outputs[0]


def wrap_atmos(mat, atmos):
    """Insert the air-haze / underwater-fog group in front of an imported material's output."""
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    src = out.inputs["Surface"].links[0].from_socket
    g = nt.nodes.new("ShaderNodeGroup")
    g.node_tree = atmos
    nt.links.new(src, g.inputs["Shader"])
    nt.links.new(g.outputs["Shader"], out.inputs["Surface"])


# --------------------------------------------------------------------------
# Shading
# --------------------------------------------------------------------------
def build_sky_group():
    ng, gi, go = new_group("SkyColor", "ShaderNodeTree", [("Direction", "NodeSocketVector")],
                           [("Sky", "NodeSocketColor")])
    d = vmath(ng, "NORMALIZE", gi.outputs["Direction"])
    z = separate(ng, d)["Z"]
    sky = color_ramp(ng, z, [(0.0, SKY_HORIZON), (0.12, srgb(205, 222, 232)), (0.45, srgb(168, 202, 226)),
                             (1.0, SKY_TOP)])
    sun = nmath(ng, "POWER", nmath(ng, "MAXIMUM", vmath(ng, "DOT_PRODUCT", d, tuple(SUN_DIR)), 0.0), 60.0)
    sky = vmath(ng, "ADD", sky, vmath(ng, "SCALE", srgb(255, 244, 222), scale=nmath(ng, "MULTIPLY", sun, 0.6)))
    below = nmath(ng, "LESS_THAN", z, 0.0)
    ng.links.new(mix_rgb(ng, below, sky, HAZE), go.inputs["Sky"])
    return ng


def build_atmos_group():
    """Air haze when the camera is above water, blue-green fog below it.
    scene["cam_under"] (keyed per frame) says which side of the waves the lens is on."""
    ng, gi, go = new_group("Atmosphere", "ShaderNodeTree", [("Shader", "NodeSocketShader")],
                           [("Shader", "NodeSocketShader")])
    geo = ng.nodes.new("ShaderNodeNewGeometry")
    dist = ng.nodes.new("ShaderNodeCameraData").outputs["View Distance"]
    inc = separate(ng, geo.outputs["Incoming"])
    a = ng.nodes.new("ShaderNodeAttribute")
    a.attribute_type = "VIEW_LAYER"
    a.attribute_name = "cam_under"
    under = a.outputs["Factor"]
    above = nmath(ng, "SUBTRACT", 1.0, under)
    fog = nmath(ng, "MULTIPLY", under,
                nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(ng, "MULTIPLY", dist, -1.0 / UW_FOG_DIST))))
    haze = nmath(ng, "MULTIPLY", above,
                 nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(ng, "MULTIPLY", dist, -1.0 / AIR_HAZE_DIST))))
    em_h = ng.nodes.new("ShaderNodeEmission")
    em_h.inputs["Color"].default_value = (*HAZE, 1.0)
    em_f = ng.nodes.new("ShaderNodeEmission")
    ng.links.new(mix_rgb(ng, nmath(ng, "MULTIPLY_ADD", inc["Z"], -0.5, 0.5, clamp=True), FOG_DEEP, FOG_SHALLOW),
                 em_f.inputs["Color"])
    m1 = ng.nodes.new("ShaderNodeMixShader")
    ng.links.new(haze, m1.inputs[0])
    ng.links.new(gi.outputs["Shader"], m1.inputs[1])
    ng.links.new(em_h.outputs[0], m1.inputs[2])
    m2 = ng.nodes.new("ShaderNodeMixShader")
    ng.links.new(fog, m2.inputs[0])
    ng.links.new(m1.outputs[0], m2.inputs[1])
    ng.links.new(em_f.outputs[0], m2.inputs[2])
    ng.links.new(m2.outputs[0], go.inputs["Shader"])
    return ng


def caustics(nt, scale=1.4):
    """Animated caustic pattern (0..1); time comes from a frame driver."""
    v = nt.nodes.new("ShaderNodeTexVoronoi")
    v.voronoi_dimensions = "4D"
    v.feature = "F1"
    v.inputs["Scale"].default_value = scale
    nt.links.new(nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"], v.inputs["Vector"])
    add_frame_driver(v.inputs["W"], "default_value", "frame*0.025")
    return nmath(nt, "POWER", nmath(nt, "SUBTRACT", 1.0, v.outputs["Distance"], clamp=True), 6.0)


def uw_shade(nt, color):
    """Stylised underwater lighting: soft light from above plus caustics."""
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    nz = vmath(nt, "DOT_PRODUCT", geo.outputs["Normal"], tuple(Vector((0.15, -0.2, 1.0)).normalized()))
    lit = nmath(nt, "MULTIPLY_ADD", nmath(nt, "MAXIMUM", nz, 0.0), 0.62, 0.42)
    lit = nmath(nt, "ADD", lit, nmath(nt, "MULTIPLY", caustics(nt), nmath(nt, "MAXIMUM", nz, 0.0)))
    col = vmath(nt, "MULTIPLY", vmath(nt, "SCALE", color, scale=lit), (0.86, 1.0, 0.98))
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(col, em.inputs["Color"])
    return em.outputs[0]


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


def build_materials(atmos):
    M = Materials()

    def obj_rgb(nt):
        return nt.nodes.new("ShaderNodeObjectInfo").outputs["Color"]

    def hybrid(color_fn, rough=0.8):
        def build(nt):
            c = color_fn(nt)
            z = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"])["Z"]
            mix = nt.nodes.new("ShaderNodeMixShader")
            nt.links.new(map_range(nt, z, -0.45, -0.05), mix.inputs[0])
            nt.links.new(uw_shade(nt, c), mix.inputs[1])
            nt.links.new(principled(nt, c, rough), mix.inputs[2])
            return mix.outputs[0]
        return build

    M.terrain = new_material("Terrain", hybrid(lambda nt: attr(nt, "Col")), atmos)
    M.rock = new_material("JettyRock", hybrid(lambda nt: attr(nt, "Col"), 0.85), atmos)
    M.painted = new_material("Painted", lambda nt: principled(nt, obj_rgb(nt), 0.7), atmos)
    M.fish = new_material("Fish", lambda nt: uw_shade(nt, vmath(nt, "MULTIPLY", attr(nt, "Col"), obj_rgb(nt))), atmos)
    M.uw_obj = new_material("UnderwaterObj", lambda nt: uw_shade(nt, obj_rgb(nt)), atmos)

    def line(nt):
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (0.95, 0.95, 0.92, 1.0)
        em.inputs["Strength"].default_value = 0.9
        return em.outputs[0]
    M.line = new_material("FishingLine", line, atmos)

    def rock(nt):
        uv = uv_scaled(nt, 1.0)
        col = image_node(nt, asset("babylon", "rock_baseColor.jpg"), False, uv).outputs["Color"]
        nrm = image_node(nt, asset("babylon", "rock_normal.jpg"), True, uv).outputs["Color"]
        orm = separate(nt, image_node(nt, asset("babylon", "rock_orm.jpg"), True, uv).outputs["Color"])
        z = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"])["Z"]
        wet = map_range(nt, z, 0.7, 0.0)                       # darker, glossier near and below the waterline
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        col = vmath(nt, "SCALE", col, scale=1.35)               # the source texture is authored for a dim underwater scene
        nt.links.new(mix_rgb(nt, nmath(nt, "MULTIPLY", wet, 0.3), col, srgb(40, 44, 40)), b.inputs["Base Color"])
        nt.links.new(nmath(nt, "MULTIPLY", orm["Y"], nmath(nt, "SUBTRACT", 1.0, nmath(nt, "MULTIPLY", wet, 0.45))),
                     b.inputs["Roughness"])
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(nrm, nm.inputs["Color"])
        nt.links.new(nm.outputs[0], b.inputs["Normal"])
        return b.outputs[0]
    M.rock_pbr = new_material("JettyRockPBR", rock, atmos)

    def ground(nt):
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        z = separate(nt, geo.outputs["Position"])["Z"]
        sand = image_node(nt, asset("babylon", "sand.jpg"), False, uv_scaled(nt, 1.3)).outputs["Color"]
        guv = uv_scaled(nt, 1.0)
        gcol = image_node(nt, asset("babylon", "ground_baseColor.jpg"), False, guv).outputs["Color"]
        gnrm = image_node(nt, asset("babylon", "ground_normal.jpg"), True, guv).outputs["Color"]
        gorm = separate(nt, image_node(nt, asset("babylon", "ground_orm.jpg"), True, guv).outputs["Color"])
        under = map_range(nt, z, 0.1, -0.7)
        wet = nmath(nt, "MULTIPLY", map_range(nt, z, 0.7, 0.05), nmath(nt, "SUBTRACT", 1.0, under))
        col = mix_rgb(nt, under, sand, gcol)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", wet, 0.4), col, srgb(120, 100, 72))
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        nt.links.new(mix_rgb(nt, under, mix_rgb(nt, wet, (0.92, 0.92, 0.92), (0.45, 0.45, 0.45)),
                             combine(nt, gorm["Y"], gorm["Y"], gorm["Y"])), b.inputs["Roughness"])
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(gnrm, nm.inputs["Color"])
        nt.links.new(under, nm.inputs["Strength"])
        bump = nt.nodes.new("ShaderNodeBump")
        bw = nt.nodes.new("ShaderNodeRGBToBW")
        nt.links.new(sand, bw.inputs[0])
        nt.links.new(bw.outputs[0], bump.inputs["Height"])
        nt.links.new(nmath(nt, "MULTIPLY", nmath(nt, "SUBTRACT", 1.0, under), 0.25), bump.inputs["Strength"])
        nt.links.new(nm.outputs[0], bump.inputs["Normal"])
        nt.links.new(bump.outputs[0], b.inputs["Normal"])
        caus = nmath(nt, "MULTIPLY", caustics(nt), nmath(nt, "MULTIPLY", under, 0.18))
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(vmath(nt, "SCALE", (0.85, 1.0, 0.95), scale=caus), em.inputs["Color"])
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(b.outputs[0], add.inputs[0])
        nt.links.new(em.outputs[0], add.inputs[1])
        return add.outputs[0]
    M.ground = new_material("SandAndSeabed", ground, atmos)

    def water(nt):
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        depth = attr(nt, "depth", "Factor")
        f = attr(nt, "foam", "Factor")
        base = color_ramp(nt, nmath(nt, "DIVIDE", depth, 8.0), [
            (0.0, srgb(170, 226, 214)), (0.08, srgb(120, 206, 202)), (0.22, srgb(72, 172, 196)),
            (0.35, srgb(56, 146, 190)), (0.6, srgb(44, 118, 166)), (1.0, srgb(34, 92, 138))])
        # Where the water turns opaque its colour is sunlit like paint; darken it with depth so deep water
        # (the rip channel, the open sea) reads darker than the see-through shallows, as it does in reality.
        base = vmath(nt, "SCALE", base, scale=map_range(nt, depth, 1.0, 3.5, 1.0, 0.45))
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.noise_dimensions = "4D"
        nz.inputs["Scale"].default_value = 0.9
        nz.inputs["Detail"].default_value = 4.0
        nt.links.new(geo.outputs["Position"], nz.inputs["Vector"])
        add_frame_driver(nz.inputs["W"], "default_value", "frame*0.02")
        m = map_range(nt, nmath(nt, "SUBTRACT", f, nmath(nt, "MULTIPLY_ADD", nz.outputs["Fac"], 2.0, -0.4)), 0.0, 0.18)
        col = mix_rgb(nt, m, base, srgb(214, 226, 226))   # kept below clipping so it cannot glare through HUD plates
        # Ripples: two scrolling samples of the downloaded water normal map, applied around +Z
        pos = separate(nt, geo.outputs["Position"])

        def ripple(scale, dx, dy):
            mp = nt.nodes.new("ShaderNodeMapping")
            nt.links.new(combine(nt, pos["X"], pos["Y"], 0.0), mp.inputs["Vector"])
            mp.inputs["Scale"].default_value = (1.0 / scale, 1.0 / scale, 1.0)
            add_frame_driver(mp.inputs["Location"], "default_value", f"frame*{dx}", index=0)
            add_frame_driver(mp.inputs["Location"], "default_value", f"frame*{dy}", index=1)
            return image_node(nt, asset("babylon", "waterbump.png"), True, mp.outputs[0]).outputs["Color"]
        rip = separate(nt, vmath(nt, "SUBTRACT", mix_rgb(nt, 0.5, ripple(5.0, 0.0011, 0.0023),
                                                          ripple(1.7, -0.0019, 0.0013)), (0.5, 0.5, 0.5)))
        tilt = nmath(nt, "MULTIPLY", nmath(nt, "SUBTRACT", 1.0, m), 0.7)
        normal = vmath(nt, "NORMALIZE", vmath(nt, "ADD", geo.outputs["Normal"],
                                              combine(nt, nmath(nt, "MULTIPLY", rip["X"], tilt),
                                                      nmath(nt, "MULTIPLY", rip["Y"], tilt), 0.0)))
        top = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, top.inputs["Base Color"])
        nt.links.new(normal, top.inputs["Normal"])
        nt.links.new(map_range(nt, m, 0.0, 1.0, 0.05, 0.6, smooth=False), top.inputs["Roughness"])
        top.inputs["IOR"].default_value = 1.333
        top.inputs["Specular IOR Level"].default_value = 0.5
        clear = nmath(nt, "MULTIPLY", nmath(nt, "SUBTRACT", 1.0, m), map_range(nt, depth, 0.3, 3.5, 0.9, 0.0))
        nt.links.new(clear, top.inputs["Transmission Weight"])      # see-through shallows, opaque depths
        nt.links.new(col, top.inputs["Emission Color"])
        top.inputs["Emission Strength"].default_value = 0.08
        under = vmath(nt, "SCALE", mix_rgb(nt, nmath(nt, "MULTIPLY", m, 0.5), srgb(150, 222, 218), srgb(205, 238, 234)),
                      scale=nmath(nt, "MULTIPLY_ADD", caustics(nt, 0.9), 0.35, 0.85))
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(under, em.inputs["Color"])
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(geo.outputs["Backfacing"], mix.inputs[0])
        nt.links.new(top.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    M.water = new_material("SurfWater", water, atmos)
    for prop, val in (("use_raytrace_refraction", True), ("thickness_mode", "SLAB")):
        try:
            setattr(M.water, prop, val)
        except (AttributeError, TypeError):
            print("note: material property not available:", prop)

    def seagrass(nt):
        uv = nt.nodes.new("ShaderNodeUVMap").outputs["UV"]
        im = image_node(nt, asset("babylon", "seagrass_card.png"), False, uv)
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(vmath(nt, "MULTIPLY", im.outputs["Color"], srgb(150, 200, 105)), b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.6
        g = nt.nodes.new("ShaderNodeGroup")
        g.node_tree = atmos
        nt.links.new(b.outputs[0], g.inputs["Shader"])
        mix = nt.nodes.new("ShaderNodeMixShader")       # cut-out after the fog, so clear areas stay clear
        nt.links.new(im.outputs["Alpha"], mix.inputs[0])
        nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[1])
        nt.links.new(g.outputs["Shader"], mix.inputs[2])
        return mix.outputs[0]
    M.seagrass = new_material("Seagrass", seagrass)

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

    def scrim(band):
        def build(nt):
            oi = nt.nodes.new("ShaderNodeObjectInfo")
            y = separate(nt, nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"])["Y"]
            if band:   # darkest along the middle of the strip
                prof = nmath(nt, "SUBTRACT", 1.0, nmath(nt, "ABSOLUTE", nmath(nt, "MULTIPLY_ADD", y, 2.0, -1.0)))
            else:      # darkest at the top edge of the frame
                prof = nmath(nt, "POWER", y, 1.6)
            tr = nt.nodes.new("ShaderNodeBsdfTransparent")
            em = nt.nodes.new("ShaderNodeEmission")
            em.inputs["Color"].default_value = (0.0, 0.02, 0.03, 1.0)
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
    M.hud_shadow = new_material("HudShadow", emissive_alpha((0.0, 0.0, 0.0), 0.45), blended=True)
    for m in (M.gfx, M.hud, M.hud_white, M.hud_accent, M.hud_shadow):
        try:
            m.use_transparency_overlap = True
        except AttributeError:
            pass
    return M


SUN_DIR = Vector((0.55, 0.2, 0.81)).normalized()


def build_world(sky_group):
    world = bpy.data.worlds.new("PastelSky")
    bpy.context.scene.world = world
    try:
        world.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = world.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new("ShaderNodeGroup")
    sky.node_tree = sky_group
    nt.links.new(nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"], sky.inputs["Direction"])
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(sky.outputs["Sky"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.15
    nt.links.new(bg.outputs[0], nt.nodes.new("ShaderNodeOutputWorld").inputs["Surface"])
    world.sun_threshold = 1000.0


def build_sun(col):
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 2.6
    light.color = srgb(255, 246, 232)
    light.angle = math.radians(3.0)
    ob = bpy.data.objects.new("Sun", light)
    col.objects.link(ob)
    ob.rotation_euler = Vector((0, 0, -1)).rotation_difference(-SUN_DIR).to_euler()
    return ob


# --------------------------------------------------------------------------
# Terrain, water, jetty
# --------------------------------------------------------------------------
def axis_coords(lo, dlo, dhi, hi, coarse, fine):
    return (list(np.arange(lo, dlo, coarse)) + list(np.arange(dlo, dhi, fine)) +
            list(np.arange(dhi, hi + 1e-6, coarse)))


def terrain_z(x, y):
    z = bed_z(x, y)
    z += 0.05 * smoothstep(1.5, 4.0, y) * smoothstep(70.0, 55.0, y) * math.sin(y * 4.2 + 0.6 * math.sin(x * 0.3))
    z += 0.9 * smoothstep(-28.0, -45.0, y) * noise.noise(Vector((x * 0.04, y * 0.04, 1.3)))
    return z


def build_terrain(col, M):
    rng = random.Random(4)
    xs = axis_coords(-160.0, -32.0, 32.0, 160.0, 3.0, 0.4)
    ys = axis_coords(-150.0, -14.0, 54.0, 170.0, 3.0, 0.4)
    bm = bmesh.new()
    layer = bm.loops.layers.float_color.new("Col")
    grid = [[bm.verts.new((float(x), float(y), terrain_z(float(x), float(y)))) for x in xs] for y in ys]
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
    bmesh.ops.triangulate(bm, faces=bm.faces)
    uv_layer = bm.loops.layers.uv.new("UVMap")             # world-aligned UVs: 1 UV unit = 3 m
    for f in bm.faces:
        for loop in f.loops:
            loop[uv_layer].uv = (loop.vert.co.x / 3.0, loop.vert.co.y / 3.0)
    dry, wet, sea = srgb(232, 214, 178), srgb(198, 178, 142), srgb(214, 198, 160)
    for f in bm.faces:
        c = f.calc_center_median()
        v = 1.0 + 0.04 * noise.noise(Vector((c.x * 0.3, c.y * 0.3, 0.0)))
        if c.z > 0.45:
            base = dry
        elif c.z > -0.35:
            base = wet
        else:
            base = sea
        color_faces([f], layer, tuple(min(1.0, ch * v) for ch in base))
    ob = mesh_object("Beach", bm, col, [M.ground])

    # Dune grass tufts
    bm = bmesh.new()
    layer = bm.loops.layers.float_color.new("Col")
    for _ in range(1400):
        x, y = rng.uniform(-70, 70), rng.uniform(-60, -30)
        if noise.noise(Vector((x * 0.08, y * 0.08, 7.0))) < -0.05:
            continue
        z = terrain_z(x, y)
        for _k in range(rng.randint(3, 6)):
            d = Vector((rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), 1.0)).normalized()
            base = Vector((x + rng.uniform(-0.4, 0.4), y + rng.uniform(-0.4, 0.4), z - 0.05))
            fs = bm_frustum(bm, base, base + d * rng.uniform(0.6, 1.1), 0.06, 0.0, segs=3)
            color_faces(fs, layer, rng.choice([srgb(132, 158, 96), srgb(160, 176, 110), srgb(120, 140, 88)]))
    mesh_object("DuneGrass", bm, col, [M.terrain])
    return ob


def build_water(col, M):
    ng, gi, go = new_group("SurfWater", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    try:
        ng.is_modifier = True
    except AttributeError:
        pass
    nodes, links = ng.nodes, ng.links
    B = NodeB(ng)
    centre = (WATER_CENTER[0], WATER_CENTER[1], 0.0)
    t = nmath(ng, "DIVIDE", nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], float(FPS))

    def grid(half, n):
        g = nodes.new("GeometryNodeMeshGrid")
        g.inputs["Size X"].default_value = g.inputs["Size Y"].default_value = 2 * half
        g.inputs["Vertices X"].default_value = g.inputs["Vertices Y"].default_value = n
        tr = nodes.new("GeometryNodeTransform")
        links.new(g.outputs["Mesh"], tr.inputs["Geometry"])
        tr.inputs["Translation"].default_value = centre
        return tr.outputs[0]

    def xy():
        p = separate(ng, nodes.new("GeometryNodeInputPosition").outputs[0])
        return p["X"], p["Y"]

    def store(geom, name, value):
        s = nodes.new("GeometryNodeStoreNamedAttribute")
        s.data_type = "FLOAT"
        s.domain = "POINT"
        links.new(geom, s.inputs["Geometry"])
        s.inputs["Name"].default_value = name
        _feed(ng, s.inputs["Value"], value)
        return s.outputs[0]

    # Near-shore surf with breaking waves, foam and depth attributes
    tri = nodes.new("GeometryNodeTriangulate")
    links.new(grid(FINE_HALF, int(2 * FINE_HALF) + 1), tri.inputs["Mesh"])
    x, y = xy()
    h, p, rip = waves(B, x, y, t)
    cheb = nmath(ng, "MAXIMUM", nmath(ng, "ABSOLUTE", nmath(ng, "SUBTRACT", x, centre[0])),
                 nmath(ng, "ABSOLUTE", nmath(ng, "SUBTRACT", y, centre[1])))
    fade = map_range(ng, cheb, 90.0, 105.0, 1.0, 0.0)
    sp = nodes.new("GeometryNodeSetPosition")
    links.new(tri.outputs[0], sp.inputs["Geometry"])
    links.new(combine(ng, 0.0, 0.0, nmath(ng, "MULTIPLY", h, fade)), sp.inputs["Offset"])
    x, y = xy()
    fine = store(sp.outputs[0], "depth", B.mul(bed(B, x, y), -1.0))
    x, y = xy()
    h, p, rip = waves(B, x, y, t)
    fine = store(fine, "foam", B.mul(foam(B, x, y, t, p, rip), fade))

    # Coarse swell ring to the horizon
    dele = nodes.new("GeometryNodeDeleteGeometry")
    dele.domain = "FACE"
    links.new(grid(COARSE_HALF, int(2 * COARSE_HALF / 6) + 1), dele.inputs["Geometry"])
    x, y = xy()
    inside = nmath(ng, "MULTIPLY",
                   nmath(ng, "LESS_THAN", nmath(ng, "ABSOLUTE", nmath(ng, "SUBTRACT", x, centre[0])), FINE_HALF),
                   nmath(ng, "LESS_THAN", nmath(ng, "ABSOLUTE", nmath(ng, "SUBTRACT", y, centre[1])), FINE_HALF))
    links.new(inside, dele.inputs["Selection"])
    tri2 = nodes.new("GeometryNodeTriangulate")
    links.new(dele.outputs[0], tri2.inputs["Mesh"])
    x, y = xy()
    cheb = nmath(ng, "MAXIMUM", nmath(ng, "ABSOLUTE", nmath(ng, "SUBTRACT", x, centre[0])),
                 nmath(ng, "ABSOLUTE", nmath(ng, "SUBTRACT", y, centre[1])))
    swell = nmath(ng, "MULTIPLY", nmath(ng, "SINE", B.add(B.mul(y, 2 * math.pi / 42.0), B.mul(t, 1.1), B.mul(x, 0.01))),
                  nmath(ng, "MULTIPLY", map_range(ng, cheb, FINE_HALF, 190.0), map_range(ng, cheb, 520.0, 690.0, 0.3, 0.0)))
    sp2 = nodes.new("GeometryNodeSetPosition")
    links.new(tri2.outputs[0], sp2.inputs["Geometry"])
    links.new(combine(ng, 0.0, 0.0, swell), sp2.inputs["Offset"])
    coarse = store(store(sp2.outputs[0], "depth", 12.0), "foam", 0.0)

    join = nodes.new("GeometryNodeJoinGeometry")
    links.new(coarse, join.inputs[0])
    links.new(fine, join.inputs[0])
    flat = nodes.new("GeometryNodeSetShadeSmooth")
    links.new(join.outputs[0], flat.inputs[0])
    flat.inputs["Shade Smooth"].default_value = True     # smooth gradients on the water
    setm = nodes.new("GeometryNodeSetMaterial")
    links.new(flat.outputs[0], setm.inputs["Geometry"])
    setm.inputs["Material"].default_value = M.water
    links.new(setm.outputs[0], go.inputs["Geometry"])

    ob = bpy.data.objects.new("Water", bpy.data.meshes.new("Water"))
    col.objects.link(ob)
    ob.modifiers.new("SurfWater", "NODES").node_group = ng
    ob.visible_shadow = False

    # Flat far ring, and an underwater curtain so underwater rays never see the sky
    bm = bmesh.new()
    cx, cy = WATER_CENTER
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    vi = [bm.verts.new((cx + sx * COARSE_HALF, cy + sy * COARSE_HALF, 0.0)) for sx, sy in corners]
    vo = [bm.verts.new((cx + sx * 9000.0, cy + sy * 9000.0, 0.0)) for sx, sy in corners]
    for i in range(4):
        j = (i + 1) % 4
        bm.faces.new((vi[i], vo[i], vo[j], vi[j]))
    far = mesh_object("WaterFar", bm, col, [M.water], recalc=False)
    far.visible_shadow = False
    bm = bmesh.new()
    seg = 64
    top = [bm.verts.new((cx + CURTAIN_R * math.cos(2 * math.pi * i / seg),
                         cy + CURTAIN_R * math.sin(2 * math.pi * i / seg), 0.005)) for i in range(seg)]
    bot = [bm.verts.new((v.co.x, v.co.y, -40.0)) for v in top]
    for i in range(seg):
        j = (i + 1) % seg
        bm.faces.new((top[i], top[j], bot[j], bot[i]))
    cur = mesh_object("UnderwaterCurtain", bm, col, [M.water], recalc=False)
    cur.visible_shadow = False
    return ob


def cap_open_shell(me):
    """The downloaded rocks are open shells (they sat on a seabed); fill the holes so no thin rim or inside shows.
    Cap corners borrow the UV of the same vertex on the rock surface. Returns the number of boundary edges."""
    bm = bmesh.new()
    bm.from_mesh(me)
    edges = [e for e in bm.edges if e.is_boundary]
    if edges:
        caps = set(bmesh.ops.holes_fill(bm, edges=edges, sides=0)["faces"])
        uv = bm.loops.layers.uv.active
        for f in caps:
            for lp in f.loops:
                src = next((o for o in lp.vert.link_loops if o.face not in caps), None)
                if uv is not None and src is not None:
                    lp[uv].uv = src[uv].uv
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.to_mesh(me)
    bm.free()
    return len(edges)


def load_rock_library():
    """Boulders from the downloaded Babylon.js rock set: one mesh per rock, 2 m across, base down."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=asset("babylon", "underwaterSceneRocksBarnaclesMussels.glb"))
    src = next(o for o in bpy.data.objects if o not in before and o.type == "MESH")
    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    bpy.context.view_layer.objects.active = src
    bpy.ops.mesh.separate(type="LOOSE")
    parts = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    rocks = sorted([o for o in parts if max(o.dimensions) > 0.9 and len(o.data.vertices) > 40],
                   key=lambda o: -max(o.dimensions))
    lib = bpy.data.collections.new("RockLibrary")      # not linked to the scene: instanced only
    capped = 0
    for i, o in enumerate(rocks):
        me = o.data
        me.transform(o.matrix_world)
        o.parent = None
        o.matrix_world = Matrix()
        lo = Vector([min(v.co[k] for v in me.vertices) for k in range(3)])
        hi = Vector([max(v.co[k] for v in me.vertices) for k in range(3)])
        me.transform(Matrix.Scale(2.0 / max(hi.x - lo.x, hi.y - lo.y), 4) @ Matrix.Translation(-(lo + hi) / 2))
        capped += cap_open_shell(me) > 0
        o.name = me.name = f"Rock{i:02d}"
        for c in list(o.users_collection):
            c.objects.unlink(o)
        lib.objects.link(o)
    for o in [o for o in bpy.data.objects if o not in before and o not in rocks]:
        bpy.data.objects.remove(o, do_unlink=True)
    print(f"rock library: {len(rocks)} rocks, {capped} open shells closed")
    return lib, rocks


def build_jetty(col, M, lib, n_rocks):
    rng = random.Random(9)
    hw_base, hw_top = 4.6, 1.2
    pts, rots, scls, idxs = [], [], [], []

    def add_rock(c, r):
        pts.append(tuple(c))
        rots.append((rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(0, 6.3)))
        scls.append((r * rng.uniform(0.95, 1.3), r * rng.uniform(0.85, 1.1), r * rng.uniform(0.75, 1.05)))
        idxs.append(rng.randrange(n_rocks))

    def pile(cx, cy, along):
        """One cross-section of rubble at (cx, cy); along = unit vector along the jetty."""
        side = Vector((along.y, -along.x, 0.0))
        b = min(bed_z(cx, cy), 0.4)
        z = b + 0.7
        while z < JETTY_CREST + 0.2:
            hw = hw_base - (z - b) / (JETTY_CREST - b + 1e-6) * (hw_base - hw_top)
            n = max(1, round(2 * hw / 1.7))
            for k in range(n):
                o = -hw + (2 * hw) * (k + 0.5) / n + rng.uniform(-0.3, 0.3)
                add_rock(Vector((cx, cy, z)) + side * o + along * rng.uniform(-0.4, 0.4), rng.uniform(0.75, 1.15))
            z += 1.15

    y = JETTY_Y0
    while y < JETTY_Y1:
        pile(JETTY_X, y, Vector((0, 1, 0)))
        y += 1.5
    for rr in (1.5, 3.0, 4.3):          # rounded head: a half disc sloping into the sea
        n = int(rr * 2.2)
        for k in range(n + 1):
            a = math.radians(-80 + 160 * k / max(n, 1))
            c = Vector((JETTY_X + rr * math.sin(a), JETTY_Y1 + rr * math.cos(a), 0.0))
            b = bed_z(c.x, c.y)
            top = JETTY_CREST - (rr / 4.6) * (JETTY_CREST - b)
            z = b + 0.7
            while z < top + 0.2:
                add_rock(c + Vector((rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), z)), rng.uniform(0.75, 1.15))
                z += 1.15
    me = bpy.data.meshes.new("JettyPoints")
    me.from_pydata(pts, [], [])
    for name, vals in (("rot", rots), ("scl", scls)):
        me.attributes.new(name, "FLOAT_VECTOR", "POINT").data.foreach_set("vector", [c for v in vals for c in v])
    me.attributes.new("idx", "INT", "POINT").data.foreach_set("value", idxs)
    ob = bpy.data.objects.new("Jetty", me)
    col.objects.link(ob)
    ng, gi, go = new_group("JettyRocks", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    ci = ng.nodes.new("GeometryNodeCollectionInfo")
    ci.inputs["Collection"].default_value = lib
    ci.inputs["Separate Children"].default_value = True
    ci.inputs["Reset Children"].default_value = True
    iop = ng.nodes.new("GeometryNodeInstanceOnPoints")
    ng.links.new(gi.outputs["Geometry"], iop.inputs["Points"])
    ng.links.new(ci.outputs[0], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True

    def named(name, dtype):
        n = ng.nodes.new("GeometryNodeInputNamedAttribute")
        n.data_type = dtype
        n.inputs["Name"].default_value = name
        return n.outputs["Attribute"]
    ng.links.new(named("idx", "INT"), iop.inputs["Instance Index"])
    ng.links.new(named("rot", "FLOAT_VECTOR"), iop.inputs["Rotation"])
    ng.links.new(named("scl", "FLOAT_VECTOR"), iop.inputs["Scale"])
    real = ng.nodes.new("GeometryNodeRealizeInstances")
    ng.links.new(iop.outputs[0], real.inputs["Geometry"])
    ng.links.new(real.outputs[0], go.inputs["Geometry"])
    ob.modifiers.new("JettyRocks", "NODES").node_group = ng
    print(f"jetty: {len(pts)} boulders from {n_rocks} downloaded rock meshes")
    return ob


# --------------------------------------------------------------------------
# Fish
# --------------------------------------------------------------------------
FISH_LIB = {}      # kind -> mesh, filled by load_fish_library()
SWIM = None        # geometry nodes group that bends the body into a swimming wave


def load_fish_library(atmos):
    """Downloaded fish, re-oriented to nose +X, centred, 1 unit long."""
    for kind, path in (("barramundi", asset("khronos", "BarramundiFish.glb")),
                       ("snapper", asset("babylon", "greySnapper_vertColor.glb"))):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=path)
        new = [o for o in bpy.data.objects if o not in before]
        src = next(o for o in new if o.type == "MESH")
        me = src.data
        me.transform(src.matrix_world)
        me.transform(Matrix.Rotation(math.pi / 2, 4, "Z"))      # models face -Y; turn them to +X
        lo = Vector([min(v.co[k] for v in me.vertices) for k in range(3)])
        hi = Vector([max(v.co[k] for v in me.vertices) for k in range(3)])
        me.transform(Matrix.Scale(1.0 / (hi.x - lo.x), 4) @ Matrix.Translation(-(lo + hi) / 2))
        for m in me.materials:
            wrap_atmos(m, atmos)
        me.name = f"Fish_{kind}"
        for o in new:
            bpy.data.objects.remove(o, do_unlink=True)
        FISH_LIB[kind] = me


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


def set_mod_input(mod, ident, value):
    """Blender 5.2 exposes modifier inputs as mod.properties.inputs.<ident>.value;
    older builds used ID properties (mod[ident])."""
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        getattr(props.inputs, ident).value = value
    else:
        mod[ident] = value


def make_fish(name, col, kind, size, rng, rate=7.0, amp=0.07):
    ob = bpy.data.objects.new(name, FISH_LIB[kind])
    col.objects.link(ob)
    ob.scale = (size, size, size)
    ob.rotation_mode = "XYZ"
    mod = ob.modifiers.new("Swim", "NODES")
    mod.node_group = SWIM
    vals = {"Phase": rng.uniform(0, 6.28), "Amp": amp, "Rate": rate}
    for item in SWIM.interface.items_tree:
        if getattr(item, "in_out", None) == "INPUT" and item.name in vals:
            set_mod_input(mod, item.identifier, vals[item.name])
    return ob


def build_seagrass(col, M):
    """Clumps of the downloaded seagrass card, kept clear of the underwater camera paths."""
    rng = random.Random(41)
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    for k in range(3):
        a = math.pi * k / 3
        d = Vector((math.cos(a), math.sin(a), 0.0)) * 0.32
        f = bm.faces.new([bm.verts.new(p) for p in (-d, d, d + Vector((0, 0, 0.85)), -d + Vector((0, 0, 0.85)))])
        for loop, c in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
            loop[uv].uv = c
    me = bpy.data.meshes.new("SeagrassClump")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M.seagrass)
    zones = [((-14.0, 10.0), (9.0, 19.5), 70), ((12.0, 15.4), (11.0, 27.0), 34),
             ((-16.6, -13.6), (36.0, 63.0), 30), ((-25.0, -14.0), (66.5, 71.0), 14)]
    placed = []
    for (x0, x1), (y0, y1), count in zones:
        for _ in range(count):
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if bed_z(x, y) > -1.0:
                continue
            ob = bpy.data.objects.new("Seagrass", me)
            col.objects.link(ob)
            s = rng.uniform(0.7, 1.35)
            ob.scale = (s, s, s * rng.uniform(0.9, 1.3))
            ob.location = (x, y, bed_z(x, y) - 0.05)
            ob.rotation_euler = (0.0, 0.0, rng.uniform(0, 6.3))
            ph = rng.uniform(0, 6.28)
            add_frame_driver(ob, "rotation_euler", f"sin(frame*0.07+{ph:.2f})*0.08", index=0)
            add_frame_driver(ob, "rotation_euler", f"sin(frame*0.05+{ph * 1.7:.2f})*0.06", index=1)
            placed.append(ob)
    print(f"seagrass: {len(placed)} clumps")
    return placed


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


# --------------------------------------------------------------------------
# Angler, cast, lure and the jig demonstration
# --------------------------------------------------------------------------
def build_angler(col, M):
    g = bed_z(ANGLER.x, ANGLER.y)
    root = empty_object("Angler", col, loc=(ANGLER.x, ANGLER.y, g), size=0.3)

    def part(name, fn, color):
        bm = bmesh.new()
        fn(bm)
        return mesh_object(name, bm, col, [M.painted], color=color, parent=root)

    waders, jacket, skin, cap = srgb(62, 84, 74), srgb(236, 112, 82), srgb(222, 172, 140), srgb(56, 78, 110)
    part("AnglerLegs", lambda bm: [bm_frustum(bm, (s * 0.11, 0, 0.0), (s * 0.11, 0, 0.88), 0.085, 0.09)
                                   for s in (-1, 1)], waders)
    part("AnglerTorso", lambda bm: bm_frustum(bm, (0, 0, 0.86), (0, 0.02, 1.46), 0.19, 0.21, segs=7), jacket)
    part("AnglerHead", lambda bm: bm_blob(bm, (0, 0.03, 1.60), (0.115, 0.115, 0.125), random.Random(2), jitter=0.03), skin)
    part("AnglerCap", lambda bm: (bm_frustum(bm, (0, 0.03, 1.68), (0, 0.03, 1.77), 0.125, 0.11, segs=8),
                                  bm_box(bm, (0, 0.17, 1.69), (0.2, 0.14, 0.02))), cap)
    pivot = empty_object("SurfRodPivot", col, parent=root, size=0.15)
    pivot.rotation_mode = "XYZ"
    yaw = math.atan2(CAST_LAND.y - ANGLER.y - 0.25, CAST_LAND.x - ANGLER.x - 0.12)
    cast = [(1, 60, (0.12, 0.25, 1.12)), (684, 60, (0.12, 0.25, 1.12)), (700, 128, (0.06, 0.05, 1.42)),
            (708, 22, (0.14, 0.42, 1.30)), (722, 32, (0.12, 0.30, 1.15)), (900, 38, (0.12, 0.28, 1.12)),
            (1560, 40, (0.12, 0.28, 1.12))]
    for f, elev, loc in cast:
        pivot.location = loc
        pivot.rotation_euler = (0.0, -math.radians(elev), yaw)
        key(pivot, "location", f)
        key(pivot, "rotation_euler", f)
    bm = bmesh.new()
    bm_frustum(bm, (-0.45, 0, 0), (0.3, 0, 0), 0.028, 0.022, segs=6)
    bm_frustum(bm, (0.3, 0, 0), (3.2, 0, 0), 0.016, 0.006, segs=5)
    mesh_object("SurfRod", bm, col, [M.painted], color=srgb(30, 32, 36), parent=pivot)
    tip = empty_object("SurfRodTip", col, loc=(3.2, 0, 0), parent=pivot, size=0.05)
    hand = empty_object("AnglerHand", col, parent=pivot, size=0.05)
    hand2 = empty_object("AnglerHand2", col, loc=(-0.35, 0, 0), parent=pivot, size=0.05)
    for name, sh, h in (("AnglerArmR", (0.2, 0.0, 1.40), hand), ("AnglerArmL", (-0.2, 0.0, 1.40), hand2)):
        bm = bmesh.new()
        bm_frustum(bm, (0, 0, 0), (0, 1, 0), 0.06, 0.05, segs=6)
        arm = mesh_object(name, bm, col, [M.painted], color=jacket, parent=root)
        arm.location = sh
        con = arm.constraints.new("STRETCH_TO")
        con.target = h
        con.rest_length = 1.0
        con.volume = "NO_VOLUME"
    return root, tip


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


def lure_path():
    """Frame -> lure position for the jig demonstration (F880..F1345), plus metadata."""
    x0 = RIP_X + 0.4
    pos = {}
    contacts = []

    def floor(y):
        return bed_z(x0, y) + 0.07
    # rest on the bottom before the demo
    for f in range(880, 915):
        pos[f] = Vector((x0, 23.6, floor(23.6)))
    # BOUNCE JIG: hops that kick sand, moving toward shore
    y = 23.6
    for f in range(915, 1045):
        k = (f - 915) % 31
        hop = (f - 915) // 31
        if k < 8:
            u = k / 8
            h = 0.42 * (1 - (1 - u) ** 2)
            dy = 0.45 * u
        elif k < 21:
            u = (k - 8) / 13
            h = 0.42 * (1 - u * u)
            dy = 0.45 + 0.35 * u
        else:
            h, dy = 0.0, 0.8
            if k == 21:
                contacts.append(f)
        yy = 23.6 - 0.8 * hop - dy
        pos[f] = Vector((x0, yy, floor(yy) + h))
        y = yy
    # STEADY RETRIEVE
    y_s = y
    for f in range(1045, 1165):
        u = (f - 1045) / 120
        e = smoothstep(0.0, 0.12, u)
        yy = y_s - 3.2 * u
        pos[f] = Vector((x0, yy, floor(yy) + 0.07 + 0.32 * e + 0.03 * math.sin(f * 0.35)))
    y = pos[1164].y
    # LIFT AND DROP (two cycles)
    for f in range(1165, 1294):
        c = (f - 1165) // 64
        k = (f - 1165) % 64
        yc = y - 1.0 * c
        if k < 14:
            u = k / 14
            h = 0.36 + 0.75 * (1 - (1 - u) ** 2)
            yy = yc - 0.6 * u
        elif k < 48:
            u = (k - 14) / 34
            h = 1.11 - 1.04 * (u ** 1.3)
            yy = yc - 0.6 - 0.25 * u
        else:
            h, yy = 0.07, yc - 0.85
        sway = 0.05 * math.sin(f * 0.5) if 14 <= k < 48 else 0.0
        pos[f] = Vector((x0 + sway, yy, floor(yy) + h - 0.07))
    pos[1164] = pos[1164]
    return pos, contacts


STRIKE = 1262


def build_jig_demo(col, M, meshes, cam_col):
    rng = random.Random(31)
    pos, contacts = lure_path()
    frames = sorted(pos)
    # Hero fish follows the lure, then strikes on the second drop
    fish = make_fish("HeroFish", col, "barramundi", 0.62, rng, rate=6.0)
    fpos = {}
    cur = pos[880] + Vector((0.2, 1.3, 0.25))
    for f in frames:
        lp = pos[f]
        if f < 1045:
            tgt, a = lp + Vector((0.15, 1.05, 0.14)), 0.10
        elif f < 1165:
            tgt, a = lp + Vector((0.10, 0.72, 0.02)), 0.12
        elif f < STRIKE - 18:
            tgt, a = Vector((lp.x + 0.2, lp.y + 0.95, max(lp.z, bed_z(lp.x, lp.y) + 0.3))), 0.08
        else:
            tgt, a = lp + Vector((0.0, 0.16, 0.0)), 0.30
        cur = cur.lerp(tgt, a)
        fpos[f] = cur.copy()
    # After the strike the lure rides in the fish's mouth while the angler reels it in
    last = fpos[STRIKE]
    hook_start = pos[STRIKE].copy()
    for f in range(STRIKE, 1346):
        u = (f - STRIKE) / (1345 - STRIKE)
        e = u * u * (3 - 2 * u)
        mouth = hook_start.lerp(Vector((RIP_X + 0.1, hook_start.y - 2.6, -0.55)), e)
        mouth += Vector((0.12 * math.sin(f * 0.9), 0.0, 0.06 * math.sin(f * 1.3)))
        pos[f] = mouth
        fpos[f] = mouth + Vector((0.05 * math.sin(f * 0.9), 0.16 + 0.04 * math.cos(f * 0.8), -0.02))
    frames = sorted(pos)
    yaws, pitches = [], []
    for f in frames:
        look = pos[f] - fpos[f] if f >= 1165 or (pos[f] - fpos[f]).length > 0.2 else Vector((0, -1, 0))
        yw, pt = heading_angles(look)
        yaws.append(yw)
        pitches.append(max(-0.5, min(0.5, pt)))
    yaws = unwrap(yaws)
    key_fish(fish, frames, [fpos[f] for f in frames], yaws, pitches)

    # Lure: jig head + paddle tail, oriented along its motion
    lure = empty_object("Lure", col, size=0.05)
    lure.rotation_mode = "XYZ"
    bm = bmesh.new()
    bm_blob(bm, (0.035, 0, 0), (0.032, 0.03, 0.032), rng, jitter=0.0)
    mesh_object("JigHead", bm, col, [M.painted], color=srgb(250, 196, 60), parent=lure)
    bm = bmesh.new()
    bm_frustum(bm, (0.0, 0, 0), (-0.12, 0, 0.0), 0.022, 0.012, segs=6)
    bm_box(bm, (-0.14, 0, 0), (0.03, 0.012, 0.05))
    mesh_object("JigTail", bm, col, [M.painted], color=srgb(238, 238, 236), parent=lure)
    lyaw = []
    for i, f in enumerate(frames):
        nxt = pos[frames[min(i + 2, len(frames) - 1)]] - pos[frames[max(i - 2, 0)]]
        if nxt.length < 1e-4:
            nxt = Vector((0, -1, 0))
        lyaw.append(heading_angles(nxt))
    yw = unwrap([a for a, _ in lyaw])
    for f, a, (_, pt) in zip(frames, yw, lyaw):
        lure.location = pos[f]
        lure.rotation_euler = (0.0, max(-0.9, min(0.9, pt)), a)
        key(lure, "location", f)
        key(lure, "rotation_euler", f)
    lure.location = pos[frames[0]]
    key(lure, "location", 1)
    # Line from the lure up toward the surface and the angler
    exit_pt = empty_object("LineExit", col, size=0.05)
    for f in frames:
        exit_pt.location = pos[f] + Vector((-0.3, -6.0, 2.6 - pos[f].z + 0.2))
        key(exit_pt, "location", f)
    line_object("LineUnderwater", col, M, lure, exit_pt, radius=0.0035)

    # Sand puffs where the bounce jig hits the bottom
    for ci, f0 in enumerate(contacts):
        p0 = pos[f0]
        for k in range(6):
            bm = bmesh.new()
            bm_blob(bm, (0, 0, 0), (0.05, 0.05, 0.04), rng, jitter=0.2)
            ob = mesh_object(f"SandPuff_{ci}_{k}", bm, col, [M.painted], color=srgb(214, 198, 160))
            d = Vector((math.cos(k * 1.05), math.sin(k * 1.05), 0.0))
            for j, s in ((-1, 0.0), (0, 0.4), (8, 1.4), (18, 0.0)):
                ob.location = p0 + d * (0.03 + 0.03 * max(j, 0)) + Vector((0, 0, 0.012 * max(j, 0)))
                ob.scale = (s, s, s)
                key(ob, "location", f0 + j)
                key(ob, "scale", f0 + j)
            ob.scale = (0, 0, 0)
            key(ob, "scale", 1)

    # Background fish crossing the channel
    for i in range(4):
        ob = make_fish(f"ChannelFish{i}", col, "snapper", rng.uniform(0.36, 0.46), rng, rate=8.0)
        x0 = rng.uniform(12.6, 14.8)
        z0 = rng.uniform(-2.0, -1.3)
        sp = rng.uniform(0.35, 0.55) * (1 if i % 2 else -1)
        y0 = 10.0 if sp > 0 else 26.0
        fr = list(range(880, 1346, 2))
        ps = [Vector((x0 + 0.3 * math.sin(f * 0.02 + i), y0 + sp * (f - 880 - 120 * i) / FPS,
                      z0 + 0.1 * math.sin(f * 0.03 + i))) for f in fr]
        ys = [math.pi / 2 if sp > 0 else -math.pi / 2] * len(fr)
        key_fish(ob, fr, ps, ys, [0.0] * len(fr))
    return pos, fpos, lure


def build_tip_school(col, M, meshes):
    rng = random.Random(17)
    for i in range(9):
        home = TIP_SCHOOL + Vector((rng.uniform(-1.6, 1.6), rng.uniform(-1.0, 1.1), rng.uniform(-0.45, 0.45)))
        ob = make_fish(f"TipFish{i}", col, "barramundi" if i % 3 else "snapper",
                       rng.uniform(0.45, 0.6), rng, rate=rng.uniform(5.5, 7.5))
        ph = rng.uniform(0, 6.28)
        yaw0 = math.radians(-55 + rng.uniform(-18, 18))
        fr = list(range(200, 400, 2))
        ps, ys = [], []
        for f in fr:
            t = frame_time(f)
            ps.append(home + Vector((0.25 * math.sin(0.5 * t + ph), 0.2 * math.sin(0.37 * t + ph * 1.3),
                                     0.08 * math.sin(0.8 * t + ph))))
            ys.append(yaw0 + 0.18 * math.sin(0.3 * t + ph))
        key_fish(ob, fr, ps, ys, [0.0] * len(fr))


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
# (frame, position, target, hold) - hold keys come to rest (ease in and out)
CAM_KEYS = [
    (1, (-2.0, 24.0, 58.0), (-2.0, 24.3, 0.0), True),
    (60, (-7.5, 16.0, 48.0), (-10.0, 22.0, 0.0), False),
    (120, (-15.2, 3.0, 30.0), (-19.0, 18.0, 0.0), False),
    (175, (-15.2, 25.0, 21.0), (-19.0, 40.0, 0.0), False),
    (222, (-14.4, 47.0, 11.0), (-18.8, 61.0, 0.0), False),
    (250, (-13.8, 56.5, 5.0), (-18.5, 65.0, -1.2), False),
    (270, (-13.1, 61.8, 0.2), (-18.2, 67.2, -2.4), False),
    (292, (-12.4, 65.0, -2.0), (-19.0, 67.6, -2.8), False),
    (318, (-12.6, 69.8, -2.0), (-19.0, 67.6, -2.9), False),
    (338, (-14.5, 73.5, -1.6), (-19.0, 67.6, -2.6), False),
    (356, (-15.0, 73.4, 0.4), (-18.0, 62.0, -1.0), False),
    (450, (-12.0, 40.0, 9.0), (-12.0, 15.0, 0.0), False),
    (800, (14.0, 40.0, 9.0), (14.5, 15.0, 0.0), False),
    (870, (19.6, 30.5, 3.4), (17.0, 23.5, -0.8), False),
    (902, (19.3, 25.6, -0.6), (16.6, 24.0, -1.6), False),
]
CAM_KEYS_END = [
    (1318, (18.6, 11.8, -0.9), (16.4, 11.0, -0.6), False),
    (1334, (18.2, 11.0, 0.5), (15.8, 14.0, 0.3), False),
    (1360, (16.0, 6.0, 3.0), (10.0, 24.0, 0.0), False),
    (1500, (-2.0, -30.0, 24.0), (-2.0, 26.0, 0.0), True),
    (1560, (-2.3, -31.2, 24.6), (-2.0, 26.0, 0.0), True),
]
TRACK = (915, 1305)   # frames where the camera tracks the lure
CAM_SMOOTH = 6.0      # gaussian sigma (frames) applied to the assembled camera path


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


def camera_frames(lure_pos):
    """Position, target and orientation for every frame."""
    a = hermite_path(CAM_KEYS)
    b = hermite_path(CAM_KEYS_END)
    keys = sorted(lure_pos)
    smooth = dict(zip(keys, smooth_series([lure_pos[f] for f in keys], 20)))
    track = {}
    for f in keys:
        s = smooth[f]
        track[f] = (Vector((s.x + 2.4, s.y + 0.45, -1.55 + 0.3 * (s.z + 2.0))),
                    Vector((s.x, s.y + 0.45, s.z + 0.32)))
    raw = {}
    for f in range(1, FRAME_END + 1):
        if f <= CAM_KEYS[-1][0]:
            p, t = a[f]
        elif f >= CAM_KEYS_END[0][0]:
            p, t = b[f]
        else:
            p, t = track[f]
        if CAM_KEYS[-1][0] - 2 < f < TRACK[0] + 18:     # blend into tracking
            w = smoothstep(CAM_KEYS[-1][0], TRACK[0] + 18, f)
            pa, ta = a.get(f, a[CAM_KEYS[-1][0]])
            p, t = pa.lerp(track[f][0], w), ta.lerp(track[f][1], w)
        if TRACK[1] - 2 < f < CAM_KEYS_END[0][0] + 2:   # blend out of tracking
            w = smoothstep(TRACK[1], CAM_KEYS_END[0][0], f)
            pb, tb = b.get(f, b[CAM_KEYS_END[0][0]])
            p, t = track[f][0].lerp(pb, w), track[f][1].lerp(tb, w)
        raw[f] = (p, t)
    # Low-pass the whole path so every hand-off between moves is smooth
    sigma, rad = CAM_SMOOTH, int(3 * CAM_SMOOTH)
    kern = [math.exp(-0.5 * (k / sigma) ** 2) for k in range(-rad, rad + 1)]
    ksum = sum(kern)
    out = {}
    q_prev = None
    for f in range(1, FRAME_END + 1):
        p, t = Vector(), Vector()
        for k, wgt in zip(range(-rad, rad + 1), kern):
            pp, tt = raw[min(FRAME_END, max(1, f + k))]
            p += pp * (wgt / ksum)
            t += tt * (wgt / ksum)
        q = look_rotation(p, t)
        if q_prev is not None and q.dot(q_prev) < 0:
            q.negate()
        q_prev = q
        out[f] = (p.copy(), t.copy(), q)
    return out


def build_camera(col, frames):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = LENS
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = SENSOR
    cd.clip_start = 0.05
    cd.clip_end = 12000.0
    cam = bpy.data.objects.new("Camera", cd)
    col.objects.link(cam)
    cam.rotation_mode = "QUATERNION"
    for f, (p, _t, q) in frames.items():
        cam.location = p
        cam.rotation_quaternion = q
        key(cam, "location", f)
        key(cam, "rotation_quaternion", f)
    bpy.context.scene.camera = cam
    sc = bpy.context.scene
    for f, (p, _t, _q) in frames.items():
        sc["cam_under"] = 1.0 if p.z < wave_z(p.x, p.y, frame_time(f)) else 0.0
        sc.keyframe_insert(data_path='["cam_under"]', frame=f)
    return cam


def crossings(frames, margin=0.4):
    """Contiguous frame spans where the lens is within `margin` of the waves."""
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
# Overlay graphics (HUD) parented to the camera
# --------------------------------------------------------------------------
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
        s = self.text(sub, "med", 0.0155, g, align="LEFT")
        s["alpha_scale"] = 0.82
        nb = self.text(str(num), "bold", 0.022, g, mat=self.M.hud)
        nb.color = (*INK, 1.0)
        tx0, tx1, _, _ = self.bounds(t)
        sx0, sx1, _, _ = self.bounds(s)
        D, gap, pad, h = 0.044, 0.014, 0.017, 0.068
        w = pad + D + gap + max(tx1 - tx0, sx1 - sx0) + pad + 0.006
        pill = self.shape(self.rounded(w, h, h / 2), INK, g)
        pill["alpha_scale"] = 0.78
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
        k = self.text(kicker, "semi", 0.0135, g, loc=(0, y, 0.001), spacing=1.5)
        k["alpha_scale"] = 0.88
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
            mat = self.M.hud_accent if clean in accents else self.M.hud_white
            sh = self.text(wd, "bold", size, g, align="LEFT", mat=self.M.hud_shadow)
            tx = self.text(wd, "bold", size, g, align="LEFT", mat=mat)
            made.append((wd, tx, sh))
        space = size * 0.3
        lines, cur, cur_w = [], [], 0.0
        for wd, tx, sh in made:
            x0, x1, _, _ = self.bounds(tx)
            w = x1 - x0
            if cur and cur_w + space + w > max_w:
                lines.append((cur, cur_w))
                cur, cur_w = [], 0.0
            cur.append((tx, sh, x0, w))
            cur_w += (space if len(cur) > 1 else 0.0) + w
        lines.append((cur, cur_w))
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
                for ob in (tx, sh):
                    key_alpha(ob, [(1, 0.0), (f_start - 6, 0.0), (f_start, 0.55), (t_word, 0.55),
                                   (t_word + 3, 1.0), (f_end, 1.0), (f_end + 6, 0.0)])
                x += w + space
                i += 1
        h = len(lines) * lh
        plate = self.shape(self.rounded(max_w_used + 0.045, h + 0.022, 0.024), INK, g, loc=(0, 0, -0.0002))
        key_alpha(plate, [(1, 0.0), (f_start - 6, 0.0), (f_start, 0.8), (f_end, 0.8), (f_end + 6, 0.0)])
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
        pill["alpha_scale"] = 0.8
        dot = self.shape(self.circle(0.0075), dot_color, g, loc=(0, 0, 0.0008))
        halo = self.shape(self.circle(0.016), dot_color, g, loc=(0, 0, 0.0006))
        halo["alpha_scale"] = 0.35
        lead = self.shape([(-0.0009, 0), (0.0009, 0), (0.0009, 1), (-0.0009, 1)], WHITE, g, loc=(0, 0, 0.0004))
        lead["alpha_scale"] = 0.9
        boxes = {}
        for f in range(max(1, f_in - 10), min(FRAME_END, f_out + 10) + 1):
            wp = anchor(f) if callable(anchor) else anchor
            s = self.project(f, wp)
            if s is None:
                s = Vector((0, -2))
            lx = max(-HUD_W + w / 2 + 0.018, min(HUD_W - w / 2 - 0.018, s.x))
            ly = min(0.27, s.y + rise + h / 2)
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
                key(ob, path, f)
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
                key(g, "location", f)
            tw = 0.75 + 0.25 * math.sin(i)
            self.fade([dot, halo], f_in + i * 3, f_out, peak=tw)

    def depth_gauge(self, under_frames):
        """Left-edge depth ruler; the pointer tracks the camera depth."""
        g = self.group((-HUD_W + 0.045, 0.0, 0))
        top, step = 0.25, 0.05
        objs = [self.shape([(-0.0012, top - 4 * step), (0.0012, top - 4 * step), (0.0012, top), (-0.0012, top)], WHITE, g)]
        title = self.text("DEPTH", "semi", 0.0115, g, loc=(0.012, top + 0.02, 0.001), align="LEFT", spacing=1.4)
        objs.append(title)
        for m in range(5):
            yy = top - m * step
            objs.append(self.shape([(-0.008, yy - 0.0012), (0.008, yy - 0.0012), (0.008, yy + 0.0012),
                                    (-0.008, yy + 0.0012)], WHITE, g))
            objs.append(self.text(f"{m} m", "med", 0.0125, g, loc=(0.014, yy, 0.001), align="LEFT"))
        ptr = self.shape([(-0.022, 0.0), (-0.006, 0.009), (-0.006, -0.009)], ORANGE, g, loc=(0, top, 0.001))
        objs.append(ptr)
        for ob in objs:
            ob["alpha_scale"] = 0.95 if ob is ptr else 0.8
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
            d = max(0.0, min(4.0, -self.frames[f][0].z))
            ptr.location = (0, top - d * step, 0.001)
            key(ptr, "location", f)
        self.boxes.append(("depth gauge", lambda f, s=set(under_frames): (-HUD_W + 0.02, top - 4 * step - 0.012,
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


# --------------------------------------------------------------------------
# World-space teaching graphics (no fog, unlit)
# --------------------------------------------------------------------------
def gfx_object(name, bm, col, M, color):
    ob = mesh_object(name, bm, col, [M.gfx], color=color, recalc=False)
    ob.visible_shadow = False
    return ob


def build_trough_graphics(col, M, windows):
    bm = bmesh.new()
    x0, x1, y0, y1, z = -14.5, 38.0, 10.5, 18.5, 0.5
    bm.faces.new([bm.verts.new(p) for p in ((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))])
    band = gfx_object("TroughBand", bm, col, M, TROUGH_BLUE)
    bm = bmesh.new()
    for yy in (y0, y1):
        x = x0
        while x < x1 - 1.0:
            bm.faces.new([bm.verts.new(p) for p in ((x, yy - 0.12, z + 0.01), (x + 1.2, yy - 0.12, z + 0.01),
                                                   (x + 1.2, yy + 0.12, z + 0.01), (x, yy + 0.12, z + 0.01))])
            x += 2.0
    dashes = gfx_object("TroughDashes", bm, col, M, DASH_YELLOW)
    for ob, peak in ((band, 0.22), (dashes, 0.95)):
        pairs = [(1, 0.0)]
        for a, b in windows:
            pairs += fade_pairs(a, b, peak, 12)
        key_alpha(ob, pairs)


def build_rip_chevrons(col, M, windows):
    n, y_lo, y_hi, speed = 6, 9.0, 42.0, 2.2
    span = y_hi - y_lo
    for i in range(n):
        bm = bmesh.new()
        w, th = 1.7, 0.42
        pts = [(-w, -0.9), (-w + th, -0.9 - th * 0.2), (0, 0.6 - th), (w - th, -0.9 - th * 0.2), (w, -0.9), (0, 0.6)]
        bm.faces.new([bm.verts.new((x, y, 0.0)) for x, y in pts[:3] + [(0, 0.6)]])
        bm.faces.new([bm.verts.new((x, y, 0.0)) for x, y in [(0, 0.6), (0, 0.6 - th), (w - th, -0.9 - th * 0.2), (w, -0.9)]])
        ob = gfx_object(f"RipChevron{i}", bm, col, M, srgb(236, 246, 255))
        pairs = [(1, 0.0)]
        for a, b in windows:
            for f in range(a - 12, b + 13):
                u = ((f / FPS) * speed / span + i / n) % 1.0
                ob.location = (RIP_X, y_lo + u * span, 0.85)
                key(ob, "location", f)
                edge = smoothstep(0.0, 0.15, u) * smoothstep(1.0, 0.8, u)
                env = smoothstep(a - 12, a, f) * smoothstep(b + 12, b, f)
                pairs.append((f, 0.85 * edge * env))
        key_alpha(ob, sorted(pairs))


def build_cast_graphics(col, M, tip_world):
    """Dashed arc from the rod tip to the landing point, plus landing ripples."""
    p0, p1 = tip_world, Vector((CAST_LAND.x, CAST_LAND.y, 0.6))
    n = 40
    for i in range(n):
        u0, u1 = i / n, (i + 0.55) / n
        a = p0.lerp(p1, u0) + Vector((0, 0, 9.0 * 4 * u0 * (1 - u0)))
        b = p0.lerp(p1, u1) + Vector((0, 0, 9.0 * 4 * u1 * (1 - u1)))
        bm = bmesh.new()
        bm_frustum(bm, a, b, 0.07, 0.07, segs=4)
        ob = gfx_object(f"CastDash{i}", bm, col, M, WHITE)
        f_show = 708 + int(28 * i / n)
        key_alpha(ob, [(1, 0.0), (f_show - 1, 0.0), (f_show + 2, 0.95), (815, 0.95), (830, 0.0)])
    for k in range(3):
        bm = bmesh.new()
        outer = Hud.circle(1.0, 32)
        inner = Hud.circle(0.82, 32)
        for i in range(32):
            j = (i + 1) % 32
            bm.faces.new([bm.verts.new((*outer[i], 0)), bm.verts.new((*outer[j], 0)),
                          bm.verts.new((*inner[j], 0)), bm.verts.new((*inner[i], 0))])
        ob = gfx_object(f"CastRipple{k}", bm, col, M, WHITE)
        ob.location = p1
        f0 = 738 + k * 14
        for f, s in ((f0, 0.3), (f0 + 26, 2.6)):
            ob.scale = (s, s, 1)
            key(ob, "scale", f)
        key_alpha(ob, [(1, 0.0), (f0 - 1, 0.0), (f0, 0.9), (f0 + 26, 0.0)])


def build_jig_trails(col, M, pos):
    """Glowing path drawn behind the lure for each technique."""
    segs = [(915, 1045, 1060), (1045, 1165, 1180), (1165, 1262, 1290)]
    for si, (a, b, gone) in enumerate(segs):
        cu = bpy.data.curves.new(f"JigPath{si}", "CURVE")
        cu.dimensions = "3D"
        sp = cu.splines.new("POLY")
        pts = [pos[f] for f in range(a, b)]
        sp.points.add(len(pts) - 1)
        for p, pt in zip(sp.points, pts):
            p.co = (pt.x, pt.y, pt.z - 0.02, 1.0)
        ob = bpy.data.objects.new(f"JigPath{si}", cu)
        col.objects.link(ob)
        ng, gi, go = new_group(f"JigTrail{si}", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                               [("Geometry", "NodeSocketGeometry")])
        ctrl = empty_object(f"JigTrailCtrl{si}", col, size=0.02)
        trim = ng.nodes.new("GeometryNodeTrimCurve")
        trim.mode = "FACTOR"
        ng.links.new(gi.outputs["Geometry"], trim.inputs["Curve"])
        oi = ng.nodes.new("GeometryNodeObjectInfo")
        oi.inputs["Object"].default_value = ctrl
        ng.links.new(separate(ng, oi.outputs["Location"])["X"], trim.inputs["End"])
        prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
        prof.inputs["Resolution"].default_value = 5
        prof.inputs["Radius"].default_value = 0.008
        c2m = ng.nodes.new("GeometryNodeCurveToMesh")
        ng.links.new(trim.outputs[0], c2m.inputs["Curve"])
        ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
        setm = ng.nodes.new("GeometryNodeSetMaterial")
        setm.inputs["Material"].default_value = M.gfx
        ng.links.new(c2m.outputs[0], setm.inputs["Geometry"])
        ng.links.new(setm.outputs[0], go.inputs["Geometry"])
        ob.modifiers.new("Trail", "NODES").node_group = ng
        ob.color = (*ORANGE, 1.0)
        ob.visible_shadow = False
        for f in range(a, b + 1):
            ctrl.location = (max(0.002, (f - a) / (b - a)), 0, 0)
            key(ctrl, "location", f)
        key_alpha(ob, [(1, 0.0), (a - 1, 0.0), (a + 2, 0.9), (gone - 10, 0.9), (gone, 0.0)])


# --------------------------------------------------------------------------
# Script (story beats and overlay timing)
# --------------------------------------------------------------------------
def build_overlays(hud, frames, lure_pos, fish_pos, wipes):
    hud.scrims()
    # A: opening and the jetty
    hud.headline("SURF FISHING BASICS", [("Where do the", []), ("fish hide at the beach?", ["fish"])], 1, 80)
    hud.caption("Read the water before you make a cast.", 16, 82)
    hud.badge(1, "THE JETTY", "rocks break the current", 92, 246)
    hud.pin(POCKETS["inside pocket"], "inside pocket", 88, 140)
    hud.pin(POCKETS["bar gap"], "bar gap", 108, 168)
    hud.pin(POCKETS["tip eddy"], "tip eddy: fish hold here", 202, 258)
    hud.caption("Bait hides in the rocks, so predators patrol the edges.", 100, 246, accents=("edges",))
    hud.pin(TIP_SCHOOL + Vector((0.4, 0.0, 0.5)), "fish at the base", 292, 336, rise=0.09)
    hud.caption("They hover low, out of the current.", 290, 336, accents=("low,",))
    # B: trough and rip
    hud.badge(2, "THE TROUGH", "right behind the first wave", 462, 628)
    trough_fish = [Vector((x, y, 0.3)) for x, y in ((-8, 13), (-3, 15.5), (2, 12.5), (6, 16), (10, 13.5),
                                                       (-11, 16.5), (4, 14.2), (-5.5, 12.2))]
    hud.glow_dots(trough_fish, 480, 626)
    hud.pin(Vector((-1.0, 14.6, 0.3)), "trough: 2 m deep", 492, 626)
    hud.caption("Waves dig a deeper trough. Fish wait there for food.", 468, 628,
                accents=("trough.", "food."))
    hud.badge(3, "THE RIP", "cast into the channel", 650, 884)
    hud.pin(Vector((RIP_X, 24.0, 0.6)), "rip: no breaking waves", 744, 866)
    hud.pin(Vector((ANGLER.x, ANGLER.y, 2.0)), "you", 690, 770, rise=0.05)
    hud.caption("The rip carries bait out. Cast in and let it sweep your jig.", 660, 884, accents=("rip", "jig."))
    # C: jig techniques
    hud.badge(4, "BOUNCE JIG", "hop it off the bottom", 922, 1042)
    hud.caption("Short hops kick up sand and draw strikes.", 930, 1040, accents=("sand",))
    hud.badge(5, "STEADY RETRIEVE", "slow and even, just off the sand", 1056, 1160)
    hud.caption("Keep it swimming at the fish's eye level.", 1056, 1160, accents=("eye", "level."))
    hud.badge(6, "LIFT AND DROP", "most bites come on the fall", 1170, 1300)
    hud.caption("Lift, then let it flutter down. Watch the line.", 1176, 1250, accents=("flutter",))
    hud.caption("Fish on!", 1264, 1306, accents=("fish", "on!"), size=0.05)
    under = [f for f, (p, _t, _q) in frames.items() if p.z < wave_z(p.x, p.y, frame_time(f))]
    hud.depth_gauge(under)
    # D: recap
    hud.pin(Vector((JETTY_X + 3.0, 34.0, 2.0)), "1  structure", 1452, 1560)
    hud.pin(Vector((-2.0, 14.5, 0.6)), "2  trough", 1462, 1560)
    hud.pin(Vector((RIP_X, 26.0, 0.8)), "3  rip", 1472, 1560)
    hud.headline("RECAP", [("Fish the structure,", ["structure,"]), ("the trough and the rip.", ["trough", "rip."])],
                 1478, 1560)
    for a, b in wipes:
        hud.wipe(a, b)


# --------------------------------------------------------------------------
# Render settings
# --------------------------------------------------------------------------
def setup_render(scale, samples):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y = RES_X, RES_Y
    sc.render.resolution_percentage = scale
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 1, FRAME_END
    ee = sc.eevee
    ee.taa_render_samples = samples
    ee.use_shadows = True
    ee.use_raytracing = True    # screen-space refraction for the see-through water
    try:
        ee.ray_tracing_method = "SCREEN"
    except (AttributeError, TypeError):
        pass
    ee.use_fast_gi = False
    ee.shadow_resolution_scale = 1.0
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def run_checks(frames, jetty, wipes, hud, fish_objs):
    print("\n=== CAMERA REPORT (frame: pos | pitch | speed m/s | above wave | above bed | rocks) ===")
    ev = jetty.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    grass = [o for o in bpy.data.objects if o.name.startswith("Seagrass")]
    kd = kdtree.KDTree(len(me.vertices) + len(grass))
    for i, v in enumerate(me.vertices):
        kd.insert(ev.matrix_world @ v.co, i)
    for j, o in enumerate(grass):          # seagrass tops count as obstacles too
        kd.insert(o.location + Vector((0, 0, 0.85 * o.scale.z)), len(me.vertices) + j)
    print("obstacle points:", len(me.vertices), "rock vertices +", len(grass), "seagrass tops")
    ev.to_mesh_clear()
    kd.balance()
    prev = None
    worst_rock, worst_bed, worst_fish = (1e9, 0), (1e9, 0), (1e9, 0, "")
    sc = bpy.context.scene
    fish_pos = {}
    for f in range(1, FRAME_END + 1, 1):
        p, t, q = frames[f]
        fwd = q @ Vector((0, 0, -1))
        pitch = math.degrees(math.asin(max(-1, min(1, fwd.z))))
        speed = 0.0 if prev is None else (p - prev).length * FPS
        prev = p
        aw = p.z - wave_z(p.x, p.y, frame_time(f))
        ab = p.z - terrain_z(p.x, p.y)
        rock = kd.find(p)[2] - 0.0
        if rock < worst_rock[0]:
            worst_rock = (rock, f)
        if ab < worst_bed[0]:
            worst_bed = (ab, f)
        if (200 <= f <= 400 or 880 <= f <= 1345) and f % 2 == 0:
            sc.frame_set(f)
            for ob in fish_objs:
                d = (ob.matrix_world.translation - p).length
                if d < worst_fish[0]:
                    worst_fish = (d, f, ob.name)
        if f % 24 == 1 or abs(aw) < 0.4 or rock < 1.2 or f == FRAME_END:
            print(f"{f:5d}: ({p.x:6.1f},{p.y:6.1f},{p.z:6.2f}) | {pitch:6.1f} | {speed:5.2f} | "
                  f"{aw:6.2f} | {ab:6.2f} | {rock:5.2f}")
    print("closest rock vertex:", worst_rock, " lowest above terrain:", worst_bed, " closest fish origin:", worst_fish)
    spans = crossings(frames)
    print("surface crossings (|lens - wave| < 0.4 m):", spans)
    for a, b in spans:
        covered = any(wa - 2 <= a and b <= wb + 2 for wa, wb in wipes)
        print(f"   frames {a}-{b} hidden by a wipe: {covered}")
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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, FRAME_END
    sc.render.fps = FPS
    c_env, c_jetty, c_life, c_people = collection("Environment"), collection("Jetty"), collection("SeaLife"), collection("Angler")
    c_gfx, c_cam = collection("TeachingGraphics"), collection("CameraAndOverlays")

    sky = build_sky_group()
    atmos = build_atmos_group()
    M = build_materials(atmos)
    build_world(sky)
    build_sun(c_env)
    build_terrain(c_env, M)
    build_water(c_env, M)
    rock_lib, rocks = load_rock_library()
    for r in rocks:
        r.data.materials.clear()
        r.data.materials.append(M.rock_pbr)
    jetty = build_jetty(c_jetty, M, rock_lib, len(rocks))

    global SWIM
    load_fish_library(atmos)
    SWIM = fish_swim_group()
    meshes = None
    build_seagrass(c_life, M)
    build_tip_school(c_life, M, meshes)
    lure_pos, fish_pos, lure = build_jig_demo(c_life, M, meshes, c_cam)
    angler, rod_tip = build_angler(c_people, M)

    frames = camera_frames(lure_pos)
    cam = build_camera(c_cam, frames)
    spans = crossings(frames)
    wipes = [(a, b) for a, b in spans]
    # While a wipe is fully opaque the lens is crossing the surface; hide the water then
    # so its plane can never slip between the lens and the overlay plane.
    for ob in (bpy.data.objects["Water"], bpy.data.objects["WaterFar"], bpy.data.objects["UnderwaterCurtain"]):
        ob.hide_render = False
        key(ob, "hide_render", 1)
        for a, b in wipes:
            for f, hidden in ((a - 3, False), (a - 2, True), (b + 2, True), (b + 3, False)):
                ob.hide_render = hidden
                key(ob, "hide_render", f)
    print("wipes (opaque from a-2 to b+2):", wipes)

    # Cast line above water: rod tip -> flying sinker -> landing point
    sinker = empty_object("CastSinker", c_people, size=0.05)
    sag = empty_object("CastSag", c_people, size=0.05)
    sc.frame_set(706)
    p0 = rod_tip.matrix_world.translation.copy()
    land = Vector((CAST_LAND.x, CAST_LAND.y, 0.0))
    tips = {}
    for f in range(660, 1561, 1):
        sc.frame_set(f)
        tips[f] = rod_tip.matrix_world.translation.copy()
    for f in range(660, 1561):
        tp = tips[f]
        if f < 706:
            s, sg = tp + Vector((0, 0, -0.5)), 0.0
        elif f < 736:
            u = (f - 706) / 30
            s = p0.lerp(land, u) + Vector((0, 0, 9.0 * 4 * u * (1 - u)))
            sg = 0.2
        else:
            s = land + Vector((0, 0, wave_z(land.x, land.y, frame_time(f))))
            sg = 0.2 + 1.2 * smoothstep(736, 760, f)
        sinker.location = s
        sag.location = (tp + s) * 0.5 + Vector((0, 0, -2 * sg))
        key(sinker, "location", f)
        key(sag, "location", f)
    cast_line = line_object("CastLine", c_people, M, rod_tip, sinker, middle=sag, radius=0.012)
    for f, hidden in ((1, False), (CAM_KEYS[-1][0] - 14, False), (CAM_KEYS[-1][0] - 13, True),
                      (CAM_KEYS_END[1][0] + 2, True), (CAM_KEYS_END[1][0] + 3, False)):
        cast_line.hide_render = hidden
        key(cast_line, "hide_render", f)

    build_trough_graphics(c_gfx, M, [(470, 628), (1458, 1560)])
    build_rip_chevrons(c_gfx, M, [(656, 896), (1468, 1560)])
    build_cast_graphics(c_gfx, M, p0)
    build_jig_trails(c_gfx, M, lure_pos)

    hud = Hud(cam, c_cam, M, frames)
    build_overlays(hud, frames, lure_pos, fish_pos, wipes)
    sc.frame_set(1)
    fish_objs = [o for o in bpy.data.objects if "Fish" in o.name and not o.name.endswith("_tail")]
    _, contacts = lure_path()
    return dict(frames=frames, jetty=jetty, wipes=wipes, hud=hud, fish=fish_objs, contacts=contacts)


def write_events(path, st):
    """Frame-accurate timing of everything the soundtrack reacts to."""
    import json
    fr = st["frames"]
    data = {
        "fps": FPS, "frames": FRAME_END,
        "camera_z": [round(fr[f][0].z, 3) for f in range(1, FRAME_END + 1)],
        "under": [1 if fr[f][0].z < wave_z(fr[f][0].x, fr[f][0].y, frame_time(f)) else 0
                  for f in range(1, FRAME_END + 1)],
        "wipes": st["wipes"],
        "overlays": sorted(st["hud"].events),
        "cast": {"backswing": 684, "forward": 700, "release": 706, "land": 736},
        "jig_contacts": st["contacts"], "strike": STRIKE, "reel_end": 1345,
        "wave_period_s": WAVE_L / WAVE_V,
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    print("events written", path)


def main():
    args = parse_args()
    st = build_scene()
    setup_render(args.scale, args.samples)
    if args.events:
        write_events(args.events, st)
    if args.check:
        run_checks(st["frames"], st["jetty"], st["wipes"], st["hud"], st["fish"])
    if args.save:
        bpy.context.scene.frame_set(1)
        sc_out = bpy.context.scene.render       # Render Animation from the Blender window resumes into gpu_render/frames
        sc_out.filepath = "//gpu_render/frames/frame_"
        sc_out.use_overwrite = False
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
            print("rendered", sc.render.filepath)
    if args.anim:
        os.makedirs(args.outdir, exist_ok=True)
        sc.frame_start, sc.frame_end, sc.frame_step = args.start, args.end, args.step
        sc.render.filepath = os.path.join(args.outdir, "frame_")
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
