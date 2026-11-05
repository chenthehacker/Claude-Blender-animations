"""Record goliath masterclass: the all tackle record goliath grouper and how anglers fish them on wrecks off Florida.
Vertical 9:16, 24 fps, about 44 s (the frame count follows the measured voiceover), Blender 5.2 EEVEE. Built to loop:
the last frame is the first frame's shot, and the voiceover's last line runs into its first.

Run with the Blender binary:
    blender --background --python record_goliath_masterclass.py -- [options]
or with the bpy Python module (pip install bpy):
    python record_goliath_masterclass.py [options]

Options:
    --save PATH        save the generated scene as a .blend file (fonts packed)
    --check            camera clearances (sea, wreck, seabed, fish, boat), camera acceleration and turn rate, cuts
                       and water surface crossings, overlay overlaps and off-screen boxes, subjects on screen, loop match
    --frames 1,300     render these frames as PNG stills into --outdir
    --outdir DIR       output directory for renders
    --scale PCT        resolution percentage (default 100 -> 1080x1920)
    --samples N        EEVEE render samples (default 32)
    --no-blur          switch motion blur off (quick tests)
    --events PATH      write overlay, action and voiceover timing as JSON (input for record_goliath_audio.py)

Everything is procedural (no imported assets). Facts in the on-screen text and the voiceover are listed with their
sources in SOURCES.md; the fish, the wreck, the depths, the rig, the boat and the fight are illustrative.

World (metres, +Z up, sea surface z = 0). Murky green blue Atlantic water off northeast Florida over a sunken steel
wreck on a sand bottom. A goliath grouper holds in a dark cavity in the hull. Overlays live in a second scene
("HUD") with its own static camera, composited over the 3D render.

Beats follow the voiceover clips (voiceover/vo1..vo5). Cuts are hard cuts, match cuts and whip pans (no wipes):
    1  hook: churning water over the wreck from above whips down under water; a zoom into a dark hole in the hull
       reveals the goliath's mouth flaring open; a second angle on the fish in the hole
    2  structure: a match cut on the fish to a cross section of the wreck, the goliath wedged in a dark cavity
    3  the rig: a close turntable of a heavy cable leader, a big circle hook and a whole fish bait
    4  the strike: the goliath inhales the bait in a burst of silt; the rod doubled over in the holder as the drag
       goes up; the fish's head turned away from the hole on the line
    5  outro: a whip pan back to the water over the wreck from above, which is frame 1
"""

import argparse
import json
import math
import os
import random
import sys

import bpy  # must come first: the pip bpy module exposes bmesh/mathutils after import
import bmesh
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector
from mathutils import noise as mnoise
from mathutils.kdtree import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
FPS = 24
SHUTTER = 0.5
RES_X, RES_Y = 1080, 1920
LENS = 35.0            # the overlay camera's lens (the 3D camera's lens is animated)
SENSOR = 36.0
HUD_DIST = 0.25
HUD_W = 0.5 * RES_X / RES_Y


def srgb(r, g, b):
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (lin(r), lin(g), lin(b))


GOLD = srgb(246, 188, 70)
BADGE = srgb(250, 172, 52)
WHITE = (1.0, 1.0, 1.0)
INK = srgb(10, 14, 18)
PIN_RED = srgb(240, 96, 80)
GLOW = srgb(255, 200, 96)

# water: murky green blue Atlantic over a wreck (fog colour by view direction, down .. up)
UW_RAMP = [(0.0, srgb(4, 12, 10)), (0.36, srgb(12, 34, 30)), (0.52, srgb(26, 62, 52)), (0.72, srgb(52, 100, 84)),
           (1.0, srgb(112, 160, 132))]
VIS = 7.5                            # fog e-folding distance under water (m): murky, silty nearshore water
AIR_HAZE = 900.0                     # sea haze e-folding distance above water (m): clear trade wind air
SUN_DIR = Vector((-0.4, 0.5, 0.77)).normalized()       # a high sun (about 50 degrees up)


SEABED_Z = -17.0                     # sand bottom (illustrative depth for a nearshore wreck)
WRECK_AT = Vector((0.0, 0.0, SEABED_Z))   # the wreck's keel line centre, lying on the sand (the hook and the strike)
WRECK_L, WRECK_B, WRECK_H = 30.0, 7.6, 6.2   # length, beam, height of the hull (illustrative)


def t_of(f):
    """Scene time in seconds at frame f, zero at frame 1, so frame 1321 would equal frame 1 (the loop)."""
    return (f - 1) / FPS


# --------------------------------------------------------------------------
# Voiceover timing. vo_timing.json (written by record_marlin_audio.py --measure) holds the measured lengths and
# sentence starts. The hook starts on frame 1 and the outro ends on the last frame, so the loop joins mid sentence:
# "...and you'll see why... In 1982, Jay de Beaubien hooked..."
# --------------------------------------------------------------------------
VO_LEN = {1: 10.76, 2: 8.36, 3: 9.40, 4: 10.19, 5: 3.47}
VO_TEXT = {
    1: "In 1961, Lynn Joyner hauled a 680 pound goliath grouper out of Florida waters. It's still the all tackle world "
       "record, more than sixty years later.",
    2: "Goliath groupers don't chase bait across open water. They hold tight to shipwrecks and bridge pilings, using "
       "dark cavities as ambush cover.",
    3: "To pull a giant away from iron structure, standard gear won't hold. Anglers use heavy cable leaders, big "
       "circle hooks, and a whole fish for bait.",
    4: "They feed by suction, flaring that huge mouth to inhale prey. Lock your drag down hard and turn the fish's "
       "head before it reaches the wreck, or it will break you off.",
    5: "Stop them early, keep them in the water, and remember...",
}
VO_GAPS = (12, 12, 12)               # frames of breath after clips 1, 2 and 3 (fast paced: no dead air)
OUTRO_GAP = 12                       # breath before the outro line


def _load_vo_timing():
    path = os.path.join(HERE, "vo_timing.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        data = json.load(fh)
    for k, v in data.get("length", {}).items():
        VO_LEN[int(k)] = float(v)
    return {int(k): v for k, v in data.get("phrases", {}).items()}


VO_PHRASES = _load_vo_timing()
VO_START = {1: 1}
for _k in range(2, 5):
    VO_START[_k] = VO_START[_k - 1] + int(round(VO_LEN[_k - 1] * FPS)) + VO_GAPS[_k - 2]
VO_START[5] = VO_START[4] + int(round(VO_LEN[4] * FPS)) + OUTRO_GAP
FRAME_END = VO_START[5] + int(round(VO_LEN[5] * FPS)) - 1   # the outro line ends on the last frame (the loop)
LOOP_S = FRAME_END / FPS             # the sea and the music repeat with this period (the film's length)


def vo_end(clip):
    return VO_START[clip] + int(round(VO_LEN[clip] * FPS))


def phrase_frame(clip, i):
    """Frame where sentence i of a clip starts (sentences split after . ? ; and :), measured or estimated."""
    if clip in VO_PHRASES and i < len(VO_PHRASES[clip]):
        return VO_START[clip] + int(round(VO_PHRASES[clip][i] * FPS))
    parts = [p for p in VO_TEXT[clip].replace("?", "?|").replace(". ", ".|").replace(";", ";|").replace(":", ":|")
             .split("|") if p.strip()]
    w = [len(p.strip()) + 8 for p in parts]
    return VO_START[clip] + int(round(VO_LEN[clip] * sum(w[:i]) / sum(w) * FPS))


P = phrase_frame
WHIP_IN = 7                          # frames 1..6: the water over the wreck from above (the loop's join), whip down
CUT1A = P(1, 1) - 4                              # hard cut inside the hook: a second angle on the fish in the hole
CUT1 = (vo_end(1) + VO_START[2]) // 2            # match cut: the goliath in the hole -> the goliath in the cross section
CUT2 = (vo_end(2) + VO_START[3]) // 2            # jump cut to the rig breakdown
CUT2B = P(3, 1) - 4                              # hard cut inside the rig beat: the turntable -> close on the hook
CUT3 = (vo_end(3) + VO_START[4]) // 2            # rapid cut to the strike
CUT3B = P(4, 1) - 3                              # hard cut: the strike -> the rod doubled over ("Lock your drag")
CUT3C = VO_START[4] + int(round(7.2 * FPS))      # hard cut: the rod -> the fish's head turned on the line
CUT4 = VO_START[5] - 6                           # whip pan back to the water from above (the loop's join)
CUTS = [WHIP_IN, CUT1A, CUT1, CUT2, CUT2B, CUT3, CUT3B, CUT3C, CUT4]
SEGMENTS = [(1, CUT1 - 1), (CUT1, CUT2 - 1), (CUT2, CUT3 - 1), (CUT3, CUT4 - 1), (CUT4, FRAME_END)]


def segment_of(f):
    for i, (a, b) in enumerate(SEGMENTS):
        if a <= f <= b:
            return i
    return len(SEGMENTS) - 1


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (u * 6 - 15) + 10)


def ease_out(u, p=3.0):
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** p


def back_out(u, s=1.6):
    u = min(1.0, max(0.0, u)) - 1.0
    return u * u * ((s + 1) * u + s) + 1.0


# --------------------------------------------------------------------------
# Helpers (shared with the other builds)
# --------------------------------------------------------------------------
def smoothstep(e0, e1, x):
    if e0 > e1:
        return 1.0 - smoothstep(e1, e0, x)
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


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


def set_mod_input(mod, ident, value):
    """Blender 5.2 exposes modifier inputs as mod.properties.inputs.<ident>.value;
    older builds used ID properties (mod[ident])."""
    props = getattr(mod, "properties", None)
    if props is not None and hasattr(props, "inputs"):
        getattr(props.inputs, ident).value = value
    else:
        mod[ident] = value


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

    def soft_plate(self, w, h, margin, loc=(0, 0, 0)):
        """Dark backing whose opacity falls to zero over its outer margin (a per-vertex 'soft' attribute)."""
        bm = bmesh.new()
        soft = bm.verts.layers.float.new("soft")
        outer = [bm.verts.new((x, y, 0.0)) for x, y in self.rounded(w, h, h / 2.5)]
        inner = [bm.verts.new((x, y, 0.0)) for x, y in self.rounded(w - 2 * margin, h - 2 * margin, max(0.004, h / 2.5 - margin))]
        for v in outer:
            v[soft] = 0.0
        for v in inner:
            v[soft] = 1.0
        n = len(outer)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((outer[i], outer[j], inner[j], inner[i]))
        bm.faces.new(inner)
        ob = mesh_object(self._name("plate"), bm, self.col, [self.M.hud_plate], color=WHITE, recalc=False)
        ob.location = loc
        return self._link(ob, None)

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
        top_y = ytop + size * 0.6
        pw, ph = (x_max - x_min) + 0.12, (top_y - bot) + 0.11
        plate = self.soft_plate(pw, ph, 0.04, loc=((x_min + x_max) / 2, (top_y + bot) / 2, -0.0004))
        plate["alpha_scale"] = 0.82
        self.fade([plate], f_start - 2, f_end, dur=5)
        self.boxes.append((f"caption {text[:22]}", lambda f, a=f_start, b=f_end: (x_min - 0.01, bot - 0.004, x_max + 0.01,
                                                                                ytop + size * 0.6)
                           if a - 4 <= f <= b + 4 else None))

    # ---- pins (label pills on leader lines pointing at the world) --------------------------
    def project(self, f, world):
        p, _t, q, lens = self.frames[f]
        m = Matrix.LocRotScale(p, q, None).inverted()
        c = m @ Vector(world)
        if c.z > -0.05:
            return None
        k = lens / SENSOR
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
        if len(best) < 16:                     # too short to fade in and out cleanly: leave it out rather than ghost
            return None
        f_in, f_out = best[0] + 6, best[-1] - 6     # the 6 frame fades stay inside the range (never across a cut)
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
        span = list(range(max(1, f_lo, f_in - 8), min(FRAME_END, f_hi, f_out + 8) + 1))   # never across a cut
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
        top.color[3] = 0.5
        bot = self.shape([(-0.3, -0.51), (0.3, -0.51), (0.3, -0.08), (-0.3, -0.08)], WHITE, mat=self.M.scrim_bot,
                         loc=(0, 0, -0.0006))
        bot.color[3] = 0.86

    def cta(self, text, f_in, f_out, y=0.10):
        """Call to action: a gold pill with dark bold text that pops in and pulses gently."""
        self.events.append((f_in - 6, "cta", text))
        g = self.group((0, y, 0))
        t = self.text(text, "bold", 0.0205, g, spacing=1.1, mat=self.M.hud)
        t.color = (*INK, 0.0)
        x0, x1, _, _ = self.bounds(t)
        t.location = (-(x0 + x1) / 2, -0.0009, 0.001)
        w, h = (x1 - x0) + 0.056, 0.058
        pill = self.shape(self.rounded(w, h, h / 2), GOLD, g)
        halo = self.shape(self.rounded(w + 0.014, h + 0.014, (h + 0.014) / 2), tuple(c * 0.6 for c in GOLD), g, loc=(0, 0, -0.0003))
        halo["alpha_scale"] = 0.35
        self.fade([pill, halo, t], f_in, f_out)
        for f in range(max(1, f_in - 6), min(FRAME_END, f_out) + 1):
            u = smoothstep(f_in - 6, f_in, f)
            s = (0.6 + 0.4 * back_out(u, 2.0)) * (1.0 + 0.025 * max(0.0, math.sin((f - f_in) * 2 * math.pi / 24)) * (f > f_in + 12))
            g.scale = (s, s, 1.0)
            key_still(g, "scale", f)
        self.boxes.append((f"cta {text[:12]}", lambda f, a=f_in, b=f_out: (-w / 2 - 0.01, y - h / 2 - 0.01, w / 2 + 0.01, y + h / 2 + 0.01)
                           if a - 4 <= f <= b + 4 else None))

    def wipe(self, f_a, f_b):
        """Slide wipe: a graphite panel with a gold leading edge sweeps up over the frame, holds while the cut
        happens behind it, then continues off the top."""
        g = self.group((0, 0, 0.004))
        panel = self.shape([(-0.31, -0.53), (0.31, -0.53), (0.31, 0.53), (-0.31, 0.53)], srgb(14, 16, 18), g)
        glow = self.shape([(-0.31, -0.014), (0.31, -0.014), (0.31, 0.014), (-0.31, 0.014)], srgb(120, 84, 24), g,
                          loc=(0, 0.542, 0.0005))
        edge = self.shape([(-0.31, -0.003), (0.31, -0.003), (0.31, 0.003), (-0.31, 0.003)], GOLD, g, loc=(0, 0.533, 0.001))
        objs = [panel, glow, edge]
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


def loft(bm, rings, n=28, mat=0, smooth=True):
    """Lofted body along X. rings: (x, half_w, half_h, z_centre[, exponent]); a ring of zero size is a pole.
    Cross-sections are superellipses (exponent 2 = ellipse, higher = boxier). x may run backwards (a cup)."""
    vr = []
    for r in rings:
        x, w, h, zc = r[:4]
        p = r[4] if len(r) > 4 else 2.0
        if w <= 1e-7 and h <= 1e-7:
            vr.append([bm.verts.new((x, 0.0, zc))])
            continue
        ring = []
        for i in range(n):
            a = 2 * math.pi * i / n
            c, s = math.cos(a), math.sin(a)
            ring.append(bm.verts.new((x, w * math.copysign(abs(c) ** (2 / p), c), zc + h * math.copysign(abs(s) ** (2 / p), s))))
        vr.append(ring)
    faces = []
    for ra, rb in zip(vr, vr[1:]):
        if len(ra) == 1 and len(rb) == 1:
            continue
        if len(ra) == 1:
            faces += [bm.faces.new((ra[0], rb[i], rb[(i + 1) % n])) for i in range(n)]
        elif len(rb) == 1:
            faces += [bm.faces.new((ra[i], ra[(i + 1) % n], rb[0])) for i in range(n)]
        else:
            faces += [bm.faces.new((ra[i], ra[(i + 1) % n], rb[(i + 1) % n], rb[i])) for i in range(n)]
    for f in faces:
        f.material_index = mat
        f.smooth = smooth
    return faces


def tube(bm, pts, r, segs=6, cap=True, mat=0):
    """Round tube along a polyline (parallel transport frames). r: radius or function of 0..1."""
    pts = [Vector(p) for p in pts]
    rings, u = [], None
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        u = basis(t)[0] if u is None else (u - t * u.dot(t)).normalized()
        v = t.cross(u)
        rr = r(i / (len(pts) - 1)) if callable(r) else r
        rings.append([bm.verts.new(p + (u * math.cos(2 * math.pi * k / segs) + v * math.sin(2 * math.pi * k / segs)) * rr)
                      for k in range(segs)])
    faces = []
    for a, b in zip(rings, rings[1:]):
        for k in range(segs):
            j = (k + 1) % segs
            faces.append(bm.faces.new((a[k], a[j], b[j], b[k])))
    if cap:
        faces.append(bm.faces.new(list(reversed(rings[0]))))
        faces.append(bm.faces.new(rings[-1]))
    for f in faces:
        f.material_index = mat
        f.smooth = True
    return faces


def torus(bm, center, axis, R, r, n=16, m=6, mat=0):
    axis = Vector(axis).normalized()
    u, v = basis(axis)
    pts = [Vector(center) + (u * math.cos(2 * math.pi * i / n) + v * math.sin(2 * math.pi * i / n)) * R for i in range(n + 1)]
    return tube(bm, pts, r, segs=m, cap=False, mat=mat)


def ellipsoid(bm, center, size, rot=(0, 0, 0), subdiv=2, mat=0, smooth=True):
    m = Matrix.LocRotScale(Vector(center), Euler(rot), Vector(size))
    res = bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0, matrix=m)
    faces = list({f for v in res["verts"] for f in v.link_faces})
    for f in faces:
        f.material_index = mat
        f.smooth = smooth
    return faces


def plate(bm, outline, thickness, mat=0, axis="Y", smooth=False):
    """Extruded flat outline. axis: the plate normal ('Y': outline in XZ, 'X': outline in YZ, 'Z': in XY)."""
    def P(a, b, d):
        return {"Y": (a, d, b), "X": (d, a, b), "Z": (a, b, d)}[axis]
    top = [bm.verts.new(P(a, b, thickness / 2)) for a, b in outline]
    bot = [bm.verts.new(P(a, b, -thickness / 2)) for a, b in outline]
    faces = [bm.faces.new(top), bm.faces.new(list(reversed(bot)))]
    n = len(outline)
    faces += [bm.faces.new((top[i], top[(i + 1) % n], bot[(i + 1) % n], bot[i])) for i in range(n)]
    for f in faces:
        f.material_index = mat
        f.smooth = smooth
    return faces


def finish(name, bm, col, mats, parent=None, recalc=True, attrs=None):
    """Mesh object from a bmesh; attrs: {name: per-vertex callback(co) -> float} stored as point attributes."""
    if recalc:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    for an, fn in (attrs or {}).items():
        a = me.attributes.new(an, "FLOAT", "POINT")
        a.data.foreach_set("value", [fn(v.co) for v in me.vertices])
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def rel_attrs(rings):
    """zrel (0 belly .. 1 back) and xrel (0 tail .. 1 nose) for a lofted body."""
    xs = [r[0] for r in rings]
    zs_lo = min(r[3] - r[2] for r in rings)
    zs_hi = max(r[3] + r[2] for r in rings)
    x0, x1 = min(xs), max(xs)
    return {"zrel": lambda c: min(1.0, max(0.0, (c.z - zs_lo) / max(1e-6, zs_hi - zs_lo))),
            "xrel": lambda c: min(1.0, max(0.0, (c.x - x0) / max(1e-6, x1 - x0)))}


def merge_into(bm, sub):
    tmp = bpy.data.meshes.new("tmp")
    sub.to_mesh(tmp)
    sub.free()
    bm.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)


def tex_coord(nt, out="Object"):
    return nt.nodes.new("ShaderNodeTexCoord").outputs[out]


def noise(nt, vec, scale, detail=2.0, rough=0.5, out="Fac"):
    n = nt.nodes.new("ShaderNodeTexNoise")
    _feed(nt, n.inputs["Vector"], vec)
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    return n.outputs[out]


def voronoi(nt, vec, scale, out="Distance", feature="F1", rand=1.0):
    n = nt.nodes.new("ShaderNodeTexVoronoi")
    n.feature = feature
    _feed(nt, n.inputs["Vector"], vec)
    n.inputs["Scale"].default_value = scale
    n.inputs["Randomness"].default_value = rand
    return n.outputs[out]


def bump(nt, height, strength, dist=0.0005):
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = strength
    b.inputs["Distance"].default_value = dist
    _feed(nt, b.inputs["Height"], height)
    return b.outputs["Normal"]


def psdf(nt, color, rough=0.4, metal=0.0, coat=0.0, coat_rough=0.03, sss=0.0, sss_scale=0.002, alpha=1.0,
         transmission=0.0, normal=None, spec=0.5, emission=None, emission_strength=0.0, ior=1.45):
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    _feed(nt, b.inputs["Base Color"], (*color, 1.0) if isinstance(color, tuple) else color)
    _feed(nt, b.inputs["Roughness"], rough)
    _feed(nt, b.inputs["Metallic"], metal)
    b.inputs["Specular IOR Level"].default_value = spec
    b.inputs["IOR"].default_value = ior
    _feed(nt, b.inputs["Coat Weight"], coat)
    b.inputs["Coat Roughness"].default_value = coat_rough
    if sss:
        _feed(nt, b.inputs["Subsurface Weight"], sss)
        b.inputs["Subsurface Scale"].default_value = sss_scale
        _feed(nt, b.inputs["Subsurface Radius"], (1.0, 0.45, 0.25))
    _feed(nt, b.inputs["Alpha"], alpha)
    _feed(nt, b.inputs["Transmission Weight"], transmission)
    if normal is not None:
        nt.links.new(normal, b.inputs["Normal"])
    if emission is not None:
        _feed(nt, b.inputs["Emission Color"], (*emission, 1.0) if isinstance(emission, tuple) else emission)
        _feed(nt, b.inputs["Emission Strength"], emission_strength)
    return b.outputs["BSDF"]


def solid(name, color, rough=0.4, metal=0.0, coat=0.0, bump_scale=0.0, bump_str=0.0, spec=0.5):
    def build(nt):
        nb = bump(nt, noise(nt, tex_coord(nt), bump_scale, 4.0), bump_str) if bump_scale else None
        return psdf(nt, color, rough=rough, metal=metal, coat=coat, normal=nb, spec=spec)
    return new_material(name, build)


def emission_mat(name, color, strength=1.0, alpha_from_object=False, blended=False):
    def build(nt):
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*color, 1.0)
        em.inputs["Strength"].default_value = strength
        if not alpha_from_object:
            return em.outputs[0]
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(oi.outputs["Alpha"], mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    return new_material(name, build, blended=blended)


class Materials:
    pass


def frame_driver_value(nt, scale=1.0):
    v = nt.nodes.new("ShaderNodeValue")
    add_frame_driver(v.outputs[0], "default_value", f"frame * {scale}")
    return v.outputs[0]


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


def look_at(pos, target, up):
    f = (target - pos).normalized()
    up = Vector(up).normalized()
    if abs(f.dot(up)) > 0.995:
        up = Vector((0, 1, 0)) if abs(f.z) > 0.9 else Vector((0, 0, 1))
    r = f.cross(up).normalized()
    u = r.cross(f)
    return Matrix((r, u, -f)).transposed().to_quaternion()


def build_hud_scene():
    """A second scene that holds only the overlays, seen by a static camera with the same lens, rendered with
    a transparent film and laid over the 3D render in the compositor (so depth of field never touches it)."""
    main = bpy.context.scene
    hs = bpy.data.scenes.new("HUD")
    hs.render.engine = "BLENDER_EEVEE"
    hs.render.resolution_x, hs.render.resolution_y = RES_X, RES_Y
    hs.render.fps = FPS
    hs.frame_start, hs.frame_end = 1, FRAME_END
    hs.render.film_transparent = True
    hs.render.use_motion_blur = False
    hs.eevee.taa_render_samples = 16
    hs.view_settings.view_transform = "Standard"
    cd = bpy.data.cameras.new("HUDCamera")
    cd.lens = LENS
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = SENSOR
    cd.clip_start = 0.01
    cd.clip_end = 10.0
    hcam = bpy.data.objects.new("HUDCamera", cd)
    hs.collection.objects.link(hcam)
    hs.camera = hcam
    col = bpy.data.collections.new("Overlays")
    hs.collection.children.link(col)
    return hs, hcam, col


def build_compositor(main, hs):
    ng = bpy.data.node_groups.new("Composite", "CompositorNodeTree")
    main.compositing_node_group = ng
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.scene = main
    img = rl.outputs["Image"]
    try:                                    # gentle bloom on the glossy highlights
        gl = ng.nodes.new("CompositorNodeGlare")
        gl.inputs["Type"].default_value = "Bloom"
        for nm, v in (("Threshold", 1.0), ("Strength", 0.25), ("Size", 0.5)):
            if nm in gl.inputs:
                gl.inputs[nm].default_value = v
        ng.links.new(img, gl.inputs["Image"])
        img = gl.outputs["Image"]
    except Exception as e:                  # noqa: BLE001 (older or newer API: skip the bloom)
        print("bloom skipped:", e)
    try:                                    # cinematic grade on the 3D only: soft S curve
        cv = ng.nodes.new("CompositorNodeCurveRGB")
        c = cv.mapping.curves[3]
        c.points[0].location = (0.0, 0.012)
        c.points[1].location = (1.0, 1.0)
        c.points.new(0.25, 0.21)
        c.points.new(0.75, 0.80)
        cv.mapping.update()
        ng.links.new(img, cv.inputs["Image"])
        img = cv.outputs["Image"]
    except Exception as e:                  # noqa: BLE001
        print("grade skipped:", e)
    hud = ng.nodes.new("CompositorNodeRLayers")
    hud.scene = hs
    ao = ng.nodes.new("CompositorNodeAlphaOver")
    ng.links.new(img, ao.inputs["Background"])
    ng.links.new(hud.outputs["Image"], ao.inputs["Foreground"])
    out = ng.nodes.new("NodeGroupOutput")
    ng.links.new(ao.outputs[0], out.inputs[0])
    return ng


def hud_materials(M):
    def scrim(top_side):
        def build(nt):
            oi = nt.nodes.new("ShaderNodeObjectInfo")
            y = separate(nt, nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"])["Y"]
            prof = nmath(nt, "POWER", y if top_side else nmath(nt, "SUBTRACT", 1.0, y), 1.7)
            tr = nt.nodes.new("ShaderNodeBsdfTransparent")
            em = nt.nodes.new("ShaderNodeEmission")
            em.inputs["Color"].default_value = (0.0, 0.004, 0.006, 1.0)
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

    def soft_plate(nt):
        """Dark backing behind a caption; its opacity comes from the 'soft' vertex attribute (0 at the rim)."""
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        k = nmath(nt, "POWER", attr(nt, "soft", "Fac"), 1.2)
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (0.0, 0.003, 0.005, 1.0)
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(nmath(nt, "MULTIPLY", oi.outputs["Alpha"], k), mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        return mix.outputs[0]
    M.hud_plate = new_material("HudSoftPlate", soft_plate, blended=True)
    for m in (M.hud, M.hud_white, M.hud_accent, M.hud_shadow, M.scrim_top, M.scrim_bot, M.hud_plate):
        m.use_transparency_overlap = True


def key_visibility(objs, spans):
    """Render objects only inside the given frame spans (they cost nothing elsewhere)."""
    for ob in objs:
        ob.hide_render = True
        key(ob, "hide_render", 1)
        for a, b in spans:
            ob.hide_render = False
            key(ob, "hide_render", max(1, a))
            ob.hide_render = True
            key(ob, "hide_render", b + 1)


def aabb_dist(p, lo, hi):
    d = [max(lo[i] - p[i], 0.0, p[i] - hi[i]) for i in range(3)]
    inside = all(lo[i] <= p[i] <= hi[i] for i in range(3))
    return -1.0 if inside else math.sqrt(sum(x * x for x in d))


def curve(points):
    """Smooth 1D curve through control points [(u, value)], cosine eased between them."""
    pts = sorted(points)

    def f(u):
        if u <= pts[0][0]:
            return pts[0][1]
        for (u0, v0), (u1, v1) in zip(pts, pts[1:]):
            if u <= u1:
                t = (u - u0) / (u1 - u0)
                t = 0.5 - 0.5 * math.cos(math.pi * t)
                return v0 + (v1 - v0) * t
        return pts[-1][1]
    return f


def spline_outline(ctrl, samples=10):
    """Closed Catmull-Rom outline through control points [(x, z)], for smooth fin edges."""
    out = []
    n = len(ctrl)
    for i in range(n):
        p0, p1, p2, p3 = (Vector(ctrl[(i + k) % n]) for k in (-1, 0, 1, 2))
        for s in range(samples):
            t = s / samples
            t2, t3 = t * t, t * t * t
            p = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append((p.x, p.y))
    return out


def loft_profile(bm, L, top, bot, wid, nr=56, na=40, power=2.3, ridge=None, mat=0):
    """Body from nose (x = +L/2, u = 0) to tail base (u = 1) with separate back (top), belly (bot) and half
    width (wid) curves; superellipse sections; per-vertex zrel (belly 0 .. back 1) and xrel (tail 0 .. nose 1)."""
    zl = bm.verts.layers.float.new("zrel")
    xl = bm.verts.layers.float.new("xrel")
    rings = []
    for i in range(nr + 1):
        u = i / nr
        x = L / 2 - u * L
        t, b, w = top(u), bot(u), wid(u)
        zc, hh = (t + b) / 2, (t - b) / 2
        ring = []
        for k in range(na):
            a = 2 * math.pi * k / na
            c, s = math.cos(a), math.sin(a)
            r = 1.0 + (ridge(u, a) if ridge else 0.0)
            y = w * r * math.copysign(abs(c) ** (2 / power), c)
            z = zc + hh * r * math.copysign(abs(s) ** (2 / power), s)
            v = bm.verts.new((x, y, z))
            v[zl] = min(1.0, max(0.0, (z - b) / max(1e-6, t - b)))
            v[xl] = 1.0 - u
            ring.append(v)
        rings.append(ring)
    for ra, rb in zip(rings, rings[1:]):
        for k in range(na):
            f = bm.faces.new((ra[k], ra[(k + 1) % na], rb[(k + 1) % na], rb[k]))
            f.material_index, f.smooth = mat, True
    for ring, x_cap, rev in ((rings[0], L / 2, True), (rings[-1], -L / 2, False)):     # rounded end caps
        cz = sum(v.co.z for v in ring) / len(ring)
        tip = bm.verts.new((x_cap + (0.02 * L if rev else -0.005 * L), 0.0, cz))
        tip[zl], tip[xl] = 0.5, 1.0 if rev else 0.0
        for k in range(na):
            vs = (ring[(k + 1) % na], ring[k], tip) if rev else (ring[k], ring[(k + 1) % na], tip)
            f = bm.faces.new(vs)
            f.material_index, f.smooth = mat, True


def soft_fin(bm, ctrl, thick, mat, plane="XZ", offset=(0, 0, 0), rot=None, samples=8):
    """Fin from a smooth spline outline, slightly thicker at the base side (first control point)."""
    outline = spline_outline(ctrl, samples)
    o = Vector(offset)
    R = rot.to_matrix() if rot is not None else None

    def P(a, b, d):
        v = Vector((a, d, b)) if plane == "XZ" else Vector((a, b, d))
        return o + (R @ v if R is not None else v)
    top = [bm.verts.new(P(a, b, thick / 2)) for a, b in outline]
    bot = [bm.verts.new(P(a, b, -thick / 2)) for a, b in outline]
    fs = [bm.faces.new(top), bm.faces.new(list(reversed(bot)))]
    n = len(outline)
    fs += [bm.faces.new((top[i], top[(i + 1) % n], bot[(i + 1) % n], bot[i])) for i in range(n)]
    for f in fs:
        f.material_index, f.smooth = mat, False          # flat: a thin membrane, not a puffy cushion


# --------------------------------------------------------------------------
# Shading: fog, caustics, depth tint (scene properties "under" and "deep" are keyed per frame)
# --------------------------------------------------------------------------
def scene_attr(nt, name):
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_type = "VIEW_LAYER"
    a.attribute_name = name
    return a.outputs["Fac"]


def iter_fcurves(idb):
    ad = idb.animation_data
    if not ad or not ad.action:
        return []
    act_ = ad.action
    out = list(getattr(act_, "fcurves", []) or [])
    for layer in getattr(act_, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                out += list(bag.fcurves)
    return out


def uw_color(nt, z_dir, deep):
    """Underwater fog colour by view direction (bright towards the surface), darker where the shot is deep."""
    c = color_ramp(nt, nmath(nt, "MULTIPLY_ADD", z_dir, 0.5, 0.5, clamp=True), UW_RAMP)
    return vmath(nt, "SCALE", c, scale=nmath(nt, "MULTIPLY_ADD", deep, -0.8, 1.0))


def build_atmos_group():
    """Grey-blue fog by distance when the lens is under water, a sea haze above it."""
    ng, gi, go = new_group("Atmosphere", "ShaderNodeTree", [("Shader", "NodeSocketShader")],
                           [("Shader", "NodeSocketShader")])
    geo = ng.nodes.new("ShaderNodeNewGeometry")
    dist = ng.nodes.new("ShaderNodeCameraData").outputs["View Distance"]
    inc = separate(ng, geo.outputs["Incoming"])
    under, deep, clear = scene_attr(ng, "under"), scene_attr(ng, "deep"), scene_attr(ng, "clear")
    dens = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY_ADD", deep, 0.6, 1.0), -1.0 / VIS)
    dens = nmath(ng, "MULTIPLY", dens, nmath(ng, "MULTIPLY_ADD", clear, -0.9, 1.0))       # the cross section reads clean
    fog = nmath(ng, "MULTIPLY", under, nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(ng, "MULTIPLY", dist, dens))))
    haze = nmath(ng, "MULTIPLY", nmath(ng, "SUBTRACT", 1.0, under),
                 nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(ng, "MULTIPLY", dist, -1.0 / AIR_HAZE))))
    em_f = ng.nodes.new("ShaderNodeEmission")
    ng.links.new(uw_color(ng, nmath(ng, "MULTIPLY", inc["Z"], -1.0), deep), em_f.inputs["Color"])
    em_h = ng.nodes.new("ShaderNodeEmission")
    em_h.inputs["Color"].default_value = (*srgb(150, 166, 176), 1.0)
    em_h.inputs["Strength"].default_value = 0.9
    s = gi.outputs["Shader"]
    for fac, em in ((haze, em_h), (fog, em_f)):
        m = ng.nodes.new("ShaderNodeMixShader")
        ng.links.new(fac, m.inputs[0])
        ng.links.new(s, m.inputs[1])
        ng.links.new(em.outputs[0], m.inputs[2])
        s = m.outputs[0]
    ng.links.new(s, go.inputs["Shader"])
    return ng


def world_pos(nt):
    return nt.nodes.new("ShaderNodeNewGeometry").outputs["Position"]


def loop_time(nt):
    """Scene seconds, zero at frame 1."""
    return frame_driver_value(nt, 1.0 / FPS)


def looped_noise(nt, p, drift, scale, detail=3.0, rough=0.55, out="Fac"):
    """Noise drifting with time that repeats every LOOP_S seconds: two copies a period apart, cross-faded."""
    t = loop_time(nt)
    a = nmath(nt, "DIVIDE", nmath(nt, "MODULO", t, LOOP_S), LOOP_S)
    tm = nmath(nt, "MODULO", t, LOOP_S)
    n1 = noise(nt, vmath(nt, "ADD", p, vmath(nt, "SCALE", drift, scale=tm)), scale, detail, rough, out=out)
    n2 = noise(nt, vmath(nt, "ADD", p, vmath(nt, "SCALE", drift, scale=nmath(nt, "SUBTRACT", tm, LOOP_S))), scale,
               detail, rough, out=out)
    if out == "Fac":
        return nmath(nt, "ADD", nmath(nt, "MULTIPLY", n1, nmath(nt, "SUBTRACT", 1.0, a)), nmath(nt, "MULTIPLY", n2, a))
    return mix_rgb(nt, a, n1, n2)


def caustics(nt, strength=1.0):
    """Animated caustic network on surfaces that face the light, fading with depth (and off above water)."""
    p = world_pos(nt)
    t = loop_time(nt)
    c = 0.0
    warp = vmath(nt, "SCALE", vmath(nt, "SUBTRACT", noise(nt, vmath(nt, "MULTIPLY", p, (0.4, 0.4, 0.15)), 1.0, 2.0, 0.5,
                                                         out="Color"), (0.5, 0.5, 0.5)), scale=0.35)
    q = vmath(nt, "ADD", vmath(nt, "MULTIPLY", p, (0.7, 0.7, 0.2)), warp)
    for sc_, speed, k in ((1.6, 0.4, 0.6), (2.7, -0.7, 0.45)):
        n = nt.nodes.new("ShaderNodeTexVoronoi")
        n.voronoi_dimensions = "4D"
        n.feature = "DISTANCE_TO_EDGE"
        _feed(nt, n.inputs["Vector"], q)
        _feed(nt, n.inputs["W"], nmath(nt, "MULTIPLY", t, speed))
        n.inputs["Scale"].default_value = sc_
        line = map_range(nt, n.outputs["Distance"], 0.0, 0.08, 1.0, 0.0)
        line = nmath(nt, "POWER", line, 2.2)
        c = nmath(nt, "MULTIPLY_ADD", line, k, c) if not isinstance(c, float) else nmath(nt, "MULTIPLY", line, k)
    nz = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Normal"])["Z"]
    z = separate(nt, p)["Z"]
    mask = nmath(nt, "MULTIPLY", map_range(nt, nz, 0.05, 0.8), nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", z, 1.0 / 9.0)))
    mask = nmath(nt, "MULTIPLY", mask, nmath(nt, "MULTIPLY", scene_attr(nt, "under"),
                                             nmath(nt, "MULTIPLY_ADD", scene_attr(nt, "deep"), -0.9, 1.0)))
    return nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", c, mask), strength)


def uw_base(nt, col, caustic=1.0):
    """Base colour under water: red absorbed with depth, darker deeper, plus the caustic light."""
    z = separate(nt, world_pos(nt))["Z"]
    deep = nmath(nt, "SUBTRACT", 1.0, nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", nmath(nt, "MINIMUM", z, 0.0), 1.0 / 8.0)))
    deep = nmath(nt, "MULTIPLY", deep, scene_attr(nt, "under"))
    tint = mix_rgb(nt, deep, (1.0, 1.0, 1.0), srgb(90, 170, 200))
    c = vmath(nt, "MULTIPLY", (*col,) if isinstance(col, tuple) else col, tint)
    if caustic:
        c = vmath(nt, "SCALE", c, scale=nmath(nt, "ADD", 1.0, caustics(nt, 0.8 * caustic)))
    return c


def build_world(sc):
    """Under water: the fog colour itself (infinitely far). Above water: a clear tropical sky with scattered trade
    wind cumulus, with a soft shoulder on camera rays (the Standard view would clip the bright sky to white), and
    deep blue sea below the horizon. Switched by the "under" value."""
    w = bpy.data.worlds.new("AtlanticWorld")
    sc.world = w
    try:
        w.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    under = nt.nodes.new("ShaderNodeValue")
    under.name = "Under"
    deep = nt.nodes.new("ShaderNodeValue")
    deep.name = "Deep"
    dirv = vmath(nt, "NORMALIZE", nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"])
    view = separate(nt, dirv)
    fogc = uw_color(nt, view["Z"], deep.outputs[0])
    sky = nt.nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "MULTIPLE_SCATTERING"
    except TypeError:
        sky.sky_type = "NISHITA"
    sky.sun_elevation = math.asin(SUN_DIR.z)
    sky.sun_rotation = -math.atan2(SUN_DIR.x, SUN_DIR.y)
    sky.sun_disc = False
    sky.altitude = 50.0
    sky.air_density = 1.3
    try:
        sky.aerosol_density = 2.0
    except AttributeError:
        pass
    lit = vmath(nt, "SCALE", sky.outputs[0], scale=0.085)
    # cloud deck: noise on the plane z = 1 seen through the view direction (static, so the loop holds)
    zc = nmath(nt, "MAXIMUM", view["Z"], 0.02)
    uv = combine(nt, nmath(nt, "DIVIDE", view["X"], zc), nmath(nt, "DIVIDE", view["Y"], zc), 0.0)
    cl = noise(nt, vmath(nt, "MULTIPLY", uv, (0.9, 1.6, 1.0)), 1.6, 8.0, 0.62)
    cl2 = noise(nt, vmath(nt, "MULTIPLY", uv, (3.0, 5.0, 1.0)), 2.0, 6.0, 0.6)
    cover = map_range(nt, nmath(nt, "MULTIPLY_ADD", cl2, 0.35, cl), 0.62, 0.78)        # scattered cumulus
    to_sun = nmath(nt, "ADD", nmath(nt, "ADD", nmath(nt, "MULTIPLY", view["X"], SUN_DIR.x),
                                     nmath(nt, "MULTIPLY", view["Y"], SUN_DIR.y)), nmath(nt, "MULTIPLY", view["Z"], SUN_DIR.z))
    sun_side = map_range(nt, to_sun, 0.6, 1.0)
    cloud_dark = mix_rgb(nt, sun_side, srgb(176, 186, 200), srgb(255, 250, 242))
    cloud_lit = mix_rgb(nt, nmath(nt, "MULTIPLY", cl2, 0.6), cloud_dark, srgb(214, 220, 228))
    near_h = map_range(nt, view["Z"], 0.012, 0.14)                  # clouds thin out into haze at the horizon
    # (fully gone by the horizon: the projected noise aliases into a comb of streaks there)
    skyc = mix_rgb(nt, nmath(nt, "MULTIPLY", cover, nmath(nt, "POWER", near_h, 1.3)), lit,
                   vmath(nt, "SCALE", cloud_lit, scale=0.75))
    knee, top = 0.55, 0.97
    sp = separate(nt, skyc)
    peak = nmath(nt, "MAXIMUM", sp["X"], nmath(nt, "MAXIMUM", sp["Y"], sp["Z"]))
    over = nmath(nt, "MAXIMUM", nmath(nt, "SUBTRACT", peak, knee), 0.0)
    rolled = nmath(nt, "ADD", nmath(nt, "MINIMUM", peak, knee),
                   nmath(nt, "DIVIDE", over, nmath(nt, "ADD", nmath(nt, "MULTIPLY", over, 1.0 / (top - knee)), 1.0)))
    soft = vmath(nt, "SCALE", skyc, scale=nmath(nt, "DIVIDE", rolled, nmath(nt, "MAXIMUM", peak, 1e-5)))
    is_cam = nt.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"]
    skyc = vmath(nt, "ADD", skyc, vmath(nt, "SCALE", vmath(nt, "SUBTRACT", soft, skyc), scale=is_cam))
    skyc = mix_rgb(nt, map_range(nt, view["Z"], 0.0, -0.03), skyc, srgb(16, 40, 40))     # the sea past the horizon
    col = mix_rgb(nt, under.outputs[0], skyc, fogc)
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(col, bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    return w, under, deep


# --------------------------------------------------------------------------
# The sea: one grid, fine at the boat and coarse to the horizon, displaced by swells whose frequencies are whole
# multiples of the loop rate (2 pi / 55 s), so the sea at frame 1321 is the sea at frame 1.
# --------------------------------------------------------------------------
G = 9.81
SWELLS = []                                      # (amplitude, kx, ky, omega, phase, steepness)
for _n, _amp, _deg, _q in ((9, 0.55, 8.0, 0.5), (11, 0.38, 33.0, 0.5), (14, 0.25, -24.0, 0.5), (19, 0.15, 61.0, 0.5),
                           (26, 0.08, -66.0, 0.45), (37, 0.045, 14.0, 0.4), (52, 0.022, -38.0, 0.35)):   # trade wind chop
    _w = 2 * math.pi * _n / LOOP_S
    _k = _w * _w / G
    _a = math.radians(_deg)
    SWELLS.append((_amp, _k * math.cos(_a), _k * math.sin(_a), _w, 1.7 * _n % (2 * math.pi), _q))
SEA_SIZE = 2000.0                                # half extent of the sea grid (m)


def sea_height(x, y, t):
    """The swell surface height at (x, y) and time t (seconds), matching the geometry (vertical part only)."""
    h = 0.0
    for a, kx, ky, w, ph, _q in SWELLS:
        h += a * math.cos(kx * x + ky * y - w * t + ph)
    return h


def sea_point(x, y, t):
    """Where the surface vertex that rests at (x, y) is at time t (with the Gerstner sideways motion)."""
    dx = dy = h = 0.0
    for a, kx, ky, w, ph, q in SWELLS:
        k = math.hypot(kx, ky)
        arg = kx * x + ky * y - w * t + ph
        dx -= q * a * kx / k * math.sin(arg)
        dy -= q * a * ky / k * math.sin(arg)
        h += a * math.cos(arg)
    return Vector((x + dx, y + dy, h))


def surface_z(x, y, t):
    """Height of the displaced (Gerstner) surface over world (x, y): find the rest point that lands there."""
    x0, y0 = x, y
    for _ in range(5):
        p = sea_point(x0, y0, t)
        x0, y0 = x0 + (x - p.x), y0 + (y - p.y)
    return sea_point(x0, y0, t).z


def sea_normal(x, y, t):
    e = 0.3
    p0, px, py = sea_point(x, y, t), sea_point(x + e, y, t), sea_point(x, y + e, t)
    return (px - p0).cross(py - p0).normalized()


def sea_group(name="SeaGN"):
    ng, gi, go = new_group(name, "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links
    grid = nodes.new("GeometryNodeMeshGrid")
    grid.inputs["Size X"].default_value = 2.0
    grid.inputs["Size Y"].default_value = 2.0
    grid.inputs["Vertices X"].default_value = 720
    grid.inputs["Vertices Y"].default_value = 720
    sx, sy, _z = gn_xyz(ng)

    def stretch(s):        # dense in the middle (about 0.2 m spacing), 2 km at the rim
        s5 = nmath(ng, "MULTIPLY", nmath(ng, "POWER", nmath(ng, "ABSOLUTE", s), 5.0), nmath(ng, "SIGN", s))
        return nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY_ADD", s, 0.03, nmath(ng, "MULTIPLY", s5, 0.97)), SEA_SIZE)
    x, y = stretch(sx), stretch(sy)
    # the grid is centred on the object (moved per shot to where the camera works, so the fine detail is there);
    # the waves themselves use world coordinates and do not move with it
    so = nodes.new("GeometryNodeSelfObject")
    oi = nodes.new("GeometryNodeObjectInfo")
    links.new(so.outputs[0], oi.inputs["Object"])
    loc = separate(ng, oi.outputs["Location"])
    wx, wy = nmath(ng, "ADD", x, loc["X"]), nmath(ng, "ADD", y, loc["Y"])
    t = nmath(ng, "DIVIDE", nmath(ng, "SUBTRACT", ng.nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], 1.0), float(FPS))
    r = nmath(ng, "SQRT", nmath(ng, "ADD", nmath(ng, "MULTIPLY", x, x), nmath(ng, "MULTIPLY", y, y)))
    dx = dy = h = None
    crest = None
    for i, (a, kx, ky, w, ph, q) in enumerate(SWELLS):
        k = math.hypot(kx, ky)
        lam = 2 * math.pi / k
        fade = (map_range(ng, r, lam * 8.0, lam * 20.0, 1.0, 0.0) if i >= 2       # no aliasing far away
                else map_range(ng, r, lam * 14.0, lam * 30.0, 1.0, 0.0))
        arg = nmath(ng, "ADD", nmath(ng, "SUBTRACT", nmath(ng, "ADD", nmath(ng, "MULTIPLY", wx, kx), nmath(ng, "MULTIPLY", wy, ky)),
                                     nmath(ng, "MULTIPLY", t, w)), ph)
        cs, sn = nmath(ng, "COSINE", arg), nmath(ng, "SINE", arg)
        amp = a if isinstance(fade, float) else nmath(ng, "MULTIPLY", fade, a)
        th = nmath(ng, "MULTIPLY", cs, amp)
        tx = nmath(ng, "MULTIPLY", sn, nmath(ng, "MULTIPLY", amp, -q * kx / k))
        ty = nmath(ng, "MULTIPLY", sn, nmath(ng, "MULTIPLY", amp, -q * ky / k))
        tc = nmath(ng, "MULTIPLY", cs, nmath(ng, "MULTIPLY", amp, q * k))   # how sharp the crest is here
        h = th if h is None else nmath(ng, "ADD", h, th)
        dx = tx if dx is None else nmath(ng, "ADD", dx, tx)
        dy = ty if dy is None else nmath(ng, "ADD", dy, ty)
        crest = tc if crest is None else nmath(ng, "ADD", crest, tc)
    sp = nodes.new("GeometryNodeSetPosition")
    links.new(grid.outputs["Mesh"], sp.inputs["Geometry"])
    links.new(combine(ng, nmath(ng, "ADD", x, dx), nmath(ng, "ADD", y, dy), h), sp.inputs["Position"])
    st = nodes.new("GeometryNodeStoreNamedAttribute")
    st.data_type = "FLOAT"
    st.domain = "POINT"
    st.inputs["Name"].default_value = "crest"
    links.new(sp.outputs[0], st.inputs["Geometry"])
    links.new(crest, st.inputs["Value"])
    st2 = nodes.new("GeometryNodeStoreNamedAttribute")
    st2.data_type = "FLOAT_VECTOR"
    st2.domain = "POINT"
    st2.inputs["Name"].default_value = "rest"
    links.new(st.outputs[0], st2.inputs["Geometry"])
    links.new(combine(ng, wx, wy, 0.0), st2.inputs["Value"])
    sm = nodes.new("GeometryNodeSetShadeSmooth")
    links.new(st2.outputs[0], sm.inputs["Geometry"])
    links.new(sm.outputs[0], go.inputs["Geometry"])
    return ng


def build_sea(col, M):
    ob = bpy.data.objects.new("Sea", bpy.data.meshes.new("Sea"))
    col.objects.link(ob)
    ng = sea_group()
    ob.modifiers.new("Swell", "NODES").node_group = ng
    setm = ng.nodes.new("GeometryNodeSetMaterial")
    setm.inputs["Material"].default_value = M.sea
    out = next(n for n in ng.nodes if n.bl_idname == "NodeGroupOutput")
    src = out.inputs[0].links[0].from_socket
    ng.links.new(src, setm.inputs["Geometry"])
    ng.links.new(setm.outputs[0], out.inputs[0])
    ob.data.materials.append(M.sea)
    ob.visible_shadow = False
    return ob


def build_surface_under(col, M, center=(0, 0, 0), size=160.0):
    """The sea surface seen from below, a flat bright plane (the swell is lost in the haze at these depths)."""
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size, matrix=Matrix.Translation(center))
    ob = finish("SurfaceFromBelow", bm, col, [M.surface_under], recalc=False)
    ob.visible_shadow = False
    return ob


def specks(name, col, mat, center, size, count, rad, seed):
    """Suspended particles: many small icosahedra in one mesh, built with numpy (fast for tens of thousands)."""
    rng = np.random.default_rng(seed)
    t = (1.0 + 5 ** 0.5) / 2
    V = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    V /= np.linalg.norm(V, axis=1)[:, None]
    F = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
                  (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
                  (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)], int)
    P = rng.uniform(-1, 1, (count, 3)) * np.array(size, float)
    S = rad * rng.uniform(0.4, 1.5, count)
    R = rng.random(count)
    verts = (P[:, None, :] + V[None, :, :] * S[:, None, None]).reshape(-1, 3)
    faces = (F[None, :, :] + 12 * np.arange(count)[:, None, None]).reshape(-1, 3)
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.validate()
    a = me.attributes.new("rand", "FLOAT", "POINT")
    a.data.foreach_set("value", np.repeat(R, 12))
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    ob.location = center
    ob.visible_shadow = False
    return ob


def build_shafts(col, M, center, n=7, seed=5, spread=8.0, length=30.0):
    """Sun shafts slanting down from the surface along the refracted sun direction (fewer, stronger)."""
    rng = random.Random(seed)
    d = -SUN_UNDER
    obs = []
    for i in range(n):
        c = Vector(center) + Vector((rng.uniform(-spread, spread), rng.uniform(-spread, spread), 0.0))
        r0, r1 = rng.uniform(0.25, 0.7), rng.uniform(0.9, 2.0)
        bm = bmesh.new()
        bm_frustum(bm, c, c + d * length, r0, r1, segs=10, cap0=False, cap1=False)
        ob = finish(f"Shaft{i}", bm, col, [M.shaft], recalc=False)
        for pl in ob.data.polygons:                          # smooth: a flat shaded frustum reads as a plank
            pl.use_smooth = True
        ob.visible_shadow = False
        ob.color = (1, 1, 1, rng.uniform(0.5, 1.0))
        obs.append(ob)
    return obs


def refracted_sun():
    """Direction towards the light as it arrives under water: Snell's law steepens a low sun."""
    h = Vector((SUN_DIR.x, SUN_DIR.y, 0.0)).normalized()
    ang = math.asin(min(1.0, math.cos(math.asin(SUN_DIR.z)) / 1.333))
    return (h * math.sin(ang) + Vector((0, 0, math.cos(ang)))).normalized()


SUN_UNDER = refracted_sun()


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------
def build_materials(atmos):
    M = Materials()
    M.atmos = atmos

    def mat(name, build, blended=False):
        return new_material(name, build, atmos, blended=blended)

    def sea(nt):
        """Murky green Atlantic seen from above: dark bottle green water, sky reflections from the Fresnel term,
        ripples, boils of churned water and drifting foam, and the wreck's long dark shape seen faintly through the
        water (painted: the water is opaque from above). World position, not the stored "rest" attribute."""
        wp = separate(nt, world_pos(nt))
        p = combine(nt, wp["X"], wp["Y"], 0.0)
        crest = attr(nt, "crest", "Fac")
        rip = looped_noise(nt, p, (0.5, 0.2, 0.0), 1.4, 6.0, 0.6)
        rip2 = looped_noise(nt, p, (-0.3, 0.45, 0.0), 6.0, 4.0, 0.55)
        boil = looped_noise(nt, vmath(nt, "MULTIPLY", p, (0.35, 0.35, 1.0)), (0.12, 0.05, 0.0), 1.0, 4.0, 0.6)
        foam_n = looped_noise(nt, vmath(nt, "MULTIPLY", p, (1.2, 1.2, 1.0)), (0.2, 0.1, 0.0), 1.2, 5.0, 0.65)
        rel = separate(nt, vmath(nt, "SUBTRACT", p, (WRECK_AT.x, WRECK_AT.y, 0.0)))
        bow = nmath(nt, "POWER", nmath(nt, "MAXIMUM", nmath(nt, "SUBTRACT", nmath(nt, "DIVIDE", rel["X"], WRECK_L / 2), 0.55),
                                       0.0), 1.6)
        hw = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY_ADD", bow, -0.88 / 0.45 ** 1.6, 1.0), WRECK_B / 2 - 0.6)
        hull_d = nmath(nt, "MAXIMUM", nmath(nt, "SUBTRACT", nmath(nt, "ABSOLUTE", rel["X"]), WRECK_L / 2 - 1.0),
                       nmath(nt, "SUBTRACT", nmath(nt, "ABSOLUTE", rel["Y"]), hw))
        over = map_range(nt, hull_d, 1.6, -1.2)                              # over the wreck (a soft edge)
        shadow = nmath(nt, "MULTIPLY", over, map_range(nt, boil, 0.3, 0.7, 0.75, 0.95))
        churn = nmath(nt, "MULTIPLY", map_range(nt, boil, 0.55, 0.72), nmath(nt, "MULTIPLY_ADD", over, 0.8, 0.2))
        foam = nmath(nt, "MULTIPLY", map_range(nt, foam_n, 0.62, 0.74), nmath(nt, "MULTIPLY_ADD", churn, 0.7, 0.04))
        foam = nmath(nt, "MINIMUM", nmath(nt, "ADD", foam, nmath(nt, "MULTIPLY", map_range(nt, crest, 0.6, 0.9), 0.25)), 1.0)
        water = color_ramp(nt, map_range(nt, crest, -0.4, 0.6), [(0.0, srgb(16, 46, 40)), (0.7, srgb(28, 72, 60)),
                                                                   (1.0, srgb(52, 108, 88))])
        water = mix_rgb(nt, nmath(nt, "MULTIPLY", churn, 0.3), water, srgb(66, 112, 86))      # silt stirred up
        water = mix_rgb(nt, nmath(nt, "MULTIPLY", shadow, 0.95), water, srgb(3, 10, 10))      # the wreck below
        col = mix_rgb(nt, foam, water, srgb(214, 224, 214))
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        _feed(nt, b.inputs["Roughness"], map_range(nt, foam, 0.0, 1.0, 0.12, 0.6, smooth=False))
        b.inputs["IOR"].default_value = 1.333
        b.inputs["Specular IOR Level"].default_value = 0.6
        h = nmath(nt, "ADD", nmath(nt, "ADD", rip, nmath(nt, "MULTIPLY", rip2, 0.35)), nmath(nt, "MULTIPLY", churn, 0.6))
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = 0.3
        bp.inputs["Distance"].default_value = 0.08
        nt.links.new(h, bp.inputs["Height"])
        nt.links.new(bp.outputs[0], b.inputs["Normal"])
        return b.outputs[0]
    M.sea = mat("Sea", sea)

    def surface_under(nt):
        """The surface from below: a bright rippling window overhead, darker total-reflection rim."""
        p = tex_coord(nt, "Object")
        rip = looped_noise(nt, p, (0.4, 0.2, 0.0), 0.35, 5.0, 0.6)
        inc = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Incoming"])["Z"]
        window = map_range(nt, nmath(nt, "ABSOLUTE", inc), 0.62, 0.8)
        col = mix_rgb(nt, window, srgb(10, 34, 30), srgb(150, 196, 168))
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(col, em.inputs["Color"])
        nt.links.new(nmath(nt, "MULTIPLY", map_range(nt, rip, 0.3, 0.7, 0.6, 1.5),
                           nmath(nt, "MULTIPLY_ADD", scene_attr(nt, "deep"), -0.9, 1.0)), em.inputs["Strength"])
        return em.outputs[0]
    M.surface_under = mat("SurfaceFromBelow", surface_under)

    def shaft(nt):
        """Additive light shaft: brightest along its core (seen side on), fading down its length."""
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.5
        core = nmath(nt, "POWER", nmath(nt, "SUBTRACT", 1.0, lw.outputs["Facing"]), 5.0)
        zw = separate(nt, world_pos(nt))["Z"]
        along = nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", zw, 1.0 / 9.0))
        along = nmath(nt, "MULTIPLY", along, map_range(nt, zw, -0.4, -4.5))       # no hard rim where it leaves the surface
        dist = nt.nodes.new("ShaderNodeCameraData").outputs["View Distance"]
        along = nmath(nt, "MULTIPLY", along, map_range(nt, dist, 0.5, 4.0))
        along = nmath(nt, "MULTIPLY", along, 0.16)
        flick = map_range(nt, looped_noise(nt, tex_coord(nt, "Object"), (0.0, 0.0, 0.25), 1.2), 0.3, 0.7, 0.55, 1.0)
        k = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", core, along), flick)
        k = nmath(nt, "MULTIPLY", k, nmath(nt, "MULTIPLY", scene_attr(nt, "under"),
                                           nmath(nt, "MULTIPLY_ADD", scene_attr(nt, "deep"), -0.9, 1.0)))
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(196, 230, 196), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", k, oi.outputs["Alpha"]), em.inputs["Strength"])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(tr.outputs[0], add.inputs[1])
        return add.outputs[0]
    M.shaft = new_material("SunShaft", shaft, blended=True)

    def plankton(nt):
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(170, 200, 196), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", attr(nt, "rand", "Fac"), 0.35), em.inputs["Strength"])
        df = nt.nodes.new("ShaderNodeBsdfDiffuse")
        df.inputs["Color"].default_value = (*srgb(120, 140, 136), 1.0)
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(df.outputs[0], add.inputs[1])
        return add.outputs[0]
    M.specks = mat("Plankton", plankton)

    def biolum(nt):
        """A stylised bioluminescent glint (the brief's particle trails): teal, brighter while young."""
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(90, 255, 220), 1.0)
        em.inputs["Strength"].default_value = 6.0
        return em.outputs[0]
    M.biolum = new_material("Bioluminescence", biolum)

    # --- the goliath grouper -------------------------------------------------------------------------------------
    def goliath(nt):
        """Goliath grouper: thick, leathery skin, brownish yellow to olive grey, paler below; irregular dark bars
        slanting across the flanks, small black spots on the head and fins; heavy subsurface scattering (thick
        skin), a dull slime sheen."""
        z, x = attr(nt, "zrel", "Fac"), attr(nt, "xrel", "Fac")
        p = tex_coord(nt, "Object")
        o = separate(nt, p)
        base = color_ramp(nt, z, [(0.0, srgb(178, 166, 132)), (0.3, srgb(150, 136, 98)), (0.55, srgb(112, 98, 64)),
                                  (0.8, srgb(84, 74, 50)), (1.0, srgb(66, 58, 42))])
        n1 = noise(nt, vmath(nt, "MULTIPLY", p, (6.0, 6.0, 6.0)), 1.0, 5.0, 0.6)
        mott = map_range(nt, noise(nt, p, 16.0, 6.0, 0.6), 0.3, 0.7)
        base = mix_rgb(nt, nmath(nt, "MULTIPLY", mott, 0.35), base, srgb(146, 140, 104))
        slant = nmath(nt, "ADD", nmath(nt, "MULTIPLY", o["X"], 50.0), nmath(nt, "MULTIPLY", o["Z"], 24.0))
        bars = map_range(nt, nmath(nt, "SINE", nmath(nt, "ADD", slant, nmath(nt, "MULTIPLY", n1, 2.5))), 0.1, 0.8)
        bars = nmath(nt, "MULTIPLY", bars, nmath(nt, "MULTIPLY", map_range(nt, z, 0.15, 0.4), map_range(nt, x, 0.12, 0.3)))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", bars, 0.7), base, srgb(40, 32, 22))
        spots = map_range(nt, voronoi(nt, p, 60.0), 0.14, 0.07)
        spots = nmath(nt, "MULTIPLY", spots, nmath(nt, "MULTIPLY", map_range(nt, x, 0.7, 0.85), map_range(nt, z, 0.2, 0.4)))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", spots, 0.8), col, srgb(14, 12, 10))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", attr(nt, "lip", "Fac"), 0.3), col, srgb(118, 98, 76))
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(uw_base(nt, col, caustic=0.6), b.inputs["Base Color"])
        _feed(nt, b.inputs["Roughness"], map_range(nt, n1, 0.0, 1.0, 0.5, 0.7, smooth=False))
        b.inputs["Coat Weight"].default_value = 0.08
        b.inputs["Coat Roughness"].default_value = 0.2
        b.inputs["Subsurface Weight"].default_value = 0.3
        b.inputs["Subsurface Scale"].default_value = 0.03
        b.inputs["Subsurface Radius"].default_value = (1.0, 0.55, 0.35)
        nt.links.new(bump(nt, nmath(nt, "MULTIPLY_ADD", noise(nt, p, 120.0, 4.0, 0.55), 0.3, nmath(nt, "MULTIPLY", n1, 0.7)), 0.35, 0.01),
                     b.inputs["Normal"])
        return b.outputs[0]
    M.goliath = mat("GoliathSkin", goliath)

    def goliath_fin(nt):
        g = separate(nt, tex_coord(nt, "Generated"))
        p = tex_coord(nt, "Object")
        edge = map_range(nt, nmath(nt, "MAXIMUM", nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", g["Z"], 0.5)),
                                   nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", g["X"], 0.5))), 0.3, 0.5)
        col = mix_rgb(nt, edge, srgb(96, 82, 56), srgb(52, 44, 32))
        spots = map_range(nt, voronoi(nt, p, 50.0), 0.16, 0.08)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", spots, 0.8), col, srgb(14, 12, 10))
        o = separate(nt, p)
        ray = nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY", o["X"], 40.0))), 8.0)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", ray, 0.15), col, srgb(30, 26, 20))
        return psdf(nt, uw_base(nt, col, caustic=0.4), rough=0.45, sss=0.25, sss_scale=0.02, alpha=0.96)
    M.goliath_fin = mat("GoliathFin", goliath_fin)
    M.throat = mat("GoliathThroat", lambda nt: psdf(nt, color_ramp(nt, attr(nt, "zrel", "Fac"),
                   [(0.0, srgb(22, 10, 9)), (1.0, srgb(70, 38, 32))]), rough=0.6, sss=0.3, sss_scale=0.02, coat=0.1))
    M.gill = mat("GoliathGillRaker", lambda nt: psdf(nt, srgb(150, 40, 40), rough=0.4, sss=0.5, sss_scale=0.01))

    def eye(nt):
        """Goliath eye: small, a dark pupil in a golden brown iris under a glossy cornea."""
        r = attr(nt, "eyer", "Fac")
        col = color_ramp(nt, r, [(0.0, srgb(2, 2, 3)), (0.45, srgb(2, 2, 3)), (0.5, srgb(150, 110, 48)),
                                 (0.66, srgb(118, 86, 40)), (0.8, srgb(70, 54, 30)), (1.0, srgb(60, 50, 36))])
        return psdf(nt, col, rough=0.04, metal=0.1, coat=1.0, coat_rough=0.02, spec=0.8)
    M.eye = mat("FishEye", eye)

    def small_eye(nt):
        """Small fish eye: silvery white sclera around a black pupil."""
        r = attr(nt, "eyer", "Fac")
        col = color_ramp(nt, r, [(0.0, srgb(2, 2, 3)), (0.66, srgb(2, 2, 3)), (0.69, srgb(200, 170, 96)),
                                 (0.72, srgb(150, 150, 140)), (0.87, srgb(120, 122, 118)), (1.0, srgb(80, 84, 86))])
        return psdf(nt, col, rough=0.05, coat=1.0, coat_rough=0.02, spec=0.8)
    M.small_eye = mat("SmallFishEye", small_eye)

    def jack(nt):
        """Jack crevalle: olive green to blue green back, brassy silver sides, a yellow wash below, a black spot on
        the gill cover edge."""
        z, x = attr(nt, "zrel", "Fac"), attr(nt, "xrel", "Fac")
        base = color_ramp(nt, z, [(0.0, srgb(226, 214, 150)), (0.35, srgb(214, 206, 170)), (0.55, srgb(170, 172, 150)),
                                  (0.72, srgb(72, 100, 84)), (1.0, srgb(40, 64, 60))])
        spot = nmath(nt, "MULTIPLY", map_range(nt, nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", x, 0.76)), 0.025, 0.008),
                     map_range(nt, nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", z, 0.62)), 0.07, 0.03))
        col = mix_rgb(nt, spot, base, srgb(10, 10, 12))
        return psdf(nt, uw_base(nt, col, caustic=0.3), rough=0.22, metal=0.45, coat=0.6, coat_rough=0.08)
    M.jack = mat("JackCrevalle", jack)
    M.jack_fin = mat("JackFin", lambda nt: psdf(nt, uw_base(nt, srgb(214, 182, 70), caustic=0.2), rough=0.4, alpha=0.92))
    M.small_fin = mat("SmallFin", lambda nt: psdf(nt, uw_base(nt, srgb(110, 120, 110), caustic=0.2), rough=0.4, alpha=0.9))

    def baitfish(nt):
        z = attr(nt, "zrel", "Fac")
        col = color_ramp(nt, z, [(0.0, srgb(210, 214, 206)), (0.55, srgb(160, 176, 160)), (0.7, srgb(50, 76, 66)),
                                 (1.0, srgb(24, 44, 40))])
        return psdf(nt, uw_base(nt, col, caustic=0.3), rough=0.25, metal=0.5, coat=0.4)
    M.baitfish = mat("Baitfish", baitfish)
    M.crab = mat("Crab", lambda nt: psdf(nt, uw_base(nt, color_ramp(nt, map_range(nt, voronoi(nt, tex_coord(nt, "Object"), 30.0), 0.0, 0.3),
                 [(0.0, srgb(120, 36, 28)), (1.0, srgb(132, 116, 92))]), caustic=0.15), rough=0.55, coat=0.1))

    # --- tackle ------------------------------------------------------------------------------------------------
    M.reel_gold = mat("ReelGold", lambda nt: psdf(nt, srgb(214, 168, 70), rough=0.22, metal=1.0, coat=0.3))
    M.reel_dark = mat("ReelDark", lambda nt: psdf(nt, srgb(26, 26, 28), rough=0.35, metal=0.6))
    M.chrome = mat("Chrome", lambda nt: psdf(nt, srgb(200, 204, 210), rough=0.12, metal=1.0))
    M.blank = mat("RodBlank", lambda nt: psdf(nt, srgb(214, 216, 210), rough=0.22, coat=0.8, coat_rough=0.05))  # pale: reads against the hull
    M.wrap = mat("RodWrap", lambda nt: psdf(nt, srgb(170, 30, 30), rough=0.3, coat=0.6))
    M.grip = mat("RodGrip", lambda nt: psdf(nt, srgb(30, 30, 32), rough=0.7, normal=bump(nt, noise(nt, tex_coord(nt), 200.0), 0.3)))
    M.line = mat("Line", lambda nt: psdf(nt, srgb(214, 220, 140), rough=0.3, emission=srgb(214, 220, 140), emission_strength=0.25))
    M.hook = mat("Hook", lambda nt: psdf(nt, srgb(58, 56, 54), rough=0.3, metal=1.0, coat=0.3))
    M.cable = mat("CableLeader", lambda nt: psdf(nt, mix_rgb(nt, nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE",
                  nmath(nt, "ADD", nmath(nt, "MULTIPLY", separate(nt, tex_coord(nt, "Object"))["X"], 900.0),
                  nmath(nt, "MULTIPLY", nmath(nt, "ARCTAN2", separate(nt, tex_coord(nt, "Object"))["Z"],
                  separate(nt, tex_coord(nt, "Object"))["Y"]), 7.0)))), 3.0), srgb(110, 112, 116), srgb(214, 216, 220)),
                  rough=0.25, metal=1.0))
    M.sleeve = mat("CrimpSleeve", lambda nt: psdf(nt, srgb(196, 140, 96), rough=0.3, metal=1.0))

    # --- the boat ----------------------------------------------------------------------------------------------
    def gelcoat(nt):
        p = tex_coord(nt, "Object")
        z = attr(nt, "zrel", "Fac")
        # zrel = (z + 1) / 3.4: the waterline (z = 0) sits at 0.294; a navy boot stripe just above it, a navy sheer
        # stripe under the rub rail; a deep navy boot band (the swell washes over it), red antifouling well below
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", map_range(nt, z, 0.2, 0.207), map_range(nt, z, 0.33, 0.323)),
                      srgb(232, 234, 232), srgb(16, 30, 52))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", map_range(nt, z, 0.68, 0.685), map_range(nt, z, 0.72, 0.715)), col,
                      srgb(16, 30, 52))
        col = mix_rgb(nt, map_range(nt, z, 0.2, 0.19), col, srgb(130, 34, 30))   # deep: rarely shows in the swell
        n = noise(nt, p, 3.0, 4.0)
        return psdf(nt, vmath(nt, "SCALE", col, scale=nmath(nt, "MULTIPLY_ADD", n, 0.06, 0.95)), rough=0.18, coat=0.8,
                    coat_rough=0.05)
    M.hull = mat("Gelcoat", gelcoat)
    M.white = mat("BoatWhite", lambda nt: psdf(nt, srgb(228, 230, 228), rough=0.25, coat=0.6, coat_rough=0.08))
    M.nonskid = mat("NonSkid", lambda nt: psdf(nt, srgb(214, 212, 204), rough=0.8,
                                               normal=bump(nt, noise(nt, tex_coord(nt), 120.0, 2.0), 0.4)))
    def sole(nt):
        """Cockpit sole seen from above: non skid panels, smooth gelcoat margins, flush fish box hatches with dark
        seams and a few scuffs (so the top down shot has something to read)."""
        o = separate(nt, tex_coord(nt, "Object"))
        x, y = o["X"], o["Y"]
        seam = None
        for cx, cy, hx, hy in ((0.95, 0.0, 0.55, 0.8), (3.55, -0.75, 0.45, 0.45), (3.55, 0.75, 0.45, 0.45)):
            d = nmath(nt, "MAXIMUM", nmath(nt, "SUBTRACT", nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", x, cx)), hx),
                      nmath(nt, "SUBTRACT", nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", y, cy)), hy))
            e = map_range(nt, nmath(nt, "ABSOLUTE", d), 0.014, 0.005)
            seam = e if seam is None else nmath(nt, "MAXIMUM", seam, e)
        margin = map_range(nt, nmath(nt, "ABSOLUTE", y), 1.75, 1.8)
        col = mix_rgb(nt, margin, srgb(196, 196, 190), srgb(230, 232, 230))
        scuff = map_range(nt, noise(nt, tex_coord(nt, "Object"), 1.6, 6.0, 0.6), 0.55, 0.75)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", scuff, 0.25), col, srgb(120, 118, 110))
        col = mix_rgb(nt, seam, col, srgb(40, 42, 44))
        return psdf(nt, col, rough=0.75, normal=bump(nt, noise(nt, tex_coord(nt), 120.0, 2.0), 0.35))
    M.sole = mat("CockpitSole", sole)
    M.teak = mat("Teak", lambda nt: psdf(nt, color_ramp(nt, noise(nt, vmath(nt, "MULTIPLY", tex_coord(nt), (1.0, 18.0, 1.0)), 6.0),
                                                        [(0.3, srgb(120, 80, 44)), (0.7, srgb(160, 112, 64))]), rough=0.5, coat=0.4))
    M.glass = mat("TintedGlass", lambda nt: psdf(nt, srgb(8, 14, 18), rough=0.03, coat=1.0, coat_rough=0.01, spec=0.9))
    M.alu = mat("TowerAluminium", lambda nt: psdf(nt, srgb(196, 198, 200), rough=0.3, metal=1.0))
    M.cushion = mat("ChairCushion", lambda nt: psdf(nt, srgb(222, 220, 212), rough=0.55, sss=0.05,
                                                    normal=bump(nt, noise(nt, tex_coord(nt), 60.0), 0.15)))
    M.dark = mat("BoatDark", lambda nt: psdf(nt, srgb(30, 32, 36), rough=0.5))
    M.spray = mat("Spray", lambda nt: psdf(nt, srgb(236, 242, 244), rough=0.4, alpha=0.7, emission=srgb(150, 160, 164), emission_strength=0.35))

    # --- the angler --------------------------------------------------------------------------------------------
    M.skin = mat("Skin", lambda nt: psdf(nt, srgb(196, 140, 106), rough=0.55, sss=0.2, sss_scale=0.01))
    M.hair = mat("Hair", lambda nt: psdf(nt, srgb(52, 40, 30), rough=0.8))
    M.cap = mat("Cap", lambda nt: psdf(nt, srgb(30, 40, 54), rough=0.75))
    M.shades = mat("Shades", lambda nt: psdf(nt, srgb(6, 8, 10), rough=0.05, metal=0.4, coat=1.0))
    M.harness = mat("Harness", lambda nt: psdf(nt, srgb(40, 42, 46), rough=0.6))
    M.oilskin = mat("Oilskin", lambda nt: psdf(nt, srgb(214, 120, 30), rough=0.35, coat=0.5))

    M.bubbles = mat("Bubbles", lambda nt: psdf(nt, srgb(226, 240, 236), rough=0.1, alpha=0.55,
                                               emission=srgb(190, 220, 210), emission_strength=0.5))

    # --- the wreck -------------------------------------------------------------------------------------------------
    def rust(nt):
        """Rusted hull steel: dark brown iron with orange rust blooms, pitted; furred with green growth on surfaces
        that face up; darker deep inside the hull (the cavities read black)."""
        p = world_pos(nt)
        n = noise(nt, vmath(nt, "MULTIPLY", p, (0.6, 0.6, 0.6)), 1.0, 6.0, 0.62)
        n2 = noise(nt, p, 6.0, 5.0, 0.6)
        col = color_ramp(nt, n, [(0.0, srgb(34, 22, 16)), (0.45, srgb(70, 40, 24)), (0.62, srgb(120, 64, 30)),
                                 (1.0, srgb(150, 92, 52))])
        up = map_range(nt, separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Normal"])["Z"], 0.2, 0.75)
        growth = nmath(nt, "MULTIPLY", up, map_range(nt, n2, 0.35, 0.6))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", growth, 0.85), col, srgb(70, 88, 52))
        inside = attr(nt, "inside", "Fac")
        col = vmath(nt, "SCALE", col, scale=nmath(nt, "MULTIPLY_ADD", inside, -0.7, 1.0))
        return psdf(nt, uw_base(nt, col, caustic=0.8), rough=0.85, metal=0.15,
                    normal=bump(nt, nmath(nt, "MULTIPLY_ADD", n2, 0.6, n), 0.8, 0.05))
    M.rust = mat("RustedSteel", rust)
    M.barnacle = mat("Barnacle", lambda nt: psdf(nt, uw_base(nt, color_ramp(nt, noise(nt, tex_coord(nt, "Object"), 30.0),
                     [(0.0, srgb(70, 64, 52)), (1.0, srgb(128, 120, 100))]), caustic=0.4), rough=0.7,
                     normal=bump(nt, noise(nt, tex_coord(nt, "Object"), 80.0), 0.5, 0.003)))

    def seafan(nt):
        """A sea fan: a flat lattice of fine branches (alpha), purple to orange."""
        o = separate(nt, tex_coord(nt, "Object"))
        uv = combine(nt, o["X"], o["Z"], 0.0)
        cells = voronoi(nt, uv, 26.0, out="Distance", feature="DISTANCE_TO_EDGE")
        net = map_range(nt, cells, 0.04, 0.0)
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        col = mix_rgb(nt, oi.outputs["Random"], srgb(120, 40, 110), srgb(200, 110, 50))
        return psdf(nt, uw_base(nt, col, caustic=0.4), rough=0.6, sss=0.3, sss_scale=0.01, alpha=net)
    M.seafan = mat("SeaFan", seafan)

    def sand(nt):
        p = world_pos(nt)
        n = noise(nt, p, 2.0, 5.0, 0.6)
        col = color_ramp(nt, n, [(0.0, srgb(90, 92, 72)), (1.0, srgb(140, 138, 108))])
        ripple = nmath(nt, "SINE", nmath(nt, "ADD", nmath(nt, "MULTIPLY", separate(nt, p)["X"], 7.0), nmath(nt, "MULTIPLY", n, 4.0)))
        return psdf(nt, uw_base(nt, col, caustic=1.0), rough=0.9, normal=bump(nt, ripple, 0.4, 0.02))
    M.sand = mat("Sand", sand)

    def cutface(nt):
        """The cut face of the cross section, drawn flat: dark steel plate with a fine diagonal hatch (a diagram)."""
        o = separate(nt, world_pos(nt))
        hatch = nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY",
                      nmath(nt, "ADD", o["X"], o["Z"]), 14.0))), 30.0)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", hatch, 0.35), srgb(120, 66, 34), srgb(220, 140, 80))
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(col, em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0
        return em.outputs[0]
    M.cutface = new_material("CutFace", cutface)

    def studio_back(nt):
        """The rig breakdown's backdrop: a soft green gradient (a studio sweep under water)."""
        z = separate(nt, tex_coord(nt, "Generated"))["Z"]
        col = color_ramp(nt, z, [(0.0, srgb(4, 12, 10)), (0.6, srgb(16, 40, 34)), (1.0, srgb(40, 80, 66))])
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(col, em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0
        return em.outputs[0]
    M.studio = new_material("StudioBackdrop", studio_back)
    M.silt = mat("Silt", lambda nt: psdf(nt, srgb(130, 136, 110), rough=0.9, alpha=0.6,
                                         emission=srgb(90, 110, 90), emission_strength=0.15))
    M.lever = mat("DragLever", lambda nt: psdf(nt, srgb(200, 30, 30), rough=0.25, coat=0.8))
    return M


# --------------------------------------------------------------------------
# Geometry nodes helpers, particles
# --------------------------------------------------------------------------
def gn_time(ng):
    """Scene seconds, zero at frame 1."""
    return nmath(ng, "DIVIDE", nmath(ng, "SUBTRACT", ng.nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], 1.0),
                 float(FPS))


def gn_xyz(ng):
    p = separate(ng, ng.nodes.new("GeometryNodeInputPosition").outputs[0])
    return p["X"], p["Y"], p["Z"]


class Particles:
    """Spray, bubbles and glints: every particle is born once with a position, velocity and life; a geometry
    nodes group moves them with scene time, so nothing needs per-frame keys."""

    def __init__(self):
        self.items = []     # (birth_s, pos, vel, life_s, size, gravity)

    def emit(self, t, pos, vel, life, size, gravity=0.0):
        self.items.append((t, Vector(pos), Vector(vel), life, size, gravity))

    def build(self, name, col, mat, streak=False):
        me = bpy.data.meshes.new(name)
        me.from_pydata([tuple(p) for _t, p, _v, _l, _s, _g in self.items], [], [])
        for an, vals in (("birth", [it[0] for it in self.items]), ("life", [it[3] for it in self.items]),
                         ("size", [it[4] for it in self.items]), ("grav", [it[5] for it in self.items])):
            a = me.attributes.new(an, "FLOAT", "POINT")
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
        # every point stays (dead ones shrink to nothing): a stable point order gives true motion vectors, while
        # deleting dead points reshuffles indices and motion blur smears particles into long false streaks
        ico = nodes.new("GeometryNodeMeshIcoSphere")
        ico.inputs["Radius"].default_value = 1.0
        ico.inputs["Subdivisions"].default_value = 1
        inst = nodes.new("GeometryNodeInstanceOnPoints")
        links.new(sp.outputs[0], inst.inputs["Points"])
        links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
        grow = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", nattr("size"), alive),
                     nmath(ng, "POWER", nmath(ng, "SUBTRACT", 1.0, nmath(ng, "DIVIDE", age, life), clamp=True), 0.4))
        if streak:   # stretched along the motion as a shutter would draw it (points born per frame get no blur)
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
        print(f"{name}: {len(self.items)} particles")
        return ob


def ctrl_swim_group(ctrl, L, name):
    """Body wave driven by a controller empty: x = phase (rad), y = amplitude (m), z = body bend (turning).
    L is the modelled length nose to tail base; the head barely moves, the tail most (a tuna swims stiff bodied)."""
    ng, gi, go = new_group(name, "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = ctrl
    c = separate(ng, oi.outputs["Location"])
    x, _y, _z = gn_xyz(ng)
    back = nmath(ng, "MAXIMUM", nmath(ng, "SUBTRACT", 0.05 * L, x), 0.0)
    w = nmath(ng, "MINIMUM", nmath(ng, "POWER", nmath(ng, "DIVIDE", back, 0.55 * L), 2.2), 1.9)
    arg = nmath(ng, "SUBTRACT", c["X"], nmath(ng, "MULTIPLY", x, 5.6 / L))
    dy = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", nmath(ng, "SINE", arg), w), c["Y"])
    dy = nmath(ng, "MULTIPLY_ADD", nmath(ng, "MULTIPLY", back, back), nmath(ng, "MULTIPLY", c["Z"], 3.0 / L), dy)
    sp = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, 0.0, dy, 0.0), sp.inputs["Offset"])
    ng.links.new(sp.outputs[0], go.inputs["Geometry"])
    return ng


def fish_mesh(name, bm, mats):
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    return me


def heading_quat(fwd, roll=0.0):
    """Rotation that turns local +X to fwd with local +Z kept up (then a roll about the heading)."""
    fwd = Vector(fwd).normalized()
    q = fwd.to_track_quat("X", "Z")
    if roll:
        q = q @ Quaternion((1, 0, 0), roll)
    return q


def fin_base(fn, x, inset=0.01):
    return (x, fn(x) - inset if fn(x) > 0 else fn(x) + inset)


# --------------------------------------------------------------------------
# The goliath grouper (Epinephelus itajara). After the species' descriptions: a huge, robust body, deepest a third
# back; a broad, flat head with small eyes; a very wide mouth with the lower jaw projecting; a low spiny dorsal
# running into a rounded soft dorsal; large rounded pectoral fins; a rounded tail. Built at a nominal 0.6 m length,
# then scaled (GOLIATH_SCALE) to about 2.3 m: illustrative, not the record fish to scale.
# --------------------------------------------------------------------------
G_L = 0.60                            # modelled length nose to tail base before scaling
GOLIATH_SCALE = 3.8                   # 0.6 x 3.8 = 2.28 m to the tail base (the tail adds about 0.5 m)
G_JAW_HINGE = Vector((0.205, 0.0, -0.034))


def pchip(points):
    """Smooth, overshoot-free curve through [(u, value)] (monotone cubic Hermite), for fair body outlines."""
    pts = sorted(points)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    n = len(pts)
    d = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(n - 1)]
    m = [d[0]] + [0.0] * (n - 2) + [d[-1]]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1, w2 = 2 * (xs[i + 1] - xs[i]) + (xs[i] - xs[i - 1]), (xs[i + 1] - xs[i]) + 2 * (xs[i] - xs[i - 1])
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])

    def f(u):
        if u <= xs[0]:
            return ys[0]
        if u >= xs[-1]:
            return ys[-1]
        i = max(0, min(n - 2, next(k for k in range(n - 1) if u <= xs[k + 1])))
        h = xs[i + 1] - xs[i]
        t = (u - xs[i]) / h
        return (ys[i] * (2 * t ** 3 - 3 * t ** 2 + 1) + m[i] * h * (t ** 3 - 2 * t ** 2 + t) +
                ys[i + 1] * (-2 * t ** 3 + 3 * t ** 2) + m[i + 1] * h * (t ** 3 - t ** 2))
    return f


def ring_section(bm, x0, x1, top, bot, wid, nr=20, na=24, power=2.2, zl=None, xl=None, xr=None, mat=0, caps=True):
    """Loft from x0 (front) to x1 (back): top/bot/wid are functions of x; superellipse sections, end caps."""
    zl = zl or bm.verts.layers.float.get("zrel") or bm.verts.layers.float.new("zrel")
    xl = xl or bm.verts.layers.float.get("xrel") or bm.verts.layers.float.new("xrel")
    rings = []
    for i in range(nr + 1):
        x = x0 + (x1 - x0) * i / nr
        t, b, w = top(x), bot(x), wid(x)
        zc, hh = (t + b) / 2, (t - b) / 2
        ring = []
        for k in range(na):
            a = 2 * math.pi * k / na
            c, s = math.cos(a), math.sin(a)
            v = bm.verts.new((x, w * math.copysign(abs(c) ** (2 / power), c), zc + hh * math.copysign(abs(s) ** (2 / power), s)))
            v[zl] = min(1.0, max(0.0, (v.co.z - b) / max(1e-6, t - b)))
            v[xl] = xr(x) if xr else 0.5
            ring.append(v)
        rings.append(ring)
    for ra, rb in zip(rings, rings[1:]):
        for k in range(na):
            f = bm.faces.new((ra[k], ra[(k + 1) % na], rb[(k + 1) % na], rb[k]))
            f.material_index, f.smooth = mat, True
    if caps:
        for ring, front in ((rings[0], True), (rings[-1], False)):
            cz = sum(v.co.z for v in ring) / na
            cx = ring[0].co.x + (0.25 if front else -0.25) * (wid(ring[0].co.x) if front else wid(ring[0].co.x))
            tip = bm.verts.new((cx, 0.0, cz))
            tip[zl], tip[xl] = 0.5, ring[0][xl]
            for k in range(na):
                vs = (ring[(k + 1) % na], ring[k], tip) if front else (ring[k], ring[(k + 1) % na], tip)
                f = bm.faces.new(vs)
                f.material_index, f.smooth = mat, True
    return rings


def goliath_curves():
    L = G_L

    def u_of(x):
        return (L / 2 - x) / L
    top_u = pchip([(0.0, 0.016), (0.04, 0.028), (0.1, 0.046), (0.2, 0.072), (0.32, 0.092), (0.45, 0.098), (0.6, 0.090),
                   (0.75, 0.066), (0.9, 0.047), (1.0, 0.041)])
    bot_u = pchip([(0.0, -0.018), (0.07, -0.028), (0.15, -0.044), (0.22, -0.068), (0.32, -0.092), (0.45, -0.100),
                   (0.6, -0.092), (0.75, -0.066), (0.9, -0.045), (1.0, -0.039)])
    wid_u = pchip([(0.0, 0.044), (0.05, 0.062), (0.12, 0.080), (0.25, 0.090), (0.42, 0.088), (0.6, 0.072), (0.8, 0.040),
                   (1.0, 0.021)])
    return (lambda x: top_u(u_of(x))), (lambda x: bot_u(u_of(x))), (lambda x: wid_u(u_of(x))), u_of


def goliath_swim_group(ctrl, name):
    """Body wave driven by a controller empty: x = phase (rad), y = amplitude (m), z = body bend (turning)."""
    ng, gi, go = new_group(name, "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = ctrl
    c = separate(ng, oi.outputs["Location"])
    x, _y, _z = gn_xyz(ng)
    back = nmath(ng, "MAXIMUM", nmath(ng, "SUBTRACT", 0.08, x), 0.0)
    w = nmath(ng, "MINIMUM", nmath(ng, "POWER", nmath(ng, "DIVIDE", back, 0.38), 1.6), 1.5)
    arg = nmath(ng, "SUBTRACT", c["X"], nmath(ng, "MULTIPLY", x, 10.5))
    dy = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", nmath(ng, "SINE", arg), w), c["Y"])
    dy = nmath(ng, "MULTIPLY_ADD", nmath(ng, "MULTIPLY", back, back), nmath(ng, "MULTIPLY", c["Z"], 6.0), dy)
    sp = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, 0.0, dy, 0.0), sp.inputs["Offset"])
    ng.links.new(sp.outputs[0], go.inputs["Geometry"])
    return ng


def make_goliath(col, M, name="Goliath"):
    """The goliath: body (with eyes), fins, a hinged lower jaw and a dark throat behind it (the mouth opens by
    rotating the jaw about G_JAW_HINGE and pushing the throat back and down: the suction gape)."""
    L = G_L
    top, bot, wid, u_of = goliath_curves()
    bm = bmesh.new()
    ring_section(bm, L / 2, -L / 2, top, bot, wid, nr=70, na=44, power=2.15, xr=lambda x: 1.0 - u_of(x))
    lipl = bm.verts.layers.float.new("lip")
    for v in bm.verts:                                   # the upper lip: a thick fold along the mouth line
        u = u_of(v.co.x)
        if u < 0.2:
            zr = (v.co.z - bot(v.co.x)) / max(1e-6, top(v.co.x) - bot(v.co.x))
            v[lipl] = math.exp(-((zr - 0.12) / 0.07) ** 2) * (1.0 - u / 0.2)
    er = bm.verts.layers.float.new("eyer")
    ex, ez = L / 2 - 0.07, 0.028
    for s in (1, -1):
        c = Vector((ex, s * wid(ex) * 0.8, ez))
        faces = ellipsoid(bm, c, (0.0105, 0.0055, 0.0105), subdiv=3, mat=1)
        axis = Vector((0.15, s * 1.0, 0.2)).normalized()
        for v in (v for f in faces for v in f.verts):
            v[er] = math.acos(max(-1.0, min(1.0, (v.co - c).normalized().dot(axis)))) / (math.pi / 2)
    me = fish_mesh(name, bm, [M.goliath, M.eye])
    body = bpy.data.objects.new(name, me)
    col.objects.link(body)
    fb = bmesh.new()
    xs = [0.15 - 0.017 * i for i in range(10)]           # low spiny dorsal, scalloped between the spine tips
    ctrl = [fin_base(top, xs[0], 0.006)]
    for i, x in enumerate(xs):
        hgt = 0.018 + 0.012 * math.sin(math.pi * min(1.0, (i + 1) / 5))
        ctrl.append((x, top(x) + hgt))
        if i < len(xs) - 1:
            ctrl.append((x - 0.008, top(x) + hgt * 0.75))
    ctrl.append(fin_base(top, xs[-1] - 0.004, 0.006))
    soft_fin(fb, ctrl, 0.003, 0, samples=5)
    x0 = -0.005                                          # rounded soft dorsal
    soft_fin(fb, [fin_base(top, x0, 0.006), (x0 - 0.01, top(x0) + 0.04), (x0 - 0.05, top(x0 - 0.05) + 0.06),
                  (x0 - 0.1, top(x0 - 0.1) + 0.058), (x0 - 0.14, top(x0 - 0.14) + 0.038), (x0 - 0.165, top(x0 - 0.165) + 0.012),
                  fin_base(top, x0 - 0.17, 0.006)], 0.004, 0, samples=8)
    xt = -L / 2                                          # rounded tail
    soft_fin(fb, [(xt + 0.03, 0.034), (xt - 0.03, 0.062), (xt - 0.08, 0.072), (xt - 0.115, 0.05), (xt - 0.128, 0.0),
                  (xt - 0.115, -0.05), (xt - 0.08, -0.072), (xt - 0.03, -0.062), (xt + 0.03, -0.034)], 0.005, 0, samples=8)
    xa = -0.075                                          # rounded anal fin
    soft_fin(fb, [fin_base(bot, xa, 0.006), (xa - 0.012, bot(xa) - 0.04), (xa - 0.05, bot(xa - 0.05) - 0.052),
                  (xa - 0.085, bot(xa - 0.085) - 0.035), fin_base(bot, xa - 0.1, 0.006)], 0.004, 0, samples=8)
    for s in (1, -1):                                    # pelvic fins
        rot = Euler((s * math.radians(16), 0.0, 0.0))
        xp = 0.1
        soft_fin(fb, [(xp + 0.03, bot(xp) + 0.014), (xp, bot(xp) - 0.04), (xp - 0.045, bot(xp) - 0.055),
                      (xp - 0.055, bot(xp) - 0.03), (xp - 0.02, bot(xp) + 0.012)], 0.004, 0, offset=(0, s * 0.03, 0), rot=rot)
    fins = bpy.data.objects.new(name + "Fins", fish_mesh(name + "Fins", fb, [M.goliath_fin]))
    col.objects.link(fins)
    fins.parent = body
    pects = []
    for s in (1, -1):                                    # big rounded pectorals, growing out of the flank
        pb = bmesh.new()
        soft_fin(pb, [(0.03, 0.006), (0.0, 0.03), (-0.04, 0.04), (-0.08, 0.026), (-0.096, 0.0), (-0.085, -0.026),
                      (-0.05, -0.036), (-0.01, -0.03), (0.03, -0.012)], 0.006, 0, samples=8)
        po = bpy.data.objects.new(f"{name}Pectoral{'L' if s > 0 else 'R'}", fish_mesh(f"{name}Pect{s}", pb, [M.goliath_fin]))
        col.objects.link(po)
        po.parent = body
        xq = L / 2 - 0.15
        po.location = (xq, s * (wid(xq) - 0.012), -0.03)
        po.rotation_euler = (s * math.radians(-8), math.radians(6), s * math.radians(-24))
        pects.append(po)
    jb = bmesh.new()                                     # the lower jaw: wide and heavy, projecting a little
    hx = G_JAW_HINGE.x
    jl = L / 2 + 0.01 - hx

    def jfrac(x):
        return min(1.0, max(0.0, x / jl))
    ring_section(jb, jl, -0.012, lambda x: 0.004 + 0.022 * (1 - jfrac(x)) * 0.0 + 0.026 * (x + 0.012) / (jl + 0.012),
                 lambda x: -0.03 + 0.024 * jfrac(x) ** 0.8, lambda x: 0.02 + 0.042 * (1 - jfrac(x)) ** 0.6, nr=16, na=28,
                 power=2.2, xr=lambda x: 0.95)
    for v in jb.verts:
        v[jb.verts.layers.float["zrel"]] *= 0.4
    jaw = bpy.data.objects.new(name + "Jaw", fish_mesh(name + "Jaw", jb, [M.goliath]))
    col.objects.link(jaw)
    jaw.parent = body
    jaw.location = G_JAW_HINGE
    jaw.rotation_mode = "XYZ"
    jaw.data.materials.append(M.throat)
    for p in jaw.data.polygons:
        p.use_smooth = True
        if p.normal.z > 0.45:                            # the inside of the mouth
            p.material_index = 1
    tb = bmesh.new()                                     # the throat: dark, red at the back (gill rakers)
    ellipsoid(tb, (0.0, 0.0, 0.0), (0.05, 0.032, 0.02), subdiv=3, mat=0)
    throat = finish(name + "Throat", tb, col, [M.throat])
    throat.parent = body
    throat.location = (L / 2 - 0.07, 0.0, -0.03)
    root = empty_object(name + "Rig", col, size=0.3)
    root.rotation_mode = "QUATERNION"
    root.scale = (GOLIATH_SCALE,) * 3
    body.parent = root
    beat = empty_object(name + "Beat", col, size=0.05)
    ng = goliath_swim_group(beat, name + "Swim")
    for ob in (body, fins):
        ob.modifiers.new("Swim", "NODES").node_group = ng
    sub = body.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 2
    js = jaw.modifiers.new("Subdivision", "SUBSURF")
    js.levels = js.render_levels = 1
    for ob in (body, fins, jaw, throat, *pects):
        ob.visible_shadow = True
    return dict(root=root, body=body, fins=fins, jaw=jaw, throat=throat, pects=pects, beat=beat, L=G_L * GOLIATH_SCALE)


def key_gape(g, f, gape, flare=0.0):
    """Open the mouth: gape 0 shut .. 1 fully open (the jaw drops about 38 degrees and the throat swells back)."""
    g["jaw"].rotation_euler = (0.0, math.radians(38.0) * gape, 0.0)
    g["throat"].location = (G_L / 2 - 0.075 - 0.02 * gape, 0.0, -0.026 - 0.012 * gape)
    g["throat"].scale = (1.0 + 0.15 * gape, 1.0 + 0.08 * gape + 0.08 * flare, 1.0 + 0.3 * gape)
    for i, po in enumerate(g["pects"]):
        sgn = 1 if i == 0 else -1
        po.rotation_euler = (sgn * math.radians(-8 - 25 * flare), math.radians(6), sgn * math.radians(-24 - 30 * flare))
        key(po, "rotation_euler", f)
    key(g["jaw"], "rotation_euler", f)
    key(g["throat"], "location", f)
    key(g["throat"], "scale", f)


def key_swim(fish, f, phase, amp, bend=0.0):
    b = fish["beat"]
    b.location = (phase, amp, bend)
    key(b, "location", f)


# --------------------------------------------------------------------------
# Small fish: skipjack tuna and a school of baitfish
# --------------------------------------------------------------------------
def small_fish(name, M, mat, fin_m, L, top_u, bot_u, wid_u, fins_spec, nr=30, na=18, eye_r=0.03):
    def u_of(x):
        return (L / 2 - x) / L
    top, bot, wid = (lambda x: top_u(u_of(x))), (lambda x: bot_u(u_of(x))), (lambda x: wid_u(u_of(x)))
    bm = bmesh.new()
    zl, xl = bm.verts.layers.float.new("zrel"), bm.verts.layers.float.new("xrel")
    er = bm.verts.layers.float.new("eyer")
    rings = []
    for i in range(nr + 1):
        x = L / 2 - L * i / nr
        t, b, w = top(x), bot(x), wid(x)
        zc, hh = (t + b) / 2, (t - b) / 2
        ring = []
        for k in range(na):
            a = 2 * math.pi * k / na
            c, s = math.cos(a), math.sin(a)
            v = bm.verts.new((x, w * math.copysign(abs(c) ** (2 / 2.2), c), zc + hh * math.copysign(abs(s) ** (2 / 2.2), s)))
            v[zl] = min(1.0, max(0.0, (v.co.z - b) / max(1e-6, t - b)))
            v[xl] = 1 - i / nr
            ring.append(v)
        rings.append(ring)
    for ra, rb in zip(rings, rings[1:]):
        for k in range(na):
            bm.faces.new((ra[k], ra[(k + 1) % na], rb[(k + 1) % na], rb[k]))
    for ring, x_cap in ((rings[0], L / 2 + 0.01 * L), (rings[-1], -L / 2 - 0.005 * L)):
        tip = bm.verts.new((x_cap, 0.0, sum(v.co.z for v in ring) / na))
        tip[zl] = 0.5
        for k in range(na):
            vs = (ring[(k + 1) % na], ring[k], tip) if x_cap > 0 else (ring[k], ring[(k + 1) % na], tip)
            bm.faces.new(vs)
    for f in bm.faces:
        f.smooth = True
    ex = L / 2 - 0.13 * L
    for s in (1, -1):
        c = Vector((ex, s * wid(ex) * 0.8, top(ex) * 0.4))
        faces = ellipsoid(bm, c, (eye_r * L, eye_r * 0.5 * L, eye_r * L), subdiv=3, mat=1)
        axis = Vector((0.1, s * 1.0, 0.05)).normalized()
        for v in (v for f in faces for v in f.verts):
            v[er] = math.acos(max(-1.0, min(1.0, (v.co - c).normalized().dot(axis)))) / (math.pi / 2)
    for ctrl in fins_spec(L, top, bot):
        soft_fin(bm, ctrl, 0.01 * L, 2, samples=5)
    return fish_mesh(name, bm, [mat, M.small_eye, fin_m])


def jack_mesh(M, L=0.5):
    """Jack crevalle: a deep, compressed body with a steep, blunt head, a deeply forked tail."""
    top = curve([(0.0, 0.0), (0.06, 0.09), (0.2, 0.17), (0.35, 0.19), (0.55, 0.15), (0.8, 0.05), (0.92, 0.022), (1.0, 0.02)])
    bot = curve([(0.0, -0.03), (0.08, -0.09), (0.3, -0.16), (0.5, -0.14), (0.75, -0.06), (0.9, -0.02), (1.0, -0.02)])
    wid = curve([(0.0, 0.0), (0.1, 0.05), (0.3, 0.07), (0.6, 0.05), (0.9, 0.016), (1.0, 0.012)])

    def fins(L, top, bot):
        t = -L / 2
        return [[(0.02 * L, top(0.02 * L) - 0.004 * L), (0.0, top(0.0) + 0.12 * L), (-0.08 * L, top(-0.08 * L) + 0.05 * L),
                 (-0.3 * L, top(-0.3 * L) + 0.03 * L), (-0.34 * L, top(-0.34 * L) - 0.004 * L)],
                [(-0.02 * L, bot(-0.02 * L) + 0.004 * L), (-0.04 * L, bot(-0.04 * L) - 0.1 * L), (-0.12 * L, bot(-0.12 * L) - 0.04 * L),
                 (-0.3 * L, bot(-0.3 * L) - 0.025 * L), (-0.33 * L, bot(-0.33 * L) + 0.004 * L)],
                [(t + 0.03 * L, 0.02 * L), (t - 0.1 * L, 0.17 * L), (t - 0.14 * L, 0.17 * L), (t - 0.06 * L, 0.0),
                 (t - 0.14 * L, -0.17 * L), (t - 0.1 * L, -0.17 * L), (t + 0.03 * L, -0.02 * L)]]
    return small_fish("JackCrevalle", M, M.jack, M.jack_fin, L, lambda u: top(u) * L, lambda u: bot(u) * L,
                      lambda u: wid(u) * L, fins, nr=28, na=18, eye_r=0.04)


def baitfish_mesh(M, L=0.22):
    top = curve([(0.0, 0.0), (0.1, 0.06), (0.35, 0.11), (0.6, 0.085), (0.9, 0.025), (1.0, 0.02)])
    bot = curve([(0.0, -0.02), (0.1, -0.06), (0.35, -0.1), (0.6, -0.075), (0.9, -0.02), (1.0, -0.02)])
    wid = curve([(0.0, 0.0), (0.12, 0.04), (0.4, 0.055), (0.9, 0.012), (1.0, 0.01)])

    def fins(L, top, bot):
        t = -L / 2
        return [[(t + 0.03 * L, 0.02 * L), (t - 0.09 * L, 0.12 * L), (t - 0.12 * L, 0.1 * L), (t - 0.07 * L, 0.0),
                 (t - 0.12 * L, -0.1 * L), (t - 0.09 * L, -0.12 * L), (t + 0.03 * L, -0.02 * L)]]
    return small_fish("Baitfish", M, M.baitfish, M.small_fin, L, lambda u: top(u) * L, lambda u: bot(u) * L,
                      lambda u: wid(u) * L, fins, nr=16, na=10, eye_r=0.04)


def school_group(fish_ob, name):
    """A bait school milling in a slow ball: each point carries its own orbit (radius, height, start angle,
    speed); geometry nodes move the points with time and instance the fish on them, tail-beat wiggle included."""
    ng, gi, go = new_group(name, "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links

    def nattr(nm):
        r = nodes.new("GeometryNodeInputNamedAttribute")
        r.data_type = "FLOAT"
        r.inputs["Name"].default_value = nm
        return r.outputs[0]
    t = gn_time(ng)
    ang = nmath(ng, "MULTIPLY_ADD", t, nattr("speed"), nattr("a0"))
    rad = nattr("rad")
    pos = combine(ng, nmath(ng, "MULTIPLY", nmath(ng, "COSINE", ang), rad), nmath(ng, "MULTIPLY", nmath(ng, "SINE", ang), rad),
                  nmath(ng, "ADD", nattr("hgt"), nmath(ng, "MULTIPLY", nmath(ng, "SINE", nmath(ng, "MULTIPLY_ADD", t, 0.7, nattr("a0"))), 0.15)))
    sp = nodes.new("GeometryNodeSetPosition")
    links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    links.new(pos, sp.inputs["Position"])
    inst = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(sp.outputs[0], inst.inputs["Points"])
    oi = nodes.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = fish_ob
    oi.inputs["As Instance"].default_value = True
    links.new(oi.outputs["Geometry"], inst.inputs["Instance"])
    yaw = nmath(ng, "ADD", nmath(ng, "ADD", ang, math.pi / 2),
                nmath(ng, "MULTIPLY", nmath(ng, "SINE", nmath(ng, "MULTIPLY_ADD", t, 14.0, nmath(ng, "MULTIPLY", nattr("a0"), 7.0))), 0.18))
    links.new(combine(ng, 0.0, 0.0, yaw), inst.inputs["Rotation"])
    links.new(combine(ng, nattr("scl"), nattr("scl"), nattr("scl")), inst.inputs["Scale"])
    links.new(inst.outputs[0], go.inputs["Geometry"])
    return ng


def build_school(col, fish_ob, center, n, r_lo, r_hi, h_span, seed, name="BaitSchool"):
    rng = random.Random(seed)
    me = bpy.data.meshes.new(name)
    me.from_pydata([(0.0, 0.0, 0.0)] * n, [], [])
    vals = {"a0": [], "rad": [], "hgt": [], "speed": [], "scl": []}
    for _ in range(n):
        vals["a0"].append(rng.uniform(0, 2 * math.pi))
        vals["rad"].append(rng.uniform(r_lo, r_hi))
        vals["hgt"].append(rng.uniform(-h_span, h_span))
        vals["speed"].append(rng.uniform(0.35, 0.5))
        vals["scl"].append(rng.uniform(0.85, 1.15))
    for k, v in vals.items():
        a = me.attributes.new(k, "FLOAT", "POINT")
        a.data.foreach_set("value", v)
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    ob.location = center
    ob.modifiers.new(name, "NODES").node_group = school_group(fish_ob, name + "GN")
    return ob


# --------------------------------------------------------------------------
# Lines
# --------------------------------------------------------------------------
def line_object(name, col, M, start, end, radius=0.0007, sag=None, count=24):
    """A line between two empties; sag (an empty) bends it as a quadratic curve through its location."""
    ng, gi, go = new_group(name + "GN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links

    def loc(ob):
        n = nodes.new("GeometryNodeObjectInfo")
        n.transform_space = "RELATIVE"
        n.inputs["Object"].default_value = ob
        return n.outputs["Location"]
    if sag is None:
        c = nodes.new("GeometryNodeCurvePrimitiveLine")
        links.new(loc(start), c.inputs["Start"])
        links.new(loc(end), c.inputs["End"])
    else:
        c = nodes.new("GeometryNodeCurveQuadraticBezier")
        c.inputs["Resolution"].default_value = count
        links.new(loc(start), c.inputs["Start"])
        links.new(loc(sag), c.inputs["Middle"])
        links.new(loc(end), c.inputs["End"])
    res = nodes.new("GeometryNodeResampleCurve")
    res.inputs["Count"].default_value = count
    links.new(c.outputs["Curve"], res.inputs["Curve"])
    prof = nodes.new("GeometryNodeCurvePrimitiveCircle")
    prof.inputs["Resolution"].default_value = 6
    prof.inputs["Radius"].default_value = radius
    c2m = nodes.new("GeometryNodeCurveToMesh")
    links.new(res.outputs[0], c2m.inputs["Curve"])
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


# --------------------------------------------------------------------------
# The boat: a sportfisher modelled 50 ft (15.2 m) with the transom at x = 0, the bow at x = +15.2, waterline z = 0,
# then scaled to about 43 ft (BOAT_SCALE); cockpit aft with the fighting chair, salon, flybridge, tuna tower and
# outriggers swung out for the spread.
# --------------------------------------------------------------------------
BOAT_LEN = 15.2
COCKPIT_SOLE = 0.72
BOAT_SCALE = 0.86                    # 15.2 m x 0.86 = 13.1 m, about 43 ft (illustrative)
CHAIR_AT = Vector((2.2, 0.0, COCKPIT_SOLE))          # base of the chair pedestal (boat frame)


def hull_section(u):
    """Half section of the hull at u (0 transom .. 1 bow): points from the keel up to the sheer."""
    x = u * BOAT_LEN
    sheer = 1.42 + 0.12 * u + 0.9 * max(0.0, u - 0.55) ** 1.6 / 0.45 ** 1.6
    bow = max(0.0, (u - 0.62) / 0.38)
    beam = 2.42 * (1.0 - 0.05 * (1 - u)) * (1.0 - bow ** 1.7)
    keel = -0.95 + 0.25 * (1 - u) + (1.45 * bow ** 1.5)
    chine_z = keel + 0.55 + 0.35 * bow
    chine_y = beam * 0.9
    return x, [(0.0, keel), (chine_y * 0.55, keel + 0.18), (chine_y, chine_z), (chine_y * 1.02, chine_z + 0.06),
               (beam, sheer - 0.3), (beam * 1.03, sheer)]


def build_boat(col, M):
    """Hull shell (open on top at the cockpit), transom, cockpit sole and coamings, covering boards, foredeck,
    salon with dark glass, flybridge with hardtop, tuna tower, outriggers, and the fighting chair."""
    root = empty_object("Boat", col, size=1.0)
    root.rotation_mode = "QUATERNION"
    bm = bmesh.new()
    zl = bm.verts.layers.float.new("zrel")
    rings = []
    n = 46
    for i in range(n + 1):
        u = (i / n) ** 0.9
        x, half = hull_section(min(u, 0.999))
        pts = [(x, -y, z) for y, z in half[::-1]] + [(x, y, z) for y, z in half[1:]]
        ring = []
        for p in pts:
            v = bm.verts.new(p)
            v[zl] = (p[2] + 1.0) / 3.4
            ring.append(v)
        rings.append(ring)
    stem = bm.verts.new((BOAT_LEN + 0.15, 0.0, 2.45))
    stem[zl] = 1.0
    for ra, rb in zip(rings, rings[1:]):
        for k in range(len(ra) - 1):
            bm.faces.new((ra[k], ra[k + 1], rb[k + 1], rb[k]))
    last = rings[-1]
    for k in range(len(last) - 1):
        bm.faces.new((last[k], last[k + 1], stem))
    bm.faces.new(rings[0])                                    # the transom
    for f in bm.faces:
        f.smooth = True
    hull = finish("BoatHull", bm, col, [M.hull], parent=root)
    # deck surfaces: cockpit sole, foredeck cap, covering boards
    bm = bmesh.new()
    xs = [i * 0.25 for i in range(0, 19)]                   # cockpit sole, 0 .. 4.5 m
    for xa, xb in zip(xs, xs[1:]):
        ya = hull_section(xa / BOAT_LEN)[1][-1][0] - 0.25
        yb = hull_section(xb / BOAT_LEN)[1][-1][0] - 0.25
        bm.faces.new([bm.verts.new((xa, -ya, COCKPIT_SOLE)), bm.verts.new((xb, -yb, COCKPIT_SOLE)),
                      bm.verts.new((xb, yb, COCKPIT_SOLE)), bm.verts.new((xa, ya, COCKPIT_SOLE))])
    sole = finish("CockpitSole", bm, col, [M.sole], parent=root)
    bm = bmesh.new()
    for s in (1, -1):                                         # covering boards along the cockpit, teak capped
        pts = []
        for i in range(19):
            x = i * 0.25
            y = hull_section(x / BOAT_LEN)[1][-1][0]
            z = hull_section(x / BOAT_LEN)[1][-1][1]
            pts.append((x, y, z))
        for (xa, ya, za), (xb, yb, zb) in zip(pts, pts[1:]):
            vs = [bm.verts.new((xa, s * (ya - 0.32), za + 0.02)), bm.verts.new((xb, s * (yb - 0.32), zb + 0.02)),
                  bm.verts.new((xb, s * (yb + 0.03), zb + 0.02)), bm.verts.new((xa, s * (ya + 0.03), za + 0.02))]
            bm.faces.new(vs if s > 0 else vs[::-1])
    yt = hull_section(0.0)[1][-1][0]
    zt = hull_section(0.0)[1][-1][1]
    bm_box(bm, (0.1, 0.0, zt + 0.01), (0.22, 2 * yt + 0.05, 0.06))     # transom cap
    caps = finish("CoveringBoards", bm, col, [M.teak], parent=root)
    bm = bmesh.new()                                          # cockpit sides (coamings), the inside of the transom
    for s in (1, -1):
        for i in range(18):
            xa, xb = i * 0.25, (i + 1) * 0.25
            ya, za = hull_section(xa / BOAT_LEN)[1][-1]
            yb, zb = hull_section(xb / BOAT_LEN)[1][-1]
            bm.faces.new([bm.verts.new((xa, s * (ya - 0.32), za + 0.02)), bm.verts.new((xb, s * (yb - 0.32), zb + 0.02)),
                          bm.verts.new((xb, s * (yb - 0.32), COCKPIT_SOLE)), bm.verts.new((xa, s * (ya - 0.32), COCKPIT_SOLE))])
        for x in (0.9, 1.9, 3.1):                             # flush rod holders in the covering boards
            y = hull_section(x / BOAT_LEN)[1][-1][0] - 0.15
            z = hull_section(x / BOAT_LEN)[1][-1][1]
            bm_frustum(bm, (x, s * y, z + 0.035), (x - 0.04, s * y, z - 0.2), 0.03, 0.028, segs=10)
    bm.faces.new([bm.verts.new((0.21, -yt + 0.3, zt)), bm.verts.new((0.21, yt - 0.3, zt)),
                  bm.verts.new((0.21, yt - 0.3, COCKPIT_SOLE)), bm.verts.new((0.21, -yt + 0.3, COCKPIT_SOLE))])
    bm_box(bm, (0.24, 0.9, (zt + COCKPIT_SOLE) / 2), (0.04, 0.75, zt - COCKPIT_SOLE - 0.1))   # transom door
    bm_box(bm, (4.25, 0.0, COCKPIT_SOLE + 0.22), (0.5, 3.6, 0.44))       # mezzanine step up to the salon
    coam = finish("Coamings", bm, col, [M.white], parent=root)
    bm = bmesh.new()                                          # foredeck from the salon front to the bow
    xs = [4.5 + i * 0.35 for i in range(31)]
    for xa, xb in zip(xs, xs[1:]):
        if xb > BOAT_LEN:
            break
        ha, hb = hull_section(xa / BOAT_LEN)[1][-1], hull_section(min(xb, BOAT_LEN - 0.01) / BOAT_LEN)[1][-1]
        bm.faces.new([bm.verts.new((xa, -ha[0], ha[1])), bm.verts.new((xb, -hb[0], hb[1])),
                      bm.verts.new((xb, hb[0], hb[1])), bm.verts.new((xa, ha[0], ha[1]))])
    deck = finish("Foredeck", bm, col, [M.white], parent=root)
    # salon: a long cabin with raked front glass and dark window bands
    bm = bmesh.new()
    rings = []
    for x, w, h, zb in ((4.5, 2.0, 1.25, 1.5), (5.0, 2.05, 1.32, 1.52), (8.6, 2.0, 1.25, 1.56), (9.6, 1.7, 0.95, 1.62),
                        (10.2, 1.2, 0.55, 1.7)):
        ring = [bm.verts.new((x, -w, zb)), bm.verts.new((x, -w, zb + h * 0.92)), bm.verts.new((x, -w * 0.94, zb + h)),
                bm.verts.new((x, w * 0.94, zb + h)), bm.verts.new((x, w, zb + h * 0.92)), bm.verts.new((x, w, zb))]
        rings.append(ring)
    for ra, rb in zip(rings, rings[1:]):
        for k in range(len(ra) - 1):
            bm.faces.new((ra[k], ra[k + 1], rb[k + 1], rb[k]))
    bm.faces.new(rings[0][::-1])
    bm.faces.new(rings[-1])
    salon = finish("Salon", bm, col, [M.white], parent=root)
    bm = bmesh.new()                                          # window bands and the aft salon door
    for s in (1, -1):
        bm_box(bm, (6.9, s * 2.06, 2.35), (3.4, 0.02, 0.42))
    bm_box(bm, (9.75, 0.0, 2.25), (0.3, 3.0, 0.45), rot=(0.0, math.radians(-38), 0.0))
    bm_box(bm, (4.48, 0.0, 2.05), (0.02, 2.6, 0.95))
    glass = finish("SalonGlass", bm, col, [M.glass], parent=root)
    # flybridge: deck over the salon reaching aft over the cockpit, helm, hardtop on struts
    bm = bmesh.new()
    bm_box(bm, (6.1, 0.0, 2.87), (6.4, 4.0, 0.14))
    for s in (1, -1):
        bm_box(bm, (6.1, s * 2.0, 3.3), (6.4, 0.06, 0.75))      # flybridge coaming
    bm_box(bm, (3.0, 0.0, 3.3), (0.06, 4.0, 0.75))
    bm_box(bm, (8.0, 0.0, 3.4), (1.0, 2.6, 0.8))             # helm console
    bm_box(bm, (6.1, 0.0, 5.05), (5.2, 4.1, 0.12))           # hardtop
    fly = finish("Flybridge", bm, col, [M.white], parent=root)
    bm = bmesh.new()
    struts = []
    for x in (3.6, 8.5):
        for s in (1, -1):
            struts.append(((x, s * 1.9, 2.95), (x, s * 1.85, 5.0)))
    tw = [((3.6, s * 1.9, 5.1), (5.0, s * 0.85, 8.3)) for s in (1, -1)] + \
         [((8.5, s * 1.9, 5.1), (7.0, s * 0.85, 8.3)) for s in (1, -1)]
    for a, b in struts + tw:
        bm_frustum(bm, a, b, 0.035, 0.035, segs=8)
    for s in (1, -1):                                          # tower rails and platform
        bm_frustum(bm, (5.0, s * 0.85, 8.3), (7.0, s * 0.85, 8.3), 0.03, 0.03, segs=8)
        bm_frustum(bm, (5.0, s * 0.85, 9.1), (7.0, s * 0.85, 9.1), 0.025, 0.025, segs=8)
        for x in (5.0, 7.0):
            bm_frustum(bm, (x, s * 0.85, 8.3), (x, s * 0.85, 9.1), 0.025, 0.025, segs=8)
    for x in (5.0, 7.0):
        bm_frustum(bm, (x, -0.85, 9.1), (x, 0.85, 9.1), 0.025, 0.025, segs=8)
    for s in (1, -1):                                          # outriggers, swung out
        base = Vector((6.4, s * 1.95, 3.0))
        tip = base + Vector((-3.0, s * 6.5, 8.0))
        bm_frustum(bm, base, tip, 0.045, 0.012, segs=8)
    tower = finish("TunaTower", bm, col, [M.alu], parent=root)
    bm = bmesh.new()
    bm_box(bm, (6.0, 0.0, 9.2), (2.2, 1.9, 0.06))
    finish("TowerTop", bm, col, [M.white], parent=root)
    chair = build_chair(col, M, root)
    return dict(root=root, hull=hull, parts=[hull, sole, caps, coam, deck, salon, glass, fly, tower], chair=chair)


def build_chair(col, M, root):
    """Fighting chair facing aft: pedestal, seat, tall back, arm rests, an adjustable footrest and the gimbal
    that takes the rod butt. A pivot empty lets it swivel toward the line."""
    pivot = empty_object("ChairPivot", col, size=0.3, parent=root)
    pivot.location = CHAIR_AT
    pivot.rotation_mode = "XYZ"
    bm = bmesh.new()
    bm_frustum(bm, (0, 0, 0.0), (0, 0, 0.06), 0.32, 0.3, segs=20)              # deck plate
    bm_frustum(bm, (0, 0, 0.06), (0, 0, 0.42), 0.11, 0.1, segs=18)             # pedestal
    bm_frustum(bm, (0, 0, 0.42), (0, 0, 0.5), 0.2, 0.22, segs=18)
    metal = mesh_object("ChairPedestal", bm, col, [M.chrome], parent=pivot)
    bm = bmesh.new()
    bm_box(bm, (0.0, 0.0, 0.58), (0.62, 0.62, 0.14))                           # seat
    bm_box(bm, (0.33, 0.0, 1.05), (0.14, 0.6, 0.9), rot=(0.0, math.radians(-12), 0.0))   # back (aft facing seat: back forward)
    for s in (1, -1):
        bm_box(bm, (0.02, s * 0.38, 0.78), (0.56, 0.1, 0.08))                # arm rests
    cush = mesh_object("ChairCushions", bm, col, [M.cushion], parent=pivot)
    bm = bmesh.new()
    bm_frustum(bm, (-0.24, 0.0, 0.52), (-0.78, 0.0, 0.3), 0.025, 0.025, segs=8)   # footrest arm
    bm_frustum(bm, (-0.78, -0.3, 0.3), (-0.78, 0.3, 0.3), 0.022, 0.022, segs=8)   # foot bar
    bm_frustum(bm, (-0.3, 0.0, 0.58), (-0.36, 0.0, 0.66), 0.045, 0.04, segs=12)   # gimbal cup
    gim = mesh_object("ChairGimbal", bm, col, [M.chrome], parent=pivot)
    for o in (metal, cush, gim):
        for p in o.data.polygons:
            p.use_smooth = o is not cush
    return dict(pivot=pivot, parts=[metal, cush, gim], gimbal=Vector((-0.33, 0.0, 0.64)), footbar=Vector((-0.78, 0.0, 0.3)),
                seat=Vector((0.02, 0.0, 0.66)))


def boat_attitude(pos, t, heading=0.0, squat=0.0):
    """Heave, pitch and roll from the swell under the hull (a 15 m hull averages the short waves)."""
    c, s = math.cos(heading), math.sin(heading)
    fwd, side = Vector((c, s, 0.0)), Vector((-s, c, 0.0))
    mid = pos + fwd * (BOAT_LEN * 0.45)
    hb = sea_height(*(mid + fwd * 5.5).xy, t)
    hs = sea_height(*(mid - fwd * 5.5).xy, t)
    hp = sea_height(*(mid + side * 2.0).xy, t)
    hst = sea_height(*(mid - side * 2.0).xy, t)
    hm = sea_height(*mid.xy, t)
    z = 0.25 * (hb + hs + hp + hst) * 0.6 + 0.4 * hm - 0.05 - squat * 0.12     # a deeper draft floods the open cockpit in a trough
    pitch = math.atan2(hb - hs, 11.0) * 0.75 - squat * math.radians(2.5)
    roll = math.atan2(hp - hst, 4.0) * 0.55
    q = Euler((roll, -pitch, heading), "XYZ").to_quaternion()
    return Vector((pos.x, pos.y, z)), q


def key_boat(boat, f, pos, q):
    boat["root"].location = pos
    boat["root"].rotation_quaternion = q
    key(boat["root"], "location", f)
    key(boat["root"], "rotation_quaternion", f)


# --------------------------------------------------------------------------
# Tackle: 130 lb class roller rod and a 14/0 lever drag reel
# --------------------------------------------------------------------------
ROD_LEN = 2.1                       # a 6 ft 10 in stand-up length chair rod (illustrative)


def build_reel(col, M, parent=None, name="Reel"):
    """A big two-speed lever drag reel in gold: spool between side plates, pillars, crank handle, and a drag lever
    that swings over a quadrant on the left side plate (free spool at the back, a red strike stop, full forward).
    Local frame: the spool axis along X, the rod along Y (the reel sits on top of the rod, +Z).
    Returns dict(root, spool, lever): rotate spool and lever about their local X."""
    root = empty_object(name, col, size=0.1, parent=parent)
    bm = bmesh.new()
    for x in (-0.068, 0.068):                                   # side plates
        bm_frustum(bm, (x - 0.012, 0, 0.085), (x + 0.012, 0, 0.085), 0.078, 0.078, segs=32)
    for a_ in range(4):                                         # pillars
        ang = math.pi / 4 + a_ * math.pi / 2
        p = (0, 0.07 * math.cos(ang), 0.085 + 0.07 * math.sin(ang))
        bm_frustum(bm, (-0.06, p[1], p[2]), (0.06, p[1], p[2]), 0.007, 0.007, segs=8)
    bm_box(bm, (0.0, 0.0, 0.012), (0.05, 0.16, 0.022))           # reel foot / clamp
    bm_frustum(bm, (-0.082, 0, 0.085), (-0.092, 0, 0.085), 0.03, 0.026, segs=20)   # the drag housing boss
    gold = mesh_object(name + "Frame", bm, col, [M.reel_gold], parent=root)
    bm = bmesh.new()                                            # the quadrant: an arc plate with stops
    for i in range(13):
        a0 = math.radians(-70 + 11 * i)
        a1 = math.radians(-70 + 11 * (i + 1))
        r0, r1 = 0.05, 0.072
        vs = [bm.verts.new((-0.0815, r * math.sin(aa), 0.085 + r * math.cos(aa))) for r, aa in ((r0, a0), (r1, a0), (r1, a1), (r0, a1))]
        bm.faces.new(vs)
    quad = mesh_object(name + "Quadrant", bm, col, [M.reel_dark], parent=root)
    bm = bmesh.new()                                            # the red strike stop on the quadrant
    bm_box(bm, (-0.083, 0.072 * math.sin(math.radians(10)), 0.085 + 0.072 * math.cos(math.radians(10))), (0.008, 0.008, 0.012))
    stop = mesh_object(name + "StrikeStop", bm, col, [M.lever], parent=root)
    spool_m = new_material(name + "SpoolLine2", lambda nt: psdf(nt, mix_rgb(nt, nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE",
                           nmath(nt, "MULTIPLY", nmath(nt, "ARCTAN2", separate(nt, tex_coord(nt, "Object"))["Z"],
                                                      separate(nt, tex_coord(nt, "Object"))["Y"]), 6.0))), 8.0),
                           srgb(206, 214, 130), srgb(120, 130, 60)), rough=0.35, emission=srgb(214, 220, 140),
                           emission_strength=0.12), M.atmos)
    bm = bmesh.new()
    bm_frustum(bm, (-0.056, 0, 0.0), (0.056, 0, 0.0), 0.062, 0.062, segs=32)   # line on the spool (around its own origin)
    spool = mesh_object(name + "Spool", bm, col, [spool_m], parent=root)
    spool.location = (0.0, 0.0, 0.085)
    spool.rotation_mode = "XYZ"
    bm = bmesh.new()
    bm_frustum(bm, (0.08, 0, 0.1), (0.13, 0, 0.1), 0.012, 0.012, segs=10)          # crank shaft
    bm_box(bm, (0.135, 0.0, 0.05), (0.014, 0.025, 0.13))                          # crank arm
    bm_frustum(bm, (0.135, 0, -0.012), (0.2, 0, -0.012), 0.016, 0.014, segs=12)   # handle knob
    dark = mesh_object(name + "Handles", bm, col, [M.reel_dark], parent=root)
    bm = bmesh.new()                                            # the lever, about its pivot at the boss
    bm_box(bm, (-0.094, 0.0, 0.055), (0.012, 0.02, 0.1))
    lever = mesh_object(name + "Lever", bm, col, [M.reel_dark], parent=root)
    bm = bmesh.new()
    bm_frustum(bm, (-0.094, 0.0, 0.1), (-0.094, 0.0, 0.125), 0.014, 0.012, segs=12)
    knob = mesh_object(name + "LeverKnob", bm, col, [M.lever], parent=lever)
    lever.location = (0.0, 0.0, 0.085)
    lever.rotation_mode = "XYZ"
    for o in (gold, spool, dark, lever, knob, quad, stop):
        for p in o.data.polygons:
            p.use_smooth = o not in (quad, stop)
    return dict(root=root, spool=spool, lever=lever)


LEVER_FREE, LEVER_STRIKE, LEVER_FULL = math.radians(-62), math.radians(10), math.radians(52)   # about X (+ forward)


def build_spread_tackle(col, M, root):
    """Four rods in the covering board holders (boat frame), straight and raked aft: the two corner rods fish the
    corner lures from their own tips; the rigger rods' lines run up to the outrigger tips. Returns the line
    departure points (boat frame) keyed by position name."""
    bm = bmesh.new()
    tips = {}
    for s, side in ((1, "port"), (-1, "starboard")):
        for x, kind, rake in ((0.9, "corner", (-0.85, s * 0.42, 0.62)), (3.1, "rigger", (-0.35, s * 0.25, 0.9))):
            y = s * 2.05
            butt = Vector((x, s * (hull_section(x / BOAT_LEN)[1][-1][0] - 0.15), 1.36))   # in the flush holders
            d = Vector(rake).normalized()
            tip = butt + d * 2.3
            bm_frustum(bm, tuple(butt), tuple(butt + d * 0.5), 0.022, 0.02, segs=10)
            bm_frustum(bm, tuple(butt + d * 0.5), tuple(tip), 0.016, 0.006, segs=10)
            tips[f"{kind}_{side}"] = tip
            bm_frustum(bm, tuple(butt + d * 0.6 + Vector((0, 0, 0.05))), tuple(butt + d * 0.6 + Vector((0, 0, 0.12))), 0.05, 0.05, segs=14)
    rods = finish("SpreadRods", bm, col, [M.blank], parent=root)
    tips["rigger_port"] = Vector((3.4, 8.45, 11.0))           # the outrigger tips (see build_boat)
    tips["rigger_starboard"] = Vector((3.4, -8.45, 11.0))
    return rods, tips


def build_rod(col, M):
    """Roller rod: a heavy blank as a quadratic Bezier (butt, middle, tip empties keyed per frame) with roller guides
    riding the curve, a long butt with the gimbal nock, foregrip, and the 14/0 reel on its seat."""
    ctrl = {k: empty_object(f"Rod{k}", col, size=0.05) for k in ("Butt", "Mid", "Tip")}
    ng, gi, go = new_group("RodGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links

    def loc(ob):
        nn = nodes.new("GeometryNodeObjectInfo")
        nn.transform_space = "RELATIVE"
        nn.inputs["Object"].default_value = ob
        return nn.outputs["Location"]
    c = nodes.new("GeometryNodeCurveQuadraticBezier")
    c.inputs["Resolution"].default_value = 64
    links.new(loc(ctrl["Butt"]), c.inputs["Start"])
    links.new(loc(ctrl["Mid"]), c.inputs["Middle"])
    links.new(loc(ctrl["Tip"]), c.inputs["End"])
    rad = nodes.new("GeometryNodeSetCurveRadius")
    links.new(c.outputs["Curve"], rad.inputs["Curve"])
    par = nodes.new("GeometryNodeSplineParameter")
    links.new(map_range(ng, par.outputs["Factor"], 0.0, 1.0, 1.0, 0.3, smooth=False), rad.inputs["Radius"])
    prof = nodes.new("GeometryNodeCurvePrimitiveCircle")
    prof.inputs["Resolution"].default_value = 12
    prof.inputs["Radius"].default_value = 0.016
    c2m = nodes.new("GeometryNodeCurveToMesh")
    links.new(rad.outputs[0], c2m.inputs["Curve"])
    links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
    setm = nodes.new("GeometryNodeSetMaterial")
    setm.inputs["Material"].default_value = M.blank
    links.new(c2m.outputs[0], setm.inputs["Geometry"])
    # roller guides: small rings with a roller, instanced on points along the blank, aligned to its tangent
    res = nodes.new("GeometryNodeResampleCurve")
    res.inputs["Count"].default_value = 7
    links.new(c.outputs["Curve"], res.inputs["Curve"])
    trim = nodes.new("GeometryNodeTrimCurve")
    trim.mode = "FACTOR"
    trim.inputs[2].default_value = 0.42
    trim.inputs[3].default_value = 1.0
    links.new(c.outputs["Curve"], trim.inputs["Curve"])
    res2 = nodes.new("GeometryNodeResampleCurve")
    res2.inputs["Count"].default_value = 6
    links.new(trim.outputs[0], res2.inputs["Curve"])
    c2p = nodes.new("GeometryNodeCurveToPoints")
    c2p.mode = "EVALUATED"
    links.new(res2.outputs[0], c2p.inputs["Curve"])
    gbm = bmesh.new()
    torus(gbm, (0, 0, 0.03), (0, 0, 1), 0.013, 0.003, n=14, m=6)
    bm_frustum(gbm, (-0.008, 0, 0.04), (0.008, 0, 0.04), 0.008, 0.008, segs=10)
    bm_frustum(gbm, (0, -0.012, 0.0), (0, -0.012, 0.03), 0.0025, 0.0025, segs=6)
    bm_frustum(gbm, (0, 0.012, 0.0), (0, 0.012, 0.03), 0.0025, 0.0025, segs=6)
    gme = bpy.data.meshes.new("RollerGuide")
    gbm.to_mesh(gme)
    gbm.free()
    gme.materials.append(M.chrome)
    gob = bpy.data.objects.new("RollerGuide", gme)
    col.objects.link(gob)
    gob.hide_render = True
    gob.hide_viewport = True
    oi = nodes.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = gob
    oi.inputs["As Instance"].default_value = True
    inst = nodes.new("GeometryNodeInstanceOnPoints")
    links.new(c2p.outputs["Points"], inst.inputs["Points"])
    links.new(oi.outputs["Geometry"], inst.inputs["Instance"])
    links.new(c2p.outputs["Rotation"], inst.inputs["Rotation"])
    join = nodes.new("GeometryNodeJoinGeometry")
    links.new(setm.outputs[0], join.inputs[0])
    links.new(inst.outputs[0], join.inputs[0])
    links.new(join.outputs[0], go.inputs["Geometry"])
    ob = bpy.data.objects.new("Rod", bpy.data.meshes.new("Rod"))
    col.objects.link(ob)
    ob.modifiers.new("Rod", "NODES").node_group = ng
    _ = res
    mount = empty_object("ReelMount", col, size=0.08)
    mount.rotation_mode = "QUATERNION"
    bm = bmesh.new()
    bm_frustum(bm, (0, -0.42, 0.0), (0, -0.05, 0.0), 0.024, 0.024, segs=16)      # aluminium butt
    bm_frustum(bm, (0, -0.44, 0.0), (0, -0.42, 0.0), 0.02, 0.024, segs=16)       # gimbal nock
    alu = mesh_object("RodButt", bm, col, [M.reel_gold], parent=mount, recalc=False)
    bm = bmesh.new()
    bm_frustum(bm, (0, 0.18, 0.0), (0, 0.5, 0.0), 0.026, 0.022, segs=16)          # foregrip
    grip = mesh_object("RodForegrip", bm, col, [M.grip], parent=mount, recalc=False)
    reel = build_reel(col, M, parent=mount, name="StrikeReel")
    reel["root"].location = (0.0, 0.05, 0.02)
    for o in (ob, alu, grip):
        o.visible_shadow = True
    return dict(ctrl=ctrl, ob=ob, mount=mount, reel=reel, guide=gob)


def place_rod(rod, f, butt, d, load_dir, bend):
    """Rod from its butt along d, bowed toward load_dir by bend (0 straight .. 1 doubled over)."""
    d = d.normalized()
    tip_dir = d.slerp(load_dir.normalized(), min(0.97, bend))
    butt = Vector(butt)
    mid = butt + d * (ROD_LEN * 0.48)
    tip = mid + (d * 0.25 + tip_dir * 0.75).normalized() * (ROD_LEN * 0.52) * (1.0 - 0.12 * bend)
    for k, p in (("Butt", butt), ("Mid", mid), ("Tip", tip)):
        rod["ctrl"][k].location = p
        key(rod["ctrl"][k], "location", f)
    side = d.cross(Vector((0, 0, 1)))
    side = side.normalized() if side.length > 1e-3 else Vector((1, 0, 0))
    up = side.cross(d).normalized()
    rod["mount"].location = butt + d * 0.44
    rod["mount"].rotation_quaternion = Matrix((side, d, up)).transposed().to_quaternion()
    key(rod["mount"], "location", f)
    key(rod["mount"], "rotation_quaternion", f)
    return tip, mid


def bezier_point(a, b, c, t):
    return a * (1 - t) ** 2 + b * 2 * (1 - t) * t + c * t * t
# --------------------------------------------------------------------------
# Choreography. Places: the wreck site (the hook, the strike, the turn, and the water above it for the loop's
# overhead shot and the boat), the cut away wreck (the cross section, XS) and the rig studio (RS), far apart and
# each rendered only in its own shots.
# --------------------------------------------------------------------------
XS = Vector((-600.0, 0.0, 0.0))
RS = Vector((600.0, 0.0, -10.0))
Z0 = SEABED_Z - 0.8                  # the wreck's keel: sunk 0.8 m into the sand
HOLE = Vector((4.2, -WRECK_B / 2, Z0 + 2.2))        # the hole in the starboard side where the goliath lives (wreck frame)
HOLE_R = (1.7, 1.3)                  # half width (along x) and half height of the hole
BULKHEADS = (-9.0, 0.0, 9.0)
TWEEN_Z = 3.9                        # the tween deck over the goliath's hold (above the keel)
SAND_IN = 1.0                        # sand inside the hull, above the keel
CUT_Y = -0.3                         # the cross section cuts the hull lengthwise here
BOAT_AT = Vector((6.0, 9.0, 0.0))    # the boat, anchored over the wreck (stern toward it)
STRIKE_F = VO_START[4] + int(round(1.6 * FPS))      # "...flaring that huge mouth to inhale prey"
LOCK_F = CUT3B + 8                   # the lever goes all the way up
TURN_F = CUT3C + 20                  # the line turns the fish's head
GAPE_F = (CUT1A - 62, CUT1A - 16)                  # the hook: the mouth opens wide (the reveal), then closes
OH_POS = Vector((0.0, -0.4, 34.0))   # the loop's overhead shot over the wreck
OH_TGT = Vector((0.0, -0.4, 0.0))


def hull_hw(x, z):
    """Half beam of the hull at length x and height z above the keel (wreck frame)."""
    u = x / (WRECK_L / 2)
    k = 1.0
    if u > 0.55:
        k = 1.0 - ((u - 0.55) / 0.45) ** 1.6 * 0.88
    elif u < -0.78:
        k = 1.0 - ((-u - 0.78) / 0.22) ** 2 * 0.3
    zr = min(1.0, max(0.0, z / WRECK_H))
    prof = 0.2 + 0.8 * (1.0 - (1.0 - zr) ** 3) ** 0.5
    return WRECK_B / 2 * k * prof


def section_pts(x, nside=36, nbot=6, y_min=None):
    """The hull section at x as a polyline: port side from the deck edge down, the flat bottom, starboard up."""
    pts = []
    for k in range(nside + 1):
        z = WRECK_H * (1 - k / nside)
        pts.append((hull_hw(x, z), z))
    hb = hull_hw(x, 0.0)
    for k in range(1, nbot):
        pts.append((hb * (1 - 2 * k / nbot), 0.0))
    for k in range(nside + 1):
        z = WRECK_H * k / nside
        pts.append((-hull_hw(x, z), z))
    if y_min is not None:
        pts = [(max(y, y_min), z) for y, z in pts]
    return pts


def hole_mask(x, z, side):
    """1 where the starboard (side -1) or port (+1) shell has rusted through."""
    if side < 0:
        dx, dz = (x - HOLE.x) / HOLE_R[0], (z - (HOLE.z - Z0)) / HOLE_R[1]
        rag = 0.18 * mnoise.noise(Vector((x * 1.3, z * 1.3, 4.0)))
        if dx * dx + dz * dz < (1.0 + rag) ** 2:
            return True
    n = mnoise.noise(Vector((x * 0.32, z * 0.45, 7.0 + side)))
    return n > 0.42 and 0.8 < z < WRECK_H - 0.4 and abs(x) < WRECK_L / 2 - 3


def build_wreck(col, M, origin, cut=False, name="Wreck"):
    """A steel freighter hull lying upright, sunk in the sand: shell plating (rusted through in places, a big hole
    on the starboard side), three bulkheads, a tween deck over the goliath's hold, a rotted main deck with its
    beams, sand inside, barnacles and sea fans. cut=True: the starboard half is cut away along CUT_Y (the cross
    section), with hatched cut faces."""
    o = Vector(origin) + Vector((0.0, 0.0, Z0 - SEABED_Z))
    nx = 200
    xs = [-WRECK_L / 2 + WRECK_L * i / nx for i in range(nx + 1)]
    bm = bmesh.new()
    rows = [[bm.verts.new(o + Vector((x, y, z))) for y, z in section_pts(x)] for x in xs]
    npts = len(rows[0])
    for i in range(nx):
        xm = (xs[i] + xs[i + 1]) / 2
        sec_m = section_pts(xm)
        for j in range(npts - 1):
            ya, za = sec_m[j]
            yb, zb = sec_m[j + 1]
            ym, zm = (ya + yb) / 2, (za + zb) / 2
            side = 1 if ym > 0.05 else (-1 if ym < -0.05 else 0)
            if cut and ym < CUT_Y:
                continue
            if side and hole_mask(xm, zm, side):
                continue
            bm.faces.new((rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]))
    st = [bm.verts.new(o + Vector((xs[0], y, z))) for y, z in section_pts(xs[0], y_min=CUT_Y if cut else None)]
    bm.faces.new(st)                                              # the stern transom
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    # rust holes: relax the stair-stepped edges of the removed panels into smooth ragged outlines on the hull
    edge_v = [v for v in bm.verts if any(e.is_boundary for e in v.link_edges)]
    inner = []
    for v in edge_v:
        lx, lz = v.co.x - o.x, v.co.z - o.z
        if 0.08 < lz < WRECK_H - 0.08 and abs(lx) < WRECK_L / 2 - 0.08 and abs(v.co.y - o.y) > 0.35:
            nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            if len(nb) == 2:
                inner.append((v, nb))
    for _ in range(8):
        new = {}
        for v, nb in inner:
            new[v] = (v.co * 0.4 + (nb[0].co + nb[1].co) * 0.3)
        for v, c in new.items():
            lx, lz = c.x - o.x, c.z - o.z
            side = 1.0 if c.y - o.y > 0 else -1.0
            v.co = Vector((c.x, o.y + side * hull_hw(lx, lz), c.z))
    shell = finish(name + "Shell", bm, col, [M.rust])
    for p in shell.data.polygons:
        p.use_smooth = True
    sol = shell.modifiers.new("Plate", "SOLIDIFY")
    sol.thickness = 0.08
    sol.offset = -1.0
    parts = [shell]
    bm = bmesh.new()                                              # bulkheads, tween deck, main deck, beams
    for xb in BULKHEADS:
        pts = section_pts(xb, y_min=CUT_Y if cut else None)
        poly = [bm.verts.new(o + Vector((xb, y * 0.99, z))) for y, z in pts]
        bm.faces.new(poly)
    yl = (CUT_Y if cut else -hull_hw(4.5, TWEEN_Z) * 0.99)
    for xa, xb_, zz in ((0.0, 9.0, TWEEN_Z), (-9.0, -0.5, TWEEN_Z)):
        hw = hull_hw((xa + xb_) / 2, zz) * 0.99
        bm_box(bm, tuple(o + Vector(((xa + xb_) / 2, (hw + max(yl, -hw)) / 2, zz))), (xb_ - xa, hw - max(yl, -hw), 0.06))
    rng = random.Random(5)
    x = -WRECK_L / 2 + 1.0
    while x < WRECK_L / 2 - 4.0:                                  # the main deck: plates with rotted gaps, beams
        w = rng.uniform(1.2, 3.2)
        hw = hull_hw(x + w / 2, WRECK_H) * 0.99
        lo = max(CUT_Y, -hw) if cut else -hw
        if rng.random() > 0.38:
            bm_box(bm, tuple(o + Vector((x + w / 2, (hw + lo) / 2, WRECK_H - 0.04))), (w, hw - lo, 0.06))
        x += w + rng.uniform(0.2, 1.6)
    for xb in [-WRECK_L / 2 + 1.5 + 1.8 * i for i in range(16)]:
        if xb > WRECK_L / 2 - 4.0:
            break
        hw = hull_hw(xb, WRECK_H) * 0.99
        lo = max(CUT_Y, -hw) if cut else -hw
        bm_box(bm, tuple(o + Vector((xb, (hw + lo) / 2, WRECK_H - 0.2))), (0.16, hw - lo, 0.3))
    inner = finish(name + "Inner", bm, col, [M.rust])
    parts.append(inner)
    bm = bmesh.new()                                              # sand inside the hull
    for i in range(nx):
        xa, xb_ = xs[i], xs[i + 1]
        ha, hb = hull_hw(xa, SAND_IN) * 0.98, hull_hw(xb_, SAND_IN) * 0.98
        la, lb = (max(CUT_Y, -ha), max(CUT_Y, -hb)) if cut else (-ha, -hb)
        bm.faces.new([bm.verts.new(o + Vector((xa, la, SAND_IN))), bm.verts.new(o + Vector((xb_, lb, SAND_IN))),
                      bm.verts.new(o + Vector((xb_, hb, SAND_IN))), bm.verts.new(o + Vector((xa, ha, SAND_IN)))])
    sand_in = finish(name + "SandIn", bm, col, [M.sand])
    parts.append(sand_in)
    if cut:                                                       # hatched cut faces (the diagram's section)
        bm = bmesh.new()
        for xb in BULKHEADS:
            bm_box(bm, tuple(o + Vector((xb, CUT_Y - 0.03, WRECK_H / 2 + 0.3))), (0.14, 0.04, WRECK_H - 0.6))
        for xa, xb_, zz in ((0.0, 9.0, TWEEN_Z), (-9.0, -0.5, TWEEN_Z)):
            bm_box(bm, tuple(o + Vector(((xa + xb_) / 2, CUT_Y - 0.03, zz))), (xb_ - xa, 0.04, 0.14))
        bm_box(bm, tuple(o + Vector((0.0, CUT_Y - 0.03, 0.06))), (WRECK_L - 0.4, 0.04, 0.16))      # the keel
        bm_box(bm, tuple(o + Vector((0.0, CUT_Y - 0.03, SAND_IN / 2))), (WRECK_L - 0.6, 0.03, SAND_IN))
        cf = finish(name + "CutFaces", bm, col, [M.cutface])
        cf.visible_shadow = False
        parts.append(cf)
    # barnacles over the shell (geometry nodes: points on the faces, a small cone on each, turned to the normal)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=7, radius1=0.02, radius2=0.007, depth=0.02,
                          matrix=Matrix.Translation((0, 0, 0.012)))
    bar = finish(name + "BarnacleShape", bm, col, [M.barnacle])
    bar.hide_render = bar.hide_viewport = True
    ng, gi, go = new_group(name + "Barnacles", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    dist = ng.nodes.new("GeometryNodeDistributePointsOnFaces")
    dist.inputs["Density"].default_value = 36.0
    ng.links.new(gi.outputs["Geometry"], dist.inputs["Mesh"])
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = bar
    oi.inputs["As Instance"].default_value = True
    inst = ng.nodes.new("GeometryNodeInstanceOnPoints")
    ng.links.new(dist.outputs["Points"], inst.inputs["Points"])
    ng.links.new(oi.outputs["Geometry"], inst.inputs["Instance"])
    ng.links.new(dist.outputs["Rotation"], inst.inputs["Rotation"])
    rv = ng.nodes.new("FunctionNodeRandomValue")
    rv.data_type = "FLOAT"
    rv.inputs["Min"].default_value = 0.4
    rv.inputs["Max"].default_value = 1.3
    ng.links.new(rv.outputs["Value"], inst.inputs["Scale"])
    join = ng.nodes.new("GeometryNodeJoinGeometry")
    ng.links.new(gi.outputs["Geometry"], join.inputs[0])
    ng.links.new(inst.outputs[0], join.inputs[0])
    ng.links.new(join.outputs[0], go.inputs["Geometry"])
    shell.modifiers.new("Barnacles", "NODES").node_group = ng
    bm = bmesh.new()                                              # sea fans on the deck edges and the rails
    for k in range(18):
        xk = rng.uniform(-WRECK_L / 2 + 2, WRECK_L / 2 - 5)
        sd = rng.choice((1, -1)) if not cut else 1
        hw = hull_hw(xk, WRECK_H)
        base = o + Vector((xk, sd * hw * 0.97, WRECK_H))
        hgt, wd = rng.uniform(0.5, 1.1), rng.uniform(0.5, 1.0)
        ang = rng.uniform(-0.6, 0.6)
        a_ = Vector((math.cos(ang), math.sin(ang), 0.0)) * wd / 2
        vs = [bm.verts.new(base - a_ * 0.15), bm.verts.new(base + a_ * 0.15), bm.verts.new(base + a_ + Vector((0, 0, hgt))),
              bm.verts.new(base - a_ + Vector((0, 0, hgt)))]
        bm.faces.new(vs)
    fans = finish(name + "SeaFans", bm, col, [M.seafan], recalc=False)
    fans.visible_shadow = False
    parts.append(fans)
    return parts


def build_sand(col, M, center, size=120.0, name="Seabed"):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=60, y_segments=60, size=size / 2,
                          matrix=Matrix.Translation((center[0], center[1], SEABED_Z)))
    for v in bm.verts:
        v.co.z += 0.25 * mnoise.noise(v.co * 0.08) + 0.06 * mnoise.noise(v.co * 0.7)
    ob = finish(name, bm, col, [M.sand])
    for p in ob.data.polygons:
        p.use_smooth = True
    return ob


# --- the goliath's paths -----------------------------------------------------------------------------------------
def wpt(v):
    return WRECK_AT + Vector((0.0, 0.0, 0.0)) + Vector(v) - Vector((0.0, 0.0, SEABED_Z)) + Vector((0.0, 0.0, SEABED_Z))


G_FACE = Vector((0.0, -1.0, 0.0))    # in the hole the goliath faces out of it (-y)


def hook_goliath(f):
    """In the hole, head just inside; for the second angle it eases half out of the hole, nosing at the light."""
    base = WRECK_AT + Vector((HOLE.x, HOLE.y + 1.25, HOLE.z - SEABED_Z - 0.1))
    k = smoothstep(CUT1A - 10, CUT1 + 20, f)
    p = base + Vector((0.0, -1.0 * k, 0.05 * math.sin(t_of(f) * 0.7)))
    fwd = Vector((0.06 * math.sin(t_of(f) * 0.4), -1.0, 0.03)).normalized()
    return p, fwd


def hook_gape(f):
    a, b = GAPE_F
    return smoothstep(a, a + 10, f) * (1.0 - smoothstep(b - 6, b + 12, f))


def xs_goliath(f):
    """Cross section: wedged lengthwise in the hold under the tween deck, facing the opening (screen left)."""
    t = t_of(f)
    p = XS + Vector((4.8, 0.9, Z0 - SEABED_Z + 2.05 + 0.04 * math.sin(t * 0.8))) + Vector((0.0, 0.0, SEABED_Z))
    p.z -= SEABED_Z
    p = XS + Vector((4.8, 0.9, Z0 + 2.05 + 0.04 * math.sin(t * 0.8)))
    fwd = Vector((-1.0, 0.04 * math.sin(t * 0.5), 0.0)).normalized()
    return p, fwd


def bait_strike(f):
    """The live jack on the cable, swimming nervously outside the hole until it is inhaled."""
    t = t_of(f)
    c = WRECK_AT + Vector((HOLE.x + 0.4, HOLE.y - 2.6, HOLE.z - SEABED_Z + 0.1))
    p = c + Vector((0.5 * math.sin(t * 1.7), 0.25 * math.sin(t * 2.3), 0.15 * math.sin(t * 2.9)))
    fwd = Vector((math.cos(t * 1.7), 0.35 * math.cos(t * 2.3), 0.0)).normalized()
    return p, fwd


def strike_goliath(f):
    """The rush: out of the hole toward the bait, mouth flaring, then back toward the dark."""
    base = WRECK_AT + Vector((HOLE.x, HOLE.y + 1.25, HOLE.z - SEABED_Z - 0.1))
    k = ease(smoothstep(STRIKE_F - 10, STRIKE_F, f)) - 0.6 * ease(smoothstep(STRIKE_F + 8, STRIKE_F + 40, f))
    bp, _ = bait_strike(STRIKE_F)
    tgt = bp - Vector((0.0, -1.0, 0.0)) * (GOLIATH_SCALE * G_L / 2 + 0.15)
    p = base.lerp(tgt, k)
    fwd = (bp - base).normalized().lerp(Vector((0, -1, 0)), 0.3).normalized()
    return p, fwd


def strike_gape(f):
    return smoothstep(STRIKE_F - 5, STRIKE_F - 1, f) * (1.0 - smoothstep(STRIKE_F + 5, STRIKE_F + 14, f))


def mouth_of(p, fwd):
    return p + heading_quat(fwd) @ Vector((GOLIATH_SCALE * (G_L / 2 + 0.005), 0.0, -0.03 * GOLIATH_SCALE))


def turn_goliath(f):
    """On the line: swimming hard for the hole, then the pull turns its head away from the wreck and up."""
    start = WRECK_AT + Vector((HOLE.x + 2.2, HOLE.y - 4.4, HOLE.z - SEABED_Z + 0.4))

    def yaw_k(g):
        kk = ease(smoothstep(TURN_F - 6, TURN_F + 30, g))
        return math.radians(210.0) - math.radians(150.0) * kk, kk

    def dir_at(g):
        y_, kk = yaw_k(g)
        return Vector((math.sin(y_), -math.cos(y_), 0.3 * kk)).normalized()
    p = start.copy()
    for s_ in range(max(0, f - CUT3C)):
        _, kk = yaw_k(CUT3C + s_)
        p = p + dir_at(CUT3C + s_) * (1.1 - 0.5 * kk) / FPS
    return p, dir_at(f)


# --- the rig (studio) ----------------------------------------------------------------------------------------
RIG_HOOK = RS + Vector((0.02, 0.0, 0.11))           # where the hook sits in the bait's back


def circle_hook_pts(scale=1.0):
    """A big circle hook as a polyline (hook frame: shank along +Z, the bend round to +X, the point turned in)."""
    pts = [Vector((0.0, 0.0, 0.075 - 0.065 * i / 8)) for i in range(9)]
    c, r = Vector((0.032, 0.0, 0.01)), 0.032
    for i in range(1, 30):
        a = math.pi + 1.5 * math.pi * i / 29
        pts.append(c + Vector((math.cos(a), 0.0, math.sin(a))) * r)
    end = pts[-1]
    for i in range(1, 7):
        pts.append(end + Vector((-0.004 * i, 0.0, -0.0012 * i)))
    return [p * scale for p in pts]


def build_rig(col, M):
    """The rig on a turntable: a whole jack crevalle bait with a big circle hook through its back, the cable
    leader from the hook's eye through a crimped loop and on up to a swivel and the main line."""
    piv = empty_object("RigPivot", col, size=0.2)
    piv.location = RS
    piv.rotation_mode = "XYZ"
    jk = bpy.data.objects.new("RigBait", jack_mesh(M, L=0.5))
    col.objects.link(jk)
    jk.parent = piv
    jk.location = (0.0, 0.0, 0.0)
    bm = bmesh.new()
    hp = circle_hook_pts(1.25)
    off = Vector((0.03, 0.0, 0.035))
    tube(bm, [p + off for p in hp], lambda u: 0.0042 if u < 0.92 else 0.0042 * (1 - (u - 0.92) / 0.08) + 0.0004, segs=10)
    eye = Vector((0.0, 0.0, 0.075 * 1.25)) + off + Vector((0.0, 0.0, 0.009))
    torus(bm, eye, (0, 1, 0), 0.008, 0.0028, n=16, m=6)
    hook = finish("RigHook", bm, col, [M.hook])
    hook.parent = piv
    hook.location = (0.03, 0.0, 0.03)
    bm = bmesh.new()                                     # cable: through the eye, crimped back on itself, then up
    e = eye + Vector((0.03, 0.0, 0.03))
    loop = [e + Vector((0.012 * math.sin(a), 0.0, 0.012 * (1 - math.cos(a)) + 0.0)) for a in [math.pi * i / 10 for i in range(11)]]
    run = []
    for i in range(40):
        u = i / 39
        run.append(e + Vector((0.0, 0.0, 0.03)) + Vector((0.05 * math.sin(u * 2.4), 0.02 * u, 0.42 * u)))
    tube(bm, loop + run, 0.0019, segs=8)
    tube(bm, [e + Vector((0.006, 0.0, 0.03)), e + Vector((0.006, 0.0, 0.06))], 0.0019, segs=8)
    cable = finish("RigCable", bm, col, [M.cable])
    cable.parent = piv
    bm = bmesh.new()
    bm_frustum(bm, tuple(e + Vector((0.003, 0.0, 0.03))), tuple(e + Vector((0.003, 0.0, 0.052))), 0.0055, 0.0055, segs=12)
    sleeve = finish("RigSleeve", bm, col, [M.sleeve])
    sleeve.parent = piv
    top = run[-1]
    bm = bmesh.new()                                     # a heavy swivel at the top of the leader
    torus(bm, top + Vector((0.0, 0.0, 0.01)), (0, 1, 0), 0.008, 0.0025, n=14, m=6)
    bm_frustum(bm, tuple(top + Vector((0.0, 0.0, 0.018))), tuple(top + Vector((0.0, 0.0, 0.05))), 0.0055, 0.0045, segs=10)
    torus(bm, top + Vector((0.0, 0.0, 0.058)), (0, 1, 0), 0.008, 0.0025, n=14, m=6)
    swivel = finish("RigSwivel", bm, col, [M.chrome])
    swivel.parent = piv
    bm = bmesh.new()
    tube(bm, [top + Vector((0.0, 0.0, 0.066)), top + Vector((0.0, 0.0, 0.6))], 0.0012, segs=6)
    mline = finish("RigLine", bm, col, [M.line])
    mline.parent = piv
    bm = bmesh.new()                                     # the studio sweep: floor in front, curving up behind
    prof = [(-4.0, -1.2)] + [(0.6 + 2.5 * math.sin(math.pi * i / 48), -1.2 + 2.5 * (1 - math.cos(math.pi * i / 48)))
                             for i in range(25)] + [(3.1, 10.0)]
    for (y0, z0), (y1, z1) in zip(prof[:-1], prof[1:]):
        bm.faces.new([bm.verts.new(RS + Vector((-6, y0, z0))), bm.verts.new(RS + Vector((6, y0, z0))),
                      bm.verts.new(RS + Vector((6, y1, z1))), bm.verts.new(RS + Vector((-6, y1, z1)))])
    back = finish("RigBackdrop", bm, col, [M.studio], recalc=False)
    back.visible_shadow = False
    pts = dict(eye=e, sleeve=e + Vector((0.003, 0.0, 0.04)), point=hp[-1] + off + Vector((0.03, 0.0, 0.03)),
               bend=hp[22] + off + Vector((0.03, 0.0, 0.03)), cable=run[10], bait=Vector((-0.12, 0.0, 0.0)))
    return dict(pivot=piv, bait=jk, parts=[jk, hook, cable, sleeve, swivel, mline], back=back, pts=pts)


def rig_rot(f):
    u = (f - CUT2) / max(1, CUT3 - CUT2)
    return math.radians(-30.0 + 70.0 * ease(u))


def rig_point(rig, key_, f):
    """A rig point (pivot local) in world space at frame f (the turntable turns about z)."""
    q = Euler((0.0, 0.0, rig_rot(f)), "XYZ").to_quaternion()
    return RS + q @ rig["pts"][key_]


# --------------------------------------------------------------------------
# Camera: frames[f] = (position, target, rotation, lens); f-stop per frame
# --------------------------------------------------------------------------
def whip(q, angle_deg, axis):
    return Quaternion(Vector(axis), math.radians(angle_deg)) @ q


def gauss_smooth(fn, f, k=6):
    acc, ws = Vector((0, 0, 0)), 0.0
    for d in range(-k, k + 1):
        w = math.exp(-0.5 * (d / (k / 2.0)) ** 2)
        acc += fn(f + d) * w
        ws += w
    return acc / ws


def overhead():
    return OH_POS.copy(), OH_TGT.copy(), look_at(OH_POS, OH_TGT, Vector((1, 0, 0)))


def camera_frames(st):
    frames, fstop = {}, {}
    Z = Vector((0, 0, 1))
    oc, ot, oq = overhead()
    # 0 the loop's join: the water over the wreck from above, then a whip pan down
    for f in range(1, WHIP_IN):
        ang = 0.0 if f <= 3 else (13.0 * (f - 3) ** 1.6)
        frames[f] = (oc, ot, whip(oq, ang, oq @ Vector((1, 0, 0))), 26.0)
        fstop[f] = 11.0
    # 1 hook: a push in and zoom toward the hole; the mouth opens in the dark
    hole_w = WRECK_AT + Vector((HOLE.x, HOLE.y, HOLE.z - SEABED_Z))
    for f in range(WHIP_IN, CUT1A):
        u = (f - WHIP_IN) / (CUT1A - WHIP_IN)
        k = 0.9 * ease(smoothstep(WHIP_IN, CUT1A - 50, f)) + 0.1 * u
        cam = hole_w + Vector((-1.6 + 3.3 * k, -9.5 + 6.4 * k, 1.6 - 1.4 * k))
        mouth = mouth_of(*hook_goliath(f))
        tgt = (hole_w + Vector((0.0, 0.6, -0.1))).lerp(mouth, ease(smoothstep(WHIP_IN + 20, CUT1A - 50, f)))
        q = look_at(cam, tgt, Z)
        if f < WHIP_IN + 6:
            kk = 1.0 - ease_out((f - WHIP_IN) / 6.0, 3.0)
            q = whip(q, 36.0 * kk, q @ Vector((1, 0, 0)))
        frames[f] = (cam, tgt, q, 24.0 + 26.0 * ease(smoothstep(WHIP_IN + 8, CUT1A - 50, f)) + 8.0 * smoothstep(CUT1A - 50, CUT1A, f))
        fstop[f] = 4.0
    # 1b side on, low along the hull: the goliath noses half out of the hole
    for f in range(CUT1A, CUT1):
        u = (f - CUT1A) / (CUT1 - CUT1A)
        gp, gf = hook_goliath(f)
        head = gp + gf * 0.8
        cam = Vector((head.x + 3.4 - 0.3 * u, hole_w.y - 1.7, head.z + 0.25))
        tgt = head + Vector((-0.5, 0.0, 0.0))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 30.0)
        fstop[f] = 4.0
    # 2 the cross section: match cut on the fish (side on, facing left), then a pull back to the whole cut hull
    for f in range(CUT1, CUT2):
        gp, gf = xs_goliath(f)
        near_cam = gp + Vector((-1.0, -4.6, 0.25))
        near_tgt = gp + Vector((-1.0, 0.0, 0.0))
        u = (f - CUT1 - 50) / max(1, CUT2 - CUT1 - 50)
        wide_cam = XS + Vector((1.5 + 1.0 * max(0.0, u), -16.5 + 1.5 * ease(max(0.0, u)), Z0 + 4.2))
        wide_tgt = XS + Vector((1.5 + 1.0 * max(0.0, u), 0.0, Z0 + 2.6))
        k = ease(smoothstep(CUT1 + 8, CUT1 + 50, f))
        cam = near_cam.lerp(wide_cam, k)
        tgt = near_tgt.lerp(wide_tgt, ease(smoothstep(CUT1 + 8, CUT1 + 42, f)))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 30.0)
        fstop[f] = 4.0 + 12.0 * k
    # 3 the rig: a close turntable; then tight on the hook and the crimped cable
    for f in range(CUT2, CUT2B):
        u = (f - CUT2) / (CUT2B - CUT2)
        cam = RS + Vector((0.12, -1.25 + 0.15 * ease(u), 0.2 + 0.03 * u))
        tgt = RS + Vector((0.05, 0.0, 0.12))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 40.0)
        fstop[f] = 5.6
    for f in range(CUT2B, CUT3):
        u = (f - CUT2B) / (CUT3 - CUT2B)
        e = rig_point(st["rig"], "eye", f)
        cam = e + Vector((0.12 - 0.05 * u, -0.42 + 0.06 * u, -0.02))
        tgt = e + Vector((0.0, 0.0, -0.085 + 0.085 * ease(u)))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 50.0)
        fstop[f] = 4.0
    # 4 the strike: low on the sand, looking up at the hole and the bait; then the rod; then the turn
    for f in range(CUT3, CUT3B):
        cam = hole_w + Vector((2.6, -6.4, -0.45))
        bp, _ = bait_strike(f)
        tgt = hole_w.lerp(bp, 0.55) + Vector((0.0, 0.0, 0.1))
        shake = 0.03 * smoothstep(STRIKE_F - 2, STRIKE_F, f) * (1 - smoothstep(STRIKE_F, STRIKE_F + 16, f))
        cam = cam + Vector((shake * math.sin(f * 2.1), 0.0, shake * math.sin(f * 2.7)))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 26.0)
        fstop[f] = 5.6
    rod = st["rod"]
    sc = bpy.context.scene
    for f in range(CUT3B, CUT3C):
        sc.frame_set(f)
        rm = rod["reel"]["root"].matrix_world.copy()
        t = t_of(f)
        shake = Vector((0.004 * math.sin(t * 7.1), 0.003 * math.sin(t * 5.3 + 1), 0.004 * math.sin(t * 6.2 + 2)))
        shake *= 1.0 + 2.0 * smoothstep(LOCK_F - 2, LOCK_F + 2, f) * (1.0 - smoothstep(LOCK_F + 2, LOCK_F + 14, f))
        cam = rm @ (Vector((-0.95, -0.45, 0.62)) + shake)
        tgt = rm @ Vector((0.0, 0.35, 0.05))
        up = (rm.to_3x3() @ Vector((0, 0, 1))).normalized()
        frames[f] = (cam, tgt, look_at(cam, tgt, up.lerp(Z, 0.6)), 26.0)
        fstop[f] = 2.8
    for f in range(CUT3C, CUT4):
        gp = gauss_smooth(lambda g: turn_goliath(g)[0], f, 8)
        cam = gp + Vector((1.8, -4.4, 0.6))
        tgt = gp + Vector((0.0, 0.0, 0.15))
        q = look_at(cam, tgt, Z)
        if f >= CUT4 - 3:
            q = whip(q, -(15.0 * (f - (CUT4 - 4)) ** 1.6), q @ Vector((1, 0, 0)))
        frames[f] = (cam, tgt, q, 28.0)
        fstop[f] = 5.6
    # 5 outro: whip in to the water over the wreck from above and hold it (frame 1's shot: the loop)
    for f in range(CUT4, FRAME_END + 1):
        q = oq
        if f < CUT4 + 7:
            kk = 1.0 - ease_out((f - CUT4) / 7.0, 3.0)
            q = whip(oq, 40.0 * kk, Z)
        frames[f] = (oc, ot, q, 26.0)
        fstop[f] = 11.0
    return frames, fstop


def build_camera(col, frames, fstop):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = LENS
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = SENSOR
    cd.clip_start = 0.01
    cd.clip_end = 3000.0
    cd.dof.use_dof = True
    cd.dof.aperture_blades = 7
    cam = bpy.data.objects.new("Camera", cd)
    col.objects.link(cam)
    cam.rotation_mode = "QUATERNION"
    prev = None
    for f in range(1, FRAME_END + 1):
        p, t, q, lens = frames[f]
        if prev is not None and prev.dot(q) < 0.0:
            q = -q
        prev = q
        cam.location = p
        cam.rotation_quaternion = q
        key(cam, "location", f)
        key(cam, "rotation_quaternion", f)
        cd.lens = lens
        cd.keyframe_insert("lens", frame=f)
        cd.dof.focus_distance = (t - p).length
        cd.dof.aperture_fstop = fstop[f]
        cd.dof.keyframe_insert("focus_distance", frame=f)
        cd.dof.keyframe_insert("aperture_fstop", frame=f)
    for fc in iter_fcurves(cam) + iter_fcurves(cd):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    for c in CUTS:
        for fc in iter_fcurves(cam) + iter_fcurves(cd):
            v_new = fc.evaluate(c)
            for kp in fc.keyframe_points:
                if abs(kp.co.x - (c - 1)) < 1e-3:
                    kp.interpolation = "CONSTANT"
            kp = fc.keyframe_points.insert(c - 0.5, v_new)
            kp.interpolation = "LINEAR"
    bpy.context.scene.camera = cam
    return cam


# --------------------------------------------------------------------------
# Scene state per frame
# --------------------------------------------------------------------------
def under_at(f):
    return 0.0 if (f < WHIP_IN or CUT3B <= f < CUT3C or f >= CUT4) else 1.0


def deep_at(f):
    if CUT2 <= f < CUT3:
        return 0.2
    return 0.45 if under_at(f) else 0.0


def clear_at(f):
    if CUT1 <= f < CUT2:
        return 0.75 * smoothstep(CUT1 + 8, CUT1 + 50, f)
    if CUT2 <= f < CUT3:
        return 1.0
    return 0.0


def set_constant(idb):
    for fc in iter_fcurves(idb):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def build_sun(col):
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 4.0
    light.specular_factor = 0.5
    light.color = srgb(255, 244, 228)
    light.angle = math.radians(2.0)
    ob = bpy.data.objects.new("Sun", light)
    col.objects.link(ob)
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (-SUN_DIR).to_track_quat("-Z", "Y")
    return ob


def build_fill(col, name, energy=0.0, size=4.0, color=(255, 236, 214)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "DISK"
    ld.size = size
    ld.color = srgb(*color)
    ld.energy = energy
    ob = bpy.data.objects.new(name, ld)
    col.objects.link(ob)
    ob.rotation_mode = "QUATERNION"
    return ob


def aim(ob, at):
    ob.rotation_quaternion = (Vector(at) - ob.location).normalized().to_track_quat("-Z", "Y")


def key_sea_center(sea):
    centres = [(1, OH_TGT), (CUT3B, BOAT_AT), (CUT4, OH_TGT)]
    for f, c in centres:
        sea.location = (c.x, c.y, 0.0)
        sea.keyframe_insert("location", frame=f - 0.5 if f > 1 else 1)
    set_constant(sea)


def key_scene_state(sc, world_vals, sun, key_l, bounce, cave):
    under_v, deep_v = world_vals
    hole_w = WRECK_AT + Vector((HOLE.x, HOLE.y, HOLE.z - SEABED_Z))
    for f in range(1, FRAME_END + 1):
        u, d, c = under_at(f), deep_at(f), clear_at(f)
        for prop, val in (("under", u), ("deep", d), ("clear", c)):
            sc[prop] = val
            sc.keyframe_insert(data_path=f'["{prop}"]', frame=f)
        under_v.outputs[0].default_value = u
        deep_v.outputs[0].default_value = d
        under_v.outputs[0].keyframe_insert("default_value", frame=f)
        deep_v.outputs[0].keyframe_insert("default_value", frame=f)
        sun.data.energy = (5.0 * (1 - 0.5 * d)) if u else 4.0
        sun.data.keyframe_insert("energy", frame=f)
        sun.rotation_quaternion = (-(SUN_UNDER if u else SUN_DIR)).to_track_quat("-Z", "Y")
        sun.keyframe_insert("rotation_quaternion", frame=f)
        key_l.data.energy, bounce.data.energy, cave.data.energy = 0.0, 0.0, 0.0
        if WHIP_IN <= f < CUT1 or CUT3 <= f < CUT3B or CUT3C <= f < CUT4:
            if f < CUT1:
                p, fwd = hook_goliath(f)
            elif f < CUT3B:
                p, fwd = strike_goliath(f)
            else:
                p, fwd = turn_goliath(f)
            key_l.location = p + Vector((0.0, -3.0, 4.0))
            aim(key_l, p)
            key_l.data.energy = 900.0
            bounce.location = p + Vector((1.5, -4.5, -1.5))
            aim(bounce, p)
            bounce.data.energy = 280.0
            if f < CUT3B:                                          # the fish's face in the dark hole
                cave.location = hole_w + Vector((0.6, -1.6, 0.5))
                aim(cave, hole_w + Vector((0.0, 0.8, -0.1)))
                cave.data.energy = 320.0
        elif CUT1 <= f < CUT2:
            key_l.location = XS + Vector((-2.0, -12.0, Z0 + 10.0))
            aim(key_l, XS + Vector((1.5, 1.0, Z0 + 2.5)))
            key_l.data.energy = 16000.0
            p, fwd = xs_goliath(f)
            bounce.location = p + Vector((0.0, -4.0, 0.3))
            aim(bounce, p)
            bounce.data.energy = 520.0
        elif CUT2 <= f < CUT3:
            key_l.location = RS + Vector((-0.8, -1.4, 1.4))
            aim(key_l, RS + Vector((0.0, 0.0, 0.1)))
            key_l.data.energy = 120.0
            bounce.location = RS + Vector((1.0, 0.6, 0.7))
            aim(bounce, RS + Vector((0.0, 0.0, 0.1)))
            bounce.data.energy = 90.0
        for ob in (key_l, bounce, cave):
            ob.data.keyframe_insert("energy", frame=f)
            key(ob, "location", f)
            key(ob, "rotation_quaternion", f)
    for idb in (sc, sc.world.node_tree, sun, key_l, key_l.data, bounce, bounce.data, cave, cave.data):
        set_constant(idb)


# --------------------------------------------------------------------------
# Animation
# --------------------------------------------------------------------------
def animate_goliath(g, spans, rate_hz=0.6, amp=0.012):
    phase = 0.0
    for f_lo, f_hi, state, gape_fn, a in spans:
        prev = None
        for f in range(f_lo, f_hi + 1):
            p, fwd = state(f)
            g["root"].location = p
            g["root"].rotation_quaternion = heading_quat(fwd)
            key(g["root"], "location", f)
            key(g["root"], "rotation_quaternion", f)
            turn = 0.0
            if prev is not None:
                turn = max(-1.0, min(1.0, prev.cross(fwd).z * 30.0))
            prev = fwd
            phase += 2 * math.pi * rate_hz / FPS
            key_swim(g, f, phase, a(f) if callable(a) else a, turn * 0.03)
            gp = gape_fn(f) if gape_fn else 0.0
            key_gape(g, f, gp, flare=gp)


def place(ob, f, p, fwd, roll=0.0):
    ob.location = p
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = heading_quat(fwd, roll)
    key(ob, "location", f)
    key(ob, "rotation_quaternion", f)


def suction(center_fn, f0, seed):
    """Silt, debris and bubbles pulled in toward the mouth from all around as it flares open (the suction)."""
    rng = random.Random(seed)
    sp = Particles()
    for _ in range(700):
        f = f0 + rng.uniform(-4.0, 1.0)
        c = center_fn(int(round(f)))
        d = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))
        if d.length < 1e-3:
            continue
        d.normalize()
        r = rng.uniform(0.25, 1.2)
        life = rng.uniform(0.12, 0.3)
        sp.emit(t_of(f), c + d * r, -d * r / life, life, rng.uniform(0.004, 0.012))
    for _ in range(500):                                          # a silt cloud kicked off the sand
        f = f0 + rng.uniform(0.0, 10.0)
        c = center_fn(int(round(f)))
        base = Vector((c.x + rng.gauss(0, 0.8), c.y + rng.gauss(0, 0.8), SEABED_Z + 0.1))
        v = Vector((rng.gauss(0, 0.5), rng.gauss(0, 0.5), rng.uniform(0.2, 0.9)))
        sp.emit(t_of(f), base, v, rng.uniform(1.5, 3.0), rng.uniform(0.03, 0.09), gravity=-0.05)
    return sp


def drift_silt(center, size, n, seed, f_lo, f_hi):
    """Suspended sediment drifting slowly in the current."""
    rng = random.Random(seed)
    sp = Particles()
    t0, t1 = t_of(f_lo) - 4.0, t_of(f_hi)
    for _ in range(n):
        p = Vector(center) + Vector((rng.uniform(-1, 1) * size[0], rng.uniform(-1, 1) * size[1], rng.uniform(-1, 1) * size[2]))
        v = Vector((rng.uniform(0.15, 0.35), rng.gauss(0, 0.05), rng.gauss(0, 0.03)))
        sp.emit(rng.uniform(t0, t1), p, v, rng.uniform(3.0, 6.0), rng.uniform(0.0015, 0.0045))
    return sp


def build_strike_line(col, M, bait, f_lo, f_hi):
    """The cable from the bait (then from the goliath's mouth) up toward the boat."""
    a = empty_object("StrikeLineA", col, size=0.05)
    b = empty_object("StrikeLineB", col, size=0.05)
    for f in range(f_lo, f_hi + 1):
        if f <= STRIKE_F:
            p, fwd = bait_strike(f)
            a.location = p + Vector((0.0, 0.0, 0.08))
        else:
            gp, gf = strike_goliath(f)
            a.location = mouth_of(gp, gf)
        b.location = WRECK_AT + Vector((BOAT_AT.x + 1.0, BOAT_AT.y - 2.0, -SEABED_Z))
        key(a, "location", f)
        key(b, "location", f)
    return line_object("StrikeLine", col, M, a, b, radius=0.002, count=30)


def build_turn_line(col, M, f_lo, f_hi):
    a = empty_object("TurnLineA", col, size=0.05)
    b = empty_object("TurnLineB", col, size=0.05)
    for f in range(f_lo, f_hi + 1):
        gp, gf = turn_goliath(f)
        a.location = mouth_of(gp, gf)
        b.location = a.location + Vector((6.0, 4.0, 14.0))
        key(a, "location", f)
        key(b, "location", f)
    return line_object("TurnLine", col, M, a, b, radius=0.0022, count=30)


def rig_boat_rod(boat, rod, f_lo, f_hi):
    """The rod in the middle gunwale holder, doubled over toward the water; the lever goes up to full."""
    sc = bpy.context.scene
    reel = rod["reel"]
    spin = 0.0
    for f in range(f_lo, f_hi + 1):
        t = t_of(f)
        pos, q = boat_attitude(Vector((BOAT_AT.x, BOAT_AT.y, 0.0)), t, heading=math.radians(180.0))
        key_boat(boat, f, pos, q)
        sc.frame_set(f)
        m = boat["root"].matrix_world
        rot = m.to_3x3().normalized()
        butt = m @ Vector((1.9, -(hull_section(1.9 / BOAT_LEN)[1][-1][0] - 0.15), 1.36))
        d = (rot @ Vector((-0.45, -0.55, 0.7))).normalized()
        load = (rot @ Vector((0.0, -0.4, -1.0))).normalized()
        bend = 0.9 + 0.03 * math.sin(f * 0.8)
        place_rod(rod, f, butt, d, load, bend)
        rate = 6.0 * (1.0 - smoothstep(LOCK_F, LOCK_F + 10, f))
        spin += 2 * math.pi * rate / FPS
        reel["spool"].rotation_euler = (-spin, 0.0, 0.0)
        key(reel["spool"], "rotation_euler", f)
        lev = LEVER_STRIKE + (LEVER_FULL - LEVER_STRIKE) * ease(smoothstep(LOCK_F - 4, LOCK_F + 4, f))
        reel["lever"].rotation_euler = (lev, 0.0, 0.0)
        key(reel["lever"], "rotation_euler", f)


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------
def build_overlays(hud, st):
    hud.scrims()
    hud.kicker("THE FISHING SERIES")
    sc = bpy.context.scene
    p1 = P(1, 1)
    hud.serif("In 1961, Lynn Joyner landed a *680 pound* goliath.", WHIP_IN + 6, p1 - 9, y=-0.30, size=0.046,
              lines=["In 1961, Lynn Joyner landed", "a *680 pound* goliath."], reveal=2.3,
              src="IGFA ALL TACKLE RECORD · MAY 20, 1961")
    hud.serif("Still the *all tackle* world record.", p1, CUT1 - 9, y=-0.30, size=0.046,
              lines=["Still the *all tackle*", "world record."], src="SPORT FISHING MAGAZINE · IGFA RECORDS")
    hud.pin(lambda f: hook_goliath(f)[0] + hook_goliath(f)[1] * 0.9 + Vector((0.0, 0.0, 0.5)), "goliath grouper",
            CUT1A + 6, CUT1 - 8, dot_color=GOLD, side=-0.05, y_band=(-0.18, 0.3))
    hud.badge(1, "WRECK CAVITY STRUCTURE", "dark iron cavities as ambush cover", CUT1 + 8, CUT2 - 8)
    s1 = P(2, 1)
    hud.serif("They don't *chase bait.*", CUT1 + 6, s1 - 9, lines=["They don't", "*chase bait.*"],
              src="FWC SPECIES PROFILE")
    hud.serif("They hold on *wrecks and pilings.*", s1, CUT2 - 9, lines=["They hold on", "*wrecks and pilings.*"],
              src="FWC · GULF AND CARIBBEAN FISHERIES INSTITUTE")
    hud.pin(lambda f: XS + Vector((4.8, 2.2, Z0 + TWEEN_Z - 0.4)), "dark cavity", CUT1 + 56, CUT2 - 10,
            dot_color=srgb(200, 160, 110), side=0.06, y_band=(-0.22, 0.32))
    hud.pin(lambda f: XS + Vector((-1.6, 3.5, Z0 + WRECK_H - 1.4)), "steel hull", CUT1 + 60, s1 - 4,
            dot_color=srgb(200, 160, 110), side=-0.05, y_band=(-0.22, 0.32))
    hud.pin(lambda f: XS + Vector((0.8, -5.3, SEABED_Z + 0.15)), "crabs: their main prey", s1 + 4, CUT2 - 10,
            dot_color=GOLD, side=0.05, rise=0.08, y_band=(-0.22, 0.32))
    hud.badge(2, "THE HEAVY RIG", "cable leader, circle hook, whole bait", CUT2 + 8, CUT3 - 8)
    hud.serif("Standard gear *won't hold.*", CUT2 + 6, CUT2B - 6, lines=["Standard gear", "*won't hold.*"],
              src="ANGLER TECHNIQUE")
    hud.serif("Cable leader, *big circle hook.*", CUT2B + 4, CUT3 - 9, lines=["Cable leader,", "*big circle hook.*"],
              src="ANGLER TECHNIQUE")
    rig = st["rig"]
    span = CUT2B - CUT2
    hud.pin(lambda f: rig_point(rig, "bait", f), "whole fish bait", CUT2 + 10, CUT2 + int(span * 0.5),
            dot_color=GOLD, side=-0.06, y_band=(-0.22, 0.3))
    hud.pin(lambda f: rig_point(rig, "cable", f), "heavy cable leader", CUT2 + int(span * 0.52), CUT2B - 8,
            dot_color=GOLD, side=0.06, y_band=(-0.22, 0.3))
    hud.pin(lambda f: rig_point(rig, "point", f), "big circle hook", CUT2B + 6, CUT2B + 6 + (CUT3 - CUT2B) // 2,
            dot_color=GOLD, side=0.06, y_band=(-0.22, 0.3))
    hud.pin(lambda f: rig_point(rig, "sleeve", f), "crimped sleeve", CUT2B + 8 + (CUT3 - CUT2B) // 2, CUT3 - 8,
            dot_color=GOLD, side=-0.06, rise=0.09, y_band=(-0.22, 0.3))
    hud.badge(3, "LOCK THE DRAG", "turn the head before the wreck", CUT3 + 6, CUT4 - 6)
    hud.serif("They feed by *suction.*", CUT3 + 4, CUT3B - 6, lines=["They feed", "by *suction.*"],
              src="SUCTION FEEDING: WAINWRIGHT ET AL. 2007")
    hud.serif("Lock your drag *down hard.*", CUT3B + 4, CUT3C - 6, lines=["Lock your drag", "*down hard.*"],
              src="ANGLER TECHNIQUE")
    hud.serif("Turn the head *before the wreck.*", CUT3C + 4, CUT4 - 4, lines=["Turn the head", "*before the wreck.*"],
              src="ANGLER TECHNIQUE")
    rod = st["rod"]

    def lever_pt(f):
        sc.frame_set(f)
        return rod["reel"]["root"].matrix_world @ Vector((-0.094, 0.0, 0.2))

    def rod_pt(f):
        sc.frame_set(f)
        return rod["ctrl"]["Mid"].matrix_world.translation.copy()
    hud.pin(lambda f: mouth_of(*strike_goliath(f)), "suction strike", STRIKE_F - 2, CUT3B - 8, dot_color=PIN_RED,
            side=0.06, y_band=(-0.2, 0.3))
    hud.pin(rod_pt, "rod doubled over", CUT3B + 2, CUT3C - 4, dot_color=GOLD, side=0.02, y_band=(-0.2, 0.3))
    hud.pin(lever_pt, "drag locked", LOCK_F + 2, CUT3C - 6, dot_color=PIN_RED, side=-0.06, rise=0.1, y_band=(-0.2, 0.3))
    hud.pin(lambda f: mouth_of(*turn_goliath(f)), "head turned", TURN_F + 4, CUT4 - 8, dot_color=GOLD, side=0.06,
            y_band=(-0.2, 0.3))
    hud.serif("Keep them *in the water.*", CUT4 + 8, FRAME_END - 9, lines=["Stop them early,", "keep them *in the water.*"],
              reveal=2.4, src="FWC: RELEASE GOLIATHS IN THE WATER")


def frames_pos(st, f):
    return st["frames"][f][0]


def setup_render(scale, samples, blur=True):
    sc = bpy.data.scenes["Main"]
    for s in bpy.data.scenes:
        s.render.resolution_x, s.render.resolution_y = RES_X, RES_Y
        s.render.resolution_percentage = scale
    sc.render.engine = "BLENDER_EEVEE"
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
    ee.use_fast_gi = True
    sc.render.use_motion_blur = blur
    sc.render.motion_blur_shutter = SHUTTER
    ee.motion_blur_steps = 1
    sc.view_settings.view_transform = "Standard"
    try:
        sc.view_settings.look = "None"
    except TypeError:
        pass
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.use_compositing = True


def build_compositor(main, hs):
    """Bloom, a high contrast underwater grade (deep greens, warm highlights) on the 3D only, then the overlays."""
    ng = bpy.data.node_groups.new("Composite", "CompositorNodeTree")
    main.compositing_node_group = ng
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.scene = main
    img = rl.outputs["Image"]
    try:
        gl = ng.nodes.new("CompositorNodeGlare")
        gl.inputs["Type"].default_value = "Bloom"
        for nm, v in (("Threshold", 1.0), ("Strength", 0.22), ("Size", 0.5)):
            if nm in gl.inputs:
                gl.inputs[nm].default_value = v
        ng.links.new(img, gl.inputs["Image"])
        img = gl.outputs["Image"]
    except Exception as e:                  # noqa: BLE001
        print("bloom skipped:", e)
    try:
        cb = ng.nodes.new("CompositorNodeColorBalance")
        vals = {"Lift": (1.0, 1.0, 1.0, 1.0), "Gamma": (1.0, 1.0, 1.0, 1.0), "Gain": (1.05, 1.02, 0.96, 1.0)}
        for sock in cb.inputs:
            if sock.bl_idname == "NodeSocketColor" and sock.name in vals:
                sock.default_value = vals[sock.name]
        ng.links.new(img, cb.inputs["Image"])
        img = cb.outputs["Image"]
    except Exception as e:                  # noqa: BLE001
        print("grade skipped:", e)
    try:
        cv = ng.nodes.new("CompositorNodeCurveRGB")
        c = cv.mapping.curves[3]
        c.points[0].location = (0.0, 0.0)
        c.points[1].location = (1.0, 1.0)
        c.points.new(0.25, 0.19)
        c.points.new(0.75, 0.82)
        cv.mapping.update()
        ng.links.new(img, cv.inputs["Image"])
        img = cv.outputs["Image"]
    except Exception as e:                  # noqa: BLE001
        print("curve skipped:", e)
    hud = ng.nodes.new("CompositorNodeRLayers")
    hud.scene = hs
    ao = ng.nodes.new("CompositorNodeAlphaOver")
    ng.links.new(img, ao.inputs["Background"])
    ng.links.new(hud.outputs["Image"], ao.inputs["Foreground"])
    out = ng.nodes.new("NodeGroupOutput")
    ng.links.new(ao.outputs[0], out.inputs[0])
    return ng


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def bvh_of(objs):
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    verts, polys = [], []
    for ob in objs:
        if ob.type != "MESH":
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        base = len(verts)
        verts.extend(ev.matrix_world @ v.co for v in me.vertices)
        polys.extend([base + i for i in pl.vertices] for pl in me.polygons)
        ev.to_mesh_clear()
    return BVHTree.FromPolygons(verts, polys)


def run_checks(frames, st):
    sc = bpy.data.scenes["Main"]
    print("\n=== CAMERA REPORT ===")
    worst = {"sea surface (above)": (1e9, 0), "surface (under)": (1e9, 0), "wreck": (1e9, 0), "wreck (cut)": (1e9, 0),
             "seabed": (1e9, 0), "goliath": (1e9, 0), "boat": (1e9, 0), "rig": (1e9, 0)}

    def upd(k, d, f):
        if d < worst[k][0]:
            worst[k] = (d, f)
    sc.frame_set(1)
    wb = bvh_of(st["wreck"])
    xb = bvh_of(st["wreck_x"])
    boat = st["boat"]
    sc.frame_set(CUT3B)
    bb = bvh_of([o for o in boat["root"].children_recursive])
    for f in range(1, FRAME_END + 1):
        p = frames[f][0]
        t = t_of(f)
        if under_at(f):
            upd("surface (under)", -p.z - 0.3, f)
        else:
            upd("sea surface (above)", p.z - sea_height(p.x, p.y, t) - 0.3, f)
        if WHIP_IN <= f < CUT1 or CUT3 <= f < CUT3B or CUT3C <= f < CUT4:
            h = wb.find_nearest(p)
            upd("wreck", (h[3] if h[0] is not None else 1e9) - 0.15, f)
            upd("seabed", p.z - SEABED_Z - 0.3, f)
            gp = hook_goliath(f)[0] if f < CUT1 else (strike_goliath(f)[0] if f < CUT3B else turn_goliath(f)[0])
            upd("goliath", (gp - p).length - GOLIATH_SCALE * G_L / 2 - 0.2, f)
        if CUT1 <= f < CUT2:
            h = xb.find_nearest(p)
            upd("wreck (cut)", (h[3] if h[0] is not None else 1e9) - 0.15, f)
            upd("goliath", (xs_goliath(f)[0] - p).length - GOLIATH_SCALE * G_L / 2 - 0.2, f)
        if CUT2 <= f < CUT3:
            upd("rig", (p - RS).length - 0.35, f)
        if CUT3B <= f < CUT3C:
            sc.frame_set(f)
            h = bb.find_nearest(p) if f == CUT3B else bvh_of([o for o in boat["root"].children_recursive]).find_nearest(p) \
                if f % 12 == 0 else (None,) * 4
            if h[0] is not None:
                upd("boat", h[3] - 0.15, f)
    for k, v in worst.items():
        print(f"closest {k}: {round(v[0], 3)} m at frame {v[1]} (negative = inside)")
    acc, turn = (0.0, 0), (0.0, 0)
    whips = set(range(1, WHIP_IN + 7)) | set(range(CUT4 - 4, CUT4 + 8)) | set(range(CUT1 + 8, CUT1 + 51))
    for f in range(3, FRAME_END):
        if any(f - 2 <= c <= f + 1 for c in CUTS) or f in whips:
            continue
        p0, p1, p2 = frames[f - 1][0], frames[f][0], frames[f + 1][0]
        a = ((p2 - p1) - (p1 - p0)).length * FPS * FPS
        if a > acc[0]:
            acc = (a, f)
        ang = math.degrees(frames[f - 1][2].rotation_difference(frames[f][2]).angle)
        ang = min(ang, 360.0 - ang) * FPS
        if ang > turn[0]:
            turn = (ang, f)
    print(f"max camera acceleration: {acc[0]:.2f} m/s^2 at frame {acc[1]};  max turn rate: {turn[0]:.1f} deg/s at frame {turn[1]}"
          " (whip pans and the pull back excluded)")
    cross = [f for f in range(2, FRAME_END + 1) if under_at(f) != under_at(f - 1)]
    print("under/above switches:", cross, "all on cuts:", all(f in CUTS for f in cross))
    jumps = [f for f in range(2, FRAME_END + 1) if (frames[f][0] - frames[f - 1][0]).length > 0.6 and f not in CUTS
             and not (CUT1 + 8 <= f <= CUT1 + 50)]
    print("camera jumps outside cuts:", jumps or "none")
    a, b = frames[1], frames[FRAME_END]
    print("loop: frame 1 vs last frame camera: position", round((a[0] - b[0]).length, 4), "m, rotation",
          round(math.degrees(a[2].rotation_difference(b[2]).angle), 4), "deg, lens", a[3], b[3])
    print("loop: sea at t=0 vs t=end (should match):", round(sea_height(5.0, 7.0, 0.0) - sea_height(5.0, 7.0, LOOP_S), 6))
    hud = st["hud"]
    live1 = [n for n, fn in hud.boxes if fn(1) is not None]
    liveN = [n for n, fn in hud.boxes if fn(FRAME_END) is not None]
    print("loop: overlays on frame 1:", live1, " on the last frame:", liveN)
    overlaps = {}
    for f in range(1, FRAME_END + 1):
        live = [(n, fn(f)) for n, fn in hud.boxes]
        live = [(n, bx) for n, bx in live if bx is not None]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                (na, a_), (nb, b_) = live[i], live[j]
                if a_[0] < b_[2] - 0.004 and b_[0] < a_[2] - 0.004 and a_[1] < b_[3] - 0.004 and b_[1] < a_[3] - 0.004:
                    overlaps.setdefault((na, nb), []).append(f)
    print("overlay overlaps:", "none" if not overlaps else "")
    for (na, nb), fs in overlaps.items():
        print(f"   {na} x {nb}: frames {fs[0]}-{fs[-1]} ({len(fs)} frames)")
    offscreen = {}
    for n, fn in hud.boxes:
        for f in range(1, FRAME_END + 1):
            bx = fn(f)
            if bx and (bx[0] < -HUD_W + 0.01 or bx[2] > HUD_W - 0.01 or bx[1] < -0.49 or bx[3] > 0.49):
                offscreen.setdefault(n, []).append(f)
    print("overlays leaving the frame:", {k: (v[0], v[-1], len(v)) for k, v in offscreen.items()} or "none")


def subject_report(frames, st):
    hud = st["hud"]
    sc = bpy.data.scenes["Main"]
    rig = st["rig"]
    subj = {GAPE_F[0] + 10: ("mouth open (hook)", lambda f: mouth_of(*hook_goliath(f))),
            (CUT1A + CUT1) // 2: ("goliath head (1b)", lambda f: hook_goliath(f)[0] + hook_goliath(f)[1] * 0.9),
            CUT1 + 2: ("goliath (cross section)", lambda f: xs_goliath(f)[0]),
            CUT2 - 20: ("goliath (wide)", lambda f: xs_goliath(f)[0]),
            CUT2 + 30: ("bait", lambda f: rig_point(rig, "bait", f)),
            CUT2B + 20: ("hook eye", lambda f: rig_point(rig, "eye", f)),
            STRIKE_F - 8: ("bait (strike)", lambda f: bait_strike(f)[0]),
            STRIKE_F: ("mouth (strike)", lambda f: mouth_of(*strike_goliath(f))),
            CUT3B + 20: ("reel", lambda f: st["rod"]["reel"]["root"].matrix_world.translation.copy()),
            TURN_F + 10: ("goliath (turn)", lambda f: turn_goliath(f)[0])}
    print("\n=== SUBJECTS ON SCREEN (x in +-0.281, y in +-0.5) ===")
    for f, (nm, fn) in sorted(subj.items()):
        sc.frame_set(f)
        s = hud.project(f, fn(f))
        print(f"frame {f}: {nm} ->", None if s is None else (round(s.x, 3), round(s.y, 3)))


def write_events(path, st):
    data = {
        "fps": FPS, "frames": FRAME_END, "loop": True,
        "cuts": [WHIP_IN, CUT1, CUT2, CUT3, CUT4], "hard_cuts": [CUT1A, CUT2B, CUT3B, CUT3C],
        "whips": [[1, WHIP_IN], [CUT4 - 4, CUT4 + 6]],
        "overlays": sorted(st["hud"].events),
        "under": [[WHIP_IN, CUT3B], [CUT3C, CUT4]], "gape": list(GAPE_F), "strike": STRIKE_F, "lock": LOCK_F,
        "turn": TURN_F, "boat": [CUT3B, CUT3C], "rig": [CUT2, CUT3], "section": [CUT1, CUT2],
        "vo_start": VO_START, "vo_len": VO_LEN, "vo_text": VO_TEXT,
        "beats": [1, CUT1, CUT2, CUT3, CUT4],
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    print("events written", path)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.name = "Main"
    sc.frame_start, sc.frame_end = 1, FRAME_END
    sc.render.fps = FPS
    for prop in ("under", "deep", "clear"):
        sc[prop] = 0.0
    c_sea, c_boat, c_w, c_x = collection("Sea"), collection("Boat"), collection("WreckSite"), collection("CrossSection")
    c_rig, c_fx, c_cam = collection("Rig"), collection("LightAndWater"), collection("Camera")
    atmos = build_atmos_group()
    M = build_materials(atmos)
    _w, w_under, w_deep = build_world(sc)
    sun = build_sun(c_fx)
    key_l = build_fill(c_fx, "UnderwaterKey", size=6.0, color=(226, 240, 220))
    bounce = build_fill(c_fx, "UnderwaterBounce", size=6.0, color=(150, 190, 160))
    cave = build_fill(c_fx, "HoleFill", size=1.5, color=(200, 214, 190))
    # above water
    sea = build_sea(c_sea, M)
    key_sea_center(sea)
    boat = build_boat(c_boat, M)
    boat["root"].scale = (BOAT_SCALE,) * 3
    rod = build_rod(c_boat, M)
    rig_boat_rod(boat, rod, CUT3B - 2, CUT3C + 1)
    bl_a = empty_object("BoatLineEnd", c_boat, size=0.1)
    for f in range(CUT3B - 2, CUT3C + 2):
        bl_a.location = Vector((BOAT_AT.x + 1.0, BOAT_AT.y - 3.2, -0.5))
        key(bl_a, "location", f)
    boat_line = line_object("BoatLine", c_boat, M, rod["ctrl"]["Tip"], bl_a, radius=0.0016, count=24)
    # the wreck site
    wreck = build_wreck(c_w, M, WRECK_AT, name="Wreck")
    sand = build_sand(c_w, M, (WRECK_AT.x, WRECK_AT.y))
    gol = make_goliath(c_w, M, "Goliath")
    animate_goliath(gol, [(WHIP_IN - 2, CUT1 + 2, hook_goliath, hook_gape, 0.01),
                          (CUT3 - 2, CUT3B + 2, strike_goliath, strike_gape, lambda f: 0.012 + 0.03 * smoothstep(STRIKE_F - 10, STRIKE_F, f)),
                          (CUT3C - 2, CUT4 + 2, turn_goliath, lambda f: 0.15 * smoothstep(TURN_F - 4, TURN_F + 4, f) *
                           (1 - smoothstep(TURN_F + 10, TURN_F + 24, f)), 0.035)], rate_hz=0.7)
    bait = bpy.data.objects.new("StrikeBait", jack_mesh(M, L=0.5))
    c_w.objects.link(bait)
    for f in range(CUT3 - 2, CUT3B + 2):
        if f <= STRIKE_F - 1:
            p, fwd = bait_strike(f)
        else:                                                   # inhaled: shrinks into the mouth
            gp, gf = strike_goliath(f)
            p, fwd = mouth_of(gp, gf) - gf * 0.3, gf
        place(bait, f, p, fwd)
        bait.scale = (1.0,) * 3 if f < STRIKE_F else (0.0,) * 3
        key(bait, "scale", f)
    strike_line = build_strike_line(c_w, M, bait, CUT3 - 2, CUT3B + 1)
    turn_line = build_turn_line(c_w, M, CUT3C - 2, CUT4 + 1)
    suck = suction(lambda f: mouth_of(*strike_goliath(f)), STRIKE_F, 8).build("Suction", c_w, M.silt)
    silt_w = drift_silt(WRECK_AT + Vector((HOLE.x, HOLE.y - 3.0, HOLE.z - SEABED_Z)), (9.0, 6.0, 3.0), 2600, 3,
                        WHIP_IN, CUT4).build("DriftSilt", c_w, M.silt)
    hole_c = WRECK_AT + Vector((HOLE.x, HOLE.y - 4.0, -SEABED_Z))
    surf1 = build_surface_under(c_w, M, center=(hole_c.x, hole_c.y, 0.0), size=200.0)
    shafts1 = build_shafts(c_w, M, (hole_c.x, hole_c.y - 2, 0.0), n=8, seed=5, spread=10.0, length=24.0)
    plank1 = specks("Sediment1", c_w, M.specks, (hole_c.x, hole_c.y, SEABED_Z + 5.0), (14.0, 10.0, 6.0), 22000, 0.0028, 31)
    # the cross section
    wreck_x = build_wreck(c_x, M, XS + Vector((0.0, 0.0, SEABED_Z)), cut=True, name="WreckX")
    sand_x = build_sand(c_x, M, (XS.x, XS.y), name="SeabedX")
    gx = make_goliath(c_x, M, "GoliathX")
    animate_goliath(gx, [(CUT1 - 2, CUT2 + 2, xs_goliath, None, 0.008)], rate_hz=0.5)
    bf = bpy.data.objects.new("SchoolBait", baitfish_mesh(M, L=0.16))
    c_x.objects.link(bf)
    bf.hide_render = bf.hide_viewport = True
    school_x = build_school(c_x, bf, XS + Vector((-1.0, -3.0, SEABED_Z + 8.6)), 120, 0.6, 2.2, 1.0, 14, "BaitSchoolX")
    bm = bmesh.new()
    rng = random.Random(9)
    for _ in range(14):                                         # crabs on the sand by the wreck
        c = XS + Vector((rng.uniform(-1.0, 2.6), rng.uniform(-6.2, -4.4), SEABED_Z + 0.06))
        ellipsoid(bm, c, (0.09, 0.07, 0.035), subdiv=2)
        for s in (1, -1):
            for k in range(3):
                a0 = c + Vector((0.03 - 0.03 * k, s * 0.06, 0.0))
                bm_frustum(bm, tuple(a0), tuple(a0 + Vector((0.02, s * 0.07, -0.04))), 0.008, 0.005, segs=5)
    crabs = finish("CrabsX", bm, c_x, [M.crab])
    surf2 = build_surface_under(c_x, M, center=(XS.x, XS.y, 0.0), size=260.0)
    shafts2 = build_shafts(c_x, M, (XS.x, XS.y - 4, 0.0), n=8, seed=7, spread=18.0, length=24.0)
    plank2 = specks("Sediment2", c_x, M.specks, (XS.x, XS.y - 6.0, SEABED_Z + 6.0), (24.0, 10.0, 7.0), 9000, 0.012, 32)
    # the rig studio
    rig = build_rig(c_rig, M)
    for f in range(CUT2 - 2, CUT3 + 2):
        rig["pivot"].rotation_euler = (0.0, 0.0, rig_rot(f))
        key(rig["pivot"], "rotation_euler", f)
    silt_r = drift_silt(RS + Vector((0.0, -0.4, 0.15)), (0.8, 0.5, 0.5), 300, 4, CUT2, CUT3).build("RigSilt", c_rig, M.silt)
    st = dict(boat=boat, rod=rod, rig=rig, wreck=wreck, wreck_x=wreck_x)
    frames, fstop = camera_frames(st)
    build_camera(c_cam, frames, fstop)
    key_scene_state(sc, (w_under, w_deep), sun, key_l, bounce, cave)
    # visibility: one set of spans per object (keys from several calls would add up)
    vis = {}

    def show(objs, spans):
        for o in objs:
            vis[o.name] = (o, tuple(spans))
    above = [(1, WHIP_IN - 1), (CUT3B, CUT3C - 1), (CUT4, FRAME_END)]
    show([sea], above)
    show([o for o in c_boat.all_objects if o.type in ("MESH", "CURVE") and o.name != "RollerGuide"], [(CUT3B, CUT3C - 1)])
    show([o for o in c_w.all_objects if o.type in ("MESH", "CURVE") and o.name not in ("WreckBarnacleShape",)],
         [(WHIP_IN, CUT1 - 1), (CUT3, CUT3B - 1), (CUT3C, CUT4 - 1)])
    show([bait, strike_line, suck], [(CUT3, CUT3B - 1)])
    show([turn_line], [(CUT3C, CUT4 - 1)])
    show([o for o in c_x.all_objects if o.type in ("MESH", "CURVE") and o.name not in ("WreckXBarnacleShape", "SchoolBait")],
         [(CUT1, CUT2 - 1)])
    show([o for o in c_rig.all_objects if o.type in ("MESH", "CURVE")], [(CUT2, CUT3 - 1)])
    groups = {}
    for o, spans in vis.values():
        groups.setdefault(spans, []).append(o)
    for spans, objs in groups.items():
        key_visibility(objs, list(spans))
    _ = (sand, silt_w, surf1, shafts1, plank1, sand_x, school_x, crabs, surf2, shafts2, plank2, silt_r, boat_line)
    hs, hcam, hcol = build_hud_scene()
    hud_materials(M)
    hud = Hud(hcam, hcol, M, frames)
    st.update(frames=frames, hud=hud, goliath=gol)
    build_overlays(hud, st)
    n = hide_when_invisible([o for o in bpy.data.objects["HUD"].children_recursive if o.type in ("MESH", "FONT")])
    print("overlay objects switched off while invisible:", n)
    build_compositor(sc, hs)
    if bpy.context.window is not None:
        bpy.context.window.scene = sc
    sc.frame_set(1)
    return st


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--save", default=None)
    p.add_argument("--check", action="store_true")
    p.add_argument("--frames", default="")
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
    sc = bpy.data.scenes["Main"]
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
            for s in bpy.data.scenes:
                s.frame_current = f
            sc.frame_set(f)
            sc.render.filepath = os.path.join(os.path.abspath(args.outdir), f"frame_{f:04d}.png")
            bpy.ops.render.render(write_still=True, scene=sc.name)


if __name__ == "__main__":
    main()
