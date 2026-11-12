"""Record musky masterclass: the IGFA all tackle record muskellunge and how anglers fish giant muskies on Wisconsin
weed edges. Vertical 9:16, 24 fps, about 43 s (the frame count follows the measured voiceover), Blender 5.2 EEVEE.
Built to loop: the last frame is the first frame's shot, and the voiceover's last line runs into its first.

Run with the Blender binary:
    blender --background --python record_musky_masterclass.py -- [options]
or with the bpy Python module (pip install bpy):
    python record_musky_masterclass.py [options]

Options:
    --save PATH        save the generated scene as a .blend file (fonts packed)
    --check            camera clearances (surface, lake bed, weeds, fish, boat, lure studio), camera acceleration and
                       turn rate, cuts and water surface crossings, overlay overlaps and off-screen boxes, subjects on
                       screen, loop match
    --frames 1,300     render these frames as PNG stills into --outdir
    --outdir DIR       output directory for renders
    --scale PCT        resolution percentage (default 100 -> 1080x1920)
    --samples N        EEVEE render samples (default 32)
    --no-blur          switch motion blur off (quick tests)
    --events PATH      write overlay, action and voiceover timing as JSON (input for record_musky_audio.py)

Everything is procedural (no imported assets). Facts in the on-screen text and the voiceover are listed with their
sources in SOURCES.md; the fish, the lake, the depths, the lures, the boat and the strikes are illustrative.

World (metres, +Z up, lake surface z = 0). A clear, weedy northern Wisconsin lake: a shallow flat of cabbage
(broadleaf pondweed) with lily pads, its outer edge dropping into deeper water. A big musky waits at the weed edge.
Overlays live in a second scene ("HUD") with its own static camera, composited over the 3D render.

Beats follow the voiceover clips (voiceover/vo1..vo5). Cuts are hard cuts, match cuts and whip pans (no wipes):
    1  hook: a bucktail ripping over the lake from above whips down under water; tracking the bucktail over the
       weed tops, a musky bursts out of the cabbage and T-bones it; a second angle on the fish with the lure
    2  structure: a match cut on the fish to a cross section of the lake: the cabbage flat, its edge and the drop off
    3  the lure: a close turntable of a double blade bucktail; then the bucktail burned over the weed tops
    4  the figure 8: from low under water beside the boat, the rod tip sweeps the lure through a wide, deep figure 8
       with a musky following, and the musky eats on the turn
    5  outro: a whip pan back to the bucktail ripping over the lake from above, which is frame 1
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

# water: a clear, slightly green northern lake (fog colour by view direction, down .. up)
UW_RAMP = [(0.0, srgb(6, 18, 16)), (0.36, srgb(16, 46, 40)), (0.52, srgb(36, 84, 70)), (0.72, srgb(74, 132, 108)),
           (1.0, srgb(150, 196, 160))]
VIS = 11.0                           # fog e-folding distance under water (m): clear lake water
AIR_HAZE = 900.0                     # haze e-folding distance above water (m)
SUN_DIR = Vector((-0.4, 0.5, 0.8)).normalized()        # a high summer sun


FLAT_Z = -2.4                        # the cabbage flat (illustrative depth)
DEEP_Z = -6.5                        # the deeper water past the drop off (illustrative)
SEABED_Z = FLAT_Z                    # the name shared helpers use for the bottom near the action
WEED_TOP = -0.55                     # cabbage tops (just under the surface on a summer flat)


def t_of(f):
    """Scene time in seconds at frame f, zero at frame 1, so frame 1321 would equal frame 1 (the loop)."""
    return (f - 1) / FPS


# --------------------------------------------------------------------------
# Voiceover timing. vo_timing.json (written by record_musky_audio.py --measure) holds the measured lengths and
# sentence starts. The hook starts on frame 1 and the outro ends on the last frame, so the loop joins mid sentence:
# "...and you'll understand why... In 1949, Cal Johnson pulled..."
# --------------------------------------------------------------------------
VO_LEN = {1: 12.20, 2: 7.0, 3: 8.5, 4: 10.0, 5: 3.2}
VO_TEXT = {
    1: "In 1949, Cal Johnson pulled a 67 pound, 8 ounce musky out of a Wisconsin lake. The IGFA lists it as the all "
       "tackle world record for the fish of ten thousand casts.",
    2: "Big muskies hunt by sight from cover. They lurk along cabbage weed edges, right next to deep water drop offs.",
    3: "To trigger a reaction strike, anglers burn double blade bucktails fast, just inches over the weed tops, where "
       "muskies hide below.",
    4: "Never pull your lure out early. Some pros say up to forty percent of their muskies hit right at the boat, so "
       "finish every cast with a wide, deep figure eight.",
    5: "Burn fast, work the turn, and you'll understand why...",
}
VO_GAPS = (10, 10, 10)               # frames of breath after clips 1, 2 and 3 (fast paced: no dead air)
OUTRO_GAP = 10                       # breath before the outro line


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
WHIP_IN = 7                          # frames 1..6: the bucktail ripping over the lake from above (the loop's join)
CUT1A = P(1, 1) - 4                              # hard cut inside the hook: a second angle on the fish with the lure
CUT1 = (vo_end(1) + VO_START[2]) // 2            # match cut: the musky -> the musky in the cross section
CUT2 = (vo_end(2) + VO_START[3]) // 2            # jump cut to the lure breakdown
CUT2B = VO_START[3] + int(round(2.2 * FPS))      # hard cut: the turntable -> the bucktail burned over the weeds
CUT3 = (vo_end(3) + VO_START[4]) // 2            # hard cut to the figure 8 beside the boat
CUT4 = VO_START[5] - 6                           # whip pan back to the lake from above (the loop's join)
CUTS = [WHIP_IN, CUT1A, CUT1, CUT2, CUT2B, CUT3, CUT4]
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
    """Under water: the fog colour itself (infinitely far). Above water: a clear summer sky with scattered cumulus,
    with a soft shoulder on camera rays (the Standard view would clip the bright sky to white). Switched by the
    "under" value."""
    w = bpy.data.worlds.new("LakeWorld")
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
# The lake surface: one grid, fine where the camera works and coarse to the horizon, displaced by a light wind chop
# whose frequencies are whole multiples of the loop rate (2 pi / LOOP_S), so the last frame runs into frame 1.
# --------------------------------------------------------------------------
G = 9.81
SWELLS = []                                      # (amplitude, kx, ky, omega, phase, steepness)
for _n, _amp, _deg, _q in ((29, 0.045, 8.0, 0.45), (36, 0.032, 33.0, 0.45), (45, 0.022, -24.0, 0.4), (57, 0.014, 61.0, 0.4),
                           (72, 0.008, -66.0, 0.35), (90, 0.005, 14.0, 0.3)):   # a light summer wind chop on a lake
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


    # the lure the overhead shot follows: the lake material draws its wake in this empty's space (+X = travel)
    wake_ref = bpy.data.objects.new("WakeRef", None)
    M.wake_ref = wake_ref

    def sea(nt):
        """A clear northern lake seen from above: dark tea green water with sky reflections from the Fresnel term,
        a light chop, the cabbage beds showing through as darker patches (painted: the water is opaque from above),
        and the bucktail's bulge and V wake (drawn in the WakeRef empty's space). World position, not "rest"."""
        wp = separate(nt, world_pos(nt))
        p = combine(nt, wp["X"], wp["Y"], 0.0)
        crest = attr(nt, "crest", "Fac")
        rip = looped_noise(nt, p, (0.5, 0.2, 0.0), 2.2, 6.0, 0.6)
        rip2 = looped_noise(nt, p, (-0.3, 0.45, 0.0), 9.0, 4.0, 0.55)
        weeds = map_range(nt, noise(nt, vmath(nt, "MULTIPLY", p, (0.25, 0.25, 1.0)), 1.0, 4.0, 0.6), 0.42, 0.62)
        weeds = nmath(nt, "MULTIPLY", weeds, map_range(nt, wp["Y"], -3.0, 1.5))          # only on the flat (y > 0)
        tcn = nt.nodes.new("ShaderNodeTexCoord")
        tcn.object = wake_ref                                                           # the lure's frame (+X ahead)
        tc = separate(nt, tcn.outputs["Object"])
        behind = nmath(nt, "MULTIPLY", tc["X"], -1.0)
        across = nmath(nt, "ABSOLUTE", tc["Y"])
        wob = nmath(nt, "MULTIPLY", nmath(nt, "SINE", nmath(nt, "MULTIPLY", behind, 9.0)), 0.015)
        arm = nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", nmath(nt, "ADD", across, wob), nmath(nt, "MULTIPLY_ADD", behind, 0.26, 0.05)))
        vline = nmath(nt, "MULTIPLY", map_range(nt, arm, 0.035, 0.0), map_range(nt, behind, 0.0, 0.25))
        vline = nmath(nt, "MULTIPLY", vline, nmath(nt, "MULTIPLY", map_range(nt, behind, 2.6, 0.3), 0.55))
        trail = nmath(nt, "MULTIPLY", map_range(nt, across, 0.16, 0.02), map_range(nt, behind, -0.05, 0.1))
        trail = nmath(nt, "MULTIPLY", trail, nmath(nt, "MULTIPLY", map_range(nt, behind, 1.8, 0.1), 0.6))
        trail = nmath(nt, "MULTIPLY", trail, map_range(nt, noise(nt, vmath(nt, "MULTIPLY", tex_coord(nt, "Object"), (6.0, 6.0, 1.0)), 1.0, 4.0), 0.35, 0.65))
        bulge = map_range(nt, vmath(nt, "LENGTH", combine(nt, nmath(nt, "MULTIPLY", tc["X"], 1.4), tc["Y"], 0.0)), 0.16, 0.0)
        foam = nmath(nt, "MINIMUM", nmath(nt, "ADD", nmath(nt, "ADD", vline, trail), bulge), 1.0)
        foam = nmath(nt, "MINIMUM", nmath(nt, "ADD", foam, nmath(nt, "MULTIPLY", map_range(nt, crest, 0.7, 0.95), 0.15)), 1.0)
        water = color_ramp(nt, map_range(nt, crest, -0.4, 0.6), [(0.0, srgb(22, 46, 34)), (0.7, srgb(36, 70, 50)),
                                                                   (1.0, srgb(70, 104, 76))])
        water = mix_rgb(nt, nmath(nt, "MULTIPLY", weeds, 0.7), water, srgb(10, 26, 14))       # the cabbage beds below
        col = mix_rgb(nt, foam, water, srgb(220, 232, 226))
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        _feed(nt, b.inputs["Roughness"], map_range(nt, foam, 0.0, 1.0, 0.06, 0.5, smooth=False))
        b.inputs["IOR"].default_value = 1.333
        b.inputs["Specular IOR Level"].default_value = 0.6
        h = nmath(nt, "ADD", nmath(nt, "ADD", rip, nmath(nt, "MULTIPLY", rip2, 0.35)), nmath(nt, "MULTIPLY", foam, 0.4))
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = 0.25
        bp.inputs["Distance"].default_value = 0.05
        nt.links.new(h, bp.inputs["Height"])
        nt.links.new(bp.outputs[0], b.inputs["Normal"])
        return b.outputs[0]
    M.sea = mat("Sea", sea)

    def surface_under(nt):
        """The surface from below: a bright rippling window overhead, darker total-reflection rim."""
        p = tex_coord(nt, "Object")
        rip = looped_noise(nt, p, (0.4, 0.2, 0.0), 0.8, 5.0, 0.6)
        inc = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Incoming"])["Z"]
        window = map_range(nt, nmath(nt, "ABSOLUTE", inc), 0.62, 0.8)
        col = mix_rgb(nt, window, srgb(14, 40, 32), srgb(176, 214, 180))
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
        along = nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", zw, 1.0 / 5.0))
        along = nmath(nt, "MULTIPLY", along, map_range(nt, zw, -0.2, -1.5))
        dist = nt.nodes.new("ShaderNodeCameraData").outputs["View Distance"]
        along = nmath(nt, "MULTIPLY", along, map_range(nt, dist, 0.5, 3.0))
        along = nmath(nt, "MULTIPLY", along, 0.2)
        flick = map_range(nt, looped_noise(nt, tex_coord(nt, "Object"), (0.0, 0.0, 0.25), 1.2), 0.3, 0.7, 0.55, 1.0)
        k = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", core, along), flick)
        k = nmath(nt, "MULTIPLY", k, nmath(nt, "MULTIPLY", scene_attr(nt, "under"),
                                           nmath(nt, "MULTIPLY_ADD", scene_attr(nt, "deep"), -0.9, 1.0)))
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(220, 240, 200), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", k, oi.outputs["Alpha"]), em.inputs["Strength"])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(tr.outputs[0], add.inputs[1])
        return add.outputs[0]
    M.shaft = new_material("SunShaft", shaft, blended=True)

    def plankton(nt):
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(200, 214, 170), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", attr(nt, "rand", "Fac"), 0.35), em.inputs["Strength"])
        df = nt.nodes.new("ShaderNodeBsdfDiffuse")
        df.inputs["Color"].default_value = (*srgb(140, 150, 110), 1.0)
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(df.outputs[0], add.inputs[1])
        return add.outputs[0]
    M.specks = mat("Plankton", plankton)
    M.silt = mat("Silt", lambda nt: psdf(nt, srgb(150, 146, 110), rough=0.9, alpha=0.6,
                                         emission=srgb(110, 116, 86), emission_strength=0.15))
    M.bubbles = mat("Bubbles", lambda nt: psdf(nt, srgb(230, 240, 236), rough=0.1, alpha=0.6,
                                               emission=srgb(200, 222, 214), emission_strength=0.5))

    # --- the musky -------------------------------------------------------------------------------------------------
    def musky(nt):
        """Muskellunge: light olive brown back, silvery olive to tan flanks with dark vertical bars and blotches, a
        cream belly; a slime sheen, some subsurface scattering in the skin."""
        z, x = attr(nt, "zrel", "Fac"), attr(nt, "xrel", "Fac")
        p = tex_coord(nt, "Object")
        o = separate(nt, p)
        base = color_ramp(nt, z, [(0.0, srgb(222, 216, 194)), (0.25, srgb(206, 198, 166)), (0.45, srgb(168, 162, 118)),
                                  (0.7, srgb(118, 116, 76)), (1.0, srgb(66, 66, 44))])
        n1 = noise(nt, vmath(nt, "MULTIPLY", p, (9.0, 9.0, 9.0)), 1.0, 5.0, 0.6)
        mott = map_range(nt, noise(nt, p, 30.0, 6.0, 0.6), 0.35, 0.7)
        base = mix_rgb(nt, nmath(nt, "MULTIPLY", mott, 0.25), base, srgb(190, 186, 150))
        bars = map_range(nt, nmath(nt, "SINE", nmath(nt, "ADD", nmath(nt, "MULTIPLY", o["X"], 82.0),
                                                     nmath(nt, "MULTIPLY", n1, 3.0))), 0.35, 0.9)
        blot = map_range(nt, voronoi(nt, vmath(nt, "MULTIPLY", p, (1.0, 2.5, 1.6)), 34.0), 0.28, 0.12)
        marks = nmath(nt, "MAXIMUM", nmath(nt, "MULTIPLY", bars, 0.85), nmath(nt, "MULTIPLY", blot, 0.6))
        marks = nmath(nt, "MULTIPLY", marks, nmath(nt, "MULTIPLY", map_range(nt, z, 0.22, 0.45), map_range(nt, x, 0.12, 0.25)))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", marks, 0.75), base, srgb(54, 52, 34))
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(uw_base(nt, col, caustic=0.7), b.inputs["Base Color"])
        _feed(nt, b.inputs["Roughness"], map_range(nt, n1, 0.0, 1.0, 0.3, 0.45, smooth=False))
        b.inputs["Coat Weight"].default_value = 0.25
        b.inputs["Coat Roughness"].default_value = 0.12
        b.inputs["Subsurface Weight"].default_value = 0.2
        b.inputs["Subsurface Scale"].default_value = 0.01
        b.inputs["Subsurface Radius"].default_value = (1.0, 0.6, 0.4)
        scales = voronoi(nt, vmath(nt, "MULTIPLY", p, (1.0, 1.6, 1.6)), 260.0, out="Distance", feature="DISTANCE_TO_EDGE")
        nt.links.new(bump(nt, map_range(nt, scales, 0.0, 0.08), 0.12, 0.002), b.inputs["Normal"])
        return b.outputs[0]
    M.musky = mat("MuskySkin", musky)

    def musky_fin(nt):
        g = separate(nt, tex_coord(nt, "Generated"))
        p = tex_coord(nt, "Object")
        edge = map_range(nt, nmath(nt, "MAXIMUM", nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", g["Z"], 0.5)),
                                   nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", g["X"], 0.5))), 0.25, 0.5)
        col = mix_rgb(nt, edge, srgb(120, 84, 54), srgb(150, 80, 50))
        spots = map_range(nt, voronoi(nt, p, 70.0), 0.15, 0.07)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", spots, 0.85), col, srgb(36, 26, 18))
        o = separate(nt, p)
        ray = nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY", o["X"], 90.0))), 8.0)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", ray, 0.2), col, srgb(60, 34, 20))
        return psdf(nt, uw_base(nt, col, caustic=0.4), rough=0.45, sss=0.3, sss_scale=0.02, alpha=0.95)
    M.musky_fin = mat("MuskyFin", musky_fin)
    M.throat = mat("MuskyThroat", lambda nt: psdf(nt, color_ramp(nt, attr(nt, "zrel", "Fac"),
                   [(0.0, srgb(40, 14, 14)), (1.0, srgb(120, 54, 48))]), rough=0.5, sss=0.3, sss_scale=0.02, coat=0.2))
    M.teeth = mat("Teeth", lambda nt: psdf(nt, srgb(236, 230, 214), rough=0.25, coat=0.6, coat_rough=0.05, sss=0.2,
                                           sss_scale=0.003))

    def eye(nt):
        """Musky eye: a dark pupil in a gold to olive iris under a glossy cornea."""
        r = attr(nt, "eyer", "Fac")
        col = color_ramp(nt, r, [(0.0, srgb(2, 2, 3)), (0.42, srgb(2, 2, 3)), (0.47, srgb(206, 170, 80)),
                                 (0.66, srgb(160, 130, 60)), (0.82, srgb(90, 80, 44)), (1.0, srgb(90, 84, 60))])
        return psdf(nt, col, rough=0.04, metal=0.1, coat=1.0, coat_rough=0.02, spec=0.8)
    M.eye = mat("FishEye", eye)

    def small_eye(nt):
        r = attr(nt, "eyer", "Fac")
        col = color_ramp(nt, r, [(0.0, srgb(2, 2, 3)), (0.62, srgb(2, 2, 3)), (0.66, srgb(190, 150, 60)),
                                 (0.75, srgb(150, 120, 60)), (1.0, srgb(90, 90, 70))])
        return psdf(nt, col, rough=0.05, coat=1.0, coat_rough=0.02, spec=0.8)
    M.small_eye = mat("SmallFishEye", small_eye)

    def perch(nt):
        """Yellow perch: golden yellow to green flanks with six or seven dark saddle bars, a white belly."""
        z, x = attr(nt, "zrel", "Fac"), attr(nt, "xrel", "Fac")
        base = color_ramp(nt, z, [(0.0, srgb(232, 228, 206)), (0.3, srgb(226, 204, 90)), (0.6, srgb(178, 168, 60)),
                                  (1.0, srgb(70, 80, 40))])
        bars = map_range(nt, nmath(nt, "SINE", nmath(nt, "MULTIPLY", x, 2 * math.pi * 6.5)), 0.2, 0.75)
        bars = nmath(nt, "MULTIPLY", bars, nmath(nt, "MULTIPLY", map_range(nt, z, 0.3, 0.6), map_range(nt, x, 0.12, 0.25)))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", bars, 0.75), base, srgb(40, 46, 22))
        return psdf(nt, uw_base(nt, col, caustic=0.4), rough=0.3, coat=0.3)
    M.perch = mat("YellowPerch", perch)
    M.perch_fin = mat("PerchFin", lambda nt: psdf(nt, uw_base(nt, srgb(232, 120, 40), caustic=0.2), rough=0.4, alpha=0.9))

    # --- the lake: bed, cabbage, lily pads --------------------------------------------------------------------------
    def lakebed(nt):
        """Sand and gravel on the flat, soft dark muck deeper down, with fallen leaves of weed."""
        p = world_pos(nt)
        z = separate(nt, p)["Z"]
        n = noise(nt, p, 1.4, 6.0, 0.6)
        n2 = noise(nt, p, 9.0, 4.0, 0.6)
        sand = color_ramp(nt, n, [(0.0, srgb(104, 96, 66)), (1.0, srgb(150, 136, 98))])
        muck = color_ramp(nt, n, [(0.0, srgb(34, 32, 22)), (1.0, srgb(62, 56, 36))])
        col = mix_rgb(nt, map_range(nt, z, FLAT_Z - 0.6, FLAT_Z - 2.4), sand, muck)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", map_range(nt, n2, 0.62, 0.7), 0.6), col, srgb(48, 70, 30))
        return psdf(nt, uw_base(nt, col, caustic=1.2), rough=0.92, normal=bump(nt, n2, 0.5, 0.02))
    M.lakebed = mat("LakeBed", lakebed)

    def cabbage(nt):
        """Cabbage (broadleaf pondweed): thin, wavy, translucent leaves, bright green to olive, the light through
        them; per plant colour from the "rand" attribute."""
        r = attr(nt, "rand", "Fac")
        col = color_ramp(nt, r, [(0.0, srgb(56, 104, 30)), (0.5, srgb(84, 128, 36)), (1.0, srgb(112, 132, 44))])
        o = tex_coord(nt, "Object")
        vein = nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY", separate(nt, o)["Z"], 140.0))), 12.0)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", vein, 0.15), col, srgb(150, 170, 80))
        return psdf(nt, uw_base(nt, col, caustic=0.6), rough=0.45, sss=0.4, sss_scale=0.01, transmission=0.15)
    M.cabbage = mat("Cabbage", cabbage)
    M.stalk = mat("CabbageStalk", lambda nt: psdf(nt, uw_base(nt, srgb(110, 120, 60), caustic=0.4), rough=0.5, sss=0.3,
                                                  sss_scale=0.01))
    M.lilypad = mat("LilyPad", lambda nt: psdf(nt, uw_base(nt, color_ramp(nt, noise(nt, tex_coord(nt, "Object"), 8.0),
                    [(0.0, srgb(44, 84, 30)), (0.85, srgb(70, 104, 38)), (1.0, srgb(110, 70, 40))]), caustic=0.0), rough=0.5, sss=0.3, sss_scale=0.01))

    def cutface(nt):
        """The cut face of the cross section, drawn flat: lake bottom soil with a fine diagonal hatch (a diagram)."""
        o = separate(nt, world_pos(nt))
        hatch = nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY",
                      nmath(nt, "ADD", o["Y"], o["Z"]), 9.0))), 30.0)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", hatch, 0.35), srgb(92, 70, 40), srgb(200, 160, 96))
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(col, em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0
        return em.outputs[0]
    M.cutface = new_material("CutFace", cutface)

    def studio_back(nt):
        """The lure breakdown's backdrop: a soft green gradient (a studio sweep under water)."""
        z = separate(nt, tex_coord(nt, "Generated"))["Z"]
        col = color_ramp(nt, z, [(0.0, srgb(6, 16, 12)), (0.6, srgb(22, 50, 36)), (1.0, srgb(52, 96, 70))])
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(col, em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0
        return em.outputs[0]
    M.studio = new_material("StudioBackdrop", studio_back)

    # --- the bucktail, line, rod and boat ---------------------------------------------------------------------------
    M.blade = mat("BladeSilver", lambda nt: psdf(nt, srgb(232, 234, 236), rough=0.08, metal=1.0,
                  normal=bump(nt, noise(nt, tex_coord(nt, "Object"), 400.0), 0.05)))
    M.blade_gold = mat("BladeGold", lambda nt: psdf(nt, srgb(236, 190, 90), rough=0.1, metal=1.0))
    M.hair = mat("BucktailHair", lambda nt: psdf(nt, color_ramp(nt, attr(nt, "rand", "Fac"),
                 [(0.0, srgb(14, 12, 12)), (0.8, srgb(30, 26, 24)), (1.0, srgb(200, 110, 30))]), rough=0.55, sss=0.2,
                 sss_scale=0.003))
    M.bead = mat("Bead", lambda nt: psdf(nt, srgb(220, 60, 30), rough=0.15, coat=1.0, coat_rough=0.02))
    M.wire = mat("Wire", lambda nt: psdf(nt, srgb(200, 202, 206), rough=0.15, metal=1.0))
    M.hook = mat("Hook", lambda nt: psdf(nt, srgb(58, 56, 54), rough=0.3, metal=1.0, coat=0.3))
    M.line = mat("Line", lambda nt: psdf(nt, srgb(214, 220, 140), rough=0.3, emission=srgb(214, 220, 140), emission_strength=0.25))
    M.blank = mat("RodBlank", lambda nt: psdf(nt, srgb(26, 28, 30), rough=0.25, coat=0.8, coat_rough=0.05))
    M.guide = mat("RodGuide", lambda nt: psdf(nt, srgb(190, 194, 200), rough=0.12, metal=1.0))
    M.alu = mat("BoatAluminium", lambda nt: psdf(nt, uw_base(nt, color_ramp(nt, noise(nt, tex_coord(nt, "Object"), 3.0),
                [(0.0, srgb(130, 136, 132)), (1.0, srgb(170, 174, 170))]), caustic=0.8), rough=0.4, metal=0.85,
                normal=bump(nt, noise(nt, tex_coord(nt, "Object"), 60.0), 0.1)))
    M.paint = mat("BoatPaint", lambda nt: psdf(nt, uw_base(nt, srgb(30, 60, 44), caustic=0.6), rough=0.45))
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


# --------------------------------------------------------------------------
# The muskellunge (Esox masquinongy). After the species' descriptions: a long, cylindrical body; a long, flat,
# duck bill snout with a huge mouth reaching back under the eye and rows of long, sharp teeth; the single dorsal and
# the anal fin set far back near the tail; a forked tail with pointed lobes; small pectoral and pelvic fins. Built at
# a nominal 1.0 m nose to tail base, then scaled (MUSKY_SCALE) to about 1.4 m with the tail: illustrative, not the
# record fish to scale.
# --------------------------------------------------------------------------
M_L = 1.0                             # modelled length nose to tail base before scaling
MUSKY_SCALE = 1.2                     # 1.0 x 1.2 = 1.2 m to the tail base (the tail adds about 0.2 m)
M_JAW_HINGE = Vector((0.345, 0.0, -0.022))


def musky_curves():
    L = M_L

    def u_of(x):
        return (L / 2 - x) / L
    top_u = pchip([(0.0, 0.005), (0.03, 0.011), (0.07, 0.019), (0.12, 0.031), (0.18, 0.046), (0.26, 0.058), (0.38, 0.068),
                   (0.5, 0.071), (0.62, 0.066), (0.74, 0.054), (0.86, 0.041), (0.95, 0.034), (1.0, 0.032)])
    bot_u = pchip([(0.0, -0.007), (0.03, -0.012), (0.07, -0.018), (0.12, -0.026), (0.18, -0.038), (0.26, -0.052),
                   (0.38, -0.066), (0.5, -0.07), (0.62, -0.064), (0.74, -0.05), (0.86, -0.038), (0.95, -0.032), (1.0, -0.03)])
    wid_u = pchip([(0.0, 0.010), (0.03, 0.019), (0.07, 0.026), (0.12, 0.033), (0.2, 0.042), (0.32, 0.05), (0.45, 0.052),
                   (0.6, 0.047), (0.75, 0.036), (0.9, 0.022), (1.0, 0.017)])
    return (lambda x: top_u(u_of(x))), (lambda x: bot_u(u_of(x))), (lambda x: wid_u(u_of(x))), u_of


def musky_swim_group(ctrl, name):
    """Body wave driven by a controller empty: x = phase (rad), y = amplitude (m), z = body bend (turning). A pike's
    S curve: the head steady, the wave growing down the long body."""
    ng, gi, go = new_group(name, "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = ctrl
    c = separate(ng, oi.outputs["Location"])
    x, _y, _z = gn_xyz(ng)
    back = nmath(ng, "MAXIMUM", nmath(ng, "SUBTRACT", 0.25, x), 0.0)
    w = nmath(ng, "MINIMUM", nmath(ng, "POWER", nmath(ng, "DIVIDE", back, 0.75), 1.5), 1.6)
    arg = nmath(ng, "SUBTRACT", c["X"], nmath(ng, "MULTIPLY", x, 6.0))
    dy = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", nmath(ng, "SINE", arg), w), c["Y"])
    dy = nmath(ng, "MULTIPLY_ADD", nmath(ng, "MULTIPLY", back, back), nmath(ng, "MULTIPLY", c["Z"], 2.0), dy)
    sp = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, 0.0, dy, 0.0), sp.inputs["Offset"])
    ng.links.new(sp.outputs[0], go.inputs["Geometry"])
    return ng


def teeth_row(bm, pts, h, r, mat, down):
    """A row of conical teeth at the given points, pointing up (lower jaw) or down (upper jaw)."""
    for i, p in enumerate(pts):
        hh = h * (0.6 + 0.4 * abs(math.sin(i * 2.3)))
        tip = p + Vector((0.0, 0.0, -hh if down else hh))
        bm_frustum(bm, tuple(p), tuple(tip), r, r * 0.08, segs=5, cap0=True, cap1=False)
    for f in bm.faces:
        f.material_index = mat


def make_musky(col, M, name="Musky"):
    """The musky: body (with eyes and upper teeth), fins, a hinged lower jaw with its teeth, a dark throat."""
    L = M_L
    top, bot, wid, u_of = musky_curves()
    bm = bmesh.new()
    ring_section(bm, L / 2, -L / 2, top, bot, wid, nr=90, na=40, power=2.1, xr=lambda x: 1.0 - u_of(x))
    mouth_line = lambda x: bot(x) + 0.35 * (top(x) - bot(x))         # noqa: E731  (where the jaws meet)
    for v in bm.verts:                                   # flatten the snout into a duck bill
        u = u_of(v.co.x)
        if u < 0.14:
            k = (1.0 - u / 0.14) ** 1.5
            v.co.z = mouth_line(v.co.x) + (v.co.z - mouth_line(v.co.x)) * (1.0 - 0.35 * k)
            v.co.y *= 1.0 + 0.25 * k
    hx0 = M_JAW_HINGE.x
    for v in bm.verts:                                   # the lower jaw is its own object: the head ends at the palate
        if v.co.x > hx0 - 0.03 and v.co.z < mouth_line(v.co.x):
            k = min(1.0, (v.co.x - (hx0 - 0.03)) / 0.03)
            v.co.z = v.co.z + (mouth_line(v.co.x) - 0.002 - v.co.z) * k
    er = bm.verts.layers.float.new("eyer")
    ex = L / 2 - 0.105
    ez = bot(ex) + 0.72 * (top(ex) - bot(ex))
    for s in (1, -1):
        c = Vector((ex, s * wid(ex) * 0.72, ez))
        faces = ellipsoid(bm, c, (0.0115, 0.006, 0.0105), subdiv=3, mat=1)
        axis = Vector((0.2, s * 1.0, 0.35)).normalized()
        for v in (v for f in faces for v in f.verts):
            v[er] = math.acos(max(-1.0, min(1.0, (v.co - c).normalized().dot(axis)))) / (math.pi / 2)
    tb0 = bmesh.new()                                    # the upper teeth, just inside the edge of the upper jaw
    pts = []
    for s in (1, -1):
        for i in range(14):
            x = L / 2 - 0.012 - i * 0.0105
            pts.append(Vector((x, s * wid(x) * 0.62, mouth_line(x) + 0.002)))
    teeth_row(tb0, pts, 0.0085, 0.0018, 0, down=True)
    me = fish_mesh(name, bm, [M.musky, M.eye])
    body = bpy.data.objects.new(name, me)
    col.objects.link(body)
    upper_teeth = bpy.data.objects.new(name + "UpperTeeth", fish_mesh(name + "UpperTeeth", tb0, [M.teeth]))
    col.objects.link(upper_teeth)
    upper_teeth.parent = body
    fb = bmesh.new()
    x0 = -0.2                                            # the dorsal, far back
    soft_fin(fb, [fin_base(top, x0, 0.004), (x0 - 0.01, top(x0) + 0.045), (x0 - 0.05, top(x0 - 0.05) + 0.075),
                  (x0 - 0.1, top(x0 - 0.1) + 0.07), (x0 - 0.15, top(x0 - 0.15) + 0.045), (x0 - 0.185, top(x0 - 0.185) + 0.012),
                  fin_base(top, x0 - 0.19, 0.004)], 0.003, 0, samples=8)
    xa = -0.23                                           # the anal fin, opposite the dorsal
    soft_fin(fb, [fin_base(bot, xa, 0.004), (xa - 0.01, bot(xa) - 0.04), (xa - 0.05, bot(xa - 0.05) - 0.065),
                  (xa - 0.1, bot(xa - 0.1) - 0.055), (xa - 0.145, bot(xa - 0.145) - 0.02), fin_base(bot, xa - 0.15, 0.004)],
             0.003, 0, samples=8)
    xt = -L / 2                                          # the forked tail, pointed lobes
    soft_fin(fb, [(xt + 0.04, 0.03), (xt - 0.05, 0.075), (xt - 0.14, 0.115), (xt - 0.15, 0.1), (xt - 0.095, 0.03),
                  (xt - 0.085, 0.0), (xt - 0.095, -0.03), (xt - 0.15, -0.1), (xt - 0.14, -0.115), (xt - 0.05, -0.075),
                  (xt + 0.04, -0.03)], 0.004, 0, samples=6)
    for s in (1, -1):                                    # pelvic fins, mid body
        rot = Euler((s * math.radians(20), 0.0, 0.0))
        xp = -0.02
        soft_fin(fb, [(xp + 0.02, bot(xp) + 0.012), (xp - 0.01, bot(xp) - 0.035), (xp - 0.05, bot(xp) - 0.045),
                      (xp - 0.06, bot(xp) - 0.02), (xp - 0.03, bot(xp) + 0.01)], 0.003, 0, offset=(0, s * 0.025, 0), rot=rot)
    fins = bpy.data.objects.new(name + "Fins", fish_mesh(name + "Fins", fb, [M.musky_fin]))
    col.objects.link(fins)
    fins.parent = body
    pects = []
    for s in (1, -1):                                    # small pectorals low behind the gill cover
        pb = bmesh.new()
        soft_fin(pb, [(0.02, 0.004), (0.0, 0.016), (-0.04, 0.022), (-0.075, 0.012), (-0.08, 0.0), (-0.06, -0.012),
                      (-0.02, -0.014), (0.02, -0.006)], 0.003, 0, samples=8)
        po = bpy.data.objects.new(f"{name}Pectoral{'L' if s > 0 else 'R'}", fish_mesh(f"{name}Pect{s}", pb, [M.musky_fin]))
        col.objects.link(po)
        po.parent = body
        xq = L / 2 - 0.2
        po.location = (xq, s * (wid(xq) - 0.008), bot(xq) + 0.016)
        po.rotation_euler = (s * math.radians(-10), math.radians(8), s * math.radians(-22))
        pects.append(po)
    jb = bmesh.new()                                     # the lower jaw: long, flat, slightly projecting
    hx = M_JAW_HINGE.x
    jl = L / 2 + 0.008 - hx
    mz = M_JAW_HINGE.z

    def jtop(x):
        return mouth_line(hx + x) - mz - 0.001

    def jbot(x):
        return bot(hx + x) - mz

    def jwid(x):
        return wid(hx + x) * 0.98 + 0.001
    ring_section(jb, jl, -0.02, jtop, jbot, jwid, nr=24, na=24, power=2.3, xr=lambda x: 0.95)
    for v in jb.verts:
        v[jb.verts.layers.float["zrel"]] *= 0.4
    tpts = []
    for s in (1, -1):
        for i in range(13):
            x = jl - 0.012 - i * 0.0105
            tpts.append(Vector((x, s * jwid(x) * 0.6, jtop(x) - 0.002)))
    nj = len(jb.faces)
    teeth_row(jb, tpts, 0.009, 0.0018, 0, down=False)
    jaw = bpy.data.objects.new(name + "Jaw", fish_mesh(name + "Jaw", jb, [M.musky]))
    col.objects.link(jaw)
    jaw.parent = body
    jaw.location = M_JAW_HINGE
    jaw.rotation_mode = "XYZ"
    jaw.data.materials.append(M.throat)
    jaw.data.materials.append(M.teeth)
    for i, p in enumerate(jaw.data.polygons):
        p.use_smooth = True
        if i >= nj:
            p.material_index = 2
        elif p.normal.z > 0.45:                          # the inside of the mouth
            p.material_index = 1
    tb = bmesh.new()                                     # the throat
    ellipsoid(tb, (0.0, 0.0, 0.0), (0.08, 0.026, 0.016), subdiv=3, mat=0)
    throat = finish(name + "Throat", tb, col, [M.throat])
    throat.parent = body
    throat.location = (L / 2 - 0.12, 0.0, -0.016)
    root = empty_object(name + "Rig", col, size=0.3)
    root.rotation_mode = "QUATERNION"
    root.scale = (MUSKY_SCALE,) * 3
    body.parent = root
    beat = empty_object(name + "Beat", col, size=0.05)
    ng = musky_swim_group(beat, name + "Swim")
    for ob in (body, fins):
        ob.modifiers.new("Swim", "NODES").node_group = ng
    sub = body.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 2
    for ob in (body, fins, jaw, throat, upper_teeth, *pects):
        ob.visible_shadow = True
    return dict(root=root, body=body, fins=fins, jaw=jaw, throat=throat, pects=pects, beat=beat, L=M_L * MUSKY_SCALE)


def key_gape(g, f, gape, flare=0.0):
    """Open the mouth: gape 0 shut .. 1 fully open (the jaw drops about 34 degrees)."""
    g["jaw"].rotation_euler = (0.0, math.radians(34.0) * gape, 0.0)
    g["throat"].location = (M_L / 2 - 0.12 - 0.01 * gape, 0.0, -0.016 - 0.01 * gape)
    g["throat"].scale = (1.0 + 0.1 * gape, 1.0 + 0.1 * flare, 1.0 + 0.4 * gape)
    for i, po in enumerate(g["pects"]):
        sgn = 1 if i == 0 else -1
        po.rotation_euler = (sgn * math.radians(-10 - 25 * flare), math.radians(8), sgn * math.radians(-22 - 30 * flare))
        key(po, "rotation_euler", f)
    key(g["jaw"], "rotation_euler", f)
    key(g["throat"], "location", f)
    key(g["throat"], "scale", f)


def key_swim(fish, f, phase, amp, bend=0.0):
    b = fish["beat"]
    b.location = (phase, amp, bend)
    key(b, "location", f)


def mouth_of(p, fwd):
    """The point between the jaws, just behind the tip of the snout (world)."""
    return p + heading_quat(fwd) @ Vector((MUSKY_SCALE * (M_L / 2 - 0.03), 0.0, -0.005 * MUSKY_SCALE))


# --------------------------------------------------------------------------
# Small fish: yellow perch (the musky's main prey here) and a school of them
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


def perch_mesh(M, L=0.2):
    """Yellow perch: a deep, compressed body, a spiny first dorsal and a soft second dorsal, a forked tail."""
    top = curve([(0.0, 0.0), (0.08, 0.07), (0.3, 0.13), (0.5, 0.12), (0.75, 0.06), (0.92, 0.035), (1.0, 0.03)])
    bot = curve([(0.0, -0.025), (0.1, -0.07), (0.35, -0.11), (0.55, -0.1), (0.8, -0.05), (0.92, -0.03), (1.0, -0.03)])
    wid = curve([(0.0, 0.0), (0.1, 0.04), (0.35, 0.06), (0.65, 0.045), (0.9, 0.015), (1.0, 0.012)])

    def fins(L, top, bot):
        t = -L / 2
        return [[(0.15 * L, top(0.15 * L) - 0.005 * L), (0.12 * L, top(0.12 * L) + 0.09 * L), (0.0, top(0.0) + 0.08 * L),
                 (-0.05 * L, top(-0.05 * L) - 0.005 * L)],
                [(-0.07 * L, top(-0.07 * L) - 0.005 * L), (-0.1 * L, top(-0.1 * L) + 0.07 * L), (-0.25 * L, top(-0.25 * L) + 0.04 * L),
                 (-0.28 * L, top(-0.28 * L) - 0.005 * L)],
                [(-0.1 * L, bot(-0.1 * L) + 0.005 * L), (-0.13 * L, bot(-0.13 * L) - 0.07 * L), (-0.25 * L, bot(-0.25 * L) - 0.03 * L),
                 (-0.27 * L, bot(-0.27 * L) + 0.005 * L)],
                [(t + 0.03 * L, 0.025 * L), (t - 0.1 * L, 0.12 * L), (t - 0.13 * L, 0.11 * L), (t - 0.07 * L, 0.0),
                 (t - 0.13 * L, -0.11 * L), (t - 0.1 * L, -0.12 * L), (t + 0.03 * L, -0.025 * L)]]
    return small_fish("YellowPerch", M, M.perch, M.perch_fin, L, lambda u: top(u) * L, lambda u: bot(u) * L,
                      lambda u: wid(u) * L, fins, nr=22, na=14, eye_r=0.045)


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
# Choreography. Places: the lake (the hook, the burn, the figure 8 and the water above it for the loop's overhead
# shot), the cut away lake (the cross section, XS) and the lure studio (LS), far apart and each rendered only in its
# own shots. The weed edge runs along x near y = 0: the cabbage flat is y > 0, the drop off y < 0.
# --------------------------------------------------------------------------
XS = Vector((-600.0, 0.0, 0.0))
LS = Vector((600.0, 0.0, -10.0))
LURE_Z = -0.38                        # the bucktail's depth when burned over the weed tops
HOOK_X0, HOOK_Y = 2.0, 0.8            # the hook: where the bucktail starts (x) and its line over the flat (y)
HOOK_V = 2.4                          # burn speed (m/s), illustrative
STRIKE_H = WHIP_IN + 44               # the T-bone in the hook
BURN_X0, BURN_Y = 44.0, 0.9           # the burn shot
OH_X0, OH_Y = -34.0, 2.6              # the loop's overhead shot: the bucktail at frame 1 (it bulges just under the top)
OH_V = 2.0
OH_H = 4.6                            # the overhead camera's height
BOAT_AT = Vector((22.0, -8.0, 0.0))   # the boat over the drop off (figure 8 beside its +y side)
F8_C = BOAT_AT + Vector((0.0, 1.35, -0.95))   # the figure 8's centre
F8_A, F8_B = 1.45, 0.55               # half length (along the boat) and half width of the eight
F8_PERIOD = 4.2                       # seconds per full eight (illustrative)
EAT_F = VO_START[4] + int(round(min(VO_LEN[4] - 1.2, 7.4) * FPS))   # the musky eats on the turn ("figure eight")
OH_FRAMES = None                      # set in camera_frames


def y_edge(x):
    return 0.45 * math.sin(x * 0.17) + 0.25 * math.sin(x * 0.41 + 1.3)


def lake_z(x, y):
    """Lake bed height at (x, y) (lake frame): the cabbage flat, then the drop off into deeper water."""
    yy = y - y_edge(x)
    n = 0.12 * mnoise.noise(Vector((x * 0.18, y * 0.18, 3.0))) + 0.04 * mnoise.noise(Vector((x * 0.9, y * 0.9, 5.0)))
    if yy >= 0.0:
        return FLAT_Z + n + 0.03 * max(0.0, yy - 10.0)
    k = min(1.0, -yy / 6.5)
    k = k * k * (3 - 2 * k)
    return FLAT_Z + (DEEP_Z - FLAT_Z) * k + n


def weed_ok(x, y, seed=0.0):
    """Where cabbage grows: on the flat, thick at the edge, in beds with gaps further in."""
    yy = y - y_edge(x)
    if yy < -0.15:
        return 0.0
    bed = 0.5 + 0.5 * mnoise.noise(Vector((x * 0.21, y * 0.21, 7.0 + seed)))
    edge = math.exp(-max(0.0, yy) / 3.0)
    return min(1.0, 0.25 + 0.9 * bed * (0.5 + 0.5 * edge) + 0.4 * edge)


# --- meshes built with numpy -------------------------------------------------------------------------------------
def np_mesh(name, verts, faces, col, mats, attrs=None, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts, float), [], faces.tolist() if isinstance(faces, np.ndarray) else faces)
    me.update()
    for an, (dom, vals) in (attrs or {}).items():
        a = me.attributes.new(an, "FLOAT", dom)
        a.data.foreach_set("value", np.asarray(vals, float))
    for m in mats:
        me.materials.append(m)
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    return ob


def build_lakebed(col, M, x_rng, y_rng, step, name, offset=Vector((0, 0, 0))):
    xs = np.arange(x_rng[0], x_rng[1] + 1e-6, step)
    ys = np.arange(y_rng[0], y_rng[1] + 1e-6, step)
    nx, ny = len(xs), len(ys)
    V = np.array([(x + offset.x, y + offset.y, lake_z(x, y) + offset.z) for y in ys for x in xs], float)
    i = np.arange(nx - 1)[None, :] + nx * np.arange(ny - 1)[:, None]
    i = i.ravel()
    F = np.stack([i, i + 1, i + nx + 1, i + nx], axis=1)
    ob = np_mesh(name, V, F, col, [M.lakebed])
    return ob


def cabbage_template(rng, height):
    """One cabbage plant: a stalk with alternate, wavy, oval leaves angled upward. Returns verts, faces, material
    index per face (0 leaf, 1 stalk) and a sway weight per vertex (0 at the root, 1 at the top)."""
    V, F, Fm = [], [], []
    segs = 7
    lean = Vector((rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), 1.0))
    pts = [Vector((0, 0, 0)) + Vector((lean.x * (k / segs) ** 2, lean.y * (k / segs) ** 2, 0)) * height +
           Vector((0, 0, height * k / segs)) for k in range(segs + 1)]
    sides = 4
    r = 0.004
    for k, p in enumerate(pts):
        for j in range(sides):
            a = 2 * math.pi * j / sides
            V.append(p + Vector((math.cos(a) * r, math.sin(a) * r, 0.0)))
    for k in range(segs):
        for j in range(sides):
            a, b = k * sides + j, k * sides + (j + 1) % sides
            F.append((a, b, b + sides, a + sides))
            Fm.append(1)
    nleaf = int(height / 0.075)
    az = rng.uniform(0, 2 * math.pi)
    for li in range(nleaf):
        h = 0.12 + (height - 0.14) * li / max(1, nleaf - 1)
        az += math.radians(137.5)
        k = h / height
        base = Vector((lean.x * k * k * height, lean.y * k * k * height, h))
        L = rng.uniform(0.15, 0.23) * (0.75 + 0.35 * (1 - k))
        W = L * rng.uniform(0.36, 0.46)
        up = math.radians(rng.uniform(30, 60))
        d = Vector((math.cos(az) * math.cos(up), math.sin(az) * math.cos(up), math.sin(up)))
        side = Vector((-math.sin(az), math.cos(az), 0.0))
        rows, cols = 5, 3
        b0 = len(V)
        for ri in range(rows):
            u = ri / (rows - 1)
            w = W * math.sin(math.pi * min(0.98, max(0.04, u))) * 0.5
            droop = Vector((0, 0, -0.25 * L * u * u))
            for ci in range(cols):
                v = (ci - 1) * w
                wave = 0.012 * math.sin(u * 9.0 + ci) * (abs(ci - 1))
                V.append(base + d * (L * u) + side * v + droop + Vector((0, 0, wave)))
        for ri in range(rows - 1):
            for ci in range(cols - 1):
                a = b0 + ri * cols + ci
                F.append((a, a + 1, a + cols + 1, a + cols))
                Fm.append(0)
    V = np.array([tuple(v) for v in V], float)
    sway = np.clip(V[:, 2] / height, 0.0, 1.0) ** 1.5
    return V, np.array(F, int), np.array(Fm, int), sway


def build_cabbage(col, M, spots, seed, name):
    """Cabbage beds: plant templates instanced at the given (x, y, base_z, height) spots, merged into one mesh.
    Attributes: rand (per plant colour), sway (0 root .. 1 top) and phase (per plant) for the sway modifier."""
    rng = random.Random(seed)
    temps = [cabbage_template(rng, 1.0) for _ in range(6)]
    Vs, Fs, Ms, rnd, sway, phase = [], [], [], [], [], []
    nv = 0
    for (x, y, z0, h) in spots:
        V, F, Fm, sw = temps[rng.randrange(len(temps))]
        a = rng.uniform(0, 2 * math.pi)
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        S = np.array([1.0, 1.0, h])
        Vt = (V * S) @ R.T + np.array([x, y, z0])
        Vs.append(Vt)
        Fs.append(F + nv)
        Ms.append(Fm)
        nv += len(V)
        rv = rng.random()
        rnd.append(np.full(len(V), rv))
        sway.append(sw * h)
        phase.append(np.full(len(V), rng.uniform(0, 2 * math.pi)))
    V = np.concatenate(Vs)
    F = np.concatenate(Fs)
    ob = np_mesh(name, V, F, col, [M.cabbage, M.stalk],
                 attrs={"rand": ("POINT", np.concatenate(rnd)), "sway": ("POINT", np.concatenate(sway)),
                        "phase": ("POINT", np.concatenate(phase))})
    ob.data.polygons.foreach_set("material_index", np.concatenate(Ms))
    ob.modifiers.new("Sway", "NODES").node_group = sway_group(name + "Sway")
    print(f"{name}: {len(spots)} plants, {len(V)} vertices")
    return ob


def sway_group(name):
    """Plants swaying in the current: loop-safe (whole numbers of cycles per film length)."""
    ng, gi, go = new_group(name, "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes = ng.nodes

    def nattr(nm):
        r = nodes.new("GeometryNodeInputNamedAttribute")
        r.data_type = "FLOAT"
        r.inputs["Name"].default_value = nm
        return r.outputs[0]
    t = gn_time(ng)
    w1, w2 = 2 * math.pi * 9 / LOOP_S, 2 * math.pi * 7 / LOOP_S
    ph, sw = nattr("phase"), nattr("sway")
    x, y, _z = gn_xyz(ng)
    a1 = nmath(ng, "ADD", nmath(ng, "MULTIPLY_ADD", t, w1, ph), nmath(ng, "MULTIPLY", x, 0.6))
    a2 = nmath(ng, "ADD", nmath(ng, "MULTIPLY_ADD", t, w2, ph), nmath(ng, "MULTIPLY", y, 0.5))
    dx = nmath(ng, "MULTIPLY", nmath(ng, "SINE", a1), nmath(ng, "MULTIPLY", sw, 0.09))
    dy = nmath(ng, "MULTIPLY", nmath(ng, "COSINE", a2), nmath(ng, "MULTIPLY", sw, 0.05))
    sp = nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, dx, dy, 0.0), sp.inputs["Offset"])
    ng.links.new(sp.outputs[0], go.inputs["Geometry"])
    return ng


def weed_spots(x_rng, y_rng, density, seed, ground=lake_z, offset=Vector((0, 0, 0)), keep=None):
    rng = random.Random(seed)
    area = (x_rng[1] - x_rng[0]) * (y_rng[1] - y_rng[0])
    out = []
    for _ in range(int(area * density)):
        x, y = rng.uniform(*x_rng), rng.uniform(*y_rng)
        if rng.random() > weed_ok(x, y):
            continue
        if keep and not keep(x, y):
            continue
        z0 = ground(x, y)
        top = WEED_TOP + rng.uniform(-0.3, 0.05)
        out.append((x + offset.x, y + offset.y, z0 - 0.03 + offset.z, max(0.4, top - z0)))
    return out


def build_lilypads(col, M, x_rng, y_rng, n, seed, name, offset=Vector((0, 0, 0)), stems=True):
    """Lily pads floating on the top (dark from below), each with its stem down to the bottom."""
    rng = random.Random(seed)
    Vs, Fs, Ms = [], [], []
    nv = 0
    for _ in range(n):
        x, y = rng.uniform(*x_rng), rng.uniform(*y_rng)
        r = rng.uniform(0.11, 0.2)
        a0 = rng.uniform(0, 2 * math.pi)
        seg = 18
        ring = [(x + r * math.cos(a0 + 0.35 + (2 * math.pi - 0.7) * k / seg), y + r * math.sin(a0 + 0.35 + (2 * math.pi - 0.7) * k / seg),
                 0.012) for k in range(seg + 1)]
        V = [(x, y, 0.012)] + ring
        F = [(0, k + 1, k + 2) for k in range(seg)]
        Vs.append(np.array(V))
        Fs.extend([tuple(i + nv for i in fc) for fc in F])
        Ms.append(np.zeros(len(F), int))
        nv += len(V)
        if stems:
            z0 = lake_z(x, y)
            p0, p1 = Vector((x, y, 0.01)), Vector((x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3), z0))
            sv, sf = [], []
            for k in range(2):
                p = p0.lerp(p1, k)
                for j in range(3):
                    a = 2 * math.pi * j / 3
                    sv.append(tuple(p + Vector((math.cos(a) * 0.004, math.sin(a) * 0.004, 0))))
            for j in range(3):
                sf.append((j, (j + 1) % 3, (j + 1) % 3 + 3, j + 3))
            Vs.append(np.array(sv))
            Fs.extend([tuple(i + nv for i in fc) for fc in sf])
            Ms.append(np.ones(len(sf), int))
            nv += len(sv)
    V = np.concatenate(Vs) + np.array(tuple(offset))
    ob = np_mesh(name, V, Fs, col, [M.lilypad, M.stalk], smooth=False)
    ob.data.polygons.foreach_set("material_index", np.concatenate(Ms))
    ob.visible_shadow = True
    return ob


# --- the bucktail --------------------------------------------------------------------------------------------------
BT_LEN = 0.26                         # a big double blade bucktail, nose loop to the end of the hair (illustrative)


def build_bucktail(col, M, name="Bucktail"):
    """A double blade bucktail along +X (nose at +X): wire shaft and line tie, two clevises with willow/Indiana
    blades (spun about the shaft by the 'spin' empty), beads, a long hair skirt and a treble hook in the hair."""
    root = empty_object(name, col, size=0.05)
    root.rotation_mode = "QUATERNION"
    bm = bmesh.new()
    tube(bm, [Vector((0.13, 0, 0)), Vector((-0.07, 0, 0))], 0.0011, segs=6)          # shaft
    torus(bm, Vector((0.137, 0, 0)), (0, 1, 0), 0.006, 0.0011, n=12, m=5)            # line tie loop
    shaft = finish(name + "Shaft", bm, col, [M.wire])
    shaft.parent = root
    bm = bmesh.new()
    for x in (0.03, 0.017, 0.004, -0.009, -0.022):                                   # beads behind the clevises
        ellipsoid(bm, (x, 0, 0), (0.0055, 0.0055, 0.0055), subdiv=2)
    beads = finish(name + "Beads", bm, col, [M.bead])
    beads.parent = root
    blades = []
    for i, (bx, L, W, mat) in enumerate(((0.105, 0.052, 0.024, M.blade), (0.07, 0.06, 0.028, M.blade_gold))):
        spin = empty_object(f"{name}Spin{i}", col, size=0.02)
        spin.parent = root
        spin.location = (bx, 0, 0)
        spin.rotation_mode = "XYZ"
        bm = bmesh.new()
        ring = []
        n = 18
        for k in range(n):                                                           # a cupped, pointed oval blade
            a = 2 * math.pi * k / n
            u = 0.5 - 0.5 * math.cos(a)
            x = -L * u
            y = W * 0.5 * math.sin(a) * (1.0 - 0.35 * u * u)
            cup = 0.004 * (1 - (2 * y / W) ** 2)
            ring.append(bm.verts.new((x, y, 0.013 + cup)))
        c = bm.verts.new((-L * 0.5, 0, 0.017))
        for k in range(n):
            bm.faces.new((ring[k], ring[(k + 1) % n], c))
        bm_frustum(bm, (0.002, 0, 0.0), (-0.004, 0, 0.012), 0.0012, 0.0012, segs=5)  # the clevis
        bl = finish(f"{name}Blade{i}", bm, col, [mat], recalc=True)
        sol = bl.modifiers.new("Thick", "SOLIDIFY")
        sol.thickness = 0.0012
        bl.parent = spin
        blades.append(spin)
    rng = random.Random(4)                                                            # the hair skirt
    bm = bmesh.new()
    rl = bm.verts.layers.float.new("rand")
    for k in range(340):
        a = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0.002, 0.006)
        p0 = Vector((-0.024, math.cos(a) * rr, math.sin(a) * rr))
        L = rng.uniform(0.09, 0.13)
        flare = rng.uniform(0.012, 0.032)
        p1 = Vector((-0.024 - L, math.cos(a) * flare, math.sin(a) * flare))
        side = Vector((0, -math.sin(a), math.cos(a))) * 0.0009
        rv = rng.random()
        vs = [bm.verts.new(p0 - side), bm.verts.new(p0 + side), bm.verts.new(p0.lerp(p1, 0.5) + side * 0.8),
              bm.verts.new(p1), bm.verts.new(p0.lerp(p1, 0.5) - side * 0.8)]
        for v in vs:
            v[rl] = rv
        bm.faces.new((vs[0], vs[1], vs[2], vs[4]))
        bm.faces.new((vs[4], vs[2], vs[3]))
    hair = finish(name + "Hair", bm, col, [M.hair], recalc=False)
    hair.parent = root
    bm = bmesh.new()                                                                  # treble hook in the hair
    tube(bm, [Vector((-0.07, 0, 0)), Vector((-0.11, 0, 0))], 0.0013, segs=5)
    for k in range(3):
        a = 2 * math.pi * k / 3
        d = Vector((0, math.cos(a), math.sin(a)))
        pts = [Vector((-0.11, 0, 0)) + d * 0.004 * j + Vector((0.004 * j * j, 0, 0)) for j in range(5)]
        tube(bm, pts, 0.0011, segs=5)
    hook = finish(name + "Treble", bm, col, [M.hook])
    hook.parent = root
    parts = [shaft, beads, hair, hook] + [b.children[0] for b in blades]
    for p in parts:
        p.visible_shadow = True
    return dict(root=root, spins=blades, parts=parts)


def key_bucktail(bt, f, p, fwd, spin_rate, roll=0.0):
    bt["root"].location = p
    bt["root"].rotation_quaternion = heading_quat(fwd, roll)
    key(bt["root"], "location", f)
    key(bt["root"], "rotation_quaternion", f)
    for i, s in enumerate(bt["spins"]):
        s.rotation_euler = (spin_rate * t_of(f) * (1.0 if i == 0 else -1.07) + i, 0.0, 0.0)
        key(s, "rotation_euler", f)


# --- paths -----------------------------------------------------------------------------------------------------------
def lure_over(f):
    """The overhead shot's bucktail, bulging just under the top: one run split across the loop's join."""
    tt = t_of(f) if f < CUT4 - 20 else (f - (FRAME_END + 1)) / FPS
    return Vector((OH_X0 + OH_V * tt, OH_Y + 0.08 * math.sin(tt * 2.1), -0.06))


def lure_hook(f):
    t = t_of(f) - t_of(WHIP_IN)
    return Vector((HOOK_X0 + HOOK_V * t, HOOK_Y + 0.05 * math.sin(t * 5.0), LURE_Z + 0.02 * math.sin(t * 7.0)))


def musky_hook(f):
    """Lurking in the cabbage at the edge, a burst up and across into the bucktail (a T-bone), then a turn back
    toward the edge with the lure across its jaws, shaking its head."""
    hit = lure_hook(STRIKE_H)
    lurk = hit + Vector((-0.2, 1.45, -0.95))
    if f <= STRIKE_H - 9:
        t = t_of(f)
        return lurk + Vector((0.03 * math.sin(t * 1.3), 0.02 * math.sin(t * 0.9), 0.0)), \
            Vector((0.25, 1.0, 0.25)).normalized()
    if f <= STRIKE_H:
        u = ease((f - (STRIKE_H - 9)) / 9.0)
        fwd = (hit - lurk).normalized()
        return lurk.lerp(hit - fwd * (MUSKY_SCALE * (M_L / 2 - 0.08)), u), fwd
    # after the hit: carried on a little, then a sweeping turn to head -x (screen left in the second angle)
    s = (f - STRIKE_H) / FPS
    k = ease(min(1.0, s / 1.4))
    yaw0 = math.atan2((hit - lurk).y, (hit - lurk).x)
    yaw = yaw0 + (math.pi - yaw0) * k
    fwd = Vector((math.cos(yaw), math.sin(yaw), 0.25 * (1 - k)))
    p0 = hit - (hit - lurk).normalized() * (MUSKY_SCALE * (M_L / 2 - 0.08))
    drift = Vector((-0.35 * s * k, 0.5 * (1 - k) * min(s, 0.6), -0.3 * k))
    shake = 0.05 * math.sin(s * 15.0) * math.exp(-s * 0.6)
    fwd = Vector((fwd.x, fwd.y + shake, fwd.z)).normalized()
    return p0 + drift, fwd


def hook_gape(f):
    return smoothstep(STRIKE_H - 6, STRIKE_H - 2, f) * (1.0 - smoothstep(STRIKE_H, STRIKE_H + 3, f)) + \
        0.18 * smoothstep(STRIKE_H, STRIKE_H + 3, f) * (1.0 - smoothstep(CUT1 - 4, CUT1, f))


def xs_musky(f):
    t = t_of(f)
    p = XS + Vector((-0.9, 0.35 + y_edge(0.0), FLAT_Z + 0.95 + 0.03 * math.sin(t * 0.8)))
    return p, Vector((0.0, -1.0, 0.02 * math.sin(t * 0.5))).normalized()


def lure_burn(f):
    t = t_of(f) - t_of(CUT2B)
    return Vector((BURN_X0 + HOOK_V * t, BURN_Y + 0.04 * math.sin(t * 4.0), LURE_Z + 0.02 * math.sin(t * 6.0)))


def musky_burn(f):
    """Rising out of the weeds below and behind the bucktail, then following it (no strike: the follow)."""
    lp = lure_burn(f)
    s = (f - CUT2B) / FPS
    k = ease(min(1.0, max(0.0, (s - 0.6) / 1.6)))
    rest = Vector((BURN_X0 + 2.6, BURN_Y + 1.1, FLAT_Z + 0.75))
    follow = lp + Vector((-0.95 + 0.2 * k, -0.15, -0.32))
    p = rest.lerp(follow, k)
    fwd = Vector((1.0, 0.05 * math.sin(s * 3.0), 0.25 * (1.0 - k) + 0.05)).normalized()
    return p, fwd


def f8_point(s):
    """The figure 8 (a lemniscate, long along the boat) at phase s (radians)."""
    return F8_C + Vector((F8_A * math.sin(s), F8_B * math.sin(s) * math.cos(s) * 2.0, -0.18 * math.cos(2 * s)))


def f8_phase(f):
    return 2 * math.pi * (t_of(f) - t_of(CUT3)) / F8_PERIOD + 0.6


def rod_tip(f):
    return f8_point(f8_phase(f) + 0.55) + Vector((0, 0, 0.45))


def lure_f8(f):
    if f <= EAT_F:
        return f8_point(f8_phase(f))
    gp, gf = musky_f8(f)
    return mouth_of(gp, gf)


def musky_f8(f):
    """Following the lure round the eight a body length behind; on the turn it closes and eats."""
    lag = 1.15 - 0.95 * ease(smoothstep(EAT_F - 16, EAT_F, f))
    if f <= EAT_F:
        s = f8_phase(f) - lag
        p = f8_point(s) + Vector((0, 0, -0.12))
        d = (f8_point(s + 0.05) - f8_point(s - 0.05)).normalized()
        p = p - d * (MUSKY_SCALE * (M_L / 2 - 0.05))
        return p, d
    p0, d0 = musky_f8(EAT_F)
    s = (f - EAT_F) / FPS
    yaw = math.atan2(d0.y, d0.x) + 1.4 * ease(min(1.0, s / 0.8))
    fwd = Vector((math.cos(yaw), math.sin(yaw), -0.3 * ease(min(1.0, s / 0.8))))
    fwd = Vector((fwd.x, fwd.y + 0.12 * math.sin(s * 16.0) * math.exp(-s), fwd.z)).normalized()
    return p0 + Vector((0.15 * s, 0.2 * s, -0.1 * s)), fwd


def f8_gape(f):
    return smoothstep(EAT_F - 5, EAT_F - 1, f) * (1.0 - smoothstep(EAT_F + 1, EAT_F + 4, f)) + \
        0.15 * smoothstep(EAT_F + 1, EAT_F + 4, f)


# --- the boat --------------------------------------------------------------------------------------------------------
def build_boat(col, M):
    """A 17 ft aluminium fishing boat (seen from below): a V hull with chines and a transom, an outboard leg."""
    root = empty_object("Boat", col, size=0.3)
    root.location = BOAT_AT
    bm = bmesh.new()
    L, B = 5.2, 2.0
    nx = 24
    rings = []
    for i in range(nx + 1):
        u = i / nx
        x = L / 2 - L * u
        bow = max(0.0, (0.25 - u) / 0.25)
        hw = B / 2 * (1.0 - 0.75 * bow ** 1.6)
        keel = -0.32 + 0.3 * bow ** 2
        sec = [(-hw, 0.55), (-hw * 0.96, -0.08), (-hw * 0.4, keel + 0.07), (0.0, keel), (hw * 0.4, keel + 0.07),
               (hw * 0.96, -0.08), (hw, 0.55)]
        rings.append([bm.verts.new((x, y, z)) for y, z in sec])
    for ra, rb in zip(rings, rings[1:]):
        for k in range(len(ra) - 1):
            bm.faces.new((ra[k], ra[k + 1], rb[k + 1], rb[k]))
    bm.faces.new(list(reversed(rings[-1])))
    hull = finish("BoatHull", bm, col, [M.alu])
    hull.parent = root
    sol = hull.modifiers.new("Plate", "SOLIDIFY")
    sol.thickness = 0.01
    bm = bmesh.new()
    bm_box(bm, (-2.75, 0.0, -0.2), (0.12, 0.16, 0.9))                                 # outboard leg
    bm_box(bm, (-2.75, 0.0, -0.62), (0.3, 0.06, 0.06))                                # cavitation plate
    for k in range(3):
        a = 2 * math.pi * k / 3
        bm_box(bm, (-2.84, 0.09 * math.cos(a), -0.58 + 0.09 * math.sin(a)), (0.02, 0.06, 0.13), rot=(a, 0, 0))
    leg = finish("Outboard", bm, col, [M.paint])
    leg.parent = root
    for ob in (hull, leg):
        ob.visible_shadow = True
    return dict(root=root, parts=[hull, leg])


def build_rod_tip(col, M):
    """The last metre of the rod, from the tip under water up out through the top (the rest is above water)."""
    tip = empty_object("RodTipEnd", col, size=0.05)
    butt = empty_object("RodTipButt", col, size=0.05)
    rod = line_object("RodTip", col, M, tip, butt, radius=0.0055, count=12)
    for n in rod.modifiers[0].node_group.nodes:
        if n.bl_idname == "GeometryNodeSetMaterial":
            n.inputs["Material"].default_value = M.blank
    return dict(tip=tip, butt=butt, ob=rod)


# --- the lure studio ---------------------------------------------------------------------------------------------
def build_studio(col, M):
    piv = empty_object("StudioPivot", col, size=0.1)
    piv.location = LS
    piv.rotation_mode = "XYZ"
    bt = build_bucktail(col, M, "StudioBucktail")
    bt["root"].parent = piv
    bm = bmesh.new()
    prof = [(-4.0, -1.2)] + [(0.6 + 2.5 * math.sin(math.pi * i / 48), -1.2 + 2.5 * (1 - math.cos(math.pi * i / 48)))
                             for i in range(25)] + [(3.1, 10.0)]
    for (y0, z0), (y1, z1) in zip(prof[:-1], prof[1:]):
        bm.faces.new([bm.verts.new(LS + Vector((-6, y0, z0))), bm.verts.new(LS + Vector((6, y0, z0))),
                      bm.verts.new(LS + Vector((6, y1, z1))), bm.verts.new(LS + Vector((-6, y1, z1)))])
    back = finish("StudioBackdrop", bm, col, [M.studio], recalc=False)
    back.visible_shadow = False
    return dict(pivot=piv, bt=bt, back=back)


def studio_rot(f):
    u = (f - CUT2) / max(1, CUT2B - CUT2)
    return math.radians(-38.0 + 62.0 * ease(u))


def studio_point(key_, f):
    pts = {"blade": Vector((0.085, 0.0, 0.016)), "hair": Vector((-0.08, 0.0, 0.0)), "nose": Vector((0.13, 0, 0))}
    q = Euler((0.0, 0.0, studio_rot(f)), "XYZ").to_quaternion()
    return LS + q @ pts[key_]


# --------------------------------------------------------------------------
# Camera: frames[f] = (position, target, rotation, lens); f-stop per frame
# --------------------------------------------------------------------------
def whip(q, angle_deg, axis):
    return Quaternion(Vector(axis), math.radians(angle_deg)) @ q


def gauss_smooth(fn, f, k=6):
    acc, wsum = Vector((0, 0, 0)), 0.0
    for d in range(-2 * k, 2 * k + 1):
        w = math.exp(-0.5 * (d / k) ** 2)
        acc += fn(f + d) * w
        wsum += w
    return acc / wsum


def overhead(f):
    lp = lure_over(f)
    c = Vector((lp.x - 0.5, lp.y - 0.2, OH_H))
    t = Vector((lp.x + 0.4, lp.y - 0.2, 0.0))
    return c, t, look_at(c, t, Vector((1, 0, 0)))


def camera_frames(st):
    frames, fstop = {}, {}
    Z = Vector((0, 0, 1))
    # 0 the loop's join: the bucktail ripping over the lake from above, then a whip pan down
    for f in list(range(1, WHIP_IN)) + list(range(CUT4, FRAME_END + 1)):
        c, t, q = overhead(f)
        if f < WHIP_IN:
            ang = 0.0 if f <= 3 else (13.0 * (f - 3) ** 1.6)
            q = whip(q, ang, q @ Vector((1, 0, 0)))
        elif f < CUT4 + 7:
            kk = 1.0 - ease_out((f - CUT4) / 7.0, 3.0)
            q = whip(q, 40.0 * kk, Z)
        frames[f] = (c, t, q, 24.0)
        fstop[f] = 8.0
    # 1 hook: tracking the bucktail over the weed tops; the musky bursts out and T-bones it
    for f in range(WHIP_IN, CUT1A):
        lp = gauss_smooth(lure_hook, f, 4) if f <= STRIKE_H else None
        if lp is None:
            lp = lure_hook(STRIKE_H).lerp(gauss_smooth(lambda g: musky_hook(g)[0], f, 4), ease(smoothstep(STRIKE_H, STRIKE_H + 20, f)))
        cam = lp + Vector((-0.95, -1.5, -0.3))
        tgt = lp + Vector((0.35, 0.55, -0.22))
        q = look_at(cam, tgt, Z)
        if f < WHIP_IN + 6:
            kk = 1.0 - ease_out((f - WHIP_IN) / 6.0, 3.0)
            q = whip(q, 36.0 * kk, q @ Vector((1, 0, 0)))
        shake = 0.025 * smoothstep(STRIKE_H - 1, STRIKE_H + 1, f) * (1 - smoothstep(STRIKE_H + 1, STRIKE_H + 14, f))
        cam = cam + Vector((shake * math.sin(f * 2.1), 0.0, shake * math.sin(f * 2.7)))
        frames[f] = (cam, tgt, q, 28.0)
        fstop[f] = 4.0
    # 1b side on, close: the musky with the bucktail across its jaws, heading screen left
    for f in range(CUT1A, CUT1):
        mp = gauss_smooth(lambda g: musky_hook(g)[0], f, 5)
        cam = mp + Vector((0.2, -2.25, 0.05))
        tgt = mp + Vector((-0.12, 0.0, 0.0))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 30.0)
        fstop[f] = 4.0
    # 2 the cross section: match cut on the fish (side on, facing screen left), then a pull back to the whole edge
    for f in range(CUT1, CUT2):
        mp, _ = xs_musky(f)
        near_cam = mp + Vector((2.25, 0.12, 0.05))
        near_tgt = mp + Vector((0.0, 0.12, 0.0))
        u = (f - CUT1 - 46) / max(1, CUT2 - CUT1 - 46)
        wide_cam = XS + Vector((9.6 - 0.8 * ease(max(0.0, u)), -0.6, -1.2))
        wide_tgt = XS + Vector((0.0, -0.6, -3.1))
        k = ease(smoothstep(CUT1 + 8, CUT1 + 46, f))
        cam = near_cam.lerp(wide_cam, k)
        tgt = near_tgt.lerp(wide_tgt, ease(smoothstep(CUT1 + 8, CUT1 + 40, f)))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 30.0 - 6.0 * k)
        fstop[f] = 4.0 + 12.0 * k
    # 3 the lure: a close turntable of the bucktail; then the bucktail burned over the weed tops
    for f in range(CUT2, CUT2B):
        u = (f - CUT2) / max(1, CUT2B - CUT2)
        cam = LS + Vector((0.02, -0.62 + 0.08 * ease(u), 0.07))
        tgt = LS + Vector((0.01, 0.0, 0.004))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 45.0)
        fstop[f] = 5.6
    for f in range(CUT2B, CUT3):
        lp = gauss_smooth(lure_burn, f, 3)
        cam = lp + Vector((-0.85, -1.5, -0.32))
        tgt = lp + Vector((-0.45, 0.25, -0.32))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 28.0)
        fstop[f] = 4.0
    # 4 the figure 8 beside the boat: a low, wide angle looking up at the boat's side
    for f in range(CUT3, CUT4):
        u = (f - CUT3) / max(1, CUT4 - CUT3)
        cam = F8_C + Vector((0.3 - 0.2 * u, 2.4 - 0.3 * ease(u), -1.45))
        tgt = F8_C + Vector((0.0, -0.1, 0.0))
        shake = 0.02 * smoothstep(EAT_F - 1, EAT_F + 1, f) * (1 - smoothstep(EAT_F + 1, EAT_F + 14, f))
        cam = cam + Vector((shake * math.sin(f * 2.3), 0.0, shake * math.sin(f * 2.9)))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 18.0)
        fstop[f] = 5.6
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
    return 0.0 if (f < WHIP_IN or f >= CUT4) else 1.0


def deep_at(f):
    if CUT2 <= f < CUT2B:
        return 0.15
    if CUT1 <= f < CUT2:
        return 0.2
    if CUT3 <= f < CUT4:
        return 0.3
    return 0.1 if under_at(f) else 0.0


def clear_at(f):
    if CUT1 <= f < CUT2:
        return 0.7 * smoothstep(CUT1 + 8, CUT1 + 46, f)
    if CUT2 <= f < CUT2B:
        return 1.0
    return 0.0


def set_constant(idb):
    for fc in iter_fcurves(idb):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def build_sun(col):
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 4.0
    light.specular_factor = 0.6
    light.color = srgb(255, 246, 232)
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
    for f in list(range(1, WHIP_IN + 2)) + list(range(CUT4 - 2, FRAME_END + 1)):
        lp = lure_over(f)
        sea.location = (lp.x, lp.y, 0.0)
        sea.keyframe_insert("location", frame=f)


def subject_at(f):
    """What the key light follows in each underwater shot."""
    if WHIP_IN <= f < CUT1:
        return musky_hook(f)[0] if f > STRIKE_H - 12 else lure_hook(f)
    if CUT2B <= f < CUT3:
        return lure_burn(f)
    if CUT3 <= f < CUT4:
        return lure_f8(f)
    return None


def key_scene_state(sc, world_vals, sun, key_l, bounce):
    under_v, deep_v = world_vals
    for f in range(1, FRAME_END + 1):
        u, d, c = under_at(f), deep_at(f), clear_at(f)
        for prop, val in (("under", u), ("deep", d), ("clear", c)):
            sc[prop] = val
            sc.keyframe_insert(data_path=f'["{prop}"]', frame=f)
        under_v.outputs[0].default_value = u
        deep_v.outputs[0].default_value = d
        under_v.outputs[0].keyframe_insert("default_value", frame=f)
        deep_v.outputs[0].keyframe_insert("default_value", frame=f)
        sun.data.energy = (6.0 * (1 - 0.5 * d)) if u else 4.5
        sun.data.keyframe_insert("energy", frame=f)
        sun.rotation_quaternion = (-(SUN_UNDER if u else SUN_DIR)).to_track_quat("-Z", "Y")
        sun.keyframe_insert("rotation_quaternion", frame=f)
        key_l.data.energy, bounce.data.energy = 0.0, 0.0
        p = subject_at(f)
        if p is not None:
            key_l.location = p + Vector((-1.0, -2.6, 2.2))
            aim(key_l, p)
            key_l.data.energy = 260.0
            bounce.location = p + Vector((1.0, -2.6, -1.2))
            aim(bounce, p)
            bounce.data.energy = 90.0
        elif CUT1 <= f < CUT2:
            key_l.location = XS + Vector((12.0, -3.0, 4.0))
            aim(key_l, XS + Vector((0.0, 1.0, -2.5)))
            key_l.data.energy = 9000.0
            mp, _ = xs_musky(f)
            bounce.location = mp + Vector((2.5, 0.0, -0.4))
            aim(bounce, mp)
            bounce.data.energy = 160.0
        elif CUT2 <= f < CUT2B:
            key_l.location = LS + Vector((-0.6, -0.9, 0.9))
            aim(key_l, LS)
            key_l.data.energy = 60.0
            bounce.location = LS + Vector((0.7, 0.4, 0.5))
            aim(bounce, LS)
            bounce.data.energy = 40.0
        for ob in (key_l, bounce):
            ob.data.keyframe_insert("energy", frame=f)
            key(ob, "location", f)
            key(ob, "rotation_quaternion", f)
    for idb in (sc, sc.world.node_tree, sun, key_l, key_l.data, bounce, bounce.data):
        set_constant(idb)


# --------------------------------------------------------------------------
# Animation
# --------------------------------------------------------------------------
def animate_musky(g, spans, rate_hz=1.1):
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
            amp = a(f) if callable(a) else a
            phase += 2 * math.pi * rate_hz * (1.0 + 4.0 * amp) / FPS
            key_swim(g, f, phase, amp, turn * 0.02)
            gp = gape_fn(f) if gape_fn else 0.0
            key_gape(g, f, gp, flare=gp)


def place(ob, f, p, fwd, roll=0.0):
    ob.location = p
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = heading_quat(fwd, roll)
    key(ob, "location", f)
    key(ob, "rotation_quaternion", f)


def lure_bubbles(path, f_lo, f_hi, seed, per_frame=10):
    """The blades churning a trail of fine bubbles behind the bucktail."""
    rng = random.Random(seed)
    sp = Particles()
    for f in range(f_lo, f_hi + 1):
        p = path(f)
        for _ in range(per_frame):
            j = Vector((rng.gauss(0, 0.012), rng.gauss(0, 0.02), rng.gauss(0, 0.02)))
            v = Vector((rng.gauss(0, 0.05), rng.gauss(0, 0.05), rng.uniform(0.08, 0.25)))
            sp.emit(t_of(f) + rng.uniform(0, 1.0 / FPS), p + j, v, rng.uniform(0.4, 1.0), rng.uniform(0.0015, 0.004))
    return sp


def strike_burst(center_fn, f0, seed, n=500):
    """Bubbles, torn weed bits and silt thrown out by a strike."""
    rng = random.Random(seed)
    sp = Particles()
    for _ in range(n):
        f = f0 + rng.uniform(-1.0, 6.0)
        c = center_fn(int(round(f)))
        d = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))
        if d.length < 1e-3:
            continue
        d.normalize()
        sp.emit(t_of(f), c + d * rng.uniform(0.02, 0.15), d * rng.uniform(0.3, 1.6) + Vector((0, 0, 0.2)),
                rng.uniform(0.3, 0.9), rng.uniform(0.002, 0.007), gravity=-0.3)
    return sp


def drift_silt(center, size, n, seed, f_lo, f_hi):
    """Suspended silt and bits of weed drifting slowly in the current."""
    rng = random.Random(seed)
    sp = Particles()
    t0, t1 = t_of(f_lo) - 4.0, t_of(f_hi)
    for _ in range(n):
        p = Vector(center) + Vector((rng.uniform(-1, 1) * size[0], rng.uniform(-1, 1) * size[1], rng.uniform(-1, 1) * size[2]))
        v = Vector((rng.uniform(0.08, 0.2), rng.gauss(0, 0.04), rng.gauss(0, 0.02)))
        sp.emit(rng.uniform(t0, t1), p, v, rng.uniform(3.0, 6.0), rng.uniform(0.0012, 0.0035))
    return sp


def build_lure_line(col, M, name, lure_fn, far_fn, f_lo, f_hi):
    a = empty_object(name + "A", col, size=0.05)
    b = empty_object(name + "B", col, size=0.05)
    for f in range(f_lo, f_hi + 1):
        a.location = lure_fn(f)
        b.location = far_fn(f)
        key(a, "location", f)
        key(b, "location", f)
    return line_object(name, col, M, a, b, radius=0.0009, count=24)


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------
def build_overlays(hud, st):
    hud.scrims()
    hud.kicker("THE FISHING SERIES")
    p1 = P(1, 1)
    hud.serif("In 1949, Cal Johnson landed a *67 lb 8 oz* musky.", WHIP_IN + 6, p1 - 9, y=-0.30, size=0.046,
              lines=["In 1949, Cal Johnson landed", "a *67 lb 8 oz* musky."], reveal=2.3,
              src="IGFA ALL TACKLE RECORD · JULY 24, 1949")
    hud.serif("The IGFA all tackle world record.", p1, CUT1 - 9, y=-0.30, size=0.046,
              lines=["The IGFA *all tackle*", "world record."], src="IGFA RECORDS · FIELD AND STREAM")
    hud.pin(lambda f: musky_hook(f)[0] + musky_hook(f)[1] * 0.25 + Vector((0.0, 0.0, 0.2)), "muskellunge",
            CUT1A + 4, CUT1 - 6, dot_color=GOLD, side=-0.05, y_band=(-0.18, 0.3))
    hud.badge(1, "CABBAGE WEED EDGE", "where big muskies wait in ambush", CUT1 + 8, CUT2 - 8)
    s1 = P(2, 1)
    hud.serif("They hunt by sight from cover.", CUT1 + 6, s1 - 9, lines=["They hunt by sight", "*from cover.*"],
              src="WISCONSIN DNR")
    hud.serif("Cabbage edges next to deep water.", s1, CUT2 - 9, lines=["Cabbage weed edges,", "*next to deep water.*"],
              src="WISCONSIN DNR MUSKY FACT SHEET")
    hud.pin(lambda f: XS + Vector((0.0, 0.9, FLAT_Z + 1.3)), "cabbage weed edge", CUT1 + 44, CUT2 - 12,
            dot_color=srgb(150, 200, 90), side=0.06, y_band=(-0.22, 0.32))
    hud.pin(lambda f: XS + Vector((0.0, -3.0, lake_z(0.0, -3.0) + 0.1)), "drop off", CUT1 + 54, CUT2 - 10,
            dot_color=srgb(200, 160, 110), side=-0.05, y_band=(-0.22, 0.32))
    hud.pin(lambda f: xs_musky(f)[0] + Vector((0.0, -0.2, 0.15)), "musky in ambush", s1 + 4, CUT2 - 10,
            dot_color=GOLD, side=-0.06, rise=0.09, y_band=(-0.22, 0.32))
    hud.badge(2, "BURN THE BUCKTAIL", "double blades, fast, over the weeds", CUT2 + 8, CUT3 - 8)
    hud.serif("Trigger a reaction strike.", CUT2 + 6, CUT2B - 6, lines=["Trigger a", "*reaction strike.*"],
              src="ANGLER TECHNIQUE")
    hud.serif("Burn it over the weed tops.", CUT2B + 4, CUT3 - 9, lines=["Burn it inches", "*over the weed tops.*"],
              src="ANGLER TECHNIQUE")
    span = CUT2B - CUT2
    hud.pin(lambda f: studio_point("blade", f), "double blades", CUT2 + 8, CUT2 + int(span * 0.55),
            dot_color=GOLD, side=0.06, y_band=(-0.22, 0.3))
    hud.pin(lambda f: studio_point("hair", f), "hair skirt", CUT2 + int(span * 0.5), CUT2B - 6,
            dot_color=GOLD, side=-0.06, y_band=(-0.22, 0.3))
    hud.pin(lambda f: lure_burn(f), "inches over the weeds", CUT2B + 4, CUT2B + 50, dot_color=GOLD, side=0.06,
            y_band=(-0.2, 0.3))
    hud.pin(lambda f: musky_burn(f)[0], "musky below", CUT2B + 54, CUT3 - 6, dot_color=PIN_RED, side=-0.06,
            y_band=(-0.2, 0.3))
    hud.badge(3, "THE FIGURE 8 FINISH", "never pull the lure out early", CUT3 + 6, CUT4 - 6)
    p4 = P(4, 1)
    hud.serif("Never pull it out early.", CUT3 + 4, p4 - 9, lines=["Never pull it", "*out early.*"],
              src="ANGLER TECHNIQUE")
    hud.serif("Up to 40 percent hit at the boat.", p4, CUT4 - 4, lines=["Up to *40 percent*", "hit at the boat."],
              src="PRO ESTIMATE: JIM SARIC, MERCURY MARINE")
    hud.pin(lambda f: f8_point(f8_phase(f) + 1.0), "wide, deep figure 8", CUT3 + 8, EAT_F - 30, dot_color=GOLD,
            side=0.05, y_band=(-0.2, 0.3))
    hud.pin(lambda f: mouth_of(*musky_f8(f)), "eats on the turn", EAT_F - 12, CUT4 - 6, dot_color=PIN_RED, side=-0.06,
            y_band=(-0.24, 0.32))
    hud.serif("Burn fast, work the turn.", CUT4 + 8, FRAME_END - 9, lines=["Burn fast,", "*work the turn.*"],
              reveal=2.4, src="ANGLER TECHNIQUE")


# --------------------------------------------------------------------------
# Render settings, compositor
# --------------------------------------------------------------------------
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
    """Bloom and a high contrast freshwater grade (rich greens, warm highlights) on the 3D only, then the overlays."""
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
        vals = {"Lift": (1.0, 1.0, 1.0, 1.0), "Gamma": (1.0, 1.0, 1.0, 1.0), "Gain": (1.06, 1.03, 0.95, 1.0)}
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
        c.points.new(0.75, 0.83)
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
    worst = {"surface (above)": (1e9, 0), "surface (under)": (1e9, 0), "lake bed": (1e9, 0), "weeds": (1e9, 0),
             "cut lake (cross section)": (1e9, 0), "musky": (1e9, 0), "boat": (1e9, 0), "lure studio": (1e9, 0),
             "lily pads": (1e9, 0)}

    def upd(k, d, f):
        if d < worst[k][0]:
            worst[k] = (d, f)
    sc.frame_set(1)
    wb = bvh_of(st["weeds"])
    pb = bvh_of(st["pads"])
    xb = bvh_of(st["xs_solid"])
    bb = bvh_of(st["boat"]["parts"])
    for f in range(1, FRAME_END + 1):
        p = frames[f][0]
        t = t_of(f)
        if under_at(f):
            upd("surface (under)", -p.z - 0.25, f)
        else:
            upd("surface (above)", p.z - sea_height(p.x, p.y, t) - 0.3, f)
        if WHIP_IN <= f < CUT1 or CUT2B <= f < CUT4:
            upd("lake bed", p.z - lake_z(p.x, p.y) - 0.3, f)
            h = wb.find_nearest(p)
            upd("weeds", (h[3] if h[0] is not None else 1e9) - 0.12, f)
            h = pb.find_nearest(p)
            upd("lily pads", (h[3] if h[0] is not None else 1e9) - 0.12, f)
            mp = musky_hook(f)[0] if f < CUT1 else (musky_burn(f)[0] if f < CUT3 else musky_f8(f)[0])
            upd("musky", (mp - p).length - MUSKY_SCALE * M_L / 2 - 0.15, f)
        if CUT3 <= f < CUT4:
            h = bb.find_nearest(p)
            upd("boat", (h[3] if h[0] is not None else 1e9) - 0.15, f)
        if CUT1 <= f < CUT2:
            h = xb.find_nearest(p)
            upd("cut lake (cross section)", (h[3] if h[0] is not None else 1e9) - 0.15, f)
            upd("musky", (xs_musky(f)[0] - p).length - MUSKY_SCALE * M_L / 2 - 0.15, f)
        if CUT2 <= f < CUT2B:
            upd("lure studio", (p - LS).length - 0.3, f)
    for k, v in worst.items():
        print(f"closest {k}: {round(v[0], 3)} m at frame {v[1]} (negative = inside)")
    acc, turn = (0.0, 0), (0.0, 0)
    whips = set(range(1, WHIP_IN + 7)) | set(range(CUT4 - 4, CUT4 + 8)) | set(range(CUT1 + 8, CUT1 + 47))
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
             and not (CUT1 + 8 <= f <= CUT1 + 46)]
    print("camera jumps outside cuts:", jumps or "none")
    a, b, b0 = frames[1], frames[FRAME_END], frames[FRAME_END - 1]
    nxt = b[0] + (b[0] - b0[0])                       # where the camera would be one frame after the last
    print("loop: frame 1 vs one frame past the last (the moving overhead shot): position",
          round((a[0] - nxt).length, 4), "m, rotation", round(math.degrees(a[2].rotation_difference(b[2]).angle), 4),
          "deg, lens", a[3], b[3])
    print("loop: lake surface at t=0 vs t=end (should match):", round(sea_height(5.0, 7.0, 0.0) - sea_height(5.0, 7.0, LOOP_S), 6))
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
    subj = {2: ("bucktail wake (overhead)", lambda f: lure_over(f) + Vector((0, 0, 0.06))),
            STRIKE_H - 12: ("musky lurking (hook)", lambda f: musky_hook(f)[0]),
            STRIKE_H: ("strike (hook)", lambda f: lure_hook(STRIKE_H)),
            (CUT1A + CUT1) // 2: ("musky (1b)", lambda f: musky_hook(f)[0]),
            CUT1 + 2: ("musky (cross section)", lambda f: xs_musky(f)[0]),
            CUT2 - 20: ("weed edge (wide)", lambda f: XS + Vector((0.0, 1.0, FLAT_Z + 0.8))),
            CUT2 + 30: ("bucktail (studio)", lambda f: LS),
            CUT2B + 20: ("bucktail (burn)", lambda f: lure_burn(f)),
            CUT3 - 10: ("musky (burn follow)", lambda f: musky_burn(f)[0]),
            CUT3 + 30: ("lure (figure 8)", lambda f: lure_f8(f)),
            EAT_F: ("eat (figure 8)", lambda f: mouth_of(*musky_f8(f))),
            FRAME_END: ("bucktail wake (overhead)", lambda f: lure_over(f) + Vector((0, 0, 0.06)))}
    print("\n=== SUBJECTS ON SCREEN (x in +-0.281, y in +-0.5) ===")
    for f, (nm, fn) in sorted(subj.items()):
        sc.frame_set(f)
        s = hud.project(f, fn(f))
        print(f"frame {f}: {nm} ->", None if s is None else (round(s.x, 3), round(s.y, 3)))


def write_events(path, st):
    data = {
        "fps": FPS, "frames": FRAME_END, "loop": True,
        "cuts": [WHIP_IN, CUT1, CUT2, CUT3, CUT4], "hard_cuts": [CUT1A, CUT2B],
        "whips": [[1, WHIP_IN], [CUT4 - 4, CUT4 + 6]],
        "overlays": sorted(st["hud"].events),
        "under": [[WHIP_IN, CUT4]], "strike": STRIKE_H, "eat": EAT_F, "burn": [CUT2B, CUT3], "studio": [CUT2, CUT2B],
        "section": [CUT1, CUT2], "figure8": [CUT3, CUT4], "overhead": [[1, WHIP_IN], [CUT4, FRAME_END]],
        "vo_start": VO_START, "vo_len": VO_LEN, "vo_text": VO_TEXT,
        "beats": [1, CUT1, CUT2, CUT3, CUT4],
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1)
    print("events written", path)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build_xs(col, M):
    """The cut away lake: the profile across the edge (y) extruded behind the cut face at x = XS.x, with the bed's
    top surface, cabbage on the flat, lily pads, and a hatched cut face (a diagram, not to scale)."""
    ys = np.arange(-16.0, 22.0 + 1e-6, 0.25)
    xs = np.arange(-16.0, 0.0 + 1e-6, 0.5)
    nx, ny = len(xs), len(ys)
    V = np.array([(XS.x + x, y, lake_z(0.0, y)) for y in ys for x in xs], float)
    i = np.arange(nx - 1)[None, :] + nx * np.arange(ny - 1)[:, None]
    i = i.ravel()
    F = np.stack([i, i + 1, i + nx + 1, i + nx], axis=1)
    top = np_mesh("XSBed", V, F, col, [M.lakebed])
    zb = DEEP_Z - 3.0
    fv = [(XS.x, y, lake_z(0.0, y)) for y in ys] + [(XS.x, y, zb) for y in ys]
    n = len(ys)
    ff = [(k, k + n, k + n + 1, k + 1) for k in range(n - 1)]
    face = np_mesh("XSCutFace", fv, ff, col, [M.cutface], smooth=False)
    face.visible_shadow = False
    spots = []
    rng = random.Random(21)
    for _ in range(1400):
        x, y = rng.uniform(-14.0, -0.3), rng.uniform(-0.2, 18.0)
        if rng.random() > weed_ok(0.0, y, seed=x * 0.1):
            continue
        z0 = lake_z(0.0, y)
        spots.append((XS.x + x, y, z0 - 0.03, max(0.4, WEED_TOP + rng.uniform(-0.3, 0.05) - z0)))
    weeds = build_cabbage(col, M, spots, 22, "XSCabbage")
    pads = build_lilypads(col, M, (-12.0, -0.5), (6.0, 16.0), 70, 23, "XSLilyPads", offset=Vector((XS.x, 0, 0)),
                          stems=False)
    return dict(top=top, face=face, weeds=weeds, pads=pads)


def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.name = "Main"
    sc.frame_start, sc.frame_end = 1, FRAME_END
    sc.render.fps = FPS
    for prop in ("under", "deep", "clear"):
        sc[prop] = 0.0
    c_sea, c_lake, c_x = collection("Lake"), collection("LakeSite"), collection("CrossSection")
    c_ls, c_fx, c_cam = collection("LureStudio"), collection("LightAndWater"), collection("Camera")
    atmos = build_atmos_group()
    M = build_materials(atmos)
    c_fx.objects.link(M.wake_ref)
    for f in list(range(1, WHIP_IN + 2)) + list(range(CUT4 - 2, FRAME_END + 1)):
        lp = lure_over(f)
        M.wake_ref.location = (lp.x, lp.y, 0.0)
        key(M.wake_ref, "location", f)
    _w, w_under, w_deep = build_world(sc)
    sun = build_sun(c_fx)
    key_l = build_fill(c_fx, "UnderwaterKey", size=4.0, color=(236, 244, 220))
    bounce = build_fill(c_fx, "UnderwaterBounce", size=4.0, color=(170, 200, 160))
    # above water
    sea = build_sea(c_sea, M)
    key_sea_center(sea)
    oh_pads = build_lilypads(c_sea, M, (OH_X0 - 12.0, OH_X0 + 14.0), (OH_Y + 0.7, OH_Y + 3.4), 90, 3, "OverheadLilyPads",
                             stems=False)
    # the lake site: bed, cabbage, pads, perch, fish, lures, boat
    bed = build_lakebed(c_lake, M, (-12.0, 80.0), (-26.0, 16.0), 0.5, "LakeBed")
    spots = weed_spots((-4.0, 34.0), (-0.6, 9.0), 3.6, 11) + weed_spots((38.0, 66.0), (-0.6, 6.0), 3.6, 12)
    weeds = build_cabbage(c_lake, M, spots, 13, "Cabbage")
    pads = build_lilypads(c_lake, M, (HOOK_X0 + 3.0, HOOK_X0 + 12.0), (2.2, 5.5), 60, 14, "LilyPads")
    perch_ob = bpy.data.objects.new("PerchFish", perch_mesh(M, L=0.2))
    c_lake.objects.link(perch_ob)
    perch_ob.hide_render = perch_ob.hide_viewport = True
    school = build_school(c_lake, perch_ob, Vector((HOOK_X0 + 7.0, -1.2, -1.5)), 60, 0.4, 1.4, 0.4, 15, "PerchSchool")
    mh = make_musky(c_lake, M, "MuskyHook")
    animate_musky(mh, [(WHIP_IN - 2, CUT1 + 2, musky_hook, hook_gape,
                        lambda f: 0.012 + 0.05 * smoothstep(STRIKE_H - 10, STRIKE_H - 2, f) * (1 - smoothstep(STRIKE_H + 10, STRIKE_H + 30, f)))])
    mb = make_musky(c_lake, M, "MuskyBurn")
    animate_musky(mb, [(CUT2B - 2, CUT3 + 2, musky_burn, None, 0.03)])
    mf = make_musky(c_lake, M, "MuskyFig8")
    animate_musky(mf, [(CUT3 - 2, CUT4 + 2, musky_f8, f8_gape,
                        lambda f: 0.025 + 0.04 * smoothstep(EAT_F - 14, EAT_F, f) * (1 - smoothstep(EAT_F + 24, EAT_F + 40, f)))])
    bt_hook = build_bucktail(c_lake, M, "BucktailHook")
    for f in range(WHIP_IN - 2, CUT1 + 2):
        if f <= STRIKE_H:
            key_bucktail(bt_hook, f, lure_hook(f), Vector((1, 0, 0)), 40.0)
        else:                                          # across the jaws
            gp, gf = musky_hook(f)
            side = gf.cross(Vector((0, 0, 1))).normalized()
            key_bucktail(bt_hook, f, mouth_of(gp, gf) - gf * 0.04 - side * 0.03, side, 4.0)
    bt_burn = build_bucktail(c_lake, M, "BucktailBurn")
    for f in range(CUT2B - 2, CUT3 + 2):
        key_bucktail(bt_burn, f, lure_burn(f), Vector((1, 0, 0)), 40.0)
    bt_f8 = build_bucktail(c_lake, M, "BucktailFig8")
    for f in range(CUT3 - 2, CUT4 + 2):
        if f <= EAT_F:
            p = lure_f8(f)
            d = (lure_f8(f + 1) - lure_f8(f - 1))
            d = d.normalized() if d.length > 1e-4 else Vector((1, 0, 0))
            key_bucktail(bt_f8, f, p, d, 30.0)
        else:
            gp, gf = musky_f8(f)
            side = gf.cross(Vector((0, 0, 1))).normalized()
            key_bucktail(bt_f8, f, mouth_of(gp, gf) - gf * 0.04 - side * 0.03, side, 3.0)
    line_hook = build_lure_line(c_lake, M, "LineHook", lambda f: bt_hook["root"].location.copy() if False else
                                (lure_hook(f) if f <= STRIKE_H else mouth_of(*musky_hook(f))),
                                lambda f: (lure_hook(f) if f <= STRIKE_H else mouth_of(*musky_hook(f))) + Vector((14.0, 0.0, 0.38)),
                                WHIP_IN - 2, CUT1 + 1)
    line_burn = build_lure_line(c_lake, M, "LineBurn", lure_burn, lambda f: lure_burn(f) + Vector((14.0, 0.0, 0.38)),
                                CUT2B - 2, CUT3 + 1)
    boat = build_boat(c_lake, M)
    rod = build_rod_tip(c_lake, M)
    for f in range(CUT3 - 2, CUT4 + 2):
        tp = rod_tip(f)
        rod["tip"].location = tp
        rod["butt"].location = BOAT_AT + Vector(((tp.x - BOAT_AT.x) * 0.3 - 0.2, 0.25, 1.1))
        key(rod["tip"], "location", f)
        key(rod["butt"], "location", f)
    line_f8 = build_lure_line(c_lake, M, "LineFig8", lure_f8, rod_tip, CUT3 - 2, CUT4 + 1)
    bub_hook = lure_bubbles(lure_hook, WHIP_IN - 30, STRIKE_H, 31).build("BubblesHook", c_lake, M.bubbles)
    bub_burn = lure_bubbles(lure_burn, CUT2B - 30, CUT3, 32).build("BubblesBurn", c_lake, M.bubbles)
    bub_f8 = lure_bubbles(lure_f8, CUT3 - 10, EAT_F, 33, per_frame=5).build("BubblesFig8", c_lake, M.bubbles)
    burst_hook = strike_burst(lambda f: lure_hook(STRIKE_H), STRIKE_H, 34).build("StrikeHook", c_lake, M.bubbles)
    burst_f8 = strike_burst(lambda f: lure_f8(EAT_F), EAT_F, 35).build("StrikeFig8", c_lake, M.bubbles)
    silt = drift_silt(Vector((24.0, -1.0, -1.6)), (30.0, 6.0, 1.2), 5000, 3, WHIP_IN, CUT4).build("DriftSilt", c_lake, M.silt)
    surf = build_surface_under(c_lake, M, center=(30.0, -2.0, 0.0), size=200.0)
    shafts = build_shafts(c_lake, M, (14.0, -2.0, 0.0), n=10, seed=5, spread=14.0, length=9.0)
    shafts += build_shafts(c_lake, M, (50.0, -1.0, 0.0), n=6, seed=6, spread=10.0, length=9.0)
    plank = specks("Specks", c_lake, M.specks, (24.0, -1.0, -1.6), (34.0, 6.0, 1.5), 30000, 0.0016, 31)
    # the cross section
    xsd = build_xs(c_x, M)
    mx = make_musky(c_x, M, "MuskyX")
    animate_musky(mx, [(CUT1 - 2, CUT2 + 2, xs_musky, None, 0.008)], rate_hz=0.6)
    school_x = build_school(c_x, perch_ob, XS + Vector((-2.5, -1.3, -1.4)), 50, 0.4, 1.4, 0.4, 16, "PerchSchoolX")
    surf_x = build_surface_under(c_x, M, center=(XS.x, 0.0, 0.0), size=200.0)
    shafts_x = build_shafts(c_x, M, (XS.x - 4.0, 3.0, 0.0), n=7, seed=7, spread=10.0, length=9.0)
    plank_x = specks("SpecksX", c_x, M.specks, (XS.x - 4.0, 2.0, -3.0), (10.0, 14.0, 3.0), 8000, 0.006, 32)
    # the lure studio
    studio = build_studio(c_ls, M)
    for f in range(CUT2 - 2, CUT2B + 2):
        studio["pivot"].rotation_euler = (0.0, 0.0, studio_rot(f))
        key(studio["pivot"], "rotation_euler", f)
        for i, s in enumerate(studio["bt"]["spins"]):
            s.rotation_euler = (8.0 * t_of(f) * (1.0 if i == 0 else -1.1) + i, 0.0, 0.0)
            key(s, "rotation_euler", f)
    silt_s = drift_silt(LS + Vector((0.0, -0.3, 0.05)), (0.5, 0.3, 0.3), 260, 4, CUT2, CUT2B).build("StudioSilt", c_ls, M.silt)
    st = dict(boat=boat, weeds=[weeds], pads=[pads], xs_solid=[xsd["top"], xsd["face"]], studio=studio)
    frames, fstop = camera_frames(st)
    build_camera(c_cam, frames, fstop)
    key_scene_state(sc, (w_under, w_deep), sun, key_l, bounce)
    vis = {}

    def show(objs, spans):
        for o in objs:
            vis[o.name] = (o, tuple(spans))
    above = [(1, WHIP_IN - 1), (CUT4, FRAME_END)]
    show([sea, oh_pads], above)
    lake_spans = [(WHIP_IN, CUT1 - 1), (CUT2B, CUT4 - 1)]
    show([o for o in c_lake.all_objects if o.type in ("MESH", "CURVE") and o.name not in ("PerchFish",)], lake_spans)
    show([o for o in mh["root"].children_recursive if o.type == "MESH"] + bt_hook["parts"] + [line_hook, bub_hook, burst_hook],
         [(WHIP_IN, CUT1 - 1)])
    show([o for o in mb["root"].children_recursive if o.type == "MESH"] + bt_burn["parts"] + [line_burn, bub_burn],
         [(CUT2B, CUT3 - 1)])
    show([o for o in mf["root"].children_recursive if o.type == "MESH"] + bt_f8["parts"] + boat["parts"] +
         [rod["ob"], line_f8, bub_f8, burst_f8], [(CUT3, CUT4 - 1)])
    show([o for o in c_x.all_objects if o.type in ("MESH", "CURVE")], [(CUT1, CUT2 - 1)])
    show([o for o in c_ls.all_objects if o.type in ("MESH", "CURVE")], [(CUT2, CUT2B - 1)])
    groups = {}
    for o, spans in vis.values():
        groups.setdefault(spans, []).append(o)
    for spans, objs in groups.items():
        key_visibility(objs, list(spans))
    _ = (bed, school, silt, surf, shafts, plank, school_x, surf_x, shafts_x, plank_x, silt_s)
    hs, hcam, hcol = build_hud_scene()
    hud_materials(M)
    hud = Hud(hcam, hcol, M, frames)
    st.update(frames=frames, hud=hud)
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
