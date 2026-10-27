"""Record marlin masterclass: the all tackle record Pacific blue marlin and big game lure trolling off Kona, Hawaii.
Vertical 9:16, 24 fps, 55 s (1320 frames), Blender 5.2 EEVEE. Built to loop: the last frame is the first frame's
shot, and the voiceover's last line runs into its first.

Run with the Blender binary:
    blender --background --python record_marlin_masterclass.py -- [options]
or with the bpy Python module (pip install bpy):
    python record_marlin_masterclass.py [options]

Options:
    --save PATH        save the generated scene as a .blend file (fonts packed)
    --check            camera clearances (sea, boat, fish, ledge), camera acceleration and turn rate, cuts and
                       water surface crossings, overlay overlaps and off-screen boxes, subjects on screen, loop match
    --frames 1,300     render these frames as PNG stills into --outdir
    --outdir DIR       output directory for renders
    --scale PCT        resolution percentage (default 100 -> 1080x1920)
    --samples N        EEVEE render samples (default 32)
    --no-blur          switch motion blur off (quick tests)
    --events PATH      write overlay, action and voiceover timing as JSON (input for record_marlin_audio.py)

Everything is procedural (no imported assets). Facts in the on-screen text and the voiceover are listed with their
sources in SOURCES.md; the fish, the ledge, the depths, the boat, the spread and the fight are illustrative.

World (metres, +Z up, sea surface z = 0). Indigo blue Pacific water off Kona under a high tropical sun with trade
wind chop. A 43 ft sportfisher trolls a four lure spread from outriggers; a giant blue marlin hunts the drop off.
Overlays live in a second scene ("HUD") with its own static camera, composited over the 3D render.

Beats follow the voiceover clips (voiceover/vo1..vo5). Cuts are hard cuts, match cuts and whip pans (no wipes):
    1  hook: the spread from above whips down under water; a giant marlin engulfs a trolled resin lure in a burst
       of white water, then turns away on the line
    2  the drop off: a match cut to a cross section of a steep volcanic ledge; upwelling pushes bait and small tuna
       up the slope, where a marlin cruises the contour
    3  the spread: a jump cut to an aerial breakdown of the boat towing four lures (short and long corners, short
       and long riggers), then under water to a smoking resin lure
    4  the strike: a close up of the lever drag reel pushed to strike as line pours off, then a hard cut to the
       marlin tail walking across the surface
    5  outro: a whip pan back to the spread from above, which is frame 1
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
FRAME_END = 1320
LOOP_S = FRAME_END / FPS             # 55 s: the sea, the boat and the music all repeat with this period
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

# water: deep, clear indigo blue Pacific (fog colour by view direction, down .. up)
UW_RAMP = [(0.0, srgb(1, 6, 22)), (0.36, srgb(4, 22, 62)), (0.52, srgb(10, 52, 120)), (0.72, srgb(30, 104, 170)),
           (1.0, srgb(96, 176, 220))]
VIS = 26.0                           # fog e-folding distance under water (m): very clear blue water
AIR_HAZE = 900.0                     # sea haze e-folding distance above water (m): clear trade wind air
SUN_DIR = Vector((-0.35, 0.55, 0.76)).normalized()     # a high tropical sun (about 50 degrees up)


def t_of(f):
    """Scene time in seconds at frame f, zero at frame 1, so frame 1321 would equal frame 1 (the loop)."""
    return (f - 1) / FPS


# --------------------------------------------------------------------------
# Voiceover timing. vo_timing.json (written by record_marlin_audio.py --measure) holds the measured lengths and
# sentence starts. The hook starts on frame 1 and the outro ends on the last frame, so the loop joins mid sentence:
# "...and you'll see why... In 1982, Jay de Beaubien hooked..."
# --------------------------------------------------------------------------
VO_LEN = {1: 12.51, 2: 10.53, 3: 11.23, 4: 14.29, 5: 3.63}
VO_TEXT = {
    1: "In 1982, Jay de Beaubien hooked a 1,376 pound Pacific blue marlin off Kona. It became the all tackle world "
       "record for the species, landed in under an hour.",
    2: "Grander marlin don't roam open water aimlessly. They hunt steep underwater volcanic drop offs, where deep "
       "current upwellings push bait and small tuna right into their path.",
    3: "To trigger a strike, run a staggered four lure spread at seven to nine knots. Large skirted resin lures on "
       "the short corners create heavy smoke trails that bring giants up from the depths.",
    4: "When a 1,000 pound fish eats, lock down 130 pound heavy tackle with 40 pounds of drag pressure to set "
       "massive double hooks into its hard jaw. That's about thirty percent of the line's breaking strength, the "
       "classic strike drag.",
    5: "Master the spread, fish the ledges, and you'll see why...",
}
VO_GAPS = (10, 10, 10)               # frames of breath after clips 1, 2 and 3; the rest is the tail walk


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
VO_START[5] = FRAME_END + 1 - int(round(VO_LEN[5] * FPS))
if VO_START[5] < VO_START[4] + int(round(VO_LEN[4] * FPS)) + 4:
    print("WARNING: the voiceover is longer than the film")


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
WHIP_IN = 7                          # frames 1..6: the spread from above (the loop's join), whip pan down, cut under
CUT1A = P(1, 1) - 4                              # hard cut inside the hook: a second angle, from below against the sun
CUT1 = (vo_end(1) + VO_START[2]) // 2            # match cut: the marlin's flank -> the marlin on the ledge diagram
CUT1W = CUT1 + 22                                # hard cut: the marlin close -> the whole drop off diagram
CUT2 = (vo_end(2) + VO_START[3]) // 2            # jump cut to the aerial spread breakdown
CUT2B = P(3, 1) - 4                              # hard cut inside the spread beat: aerial -> a smoking lure, under water
CUT3 = (vo_end(3) + VO_START[4]) // 2            # rapid cut to the reel close up
CUT3B = VO_START[4] + int(round(6.4 * FPS))      # hard cut: the reel -> the marlin tail walking
CUT4 = VO_START[5] - 6                           # whip pan back to the spread from above (the loop's join)
CUTS = [WHIP_IN, CUT1A, CUT1, CUT1W, CUT2, CUT2B, CUT3, CUT3B, CUT4]
SEGMENTS = [(1, CUT1 - 1), (CUT1, CUT2 - 1), (CUT2, CUT3 - 1), (CUT3, CUT4 - 1), (CUT4, FRAME_END)]
LOOP_HOLD = 26                       # the last frames hold the overhead shot that frame 1 opens on


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
    w = bpy.data.worlds.new("KonaWorld")
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
    sky.air_density = 1.0
    try:
        sky.aerosol_density = 1.2
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
    skyc = mix_rgb(nt, map_range(nt, view["Z"], 0.0, -0.03), skyc, srgb(8, 28, 66))     # the sea past the horizon
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
    rng = random.Random(seed)
    bm = bmesh.new()
    rand = bm.verts.layers.float.new("rand")
    for _ in range(count):
        p = Vector((rng.uniform(-1, 1) * size[0], rng.uniform(-1, 1) * size[1], rng.uniform(-1, 1) * size[2]))
        s = rad * rng.uniform(0.4, 1.5)
        res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=s, matrix=Matrix.Translation(p))
        r = rng.random()
        for v in res["verts"]:
            v[rand] = r
    ob = finish(name, bm, col, [mat])
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
        """Deep Pacific off Kona seen from above: indigo blue water glowing a little in the sun, sky reflections from
        the Fresnel term, ripples, a few whitecaps on the sharp crests. (World position, not the stored "rest"
        attribute, which came out corrupt away from the grid centre in an earlier build.)"""
        wp = separate(nt, world_pos(nt))
        p = combine(nt, wp["X"], wp["Y"], 0.0)
        crest = attr(nt, "crest", "Fac")
        rip = looped_noise(nt, vmath(nt, "MULTIPLY", p, (1.0, 1.0, 1.0)), (0.6, 0.25, 0.0), 1.4, 6.0, 0.6)
        rip2 = looped_noise(nt, vmath(nt, "MULTIPLY", p, (1.0, 1.0, 1.0)), (-0.3, 0.5, 0.0), 6.0, 4.0, 0.55)
        foam_n = looped_noise(nt, vmath(nt, "MULTIPLY", p, (0.6, 1.4, 1.0)), (0.3, 0.12, 0.0), 0.9, 5.0, 0.65)
        cap = nmath(nt, "MULTIPLY", map_range(nt, crest, 0.45, 0.8), map_range(nt, foam_n, 0.5, 0.64))
        streak = nmath(nt, "MULTIPLY", map_range(nt, noise(nt, vmath(nt, "MULTIPLY", p, (0.25, 2.2, 1.0)), 1.2, 4.0, 0.6),
                                                 0.6, 0.7), 0.35)
        foam = nmath(nt, "MINIMUM", nmath(nt, "ADD", cap, nmath(nt, "MULTIPLY", streak, map_range(nt, crest, 0.0, 0.5))), 1.0)
        dist = nt.nodes.new("ShaderNodeCameraData").outputs["View Distance"]
        foam = nmath(nt, "MULTIPLY", foam, map_range(nt, dist, 300.0, 900.0, 1.0, 0.25))
        water = color_ramp(nt, map_range(nt, crest, -0.4, 0.6), [(0.0, srgb(3, 14, 48)), (0.7, srgb(6, 30, 88)),
                                                                   (1.0, srgb(18, 74, 140))])
        col = mix_rgb(nt, foam, water, srgb(214, 222, 224))
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(col, b.inputs["Base Color"])
        rough = nmath(nt, "ADD", map_range(nt, foam, 0.0, 1.0, 0.13, 0.6, smooth=False),
                      map_range(nt, dist, 200.0, 1400.0, 0.0, 0.25))           # far grazing facets: no comb of glints
        _feed(nt, b.inputs["Roughness"], rough)
        b.inputs["IOR"].default_value = 1.333
        b.inputs["Specular IOR Level"].default_value = 0.6
        h = nmath(nt, "ADD", nmath(nt, "MULTIPLY", rip, 1.0), nmath(nt, "MULTIPLY", rip2, 0.35))
        bstr = map_range(nt, dist, 20.0, 400.0, 0.32, 0.12)
        bp = nt.nodes.new("ShaderNodeBump")
        nt.links.new(bstr, bp.inputs["Strength"])
        bp.inputs["Distance"].default_value = 0.08
        nt.links.new(h, bp.inputs["Height"])
        nt.links.new(bp.outputs[0], b.inputs["Normal"])
        return b.outputs[0]
    M.sea = mat("Sea", sea)

    def wake(nt):
        """White water astern: dense churn behind the transom thinning into streaks down the wake (alpha)."""
        p = attr(nt, "rest", "Vector")              # (the wake object's own rest attribute, set in its GN)
        o = separate(nt, p)
        along = nmath(nt, "ABSOLUTE", o["X"])
        wdt = nmath(nt, "ADD", 2.2, nmath(nt, "MULTIPLY", along, 0.18))
        across = nmath(nt, "DIVIDE", nmath(nt, "ABSOLUTE", o["Y"]), wdt)
        n1 = looped_noise(nt, vmath(nt, "MULTIPLY", p, (0.6, 1.2, 1.0)), (0.0, 0.0, 0.0), 1.3, 6.0, 0.65)
        n2 = noise(nt, vmath(nt, "MULTIPLY", p, (0.6, 1.6, 1.0)), 1.8, 5.0, 0.6)
        core = nmath(nt, "MULTIPLY", map_range(nt, along, 0.0, 18.0, 1.0, 0.0), map_range(nt, across, 0.6, 0.15))
        edge = nmath(nt, "MULTIPLY", map_range(nt, nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", across, 0.85)), 0.15, 0.0),
                     map_range(nt, along, 0.0, 55.0, 0.8, 0.0))
        foam = nmath(nt, "MULTIPLY", nmath(nt, "MAXIMUM", core, edge), map_range(nt, nmath(nt, "ADD", n1, nmath(nt, "MULTIPLY", n2, 0.4)), 0.45, 0.75))
        foam = nmath(nt, "MULTIPLY", foam, map_range(nt, along, 0.0, 1.5))
        return psdf(nt, srgb(226, 234, 236), rough=0.5, alpha=nmath(nt, "MINIMUM", nmath(nt, "MULTIPLY", foam, 0.9), 1.0))
    M.wake = mat("WakeFoam", wake)

    def lure_foam(nt):
        """The short streak of white water and bubbles a popping lure leaves on the surface (alpha)."""
        p = attr(nt, "rest", "Vector")
        o = separate(nt, p)
        along = nmath(nt, "ABSOLUTE", o["X"])
        across = nmath(nt, "DIVIDE", nmath(nt, "ABSOLUTE", o["Y"]), nmath(nt, "ADD", 0.12, nmath(nt, "MULTIPLY", along, 0.08)))
        n1 = looped_noise(nt, vmath(nt, "MULTIPLY", p, (3.0, 6.0, 1.0)), (0.0, 0.0, 0.0), 1.0, 5.0, 0.65)
        k = nmath(nt, "MULTIPLY", nmath(nt, "POWER", map_range(nt, along, 0.0, 5.0, 1.0, 0.0), 2.0), map_range(nt, across, 1.0, 0.3))
        foam = nmath(nt, "MULTIPLY", k, map_range(nt, n1, 0.4, 0.7))
        return psdf(nt, srgb(232, 240, 244), rough=0.45, alpha=nmath(nt, "MINIMUM", nmath(nt, "MULTIPLY", foam, 0.85), 1.0))
    M.lure_foam = mat("LureFoam", lure_foam)

    def surface_under(nt):
        """The surface from below: a bright rippling window overhead, darker total-reflection rim."""
        p = tex_coord(nt, "Object")
        rip = looped_noise(nt, p, (0.4, 0.2, 0.0), 0.35, 5.0, 0.6)
        inc = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Incoming"])["Z"]
        window = map_range(nt, nmath(nt, "ABSOLUTE", inc), 0.62, 0.8)
        col = mix_rgb(nt, window, srgb(6, 32, 86), srgb(170, 222, 246))
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
        em.inputs["Color"].default_value = (*srgb(200, 236, 255), 1.0)
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

    # --- the blue marlin ----------------------------------------------------------------------------------------
    def marlin(nt):
        """Pacific blue marlin: a deep cobalt to blue-black back, an electric blue iridescent band along the upper
        flank, silvery white lower sides and belly, and rows of pale cobalt vertical bars that light up when the
        fish is excited (scene "lit"); a slick coat of slime."""
        z, x = attr(nt, "zrel", "Fac"), attr(nt, "xrel", "Fac")
        p = tex_coord(nt, "Object")
        pz = separate(nt, p)
        base = color_ramp(nt, z, [(0.0, srgb(206, 212, 222)), (0.26, srgb(186, 194, 210)), (0.38, srgb(118, 132, 162)),
                                  (0.47, srgb(28, 50, 104)), (0.56, srgb(8, 20, 60)), (1.0, srgb(2, 6, 22))])
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.45
        irid = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", map_range(nt, z, 0.46, 0.56), map_range(nt, z, 0.78, 0.64)),
                     map_range(nt, lw.outputs["Facing"], 0.1, 0.7, 0.5, 1.0))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", irid, 0.6), base, srgb(30, 120, 230))
        wav = nmath(nt, "MULTIPLY", pz["X"], 13.0)                  # about 15 bars along the body
        bars = nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "ADD", wav,
                     nmath(nt, "MULTIPLY", noise(nt, p, 3.0, 2.0), 0.6)))), 18.0)
        bars = nmath(nt, "MULTIPLY", bars, map_range(nt, noise(nt, vmath(nt, "MULTIPLY", p, (1.0, 1.0, 6.0)), 4.0, 3.0), 0.35, 0.65))
        flank = nmath(nt, "MULTIPLY", map_range(nt, z, 0.28, 0.42), map_range(nt, z, 0.74, 0.6))
        flank = nmath(nt, "MULTIPLY", flank, nmath(nt, "MULTIPLY", map_range(nt, x, 0.12, 0.22), map_range(nt, x, 0.88, 0.7)))
        lit = scene_attr(nt, "lit")
        barw = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", bars, flank), nmath(nt, "MULTIPLY_ADD", lit, 0.45, 0.25))
        col = mix_rgb(nt, barw, col, srgb(96, 160, 236))
        seam = attr(nt, "seam", "Fac")
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", seam, 0.8), col, srgb(14, 16, 20))
        n = noise(nt, p, 40.0, 4.0, 0.5)
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(uw_base(nt, col, caustic=0.3), b.inputs["Base Color"])
        _feed(nt, b.inputs["Metallic"], map_range(nt, z, 0.2, 0.6, 0.5, 0.15))
        _feed(nt, b.inputs["Roughness"], map_range(nt, n, 0.0, 1.0, 0.26, 0.38, smooth=False))
        b.inputs["Coat Weight"].default_value = 0.45
        b.inputs["Coat Roughness"].default_value = 0.06
        b.inputs["Subsurface Weight"].default_value = 0.05
        b.inputs["Subsurface Scale"].default_value = 0.01
        b.inputs["Emission Color"].default_value = (*srgb(90, 170, 255), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", barw, nmath(nt, "MULTIPLY", lit, 0.22)), b.inputs["Emission Strength"])
        nt.links.new(bump(nt, nmath(nt, "MULTIPLY_ADD", n, 0.2, nmath(nt, "MULTIPLY", attr(nt, "gill", "Fac"), -1.0)), 0.25, 0.004),
                     b.inputs["Normal"])
        return b.outputs[0]
    M.marlin = mat("MarlinSkin", marlin)
    M.bill = mat("MarlinBill", lambda nt: psdf(nt, uw_base(nt, color_ramp(nt, attr(nt, "zrel", "Fac"),
                 [(0.0, srgb(70, 80, 96)), (1.0, srgb(10, 16, 30))]), caustic=0.2), rough=0.35, coat=0.4))

    def fin(name, base, edge, alpha=1.0, rays=60.0, metal=0.2):
        def build(nt):
            g = separate(nt, tex_coord(nt, "Generated"))
            e = nmath(nt, "MAXIMUM", nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", g["Z"], 0.5)),
                      nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", g["X"], 0.5)))
            col = color_ramp(nt, nmath(nt, "MULTIPLY", e, 2.0), [(0.0, base), (0.75, base), (1.0, edge)])
            o = separate(nt, tex_coord(nt, "Object"))
            ray = nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY", o["X"], rays))), 6.0)
            col = mix_rgb(nt, nmath(nt, "MULTIPLY", ray, 0.2), col, srgb(10, 14, 22))
            return psdf(nt, uw_base(nt, col, caustic=0.2), rough=0.3, metal=metal, coat=0.5, coat_rough=0.1, alpha=alpha)
        return mat(name, build)
    M.fin_dark = fin("MarlinFinDark", srgb(14, 26, 60), srgb(6, 10, 24))
    M.fin_dorsal1 = fin("MarlinDorsal", srgb(16, 40, 104), srgb(6, 12, 34), rays=90.0, metal=0.3)
    M.fin_dorsal2 = fin("MarlinSecondDorsal", srgb(14, 30, 72), srgb(8, 14, 30))
    M.fin_pale = fin("MarlinPaleFin", srgb(150, 160, 176), srgb(60, 70, 90), metal=0.4)

    def eye(nt):
        """Fish eye: a pale silvery outer ring (the sclera the viewer reads), a thin gold iris and a large black
        pupil, under a clear glossy cornea. "eyer" is the angle from the eye's axis (0 at the pupil's centre)."""
        r = attr(nt, "eyer", "Fac")
        col = color_ramp(nt, r, [(0.0, srgb(2, 2, 3)), (0.54, srgb(2, 2, 3)), (0.555, srgb(150, 118, 52)),
                                 (0.575, srgb(196, 160, 84)), (0.595, srgb(214, 216, 210)), (0.68, srgb(232, 234, 230)),
                                 (0.73, srgb(150, 156, 162)), (0.79, srgb(40, 44, 52)), (1.0, srgb(24, 28, 34))])
        return psdf(nt, col, rough=0.04, metal=0.1, coat=1.0, coat_rough=0.02, spec=0.8)
    M.eye = mat("FishEye", eye)

    def small_eye(nt):
        """Small fish eye: silvery white sclera around a black pupil."""
        r = attr(nt, "eyer", "Fac")
        col = color_ramp(nt, r, [(0.0, srgb(2, 2, 3)), (0.66, srgb(2, 2, 3)), (0.69, srgb(200, 170, 96)),
                                 (0.72, srgb(226, 228, 224)), (0.87, srgb(214, 218, 216)), (1.0, srgb(120, 126, 130))])
        return psdf(nt, col, rough=0.05, coat=1.0, coat_rough=0.02, spec=0.8)
    M.small_eye = mat("SmallFishEye", small_eye)

    def skipjack(nt):
        """Skipjack tuna: dark purple-blue back, silver sides and belly with three to five dark stripes along the
        lower flank."""
        z = attr(nt, "zrel", "Fac")
        base = color_ramp(nt, z, [(0.0, srgb(226, 228, 232)), (0.5, srgb(190, 198, 210)), (0.62, srgb(60, 70, 120)),
                                  (1.0, srgb(16, 22, 56))])
        stripes = nmath(nt, "MULTIPLY", nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY", z, 26.0))), 6.0),
                        nmath(nt, "MULTIPLY", map_range(nt, z, 0.08, 0.16), map_range(nt, z, 0.5, 0.42)))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", stripes, 0.8), base, srgb(40, 44, 60))
        return psdf(nt, uw_base(nt, col, caustic=0.25), rough=0.22, metal=0.45, coat=0.6, coat_rough=0.08)
    M.skipjack = mat("Skipjack", skipjack)
    M.small_fin = mat("SmallFin", lambda nt: psdf(nt, uw_base(nt, srgb(110, 120, 120), caustic=0.2), rough=0.4, alpha=0.9))

    def baitfish(nt):
        z = attr(nt, "zrel", "Fac")
        col = color_ramp(nt, z, [(0.0, srgb(210, 214, 218)), (0.55, srgb(160, 176, 184)), (0.7, srgb(40, 70, 90)),
                                 (1.0, srgb(20, 40, 56))])
        return psdf(nt, uw_base(nt, col, caustic=0.3), rough=0.25, metal=0.5, coat=0.4)
    M.baitfish = mat("Baitfish", baitfish)

    # --- tackle ------------------------------------------------------------------------------------------------
    M.reel_gold = mat("ReelGold", lambda nt: psdf(nt, srgb(214, 168, 70), rough=0.22, metal=1.0, coat=0.3))
    M.reel_dark = mat("ReelDark", lambda nt: psdf(nt, srgb(26, 26, 28), rough=0.35, metal=0.6))
    M.chrome = mat("Chrome", lambda nt: psdf(nt, srgb(200, 204, 210), rough=0.12, metal=1.0))
    M.blank = mat("RodBlank", lambda nt: psdf(nt, srgb(214, 216, 210), rough=0.22, coat=0.8, coat_rough=0.05))  # pale: reads against the hull
    M.wrap = mat("RodWrap", lambda nt: psdf(nt, srgb(170, 30, 30), rough=0.3, coat=0.6))
    M.grip = mat("RodGrip", lambda nt: psdf(nt, srgb(30, 30, 32), rough=0.7, normal=bump(nt, noise(nt, tex_coord(nt), 200.0), 0.3)))
    M.line = mat("Line", lambda nt: psdf(nt, srgb(214, 220, 140), rough=0.3, emission=srgb(214, 220, 140), emission_strength=0.25))
    M.hook = mat("Hook", lambda nt: psdf(nt, srgb(150, 150, 156), rough=0.25, metal=1.0))

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

    # --- lures --------------------------------------------------------------------------------------------------
    def resin(color):
        """A resin lure head: clear cast resin over a coloured abalone like insert, a pale glow where the sun gets in."""
        def build(nt):
            p = tex_coord(nt, "Object")
            n = noise(nt, vmath(nt, "MULTIPLY", p, (60.0, 60.0, 60.0)), 1.0, 4.0, 0.6)
            ins = mix_rgb(nt, map_range(nt, n, 0.3, 0.7), srgb(*color), srgb(230, 236, 240))
            return psdf(nt, ins, rough=0.04, coat=1.0, coat_rough=0.01, transmission=0.35, ior=1.5, spec=0.9,
                        emission=srgb(*color), emission_strength=0.25)
        return build
    M.resin = [mat(f"LureResin{i}", resin(c)) for i, c in enumerate(((60, 140, 255), (255, 90, 160), (60, 230, 170),
                                                                      (190, 200, 214)))]

    def skirt(c1, c2):
        def build(nt):
            o = separate(nt, tex_coord(nt, "Object"))
            st = nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY", attr(nt, "ang", "Fac"), 31.4))), 3.0)
            col = mix_rgb(nt, st, srgb(*c1), srgb(*c2))
            flake = map_range(nt, voronoi(nt, tex_coord(nt, "Object"), 400.0), 0.15, 0.0)
            col = mix_rgb(nt, nmath(nt, "MULTIPLY", flake, 0.5), col, srgb(255, 255, 255))
            _ = o
            return psdf(nt, uw_base(nt, col, caustic=0.2), rough=0.35, coat=0.6, sss=0.2, sss_scale=0.004)
        return build
    M.skirts = [mat(f"LureSkirt{i}", skirt(c1, c2)) for i, (c1, c2) in enumerate(((((20, 40, 160)), (210, 220, 240)),
                                                                                  ((230, 40, 120), (40, 20, 60)),
                                                                                  ((30, 200, 120), (230, 220, 60)),
                                                                                  ((150, 160, 180), (30, 60, 200))))]
    M.bubbles = mat("BubbleTrail", lambda nt: psdf(nt, srgb(236, 246, 255), rough=0.1, alpha=0.55, transmission=0.0,
                                                   emission=srgb(200, 226, 255), emission_strength=0.6))

    # --- the drop off diagram -----------------------------------------------------------------------------------
    def basalt(nt):
        """Volcanic rock of the drop off: dark basalt with lighter ledges and a little coral crust on the shelf."""
        p = world_pos(nt)
        o = separate(nt, p)
        n = noise(nt, vmath(nt, "MULTIPLY", p, (0.7, 0.7, 0.7)), 1.0, 6.0, 0.6)
        strata = map_range(nt, nmath(nt, "SINE", nmath(nt, "ADD", nmath(nt, "MULTIPLY", o["Z"], 1.6), nmath(nt, "MULTIPLY", n, 3.0))), 0.6, 1.0)
        col = color_ramp(nt, n, [(0.0, srgb(10, 10, 14)), (0.5, srgb(26, 26, 32)), (1.0, srgb(52, 50, 56))])
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", strata, 0.4), col, srgb(70, 66, 70))
        shelf = map_range(nt, o["Z"], -14.0, -6.0)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", shelf, map_range(nt, noise(nt, p, 3.0, 4.0), 0.5, 0.65)), col, srgb(120, 110, 86))
        return psdf(nt, uw_base(nt, col, caustic=0.6), rough=0.8, normal=bump(nt, n, 0.6, 0.08))
    M.basalt = mat("Basalt", basalt)

    def cutface(nt):
        """The cut face of the cross section, drawn flat: dark rock with a fine diagonal hatch (a diagram, not rock)."""
        o = separate(nt, world_pos(nt))
        hatch = nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY",
                      nmath(nt, "ADD", o["X"], o["Z"]), 9.0))), 30.0)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", hatch, 0.35), srgb(16, 18, 26), srgb(70, 90, 120))
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(col, em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0
        return em.outputs[0]
    M.cutface = new_material("CutFace", cutface)

    def upwell(nt):
        """Upwelling flow arrows: a soft cyan glow (a diagram element), faded by object alpha."""
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(110, 230, 255), 1.0)
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        nt.links.new(nmath(nt, "MULTIPLY", oi.outputs["Alpha"], 3.0), em.inputs["Strength"])
        return em.outputs[0]
    M.upwell = new_material("Upwelling", upwell)

    def depth_column(nt):
        """Backdrop of the cross section: sunlit blue near the surface fading to black at depth (by world height)."""
        z = separate(nt, world_pos(nt))["Z"]
        col = color_ramp(nt, map_range(nt, z, -90.0, 0.0, 0.0, 1.0, smooth=False),
                         [(0.0, srgb(0, 2, 8)), (0.45, srgb(2, 12, 40)), (0.8, srgb(8, 46, 110)), (1.0, srgb(40, 120, 200))])
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(col, em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0
        return em.outputs[0]
    M.column = new_material("DepthColumn", depth_column)
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
# The Pacific blue marlin (Makaira nigricans). Proportions after the species' descriptions: a long, round spear
# (the upper jaw); an elongate body, deepest just behind the head with a high nape; a tall, pointed first dorsal
# fin falling quickly to a long low rear part, a small second dorsal; two anal fins; long, sickle pectorals that
# fold flat against the body; thin pelvic fins; a tall, stiff crescent tail on a slender stock with two keels per
# side; cobalt vertical bars on the flanks. Modelled about 4.9 m from the bill tip to the tail tips (illustrative;
# not the record fish to scale).
# --------------------------------------------------------------------------
MARLIN_L = 3.85                      # tip of the lower jaw to the base of the tail (m)
BILL_L = 0.82                        # the spear beyond the lower jaw


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


def marlin_curves():
    L = MARLIN_L
    pts_top = [(0.0, 0.004), (0.03, 0.03), (0.08, 0.066), (0.15, 0.1), (0.22, 0.118), (0.3, 0.121), (0.42, 0.112),
               (0.55, 0.092), (0.7, 0.062), (0.82, 0.036), (0.92, 0.02), (1.0, 0.014)]
    pts_bot = [(0.0, -0.01), (0.03, -0.032), (0.08, -0.064), (0.15, -0.09), (0.22, -0.102), (0.3, -0.105),
               (0.42, -0.098), (0.55, -0.08), (0.7, -0.052), (0.82, -0.03), (0.92, -0.017), (1.0, -0.012)]
    pts_wid = [(0.0, 0.0), (0.03, 0.022), (0.08, 0.044), (0.15, 0.06), (0.25, 0.068), (0.35, 0.066), (0.5, 0.056),
               (0.7, 0.038), (0.85, 0.024), (0.93, 0.019), (1.0, 0.016)]
    top_u, bot_u, wid_u = (pchip([(u, v * L) for u, v in pts]) for pts in (pts_top, pts_bot, pts_wid))

    def u_of(x):
        return (L / 2 - x) / L
    return (lambda x: top_u(u_of(x))), (lambda x: bot_u(u_of(x))), (lambda x: wid_u(u_of(x))), u_of


def marlin_surface(x, z_rel, side=1):
    """A point on the skin at body position x, at relative height z_rel (-1 belly .. 1 back) on one side."""
    top, bot, wid, _u = marlin_curves()
    zc, hh, w = (top(x) + bot(x)) / 2, (top(x) - bot(x)) / 2, wid(x)
    s = max(-1.0, min(1.0, z_rel))
    y = w * max(0.0, 1.0 - abs(s) ** 2.1) ** (1 / 2.1)
    return Vector((x, side * y, zc + hh * s))


def marlin_mesh(M):
    L = MARLIN_L
    top, bot, wid, u_of = marlin_curves()
    bm = bmesh.new()
    zl, xl = bm.verts.layers.float.new("zrel"), bm.verts.layers.float.new("xrel")
    sl, gl = bm.verts.layers.float.new("seam"), bm.verts.layers.float.new("gill")
    er = bm.verts.layers.float.new("eyer")
    nr, na, pw = 150, 64, 2.1
    rings = []
    for i in range(1, nr + 1):
        u = (i / nr) ** 1.2
        x = L / 2 - u * L
        t, b_, w = top(x), bot(x), wid(x)
        zc, hh = (t + b_) / 2, (t - b_) / 2
        keel = smoothstep(0.86, 0.92, u)
        ring = []
        for k in range(na):
            a = 2 * math.pi * k / na
            c, s = math.cos(a), math.sin(a)
            y = w * math.copysign(abs(c) ** (2 / pw), c)
            z = zc + hh * math.copysign(abs(s) ** (2 / pw), s)
            kz = (z - zc) / max(hh, 1e-4)
            ridge = keel * 0.02 * (math.exp(-((kz - 0.3) / 0.14) ** 2) + math.exp(-((kz + 0.3) / 0.14) ** 2))   # two keels
            y += math.copysign(ridge, c)
            v = bm.verts.new((x, y, z))
            v[zl] = min(1.0, max(0.0, (z - b_) / max(1e-6, t - b_)))
            v[xl] = 1.0 - u
            if u < 0.09:                                  # the mouth: a seam from the jaw tip back under the eye
                sz = -0.1 + 0.05 * (u / 0.09)
                d = (kz - sz) * hh
                v[sl] = math.exp(-(d / 0.007) ** 2) * smoothstep(0.09, 0.075, u) * smoothstep(0.0, 0.25, abs(c))
            ug = 0.165 + 0.015 * math.cos(kz * 1.4) - 0.025 * max(0.0, -kz) ** 1.5
            if abs(u - ug) < 0.04 and kz < 0.6:
                d = (u - ug) * L
                v[gl] = math.exp(-(d / 0.01) ** 2) * smoothstep(0.6, 0.42, kz)
            ring.append(v)
        rings.append(ring)
    for ra, rb in zip(rings, rings[1:]):
        for k in range(na):
            bm.faces.new((ra[k], ra[(k + 1) % na], rb[(k + 1) % na], rb[k]))
    x0 = L / 2
    tip = bm.verts.new((x0 + 0.004, 0.0, (top(x0) + bot(x0)) / 2))
    tip[zl], tip[xl] = 0.4, 1.0
    for k in range(na):
        bm.faces.new((rings[0][(k + 1) % na], rings[0][k], tip))
    xe = -L / 2
    tail = bm.verts.new((xe - 0.01, 0.0, (top(xe) + bot(xe)) / 2))
    tail[zl], tail[xl] = 0.5, 0.0
    for k in range(na):
        bm.faces.new((rings[-1][k], rings[-1][(k + 1) % na], tail))
    for f in bm.faces:
        f.smooth = True
    # the spear: an elliptical cone from the upper jaw, slightly upturned, rough textured (its own material slot)
    bill = []
    zb = top(x0 - 0.05) * 0.35
    for i in range(25):
        s_ = i / 24
        xb = x0 - 0.12 + s_ * (BILL_L + 0.12)
        r = 0.032 * (1.0 - s_) ** 0.85 + 0.002
        ring = [bm.verts.new((xb, 1.25 * r * math.cos(2 * math.pi * k / 16), zb + 0.012 * s_ * s_ + r * math.sin(2 * math.pi * k / 16)))
                for k in range(16)]
        for v in ring:
            v[zl] = 0.5 + 0.5 * math.sin(math.atan2(v.co.z - zb, v.co.y + 1e-6))
        bill.append(ring)
    for ra, rb in zip(bill, bill[1:]):
        for k in range(16):
            f = bm.faces.new((ra[k], ra[(k + 1) % 16], rb[(k + 1) % 16], rb[k]))
            f.material_index = 2
            f.smooth = True
    btip = bm.verts.new((x0 + BILL_L + 0.006, 0.0, zb + 0.012))
    for k in range(16):
        f = bm.faces.new((bill[-1][k], bill[-1][(k + 1) % 16], btip))
        f.material_index = 2
    ex = L / 2 - 0.075 * L
    for s in (1, -1):
        c = marlin_surface(ex, 0.3, s) - Vector((0.0, s * 0.008, 0.0))
        faces = ellipsoid(bm, c, (0.034, 0.016, 0.032), subdiv=5, mat=1)
        axis = Vector((0.08, s * 1.0, 0.1)).normalized()
        for v in (v for f in faces for v in f.verts):
            d = (v.co - c).normalized()
            v[er] = math.acos(max(-1.0, min(1.0, d.dot(axis)))) / (math.pi / 2)
    return fish_mesh("Marlin", bm, [M.marlin, M.eye, M.bill])


def make_marlin(col, M, name="Marlin"):
    L = MARLIN_L
    top, bot, wid, u_of = marlin_curves()
    me = marlin_mesh(M)
    body = bpy.data.objects.new(name, me)
    col.objects.link(body)

    def xu(u):
        return L / 2 - u * L
    fb = bmesh.new()
    x1 = xu(0.165)                                       # first dorsal: a tall pointed lobe, then long and low
    soft_fin(fb, [fin_base(top, x1, 0.02), (x1 - 0.03, top(x1) + 0.42), (x1 - 0.14, top(x1 - 0.14) + 0.62),
                  (x1 - 0.22, top(x1 - 0.22) + 0.34), (x1 - 0.36, top(x1 - 0.36) + 0.16), (x1 - 0.9, top(x1 - 0.9) + 0.11),
                  (x1 - 1.5, top(x1 - 1.5) + 0.075), (x1 - 1.72, top(x1 - 1.72) + 0.03), fin_base(top, x1 - 1.74, 0.02)],
             0.012, 0, samples=10)
    x2 = xu(0.865)                                       # small second dorsal
    soft_fin(fb, [fin_base(top, x2, 0.01), (x2 - 0.02, top(x2) + 0.07), (x2 - 0.09, top(x2) + 0.06),
                  fin_base(top, x2 - 0.1, 0.01)], 0.008, 1, samples=5)
    xa = xu(0.6)                                         # first anal fin, pointed
    soft_fin(fb, [fin_base(bot, xa, 0.02), (xa - 0.04, bot(xa) - 0.2), (xa - 0.12, bot(xa) - 0.24),
                  (xa - 0.2, bot(xa - 0.2) - 0.08), (xa - 0.42, bot(xa - 0.42) - 0.03), fin_base(bot, xa - 0.44, 0.02)],
             0.01, 1, samples=8)
    xa2 = xu(0.865)
    soft_fin(fb, [fin_base(bot, xa2, 0.01), (xa2 - 0.02, bot(xa2) - 0.065), (xa2 - 0.09, bot(xa2) - 0.055),
                  fin_base(bot, xa2 - 0.1, 0.01)], 0.008, 1, samples=5)
    xt = -L / 2                                          # the tall crescent tail
    soft_fin(fb, [(xt + 0.09, 0.03), (xt - 0.04, 0.2), (xt - 0.22, 0.48), (xt - 0.44, 0.72), (xt - 0.47, 0.73),
                  (xt - 0.38, 0.58), (xt - 0.22, 0.28), (xt - 0.15, 0.06), (xt - 0.14, 0.0), (xt - 0.15, -0.06),
                  (xt - 0.22, -0.28), (xt - 0.38, -0.58), (xt - 0.47, -0.73), (xt - 0.44, -0.72), (xt - 0.22, -0.48),
                  (xt - 0.04, -0.2), (xt + 0.09, -0.03)], 0.016, 2, samples=6)
    fins_me = fish_mesh(name + "Fins", fb, [M.fin_dorsal1, M.fin_dorsal2, M.fin_dark])
    fins = bpy.data.objects.new(name + "Fins", fins_me)
    col.objects.link(fins)
    fins.parent = body
    paired = []
    for s in (1, -1):                                    # long sickle pectorals, held out a little from the side
        pb = bmesh.new()
        soft_fin(pb, [(0.04, 0.03), (-0.1, 0.06), (-0.4, 0.055), (-0.7, 0.01), (-0.76, -0.02), (-0.5, -0.04),
                      (-0.18, -0.06), (0.04, -0.045)], 0.014, 0, samples=6)
        pm = fish_mesh(f"{name}Pectoral{'L' if s > 0 else 'R'}", pb, [M.fin_dark])
        po = bpy.data.objects.new(pm.name, pm)
        col.objects.link(po)
        po.parent = body
        po.location = marlin_surface(xu(0.18), -0.35, s) - Vector((0.0, s * 0.012, 0.0))
        po.rotation_euler = (s * math.radians(30), math.radians(14), s * math.radians(-30))
        paired.append(po)
        vb = bmesh.new()                                 # thin pelvic fins
        soft_fin(vb, [(0.01, 0.0), (-0.1, -0.012), (-0.42, -0.03), (-0.3, -0.008), (0.0, 0.01)], 0.006, 0, samples=5)
        vm = fish_mesh(f"{name}Pelvic{'L' if s > 0 else 'R'}", vb, [M.fin_dark])
        vo = bpy.data.objects.new(vm.name, vm)
        col.objects.link(vo)
        vo.parent = body
        vo.location = marlin_surface(xu(0.21), -0.95, s) - Vector((0.0, s * 0.01, -0.005))
        vo.rotation_euler = (s * math.radians(-30), math.radians(-8), s * math.radians(-5))
        paired.append(vo)
    root = empty_object(name + "Rig", col, size=0.5)
    root.rotation_mode = "QUATERNION"
    body.parent = root
    beat = empty_object(name + "Beat", col, size=0.05)
    ng = ctrl_swim_group(beat, L, name + "Swim")
    for ob in (body, fins):
        ob.modifiers.new("Swim", "NODES").node_group = ng
    sub = body.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 0
    sub.render_levels = 1
    return dict(root=root, body=body, fins=fins, paired=paired, beat=beat, L=L)


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


def skipjack_mesh(M, L=0.6):
    """Skipjack tuna: a plump torpedo, deepest a third back, a short first dorsal and a forked tail."""
    top = curve([(0.0, 0.0), (0.08, 0.07), (0.3, 0.14), (0.5, 0.12), (0.75, 0.05), (0.92, 0.02), (1.0, 0.018)])
    bot = curve([(0.0, -0.02), (0.08, -0.07), (0.3, -0.13), (0.5, -0.11), (0.75, -0.045), (0.92, -0.018), (1.0, -0.016)])
    wid = curve([(0.0, 0.0), (0.1, 0.06), (0.35, 0.1), (0.6, 0.075), (0.9, 0.02), (1.0, 0.014)])

    def fins(L, top, bot):
        t = -L / 2
        return [[(0.08 * L, top(0.08 * L) - 0.005 * L), (0.06 * L, top(0.06 * L) + 0.08 * L), (-0.06 * L, top(-0.06 * L) + 0.02 * L),
                 (-0.1 * L, top(-0.1 * L) - 0.005 * L)],
                [(t + 0.03 * L, 0.02 * L), (t - 0.08 * L, 0.14 * L), (t - 0.12 * L, 0.14 * L), (t - 0.07 * L, 0.0),
                 (t - 0.12 * L, -0.14 * L), (t - 0.08 * L, -0.14 * L), (t + 0.03 * L, -0.02 * L)]]
    return small_fish("Skipjack", M, M.skipjack, M.fin_dark, L, lambda u: top(u) * L, lambda u: bot(u) * L,
                      lambda u: wid(u) * L, fins, nr=24, na=14, eye_r=0.035)


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
# Trolling lures: a cast resin head with a cupped face (it pushes water and pulls air down: the "smoke"), a
# vinyl skirt that flutters behind it, a pair of hooks inside the skirt. Built along +X (the nose); about 0.36 m long.
# --------------------------------------------------------------------------
LURE_HEAD = 0.11
LURE_LEN = 0.36


def skirt_flutter_group(name):
    ng, gi, go = new_group(name, "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links

    def nattr(nm):
        r = nodes.new("GeometryNodeInputNamedAttribute")
        r.data_type = "FLOAT"
        r.inputs["Name"].default_value = nm
        return r.outputs[0]
    t = gn_time(ng)
    sx, ang = nattr("sx"), nattr("ang")
    arg = nmath(ng, "ADD", nmath(ng, "MULTIPLY", t, 2 * math.pi * round(6.0 * LOOP_S) / LOOP_S),     # 6 Hz, loops
                nmath(ng, "ADD", nmath(ng, "MULTIPLY", sx, 40.0), nmath(ng, "MULTIPLY", ang, 9.0)))
    amp = nmath(ng, "MULTIPLY", sx, 0.05)
    off = combine(ng, 0.0, nmath(ng, "MULTIPLY", nmath(ng, "SINE", arg), amp),
                  nmath(ng, "MULTIPLY", nmath(ng, "COSINE", nmath(ng, "MULTIPLY", arg, 1.3)), amp))
    sp = nodes.new("GeometryNodeSetPosition")
    links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    links.new(off, sp.inputs["Offset"])
    links.new(sp.outputs[0], go.inputs["Geometry"])
    return ng


def make_lure(col, M, i, name, flutter=None):
    """Lure i (colour set i of M.resin / M.skirts). Returns the root empty (the nose at its origin)."""
    root = empty_object(name, col, size=0.05)
    root.rotation_mode = "QUATERNION"
    bm = bmesh.new()
    n = 20
    front, back = [], []
    for k in range(n):                                   # the head: a short tapered cylinder
        a = 2 * math.pi * k / n
        front.append(bm.verts.new((0.0, 0.026 * math.cos(a), 0.026 * math.sin(a))))
        back.append(bm.verts.new((-LURE_HEAD, 0.021 * math.cos(a), 0.021 * math.sin(a))))
    for k in range(n):
        bm.faces.new((front[k], back[k], back[(k + 1) % n], front[(k + 1) % n]))
    cup = bm.verts.new((-0.022, 0.0, -0.004))            # the cupped, slightly slanted face
    for k in range(n):
        bm.faces.new((front[(k + 1) % n], cup, front[k]))
    bc = bm.verts.new((-LURE_HEAD - 0.004, 0.0, 0.0))
    for k in range(n):
        bm.faces.new((back[k], bc, back[(k + 1) % n]))
    for f in bm.faces:
        f.smooth = True
    head = finish(name + "Head", bm, col, [M.resin[i % len(M.resin)]])
    head.parent = root
    bm = bmesh.new()                                     # the skirt: thin strips flaring back from the head
    sxl, angl = bm.verts.layers.float.new("sx"), bm.verts.layers.float.new("ang")
    strips, segs = 28, 8
    for s in range(strips):
        a = 2 * math.pi * s / strips
        cols = []
        for j in range(segs + 1):
            u = j / segs
            x = -LURE_HEAD + 0.01 - u * (LURE_LEN - LURE_HEAD)
            r = 0.022 + 0.022 * u ** 0.8
            w = 0.0055 * (1.0 - 0.5 * u)
            c, sn = math.cos(a), math.sin(a)
            tang = Vector((0.0, -sn, c))
            p = Vector((x, r * c, r * sn))
            pair = (bm.verts.new(p - tang * w), bm.verts.new(p + tang * w))
            for v in pair:
                v[sxl] = u * (LURE_LEN - LURE_HEAD)
                v[angl] = s / strips
            cols.append(pair)
        for (a0, b0), (a1, b1) in zip(cols, cols[1:]):
            f = bm.faces.new((a0, b0, b1, a1))
            f.smooth = True
    skirt = finish(name + "Skirt", bm, col, [M.skirts[i % len(M.skirts)]], recalc=False)
    skirt.parent = root
    skirt.modifiers.new("Flutter", "NODES").node_group = flutter or skirt_flutter_group(name + "Flutter")
    bm = bmesh.new()                                     # two hooks inside the skirt, points up and to the side
    for k, (yy, ang) in enumerate(((0.012, 0.6), (-0.012, -0.6))):
        pts = []
        for j in range(10):
            u = j / 9
            if u < 0.6:
                pts.append(Vector((-0.12 - 0.17 * u / 0.6, yy, 0.0)))
            else:
                th = (u - 0.6) / 0.4 * math.pi
                pts.append(Vector((-0.29 - 0.03 * math.sin(th), yy + 0.0, 0.03 * (1 - math.cos(th)))))
        rot = Matrix.Rotation(ang, 4, "X")
        pts = [rot @ p for p in pts]
        for p0, p1 in zip(pts, pts[1:]):
            bm_frustum(bm, tuple(p0), tuple(p1), 0.0022, 0.0022, segs=6)
    hooks = finish(name + "Hooks", bm, col, [M.hook])
    hooks.parent = root
    for ob in (head, skirt, hooks):
        ob.visible_shadow = False if ob is skirt else True
    return dict(root=root, head=head, skirt=skirt, hooks=hooks)


def lure_trail(name, col, M, path_fn, f_lo, f_hi, rate=40, seed=3, spread=0.02, life=1.6):
    """The "smoke": air bubbles shed from the lure's cupped face, left behind in the water, rising and swirling a
    little before they shrink away. path_fn(f) -> (nose position, unit heading)."""
    rng = random.Random(seed)
    sp = Particles()
    for f in range(f_lo - int(life * FPS), f_hi + 1):
        p, fwd = path_fn(f)
        for _ in range(max(1, int(rate / FPS + rng.random()))):
            sub = rng.random()
            q, fw2 = path_fn(f + sub)
            o = Vector((rng.gauss(0, spread), rng.gauss(0, spread), rng.gauss(0, spread)))
            v = Vector((rng.gauss(0, 0.08), rng.gauss(0, 0.08), rng.uniform(0.05, 0.25)))
            sp.emit(t_of(f + sub), q - fw2 * rng.uniform(0.02, 0.1) + o, v, rng.uniform(0.6, 1.0) * life,
                    rng.uniform(0.003, 0.009), gravity=-0.2)
        _ = (p, fwd)
    return sp.build(name, col, M.bubbles)


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
            butt = Vector((x, y, 1.0))
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
# Choreography. Places: the boat (the spread, the reel, the tail walk and the loop's overhead shot), open water for
# the hook and the smoking lure (UW1) and the drop off cross section (UW2), far apart and each rendered only in its
# own shots.
# --------------------------------------------------------------------------
UW1 = Vector((620.0, 0.0, 0.0))
UW2 = Vector((-620.0, 0.0, 0.0))
TROLL_V = 4.1                        # m/s: 8 knots (8 x 0.5144 = 4.12 m/s), the middle of the 7 to 9 knot range
HOOK_STRIKE = WHIP_IN + 38           # the marlin engulfs the lure
LOCK_F = VO_START[4] + int(round(2.6 * FPS))          # "lock down": the lever goes up to the strike stop
SPREAD = {                                             # lure positions astern of the transom (boat frame, m)
    "short corner": (Vector((-17.0, 3.0)), "corner_port", 1),
    "long corner": (Vector((-27.0, -3.6)), "corner_starboard", 0),
    "short rigger": (Vector((-38.0, 9.6)), "rigger_port", 2),
    "long rigger": (Vector((-50.0, -10.2)), "rigger_starboard", 3),
}
OH_BACK = 22.0                       # the overhead shot looks down 22 m astern, between the corner lures
OH_H = 12.0


def wrap_t(t):
    """Time folded around the loop point (frame 1320 -> just under 0), so the loop's boat runs straight through it."""
    return t if t < LOOP_S / 2 else t - LOOP_S


def loop_boat_x(f):
    return 260.0 + TROLL_V * wrap_t(t_of(f))


def spread_boat_x(f):
    return -40.0 + TROLL_V * (t_of(f) - t_of(CUT2))


def strike_boat_x(f):
    return 90.0 + TROLL_V * (t_of(f) - t_of(CUT3))


def boat_x_at(f):
    if CUT2 <= f < CUT2B + 2:
        return spread_boat_x(f)
    if CUT3 - 2 <= f < CUT4 + 2:
        return strike_boat_x(f)
    return loop_boat_x(f)


def lure_world(name, f, bx=None):
    """A spread lure: popping on the surface astern, a little sideways wander."""
    off, _tip, _i = SPREAD[name]
    bx = boat_x_at(f) if bx is None else bx
    t = t_of(f)
    ph = hash(name) % 7
    x = bx + off.x + 0.25 * math.sin(t * 0.9 + ph)
    y = off.y + 0.3 * math.sin(t * 0.55 + ph)
    z = surface_z(x, y, t) - 0.06 + 0.07 * math.sin(t * 2 * math.pi * 0.7 + ph)      # pop, smoke, pop
    return Vector((x, y, z))


def hook_lure(f):
    t = t_of(f) - t_of(WHIP_IN)
    return UW1 + Vector((-25.0 + TROLL_V * t, 0.5 * math.sin(t * 0.5), -0.5 + 0.06 * math.sin(t * 9.0)))


def smoke_lure(f):
    t = t_of(f) - t_of(CUT2B)
    return UW1 + Vector((40.0 + TROLL_V * t, -3.0 + 0.3 * math.sin(t * 0.7), -0.55 + 0.05 * math.sin(t * 8.0)))


def _heading(p0, p1, fallback=Vector((1, 0, 0))):
    d = p1 - p0
    return d.normalized() if d.length > 1e-5 else fallback


def build_hook_path():
    """The marlin in the hook: rising from below and behind to engulf the lure, then turning away and diving on the
    line. Precomputed (root position, heading) per frame."""
    L = MARLIN_L
    f0, fs, f1 = WHIP_IN - 12, HOOK_STRIKE, CUT1 + 3
    fwd_s = Vector((1.0, -0.12, 0.22)).normalized()
    path = {}
    o0 = Vector((-10.0, 2.6, -4.8))
    for f in range(f0, fs + 1):
        u = (f - f0) / (fs - f0)
        oend = -fwd_s * (L / 2)
        path[f] = hook_lure(f) + o0.lerp(oend, ease_out(u, 2.4))
    yaw, pitch = math.atan2(fwd_s.y, fwd_s.x), math.asin(fwd_s.z)
    p = path[fs].copy()
    for f in range(fs + 1, f1 + 1):
        k = smoothstep(fs, fs + 30, f)
        y_ = yaw + math.radians(62) * k + math.radians(9) * math.sin((f - fs) * 0.11) * k
        pt = pitch + (math.radians(-13) - pitch) * smoothstep(fs, fs + 24, f) + math.radians(4) * math.sin((f - fs) * 0.07)
        d = Vector((math.cos(pt) * math.cos(y_), math.cos(pt) * math.sin(y_), math.sin(pt)))
        speed = 6.0 - 1.2 * smoothstep(fs, fs + 60, f)
        p = p + d * speed / FPS
        p.z = min(p.z, -1.6) if f > fs + 20 else p.z
        path[f] = p.copy()
    keys = sorted(path)
    pos = smooth_vectors([path[f] for f in keys], 3)
    out = {}
    for i, f in enumerate(keys):
        a, b = pos[max(0, i - 1)], pos[min(len(pos) - 1, i + 1)]
        fwd = _heading(a, b)
        if f <= fs:                                 # the last approach frames: aimed at the lure
            fwd = fwd.lerp(fwd_s, smoothstep(fs - 12, fs, f)).normalized()
        out[f] = (pos[i], fwd)
    return out


def smooth_vectors(vals, k):
    arr = np.array([list(v) for v in vals])
    out = arr.copy()
    for i in range(len(arr)):
        lo, hi = max(0, i - k), min(len(arr), i + k + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    return [Vector(v) for v in out]


HOOK_PATH = build_hook_path()


def hook_marlin(f):
    keys = sorted(HOOK_PATH)
    f = min(max(f, keys[0]), keys[-1])
    return HOOK_PATH[f]


def mouth_world(p, fwd, roll=0.0):
    return p + heading_quat(fwd, roll) @ Vector((MARLIN_L / 2 - 0.02, 0.0, -0.01))


def rise_marlin(f):
    """The marlin rising from the depths behind the smoking lure (the spread beat, under water)."""
    u = (f - CUT2B) / max(1, CUT3 - CUT2B)
    lure = smoke_lure(f)
    p = lure + Vector((-13.0, 5.0, -12.0)).lerp(Vector((-6.2, 1.4, -1.8)), ease(u))
    q = lure + Vector((-13.0, 5.0, -12.0)).lerp(Vector((-6.2, 1.4, -1.8)), ease(min(1.0, u + 0.02))) + Vector((0.12, 0, 0))
    return p, _heading(p, q)


# --- the drop off (UW2 local: x along the profile, z depth, the cut face at y = LEDGE_Y0) -----------------------
LEDGE_Y0 = -4.0


def ledge_z(x):
    """The profile: a shelf at about -10 m, a steep volcanic drop with two terraces, then the deep floor."""
    if x < -6.0:
        return -10.0 + 0.6 * math.sin(x * 0.4) + 0.3 * math.sin(x * 1.3)
    if x < 4.0:
        return -10.0 - 20.0 * ease((x + 6.0) / 10.0)
    if x < 8.0:
        return -30.0 - 3.0 * ease((x - 4.0) / 4.0)
    if x < 16.0:
        return -33.0 - 24.0 * ease((x - 8.0) / 8.0)
    if x < 20.0:
        return -57.0 - 2.0 * ease((x - 16.0) / 4.0)
    if x < 30.0:
        return -59.0 - 25.0 * ease((x - 20.0) / 10.0)
    return -84.0 - 0.5 * math.sin(x * 0.3)


def ledge_marlin(f):
    t = t_of(f) - t_of(CUT1)
    x = -14.0 + 1.9 * t
    z = -16.5 + 0.6 * math.sin(t * 0.5)
    y = 2.0 + 0.8 * math.sin(t * 0.35)
    fwd = Vector((1.9, 0.8 * 0.35 * math.cos(t * 0.35), 0.6 * 0.5 * math.cos(t * 0.5))).normalized()
    return UW2 + Vector((x, y, z)), fwd


# --- the tail walk -------------------------------------------------------------------------------------------
def tail_walk(f):
    """Root position, heading (nose direction), roll for the marlin walking on its tail across the surface."""
    t = t_of(f) - t_of(CUT3B)
    bx0 = strike_boat_x(CUT3B)
    x = bx0 - 44.0 + 0.4 * math.sin(t * 1.3)
    y = -16.0 + 3.0 * t
    rise = ease_out(min(1.0, (t + 0.18) / 0.45), 2.0)            # already bursting out on the cut
    zc = -1.8 + 3.75 * rise + 0.22 * math.sin(2 * math.pi * 3.2 * t)
    zc -= 0.15 * smoothstep(0.0, 2.0, t)
    tilt = math.radians(68 + 6 * math.sin(2 * math.pi * 1.6 * t))
    yaw = math.radians(90 + 18 * math.sin(2 * math.pi * 1.5 * t))
    fwd = Vector((math.cos(tilt) * math.cos(yaw), math.cos(tilt) * math.sin(yaw), math.sin(tilt)))
    roll = math.radians(28) * math.sin(2 * math.pi * 2.9 * t)
    return Vector((x, y, zc)), fwd, roll


def tail_contact(f):
    p, fwd, roll = tail_walk(f)
    tail = p - fwd * (MARLIN_L / 2 + 0.25)
    return Vector((tail.x, tail.y, surface_z(tail.x, tail.y, t_of(f))))


# --------------------------------------------------------------------------
# Camera: frames[f] = (position, target, rotation, lens); f-stop per frame
# --------------------------------------------------------------------------
def whip(q, angle_deg, axis):
    """Rotation q turned by angle_deg about a world axis (the whip pans)."""
    return Quaternion(Vector(axis), math.radians(angle_deg)) @ q


def gauss_smooth(fn, f, k=6):
    acc, ws = Vector((0, 0, 0)), 0.0
    for d in range(-k, k + 1):
        w = math.exp(-0.5 * (d / (k / 2.0)) ** 2)
        acc += fn(f + d) * w
        ws += w
    return acc / ws


def overhead(f):
    bx = loop_boat_x(f)
    tgt = Vector((bx - OH_BACK, -0.2, 0.0))
    cam = tgt + Vector((0.0, -0.4, OH_H))
    return cam, tgt, look_at(cam, tgt, Vector((1, 0, 0)))


def camera_frames(st):
    frames, fstop = {}, {}
    Z = Vector((0, 0, 1))
    # 0 the loop's join: the spread from above, then a whip pan down into the water
    for f in range(1, WHIP_IN):
        cam, tgt, q = overhead(f)
        ang = 0.0 if f <= 3 else (13.0 * (f - 3) ** 1.6)
        frames[f] = (cam, tgt, whip(q, ang, q @ Vector((1, 0, 0))), 24.0)
        fstop[f] = 11.0
    # 1 hook: alongside the lure as the marlin comes up behind it, then with the marlin as it turns away
    def hook_cam_pos(g):
        lure = hook_lure(g)
        a = lure + Vector((-1.4, -3.3, -0.75))
        mp, fwd = hook_marlin(g)
        side = Z.cross(fwd).normalized()
        b = mp - side * 5.8 + fwd * 0.6 + Vector((0.0, 0.0, -0.7))
        k = smoothstep(HOOK_STRIKE + 2, HOOK_STRIKE + 34, g)
        c = a.lerp(b, k)
        c.z = min(c.z, -0.9)
        return c

    def hook_tgt(g):
        lure = hook_lure(g)
        a = lure + Vector((-0.7, 0.0, -0.5))
        mp, fwd = hook_marlin(g)
        b = mp + fwd * 0.4
        return a.lerp(b, smoothstep(HOOK_STRIKE - 6, HOOK_STRIKE + 20, g))
    for f in range(WHIP_IN, CUT1A):
        cam = gauss_smooth(hook_cam_pos, f, 6)
        tgt = gauss_smooth(hook_tgt, f, 4)
        q = look_at(cam, tgt, Z)
        if f < WHIP_IN + 6:
            k = 1.0 - ease_out((f - WHIP_IN) / 6.0, 3.0)
            q = whip(q, 36.0 * k, q @ Vector((1, 0, 0)))
        frames[f] = (cam, tgt, q, 24.0)
        fstop[f] = 4.0
    # 1b a second angle, side on and a little below, the marlin running screen right on the line toward the light
    def side_cam(g):
        mp, fwd = hook_marlin(g)
        fh = Vector((fwd.x, fwd.y, 0.0)).normalized()
        u = (g - CUT1A) / max(1, CUT1 - CUT1A)
        d = Quaternion(Z, math.radians(-22 + 40 * u)) @ Z.cross(fh)      # a slow orbit round the running fish
        return mp - d * (6.6 - 1.2 * u) + fh * 0.4 + Vector((0.0, 0.0, -1.4 + 0.5 * u))
    for f in range(CUT1A, CUT1):
        cam = gauss_smooth(side_cam, f, 6)
        mp, fwd = hook_marlin(f)
        tgt = gauss_smooth(lambda g: hook_marlin(g)[0] + hook_marlin(g)[1] * 0.4 + Vector((0, 0, 0.25)), f, 4)
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 28.0)
        fstop[f] = 5.6
    # 2 the drop off: match cut on the marlin running screen right, then a hard cut to the whole diagram, which
    # drifts in slowly
    for f in range(CUT1, CUT1W):
        mp, fwd = ledge_marlin(f)
        cam = mp + Vector((0.4 + 0.02 * (f - CUT1), -6.8, -0.9))
        tgt = mp + Vector((0.4, 0.0, 0.25))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 28.0)
        fstop[f] = 5.6
    for f in range(CUT1W, CUT2):
        u = (f - CUT1W) / (CUT2 - CUT1W)
        cam = UW2 + Vector((0.5 + 1.5 * u, -76.0 + 6.0 * ease(u), -41.0))
        tgt = UW2 + Vector((0.5 + 1.5 * u, 0.0, -42.0))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 28.0)
        fstop[f] = 16.0
    # 3 the spread from the air: a high chase behind the spread looking forward and down at the boat, the four
    # lures fanned and staggered astern of it
    for f in range(CUT2, CUT2B):
        u = (f - CUT2) / (CUT2B - CUT2)
        bx = spread_boat_x(f)
        tgt = Vector((bx - 30.0 + 3.0 * u, 0.0, 0.0))
        cam = Vector((bx - 91.0 + 4.0 * u, -14.0 + 4.0 * u, 50.0 - 3.0 * u))
        frames[f] = (cam, tgt, look_at(cam, tgt, Vector((0, 0, 1))), 30.0)
        fstop[f] = 16.0
    # 3b under water with the smoking lure; a marlin rises out of the blue behind it
    for f in range(CUT2B, CUT3):
        u = (f - CUT2B) / (CUT3 - CUT2B)
        lure = smoke_lure(f)
        cam = lure + Vector((1.2 + 0.3 * u, -1.0 - 0.25 * u, -0.32))       # ahead of it, looking back along its trail
        tgt = lure + Vector((-0.9 - 1.2 * u, 0.35, -0.3 - 0.4 * u))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 30.0)
        fstop[f] = 4.0
    # 4 the reel, close, from the cockpit (rides with the boat), then the tail walk from low on the water
    rod = st["rod"]
    sc = bpy.context.scene
    for f in range(CUT3, CUT3B):
        sc.frame_set(f)
        rm = rod["reel"]["root"].matrix_world.copy()
        t = t_of(f)
        shake = Vector((0.004 * math.sin(t * 7.1), 0.003 * math.sin(t * 5.3 + 1), 0.004 * math.sin(t * 6.2 + 2)))
        cam = rm @ (Vector((-0.56, -0.18, 0.3)) + shake)
        tgt = rm @ Vector((0.0, 0.03, 0.08))
        up = (rm.to_3x3() @ Vector((0, 0, 1))).normalized()
        frames[f] = (cam, tgt, look_at(cam, tgt, up.lerp(Z, 0.6)), 38.0)
        fstop[f] = 2.8
    def tw_cam(g):
        p, fwd, roll = tail_walk(g)
        c = Vector((p.x + 17.0, p.y - 0.8, 0.0))
        c.z = sea_height(c.x, c.y, t_of(g)) * 0.4 + 1.5
        return c
    for f in range(CUT3B, CUT4):
        cam = gauss_smooth(tw_cam, f, 8)
        tgt = gauss_smooth(lambda g: tail_walk(g)[0] + Vector((0.0, 0.0, 0.4)), f, 6)
        q = look_at(cam, tgt, Z)
        if f >= CUT4 - 3:                                       # whip out
            q = whip(q, -(16.0 * (f - (CUT4 - 4)) ** 1.6), Z)
        frames[f] = (cam, tgt, q, 40.0)
        fstop[f] = 5.6
    # 5 outro: whip in to the spread from above and hold it (the hold is frame 1's shot: the loop)
    for f in range(CUT4, FRAME_END + 1):
        cam, tgt, q = overhead(f)
        if f < CUT4 + 7:
            k = 1.0 - ease_out((f - CUT4) / 7.0, 3.0)
            q = whip(q, 40.0 * k, Z)
        frames[f] = (cam, tgt, q, 24.0)
        fstop[f] = 11.0
    return frames, fstop


def build_camera(col, frames, fstop):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = LENS
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = SENSOR
    cd.clip_start = 0.02
    cd.clip_end = 5000.0
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
    # cuts must not interpolate: hold the old shot to f - 0.5, then the new shot (CONSTANT steps)
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
    return 1.0 if (WHIP_IN <= f < CUT2) or (CUT2B <= f < CUT3) else 0.0


def deep_at(f):
    if WHIP_IN <= f < CUT1 or CUT2B <= f < CUT3:
        return 0.12
    if CUT1 <= f < CUT2:
        return 0.3
    return 0.0


def clear_at(f):
    return 1.0 if CUT1 <= f < CUT2 else 0.0


def lit_at(f):
    if WHIP_IN <= f < CUT1:
        return 0.35 + 0.65 * smoothstep(HOOK_STRIKE - 30, HOOK_STRIKE, f)
    if CUT1 <= f < CUT2:
        return 0.5
    if CUT2B <= f < CUT3:
        return 0.3 + 0.7 * smoothstep(CUT2B, CUT3, f)
    return 1.0


def set_constant(idb):
    for fc in iter_fcurves(idb):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def build_sun(col):
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 4.5
    light.specular_factor = 0.5
    light.color = srgb(255, 244, 228)
    light.angle = math.radians(1.5)
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


def mask_sea_in_cockpit(mat, boat_root):
    """The hull is an open shell: make the sea transparent inside the cockpit (in the boat's own frame)."""
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    src = out.inputs["Surface"].links[0].from_socket
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = boat_root
    o = separate(nt, tc.outputs["Object"])
    inside = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", map_range(nt, o["X"], 0.0, 0.08), map_range(nt, o["X"], 4.7, 4.6)),
                   map_range(nt, nmath(nt, "ABSOLUTE", o["Y"]), 2.3, 2.2))
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(inside, mix.inputs["Fac"])
    nt.links.new(src, mix.inputs[1])
    nt.links.new(nt.nodes.new("ShaderNodeBsdfTransparent").outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])


def key_sea_center(sea):
    """Centre the sea grid (finest near its middle) where each above water shot works; held constant per shot."""
    centres = [(1, Vector((loop_boat_x(1) - OH_BACK, 0.0, 0.0))),
               (CUT2, Vector((spread_boat_x((CUT2 + CUT2B) // 2) - 22.0, 0.0, 0.0))),
               (CUT3, Vector((strike_boat_x(CUT3), -1.0, 0.0))),
               (CUT3B, tail_walk(CUT3B)[0] + Vector((8.0, 0.0, 0.0))),
               (CUT4, Vector((loop_boat_x(FRAME_END) - OH_BACK, 0.0, 0.0)))]
    for f, c in centres:
        sea.location = (c.x, c.y, 0.0)
        sea.keyframe_insert("location", frame=f - 0.5 if f > 1 else 1)
    set_constant(sea)


def key_scene_state(sc, world_vals, sun, key_l, bounce):
    under_v, deep_v = world_vals
    for f in range(1, FRAME_END + 1):
        u, d, c, lt = under_at(f), deep_at(f), clear_at(f), lit_at(f)
        for prop, val in (("under", u), ("deep", d), ("clear", c), ("lit", lt)):
            sc[prop] = val
            sc.keyframe_insert(data_path=f'["{prop}"]', frame=f)
        under_v.outputs[0].default_value = u
        deep_v.outputs[0].default_value = d
        under_v.outputs[0].keyframe_insert("default_value", frame=f)
        deep_v.outputs[0].keyframe_insert("default_value", frame=f)
        sun.data.energy = (5.5 * (1 - 0.5 * d)) if u else 4.5
        sun.data.keyframe_insert("energy", frame=f)
        sun.rotation_quaternion = (-(SUN_UNDER if u else SUN_DIR)).to_track_quat("-Z", "Y")
        sun.keyframe_insert("rotation_quaternion", frame=f)
        key_l.data.energy, bounce.data.energy = 0.0, 0.0
        if WHIP_IN <= f < CUT1 or CUT2B <= f < CUT3:
            p, fwd = hook_marlin(f) if f < CUT1 else rise_marlin(f)
            key_l.location = p + Vector((0.0, 0.0, 4.5)) - fwd * 0.5
            key_l.rotation_quaternion = (p - key_l.location).normalized().to_track_quat("-Z", "Y")
            key_l.data.energy = 1100.0
            side = Vector((0, 0, 1)).cross(Vector((fwd.x, fwd.y, 0.0)).normalized())
            bounce.location = p - side * 3.5 + Vector((0.0, 0.0, -3.0))
            bounce.rotation_quaternion = (p - bounce.location).normalized().to_track_quat("-Z", "Y")
            bounce.data.energy = 320.0
        elif CUT1 <= f < CUT2:
            key_l.location = UW2 + Vector((0.0, -20.0, -8.0))
            key_l.rotation_quaternion = Vector((0.1, 0.5, -0.86)).to_track_quat("-Z", "Y")
            key_l.data.energy = 9000.0
            mp, fwd = ledge_marlin(f)
            bounce.location = mp + Vector((0.0, -5.0, -2.5))
            bounce.rotation_quaternion = (mp - bounce.location).normalized().to_track_quat("-Z", "Y")
            bounce.data.energy = 600.0
        for ob in (key_l, bounce):
            ob.data.keyframe_insert("energy", frame=f)
            key(ob, "location", f)
            key(ob, "rotation_quaternion", f)
    for idb in (sc, sc.world.node_tree, sun, key_l, key_l.data, bounce, bounce.data):
        set_constant(idb)


# --------------------------------------------------------------------------
# Animation
# --------------------------------------------------------------------------
def animate_marlin(fish, spans, rate_hz=1.4, amp=0.07):
    """spans: [(f_lo, f_hi, state)], state(f) -> (pos, fwd[, roll]); the swim phase runs on through the spans."""
    phase = 0.0
    for f_lo, f_hi, state, rate, a in spans:
        prev = None
        for f in range(f_lo, f_hi + 1):
            r = state(f)
            p, fwd = r[0], r[1]
            roll = r[2] if len(r) > 2 else 0.0
            fish["root"].location = p
            fish["root"].rotation_quaternion = heading_quat(fwd, roll)
            key(fish["root"], "location", f)
            key(fish["root"], "rotation_quaternion", f)
            turn = 0.0
            if prev is not None:
                turn = max(-1.0, min(1.0, prev.cross(fwd).z * 30.0))
            prev = fwd
            phase += 2 * math.pi * rate / FPS
            key_swim(fish, f, phase, a(f) if callable(a) else a, turn * 0.04)


def place_lure(lure, f, p, fwd, roll=0.0):
    lure["root"].location = p
    lure["root"].rotation_quaternion = heading_quat(fwd, roll)
    key(lure["root"], "location", f)
    key(lure["root"], "rotation_quaternion", f)


def strike_burst(center_fn, f0, n, seed, mat_spray=None):
    """White water at the strike: a cloud of bubbles bursting from the marlin's mouth as it engulfs the lure."""
    rng = random.Random(seed)
    sp = Particles()
    for k in range(n):
        f = f0 + rng.uniform(-1.0, 6.0)
        c = center_fn(int(f))
        d = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))).normalized()
        sp.emit(t_of(f), c + d * rng.uniform(0.0, 0.35), d * rng.uniform(0.3, 1.8) + Vector((0, 0, rng.uniform(0.2, 0.8))),
                rng.uniform(0.8, 2.2), rng.uniform(0.006, 0.03), gravity=-0.6)
    return sp


def tail_spray(f_lo, f_hi):
    """Spray thrown from the thrashing tail and flung off the head shakes during the tail walk."""
    rng = random.Random(21)
    sp = Particles()
    for f in range(f_lo, f_hi + 1):
        t = t_of(f)
        c = tail_contact(f)
        p, fwd, roll = tail_walk(f)
        n = 70 if f > f_lo + 3 else 400
        for _ in range(n):
            d = Vector((rng.gauss(0, 1), rng.gauss(0, 1), 0.0)).normalized()
            v = d * rng.uniform(0.5, 3.5) + Vector((0.0, -2.4, rng.uniform(1.0, 5.5)))
            sp.emit(t + rng.random() / FPS, c + d * rng.uniform(0.0, 0.6), v, rng.uniform(0.5, 1.2),
                    rng.uniform(0.008, 0.03), 9.81)
        for _ in range(16):                                        # droplets off the head and body
            s = rng.uniform(-0.2, 0.55) * MARLIN_L
            q = p + fwd * s
            side = fwd.cross(Vector((0, 0, 1))).normalized()
            v = side * rng.uniform(-3.5, 3.5) + Vector((0, 0, rng.uniform(0.0, 2.5)))
            sp.emit(t + rng.random() / FPS, q, v, rng.uniform(0.4, 0.9), rng.uniform(0.005, 0.016), 9.81)
    return sp


def build_ledge(col, M):
    """The cross section: the drop off as a slab cut at y = LEDGE_Y0 (the diagram's face), extruded back into the
    blue; a backdrop shading from sunlit blue to black with depth."""
    bm = bmesh.new()
    xs = [-48.0 + 0.5 * i for i in range(201)]
    ys = [LEDGE_Y0 + 1.0 * j for j in range(45)]
    grid = []
    for y in ys:
        row = []
        for x in xs:
            z = ledge_z(x) + 0.7 * mnoise.noise(Vector((x * 0.2, y * 0.2, 0.0))) + 0.25 * mnoise.noise(Vector((x, y, 3.0)))
            row.append(bm.verts.new((x, y, z)))
        grid.append(row)
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
    front_co = [v.co.copy() for v in grid[0]]
    top = finish("LedgeRock", bm, col, [M.basalt])
    for pl in top.data.polygons:
        pl.use_smooth = True
    bm = bmesh.new()                                     # the cut face, down to well below the floor
    bottom = [bm.verts.new((c.x, c.y - 0.02, -110.0)) for c in front_co]
    topv = [bm.verts.new((c.x, c.y - 0.02, c.z)) for c in front_co]
    for i in range(len(front_co) - 1):
        bm.faces.new((bottom[i], bottom[i + 1], topv[i + 1], topv[i]))
    face = finish("LedgeCutFace", bm, col, [M.cutface], recalc=False)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * 220.0, 0.0, v.co.y * 70.0))
    back = finish("LedgeBackdrop", bm, col, [M.column], recalc=False)
    back.location = (0.0, 120.0, -60.0)
    for ob in (top, face, back):
        ob.parent = None
        ob.location = ob.location + UW2
    top.visible_shadow = False
    back.visible_shadow = False
    face.visible_shadow = False
    return top, face, back


def upwelling_flow(f_lo, f_hi):
    """Streaks of glow drifting up the slope from the deep and over the lip of the shelf: the upwelling (drawn)."""
    rng = random.Random(33)
    sp = Particles()
    paths = []
    for k in range(5):
        y = rng.uniform(-2.5, 3.0)
        pts = []
        for i in range(40):
            x = 30.0 - 36.0 * i / 39
            z = ledge_z(x) + 2.5 + 1.2 * k / 4 + 6.0 * smoothstep(-2.0, -6.0, x)
            pts.append(UW2 + Vector((x, y, z)))
        paths.append(pts)
    t0, t1 = t_of(f_lo) - 3.0, t_of(f_hi)
    for _ in range(900):
        pts = rng.choice(paths)
        i = rng.randrange(len(pts) - 1)
        a, b = pts[i], pts[i + 1]
        v = (b - a).normalized() * rng.uniform(2.2, 3.4)
        sp.emit(rng.uniform(t0, t1), a.lerp(b, rng.random()) + Vector((0, rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3))),
                v, rng.uniform(1.2, 2.2), rng.uniform(0.05, 0.1))
    return sp


def build_spread_lines(col, M, boat, tips, lures, spans):
    """Lines from the rod tips and outrigger tips down to the lures (thickened so they read from the air)."""
    lines = []
    for name, (off, tip_key, i) in SPREAD.items():
        start = empty_object(f"LineStart_{i}", col, size=0.05, parent=boat["root"])
        start.location = tips[tip_key]
        end = empty_object(f"LineEnd_{i}", col, size=0.05)
        sag = empty_object(f"LineSag_{i}", col, size=0.05)
        for f_lo, f_hi in spans:
            for f in range(f_lo, f_hi + 1):
                bpy.context.scene.frame_set(f)
                a = start.matrix_world.translation.copy()
                b = lure_world(name, f) + Vector((0.03, 0.0, 0.02))
                end.location = b
                sag.location = a.lerp(b, 0.55) + Vector((0.0, 0.0, -0.18 * (a - b).length / 10.0))
                key(end, "location", f)
                key(sag, "location", f)
        lines.append(line_object(f"SpreadLine_{i}", col, M, start, end, radius=0.006, sag=sag, count=32))
    return lines


def rig_strike(boat, rod, f_lo, f_hi):
    """The boat trolling on; the starboard corner rod in its holder, bowing over as line pours off."""
    sc = bpy.context.scene
    reel = rod["reel"]
    spin = 0.0
    for f in range(f_lo, f_hi + 1):
        t = t_of(f)
        pos, q = boat_attitude(Vector((strike_boat_x(f), 0.0, 0.0)), t)
        key_boat(boat, f, pos, q)
        sc.frame_set(f)
        m = boat["root"].matrix_world
        rot = m.to_3x3().normalized()
        butt = m @ Vector((1.9, -2.05, 1.0))                 # the middle holder (the corner rods stay set)
        d = (rot @ Vector((-0.85, -0.42, 0.62))).normalized()
        load = (rot @ Vector((-1.0, -0.25, -0.12))).normalized()
        bend = 0.35 + 0.4 * smoothstep(CUT3, CUT3 + 12, f) + 0.03 * math.sin(f * 0.9)
        place_rod(rod, f, butt, d, load, bend)
        rate = (11.0 if f < LOCK_F else 11.0 - 4.0 * smoothstep(LOCK_F, LOCK_F + 30, f)) * smoothstep(CUT3 - 6, CUT3 + 2, f)
        spin += 2 * math.pi * rate / FPS
        reel["spool"].rotation_euler = (-spin, 0.0, 0.0)
        key(reel["spool"], "rotation_euler", f)
        lev = math.radians(-22.0) + (LEVER_STRIKE - math.radians(-22.0)) * ease(smoothstep(LOCK_F, LOCK_F + 8, f))
        reel["lever"].rotation_euler = (lev, 0.0, 0.0)
        key(reel["lever"], "rotation_euler", f)


def key_loop_boat(boat, f_ranges):
    for f_lo, f_hi, fn in f_ranges:
        for f in range(f_lo, f_hi + 1):
            pos, q = boat_attitude(Vector((fn(f), 0.0, 0.0)), t_of(f))
            key_boat(boat, f, pos, q)


def gerstner_gn(ng, wx, wy, t):
    dx = dy = h = None
    for a, kx, ky, w, ph, q in SWELLS:
        k = math.hypot(kx, ky)
        arg = nmath(ng, "ADD", nmath(ng, "SUBTRACT", nmath(ng, "ADD", nmath(ng, "MULTIPLY", wx, kx), nmath(ng, "MULTIPLY", wy, ky)),
                                     nmath(ng, "MULTIPLY", t, w)), ph)
        sn = nmath(ng, "SINE", arg)
        tx, ty = nmath(ng, "MULTIPLY", sn, -q * a * kx / k), nmath(ng, "MULTIPLY", sn, -q * a * ky / k)
        tz = nmath(ng, "MULTIPLY", nmath(ng, "COSINE", arg), a)
        dx = tx if dx is None else nmath(ng, "ADD", dx, tx)
        dy = ty if dy is None else nmath(ng, "ADD", dy, ty)
        h = tz if h is None else nmath(ng, "ADD", h, tz)
    return dx, dy, h


def store_rest_after(ng, sp, x, y, go):
    st = ng.nodes.new("GeometryNodeStoreNamedAttribute")
    st.data_type = "FLOAT_VECTOR"
    st.domain = "POINT"
    st.inputs["Name"].default_value = "rest"
    ng.links.new(sp.outputs[0], st.inputs["Geometry"])
    ng.links.new(combine(ng, x, y, 0.0), st.inputs["Value"])
    ng.links.new(st.outputs[0], go.inputs["Geometry"])


def wake_object(col, M, name, length=60.0, width=2.2, spread=0.18, nx=None, mat=None):
    """White water astern: a strip that rides the swell (foam by alpha)."""
    bm = bmesh.new()
    nx = nx or max(8, int(length))
    ny = 14
    for i in range(nx):
        for j in range(ny):
            x0, x1 = -length * i / nx, -length * (i + 1) / nx
            w0, w1 = width + spread * (-x0), width + spread * (-x1)
            y0, y1 = -1 + 2 * j / ny, -1 + 2 * (j + 1) / ny
            bm.faces.new([bm.verts.new((x0, w0 * y0, 0.0)), bm.verts.new((x1, w1 * y0, 0.0)),
                          bm.verts.new((x1, w1 * y1, 0.0)), bm.verts.new((x0, w0 * y1, 0.0))])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    ob = finish(name, bm, col, [mat or M.wake], recalc=False)
    for pl in ob.data.polygons:
        pl.use_smooth = True
    ng, gi, go = new_group(name + "GN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    so = ng.nodes.new("GeometryNodeSelfObject")
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    ng.links.new(so.outputs[0], oi.inputs["Object"])
    loc = separate(ng, oi.outputs["Location"])
    x, y, _z = gn_xyz(ng)
    wx, wy = nmath(ng, "ADD", x, loc["X"]), nmath(ng, "ADD", y, loc["Y"])
    t = gn_time(ng)
    dx, dy, h = gerstner_gn(ng, wx, wy, t)
    sp = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, nmath(ng, "ADD", x, dx), nmath(ng, "ADD", y, dy),
                         nmath(ng, "SUBTRACT", nmath(ng, "ADD", h, 0.04), loc["Z"])), sp.inputs["Position"])
    store_rest_after(ng, sp, x, y, go)
    ob.modifiers.new(name, "NODES").node_group = ng
    ob.visible_shadow = False
    return ob


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------
def build_overlays(hud, st):
    hud.scrims()
    hud.kicker("THE FISHING SERIES")
    sc = bpy.context.scene
    # 1 hook (nothing but the kicker on frame 1, as on the last frame)
    p1 = P(1, 1)
    hud.serif("In 1982, Jay de Beaubien landed a *1,376 pound* blue marlin.", WHIP_IN + 6, p1 - 9, y=-0.30, size=0.046,
              lines=["In 1982, Jay de Beaubien", "landed a *1,376 pound* marlin."], reveal=2.3,
              src="IGFA ALL TACKLE RECORD, MAY 31, 1982")
    hud.serif("The world record, *landed in under an hour.*", p1, CUT1 - 9, y=-0.30, size=0.046,
              lines=["The world record,", "*landed in under an hour.*"], src="PACIFIC BLUE MARLIN · MARLIN MAGAZINE")
    hud.pin(lambda f: hook_lure(f) + Vector((0.12, 0.0, 0.05)), "trolled resin lure", WHIP_IN + 10, HOOK_STRIKE - 8,
            dot_color=GOLD, side=0.05, y_band=(-0.18, 0.3))
    hud.pin(lambda f: hook_marlin(f)[0] + hook_marlin(f)[1] * 0.6 + Vector((0, 0, 0.35)), "pacific blue marlin",
            HOOK_STRIKE + 30, CUT1A - 6, dot_color=GOLD, side=-0.05, y_band=(-0.18, 0.3))
    # 2 the drop off
    hud.badge(1, "THE DROP OFF LEDGE", "where upwelling lifts the bait", CUT1 + 8, CUT2 - 8)
    l1 = P(2, 1)
    hud.serif("Grander marlin *hunt the ledges.*", CUT1 + 6, l1 - 9, lines=["Grander marlin", "*hunt the ledges.*"],
              src="ANGLER TECHNIQUE · DIAGRAM NOT TO SCALE")
    hud.serif("Upwellings push *bait and small tuna* up.", l1, CUT2 - 9, lines=["Upwellings push", "*bait and small tuna* up."],
              src="KONA LEDGES: FISHTRACK · DIET: BROCK 1984")
    hud.pin(lambda f: UW2 + Vector((11.0, LEDGE_Y0, ledge_z(11.0) + 0.5)), "steep volcanic drop off", CUT1W + 10, l1 - 4,
            dot_color=srgb(170, 170, 190), side=0.06, y_band=(-0.22, 0.32))
    hud.pin(lambda f: UW2 + Vector((18.0, 0.0, ledge_z(18.0) + 4.0)), "upwelling", l1 + 4, CUT2 - 10,
            dot_color=srgb(110, 230, 255), side=0.05, y_band=(-0.22, 0.32))
    hud.pin(lambda f: UW2 + Vector((-3.0, 4.0, -12.0)), "bait and small tuna", l1 + 30, CUT2 - 10,
            dot_color=GOLD, side=-0.06, rise=0.09, y_band=(-0.22, 0.32))
    # 3 the spread
    hud.badge(2, "THE TROLLING SPREAD", "four lures, staggered astern", CUT2 + 8, CUT3 - 8)
    s_mid = VO_START[3] + int(round(2.9 * FPS))
    hud.serif("A staggered *four lure spread.*", CUT2 + 6, s_mid - 9, lines=["A staggered", "*four lure spread.*"],
              src="ANGLER TECHNIQUE")
    hud.serif("Troll at *7 to 9 knots.*", s_mid, CUT2B - 6, lines=["Troll at", "*7 to 9 knots.*"], src="ANGLER TECHNIQUE")
    hud.serif("Resin heads trail *smoke.*", CUT2B + 4, CUT3 - 9, lines=["Resin heads", "trail *smoke.*"],
              src="ANGLER TECHNIQUE")
    span = CUT2B - CUT2
    for k, name in enumerate(("short corner", "long corner", "short rigger", "long rigger")):
        lo = CUT2 + 8 + int(span * 0.235 * k)
        hud.pin(lambda f, n=name: lure_world(n, f), name, lo, lo + int(span * 0.215),
                dot_color=GOLD, side=(0.05 if SPREAD[name][0].y > 0 else -0.05), rise=0.06, y_band=(-0.24, 0.3))
    hud.pin(lambda f: smoke_lure(f) + Vector((-0.5, 0.0, 0.05)), "smoke trail", CUT2B + 8, CUT3 - 10,
            dot_color=srgb(200, 230, 255), side=-0.05, y_band=(-0.18, 0.3))
    # 4 the drag
    hud.badge(3, "THE DRAG", "130 lb tackle, about 40 lb of drag", CUT3 + 6, CUT4 - 6)
    d1 = LOCK_F
    hud.serif("When a giant eats, *lock it down.*", CUT3 + 4, d1 - 9, lines=["When a giant eats,", "*lock it down.*"], reveal=2.4)
    hud.serif("*40 lb of drag* on 130 lb tackle.", d1, CUT3B - 6, lines=["*40 lb of drag*", "on 130 lb tackle."],
              src="ANGLER TECHNIQUE")
    d3 = P(4, 1)
    hud.serif("Set the hooks in its *hard jaw.*", CUT3B + 4, d3 - 9, lines=["Set the hooks", "in its *hard jaw.*"],
              src="ANGLER TECHNIQUE")
    hud.serif("About *30 percent* of line strength.", d3, CUT4 - 4, lines=["About *30 percent*", "of line strength."],
              src="STRIKE DRAG RULE: MARLIN MAGAZINE")
    rod = st["rod"]

    def stop_pt(f):
        sc.frame_set(f)
        return rod["reel"]["root"].matrix_world @ Vector((-0.084, 0.072 * math.sin(math.radians(10)),
                                                         0.085 + 0.072 * math.cos(math.radians(10))))

    def spool_pt(f):
        sc.frame_set(f)
        return rod["reel"]["root"].matrix_world @ Vector((0.0, 0.0, 0.15))
    hud.pin(spool_pt, "line pouring off", CUT3 + 8, d1 - 10, dot_color=GOLD, side=0.06, rise=0.1, y_band=(-0.2, 0.3))
    hud.pin(stop_pt, "strike", d1 + 4, CUT3B - 6, dot_color=PIN_RED, side=-0.06, rise=0.1, y_band=(-0.2, 0.3))
    hud.pin(lambda f: tail_walk(f)[0] + tail_walk(f)[1] * 1.2, "tail walk", CUT3B + 18, d3 - 6, dot_color=GOLD,
            side=0.06, y_band=(-0.2, 0.32))
    # 5 outro (the last caption is gone before the end: the last frame matches the first)
    hud.serif("Master the spread, *fish the ledges...*", CUT4 + 8, FRAME_END - 9, lines=["Master the spread,", "*fish the ledges...*"],
              reveal=2.6)


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
    """Bloom, a soft S curve and a warm tropical grade on the 3D only, then the overlays on top."""
    ng = bpy.data.node_groups.new("Composite", "CompositorNodeTree")
    main.compositing_node_group = ng
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.scene = main
    img = rl.outputs["Image"]
    try:
        gl = ng.nodes.new("CompositorNodeGlare")
        gl.inputs["Type"].default_value = "Bloom"
        for nm, v in (("Threshold", 1.0), ("Strength", 0.25), ("Size", 0.5)):
            if nm in gl.inputs:
                gl.inputs[nm].default_value = v
        ng.links.new(img, gl.inputs["Image"])
        img = gl.outputs["Image"]
    except Exception as e:                  # noqa: BLE001
        print("bloom skipped:", e)
    try:
        cb = ng.nodes.new("CompositorNodeColorBalance")
        vals = {"Lift": (1.0, 1.0, 1.003, 1.0), "Gamma": (1.0, 1.0, 1.0, 1.0), "Gain": (1.04, 1.01, 0.97, 1.0)}
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
        c.points.new(0.25, 0.21)
        c.points.new(0.75, 0.80)
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
def run_checks(frames, st):
    sc = bpy.data.scenes["Main"]
    print("\n=== CAMERA REPORT ===")
    worst = {"sea surface (above)": (1e9, 0), "surface (under)": (1e9, 0), "boat": (1e9, 0), "rod/reel": (1e9, 0),
             "marlin": (1e9, 0), "ledge": (1e9, 0), "lure": (1e9, 0)}

    def upd(k, d, f):
        if d < worst[k][0]:
            worst[k] = (d, f)
    from mathutils.bvhtree import BVHTree
    boat = st["boat"]
    sc.frame_set(CUT3)
    dg = bpy.context.evaluated_depsgraph_get()
    inv = boat["root"].matrix_world.inverted()
    verts, polys = [], []
    for ob in boat["root"].children_recursive:
        if ob.type != "MESH":
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        mw = inv @ ev.matrix_world
        base = len(verts)
        verts.extend(mw @ v.co for v in me.vertices)
        polys.extend([base + i for i in pl.vertices] for pl in me.polygons)
        ev.to_mesh_clear()
    boat_bvh = BVHTree.FromPolygons(verts, polys)
    lt = st["ledge"]
    me = lt.evaluated_get(dg).to_mesh()
    ledge_bvh = BVHTree.FromPolygons([lt.matrix_world @ v.co for v in me.vertices], [list(p.vertices) for p in me.polygons])
    for f in range(1, FRAME_END + 1):
        p = frames[f][0]
        t = t_of(f)
        if under_at(f):
            upd("surface (under)", -p.z - 0.3, f)
        else:
            upd("sea surface (above)", p.z - sea_height(p.x, p.y, t) - 0.3, f)
        if not under_at(f):
            sc.frame_set(f)
            lp = boat["root"].matrix_world.inverted() @ p
            hit = boat_bvh.find_nearest(lp)
            upd("boat", (hit[3] * BOAT_SCALE if hit[0] is not None else 1e9) - 0.2, f)
        if CUT3 <= f < CUT3B:
            sc.frame_set(f)
            rm = st["rod"]["reel"]["root"].matrix_world.translation
            upd("rod/reel", (rm - p).length - 0.2, f)
        if WHIP_IN <= f < CUT1:
            fp, _fw = hook_marlin(f)
            upd("marlin", (fp - p).length - MARLIN_L / 2 - 0.4, f)
            upd("lure", (hook_lure(f) - p).length - 0.3, f)
        if CUT1 <= f < CUT2:
            hit = ledge_bvh.find_nearest(p)
            upd("ledge", hit[3] if hit[0] is not None else 1e9, f)
            upd("marlin", (ledge_marlin(f)[0] - p).length - MARLIN_L / 2 - 0.4, f)
        if CUT2B <= f < CUT3:
            upd("lure", (smoke_lure(f) - p).length - 0.3, f)
            upd("marlin", (rise_marlin(f)[0] - p).length - MARLIN_L / 2 - 0.4, f)
        if CUT3B <= f < CUT4:
            upd("marlin", (tail_walk(f)[0] - p).length - MARLIN_L / 2 - 0.6, f)
    for k, v in worst.items():
        print(f"closest {k}: {round(v[0], 3)} m at frame {v[1]} (negative = inside)")
    acc, turn = (0.0, 0), (0.0, 0)
    whips = set(range(1, WHIP_IN + 7)) | set(range(CUT4 - 4, CUT4 + 8))
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
          " (whip pans excluded)")
    cross = [f for f in range(2, FRAME_END + 1) if under_at(f) != under_at(f - 1)]
    print("under/above switches:", cross, "all on cuts:", all(f in CUTS for f in cross))
    jumps = [f for f in range(2, FRAME_END + 1) if (frames[f][0] - frames[f - 1][0]).length > 0.6 and f not in CUTS]
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
    subj = {WHIP_IN + 12: ("lure", hook_lure),
            HOOK_STRIKE: ("strike: marlin", lambda f: hook_marlin(f)[0]),
            (CUT1A + CUT1) // 2: ("marlin (side)", lambda f: hook_marlin(f)[0]),
            CUT1 + 3: ("ledge marlin", lambda f: ledge_marlin(f)[0]),
            (CUT1 + CUT2) // 2: ("ledge face", lambda f: UW2 + Vector((11.0, LEDGE_Y0, ledge_z(11.0)))),
            CUT2 + 30: ("short corner lure", lambda f: lure_world("short corner", f)),
            CUT2B - 10: ("long rigger lure", lambda f: lure_world("long rigger", f)),
            CUT2B + 20: ("smoking lure", smoke_lure),
            CUT3 - 10: ("rising marlin", lambda f: rise_marlin(f)[0]),
            CUT3 + 30: ("reel", lambda f: st["rod"]["reel"]["root"].matrix_world.translation.copy()),
            CUT3B + 20: ("tail walk", lambda f: tail_walk(f)[0]),
            FRAME_END - 5: ("corner lure (loop)", lambda f: lure_world("short corner", f))}
    print("\n=== SUBJECTS ON SCREEN (x in +-0.281, y in +-0.5) ===")
    for f, (nm, fn) in sorted(subj.items()):
        sc.frame_set(f)
        s = hud.project(f, fn(f))
        print(f"frame {f}: {nm} ->", None if s is None else (round(s.x, 3), round(s.y, 3)))


def write_events(path, st):
    data = {
        "fps": FPS, "frames": FRAME_END, "loop": True,
        "cuts": [WHIP_IN, CUT1, CUT2, CUT3, CUT4], "hard_cuts": [CUT1A, CUT2B, CUT3B],
        "whips": [[1, WHIP_IN], [CUT4 - 4, CUT4 + 6]],
        "overlays": sorted(st["hud"].events),
        "under": [[WHIP_IN, CUT2], [CUT2B, CUT3]], "hook_strike": HOOK_STRIKE, "reel_start": CUT3, "lock": LOCK_F,
        "tail_walk": [CUT3B, CUT4], "spread": [CUT2, CUT2B], "ledge": [CUT1, CUT2],
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
    for prop in ("under", "deep", "clear", "lit"):
        sc[prop] = 0.0
    c_sea, c_boat, c_uw1, c_uw2 = collection("Sea"), collection("Boat"), collection("OpenWater"), collection("DropOff")
    c_tail, c_fx, c_cam = collection("TailWalk"), collection("LightAndWater"), collection("Camera")
    atmos = build_atmos_group()
    M = build_materials(atmos)
    _w, w_under, w_deep = build_world(sc)
    sun = build_sun(c_fx)
    key_l = build_fill(c_fx, "UnderwaterKey", size=6.0, color=(214, 236, 255))
    bounce = build_fill(c_fx, "UnderwaterBounce", size=8.0, color=(90, 170, 230))
    # above water: the sea, the boat, the spread
    sea = build_sea(c_sea, M)
    key_sea_center(sea)
    boat = build_boat(c_boat, M)
    boat["root"].scale = (BOAT_SCALE,) * 3
    mask_sea_in_cockpit(M.sea, boat["root"])
    rods, tips = build_spread_tackle(c_boat, M, boat["root"])
    rod = build_rod(c_boat, M)
    loop_lo, loop_hi = (1, WHIP_IN + 1), (CUT4 - 2, FRAME_END)
    key_loop_boat(boat, [(loop_lo[0], loop_lo[1], loop_boat_x), (loop_hi[0], loop_hi[1], loop_boat_x),
                         (CUT2 - 2, CUT2B + 1, spread_boat_x)])
    rig_strike(boat, rod, CUT3 - 2, CUT4 + 1)
    flutter = skirt_flutter_group("SkirtFlutter")
    spread_spans = [loop_lo, (CUT2 - 2, CUT2B + 1), loop_hi]
    lures, lure_wakes = {}, []
    for name, (off, tip_key, i) in SPREAD.items():
        lu = make_lure(c_boat, M, i, f"Lure_{i}", flutter=flutter)
        wk = wake_object(c_sea, M, f"LureWake_{i}", length=5.0, width=0.16, spread=0.09, nx=10, mat=M.lure_foam)
        for f_lo, f_hi in spread_spans:
            for f in range(f_lo, f_hi + 1):
                p = lure_world(name, f)
                place_lure(lu, f, p, Vector((1.0, 0.0, 0.08)))
                wk.location = (p.x - 0.15, p.y, 0.0)
                key(wk, "location", f)
        lures[name] = lu
        lure_wakes.append(wk)
    spread_lines = build_spread_lines(c_boat, M, boat, tips, lures, spread_spans)
    wake = wake_object(c_sea, M, "Wake", length=60.0, width=1.5, spread=0.09)
    for f_lo, f_hi in spread_spans + [(CUT3 - 2, CUT4 + 1)]:
        for f in range(f_lo, f_hi + 1):
            wake.location = (boat_x_at(f), 0.0, 0.0)
            key(wake, "location", f)
    strike_line_end = empty_object("StrikeLineEnd", c_boat, size=0.1)
    for f in range(CUT3 - 2, CUT4 + 2):
        bx = strike_boat_x(f)
        strike_line_end.location = Vector((bx - 40.0, -9.0, surface_z(bx - 40.0, -9.0, t_of(f))))
        key(strike_line_end, "location", f)
    strike_line = line_object("StrikeLine", c_boat, M, rod["ctrl"]["Tip"], strike_line_end, radius=0.0014, count=40)
    # open water: the hook and the smoking lure
    marlin = make_marlin(c_uw1, M, "Marlin")
    animate_marlin(marlin, [(WHIP_IN - 12, CUT1 + 3, hook_marlin, 1.4, lambda f: 0.06 if f < HOOK_STRIKE else 0.11),
                            (CUT2B - 2, CUT3 + 2, rise_marlin, 1.1, 0.06)])
    hook_lu = make_lure(c_uw1, M, 1, "HookLure", flutter=flutter)
    for f in range(WHIP_IN - 3, CUT1 + 3):
        if f <= HOOK_STRIKE:
            place_lure(hook_lu, f, hook_lure(f), Vector((1.0, 0.0, 0.0)))
        else:
            mp, fwd = hook_marlin(f)
            place_lure(hook_lu, f, mouth_world(mp, fwd) + fwd * 0.1, fwd)
    smoke_lu = make_lure(c_uw1, M, 0, "SmokeLure", flutter=flutter)
    for f in range(CUT2B - 3, CUT3 + 3):
        place_lure(smoke_lu, f, smoke_lure(f), Vector((1.0, 0.0, 0.02)), 0.15 * math.sin(t_of(f) * 6.0))
    hook_trail = lure_trail("HookTrail", c_uw1, M, lambda f: (hook_lure(min(f, HOOK_STRIKE)), Vector((1, 0, 0))),
                            WHIP_IN, HOOK_STRIKE, rate=260, seed=3, spread=0.025, life=1.4)
    smoke_trail = lure_trail("SmokeTrail", c_uw1, M, lambda f: (smoke_lure(f), Vector((1, 0, 0))), CUT2B, CUT3,
                             rate=520, seed=4, spread=0.03, life=1.8)
    burst = strike_burst(lambda f: mouth_world(*hook_marlin(f)), HOOK_STRIKE, 1400, 8).build("StrikeBurst", c_uw1, M.bubbles)
    ln_a = empty_object("HookLineA", c_uw1, size=0.05)
    ln_b = empty_object("HookLineB", c_uw1, size=0.05)
    for f in range(WHIP_IN - 3, CUT1 + 3):
        a = hook_lure(f) if f <= HOOK_STRIKE else mouth_world(*hook_marlin(f))
        ln_a.location = a
        ln_b.location = Vector((a.x + 16.0, a.y - 1.0, -0.05))
        key(ln_a, "location", f)
        key(ln_b, "location", f)
    hook_line = line_object("HookLine", c_uw1, M, ln_a, ln_b, radius=0.0016, count=30)
    surf1 = build_surface_under(c_uw1, M, center=(UW1.x + 20, UW1.y, 0.0), size=200.0)
    shafts1 = build_shafts(c_uw1, M, (UW1.x + 10, UW1.y + 3, 0.0), n=8, seed=5, spread=14.0, length=34.0)
    pa, pb = hook_marlin(WHIP_IN)[0], hook_marlin(CUT1)[0]
    mid = (pa + pb) / 2
    half = Vector((abs(pb.x - pa.x) / 2 + 10.0, abs(pb.y - pa.y) / 2 + 10.0, 0.0))
    plank1 = specks("Plankton1", c_uw1, M.specks, (mid.x, mid.y, -4.0), (half.x, half.y, 3.6), 26000, 0.004, 31)
    shafts1b = build_shafts(c_uw1, M, (mid.x, mid.y, 0.0), n=9, seed=6, spread=max(half.x, half.y) * 0.8, length=34.0)
    shafts1c = build_shafts(c_uw1, M, (UW1.x + 48, UW1.y - 2, 0.0), n=6, seed=9, spread=10.0, length=34.0)
    plank1b = specks("Plankton1b", c_uw1, M.specks, tuple(UW1 + Vector((52.0, -3.0, -3.0))), (14.0, 8.0, 3.0), 7000, 0.004, 34)
    # the drop off
    ledge, ledge_face, ledge_back = build_ledge(c_uw2, M)
    sk = bpy.data.objects.new("SchoolSkipjack", skipjack_mesh(M))
    c_uw2.objects.link(sk)
    sk.hide_render = sk.hide_viewport = True
    bf = bpy.data.objects.new("SchoolBait", baitfish_mesh(M, L=0.18))
    c_uw2.objects.link(bf)
    bf.hide_render = bf.hide_viewport = True
    school_t = build_school(c_uw2, sk, UW2 + Vector((-3.0, 4.0, -12.5)), 70, 1.2, 4.5, 1.5, 14, "SkipjackSchool")
    school_b = build_school(c_uw2, bf, UW2 + Vector((-7.0, 5.0, -7.0)), 360, 1.0, 4.0, 1.4, 15, "BaitBall")
    lmar = make_marlin(c_uw2, M, "LedgeMarlin")
    animate_marlin(lmar, [(CUT1 - 2, CUT2 + 2, ledge_marlin, 1.2, 0.06)])
    flow = upwelling_flow(CUT1, CUT2).build("Upwelling", c_uw2, M.upwell, streak=True)
    surf2 = build_surface_under(c_uw2, M, center=(UW2.x, UW2.y, 0.0), size=260.0)
    shafts2 = build_shafts(c_uw2, M, (UW2.x - 5, UW2.y + 6, 0.0), n=8, seed=7, spread=30.0, length=60.0)
    plank2 = specks("Plankton2", c_uw2, M.specks, tuple(UW2 + Vector((0.0, -10.0, -30.0))), (50.0, 20.0, 30.0), 5000, 0.01, 32)
    # the tail walk
    tmar = make_marlin(c_tail, M, "TailMarlin")
    animate_marlin(tmar, [(CUT3B - 2, CUT4 + 2, tail_walk, 3.2, 0.2)])
    tspray = tail_spray(CUT3B - 2, CUT4 + 1).build("TailSpray", c_tail, M.spray, streak=True)
    tw_a = empty_object("TailLineA", c_tail, size=0.05)
    tw_b = empty_object("TailLineB", c_tail, size=0.05)
    for f in range(CUT3B - 2, CUT4 + 2):
        p, fwd, roll = tail_walk(f)
        a = mouth_world(p, fwd, roll)
        tw_a.location = a
        tw_b.location = Vector((strike_boat_x(f) - 2.0, -2.0, 1.6))
        key(tw_a, "location", f)
        key(tw_b, "location", f)
    tail_line = line_object("TailLine", c_tail, M, tw_a, tw_b, radius=0.0025, count=30)
    # camera and state
    st = dict(boat=boat, rod=rod, ledge=ledge)
    frames, fstop = camera_frames(st)
    build_camera(c_cam, frames, fstop)
    key_scene_state(sc, (w_under, w_deep), sun, key_l, bounce)
    # visibility: one set of spans per object (keys from several calls would add up)
    above = [(1, WHIP_IN - 1), (CUT2, CUT2B - 1), (CUT3, FRAME_END)]
    vis = {}

    def show(objs, spans):
        for o in objs:
            vis[o.name] = (o, tuple(spans))
    show([sea], above)
    show([o for o in c_boat.all_objects if o.type in ("MESH", "CURVE") and o.name != "RollerGuide"],
         [(1, WHIP_IN - 1), (CUT2, CUT2B - 1), (CUT3, CUT4 - 1), (CUT4, FRAME_END)])
    show([o for lu in lures.values() for o in (lu["head"], lu["skirt"], lu["hooks"])] + lure_wakes + spread_lines,
         [(1, WHIP_IN - 1), (CUT2, CUT2B - 1), (CUT4, FRAME_END)])
    show([rod["ob"]] + [o for o in rod["mount"].children_recursive if o.type == "MESH"] + [strike_line], [(CUT3, CUT4 - 1)])
    show([wake], [(1, WHIP_IN - 1), (CUT2, CUT2B - 1), (CUT3, FRAME_END)])
    uw1_objs = [o for o in c_uw1.all_objects if o.type in ("MESH", "CURVE")]
    show(uw1_objs, [(WHIP_IN, CUT1 - 1), (CUT2B, CUT3 - 1)])
    show([hook_lu[k] for k in ("head", "skirt", "hooks")] + [hook_line, hook_trail, burst], [(WHIP_IN, CUT1 - 1)])
    show([smoke_lu[k] for k in ("head", "skirt", "hooks")] + [smoke_trail], [(CUT2B, CUT3 - 1)])
    show([o for o in c_uw2.all_objects if o.type in ("MESH", "CURVE") and o.name not in ("SchoolSkipjack", "SchoolBait")],
         [(CUT1, CUT2 - 1)])
    show([o for o in c_tail.all_objects if o.type in ("MESH", "CURVE")], [(CUT3B, CUT4 - 1)])
    groups = {}
    for o, spans in vis.values():
        groups.setdefault(spans, []).append(o)
    for spans, objs in groups.items():
        key_visibility(objs, list(spans))
    _ = (rods, shafts1, shafts1b, shafts1c, plank1, plank1b, surf1, ledge_face, ledge_back, school_t, school_b, flow, surf2, shafts2, plank2,
         tspray, tail_line)
    hs, hcam, hcol = build_hud_scene()
    hud_materials(M)
    hud = Hud(hcam, hcol, M, frames)
    st.update(frames=frames, hud=hud, marlin=marlin, lures=lures)
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
