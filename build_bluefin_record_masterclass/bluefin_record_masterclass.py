"""Bluefin record masterclass: giant Atlantic bluefin tuna breakdown. Vertical 9:16, 24 fps, 55 s (1320 frames),
Blender 5.2 EEVEE. Built to loop: the last frame is the first frame's shot, and the voiceover's last line runs
into its first.

Run with the Blender binary:
    blender --background --python bluefin_record_masterclass.py -- [options]
or with the bpy Python module (pip install bpy):
    python bluefin_record_masterclass.py [options]

Options:
    --save PATH        save the generated scene as a .blend file (fonts packed)
    --check            camera clearances (sea, boat, fish), camera acceleration and turn rate, cuts and water
                       surface crossings, overlay overlaps and off-screen boxes, subjects on screen, loop match
    --frames 1,300     render these frames as PNG stills into --outdir
    --outdir DIR       output directory for renders
    --scale PCT        resolution percentage (default 100 -> 1080x1920)
    --samples N        EEVEE render samples (default 32)
    --no-blur          switch motion blur off (quick tests)
    --events PATH      write overlay, action and voiceover timing as JSON (input for bluefin_audio.py)

Everything is procedural except the angler's body (Cesium Man, CC BY 4.0, see assets/CREDITS.md). Facts in the
on-screen text and the voiceover are listed with their sources in SOURCES.md; the fish, the depths, the boat and
the fight are illustrative.

World (metres, +Z up, sea surface z = 0). Cold, grey-blue North Atlantic water off Nova Scotia under a broken
overcast sky with long swells. A 50 ft sportfisher backs down on a giant bluefin; the angler is strapped into the
fighting chair. Overlays live in a second scene ("HUD") with its own static camera, composited over the 3D render.

Beats follow the voiceover clips (voiceover/vo1..vo5). Cuts are hard cuts, match cuts and whip pans (no wipes):
    1  hook: the dark horizon whips down into an underwater tracking shot of a giant bluefin towing a heavy line
    2  gear: a match cut to a top down breakdown of the cockpit (fighting chair, 14/0 reel, roller rod), then
       along the line to the trolled mackerel at a temperature break
    3  thermal strike zone: a cross section of the water column; bluefin hold along the thermocline, then dive
    4  the fight: a low angle at the transom as the boat backs down hard into the swell, the rod doubled over
    5  outro: a whip pan back to the dark horizon, which is frame 1
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

# water: cold, grey-blue North Atlantic (fog colour by view direction, down .. up)
UW_RAMP = [(0.0, srgb(2, 6, 10)), (0.36, srgb(8, 22, 32)), (0.52, srgb(22, 52, 66)), (0.72, srgb(52, 96, 112)),
           (1.0, srgb(112, 160, 172))]
VIS = 14.0                           # fog e-folding distance under water (m): clear ocean water, still moody
AIR_HAZE = 420.0                     # sea haze e-folding distance above water (m)
SUN_DIR = Vector((-0.42, 0.88, 0.22)).normalized()     # towards a low autumn sun behind broken cloud


def t_of(f):
    """Scene time in seconds at frame f, zero at frame 1, so frame 1321 would equal frame 1 (the loop)."""
    return (f - 1) / FPS


# --------------------------------------------------------------------------
# Voiceover timing. vo_timing.json (written by bluefin_audio.py --measure) holds the measured lengths and sentence
# starts. The hook starts on frame 1 and the outro ends on the last frame, so the loop joins mid sentence:
# "...And it all started when... In 1979, Ken Fraser hooked..."
# --------------------------------------------------------------------------
VO_LEN = {1: 10.68, 2: 12.51, 3: 14.52, 4: 9.56, 5: 5.41}
VO_TEXT = {
    1: "In 1979, Ken Fraser hooked a 1,496 pound bluefin tuna off Nova Scotia. A record that hasn't been touched in "
       "over 40 years.",
    2: "To target fish this size, average gear will snap in seconds. You need 130 pound class monofilament, spooled "
       "on a massive fourteen aught reel, trolling fresh mackerel or herring deep behind the boat.",
    3: "Giant bluefin don't just swim anywhere; they track temperature breaks. Tagging studies show their dives "
       "follow the thermocline, the boundary between warm surface water and the cold water below. And they can "
       "dive past three thousand feet.",
    4: "When a half ton fish hits, the angler doesn't fight alone. The captain must back the boat down hard to keep "
       "line on the spool, while the angler holds heavy drag.",
    5: "That's how you conquer the ultimate apex predator of the sea. And it all started when...",
}
VO_GAPS = (8, 10, 10)                # frames of breath after clips 1, 2 and 3; the rest goes before the outro


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
WHIP_IN = 7                          # frames 1..6: the horizon (the loop's join), whip pan down, cut under water
CUT1 = (vo_end(1) + VO_START[2]) // 2            # match cut: towed line -> the line leaving the rod tip, top down
CUT2 = (vo_end(2) + VO_START[3]) // 2            # jump cut to the water column cross section
CUT3 = (vo_end(3) + VO_START[4]) // 2            # rapid cut to the fight at the transom
CUT4 = VO_START[5] - 6                           # whip pan back to the horizon (the loop's join)
CUT1B = P(2, 1) + int((CUT2 - P(2, 1)) * 0.55)  # hard cut inside the gear beat: cockpit -> the bait far astern
CUTS = [WHIP_IN, CUT1, CUT1B, CUT2, CUT3, CUT4]
SEGMENTS = [(1, CUT1 - 1), (CUT1, CUT2 - 1), (CUT2, CUT3 - 1), (CUT3, CUT4 - 1), (CUT4, FRAME_END)]
LOOP_HOLD = 26                       # the last frames hold the horizon shot that frame 1 opens on


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
    dens = nmath(ng, "MULTIPLY", dens, nmath(ng, "MULTIPLY_ADD", clear, -0.65, 1.0))      # the cross section reads clean
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
    """Under water: the fog colour itself (infinitely far). Above water: a low autumn sky under broken overcast,
    with a soft shoulder on camera rays (the Standard view would clip the bright breaks to white), and a dark
    sea below the horizon. Switched by the "under" value."""
    w = bpy.data.worlds.new("NorthAtlanticWorld")
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
    sky.air_density = 1.4
    try:
        sky.aerosol_density = 2.2
    except AttributeError:
        pass
    lit = vmath(nt, "SCALE", sky.outputs[0], scale=0.10)
    # cloud deck: noise on the plane z = 1 seen through the view direction (static, so the loop holds)
    zc = nmath(nt, "MAXIMUM", view["Z"], 0.02)
    uv = combine(nt, nmath(nt, "DIVIDE", view["X"], zc), nmath(nt, "DIVIDE", view["Y"], zc), 0.0)
    cl = noise(nt, vmath(nt, "MULTIPLY", uv, (0.9, 1.6, 1.0)), 1.6, 8.0, 0.62)
    cl2 = noise(nt, vmath(nt, "MULTIPLY", uv, (3.0, 5.0, 1.0)), 2.0, 6.0, 0.6)
    cover = map_range(nt, nmath(nt, "MULTIPLY_ADD", cl2, 0.35, cl), 0.48, 0.7)
    to_sun = nmath(nt, "ADD", nmath(nt, "ADD", nmath(nt, "MULTIPLY", view["X"], SUN_DIR.x),
                                     nmath(nt, "MULTIPLY", view["Y"], SUN_DIR.y)), nmath(nt, "MULTIPLY", view["Z"], SUN_DIR.z))
    sun_side = map_range(nt, to_sun, 0.6, 1.0)
    cloud_dark = mix_rgb(nt, sun_side, srgb(70, 78, 86), srgb(196, 170, 140))
    cloud_lit = mix_rgb(nt, nmath(nt, "MULTIPLY", cl2, 0.6), cloud_dark, srgb(150, 154, 158))
    near_h = map_range(nt, view["Z"], 0.012, 0.14)                  # clouds thin out into haze at the horizon
    # (fully gone by the horizon: the projected noise aliases into a comb of streaks there)
    skyc = mix_rgb(nt, nmath(nt, "MULTIPLY", cover, nmath(nt, "POWER", near_h, 1.3)), lit,
                   vmath(nt, "SCALE", cloud_lit, scale=0.55))
    knee, top = 0.55, 0.97
    sp = separate(nt, skyc)
    peak = nmath(nt, "MAXIMUM", sp["X"], nmath(nt, "MAXIMUM", sp["Y"], sp["Z"]))
    over = nmath(nt, "MAXIMUM", nmath(nt, "SUBTRACT", peak, knee), 0.0)
    rolled = nmath(nt, "ADD", nmath(nt, "MINIMUM", peak, knee),
                   nmath(nt, "DIVIDE", over, nmath(nt, "ADD", nmath(nt, "MULTIPLY", over, 1.0 / (top - knee)), 1.0)))
    soft = vmath(nt, "SCALE", skyc, scale=nmath(nt, "DIVIDE", rolled, nmath(nt, "MAXIMUM", peak, 1e-5)))
    is_cam = nt.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"]
    skyc = vmath(nt, "ADD", skyc, vmath(nt, "SCALE", vmath(nt, "SUBTRACT", soft, skyc), scale=is_cam))
    skyc = mix_rgb(nt, map_range(nt, view["Z"], 0.0, -0.03), skyc, srgb(12, 22, 30))    # the sea past the horizon
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
for _n, _amp, _deg, _q in ((9, 0.85, 8.0, 0.55), (11, 0.55, 33.0, 0.5), (14, 0.34, -24.0, 0.5), (19, 0.17, 61.0, 0.45),
                           (26, 0.085, -66.0, 0.4), (37, 0.04, 14.0, 0.35), (52, 0.02, -38.0, 0.3)):
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
        """Cold North Atlantic seen from above: dark slate water, sky reflections from the Fresnel term, ripples,
        whitecaps on the sharp crests, streaks of foam lying along the wind."""
        rest = separate(nt, attr(nt, "rest", "Vector"))
        p = combine(nt, rest["X"], rest["Y"], 0.0)
        crest = attr(nt, "crest", "Fac")
        rip = looped_noise(nt, vmath(nt, "MULTIPLY", p, (1.0, 1.0, 1.0)), (0.6, 0.25, 0.0), 1.4, 6.0, 0.6)
        rip2 = looped_noise(nt, vmath(nt, "MULTIPLY", p, (1.0, 1.0, 1.0)), (-0.3, 0.5, 0.0), 6.0, 4.0, 0.55)
        foam_n = looped_noise(nt, vmath(nt, "MULTIPLY", p, (0.6, 1.4, 1.0)), (0.3, 0.12, 0.0), 0.9, 5.0, 0.65)
        cap = nmath(nt, "MULTIPLY", map_range(nt, crest, 0.38, 0.72), map_range(nt, foam_n, 0.45, 0.62))
        streak = nmath(nt, "MULTIPLY", map_range(nt, noise(nt, vmath(nt, "MULTIPLY", p, (0.25, 2.2, 1.0)), 1.2, 4.0, 0.6),
                                                 0.6, 0.7), 0.35)
        foam = nmath(nt, "MINIMUM", nmath(nt, "ADD", cap, nmath(nt, "MULTIPLY", streak, map_range(nt, crest, 0.0, 0.5))), 1.0)
        dist = nt.nodes.new("ShaderNodeCameraData").outputs["View Distance"]
        foam = nmath(nt, "MULTIPLY", foam, map_range(nt, dist, 300.0, 900.0, 1.0, 0.25))
        water = color_ramp(nt, map_range(nt, crest, -0.4, 0.6), [(0.0, srgb(6, 16, 22)), (0.7, srgb(10, 30, 38)),
                                                                   (1.0, srgb(24, 62, 70))])
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
        p = attr(nt, "rest", "Vector")
        o = separate(nt, p)
        along = nmath(nt, "ABSOLUTE", o["X"])
        wdt = nmath(nt, "ADD", 2.2, nmath(nt, "MULTIPLY", along, 0.18))
        across = nmath(nt, "DIVIDE", nmath(nt, "ABSOLUTE", o["Y"]), wdt)
        n1 = looped_noise(nt, vmath(nt, "MULTIPLY", p, (0.6, 1.2, 1.0)), (0.0, 0.0, 0.0), 1.3, 6.0, 0.65)
        n2 = noise(nt, vmath(nt, "MULTIPLY", p, (0.6, 1.6, 1.0)), 1.8, 5.0, 0.6)
        core = nmath(nt, "MULTIPLY", map_range(nt, along, 0.0, 30.0, 1.0, 0.1), map_range(nt, across, 0.75, 0.2))
        edge = nmath(nt, "MULTIPLY", map_range(nt, nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", across, 0.85)), 0.15, 0.0),
                     map_range(nt, along, 0.0, 55.0, 0.8, 0.0))
        foam = nmath(nt, "MULTIPLY", nmath(nt, "MAXIMUM", core, edge), map_range(nt, nmath(nt, "ADD", n1, nmath(nt, "MULTIPLY", n2, 0.4)), 0.45, 0.75))
        foam = nmath(nt, "MULTIPLY", foam, map_range(nt, along, 0.0, 1.5))
        return psdf(nt, srgb(226, 234, 236), rough=0.5, alpha=nmath(nt, "MINIMUM", nmath(nt, "MULTIPLY", foam, 1.3), 1.0))
    M.wake = mat("WakeFoam", wake)

    def breakline(nt):
        """The temperature break: a ragged scum line of weed and foam, greener water on its far side (alpha)."""
        p = attr(nt, "rest", "Vector")
        o = separate(nt, p)
        wob = nmath(nt, "MULTIPLY", nmath(nt, "SUBTRACT", noise(nt, vmath(nt, "MULTIPLY", p, (0.05, 0.05, 1.0)), 1.0, 3.0), 0.5), 1.2)
        dy = nmath(nt, "SUBTRACT", nmath(nt, "SUBTRACT", BREAK_Y, o["Y"]), wob)
        line = nmath(nt, "MULTIPLY", map_range(nt, dy, 0.0, 0.35), map_range(nt, dy, 1.4, 0.5))
        weed = map_range(nt, noise(nt, vmath(nt, "MULTIPLY", p, (0.5, 2.0, 1.0)), 2.5, 6.0, 0.65), 0.36, 0.55)
        line = nmath(nt, "MULTIPLY", line, weed)
        tint = nmath(nt, "MULTIPLY", map_range(nt, dy, 0.3, 2.5), map_range(nt, dy, 14.0, 9.0))
        col = mix_rgb(nt, line, srgb(34, 92, 70), srgb(196, 164, 84))
        alpha = nmath(nt, "MAXIMUM", nmath(nt, "MULTIPLY", tint, 0.7), line)
        return psdf(nt, col, rough=0.25, coat=0.4, alpha=alpha)
    M.breakline = mat("TemperatureBreak", breakline)

    def surface_under(nt):
        """The surface from below: a bright rippling window overhead, darker total-reflection rim."""
        p = tex_coord(nt, "Object")
        rip = looped_noise(nt, p, (0.4, 0.2, 0.0), 0.35, 5.0, 0.6)
        inc = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Incoming"])["Z"]
        window = map_range(nt, nmath(nt, "ABSOLUTE", inc), 0.62, 0.8)
        col = mix_rgb(nt, window, srgb(10, 30, 40), srgb(150, 196, 204))
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
        em.inputs["Color"].default_value = (*srgb(196, 230, 236), 1.0)
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

    # --- the bluefin -----------------------------------------------------------------------------------------
    def tuna(nt):
        """Atlantic bluefin: deep metallic blue-black back, a faint iridescent band, silvery white lower sides and
        belly crossed by colourless lines alternating with rows of colourless dots; a slick coat of slime."""
        z, x = attr(nt, "zrel", "Fac"), attr(nt, "xrel", "Fac")
        p = tex_coord(nt, "Object")
        pz = separate(nt, p)
        base = color_ramp(nt, z, [(0.0, srgb(206, 210, 214)), (0.34, srgb(186, 192, 198)), (0.5, srgb(128, 140, 152)),
                                  (0.58, srgb(52, 70, 96)), (0.66, srgb(16, 26, 46)), (1.0, srgb(6, 10, 22))])
        band = nmath(nt, "MULTIPLY", map_range(nt, z, 0.5, 0.56), map_range(nt, z, 0.64, 0.58))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", band, 0.55), base, srgb(60, 120, 170))
        lines = nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY", pz["X"], 46.0))), 24.0)
        dots = map_range(nt, voronoi(nt, vmath(nt, "MULTIPLY", p, (23.0, 40.0, 40.0)), 1.0), 0.22, 0.12)
        alt = map_range(nt, nmath(nt, "SINE", nmath(nt, "MULTIPLY", pz["X"], 23.0)), -0.2, 0.2)
        mark = nmath(nt, "ADD", nmath(nt, "MULTIPLY", lines, alt), nmath(nt, "MULTIPLY", dots, nmath(nt, "SUBTRACT", 1.0, alt)))
        lower = nmath(nt, "MULTIPLY", map_range(nt, z, 0.12, 0.25), map_range(nt, z, 0.55, 0.42))
        lower = nmath(nt, "MULTIPLY", lower, map_range(nt, x, 0.08, 0.2))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", mark, lower), 0.35), col, srgb(236, 240, 244))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", attr(nt, "keelw", "Fac"), 0.9), col, srgb(10, 12, 16))   # black keel
        seam = attr(nt, "seam", "Fac")
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", seam, 0.8), col, srgb(14, 16, 20))
        n = noise(nt, p, 40.0, 4.0, 0.5)
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(uw_base(nt, col, caustic=0.25), b.inputs["Base Color"])
        _feed(nt, b.inputs["Metallic"], map_range(nt, z, 0.2, 0.7, 0.55, 0.25))
        _feed(nt, b.inputs["Roughness"], map_range(nt, n, 0.0, 1.0, 0.22, 0.34, smooth=False))
        b.inputs["Coat Weight"].default_value = 0.6
        b.inputs["Coat Roughness"].default_value = 0.08
        b.inputs["Subsurface Weight"].default_value = 0.06
        b.inputs["Subsurface Scale"].default_value = 0.01
        nt.links.new(bump(nt, nmath(nt, "MULTIPLY_ADD", n, 0.2, nmath(nt, "MULTIPLY", attr(nt, "gill", "Fac"), -1.0)), 0.25, 0.004),
                     b.inputs["Normal"])
        return b.outputs[0]
    M.tuna = mat("BluefinSkin", tuna)

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
    M.fin_dark = fin("BluefinFinDark", srgb(20, 30, 48), srgb(10, 14, 22))
    M.fin_dorsal1 = fin("BluefinFirstDorsal", srgb(54, 62, 70), srgb(120, 108, 62))
    M.fin_dorsal2 = fin("BluefinSecondDorsal", srgb(92, 66, 54), srgb(40, 34, 32))
    M.fin_pale = fin("BluefinPaleFin", srgb(160, 166, 172), srgb(70, 78, 88), metal=0.4)

    def finlet(nt):
        g = separate(nt, tex_coord(nt, "Generated"))
        e = map_range(nt, nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", g["Z"], 0.5)), 0.32, 0.46)
        col = mix_rgb(nt, e, srgb(214, 178, 52), srgb(18, 18, 20))
        return psdf(nt, uw_base(nt, col, caustic=0.2), rough=0.32, coat=0.5, coat_rough=0.1)
    M.finlet = mat("BluefinFinlet", finlet)
    M.keel = mat("BluefinKeel", lambda nt: psdf(nt, uw_base(nt, srgb(14, 16, 22), caustic=0.1), rough=0.25, coat=0.6))

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

    def mackerel(nt):
        """Atlantic mackerel: green-blue back with dark wavy bars, silver-white belly."""
        z = attr(nt, "zrel", "Fac")
        p = tex_coord(nt, "Object")
        pz = separate(nt, p)
        base = color_ramp(nt, z, [(0.0, srgb(226, 228, 230)), (0.45, srgb(196, 206, 208)), (0.58, srgb(60, 120, 120)),
                                  (1.0, srgb(24, 70, 84))])
        wav = nmath(nt, "SINE", nmath(nt, "ADD", nmath(nt, "MULTIPLY", pz["X"], 160.0),
                                     nmath(nt, "MULTIPLY", nmath(nt, "SINE", nmath(nt, "MULTIPLY", pz["Z"], 90.0)), 2.2)))
        bars = nmath(nt, "MULTIPLY", map_range(nt, wav, 0.3, 0.7), map_range(nt, z, 0.55, 0.65))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", bars, 0.9), base, srgb(8, 16, 22))
        return psdf(nt, uw_base(nt, col, caustic=0.25), rough=0.25, metal=0.35, coat=0.6, coat_rough=0.08)
    M.mackerel = mat("Mackerel", mackerel)
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

    # --- the water column diagram -------------------------------------------------------------------------------
    def thermo_sheet(nt):
        """The thermocline drawn as a thin shimmering sheet: faint, brighter where it is seen edge on."""
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.3
        edge = nmath(nt, "POWER", lw.outputs["Facing"], 2.0)
        p = tex_coord(nt, "Object")
        shimmer = map_range(nt, looped_noise(nt, vmath(nt, "MULTIPLY", p, (0.4, 0.4, 1.0)), (0.2, 0.0, 0.0), 1.5), 0.3, 0.7, 0.4, 1.0)
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(120, 220, 230), 1.0)
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        nt.links.new(nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY_ADD", edge, 0.12, 0.012), shimmer),
                           oi.outputs["Alpha"]), em.inputs["Strength"])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(tr.outputs[0], add.inputs[1])
        return add.outputs[0]
    M.thermo = new_material("ThermoclineSheet", thermo_sheet, blended=True)

    def column(nt):
        """Backdrop of the cross section: warm green-teal surface water fading through the thermocline into cold,
        black deep water (by world height)."""
        z = separate(nt, world_pos(nt))["Z"]
        col = color_ramp(nt, map_range(nt, z, -60.0, 0.0, 0.0, 1.0, smooth=False),
                         [(0.0, srgb(1, 3, 8)), (0.4, srgb(3, 10, 24)), (0.528, srgb(8, 28, 54)), (0.548, srgb(34, 96, 102)),
                          (0.75, srgb(58, 126, 118)), (1.0, srgb(100, 165, 150))])      # warm above the layer, cold below
        line = map_range(nt, nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", z, THERMO_Z)), 0.9, 0.15)   # the layer, drawn
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", line, 0.8), col, srgb(150, 236, 240))
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(col, em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0
        return em.outputs[0]
    M.column = new_material("WaterColumn", column)
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
# The giant bluefin (Thunnus thynnus). Proportions after the species' descriptions: a deep, round, torpedo body,
# deepest near the first dorsal fin; a pointed conical snout and a large eye; short pectoral fins that do not
# reach the gap between the dorsal fins; a spiny first dorsal folding into a groove, a tall second dorsal and anal
# fin; 8 to 10 yellow finlets above and 7 to 9 below; a strong lateral keel between two small keels on a slender
# tail stock; a stiff, crescent (lunate) tail. Modelled 3.26 m long, the length reported for Fraser's fish.
# --------------------------------------------------------------------------
TUNA_L = 2.85                        # snout to the base of the tail (m); the tail reaches about 0.41 m further


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


def tuna_curves():
    L = TUNA_L
    # profile: a straight conical head (no forehead step), deepest at u 0.4, a long even taper to a slender,
    # keeled tail stock that narrows into the tail
    pts_top = [(0.0, -0.012), (0.02, 0.03), (0.06, 0.098), (0.12, 0.186), (0.2, 0.282), (0.3, 0.37), (0.4, 0.418),
               (0.5, 0.405), (0.62, 0.33), (0.74, 0.215), (0.85, 0.102), (0.93, 0.05), (1.0, 0.026)]
    pts_bot = [(0.0, -0.03), (0.02, -0.058), (0.06, -0.104), (0.12, -0.17), (0.2, -0.252), (0.3, -0.338), (0.4, -0.39),
               (0.5, -0.378), (0.62, -0.305), (0.74, -0.196), (0.85, -0.091), (0.93, -0.044), (1.0, -0.022)]
    pts_wid = [(0.0, 0.0), (0.02, 0.034), (0.06, 0.085), (0.12, 0.15), (0.2, 0.215), (0.3, 0.27), (0.4, 0.3),
               (0.5, 0.29), (0.62, 0.238), (0.74, 0.16), (0.85, 0.084), (0.93, 0.058), (1.0, 0.034)]
    top_u, bot_u, wid_u = pchip(pts_top), pchip(pts_bot), pchip(pts_wid)

    def u_of(x):
        return (L / 2 - x) / L
    return (lambda x: top_u(u_of(x))), (lambda x: bot_u(u_of(x))), (lambda x: wid_u(u_of(x))), u_of


def tuna_surface(x, z_rel, side=1):
    """A point on the skin at body position x, at relative height z_rel (-1 belly .. 1 back) on one side."""
    top, bot, wid, _u = tuna_curves()
    zc, hh, w = (top(x) + bot(x)) / 2, (top(x) - bot(x)) / 2, wid(x)
    s = max(-1.0, min(1.0, z_rel))
    y = w * max(0.0, 1.0 - abs(s) ** 2.1) ** (1 / 2.1)
    return Vector((x, side * y, zc + hh * s))


def tuna_mesh(M):
    L = TUNA_L
    top, bot, wid, u_of = tuna_curves()
    bm = bmesh.new()
    zl, xl = bm.verts.layers.float.new("zrel"), bm.verts.layers.float.new("xrel")
    sl, gl, kl = bm.verts.layers.float.new("seam"), bm.verts.layers.float.new("gill"), bm.verts.layers.float.new("keelw")
    er = bm.verts.layers.float.new("eyer")
    nr, na, pw = 140, 72, 2.1
    rings = []
    for i in range(1, nr + 1):
        u = (i / nr) ** 1.25
        x = L / 2 - u * L
        t, b, w = top(x), bot(x), wid(x)
        zc, hh = (t + b) / 2, (t - b) / 2
        keel = smoothstep(0.83, 0.9, u)
        ring = []
        for k in range(na):
            a = 2 * math.pi * k / na
            c, s = math.cos(a), math.sin(a)
            y = w * math.copysign(abs(c) ** (2 / pw), c)
            z = zc + hh * math.copysign(abs(s) ** (2 / pw), s)
            kz = (z - zc) / max(hh, 1e-4)
            ridge = keel * (0.05 * math.exp(-(kz / 0.16) ** 2) + 0.014 * smoothstep(0.93, 0.97, u) *
                            (math.exp(-((kz - 0.62) / 0.12) ** 2) + math.exp(-((kz + 0.62) / 0.12) ** 2)))
            y += math.copysign(ridge, c)
            v = bm.verts.new((x, y, z))
            v[zl] = min(1.0, max(0.0, (z - b) / max(1e-6, t - b)))
            v[xl] = 1.0 - u
            v[kl] = min(1.0, ridge / 0.03)
            # the mouth: a fine seam from the snout back to below the eye, curving down a little
            if u < 0.1:
                sz = -0.38 + 0.12 * (u / 0.1)
                d = (kz - sz) * hh
                v[sl] = math.exp(-(d / 0.006) ** 2) * smoothstep(0.1, 0.085, u) * smoothstep(0.0, 0.25, abs(c))
            # the gill cover: its free edge bows back at mid height and sweeps forward under the throat
            ug = 0.205 + 0.02 * math.cos(kz * 1.4) - 0.03 * max(0.0, -kz) ** 1.5
            if abs(u - ug) < 0.04 and kz < 0.62:
                d = (u - ug) * L
                v[gl] = math.exp(-(d / 0.008) ** 2) * smoothstep(0.62, 0.45, kz)
                v.co += Vector((0.0, math.copysign(1.0, c) * abs(c), s)).normalized() * (
                    -0.006 * v[gl] + 0.004 * smoothstep(0.0, 0.03, -d) * smoothstep(0.08, 0.02, -d) * smoothstep(0.62, 0.45, kz))
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
    ex = L / 2 - 0.105 * L
    for s in (1, -1):
        c = tuna_surface(ex, 0.34, s) - Vector((0.0, s * 0.006, 0.0))
        faces = ellipsoid(bm, c, (0.036, 0.016, 0.034), subdiv=5, mat=1)
        axis = Vector((0.08, s * 1.0, 0.12)).normalized()
        for v in (v for f in faces for v in f.verts):        # (BMVert sets are unreliable: walk the new faces)
            d = (v.co - c).normalized()
            v[er] = math.acos(max(-1.0, min(1.0, d.dot(axis)))) / (math.pi / 2)
    return fish_mesh("Bluefin", bm, [M.tuna, M.eye])


def make_tuna(col, M, name="Bluefin"):
    L = TUNA_L
    top, bot, wid, u_of = tuna_curves()
    me = tuna_mesh(M)
    body = bpy.data.objects.new(name, me)
    col.objects.link(body)

    def xu(u):
        return L / 2 - u * L
    fb = bmesh.new()
    x1 = xu(0.33)                                        # spiny first dorsal: tall in front, falling to a low rear
    soft_fin(fb, [fin_base(top, x1, 0.02), (x1 - 0.015, top(x1) + 0.26), (x1 - 0.06, top(x1 - 0.06) + 0.22),
                  (x1 - 0.16, top(x1 - 0.16) + 0.11), (x1 - 0.32, top(x1 - 0.32) + 0.045), (x1 - 0.46, top(x1 - 0.46) + 0.02),
                  fin_base(top, x1 - 0.47, 0.02)], 0.008, 0, samples=8)
    x2 = xu(0.515)                                       # tall, swept second dorsal
    soft_fin(fb, [fin_base(top, x2, 0.02), (x2 - 0.03, top(x2) + 0.16), (x2 - 0.12, top(x2) + 0.33),
                  (x2 - 0.16, top(x2) + 0.3), (x2 - 0.17, top(x2 - 0.17) + 0.1), fin_base(top, x2 - 0.2, 0.02)],
             0.01, 1, samples=8)
    xa = xu(0.56)                                        # anal fin, its mirror below
    soft_fin(fb, [fin_base(bot, xa, 0.02), (xa - 0.03, bot(xa) - 0.15), (xa - 0.12, bot(xa) - 0.3),
                  (xa - 0.16, bot(xa) - 0.27), (xa - 0.17, bot(xa - 0.17) - 0.09), fin_base(bot, xa - 0.2, 0.02)],
             0.01, 1, samples=8)
    for i in range(9):                                   # finlets: 9 above, 8 below, yellow edged with black
        u = 0.625 + i * 0.033
        x = xu(u)
        hgt = 0.052 - 0.016 * i / 8
        soft_fin(fb, [fin_base(top, x, 0.008), (x - 0.008, top(x) + hgt * 0.75), (x - 0.042, top(x) + hgt),
                      (x - 0.036, top(x) + hgt * 0.45), fin_base(top, x - 0.04, 0.008)], 0.005, 2, samples=5)
        if i < 8:
            u = 0.655 + i * 0.033
            x = xu(u)
            soft_fin(fb, [fin_base(bot, x, 0.008), (x - 0.008, bot(x) - hgt * 0.75), (x - 0.042, bot(x) - hgt),
                          (x - 0.036, bot(x) - hgt * 0.45), fin_base(bot, x - 0.04, 0.008)], 0.005, 2, samples=5)
    xt = -L / 2                                          # the lunate tail, stiff and swept
    soft_fin(fb, [(xt + 0.07, 0.03), (xt - 0.03, 0.15), (xt - 0.16, 0.33), (xt - 0.31, 0.5), (xt - 0.33, 0.51),
                  (xt - 0.27, 0.42), (xt - 0.16, 0.21), (xt - 0.11, 0.05), (xt - 0.105, 0.0), (xt - 0.11, -0.05),
                  (xt - 0.16, -0.21), (xt - 0.27, -0.42), (xt - 0.33, -0.51), (xt - 0.31, -0.5), (xt - 0.16, -0.33),
                  (xt - 0.03, -0.15), (xt + 0.07, -0.03)], 0.014, 3, samples=6)
    fins_me = fish_mesh(name + "Fins", fb, [M.fin_dorsal1, M.fin_dorsal2, M.finlet, M.fin_dark])
    fins = bpy.data.objects.new(name + "Fins", fins_me)
    col.objects.link(fins)
    fins.parent = body
    paired = []
    for s in (1, -1):                                    # short, sickle pectorals lying back along the side
        pb = bmesh.new()
        soft_fin(pb, [(0.03, 0.02), (-0.08, 0.05), (-0.28, 0.045), (-0.48, 0.0), (-0.52, -0.025), (-0.33, -0.04),
                      (-0.12, -0.055), (0.03, -0.04)], 0.014, 0, samples=6)
        pm = fish_mesh(f"{name}Pectoral{'L' if s > 0 else 'R'}", pb, [M.fin_dark])
        po = bpy.data.objects.new(pm.name, pm)
        col.objects.link(po)
        po.parent = body
        xq = xu(0.215)
        base = tuna_surface(xq, -0.1, s)
        po.location = base - Vector((0.0, s * 0.01, 0.0))
        po.rotation_euler = (s * math.radians(24), math.radians(10), s * math.radians(-24))
        paired.append(po)
        vb = bmesh.new()                                 # small pelvic fins under the pectorals
        soft_fin(vb, [(0.01, 0.0), (-0.06, -0.01), (-0.17, -0.035), (-0.12, -0.012), (0.0, 0.01)], 0.008, 0, samples=5)
        vm = fish_mesh(f"{name}Pelvic{'L' if s > 0 else 'R'}", vb, [M.fin_pale])
        vo = bpy.data.objects.new(vm.name, vm)
        col.objects.link(vo)
        vo.parent = body
        xv = xu(0.27)
        vo.location = tuna_surface(xv, -0.93, s) - Vector((0.0, s * 0.01, -0.005))
        vo.rotation_euler = (s * math.radians(-35), math.radians(-6), s * math.radians(-6))
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
# Small fish: the trolled mackerel and a school of baitfish
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


def mackerel_mesh(M, L=0.36):
    top = curve([(0.0, 0.0), (0.08, 0.05), (0.3, 0.1), (0.5, 0.095), (0.75, 0.05), (0.92, 0.022), (1.0, 0.02)])
    bot = curve([(0.0, -0.02), (0.08, -0.05), (0.3, -0.09), (0.5, -0.085), (0.75, -0.045), (0.92, -0.02), (1.0, -0.02)])
    wid = curve([(0.0, 0.0), (0.1, 0.04), (0.35, 0.07), (0.6, 0.055), (0.9, 0.018), (1.0, 0.012)])

    def fins(L, top, bot):
        t = -L / 2
        return [[(0.12 * L, top(0.12 * L) - 0.005 * L), (0.1 * L, top(0.1 * L) + 0.07 * L), (0.0, top(0.0) + 0.04 * L),
                 (-0.04 * L, top(-0.04 * L) - 0.005 * L)],
                [(t + 0.03 * L, 0.02 * L), (t - 0.08 * L, 0.11 * L), (t - 0.12 * L, 0.11 * L), (t - 0.07 * L, 0.0),
                 (t - 0.12 * L, -0.11 * L), (t - 0.08 * L, -0.11 * L), (t + 0.03 * L, -0.02 * L)]]
    return small_fish("Mackerel", M, M.mackerel, M.small_fin, L, lambda u: top(u) * L, lambda u: bot(u) * L,
                      lambda u: wid(u) * L, fins, nr=28, na=16, eye_r=0.035)


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
# The boat: a 50 ft (15.2 m) sportfisher. Modelled with the transom at x = 0, the bow at x = +15.2, waterline z = 0;
# cockpit aft (sole 0.7 m above the water) with the fighting chair, salon, flybridge, tuna tower and outriggers.
# --------------------------------------------------------------------------
BOAT_LEN = 15.2
COCKPIT_SOLE = 0.72
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


def build_reel(col, M, parent=None):
    """A big two-speed lever drag reel in gold: spool between side plates, pillars, drag lever, crank handle.
    Local frame: the spool axis along X, the rod along Y (the reel sits on top of the rod, +Z)."""
    root = empty_object("Reel14", col, size=0.1, parent=parent)
    bm = bmesh.new()
    for x in (-0.068, 0.068):                                   # side plates
        bm_frustum(bm, (x - 0.012, 0, 0.085), (x + 0.012, 0, 0.085), 0.078, 0.078, segs=28)
    for a in range(4):                                          # pillars
        ang = math.pi / 4 + a * math.pi / 2
        p = (0, 0.07 * math.cos(ang), 0.085 + 0.07 * math.sin(ang))
        bm_frustum(bm, (-0.06, p[1], p[2]), (0.06, p[1], p[2]), 0.007, 0.007, segs=8)
    bm_box(bm, (0.0, 0.0, 0.012), (0.05, 0.16, 0.022))           # reel foot / clamp
    gold = mesh_object("ReelFrame", bm, col, [M.reel_gold], parent=root)
    bm = bmesh.new()
    bm_frustum(bm, (-0.056, 0, 0.085), (0.056, 0, 0.085), 0.062, 0.062, segs=28)   # line on the spool
    spool = mesh_object("ReelSpool", bm, col, [M.line], parent=root)
    bm = bmesh.new()
    bm_frustum(bm, (0.08, 0, 0.1), (0.13, 0, 0.1), 0.012, 0.012, segs=10)          # crank shaft
    bm_box(bm, (0.135, 0.0, 0.05), (0.014, 0.025, 0.13))                          # crank arm
    bm_frustum(bm, (0.135, 0, -0.012), (0.2, 0, -0.012), 0.016, 0.014, segs=12)   # handle knob
    bm_box(bm, (-0.088, -0.03, 0.16), (0.012, 0.02, 0.1), rot=(math.radians(-30), 0, 0))   # drag lever
    bm_frustum(bm, (-0.088, -0.06, 0.205), (-0.088, -0.066, 0.215), 0.012, 0.012, segs=10)
    dark = mesh_object("ReelHandles", bm, col, [M.reel_dark], parent=root)
    for o in (gold, spool, dark):
        for p in o.data.polygons:
            p.use_smooth = True
    return root


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
    reel = build_reel(col, M, parent=mount)
    reel.location = (0.0, 0.05, 0.02)
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
# The angler, strapped into the chair: Cesium Man (Khronos sample, CC BY 4.0) re-dressed in orange foul weather
# gear, a cap and sunglasses; seated with IK hands on the rod and the legs bent to the footrest.
# --------------------------------------------------------------------------
ANGLER_GLB = os.path.join(HERE, "assets", "khronos", "CesiumMan.glb")


def import_glb(path, **kw):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path, **kw)
    return [o for o in bpy.data.objects if o not in before]


def attr_vec(nt, name):
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_name = name
    return a.outputs["Vector"]


def build_angler(col, M):
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
    if arm.parent is not None:
        mw = arm.matrix_world.copy()
        arm.parent = None
        arm.matrix_world = mw
    arm.animation_data_clear()
    for b in arm.pose.bones:
        b.rotation_mode = "QUATERNION"
        b.rotation_quaternion = (1, 0, 0, 0)
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
    for m in me.materials:
        for nd in (m.node_tree.nodes if m and m.node_tree else []):
            if nd.bl_idname == "ShaderNodeTexImage" and nd.image:
                bpy.data.images.remove(nd.image)
    me.materials.clear()

    def clothes(nt):
        r = separate(nt, attr_vec(nt, "rest"))
        ax = nmath(nt, "ABSOLUTE", r["X"])
        hand_m = map_range(nt, ax, 0.40, 0.43)
        boot_m = map_range(nt, r["Z"], 0.11, 0.08)
        bib_m = map_range(nt, r["Z"], 0.6, 0.64)
        n = noise(nt, attr_vec(nt, "rest"), 60.0)
        col_ = mix_rgb(nt, bib_m, srgb(214, 110, 28), srgb(64, 70, 78))          # orange bib trousers, grey top
        col_ = mix_rgb(nt, boot_m, col_, srgb(230, 230, 226))                     # white deck boots
        col_ = mix_rgb(nt, hand_m, col_, srgb(196, 140, 106))
        col_ = vmath(nt, "SCALE", col_, scale=nmath(nt, "MULTIPLY_ADD", n, 0.2, 0.9))
        return psdf(nt, col_, rough=0.5, coat=0.3, normal=bump(nt, n, 0.2))
    me.materials.append(new_material("AnglerClothes", clothes, M.atmos))
    for p in me.polygons:
        p.use_smooth = True
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
    tgt = {s: empty_object(f"AnglerHand{s}", col, size=0.03) for s in ("L", "R")}
    for s, bn in (("L", "Skeleton_arm_joint_L__2_"), ("R", "Skeleton_arm_joint_R__3_")):
        c = arm.pose.bones[bn].constraints.new("IK")
        c.target = tgt[s]
        c.chain_count = 2
    feet = {s: empty_object(f"AnglerFoot{s}", col, size=0.03) for s in ("L", "R")}
    knees = {s: empty_object(f"AnglerKnee{s}", col, size=0.03) for s in ("L", "R")}
    for s in ("L", "R"):
        c = arm.pose.bones[f"leg_joint_{s}_2"].constraints.new("IK")
        c.target = feet[s]
        c.pole_target = knees[s]
        c.pole_angle = math.radians(-90)
        c.chain_count = 2
    arm["base_q"] = list(arm.matrix_world.to_quaternion())
    arm.scale = (1.24, 1.24, 1.24)
    arm.rotation_mode = "QUATERNION"
    return dict(arm=arm, body=body, hands=tgt, feet=feet, knees=knees, head=hroot)


HIP_REST = Vector((0.005, -0.024, 0.614))            # the hip joint in the model's rest frame (before scaling)


def seat_angler(ang, f, seat_w, facing, lean, foot_w, knee_up, hands=None, keyframe=True):
    """Put the angler's hips on the seat, facing 'facing', leaning back by 'lean' (radians), feet on the bar.
    The model faces -Y at rest; its glTF Y-up fix lives in base_q, so new rotations compose on top of it."""
    arm = ang["arm"]
    yaw = math.atan2(facing.y, facing.x) - math.atan2(-1.0, 0.0)
    right = facing.cross(Vector((0, 0, 1))).normalized()
    base = Quaternion(arm["base_q"])
    q = Quaternion(right, lean) @ Euler((0, 0, yaw)).to_quaternion() @ base
    arm.rotation_quaternion = q
    hip_world = (q @ base.inverted()) @ (HIP_REST * 1.24)
    arm.location = Vector(seat_w) - hip_world
    for s, side in (("L", 1), ("R", -1)):
        lat = -right * (0.11 * side)
        ang["feet"][s].location = Vector(foot_w) + lat
        ang["knees"][s].location = (Vector(seat_w) + Vector(foot_w)) / 2 + lat + Vector((0, 0, knee_up))
        if keyframe:
            key(ang["feet"][s], "location", f)
            key(ang["knees"][s], "location", f)
    if hands:
        for s in ("L", "R"):
            ang["hands"][s].location = hands[s]
            if keyframe:
                key(ang["hands"][s], "location", f)
    if keyframe:
        key(arm, "rotation_quaternion", f)
        key(arm, "location", f)


# --------------------------------------------------------------------------
# Choreography. Three places: the boat near the origin (the sea grid is finest there), open water for the hook
# (UW1) and the water column cross section (UW2), far apart and each rendered only in its own shots.
# --------------------------------------------------------------------------
UW1 = Vector((420.0, 0.0, 0.0))
UW2 = Vector((-420.0, 0.0, 0.0))
HOOK_DEPTH = -7.0
THERMO_Z = -28.0                     # the drawn thermocline in the cross section (illustrative, not to scale)
TROLL_SPEED = 3.0                    # m/s
BACK_SPEED = 1.8                     # m/s, backing down
BAIT_BACK = 38.0                     # the mackerel trails this far astern
BREAK_Y = -1.0                       # the temperature break runs along the boat's track, just to port of the bait
DIVE_F = P(3, 3)                     # "And they can dive past three thousand feet"
HZ_POS = Vector((-30.0, -150.0, 3.2))                            # the horizon shot (the loop's join)
HZ_TGT = HZ_POS + Vector((0.9, 0.3, 0.035)).normalized() * 100.0     # away from the sun: the dark horizon of the brief


def smooth_vectors(vals, k):
    arr = np.array([list(v) for v in vals])
    out = arr.copy()
    for i in range(len(arr)):
        lo, hi = max(0, i - k), min(len(arr), i + k + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    return [Vector(v) for v in out]


def hook_fish(f):
    """The giant towing the line: position, heading (unit), bank. Cruising about 1.7 m/s with slow weaving turns."""
    t = t_of(f) - t_of(WHIP_IN)
    x = UW1.x - 14.0 + 1.7 * t
    y = UW1.y + 0.9 * math.sin(t * 0.45) + 0.25 * math.sin(t * 1.3)
    z = HOOK_DEPTH + 0.5 * math.sin(t * 0.33)
    dx = 1.7
    dy = 0.9 * 0.45 * math.cos(t * 0.45) + 0.25 * 1.3 * math.cos(t * 1.3)
    dz = 0.5 * 0.33 * math.cos(t * 0.33)
    fwd = Vector((dx, dy, dz)).normalized()
    return Vector((x, y, z)), fwd, -dy * 0.25


def gear_boat_x(f):
    return -16.0 + TROLL_SPEED * (t_of(f) - t_of(CUT1))


def fight_boat_x(f):
    return 9.0 - BACK_SPEED * (t_of(f) - t_of(CUT3))


def thermo_fish(i, f):
    """Bluefin i in the cross section: cruising along the thermocline; the lead fish dives on its line."""
    t = t_of(f) - t_of(CUT2)
    base = [Vector((-9.0, 0.8, THERMO_Z + 0.6)), Vector((-12.5, -1.6, THERMO_Z - 0.8)), Vector((-15.5, 1.9, THERMO_Z + 0.2))][i]
    speed = 1.9
    p = UW2 + base + Vector((speed * t, 0.3 * math.sin(t * 0.6 + i), 0.35 * math.sin(t * 0.4 + 2 * i)))
    fwd = Vector((1.0, 0.18 * math.cos(t * 0.6 + i), 0.14 * math.cos(t * 0.4 + 2 * i))).normalized()
    if i == 0 and f > DIVE_F - 6:
        td = t_of(f) - t_of(DIVE_F - 6)
        k = smoothstep(0.0, 1.6, td)
        drop = 0.0
        steps = int(td * FPS)
        for s_ in range(steps):                               # integrate the dive (pitching down to about 55 deg)
            ks = smoothstep(0.0, 1.6, s_ / FPS)
            drop += math.sin(math.radians(55) * ks) * 2.6 / FPS
        p = p + Vector((-0.6 * drop * 0.3, 0.0, -drop))
        fwd = Vector((math.cos(math.radians(55) * k), 0.0, -math.sin(math.radians(55) * k))).normalized()
    return p, fwd


def bait_pos(f):
    """The trolled mackerel, just under the surface far astern, with a little swimming wobble."""
    t = t_of(f)
    x = gear_boat_x(f) - BAIT_BACK
    y = -0.35 + 0.25 * math.sin(t * 2.1)
    return Vector((x, y, surface_z(x, y, t) + 0.06))             # skipping on the surface, as trolled baits do


# --------------------------------------------------------------------------
# Camera: frames[f] = (position, target, rotation, lens); f-stop per frame
# --------------------------------------------------------------------------
def whip(q, angle_deg, axis):
    """Rotation q turned by angle_deg about a world axis (the whip pans)."""
    return Quaternion(Vector(axis), math.radians(angle_deg)) @ q


def camera_frames():
    frames, fstop = {}, {}
    Z = Vector((0, 0, 1))
    hz_q = look_at(HZ_POS, HZ_TGT, Z)
    # 0 the loop's join: the horizon, then a whip pan down into the water
    for f in range(1, WHIP_IN):
        ang = 0.0 if f <= 3 else -(12.0 * (f - 3) ** 1.6)
        frames[f] = (HZ_POS.copy(), HZ_TGT.copy(), whip(hz_q, ang, hz_q @ Vector((1, 0, 0))), 26.0)
        fstop[f] = 11.0
    # 1 hook: tracking alongside the giant, a little below her, so she reads against the bright surface
    for f in range(WHIP_IN, CUT1):
        p, fwd, _b = hook_fish(f)
        u = (f - WHIP_IN) / (CUT1 - WHIP_IN)
        side = fwd.cross(Z).normalized()
        cam = p - side * (5.4 - 0.8 * u) + fwd * (0.5 - 1.4 * u) + Vector((0, 0, -0.65))
        tgt = p - fwd * (0.25 + 0.15 * u) + Vector((0, 0, 0.12))          # centred on the body, tail in frame
        q = look_at(cam, tgt, Z)
        if f < WHIP_IN + 6:                                       # the whip lands: residual turn settles fast
            k = 1.0 - ease_out((f - WHIP_IN) / 6.0, 3.0)
            q = whip(q, 38.0 * k, q @ Vector((1, 0, 0)))
        frames[f] = (cam, tgt, q, 22.0 + 4.0 * u)
        fstop[f] = 4.0
    # 2 gear: top down over the cockpit (bow at the top of the frame), easing aft over the transom along the line;
    # then a hard cut to the trolled mackerel far astern at the temperature break (a long fast move only smears)
    for f in range(CUT1, CUT2):
        bx = gear_boat_x(f)
        if f < CUT1B:
            u = (f - CUT1) / (CUT1B - CUT1)
            k = ease(smoothstep(0.25, 1.0, u))
            cam = Vector((bx + CHAIR_AT.x - 0.3, 0.15, 6.4)).lerp(Vector((bx - 2.6, 0.0, 7.6)), k)
            lens = 30.0
        else:
            u = (f - CUT1B) / (CUT2 - CUT1B)
            cam = Vector((bait_pos(f).x - 0.6, -1.05, 5.0 - 0.5 * ease(u)))
            lens = 28.0
        tgt = Vector((cam.x - 0.6, cam.y - 0.05, 0.0))
        q = look_at(cam, tgt, Vector((1, 0, 0)))
        frames[f] = (cam, tgt, q, lens)
        fstop[f] = 11.0
    # 3 the water column, side on, long and clean; the camera follows the school, then tilts after the diving fish
    for f in range(CUT2, CUT3):
        t = t_of(f) - t_of(CUT2)
        lead, _fw = thermo_fish(0, min(f, DIVE_F - 6))
        base_x = UW2.x - 10.0 + 1.75 * t
        cam = Vector((base_x, -19.0, THERMO_Z + 0.4))             # level with the layer: it reads as a line
        tgt = Vector((base_x + 1.0, 0.0, THERMO_Z + 0.7))
        if f > DIVE_F - 4:
            d, _ = thermo_fish(0, f)
            k = smoothstep(DIVE_F - 4, DIVE_F + 40, f)
            tgt = tgt.lerp(d + Vector((0.0, 0.0, 1.0)), 0.7 * k)
            cam = cam + Vector((0.0, 0.0, -3.0 * k))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 32.0)
        fstop[f] = 16.0
        _ = lead
    # 4 the fight: low on the water off the starboard quarter as she backs down toward the lens
    for f in range(CUT3, CUT4):
        u = (f - CUT3) / (CUT4 - CUT3)
        t = t_of(f)
        bx = fight_boat_x(f)
        cx = bx - 7.6 + 1.6 * u                                    # off the quarter, so the bent rod reads side on
        cy = -7.6 + 1.0 * u
        wsum = sum(math.exp(-0.5 * (k / 4.0) ** 2) for k in range(-10, 11))
        h = sum(sea_height(cx, cy, t + k / FPS) * math.exp(-0.5 * (k / 4.0) ** 2) for k in range(-10, 11)) / wsum
        hb = sum(boat_attitude(Vector((fight_boat_x(f + k), 0.0, 0.0)), t + k / FPS, squat=1.0)[0].z
                 * math.exp(-0.5 * (k / 4.0) ** 2) for k in range(-10, 11)) / wsum
        # low, level with the boat's waterline + 2.1 m (under the angler's eye line, just over the coaming), and never
        # closer than 0.9 m to the swell under the lens
        cam = Vector((cx, cy, max(hb + 2.1, h + 0.9)))
        cam += Vector((0.0, 0.03 * math.sin(f * 0.68), 0.035 * math.sin(f * 0.41) + 0.015 * math.sin(f * 0.93)))
        tgt = Vector((bx + 2.0, 0.2, hb + 2.0))
        q = look_at(cam, tgt, Z)
        if f >= CUT4 - 3:                                          # whip out
            q = whip(q, -(16.0 * (f - (CUT4 - 4)) ** 1.6), Z)
        frames[f] = (cam, tgt, q, 30.0)
        fstop[f] = 8.0
    # 5 outro: whip in to the horizon and hold it (the hold is frame 1's shot: the loop)
    for f in range(CUT4, FRAME_END + 1):
        q = hz_q
        if f < CUT4 + 7:
            k = 1.0 - ease_out((f - CUT4) / 7.0, 3.0)
            q = whip(hz_q, 42.0 * k, Z)
        frames[f] = (HZ_POS.copy(), HZ_TGT.copy(), q, 26.0)
        fstop[f] = 11.0
    return frames, fstop


def build_camera(col, frames, fstop):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = LENS
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = SENSOR
    cd.clip_start = 0.05
    cd.clip_end = 5000.0
    cd.dof.use_dof = True
    cd.dof.aperture_blades = 7
    cam = bpy.data.objects.new("Camera", cd)
    col.objects.link(cam)
    cam.rotation_mode = "QUATERNION"
    prev = None
    for f in range(1, FRAME_END + 1):
        p, t, q, lens = frames[f]
        if prev is not None and prev.dot(q) < 0.0:          # keep the sign continuous (q and -q are one rotation)
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
    # cuts must not interpolate: the motion blur shutter (frame -+0.25) would smear the cut frame between the two
    # shots. Hold the old shot to f - 0.5, then the new shot from there (CONSTANT steps)
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
# Scene state per frame: under water or not, how deep the shot is, how clear the cross section is
# --------------------------------------------------------------------------
def under_at(f):
    return 1.0 if (WHIP_IN <= f < CUT1) or (CUT2 <= f < CUT3) else 0.0


def deep_at(f):
    if WHIP_IN <= f < CUT1:
        return 0.25
    if CUT2 <= f < CUT3:
        return 0.45 + 0.45 * smoothstep(DIVE_F, DIVE_F + 50, f)
    return 0.0


def clear_at(f):
    return 0.65 if CUT2 <= f < CUT3 else 0.0


def set_constant(idb):
    for fc in iter_fcurves(idb):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def build_sun(col):
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 3.0
    light.specular_factor = 0.4      # overcast: soft glints (full strength sparkles the rippled sea into noise)
    light.color = srgb(255, 226, 196)
    light.angle = math.radians(2.5)
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
    """The hull is an open shell, so in a steep swell the sea plane shows above the cockpit sole: make the sea
    transparent inside the cockpit (in the boat's own frame), so the cockpit stays dry."""
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
    centres = [(1, HZ_POS + Vector((25.0, 10.0, 0.0))), (CUT1, Vector((-15.0, -0.5, 0.0))),
               (CUT3, Vector((-5.0, -3.0, 0.0))), (CUT4, HZ_POS + Vector((25.0, 10.0, 0.0)))]
    for f, c in centres:
        sea.location = (c.x, c.y, 0.0)
        sea.keyframe_insert("location", frame=f - 0.5 if f > 1 else 1)      # changes between shutters, at a cut
    set_constant(sea)


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
        sun.data.energy = (4.0 * (1 - 0.6 * d)) if u else 3.0
        sun.data.keyframe_insert("energy", frame=f)
        sun.rotation_quaternion = (-(SUN_UNDER if u else SUN_DIR)).to_track_quat("-Z", "Y")
        sun.keyframe_insert("rotation_quaternion", frame=f)
        if WHIP_IN <= f < CUT1:                                       # soft top light riding with the giant
            p, fwd, _b = hook_fish(f)
            key_l.location = p + Vector((0.0, 0.0, 4.0)) - fwd * 0.5
            key_l.rotation_quaternion = (p - key_l.location).normalized().to_track_quat("-Z", "Y")
            key_l.data.energy = 900.0
            side = fwd.cross(Vector((0, 0, 1))).normalized()       # the camera's side, low: lifts the silver belly
            bounce.location = p - side * 3.0 + Vector((0.0, 0.0, -3.2))
            bounce.rotation_quaternion = (p - bounce.location).normalized().to_track_quat("-Z", "Y")
            bounce.data.energy = 260.0
        elif CUT2 <= f < CUT3:
            cam = UW2 + Vector((-10.0 + 1.75 * (t_of(f) - t_of(CUT2)), 0.0, THERMO_Z + 9.0))
            key_l.location = cam
            key_l.rotation_quaternion = Vector((0, 0, -1)).to_track_quat("-Z", "Y")
            key_l.data.energy = 2500.0
            bounce.location = cam + Vector((0.0, -12.0, -16.0))
            bounce.rotation_quaternion = Vector((0.0, 0.6, 0.8)).to_track_quat("-Z", "Y")
            bounce.data.energy = 900.0
        else:
            key_l.data.energy = 0.0
        if not (WHIP_IN <= f < CUT1 or CUT2 <= f < CUT3):
            bounce.data.energy = 0.0
        key_l.data.keyframe_insert("energy", frame=f)
        bounce.data.keyframe_insert("energy", frame=f)
        key(bounce, "location", f)
        key(bounce, "rotation_quaternion", f)
        key(key_l, "location", f)
        key(key_l, "rotation_quaternion", f)
    set_constant(sc)
    set_constant(sc.world.node_tree)
    set_constant(sun)
    set_constant(key_l)
    set_constant(key_l.data)
    set_constant(bounce)
    set_constant(bounce.data)


# --------------------------------------------------------------------------
# Animation: the giant (hook), the school in the cross section, the boat, rod, line and angler
# --------------------------------------------------------------------------
def animate_tuna(fish, f_lo, f_hi, state, rate_hz=1.3, amp=0.075):
    phase = 0.0
    prev = None
    for f in range(f_lo, f_hi + 1):
        p, fwd = state(f)[:2]
        fish["root"].location = p
        fish["root"].rotation_quaternion = heading_quat(fwd)
        key(fish["root"], "location", f)
        key(fish["root"], "rotation_quaternion", f)
        turn = 0.0
        if prev is not None:
            turn = max(-1.0, min(1.0, prev.cross(fwd).z * 30.0))
        prev = fwd
        phase += 2 * math.pi * rate_hz / FPS
        key_swim(fish, f, phase, amp, turn * 0.04)


def mouth_corner_world(fish, f, p, fwd):
    q = heading_quat(fwd)
    return p + q @ Vector((TUNA_L / 2 - 0.09 * TUNA_L, -0.07, -0.05))


def build_tow_line(col, M, fish, f_lo, f_hi):
    """The heavy line from the corner of her jaw, trailing back and up toward the surface (and the boat)."""
    a = empty_object("TowStart", col, size=0.05)
    b = empty_object("TowEnd", col, size=0.2)
    m = empty_object("TowSag", col, size=0.1)
    for f in range(f_lo, f_hi + 1):
        p, fwd, _b = hook_fish(f)
        s = mouth_corner_world(fish, f, p, fwd)
        a.location = s
        b.location = s + Vector((-16.0, -2.0, 10.5))
        m.location = s + Vector((-7.5, -1.2, 3.2))
        for ob in (a, b, m):
            key(ob, "location", f)
    return line_object("TowLine", col, M, a, b, radius=0.0022, sag=m, count=40)


def hook_glints(f_lo, f_hi, fish_state):
    """Stylised bioluminescent glints shed from her tail and flanks as she swims (the brief's particle trails)."""
    rng = random.Random(4)
    sp = Particles()
    for f in range(f_lo, f_hi + 1):
        p, fwd, _b = fish_state(f)
        side = fwd.cross(Vector((0, 0, 1))).normalized()
        for _ in range(7):
            along = rng.uniform(-0.5, 0.1) * TUNA_L                  # shed mostly from the rear half: trails
            src = p + fwd * along + side * rng.uniform(-0.35, 0.35) + Vector((0, 0, rng.uniform(-0.35, 0.35)))
            if along > -0.3:
                src = src + side * rng.choice((-1, 1)) * 0.32
            sp.emit(t_of(f), src, (rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), rng.uniform(0.0, 0.12)),
                    rng.uniform(0.9, 2.2), rng.uniform(0.004, 0.009))
    return sp


def tow_bubbles(f_lo, f_hi, fish, fish_state):
    rng = random.Random(6)
    sp = Particles()
    for f in range(f_lo, f_hi + 1, 2):
        p, fwd, _b = fish_state(f)
        s = mouth_corner_world(fish, f, p, fwd)
        sp.emit(t_of(f), s + Vector((rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.0)),
                (rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.4, 0.8)), rng.uniform(1.0, 2.0),
                rng.uniform(0.003, 0.007))
    return sp


def chair_frame(boat):
    return boat["chair"]["pivot"].matrix_world


def rig_fight(boat, rod, ang, sc, f_lo, f_hi, mode):
    """Per frame: the boat on the swell, the rod in the chair gimbal bowed toward the line, the angler's hands and
    lean; returns the rod tips (for the line) and mids (for pins)."""
    tips, mids, reels = {}, {}, {}
    for f in range(f_lo, f_hi + 1):
        t = t_of(f)
        if mode == "troll":
            bx = gear_boat_x(f)
            pos, q = boat_attitude(Vector((bx, 0.0, 0.0)), t)
            swivel, bend, lean = math.radians(0.0), 0.22, math.radians(-10)
        else:
            bx = fight_boat_x(f)
            stern_dip = max(0.0, -sea_height(bx, 0.0, t))
            pos, q = boat_attitude(Vector((bx, 0.0, 0.0)), t, squat=1.0)
            swivel = math.radians(8.0 * math.sin(t * 0.7))
            bend = 0.9 + 0.04 * math.sin(f * 0.4) + 0.03 * math.sin(f * 1.1)
            lean = math.radians(-24 - 6 * math.sin(f * 0.21) - 4 * stern_dip)
        key_boat(boat, f, pos, q)
        boat["chair"]["pivot"].rotation_euler = (0.0, 0.0, swivel)
        key(boat["chair"]["pivot"], "rotation_euler", f)
        sc.frame_set(f)
        bpy.context.view_layer.update()
        pm = chair_frame(boat)
        gim, foot, seat = pm @ boat["chair"]["gimbal"], pm @ boat["chair"]["footbar"], pm @ boat["chair"]["seat"]
        aft = (pm.to_3x3() @ Vector((-1.0, 0.0, 0.0))).normalized()
        if mode == "troll":
            d = (aft * 0.55 + Vector((0, 0, 0.84))).normalized()
            load = aft * 1.0 + Vector((0, 0, -0.25))
        else:
            d = (aft * 0.72 + Vector((0, 0, 0.7))).normalized()
            load = aft * 0.8 + Vector((0, 0, -0.6))
        tip, mid = place_rod(rod, f, gim, d, load, bend)
        tips[f], mids[f] = tip, mid
        reels[f] = gim + d * 0.46
        side = d.cross(Vector((0, 0, 1))).normalized()
        hands = {"L": gim + d * 0.74, "R": gim + d * 0.48 - side * 0.17 + Vector((0, 0, -0.03))}
        if mode == "fight":
            pump = 0.05 * math.sin(f * 0.21)
            hands["L"] = hands["L"] + d * pump
        seat_angler(ang, f, seat + Vector((0, 0, 0.06)), aft, lean, foot + Vector((0, 0, 0.04)), 0.32, hands=hands)
    return tips, mids, reels


def build_lines_above(col, M, rod, tips, f_lo, f_hi, mode):
    """The line from the rod tip: in the troll, out over the wake to where it cuts in, then down to the bait;
    in the fight, straight down into the water astern."""
    entry = empty_object(f"LineEntry_{mode}", col, size=0.1)
    sag = empty_object(f"LineSag_{mode}", col, size=0.1)
    under_end = empty_object(f"LineBait_{mode}", col, size=0.05)
    for f in range(f_lo, f_hi + 1):
        t = t_of(f)
        tip = tips[f]
        if mode == "troll":
            ex = tip.x - 13.0
            entry.location = Vector((ex, -0.3, surface_z(ex, -0.3, t) + 0.02))
            sag.location = tip.lerp(entry.location, 0.5) + Vector((0, 0, -1.2))
            under_end.location = bait_pos(f) + Vector((0.16, 0.0, 0.0))
        else:
            ex = tip.x - 6.5
            entry.location = Vector((ex, -0.6 + 0.4 * math.sin(t * 0.7), surface_z(ex, -0.6, t)))
            sag.location = tip.lerp(entry.location, 0.5)
            under_end.location = entry.location + Vector((-8.0, -1.0, -14.0))
        for ob in (entry, sag, under_end):
            key(ob, "location", f)
    a = line_object(f"LineAbove_{mode}", col, M, rod["ctrl"]["Tip"], entry, radius=0.0016, sag=sag)
    b = line_object(f"LineUnder_{mode}", col, M, entry, under_end, radius=0.0016)
    return [a, b], entry


def transom_spray(boat_x_fn, f_lo, f_hi):
    """Water bursting up and over the transom as she backs into the swell, and sheets thrown off the corners."""
    rng = random.Random(9)
    sp = Particles()
    for f in range(f_lo, f_hi + 1):
        t = t_of(f)
        bx = boat_x_fn(f)
        h = sea_height(bx - 0.5, 0.0, t)
        burst = 24 + int(100 * smoothstep(-0.2, 0.4, h))
        for _ in range(burst):
            y = rng.uniform(-2.3, 2.3)
            p = Vector((bx - 0.25, y, h + rng.uniform(-0.1, 0.3)))
            v = Vector((rng.uniform(-2.2, 0.6), rng.uniform(-0.8, 0.8) + 0.6 * y / 2.3, rng.uniform(1.8, 5.5)))
            sp.emit(t, p, v, rng.uniform(0.6, 1.3), rng.uniform(0.008, 0.026), 9.81)
    return sp


def gerstner_gn(ng, wx, wy, t):
    """The sea's own surface motion in a node tree: (dx, dy, h) for the point resting at world (wx, wy), so sheets
    laid on the sea (wake, the break) sit exactly on it instead of cutting through it on the slopes."""
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
    """Keep the undisplaced position as "rest" so foam and weed stay put on the water as the surface heaves."""
    st = ng.nodes.new("GeometryNodeStoreNamedAttribute")
    st.data_type = "FLOAT_VECTOR"
    st.domain = "POINT"
    st.inputs["Name"].default_value = "rest"
    ng.links.new(sp.outputs[0], st.inputs["Geometry"])
    ng.links.new(combine(ng, x, y, 0.0), st.inputs["Value"])
    ng.links.new(st.outputs[0], go.inputs["Geometry"])


def wake_object(col, M, name, length=60.0, width=2.2, spread=0.18):
    """White water in the wake: a strip astern of the transom that rides the swell (foam by alpha)."""
    bm = bmesh.new()
    nx, ny = int(length), 14
    for i in range(nx):
        for j in range(ny):
            x0, x1 = -length * i / nx, -length * (i + 1) / nx
            w0, w1 = width + spread * (-x0), width + spread * (-x1)
            y0, y1 = -1 + 2 * j / ny, -1 + 2 * (j + 1) / ny
            bm.faces.new([bm.verts.new((x0, w0 * y0, 0.0)), bm.verts.new((x1, w1 * y0, 0.0)),
                          bm.verts.new((x1, w1 * y1, 0.0)), bm.verts.new((x0, w0 * y1, 0.0))])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    ob = finish(name, bm, col, [M.wake], recalc=False)
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
                         nmath(ng, "SUBTRACT", nmath(ng, "ADD", h, 0.05), loc["Z"])), sp.inputs["Position"])
    store_rest_after(ng, sp, x, y, go)
    ob.modifiers.new(name, "NODES").node_group = ng
    ob.visible_shadow = False
    return ob


def break_object(col, M):
    """The temperature break: greener, warmer water on one side of a scum line of weed and foam (the bait runs
    along it). A wide strip on the swell, parallel to the boat's track."""
    bm = bmesh.new()
    nx, ny = 90, 24
    for i in range(nx):
        for j in range(ny):
            x0, x1 = -70.0 + 140.0 * i / nx, -70.0 + 140.0 * (i + 1) / nx
            y0, y1 = BREAK_Y - 14.0 * j / ny, BREAK_Y - 14.0 * (j + 1) / ny
            bm.faces.new([bm.verts.new((x0, y0, 0.0)), bm.verts.new((x0, y1, 0.0)), bm.verts.new((x1, y1, 0.0)),
                          bm.verts.new((x1, y0, 0.0))])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    ob = finish("TemperatureBreak", bm, col, [M.breakline], recalc=False)
    for pl in ob.data.polygons:
        pl.use_smooth = True
    ng, gi, go = new_group("BreakGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    x, y, _z = gn_xyz(ng)
    t = gn_time(ng)
    dx, dy, h = gerstner_gn(ng, x, y, t)
    sp = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, nmath(ng, "ADD", x, dx), nmath(ng, "ADD", y, dy), nmath(ng, "ADD", h, 0.04)), sp.inputs["Position"])
    store_rest_after(ng, sp, x, y, go)
    ob.modifiers.new("Break", "NODES").node_group = ng
    ob.visible_shadow = False
    return ob


def build_column(col, M):
    """The cross section's backdrop and the thermocline drawn as a sheet; a bait school above it."""
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * 160.0, 0.0, v.co.y * 90.0))
    back = finish("ColumnBackdrop", bm, col, [M.column], recalc=False)
    back.location = UW2 + Vector((20.0, 34.0, -80.0))
    back.visible_shadow = False
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=60, y_segments=30, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * 110.0, v.co.y * 40.0, 0.25 * math.sin(v.co.x * 9.0) + 0.15 * math.sin(v.co.y * 11.0)))
    sheet = finish("ThermoclineSheet", bm, col, [M.thermo], recalc=False)
    sheet.location = UW2 + Vector((20.0, 6.0, THERMO_Z))
    sheet.visible_shadow = False
    return back, sheet


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------
def build_overlays(hud, st):
    hud.scrims()
    hud.kicker("THE FISHING SERIES")
    # 1 hook (the first caption pops after the whip; nothing but the kicker on frame 1, as on the last frame)
    p1 = P(1, 1)
    hud.serif("In 1979, Ken Fraser landed a *1,496 pound* bluefin.", WHIP_IN + 2, p1 - 9, y=-0.30, size=0.048,
              lines=["In 1979, Ken Fraser landed", "a *1,496 pound* bluefin."], reveal=2.3,
              src="FRASER'S RECORD FISH, OCTOBER 26, 1979")
    hud.serif("A record untouched for *over 40 years.*", p1, CUT1 - 9, y=-0.30, size=0.048,
              lines=["A record untouched", "for *over 40 years.*"], src="IGFA ALL TACKLE WORLD RECORD")

    hud.pin(lambda f: hook_fish(f)[0] + hook_fish(f)[1] * 0.3 + Vector((0.0, 0.0, 0.3)), "128.5 in, fraser's fish",
            p1 + 6, CUT1 - 10, dot_color=GOLD, side=-0.04, y_band=(-0.15, 0.3))
    def tow_pt(f, u=0.1):
        """A point on the tow line itself (the same quadratic Bezier as build_tow_line)."""
        a = mouth_corner_world(st["tuna"], f, *hook_fish(f)[:2])
        m, b = a + Vector((-7.5, -1.2, 3.2)), a + Vector((-16.0, -2.0, 10.5))
        return a * (1 - u) ** 2 + m * (2 * u * (1 - u)) + b * u * u
    hud.pin(tow_pt, "heavy line",
            WHIP_IN + 30, p1 - 8, dot_color=GOLD, side=0.05, y_band=(-0.15, 0.3))
    # 2 gear
    hud.badge(1, "HEAVY TROLLING RIG", "130 lb class line, 14/0 reel", CUT1 + 8, CUT2 - 8)
    s1, s2 = P(2, 0), P(2, 1)
    s3 = CUT1B
    hud.serif("Average gear will *snap in seconds.*", max(CUT1 + 6, s1), s2 - 9, lines=["Average gear will", "*snap in seconds.*"])
    hud.serif("*130 lb class* line on a *14/0 reel.*", s2, s3 - 9, lines=["*130 lb class* line", "on a *14/0 reel.*"],
              src="FRASER'S TACKLE: OUTDOOR LIFE")
    hud.serif("Troll fresh *mackerel or herring.*", s3, CUT2 - 9, lines=["Troll fresh", "*mackerel or herring.*"],
              src="ANGLER TECHNIQUE")
    rod, boat = st["rod"], st["boat"]
    sc = bpy.context.scene

    def chair_pt(f):
        sc.frame_set(f)
        return chair_frame(boat) @ Vector((0.0, 0.0, 0.9))

    def reel_pt(f):
        sc.frame_set(f)
        return rod["reel"].matrix_world @ Vector((0.0, 0.0, 0.09))

    def guide_pt(f):
        sc.frame_set(f)
        return rod["ctrl"]["Mid"].matrix_world.translation.lerp(rod["ctrl"]["Tip"].matrix_world.translation, 0.6)
    span = CUT2 - CUT1
    hud.pin(chair_pt, "fighting chair", CUT1 + 10, CUT1 + int(span * 0.15), dot_color=GOLD, side=0.07)
    hud.pin(reel_pt, "14/0 reel", CUT1 + int(span * 0.195), CUT1 + int(span * 0.36), dot_color=GOLD, side=-0.08, rise=0.1)
    hud.pin(guide_pt, "roller guides", CUT1 + int(span * 0.22), CUT1 + int(span * 0.40), dot_color=GOLD, side=0.08)

    def line_pt(f):
        """The point of the line (tip, entry, bait) just ahead of the camera's ground point, so the pin rides along."""
        sc.frame_set(f)
        pts = [rod["ctrl"]["Tip"].matrix_world.translation.copy(), st["troll_entry"].matrix_world.translation.copy(),
               bait_pos(f)]
        cx = min(pts[0].x, max(pts[2].x, frames_pos(st, f).x + 0.8))
        for a, b in zip(pts, pts[1:]):
            if b.x <= cx <= a.x:
                return a.lerp(b, (a.x - cx) / max(1e-6, a.x - b.x))
        return pts[-1]
    hud.pin(line_pt, "130 lb mono", CUT1 + int(span * 0.455), CUT1B - 8, dot_color=GOLD, side=-0.07)
    # the dot sits just ahead of its nose, so it never hides the bait
    hud.pin(lambda f: bait_pos(f) + Vector((0.45, 0.0, 0.1)), "fresh mackerel", CUT1B + 6, CUT2 - 8,
            dot_color=GOLD, side=0.08, y_band=(-0.17, 0.3))
    hud.pin(lambda f: Vector((bait_pos(f).x - 1.0, BREAK_Y - 0.8, 0.15)), "temperature break", CUT1B + 20,
            CUT2 - 8, dot_color=srgb(120, 210, 220), side=0.02, rise=0.06, y_band=(-0.22, 0.3))
    # 3 the water column
    hud.badge(2, "THERMOCLINE DEPTH", "where their dives level off", CUT2 + 8, CUT3 - 8)
    t1, t2, t3 = P(3, 0), P(3, 2), DIVE_F
    hud.serif("Giant bluefin *track temperature breaks.*", CUT2 + 6, t2 - 9, lines=["Giant bluefin track", "*temperature breaks.*"],
              src="ANGLER TECHNIQUE")
    hud.serif("Their dives *follow the thermocline.*", t2, t3 - 9, lines=["Their dives follow", "*the thermocline.*"],
              src="TAGGING: WALLI ET AL. 2009, PLOS ONE")
    hud.serif("They can dive past *3,000 feet.*", t3, CUT3 - 9, lines=["They can dive", "past *3,000 feet.*"],
              src="TAGGING: BLOCK ET AL. 2001, SCIENCE")
    _ = t1

    def cam_x(f):
        return frames_pos(st, f).x
    hud.pin(lambda f: Vector((cam_x(f) + 0.5, 0.0, THERMO_Z + 5.5)), "warm surface water", CUT2 + 14, t2 - 4,
            dot_color=srgb(120, 210, 200), side=0.04, y_band=(-0.2, 0.33))
    hud.pin(lambda f: Vector((cam_x(f) + 5.5, 4.0, THERMO_Z)), "thermocline", CUT2 + 30, t3 - 4,
            dot_color=srgb(120, 220, 230), side=0.0, rise=0.11, y_band=(-0.25, 0.33))
    hud.pin(lambda f: Vector((cam_x(f) - 1.0, 6.0, THERMO_Z - 4.5)), "cold deep water", t2, t3 - 4,
            dot_color=srgb(60, 110, 200), side=0.05, rise=0.05, y_band=(-0.25, 0.33))
    hud.pin(lambda f: thermo_fish(0, f)[0], "dives past 3,000 ft", t3 + 12, CUT3 - 10, dot_color=GOLD, side=0.06,
            y_band=(-0.25, 0.33))
    # 4 the fight
    hud.badge(3, "BACK DOWN HARD", "keep line on the spool", CUT3 + 6, CUT4 - 6)
    f1, f2 = P(4, 0), P(4, 1)
    f3 = f2 + int((CUT4 - f2) * 0.55)
    hud.serif("The angler *doesn't fight alone.*", CUT3 + 4, f2 - 9, lines=["The angler", "*doesn't fight alone.*"], reveal=2.4)
    hud.serif("The captain *backs down hard.*", f2, f3 - 9, lines=["The captain", "*backs down hard.*"], src="ANGLER TECHNIQUE")
    hud.serif("Line on the spool. *Heavy drag.*", f3, CUT4 - 4, lines=["Line on the spool.", "*Heavy drag.*"],
              src="ANGLER TECHNIQUE")
    _ = f1

    def transom_pt(f):
        sc.frame_set(f)
        return boat["root"].matrix_world @ Vector((0.0, -1.2, 1.2))

    def rod_pt(f):
        sc.frame_set(f)
        return rod["ctrl"]["Mid"].matrix_world.translation.lerp(rod["ctrl"]["Tip"].matrix_world.translation, 0.4)
    hud.pin(transom_pt, "backing down", CUT3 + 16, f2 + 20, dot_color=GOLD, side=-0.14, rise=0.09)
    hud.pin(rod_pt, "rod doubled over", f2 + 30, CUT4 - 8, dot_color=GOLD, side=0.06)
    # 5 outro (the last caption is gone before the end: the last frame matches the first)
    hud.serif("The ultimate *apex predator* of the sea.", CUT4 + 8, P(5, 1) - 9, lines=["The ultimate", "*apex predator* of the sea."])
    hud.serif("And it all started *when...*", P(5, 1), FRAME_END - 9, lines=["And it all started", "*when...*"], reveal=3.0)


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
    """Bloom, a soft S curve and a teal and orange grade on the 3D only, then the overlays on top."""
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
        vals = {"Lift": (0.97, 1.0, 1.04, 1.0), "Gamma": (1.0, 1.0, 1.0, 1.0), "Gain": (1.05, 1.0, 0.94, 1.0)}
        for sock in cb.inputs:                  # (5.x: a factor and a colour socket share each name)
            if sock.bl_idname == "NodeSocketColor" and sock.name in vals:
                sock.default_value = vals[sock.name]
        ng.links.new(img, cb.inputs["Image"])
        img = cb.outputs["Image"]
    except Exception as e:                  # noqa: BLE001
        print("teal/orange grade skipped:", e)
    try:
        cv = ng.nodes.new("CompositorNodeCurveRGB")
        c = cv.mapping.curves[3]
        c.points[0].location = (0.0, 0.01)
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


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def run_checks(frames, st):
    sc = bpy.data.scenes["Main"]
    print("\n=== CAMERA REPORT ===")
    worst = {"sea surface (above)": (1e9, 0), "surface (under)": (1e9, 0), "boat": (1e9, 0), "fish": (1e9, 0),
             "angler/rod": (1e9, 0)}

    def upd(k, d, f):
        if d < worst[k][0]:
            worst[k] = (d, f)
    boat = st["boat"]
    # exact clearance: one BVH of every boat mesh (evaluated, in the boat's own frame), built once at the troll's start
    from mathutils.bvhtree import BVHTree
    sc.frame_set(CUT1)
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
    for f in range(1, FRAME_END + 1):
        p = frames[f][0]
        t = t_of(f)
        if under_at(f):
            upd("surface (under)", -p.z - 0.3, f)
        else:
            upd("sea surface (above)", p.z - sea_height(p.x, p.y, t) - 0.3, f)
        if CUT1 <= f < CUT2 or CUT3 <= f < CUT4:
            sc.frame_set(f)
            m = boat["root"].matrix_world.inverted()
            lp = m @ p
            hit = boat_bvh.find_nearest(lp)
            upd("boat", (hit[3] if hit[0] is not None else 1e9) - 0.3, f)
            d = min((st["rod"]["ctrl"][k].matrix_world.translation - p).length for k in ("Butt", "Mid", "Tip"))
            upd("angler/rod", d - 0.2, f)
        if WHIP_IN <= f < CUT1:
            fp, _fw, _b = hook_fish(f)
            upd("fish", (fp - p).length - TUNA_L / 2 - 0.3, f)
        if CUT2 <= f < CUT3:
            for i in range(3):
                upd("fish", (thermo_fish(i, f)[0] - p).length - TUNA_L / 2 - 0.3, f)
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
    print("loop: sea at t=0 vs t=55 s (should match):", round(sea_height(5.0, 7.0, 0.0) - sea_height(5.0, 7.0, LOOP_S), 6))
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
    rod = st["rod"]
    subj = {WHIP_IN + 10: ("bluefin", lambda f: hook_fish(f)[0]),
            (WHIP_IN + CUT1) // 2: ("bluefin", lambda f: hook_fish(f)[0]),
            CUT1 + 20: ("chair", lambda f: chair_frame(st["boat"]) @ Vector((0, 0, 0.9))),
            264: ("reel", lambda f: rod["reel"].matrix_world.translation.copy()),
            CUT2 - 30: ("mackerel", lambda f: bait_pos(f)),
            CUT2 + 40: ("lead bluefin", lambda f: thermo_fish(0, f)[0]),
            DIVE_F + 30: ("diving bluefin", lambda f: thermo_fish(0, f)[0]),
            CUT3 + 40: ("rod middle", lambda f: rod["ctrl"]["Mid"].matrix_world.translation.copy()),
            CUT4 - 20: ("transom", lambda f: st["boat"]["root"].matrix_world.translation.copy())}
    print("\n=== SUBJECTS ON SCREEN (x in +-0.281, y in +-0.5) ===")
    for f, (nm, fn) in sorted(subj.items()):
        sc.frame_set(f)
        s = hud.project(f, fn(f))
        print(f"frame {f}: {nm} ->", None if s is None else (round(s.x, 3), round(s.y, 3)))


def write_events(path, st):
    data = {
        "fps": FPS, "frames": FRAME_END, "loop": True,
        "cuts": [WHIP_IN, CUT1, CUT2, CUT3, CUT4], "hard_cuts": [CUT1B], "whips": [[1, WHIP_IN], [CUT4 - 4, CUT4 + 6]],
        "overlays": sorted(st["hud"].events),
        "under": [[WHIP_IN, CUT1], [CUT2, CUT3]], "dive": DIVE_F, "fight": [CUT3, CUT4], "troll": [CUT1, CUT2],
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
    c_sea, c_life, c_boat = collection("Sea"), collection("Fish"), collection("Boat")
    c_uw1, c_uw2, c_fx, c_cam = collection("OpenWater"), collection("WaterColumn"), collection("LightAndWater"), collection("Camera")
    atmos = build_atmos_group()
    M = build_materials(atmos)
    _w, w_under, w_deep = build_world(sc)
    sun = build_sun(c_fx)
    key_l = build_fill(c_fx, "UnderwaterKey", size=6.0, color=(214, 236, 240))
    frames, fstop = camera_frames()
    # above water
    sea = build_sea(c_sea, M)
    key_sea_center(sea)
    wake = wake_object(c_sea, M, "Wake")
    brk = break_object(c_sea, M)
    boat = build_boat(c_boat, M)
    mask_sea_in_cockpit(M.sea, boat["root"])
    ang = build_angler(c_boat, M)
    rod = build_rod(c_boat, M)
    tips_t, mids_t, reels_t = rig_fight(boat, rod, ang, sc, CUT1 - 2, CUT2 + 1, "troll")
    tips_f, mids_f, reels_f = rig_fight(boat, rod, ang, sc, CUT3 - 2, CUT4 + 1, "fight")
    tips = {**tips_t, **tips_f}
    lines_t, entry_t = build_lines_above(c_boat, M, rod, tips_t, CUT1 - 2, CUT2 + 1, "troll")
    lines_f, entry_f = build_lines_above(c_boat, M, rod, tips_f, CUT3 - 2, CUT4 + 1, "fight")
    for f in range(CUT1 - 2, CUT4 + 2):                                # the wake rides astern while she trolls
        bx = gear_boat_x(f) if f < CUT2 + 2 else fight_boat_x(f)
        wake.location = (bx, 0.0, 0.0)
        key(wake, "location", f)
    wash = wake_object(c_sea, M, "TransomWash", length=12.0, width=2.7, spread=0.12)   # white water piling astern
    for f in range(CUT3 - 2, CUT4 + 2):
        wash.location = (fight_boat_x(f) + 0.3, 0.0, 0.0)
        key(wash, "location", f)
    mk = bpy.data.objects.new("MackerelBait", mackerel_mesh(M))
    c_boat.objects.link(mk)
    mk.rotation_mode = "QUATERNION"
    for f in range(CUT1 - 2, CUT2 + 2):
        mk.location = bait_pos(f)
        mk.rotation_quaternion = heading_quat(Vector((1.0, 0.15 * math.cos(t_of(f) * 2.1), 0.0)),
                                              1.05 + 0.35 * math.sin(t_of(f) * 9.0))   # rolled: the silver flank flashes
        key(mk, "location", f)
        key(mk, "rotation_quaternion", f)
    mk.scale = (1.3, 1.3, 1.3)
    rng_b = random.Random(12)
    bubbles = Particles()                                  # the skipping bait leaves a short trail of white water
    for f in range(CUT1 - 30, CUT2 + 2):
        bp = bait_pos(f)
        for _ in range(10):
            q = bp + Vector((rng_b.uniform(-0.35, 0.05), rng_b.uniform(-0.06, 0.06), 0.0))
            q.z = surface_z(q.x, q.y, t_of(f)) + 0.01
            bubbles.emit(t_of(f), q, (rng_b.uniform(-0.1, 0.1), rng_b.uniform(-0.12, 0.12), 0.0), rng_b.uniform(0.35, 0.8),
                         rng_b.uniform(0.012, 0.03))
    bait_foam = bubbles.build("BaitFoam", c_boat, M.spray)
    spray = transom_spray(fight_boat_x, CUT3 - 2, CUT4 + 1).build("TransomSpray", c_boat, M.spray, streak=True)
    # open water (the hook)
    tuna = make_tuna(c_uw1, M, "Bluefin")
    animate_tuna(tuna, WHIP_IN - 2, CUT1 + 2, hook_fish, rate_hz=1.25, amp=0.07)
    tow = build_tow_line(c_uw1, M, tuna, WHIP_IN - 2, CUT1 + 2)
    glints = hook_glints(WHIP_IN - 40, CUT1 + 2, hook_fish).build("Glints", c_uw1, M.biolum)
    bub = tow_bubbles(WHIP_IN - 40, CUT1 + 2, tuna, hook_fish).build("TowBubbles", c_uw1, M.spray)
    surf1 = build_surface_under(c_uw1, M, center=(UW1.x + 10, UW1.y, 0.0), size=120.0)
    shafts1 = build_shafts(c_uw1, M, (UW1.x + 8, UW1.y + 4, 0.0), n=7, seed=5, spread=10.0, length=34.0)
    plank1 = specks("Plankton1", c_uw1, M.specks, tuple(UW1 + Vector((8.0, -3.0, HOOK_DEPTH))), (20.0, 8.0, 5.0), 7000, 0.004, 31)
    # the water column (cross section)
    back, sheet = build_column(c_uw2, M)
    school_fish = bpy.data.objects.new("SchoolFish", baitfish_mesh(M))
    c_uw2.objects.link(school_fish)
    school_fish.hide_render = True
    school_fish.hide_viewport = True
    school = build_school(c_uw2, school_fish, UW2 + Vector((6.0, 2.5, THERMO_Z + 7.5)), 420, 1.0, 3.6, 1.6, 12)
    school2 = build_school(c_uw2, school_fish, UW2 + Vector((26.0, 4.0, THERMO_Z + 6.0)), 300, 0.8, 3.0, 1.4, 13, "BaitSchool2")
    pod = []
    for i in range(3):
        fi = make_tuna(c_uw2, M, f"PodBluefin{i}")
        animate_tuna(fi, CUT2 - 2, CUT3 + 2, lambda f, i=i: thermo_fish(i, f), rate_hz=1.15 + 0.1 * i, amp=0.07)
        pod.append(fi)
    surf2 = build_surface_under(c_uw2, M, center=(UW2.x + 20, UW2.y, 0.0), size=160.0)
    shafts2 = build_shafts(c_uw2, M, (UW2.x + 20, UW2.y + 2, 0.0), n=9, seed=7, spread=26.0, length=40.0)
    plank2 = specks("Plankton2", c_uw2, M.specks, tuple(UW2 + Vector((20.0, -6.0, THERMO_Z + 2.0))), (40.0, 12.0, 14.0), 6000, 0.006, 32)
    build_camera(c_cam, frames, fstop)
    bounce = build_fill(c_fx, "UnderwaterBounce", size=8.0, color=(110, 186, 196))
    key_scene_state(sc, (w_under, w_deep), sun, key_l, bounce)
    # visibility: each place renders only in its own shots (the boat set in the troll and the fight)
    above_spans = [(1, WHIP_IN - 1), (CUT1, CUT2 - 1), (CUT3, FRAME_END)]
    boat_spans = [(CUT1, CUT2 - 1), (CUT3, CUT4 - 1)]
    key_visibility([sea], above_spans)
    own = {o.name for o in [wake, wash, brk, mk, bait_foam, spray] + lines_t + lines_f}       # keyed below for one span only (keys add up)
    key_visibility([o for o in c_boat.all_objects if o.type in ("MESH", "CURVE") and o.name not in own
                    and o.name != "RollerGuide"], boat_spans)
    key_visibility([wake, mk, bait_foam] + lines_t, [(CUT1, CUT2 - 1)])
    key_visibility([brk], [(CUT1B, CUT2 - 1)])                       # the break only in the bait shot (not under the boat)
    key_visibility(lines_f + [spray, wash], [(CUT3, CUT4 - 1)])
    key_visibility([o for o in c_uw1.all_objects if o.type in ("MESH", "CURVE")], [(WHIP_IN, CUT1 - 1)])
    key_visibility([o for o in c_uw2.all_objects if o.type in ("MESH", "CURVE") and o.name != "SchoolFish"], [(CUT2, CUT3 - 1)])
    _ = (glints, bub, surf1, shafts1, plank1, back, sheet, school, school2, surf2, shafts2, plank2, entry_f)
    hs, hcam, hcol = build_hud_scene()
    hud_materials(M)
    hud = Hud(hcam, hcol, M, frames)
    st = dict(frames=frames, hud=hud, tuna=tuna, boat=boat, rod=rod, angler=ang, troll_entry=entry_t, tips=tips,
              mids={**mids_t, **mids_f}, reels={**reels_t, **reels_f}, pod=pod)
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
            print("rendered", sc.render.filepath, flush=True)


if __name__ == "__main__":
    main()
