"""Night squid masterclass: eging (squid jigging) under a pier light at midnight. Vertical 9:16, 24 fps, 55 s
(1320 frames), Blender 5.2 EEVEE. Built to loop: the last frame is the first frame's shot, and the voiceover's last
line runs into its first.

Run with the Blender binary:
    blender --background --python night_squid_masterclass.py -- [options]
or with the bpy Python module (pip install bpy):
    python night_squid_masterclass.py [options]

Options:
    --save PATH        save the generated scene as a .blend file (fonts packed)
    --check            camera clearances (pier, squid, jig), camera acceleration and turn rate, cuts and water
                       surface crossings, overlay overlaps and off-screen boxes, subjects on screen, loop match
    --frames 1,300     render these frames as PNG stills into --outdir
    --outdir DIR       output directory for renders
    --scale PCT        resolution percentage (default 100 -> 1080x1920)
    --samples N        EEVEE render samples (default 32)
    --no-blur          switch motion blur off (quick tests)
    --events PATH      write overlay, action and voiceover timing as JSON (input for night_squid_audio.py)

Everything is procedural. Facts in the on-screen text and the voiceover are listed with their sources in SOURCES.md;
the squid, the pier, the light cone, the plankton glow and the strike are illustrative.

World (metres, +Z up, water surface z = 0). A wooden pier stands in a calm harbour at midnight; a green underwater
lamp on one piling throws a hard edged cone of light into black water. Baitfish crowd the light; bigfin reef squid
wait in the dark just outside it. Overlays live in a second scene ("HUD") with its own static camera.

Beats follow the voiceover clips (voiceover/vo1..vo5). Cuts are hard cuts, match cuts and whip pans (no wipes):
    1  hook: the pier light seen from above whips into a macro of a squid's eye, its colour cells flickering as a
       glowing jig drifts past
    2  the light trap: a match cut (the round pupil to the round lamp) to a cross section of the pier light: the
       cone, the shadow line, baitfish in the light and squid waiting in the dark
    3  the jig: a close breakdown of a cloth wrapped egi with a glowing body, weighted nose and barbless crown
    4  hop and fall: a low underwater tracking shot: three sharp jerks dart the jig upward, then it glides down,
       and a squid strikes on the fall
    5  outro: a whip pan back to the pier light from above, which is frame 1
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
LOOP_S = FRAME_END / FPS             # 55 s: the harbour swell and the music repeat with this period
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
LAMP = srgb(120, 255, 170)          # the underwater pier lamp: green white
CYAN = srgb(110, 230, 220)

# water: ink black harbour at night, faint green near the lamp (fog colour by view direction, down .. up)
UW_RAMP = [(0.0, srgb(0, 1, 2)), (0.4, srgb(1, 4, 6)), (0.6, srgb(3, 10, 12)), (0.8, srgb(6, 20, 20)),
           (1.0, srgb(10, 30, 30))]
VIS = 9.0                            # fog e-folding distance under water (m): turbid harbour water
AIR_HAZE = 260.0                     # night haze above water (m)
MOON_DIR = Vector((0.35, -0.6, 0.72)).normalized()


def t_of(f):
    """Scene time in seconds at frame f, zero at frame 1, so frame 1321 would equal frame 1 (the loop)."""
    return (f - 1) / FPS


# --------------------------------------------------------------------------
# Voiceover timing. vo_timing.json (written by night_squid_audio.py --measure) holds the measured lengths and
# sentence starts. The hook starts on frame 1 and the outro ends on the last frame, so the loop joins mid sentence:
# "...and you'll quickly realize... Stop casting blindly into the dark for squid."
# --------------------------------------------------------------------------
VO_LEN = {1: 5.72, 2: 11.15, 3: 9.48, 4: 9.09, 5: 4.28}
VO_TEXT = {
    1: "Stop casting blindly into the dark for squid. You're completely ignoring how their eyes hunt.",
    2: "Squid don't sit directly in dock lights. They lurk right along the shadow line. High power green or white "
       "lights draw in small baitfish, and squid ambush them from the pitch black edge.",
    3: "To trigger them, use a cloth wrapped eggy jig with a glowing body. Squid eyes are tuned to blue green light, "
       "so a bright silhouette stands out in dark water.",
    4: "Work the jig with three aggressive vertical rod jerks to make it dart upward, then pause. Most strikes come "
       "on the slow, horizontal drift down.",
    5: "Master the shadow line, lock in your drift, and you'll quickly realize...",
}
VO_GAPS = (40, 80, 90)               # frames of breath after clips 1, 2 and 3; the rest goes before the outro


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
WHIP_IN = 7                          # frames 1..6: the pier light from above (the loop's join), whip, cut under water
CUT1 = (vo_end(1) + VO_START[2]) // 2            # match cut: the round pupil -> the round lamp
CUT2 = (vo_end(2) + VO_START[3]) // 2            # jump cut to the jig breakdown
CUT3 = (vo_end(3) + VO_START[4]) // 2            # jump cut to the hop and fall
CUT4 = VO_START[5] - 6                           # whip pan back to the pier from above (the loop's join)
CUTS = [WHIP_IN, CUT1, CUT2, CUT3, CUT4]
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





# --------------------------------------------------------------------------
# The pier lamp: a green white light on a piling, pointing out and down. Its cone is drawn three ways that agree:
# a spot light (lights the squid, baitfish and pilings), an additive cone sheet (the beam in the water), and a mask
# used by the plankton and the seabed (lit inside the cone only). The cone's edge is the shadow line.
# --------------------------------------------------------------------------
LAMP_POS = Vector((0.0, 0.0, -0.8))
LAMP_DIR = Vector((0.62, 0.0, -0.785)).normalized()
LAMP_HALF = math.radians(23.0)
SEABED_Z = -7.5


def in_cone(p, margin=0.0):
    """How far point p sits inside the lamp cone: > 0 inside (radians from the edge), < 0 outside."""
    v = Vector(p) - LAMP_POS
    if v.length < 1e-6:
        return LAMP_HALF
    return LAMP_HALF - v.angle(LAMP_DIR) - margin


def cone_mask(nt, soft=0.04):
    """Shader: 1 inside the lamp cone, 0 outside, with a narrow soft edge; dims with distance from the lamp."""
    v = vmath(nt, "SUBTRACT", world_pos(nt), tuple(LAMP_POS))
    d = vmath(nt, "LENGTH", v)
    cosang = nmath(nt, "DIVIDE", vmath(nt, "DOT_PRODUCT", v, tuple(LAMP_DIR)), nmath(nt, "MAXIMUM", d, 1e-4))
    edge = map_range(nt, cosang, math.cos(LAMP_HALF + soft), math.cos(LAMP_HALF - soft), smooth=True)
    fall = nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", d, -1.0 / 9.0))
    return nmath(nt, "MULTIPLY", edge, fall)


def build_world(sc):
    """Under water: the black fog colour itself. Above water: a moonless navy night sky with faint stars and a
    town glow low on one horizon. Switched by the "under" value."""
    w = bpy.data.worlds.new("HarbourNight")
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
    z = view["Z"]
    sky = color_ramp(nt, map_range(nt, z, -0.05, 0.6, smooth=False),
                     [(0.0, srgb(10, 14, 22)), (0.12, srgb(16, 22, 34)), (0.4, srgb(6, 10, 20)), (1.0, srgb(2, 4, 10))])
    glow = nmath(nt, "MULTIPLY", map_range(nt, z, 0.12, 0.0), map_range(nt, view["X"], -0.2, -0.9))
    sky = mix_rgb(nt, nmath(nt, "MULTIPLY", glow, 0.6), sky, srgb(46, 34, 30))
    stars = nt.nodes.new("ShaderNodeTexVoronoi")
    stars.inputs["Scale"].default_value = 420.0
    _feed(nt, stars.inputs["Vector"], dirv)
    st = nmath(nt, "MULTIPLY", map_range(nt, stars.outputs["Distance"], 0.03, 0.0), map_range(nt, z, 0.08, 0.3))
    sky = vmath(nt, "ADD", sky, vmath(nt, "SCALE", (0.5, 0.55, 0.6), scale=nmath(nt, "MULTIPLY", st, 0.6)))
    sky = mix_rgb(nt, map_range(nt, z, 0.0, -0.03), sky, srgb(2, 3, 5))
    col = mix_rgb(nt, under.outputs[0], sky, fogc)
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(col, bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    return w, under, deep


# --------------------------------------------------------------------------
# The harbour surface: a calm, low swell whose frequencies are whole multiples of the loop rate (2 pi / 55 s), so
# the water at frame 1321 is the water at frame 1.
# --------------------------------------------------------------------------
G = 9.81
SWELLS = []                                      # (amplitude, kx, ky, omega, phase, steepness)
for _n, _amp, _deg, _q in ((16, 0.05, 12.0, 0.35), (22, 0.035, 47.0, 0.3), (30, 0.022, -28.0, 0.3),
                           (41, 0.013, 71.0, 0.25), (57, 0.008, -52.0, 0.2), (79, 0.005, 18.0, 0.2)):
    _w = 2 * math.pi * _n / LOOP_S
    _k = _w * _w / G
    _a = math.radians(_deg)
    SWELLS.append((_amp, _k * math.cos(_a), _k * math.sin(_a), _w, 1.3 * _n % (2 * math.pi), _q))
SEA_SIZE = 600.0


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




# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------
def build_materials(atmos):
    M = Materials()
    M.atmos = atmos

    def mat(name, build, blended=False):
        return new_material(name, build, atmos, blended=blended)

    # --- water ------------------------------------------------------------------------------------------------
    def sea(nt):
        """Harbour water at night from above: nearly black, mirror like, catching the deck lamp and the sky; a faint
        green glow where the underwater lamp shines up through it."""
        wp = separate(nt, world_pos(nt))          # (the stored "rest" attribute came out corrupt away from the centre)
        p = combine(nt, wp["X"], wp["Y"], 0.0)
        rip = looped_noise(nt, vmath(nt, "MULTIPLY", p, (1.0, 1.0, 1.0)), (0.3, 0.15, 0.0), 2.2, 5.0, 0.6)
        rip2 = looped_noise(nt, vmath(nt, "MULTIPLY", p, (1.0, 1.0, 1.0)), (-0.2, 0.3, 0.0), 7.0, 3.0, 0.55)
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        _feed(nt, b.inputs["Base Color"], (*srgb(2, 5, 6), 1.0))
        b.inputs["Roughness"].default_value = 0.06
        b.inputs["IOR"].default_value = 1.333
        h = nmath(nt, "ADD", rip, nmath(nt, "MULTIPLY", rip2, 0.35))
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = 0.25
        bp.inputs["Distance"].default_value = 0.04
        nt.links.new(h, bp.inputs["Height"])
        nt.links.new(bp.outputs[0], b.inputs["Normal"])
        # the lamp's glow from below the surface (the lamp is opaque from above, so it is painted on)
        d = vmath(nt, "LENGTH", vmath(nt, "MULTIPLY", vmath(nt, "SUBTRACT", p, (LAMP_POS.x + 1.4, LAMP_POS.y, 0.0)),
                                      (0.8, 1.0, 0.0)))
        g = nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", d, d), -0.3))
        g = nmath(nt, "MULTIPLY", g, map_range(nt, rip2, 0.3, 0.7, 0.8, 1.15))
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*LAMP, 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", g, 1.8), em.inputs["Strength"])
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(b.outputs[0], add.inputs[0])
        nt.links.new(em.outputs[0], add.inputs[1])
        return add.outputs[0]
    M.sea = mat("Sea", sea)

    def surface_under(nt):
        """The surface from below: black, a dim green sheen near the lamp, rippling."""
        p = tex_coord(nt, "Object")
        rip = looped_noise(nt, p, (0.2, 0.1, 0.0), 0.5, 4.0, 0.6)
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(mix_rgb(nt, cone_mask(nt, 0.3), srgb(1, 3, 4), srgb(30, 90, 70)), em.inputs["Color"])
        nt.links.new(map_range(nt, rip, 0.3, 0.7, 0.4, 1.0), em.inputs["Strength"])
        return em.outputs[0]
    M.surface_under = new_material("SurfaceFromBelow", surface_under)

    def beam(nt):
        """The lamp's beam in the water: an additive cone sheet, even across (a hard edge: the shadow line), dimming
        with distance from the lamp, with drifting turbidity in it."""
        p = world_pos(nt)
        turb = map_range(nt, looped_noise(nt, vmath(nt, "MULTIPLY", p, (0.6, 0.6, 0.6)), (0.05, 0.02, -0.04), 1.4, 4.0, 0.6),
                         0.3, 0.7, 0.55, 1.0)
        d = vmath(nt, "LENGTH", vmath(nt, "SUBTRACT", p, tuple(LAMP_POS)))
        fall = nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", d, -1.0 / 6.5))
        k = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", fall, turb), 0.16)
        k = nmath(nt, "MULTIPLY", k, nmath(nt, "MULTIPLY_ADD", scene_attr(nt, "beam"), 1.0, 0.0))
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*LAMP, 1.0)
        nt.links.new(k, em.inputs["Strength"])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(tr.outputs[0], add.inputs[1])
        return add.outputs[0]
    M.beam = new_material("LampBeam", beam, blended=True)

    def plankton(nt):
        """Suspended particles: invisible in the dark, glittering inside the lamp cone."""
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(170, 255, 210), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", cone_mask(nt, 0.02), attr(nt, "rand", "Fac")), 3.0),
                     em.inputs["Strength"])
        return em.outputs[0]
    M.specks = new_material("Plankton", plankton)

    def sparks(nt):
        """Stylised glowing plankton sparks around the squid (an effect, not the squid's own light)."""
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*CYAN, 1.0)
        em.inputs["Strength"].default_value = 5.0
        return em.outputs[0]
    M.sparks = new_material("Sparks", sparks)

    # --- the squid ---------------------------------------------------------------------------------------------
    def squid_skin(nt, fin=False):
        """Translucent mantle, cream below, flushed above, freckled with chromatophores (two sizes, brown red and
        yellow) that open and close in waves; a faint blue green iridescent sheen at grazing angles."""
        p = tex_coord(nt, "Object")
        z, x = attr(nt, "zrel", "Fac"), attr(nt, "xrel", "Fac")
        base = color_ramp(nt, z, [(0.0, srgb(232, 220, 212)), (0.45, srgb(226, 206, 196)), (0.7, srgb(206, 168, 150)),
                                  (1.0, srgb(188, 140, 122))])
        flash = scene_attr(nt, "flash")
        wave = looped_noise(nt, vmath(nt, "MULTIPLY", p, (9.0, 14.0, 14.0)), (-1.2, 0.0, 0.0), 1.0, 2.0, 0.5)
        band = map_range(nt, nmath(nt, "SINE", nmath(nt, "ADD", nmath(nt, "MULTIPLY", separate(nt, p)["X"], 60.0),
                                                     nmath(nt, "MULTIPLY", loop_time(nt), 2 * math.pi * 1.2 * 55 / LOOP_S))),
                         -0.2, 1.0)
        opening = nmath(nt, "ADD", nmath(nt, "MULTIPLY_ADD", z, 0.35, 0.22),
                        nmath(nt, "MULTIPLY", map_range(nt, wave, 0.35, 0.7), 0.3))
        opening = nmath(nt, "ADD", opening, nmath(nt, "MULTIPLY", flash, nmath(nt, "MULTIPLY", band, 0.55)))
        col = base
        for scale, color, k in ((340.0, srgb(118, 32, 22), 1.0), (190.0, srgb(196, 118, 40), 0.85), (90.0, srgb(90, 24, 20), 0.7)):
            jit = map_range(nt, noise(nt, p, scale * 0.35, 2.0, 0.5), 0.3, 0.7, 0.55, 1.35)   # cells open unevenly
            v = nmath(nt, "MULTIPLY", voronoi(nt, p, scale, out="Distance"), jit)
            hi = nmath(nt, "MULTIPLY", opening, 0.42 * k)                      # the cell's open radius
            spot = nmath(nt, "DIVIDE", nmath(nt, "SUBTRACT", hi, v),
                         nmath(nt, "ADD", nmath(nt, "MULTIPLY", opening, 0.12 * k), 1e-4), clamp=True)
            col = mix_rgb(nt, nmath(nt, "MULTIPLY", spot, 0.9), col, color)
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.4
        sheen = nmath(nt, "MULTIPLY", nmath(nt, "POWER", lw.outputs["Facing"], 3.0), 0.12)
        alpha = 0.82 if fin else 1.0
        return psdf(nt, col, rough=0.3, coat=0.7, coat_rough=0.05, sss=0.55, sss_scale=0.012, alpha=alpha,
                    emission=srgb(80, 220, 210), emission_strength=sheen)
    M.squid = mat("SquidSkin", squid_skin)
    M.squid_fin = mat("SquidFin", lambda nt: squid_skin(nt, fin=True))

    def squid_eye(nt):
        """Squid eye: a large round black pupil, a silvery gold iris, a clear glossy surface."""
        r = attr(nt, "eyer", "Fac")
        col = color_ramp(nt, r, [(0.0, srgb(2, 2, 3)), (0.56, srgb(2, 2, 3)), (0.585, srgb(150, 118, 60)),
                                 (0.63, srgb(206, 182, 120)), (0.72, srgb(214, 216, 212)), (0.86, srgb(190, 176, 168)),
                                 (1.0, srgb(206, 176, 160))])
        return psdf(nt, col, rough=0.03, metal=0.2, coat=1.0, coat_rough=0.02, spec=0.8)
    M.squid_eye = mat("SquidEye", squid_eye)

    def small_eye(nt):
        r = attr(nt, "eyer", "Fac")
        col = color_ramp(nt, r, [(0.0, srgb(2, 2, 3)), (0.66, srgb(2, 2, 3)), (0.69, srgb(200, 170, 96)),
                                 (0.72, srgb(226, 228, 224)), (0.87, srgb(214, 218, 216)), (1.0, srgb(120, 126, 130))])
        return psdf(nt, col, rough=0.05, coat=1.0, coat_rough=0.02, spec=0.8)
    M.small_eye = mat("SmallEye", small_eye)

    # --- the egi -------------------------------------------------------------------------------------------------
    def egi_cloth(nt):
        """Woven cloth over a glowing body: orange and pink marbled cloth with a fine weave; the luminous body glows
        green through the gaps in the weave."""
        p = tex_coord(nt, "Object")
        o = separate(nt, p)
        k = 2600.0                                                          # a fine twill, about 1.2 mm repeat
        w1 = nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY", nmath(nt, "ADD", o["X"], o["Z"]), k)))
        w2 = nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY", nmath(nt, "SUBTRACT", o["X"], o["Z"]), k * 0.5)))
        weave = map_range(nt, nmath(nt, "MULTIPLY", w1, w2), 0.1, 0.8)
        marble = noise(nt, vmath(nt, "MULTIPLY", p, (60.0, 160.0, 160.0)), 1.0, 4.0, 0.6)
        z = attr(nt, "zrel", "Fac")
        base = color_ramp(nt, marble, [(0.3, srgb(236, 74, 26)), (0.55, srgb(240, 104, 120)), (0.75, srgb(186, 34, 40))])
        base = mix_rgb(nt, map_range(nt, z, 0.78, 0.95), base, srgb(120, 30, 30))            # darker back
        base = mix_rgb(nt, map_range(nt, z, 0.3, 0.1), base, srgb(246, 160, 120))           # paler belly
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", weave, 0.25), base, vmath(nt, "SCALE", base, scale=0.65))
        glow = nmath(nt, "MULTIPLY", map_range(nt, weave, 0.6, 0.0, 0.25, 1.0), nmath(nt, "MULTIPLY", scene_attr(nt, "glow"), 0.55))
        return psdf(nt, col, rough=0.75, normal=bump(nt, weave, 0.35, 0.0002), emission=srgb(110, 255, 180),
                    emission_strength=glow)
    M.egi_cloth = mat("EgiCloth", egi_cloth)
    M.egi_lead = mat("EgiLead", lambda nt: psdf(nt, srgb(176, 180, 186), rough=0.3, metal=0.9, coat=0.4))
    M.egi_fin = mat("EgiFin", lambda nt: psdf(nt, srgb(250, 130, 60), rough=0.5, alpha=0.85, sss=0.3))
    M.chrome = mat("Chrome", lambda nt: psdf(nt, srgb(214, 218, 224), rough=0.18, metal=1.0, emission=srgb(120, 130, 140),
                                             emission_strength=0.08))
    M.line = mat("Line", lambda nt: psdf(nt, srgb(230, 236, 200), rough=0.3, emission=srgb(200, 230, 200),
                                         emission_strength=0.15))

    def trail(nt):
        """The jig's path drawn as a glowing guide line (a graphic, not water)."""
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*GOLD, 1.0)
        em.inputs["Strength"].default_value = 2.5
        return em.outputs[0]
    M.trail = new_material("JigTrail", trail)

    # --- baitfish ------------------------------------------------------------------------------------------------
    def baitfish(nt):
        z = attr(nt, "zrel", "Fac")
        col = color_ramp(nt, z, [(0.0, srgb(214, 218, 222)), (0.55, srgb(170, 184, 190)), (0.7, srgb(50, 80, 90)),
                                 (1.0, srgb(24, 44, 56))])
        return psdf(nt, col, rough=0.2, metal=0.6, coat=0.4)
    M.baitfish = mat("Baitfish", baitfish)
    M.small_fin = mat("SmallFin", lambda nt: psdf(nt, srgb(150, 160, 160), rough=0.4, alpha=0.85))

    # --- the pier ------------------------------------------------------------------------------------------------
    def piling(nt):
        """Old timber piling: dark wet wood above, furred with weed and barnacles below the water line."""
        p = tex_coord(nt, "Object")
        wz = separate(nt, world_pos(nt))["Z"]
        grain = noise(nt, vmath(nt, "MULTIPLY", p, (8.0, 8.0, 0.6)), 3.0, 6.0, 0.6)
        wood = color_ramp(nt, grain, [(0.3, srgb(46, 36, 28)), (0.7, srgb(80, 64, 48))])
        growth = noise(nt, vmath(nt, "MULTIPLY", p, (30.0, 30.0, 30.0)), 1.0, 5.0, 0.7)
        bio = color_ramp(nt, growth, [(0.35, srgb(40, 52, 36)), (0.55, srgb(70, 82, 60)), (0.7, srgb(150, 150, 130))])
        col = mix_rgb(nt, map_range(nt, wz, 0.2, -0.2), wood, bio)
        return psdf(nt, col, rough=0.85, normal=bump(nt, growth, 0.6, 0.004))
    M.piling = mat("Piling", piling)

    def deck(nt):
        """Weathered deck planks."""
        p = tex_coord(nt, "Object")
        o = separate(nt, p)
        plank = nmath(nt, "FRACT", nmath(nt, "MULTIPLY", o["Y"], 1.0 / 0.16))
        seam = map_range(nt, nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", plank, 0.5)), 0.47, 0.5)
        grain = noise(nt, vmath(nt, "MULTIPLY", p, (0.8, 18.0, 1.0)), 3.0, 5.0, 0.6)
        col = color_ramp(nt, grain, [(0.3, srgb(70, 60, 50)), (0.7, srgb(110, 96, 80))])
        col = mix_rgb(nt, seam, col, srgb(14, 12, 10))
        return psdf(nt, col, rough=0.8, normal=bump(nt, nmath(nt, "SUBTRACT", grain, seam), 0.4, 0.003))
    M.deck = mat("Deck", deck)
    M.metal = mat("LampMetal", lambda nt: psdf(nt, srgb(60, 64, 66), rough=0.4, metal=0.8))
    M.lens = new_material("LampLens", lambda nt: emission_shader(nt, LAMP, 40.0))
    M.deck_lamp = new_material("DeckLampGlass", lambda nt: emission_shader(nt, srgb(255, 214, 160), 30.0))
    M.rope = mat("Rope", lambda nt: psdf(nt, srgb(150, 130, 100), rough=0.9))

    def seabed(nt):
        """Silt and shell grit: black in the dark, pale green inside the lamp's cone."""
        p = tex_coord(nt, "Object")
        n = noise(nt, vmath(nt, "MULTIPLY", p, (2.0, 2.0, 2.0)), 3.0, 6.0, 0.6)
        col = color_ramp(nt, n, [(0.3, srgb(40, 38, 32)), (0.7, srgb(70, 66, 56))])
        return psdf(nt, col, rough=0.95, normal=bump(nt, n, 0.4, 0.02))
    M.seabed = mat("Seabed", seabed)
    return M


def emission_shader(nt, color, strength):
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    em.inputs["Strength"].default_value = strength
    return em.outputs[0]



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
# The bigfin reef squid (Sepioteuthis lessoniana): a long torpedo mantle with broad oval fins running almost its
# whole length (the genus' mark), a short head with very large eyes, eight arms and two longer feeding tentacles
# with clubs, kept tucked between the arms until the strike. Colour changes with chromatophores (skin cells it
# opens and closes). Modelled with a 0.28 m mantle. Local +X points from the mantle tip towards the arms.
# --------------------------------------------------------------------------
SQ_ML = 0.28                         # mantle length (m)
SQ_HEAD = 0.055                      # collar to the arm crown
SQ_EYE_X, SQ_EYE_Y, SQ_EYE_R = 0.028, 0.031, 0.0165


def squid_profile():
    """Mantle half width along x (x from -SQ_ML at the tip to 0 at the collar), then the head."""
    m = pchip([(0.0, 0.0), (0.05, 0.016), (0.22, 0.034), (0.5, 0.045), (0.78, 0.048), (1.0, 0.044)])
    h = pchip([(0.0, 0.043), (0.18, 0.034), (0.5, 0.04), (0.78, 0.036), (1.0, 0.027)])

    def r(x):
        if x <= 0.0:
            return m((x + SQ_ML) / SQ_ML)
        return h(min(1.0, x / SQ_HEAD))
    return r


def squid_mesh(M):
    r = squid_profile()
    bm = bmesh.new()
    zl, xl = bm.verts.layers.float.new("zrel"), bm.verts.layers.float.new("xrel")
    er = bm.verts.layers.float.new("eyer")
    fl = bm.verts.layers.float.new("finw")
    na, rings = 40, []
    xs = [-SQ_ML + SQ_ML * (i / 60) ** 0.85 for i in range(61)] + [SQ_HEAD * i / 14 for i in range(1, 15)]
    for x in xs:
        rr = max(r(x), 1e-4)
        ring = []
        for k in range(na):
            a = 2 * math.pi * k / na
            c, s = math.cos(a), math.sin(a)
            v = bm.verts.new((x, rr * c, 0.84 * rr * s))
            v[zl] = 0.5 + 0.5 * s
            v[xl] = (x + SQ_ML) / (SQ_ML + SQ_HEAD)
            ring.append(v)
        rings.append(ring)
    for ra, rb in zip(rings, rings[1:]):
        for k in range(na):
            bm.faces.new((ra[k], ra[(k + 1) % na], rb[(k + 1) % na], rb[k]))
    tip = bm.verts.new((-SQ_ML - 0.004, 0.0, 0.0))
    tip[zl], tip[xl] = 0.5, 0.0
    for k in range(na):
        bm.faces.new((rings[0][(k + 1) % na], rings[0][k], tip))
    cap = bm.faces.new(list(reversed(rings[-1])))          # the arm crown closes the head
    for f in bm.faces:
        f.smooth = True
    cap.smooth = True
    # eyes: big and bulging on the sides of the head
    for s in (1, -1):
        c = Vector((SQ_EYE_X, s * SQ_EYE_Y, 0.004))
        faces = ellipsoid(bm, c, (SQ_EYE_R, SQ_EYE_R * 0.62, SQ_EYE_R * 0.95), subdiv=5, mat=1)
        axis = Vector((0.15, s * 1.0, 0.08)).normalized()
        for v in (v for f in faces for v in f.verts):
            v[er] = math.acos(max(-1.0, min(1.0, (v.co - c).normalized().dot(axis)))) / (math.pi / 2)
            v[zl], v[xl] = 0.6, 0.9
    # fins: broad ovals from near the tip almost to the collar, thin, tapering to a soft edge
    x0, x1 = -SQ_ML + 0.012, -0.012
    for s in (1, -1):
        nx, ny = 26, 8
        grid = []
        for i in range(nx + 1):
            x = x0 + (x1 - x0) * i / nx
            u = i / nx
            w = 0.074 * math.sin(math.pi * u) ** 0.75 * (1.0 - 0.18 * u)        # broadest just behind the middle
            base = r(x) * 0.9
            row = []
            for j in range(ny + 1):
                t = j / ny
                y = s * (base + w * t)
                th = 0.0032 * (1.0 - t) ** 0.6 + 0.0004
                top = bm.verts.new((x, y, th))
                bot = bm.verts.new((x, y, -th))
                for v in (top, bot):
                    v[fl] = t
                    v[zl] = 0.75 if v is top else 0.25
                    v[xl] = (x + SQ_ML) / (SQ_ML + SQ_HEAD)
                row.append((top, bot))
            grid.append(row)
        for i in range(nx):
            for j in range(ny):
                a, b, c_, d = grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]
                f1 = bm.faces.new((a[0], b[0], c_[0], d[0]) if s > 0 else (d[0], c_[0], b[0], a[0]))
                f2 = bm.faces.new((d[1], c_[1], b[1], a[1]) if s > 0 else (a[1], b[1], c_[1], d[1]))
                for f in (f1, f2):
                    f.material_index, f.smooth = 2, True
        for i in range(nx):                                    # outer rim
            a, b = grid[i][ny], grid[i + 1][ny]
            f = bm.faces.new((a[0], a[1], b[1], b[0]) if s > 0 else (b[0], b[1], a[1], a[0]))
            f.material_index, f.smooth = 2, True
    return fish_mesh("SquidBody", bm, [M.squid, M.squid_eye, M.squid_fin])


def squid_arm(bm, base, direction, length, r0, curl, club=False, mat=0):
    """One arm: a tapering tube from the crown, curling a little; a tentacle ends in a flattened club."""
    d = Vector(direction).normalized()
    side = d.cross(Vector((0, 0, 1)))
    if side.length < 1e-3:
        side = Vector((0, 1, 0))
    side.normalize()
    up = side.cross(d).normalized()
    pts, n = [], 18
    for i in range(n + 1):
        t = i / n
        bend = curl * t * t
        pts.append(Vector(base) + d * (length * t) + up * (bend * length * 0.25) + side * (0.004 * math.sin(3 * t)))

    def rad(t):
        rr = r0 * (1.0 - 0.85 * t) + 0.0006
        if club and t > 0.78:
            rr *= 1.0 + 1.6 * math.sin(math.pi * (t - 0.78) / 0.22)
        return rr
    return tube(bm, pts, rad, segs=8, cap=True, mat=mat)


def squid_arms_mesh(M):
    """The eight arms: thick at the crown, tapering to fine tips, gathered forward into a loose cone (dorsal pair
    shortest, ventral pair longest), each curling a little."""
    bm = bmesh.new()
    lengths = {0: 0.098, 1: 0.115, 2: 0.128, 3: 0.138}              # by pair, dorsal (0) to ventral (3)
    for k in range(8):
        a = 2 * math.pi * (k + 0.5) / 8
        ring = Vector((0.0, math.cos(a), 0.86 * math.sin(a)))
        pair = min(3, int(round((1.0 - math.sin(a)) * 1.5)))
        base = Vector((SQ_HEAD - 0.006, 0.0, 0.0)) + ring * 0.017
        length = lengths[pair] * (0.96 + 0.08 * ((k * 37) % 5) / 4)
        tip = Vector((SQ_HEAD + length, 0.0, -0.006)) + ring * 0.012
        mid = base.lerp(tip, 0.5) + ring * 0.006
        pts = []
        for i in range(19):
            t = i / 18
            q = (1 - t) ** 2 * base + 2 * t * (1 - t) * mid + t * t * tip                 # a soft bow
            curl = Vector((0.0, 0.0, -0.01 * t ** 3)) + ring * (-0.004 * math.sin(math.pi * t))
            pts.append(q + curl)
        tube(bm, pts, lambda t: 0.0092 * (1.0 - t) ** 1.15 + 0.0006, segs=10, cap=True, mat=0)
    me = fish_mesh("SquidArms", bm, [M.squid])
    for an, val in (("zrel", 0.62), ("xrel", 1.0)):
        at = me.attributes.new(an, "FLOAT", "POINT")
        at.data.foreach_set("value", [val] * len(me.vertices))
    return me


def squid_tentacles_mesh(M):
    """The two feeding tentacles, built at full strike length (0.34 m); a shape key "Tucked" pulls them back."""
    bm = bmesh.new()
    for s in (1, -1):
        base = Vector((SQ_HEAD - 0.006, s * 0.006, -0.012))
        squid_arm(bm, base, Vector((1.0, s * 0.04, -0.03)), 0.34, 0.0036, curl=0.05, club=True, mat=0)
    me = fish_mesh("SquidTentacles", bm, [M.squid])
    for an, val in (("zrel", 0.45), ("xrel", 1.0)):
        at = me.attributes.new(an, "FLOAT", "POINT")
        at.data.foreach_set("value", [val] * len(me.vertices))
    return me


def make_squid(col, M, name="Squid"):
    root = empty_object(name + "Rig", col, size=0.1)
    root.rotation_mode = "QUATERNION"
    body = bpy.data.objects.new(name, squid_mesh(M))
    col.objects.link(body)
    body.parent = root
    arms = bpy.data.objects.new(name + "Arms", squid_arms_mesh(M))
    col.objects.link(arms)
    arms.parent = root
    tent = bpy.data.objects.new(name + "Tentacles", squid_tentacles_mesh(M))
    col.objects.link(tent)
    tent.parent = root
    tent.shape_key_add(name="Basis")
    sk = tent.shape_key_add(name="Tucked")
    x0 = SQ_HEAD - 0.006
    for i, v in enumerate(tent.data.vertices):
        co = v.co
        sk.data[i].co = Vector((x0 + (co.x - x0) * 0.33, co.y * 0.6, co.z * 0.5 - 0.002))
    sk.value = 1.0
    beat = empty_object(name + "Beat", col, size=0.02)          # x: fin wave phase (rad), y: fin amplitude (m)
    ng = squid_fin_group(beat, name + "Fins")
    body.modifiers.new("Fins", "NODES").node_group = ng
    return dict(root=root, body=body, arms=arms, tent=tent, beat=beat, tucked=sk)


def squid_fin_group(ctrl, name):
    """Fin undulation: waves running back along both fins, growing towards the fin edge (attribute finw)."""
    ng, gi, go = new_group(name, "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = ctrl
    c = separate(ng, oi.outputs["Location"])
    x, _y, _z = gn_xyz(ng)
    fw = ng.nodes.new("GeometryNodeInputNamedAttribute")
    fw.data_type = "FLOAT"
    fw.inputs["Name"].default_value = "finw"
    arg = nmath(ng, "ADD", c["X"], nmath(ng, "MULTIPLY", x, 2 * math.pi * 2.2 / SQ_ML))
    dz = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", nmath(ng, "SINE", arg), c["Y"]),
               nmath(ng, "POWER", fw.outputs[0], 1.4))
    sp = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, 0.0, 0.0, dz), sp.inputs["Offset"])
    ng.links.new(sp.outputs[0], go.inputs["Geometry"])
    return ng


def key_squid(sq, f, pos, fwd, roll=0.0, fin_phase=0.0, fin_amp=0.012, tucked=1.0):
    sq["root"].location = pos
    sq["root"].rotation_quaternion = heading_quat(fwd, roll)
    key(sq["root"], "location", f)
    key(sq["root"], "rotation_quaternion", f)
    sq["beat"].location = (fin_phase, fin_amp, 0.0)
    key(sq["beat"], "location", f)
    sq["tucked"].value = tucked
    sq["tucked"].keyframe_insert("value", frame=f)


# --------------------------------------------------------------------------
# The egi (Japanese squid jig): a shrimp shaped body wrapped in cloth, big eyes, a lead sinker under the chin
# that sets its sink angle, two feather fins, and a crown (kasa) of barbless needle hooks at the tail. Size 3.5
# class, about 10.5 cm long. This one has a glowing body under the cloth. Local +X points to the nose.
# --------------------------------------------------------------------------
EGI_L = 0.105


def egi_profile():
    top = pchip([(0.0, 0.0015), (0.07, 0.0085), (0.25, 0.0135), (0.5, 0.0118), (0.8, 0.0062), (1.0, 0.0028)])
    bot = pchip([(0.0, -0.0015), (0.07, -0.0075), (0.25, -0.0115), (0.5, -0.0098), (0.8, -0.005), (1.0, -0.0024)])
    wid = pchip([(0.0, 0.0012), (0.07, 0.0062), (0.25, 0.0088), (0.55, 0.0074), (0.85, 0.0036), (1.0, 0.0022)])
    return top, bot, wid                      # u = 0 at the nose, 1 at the tail


def egi_mesh(M):
    top, bot, wid = egi_profile()
    bm = bmesh.new()
    loft_profile(bm, EGI_L, top, bot, wid, nr=60, na=36, power=2.2, mat=0)
    er = bm.verts.layers.float.new("eyer")
    for s in (1, -1):                                          # big 3D eyes just behind the nose
        c = Vector((EGI_L / 2 - 0.016, s * 0.0062, 0.0035))
        faces = ellipsoid(bm, c, (0.0046, 0.0032, 0.0046), subdiv=4, mat=1)
        axis = Vector((0.2, s * 1.0, 0.1)).normalized()
        for v in (v for f in faces for v in f.verts):
            v[er] = math.acos(max(-1.0, min(1.0, (v.co - c).normalized().dot(axis)))) / (math.pi / 2)
    # lead sinker: a curved, weighted plate under the chin (it sets the slow, nose down sink angle)
    sink = []
    for i in range(9):
        t = i / 8
        x = EGI_L / 2 - 0.004 - 0.032 * t
        z = bot(0.04 + 0.3 * t) - 0.0018 - 0.0035 * math.sin(math.pi * t)
        sink.append((x, z))
    outline = sink + [(x, z + 0.0035 + 0.002 * math.sin(math.pi * (1 - k / 8))) for k, (x, z) in enumerate(reversed(sink))]
    plate(bm, outline, 0.0072, mat=2, axis="Y", smooth=True)
    # feather fins: two small soft fins at the shoulders, swept back
    for s in (1, -1):
        soft_fin(bm, [(0.0, 0.0), (-0.008, s * 0.009), (-0.02, s * 0.011), (-0.028, s * 0.006), (-0.016, 0.0)], 0.0007, 3,
                 plane="XY", offset=(0.008, s * 0.0082, 0.003), samples=6)
    # tail crown (kasa): a short stem and two rings of straight, barbless needles flaring back like an umbrella
    xt = -EGI_L / 2 - 0.001
    tube(bm, [(xt + 0.004, 0, 0), (xt - 0.006, 0, 0)], 0.0011, segs=8, mat=4)
    for ring, (dx, n, spread, ln) in enumerate(((-0.004, 12, 0.55, 0.013), (-0.0065, 12, 0.85, 0.0125))):
        for k in range(n):
            a = 2 * math.pi * (k + 0.5 * ring) / n
            d = Vector((-math.cos(spread), math.sin(spread) * math.cos(a), math.sin(spread) * math.sin(a)))
            b = Vector((xt + dx, 0.0011 * math.cos(a), 0.0011 * math.sin(a)))
            tube(bm, [b, b + d * ln], lambda t: 0.0004 * (1.0 - 0.6 * t), segs=5, mat=4)
    tube(bm, [(EGI_L / 2 + 0.001, 0, 0.001), (EGI_L / 2 + 0.006, 0, 0.002)], 0.0005, segs=6, mat=4)    # line tie
    return fish_mesh("Egi", bm, [M.egi_cloth, M.small_eye, M.egi_lead, M.egi_fin, M.chrome])


def make_egi(col, M, name="Egi"):
    ob = bpy.data.objects.new(name, egi_mesh(M))
    col.objects.link(ob)
    ob.rotation_mode = "QUATERNION"
    return ob



# --------------------------------------------------------------------------
# The pier: two rows of timber pilings (x = -0.2 and x = -2.8, every 3 m along y), cross beams and a plank deck at
# z = 2.2; a deck lamp on a post; the green lamp clamped to the outer piling at y = 0, below the water.
# --------------------------------------------------------------------------
PILE_XS = (-0.2, -2.8)
PILE_YS = tuple(-12.0 + 3.0 * i for i in range(9))
PILE_R = 0.16
DECK_Z = 2.2
DECK_LAMP = Vector((-0.45, 2.2, DECK_Z + 2.6))


def build_pier(col, M):
    bm = bmesh.new()
    for x in PILE_XS:
        for y in PILE_YS:
            bm_frustum(bm, (x, y, SEABED_Z - 0.3), (x, y, DECK_Z - 0.1), PILE_R * 1.05, PILE_R, segs=14)
    piles = finish("Pilings", bm, col, [M.piling])
    for p in piles.data.polygons:
        p.use_smooth = True
    bm = bmesh.new()
    for y in PILE_YS:                                          # cross beams under the deck
        bm_box(bm, (-1.5, y, DECK_Z - 0.32), (3.3, 0.22, 0.26))
    for x in PILE_XS:                                          # stringers
        bm_box(bm, (x, 0.0, DECK_Z - 0.16), (0.2, 24.6, 0.2))
    bm_box(bm, (-0.05, 0.0, DECK_Z + 0.45), (0.12, 24.4, 0.1))  # hand rail on the water side
    for y in PILE_YS:
        bm_box(bm, (-0.05, y, DECK_Z + 0.22), (0.1, 0.1, 0.46))
    beams = finish("PierBeams", bm, col, [M.piling])
    bm = bmesh.new()
    bm_box(bm, (-1.5, 0.0, DECK_Z), (3.4, 24.8, 0.08))
    deck = finish("Deck", bm, col, [M.deck])
    bm = bmesh.new()                                           # deck lamp: post, arm, shade
    px = DECK_LAMP.x - 0.35                                    # the post stands back; an arm carries the shade
    bm_frustum(bm, (px, DECK_LAMP.y, DECK_Z), (px, DECK_LAMP.y, DECK_LAMP.z + 0.2), 0.05, 0.04, segs=10)
    bm_box(bm, ((px + DECK_LAMP.x) / 2, DECK_LAMP.y, DECK_LAMP.z + 0.17), (0.4, 0.04, 0.04))
    bm_frustum(bm, (DECK_LAMP.x, DECK_LAMP.y, DECK_LAMP.z + 0.12), (DECK_LAMP.x, DECK_LAMP.y, DECK_LAMP.z - 0.12), 0.06, 0.2, segs=14)
    for y in (-9.0, -3.0, 6.0):                                # bollards
        bm_frustum(bm, (-0.45, y, DECK_Z), (-0.45, y, DECK_Z + 0.35), 0.11, 0.09, segs=12)
    post = finish("DeckLampPost", bm, col, [M.metal])
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, segments=20, radius=0.17,
                            matrix=Matrix.Translation((DECK_LAMP.x, DECK_LAMP.y, DECK_LAMP.z - 0.125)))
    glass = finish("DeckLampGlass", bm, col, [M.deck_lamp])
    bm = bmesh.new()                                           # a coil of rope by the lamp
    torus(bm, (-1.1, 3.4, DECK_Z + 0.08), (0, 0, 1), 0.28, 0.035, n=24, m=8)
    torus(bm, (-1.1, 3.4, DECK_Z + 0.14), (0, 0, 1), 0.24, 0.035, n=24, m=8)
    rope = finish("Rope", bm, col, [M.rope])
    return [piles, beams, deck, post, glass, rope]


def build_lamp(col, M):
    """The underwater lamp: a clamp on the piling, a short barrel along LAMP_DIR, a bright round lens."""
    bm = bmesh.new()
    back = LAMP_POS - LAMP_DIR * 0.22
    bm_frustum(bm, tuple(back), tuple(LAMP_POS), 0.09, 0.11, segs=16)
    bm_box(bm, ((PILE_XS[0] + back.x) / 2, 0.0, back.z), (abs(back.x - PILE_XS[0]) + 0.1, 0.06, 0.06))   # clamp
    body = finish("LampBody", bm, col, [M.metal])
    bm = bmesh.new()
    u, v = basis(LAMP_DIR)
    ring = [bm.verts.new(tuple(LAMP_POS + LAMP_DIR * 0.002 + (u * math.cos(a) + v * math.sin(a)) * 0.1))
            for a in [2 * math.pi * k / 24 for k in range(24)]]
    bm.faces.new(ring)
    lens = finish("LampLens", bm, col, [M.lens])
    lens.visible_shadow = False
    return body, lens


def build_beam(col, M, length=9.6):
    """The beam: an open cone sheet from the lens along LAMP_DIR (drawn additive, even across: a hard edge)."""
    bm = bmesh.new()
    r1 = 0.1 + length * math.tan(LAMP_HALF)
    bm_frustum(bm, tuple(LAMP_POS + LAMP_DIR * 0.01), tuple(LAMP_POS + LAMP_DIR * length), 0.1, r1, segs=48,
               cap0=False, cap1=False)
    ob = finish("LampBeam", bm, col, [M.beam], recalc=False)
    for p in ob.data.polygons:
        p.use_smooth = True
    ob.visible_shadow = False
    return ob


def build_lamp_light(col):
    ld = bpy.data.lights.new("PierLamp", "SPOT")
    ld.energy = 2600.0
    ld.color = LAMP
    ld.spot_size = 2 * LAMP_HALF
    ld.spot_blend = 0.06
    ld.shadow_soft_size = 0.08
    ob = bpy.data.objects.new("PierLamp", ld)
    col.objects.link(ob)
    ob.location = LAMP_POS + LAMP_DIR * 0.03
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = LAMP_DIR.to_track_quat("-Z", "Y")
    dl = bpy.data.lights.new("DeckLamp", "POINT")             # warm deck lamp (above water)
    dl.energy = 420.0
    dl.color = srgb(255, 200, 140)
    dl.shadow_soft_size = 0.15
    do = bpy.data.objects.new("DeckLamp", dl)
    col.objects.link(do)
    do.location = DECK_LAMP - Vector((0, 0, 0.2))
    return ob, do


def build_seabed(col, M, center=(0.0, 0.0), size=40.0):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=40, y_segments=40, size=size,
                          matrix=Matrix.Translation((center[0], center[1], SEABED_Z)))
    for v in bm.verts:
        v.co.z += 0.18 * mnoise.noise(v.co * 0.35) + 0.05 * mnoise.noise(v.co * 1.7)
    ob = finish("Seabed", bm, col, [M.seabed])
    for p in ob.data.polygons:
        p.use_smooth = True
    return ob


def build_glow_fill(col, name, loc, target, energy, size=2.0, color=None):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "DISK"
    ld.size = size
    ld.energy = energy
    ld.color = color or LAMP
    ob = bpy.data.objects.new(name, ld)
    col.objects.link(ob)
    ob.location = loc
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y")
    return ob


def build_point(col, name, loc, energy, color, radius=0.02):
    ld = bpy.data.lights.new(name, "POINT")
    ld.energy = energy
    ld.color = color
    ld.shadow_soft_size = radius
    ob = bpy.data.objects.new(name, ld)
    col.objects.link(ob)
    ob.location = loc
    return ob



# --------------------------------------------------------------------------
# Choreography. Places: the pier (cross section, hop and fall, the view from above), the macro set A (the squid's
# eye) and the jig set B (the breakdown), far apart and each rendered only in its own shots.
# --------------------------------------------------------------------------
A = Vector((40.0, 0.0, -3.0))
B = Vector((-40.0, 0.0, -3.0))
H = Vector((5.0, 4.2, -4.6))                     # where the jig is worked, outside the cone (the dark side)
OH_POS = Vector((3.2, -2.4, 15.0))               # the pier light from above (the loop's join)
OH_TGT = Vector((1.5, 0.6, 0.0))
W_POS = Vector((1.78, -5.6, -3.3))                # the wide cross section (framed on the lamp and the three squid)
W_TGT = Vector((1.78, 0.0, -3.45))
JERKS = None                                     # set in hop_times()


def hop_times():
    j0 = VO_START[4] + int(round(0.8 * FPS))
    jerks = [j0, j0 + 18, j0 + 36]
    pause = jerks[-1] + 14
    fall0 = max(pause + 30, P(4, 1) - 8)
    strike = min(CUT4 - 70, P(4, 1) + int(round(4.6 * FPS)))
    return jerks, pause, fall0, strike


JERKS, PAUSE_F, FALL_F, STRIKE_F = hop_times()


def axis_point(d):
    return LAMP_POS + LAMP_DIR * d


def cone_radius(d):
    return 0.1 + d * math.tan(LAMP_HALF)


N_OUT = Vector((0.785, 0.0, 0.62)).normalized()    # in the cross section plane, perpendicular to the beam (upper side)


def hook_squid(f):
    t = t_of(f)
    p = A + Vector((0.02 * math.sin(t * 0.8), 0.0, 0.015 * math.sin(t * 1.1)))
    fwd = Vector((1.0, 0.06 * math.sin(t * 0.5), 0.03 * math.sin(t * 0.7))).normalized()
    return p, fwd


def hook_eye(f):
    p, fwd = hook_squid(f)
    return p + heading_quat(fwd) @ Vector((SQ_EYE_X, -SQ_EYE_Y, 0.004))


def hook_jig(f):
    """The glowing jig drifting across behind the squid, out of focus: placed along the macro camera's own line of
    sight (about 0.5 m past the eye), so it stays in frame while the camera pushes in."""
    u = (f - WHIP_IN) / max(1, CUT1 - WHIP_IN)
    cam, tgt = hook_cam(f)
    d = (tgt - cam).normalized()
    right = d.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(d)
    p = tgt + d * 0.32 + right * (0.09 - 0.14 * ease(u)) + up * (0.06 + 0.012 * math.sin(t_of(f) * 3.0))
    return p, (-right * 0.9 - d * 0.3).normalized()


def hook_cam(f):
    u = (f - WHIP_IN) / (CUT1 - WHIP_IN)
    e = hook_eye(f)
    return e + Vector((-0.035 + 0.01 * u, -0.14 + 0.035 * ease(u), 0.022)), e + Vector((0.012, 0.0, 0.0))


TRAP = ((2.4, -1.0, 0.6), (3.5, 1.0, -0.5), (3.9, -1.0, 1.2))     # (distance along the beam, side, y)
SCHOOL_D = 3.4                                                    # the baitfish ball, inside the cone


def trap_squid(i, f):
    """Squid i in the cross section: hovering in the dark just outside the cone edge, facing the light. Squid 1
    darts in to strike at the baitfish during the line about the ambush."""
    t = t_of(f)
    d, side, y = TRAP[i]
    centre = axis_point(d)
    p = centre + N_OUT * side * (cone_radius(d) + 0.4) + Vector((0.0, y, 0.0))
    p += Vector((0.05 * math.sin(t * 0.7 + i), 0.04 * math.sin(t * 0.5 + 2 * i), 0.04 * math.sin(t * 0.9 + i)))
    target = centre + Vector((0.0, y * 0.5, 0.0))
    tucked = 1.0
    if i == 1:
        s0 = P(2, 2) + 26
        k = smoothstep(s0, s0 + 10, f) * (1.0 - smoothstep(s0 + 22, s0 + 46, f))
        p = p.lerp(centre + N_OUT * (cone_radius(d) - 0.25) + Vector((0.0, -0.2, 0.0)), k)
        tucked = 1.0 - (smoothstep(s0 + 6, s0 + 9, f) - smoothstep(s0 + 13, s0 + 20, f))
    fwd = (target - p).normalized()
    return p, fwd, tucked


def jig_breakdown(f):
    u = (f - CUT2) / max(1, CUT3 - CUT2)
    yaw = math.radians(-35.0 + 85.0 * ease(u))
    q = Euler((math.radians(4.0 * math.sin(t_of(f) * 1.3)), math.radians(8.0), yaw), "XYZ").to_quaternion()
    p = B + Vector((0.0, 0.0, 0.004 * math.sin(t_of(f) * 1.6)))
    return p, q


def jig_hop(f):
    """The worked jig: resting low, three sharp darts upward (rod jerks), a pause, then a slow horizontal glide down.
    Returns position, heading (nose direction) and roll."""
    base = H + Vector((0.0, 0.0, -0.95))
    p = base.copy()
    nose = Vector((1.0, 0.0, -0.25)).normalized()
    roll = 0.0
    for k, j in enumerate(JERKS):
        if f >= j:
            up = 0.36 * ease_out((f - j) / 7.0, 3.0)
            p += Vector((0.06, 0.0, up))
            if f < j + 12:
                nose = Vector((0.35, 0.0, 1.0)).normalized().lerp(nose, smoothstep(j + 7, j + 12, f))
        nxt = JERKS[k + 1] if k + 1 < len(JERKS) else PAUSE_F
        if f >= j + 7:
            p += Vector((0.0, 0.0, -0.07 * smoothstep(j + 7, nxt, min(f, nxt))))     # slack: it settles back a little
    if f >= PAUSE_F:
        p += Vector((0.0, 0.0, -0.05 * smoothstep(PAUSE_F, FALL_F, f)))
    if f >= FALL_F:
        u = (f - FALL_F) / max(1, CUT4 + 30 - FALL_F)
        p += Vector((1.7 * u, 0.15 * math.sin(u * 3.0), -1.15 * u))
        nose = Vector((1.0, 0.0, -0.2)).normalized()
        roll = 0.16 * math.sin(t_of(f) * 2.4)
    if f >= STRIKE_F + 8:                                     # held: drawn back with the squid
        hp, hf, _ = hop_squid(f)
        p = hp + hf * 0.17
    return p, nose, roll


def hop_squid(f):
    """The squid: following from the dark, closing in during the drift, striking with its tentacles, then wrapping
    the jig in its arms."""
    jp = jig_hop(min(f, STRIKE_F))[0]
    far = H + Vector((1.1, 1.2, -1.0))
    t = t_of(f)
    close = jp + Vector((-0.36, 0.18, -0.15))
    k = smoothstep(FALL_F - 40, STRIKE_F, f)
    p = far.lerp(close, k) + Vector((0.03 * math.sin(t * 0.8), 0.02 * math.sin(t * 0.6), 0.03 * math.sin(t * 1.1)))
    if f >= STRIKE_F:
        p = close.lerp(jp + Vector((-0.2, 0.1, -0.08)), smoothstep(STRIKE_F, STRIKE_F + 7, f))
    fwd = (jp - p).normalized() if (jp - p).length > 1e-3 else Vector((1, 0, 0))
    tucked = 1.0 - (smoothstep(STRIKE_F - 3, STRIKE_F, f) - smoothstep(STRIKE_F + 6, STRIKE_F + 14, f))
    return p, fwd, tucked


# --------------------------------------------------------------------------
# Camera: frames[f] = (position, target, rotation, lens); f-stop per frame
# --------------------------------------------------------------------------
def whip(q, angle_deg, axis):
    """Rotation q turned by angle_deg about a world axis (the whip pans)."""
    return Quaternion(Vector(axis), math.radians(angle_deg)) @ q


def smooth_path(fn, f, k=6):
    acc, ws = Vector((0, 0, 0)), 0.0
    for d in range(-k, k + 1):
        w = math.exp(-0.5 * (d / (k / 2.0)) ** 2)
        acc += fn(f + d) * w
        ws += w
    return acc / ws


def camera_frames():
    frames, fstop = {}, {}
    Z = Vector((0, 0, 1))
    oh_q = look_at(OH_POS, OH_TGT, Vector((0, 1, 0)))
    # 0 the loop's join: the pier light from above, then a whip pan down into the water
    for f in range(1, WHIP_IN):
        ang = 0.0 if f <= 3 else (13.0 * (f - 3) ** 1.6)
        frames[f] = (OH_POS.copy(), OH_TGT.copy(), whip(oh_q, ang, oh_q @ Vector((1, 0, 0))), 30.0)
        fstop[f] = 11.0
    # 1 hook: a macro on the squid's eye, pushing in slowly; the glowing jig drifts past behind, out of focus
    for f in range(WHIP_IN, CUT1):
        cam, tgt = hook_cam(f)
        q = look_at(cam, tgt, Z)
        if f < WHIP_IN + 6:
            k = 1.0 - ease_out((f - WHIP_IN) / 6.0, 3.0)
            q = whip(q, 36.0 * k, q @ Vector((1, 0, 0)))
        frames[f] = (cam, tgt, q, 42.0)
        fstop[f] = 2.8
    # 2 the light trap: match cut on the round lamp lens (the round pupil), pulling back to the cross section
    p0 = LAMP_POS + LAMP_DIR * 0.5 + Vector((0.0, -0.32, 0.06))
    for f in range(CUT1, CUT2):
        u = (f - CUT1) / (CUT2 - CUT1)
        k = ease(min(1.0, (f - CUT1) / 56.0))
        wide = W_POS + Vector((0.25 * u, 0.35 * u, 0.1 * u))
        cam = p0.lerp(wide, k)
        tgt = LAMP_POS.lerp(W_TGT + Vector((0.2 * u, 0.0, 0.05 * u)), k)
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 34.0 - 10.0 * k)
        fstop[f] = 5.6 if k < 0.5 else 11.0
    # 3 the jig: a close, slow orbit as it turns (a turntable breakdown)
    for f in range(CUT2, CUT3):
        u = (f - CUT2) / (CUT3 - CUT2)
        cam = B + Vector((0.015, -0.43 + 0.06 * ease(u), 0.045 + 0.01 * math.sin(u * math.pi)))
        tgt = B + Vector((-0.004, 0.0, 0.0))
        frames[f] = (cam, tgt, look_at(cam, tgt, Z), 50.0)
        fstop[f] = 5.6
    # 4 hop and fall: low and below the jig, looking up towards the faint lamp glow, tracking it
    for f in range(CUT3, CUT4):
        js = smooth_path(lambda g: jig_hop(g)[0], f, 8)
        sq = hop_squid(f)[0]
        k = smoothstep(FALL_F, STRIKE_F, f)
        look = js.lerp(js.lerp(sq, 0.45), k)
        cam = js + Vector((-0.3 - 0.12 * k, -0.82 - 0.25 * k, -0.34))
        cam.z = max(cam.z, SEABED_Z + 0.5)
        q = look_at(cam, look, Z)
        if f >= CUT4 - 3:                                       # whip out
            q = whip(q, -(15.0 * (f - (CUT4 - 4)) ** 1.6), q @ Vector((1, 0, 0)))
        frames[f] = (cam, look, q, 34.0)
        fstop[f] = 4.0
    # 5 outro: whip in to the pier light from above and hold (the hold is frame 1's shot: the loop)
    for f in range(CUT4, FRAME_END + 1):
        q = oh_q
        if f < CUT4 + 7:
            k = 1.0 - ease_out((f - CUT4) / 7.0, 3.0)
            q = whip(oh_q, -40.0 * k, oh_q @ Vector((1, 0, 0)))
        frames[f] = (OH_POS.copy(), OH_TGT.copy(), q, 30.0)
        fstop[f] = 11.0
    return frames, fstop


def build_camera(col, frames, fstop):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = LENS
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = SENSOR
    cd.clip_start = 0.01
    cd.clip_end = 2000.0
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
# Scene state per frame
# --------------------------------------------------------------------------
def under_at(f):
    return 1.0 if WHIP_IN <= f < CUT4 else 0.0


def deep_at(f):
    if WHIP_IN <= f < CUT1 or CUT2 <= f < CUT3:
        return 0.35
    return 0.2 if under_at(f) else 0.0


def clear_at(f):
    if CUT1 <= f < CUT2:
        return 1.0
    if CUT3 <= f < CUT4:
        return 0.55
    return 0.3 if under_at(f) else 0.0


def flash_at(f):
    """How hard the chromatophores flash: the hook squid reacts as the jig passes; the hop squid at the strike."""
    if WHIP_IN <= f < CUT1:
        return 0.15 + 0.85 * smoothstep(WHIP_IN + 20, WHIP_IN + 60, f)
    if CUT3 <= f < CUT4:
        return 0.2 + 0.8 * smoothstep(STRIKE_F - 20, STRIKE_F, f)
    return 0.25


def set_constant(idb):
    for fc in iter_fcurves(idb):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def key_scene_state(sc, world_vals):
    under_v, deep_v = world_vals
    for f in range(1, FRAME_END + 1):
        vals = (("under", under_at(f)), ("deep", deep_at(f)), ("clear", clear_at(f)), ("flash", flash_at(f)),
                ("glow", 1.0), ("beam", 1.0))
        for prop, val in vals:
            sc[prop] = val
            sc.keyframe_insert(data_path=f'["{prop}"]', frame=f)
        under_v.outputs[0].default_value = under_at(f)
        deep_v.outputs[0].default_value = deep_at(f)
        under_v.outputs[0].keyframe_insert("default_value", frame=f)
        deep_v.outputs[0].keyframe_insert("default_value", frame=f)
    set_constant(sc)
    set_constant(sc.world.node_tree)


def animate_squid(sq, f_lo, f_hi, state, fin_hz=1.4, amp=0.01):
    for f in range(f_lo, f_hi + 1):
        r = state(f)
        p, fwd = r[0], r[1]
        tucked = r[2] if len(r) > 2 else 1.0
        key_squid(sq, f, p, fwd, roll=0.0, fin_phase=2 * math.pi * fin_hz * t_of(f), fin_amp=amp, tucked=tucked)


def animate_jig(ob, f_lo, f_hi, state):
    for f in range(f_lo, f_hi + 1):
        r = state(f)
        ob.location = r[0]
        if isinstance(r[1], Quaternion):
            ob.rotation_quaternion = r[1]
        else:
            ob.rotation_quaternion = heading_quat(r[1], r[2] if len(r) > 2 else 0.0)
        key(ob, "location", f)
        key(ob, "rotation_quaternion", f)


def sparks(center, radius, f_lo, f_hi, n, seed):
    """Stylised glowing plankton sparks drifting slowly around a subject (an effect, not the squid's own light)."""
    rng = random.Random(seed)
    sp = Particles()
    t0, t1 = t_of(f_lo), t_of(f_hi)
    for _ in range(n):
        birth = rng.uniform(t0 - 3.0, t1)
        p = Vector(center) + Vector((rng.gauss(0, radius), rng.gauss(0, radius), rng.gauss(0, radius * 0.7)))
        v = Vector((rng.gauss(0, 0.02), rng.gauss(0, 0.02), rng.uniform(-0.01, 0.03)))
        sp.emit(birth, p, v, rng.uniform(2.5, 5.0), rng.uniform(0.0012, 0.003))
    return sp


def build_trail(col, M, f_lo, f_hi):
    """A short glowing comet trail behind the jig (the last second or so of its path)."""
    pts = [jig_hop(f)[0] for f in range(f_lo, f_hi + 1, 2)]
    cu = bpy.data.curves.new("JigTrail", "CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for pt, v in zip(sp.points, pts):
        pt.co = (*v, 1.0)
    ob = bpy.data.objects.new("JigTrailPath", cu)
    col.objects.link(ob)
    ob.hide_render = True
    ctrl = empty_object("TrailCtrl", col, size=0.05)
    cum = [0.0]                                  # trimming works by length: map each frame to its arc length
    for p0, p1 in zip(pts, pts[1:]):
        cum.append(cum[-1] + (p1 - p0).length)
    total = max(1e-6, cum[-1])
    for f in range(f_lo, f_hi + 1):
        u = (f - f_lo) / 2.0
        i = min(int(u), len(cum) - 2)
        s = cum[i] + (cum[i + 1] - cum[i]) * min(1.0, u - i)
        ctrl.location = (max(0.002, s / total), max(0.0, s - 0.3) / total, 0.0)
        key(ctrl, "location", f)
    ng, gi, go = new_group("TrailGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    oi.transform_space = "RELATIVE"
    oi.inputs["Object"].default_value = ob
    ci = ng.nodes.new("GeometryNodeObjectInfo")
    ci.inputs["Object"].default_value = ctrl
    trim = ng.nodes.new("GeometryNodeTrimCurve")
    trim.mode = "FACTOR"
    ng.links.new(oi.outputs["Geometry"], trim.inputs["Curve"])
    cl = separate(ng, ci.outputs["Location"])
    ng.links.new(cl["Y"], trim.inputs[2])
    ng.links.new(cl["X"], trim.inputs[3])
    prof = ng.nodes.new("GeometryNodeCurvePrimitiveCircle")
    prof.inputs["Resolution"].default_value = 6
    prof.inputs["Radius"].default_value = 0.0018
    c2m = ng.nodes.new("GeometryNodeCurveToMesh")
    ng.links.new(trim.outputs[0], c2m.inputs["Curve"])
    ng.links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
    setm = ng.nodes.new("GeometryNodeSetMaterial")
    setm.inputs["Material"].default_value = M.trail
    ng.links.new(c2m.outputs[0], setm.inputs["Geometry"])
    ng.links.new(setm.outputs[0], go.inputs["Geometry"])
    tr = bpy.data.objects.new("JigTrail", bpy.data.meshes.new("JigTrail"))
    col.objects.link(tr)
    tr.modifiers.new("Trail", "NODES").node_group = ng
    tr.visible_shadow = False
    return tr


def build_hop_line(col, M, egi, f_lo, f_hi):
    """The line from the jig's nose up towards the rod (out of frame above the surface): taut on the jerks, a soft
    belly on the fall."""
    a = empty_object("LineJig", col, size=0.02)
    b = empty_object("LineTop", col, size=0.05)
    m = empty_object("LineMid", col, size=0.05)
    for f in range(f_lo, f_hi + 1):
        p, nose, roll = jig_hop(f)
        tip = p + heading_quat(nose, roll) @ Vector((EGI_L / 2 + 0.006, 0.0, 0.002))
        top = Vector((p.x - 2.2, p.y - 0.4, -0.05))
        slack = 0.0 if f < FALL_F else 0.35 * smoothstep(FALL_F, FALL_F + 30, f)
        a.location, b.location = tip, top
        m.location = tip.lerp(top, 0.5) + Vector((0.4 * slack, 0.0, -slack))
        for ob in (a, b, m):
            key(ob, "location", f)
    return line_object("JigLine", col, M, a, b, radius=0.0007, sag=m, count=40)


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------
def build_overlays(hud, st):
    hud.kicker("THE FISHING SERIES")
    # 1 hook
    p1 = P(1, 1)
    hud.serif("Why most anglers fish *blindly* for squid.", WHIP_IN + 2, p1 - 9, y=-0.30, size=0.048,
              lines=["Why most anglers", "fish *blindly* for squid."], reveal=2.3)
    hud.serif("Squid hunt with *their eyes.*", p1, CUT1 - 9, y=-0.30, size=0.048, lines=["Squid hunt", "with *their eyes.*"])
    # 2 the light trap
    hud.badge(1, "THE LIGHT CONE", "squid hold on the shadow line", CUT1 + 8, CUT2 - 8)
    t1, t2 = P(2, 1), P(2, 2)
    hud.serif("Squid don't sit *in the light.*", CUT1 + 6, t1 - 9, lines=["Squid don't sit", "*in the light.*"])
    hud.serif("They lurk on the *shadow line.*", t1, t2 - 9, lines=["They lurk on", "the *shadow line.*"],
              src="CONSISTENT WITH JIGGING BOAT LIGHT STUDIES")
    hud.serif("Light draws *baitfish.* Squid *ambush* from the dark.", t2, CUT2 - 9,
              lines=["Light draws *baitfish.*", "Squid *ambush* from the dark."], src="ANGLER TECHNIQUE")
    hud.pin(lambda f: axis_point(4.3), "light cone", CUT1 + 46, t1 - 4, dot_color=srgb(150, 255, 190), side=0.06,
            y_band=(-0.2, 0.32))
    hud.pin(lambda f: axis_point(3.0) + N_OUT * cone_radius(3.0), "shadow line", t1 + 4, t2 - 4,
            dot_color=srgb(150, 255, 190), side=0.07, rise=0.09, y_band=(-0.2, 0.32))
    hud.pin(lambda f: axis_point(SCHOOL_D), "baitfish in the light", t2 + 4, t2 + 70, dot_color=GOLD, side=-0.05,
            y_band=(-0.2, 0.32))
    hud.pin(lambda f: trap_squid(0, f)[0], "squid wait in the dark", t2 + 78, CUT2 - 8, dot_color=GOLD, side=-0.06,
            y_band=(-0.22, 0.32))
    # 3 the jig
    hud.badge(2, "GLOWING EGI JIG", "cloth body, barbless crown", CUT2 + 8, CUT3 - 8)
    j1 = P(3, 1)
    jm = (j1 + CUT3) // 2
    hud.serif("A cloth wrapped *egi* jig.", CUT2 + 6, j1 - 9, lines=["A cloth wrapped", "*egi* jig."], src="ANGLER TECHNIQUE")
    hud.serif("Squid eyes are tuned to *blue green* light.", j1, jm - 9, lines=["Squid eyes are tuned", "to *blue green* light."],
              src="CEPHALOPOD VISION: ONE BLUE GREEN PIGMENT")
    hud.serif("A glowing *silhouette* stands out.", jm, CUT3 - 9, lines=["A glowing *silhouette*", "stands out."],
              src="ANGLER TECHNIQUE")
    egi = st["egi_b"]

    def on_egi(local):
        def fn(f):
            p, q = jig_breakdown(f)
            return p + q @ Vector(local)
        return fn
    span = CUT3 - CUT2
    for (lo, hi), local, label, side, rise in (((0.05, 0.24), (0.0, 0.0, 0.0125), "cloth wrap", 0.05, 0.08),
                                               ((0.27, 0.46), (-0.012, 0.0, -0.0095), "glowing body", -0.05, 0.15),
                                               ((0.49, 0.68), (0.036, 0.0, -0.016), "weighted nose", 0.05, 0.16),
                                               ((0.71, 0.93), (-0.064, 0.0, 0.0), "barbless crown", -0.05, 0.1)):
        hud.pin(on_egi(local), label, CUT2 + int(span * lo), CUT2 + int(span * hi), dot_color=GOLD, side=side,
                rise=rise, y_band=(-0.22, 0.3))
    _ = egi
    # 4 hop and fall
    hud.badge(3, "HOP AND FALL", "three jerks, then the drift", CUT3 + 6, CUT4 - 6)
    h1 = P(4, 1)
    hud.serif("Three sharp *rod jerks.*", CUT3 + 4, PAUSE_F - 9, lines=["Three sharp", "*rod jerks.*"], src="ANGLER TECHNIQUE")
    hud.serif("Then *pause.*", PAUSE_F, h1 - 9, lines=["Then *pause.*"])
    hud.serif("Most strikes come *on the fall.*", h1, CUT4 - 4, lines=["Most strikes come", "*on the fall.*"],
              src="ANGLER TECHNIQUE")
    hud.pin(lambda f: jig_hop(f)[0], "dart up", JERKS[0] - 4, PAUSE_F - 6, dot_color=GOLD, side=0.06, y_band=(-0.2, 0.3))
    hud.pin(lambda f: jig_hop(f)[0], "the drift down", h1 + 4, STRIKE_F - 14, dot_color=GOLD, side=0.06, y_band=(-0.2, 0.3))
    hud.pin(lambda f: jig_hop(f)[0], "strike", STRIKE_F - 4, CUT4 - 10, dot_color=PIN_RED, side=-0.06, y_band=(-0.2, 0.3))
    # 5 outro
    hud.serif("Master the *shadow line.*", CUT4 + 8, (CUT4 + FRAME_END) // 2 - 9, lines=["Master the", "*shadow line.*"])
    hud.serif("And you'll quickly *realize...*", (CUT4 + FRAME_END) // 2, FRAME_END - 9,
              lines=["And you'll quickly", "*realize...*"], reveal=3.0)


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
    """Bloom on the lamp and the glow, a high contrast night grade (crushed blacks, cool shadows), overlays on top."""
    ng = bpy.data.node_groups.new("Composite", "CompositorNodeTree")
    main.compositing_node_group = ng
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.scene = main
    img = rl.outputs["Image"]
    try:
        gl = ng.nodes.new("CompositorNodeGlare")
        gl.inputs["Type"].default_value = "Bloom"
        for nm, v in (("Threshold", 0.8), ("Strength", 0.35), ("Size", 0.6)):
            if nm in gl.inputs:
                gl.inputs[nm].default_value = v
        ng.links.new(img, gl.inputs["Image"])
        img = gl.outputs["Image"]
    except Exception as e:                  # noqa: BLE001
        print("bloom skipped:", e)
    try:
        cb = ng.nodes.new("CompositorNodeColorBalance")
        vals = {"Lift": (1.0, 1.002, 1.004, 1.0), "Gamma": (1.0, 1.0, 1.0, 1.0), "Gain": (1.0, 1.03, 1.0, 1.0)}
        for sock in cb.inputs:                  # (5.x: a factor and a colour socket share each name)
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
        c.points.new(0.2, 0.14)
        c.points.new(0.7, 0.78)
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
    worst = {"water surface (above)": (1e9, 0), "surface (under)": (1e9, 0), "pier": (1e9, 0), "seabed": (1e9, 0),
             "squid": (1e9, 0), "jig": (1e9, 0)}

    def upd(k, d, f):
        if d < worst[k][0]:
            worst[k] = (d, f)
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    verts, polys = [], []
    for ob in st["pier"]:
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        base = len(verts)
        verts.extend(ev.matrix_world @ v.co for v in me.vertices)
        polys.extend([base + i for i in pl.vertices] for pl in me.polygons)
        ev.to_mesh_clear()
    pier_bvh = BVHTree.FromPolygons(verts, polys)
    for f in range(1, FRAME_END + 1):
        p = frames[f][0]
        t = t_of(f)
        if under_at(f):
            if abs(p.x) < 30:
                upd("surface (under)", -p.z - 0.2, f)
                upd("seabed", p.z - SEABED_Z - 0.3, f)
        else:
            upd("water surface (above)", p.z - sea_height(p.x, p.y, t) - 0.3, f)
        hit = pier_bvh.find_nearest(p)
        if hit[0] is not None:
            upd("pier", hit[3] - 0.25, f)
        if WHIP_IN <= f < CUT1:
            upd("squid", (hook_squid(f)[0] - p).length - 0.08, f)     # the eye macro is meant to be close
        if CUT1 <= f < CUT2:
            for i in range(3):
                upd("squid", (trap_squid(i, f)[0] - p).length - 0.3, f)
        if CUT2 <= f < CUT3:
            upd("jig", (jig_breakdown(f)[0] - p).length - 0.12, f)
        if CUT3 <= f < CUT4:
            upd("jig", (jig_hop(f)[0] - p).length - 0.2, f)
            upd("squid", (hop_squid(f)[0] - p).length - 0.3, f)
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
    print("loop: water at t=0 vs t=55 s (should match):", round(sea_height(5.0, 7.0, 0.0) - sea_height(5.0, 7.0, LOOP_S), 6))
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
    print("cone check: hop jig outside the cone at the strike:", round(math.degrees(in_cone(jig_hop(STRIKE_F)[0])), 1),
          "deg (negative = outside);  trap squid:",
          [round(math.degrees(in_cone(trap_squid(i, CUT1 + 60)[0])), 1) for i in range(3)])


def subject_report(frames, st):
    hud = st["hud"]
    sc = bpy.data.scenes["Main"]
    subj = {WHIP_IN + 20: ("squid eye", hook_eye), (WHIP_IN + CUT1) // 2: ("jig (background)", lambda f: hook_jig(f)[0]),
            CUT1 + 60: ("lamp", lambda f: LAMP_POS), CUT1 + 120: ("baitfish", lambda f: axis_point(4.0)),
            CUT1 + 140: ("squid 0", lambda f: trap_squid(0, f)[0]), CUT1 + 200: ("squid 2", lambda f: trap_squid(2, f)[0]),
            CUT2 + 40: ("jig", lambda f: jig_breakdown(f)[0]), JERKS[1]: ("jig on a jerk", lambda f: jig_hop(f)[0]),
            STRIKE_F: ("strike: jig", lambda f: jig_hop(f)[0]), STRIKE_F + 1: ("strike: squid", lambda f: hop_squid(f)[0]),
            CUT4 + 20: ("lamp glow from above", lambda f: Vector((LAMP_POS.x + 1.4, 0.0, 0.0)))}
    print("\n=== SUBJECTS ON SCREEN (x in +-0.281, y in +-0.5) ===")
    for f, (nm, fn) in sorted(subj.items()):
        sc.frame_set(f)
        s = hud.project(f, fn(f))
        print(f"frame {f}: {nm} ->", None if s is None else (round(s.x, 3), round(s.y, 3)))


def write_events(path, st):
    data = {
        "fps": FPS, "frames": FRAME_END, "loop": True,
        "cuts": [WHIP_IN, CUT1, CUT2, CUT3, CUT4], "whips": [[1, WHIP_IN], [CUT4 - 4, CUT4 + 6]],
        "overlays": sorted(st["hud"].events), "jerks": JERKS, "pause": PAUSE_F, "fall": FALL_F, "strike": STRIKE_F,
        "under": [[WHIP_IN, CUT4]], "trap_strike": P(2, 2) + 26,
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
    for prop in ("under", "deep", "clear", "flash", "glow", "beam"):
        sc[prop] = 0.0
    c_sea, c_pier, c_a, c_b = collection("Harbour"), collection("Pier"), collection("EyeMacro"), collection("JigSet")
    c_hop, c_fx, c_cam = collection("HopAndFall"), collection("Lights"), collection("Camera")
    atmos = build_atmos_group()
    M = build_materials(atmos)
    _w, w_under, w_deep = build_world(sc)
    frames, fstop = camera_frames()
    moon = bpy.data.lights.new("Moon", "SUN")
    moon.energy = 0.06
    moon.color = srgb(170, 190, 230)
    mo = bpy.data.objects.new("Moon", moon)
    c_fx.objects.link(mo)
    mo.rotation_mode = "QUATERNION"
    mo.rotation_quaternion = (-MOON_DIR).to_track_quat("-Z", "Y")
    # the pier and its light
    sea = build_sea(c_sea, M)
    sea.location = (OH_TGT.x, OH_TGT.y, 0.0)
    pier = build_pier(c_pier, M)
    lamp_body, lens = build_lamp(c_pier, M)
    beam = build_beam(c_pier, M)
    lamp_l, deck_l = build_lamp_light(c_fx)
    seabed = build_seabed(c_pier, M, center=(4.0, 1.0), size=26.0)
    surf = build_surface_under(c_pier, M, center=(3.0, 0.0, -0.02), size=60.0)
    plank = specks("Plankton", c_pier, M.specks, (3.2, 1.6, -3.9), (5.5, 5.5, 3.6), 3200, 0.005, 41)
    school_fish = bpy.data.objects.new("SchoolFish", baitfish_mesh(M, L=0.085))
    c_pier.objects.link(school_fish)
    school_fish.hide_render = True
    school_fish.hide_viewport = True
    school = build_school(c_pier, school_fish, axis_point(SCHOOL_D), 170, 0.15, 1.0, 0.4, 21)
    trap = []
    for i in range(3):
        s = make_squid(c_pier, M, f"TrapSquid{i}")
        animate_squid(s, CUT1 - 2, CUT2 + 2, lambda f, i=i: trap_squid(i, f), fin_hz=1.1 + 0.15 * i, amp=0.009)
        trap.append(s)
    trap_lights = []                 # scattered light from the beam: each squid on the shadow line catches a little
    for i in range(3):
        lt = build_point(c_fx, f"TrapGlow{i}", Vector((0, 0, 0)), 18.0, srgb(170, 255, 205), radius=0.25)
        lt.data.use_custom_distance = True
        lt.data.cutoff_distance = 2.0
        for f in range(CUT1 - 2, CUT2 + 2):
            p, fwd, _t = trap_squid(i, f)
            lt.location = p - N_OUT * TRAP[i][1] * 0.45 + Vector((0.0, -0.25, 0.0))
            key(lt, "location", f)
        trap_lights.append(lt)
    # the hop and fall (at the pier, outside the cone)
    hop_sq = make_squid(c_hop, M, "HopSquid")
    animate_squid(hop_sq, CUT3 - 2, CUT4 + 2, hop_squid, fin_hz=1.6, amp=0.012)
    hop_egi = make_egi(c_hop, M, "HopEgi")
    animate_jig(hop_egi, CUT3 - 2, CUT4 + 2, jig_hop)
    hop_line = build_hop_line(c_hop, M, hop_egi, CUT3 - 2, CUT4 + 2)
    trail = build_trail(c_hop, M, CUT3, CUT4 + 2)
    hop_fill = build_glow_fill(c_fx, "HopFill", H + Vector((-1.0, -1.5, 3.5)), H, 140.0, size=4.0)
    hop_glow = build_point(c_fx, "HopJigGlow", H, 5.0, srgb(130, 255, 190))
    for f in range(CUT3 - 2, CUT4 + 2):
        hop_glow.location = jig_hop(f)[0]
        key(hop_glow, "location", f)
    hop_sparks = sparks(H + Vector((0.6, 0.2, -0.3)), 0.9, CUT3, CUT4, 260, 52).build("HopSparks", c_hop, M.sparks)
    # A: the eye macro
    eye_sq = make_squid(c_a, M, "EyeSquid")
    animate_squid(eye_sq, WHIP_IN - 2, CUT1 + 2, lambda f: (*hook_squid(f), 1.0), fin_hz=1.2, amp=0.008)
    eye_egi = make_egi(c_a, M, "EyeEgi")
    animate_jig(eye_egi, WHIP_IN - 2, CUT1 + 2, hook_jig)
    a_key = build_glow_fill(c_fx, "EyeKey", A + Vector((0.25, -0.2, 0.55)), A, 6.0, size=0.6, color=srgb(225, 250, 236))
    a_rim = build_glow_fill(c_fx, "EyeRim", A + Vector((-0.4, 0.45, 0.2)), A, 6.0, size=0.5, color=CYAN)
    a_glow = build_point(c_fx, "EyeJigGlow", A, 2.5, srgb(130, 255, 190))
    for f in range(WHIP_IN - 2, CUT1 + 2):
        a_glow.location = hook_jig(f)[0]
        key(a_glow, "location", f)
    a_sparks = sparks(A + Vector((0.1, 0.25, 0.0)), 0.35, WHIP_IN, CUT1, 160, 53).build("EyeSparks", c_a, M.sparks)
    # B: the jig breakdown
    egi_b = make_egi(c_b, M, "BreakdownEgi")
    animate_jig(egi_b, CUT2 - 2, CUT3 + 2, jig_breakdown)
    b_key = build_glow_fill(c_fx, "JigKey", B + Vector((0.15, -0.35, 0.45)), B, 7.0, size=0.5,
                            color=srgb(220, 255, 236))
    b_rim = build_glow_fill(c_fx, "JigRim", B + Vector((-0.35, 0.35, 0.15)), B, 6.0, size=0.4, color=CYAN)
    b_sparks = sparks(B + Vector((0.0, 0.15, 0.0)), 0.25, CUT2, CUT3, 140, 54).build("JigSparks", c_b, M.sparks)
    for lt in (a_key, a_rim, a_glow, b_key, b_rim):
        lt.data.use_custom_distance = True
        lt.data.cutoff_distance = 3.0
    build_camera(c_cam, frames, fstop)
    key_scene_state(sc, (w_under, w_deep))
    # visibility: each place renders only in its own shots
    above = [(1, WHIP_IN - 1), (CUT4, FRAME_END)]
    pier_uw = [(CUT1, CUT2 - 1), (CUT3, CUT4 - 1)]
    vis = {}

    def show(objs, spans):           # the last call for an object wins: keys from several calls would add up
        for o in objs:
            vis[o.name] = (o, tuple(spans))
    show([sea], above)
    show([o for o in c_pier.all_objects if o.type in ("MESH", "CURVE") and o.name not in ("SchoolFish",)], pier_uw + above)
    show([surf, beam, plank, school, seabed], [(CUT1, CUT2 - 1), (CUT3, CUT4 - 1)])
    show([o for o in c_hop.all_objects if o.type in ("MESH", "CURVE") and o.name != "JigTrailPath"], [(CUT3, CUT4 - 1)])
    show([o for s in trap for o in (s["body"], s["arms"], s["tent"])], [(CUT1, CUT2 - 1)])
    show(trap_lights, [(CUT1 - 1, CUT2 - 1)])
    show([trail], [(FALL_F + 10, STRIKE_F - 3)])                    # the trail only marks the slow drift down
    show([hop_fill, hop_glow], [(CUT3 - 1, CUT4 - 1)])
    show([o for o in c_a.all_objects if o.type in ("MESH", "CURVE")], [(WHIP_IN, CUT1 - 1)])
    show([o for o in c_b.all_objects if o.type in ("MESH", "CURVE")], [(CUT2, CUT3 - 1)])
    groups = {}
    for o, spans in vis.values():
        groups.setdefault(spans, []).append(o)
    for spans, objs in groups.items():
        key_visibility(objs, list(spans))
    _ = (lamp_body, lens, lamp_l, deck_l, hop_line, trail, hop_fill, hop_sparks, a_sparks, b_sparks)
    hs, hcam, hcol = build_hud_scene()
    hud_materials(M)
    hud = Hud(hcam, hcol, M, frames)
    st = dict(frames=frames, hud=hud, pier=pier, egi_b=egi_b, trap=trap, hop_squid=hop_sq, hop_egi=hop_egi)
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
