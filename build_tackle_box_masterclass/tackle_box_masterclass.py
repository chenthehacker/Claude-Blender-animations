"""Tackle box masterclass: gear breakdown. Vertical 9:16, 24 fps, 55 s (1320 frames), Blender 5.2 EEVEE.

Run with the Blender binary:
    blender --background --python tackle_box_masterclass.py -- [options]
or with the bpy Python module (pip install bpy):
    python tackle_box_masterclass.py [options]

Options:
    --save PATH        save the generated scene as a .blend file (fonts packed)
    --check            camera clearances (box, trays, lures), camera acceleration and turn rate, cuts against
                       wipes, overlay overlaps and off-screen boxes, key subjects projected to the screen
    --frames 1,300     render these frames as PNG stills into --outdir
    --outdir DIR       output directory for renders
    --scale PCT        resolution percentage (default 100 -> 1080x1920)
    --samples N        EEVEE render samples (default 32)
    --no-blur          switch motion blur off (quick tests)
    --events PATH      write overlay, action and voiceover timing as JSON (input for tackle_box_audio.py)

The lures, box, studio and baitfish are all modelled procedurally here; nothing is imported. Facts in the
on-screen text and the voiceover are listed with their sources in SOURCES.md.

Layout (metres, +Z up). Set 1, the tackle box, sits at the origin: a three level cantilever box whose middle
tray and top tray swing out towards +Y on parallel arms (bottom bin y = 0, middle tray y = 0.23, top tray
y = 0.46 when open). Set 3 (the clarity rule) is at x = 4, set 4 (the forage match) at x = 8. One large
backdrop sphere, a studio to nature gradient, surrounds all three; the depth of field turns its soft
light blobs into bokeh.

Overlays (kicker, pill badges, serif captions, pins) live in a second scene ("HUD") with its own static
camera and are laid over the 3D render in the compositor, so the depth of field never blurs the text.

Beats (the brief asked for 0-5, 5-20, 20-40, 40-50, 50-55 s; the cuts follow the measured voiceover
clips instead, see VO below, so they land up to 2.7 s later than the brief's marks):
    1  F1-149      hook: macro push into an overflowing box; the junk flies out, the keepers snap into a grid
    2  F150-543    the layout: top-down dolly over the three tiers (surface, mid-column, bottom)
    3  F544-971    the clarity rule: orbit around a loud bright spinnerbait and a natural ghost jerkbait
    4  F972-1187   the forage match: side cross-section, the cutaway swimbait shrinks to the baitfish size
    5  F1188-1320  outro: rapid pull-back to the organised box, recap and the call to action
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

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
FPS = 24
SHUTTER = 0.5          # motion blur exposure, in frames
FRAME_END = 1320
RES_X, RES_Y = 1080, 1920
LENS = 35.0
SENSOR = 36.0          # vertical sensor fit
HUD_DIST = 0.25        # overlay plane distance in front of the (separate, static) overlay camera
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
INK = srgb(10, 14, 18)
PIN_RED = srgb(240, 96, 80)
GLOW = srgb(255, 200, 96)
TEAL = srgb(12, 86, 92)

CUT1, CUT2, CUT3, CUT4 = 158, 566, 948, 1146
WIPES = [(CUT1 - 4, CUT1 + 3), (CUT2 - 4, CUT2 + 3), (CUT3 - 4, CUT3 + 3), (CUT4 - 4, CUT4 + 3)]   # opaque a-2..b+2
SEGMENTS = [(1, CUT1 - 1), (CUT1, CUT2 - 1), (CUT2, CUT3 - 1), (CUT3, CUT4 - 1), (CUT4, FRAME_END)]

S3 = Vector((4.0, 0.0, 0.15))    # set 3: the clarity rule
S4 = Vector((8.0, 0.0, 0.15))    # set 4: the forage match

# Voiceover: five clips (voiceover/vo1..vo5), one per beat. VO_START is the frame each clip starts on;
# VO_LEN its length in seconds as measured from the files (vo_timing.json, written by tackle_box_audio.py,
# overrides these defaults). Caption timings below are offsets into each clip.
VO_START = {1: 2, 2: 164, 3: 572, 4: 954, 5: 1140}
VO_LEN = {1: 6.35, 2: 16.27, 3: 14.99, 4: 7.47, 5: 6.03}
VO_TEXT = {
    1: "Stop carrying a fifty pound tackle box full of junk you never use. Here's how to pack like a pro.",
    2: "First, ditch the chaos. Never organize by brand. Organize by water column. Top tray: surface topwaters "
       "for low-light feeding. Middle: suspending jerkbaits to cover ground. Bottom: heavy jigs to anchor your setup.",
    3: "Next, choosing the right lure comes down to one golden rule: water clarity. Murky water? Throw "
       "high-contrast brights with heavy vibration. Crystal clear? Switch to subtle, natural ghost patterns "
       "that fool line-shy fish.",
    4: "Always match your profile size to the local forage. If the baitfish are small, downsize your "
       "presentation immediately.",
    5: "Lightweight, tactical, and ready for anything. Hit subscribe for more insider masterclasses.",
}


def _load_vo_timing():
    """Measured clip lengths and phrase starts (seconds into each clip), if the audio script wrote them."""
    path = os.path.join(HERE, "vo_timing.json")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        data = json.load(fh)
    for k, v in data.get("length", {}).items():
        VO_LEN[int(k)] = float(v)
    return {int(k): v for k, v in data.get("phrases", {}).items()}


VO_PHRASES = _load_vo_timing()


def phrase_frame(clip, i):
    """Frame where sentence i of a voiceover clip starts: measured if available, else estimated from its
    characters (each sentence end counts as a short pause)."""
    if clip in VO_PHRASES and i < len(VO_PHRASES[clip]):
        return VO_START[clip] + int(round(VO_PHRASES[clip][i] * FPS))
    parts = [p for p in VO_TEXT[clip].replace("?", "?|").replace(".", ".|").replace(":", ":|").split("|") if p.strip()]
    w = [len(p.strip()) + 8 for p in parts]
    t = VO_LEN[clip] * sum(w[:i]) / sum(w)
    return VO_START[clip] + int(round(t * FPS))


def vo_end(clip):
    return VO_START[clip] + int(round(VO_LEN[clip] * FPS))


def frame_time(f):
    return f / FPS


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
    """Ease out with a small overshoot (a snap)."""
    u = min(1.0, max(0.0, u)) - 1.0
    return u * u * ((s + 1) * u + s) + 1.0


# --------------------------------------------------------------------------
# Generic helpers (shared with the other builds)
# --------------------------------------------------------------------------
def smoothstep(e0, e1, x):
    if e0 > e1:
        return 1.0 - smoothstep(e1, e0, x)
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)



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


# --------------------------------------------------------------------------
# Geometry helpers for the tackle
# --------------------------------------------------------------------------
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


def loft_half(bm, rings, n=16, mat=0, cap_mat=1):
    """The +Y half of a loft, closed by a flat cap on the y = 0 plane (a cutaway section)."""
    vr = []
    for x, w, h, zc in rings:
        vr.append([bm.verts.new((x, w * math.cos(a), zc + h * math.sin(a)))
                   for a in np.linspace(-math.pi / 2, math.pi / 2, n + 1)])
    for ra, rb in zip(vr, vr[1:]):
        for i in range(n):
            f = bm.faces.new((ra[i], ra[i + 1], rb[i + 1], rb[i]))
            f.material_index, f.smooth = mat, True
        f = bm.faces.new((ra[n], rb[n], rb[0], ra[0]))           # the cut face, one strip per ring pair
        f.material_index, f.smooth = cap_mat, False
    return vr


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


def body_profile(L, hmax, wmax, hpeak=0.35, nose=0.25, tail=0.06, tail_w=None, zc=0.0, n=26, p=2.0, nose_x=None):
    """Rings for a fish-like body from nose (x = +L/2) to tail (x = -L/2)."""
    out = []
    tail_w = wmax * 0.3 if tail_w is None else tail_w
    for i in range(n + 1):
        u = i / n                                  # 0 nose .. 1 tail
        x = L / 2 - u * L
        if u < hpeak:
            k = math.sin(0.5 * math.pi * u / hpeak) ** nose
        else:
            v = (u - hpeak) / (1 - hpeak)
            k = 1 - (1 - tail) * v ** 1.6
        h = hmax * k
        w = wmax * k if u < hpeak else wmax * k * (1 - 0.3 * (u - hpeak)) + 0 * tail_w
        if i == 0 or i == n:
            out.append((x, 0.0, 0.0, zc))
        else:
            out.append((x, max(w, 1e-4), max(h, 1e-4), zc, p))
    return out


# --------------------------------------------------------------------------
# Hooks and hardware
# --------------------------------------------------------------------------
def treble_bm(bm, s=1.0, mat=0):
    """Treble hook with its eye at the origin and the shank down -Z. s: size factor (1 = a #6 treble)."""
    R, shank, wire = 0.0032 * s, 0.011 * s, 0.00045 * s
    torus(bm, (0, 0, -0.0016 * s), (0, 1, 0), 0.0013 * s, wire * 0.9, n=12, m=5, mat=mat)
    tube(bm, [(0, 0, -0.003 * s), (0, 0, -shank)], wire, segs=6, mat=mat)
    for k in range(3):
        a = 2 * math.pi * k / 3 + math.pi / 2
        d = Vector((math.cos(a), math.sin(a), 0.0))
        z = Vector((0, 0, 1))
        c = Vector((0, 0, -shank)) + d * R
        pts = [c - d * R * math.cos(t) - z * R * math.sin(t) for t in np.linspace(0, math.pi, 9)]
        tip = c + d * R
        pts += [tip + z * (0.002 * s * j) - d * (0.0004 * s * j) for j in (1, 2, 3)]
        tube(bm, pts, lambda u: wire * (1.0 if u < 0.75 else max(0.12, 1 - (u - 0.75) * 3.6)), segs=6, mat=mat)
        b0 = pts[-2]
        tube(bm, [b0, b0 - z * 0.0016 * s + d * 0.0006 * s], lambda u: wire * 0.7 * (1 - 0.85 * u), segs=4, mat=mat)  # barb


def single_hook_bm(bm, shank, gap, s=1.0, mat=0, start=(0, 0, 0), up=True):
    """Single hook: shank back along -X from start, bend, point up (+Z)."""
    wire = 0.0007 * s
    p0 = Vector(start)
    p1 = p0 + Vector((-shank, 0, 0))
    R = gap / 2
    c = p1 + Vector((0, 0, R if up else -R))
    sgn = 1 if up else -1
    pts = [p0, p0 + Vector((-shank * 0.5, 0, 0))]
    pts += [c + Vector((-R * math.sin(t), 0, -sgn * R * math.cos(t))) for t in np.linspace(0, math.pi, 10)]
    tip = c + Vector((0, 0, sgn * R))
    pts += [tip + Vector((shank * 0.25 * j / 3, 0, sgn * 0.0006 * j)) for j in (1, 2, 3)]
    tube(bm, pts, lambda u: wire * (1.0 if u < 0.8 else max(0.12, 1 - (u - 0.8) * 4.5)), segs=6, mat=mat)
    return p1, tip


def eye_pair(bm, x, w, z, r, mat):
    """Two domed 3D eyes on the sides of a body (half width w at x)."""
    for sy in (1, -1):
        ellipsoid(bm, (x, sy * (w - r * 0.25), z), (r, r * 0.45, r), subdiv=2, mat=mat)


def hang_treble(bm, x, z, s, mat_hook, mat_ring):
    """Split ring under the body at (x, z) and a treble hanging from it."""
    torus(bm, (x, 0, z - 0.0026 * s), (0, 1, 0), 0.0024 * s, 0.0004 * s, n=12, m=5, mat=mat_ring)
    sub = bmesh.new()
    treble_bm(sub, s, mat=mat_hook)
    for v in sub.verts:
        v.co = Vector((v.co.x, v.co.y, v.co.z)) + Vector((x, 0, z - 0.0045 * s))
    tmp = bpy.data.meshes.new("tmp")
    sub.to_mesh(tmp)
    sub.free()
    bm.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)


def merge_into(bm, sub):
    tmp = bpy.data.meshes.new("tmp")
    sub.to_mesh(tmp)
    sub.free()
    bm.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------
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


def flake_mask(nt, scale=2600.0, density=0.12):
    """Sparse glitter flakes in object space (0..1)."""
    d = voronoi(nt, tex_coord(nt), scale)
    return map_range(nt, d, density, 0.0, 0.0, 1.0)


def lure_paint(name, back, side, belly, pattern="none", flake=0.0, chrome=False, ghost=False, bars=None,
               coat=1.0, rough=0.25):
    """Hard bait paint: belly-side-back gradient (from the zrel attribute), optional bars, glitter flake, a
    gloss clear coat; chrome sides are metallic, ghost bodies see-through with a foil core."""
    def build(nt):
        z = attr(nt, "zrel", "Fac")
        x = attr(nt, "xrel", "Fac")
        col = color_ramp(nt, z, [(0.0, belly), (0.38, side), (0.62, side), (0.86, back), (1.0, back)])
        if bars is not None:
            wob = nmath(nt, "MULTIPLY", noise(nt, tex_coord(nt), 160.0), 1.6)
            s = nmath(nt, "SINE", nmath(nt, "ADD", nmath(nt, "MULTIPLY", x, bars[1] * 6.283), wob))
            m = nmath(nt, "MULTIPLY", map_range(nt, s, 0.25, 0.65), map_range(nt, z, 0.25, 0.55))
            col = mix_rgb(nt, m, col, bars[0])
        if pattern == "scale":
            v = voronoi(nt, tex_coord(nt), 900.0, out="Distance", feature="DISTANCE_TO_EDGE")
            m = nmath(nt, "MULTIPLY", map_range(nt, v, 0.0, 0.06, 0.35, 0.0), map_range(nt, z, 0.15, 0.5))
            col = mix_rgb(nt, m, col, tuple(c * 0.55 for c in side))
        fm = flake_mask(nt) if flake else 0.0
        if flake:
            col = mix_rgb(nt, nmath(nt, "MULTIPLY", fm, flake), col, (1.0, 0.95, 0.85))
        metal = 0.0
        rgh = rough
        if chrome:                                  # metallic flank, painted back
            metal = map_range(nt, z, 0.80, 0.70, 0.0, 1.0)
            rgh = map_range(nt, z, 0.80, 0.70, rough, 0.07)
        if flake:
            metal = nmath(nt, "MAXIMUM", metal, nmath(nt, "MULTIPLY", fm, 0.9)) if not isinstance(metal, float) else \
                nmath(nt, "MULTIPLY", fm, 0.9)
        if ghost:
            alpha = map_range(nt, z, 0.92, 0.75, 0.2, 0.75)
            return psdf(nt, col, rough=0.12, metal=metal, coat=coat, alpha=alpha, spec=0.8)
        return psdf(nt, col, rough=rgh, metal=metal, coat=coat, coat_rough=0.02)
    m = new_material(name, build)
    if ghost:
        m.surface_render_method = "DITHERED"
    return m


def soft_plastic(name, color, flake_col=None, flake=0.0, translucent=0.9, belly=None):
    """Soft plastic: subsurface scattering with glitter, slightly glossy."""
    def build(nt):
        col = color
        if belly is not None:
            col = color_ramp(nt, attr(nt, "zrel", "Fac"), [(0.0, belly), (0.45, color), (1.0, tuple(c * 0.6 for c in color))])
        metal = 0.0
        if flake:
            fm = nmath(nt, "MULTIPLY", flake_mask(nt, 1800.0, 0.16), flake)
            col = mix_rgb(nt, fm, col, flake_col or (0.9, 0.8, 0.5))
            metal = nmath(nt, "MULTIPLY", fm, 0.8)
        nb = bump(nt, noise(nt, tex_coord(nt), 700.0, 3.0), 0.08)
        return psdf(nt, col, rough=0.38, metal=metal, coat=0.25, sss=translucent, sss_scale=0.004, normal=nb, spec=0.45)
    return new_material(name, build)


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


def build_materials():
    M = Materials()
    # box: tactical graphite polymer with a fine moulded grain, gold anodised accents, aluminium arms
    M.box = solid("BoxPolymer", srgb(20, 22, 25), rough=0.42, bump_scale=1400.0, bump_str=0.12)
    M.tray = solid("TrayPolymer", srgb(34, 38, 34), rough=0.36, bump_scale=1400.0, bump_str=0.08)
    M.tray_floor = solid("TrayFloor", srgb(22, 24, 27), rough=0.6, bump_scale=900.0, bump_str=0.1)
    M.accent = solid("BoxAccent", srgb(232, 142, 36), rough=0.32, coat=0.4)
    M.alu = solid("AnodisedAluminium", srgb(120, 124, 128), rough=0.28, metal=1.0, bump_scale=60.0, bump_str=0.02)
    M.steel = solid("Stainless", srgb(200, 202, 205), rough=0.18, metal=1.0)
    M.rubber = solid("Rubber", srgb(16, 16, 17), rough=0.85)
    M.hook = solid("BlackNickel", srgb(46, 48, 52), rough=0.16, metal=1.0)
    M.ring = solid("SplitRing", srgb(190, 192, 196), rough=0.14, metal=1.0)
    M.lip = new_material("ClearLip", lambda nt: psdf(nt, (0.9, 0.92, 0.95), rough=0.05, alpha=0.32, spec=1.0, coat=1.0))
    M.lip.surface_render_method = "DITHERED"
    M.eye = new_material("LureEye", lambda nt: psdf(nt, color_ramp(
        nt, nmath(nt, "LENGTH", 0.0) if False else vmath(nt, "LENGTH", vmath(nt, "SUBTRACT", tex_coord(nt, "Generated"), (0.5, 0.5, 0.5))),
        [(0.0, (0.0, 0.0, 0.0)), (0.17, (0.0, 0.0, 0.0)), (0.2, srgb(250, 196, 40)), (0.34, srgb(214, 120, 20)),
         (0.40, srgb(30, 20, 10)), (1.0, srgb(30, 20, 10))]), rough=0.08, coat=1.0))
    M.lead = solid("PaintedLead", srgb(40, 44, 48), rough=0.45, coat=0.6)
    M.blade_gold = solid("BladeGold", srgb(236, 190, 90), rough=0.06, metal=1.0)
    M.blade_silver = solid("BladeSilver", srgb(230, 232, 236), rough=0.05, metal=1.0)
    M.wire = solid("SpringWire", srgb(196, 198, 202), rough=0.2, metal=1.0)

    def skirt(nt):
        c = attr(nt, "Col", "Color")
        return psdf(nt, c, rough=0.35, sss=0.4, sss_scale=0.002, coat=0.3, spec=0.45)
    M.skirt = new_material("SiliconeSkirt", skirt)

    # hard baits (colourways named for the patterns they imitate)
    M.popper_bone = lure_paint("PopperBone", srgb(236, 226, 200), srgb(242, 236, 220), srgb(250, 246, 236), flake=0.0)
    M.popper_frog = lure_paint("PopperFrog", srgb(40, 70, 26), srgb(96, 150, 46), srgb(236, 228, 160), pattern="scale")
    M.popper_shad = lure_paint("PopperShad", srgb(40, 58, 70), srgb(196, 204, 210), srgb(246, 246, 246), chrome=True)
    M.walker_chrome = lure_paint("WalkerChrome", srgb(22, 40, 96), srgb(214, 218, 224), srgb(240, 240, 240), chrome=True)
    M.walker_black = lure_paint("WalkerBlack", srgb(14, 14, 16), srgb(30, 30, 34), srgb(236, 236, 236))
    M.walker_bone = lure_paint("WalkerBone", srgb(120, 110, 90), srgb(232, 222, 196), srgb(248, 244, 232), flake=0.4)
    M.jerk_ghost = lure_paint("JerkGhostMinnow", srgb(120, 140, 130), srgb(168, 186, 182), srgb(200, 208, 204),
                              pattern="scale", flake=0.8, ghost=True)
    M.jerk_chrome = lure_paint("JerkChromeBlue", srgb(20, 50, 120), srgb(210, 214, 220), srgb(240, 240, 240), chrome=True, pattern="scale")
    M.jerk_tiger = lure_paint("JerkFiretiger", srgb(40, 120, 40), srgb(214, 232, 40), srgb(246, 120, 30),
                              bars=(srgb(20, 30, 20), 5.0))
    M.jerk_clown = lure_paint("JerkClown", srgb(230, 40, 30), srgb(250, 214, 40), srgb(250, 230, 120), flake=0.5)
    M.spin_head = lure_paint("SpinnerHead", srgb(190, 240, 30), srgb(214, 246, 60), srgb(250, 250, 250))
    # soft plastics
    M.swim_shad = soft_plastic("SwimbaitShad", srgb(70, 90, 100), (0.95, 0.95, 1.0), 0.6, belly=srgb(200, 204, 204))
    M.swim_gold = soft_plastic("SwimbaitGoldShiner", srgb(200, 170, 90), (1.0, 0.85, 0.4), 0.7, belly=srgb(246, 240, 220))
    M.craw_pumpkin = soft_plastic("CrawGreenPumpkin", srgb(70, 78, 40), (0.2, 0.25, 0.1), 0.8)
    M.craw_brown = soft_plastic("CrawBrownOrange", srgb(96, 56, 30), (1.0, 0.45, 0.1), 0.7)
    M.worm_pumpkin = soft_plastic("WormGreenPumpkin", srgb(62, 74, 36), (0.15, 0.2, 0.08), 0.8)
    M.worm_black = soft_plastic("WormBlackBlue", srgb(16, 18, 30), (0.2, 0.35, 1.0), 0.9)
    M.worm_water = soft_plastic("WormWatermelon", srgb(70, 96, 52), (0.8, 0.1, 0.1), 0.6)
    M.cut_inner = new_material("SwimbaitCutFace", lambda nt: psdf(nt, color_ramp(nt, nmath(nt, "ABSOLUTE", nmath(
        nt, "SUBTRACT", attr(nt, "zrel", "Fac"), 0.5)), [(0.0, srgb(226, 214, 190)), (0.36, srgb(214, 200, 176)),
        (0.44, srgb(120, 130, 136)), (1.0, srgb(80, 96, 106))]), rough=0.55, sss=0.6, sss_scale=0.003, spec=0.3))
    # junk
    M.bobber_red = solid("BobberRed", srgb(200, 30, 26), rough=0.3, coat=0.8)
    M.bobber_white = solid("BobberWhite", srgb(236, 236, 230), rough=0.3, coat=0.8)
    M.rust = solid("RustySpoon", srgb(110, 60, 34), rough=0.7, metal=0.6, bump_scale=300.0, bump_str=0.4)
    M.line = new_material("TangledLine", lambda nt: psdf(nt, srgb(150, 210, 170), rough=0.2, alpha=0.75, spec=0.8))
    M.line.surface_render_method = "DITHERED"
    M.faded = solid("FadedPlastic", srgb(150, 140, 120), rough=0.65)
    # baitfish
    def shad(nt):
        z = attr(nt, "zrel", "Fac")
        col = color_ramp(nt, z, [(0.0, srgb(226, 228, 230)), (0.45, srgb(176, 184, 190)), (0.7, srgb(80, 100, 112)),
                                 (1.0, srgb(34, 48, 60))])
        v = voronoi(nt, tex_coord(nt), 700.0, out="Distance", feature="DISTANCE_TO_EDGE")
        col = mix_rgb(nt, map_range(nt, v, 0.0, 0.08, 0.25, 0.0), col, srgb(30, 40, 50))
        metal = map_range(nt, z, 0.8, 0.45, 0.0, 0.7)
        return psdf(nt, col, rough=0.28, metal=metal, coat=0.5, spec=0.6)
    M.shad = new_material("Baitfish", shad)
    M.fin = new_material("Fin", lambda nt: psdf(nt, srgb(110, 120, 126), rough=0.35, alpha=0.6, sss=0.3))
    M.fin.surface_render_method = "DITHERED"
    # set dressing
    M.cell_glow = emission_mat("CellGlow", GOLD, 1.0)          # strength comes from the object colour (keyed)
    nt = M.cell_glow.node_tree
    em = [n for n in nt.nodes if n.type == "EMISSION"][0]
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    nt.links.new(nmath(nt, "MULTIPLY", oi.outputs["Alpha"], 9.0), em.inputs["Strength"])
    M.ring_fx = emission_mat("VibrationRing", srgb(214, 255, 90), 1.4, alpha_from_object=True, blended=True)
    M.silt = emission_mat("MurkySilt", srgb(120, 108, 60), 0.6)
    M.sparkle = new_material("ClearSparkle", sparkle_shader)
    M.grid = new_material("BlueprintGrid", grid_shader)
    M.shaft = new_material("LightShaft", shaft_shader, blended=True)
    M.backdrop = new_material("StudioBackdrop", backdrop_shader)
    M.split_field = new_material("ClaritySplitField", split_field_shader)
    for m in (M.shaft, M.ring_fx):
        m.use_transparent_shadow = True
    return M


def frame_driver_value(nt, scale=1.0):
    v = nt.nodes.new("ShaderNodeValue")
    add_frame_driver(v.outputs[0], "default_value", f"frame * {scale}")
    return v.outputs[0]


def sparkle_shader(nt):
    r = attr(nt, "rand", "Fac")
    t = frame_driver_value(nt, 1.0 / FPS)
    s = nmath(nt, "SINE", nmath(nt, "MULTIPLY_ADD", t, nmath(nt, "MULTIPLY_ADD", r, 5.0, 3.0), nmath(nt, "MULTIPLY", r, 40.0)))
    k = nmath(nt, "POWER", map_range(nt, s, 0.4, 1.0, 0.0, 1.0, smooth=False), 3.0)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.85, 1.0, 1.0, 1.0)
    nt.links.new(nmath(nt, "MULTIPLY", k, 14.0), em.inputs["Strength"])
    return em.outputs[0]


def grid_shader(nt):
    """Dark blueprint panel with a fine and a coarse grid (object space, metres)."""
    co = tex_coord(nt)
    sx = separate(nt, co)

    def lines(step, width):
        gx = nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", nmath(nt, "FRACT", nmath(nt, "DIVIDE", sx["X"], step)), 0.5))
        gz = nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", nmath(nt, "FRACT", nmath(nt, "DIVIDE", sx["Z"], step)), 0.5))
        return nmath(nt, "MAXIMUM", map_range(nt, gx, 0.5 - width, 0.5, 0.0, 1.0), map_range(nt, gz, 0.5 - width, 0.5, 0.0, 1.0))
    g = nmath(nt, "ADD", nmath(nt, "MULTIPLY", lines(0.01, 0.06), 0.35), lines(0.05, 0.03))
    radial = map_range(nt, vmath(nt, "LENGTH", co), 0.1, 0.75, 1.0, 0.25)
    base = mix_rgb(nt, radial, srgb(4, 10, 16), srgb(10, 34, 48))
    col = mix_rgb(nt, nmath(nt, "MULTIPLY", g, radial), base, srgb(60, 150, 170))
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(col, em.inputs["Color"])
    em.inputs["Strength"].default_value = 1.0
    return em.outputs[0]


def shaft_shader(nt):
    """Soft volumetric-looking light shaft: brightest on its axis (facing ratio), fading along its length."""
    lw = nt.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    face = nmath(nt, "POWER", nmath(nt, "SUBTRACT", 1.0, lw.outputs["Facing"]), 2.5)
    along = separate(nt, tex_coord(nt, "Generated"))["Z"]
    fall = nmath(nt, "MULTIPLY", map_range(nt, along, 0.0, 0.35), map_range(nt, along, 1.0, 0.55))
    a = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", face, fall), 0.10)
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.93, 0.80, 1.0)
    em.inputs["Strength"].default_value = 1.0
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    add = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(a, add.inputs[0])
    nt.links.new(tr.outputs[0], add.inputs[1])
    nt.links.new(em.outputs[0], add.inputs[2])
    return add.outputs[0]


def backdrop_shader(nt):
    """Studio to nature: charcoal studio below, a warm horizon glow, soft green-gold foliage light above,
    with round light blobs that the depth of field turns into bokeh. Unlit (emission)."""
    co = tex_coord(nt)
    n = vmath(nt, "NORMALIZE", co)
    z = separate(nt, n)["Z"]
    col = color_ramp(nt, z, [(0.0, srgb(6, 7, 8)), (0.36, srgb(14, 15, 17)), (0.50, srgb(46, 42, 36)),
                             (0.58, srgb(38, 48, 40)), (0.75, srgb(20, 42, 36)), (1.0, srgb(10, 26, 28))])
    # foliage light: soft noise, only above the horizon
    fol = nmath(nt, "MULTIPLY", noise(nt, n, 3.0, 4.0, 0.6), map_range(nt, z, 0.48, 0.70))
    col = mix_rgb(nt, nmath(nt, "MULTIPLY", map_range(nt, fol, 0.45, 0.75), 0.6), col, srgb(70, 96, 60))
    # bokeh discs (cells of a 3D voronoi, gated to a sparse random subset)
    vo = nt.nodes.new("ShaderNodeTexVoronoi")
    vo.inputs["Scale"].default_value = 9.0
    _feed(nt, vo.inputs["Vector"], n)
    disc = map_range(nt, vo.outputs["Distance"], 0.20, 0.16, 0.0, 1.0)
    pick = map_range(nt, separate(nt, vo.outputs["Color"])["X"], 0.72, 0.78)
    band = map_range(nt, z, 0.46, 0.62)
    k = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", disc, pick), band)
    tint = mix_rgb(nt, separate(nt, vo.outputs["Color"])["Y"], srgb(255, 200, 120), srgb(190, 230, 150))
    col = mix_rgb(nt, nmath(nt, "MULTIPLY", k, 0.55), col, tint)
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(col, em.inputs["Color"])
    em.inputs["Strength"].default_value = 1.0
    return em.outputs[0]


def split_field_shader(nt):
    """Backdrop cylinder of the clarity set: murky olive-brown water above, clear aqua water below (z in
    object space, the set centre at 0), with drifting silt in the murky half and caustic light in the clear."""
    co = tex_coord(nt)
    z = separate(nt, co)["Z"]
    murky = mix_rgb(nt, noise(nt, co, 4.0, 3.0), srgb(34, 32, 16), srgb(62, 56, 28))
    t = frame_driver_value(nt, 0.02)
    cz = vmath(nt, "ADD", vmath(nt, "MULTIPLY", co, (14.0, 14.0, 14.0)), combine(nt, 0.0, t, t))
    caust = map_range(nt, voronoi(nt, cz, 1.0, out="Distance", feature="DISTANCE_TO_EDGE"), 0.06, 0.0, 0.0, 1.0)
    clear = mix_rgb(nt, nmath(nt, "MULTIPLY", caust, 0.3), srgb(14, 84, 96), srgb(120, 210, 210))
    col = mix_rgb(nt, map_range(nt, z, -0.06, 0.06), clear, murky)
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(col, em.inputs["Color"])
    em.inputs["Strength"].default_value = 1.0
    return em.outputs[0]


# --------------------------------------------------------------------------
# Lures. Local frame: nose towards +X, back +Z, sides +-Y, origin at the body centre.
# Slots: 0 body, 1 eye, 2 hook, 3 split ring, 4 extra (lip, skirt, blade, cut face, fin)
# --------------------------------------------------------------------------
class Lure:
    def __init__(self, ob, kind, length, half_w, extent):
        self.ob, self.kind, self.length, self.half_w = ob, kind, length, half_w
        self.extent = extent          # how far below the body centre the lowest part reaches (hooks), metres


def _lure_object(name, bm, col, M, body_mat, extra=None, attrs=None):
    return finish(name, bm, col, [body_mat, M.eye, M.hook, M.ring, extra or M.lip], attrs=attrs, recalc=False)


def make_popper(name, col, M, paint, L=0.072):
    r = 0.0125
    rings = [(L / 2 - 0.007, 0.0, 0.0, 0.0), (L / 2 - 0.005, r * 0.5, r * 0.5, 0.0), (L / 2 - 0.0015, r * 0.86, r * 0.86, 0.0),
             (L / 2, r * 0.98, r * 0.98, 0.0), (L / 2 - 0.004, r, r, 0.0)]
    for i in range(1, 15):
        u = i / 15
        rr = r * (1 - 0.62 * u ** 1.5)
        rings.append((L / 2 - 0.004 - u * (L - 0.007), rr, rr, 0.0))
    rings.append((-L / 2, 0.0, 0.0, 0.0))
    bm = bmesh.new()
    loft(bm, rings, n=28)
    eye_pair(bm, L / 2 - 0.013, r * 0.95, 0.004, 0.0034, 1)
    torus(bm, (L / 2 - 0.0035, 0, 0.0055), (0, 1, 0), 0.0016, 0.0004, n=10, m=5, mat=3)           # line tie
    for x in (0.004, -L / 2 + 0.002):
        tube(bm, [(x, 0, -r * 0.6), (x, 0, -r - 0.001)], 0.0005, mat=3)
        hang_treble(bm, x, -r - 0.0012, 0.9, 2, 3)
    ob = _lure_object(name, bm, col, M, paint, attrs=rel_attrs(rings))
    return Lure(ob, "popper", L, r, r + 0.019)


def make_walker(name, col, M, paint, L=0.105):
    rings = body_profile(L, 0.0092, 0.0092, hpeak=0.32, nose=0.45, tail=0.22, n=30)
    bm = bmesh.new()
    loft(bm, rings, n=28)
    eye_pair(bm, L / 2 - 0.014, 0.0084, 0.0025, 0.003, 1)
    torus(bm, (L / 2 + 0.0008, 0, 0.0), (0, 0, 1), 0.0015, 0.0004, n=10, m=5, mat=3)
    for x in (0.008, -L / 2 + 0.003):
        hang_treble(bm, x, -0.0082 if x > 0 else -0.004, 0.85, 2, 3)
    ob = _lure_object(name, bm, col, M, paint, attrs=rel_attrs(rings))
    return Lure(ob, "walker", L, 0.0092, 0.028)


def make_jerkbait(name, col, M, paint, L=0.11):
    rings = body_profile(L, 0.0105, 0.0062, hpeak=0.3, nose=0.4, tail=0.18, n=32, p=2.2)
    bm = bmesh.new()
    loft(bm, rings, n=28)
    eye_pair(bm, L / 2 - 0.012, 0.0056, 0.0022, 0.0029, 1)
    # clear diving lip under the chin, angled down and forward
    lip = bmesh.new()
    outline = [(0.0, 0.0045), (0.0, -0.0045), (0.016, -0.0052), (0.019, 0.0), (0.016, 0.0052)]
    plate(lip, outline, 0.0007, mat=4, axis="Z")
    rot = Matrix.Rotation(math.radians(-28), 4, "Y")
    for v in lip.verts:
        v.co = rot @ v.co + Vector((L / 2 - 0.006, 0.0, -0.004))
    merge_into(bm, lip)
    torus(bm, (L / 2 + 0.006, 0, -0.0072), (0, 1, 0), 0.0013, 0.0004, n=10, m=5, mat=3)
    for x in (0.02, -0.012, -L / 2 + 0.006):
        hang_treble(bm, x, -0.0095 if x > -0.03 else -0.006, 0.75, 2, 3)
    ob = _lure_object(name, bm, col, M, paint, attrs=rel_attrs(rings))
    return Lure(ob, "jerkbait", L + 0.012, 0.0062, 0.026)


def paddle_rings(L):
    rings = []
    body = 0.72 * L
    for i in range(29):
        u = i / 28
        x = L / 2 - u * body
        if u < 0.2:
            k = math.sin(0.5 * math.pi * u / 0.2) ** 0.5
        else:
            k = 1 - 0.78 * ((u - 0.2) / 0.8) ** 1.3
        h, w = 0.0125 * L / 0.1 * k, 0.0075 * L / 0.1 * k
        rings.append((x, max(w, 1e-4) if i else 0.0, max(h, 1e-4) if i else 0.0, 0.0))
    return rings, L / 2 - body


def make_swimbait(name, col, M, paint, L=0.10, cutaway=False):
    rings, x_tail = paddle_rings(L)
    bm = bmesh.new()
    if cutaway:
        loft_half(bm, [r for r in rings if r[1] > 0] + [(x_tail - 0.0005, 0.0006, 0.0018 * L / 0.1, 0.0)], n=16, mat=0, cap_mat=4)
    else:
        loft(bm, rings + [(x_tail - 0.0005, 0.0, 0.0, 0.0)], n=28)
    # stem and paddle tail (a disc facing backwards)
    s = L / 0.1
    stem = [(x_tail + 0.002 * s, 0.0016 * s, 0.0024 * s, 0.0), (x_tail - 0.008 * s, 0.0013 * s, 0.0018 * s, 0.0)]
    if cutaway:
        loft_half(bm, stem, n=8, mat=0, cap_mat=4)
    else:
        loft(bm, [stem[0]] + [stem[1]], n=12)
    pad = [(0.0105 * s * math.cos(a) * 0.62, 0.0105 * s * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 22)[:-1]]
    pb = bmesh.new()
    plate(pb, pad, 0.0026 * s, mat=0, axis="X", smooth=True)
    for v in pb.verts:
        v.co = Matrix.Rotation(math.radians(12), 4, "Y") @ v.co + Vector((x_tail - 0.0095 * s, 0, 0))
        if cutaway and v.co.y < 0:
            v.co.y = 0.0
    merge_into(bm, pb)
    eye_pair(bm, L / 2 - 0.011 * s, 0.0068 * s, 0.003 * s, 0.0026 * s, 1) if not cutaway else \
        ellipsoid(bm, (L / 2 - 0.011 * s, 0.0068 * s - 0.0007 * s, 0.003 * s), (0.0026 * s, 0.0012 * s, 0.0026 * s), mat=1)
    # weighted swimbait hook: eye out of the nose, shank inside the body, point out of the back
    y_h = -0.0004 if cutaway else 0.0
    e = Vector((L / 2 + 0.002 * s, y_h, 0.0045 * s))
    torus(bm, e, (0, 1, 0), 0.0016 * s, 0.00055 * s, n=10, m=5, mat=2)
    shank = [e + Vector((-0.0014 * s, 0, -0.001 * s)), Vector((L / 2 - 0.012 * s, y_h, -0.002 * s)),
             Vector((-0.018 * s, y_h, -0.003 * s))]
    bend = [Vector((-0.018 * s - 0.004 * s * math.sin(t), y_h, -0.003 * s + 0.0045 * s * (1 - math.cos(t)))) for t in np.linspace(0.3, math.pi, 7)]
    tipb = bend[-1]
    pts = shank + bend + [tipb + Vector((0.004 * s * j / 3, 0, 0.0045 * s * j / 3)) for j in (1, 2, 3)]
    tube(bm, pts, lambda u: 0.00075 * s * (1.0 if u < 0.85 else max(0.15, 1 - (u - 0.85) * 5.5)), segs=6, mat=2)
    ellipsoid(bm, (L / 2 - 0.022 * s, y_h, -0.006 * s), (0.008 * s, 0.0032 if not cutaway else 0.0016, 0.0035 * s), mat=2)  # belly weight
    ob = finish(name, bm, col, [paint, M.eye, M.hook, M.ring, M.cut_inner], attrs=rel_attrs(rings), recalc=not cutaway)
    return Lure(ob, "swimbait", L + 0.02 * s, 0.0075 * s, 0.013 * s)


def skirt_strands(bm, root, axis, n, length, flare, colors, rng, width=0.0011, seg=6, mat=4, layer=None):
    """Silicone skirt: n thin ribbons from a collar, flaring back along axis (unit vector)."""
    axis = Vector(axis).normalized()
    u, v = basis(axis)
    for i in range(n):
        a = 2 * math.pi * (i + rng.uniform(-0.3, 0.3)) / n
        d = u * math.cos(a) + v * math.sin(a)
        L = length * rng.uniform(0.85, 1.08)
        side = axis.cross(d).normalized()
        sag = Vector((0, 0, -1)) * rng.uniform(0.0, 0.25)
        pts = []
        for k in range(seg + 1):
            t = k / seg
            pts.append(Vector(root) + d * (0.0025 + flare * L * t ** 0.8) + axis * (L * t) + sag * L * t * t +
                       side * (0.0015 * math.sin(3 * t + i)))
        vs = [(bm.verts.new(p - side * width / 2), bm.verts.new(p + side * width / 2)) for p in pts]
        c = colors[i % len(colors)]
        for (a0, a1), (b0, b1) in zip(vs, vs[1:]):
            f = bm.faces.new((a0, a1, b1, b0))
            f.material_index = mat
            f.smooth = True
            if layer is not None:
                for loop in f.loops:
                    loop[layer] = (*c, 1.0)


def make_jig(name, col, M, head_mat, colors, seed, L=0.085, trailer=None):
    rng = random.Random(seed)
    bm = bmesh.new()
    lay = bm.loops.layers.color.new("Col")
    hx = L / 2 - 0.012
    ellipsoid(bm, (hx, 0, 0), (0.012, 0.0095, 0.0105), subdiv=3, mat=0)
    eye_pair(bm, hx + 0.004, 0.0094, 0.003, 0.0024, 1)
    torus(bm, (hx + 0.006, 0, 0.0112), (0, 1, 0), 0.0021, 0.0006, n=10, m=5, mat=2)      # line tie
    tube(bm, [(hx - 0.010, 0, -0.0005), (hx - 0.016, 0, -0.0004)], 0.0042, segs=12, mat=0)      # collar
    p1, tip = single_hook_bm(bm, 0.05, 0.016, s=1.25, mat=2, start=(hx - 0.004, 0, -0.001))
    # weed guard: a fan of stiff fibres from the head over the point
    for k in range(9):
        off = Vector((0, (k - 4) * 0.0007, 0))
        tube(bm, [Vector((hx - 0.004, 0, 0.004)) + off, tip + Vector((0.003, 0, 0.007)) + off * 2.0], 0.00035, segs=4, mat=2)
    skirt_strands(bm, (hx - 0.016, 0, -0.0004), (-1, 0, 0), 46, 0.046, 0.12, colors, rng, layer=lay)
    if trailer is not None:                    # a small craw-style trailer on the hook
        trailer_rings = body_profile(0.03, 0.004, 0.0055, hpeak=0.4, nose=0.6, tail=0.4, n=14)
        sub = bmesh.new()
        loft(sub, [(r[0] + p1.x - 0.006, r[1], r[2], r[3] + p1.z) + tuple(r[4:]) for r in trailer_rings], n=16, mat=5)
        merge_into(bm, sub)
    ob = finish(name, bm, col, [head_mat, M.eye, M.hook, M.ring, M.skirt], recalc=False)
    ob.data.materials.append(trailer or M.skirt)
    return Lure(ob, "jig", L + 0.02, 0.0095, 0.012)


def make_craw(name, col, M, paint, L=0.085):
    rings = []
    for i in range(25):
        u = i / 24
        x = L / 2 - u * L * 0.72
        k = math.sin(math.pi * min(1.0, u * 1.25 + 0.02)) ** 0.6
        seg_ = 1.0 - 0.07 * (0.5 + 0.5 * math.cos(u * 50.0)) * (u > 0.35)
        rings.append((x, max(1e-4, 0.0068 * k * seg_) if 0 < i < 24 else 0.0, max(1e-4, 0.0058 * k * seg_) if 0 < i < 24 else 0.0, 0.0))
    bm = bmesh.new()
    loft(bm, rings, n=20)
    # two flat claws spreading forward of the head (the body runs head +X, claws extend further +X)
    for sy in (1, -1):
        cl = body_profile(0.04, 0.0026, 0.0085, hpeak=0.55, nose=0.6, tail=0.3, n=14, p=2.4)
        sub = bmesh.new()
        loft(sub, cl, n=14, mat=0)
        m = Matrix.Translation((L / 2 + 0.012, sy * 0.008, 0)) @ Matrix.Rotation(math.radians(sy * 14), 4, "Z") @ \
            Matrix.Rotation(math.radians(90), 4, "X")
        for v in sub.verts:
            v.co = m @ v.co
        merge_into(bm, sub)
    for sy in (1, -1):                         # antennae
        pts = [Vector((L / 2 - 0.004, sy * 0.002, 0.004)) + Vector((0.012 * t, sy * 0.01 * t * t, 0.002 * t)) for t in np.linspace(0, 3, 10)]
        tube(bm, pts, lambda u: 0.0005 * (1 - 0.8 * u), segs=4, mat=0)
    ob = finish(name, bm, col, [paint, M.eye, M.hook, M.ring, M.lip], attrs=rel_attrs(rings))
    return Lure(ob, "craw", L + 0.04, 0.0068, 0.009)


def make_worm(name, col, M, paint, L=0.12, curve=0.25, seed=0):
    rng = random.Random(seed)
    pts = []
    for i in range(30):
        t = i / 29
        pts.append(Vector((L / 2 - L * t, curve * 0.02 * math.sin(math.pi * t + rng.uniform(-0.2, 0.2)), 0.0)))
    bm = bmesh.new()
    tube(bm, pts, lambda u: 0.0048 * (1 - 0.35 * u) * min(1.0, 0.4 + 6 * min(u, 1 - u)), segs=14, mat=0)
    ob = finish(name, bm, col, [paint, M.eye, M.hook, M.ring, M.lip])
    return Lure(ob, "worm", L, 0.0048, 0.005)


def make_spinnerbait(name, col, M, seed=5):
    """Spinnerbait: painted head and skirt on a single hook, R-bend wire arm, a Colorado and a willow blade on
    swivels (the blades are separate objects so they can spin)."""
    rng = random.Random(seed)
    bm = bmesh.new()
    lay = bm.loops.layers.color.new("Col")
    ellipsoid(bm, (0.0, 0, 0), (0.011, 0.007, 0.0085), subdiv=3, mat=0)
    eye_pair(bm, 0.004, 0.0068, 0.002, 0.0022, 1)
    tube(bm, [(-0.009, 0, 0), (-0.013, 0, 0)], 0.0035, segs=12, mat=0)
    single_hook_bm(bm, 0.048, 0.016, s=1.2, mat=2, start=(-0.004, 0, -0.0008))
    skirt_strands(bm, (-0.013, 0, 0), (-1, 0, 0), 50, 0.052, 0.14,
                  [srgb(214, 250, 40), srgb(250, 250, 250), srgb(214, 250, 40), srgb(40, 90, 220)], rng, layer=lay, mat=5)
    tie = Vector((0.010, 0, 0.004))
    kink = Vector((0.016, 0, 0.026))
    arm_end = Vector((-0.040, 0, 0.046))
    tube(bm, [(0.006, 0, 0.005), tie, kink + Vector((0.002, 0, -0.003)), kink, kink + Vector((-0.004, 0, 0.002)),
              (kink * 0.6 + arm_end * 0.4), arm_end], 0.0006, segs=6, mat=4)
    torus(bm, kink + Vector((0.0006, 0, -0.0008)), (0, 1, 0), 0.0016, 0.0005, n=10, m=5, mat=4)      # R-bend loop
    ob = finish(name, bm, col, [M.spin_head, M.eye, M.hook, M.ring, M.wire, M.skirt], recalc=False,
                attrs={"zrel": lambda c: min(1.0, max(0.0, (c.z + 0.0085) / 0.017)),
                       "xrel": lambda c: min(1.0, max(0.0, (c.x + 0.011) / 0.022))})
    blades = []
    arm_dir = (arm_end - kink).normalized()
    for k, (pos, kind, mat) in enumerate(((arm_end + arm_dir * 0.002, "willow", M.blade_gold),
                                          (kink * 0.45 + arm_end * 0.55, "colorado", M.blade_silver))):
        piv = empty_object(f"{name}_BladePivot{k}", col, loc=pos, parent=ob, size=0.004)
        piv.rotation_mode = "QUATERNION"
        piv.rotation_quaternion = Vector((1, 0, 0)).rotation_difference(arm_dir)
        bb = bmesh.new()
        if kind == "willow":
            Lb, Wb = 0.034, 0.0075
        else:
            Lb, Wb = 0.018, 0.0085
        outline = [(Lb * (0.5 - 0.5 * math.cos(t)), Wb * math.sin(t) * (1 - 0.3 * (0.5 - 0.5 * math.cos(t)))) for t in np.linspace(0, math.pi, 12)]
        outline += [(x, -y) for x, y in reversed(outline[1:-1])]
        grid = []
        for x, y in outline:
            cup = 0.0018 * (1 - (y / Wb) ** 2) if Wb else 0.0
            grid.append((x, y, cup))
        top = [bb.verts.new((x + 0.003, y, z + 0.0)) for x, y, z in grid]
        f = bb.faces.new(top)
        f.smooth = True
        bmesh.ops.triangulate(bb, faces=[f])
        bmesh.ops.solidify(bb, geom=list(bb.faces), thickness=0.0004)
        tilt = Matrix.Rotation(math.radians(-16 if kind == "willow" else -24), 4, "Y") @ Matrix.Rotation(math.radians(40), 4, "X")
        for v in bb.verts:
            v.co = tilt @ v.co
        torus(bb, (0.0012, 0, 0), (0, 0, 1), 0.0012, 0.0003, n=8, m=4)                           # swivel ring
        bo = finish(f"{name}_Blade{k}", bb, col, [mat], parent=piv)
        blades.append(piv)
    return Lure(ob, "spinnerbait", 0.075, 0.007, 0.012), blades


def make_shad(name, col, M, L=0.065):
    rings = body_profile(L, 0.0105, 0.0042, hpeak=0.3, nose=0.5, tail=0.12, n=28, p=2.0)
    rings = [(x, w, h, z - 0.0015 * math.sin(math.pi * (L / 2 - x) / L)) + tuple(r) for x, w, h, z, *r in rings]
    bm = bmesh.new()
    loft(bm, rings, n=24)
    eye_pair(bm, L / 2 - 0.0075, 0.0034, 0.0012, 0.0016, 1)
    tail = [(-L / 2 + 0.004, 0.0012), (-L / 2 - 0.006, 0.0045), (-L / 2 - 0.013, 0.0095), (-L / 2 - 0.0095, 0.002),
            (-L / 2 - 0.008, 0.0), (-L / 2 - 0.0095, -0.002), (-L / 2 - 0.013, -0.0095), (-L / 2 - 0.006, -0.0045),
            (-L / 2 + 0.004, -0.0012)]
    plate(bm, tail, 0.0005, mat=4)
    dorsal = [(0.003, 0.0088), (-0.002, 0.0135), (-0.004, 0.0128), (-0.009, 0.0075)]
    plate(bm, dorsal, 0.0004, mat=4)
    anal = [(-0.012, -0.0072), (-0.016, -0.0102), (-0.022, -0.0058)]
    plate(bm, anal, 0.0004, mat=4)
    for sy in (1, -1):                                   # pectoral fins
        pec = bmesh.new()
        plate(pec, [(0.0, 0.0), (-0.009, -0.003), (-0.007, 0.0012)], 0.0003, mat=4)
        for v in pec.verts:
            v.co = Vector((v.co.x + L / 2 - 0.016, sy * (0.0036 + 0.001 * abs(v.co.x) * 100 * 0.1), v.co.z - 0.004))
        merge_into(bm, pec)
    ob = finish(name, bm, col, [M.shad, M.eye, M.hook, M.ring, M.fin], attrs=rel_attrs(rings), recalc=False)
    return ob


def make_junk(name, col, M, kind, rng):
    bm = bmesh.new()
    if kind == "bobber":
        res = ellipsoid(bm, (0, 0, 0), (0.013, 0.013, 0.013), subdiv=3, mat=0)
        for f in res:
            if f.calc_center_median().z > 0:
                f.material_index = 1
        tube(bm, [(0, 0, 0.012), (0, 0, 0.022)], 0.0016, segs=8, mat=1)
        tube(bm, [(0, 0, -0.012), (0, 0, -0.017)], 0.001, segs=6, mat=1)
        ob = finish(name, bm, col, [M.bobber_red, M.bobber_white])
        return ob, 0.015
    if kind == "spoon":
        out = [(0.03 * math.cos(t), 0.011 * math.sin(t) * (1 - 0.35 * math.cos(t))) for t in np.linspace(0, 2 * math.pi, 26)[:-1]]
        vs = [bm.verts.new((x, y, 0.004 * (1 - (y / 0.012) ** 2) + 0.002 * math.sin(x * 60))) for x, y in out]
        f = bm.faces.new(vs)
        bmesh.ops.triangulate(bm, faces=[f])
        bmesh.ops.solidify(bm, geom=list(bm.faces), thickness=0.0008)
        sub = bmesh.new()
        treble_bm(sub, 0.8, mat=0)
        for v in sub.verts:
            v.co = Matrix.Rotation(math.radians(90), 4, "Y") @ v.co + Vector((-0.032, 0, 0))
        merge_into(bm, sub)
        ob = finish(name, bm, col, [M.rust])
        return ob, 0.03
    if kind == "line":
        pts = []
        p = Vector((0, 0, 0))
        d = Vector((1, 0, 0))
        for i in range(140):
            d = (d + Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.7, 0.7))) * 0.55).normalized()
            p = p + d * 0.004
            p = p * 0.97 if p.length > 0.02 else p
            pts.append(p.copy())
        tube(bm, pts, 0.00035, segs=4, mat=0)
        ob = finish(name, bm, col, [M.line])
        return ob, 0.022
    # a faded, broken old crankbait
    rings = body_profile(0.06, 0.012, 0.0075, hpeak=0.35, nose=0.5, tail=0.3, n=16)
    loft(bm, rings, n=16)
    ob = finish(name, bm, col, [M.faded])
    return ob, 0.03


# --------------------------------------------------------------------------
# The cantilever tackle box (set 1)
# --------------------------------------------------------------------------
BOX_W, BOX_D = 0.42, 0.22            # outer width (x) and depth (y) of the body and the trays
BODY_H, BIN_FLOOR = 0.12, 0.07        # body height; the bottom bin's floor (a 5 cm deep well on top)
TRAY_H, WALL = 0.045, 0.006
ARM_L = 0.23 / (2 * math.cos(math.radians(30)))      # parallel arms: 150 deg closed -> 30 deg open moves 0.23 m
ARM_ATTACH = (-0.09, -0.03)                          # arm attachment points along y on each tray (tray local)
ARM_Z = 0.03
COLS, ROWS = 3, 2


def container(bm, w, d, h, wall, floor, mat=0, floor_mat=None):
    """Open-topped box: a cube whose top face is inset and pushed down to make the well."""
    res = bmesh.ops.create_cube(bm, size=1.0)
    for v in res["verts"]:
        v.co = Vector((v.co.x * w, v.co.y * d, (v.co.z + 0.5) * h))
    top = [f for f in bm.faces if f.normal.z > 0.9 and f.calc_center_median().z > h * 0.99][0]
    bmesh.ops.inset_region(bm, faces=[top], thickness=wall, depth=0.0)
    ext = bmesh.ops.extrude_face_region(bm, geom=[top])
    bmesh.ops.delete(bm, geom=[top], context="FACES_ONLY")          # the op keeps the original face as a lid
    vs = [e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=vs, vec=(0, 0, -(h - floor)))
    for f in bm.faces:
        f.material_index = mat
    if floor_mat is not None:
        for f in bm.faces:
            c = f.calc_center_median()
            if f.normal.z > 0.9 and abs(c.z - floor) < 1e-4:
                f.material_index = floor_mat


def dividers(bm, z0, h, w_in, d_in, mat=0):
    t = 0.003
    for i in range(1, COLS):
        x = -w_in / 2 + w_in * i / COLS
        bm_box(bm, (x, 0, z0 + h / 2), (t, d_in, h))
    for j in range(1, ROWS):
        y = -d_in / 2 + d_in * j / ROWS
        bm_box(bm, (0, y, z0 + h / 2), (w_in, t, h))


def cell_centers(floor_z):
    w_in, d_in = BOX_W - 2 * WALL, BOX_D - 2 * WALL
    out = []
    for j in range(ROWS - 1, -1, -1):          # back row first
        for i in range(COLS):
            out.append(Vector((-w_in / 2 + w_in * (i + 0.5) / COLS, -d_in / 2 + d_in * (j + 0.5) / ROWS, floor_z)))
    return out


def bevel(ob, width=0.0025, segs=2):
    mod = ob.modifiers.new("Bevel", "BEVEL")
    mod.width = width
    mod.segments = segs
    mod.limit_method = "ANGLE"
    mod.harden_normals = True
    for p in ob.data.polygons:
        p.use_smooth = True
    return mod


def arm_angle(u):
    """Arm angle (deg) for opening fraction u: 150 closed .. 30 open, with a small overshoot (the snap)."""
    return 150.0 - 120.0 * back_out(u, 1.2)


def tray_offset(phi_deg):
    """Offset of a tray relative to its closed seat when its arms stand at phi."""
    p, p0 = math.radians(phi_deg), math.radians(150.0)
    return Vector((0.0, ARM_L * (math.cos(p) - math.cos(p0)), ARM_L * (math.sin(p) - math.sin(p0))))


OPEN_MID = (62, 92)          # middle tray swings out over these frames
OPEN_TOP = (68, 98)          # the top tray swings out from the middle tray
LID_OPEN = 104.0             # lid angle (deg) past vertical stays open behind the top tray


def open_u(f, span):
    return min(1.0, max(0.0, (f - span[0]) / (span[1] - span[0])))


def mid_matrix(f):
    """Middle tray relative to the box rig."""
    return Matrix.Translation(Vector((0, 0, BODY_H)) + tray_offset(arm_angle(open_u(f, OPEN_MID))))


def top_matrix(f):
    return mid_matrix(f) @ Matrix.Translation(Vector((0, 0, TRAY_H)) + tray_offset(arm_angle(open_u(f, OPEN_TOP))))


def rig_matrix(f):
    """The whole box floats: a slow bob and sway."""
    z = 0.004 * math.sin(2 * math.pi * f / 96.0)
    rx = math.radians(0.5) * math.sin(2 * math.pi * f / 130.0)
    ry = math.radians(0.4) * math.sin(2 * math.pi * f / 170.0 + 1.0)
    return Matrix.Translation((0, 0, z)) @ Euler((rx, ry, 0.0)).to_matrix().to_4x4()


def build_box(col, M):
    rig = empty_object("BoxRig", col, size=0.05)
    rig.rotation_mode = "XYZ"
    for f in range(1, FRAME_END + 1):
        m = rig_matrix(f)
        rig.location = m.translation
        rig.rotation_euler = m.to_euler()
        key(rig, "location", f)
        key(rig, "rotation_euler", f)
    # body: a solid block with a shallow bin on top, ribs, latches and feet
    bm = bmesh.new()
    container(bm, BOX_W, BOX_D, BODY_H, WALL, BIN_FLOOR, mat=0, floor_mat=1)
    for z in (0.025, 0.05):                                            # moulded ribs round the body
        bm_box(bm, (0, 0, z), (BOX_W + 0.004, BOX_D + 0.004, 0.006))
    dividers(bm, BIN_FLOOR, BODY_H - BIN_FLOOR - 0.006, BOX_W - 2 * WALL, BOX_D - 2 * WALL)
    body = finish("BoxBody", bm, col, [M.box, M.tray_floor], parent=rig)
    bevel(body, 0.003)
    bm = bmesh.new()
    for x in (-0.12, 0.12):                                            # front latches
        bm_box(bm, (x, -BOX_D / 2 - 0.004, BODY_H - 0.02), (0.05, 0.008, 0.03))
    for x in (-0.18, 0.18):
        for y in (-0.08, 0.08):
            bm_box(bm, (x, y, 0.003), (0.03, 0.03, 0.008))
    acc = finish("BoxLatches", bm, col, [M.accent], parent=rig)
    bevel(acc, 0.0025)
    # trays
    trays = {}
    for name, seat_parent in (("MidTray", body), ("TopTray", None)):
        bm = bmesh.new()
        container(bm, BOX_W, BOX_D, TRAY_H, WALL, 0.004, mat=0, floor_mat=1)
        dividers(bm, 0.004, TRAY_H - 0.009, BOX_W - 2 * WALL, BOX_D - 2 * WALL)
        bm_box(bm, (0, -BOX_D / 2 - 0.002, TRAY_H - 0.01), (0.12, 0.006, 0.012))         # finger pull
        tr = finish(name, bm, col, [M.tray, M.tray_floor], parent=rig)
        bevel(tr, 0.0022)
        tr.matrix_parent_inverse = Matrix()
        trays[name] = tr
    mid, top = trays["MidTray"], trays["TopTray"]
    for f in range(1, FRAME_END + 1):
        for ob, mf in ((mid, mid_matrix), (top, top_matrix)):
            ob.location = mf(f).translation
            key(ob, "location", f)
        if f > OPEN_TOP[1] + 4:
            break
    # parallel arms (aluminium) with rivets: body -> middle tray, middle tray -> top tray
    arms = []
    for parent, child_span, seat_z in ((body, OPEN_MID, BODY_H), (mid, OPEN_TOP, TRAY_H)):
        for ay in ARM_ATTACH:
            pivot_y = ay - ARM_L * math.cos(math.radians(150))
            pivot_z = seat_z + ARM_Z - ARM_L * math.sin(math.radians(150))
            for sx in (-1, 1):
                bm = bmesh.new()
                bm_box(bm, (ARM_L / 2, 0, 0), (ARM_L + 0.014, 0.004, 0.012))
                for x in (0.0, ARM_L):
                    bm_frustum(bm, (x, -0.003, 0), (x, 0.003, 0), 0.0035, 0.0035, segs=10)
                a = finish(f"Arm_{parent.name}_{ay}_{sx}", bm, col, [M.alu], parent=parent)
                bevel(a, 0.0012)
                # the bar is built along +X: rotate it so +X becomes +Y, then tilt it up by phi about X
                a.location = (sx * (BOX_W / 2 + 0.006), pivot_y, pivot_z)
                a.rotation_mode = "XYZ"
                for f in range(1, child_span[1] + 6):
                    phi = arm_angle(open_u(f, child_span))
                    a.rotation_euler = (math.radians(phi), 0.0, math.radians(90))
                    key(a, "rotation_euler", f)
                arms.append(a)
    # snap glow outlines, one per cell (keyed when a lure lands)
    glows = {}
    for name, parent, fz in (("Body", body, BIN_FLOOR), ("MidTray", mid, 0.004), ("TopTray", top, 0.004)):
        for k, c in enumerate(cell_centers(fz)):
            w = (BOX_W - 2 * WALL) / COLS - 0.008
            d = (BOX_D - 2 * WALL) / ROWS - 0.008
            bm = bmesh.new()
            t = 0.0016
            for (cx, cy, sx_, sy_) in ((0, d / 2, w, t), (0, -d / 2, w, t), (w / 2, 0, t, d), (-w / 2, 0, t, d)):
                bm_box(bm, (c.x + cx, c.y + cy, fz + 0.0006), (sx_, sy_, 0.0008))
            g = finish(f"Glow_{name}_{k}", bm, col, [M.cell_glow], parent=parent)
            g.visible_shadow = False
            g.color = (1, 1, 1, 0.0)
            key(g, "color", 1, index=3)
            glows[(name, k)] = g
    return dict(rig=rig, body=body, mid=mid, top=top, arms=arms, glows=glows)


def tier_matrix(tier, f):
    return {"Body": lambda f: Matrix(), "MidTray": mid_matrix, "TopTray": top_matrix}[tier](f)


TIER_FLOOR = {"Body": BIN_FLOOR, "MidTray": 0.004, "TopTray": 0.004}


def cell_pose(tier, k, lure, f):
    """Rig-local matrix of a lure lying on its side in a cell (back towards +Y, hooks towards -Y)."""
    c = cell_centers(TIER_FLOOR[tier])[k]
    lift = max(lure.half_w, 0.0045) + 0.0012
    yaw = math.radians(180) if lure.kind == "craw" else 0.0
    local = Matrix.Translation((c.x, c.y + 0.006, c.z + lift)) @ Matrix.Rotation(yaw, 4, "Z") @ \
        Matrix.Rotation(math.radians(-90), 4, "X")
    return tier_matrix(tier, f) @ local


# --------------------------------------------------------------------------
# Lure inventory, the messy pile and the snap into the grid
# --------------------------------------------------------------------------
def build_lures(col, M):
    """The 18 keepers (6 per tier) and the junk. Returns dict with lists."""
    def jigcols(a, b, c):
        return [a, b, a, c]
    surface = [make_popper("PopperBone", col, M, M.popper_bone), make_walker("WalkerChrome", col, M, M.walker_chrome),
               make_popper("PopperFrog", col, M, M.popper_frog), make_walker("WalkerBlack", col, M, M.walker_black),
               make_popper("PopperShad", col, M, M.popper_shad), make_walker("WalkerBone", col, M, M.walker_bone)]
    midcol = [make_jerkbait("JerkGhost", col, M, M.jerk_ghost), make_jerkbait("JerkChrome", col, M, M.jerk_chrome),
              make_swimbait("SwimShad", col, M, M.swim_shad), make_jerkbait("JerkFiretiger", col, M, M.jerk_tiger),
              make_jerkbait("JerkClown", col, M, M.jerk_clown), make_swimbait("SwimGold", col, M, M.swim_gold)]
    bottom = [make_jig("JigBlackBlue", col, M, M.lead, jigcols(srgb(14, 14, 18), srgb(30, 60, 200), srgb(120, 30, 150)), 11),
              make_craw("CrawPumpkin", col, M, M.craw_pumpkin),
              make_jig("JigPumpkin", col, M, M.lead, jigcols(srgb(60, 70, 34), srgb(150, 90, 30), srgb(20, 30, 14)), 12,
                       trailer=M.craw_pumpkin),
              make_worm("WormPumpkin", col, M, M.worm_pumpkin, seed=1), make_craw("CrawBrown", col, M, M.craw_brown),
              make_jig("JigBrown", col, M, M.lead, jigcols(srgb(90, 50, 26), srgb(230, 110, 30), srgb(30, 20, 14)), 13,
                       trailer=M.craw_brown)]
    extra_worms = [make_worm("WormBlack", col, M, M.worm_black, seed=2, curve=-0.3),
                   make_worm("WormWatermelon", col, M, M.worm_water, seed=3, curve=0.5)]
    rng = random.Random(21)
    junk = []
    kinds = ["bobber", "spoon", "line", "old", "bobber", "spoon", "line", "old", "bobber", "line", "old", "spoon",
             "bobber", "line", "old", "spoon"]
    for i, kind in enumerate(kinds):
        ob, r = make_junk(f"Junk_{i}_{kind}", col, M, kind, rng)
        junk.append((ob, r))
    return dict(tiers={"TopTray": surface, "MidTray": midcol, "Body": bottom}, extra_worms=extra_worms, junk=junk)


FLIGHT = 13                   # frames a lure takes from the pile to its cell
LAND = {"Body": 96, "MidTray": 112, "TopTray": 126}   # first landing per tier (bottom bin first: it is uncovered first)
LAND_STEP = 2.4
JUNK_OUT = (70, 90)


def landing_frames():
    out = {}
    for tier, f0 in LAND.items():
        for k in range(6):
            out[(tier, k)] = int(round(f0 + k * LAND_STEP))
    return out


def pile_layout(items, rng):
    """Heap the items on the closed top tray (top tray local coordinates): each one rests on the floor or
    on the items already placed (bounding spheres), with a bias towards the middle so it overflows."""
    placed = []
    for name, r in items:
        best = None
        for _ in range(40):
            x = max(-0.15, min(0.15, rng.gauss(0.0, 0.055)))
            y = max(-0.07, min(0.07, rng.gauss(0.0, 0.032)))
            z = 0.004 + r * 0.6
            for (px, py, pz, pr) in placed:
                d = math.hypot(x - px, y - py)
                rr = (r + pr) * 0.62
                if d < rr:
                    z = max(z, pz + math.sqrt(rr * rr - d * d))
            if best is None or z < best[2]:
                best = (x, y, z)
        placed.append((*best, r))
    out = []
    for (x, y, z, r) in placed:
        rot = Euler((rng.uniform(-0.9, 0.9), rng.uniform(-0.7, 0.7), rng.uniform(0, 2 * math.pi))).to_matrix().to_4x4()
        out.append(Matrix.Translation((x, y, z)) @ rot)
    return out


def key_matrix(ob, m, f):
    loc, rot, scl = m.decompose()
    ob.location = loc
    ob.rotation_quaternion = rot
    ob.scale = scl
    for p in ("location", "rotation_quaternion", "scale"):
        key(ob, p, f)


def animate_pile(box, inv, M):
    rig = box["rig"]
    rng = random.Random(5)
    keepers = [(tier, k, lu) for tier, lst in inv["tiers"].items() for k, lu in enumerate(lst)]
    items = [(lu.ob.name, max(0.018, lu.length * 0.32)) for _t, _k, lu in keepers]
    items += [(ob.name, r) for ob, r in inv["junk"]]
    order = list(range(len(items)))
    rng.shuffle(order)
    poses_sorted = pile_layout([items[i] for i in order], rng)
    poses = [None] * len(items)
    for j, i in enumerate(order):
        poses[i] = poses_sorted[j]
    lands = landing_frames()
    info = {}
    for idx, (tier, k, lu) in enumerate(keepers):
        ob = lu.ob
        ob.parent = rig
        ob.rotation_mode = "QUATERNION"
        chaos = poses[idx]
        fl = lands[(tier, k)]
        f0 = fl - FLIGHT
        for f in range(1, FRAME_END + 1):
            start = top_matrix(f) @ chaos
            end = cell_pose(tier, k, lu, f)
            if f <= f0:
                m = start
            elif f < fl:
                u = (f - f0) / FLIGHT
                e = ease(u)
                loc = start.translation.lerp(end.translation, e) + Vector((0, 0, 0.07 * math.sin(math.pi * min(1, u * 1.05))))
                q = start.to_quaternion().slerp(end.to_quaternion(), ease(min(1.0, u * 1.2)))
                m = Matrix.Translation(loc) @ q.to_matrix().to_4x4()
            else:
                d = f - fl                                  # settle: a quick damped bounce
                dz = -0.003 * math.exp(-d / 2.5) * math.cos(d * 1.6)
                m = Matrix.Translation((0, 0, dz)) @ end
            key_matrix(ob, m, f)
            if f > fl + 10 and f > OPEN_TOP[1] + 4:
                break
        info[ob.name] = dict(tier=tier, cell=k, land=fl, lure=lu)
        g = box["glows"][(tier, k)]                          # the cell flashes gold as the lure snaps in
        for fr, a in ((fl - 1, 0.0), (fl, 1.0), (fl + 3, 0.6), (fl + 12, 0.0)):
            g.color[3] = a
            key(g, "color", fr, index=3)
    # the two spare worms ride in the worm cell with the first one
    worm_tier, worm_k = "Body", 3
    for j, lu in enumerate(inv["extra_worms"]):
        ob = lu.ob
        ob.parent = rig
        ob.rotation_mode = "QUATERNION"
        fl = lands[(worm_tier, worm_k)] + 2 + j
        chaos = Matrix.Translation((0.05 * (j - 0.5), 0.03, 0.02)) @ Euler((0.4, 0.2, 1.0 + j)).to_matrix().to_4x4()
        for f in range(1, fl + 12):
            start = top_matrix(f) @ chaos
            end = cell_pose(worm_tier, worm_k, lu, f) @ Matrix.Translation((0, 0.011 * (j + 1) * (1 if j else -1), 0.0))
            u = min(1.0, max(0.0, (f - (fl - FLIGHT)) / FLIGHT))
            e = ease(u)
            loc = start.translation.lerp(end.translation, e) + Vector((0, 0, 0.06 * math.sin(math.pi * u)))
            q = start.to_quaternion().slerp(end.to_quaternion(), e)
            key_matrix(ob, Matrix.Translation(loc) @ q.to_matrix().to_4x4(), f)
    # junk: thrown out of the box, spinning and shrinking away
    for idx, (ob, r) in enumerate(inv["junk"]):
        ob.parent = rig
        ob.rotation_mode = "QUATERNION"
        chaos = poses[len(keepers) + idx]
        f0 = JUNK_OUT[0] + int(idx * 1.3)
        side = -1 if idx % 2 else 1
        for f in range(1, f0 + 18):
            start = top_matrix(f) @ chaos
            if f <= f0:
                m = start
            else:
                u = (f - f0) / 16.0
                loc = start.translation + Vector((side * 0.35 * u, -0.15 * u, 0.30 * u - 0.1 * u * u))
                q = start.to_quaternion() @ Quaternion((0.3, 1, 0.2), 9.0 * u)
                s = max(0.001, 1.0 - ease(u))
                m = Matrix.Translation(loc) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((s, s, s, 1))
            key_matrix(ob, m, f)
        ob.hide_render = False
        key(ob, "hide_render", f0 + 16)
        ob.hide_render = True
        key(ob, "hide_render", f0 + 17)
    return info


# --------------------------------------------------------------------------
# Set 3: the clarity rule. Set 4: the forage match.
# --------------------------------------------------------------------------
SPIN_Z, GHOST_Z = 0.075, -0.075


def specks(name, col, mat, center, radius, n, size, seed, flat_y=False):
    rng = random.Random(seed)
    bm = bmesh.new()
    rand = bm.verts.layers.float.new("rand")
    for _ in range(n):
        while True:
            p = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
            if p.length <= 1:
                break
        p = p * radius
        s = size * rng.uniform(0.5, 1.4)
        res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=s, matrix=Matrix.Translation(p))
        r = rng.random()
        for v in res["verts"]:
            v[rand] = r
    ob = finish(name, bm, col, [mat])
    ob.location = center
    ob.visible_shadow = False
    return ob


def build_clarity_set(col, M, cam):
    spin, blades = make_spinnerbait("ClaritySpinnerbait", col, M)
    ghost = make_jerkbait("ClarityGhost", col, M, M.jerk_ghost)
    spin.ob.location = S3 + Vector((0.0, 0.0, SPIN_Z))
    ghost.ob.location = S3 + Vector((0.0, 0.0, GHOST_Z))
    for ob, ph in ((spin.ob, 0.0), (ghost.ob, 1.7)):
        ob.rotation_mode = "XYZ"
        base = ob.location.copy()
        for f in range(CUT2 - 10, CUT3 + 10, 2):
            t = f / FPS
            ob.rotation_euler = (0.05 * math.sin(t * 1.3 + ph), 0.08 * math.sin(t * 0.9 + ph), math.radians(-12) + 0.18 * math.sin(t * 0.5 + ph))
            ob.location = base + Vector((0, 0, 0.004 * math.sin(t * 1.7 + ph)))
            key(ob, "rotation_euler", f)
            key(ob, "location", f)
    for k, piv in enumerate(blades):                    # blades spin about the arm (about 6 and 9 turns a second)
        rate = (6.0, 9.0)[k] * 2 * math.pi / FPS
        piv.rotation_mode = "XYZ"
        q0 = piv.rotation_quaternion.copy() if piv.rotation_mode == "QUATERNION" else None
        e0 = piv.rotation_euler.copy()
        piv.rotation_mode = "QUATERNION"
        base_q = piv.rotation_quaternion.copy()
        prefs = bpy.context.preferences.edit
        old = prefs.keyframe_new_interpolation_type
        prefs.keyframe_new_interpolation_type = "LINEAR"
        for f in range(CUT2 - 10, CUT3 + 10, 1):
            piv.rotation_quaternion = base_q @ Quaternion((1, 0, 0), rate * f)
            key(piv, "rotation_quaternion", f)
        prefs.keyframe_new_interpolation_type = old
    # vibration rings radiating from the bright lure (face the camera)
    rings = []
    for i in range(3):
        bm = bmesh.new()
        n = 48
        inner = [bm.verts.new((0.955 * math.cos(2 * math.pi * k / n), 0.92 * math.sin(2 * math.pi * k / n), 0)) for k in range(n)]
        outer = [bm.verts.new((math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n), 0)) for k in range(n)]
        for k in range(n):
            bm.faces.new((inner[k], inner[(k + 1) % n], outer[(k + 1) % n], outer[k]))
        rg = finish(f"VibrationRing{i}", bm, col, [M.ring_fx], recalc=False)
        rg.location = S3 + Vector((0.0, 0.0, SPIN_Z))
        rg.visible_shadow = False
        c = rg.constraints.new("TRACK_TO")
        c.target = cam
        c.track_axis = "TRACK_Z"
        c.up_axis = "UP_Y"
        rg.color = (1, 1, 1, 0)
        key(rg, "color", 1, index=3)
        rings.append(rg)
    # haze in the murky half, sparkle in the clear half, and the split backdrop
    silt = specks("MurkySilt", col, M.silt, S3 + Vector((0, 0, SPIN_Z)), 0.13, 260, 0.0011, 31)
    sparkle = specks("ClearSparkle", col, M.sparkle, S3 + Vector((0, 0, GHOST_Z)), 0.12, 120, 0.0007, 32)
    for ob, rate in ((silt, 0.05), (sparkle, -0.03)):
        ob.rotation_mode = "XYZ"
        for f in (CUT2 - 10, CUT3 + 10):
            ob.rotation_euler = (0.0, 0.0, rate * f / FPS)
            key(ob, "rotation_euler", f)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=48, radius1=1.4, radius2=1.4, depth=1.6)
    field = finish("ClarityField", bm, col, [M.split_field])
    field.location = S3
    field.visible_shadow = False
    return dict(spin=spin, ghost=ghost, blades=blades, rings=rings, field=field)


def key_rings(rings, f_lo, f_hi):
    """Expanding, fading rings every 14 frames while the murky line is on screen."""
    period = 14
    for i, rg in enumerate(rings):
        for f in range(f_lo - 2, f_hi + 16):
            ph = ((f - f_lo) + i * period / 3) % period / period
            on = f_lo <= f <= f_hi
            s = 0.018 + 0.09 * ph
            rg.scale = (s, s, s)
            rg.color[3] = (1.0 - ph) ** 1.5 * 0.3 if on else 0.0
            key(rg, "scale", f)
            key(rg, "color", f, index=3)


def fish_swim_group():
    ng, gi, go = new_group("BaitSwim", "GeometryNodeTree",
                           [("Geometry", "NodeSocketGeometry"), ("Phase", "NodeSocketFloat"),
                            ("Amp", "NodeSocketFloat"), ("Rate", "NodeSocketFloat")],
                           [("Geometry", "NodeSocketGeometry")])
    pos = separate(ng, ng.nodes.new("GeometryNodeInputPosition").outputs[0])
    t = nmath(ng, "DIVIDE", ng.nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], float(FPS))
    w = nmath(ng, "POWER", map_range(ng, pos["X"], 0.012, -0.045, 0.0, 1.0, smooth=False), 1.5)
    arg = nmath(ng, "MULTIPLY_ADD", t, gi.outputs["Rate"], nmath(ng, "MULTIPLY_ADD", pos["X"], -70.0, gi.outputs["Phase"]))
    dy = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", nmath(ng, "SINE", arg), w), gi.outputs["Amp"])
    sp = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, 0.0, dy, 0.0), sp.inputs["Offset"])
    ng.links.new(sp.outputs[0], go.inputs["Geometry"])
    return ng


SHRINK = 0.62                       # the swimbait is scaled to the baitfish length
BAIT_LEN = 0.065


def shrink_frame():
    return phrase_frame(4, 1) + 30          # "downsize" lands about 1.2 s into the second sentence


def build_forage_set(col, M):
    ng = fish_swim_group()
    fish = []
    for i, (dx, dz, ph) in enumerate(((-0.03, 0.07, 0.0), (0.035, 0.085, 2.0), (0.01, 0.045, 4.1))):
        ob = make_shad(f"Baitfish{i}", col, M, BAIT_LEN)
        ob.location = S4 + Vector((dx, 0.012 * i, dz))
        mod = ob.modifiers.new("Swim", "NODES")
        mod.node_group = ng
        vals = {"Phase": ph, "Amp": 0.0035, "Rate": 14.0}
        for item in ng.interface.items_tree:
            if getattr(item, "in_out", "") == "INPUT" and item.name in vals:
                set_mod_input(mod, item.identifier, vals[item.name])
        base = ob.location.copy()
        for f in range(CUT3 - 10, CUT4 + 10, 3):
            t = f / FPS
            ob.location = base + Vector((0.004 * math.sin(t * 0.8 + ph), 0, 0.003 * math.sin(t * 1.3 + ph)))
            key(ob, "location", f)
        fish.append(ob)
    bait = make_swimbait("CutawaySwimbait", col, M, M.swim_shad, L=0.105, cutaway=True)
    ob = bait.ob
    ob.location = S4 + Vector((0.0, 0.0, -0.045))          # nose to screen right, the cut face to the camera
    ob.rotation_euler = (math.radians(28), 0.0, 0.0)        # rolled: the outer skin shows along the back
    fs = shrink_frame()
    for f, s in ((CUT3 - 10, 1.0), (fs, 1.0)):
        ob.scale = (s, s, s)
        key(ob, "scale", f)
    for f in range(fs + 1, fs + 14):
        u = (f - fs) / 12
        s = 1.0 + (SHRINK - 1.0) * back_out(u, 2.2)
        ob.scale = (s, s, s)
        key(ob, "scale", f)
    bm = bmesh.new()
    bm_box(bm, (0, 0, 0), (1.4, 0.004, 1.8))
    grid = finish("BlueprintPanel", bm, col, [M.grid])
    grid.location = S4 + Vector((0, 0.30, 0.0))
    grid.visible_shadow = False
    return dict(fish=fish, bait=bait)


def bait_ends(f, which):
    """World nose and tail points of the baitfish (top) or the swimbait (scaled) for the dimension brackets."""
    sc = bpy.context.scene
    if which == "fish":
        c = S4 + Vector((0.01, 0.024, 0.045))
        return c + Vector((-BAIT_LEN / 2 - 0.012, 0, -0.004)), c + Vector((BAIT_LEN / 2, 0, -0.004))
    fs = shrink_frame()
    if f <= fs:
        s = 1.0
    elif f >= fs + 13:
        s = SHRINK
    else:
        s = 1.0 + (SHRINK - 1.0) * back_out((f - fs) / 12, 2.2)
    c = S4 + Vector((0.0, 0.0, -0.045))
    half = 0.5 * 0.105
    return c + Vector((-half * s - 0.03 * s, 0, -0.012 * s)), c + Vector((half * s + 0.001, 0, -0.012 * s))


# --------------------------------------------------------------------------
# World, lights, light shafts
# --------------------------------------------------------------------------
def build_world():
    """Studio to nature gradient as the world itself, so glossy lures reflect it and the depth of field
    blurs it (it counts as infinitely far)."""
    w = bpy.data.worlds.new("StudioToNature")
    bpy.context.scene.world = w
    try:
        w.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    co = nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"]
    n = vmath(nt, "NORMALIZE", co)
    z = separate(nt, n)["Z"]
    zz = map_range(nt, z, -1.0, 1.0, 0.0, 1.0, smooth=False)
    col = color_ramp(nt, zz, [(0.0, srgb(6, 7, 8)), (0.36, srgb(14, 15, 17)), (0.49, srgb(52, 46, 38)),
                              (0.56, srgb(40, 50, 40)), (0.72, srgb(20, 44, 38)), (1.0, srgb(10, 26, 28))])
    fol = nmath(nt, "MULTIPLY", noise(nt, n, 3.0, 4.0, 0.6), map_range(nt, zz, 0.48, 0.70))
    col = mix_rgb(nt, nmath(nt, "MULTIPLY", map_range(nt, fol, 0.45, 0.75), 0.6), col, srgb(74, 100, 62))
    vo = nt.nodes.new("ShaderNodeTexVoronoi")
    vo.inputs["Scale"].default_value = 7.0
    _feed(nt, vo.inputs["Vector"], n)
    disc = map_range(nt, vo.outputs["Distance"], 0.22, 0.17, 0.0, 1.0)
    pick = map_range(nt, separate(nt, vo.outputs["Color"])["X"], 0.70, 0.76)
    band = map_range(nt, zz, 0.47, 0.62)
    k = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", disc, pick), band)
    tint = mix_rgb(nt, separate(nt, vo.outputs["Color"])["Y"], srgb(255, 200, 120), srgb(190, 236, 150))
    col = mix_rgb(nt, nmath(nt, "MULTIPLY", k, 0.7), col, tint)
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(col, bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    return w


def area_light(name, col, loc, target, size, energy, color=(1, 1, 1), size_y=None, shadow=True):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "RECTANGLE"
    ld.size = size
    ld.size_y = size_y or size
    ld.energy = energy
    ld.color = color
    ld.use_shadow = shadow
    ob = bpy.data.objects.new(name, ld)
    col.objects.link(ob)
    ob.location = loc
    d = (Vector(target) - Vector(loc)).normalized()
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = d.to_track_quat("-Z", "Y")
    return ob


def build_lights(col):
    c = Vector((0, 0.22, 0.14))
    area_light("BoxKey", col, (-0.6, -0.55, 1.0), c, 0.7, 80.0, (1.0, 0.95, 0.88))
    area_light("BoxRim", col, (0.7, 1.0, 0.7), c, 0.5, 90.0, (0.80, 0.90, 1.0))
    area_light("BoxFill", col, (0.8, -0.7, 0.3), c, 0.9, 8.0, (1.0, 0.98, 0.95), shadow=False)
    area_light("BoxTop", col, (0.0, 0.25, 1.4), c, 1.0, 14.0, (1.0, 0.98, 0.95))
    area_light("ClarityKey", col, S3 + Vector((-0.45, -0.5, 0.45)), S3, 0.5, 22.0, (1.0, 0.95, 0.9))
    area_light("ClarityRim", col, S3 + Vector((0.45, 0.5, 0.35)), S3, 0.4, 25.0, (0.85, 0.95, 1.0))
    area_light("ClarityUnder", col, S3 + Vector((0.0, -0.3, -0.55)), S3 + Vector((0, 0, GHOST_Z)), 0.5, 5.0, (0.6, 1.0, 0.95), shadow=False)
    area_light("ForageKey", col, S4 + Vector((-0.3, -0.6, 0.45)), S4, 0.6, 15.0, (1.0, 0.96, 0.92))
    area_light("ForageRim", col, S4 + Vector((0.4, 0.4, 0.3)), S4, 0.4, 18.0, (0.8, 0.92, 1.0))


def build_shafts(col, M):
    """Two soft light shafts falling from upper left behind the box (studio haze), fake volumetrics."""
    out = []
    for i, (top, bot, r0, r1) in enumerate((((-0.55, 0.75, 1.3), (0.05, 0.25, 0.05), 0.05, 0.22),
                                            ((-0.25, 0.95, 1.35), (0.25, 0.45, 0.02), 0.04, 0.16))):
        bm = bmesh.new()
        bm_frustum(bm, bot, top, r1, r0, segs=24, cap0=False, cap1=False)
        ob = finish(f"LightShaft{i}", bm, col, [M.shaft], recalc=False)
        ob.visible_shadow = False
        out.append(ob)
    return out


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------
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


def ghost_cell(f):
    return rig_matrix(f) @ cell_pose("MidTray", 0, Lure(None, "jerkbait", 0.12, 0.0062, 0.026), f).translation


def camera_frames():
    """frames[f] = (position, target, rotation); also f-stop per frame (depth of field)."""
    frames, fstop = {}, {}
    Z, Y = Vector((0, 0, 1)), Vector((0, 1, 0))
    # 1 hook: macro push into the messy pile, then up and back as the grid snaps together
    k1 = [(1, (0.12, -0.27, 0.40), (0.0, 0.0, 0.26), True),
          (58, (0.06, -0.165, 0.335), (0.0, 0.01, 0.265), False),
          (78, (0.06, -0.20, 0.36), (0.0, 0.07, 0.235), False),
          (118, (0.05, -0.43, 0.80), (0.0, 0.20, 0.12), False),
          (CUT1 - 1, (0.03, -0.38, 0.93), (0.0, 0.24, 0.12), True)]
    for f, (p, t) in hermite_path(k1).items():
        frames[f] = (p, t, look_at(p, t, Z))
        fstop[f] = 7.0
    # 2 the layout: top-down dolly, top tray -> middle tray -> bottom bin, synced to the voiceover
    a, b, c = phrase_frame(2, TOP_TRAY), phrase_frame(2, MID_TRAY), phrase_frame(2, BOTTOM_BIN)
    k2 = [(CUT1, (0.0, 0.21, 1.12), (0.0, 0.24, 0.10), True),
          (CUT1 + 60, (0.0, 0.235, 1.02), (0.0, 0.24, 0.10), False),
          (a - 4, (0.035, 0.455, 0.80), (0.035, 0.46, 0.21), True),
          (b - 8, (-0.035, 0.445, 0.76), (-0.035, 0.46, 0.21), True),
          (b + 14, (0.035, 0.235, 0.74), (0.035, 0.23, 0.165), True),
          (c - 8, (-0.035, 0.225, 0.72), (-0.035, 0.23, 0.165), True),
          (c + 14, (0.035, 0.005, 0.70), (0.035, 0.0, 0.07), True),
          (CUT2 - 1, (-0.02, 0.0, 0.72), (-0.02, 0.01, 0.07), True)]
    for f, (p, t) in hermite_path(k2).items():
        frames[f] = (p, t, look_at(p, t, Y))
        fstop[f] = 11.0
    # 3 the clarity rule: orbit (angle, radius, height, target height) around the two lures
    m1, m2 = phrase_frame(3, MURKY), phrase_frame(3, CLEAR)
    k3 = [(CUT2, (-55.0, 0.42, 0.05, 0.0), None, True),
          (m1 - 6, (-18.0, 0.37, 0.035, 0.0), None, False),
          (m1 + 20, (2.0, 0.25, 0.085, SPIN_Z), None, False),
          (m2 - 10, (34.0, 0.235, 0.08, SPIN_Z - 0.002), None, False),
          (m2 + 18, (58.0, 0.25, -0.05, GHOST_Z + 0.004), None, False),
          (CUT3 - 1, (112.0, 0.225, -0.06, GHOST_Z), None, True)]
    pol = hermite_path([(f, v[:3], (0, 0, 0), h) for f, v, _x, h in k3])
    tz_path = hermite_path([(f, (v[3], 0, 0), (0, 0, 0), h) for f, v, _x, h in k3])
    for f, (v, _t) in pol.items():
        th, R, h = math.radians(v.x), v.y, v.z
        tzv = tz_path[f][0].x
        p = S3 + Vector((R * math.sin(th), -R * math.cos(th), h))
        t = S3 + Vector((0, 0, tzv))
        frames[f] = (p, t, look_at(p, t, Z))
        fstop[f] = 5.6
    # 4 the forage match: side-on cross-section with a slow push in
    k4 = [(CUT3, S4 + Vector((0.02, -0.40, -0.006)), S4 + Vector((0.0, 0.0, -0.008)), True),
          (shrink_frame() - 6, S4 + Vector((0.0, -0.345, -0.008)), S4 + Vector((0.0, 0.0, -0.01)), False),
          (CUT4 - 1, S4 + Vector((-0.012, -0.315, -0.01)), S4 + Vector((-0.004, 0.0, -0.012)), True)]
    for f, (p, t) in hermite_path(k4).items():
        frames[f] = (p, t, look_at(p, t, Z))
        fstop[f] = 9.0
    # 5 outro: from the ghost jerkbait in its cell, a rapid pull-back to the whole organised box
    g = ghost_cell(CUT4)
    p0, t0 = g + Vector((0.0, -0.06, 0.13)), g + Vector((0.0, 0.0, 0.0))
    p1, t1 = Vector((0.40, -0.86, 0.95)), Vector((0.0, 0.22, 0.13))
    p2, t2 = Vector((0.46, -0.98, 1.03)), Vector((0.0, 0.23, 0.13))
    fq = CUT4 + 30
    for f in range(CUT4, FRAME_END + 1):
        if f <= fq:
            u = ease_out((f - CUT4) / (fq - CUT4), 4.0)
            p, t = p0.lerp(p1, u), t0.lerp(t1, u)
            up = Y.lerp(Z, ease(min(1.0, u * 1.3))).normalized()
        else:
            u = ease((f - fq) / (FRAME_END - fq))
            p, t = p1.lerp(p2, u), t1.lerp(t2, u)
            up = Z
        frames[f] = (p, t, look_at(p, t, up))
        fstop[f] = 8.0
    return frames, fstop


def build_camera(col, frames, fstop):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = LENS
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = SENSOR
    cd.clip_start = 0.01
    cd.clip_end = 200.0
    cd.dof.use_dof = True
    cd.dof.aperture_blades = 7
    cam = bpy.data.objects.new("Camera", cd)
    col.objects.link(cam)
    cam.rotation_mode = "QUATERNION"
    for f, (p, t, q) in frames.items():
        cam.location = p
        cam.rotation_quaternion = q
        key(cam, "location", f)
        key(cam, "rotation_quaternion", f)
        cd.dof.focus_distance = (t - p).length
        cd.dof.aperture_fstop = fstop[f]
        cd.dof.keyframe_insert("focus_distance", frame=f)
        cd.dof.keyframe_insert("aperture_fstop", frame=f)
    bpy.context.scene.camera = cam
    return cam


# --------------------------------------------------------------------------
# Overlay scene and compositing
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------
def P(clip, i):
    return phrase_frame(clip, i)


# sentence indices inside the clips (sentences split at . ? and :)
TOP_TRAY, MID_TRAY, BOTTOM_BIN = 3, 5, 7      # clip 2: "Top tray:", "Middle:", "Bottom:"
MURKY, CLEAR = 2, 4                           # clip 3: "Murky water?", "Crystal clear?"


def build_overlays(hud, box, clarity):
    hud.scrims()
    hud.kicker("THE FISHING SERIES")
    # 1 hook
    p1 = P(1, 1)
    hud.serif("Stop carrying a *50-pound* tackle box full of junk.", -8, p1 - 9, y=-0.30, size=0.054,
              lines=["Stop carrying a *50-pound*", "tackle box full of junk."], reveal=2.4)
    hud.serif("Here's how to pack like a *pro.*", p1, CUT1 - 7, y=-0.30, size=0.056, reveal=3.0)
    # 2 the layout
    hud.badge(1, "THE CORE LAYOUT", "organize by water column, not brand", max(CUT1 + 10, P(2, 1) - 6), CUT2 - 8)
    hud.serif("First, ditch the *chaos.*", max(CUT1 + 8, P(2, 0)), P(2, 1) - 9)
    hud.serif("Organize by *water column,* not brand.", P(2, 1), P(2, TOP_TRAY) - 9, src="ANGLER TECHNIQUE",
              lines=["Organize by *water column,*", "not by brand."])
    hud.serif("Top tray: *surface topwaters.*", P(2, TOP_TRAY), P(2, MID_TRAY) - 9, src="FOR LOW LIGHT FEEDING",
              lines=["Top tray:", "*surface topwaters.*"])
    hud.serif("Middle: *suspending jerkbaits.*", P(2, MID_TRAY), P(2, BOTTOM_BIN) - 9, src="TO COVER GROUND",
              lines=["Middle:", "*suspending jerkbaits.*"])
    hud.serif("Bottom: *heavy jigs.*", P(2, BOTTOM_BIN), CUT2 - 8, src="TO WORK THE BOTTOM",
              lines=["Bottom:", "*heavy jigs.*"])

    def tray_pt(mf, y=0.0, z=0.03):
        return lambda f: rig_matrix(f) @ mf(f) @ Vector((0.0, y, z))
    hud.pin(tray_pt(top_matrix, -0.06), "surface", P(2, TOP_TRAY) + 4, P(2, MID_TRAY) - 6, dot_color=GOLD)
    hud.pin(tray_pt(mid_matrix, -0.06), "mid-column", P(2, MID_TRAY) + 10, P(2, BOTTOM_BIN) - 6, dot_color=GOLD)
    hud.pin(tray_pt(lambda f: Matrix.Translation((0, 0, BIN_FLOOR - 0.03)), -0.06), "bottom", P(2, BOTTOM_BIN) + 10,
            CUT2 - 10, dot_color=GOLD)
    # 3 the clarity rule
    hud.badge(2, "THE CLARITY RULE", "let water clarity pick the color", CUT2 + 8, CUT3 - 8)
    hud.serif("One golden rule: *water clarity.*", CUT2 + 10, P(3, MURKY) - 9, lines=["One golden rule:", "*water clarity.*"])
    hud.serif("Murky water? *High contrast,* heavy vibration.", P(3, MURKY), P(3, CLEAR) - 9,
              src="SOURCE: MINNESOTA DNR", lines=["Murky water? *High contrast,*", "heavy vibration."])
    hud.serif("Crystal clear? *Natural ghost patterns.*", P(3, CLEAR), CUT3 - 8, src="ANGLER TECHNIQUE",
              lines=["Crystal clear?", "*Natural ghost patterns.*"])
    hud.pin(S3 + Vector((0.0, 0.0, SPIN_Z + 0.02)), "bright + loud", P(3, MURKY) + 8, P(3, CLEAR) - 8,
            dot_color=srgb(214, 255, 90), y_band=(-0.20, 0.30))
    hud.pin(S3 + Vector((0.0, 0.0, GHOST_Z + 0.012)), "natural + subtle", P(3, CLEAR) + 10, CUT3 - 12,
            dot_color=srgb(150, 236, 230), y_band=(-0.24, 0.30))
    # 4 the forage match
    hud.badge(3, "THE FORAGE MATCH", "match lure size to the local bait", CUT3 + 8, CUT4 - 8)
    hud.serif("Match your profile to the *local forage.*", max(CUT3 + 10, P(4, 0)), P(4, 1) - 9, src="ANGLER TECHNIQUE",
              lines=["Match your profile", "to the *local forage.*"])
    hud.serif("Small baitfish? *Downsize.*", P(4, 1), CUT4 - 8, lines=["Small baitfish?", "*Downsize.*"])
    hud.dimension(lambda f: bait_ends(f, "fish")[0], lambda f: bait_ends(f, "fish")[1], "BAITFISH",
                  CUT3 + 24, CUT4 - 10, drop=0.03)
    hud.dimension(lambda f: bait_ends(f, "lure")[0], lambda f: bait_ends(f, "lure")[1], "LURE",
                  CUT3 + 36, CUT4 - 10, drop=0.04)
    fs = shrink_frame()
    hud.pin(S4 + Vector((0.0, -0.006, -0.03)), "matched", fs + 16, CUT4 - 12, dot_color=GOLD, rise=0.05, side=0.10,
            y_band=(-0.3, 0.3))
    # 5 outro: recap, the final card and the call to action
    hud.recap([(1, "THE CORE LAYOUT"), (2, "THE CLARITY RULE"), (3, "THE FORAGE MATCH")], CUT4 + 8, P(5, 1) - 10,
              gap=8)
    hud.serif("Lightweight, tactical, *ready for anything.*", CUT4 + 12, FRAME_END + 10, y=-0.30, size=0.05,
              lines=["Lightweight, tactical,", "*ready for anything.*"], reveal=3.0)
    hud.cta("SUBSCRIBE FOR MORE MASTERCLASSES", P(5, 1), FRAME_END + 10, y=0.25)
    for a, b in WIPES:
        hud.wipe(a, b)


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


# --------------------------------------------------------------------------
# Render settings
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


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def box_aabbs(f):
    """Rig-local boxes (min, max) of the body, the trays and the standing lid at frame f."""
    hw, hd = BOX_W / 2 + 0.012, BOX_D / 2 + 0.006
    out = [((-hw, -hd, 0.0), (hw, hd, BODY_H))]
    for mf in (mid_matrix, top_matrix):
        o = mf(f).translation
        out.append(((-hw, o.y - hd, o.z), (hw, o.y + hd, o.z + TRAY_H)))
    return out


def aabb_dist(p, lo, hi):
    d = [max(lo[i] - p[i], 0.0, p[i] - hi[i]) for i in range(3)]
    inside = all(lo[i] <= p[i] <= hi[i] for i in range(3))
    return -1.0 if inside else math.sqrt(sum(x * x for x in d))


def run_checks(frames, st):
    sc = bpy.data.scenes["Main"]
    print("\n=== CAMERA REPORT ===")
    worst = {"box": (1e9, 0), "lure": (1e9, 0, ""), "set3": (1e9, 0, ""), "set4": (1e9, 0, "")}
    for f in range(1, FRAME_END + 1):
        p = frames[f][0]
        seg = segment_of(f)
        if seg in (0, 1, 4):
            pl = rig_matrix(f).inverted() @ p
            d = min(aabb_dist(pl, lo, hi) for lo, hi in box_aabbs(f))
            if d < worst["box"][0]:
                worst["box"] = (d, f)
    movers = [lu.ob for lst in st["inv"]["tiers"].values() for lu in lst]
    for f in range(1, FRAME_END + 1, 2):
        seg = segment_of(f)
        p = frames[f][0]
        sc.frame_set(f)
        if seg in (0, 1, 4):
            for ob in movers:
                d = (ob.matrix_world.translation - p).length - 0.065
                if d < worst["lure"][0]:
                    worst["lure"] = (d, f, ob.name)
        elif seg == 2:
            for nm, z in (("spinnerbait", SPIN_Z), ("ghost", GHOST_Z)):
                d = (S3 + Vector((0, 0, z)) - p).length - 0.065
                if d < worst["set3"][0]:
                    worst["set3"] = (d, f, nm)
        else:
            for ob in st["forage"]["fish"] + [st["forage"]["bait"].ob]:
                d = (ob.matrix_world.translation - p).length - 0.06
                if d < worst["set4"][0]:
                    worst["set4"] = (d, f, ob.name)
    for k, v in worst.items():
        print(f"closest {k}: {tuple(round(x, 3) if isinstance(x, float) else x for x in v)} (negative = inside)")
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
    hud = st["hud"]
    sc = bpy.data.scenes["Main"]
    subj = {1: ("pile centre", lambda f: rig_matrix(f) @ top_matrix(f) @ Vector((0, 0, 0.06))),
            100: ("bottom bin", lambda f: rig_matrix(f) @ Vector((0, 0, BIN_FLOOR))),
            264: ("top tray", lambda f: rig_matrix(f) @ top_matrix(f) @ Vector((0, 0, 0.02))),
            400: ("middle tray", lambda f: rig_matrix(f) @ mid_matrix(f) @ Vector((0, 0, 0.02))),
            528: ("bottom bin", lambda f: rig_matrix(f) @ Vector((0, 0, BIN_FLOOR))),
            700: ("spinnerbait", lambda f: S3 + Vector((0, 0, SPIN_Z))),
            960: ("ghost jerkbait", lambda f: S3 + Vector((0, 0, GHOST_Z))),
            1100: ("swimbait", lambda f: S4 + Vector((0, 0, -0.045))),
            1320: ("box centre", lambda f: rig_matrix(f) @ Vector((0, 0.22, 0.12)))}
    print("\n=== SUBJECTS ON SCREEN (x in +-0.281, y in +-0.5) ===")
    for f, (nm, fn) in sorted(subj.items()):
        sc.frame_set(f)
        s = hud.project(f, fn(f))
        print(f"frame {f}: {nm} ->", None if s is None else (round(s.x, 3), round(s.y, 3)))


def write_events(path, st):
    data = {
        "fps": FPS, "frames": FRAME_END,
        "wipes": WIPES, "cuts": [s[0] for s in SEGMENTS[1:]],
        "overlays": sorted(st["hud"].events),
        "snaps": sorted(v["land"] for v in st["info"].values()),
        "junk": [JUNK_OUT[0] + int(1.3 * i) for i in range(len(st["inv"]["junk"]))],
        "trays": [list(OPEN_MID), list(OPEN_TOP)],
        "spin": [CUT2, CUT3 - 1],
        "rings": [P(3, MURKY), P(3, CLEAR) - 4],
        "shrink": shrink_frame(),
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
    c_box, c_lure, c_sets = collection("TackleBox"), collection("Lures"), collection("Sets")
    c_fx, c_cam = collection("LightsAndHaze"), collection("Camera")
    M = build_materials()
    build_world()
    build_lights(c_fx)
    frames, fstop = camera_frames()
    cam = build_camera(c_cam, frames, fstop)
    box = build_box(c_box, M)
    inv = build_lures(c_lure, M)
    info = animate_pile(box, inv, M)
    shafts = build_shafts(c_fx, M)
    clarity = build_clarity_set(c_sets, M, cam)
    key_rings(clarity["rings"], P(3, MURKY), P(3, CLEAR) - 4)
    forage = build_forage_set(c_sets, M)
    # each set renders only in its own shots (plus the frames under the wipes)
    box_objs = [o for o in c_box.all_objects] + [o for o in c_lure.all_objects] + shafts
    box_objs += [o for o in c_fx.objects if o.name.startswith("Box")]
    set3 = [o for o in c_sets.all_objects if o.name.startswith(("Clarity", "Vibration", "MurkySilt", "ClearSparkle"))]
    set3 += [o for o in c_fx.objects if o.name.startswith("Clarity")]
    set4 = [o for o in c_sets.all_objects if o.name.startswith(("Baitfish", "CutawaySwimbait", "BlueprintPanel"))]
    set4 += [o for o in c_fx.objects if o.name.startswith("Forage")]
    key_visibility(box_objs, [(1, CUT2 + 2), (CUT4 - 3, FRAME_END)])
    key_visibility(set3, [(CUT2 - 3, CUT3 + 2)])
    key_visibility(set4, [(CUT3 - 3, CUT4 + 2)])
    hs, hcam, hcol = build_hud_scene()
    hud_materials(M)
    hud = Hud(hcam, hcol, M, frames)
    build_overlays(hud, box, clarity)
    n = hide_when_invisible([o for o in bpy.data.objects["HUD"].children_recursive if o.type in ("MESH", "FONT")])
    print("overlay objects switched off while invisible:", n)
    build_compositor(sc, hs)
    bpy.context.window_manager  # noqa: B018 (keep the main scene active)
    sc.frame_set(1)
    return dict(frames=frames, hud=hud, box=box, inv=inv, info=info, clarity=clarity, forage=forage)


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
    if bpy.context.window is not None:
        bpy.context.window.scene = sc
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
            bpy.ops.render.render(write_still=True, scene=sc.name)
            print("rendered", sc.render.filepath, flush=True)


if __name__ == "__main__":
    main()
