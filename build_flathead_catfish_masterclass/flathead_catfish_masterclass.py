"""Flathead catfish masterclass: apex river predator breakdown. Vertical 9:16, 24 fps, 55 s (1320 frames),
Blender 5.2 EEVEE.

Run with the Blender binary:
    blender --background --python flathead_catfish_masterclass.py -- [options]
or with the bpy Python module (pip install bpy):
    python flathead_catfish_masterclass.py [options]

Options:
    --save PATH        save the generated scene as a .blend file (fonts packed)
    --check            camera clearances (terrain, logjam, fish), camera acceleration and turn rate, cuts and water
                       surface crossings against wipes, overlay overlaps and off-screen boxes, subjects on screen
    --frames 1,300     render these frames as PNG stills into --outdir
    --outdir DIR       output directory for renders
    --scale PCT        resolution percentage (default 100 -> 1080x1920)
    --samples N        EEVEE render samples (default 32)
    --no-blur          switch motion blur off (quick tests)
    --events PATH      write overlay, action and voiceover timing as JSON (input for flathead_audio.py)

Everything is procedural except the angler's body (Cesium Man, CC BY 4.0, see assets/CREDITS.md). Facts in the
on-screen text and the voiceover are listed with their sources in SOURCES.md; the 40+ lb fish, the depths and
the distances are illustrative.

World (metres, +Z up, river surface z = 0). A river runs along +Y and bends; at the outside of the bend the
current has scoured a hole about 6.1 m (20 ft) deep under an undercut clay bank. A logjam with root balls fills
the hole, and the big flathead rests inside it. Tea-coloured, sediment-laden water at dusk. Above the water:
wooded banks, a dusk sky, and the angler in a boat anchored upstream of the hole.

Overlays live in a second scene ("HUD") with its own static camera, composited over the 3D render, so the
lens changes (the dolly zoom) and depth of field never touch the text.

Beats follow the voiceover clips (voiceover/vo1..vo5); cuts sit at the end of each clip:
    1  hook: a dolly zoom through the logjam onto the flathead's broad head and wide jaw
    2  habitat: an aerial pan down the river bend, a wipe under the surface, and a descent into the scour hole
    3  bait and rig: macro on a live bluegill on a circle hook and slip sinker rig, kicking near the wood
    4  the strike: the flathead leaves the wood and inhales the bait; a wipe to the surface as the rod loads
    5  outro: a rapid pull-back from the bent rod to the angler and the boat at dusk, recap and CTA
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

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
FPS = 24
SHUTTER = 0.5
FRAME_END = 1320
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

# water: tea coloured, sediment laden river at dusk (fog colour by view direction, down .. up)
UW_RAMP = [(0.0, srgb(14, 10, 4)), (0.38, srgb(44, 32, 12)), (0.52, srgb(84, 60, 24)), (0.72, srgb(130, 96, 44)),
           (1.0, srgb(190, 150, 80))]
VIS = 8.0                            # fog e-folding distance under water (m): stained river, still readable
AIR_HAZE = 700.0                     # air haze e-folding distance above water (m)
SUN_DIR = Vector((-0.55, 0.80, 0.16)).normalized()     # towards the low dusk sun (about 9 degrees up)
SET2 = Vector((0.0, -24.0, 0.0))     # the boat, anchored upstream of the hole

# --------------------------------------------------------------------------
# Voiceover timing. VO_LEN defaults are estimates; vo_timing.json (written by record_bass_audio.py --measure)
# replaces them with the measured lengths and sentence starts. Clips run back to back with a short gap and
# each cut sits at the end of a clip.
# --------------------------------------------------------------------------
VO_LEN = {1: 6.0, 2: 13.6, 3: 14.6, 4: 10.4, 5: 5.0}
VO_GAP = 2
VO_TEXT = {
    1: "Stop throwing dead stink bait on the bottom if you want flatheads. You're catching channel cats, not river "
       "monsters.",
    2: "Flathead catfish aren't scavengers. They are apex solitary predators. During the day, they lock themselves "
       "deep inside heavy cover: root balls, logjams, and undercut clay banks in deep river scour holes.",
    3: "When the sun goes down, they leave the wood to hunt. To catch them, you must use live, struggling bait. Rig a "
       "six to eight inch live bluegill or bullhead on an eight aught circle hook with a heavy slip sinker rig.",
    4: "The vibrations drive them wild. When a flathead hits, let the rod load completely before locking down your "
       "drag. Their massive, bony mouth requires maximum pressure.",
    5: "Ditch the dead bait, target heavy wood, and hit subscribe for more trophy breakdown guides.",
}


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
VO_START = {}
_f = 2
for _k in range(1, 6):
    VO_START[_k] = _f
    _f += int(round(VO_LEN[_k] * FPS)) + VO_GAP
if _f - VO_GAP > FRAME_END:
    print(f"WARNING: the voiceover runs to frame {_f - VO_GAP}, past the end ({FRAME_END})")


def vo_end(clip):
    return VO_START[clip] + int(round(VO_LEN[clip] * FPS))


def phrase_frame(clip, i):
    """Frame where sentence i of a clip starts (sentences split after . ? and :), measured or estimated."""
    if clip in VO_PHRASES and i < len(VO_PHRASES[clip]):
        return VO_START[clip] + int(round(VO_PHRASES[clip][i] * FPS))
    parts = [p for p in VO_TEXT[clip].replace("?", "?|").replace(".", ".|").replace(":", ":|").split("|") if p.strip()]
    w = [len(p.strip()) + 8 for p in parts]
    return VO_START[clip] + int(round(VO_LEN[clip] * sum(w[:i]) / sum(w) * FPS))


CUT1, CUT2, CUT3, CUT4 = (min(vo_end(k) + 1, VO_START[k + 1]) for k in (1, 2, 3, 4))
SEGMENTS = [(1, CUT1 - 1), (CUT1, CUT2 - 1), (CUT2, CUT3 - 1), (CUT3, CUT4 - 1), (CUT4, FRAME_END)]
DIVE = phrase_frame(2, 2)            # wipe under the surface: "During the day, they lock themselves ..."
STRIKE = phrase_frame(4, 1) + 8      # she takes the bait on "When a flathead hits"
SURF = STRIKE + 44                   # wipe to the surface as the rod loads
WIPES = sorted([(c - 4, c + 3) for c in (CUT1, CUT2, CUT3, DIVE, SURF)])   # beat 4 runs into beat 5 without a cut


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
# Shading: fog, caustics, depth tint (scene properties "under" and "gloom" are keyed per frame)
# --------------------------------------------------------------------------
def scene_attr(nt, name):
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_type = "VIEW_LAYER"
    a.attribute_name = name
    return a.outputs["Fac"]


def uw_color(nt, z_dir, gloom):
    """Underwater fog colour by view direction (bright towards the surface), darkened by the gloom of the hook."""
    c = color_ramp(nt, nmath(nt, "MULTIPLY_ADD", z_dir, 0.5, 0.5, clamp=True), UW_RAMP)
    return vmath(nt, "SCALE", c, scale=nmath(nt, "MULTIPLY_ADD", gloom, -0.82, 1.0))


def build_atmos_group():
    """Blue-green fog by distance when the lens is under water, a faint air haze above it."""
    ng, gi, go = new_group("Atmosphere", "ShaderNodeTree", [("Shader", "NodeSocketShader")],
                           [("Shader", "NodeSocketShader")])
    geo = ng.nodes.new("ShaderNodeNewGeometry")
    dist = ng.nodes.new("ShaderNodeCameraData").outputs["View Distance"]
    inc = separate(ng, geo.outputs["Incoming"])
    under, gloom = scene_attr(ng, "under"), scene_attr(ng, "gloom")
    dens = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY_ADD", gloom, 0.7, 1.0), -1.0 / VIS)
    fog = nmath(ng, "MULTIPLY", under, nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(ng, "MULTIPLY", dist, dens))))
    haze = nmath(ng, "MULTIPLY", nmath(ng, "SUBTRACT", 1.0, under),
                 nmath(ng, "SUBTRACT", 1.0, nmath(ng, "EXPONENT", nmath(ng, "MULTIPLY", dist, -1.0 / AIR_HAZE))))
    em_f = ng.nodes.new("ShaderNodeEmission")
    ng.links.new(uw_color(ng, nmath(ng, "MULTIPLY", inc["Z"], -1.0), gloom), em_f.inputs["Color"])
    em_h = ng.nodes.new("ShaderNodeEmission")
    em_h.inputs["Color"].default_value = (*srgb(176, 196, 214), 1.0)
    em_h.inputs["Strength"].default_value = 0.8
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


def caustics(nt, strength=1.0):
    """Animated caustic network on surfaces that face the light, fading with depth (and off above water)."""
    p = world_pos(nt)
    t = frame_driver_value(nt, 1.0 / FPS)
    c = 0.0
    warp = vmath(nt, "SCALE", vmath(nt, "SUBTRACT", noise(nt, vmath(nt, "MULTIPLY", p, (0.6, 0.6, 0.2)), 1.0, 2.0, 0.5,
                                                         out="Color"), (0.5, 0.5, 0.5)), scale=0.35)
    q = vmath(nt, "ADD", vmath(nt, "MULTIPLY", p, (1.0, 1.0, 0.25)), warp)
    for sc_, speed, k in ((2.3, 0.5, 0.6), (3.9, -0.8, 0.45)):
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
    mask = nmath(nt, "MULTIPLY", map_range(nt, nz, 0.05, 0.8), nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", z, 1.0 / 7.0)))
    mask = nmath(nt, "MULTIPLY", mask, nmath(nt, "MULTIPLY", scene_attr(nt, "under"),
                                             nmath(nt, "MULTIPLY_ADD", scene_attr(nt, "gloom"), -0.85, 1.0)))
    return nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", c, mask), strength)


def uw_base(nt, col, caustic=1.0):
    """Base colour under water: red absorbed with depth, darker deeper, plus the caustic light."""
    z = separate(nt, world_pos(nt))["Z"]
    deep = nmath(nt, "SUBTRACT", 1.0, nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", nmath(nt, "MINIMUM", z, 0.0), 1.0 / 6.0)))
    deep = nmath(nt, "MULTIPLY", deep, scene_attr(nt, "under"))
    tint = mix_rgb(nt, deep, (1.0, 1.0, 1.0), srgb(110, 190, 190))
    c = vmath(nt, "MULTIPLY", (*col,) if isinstance(col, tuple) else col, tint)
    if caustic:
        c = vmath(nt, "SCALE", c, scale=nmath(nt, "ADD", 1.0, caustics(nt, 0.8 * caustic)))
    return c


def build_world(sc):
    """Under water: the fog colour itself (infinitely far). Above water: a scattering sky with a soft shoulder
    on camera rays (the Standard view would clip the bright horizon to white). Switched by the "under" value."""
    w = bpy.data.worlds.new("ReservoirWorld")
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
    gloom = nt.nodes.new("ShaderNodeValue")
    gloom.name = "Gloom"
    view = separate(nt, vmath(nt, "NORMALIZE", nt.nodes.new("ShaderNodeTexCoord").outputs["Generated"]))
    fogc = uw_color(nt, view["Z"], gloom.outputs[0])
    sky = nt.nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "MULTIPLE_SCATTERING"
    except TypeError:
        sky.sky_type = "NISHITA"
    sky.sun_elevation = math.asin(SUN_DIR.z)
    sky.sun_rotation = -math.atan2(SUN_DIR.x, SUN_DIR.y)
    sky.sun_disc = False
    sky.altitude = 100.0
    sky.air_density = 1.0
    try:
        sky.aerosol_density = 1.6
    except AttributeError:
        pass
    lit = vmath(nt, "SCALE", sky.outputs[0], scale=0.12)
    knee, top = 0.55, 0.97
    sp = separate(nt, lit)
    peak = nmath(nt, "MAXIMUM", sp["X"], nmath(nt, "MAXIMUM", sp["Y"], sp["Z"]))
    over = nmath(nt, "MAXIMUM", nmath(nt, "SUBTRACT", peak, knee), 0.0)
    rolled = nmath(nt, "ADD", nmath(nt, "MINIMUM", peak, knee),
                   nmath(nt, "DIVIDE", over, nmath(nt, "ADD", nmath(nt, "MULTIPLY", over, 1.0 / (top - knee)), 1.0)))
    soft = vmath(nt, "SCALE", lit, scale=nmath(nt, "DIVIDE", rolled, nmath(nt, "MAXIMUM", peak, 1e-5)))
    is_cam = nt.nodes.new("ShaderNodeLightPath").outputs["Is Camera Ray"]
    skyc = vmath(nt, "ADD", lit, vmath(nt, "SCALE", vmath(nt, "SUBTRACT", soft, lit), scale=is_cam))
    # below the horizon above water: the reservoir far away (dark blue-green), so reflections are not sky
    skyc = mix_rgb(nt, map_range(nt, view["Z"], 0.0, -0.04), skyc, srgb(10, 34, 38))
    col = mix_rgb(nt, under.outputs[0], skyc, fogc)
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(col, bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    return w, under, gloom


def build_materials(atmos):
    M = Materials()
    M.atmos = atmos

    def mat(name, build, blended=False):
        return new_material(name, build, atmos, blended=blended)

    def terrain(nt):
        """River bed and banks: dark silt in the deep water, orange-brown clay on the steep cut banks, wet mud at the
        waterline, dusk grass and leaf litter above it."""
        p = tex_coord(nt, "Object")
        nz = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Normal"])["Z"]
        z = separate(nt, p)["Z"]
        n1 = noise(nt, p, 0.4, 4.0, 0.6)
        n2 = noise(nt, p, 3.5, 6.0, 0.6)
        steep = map_range(nt, nz, 0.8, 0.45)
        silt = color_ramp(nt, n1, [(0.3, srgb(50, 40, 28)), (0.55, srgb(72, 58, 38)), (0.75, srgb(60, 50, 34))])
        strata = nmath(nt, "SINE", nmath(nt, "MULTIPLY_ADD", z, 9.0, nmath(nt, "MULTIPLY", n1, 5.0)))
        clay = color_ramp(nt, map_range(nt, nmath(nt, "MULTIPLY_ADD", strata, 0.3, n2), 0.2, 1.1, smooth=False),
                          [(0.0, srgb(96, 62, 36)), (0.5, srgb(132, 88, 52)), (1.0, srgb(110, 74, 44))])
        col = mix_rgb(nt, steep, silt, clay)
        leaves = map_range(nt, voronoi(nt, p, 3.0), 0.12, 0.06)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", leaves, map_range(nt, nz, 0.7, 0.9)), 0.55), col,
                      srgb(38, 26, 16))
        wet = nmath(nt, "MULTIPLY", map_range(nt, z, -0.6, -0.1), map_range(nt, z, 0.45, 0.15))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", wet, 0.7), col, srgb(46, 34, 22))
        grass = color_ramp(nt, n2, [(0.3, srgb(46, 58, 26)), (0.6, srgb(70, 80, 36)), (0.8, srgb(88, 76, 42))])
        dry = nmath(nt, "MULTIPLY", map_range(nt, z, 0.3, 0.7), map_range(nt, nz, 0.55, 0.8))
        col = mix_rgb(nt, dry, col, grass)
        h = nmath(nt, "ADD", n2, nmath(nt, "MULTIPLY", steep, nmath(nt, "MULTIPLY", strata, 0.5)))
        rough = map_range(nt, wet, 0.0, 1.0, 0.92, 0.45, smooth=False)
        return psdf(nt, uw_base(nt, col, caustic=0.35), rough=rough, spec=0.3, normal=bump(nt, h, 0.5, 0.05))
    M.terrain = mat("RiverBed", terrain)

    def rock(nt):
        p = tex_coord(nt, "Object")
        n = noise(nt, p, 2.4, 6.0, 0.6)
        col = color_ramp(nt, n, [(0.3, srgb(70, 66, 58)), (0.6, srgb(112, 102, 86)), (0.8, srgb(90, 92, 78))])
        nz = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Normal"])["Z"]
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", map_range(nt, nz, 0.4, 0.9), 0.6), col, srgb(66, 88, 42))
        return psdf(nt, uw_base(nt, col), rough=0.85, spec=0.3, normal=bump(nt, noise(nt, p, 9.0, 6.0, 0.6), 0.6, 0.02))
    M.rock = mat("Boulder", rock)

    def wood(nt):
        """Waterlogged timber: dark furrowed bark, paler worn wood where the bark is gone, a film of silt on top."""
        p = tex_coord(nt, "Object")
        crack = voronoi(nt, vmath(nt, "MULTIPLY", p, (1.0, 1.0, 3.0)), 7.0, out="Distance")
        furrow = map_range(nt, crack, 0.02, 0.18)
        streak = noise(nt, p, 18.0, 6.0, 0.6)
        bark = color_ramp(nt, nmath(nt, "MULTIPLY_ADD", furrow, 0.6, nmath(nt, "MULTIPLY", streak, 0.4)),
                          [(0.15, srgb(38, 30, 22)), (0.5, srgb(78, 64, 48)), (0.85, srgb(104, 88, 66))])
        worn = map_range(nt, noise(nt, p, 0.9, 3.0, 0.6), 0.6, 0.72)
        col = mix_rgb(nt, worn, bark, srgb(140, 120, 92))
        nz = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Normal"])["Z"]
        film = nmath(nt, "MULTIPLY", map_range(nt, nz, 0.2, 0.85), map_range(nt, noise(nt, p, 1.6, 3.0), 0.35, 0.6))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", film, 0.7), col, srgb(78, 64, 40))
        return psdf(nt, uw_base(nt, col, caustic=0.3), rough=0.85, spec=0.3,
                    normal=bump(nt, nmath(nt, "MULTIPLY_ADD", furrow, 1.0, nmath(nt, "MULTIPLY", streak, 0.3)), 0.8, 0.015))
    M.wood = mat("Driftwood", wood)

    def clay(nt):
        p = tex_coord(nt, "Object")
        col = color_ramp(nt, noise(nt, p, 1.8, 5.0, 0.6), [(0.3, srgb(66, 54, 40)), (0.7, srgb(98, 80, 58))])
        return psdf(nt, uw_base(nt, col), rough=0.95, spec=0.2, normal=bump(nt, noise(nt, p, 7.0, 6.0), 0.6, 0.03))
    M.rootclay = mat("RootClay", clay)

    def specks(nt):
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(214, 176, 120), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", attr(nt, "rand", "Fac"), 0.35), em.inputs["Strength"])
        df = nt.nodes.new("ShaderNodeBsdfDiffuse")
        df.inputs["Color"].default_value = (*srgb(150, 120, 84), 1.0)
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(df.outputs[0], add.inputs[1])
        _ = oi
        return add.outputs[0]
    M.specks = mat("Sediment", specks)

    def shaft(nt):
        """Additive light shaft: brightest along its core (seen side on), fading down its length."""
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.5
        core = nmath(nt, "POWER", nmath(nt, "SUBTRACT", 1.0, lw.outputs["Facing"]), 5.0)
        zw = separate(nt, world_pos(nt))["Z"]
        along = nmath(nt, "EXPONENT", nmath(nt, "MULTIPLY", zw, 1.0 / 3.2))       # fades with depth
        dist = nt.nodes.new("ShaderNodeCameraData").outputs["View Distance"]
        along = nmath(nt, "MULTIPLY", along, map_range(nt, dist, 0.5, 3.0))         # never blinding up close
        along = nmath(nt, "MULTIPLY", along, 0.22)
        flick = map_range(nt, noise(nt, vmath(nt, "ADD", tex_coord(nt, "Object"),
                                               combine(nt, 0.0, 0.0, frame_driver_value(nt, 0.02))), 1.6), 0.3, 0.7, 0.55, 1.0)
        k = nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", core, along), flick)
        k = nmath(nt, "MULTIPLY", k, nmath(nt, "MULTIPLY", scene_attr(nt, "under"),
                                           nmath(nt, "MULTIPLY_ADD", scene_attr(nt, "gloom"), -0.9, 1.0)))
        oi = nt.nodes.new("ShaderNodeObjectInfo")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(255, 196, 120), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", k, oi.outputs["Alpha"]), em.inputs["Strength"])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(tr.outputs[0], add.inputs[1])
        return add.outputs[0]
    M.shaft = new_material("SunShaft", shaft, blended=True)

    def surface_under(nt):
        """The water surface seen from below: bright, rippling, with a darker total-reflection rim."""
        p = tex_coord(nt, "Object")
        t = frame_driver_value(nt, 0.03)
        rip = noise(nt, vmath(nt, "ADD", p, combine(nt, t, nmath(nt, "MULTIPLY", t, 0.6), 0.0)), 0.7, 5.0, 0.6)
        inc = separate(nt, nt.nodes.new("ShaderNodeNewGeometry").outputs["Incoming"])["Z"]
        window = map_range(nt, nmath(nt, "ABSOLUTE", inc), 0.62, 0.8)
        col = mix_rgb(nt, window, srgb(46, 32, 14), srgb(210, 150, 80))
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(col, em.inputs["Color"])
        nt.links.new(map_range(nt, rip, 0.3, 0.7, 0.6, 1.6), em.inputs["Strength"])
        return em.outputs[0]
    M.surface_under = mat("SurfaceFromBelow", surface_under)

    # --- fish -------------------------------------------------------------------------------------
    def flathead(nt):
        """Flathead catfish: smooth scaleless skin, olive brown and yellow mottled back and flanks breaking up into a
        pale cream belly; a soft wet sheen and a little subsurface glow (the brief asks for skin SSS)."""
        z, x = attr(nt, "zrel", "Fac"), attr(nt, "xrel", "Fac")
        p = tex_coord(nt, "Object")
        base = color_ramp(nt, z, [(0.0, srgb(230, 214, 168)), (0.22, srgb(214, 190, 132)), (0.42, srgb(150, 124, 70)),
                                  (0.62, srgb(108, 86, 48)), (0.85, srgb(74, 60, 36)), (1.0, srgb(58, 48, 30))])
        mot1 = map_range(nt, noise(nt, vmath(nt, "MULTIPLY", p, (0.8, 1.0, 1.2)), 7.0, 3.0, 0.55), 0.5, 0.64)
        mot2 = map_range(nt, voronoi(nt, vmath(nt, "MULTIPLY", p, (0.7, 1.0, 1.0)), 9.0), 0.5, 0.2)
        mott = nmath(nt, "MULTIPLY", nmath(nt, "MAXIMUM", mot1, nmath(nt, "MULTIPLY", mot2, 0.7)), map_range(nt, z, 0.32, 0.55))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", mott, 0.75), base, srgb(44, 34, 20))
        yel = nmath(nt, "MULTIPLY", map_range(nt, noise(nt, p, 4.0, 2.0, 0.5), 0.55, 0.7), map_range(nt, z, 0.3, 0.5))
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", yel, 0.35), col, srgb(176, 150, 66))
        pores = map_range(nt, noise(nt, p, 160.0, 2.0, 0.5), 0.45, 0.75)
        col = vmath(nt, "SCALE", col, scale=nmath(nt, "MULTIPLY_ADD", pores, -0.05, 1.02))
        b = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(uw_base(nt, col, caustic=0.3), b.inputs["Base Color"])
        _feed(nt, b.inputs["Roughness"], map_range(nt, pores, 0.0, 1.0, 0.24, 0.36, smooth=False))
        b.inputs["Specular IOR Level"].default_value = 0.55
        b.inputs["Coat Weight"].default_value = 0.55
        b.inputs["Coat Roughness"].default_value = 0.1
        b.inputs["Subsurface Weight"].default_value = 0.18
        b.inputs["Subsurface Scale"].default_value = 0.012
        b.inputs["Subsurface Radius"].default_value = (1.0, 0.65, 0.4)
        nt.links.new(bump(nt, nmath(nt, "MULTIPLY_ADD", pores, 0.3, mott), 0.1, 0.001), b.inputs["Normal"])
        return b.outputs[0]
    M.flathead = mat("FlatheadCatfish", flathead)

    def fin(name, base, tip, rays=140.0, alpha=0.92, axis="X", ray_k=0.25):
        def build(nt):
            g = separate(nt, tex_coord(nt, "Generated"))
            edge = nmath(nt, "MAXIMUM", nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", g["Z"], 0.5)),
                         nmath(nt, "ABSOLUTE", nmath(nt, "SUBTRACT", g["X"], 0.5)))
            col = color_ramp(nt, nmath(nt, "MULTIPLY", edge, 2.0), [(0.0, base), (1.0, tip)])
            o = separate(nt, tex_coord(nt, "Object"))
            ray = nmath(nt, "POWER", nmath(nt, "ABSOLUTE", nmath(nt, "SINE", nmath(nt, "MULTIPLY", o[axis], rays))), 5.0)
            col = mix_rgb(nt, nmath(nt, "MULTIPLY", ray, ray_k), col, srgb(40, 48, 26))
            base_c = uw_base(nt, col, caustic=0.3)
            b = psdf(nt, base_c, rough=0.35, sss=0.2, sss_scale=0.01, coat=0.2, alpha=alpha, emission=base_c,
                     emission_strength=0.12)                             # a faint glow: light through the membrane
            b.node.inputs["Subsurface Radius"].default_value = (0.6, 0.8, 0.5)
            tl = nt.nodes.new("ShaderNodeBsdfTranslucent")          # light comes through the membrane
            nt.links.new(base_c, tl.inputs["Color"])
            mix = nt.nodes.new("ShaderNodeMixShader")
            mix.inputs[0].default_value = 0.4
            nt.links.new(b, mix.inputs[1])
            nt.links.new(tl.outputs[0], mix.inputs[2])
            return mix.outputs[0]
        return mat(name, build)
    M.fh_fin = fin("FlatheadFin", srgb(132, 108, 66), srgb(78, 62, 38), alpha=1.0, rays=200.0, ray_k=0.3)
    M.fh_fin_h = fin("FlatheadFinRaysAcross", srgb(132, 108, 66), srgb(78, 62, 38), alpha=1.0, rays=200.0, axis="Z",
                     ray_k=0.3)
    M.barbel = mat("Barbel", lambda nt: psdf(nt, uw_base(nt, color_ramp(nt, separate(nt, tex_coord(nt, "Object"))["X"],
                                                                       [(0.0, srgb(70, 56, 34)), (1.0, srgb(150, 126, 84))]),
                                                         caustic=0.2), rough=0.35, sss=0.3, sss_scale=0.006, coat=0.4))

    def eye(nt):
        g = separate(nt, tex_coord(nt, "Object"))
        r = vmath(nt, "LENGTH", combine(nt, g["X"], 0.0, g["Z"]))
        col = color_ramp(nt, r, [(0.0, srgb(4, 4, 4)), (0.42, srgb(4, 4, 4)), (0.47, srgb(150, 104, 30)),
                                 (0.8, srgb(196, 150, 60)), (1.0, srgb(40, 34, 20))])
        shine = map_range(nt, r, 0.40, 0.2)                       # a faint warm eyeshine in the pupil
        return psdf(nt, col, rough=0.05, coat=1.0, coat_rough=0.02, spec=0.8, emission=srgb(255, 190, 90),
                    emission_strength=nmath(nt, "MULTIPLY", nmath(nt, "MULTIPLY", shine, scene_attr(nt, "eyeshine")), 3.0))
    M.eye = mat("FishEye", eye)
    M.mouth = mat("MouthInside", lambda nt: psdf(nt, srgb(60, 30, 30), rough=0.4, sss=0.3, sss_scale=0.01))
    M.throat = mat("Throat", lambda nt: psdf(nt, srgb(150, 120, 104), rough=0.55, sss=0.3, sss_scale=0.01))

    def bluegill(nt):
        z, x = attr(nt, "zrel", "Fac"), attr(nt, "xrel", "Fac")
        o = separate(nt, tex_coord(nt, "Object"))
        base = color_ramp(nt, z, [(0.0, srgb(222, 160, 70)), (0.3, srgb(176, 150, 80)), (0.6, srgb(90, 104, 74)),
                                  (1.0, srgb(48, 62, 54))])
        bars = map_range(nt, nmath(nt, "SINE", nmath(nt, "MULTIPLY", o["X"], 120.0)), 0.3, 0.9)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", bars, nmath(nt, "MULTIPLY", map_range(nt, z, 0.3, 0.5), 0.45)), base,
                      srgb(40, 50, 44))
        cheek = map_range(nt, vmath(nt, "LENGTH", combine(nt, nmath(nt, "SUBTRACT", x, 0.78), 0.0,
                                                          nmath(nt, "SUBTRACT", z, 0.45))), 0.12, 0.06)
        col = mix_rgb(nt, nmath(nt, "MULTIPLY", cheek, 0.6), col, srgb(60, 90, 140))
        return psdf(nt, uw_base(nt, col, caustic=0.4), rough=0.3, coat=0.3, spec=0.5)
    M.bluegill = mat("Bluegill", bluegill)
    M.small_fin = fin("SmallFin", srgb(120, 120, 100), srgb(60, 60, 50), rays=300.0)

    def bubble(nt):
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.35
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(220, 250, 245), 1.0)
        nt.links.new(nmath(nt, "MULTIPLY", lw.outputs["Fresnel"], 2.0), em.inputs["Strength"])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(tr.outputs[0], add.inputs[1])
        return add.outputs[0]
    M.bubble = new_material("Bubble", bubble, blended=True)

    def silt(nt):
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*srgb(150, 140, 110), 1.0)
        em.inputs["Strength"].default_value = 0.25
        df = nt.nodes.new("ShaderNodeBsdfDiffuse")
        df.inputs["Color"].default_value = (*srgb(120, 110, 84), 1.0)
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(df.outputs[0], add.inputs[1])
        return add.outputs[0]
    M.silt = mat("SuctionDebris", silt)
    M.hook = mat("HookSteel", lambda nt: psdf(nt, srgb(40, 42, 46), rough=0.22, metal=1.0, coat=0.6))
    M.lead = mat("LeadSinker", lambda nt: psdf(nt, srgb(110, 112, 116), rough=0.55, metal=0.7))
    M.brass = mat("SwivelBrass", lambda nt: psdf(nt, srgb(176, 140, 70), rough=0.25, metal=1.0))
    M.bead = mat("Bead", lambda nt: psdf(nt, srgb(220, 50, 30), rough=0.08, coat=1.0, transmission=0.0, sss=0.4, sss_scale=0.004))

    def line(nt):
        """Fluorocarbon: nearly invisible, a thin glint; a faint emission keeps it legible at a distance."""
        b = psdf(nt, srgb(190, 200, 196), rough=0.1, spec=1.0, coat=1.0,
                 emission=srgb(170, 210, 200), emission_strength=0.35)
        return b
    M.line = mat("Fluorocarbon", line)

    # --- above water ------------------------------------------------------------------------------------
    def water_top(nt):
        p = tex_coord(nt, "Object")
        t = frame_driver_value(nt, 0.04)
        rip = noise(nt, vmath(nt, "ADD", vmath(nt, "MULTIPLY", p, (1.0, 1.0, 0.0)), combine(nt, t, nmath(nt, "MULTIPLY", t, -0.5), 0.0)),
                    2.2, 6.0, 0.55)
        b = psdf(nt, srgb(20, 13, 6), rough=0.06, spec=0.6, ior=1.333, normal=bump(nt, rip, 0.3, 0.02))
        return b
    M.water_top = mat("RiverSurface", water_top)
    M.canopy = mat("Canopy", lambda nt: psdf(nt, color_ramp(nt, noise(nt, tex_coord(nt, "Object"), 1.2, 6.0, 0.7),
                                                             [(0.3, srgb(22, 32, 16)), (0.7, srgb(52, 62, 30))]), rough=0.9,
                                               normal=bump(nt, noise(nt, tex_coord(nt, "Object"), 9.0, 4.0), 0.6, 0.05)))
    M.bark = mat("TreeBark", lambda nt: psdf(nt, srgb(44, 36, 28), rough=0.9))
    M.hull = mat("HullGelcoat", lambda nt: psdf(nt, color_ramp(nt, attr(nt, "zrel", "Fac"),
                                                               [(0.0, srgb(28, 28, 32)), (0.48, srgb(30, 30, 34)),
                                                                (0.5, srgb(178, 24, 30)), (0.86, srgb(150, 18, 26)),
                                                                (0.88, srgb(220, 220, 224)), (1.0, srgb(220, 220, 224))]),
                                                 rough=0.18, metal=0.25, coat=1.0, coat_rough=0.03))
    M.deck = mat("DeckCarpet", lambda nt: psdf(nt, srgb(52, 54, 58), rough=0.95, normal=bump(nt, noise(nt, tex_coord(nt), 120.0), 0.3, 0.003)))
    M.motor = mat("OutboardCowl", lambda nt: psdf(nt, srgb(16, 16, 18), rough=0.25, coat=1.0))
    M.seat = mat("Seat", lambda nt: psdf(nt, srgb(58, 60, 64), rough=0.5, coat=0.3))
    M.chrome = mat("Chrome", lambda nt: psdf(nt, srgb(200, 202, 206), rough=0.12, metal=1.0))
    M.rod = mat("RodBlank", lambda nt: psdf(nt, srgb(22, 22, 26), rough=0.2, coat=1.0, coat_rough=0.02))
    M.cork = mat("Cork", lambda nt: psdf(nt, srgb(150, 112, 74), rough=0.8, normal=bump(nt, noise(nt, tex_coord(nt), 260.0), 0.5, 0.002)))
    M.reel = mat("Reel", lambda nt: psdf(nt, srgb(40, 42, 46), rough=0.3, metal=0.8, coat=0.5))
    M.skin = mat("Skin", lambda nt: psdf(nt, srgb(196, 140, 106), rough=0.55, sss=0.2, sss_scale=0.01))
    M.hair = mat("Beard", lambda nt: psdf(nt, srgb(70, 52, 38), rough=0.8))
    M.cap = mat("Cap", lambda nt: psdf(nt, srgb(28, 40, 58), rough=0.8))
    M.shades = mat("Shades", lambda nt: psdf(nt, srgb(8, 10, 12), rough=0.05, coat=1.0))
    M.spray = mat("Spray", lambda nt: psdf(nt, srgb(235, 245, 245), rough=0.1, emission=srgb(220, 235, 235), emission_strength=0.6))
    return M


# --------------------------------------------------------------------------
# Terrain: a river bend with a scour hole at the outside of the bend, banks above water
# --------------------------------------------------------------------------
RIVER_W = 9.0                         # half width of the wetted channel (m)
HOLE = Vector((5.2, 1.0, -6.1))       # floor of the scour hole: 6.1 m = 20 ft deep
FH_HOLD = HOLE + Vector((-0.7, 0.2, 0.62))   # where the flathead rests inside the jam


def river_cx(y):
    return 6.0 * math.sin(y / 14.0)


def ground_z(x, y):
    s = (x - river_cx(y)) / RIVER_W
    a = abs(s)
    if a < 1.0:
        z = -3.0 * (1.0 - a * a) ** 0.55
    else:
        z = 1.5 * smoothstep(1.0, 1.28, a) + 0.4 * smoothstep(1.5, 3.5, a)
    dx, dy = x - HOLE.x, y - HOLE.y
    z -= 3.6 * math.exp(-(dx * dx / (2 * 2.8 ** 2) + dy * dy / (2 * 5.5 ** 2)))
    if s > 0.85:                                  # the undercut: the bank is eaten away below the waterline
        under = smoothstep(0.85, 1.0, s) * smoothstep(1.2, 1.05, s) * math.exp(-dy * dy / (2 * 6.0 ** 2))
        z -= 1.1 * under * smoothstep(0.6, -1.0, z)
    z += 0.18 * mnoise.fractal(Vector((x * 0.3, y * 0.3, 0.0)), 0.6, 2.1, 3)
    if a > 1.05:
        z += 0.25 * mnoise.noise(Vector((x * 0.12, y * 0.12, 3.0)))
    return z


def axis_coords(lo, hi, f_lo, f_hi, fine, coarse):
    out, v = [], lo
    while v < hi:
        out.append(v)
        v += fine if f_lo <= v <= f_hi else coarse
    out.append(hi)
    return out


def build_terrain(col, M):
    xs = axis_coords(-40.0, 46.0, -12.0, 16.0, 0.2, 0.8)
    ys = axis_coords(-60.0, 50.0, -14.0, 16.0, 0.2, 0.8)
    bm = bmesh.new()
    rows = []
    for y in ys:
        rows.append([bm.verts.new((x, y, ground_z(x, y))) for x in xs])
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            f = bm.faces.new((rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]))
            f.smooth = True
    ob = finish("RiverBed", bm, col, [M.terrain])
    print(f"terrain: {len(ob.data.vertices)} verts")
    return ob


# --------------------------------------------------------------------------
# Structure: the logjam in the scour hole, root balls, the undercut clay lip, bank trees
# --------------------------------------------------------------------------
def bent_path(p0, d, length, n, rng, wander=0.18, droop=0.0):
    pts = [Vector(p0)]
    d = Vector(d).normalized()
    step = length / (n - 1)
    for i in range(1, n):
        d = (d + Vector((rng.uniform(-wander, wander), rng.uniform(-wander, wander), rng.uniform(-wander, wander) - droop))).normalized()
        pts.append(pts[-1] + d * step)
    return pts


def root_ball(bm, a, axis, rng, radius=1.8, n=26):
    """Roots radiating in the plane facing away from the trunk, with rootlets and a few long dangling roots."""
    axis = Vector(axis).normalized()
    u = axis.cross(Vector((0, 0, 1)))
    u = u.normalized() if u.length > 1e-3 else Vector((1, 0, 0))
    v = axis.cross(u).normalized()
    for k in range(n):
        ang = 2 * math.pi * k / n + rng.uniform(-0.1, 0.1)
        dirv = (u * math.cos(ang) + v * math.sin(ang) - axis * rng.uniform(0.05, 0.5)).normalized()
        L = rng.uniform(0.7, 1.0) * radius
        pts = bent_path(a - axis * 0.2, dirv, L, 7, rng, 0.22, droop=0.04)
        r0 = rng.uniform(0.06, 0.14)
        tube(bm, pts, lambda t, r0=r0: r0 * (1 - 0.88 * t), segs=7)
        j = rng.randint(2, 5)
        sub = bent_path(pts[j], (dirv + Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.8, 0.4)))).normalized(),
                        rng.uniform(0.3, 0.8), 5, rng, 0.3, droop=0.08)
        tube(bm, sub, lambda t: 0.03 * (1 - 0.9 * t), segs=5)


LOGS = []                              # (polyline points, radius) of every log, for clearance checks


def seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-9)))
    return (p - (a + ab * t)).length


def frame_logs(bm, path, rng, clear=()):
    """Logs that frame the hook's dolly: crossing just above and beside the camera path, so the move reads as
    pushing through the jam."""
    out = []
    for frac, off_up, off_side, tilt in ((0.25, 0.75, 0.0, 0.0), (0.5, 0.1, -0.85, 1.1), (0.7, -0.55, 0.6, 0.5)):
        i = int(frac * (len(path) - 1))
        p = path[i]
        d = (path[min(i + 3, len(path) - 1)] - path[max(i - 3, 0)]).normalized()
        side = d.cross(Vector((0, 0, 1))).normalized()
        up = side.cross(d).normalized()
        c = p + up * off_up + side * off_side
        axis = (side * math.cos(tilt) + up * math.sin(tilt)).normalized()
        r = rng.uniform(0.2, 0.3)
        pts = [c - axis * 2.5 + axis * 5.0 * k / 8 for k in range(9)]
        if any(seg_dist(cp, a, b) < cr + r for cp, cr in clear for a, b in zip(pts, pts[1:])):
            pts = [c - axis * 1.2 + axis * 2.4 * k / 8 for k in range(9)]          # shorter, if the long one is in the way
            if any(seg_dist(cp, a, b) < cr + r for cp, cr in clear for a, b in zip(pts, pts[1:])):
                continue
        tube(bm, pts, lambda t, r=r: r * (1 - 0.25 * t), segs=12)
        LOGS.append((pts, r))
        out.append((pts, r))
    return out


def build_logjam(col, M, keep_clear=(), frame_path=None):
    """Waterlogged trunks piled into the scour hole, most lying with the current, a few raking up towards the
    surface, two root balls, smaller branches wedged between them. The flathead's den stays open."""
    rng = random.Random(31)
    bm = bmesh.new()
    clear = [(FH_HOLD, 0.75)] + list(keep_clear)
    placed = 0
    tries = 0
    roots = []
    while placed < 16 and tries < 600:
        tries += 1
        c = HOLE + Vector((rng.uniform(-3.0, 2.6), rng.uniform(-4.5, 4.5), 0.0))
        c.z = ground_z(c.x, c.y) + rng.uniform(0.3, 2.6)
        yaw = rng.uniform(-0.55, 0.55) + (math.pi / 2 if rng.random() < 0.8 else rng.uniform(0, math.pi))
        pitch = rng.uniform(-0.15, 0.15) if placed > 3 else rng.uniform(0.35, 0.7)
        d = Vector((math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)))
        L = rng.uniform(5.0, 10.0)
        r = rng.uniform(0.18, 0.42)
        pts = [c - d * L * 0.5 + d * L * i / 10 for i in range(11)]
        for i, p in enumerate(pts):                     # sag and kinks
            p.z += 0.1 * math.sin(i * 1.7 + placed)
        if any(p.z > -0.25 for p in pts[2:-2]) and pitch < 0.3:
            continue
        top = max(p.z for p in pts)
        if top > 0.4:                                   # the raking logs may break the surface, but only a little
            continue
        if any(seg_dist(cp, a, b) < cr + r for cp, cr in clear for a, b in zip(pts, pts[1:])):
            continue
        tube(bm, pts, lambda t, r=r: r * (1 - 0.35 * t), segs=12)
        LOGS.append((pts, r))
        for j in range(rng.randint(1, 3)):              # broken limbs
            p0 = pts[rng.randint(2, 8)]
            dd = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.3, 0.8))).normalized()
            br = bent_path(p0, dd, rng.uniform(0.8, 2.0), 5, rng, 0.2)
            if any(seg_dist(cp, br[0], br[-1]) < cr + 0.1 for cp, cr in clear):
                continue
            tube(bm, br, lambda t: 0.07 * (1 - 0.8 * t), segs=6)
            LOGS.append((br, 0.07))
        if len(roots) < 2 and r > 0.3:
            roots.append((pts[0], -d))
        placed += 1
    for a, axis in roots:
        root_ball(bm, a, axis, rng, radius=1.7)
    if frame_path:
        frame_logs(bm, frame_path, rng, [(cp, cr) for cp, cr in clear if all((cp - q).length > 0.05 for q in frame_path)])
    jam = finish("Logjam", bm, col, [M.wood])
    print(f"logjam: {placed} logs ({tries} tries), {len(roots)} root balls")
    bm = bmesh.new()
    for a, axis in roots:
        bm_blob(bm, a + axis * 0.15, (0.75, 0.75, 0.65), rng, subdiv=3, jitter=0.18)
    for f in bm.faces:
        f.smooth = True
    clay = finish("RootBallClay", bm, col, [M.rootclay])
    return jam, clay, roots


def build_undercut(col, M):
    """The undercut bank: a lip of clay and turf overhanging the eroded cavity, with roots hanging out of it."""
    rng = random.Random(12)
    bm = bmesh.new()
    zl = None
    rings = []
    ys = [HOLE.y - 7.0 + 14.0 * i / 20 for i in range(21)]
    for y in ys:
        xb = river_cx(y) + RIVER_W * 1.02
        over = 1.3 * math.exp(-((y - HOLE.y) ** 2) / (2 * 4.0 ** 2))
        prof = [(xb + 0.6, 1.4), (xb - over, 1.3), (xb - over - 0.15, 0.9), (xb - over * 0.8, 0.35), (xb - over * 0.4, 0.05),
                (xb + 0.3, -0.05), (xb + 0.8, 0.6)]
        rings.append([bm.verts.new((px, y, pz + 0.05 * mnoise.noise(Vector((px, y, pz))))) for px, pz in prof])
    n = len(rings[0])
    for ra, rb in zip(rings, rings[1:]):
        for k in range(n):
            f = bm.faces.new((ra[k], ra[(k + 1) % n], rb[(k + 1) % n], rb[k]))
            f.smooth = True
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    lip = finish("UndercutLip", bm, col, [M.rootclay])
    bm = bmesh.new()
    for k in range(26):                                 # roots hanging into the water from the lip
        y = HOLE.y + rng.uniform(-5.0, 5.0)
        over = 1.3 * math.exp(-((y - HOLE.y) ** 2) / (2 * 4.0 ** 2))
        x = river_cx(y) + RIVER_W * 1.02 - over * rng.uniform(0.3, 0.95)
        pts = bent_path((x, y, 0.2), (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), -1), rng.uniform(0.6, 1.9), 6, rng, 0.2)
        tube(bm, pts, lambda t: 0.03 * (1 - 0.85 * t), segs=5)
    roots = finish("UndercutRoots", bm, col, [M.wood])
    return lip, roots


def build_trees(col, M):
    """Wooded banks for the aerial and the boat shots: trunks with lumpy canopies along both sides of the river."""
    rng = random.Random(77)
    bt, bc = bmesh.new(), bmesh.new()
    for side in (1, -1):
        y = -70.0
        while y < 55.0:
            y += rng.uniform(2.5, 5.0)
            x = river_cx(y) + side * (RIVER_W * rng.uniform(1.35, 1.9) + rng.uniform(0, 6))
            z = ground_z(x, y)
            h = rng.uniform(6.0, 11.0)
            lean = Vector((-side * rng.uniform(0.0, 0.15), rng.uniform(-0.08, 0.08), 1.0)).normalized()
            top = Vector((x, y, z)) + lean * h
            tube(bt, [Vector((x, y, z - 0.5)), Vector((x, y, z)).lerp(top, 0.5), top], lambda t: 0.28 * (1 - 0.7 * t), segs=7)
            for _k in range(rng.randint(6, 9)):
                c = Vector((x, y, z)).lerp(top, rng.uniform(0.45, 1.0)) + Vector((rng.uniform(-2.2, 2.2), rng.uniform(-2.2, 2.2), 0.0))
                s = rng.uniform(1.6, 3.0)
                bm_blob(bc, c, (s, s, s * 0.85), rng, subdiv=2, jitter=0.3)
            if rng.random() < 0.7:                          # bushes on the bank
                bx = river_cx(y) + side * RIVER_W * rng.uniform(1.25, 1.4)
                s = rng.uniform(0.8, 1.6)
                bm_blob(bc, (bx, y + rng.uniform(-1, 1), ground_z(bx, y) + s * 0.4), (s, s, s * 0.7), rng, subdiv=2, jitter=0.3)
    for f in bc.faces:
        f.smooth = True
    trunks = finish("BankTrunks", bt, col, [M.bark])
    canopy = finish("BankCanopy", bc, col, [M.canopy])
    canopy.visible_shadow = False
    return trunks, canopy


def build_shafts(col, M):
    """Dusk light shafts: few, warm and dim, bent by refraction (about 48 degrees from vertical for a sun 9 degrees
    up), additive (fake volumetrics: a world volume would block the distant light)."""
    rng = random.Random(9)
    h = Vector((SUN_DIR.x, SUN_DIR.y, 0.0)).normalized()
    ang = math.asin(min(1.0, math.cos(math.asin(SUN_DIR.z)) / 1.333))
    d = (-h * math.sin(ang) + Vector((0, 0, -math.cos(ang)))).normalized()
    bm = bmesh.new()
    for i in range(9):
        top = HOLE + Vector((rng.uniform(-8, 6), rng.uniform(-9, 9), 0.0))
        top.z = 0.0
        L = rng.uniform(6.0, 9.0)
        r0, r1 = rng.uniform(0.2, 0.5), rng.uniform(0.7, 1.4)
        bm_frustum(bm, top + d * L, top, r1, r0, segs=20, cap0=False, cap1=False)
    ob = finish("SunShafts", bm, col, [M.shaft], recalc=False)
    ob.visible_shadow = False
    try:
        ob.visible_diffuse = ob.visible_glossy = False
    except AttributeError:
        pass
    return [ob]



def specks(name, col, mat, center, radius, n, size, seed):
    rng = random.Random(seed)
    bm = bmesh.new()
    rand = bm.verts.layers.float.new("rand")
    for _ in range(n):
        while True:
            p = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
            if p.length <= 1:
                break
        p = Vector((p.x * radius[0], p.y * radius[1], p.z * radius[2]))
        s = size * rng.uniform(0.4, 1.5)
        res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=s, matrix=Matrix.Translation(p))
        r = rng.random()
        for v in res["verts"]:
            v[rand] = r
    ob = finish(name, bm, col, [mat])
    ob.location = center
    ob.visible_shadow = False
    return ob


def build_sediment(col, M):
    """Suspended sediment around each underwater shot, drifting downstream (+Y) with the current."""
    obs = []
    for i, (c, r, n) in enumerate(((tuple(FH_HOLD + Vector((0.0, -1.0, 0.4))), (4.0, 5.0, 2.2), 3200),
                                   (tuple(HOLE + Vector((-3.0, -4.0, 2.5))), (6.0, 7.0, 3.0), 3200),
                                   (tuple(HOLE + Vector((-2.5, -5.0, 0.8))), (3.5, 4.0, 1.6), 2600))):
        ob = specks(f"Sediment{i}", col, M.specks, c, r, n, 0.003, 30 + i)
        for f, dv in ((1, Vector((0, 0, 0))), (FRAME_END, Vector((0.2, 3.5, -0.3)))):
            ob.location = Vector(c) + dv
            key(ob, "location", f)
        for fc in iter_fcurves(ob):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
        obs.append(ob)
    return obs


def iter_fcurves(idb):
    ad = idb.animation_data
    if not ad or not ad.action:
        return []
    act = ad.action
    out = list(getattr(act, "fcurves", []) or [])
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                out += list(bag.fcurves)
    return out


def build_surface_under(col, M):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=60.0, matrix=Matrix.Translation((0, -5, 0)))
    ob = finish("SurfaceFromBelow", bm, col, [M.surface_under], recalc=False)
    ob.visible_shadow = False
    return ob


# --------------------------------------------------------------------------
# Action clock: render frame -> action frame. The strike plays as a speed ramp (about a third of real speed),
# so everything that takes part in it (the flathead, the bait, the line, the suction debris) runs on this clock.
# --------------------------------------------------------------------------
def _clock():
    t, out = 1.0, {1: 1.0}
    for f in range(2, FRAME_END + 2):
        slow = smoothstep(STRIKE - 18, STRIKE - 8, f) * (1.0 - smoothstep(STRIKE + 12, STRIKE + 24, f))
        t += 1.0 - 0.68 * slow
        out[f] = t
    return out


ACT = _clock()
A_STRIKE = ACT[STRIKE]


def act(f):
    """Action frame (float) at render frame f (fractional frames interpolate)."""
    f0 = int(math.floor(f))
    f0 = max(1, min(FRAME_END, f0))
    return ACT[f0] + (ACT[f0 + 1] - ACT[f0]) * (f - f0)


def gn_time(ng, clock=None):
    """Seconds: scene time, or the action clock (an empty whose x location is keyed in seconds)."""
    if clock is None:
        return nmath(ng, "DIVIDE", ng.nodes.new("GeometryNodeInputSceneTime").outputs["Frame"], float(FPS))
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = clock
    return separate(ng, oi.outputs["Location"])["X"]


def gn_xyz(ng):
    p = separate(ng, ng.nodes.new("GeometryNodeInputPosition").outputs[0])
    return p["X"], p["Y"], p["Z"]


def build_clock(col):
    ob = empty_object("ActionClock", col, size=0.05)
    for f in range(1, FRAME_END + 1):
        ob.location = (ACT[f] / FPS, 0.0, 0.0)
        key(ob, "location", f, index=0)
    for fc in iter_fcurves(ob):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    return ob


class Particles:
    """Bubbles, debris and spray: every particle is born once with a position, velocity and life; a geometry
    nodes group moves them with time (scene time or the action clock), so nothing needs per-frame keys."""

    def __init__(self):
        self.items = []     # (birth_s, pos, vel, life_s, size, gravity)

    def emit(self, t, pos, vel, life, size, gravity=0.0):
        self.items.append((t, Vector(pos), Vector(vel), life, size, gravity))

    def build(self, name, col, mat, clock=None, streak=False):
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
        t = gn_time(ng, clock)
        age = nmath(ng, "SUBTRACT", t, nattr("birth"))
        life = nattr("life")
        alive = nmath(ng, "MULTIPLY", nmath(ng, "GREATER_THAN", age, 0.0), nmath(ng, "LESS_THAN", age, life))
        sp = nodes.new("GeometryNodeSetPosition")
        links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
        disp = vmath(ng, "ADD", vmath(ng, "SCALE", nattr("vel", "FLOAT_VECTOR"), scale=age),
                     combine(ng, 0.0, 0.0, nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", age, age),
                                                nmath(ng, "MULTIPLY", nattr("grav"), -0.5))))
        links.new(disp, sp.inputs["Offset"])
        dele = nodes.new("GeometryNodeDeleteGeometry")
        dele.domain = "POINT"
        links.new(sp.outputs[0], dele.inputs["Geometry"])
        links.new(nmath(ng, "SUBTRACT", 1.0, alive), dele.inputs["Selection"])
        ico = nodes.new("GeometryNodeMeshIcoSphere")
        ico.inputs["Radius"].default_value = 1.0
        ico.inputs["Subdivisions"].default_value = 1
        inst = nodes.new("GeometryNodeInstanceOnPoints")
        links.new(dele.outputs[0], inst.inputs["Points"])
        links.new(ico.outputs["Mesh"], inst.inputs["Instance"])
        grow = nmath(ng, "MULTIPLY", nattr("size"),
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


# --------------------------------------------------------------------------
# Fish geometry
# --------------------------------------------------------------------------
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


def fish_mesh(name, bm, mats):
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    return me


def swim_group():
    """Body wave for the school fish: the tail swings most; inputs per object (phase, amplitude, rate, length)."""
    ng, gi, go = new_group("SchoolSwim", "GeometryNodeTree",
                           [("Geometry", "NodeSocketGeometry"), ("Phase", "NodeSocketFloat"), ("Amp", "NodeSocketFloat"),
                            ("Rate", "NodeSocketFloat"), ("Len", "NodeSocketFloat")],
                           [("Geometry", "NodeSocketGeometry")])
    x, _y, _z = gn_xyz(ng)
    L = gi.outputs["Len"]
    w = nmath(ng, "POWER", nmath(ng, "DIVIDE", nmath(ng, "SUBTRACT", nmath(ng, "MULTIPLY", L, 0.15), x),
                                 nmath(ng, "MULTIPLY", L, 0.65)), 1.5)
    w = nmath(ng, "MINIMUM", nmath(ng, "MAXIMUM", w, 0.0), 1.4)
    k = nmath(ng, "DIVIDE", 2 * math.pi / 0.9, L)
    arg = nmath(ng, "SUBTRACT", nmath(ng, "MULTIPLY_ADD", gn_time(ng), gi.outputs["Rate"], gi.outputs["Phase"]),
                nmath(ng, "MULTIPLY", x, k))
    dy = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", nmath(ng, "SINE", arg), w), gi.outputs["Amp"])
    sp = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, 0.0, dy, 0.0), sp.inputs["Offset"])
    ng.links.new(sp.outputs[0], go.inputs["Geometry"])
    return ng


def ctrl_swim_group(ctrl, L, name="FlatheadSwim"):
    """Body wave driven by a controller empty: x = phase (rad), y = amplitude (m), z = body bend (turning).
    L is the modelled length nose to tail base (the head barely moves, the tail most)."""
    ng, gi, go = new_group(name, "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    oi = ng.nodes.new("GeometryNodeObjectInfo")
    oi.inputs["Object"].default_value = ctrl
    c = separate(ng, oi.outputs["Location"])
    x, _y, _z = gn_xyz(ng)
    back = nmath(ng, "MAXIMUM", nmath(ng, "SUBTRACT", 0.17 * L, x), 0.0)
    w = nmath(ng, "MINIMUM", nmath(ng, "POWER", nmath(ng, "DIVIDE", back, 0.67 * L), 1.5), 1.45)
    arg = nmath(ng, "SUBTRACT", c["X"], nmath(ng, "MULTIPLY", x, 6.9 / L))
    dy = nmath(ng, "MULTIPLY", nmath(ng, "MULTIPLY", nmath(ng, "SINE", arg), w), c["Y"])
    dy = nmath(ng, "MULTIPLY_ADD", nmath(ng, "MULTIPLY", back, back), nmath(ng, "MULTIPLY", c["Z"], 3.6 / L), dy)
    sp = ng.nodes.new("GeometryNodeSetPosition")
    ng.links.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    ng.links.new(combine(ng, 0.0, dy, 0.0), sp.inputs["Offset"])
    ng.links.new(sp.outputs[0], go.inputs["Geometry"])
    return ng


FH_L = 1.0                          # modelled length nose to tail base (m): about 1.17 m to the tail tip
FH_SCALE = 1.0                      # an illustrative 40+ lb flathead
FH_JAW_HINGE = Vector((FH_L / 2 - 0.145, 0.0, -0.034))


def fh_curves():
    L = FH_L

    def u_of(x):
        return (L / 2 - x) / L
    top_u = curve([(0.0, 0.010), (0.05, 0.034), (0.12, 0.060), (0.25, 0.090), (0.40, 0.104), (0.6, 0.094),
                   (0.8, 0.066), (1.0, 0.046)])
    bot_u = curve([(0.0, -0.016), (0.07, -0.028), (0.14, -0.038), (0.2, -0.066), (0.32, -0.088), (0.47, -0.094),
                   (0.66, -0.078), (0.85, -0.052), (1.0, -0.040)])
    wid_u = curve([(0.0, 0.062), (0.05, 0.100), (0.12, 0.126), (0.22, 0.121), (0.36, 0.102), (0.56, 0.078),
                   (0.8, 0.046), (1.0, 0.024)])
    return (lambda x: top_u(u_of(x))), (lambda x: bot_u(u_of(x))), (lambda x: wid_u(u_of(x))), u_of


def fin_base(fn, x, inset=0.01):
    return (x, fn(x) - inset if fn(x) > 0 else fn(x) + inset)


def make_flathead(col, M):
    """Flathead catfish: a broad, flattened head with small eyes set high, the lower jaw jutting past the upper,
    eight barbels (two long maxillary, four chin, two short nasal), a long fleshy adipose fin, a small dorsal,
    a long rounded anal fin and a broad, barely notched tail. Smooth scaleless skin."""
    L = FH_L
    top, bot, wid, u_of = fh_curves()
    bm = bmesh.new()
    ring_section(bm, L / 2, -L / 2, top, bot, wid, nr=70, na=44, power=2.5, xr=lambda x: 1.0 - u_of(x))
    ex = L / 2 - 0.085
    for s in (1, -1):
        ellipsoid(bm, (ex, s * wid(ex) * 0.74, top(ex) * 0.72), (0.0095, 0.0055, 0.0085), subdiv=3, mat=1)
    me = fish_mesh("Flathead", bm, [M.flathead, M.eye])
    body = bpy.data.objects.new("Flathead", me)
    col.objects.link(body)
    for p in me.polygons:
        p.use_smooth = True
    fb = bmesh.new()
    xd = L / 2 - 0.36                                   # small rounded dorsal with a stout first ray
    soft_fin(fb, [fin_base(top, xd), (xd - 0.005, top(xd) + 0.07), (xd - 0.04, top(xd - 0.04) + 0.075),
                  (xd - 0.085, top(xd - 0.085) + 0.04), fin_base(top, xd - 0.1)], 0.003, 2, samples=8)
    xa = -0.24                                          # long, low, fleshy adipose fin
    soft_fin(fb, [fin_base(top, xa, 0.006), (xa - 0.02, top(xa - 0.02) + 0.03), (xa - 0.12, top(xa - 0.12) + 0.036),
                  (xa - 0.19, top(xa - 0.19) + 0.022), fin_base(top, xa - 0.21, 0.006)], 0.006, 2, samples=8)
    xt = -L / 2                                         # broad tail, nearly square with a shallow notch
    soft_fin(fb, [(xt + 0.03, 0.036), (xt - 0.06, 0.085), (xt - 0.165, 0.098), (xt - 0.18, 0.06), (xt - 0.17, 0.004),
                  (xt - 0.18, -0.055), (xt - 0.165, -0.092), (xt - 0.06, -0.08), (xt + 0.03, -0.032)], 0.003, 4, samples=8)
    xn = -0.1                                           # long rounded anal fin
    soft_fin(fb, [fin_base(bot, xn), (xn - 0.02, bot(xn) - 0.05), (xn - 0.12, bot(xn - 0.12) - 0.06),
                  (xn - 0.22, bot(xn - 0.22) - 0.04), fin_base(bot, xn - 0.24)], 0.003, 2, samples=8)
    for s in (1, -1):                                   # pelvic fins
        rot = Euler((s * math.radians(20), 0.0, 0.0))
        xp = -0.02
        soft_fin(fb, [(xp + 0.02, bot(xp) + 0.012), (xp, bot(xp) - 0.05), (xp - 0.05, bot(xp) - 0.06),
                      (xp - 0.06, bot(xp) - 0.03), (xp - 0.03, bot(xp) + 0.01)], 0.003, 2, offset=(0, s * 0.04, 0), rot=rot)
    fbe = fish_mesh("FlatheadFins", fb, [M.flathead, M.eye, M.fh_fin, M.fh_fin, M.fh_fin_h])
    fins = bpy.data.objects.new("FlatheadFins", fbe)
    col.objects.link(fins)
    fins.parent = body
    pects = []
    for s in (1, -1):                                   # pectorals with a stiff leading spine; they scull
        pb = bmesh.new()
        soft_fin(pb, [(0.006, 0.0), (-0.04, 0.016), (-0.1, 0.004), (-0.118, -0.022), (-0.085, -0.04), (-0.015, -0.018)],
                 0.0025, 0, samples=8)
        tube(pb, [(0.006, 0.0, 0.0), (-0.05, 0.0, 0.012), (-0.1, 0.0, 0.004)], lambda t: 0.004 * (1 - 0.7 * t), segs=6, mat=0)
        pm = fish_mesh(f"FlatheadPectoral{'L' if s > 0 else 'R'}", pb, [M.fh_fin_h])
        po = bpy.data.objects.new(pm.name, pm)
        col.objects.link(po)
        po.parent = body
        xq = L / 2 - 0.2
        po.location = (xq, s * (wid(xq) - 0.008), bot(xq) * 0.45)
        po.rotation_euler = (s * math.radians(-6), math.radians(4), s * math.radians(-30))
        pects.append(po)
    # barbels on the head: two long maxillary (mouth corners) and two short nasal ones
    bb = bmesh.new()
    for s in (1, -1):
        p0 = Vector((L / 2 - 0.035, s * (wid(L / 2 - 0.035) - 0.004), -0.012))
        pts = [p0 + Vector((-0.02 * i, s * 0.028 * i, -0.004 * i * i)) for i in range(7)]
        tube(bb, pts, lambda t: 0.0045 * (1 - 0.85 * t), segs=6)
        n0 = Vector((L / 2 - 0.03, s * 0.035, top(L / 2 - 0.03) - 0.004))
        tube(bb, [n0, n0 + Vector((0.008, s * 0.006, 0.016)), n0 + Vector((0.01, s * 0.01, 0.026))],
             lambda t: 0.002 * (1 - 0.6 * t), segs=5)
    barbels = finish("FlatheadBarbels", bb, col, [M.barbel])
    barbels.parent = body
    for p in barbels.data.polygons:
        p.use_smooth = True
    # lower jaw (hinged, jutting past the upper) with the four chin barbels; the pale mouth behind it
    jb = bmesh.new()
    hx = FH_JAW_HINGE.x
    jl = L / 2 + 0.024 - hx

    def jtop(x):
        return 0.002 + 0.024 * max(0.0, (x + 0.015) / (jl + 0.015))
    ring_section(jb, jl, -0.015, jtop,
                 lambda x: -0.03 + 0.02 * min(1.0, max(0.0, x / jl)) ** 0.8,
                 lambda x: 0.11 - 0.035 * min(1.0, max(0.0, x / jl)) ** 1.5, nr=16, na=28, power=2.6, xr=lambda x: 0.95)
    for v in jb.verts:
        v[jb.verts.layers.float["zrel"]] *= 0.45
    for s in (1, -1):
        for k in (0.03, 0.065):
            c0 = Vector((jl - 0.035, s * k, -0.026))
            tube(jb, [c0, c0 + Vector((0.006, s * 0.006, -0.022)), c0 + Vector((-0.004, s * 0.012, -0.044))],
                 lambda t: 0.0028 * (1 - 0.7 * t), segs=5, mat=1)
    jm = fish_mesh("FlatheadJaw", jb, [M.flathead, M.barbel])
    jaw = bpy.data.objects.new("FlatheadJaw", jm)
    col.objects.link(jaw)
    jaw.parent = body
    jaw.location = FH_JAW_HINGE
    for p in jm.polygons:
        p.use_smooth = True
    tb = bmesh.new()
    ellipsoid(tb, (0.0, 0.0, 0.0), (0.07, 0.085, 0.016), subdiv=3, mat=0)
    throat = finish("FlatheadMouth", tb, col, [M.throat])
    throat.parent = body
    throat.location = (L / 2 - 0.085, 0.0, -0.028)
    root = empty_object("FlatheadRig", col, size=0.3)
    root.rotation_mode = "QUATERNION"
    root.scale = (FH_SCALE,) * 3
    body.parent = root
    beat = empty_object("FlatheadBeat", col, size=0.05)
    ng = ctrl_swim_group(beat, L)
    for ob in (body, fins):
        ob.modifiers.new("Swim", "NODES").node_group = ng
    sub = body.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = 1
    sub.render_levels = 2
    jsub = jaw.modifiers.new("Subdivision", "SUBSURF")
    jsub.levels = jsub.render_levels = 1
    return dict(root=root, body=body, fins=fins, jaw=jaw, throat=throat, pects=pects, beat=beat, barbels=barbels)


def small_fish(name, M, mat, fin_m, L, top_u, bot_u, wid_u, fins_spec, nr=26, na=16):
    """School fish mesh (one shared mesh): body plus fins in one object."""
    def u_of(x):
        return (L / 2 - x) / L
    top, bot, wid = (lambda x: top_u(u_of(x))), (lambda x: bot_u(u_of(x))), (lambda x: wid_u(u_of(x)))
    bm = bmesh.new()
    ring_section(bm, L / 2, -L / 2, top, bot, wid, nr=nr, na=na, power=2.2, xr=lambda x: 1.0 - u_of(x))
    ex = L / 2 - 0.12 * L
    for s in (1, -1):
        ellipsoid(bm, (ex, s * wid(ex) * 0.85, top(ex) * 0.35), (0.032 * L, 0.016 * L, 0.032 * L), subdiv=2, mat=1)
    for ctrl in fins_spec(L, top, bot):
        soft_fin(bm, ctrl, 0.012 * L, 2, samples=5)
    return fish_mesh(name, bm, [mat, M.eye, fin_m])


def bluegill_mesh(M):
    top = curve([(0.0, 0.03), (0.1, 0.13), (0.35, 0.24), (0.6, 0.2), (0.85, 0.08), (1.0, 0.06)])
    bot = curve([(0.0, -0.04), (0.12, -0.14), (0.38, -0.25), (0.62, -0.2), (0.85, -0.08), (1.0, -0.06)])
    wid = curve([(0.0, 0.02), (0.2, 0.06), (0.45, 0.07), (0.8, 0.03), (1.0, 0.015)])

    def fins(L, top, bot):
        t = -L / 2
        return [[(0.15 * L, top(0.15 * L) - 0.01 * L), (0.1 * L, top(0.1 * L) + 0.1 * L), (-0.25 * L, top(-0.25 * L) + 0.12 * L),
                 (-0.32 * L, top(-0.32 * L) - 0.01 * L)],
                [(-0.05 * L, bot(-0.05 * L) + 0.01 * L), (-0.1 * L, bot(-0.1 * L) - 0.1 * L), (-0.3 * L, bot(-0.3 * L) - 0.08 * L),
                 (-0.32 * L, bot(-0.32 * L) + 0.01 * L)],
                [(t + 0.03 * L, 0.05 * L), (t - 0.12 * L, 0.15 * L), (t - 0.2 * L, 0.12 * L), (t - 0.17 * L, 0.0),
                 (t - 0.2 * L, -0.12 * L), (t - 0.12 * L, -0.15 * L), (t + 0.03 * L, -0.05 * L)]]
    L = 0.19
    return small_fish("Bluegill", M, M.bluegill, M.small_fin, L, lambda u: top(u) * L, lambda u: bot(u) * L,
                      lambda u: wid(u) * L, fins, nr=22), L


def heading_quat(fwd, roll=0.0):
    """Rotation that turns local +X to fwd with local +Z kept up (then a roll about the heading)."""
    fwd = Vector(fwd).normalized()
    q = fwd.to_track_quat("X", "Z")
    if roll:
        q = q @ Quaternion((1, 0, 0), roll)
    return q


# --------------------------------------------------------------------------
# The flathead's moves (in action frames), the live bait rig and the line
# --------------------------------------------------------------------------
F0 = Vector((-0.25, -1.0, 0.0)).normalized()               # she rests facing upstream, into the current
NOSE = FH_L / 2 * FH_SCALE
SINKER = FH_HOLD + F0 * 3.2
SINKER.z = ground_z(SINKER.x, SINKER.y) + 0.035
SWIVEL = SINKER + Vector((0.0, 0.07, 0.0))                 # bead and swivel just below the egg sinker
LEADER = 0.55                                               # leader length (m)
BAIT_LEN = 0.19                                             # the bluegill (7.5 in)
LINE_FAR = Vector((0.0, 0.0, 0.0))                          # set once the boat is placed (the rod is upstream)


def bait_state(a):
    """The live bluegill at action frame a: hook point, body heading, roll and kick strength. It hangs on the
    leader downstream of the sinker, facing into the current, and keeps kicking and swinging."""
    t = a / FPS
    sway = 0.42 * math.sin(t * 2 * math.pi / 3.1) + 0.18 * math.sin(t * 2 * math.pi / 1.3 + 1.0)
    burst = max(0.0, math.sin(t * 2 * math.pi / 2.2)) ** 6                  # struggling bursts
    lift = 0.32 + 0.12 * math.sin(t * 2 * math.pi / 4.0) + 0.1 * burst
    d = Vector((math.sin(sway) * 0.9, math.cos(sway), lift)).normalized()  # downstream (+Y) and up
    hook = SWIVEL + d * LEADER
    yaw = 0.55 * math.sin(t * 2 * math.pi / 0.9) * (0.4 + 0.6 * burst)
    fwd = (Matrix.Rotation(yaw, 3, "Z") @ Vector((0.15 * math.sin(t * 1.3), -1.0, -0.15))).normalized()
    roll = 0.25 * math.sin(t * 2 * math.pi / 1.7)
    return hook, fwd, roll, 0.5 + 0.5 * burst


def strike_point():
    hook, fwd, _r, _k = bait_state(A_STRIKE)
    return hook + fwd * 0.03                                # the bluegill's head, just ahead of the hook


E_STRIKE = strike_point()


def fh_state(a):
    """Flathead at action frame a: (position, forward, amplitude, rate, bend, jaw opening)."""
    s = a - A_STRIKE
    pos = FH_HOLD + Vector((0, 0, 0.01 * math.sin(a * 0.04)))
    fwd = F0.copy()
    amp, rate, bnd = 0.004, 2.6, 0.0
    jaw = 0.08 + 0.06 * max(0.0, math.sin(a * 0.09))       # slow breathing
    to = (E_STRIKE - FH_HOLD)
    to.z *= 0.6
    to = to.normalized()
    stalk = E_STRIKE - to * (NOSE + 0.6)
    if s > -90:                                             # the vibrations draw her out of the wood
        k = ease((s + 90) / 76.0)
        pos = FH_HOLD.lerp(stalk, k) + Vector((0, 0, 0.01 * math.sin(a * 0.04)))
        fwd = F0.lerp(to, min(1.0, k * 1.4)).normalized()
        amp = 0.004 + 0.008 * math.sin(math.pi * min(1.0, k))
        rate = 2.6 + 3.0 * k
    if s > -12:                                             # the lunge and the vacuum
        lunge = smoothstep(-8, 0, s)
        back = smoothstep(-12, -8, s) * (1 - smoothstep(-8, -4, s))
        pos = stalk + to * (0.6 * lunge - 0.05 * back)
        fwd = to
        amp = 0.012 + 0.05 * smoothstep(-10, -6, s) * (1 - smoothstep(4, 24, s))
        rate = 5.6 + 20.0 * smoothstep(-10, -6, s) * (1 - smoothstep(6, 30, s))
        jaw = max(jaw, smoothstep(-6, -1, s) * (1 - smoothstep(2, 8, s)))
    if s > 3:                                               # she turns and bulls back for the wood
        t = smoothstep(3, 40, s)
        start = stalk + to * 0.6
        home = (FH_HOLD - start)
        home.z = 0.0
        away = home.normalized() if home.length > 1e-3 else -to
        fwd = to.slerp(away, t).normalized()
        dist = min(home.length * 0.8, 0.8 * smoothstep(8, 60, s) + 0.01 * max(0.0, s - 60))
        pos = start + away * dist + Vector((0, 0, -0.08 * t))
        bnd = 0.32 * math.sin(math.pi * min(1.0, max(0.0, (s - 3) / 26)))
        amp = max(amp, 0.02 * smoothstep(3, 10, s))
        rate = max(rate, 8.0)
    return pos, fwd, amp, rate, bnd, jaw


def animate_flathead(fh, f_lo, f_hi):
    root, beat = fh["root"], fh["beat"]
    phase, prev = 0.0, None
    mouth = {}
    for f in range(f_lo, f_hi + 1):
        a = act(f)
        pos, fwd, amp, rate, bnd, jaw = fh_state(a)
        phase += rate * ((a - prev) if prev is not None else 1.0) / FPS
        prev = a
        root.location = pos
        root.rotation_quaternion = heading_quat(fwd, 0.0)
        key(root, "location", f)
        key(root, "rotation_quaternion", f)
        beat.location = (phase, amp / FH_SCALE, bnd)
        key(beat, "location", f)
        fh["jaw"].rotation_euler = (0.0, math.radians(38) * jaw, 0.0)
        key(fh["jaw"], "rotation_euler", f)
        sc_ = max(0.05, min(1.0, jaw * 1.3))
        fh["throat"].scale = (1.0, 0.7 + 0.3 * sc_, sc_)
        fh["throat"].location = (FH_L / 2 - 0.085, 0.0, -0.028 - 0.006 * jaw)
        key(fh["throat"], "scale", f)
        key(fh["throat"], "location", f)
        for s_, po in zip((1, -1), fh["pects"]):
            po.rotation_euler = (s_ * math.radians(-6), math.radians(4), s_ * math.radians(-30 - 12 * math.sin(phase * 0.8)))
            key(po, "rotation_euler", f)
        m = Matrix.LocRotScale(pos, heading_quat(fwd), Vector((FH_SCALE,) * 3))
        mouth[f] = m @ Vector((FH_L / 2 + 0.01, 0.0, -0.02))
    return mouth


def circle_hook(bm, size=0.05, wire=0.0014, mat=0):
    """Circle hook (about an 8/0): shank, a round bend, and the point turned in towards the shank, with an eye."""
    pts = [(0.0, 0.0, size * 0.55), (0.0, 0.0, size * 0.1)]
    r = size * 0.32
    cx, cz = r, size * 0.1
    for i in range(1, 15):
        a = math.pi + math.pi * 1.25 * i / 14                    # around the bend and back up
        pts.append((cx + r * math.cos(a), 0.0, cz + r * math.sin(a)))
    tip = Vector(pts[-1])
    pts.append(tuple(tip + Vector((-r * 0.45, 0.0, r * 0.05))))  # the point turns in
    tube(bm, pts, lambda t: wire * (1 - 0.6 * max(0.0, t - 0.85) / 0.15), segs=6, cap=True, mat=mat)
    torus(bm, (0.0, 0.0, size * 0.6), (1, 0, 0), size * 0.06, wire * 0.8, n=12, m=5, mat=mat)


def make_rig(col, M):
    """Live bluegill on an 8/0 circle hook (through the back, in front of the dorsal fin), a fluorocarbon
    leader to a barrel swivel, a bead, and an egg sinker sliding on the main line."""
    me, L = bluegill_mesh(M)
    gill = bpy.data.objects.new("LiveBluegill", me)
    col.objects.link(gill)
    gill.rotation_mode = "QUATERNION"
    s = BAIT_LEN / L
    gill.scale = (s, s, s)
    ng = swim_group()
    mod = gill.modifiers.new("Swim", "NODES")
    mod.node_group = ng
    for nm, val in (("Phase", 0.0), ("Amp", 0.12 * L), ("Rate", 9.0), ("Len", L)):
        ident = next(it.identifier for it in ng.interface.items_tree if getattr(it, "name", "") == nm)
        set_mod_input(mod, ident, val)
    hb = bmesh.new()
    circle_hook(hb)
    hm = fish_mesh("CircleHook", hb, [M.hook])
    hook = bpy.data.objects.new("CircleHook", hm)
    col.objects.link(hook)
    hook.parent = gill
    hook.location = (0.02 * L / 0.19, 0.0, 0.11 * L)        # just in front of the dorsal fin, above the back
    hook.scale = (1 / s, 1 / s, 1 / s)
    hook.rotation_euler = (0.0, math.radians(-15), 0.0)
    tie = empty_object("HookEye", col, size=0.01, parent=hook)
    tie.location = (0.0, 0.0, 0.03)
    sb = bmesh.new()
    ellipsoid(sb, (0, 0, 0), (0.028, 0.019, 0.019), subdiv=3)                    # 2 oz egg sinker
    sink = finish("EggSinker", sb, col, [M.lead])
    sink.location = SINKER
    sink.rotation_euler = (0.0, 0.0, math.atan2(-SWIVEL.y + SINKER.y, 1.0))
    for p in sink.data.polygons:
        p.use_smooth = True
    bb = bmesh.new()
    ellipsoid(bb, (0, 0, 0), (0.0055, 0.0055, 0.0055), subdiv=2)
    bead = finish("Bead", bb, col, [M.bead])
    bead.location = SINKER.lerp(SWIVEL, 0.55)
    wb = bmesh.new()
    bm_frustum(wb, (0, -0.009, 0), (0, 0.009, 0), 0.0035, 0.0035, segs=10)
    torus(wb, (0, -0.013, 0), (1, 0, 0), 0.004, 0.0009, n=10, m=4)
    torus(wb, (0, 0.013, 0), (1, 0, 0), 0.004, 0.0009, n=10, m=4)
    swivel = finish("BarrelSwivel", wb, col, [M.brass])
    swivel.location = SWIVEL
    for ob in (gill, hook, sink, bead, swivel):
        ob.visible_shadow = True
    return dict(gill=gill, hook=hook, tie=tie, sinker=sink, bead=bead, swivel=swivel, len=L)


def animate_rig(rig, mouth, f_lo, f_hi):
    """The bluegill kicks on the leader until the strike, then it is inside her mouth and the leader runs to her
    jaw. Keyed every frame on the action clock."""
    gill = rig["gill"]
    L = rig["len"]
    for f in range(f_lo, f_hi + 1):
        a = act(f)
        if a <= A_STRIKE - 3:
            hook, fwd, roll, kick = bait_state(a)
        else:
            hook0, fwd, roll, kick = bait_state(A_STRIKE - 3)
            k = smoothstep(A_STRIKE - 3, A_STRIKE, a) ** 1.5
            hook = hook0.lerp(mouth[f], k)
        q = heading_quat(fwd, roll)
        off = q @ Vector((0.02 * BAIT_LEN / 0.19, 0.0, 0.11 * BAIT_LEN))
        gill.location = hook - off
        gill.rotation_quaternion = q
        key(gill, "location", f)
        key(gill, "rotation_quaternion", f)
    fh = next(f for f in range(f_lo, f_hi + 1) if act(f) >= A_STRIKE + 1)
    for ob in (gill, rig["hook"]):
        ob.hide_render = False
        key(ob, "hide_render", 1)
        ob.hide_render = True
        key(ob, "hide_render", fh)
    return fh


def line_object(name, col, M, start, end, radius=0.0007):
    ng, gi, go = new_group(name + "GN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links

    def loc(ob):
        n = nodes.new("GeometryNodeObjectInfo")
        n.transform_space = "RELATIVE"
        n.inputs["Object"].default_value = ob
        return n.outputs["Location"]
    c = nodes.new("GeometryNodeCurvePrimitiveLine")
    links.new(loc(start), c.inputs["Start"])
    links.new(loc(end), c.inputs["End"])
    res = nodes.new("GeometryNodeResampleCurve")
    res.inputs["Count"].default_value = 24
    links.new(c.outputs["Curve"], res.inputs["Curve"])
    prof = nodes.new("GeometryNodeCurvePrimitiveCircle")
    prof.inputs["Resolution"].default_value = 5
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


def build_rig_lines(col, M, rig, mouth, f_lo, f_hi):
    """Leader from the swivel to the hook eye (then to her jaw); main line from the swivel up through the sinker
    and away upstream to the rod."""
    sw = empty_object("LeaderStart", col, size=0.01)
    sw.location = SWIVEL
    end = empty_object("LeaderEnd", col, size=0.01)
    far = empty_object("MainLineFar", col, size=0.2)
    sc = bpy.context.scene
    for f in range(f_lo, f_hi + 1):
        sc.frame_set(f)
        a = act(f)
        p = rig["tie"].matrix_world.translation.copy() if a < A_STRIKE + 1 else mouth[f]
        end.location = p
        key(end, "location", f)
        far.location = LINE_FAR
        key(far, "location", f)
    leader = line_object("Leader", col, M, sw, end, radius=0.0006)
    main = line_object("MainLineUnder", col, M, sw, far, radius=0.0006)
    return leader, main


def build_vibration_rings(col, M, rig, spans):
    """Pressure waves from the kicking bait (a visual aid, not to scale): rings that spread and fade."""
    mat = new_material("VibrationRing", emissive_alpha(srgb(255, 214, 140)), blended=True)
    rings = []
    for i in range(10):
        bm = bmesh.new()
        torus(bm, (0, 0, 0), (0, 0, 1), 1.0, 0.012, n=48, m=4)
        ob = finish(f"VibrationRing{i}", bm, col, [mat], recalc=False)
        ob.visible_shadow = False
        ob.color = (1, 1, 1, 0)
        rings.append(ob)
    sc = bpy.context.scene
    starts = []
    for a, b in spans:
        starts += list(range(a, b - 30, 16))
    for i, f0 in enumerate(starts):
        ob = rings[i % len(rings)]
        for f in range(f0, f0 + 32):
            sc.frame_set(f)
            u = (f - f0) / 31.0
            ob.location = rig["gill"].matrix_world.translation
            r = 0.05 + 0.65 * u
            ob.scale = (r, r, r)
            ob.color[3] = 0.55 * (1 - u) ** 1.5 * smoothstep(0.0, 0.1, u)
            key(ob, "location", f)
            key(ob, "scale", f)
            key(ob, "color", f, index=3)
        ob.color[3] = 0.0
        key(ob, "color", f0 - 1, index=3)
        key(ob, "color", f0 + 32, index=3)
    return rings


def suction_debris(mouth, f_strike):
    """Silt and bubbles pulled into her mouth from all around as it opens."""
    rng = random.Random(77)
    deb = Particles()
    p = mouth[f_strike]
    t0 = (A_STRIKE - 7) / FPS
    for _ in range(420):
        d = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
        if d.length > 1 or d.length < 0.2:
            continue
        r = rng.uniform(0.1, 0.45)
        start = p + d.normalized() * r
        life = rng.uniform(0.12, 0.3)
        deb.emit(t0 + rng.uniform(0.0, 0.2), start, (p - start) / life, life, rng.uniform(0.0015, 0.0035))
    bub = Particles()
    for _ in range(90):
        start = p + Vector((rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(-0.03, 0.06)))
        bub.emit((A_STRIKE + rng.uniform(2, 14)) / FPS, start, Vector((rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1),
                                                                      rng.uniform(0.25, 0.6))), rng.uniform(1.0, 2.2),
                 rng.uniform(0.002, 0.007))
    return deb, bub


# --------------------------------------------------------------------------
# Above water: the boat, the angler, the rod bowed by the fish
# --------------------------------------------------------------------------
BOAT = Vector((river_cx(SET2.y), SET2.y, 0.0))      # anchored upstream of the hole, bow into the current
BOAT_YAW = -math.pi / 2                             # the hull is modelled bow +X; this turns the bow upstream (-Y)


def boat_pt(local):
    return BOAT + Matrix.Rotation(BOAT_YAW, 3, "Z") @ Vector(local)


ANGLER_AT = boat_pt((-1.9, 0.2, 0.40))              # standing on the rear deck, facing downstream to the hole
ANGLER_FACING = Vector((0.12, 1.0, 0.0)).normalized()
WATER_ENTRY = BOAT + Vector((1.4, 10.5, 0.0))       # where the tight line cuts into the water
LINE_FAR = WATER_ENTRY.copy()                       # the underwater main line runs up to the entry point
ROD_LEN = 2.29                                     # 7 ft 6 in heavy rod


def water_waves_group():
    ng, gi, go = new_group("LakeSurfaceGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links
    grid = nodes.new("GeometryNodeMeshGrid")
    grid.inputs["Size X"].default_value = 100.0
    grid.inputs["Size Y"].default_value = 130.0
    grid.inputs["Vertices X"].default_value = 300
    grid.inputs["Vertices Y"].default_value = 380
    x, y, _z = gn_xyz(ng)
    t = gn_time(ng)
    h = 0.0
    for kx, ky, w, a in ((0.25, 0.71, 1.1, 0.02), (-0.31, 0.92, 1.6, 0.015), (1.3, -0.9, 2.6, 0.008), (2.2, 1.7, 3.4, 0.005)):
        arg = nmath(ng, "SUBTRACT", nmath(ng, "ADD", nmath(ng, "MULTIPLY", x, kx), nmath(ng, "MULTIPLY", y, ky)), nmath(ng, "MULTIPLY", t, w))
        term = nmath(ng, "MULTIPLY", nmath(ng, "SINE", arg), a)
        h = term if isinstance(h, float) else nmath(ng, "ADD", h, term)
    sp = nodes.new("GeometryNodeSetPosition")
    links.new(grid.outputs["Mesh"], sp.inputs["Geometry"])
    links.new(combine(ng, 0.0, 0.0, h), sp.inputs["Offset"])
    sm = nodes.new("GeometryNodeSetShadeSmooth")
    links.new(sp.outputs[0], sm.inputs["Geometry"])
    links.new(sm.outputs[0], go.inputs["Geometry"])
    return ng


def build_lake(col, M):
    """The river surface (rippling, reflecting the dusk sky), the land beyond the modelled banks, far wooded hills."""
    ob = bpy.data.objects.new("RiverSurface", bpy.data.meshes.new("RiverSurface"))
    col.objects.link(ob)
    ob.modifiers.new("Waves", "NODES").node_group = water_waves_group()
    ob.data.materials.append(M.water_top)
    ng = ob.modifiers["Waves"].node_group
    setm = ng.nodes.new("GeometryNodeSetMaterial")
    setm.inputs["Material"].default_value = M.water_top
    out = next(n for n in ng.nodes if n.bl_idname == "NodeGroupOutput")
    src = out.inputs[0].links[0].from_socket
    ng.links.new(src, setm.inputs["Geometry"])
    ng.links.new(setm.outputs[0], out.inputs[0])
    ob.location = (3.0, -5.0, 0.0)
    ob.visible_shadow = False
    bm = bmesh.new()                                      # land beyond the modelled terrain (a frame around it)
    x0, x1, y0, y1, R, z = -40.0, 46.0, -60.0, 50.0, 700.0, 1.75
    for a_, b_ in (((-R, -R), (R, y0)), ((-R, y1), (R, R)), ((-R, y0), (x0, y1)), ((x1, y0), (R, y1))):
        vs = [bm.verts.new((a_[0], a_[1], z)), bm.verts.new((b_[0], a_[1], z)), bm.verts.new((b_[0], b_[1], z)),
              bm.verts.new((a_[0], b_[1], z))]
        bm.faces.new(vs)
    land = finish("FarLand", bm, col, [M.terrain], recalc=False)
    land.visible_shadow = False
    rng = random.Random(41)
    bm = bmesh.new()
    n, rings = 180, []
    for j, (r, hk) in enumerate(((330.0, 0.0), (350.0, 0.55), (420.0, 1.0), (560.0, 0.8), (800.0, 0.0))):
        ring = []
        for i in range(n):
            a = 2 * math.pi * i / n
            hh = 18.0 + 10.0 * math.sin(a * 3 + 1.0) + 7.0 * math.sin(a * 7 + 2.0) + 4.0 * mnoise.noise(Vector((math.cos(a) * 4, math.sin(a) * 4, 0)))
            rr = r + 40.0 * math.sin(a * 2 + 0.4)
            ring.append(bm.verts.new((rr * math.cos(a), rr * math.sin(a), 1.5 + hh * hk)))
        rings.append(ring)
    for ra, rb in zip(rings, rings[1:]):
        for i in range(n):
            bm.faces.new((ra[i], ra[(i + 1) % n], rb[(i + 1) % n], rb[i]))
    hills = finish("FarHills", bm, col, [M.canopy])
    hills.visible_shadow = False
    _ = rng
    return [ob, land, hills]


def build_boat(col, M):
    """Bass boat: V hull with chines, flat casting decks, console, seats, outboard and bow trolling motor."""
    bm = bmesh.new()
    zl = bm.verts.layers.float.new("zrel")
    xl = bm.verts.layers.float.new("xrel")
    stations = []
    for i in range(25):
        u = i / 24                                   # 0 stern .. 1 bow
        x = -3.1 + 6.2 * u
        beam = 1.15 * (1 - (max(0.0, u - 0.55) / 0.45) ** 1.8 * 0.92)
        keel = -0.28 + 0.22 * max(0.0, u - 0.7) / 0.3
        deck = 0.36 + 0.06 * max(0.0, u - 0.7) / 0.3
        chine = keel + 0.14
        pts = [(0.0, keel), (beam * 0.82, chine), (beam, chine + 0.06), (beam * 1.02, deck)]
        ring = []
        for y, z in pts[::-1]:
            ring.append(bm.verts.new((x, -y, z)))
        for y, z in pts[1:]:
            ring.append(bm.verts.new((x, y, z)))
        for v in ring:
            v[zl] = (v.co.z + 0.3) / 0.75
            v[xl] = u
        stations.append(ring)
    for ra, rb in zip(stations, stations[1:]):
        for k in range(len(ra) - 1):
            bm.faces.new((ra[k], ra[k + 1], rb[k + 1], rb[k]))
    bm.faces.new(stations[0])
    for ra, rb in zip(stations, stations[1:]):
        bm.faces.new((ra[0], ra[-1], rb[-1], rb[0])).material_index = 1
    for f in bm.faces:
        f.smooth = False
    hull = finish("BoatHull", bm, col, [M.hull, M.deck])
    hull.location = BOAT
    hull.rotation_euler = (0, 0, BOAT_YAW)
    rng = random.Random(2)
    bm = bmesh.new()
    bm_box(bm, (0.15, 0.62, 0.62), (0.7, 0.5, 0.5))                  # console
    bm_box(bm, (0.15, 0.62, 0.92), (0.66, 0.06, 0.12), rot=(0.4, 0, 0))   # windscreen
    for x, y in ((-0.7, 0.45), (-0.7, -0.45), (2.6, 0.0)):
        bm_frustum(bm, (x, y, 0.36), (x, y, 0.72), 0.03, 0.03, segs=10)
    for x, y in ((-0.7, 0.45), (-0.7, -0.45)):
        bm_box(bm, (x, y, 0.8), (0.46, 0.46, 0.14))
    seats = finish("BoatSeats", bm, col, [M.seat])
    seats.location = BOAT
    seats.rotation_euler = (0, 0, BOAT_YAW)
    bm = bmesh.new()
    ellipsoid(bm, (-3.42, 0.0, 0.78), (0.32, 0.27, 0.36), subdiv=3)
    bm_frustum(bm, (-3.38, 0.0, 0.45), (-3.36, 0.0, -0.6), 0.09, 0.07, segs=12)
    bm_box(bm, (-3.3, 0.0, -0.62), (0.4, 0.07, 0.12))
    bm_frustum(bm, (3.0, 0.38, 0.45), (3.0, 0.38, -0.7), 0.018, 0.018, segs=8)     # trolling motor shaft
    ellipsoid(bm, (3.0, 0.38, -0.74), (0.16, 0.05, 0.05), subdiv=2)
    motor = finish("Outboard", bm, col, [M.motor])
    motor.location = BOAT
    motor.rotation_euler = (0, 0, BOAT_YAW)
    for f in motor.data.polygons:
        f.use_smooth = True
    return [hull, seats, motor]


def import_glb(path, **kw):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path, **kw)
    return [o for o in bpy.data.objects if o not in before]


def attr_vec(nt, name):
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_name = name
    return a.outputs["Vector"]


ANGLER_GLB = os.path.join(HERE, "assets", "khronos", "CesiumMan.glb")


def build_angler(col, M):
    """CesiumMan (Khronos sample, CC BY 4.0) re-dressed as the angler: the mesh is recoloured into a
    performance shirt, trousers and boots by its rest height, the logo head is replaced by a modelled head,
    cap and sunglasses, and IK puts both hands on the rod."""
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
        for n in (m.node_tree.nodes if m and m.node_tree else []):
            if n.bl_idname == "ShaderNodeTexImage" and n.image:
                bpy.data.images.remove(n.image)
    me.materials.clear()

    def clothes(nt):
        r = separate(nt, attr_vec(nt, "rest"))
        ax = nmath(nt, "ABSOLUTE", r["X"])
        hand_m = map_range(nt, ax, 0.40, 0.43)
        shirt_m = map_range(nt, r["Z"], 0.62, 0.66)
        boot_m = map_range(nt, r["Z"], 0.11, 0.08)
        n = noise(nt, attr_vec(nt, "rest"), 60.0)
        col_ = mix_rgb(nt, shirt_m, srgb(64, 70, 74), srgb(70, 120, 150))
        col_ = mix_rgb(nt, boot_m, col_, srgb(40, 38, 36))
        col_ = mix_rgb(nt, hand_m, col_, srgb(196, 140, 106))
        col_ = vmath(nt, "SCALE", col_, scale=nmath(nt, "MULTIPLY_ADD", n, 0.25, 0.88))
        return psdf(nt, col_, rough=0.7, coat=0.1, normal=bump(nt, n, 0.25))
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
    arm["base_q"] = list(arm.matrix_world.to_quaternion())
    arm.location = ANGLER_AT
    arm.scale = (1.24, 1.24, 1.24)                     # 1.45 m model -> 1.8 m angler
    key_lean(arm, 1, 0.0, keyframe=False)
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


def build_rod(col, M):
    """Heavy catfish rod: a quadratic Bezier blank (butt, middle, tip empties keyed per frame), cork grip
    and a low profile baitcasting reel on top."""
    ctrl = {k: empty_object(f"Rod{k}", col, size=0.05) for k in ("Butt", "Mid", "Tip")}
    ng, gi, go = new_group("RodGN", "GeometryNodeTree", [("Geometry", "NodeSocketGeometry")],
                           [("Geometry", "NodeSocketGeometry")])
    nodes, links = ng.nodes, ng.links

    def loc(ob):
        n = nodes.new("GeometryNodeObjectInfo")
        n.transform_space = "RELATIVE"
        n.inputs["Object"].default_value = ob
        return n.outputs["Location"]
    c = nodes.new("GeometryNodeCurveQuadraticBezier")
    c.inputs["Resolution"].default_value = 48
    links.new(loc(ctrl["Butt"]), c.inputs["Start"])
    links.new(loc(ctrl["Mid"]), c.inputs["Middle"])
    links.new(loc(ctrl["Tip"]), c.inputs["End"])
    rad = nodes.new("GeometryNodeSetCurveRadius")
    links.new(c.outputs["Curve"], rad.inputs["Curve"])
    par = nodes.new("GeometryNodeSplineParameter")
    links.new(map_range(ng, par.outputs["Factor"], 0.0, 1.0, 1.0, 0.16, smooth=False), rad.inputs["Radius"])
    prof = nodes.new("GeometryNodeCurvePrimitiveCircle")
    prof.inputs["Resolution"].default_value = 10
    prof.inputs["Radius"].default_value = 0.0075
    c2m = nodes.new("GeometryNodeCurveToMesh")
    links.new(rad.outputs[0], c2m.inputs["Curve"])
    links.new(prof.outputs["Curve"], c2m.inputs["Profile Curve"])
    setm = nodes.new("GeometryNodeSetMaterial")
    setm.inputs["Material"].default_value = M.rod
    links.new(c2m.outputs[0], setm.inputs["Geometry"])
    links.new(setm.outputs[0], go.inputs["Geometry"])
    ob = bpy.data.objects.new("Rod", bpy.data.meshes.new("Rod"))
    col.objects.link(ob)
    ob.modifiers.new("Rod", "NODES").node_group = ng
    mount = empty_object("ReelMount", col, size=0.08)
    mount.rotation_mode = "QUATERNION"
    bm = bmesh.new()
    bm_frustum(bm, (0, -0.05, 0.0), (0, 0.06, 0.0), 0.016, 0.016, segs=14)      # grip section around the blank
    cork = mesh_object("RodGrip", bm, col, [M.cork], parent=mount, recalc=False)
    bm = bmesh.new()
    ellipsoid(bm, (0.0, 0.05, 0.035), (0.034, 0.045, 0.026), subdiv=2)
    bm_frustum(bm, (0.034, 0.05, 0.035), (0.06, 0.05, 0.035), 0.01, 0.006, segs=8)     # handle
    reel = mesh_object("Reel", bm, col, [M.reel], parent=mount, recalc=False)
    for o in (ob, cork, reel):
        o.visible_shadow = True
    return dict(ctrl=ctrl, ob=ob, mount=mount)


def hand_point(f):
    A = ANGLER_AT
    F = ANGLER_FACING
    R = F.cross(Vector((0, 0, 1))).normalized()
    pump = 0.05 * math.sin(f * 0.21) + 0.03 * math.sin(f * 0.53)
    return A + F * (0.20 - pump) + R * 0.12 + Vector((0, 0, 1.10 + 0.03 * math.sin(f * 0.17)))


def rod_pose(f):
    """Butt, direction and bend: rod held high and bowed hard towards the fish (it is turning her head)."""
    H = hand_point(f)
    flat = Vector((WATER_ENTRY.x - H.x, WATER_ENTRY.y - H.y, 0.0)).normalized()
    side = flat.cross(Vector((0, 0, 1))).normalized()
    swing = math.radians(-14.0 + 6.0 * math.sin(f * 0.05))
    flat = (Matrix.Rotation(swing, 3, "Z") @ flat).normalized()
    el = math.radians(62.0 + 6.0 * math.sin(f * 0.13))
    d = (flat * math.cos(el) + Vector((0, 0, math.sin(el)))).normalized()
    bend = 0.86 + 0.05 * math.sin(f * 0.41) + 0.03 * math.sin(f * 0.97)
    _ = side
    return H - d * 0.38, d, bend


def place_rod(rod, f, butt, d, load_dir, bend):
    d = d.normalized()
    tip_dir = d.slerp(load_dir.normalized(), min(0.97, bend)) if hasattr(d, "slerp") else d
    butt = Vector(butt)
    mid = butt + d * (ROD_LEN * 0.5)
    tip = mid + (d * 0.3 + tip_dir * 0.7).normalized() * (ROD_LEN * 0.5) * (1.0 - 0.1 * bend)
    for k, p in (("Butt", butt), ("Mid", mid), ("Tip", tip)):
        rod["ctrl"][k].location = p
        key(rod["ctrl"][k], "location", f)
    side = d.cross(Vector((0, 0, 1)))
    side = side.normalized() if side.length > 1e-3 else Vector((1, 0, 0))
    up = side.cross(d).normalized()
    rod["mount"].location = butt + d * 0.36
    rod["mount"].rotation_quaternion = Matrix((side, d, up)).transposed().to_quaternion()
    key(rod["mount"], "location", f)
    key(rod["mount"], "rotation_quaternion", f)
    return tip, mid


def bezier_point(a, b, c, t):
    return a * (1 - t) ** 2 + b * 2 * (1 - t) * t + c * t * t


def animate_fight(rod, ang, f_lo, f_hi):
    tips, mids, butts = {}, {}, {}
    for f in range(f_lo, f_hi + 1):
        butt, d, bend = rod_pose(f)
        H = hand_point(f)
        load = WATER_ENTRY - H
        tips[f], mids[f] = place_rod(rod, f, butt, d, load, bend)
        butts[f] = butt
        for s, off in (("R", 0.36), ("L", 0.05)):
            ang["hands"][s].location = butt + d * off
            key(ang["hands"][s], "location", f)
        lean = math.radians(12 + 4 * math.sin(f * 0.07))       # positive leans back, away from the fish
        key_lean(ang["arm"], f, lean)
    return tips, mids, butts


def build_surface_line(col, M, rod, f_lo, f_hi):
    far = empty_object("LineEntry", col, size=0.1)
    for f in range(f_lo, f_hi + 1):
        far.location = WATER_ENTRY + Vector((0.25 * math.sin(f * 0.09), 0.18 * math.sin(f * 0.13), -0.02))
        key(far, "location", f)
    return line_object("MainLineAbove", col, M, rod["ctrl"]["Tip"], far, radius=0.0011), far


def entry_spray(f_lo, f_hi):
    """Spray and a little hiss where the tight line slices through the surface."""
    rng = random.Random(8)
    sp = Particles()
    for f in range(f_lo, f_hi + 1):
        for _ in range(5):
            p = WATER_ENTRY + Vector((0.25 * math.sin(f * 0.09), 0.18 * math.sin(f * 0.13), 0.0))
            sp.emit(f / FPS, p + Vector((rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0.0)),
                    (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(0.6, 1.6)), rng.uniform(0.25, 0.5),
                    rng.uniform(0.004, 0.009), 9.81)
    return sp


# --------------------------------------------------------------------------
# Camera: frames[f] = (position, target, rotation, lens); f-stop per frame
# --------------------------------------------------------------------------
def P(clip, i):
    return phrase_frame(clip, i)


CTA_F = VO_START[5] + int(round(VO_LEN[5] * FPS * 0.42))      # "... and hit subscribe"


def smooth_vectors(vals, k):
    arr = np.array([list(v) for v in vals])
    out = arr.copy()
    for i in range(len(arr)):
        lo, hi = max(0, i - k), min(len(arr), i + k + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    return [Vector(v) for v in out]


def bait_centre(f):
    hook, fwd, _r, _k = bait_state(act(f))
    return hook + Vector((0, 0, -0.03))


def camera_frames():
    frames, fstop = {}, {}
    Z = Vector((0, 0, 1))
    # 1 hook: dolly zoom through the jam onto her head (lens widens as the camera closes in, so she keeps her size)
    dc = (Matrix.Rotation(math.radians(14), 3, "Z") @ F0).normalized()
    dc = (dc + Vector((0, 0, 0.14))).normalized()
    centre = FH_HOLD + F0 * 0.42 + Vector((0, 0, 0.03))
    for f in range(1, CUT1):
        u = ease((f - 1) / (CUT1 - 2))
        lens = 85.0 * (22.0 / 85.0) ** u
        d = 3.0 * lens / 85.0
        p = centre + dc * d
        frames[f] = (p, centre, look_at(p, centre, Z), lens)
        fstop[f] = 4.0
    # 2 habitat: aerial down the river bend, then (under the wipe) a descent into the scour hole
    k2a = [(CUT1, (-9.0, -34.0, 27.0), (3.0, -4.0, 0.0), True),
           ((CUT1 + DIVE) // 2, (-3.5, -17.0, 14.0), (4.5, -1.0, -1.0), False),
           (DIVE - 1, (2.0, -7.5, 4.0), (4.6, 0.5, -2.0), True)]
    for f, (p, t) in hermite_path(k2a).items():
        frames[f] = (p, t, look_at(p, t, Z), 24.0)
        fstop[f] = 11.0
    s3 = P(2, 3)
    k2b = [(DIVE, (1.6, -6.4, -0.9), (2.6, -2.6, -4.4), True),
           (s3 + 6, (2.2, -5.4, -1.6), (3.0, -0.8, -5.2), False),
           (s3 + int((CUT2 - s3) * 0.62), (4.4, -3.6, -1.5), (8.7, 1.0, -1.1), False),
           (CUT2 - 1, (3.0, -2.2, -3.5), tuple(FH_HOLD + Vector((0, 0, 0.1))), True)]
    for f, (p, t) in hermite_path(k2b).items():
        frames[f] = (p, t, look_at(p, t, Z), 24.0)
        fstop[f] = 4.0
    # 3 bait and rig: macro orbit around the kicking bluegill, the jam behind it; wide for "live, struggling"
    s1, s2 = P(3, 1), P(3, 2)
    raw = [bait_centre(f) for f in range(CUT2 - 40, CUT3 + 40)]
    sm = smooth_vectors(raw, 24)
    base = {f: sm[f - (CUT2 - 40)] for f in range(CUT2, CUT3)}
    for f in range(CUT2, CUT3):
        b = base[f]
        u = (f - CUT2) / max(1, CUT3 - 1 - CUT2)
        wide = smoothstep(s1 - 16, s1 + 14, f) * (1 - smoothstep(s2 - 14, s2 + 16, f))
        macro = smoothstep(s2 - 14, s2 + 16, f)
        th = math.radians(-112 + 40 * ease(u))                  # side on to the bait, from the channel side
        r = 0.85 + 0.6 * wide + 0.4 * macro
        p = b + Vector((r * math.sin(th), -r * math.cos(th), 0.12 + 0.1 * wide))
        t = b.lerp(FH_HOLD + Vector((0, 0, 0.1)), 0.18 * wide).lerp(SWIVEL, 0.45 * macro)
        p.z = max(p.z, ground_z(p.x, p.y) + 0.35)
        lens = 38.0 - 8.0 * wide - 2.0 * macro
        frames[f] = (p, t, look_at(p, t, Z), lens)
        fstop[f] = 2.8 + 1.4 * wide
    # 4 the strike: low and side on between the wood and the bait; then follows her back to cover
    b0 = bait_centre(CUT3)
    side = Vector((-1.0, 0.25, 0.0)).normalized()                 # the channel side, looking towards the bank
    mid = b0.lerp(FH_HOLD, 0.45)
    for f in range(CUT3, SURF):
        a = act(f)
        s = a - A_STRIKE
        push = smoothstep(-60, -4, s)
        p = mid + side * (1.9 - 0.3 * push) + Vector((0, 0, 0.45))
        t = mid.lerp(E_STRIKE, 0.4 + 0.3 * push)
        jolt = math.exp(-max(0.0, s) / 4.0) * (s > 0) * 0.03
        p += Vector((math.sin(f * 2.1), math.cos(f * 1.7), math.sin(f * 1.3))) * jolt
        if s > 4:
            k = smoothstep(4, 50, s)
            pos, *_ = fh_state(a)
            t = t.lerp(pos + Vector((0, 0, 0.1)), k)
            p = p + Vector((0, 0, 0.35 * k)) + side * (0.3 * k)
        frames[f] = (p, t, look_at(p, t, Z), 24.0)
        fstop[f] = 4.0
    return frames, fstop


def surface_camera(frames, fstop, mids, tips):
    """4b and 5, above water: over the angler's shoulder on the loaded rod, then a rapid pull back to the boat."""
    Z = Vector((0, 0, 1))
    near = {}
    fs = list(range(SURF, FRAME_END + 1))
    smooth_hand = dict(zip(fs, smooth_vectors([hand_point(f) for f in fs], 18)))
    for f in fs:
        tip = tips[f]
        H = hand_point(f)
        flat = Vector((WATER_ENTRY.x - H.x, WATER_ENTRY.y - H.y, 0.0)).normalized()
        side_ = flat.cross(Z).normalized()
        shake = Vector((math.sin(f * 0.9) * 0.6 + math.sin(f * 2.3) * 0.4, math.sin(f * 1.1), math.sin(f * 1.7))) * 0.006
        pn = smooth_hand[f] - flat * 1.35 + side_ * 0.55 + Vector((0, 0, 0.3)) + shake
        tn = tip.lerp(WATER_ENTRY, 0.1) + Vector((0, 0, -0.3))
        near[f] = (pn, tn)
    pw = ANGLER_AT + Vector((-7.5, 7.0, 3.2))             # downstream and to the side: the angler faces the camera
    tw = ANGLER_AT + Vector((0.4, 1.6, 3.2))              # aims high: the boat sits below the recap and the call to action
    pw2 = pw + Vector((-1.5, 1.2, 0.6))
    fq = CUT4 + 34
    for f in fs:
        pn, tn = near[f]
        if f < CUT4:
            p, t, lens = pn, tn, 30.0
        elif f <= fq:
            x = (f - CUT4) / (fq - CUT4)
            u = ease_out(x, 3.0) * smoothstep(0.0, 0.16, x)
            p, t = pn.lerp(pw, u), tn.lerp(tw, u)
            lens = 30.0 - 6.0 * u
        else:
            u = ease((f - fq) / max(1, FRAME_END - fq))
            p, t, lens = pw.lerp(pw2, u), tw, 24.0
        frames[f] = (p, t, look_at(p, t, Z), lens)
        fstop[f] = 5.6 if f < CUT4 else 8.0


def build_camera(col, frames, fstop):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = LENS
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = SENSOR
    cd.clip_start = 0.02
    cd.clip_end = 3000.0
    cd.dof.use_dof = True
    cd.dof.aperture_blades = 7
    cam = bpy.data.objects.new("Camera", cd)
    col.objects.link(cam)
    cam.rotation_mode = "QUATERNION"
    for f in range(1, FRAME_END + 1):
        p, t, q, lens = frames[f]
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
    bpy.context.scene.camera = cam
    return cam

# --------------------------------------------------------------------------
# Scene state per frame: under water or not, the hook's gloom and eyeshine, the dusk sun, the hook's rim light
# --------------------------------------------------------------------------
def under_at(f):
    return 0.0 if (CUT1 <= f < DIVE) or f >= SURF else 1.0


def gloom_at(f):
    if f < CUT1:
        return 0.55 - 0.35 * ease((f - 1) / (CUT1 - 2))
    return 0.15                                             # dusk: the river stays dim


def set_constant(idb):
    for fc in iter_fcurves(idb):
        for kp in fc.keyframe_points:
            kp.interpolation = "CONSTANT"


def key_scene_state(sc, world_vals, sun, rim, key_l):
    under_v, gloom_v = world_vals
    for f in range(1, FRAME_END + 1):
        u, g = under_at(f), gloom_at(f)
        shine = 1.6 if f < CUT1 else (0.8 if CUT3 <= f < SURF else 0.0)
        for prop, val in (("under", u), ("gloom", g), ("eyeshine", shine)):
            sc[prop] = val
            sc.keyframe_insert(data_path=f'["{prop}"]', frame=f)
        under_v.outputs[0].default_value = u
        gloom_v.outputs[0].default_value = g
        under_v.outputs[0].keyframe_insert("default_value", frame=f)
        gloom_v.outputs[0].keyframe_insert("default_value", frame=f)
        sun.data.energy = (4.5 * (1 - 0.4 * g)) if u else 2.8
        sun.data.keyframe_insert("energy", frame=f)
        sun.rotation_quaternion = (-(SUN_UNDER if u else SUN_DIR)).to_track_quat("-Z", "Y")
        sun.keyframe_insert("rotation_quaternion", frame=f)
        rim.data.energy = 14.0 if f < CUT1 else 0.0
        rim.data.keyframe_insert("energy", frame=f)
        key_l.data.energy = 30.0 if f < CUT1 else 0.0
        key_l.data.keyframe_insert("energy", frame=f)
    set_constant(sc)
    set_constant(sc.world.node_tree)
    set_constant(sun)


def build_sun(col):
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 2.6
    light.color = srgb(255, 196, 140)                      # low, warm dusk sun
    light.angle = math.radians(1.5)
    ob = bpy.data.objects.new("Sun", light)
    col.objects.link(ob)
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (-SUN_DIR).to_track_quat("-Z", "Y")
    return ob


def refracted_sun():
    """Direction towards the light as it arrives under water: Snell's law bends a sun 9 degrees up to about
    48 degrees from vertical."""
    h = Vector((SUN_DIR.x, SUN_DIR.y, 0.0)).normalized()
    ang = math.asin(min(1.0, math.cos(math.asin(SUN_DIR.z)) / 1.333))
    return (h * math.sin(ang) + Vector((0, 0, math.cos(ang)))).normalized()


SUN_UNDER = refracted_sun()


def build_key(col):
    """Soft key light from above and in front of her head for the hook (dusk water is too dark on its own)."""
    ld = bpy.data.lights.new("HookKey", "AREA")
    ld.shape = "DISK"
    ld.size = 1.5
    ld.color = srgb(255, 214, 170)
    ld.energy = 0.0
    ob = bpy.data.objects.new("HookKey", ld)
    col.objects.link(ob)
    loc = FH_HOLD + F0 * 1.6 + Vector((-0.4, 0.0, 1.5))
    ob.location = loc
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (FH_HOLD + F0 * 0.4 - loc).normalized().to_track_quat("-Z", "Y")
    return ob


def build_rim(col, frames):
    """Back light for the hook: it draws her broad head out of the dark wood."""
    ld = bpy.data.lights.new("HookRim", "AREA")
    ld.shape = "DISK"
    ld.size = 1.0
    ld.color = srgb(255, 200, 140)
    ld.energy = 0.0
    ob = bpy.data.objects.new("HookRim", ld)
    col.objects.link(ob)
    loc = FH_HOLD - F0 * 1.0 + Vector((0.3, 0.0, 0.9))
    ob.location = loc
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = (FH_HOLD + F0 * 0.3 - loc).normalized().to_track_quat("-Z", "Y")
    return ob


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------
def best_anchor(hud, cands, lo, hi, band=(-0.15, 0.28), xmax=0.2):
    """The candidate world point that stays longest inside the pin band over frames lo..hi."""
    def score(w):
        n = 0
        for f in range(lo, hi + 1, 2):
            s = hud.project(f, w)
            if s is not None and abs(s.x) < xmax and band[0] < s.y < band[1]:
                n += 1
        return n
    return max(cands, key=score)


def build_overlays(hud, st):
    hud.scrims()
    hud.kicker("THE FISHING SERIES")
    # 1 hook
    p1 = P(1, 1)
    hud.serif("Stop throwing *dead stink bait* on the bottom for flatheads.", -40, p1 - 9, y=-0.30, size=0.05,
              lines=["Stop throwing *dead stink bait*", "on the bottom for flatheads."], reveal=2.4)
    hud.serif("You're catching *channel cats,* not river monsters.", p1, CUT1 - 7, y=-0.30, size=0.05,
              lines=["You're catching *channel cats,*", "not river monsters."], reveal=2.6)
    # 2 habitat
    hud.badge(1, "APEX AMBUSH LAIR", "deep wood in the scour hole", max(CUT1 + 10, P(2, 1) - 6), CUT2 - 8)
    hud.serif("Flatheads aren't scavengers. They're *apex predators.*", max(CUT1 + 8, P(2, 0)), DIVE - 8,
              src="LIVE PREY ONLY: TEXAS PARKS AND WILDLIFE", lines=["Flatheads aren't scavengers.", "They're *apex predators.*"])
    hud.serif("By day they hide in *heavy cover.*", DIVE + 6, P(2, 3) - 9, src="SOURCE: MINNESOTA DNR",
              lines=["By day they hide in", "*heavy cover.*"])
    hud.serif("Root balls, logjams, *undercut banks.*", P(2, 3), CUT2 - 8, src="SOURCE: SOUTH CAROLINA DNR",
              lines=["Root balls, logjams,", "*undercut banks.*"])
    a = P(2, 3)
    span = CUT2 - 10 - a
    w = [(a + 4, a + int(span * 0.42)), (a + int(span * 0.3), a + int(span * 0.72)), (a + int(span * 0.58), CUT2 - 10),
         (a + int(span * 0.15), a + int(span * 0.55))]
    roots = [r[0] + Vector((0, 0, 0.6)) for r in st["roots"]]
    logs = [pts[k] + Vector((0, 0, rr + 0.05)) for pts, rr in LOGS if rr > 0.15 for k in (3, 5, 7)]
    lip = [Vector((river_cx(y) + RIVER_W * 0.97, y, z)) for y in (HOLE.y - 3, HOLE.y - 1.5, HOLE.y, HOLE.y + 1.5)
           for z in (-1.4, -0.9)]                                  # the eroded cavity under the lip
    floor = [Vector((HOLE.x + dx, HOLE.y + dy, ground_z(HOLE.x + dx, HOLE.y + dy) + 0.2))
             for dx in (-1.5, -0.5, 0.5) for dy in (-3, -1.5, 0, 1.5)]
    pr = best_anchor(hud, roots, *w[0])
    pl = best_anchor(hud, [p for p in logs if (p - pr).length > 1.5], *w[1])
    pu = best_anchor(hud, lip, *w[2])
    pf = best_anchor(hud, [p for p in floor if (p - pl).length > 1.2], *w[3])
    hud.pin(pr, "root ball", *w[0], dot_color=GOLD, y_band=(-0.2, 0.32))
    hud.pin(pl, "logjam", *w[1], dot_color=GOLD, side=-0.06, y_band=(-0.2, 0.32))
    hud.pin(pu, "undercut bank", *w[2], dot_color=GOLD, side=0.06, y_band=(-0.2, 0.32))
    hud.pin(pf, "20 ft scour hole", *w[3], dot_color=srgb(240, 190, 120), rise=0.05, side=0.08, y_band=(-0.2, 0.32))
    # 3 bait and rig
    hud.badge(2, "LIVE BAIT ONLY", "big, struggling, near the wood", CUT2 + 8, CUT3 - 8)
    hud.serif("At dusk they *leave the wood* to hunt.", CUT2 + 10, P(3, 1) - 9, src="NIGHT FEEDING: MINNESOTA DNR",
              lines=["At dusk they", "*leave the wood* to hunt."])
    hud.serif("Use *live, struggling* bait.", P(3, 1), P(3, 2) - 9, src="ANGLER TECHNIQUE", lines=["Use *live,*", "*struggling* bait."])
    hud.serif("6 to 8 inch bluegill, *8/0 circle hook.*", P(3, 2), CUT3 - 8,
              src="ANGLER TECHNIQUE: CHECK YOUR LOCAL BAIT RULES", lines=["6 to 8 inch bluegill,", "*8/0 circle hook.*"])
    rig = st["rig"]
    s2 = P(3, 2)
    hud.pin(lambda f: (bpy.context.scene.frame_set(f), rig["hook"].matrix_world.translation.copy())[1],
            "8/0 circle hook", s2 + 8, s2 + int((CUT3 - s2) * 0.62), dot_color=GOLD, side=-0.06)
    hud.pin(SINKER + Vector((0, 0, 0.02)), "egg sinker", s2 + int((CUT3 - s2) * 0.4), CUT3 - 10, dot_color=GOLD,
            side=0.07, y_band=(-0.32, 0.3))

    def gill_end(which):
        def fn(f):
            bpy.context.scene.frame_set(f)
            m = rig["gill"].matrix_world
            return m @ Vector((rig["len"] / 2 if which == 0 else -rig["len"] / 2, 0.0, 0.0))
        return fn
    hud.dimension(gill_end(0), gill_end(1), "6 TO 8 IN", s2 + 16, s2 + int((CUT3 - s2) * 0.9), drop=0.09)
    # 4 the strike, then the rod
    hud.badge(3, "STRUCTURAL HOLD", "let it load, then lock the drag", CUT3 + 8, CUT4 - 8)
    hud.serif("The vibrations *draw them out.*", CUT3 + 4, P(4, 1) - 6, src="LATERAL LINE: MINNESOTA DNR",
              lines=["The vibrations", "*draw them out.*"], reveal=2.4)
    hud.serif("Let the rod *load completely.*", SURF + 4, P(4, 2) - 10, src="ANGLER TECHNIQUE",
              lines=["Let the rod", "*load completely.*"])
    hud.serif("A bony mouth needs *maximum pressure.*", P(4, 2), CUT4 - 8, src="ANGLER TECHNIQUE",
              lines=["A bony mouth needs", "*maximum pressure.*"])
    mids, tips = st["mids"], st["tips"]
    hud.pin(lambda f: mids[f].lerp(tips[f], 0.3), "rod fully loaded", SURF + 10, SURF + int((CUT4 - SURF) * 0.55),
            dot_color=GOLD, side=-0.05)
    entry = st["entry"]

    def line_pt(f):
        bpy.context.scene.frame_set(f)
        return tips[f].lerp(entry.matrix_world.translation, 0.25)
    hud.pin(line_pt, "drag locked down", SURF + int((CUT4 - SURF) * 0.45), CUT4 - 6, dot_color=srgb(240, 190, 120),
            side=0.06, y_band=(-0.2, 0.4))
    # 5 outro
    hud.recap([(1, "APEX AMBUSH LAIR"), (2, "LIVE BAIT ONLY"), (3, "STRUCTURAL HOLD")], CUT4 + 10, FRAME_END + 10, gap=7,
              y0=0.27, step=0.08)
    hud.serif("Ditch the dead bait. *Target heavy wood.*", CUT4 + 4, FRAME_END + 10, y=-0.30, size=0.05, reveal=3.0,
              lines=["Ditch the dead bait.", "*Target heavy wood.*"])
    hud.cta("SUBSCRIBE FOR MORE TROPHY BREAKDOWNS", CTA_F, FRAME_END + 10, y=0.02)
    for wa, wb in WIPES:
        hud.wipe(wa, wb)


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
def run_checks(frames, st):
    sc = bpy.data.scenes["Main"]
    print("\n=== CAMERA REPORT ===")
    from mathutils import kdtree
    pts = []
    for nm in ("Logjam", "RootBallClay", "UndercutLip", "UndercutRoots", "BankTrunks", "BankCanopy"):
        ob = bpy.data.objects[nm]
        pts += [ob.matrix_world @ v.co for v in ob.data.vertices]
    kd = kdtree.KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert(p, i)
    kd.balance()
    worst = {"ground": (1e9, 0), "wood, bank, trees": (1e9, 0), "flathead": (1e9, 0), "bait": (1e9, 0),
             "surface (under)": (1e9, 0), "surface (above)": (1e9, 0), "boat": (1e9, 0), "angler/rod": (1e9, 0)}

    def upd(k, d, f):
        if d < worst[k][0]:
            worst[k] = (d, f)
    for f in range(1, FRAME_END + 1):
        p = frames[f][0]
        upd("ground", p.z - ground_z(p.x, p.y), f)
        upd("wood, bank, trees", kd.find(p)[2] - 0.15, f)
        if under_at(f):
            upd("surface (under)", -p.z, f)
        else:
            upd("surface (above)", p.z - 0.15, f)
        if f >= SURF:
            lo = boat_pt((-3.6, -1.3, -0.4))
            hi = boat_pt((3.2, 1.3, 1.0))
            upd("boat", aabb_dist(p, Vector(map(min, lo, hi)), Vector(map(max, lo, hi))), f)
    for f in range(1, FRAME_END + 1, 2):
        p = frames[f][0]
        if f < SURF and under_at(f):
            a = act(f)
            fp, *_ = fh_state(a)
            upd("flathead", (fp - p).length - 0.65, f)
            if CUT2 - 4 <= f:
                upd("bait", (bait_state(a)[0] - p).length - 0.12, f)
        elif f >= SURF:
            sc.frame_set(f)
            d = min((st["rod"]["ctrl"][k].matrix_world.translation - p).length for k in ("Mid", "Tip"))
            upd("angler/rod", min(d - 0.15, (ANGLER_AT + Vector((0, 0, 1.0)) - p).length - 0.6), f)
    for k, v in worst.items():
        print(f"closest {k}: {round(v[0], 3)} m at frame {v[1]} (negative = inside)")
    acc, turn = (0.0, 0), (0.0, 0)
    cuts = [CUT1, CUT2, CUT3, DIVE, SURF]
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
    cross = [f for f in range(2, FRAME_END + 1) if (frames[f][0].z < 0) != (frames[f - 1][0].z < 0)]
    print("water surface crossings:", cross, "all inside wipes:",
          all(any(wa - 2 <= f - 1 and f <= wb + 2 for wa, wb in WIPES) for f in cross))
    jumps = [f for f in range(2, FRAME_END + 1) if (frames[f][0] - frames[f - 1][0]).length > 0.6 and f not in cuts]
    print("camera jumps outside cuts:", jumps or "none")
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
    rod = st["rod"]
    subj = {1: ("flathead head", lambda f: fh_state(act(f))[0] + fh_state(act(f))[1] * 0.4),
            100: ("flathead head", lambda f: fh_state(act(f))[0] + fh_state(act(f))[1] * 0.4),
            CUT1 + 30: ("scour hole", lambda f: HOLE),
            264: ("scour hole", lambda f: HOLE),
            CUT2 - 20: ("flathead den", lambda f: FH_HOLD),
            528: ("bluegill", lambda f: bait_state(act(f))[0]),
            700: ("bluegill", lambda f: bait_state(act(f))[0]),
            912: ("flathead", lambda f: fh_state(act(f))[0]),
            STRIKE: ("mouth", lambda f: st["mouth"][f]),
            SURF + 20: ("rod middle", lambda f: rod["ctrl"]["Mid"].matrix_world.translation.copy()),
            1320: ("angler", lambda f: ANGLER_AT + Vector((0, 0, 1.2)))}
    print("\n=== SUBJECTS ON SCREEN (x in +-0.281, y in +-0.5) ===")
    for f, (nm, fn) in sorted(subj.items()):
        sc.frame_set(f)
        s = hud.project(f, fn(f))
        print(f"frame {f}: {nm} ->", None if s is None else (round(s.x, 3), round(s.y, 3)))


def write_events(path, st):
    data = {
        "fps": FPS, "frames": FRAME_END,
        "wipes": WIPES, "cuts": [CUT1, CUT2, CUT3, DIVE, SURF],
        "overlays": sorted(st["hud"].events),
        "strike": STRIKE, "slowmo": [STRIKE - 18, STRIKE + 24], "surface": SURF, "dive": DIVE,
        "aerial": [CUT1, DIVE], "kicks": [CUT2, STRIKE], "pullback": [CUT4, CUT4 + 34],
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
    for prop in ("under", "gloom", "eyeshine"):
        sc[prop] = 0.0
    c_env, c_life, c_tackle = collection("River"), collection("Fish"), collection("Tackle")
    c_above, c_fx, c_cam = collection("AboveWater"), collection("LightAndWater"), collection("Camera")
    atmos = build_atmos_group()
    M = build_materials(atmos)
    _w, w_under, w_gloom = build_world(sc)
    sun = build_sun(c_fx)
    clock = build_clock(c_fx)
    frames, fstop = camera_frames()
    terrain = build_terrain(c_env, M)
    keep = [(frames[f][0], 0.5) for f in range(1, SURF, 3) if under_at(f)]
    keep += [(bait_state(act(f))[0], 0.35) for f in range(CUT2, STRIKE, 6)]
    keep += [(fh_state(act(f))[0], 0.45) for f in range(CUT3, SURF, 4)]
    hook_path = [frames[f][0] for f in range(1, CUT1)]
    jam, rclay, roots = build_logjam(c_env, M, keep_clear=keep, frame_path=hook_path)
    lip, lroots = build_undercut(c_env, M)
    trunks, canopy = build_trees(c_above, M)
    shafts = build_shafts(c_fx, M)
    sediment = build_sediment(c_fx, M)
    surf_under = build_surface_under(c_fx, M)
    fh = make_flathead(c_life, M)
    mouth = animate_flathead(fh, 1, SURF + 3)
    rig = make_rig(c_tackle, M)
    animate_rig(rig, mouth, CUT2 - 6, SURF + 3)
    leader, main_line = build_rig_lines(c_tackle, M, rig, mouth, CUT2 - 6, SURF + 3)
    rings = build_vibration_rings(c_fx, M, rig, [(P(3, 1), P(3, 2) - 10), (CUT3, STRIKE - 20)])
    deb, bub = suction_debris(mouth, STRIKE)
    debris = deb.build("SuctionDebris", c_fx, M.silt, clock=clock, streak=True)
    bubbles = bub.build("StrikeBubbles", c_fx, M.bubble, clock=clock)
    lake = build_lake(c_above, M)
    boat = build_boat(c_above, M)
    ang = build_angler(c_above, M)
    rod = build_rod(c_above, M)
    tips, mids, _butts = animate_fight(rod, ang, SURF - 4, FRAME_END)
    line_a, entry = build_surface_line(c_above, M, rod, SURF - 4, FRAME_END)
    entry_spray(SURF - 4, FRAME_END).build("LineSpray", c_above, M.spray, streak=True)
    surface_camera(frames, fstop, mids, tips)
    build_camera(c_cam, frames, fstop)
    rim = build_rim(c_fx, frames)
    key_l = build_key(c_fx)
    key_scene_state(sc, (w_under, w_gloom), sun, rim, key_l)
    # visibility: each group renders only in the shots that need it (plus the frames under the wipes)
    under_spans = [(1, CUT1 + 3), (DIVE - 3, SURF + 3)]
    above_spans = [(CUT1 - 3, DIVE + 3), (SURF - 3, FRAME_END)]
    under_objs = [surf_under, debris, bubbles, leader, main_line] + shafts + sediment + rings
    under_objs += [o for o in c_life.all_objects] + [o for o in c_tackle.all_objects]
    key_visibility([o for o in under_objs if o.type in ("MESH", "CURVE")], under_spans)
    key_visibility([o for o in c_above.all_objects if o.type in ("MESH", "CURVE")], [(SURF - 3, FRAME_END)])
    key_visibility(lake[:1] + lake[1:] + [trunks, canopy], above_spans)
    key_visibility([o for o in c_tackle.all_objects if o.type == "MESH"] + [leader, main_line] + rings,
                   [(CUT2 - 3, SURF + 3)])
    hs, hcam, hcol = build_hud_scene()
    hud_materials(M)
    hud = Hud(hcam, hcol, M, frames)
    st = dict(frames=frames, hud=hud, roots=roots, fh=fh, mouth=mouth, rig=rig, rod=rod, mids=mids, tips=tips,
              entry=entry, angler=ang)
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
