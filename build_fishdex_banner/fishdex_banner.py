"""FishDex YouTube channel banner (2560 x 1440), built entirely in Blender with Python.

Run it from Blender's Scripting workspace (Text > Open, then Run Script), or headless:
    blender --background --python fishdex_banner.py -- --render banner.png
    blender --background --python fishdex_banner.py -- --preview preview.png     (quick 25% CPU check)

It does not touch your open scene: everything is built in a new scene called "FishDex_Banner", in four
collections (Branding, Environment, Marine_Life, Lighting). Render with F12.

Composition. The camera is a 100 mm lens looking down +Y at the focal plane y = 0, placed so that this plane
is exactly 25.6 units wide in frame: 1 unit on the plane = 100 pixels of the 2560 x 1440 banner. YouTube's
safe area (1546 x 423 px, the only part visible on every device) is then the box |x| <= 7.73, |z| <= 2.115 on
that plane. The emblem and the "FishDex" word are built there, centred on (0, 0, 0), and scaled so that they
fit inside it with a margin (the script prints the measured pixel box at the end). Marine life sits outside
it, in the four corners and further back in depth.

Water. A volume box (density WATER_DENSITY) fills the water behind the focal plane; the camera sits outside
it, so the branding stays crisp while distant fish fade into blue. A sun shines straight down through a
"caustic sheet" near the surface: a procedural pattern that lets light through in bright veins, so the
sun casts dappled caustics on the seabed and the fish and visible light shafts through the volume. (Real
refracted caustics in Cycles need the slower MNEE shadow caustics; this projected pattern is the usual
production shortcut and renders cleanly.)

Branding: the emblem follows the channel avatar (viewfinder brackets, a white fish whose tail fans out like
book pages, an open book) inside a rounded square; the word is Playfair Display, "Fish" white and "Dex" gold
italic like the channel's thumbnails. The fonts are looked up in a 'fonts' folder next to this script first.

Fish (low to mid poly, smooth shaded, subdivided once): upper left, freshwater (sockeye salmon in spawning
colours with the hooked jaw and hump; black sea bass; arapaima); upper right, pelagic and reef (mahi mahi,
blue marlin, reef fish); far background, a whale shark; lower left, deep sea (anglerfish with a glowing lure
and its own point light); lower right, open ocean (great white shark, a swirling school of tuna). The brief placed the shark both "upper background"
and "lower right": it is in the lower right quadrant, set back in depth and rising towards the upper right.
"""

import math
import os
import random
import sys

import bpy
import bmesh
from mathutils import Euler, Matrix, Vector, noise

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
RES_X, RES_Y = 2560, 1440
SAFE_W, SAFE_H = 1546, 423              # YouTube banner safe area, pixels
LENS = 100.0                            # mm (the brief allows 85 to 105)
SENSOR = 36.0                           # mm, horizontal fit
PX_PER_UNIT = 100.0                     # on the focal plane y = 0
CAM_DIST = (RES_X / PX_PER_UNIT) * LENS / SENSOR      # 71.1 units: the plane is 25.6 units wide
SAMPLES = 256
WATER_DENSITY = 0.02
EMBLEM_SHAPE = "rounded_square"         # or "circle" (the channel avatar is a circle)
SEED = 7

BLUE_DEEP = (0.004, 0.025, 0.07)
NAVY = (0.006, 0.035, 0.12)
LIGHT_BLUE = (0.32, 0.62, 1.0)


def srgb(r, g, b):
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (lin(r), lin(g), lin(b))


# --------------------------------------------------------------------------
# Scene, collections, small helpers
# --------------------------------------------------------------------------
def new_scene():
    sc = bpy.data.scenes.new("FishDex_Banner")
    if bpy.context.window is not None:
        bpy.context.window.scene = sc
    cols = {}
    for name in ("Branding", "Environment", "Marine_Life", "Lighting"):
        c = bpy.data.collections.new(name)
        sc.collection.children.link(c)
        cols[name] = c
    return sc, cols


def link(ob, col):
    col.objects.link(ob)
    return ob


def mesh_from_bm(name, bm, col, mats, smooth=True, subsurf=0, recalc=True):
    if recalc:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    for p in me.polygons:
        p.use_smooth = smooth
    ob = link(bpy.data.objects.new(name, me), col)
    if subsurf:
        mod = ob.modifiers.new("Subdivision", "SUBSURF")
        mod.levels = subsurf
        mod.render_levels = subsurf
    return ob


def add_attr(ob, name, fn):
    a = ob.data.attributes.new(name, "FLOAT", "POINT")
    a.data.foreach_set("value", [fn(v.co) for v in ob.data.vertices])


def basis(n):
    n = n.normalized()
    ref = Vector((0, 0, 1)) if abs(n.z) < 0.9 else Vector((1, 0, 0))
    u = n.cross(ref).normalized()
    return u, n.cross(u)


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------
def new_mat(name):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = m.node_tree
    nt.nodes.clear()
    return m, nt, nt.nodes.new("ShaderNodeOutputMaterial")


def principled(nt, color, rough=0.4, metal=0.0, coat=0.0, sss=0.0, sss_scale=0.05, emission=None, strength=0.0,
               transmission=0.0, alpha=1.0):
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    if isinstance(color, tuple):
        b.inputs["Base Color"].default_value = (*color, 1.0)
    else:
        nt.links.new(color, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    if isinstance(metal, float):
        b.inputs["Metallic"].default_value = metal
    else:
        nt.links.new(metal, b.inputs["Metallic"])
    b.inputs["Coat Weight"].default_value = coat
    if sss:
        b.inputs["Subsurface Weight"].default_value = sss
        b.inputs["Subsurface Scale"].default_value = sss_scale
    if emission is not None:
        b.inputs["Emission Color"].default_value = (*emission, 1.0)
        b.inputs["Emission Strength"].default_value = strength
    b.inputs["Transmission Weight"].default_value = transmission
    b.inputs["Alpha"].default_value = alpha
    return b


def solid_mat(name, color, **kw):
    m, nt, out = new_mat(name)
    nt.links.new(principled(nt, color, **kw).outputs[0], out.inputs[0])
    return m


def emission_mat(name, color, strength):
    m, nt, out = new_mat(name)
    e = nt.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1.0)
    e.inputs["Strength"].default_value = strength
    nt.links.new(e.outputs[0], out.inputs[0])
    return m


def ramp(nt, fac, stops):
    r = nt.nodes.new("ShaderNodeValToRGB")
    nt.links.new(fac, r.inputs[0])
    els = r.color_ramp.elements
    els[0].position, els[0].color = stops[0][0], (*stops[0][1], 1.0)
    els[1].position, els[1].color = stops[-1][0], (*stops[-1][1], 1.0)
    for pos, col in stops[1:-1]:
        els.new(pos).color = (*col, 1.0)
    return r.outputs["Color"]


def attr(nt, name):
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_name = name
    return a.outputs["Fac"]


def math_node(nt, op, a, b=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def gradient_mat(name, kind, stops, emit=0.0):
    """Flat logo colour from a ramp: 'vertical' (bottom to top) or 'radial' (centre to edge), on the object's
    generated coordinates; a little emission keeps it vivid under the water light."""
    m, nt, out = new_mat(name)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    if kind == "vertical":
        fac = sep.outputs["Z"]
    else:
        dx = math_node(nt, "SUBTRACT", sep.outputs["X"], 0.42)
        dz = math_node(nt, "SUBTRACT", sep.outputs["Z"], 0.6)
        fac = math_node(nt, "MULTIPLY", math_node(nt, "SQRT", math_node(nt, "ADD", math_node(nt, "MULTIPLY", dx, dx),
                                                                         math_node(nt, "MULTIPLY", dz, dz))), 1.5)
    col = ramp(nt, fac, stops)
    b = principled(nt, col, rough=0.25, coat=0.7)
    nt.links.new(col, b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = emit
    nt.links.new(b.outputs[0], out.inputs[0])
    return m


def fish_mat(name, stops, along=None, spots=None, metal=0.15, sss=0.15, rough=0.35, coat=0.5):
    """Fish skin: colour from the belly-to-back attribute 'zrel' (optionally blended with a nose-to-tail
    ramp on 'xrel'), optional dark spots, a light metallic sheen and subsurface scattering."""
    m, nt, out = new_mat(name)
    col = ramp(nt, attr(nt, "zrel"), stops)
    if along is not None:                     # e.g. the sockeye's green head over its red body
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        nt.links.new(ramp(nt, attr(nt, "xrel"), along[0]), mix.inputs[0])
        nt.links.new(col, mix.inputs[6])
        mix.inputs[7].default_value = (*along[1], 1.0)
        col = mix.outputs[2]
    if spots is not None:
        vo = nt.nodes.new("ShaderNodeTexVoronoi")
        vo.inputs["Scale"].default_value = spots[1]
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(tc.outputs["Object"], vo.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.16, 0.10
        nt.links.new(vo.outputs["Distance"], mr.inputs["Value"])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        nt.links.new(mr.outputs["Result"], mix.inputs[0])
        nt.links.new(col, mix.inputs[6])
        mix.inputs[7].default_value = (*spots[0], 1.0)
        col = mix.outputs[2]
    metal_s = math_node(nt, "MULTIPLY", math_node(nt, "SUBTRACT", 1.0, attr(nt, "zrel")), metal * 2.0)
    b = principled(nt, col, rough=rough, metal=metal_s, coat=coat, sss=sss, sss_scale=0.08)
    nt.links.new(b.outputs[0], out.inputs[0])
    return m


# --------------------------------------------------------------------------
# Geometry: lofted bodies, tubes, fins
# --------------------------------------------------------------------------
def loft(bm, rings, n=20, mat=0):
    """Body along +X from rings (x, half_width, half_height, z_centre); zero size makes a pole."""
    vr = []
    for x, w, h, zc in rings:
        if w <= 1e-6 and h <= 1e-6:
            vr.append([bm.verts.new((x, 0.0, zc))])
        else:
            vr.append([bm.verts.new((x, w * math.cos(2 * math.pi * i / n), zc + h * math.sin(2 * math.pi * i / n)))
                       for i in range(n)])
    for ra, rb in zip(vr, vr[1:]):
        if len(ra) == 1:
            fs = [bm.faces.new((ra[0], rb[i], rb[(i + 1) % n])) for i in range(n)]
        elif len(rb) == 1:
            fs = [bm.faces.new((ra[i], ra[(i + 1) % n], rb[0])) for i in range(n)]
        else:
            fs = [bm.faces.new((ra[i], ra[(i + 1) % n], rb[(i + 1) % n], rb[i])) for i in range(n)]
        for f in fs:
            f.material_index = mat


def body_rings(L, h, w, peak=0.35, nose=0.5, tail=0.08, hump=0.0, droop=0.0, n=18):
    """Fish body from nose (x = L/2) to tail stalk (x = -L/2)."""
    out = []
    for i in range(n + 1):
        u = i / n
        x = L / 2 - u * L
        if u < peak:
            k = math.sin(0.5 * math.pi * u / peak) ** nose
        else:
            k = 1 - (1 - tail) * ((u - peak) / (1 - peak)) ** 1.5
        hh = h * k * (1 + hump * math.sin(math.pi * min(1.0, u / 0.7)))
        zc = hump * h * 0.5 * math.sin(math.pi * min(1.0, u / 0.7)) - droop * h * (1 - u) ** 3
        out.append((x, 0.0, 0.0, zc) if i in (0, n) else (x, w * k, hh, zc))
    return out


def fin(bm, outline, thick, mat, plane="XZ", offset=(0, 0, 0)):
    """Flat fin from an outline; plane XZ (vertical, along the body) or XY (horizontal)."""
    o = Vector(offset)

    def P(a, b, d):
        return o + (Vector((a, d, b)) if plane == "XZ" else Vector((a, b, d)))
    top = [bm.verts.new(P(a, b, thick / 2)) for a, b in outline]
    bot = [bm.verts.new(P(a, b, -thick / 2)) for a, b in outline]
    fs = [bm.faces.new(top), bm.faces.new(list(reversed(bot)))]
    n = len(outline)
    fs += [bm.faces.new((top[i], top[(i + 1) % n], bot[(i + 1) % n], bot[i])) for i in range(n)]
    for f in fs:
        f.material_index = mat


def tube(bm, pts, r, segs=8, mat=0):
    pts = [Vector(p) for p in pts]
    rings, u = [], None
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        u = basis(t)[0] if u is None else (u - t * u.dot(t)).normalized()
        v = t.cross(u)
        rr = r(i / (len(pts) - 1)) if callable(r) else r
        rings.append([bm.verts.new(p + (u * math.cos(2 * math.pi * k / segs) + v * math.sin(2 * math.pi * k / segs)) * rr)
                      for k in range(segs)])
    for a, b in zip(rings, rings[1:]):
        for k in range(segs):
            f = bm.faces.new((a[k], a[(k + 1) % segs], b[(k + 1) % segs], b[k]))
            f.material_index = mat
    for ring, rev in ((rings[0], True), (rings[-1], False)):
        f = bm.faces.new(list(reversed(ring)) if rev else ring)
        f.material_index = mat


def sphere(bm, center, size, mat=0, subdiv=2):
    m = Matrix.LocRotScale(Vector(center), None, Vector(size))
    res = bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0, matrix=m)
    for f in {f for v in res["verts"] for f in v.link_faces}:
        f.material_index = mat


def eyes(bm, x, w, z, r, mat):
    for s in (1, -1):
        sphere(bm, (x, s * w * 0.85, z), (r, r * 0.6, r), mat)


def finish_fish(name, bm, col, mats, rings):
    xs = [r[0] for r in rings]
    zlo = min(r[3] - r[2] for r in rings)
    zhi = max(r[3] + r[2] for r in rings)
    ob = mesh_from_bm(name, bm, col, mats, subsurf=1)
    add_attr(ob, "zrel", lambda c: min(1.0, max(0.0, (c.z - zlo) / (zhi - zlo))))
    add_attr(ob, "xrel", lambda c: min(1.0, max(0.0, (c.x - min(xs)) / (max(xs) - min(xs)))))
    return ob


def tail_fork(L, span, depth, lobe_up=1.0):
    x0 = -L / 2
    return [(x0 + 0.02 * L, 0.06 * span), (x0 - depth, span * lobe_up), (x0 - depth * 0.55, 0.0),
            (x0 - depth, -span), (x0 + 0.02 * L, -0.06 * span)]


# --------------------------------------------------------------------------
# Species (each built with nose +X, back +Z, about its own origin)
# --------------------------------------------------------------------------
def make_sockeye(col, M):
    L, h, w = 3.0, 0.42, 0.22
    rings = body_rings(L, h, w, peak=0.32, hump=0.35, droop=0.1)
    bm = bmesh.new()
    loft(bm, rings)
    eyes(bm, L / 2 - 0.42, w, 0.12, 0.07, 1)
    # the kype: the spawning male's hooked lower jaw curling up past the snout
    tube(bm, [(L / 2 - 0.35, 0, -0.16), (L / 2 - 0.05, 0, -0.17), (L / 2 + 0.12, 0, -0.08), (L / 2 + 0.12, 0, 0.05)],
         lambda u: 0.07 * (1 - 0.7 * u), mat=0)
    fin(bm, tail_fork(L, 0.55, 0.55), 0.03, 2)
    fin(bm, [(0.15, 0.45), (-0.25, 0.85), (-0.5, 0.55)], 0.03, 2)            # dorsal on the hump
    fin(bm, [(-0.95, 0.35), (-1.05, 0.5), (-1.15, 0.33)], 0.02, 2)           # adipose
    fin(bm, [(-0.6, -0.3), (-0.75, -0.55), (-1.0, -0.28)], 0.02, 2)          # anal
    return finish_fish("SockeyeSalmon", bm, col, [M["sockeye"], M["eye"], M["fin_red"]], rings)


def make_bass(col, M):
    L, h, w = 2.4, 0.5, 0.22
    rings = body_rings(L, h, w, peak=0.38, nose=0.6, tail=0.22)
    bm = bmesh.new()
    loft(bm, rings)
    eyes(bm, L / 2 - 0.32, w * 0.9, 0.15, 0.08, 1)
    fin(bm, [(0.45, 0.42), (0.3, 0.85), (0.0, 0.72), (-0.2, 0.8), (-0.5, 0.65), (-0.75, 0.42)], 0.03, 2)   # spiny dorsal
    fin(bm, tail_fork(L, 0.6, 0.45) + [], 0.03, 2)
    fin(bm, [(-0.3, -0.42), (-0.5, -0.75), (-0.8, -0.4)], 0.03, 2)
    return finish_fish("BlackSeaBass", bm, col, [M["bass"], M["eye"], M["fin_dark"]], rings)


def make_mahi(col, M):
    L, h, w = 3.4, 0.55, 0.16
    rings = body_rings(L, h, w, peak=0.12, nose=0.25, tail=0.08)
    rings = [(x, wd, hh, zc + (0.2 * h * (x / L)) if x > 0 else zc) for x, wd, hh, zc in rings]
    bm = bmesh.new()
    loft(bm, rings)
    eyes(bm, L / 2 - 0.3, w * 0.9, -0.05, 0.08, 1)
    dorsal = [(L / 2 - 0.25, 0.55)] + [(L / 2 - 0.25 - t * (L - 0.6), 0.55 + 0.45 * math.sin(math.pi * t) ** 0.6) for t in
                                       (0.1, 0.3, 0.5, 0.7, 0.9)] + [(-L / 2 + 0.35, 0.2), (L / 2 - 0.4, 0.4)]
    fin(bm, dorsal, 0.03, 2)                                          # the long dorsal fin, head to tail
    fin(bm, tail_fork(L, 0.85, 0.75), 0.03, 2)
    fin(bm, [(-0.2, -0.45), (-0.6, -0.75), (-1.4, -0.25)], 0.03, 2)
    return finish_fish("MahiMahi", bm, col, [M["mahi"], M["eye"], M["fin_blue"]], rings)


def make_reef(col, M, name, mat):
    L, h, w = 0.9, 0.42, 0.12
    rings = body_rings(L, h, w, peak=0.4, nose=0.7, tail=0.25)
    bm = bmesh.new()
    loft(bm, rings)
    eyes(bm, L / 2 - 0.13, w * 0.9, 0.08, 0.05, 1)
    fin(bm, [(0.2, 0.35), (-0.05, 0.6), (-0.35, 0.3)], 0.02, 0)
    fin(bm, [(0.1, -0.35), (-0.1, -0.55), (-0.35, -0.28)], 0.02, 0)
    fin(bm, tail_fork(L, 0.3, 0.22), 0.02, 0)
    return finish_fish(name, bm, col, [mat, M["eye"]], rings)


def make_shark(col, M):
    L, h, w = 9.0, 0.95, 0.8
    rings = body_rings(L, h, w, peak=0.32, nose=0.55, tail=0.06)
    bm = bmesh.new()
    loft(bm, rings, n=24)
    eyes(bm, L / 2 - 1.0, w * 0.75, 0.15, 0.1, 1)
    fin(bm, [(0.9, 0.8), (-0.2, 2.2), (-0.7, 0.75)], 0.12, 0)                          # first dorsal
    fin(bm, [(-L / 2 + 0.2, 0.1), (-L / 2 - 1.2, 2.0), (-L / 2 - 0.65, 0.05), (-L / 2 - 0.95, -1.4), (-L / 2 + 0.15, -0.1)],
        0.1, 0)                                                                           # crescent tail, upper lobe longer
    for s in (1, -1):                                                                     # pectoral fins
        fin(bm, [(1.2, 0.0), (0.0, s * 2.2), (-0.1, s * 1.9), (0.4, 0.0)], 0.08, 0, plane="XY",
            offset=(0, s * 0.55, -0.55))
    for k in range(5):                                                                    # gill slits
        x = L / 2 - 1.6 - k * 0.18
        for s in (1, -1):
            tube(bm, [(x, s * w * 0.93, 0.25), (x - 0.05, s * w * 0.93, -0.3)], 0.025, segs=6, mat=1)
    return finish_fish("GreatWhiteShark", bm, col, [M["shark"], M["eye"]], rings)


def make_angler(col, M):
    L, h, w = 1.8, 0.75, 0.75
    rings = body_rings(L, h, w, peak=0.35, nose=0.35, tail=0.15)
    bm = bmesh.new()
    loft(bm, rings)
    eyes(bm, L / 2 - 0.4, w * 0.75, 0.35, 0.07, 1)
    # gaping lower jaw and needle teeth
    tube(bm, [(L / 2 - 0.6, 0, -0.45), (L / 2 + 0.05, 0, -0.35), (L / 2 + 0.2, 0, -0.1)], lambda u: 0.22 * (1 - 0.4 * u), mat=0)
    rng = random.Random(3)
    for k in range(14):
        a = math.pi * (k / 13.0 - 0.5)
        base = Vector((L / 2 + 0.05 * math.cos(a), 0.45 * math.sin(a), -0.08 if k % 2 else 0.1))
        tip = base + Vector((0.05, 0.0, -0.22 if k % 2 == 0 else 0.22)) * rng.uniform(0.7, 1.1)
        tube(bm, [base, tip], lambda u: 0.025 * (1 - 0.9 * u), segs=5, mat=3)
    # illicium: the fishing rod spine arching forward, the esca (lure) at its tip
    rod = [Vector((L / 2 - 0.5, 0, h * 0.95)) + Vector((0.75 * t, 0, 0.8 * math.sin(math.pi * t * 0.8))) for t in
           (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    tube(bm, rod, 0.025, segs=6, mat=0)
    fin(bm, tail_fork(L, 0.35, 0.3), 0.03, 0)
    ob = finish_fish("Anglerfish", bm, col, [M["angler"], M["eye"], M["lure"], M["teeth"]], rings)
    bm = bmesh.new()
    sphere(bm, rod[-1] + Vector((0.05, 0, -0.05)), (0.09, 0.09, 0.09), 0, subdiv=3)
    esca = mesh_from_bm("Anglerfish_Lure", bm, col, [M["lure"]])
    esca.parent = ob
    return ob, esca, rod[-1] + Vector((0.05, 0, -0.05))


def make_tuna(col, M, name="Tuna"):
    L, h, w = 1.2, 0.2, 0.15
    rings = body_rings(L, h, w, peak=0.38, nose=0.55, tail=0.05)
    bm = bmesh.new()
    loft(bm, rings, n=14)
    eyes(bm, L / 2 - 0.15, w * 0.9, 0.04, 0.035, 1)
    fin(bm, tail_fork(L, 0.32, 0.28), 0.02, 0)
    fin(bm, [(0.1, 0.18), (-0.05, 0.36), (-0.15, 0.18)], 0.015, 0)
    return finish_fish(name, bm, col, [M["tuna"], M["eye"]], rings)




def make_marlin(col, M):
    """Blue marlin: long spear bill, tall sail-like front dorsal, deeply forked tail, cobalt back, silver belly."""
    L, h, w = 6.0, 0.62, 0.4
    rings = body_rings(L, h, w, peak=0.3, nose=0.6, tail=0.05)
    bm = bmesh.new()
    loft(bm, rings, n=22)
    tube(bm, [(L / 2 - 0.1, 0, 0.05), (L / 2 + 0.7, 0, 0.03), (L / 2 + 1.7, 0, 0.0)], lambda u: 0.07 * (1 - 0.9 * u), mat=0)
    eyes(bm, L / 2 - 0.45, w * 0.8, 0.12, 0.1, 1)
    fin(bm, [(L / 2 - 0.9, 0.5), (L / 2 - 1.2, 1.6), (L / 2 - 1.8, 1.45), (L / 2 - 2.6, 0.9), (L / 2 - 3.4, 0.45)], 0.05, 2)
    fin(bm, tail_fork(L, 1.5, 1.2), 0.05, 2)
    for s in (1, -1):
        fin(bm, [(L / 2 - 1.4, 0.0), (L / 2 - 2.6, s * 1.0), (L / 2 - 2.3, 0.0)], 0.04, 2, plane="XY", offset=(0, s * 0.3, -0.25))
    return finish_fish("BlueMarlin", bm, col, [M["marlin"], M["eye"], M["fin_marlin"]], rings)


def make_whale_shark(col, M):
    """Whale shark: broad flat head with a wide terminal mouth, ridged dark blue-grey back with pale spots."""
    L, h, w = 18.0, 1.6, 1.9
    rings = body_rings(L, h, w, peak=0.25, nose=0.3, tail=0.05)
    rings = [(x, wd * (1.15 if x > L / 4 else 1.0), hh * (0.8 if x > L / 4 else 1.0), zc) for x, wd, hh, zc in rings]
    bm = bmesh.new()
    loft(bm, rings, n=28)
    eyes(bm, L / 2 - 1.3, w * 1.05, 0.2, 0.14, 1)
    fin(bm, [(1.2, 1.3), (-0.5, 3.4), (-1.4, 1.2)], 0.25, 0)
    fin(bm, [(-L / 2 + 0.4, 0.2), (-L / 2 - 2.2, 3.8), (-L / 2 - 1.2, 0.0), (-L / 2 - 1.8, -2.4), (-L / 2 + 0.3, -0.2)], 0.2, 0)
    for s in (1, -1):
        fin(bm, [(L / 2 - 3.0, 0.0), (L / 2 - 5.2, s * 3.6), (L / 2 - 5.6, s * 3.2), (L / 2 - 4.4, 0.0)], 0.16, 0, plane="XY",
            offset=(0, s * 1.2, -0.9))
    return finish_fish("WhaleShark", bm, col, [M["whale_shark"], M["eye"]], rings)


def make_arapaima(col, M):
    """Arapaima: long cylindrical Amazon giant, olive-grey front shading to red-edged scales towards the tail,
    with the dorsal and anal fins set far back near the tail."""
    L, h, w = 6.5, 0.6, 0.5
    rings = body_rings(L, h, w, peak=0.3, nose=0.45, tail=0.2)
    rings = [(x, wd, hh * (0.85 if x > 0 else 1.0), zc) for x, wd, hh, zc in rings]
    bm = bmesh.new()
    loft(bm, rings, n=22)
    eyes(bm, L / 2 - 0.45, w * 0.8, 0.16, 0.09, 1)
    fin(bm, [(-L / 2 + 1.9, 0.45), (-L / 2 + 0.9, 0.95), (-L / 2 + 0.4, 0.45)], 0.05, 2)
    fin(bm, [(-L / 2 + 1.9, -0.45), (-L / 2 + 0.9, -0.9), (-L / 2 + 0.4, -0.45)], 0.05, 2)
    fin(bm, [(-L / 2 + 0.2, 0.25), (-L / 2 - 0.9, 0.7), (-L / 2 - 1.05, 0.0), (-L / 2 - 0.9, -0.7), (-L / 2 + 0.2, -0.25)], 0.05, 2)
    return finish_fish("Arapaima", bm, col, [M["arapaima"], M["eye"], M["fin_red"]], rings)


# --------------------------------------------------------------------------
# Branding: emblem (viewfinder, fish, open book) and the FishDex word
# --------------------------------------------------------------------------
def rounded_rect(w, h, r, n=10):
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180),
                       (w / 2 - r, -h / 2 + r, 270)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def extrude_outline(name, outline, depth, col, mat, z_front=0.0, bevel=0.0, holes=()):
    """Flat outline in the XZ plane (facing -Y, the camera) extruded towards +Y."""
    bm = bmesh.new()
    vs = [bm.verts.new((x, 0.0, z)) for x, z in outline]
    f = bm.faces.new(vs)
    bmesh.ops.triangulate(bm, faces=[f])
    res = bmesh.ops.extrude_face_region(bm, geom=list(bm.faces))
    moved = [e for e in res["geom"] if isinstance(e, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=moved, vec=(0, depth, 0))
    ob = mesh_from_bm(name, bm, col, [mat], smooth=False)
    ob.location.y = z_front
    if bevel:
        mod = ob.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 3
        mod.limit_method = "ANGLE"
    return ob


def emblem_fish_outline(S):
    """The FishDex fish (head left, as in the channel avatar): a smooth lens-shaped body ending in a narrow
    tail stalk on the right. The tail itself is the fan of 'pages' built separately."""
    top, bot = [], []
    for i in range(29):
        u = i / 28                                         # 0 snout (left) .. 1 tail stalk (right)
        x = (-0.30 + 0.50 * u) * S
        h = 0.155 * S * math.sin(math.pi * min(1.0, u * 1.08)) ** 0.75 + 0.012 * S
        top.append((x, h + 0.01 * S * (1 - u)))
        bot.append((x, -0.8 * h))
    return top + list(reversed(bot))


def make_emblem(col, M, size):
    """The FishDex emblem as on the channel avatar, inside a rounded square: a blue gradient plate, light blue
    viewfinder brackets in the four corners, a white fish (head left, with an eye) whose tail fans out like
    book pages, over an open book with light blue page edges."""
    S = size
    root = link(bpy.data.objects.new("FishDex_Emblem", None), col)
    plate = extrude_outline("Emblem_Plate", rounded_rect(S, S, S * 0.22) if EMBLEM_SHAPE != "circle" else
                            [(S / 2 * math.cos(2 * math.pi * i / 72), S / 2 * math.sin(2 * math.pi * i / 72)) for i in range(72)],
                            0.12 * S, col, M["emblem"], bevel=0.02 * S)
    plate.parent = root
    plate.location = (0, 0.02, 0)

    def part(name, outline, mat, depth=0.03, y=-0.03):
        ob = extrude_outline(name, outline, depth * S, col, mat)
        ob.parent = root
        ob.location = (0, y * S, 0)
        return ob
    # viewfinder brackets: thick L shapes with rounded outer corners
    k, t, arm, r = 0.37 * S, 0.055 * S, 0.17 * S, 0.06 * S
    arc = []
    for i in range(9):                                     # rounded outer corner of the top left bracket
        a = math.radians(90 + 90 * i / 8)
        arc.append((-k + r + r * math.cos(a), k - r + r * math.sin(a)))
    base = [(-k + arm, k)] + arc + [(-k, k - arm), (-k + t, k - arm), (-k + t, k - t), (-k + arm, k - t)]
    for sx, sz in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        pts = [(-x * sx, z * sz) for x, z in base]         # base is the top left one: mirror it into each corner
        if sx * sz < 0:
            pts = list(reversed(pts))
        part(f"Viewfinder_{sx}_{sz}", pts, M["bracket"])
    # open book: two pages whose top edges rise from the spine to the outer corners, with page edges below
    for side in (1, -1):
        for j, (z0, th, mat) in enumerate(((-0.105, 0.05, M["book"]), (-0.175, 0.026, M["bracket"]), (-0.225, 0.022, M["bracket"]))):
            pts = []
            for i in range(17):
                u = i / 16
                x = side * (0.012 + u * 0.32) * S
                z = (z0 + 0.075 * u ** 1.3 - 0.02 * math.sin(math.pi * u)) * S
                pts.append((x, z))
            outline = pts + [(x, z - th * S) for x, z in reversed(pts)]
            if side < 0:
                outline = list(reversed(outline))
            part(f"Book_{side}_{j}", outline, mat, depth=0.03, y=-0.035)
    # the fish (white) and its eye
    part("Emblem_Fish", [(x - 0.03 * S, z + 0.075 * S) for x, z in emblem_fish_outline(S)], M["white"], depth=0.04, y=-0.06)
    bm = bmesh.new()
    ex, ez, er = -0.215 * S, 0.11 * S, 0.028 * S
    ring = [bm.verts.new((ex + er * math.cos(2 * math.pi * i / 24), 0.0, ez + er * math.sin(2 * math.pi * i / 24))) for i in range(24)]
    bm.faces.new(ring)
    eye = mesh_from_bm("Emblem_FishEye", bm, col, [M["emblem_dark"]], smooth=False)
    eye.parent = root
    eye.location = (0, -0.061 * S, 0)
    # the tail: a fan of page-like blades spreading to the right of the tail stalk
    sx0, sz0 = 0.15 * S, 0.075 * S
    for i, ang in enumerate((40, 20, 0, -20, -40)):
        a0, a1 = math.radians(ang - 8), math.radians(ang + 8)
        L1 = 0.21 * S
        blade = [(sx0, sz0), (sx0 + L1 * math.cos(a0), sz0 + L1 * math.sin(a0)),
                 (sx0 + L1 * 1.02 * math.cos((a0 + a1) / 2), sz0 + L1 * 1.02 * math.sin((a0 + a1) / 2)),
                 (sx0 + L1 * math.cos(a1), sz0 + L1 * math.sin(a1))]
        part(f"Emblem_TailPage_{i}", blade, M["white"] if i % 2 == 0 else M["page_shade"], depth=0.04, y=-0.06)
    return root


def find_font(names):
    heres = []
    if "__file__" in globals() and os.path.exists(str(__file__)):
        heres.append(os.path.dirname(os.path.abspath(__file__)))
    if bpy.data.filepath:
        heres.append(os.path.dirname(bpy.data.filepath))
    dirs = [d for h in heres for d in (os.path.join(h, "fonts"), h)] + [ os.path.expanduser("~/Library/Fonts"), "/Library/Fonts",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"), "C:/Windows/Fonts",
            os.path.expanduser("~/.fonts"), os.path.expanduser("~/.local/share/fonts"), "/usr/share/fonts/truetype"]
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return bpy.data.fonts.load(p, check_existing=True)
    print(f"WARNING: {names[0]} not found (put it in a 'fonts' folder next to this script); using Blender's font")
    return bpy.data.fonts.get("Bfont Regular") or bpy.data.fonts.load("<builtin>")


def text_object(name, body, font, size, mat, col):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.font = font
    cu.size = size
    cu.align_x, cu.align_y = "LEFT", "CENTER"
    cu.extrude = size * 0.07
    cu.bevel_depth = size * 0.012
    cu.bevel_resolution = 3
    cu.materials.append(mat)
    ob = link(bpy.data.objects.new(name, cu), col)
    ob.rotation_euler = (math.radians(90), 0, 0)          # stand up, facing the camera (-Y)
    return ob


def make_wordmark(col, M, height):
    """'Fish' in white Playfair Display SemiBold, 'Dex' in gold Playfair Bold Italic (the thumbnails' style)."""
    roman = find_font(["PlayfairDisplay-SemiBold.ttf", "PlayfairDisplay-Bold.ttf", "PlayfairDisplay-Regular.ttf"])
    italic = find_font(["PlayfairDisplay-BoldItalic.ttf", "PlayfairDisplay-SemiBoldItalic.ttf", "PlayfairDisplay-Italic.ttf"])
    root = link(bpy.data.objects.new("FishDex_Wordmark", None), col)
    a = text_object("Wordmark_Fish", "Fish", roman, height, M["text"], col)
    b = text_object("Wordmark_Dex", "Dex", italic, height, M["gold"], col)
    for ob in (a, b):
        ob.parent = root
    bpy.context.view_layer.update()
    wa = a.dimensions.x
    b.location.x = wa + height * 0.04
    bpy.context.view_layer.update()
    return root, wa + height * 0.04 + b.dimensions.x


def build_branding(col, M):
    group = link(bpy.data.objects.new("Branding_Group", None), col)
    emblem_size = 3.4
    emblem = make_emblem(col, M, emblem_size)
    word, tw = make_wordmark(col, M, 2.9)
    emblem.parent = group
    word.parent = group
    gap = 0.6
    total = emblem_size + gap + tw
    emblem.location = (-total / 2 + emblem_size / 2, 0, 0)
    word.location = (-total / 2 + emblem_size + gap, 0, -0.12)
    # fit inside the safe area with a 6% margin (centred on the origin, the safe area's centre)
    s = min(1.0, SAFE_W / PX_PER_UNIT * 0.94 / total, SAFE_H / PX_PER_UNIT * 0.94 / emblem_size)
    group.scale = (s, s, s)
    return group, emblem, word


# --------------------------------------------------------------------------
# Environment: water volume, seabed, rocks, kelp, coral, bubbles, caustic sheet
# --------------------------------------------------------------------------
def build_world(sc):
    w = bpy.data.worlds.new("DeepBlue")
    sc.world = w
    try:
        w.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    col = ramp(nt, sep.outputs["Z"], [(0.0, (0.0015, 0.008, 0.025)), (0.5, BLUE_DEEP), (1.0, (0.02, 0.12, 0.28))])
    bg = nt.nodes.new("ShaderNodeBackground")
    nt.links.new(col, bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(bg.outputs[0], out.inputs["Surface"])


def build_environment(col, M):
    rng = random.Random(SEED)
    # water volume: starts just behind the branding, so the camera looks into it from outside
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.LocRotScale(Vector((0, 46, 1.5)), None, Vector((180, 96, 40))))
    vol = mesh_from_bm("WaterVolume", bm, col, [M["water"]], smooth=False)
    vol.visible_shadow = True
    # seabed: rolling sand, lower towards the back
    bm = bmesh.new()
    nx, ny = 90, 50
    verts = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            x = -90 + 180 * i / nx
            y = -6 + 100 * j / ny
            z = -9.5 - 0.06 * max(0.0, y) + 0.9 * noise.noise(Vector((x * 0.05, y * 0.05, 0.3))) + 0.25 * noise.noise(Vector((x * 0.2, y * 0.2, 1.7)))
            verts[i, j] = bm.verts.new((x, y, z))
    for i in range(nx):
        for j in range(ny):
            bm.faces.new((verts[i, j], verts[i + 1, j], verts[i + 1, j + 1], verts[i, j + 1]))
    mesh_from_bm("Seabed", bm, col, [M["sand"]])
    # rocks
    bm = bmesh.new()
    for _ in range(40):
        x = rng.uniform(-40, 40)
        y = rng.uniform(4, 60)
        if abs(x) < 9 and y < 12:
            continue
        s = rng.uniform(0.8, 3.0)
        z = -9.6 - 0.06 * y
        res = bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0,
                                         matrix=Matrix.LocRotScale(Vector((x, y, z)), Euler((rng.random(), rng.random(), rng.random())),
                                                                   Vector((s * rng.uniform(1, 1.6), s, s * rng.uniform(0.5, 0.9)))))
        for v in res["verts"]:
            v.co += Vector((rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15))) * s
    mesh_from_bm("Rocks", bm, col, [M["rock"]])
    # kelp and eelgrass on the left (freshwater and shallows), coral heads on the right (reef)
    bm = bmesh.new()
    for _ in range(46):
        x = rng.uniform(-36, -9)
        y = rng.uniform(2, 40)
        z0 = -9.6 - 0.06 * y
        hgt = rng.uniform(4, 11)
        ph = rng.uniform(0, 6.28)
        pts = [Vector((x + 0.6 * math.sin(ph + t * 2.5) * t, y, z0 + hgt * t)) for t in [k / 10 for k in range(11)]]
        tube(bm, pts, lambda u: 0.16 * (1 - 0.7 * u), segs=6)
    mesh_from_bm("Kelp", bm, col, [M["kelp"]])
    bm = bmesh.new()
    for _ in range(30):
        x = rng.uniform(9, 38)
        y = rng.uniform(2, 34)
        z0 = -9.6 - 0.06 * y
        for _b in range(rng.randint(3, 7)):
            c = Vector((x + rng.uniform(-1, 1), y + rng.uniform(-1, 1), z0 + rng.uniform(0.2, 1.6)))
            r = rng.uniform(0.4, 1.1)
            sphere(bm, c, (r, r, r * rng.uniform(0.7, 1.3)), rng.randint(0, 2))
    mesh_from_bm("Coral", bm, col, [M["coral_a"], M["coral_b"], M["coral_c"]], subsurf=1)
    # bubble columns
    bm = bmesh.new()
    for cx, cy in ((-12.5, 6.0), (13.0, 9.0), (-20.0, 18.0)):
        z = -9.0
        while z < 10:
            r = rng.uniform(0.04, 0.12)
            sphere(bm, (cx + rng.uniform(-0.3, 0.3), cy + rng.uniform(-0.3, 0.3), z), (r, r, r), 0, subdiv=1)
            z += rng.uniform(0.3, 0.9)
    mesh_from_bm("Bubbles", bm, col, [M["bubble"]])
    # caustic sheet near the surface (light passes through its bright veins)
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=110, matrix=Matrix.Translation((0, 40, 16)))
    sheet = mesh_from_bm("CausticSheet", bm, col, [M["caustic"]], smooth=False)
    sheet.visible_camera = False
    return vol


# --------------------------------------------------------------------------
# Marine life layout
# --------------------------------------------------------------------------
def place(ob, loc, yaw=0.0, pitch=0.0, roll=0.0, scale=1.0):
    ob.location = loc
    ob.rotation_euler = (math.radians(roll), math.radians(pitch), math.radians(yaw))
    ob.scale = (scale, scale, scale)


def build_marine_life(col, M, lights_col):
    """Fish around the branding, outside the safe area (|x| > 7.7 or |z| > 2.1 on the focal plane, scaled up
    with depth). Positions are (x, depth y, z); y > 0 is behind the branding, inside the water."""
    # upper left: freshwater
    place(make_sockeye(col, M), (-10.6, -1.0, 4.9), yaw=8, pitch=-3, scale=1.3)
    s2 = make_sockeye(col, M)
    s2.name = "SockeyeSalmon_Back"
    place(s2, (-7.6, 12.0, 6.6), yaw=-6, scale=1.05)
    place(make_bass(col, M), (-5.4, 3.0, 4.6), yaw=-172, pitch=2, scale=1.15)
    place(make_arapaima(col, M), (-14.6, 6.0, 2.7), yaw=4, pitch=2, scale=0.95)
    # upper right: pelagic and reef
    place(make_mahi(col, M), (10.4, -1.0, 4.1), yaw=176, pitch=3, scale=1.15)
    place(make_marlin(col, M), (14.8, 12.0, 6.9), yaw=188, pitch=-12, scale=0.9)
    for i, (mat, loc, yaw, sc) in enumerate(((M["reef_yellow"], (8.6, -1.0, 2.75), 170, 1.1),
                                             (M["reef_blue"], (6.4, 0.0, 5.6), -12, 1.15),
                                             (M["reef_orange"], (12.4, 2.0, 2.6), 165, 1.0),
                                             (M["reef_yellow"], (17.0, 8.0, 3.2), 178, 1.2),
                                             (M["reef_blue"], (10.2, 6.0, 7.0), 172, 1.0))):
        place(make_reef(col, M, f"ReefFish_{i}", mat), loc, yaw=yaw, scale=sc)
    # far background, top: a whale shark gliding across, faded by the water
    place(make_whale_shark(col, M), (1.0, 34.0, 8.6), yaw=176, pitch=2, scale=0.95)
    # lower left: deep sea
    angler, esca, tip = make_angler(col, M)
    place(angler, (-10.4, -1.0, -4.5), yaw=16, scale=1.6)
    bpy.context.view_layer.update()
    ld = bpy.data.lights.new("AnglerLureLight", "POINT")
    ld.energy = 80.0
    ld.color = (0.45, 1.0, 0.85)
    ld.shadow_soft_size = 0.1
    lo = link(bpy.data.objects.new("AnglerLureLight", ld), lights_col)
    lo.parent = angler
    lo.location = tip + Vector((0.05, 0, -0.05))
    # lower right: open ocean apex predator and a swirling tuna school
    place(make_shark(col, M), (15.5, 13.0, -3.3), yaw=192, pitch=-6, scale=0.95)
    proto = make_tuna(col, M)
    proto.hide_render = True
    proto.hide_viewport = True
    rng = random.Random(SEED + 1)
    center = Vector((9.6, -1.0, -5.0))
    n = 46
    for i in range(n):                            # a rising helix that tightens: the school's swirl
        t = i / n
        a = 2 * math.pi * 2.3 * t + rng.uniform(-0.15, 0.15)
        r = 2.8 - 1.3 * t + rng.uniform(-0.2, 0.2)
        p = center + Vector((r * math.cos(a), 0.6 * r * math.sin(a) + 1.5 * t, -1.0 + 2.0 * t + rng.uniform(-0.15, 0.15)))
        tangent = Vector((-math.sin(a), 0.6 * math.cos(a), 0.15)).normalized()
        ob = proto.copy()
        ob.name = f"Tuna_{i:02d}"
        ob.hide_render = False
        ob.hide_viewport = False
        col.objects.link(ob)
        ob.location = p
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = Vector((1, 0, 0)).rotation_difference(tangent)
        s = rng.uniform(0.8, 1.05)
        ob.scale = (s, s, s)
    return angler


# --------------------------------------------------------------------------
# Materials table
# --------------------------------------------------------------------------
def build_materials():
    M = {}
    M["emblem"] = gradient_mat("EmblemBlue", "radial", [(0.0, srgb(44, 132, 222)), (0.55, srgb(22, 88, 176)),
                                                        (1.0, srgb(10, 50, 112))], emit=0.35)
    M["emblem_dark"] = solid_mat("EmblemEye", srgb(14, 64, 140), rough=0.3)
    M["bracket"] = gradient_mat("EmblemLightBlue", "vertical", [(0.0, srgb(70, 150, 240)), (1.0, srgb(150, 215, 255))], emit=0.9)
    M["book"] = gradient_mat("EmblemBookBlue", "vertical", [(0.0, srgb(60, 130, 230)), (1.0, srgb(130, 200, 255))], emit=0.8)
    M["white"] = solid_mat("EmblemWhite", (0.95, 0.97, 1.0), rough=0.2, coat=0.6, emission=(1, 1, 1), strength=0.7)
    M["page_shade"] = solid_mat("EmblemPageShade", srgb(206, 226, 248), rough=0.25, emission=srgb(206, 226, 248), strength=0.6)
    M["text"] = solid_mat("WordmarkWhite", (0.95, 0.97, 1.0), rough=0.15, coat=1.0, emission=(1.0, 1.0, 1.0), strength=0.6)
    M["gold"] = solid_mat("WordmarkGold", srgb(246, 188, 70), rough=0.2, metal=0.3, coat=1.0, emission=srgb(246, 188, 70), strength=0.45)
    M["marlin"] = fish_mat("BlueMarlin", [(0.0, srgb(236, 240, 245)), (0.42, srgb(190, 200, 212)), (0.55, srgb(40, 80, 170)),
                                          (1.0, srgb(14, 34, 96))], metal=0.4)
    M["fin_marlin"] = solid_mat("FinMarlin", srgb(24, 60, 150), rough=0.35, sss=0.3)
    M["whale_shark"] = fish_mat("WhaleShark", [(0.0, srgb(210, 216, 220)), (0.4, srgb(200, 206, 212)), (0.5, srgb(60, 80, 100)),
                                               (1.0, srgb(40, 56, 74))], spots=(srgb(220, 228, 235), 2.2), metal=0.0,
                                sss=0.05, rough=0.55, coat=0.2)
    M["arapaima"] = fish_mat("Arapaima", [(0.0, srgb(200, 170, 140)), (0.5, srgb(110, 112, 90)), (1.0, srgb(60, 66, 54))],
                             along=([(0.0, (1, 1, 1)), (0.35, (1, 1, 1)), (0.6, (0, 0, 0)), (1.0, (0, 0, 0))], srgb(170, 40, 30)),
                             metal=0.1, sss=0.15)
    M["eye"] = solid_mat("FishEye", (0.01, 0.01, 0.012), rough=0.05, coat=1.0)
    M["teeth"] = solid_mat("Teeth", (0.85, 0.85, 0.8), rough=0.3, sss=0.3)
    M["lure"] = emission_mat("AnglerLure", (0.45, 1.0, 0.85), 25.0)
    M["sockeye"] = fish_mat("Sockeye", [(0.0, srgb(230, 200, 180)), (0.3, srgb(200, 30, 26)), (1.0, srgb(150, 18, 18))],
                            along=([(0.0, (0, 0, 0)), (0.74, (0, 0, 0)), (0.82, (1, 1, 1)), (1.0, (1, 1, 1))], srgb(40, 90, 50)),
                            metal=0.05, sss=0.2)
    M["fin_red"] = solid_mat("FinRed", srgb(150, 30, 26), rough=0.4, sss=0.4)
    M["bass"] = fish_mat("BlackSeaBass", [(0.0, srgb(70, 74, 86)), (0.5, srgb(28, 30, 40)), (1.0, srgb(12, 14, 20))],
                         spots=(srgb(150, 160, 175), 9.0), metal=0.1)
    M["fin_dark"] = solid_mat("FinDark", srgb(20, 22, 30), rough=0.4, sss=0.3)
    M["mahi"] = fish_mat("MahiMahi", [(0.0, srgb(250, 230, 90)), (0.45, srgb(170, 210, 60)), (0.8, srgb(40, 160, 110)),
                                      (1.0, srgb(20, 90, 150))], spots=(srgb(40, 110, 200), 7.0), metal=0.25)
    M["fin_blue"] = solid_mat("FinBlue", srgb(30, 100, 190), rough=0.35, sss=0.3)
    M["reef_yellow"] = fish_mat("ReefYellow", [(0.0, srgb(255, 236, 80)), (1.0, srgb(255, 200, 20))], metal=0.05)
    M["reef_blue"] = fish_mat("ReefBlue", [(0.0, srgb(255, 220, 60)), (0.35, srgb(30, 90, 230)), (1.0, srgb(20, 50, 160))], metal=0.1)
    M["reef_orange"] = fish_mat("ReefOrange", [(0.0, srgb(255, 160, 60)), (1.0, srgb(240, 90, 20))],
                                spots=((0.95, 0.95, 0.95), 4.0), metal=0.05)
    M["shark"] = fish_mat("GreatWhite", [(0.0, srgb(240, 240, 236)), (0.4, srgb(230, 230, 228)), (0.48, srgb(110, 118, 126)),
                                         (1.0, srgb(80, 88, 98))], metal=0.05, sss=0.05, rough=0.5, coat=0.2)
    M["angler"] = fish_mat("Anglerfish", [(0.0, srgb(60, 40, 34)), (1.0, srgb(30, 20, 18))], spots=(srgb(15, 10, 9), 6.0),
                           metal=0.0, sss=0.1, rough=0.6, coat=0.6)
    M["tuna"] = fish_mat("Tuna", [(0.0, srgb(225, 230, 235)), (0.45, srgb(170, 180, 190)), (0.7, srgb(30, 50, 110)),
                                  (1.0, srgb(15, 25, 60))], metal=0.45, sss=0.05, rough=0.25)
    M["sand"] = solid_mat("Sand", srgb(150, 140, 110), rough=0.9)
    M["rock"] = solid_mat("Rock", srgb(60, 66, 70), rough=0.8)
    M["kelp"] = solid_mat("Kelp", srgb(40, 90, 50), rough=0.5, sss=0.4)
    M["coral_a"] = solid_mat("CoralPink", srgb(240, 110, 130), rough=0.6, sss=0.4)
    M["coral_b"] = solid_mat("CoralOrange", srgb(250, 150, 60), rough=0.6, sss=0.4)
    M["coral_c"] = solid_mat("CoralPurple", srgb(150, 80, 200), rough=0.6, sss=0.4)
    M["bubble"] = solid_mat("Bubble", (1, 1, 1), rough=0.0, transmission=1.0)
    # water: absorbing, scattering volume (no surface)
    m, nt, out = new_mat("WaterVolume")
    pv = nt.nodes.new("ShaderNodeVolumePrincipled")
    pv.inputs["Color"].default_value = (0.35, 0.65, 0.9, 1.0)
    pv.inputs["Density"].default_value = WATER_DENSITY
    pv.inputs["Absorption Color"].default_value = (0.02, 0.2, 0.45, 1.0)
    pv.inputs["Anisotropy"].default_value = 0.6
    nt.links.new(pv.outputs[0], out.inputs["Volume"])
    M["water"] = m
    # caustic sheet: transparent along voronoi cell edges (bright veins), mostly opaque elsewhere
    m, nt, out = new_mat("CausticSheet")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.6, 0.6, 0.6)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    vo = nt.nodes.new("ShaderNodeTexVoronoi")
    vo.feature = "DISTANCE_TO_EDGE"
    nt.links.new(mp.outputs[0], vo.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.0, 0.09
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 1.0, 0.04
    nt.links.new(vo.outputs["Distance"], mr.inputs["Value"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    hold = nt.nodes.new("ShaderNodeBsdfDiffuse")
    hold.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mr.outputs["Result"], mix.inputs[0])
    nt.links.new(hold.outputs[0], mix.inputs[1])
    nt.links.new(tr.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    M["caustic"] = m
    return M


# --------------------------------------------------------------------------
# Lights, camera, render settings
# --------------------------------------------------------------------------
def build_lights(col, group):
    sd = bpy.data.lights.new("Sun", "SUN")
    sd.energy = 9.0
    sd.angle = math.radians(1.5)
    sd.color = (0.92, 0.97, 1.0)
    sun = link(bpy.data.objects.new("Sun", sd), col)
    sun.rotation_euler = (math.radians(14), math.radians(-10), 0)      # almost straight down, slanting slightly
    # dedicated soft spot on the branding, from in front and above the camera line
    spd = bpy.data.lights.new("BrandingSpot", "SPOT")
    spd.energy = 9000.0
    spd.spot_size = math.radians(24)
    spd.spot_blend = 0.6
    spd.shadow_soft_size = 2.5
    spd.color = (0.9, 0.95, 1.0)
    spot = link(bpy.data.objects.new("BrandingSpot", spd), col)
    spot.location = (0.0, -26.0, 9.0)
    spot.rotation_mode = "QUATERNION"
    spot.rotation_quaternion = (Vector((0, 0, 0)) - spot.location).to_track_quat("-Z", "Y")
    # soft blue rim from behind the branding
    rd = bpy.data.lights.new("BrandingRim", "AREA")
    rd.energy = 2500.0
    rd.size = 10.0
    rd.color = (0.35, 0.65, 1.0)
    rim = link(bpy.data.objects.new("BrandingRim", rd), col)
    rim.location = (0.0, 9.0, 6.0)
    rim.rotation_mode = "QUATERNION"
    rim.rotation_quaternion = (Vector((0, 0, 0)) - rim.location).to_track_quat("-Z", "Y")


def build_camera(sc, col):
    cd = bpy.data.cameras.new("BannerCamera")
    cd.lens = LENS
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = SENSOR
    cd.clip_start = 0.5
    cd.clip_end = 400.0
    cam = link(bpy.data.objects.new("BannerCamera", cd), col)
    cam.location = (0.0, -CAM_DIST, 0.0)
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)               # looking down +Y at the focal plane
    sc.camera = cam
    return cam


def enable_gpu():
    """Pick the first available Cycles GPU backend and enable its devices; fall back to the CPU."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return "CPU"
    for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        try:
            prefs.refresh_devices()
        except AttributeError:
            prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type != "CPU"]
        if gpus:
            for d in prefs.devices:
                d.use = d.type != "CPU"
            return backend
    return "CPU"


def setup_render(sc, preview=False):
    sc.render.engine = "CYCLES"
    backend = enable_gpu()
    sc.cycles.device = "GPU" if backend != "CPU" else "CPU"
    sc.render.resolution_x, sc.render.resolution_y = RES_X, RES_Y
    sc.render.resolution_percentage = 25 if preview else 100
    sc.cycles.samples = 24 if preview else SAMPLES
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.max_bounces = 8
    sc.cycles.transparent_max_bounces = 16
    sc.cycles.volume_bounces = 1
    try:
        sc.cycles.volume_step_rate = 2.0 if preview else 1.0
    except AttributeError:
        pass
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    try:
        sc.view_settings.view_transform = "Standard"      # keeps the logo blues and the white text exact
        sc.view_settings.look = "None"
    except TypeError:
        pass
    sc.render.image_settings.file_format = "PNG"
    print(f"Cycles device: {sc.cycles.device} ({backend}), samples {sc.cycles.samples}")


def report_safe_zone(sc, cam, group):
    """Project every branding vertex to pixels and compare its bounding box with the safe area."""
    from bpy_extras.object_utils import world_to_camera_view
    dg = bpy.context.evaluated_depsgraph_get()
    xs, ys = [], []
    for ob in [group] + list(group.children_recursive):
        if ob.type not in ("MESH", "FONT", "CURVE"):
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        for v in me.vertices:
            c = world_to_camera_view(sc, cam, ev.matrix_world @ v.co)
            xs.append(c.x * RES_X)
            ys.append((1 - c.y) * RES_Y)
        ev.to_mesh_clear()
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx0, sx1 = (RES_X - SAFE_W) / 2, (RES_X + SAFE_W) / 2
    sy0, sy1 = (RES_Y - SAFE_H) / 2, (RES_Y + SAFE_H) / 2
    ok = sx0 <= x0 and x1 <= sx1 and sy0 <= y0 and y1 <= sy1
    print(f"branding box: x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f} px ({x1 - x0:.0f} x {y1 - y0:.0f}); "
          f"safe area x {sx0:.0f}-{sx1:.0f}, y {sy0:.0f}-{sy1:.0f}: {'INSIDE' if ok else 'OUTSIDE'}")
    return ok


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build(preview=False):
    sc, cols = new_scene()
    M = build_materials()
    build_world(sc)
    group, _emblem, _text = build_branding(cols["Branding"], M)
    build_environment(cols["Environment"], M)
    build_marine_life(cols["Marine_Life"], M, cols["Lighting"])
    build_lights(cols["Lighting"], group)
    cam = build_camera(sc, cols["Lighting"])
    setup_render(sc, preview)
    bpy.context.view_layer.update()
    report_safe_zone(sc, cam, group)
    return sc


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out, preview = None, False
    if "--render" in argv:
        out = argv[argv.index("--render") + 1]
    if "--preview" in argv:
        out, preview = argv[argv.index("--preview") + 1], True
    sc = build(preview)
    if out:
        sc.render.filepath = os.path.abspath(out)
        bpy.ops.render.render(write_still=True, scene=sc.name)
        print("saved", sc.render.filepath)


if __name__ == "__main__":
    main()
