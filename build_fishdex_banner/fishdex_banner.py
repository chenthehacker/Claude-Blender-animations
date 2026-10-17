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

Fish (low to mid poly, smooth shaded, subdivided once): upper left, freshwater (sockeye salmon in spawning
colours with the hooked jaw and hump; black sea bass); upper right, pelagic and reef (mahi mahi, reef fish);
lower left, deep sea (anglerfish with a glowing lure and its own point light); lower right, open ocean (great
white shark in the background, a swirling school of tuna). The brief placed the shark both "upper background"
and "lower right": it is in the lower right quadrant, set back in depth and rising towards the upper right.
"""

import math
import os
import random
import sys

import bpy
import bmesh
from mathutils import Euler, Matrix, Quaternion, Vector, noise

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


def fish_outline(s=1.0):
    """The emblem fish: a lens-shaped body and a forked tail, facing right."""
    pts = []
    for i in range(25):
        t = math.pi * i / 24
        pts.append((0.55 * s * math.cos(t), 0.36 * s * math.sin(t) ** 0.9))
    body_top = pts
    body_bot = [(x, -z) for x, z in reversed(pts)]
    tail = [(-0.5 * s, 0.0), (-0.85 * s, 0.32 * s), (-0.76 * s, 0.0), (-0.85 * s, -0.32 * s)]
    return body_top[:-1] + [tail[0], tail[1], tail[2], tail[3]] + body_bot[1:]


def make_emblem(col, M, size):
    root = link(bpy.data.objects.new("FishDex_Emblem", None), col)
    if EMBLEM_SHAPE == "circle":
        out = [(size / 2 * math.cos(2 * math.pi * i / 64), size / 2 * math.sin(2 * math.pi * i / 64)) for i in range(64)]
    else:
        out = rounded_rect(size, size, size * 0.22)
    plate = extrude_outline("Emblem_Plate", out, 0.35 * size / 3, col, M["emblem"], bevel=0.06 * size / 3)
    plate.parent = root
    plate.location = (0, 0.02, 0)
    # viewfinder corner brackets
    k, t, arm = size * 0.33, size * 0.04, size * 0.16
    for sx in (1, -1):
        for sz in (1, -1):
            L = [(0, 0), (arm, 0), (arm, t), (t, t), (t, arm), (0, arm)]
            L = [(sx * (-k + x), sz * (k - z)) if False else (sx * (k - x), sz * (k - z)) for x, z in L]
            if (sx > 0) != (sz > 0):
                L = list(reversed(L))
            br = extrude_outline(f"Viewfinder_{sx}_{sz}", L, 0.06, col, M["bracket"])
            br.parent = root
            br.location = (0, -0.05, 0)
    # open book under the fish: two curved page fans
    for side in (1, -1):
        for j in range(3):
            pts = []
            for i in range(13):
                u = i / 12
                x = side * (0.03 + u * 0.36) * size
                z = (-0.17 - 0.035 * j) * size + 0.05 * size * math.sin(math.pi * u) - 0.02 * size * u
                pts.append((x, z))
            th = 0.022 * size
            outline = pts + [(x, z - th) for x, z in reversed(pts)]
            if side < 0:
                outline = list(reversed(outline))
            pg = extrude_outline(f"Book_{side}_{j}", outline, 0.05, col, M["book"] if j else M["bracket"])
            pg.parent = root
            pg.location = (0, -0.05, 0)
    # the fish, with an eye hole cut as a dark disc in front
    fo = [(x + 0.05 * size, z + 0.05 * size) for x, z in fish_outline(size * 0.42)]
    fish = extrude_outline("Emblem_Fish", fo, 0.08, col, M["white"], bevel=0.01)
    fish.parent = root
    fish.location = (0, -0.07, 0)
    bm = bmesh.new()
    ring = [bm.verts.new((0.05 * size + 0.13 * size + 0.04 * size * math.cos(a), 0.0, 0.05 * size + 0.05 * size + 0.04 * size * math.sin(a)))
            for a in [2 * math.pi * i / 20 for i in range(20)]]
    bm.faces.new(ring)
    eye = mesh_from_bm("Emblem_FishEye", bm, col, [M["emblem"]], smooth=False)
    eye.parent = root
    eye.location = (0, -0.1, 0)
    return root


def find_bold_font():
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return bpy.data.fonts.load(p, check_existing=True), True
            except RuntimeError:
                pass
    return bpy.data.fonts.get("Bfont Regular") or bpy.data.fonts.load("<builtin>"), False


def make_wordmark(col, M, height):
    font, bold = find_bold_font()
    cu = bpy.data.curves.new("FishDex_Text", "FONT")
    cu.body = "FishDex"
    cu.font = font
    cu.size = height
    cu.align_x, cu.align_y = "LEFT", "CENTER"
    cu.extrude = height * 0.09
    cu.bevel_depth = height * 0.018
    cu.bevel_resolution = 3
    if not bold:
        cu.offset = height * 0.025          # thicken the built-in font when no bold face is installed
    cu.materials.append(M["text"])
    ob = link(bpy.data.objects.new("FishDex_Text", cu), col)
    ob.rotation_euler = (math.radians(90), 0, 0)          # stand it up, facing the camera (-Y)
    return ob, font


def build_branding(col, M):
    group = link(bpy.data.objects.new("Branding_Group", None), col)
    emblem_size = 3.3
    emblem = make_emblem(col, M, emblem_size)
    text, _font = make_wordmark(col, M, 2.6)
    emblem.parent = group
    text.parent = group
    bpy.context.view_layer.update()
    tw = text.dimensions.x
    gap = 0.55
    total = emblem_size + gap + tw
    emblem.location = (-total / 2 + emblem_size / 2, 0, 0)
    text.location = (-total / 2 + emblem_size + gap, 0, -0.1)
    # fit inside the safe area with a 6% margin (it is centred on the origin, the safe area's centre)
    limit_w = SAFE_W / PX_PER_UNIT * 0.94
    limit_h = SAFE_H / PX_PER_UNIT * 0.94
    s = min(1.0, limit_w / total, limit_h / emblem_size)
    group.scale = (s, s, s)
    return group, emblem, text


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
    # upper left: freshwater
    sock = make_sockeye(col, M)
    place(sock, (-10.0, 4.0, 4.4), yaw=12, pitch=-4)
    bass = make_bass(col, M)
    place(bass, (-15.5, 12.0, 7.6), yaw=-160, pitch=3, scale=1.2)
    sock2 = make_sockeye(col, M)
    sock2.name = "SockeyeSalmon_Far"
    place(sock2, (-21.0, 22.0, 3.0), yaw=20, scale=0.95)
    # upper right: pelagic and reef
    mahi = make_mahi(col, M)
    place(mahi, (11.2, 5.0, 4.8), yaw=168, pitch=4)
    for i, (mat, loc, yaw) in enumerate(((M["reef_yellow"], (16.5, 3.0, 2.5), 170), (M["reef_blue"], (18.0, 6.0, 6.8), -10),
                                         (M["reef_orange"], (14.5, 9.0, 8.4), 160), (M["reef_yellow"], (21.0, 12.0, 4.2), 175))):
        rf = make_reef(col, M, f"ReefFish_{i}", mat)
        place(rf, loc, yaw=yaw, scale=1.1)
    # lower left: deep sea
    angler, esca, tip = make_angler(col, M)
    place(angler, (-12.5, 2.0, -5.0), yaw=18, scale=1.25)
    bpy.context.view_layer.update()
    ld = bpy.data.lights.new("AnglerLureLight", "POINT")
    ld.energy = 60.0
    ld.color = (0.45, 1.0, 0.85)
    ld.shadow_soft_size = 0.1
    lo = link(bpy.data.objects.new("AnglerLureLight", ld), lights_col)
    lo.parent = angler
    lo.location = tip + Vector((0.05, 0, -0.05))
    # lower right: open ocean apex predators and a tuna school
    shark = make_shark(col, M)
    place(shark, (17.0, 26.0, -3.0), yaw=196, pitch=-9, scale=1.0)
    proto = make_tuna(col, M)
    proto.hide_render = True
    proto.hide_viewport = True
    rng = random.Random(SEED + 1)
    center = Vector((12.5, 6.0, -5.2))
    n = 42
    for i in range(n):                            # a swirling school: a rising helix that tightens
        t = i / n
        a = 2 * math.pi * 2.2 * t + rng.uniform(-0.15, 0.15)
        r = 3.6 - 1.8 * t + rng.uniform(-0.25, 0.25)
        p = center + Vector((r * math.cos(a), 0.6 * r * math.sin(a) + 2.0 * t, -1.2 + 2.6 * t + rng.uniform(-0.2, 0.2)))
        tangent = Vector((-math.sin(a), 0.6 * math.cos(a), 0.18)).normalized()
        ob = proto.copy()
        ob.name = f"Tuna_{i:02d}"
        ob.hide_render = False
        ob.hide_viewport = False
        col.objects.link(ob)
        ob.location = p
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = Vector((1, 0, 0)).rotation_difference(tangent)
        s = rng.uniform(0.75, 1.0)
        ob.scale = (s, s, s)
    return dict(sockeye=sock, bass=bass, mahi=mahi, angler=angler, shark=shark)


# --------------------------------------------------------------------------
# Materials table
# --------------------------------------------------------------------------
def build_materials():
    M = {}
    M["emblem"] = solid_mat("EmblemNavy", srgb(14, 70, 150), rough=0.25, coat=0.8)
    M["bracket"] = solid_mat("EmblemLightBlue", srgb(120, 196, 255), rough=0.2, emission=srgb(120, 196, 255), strength=0.8)
    M["book"] = solid_mat("EmblemBookBlue", srgb(70, 150, 240), rough=0.25, emission=srgb(70, 150, 240), strength=0.5)
    M["white"] = solid_mat("EmblemWhite", (0.9, 0.93, 0.96), rough=0.2, coat=0.6, emission=(1, 1, 1), strength=0.35)
    M["text"] = solid_mat("WordmarkWhite", (0.92, 0.95, 1.0), rough=0.18, coat=1.0, emission=(0.85, 0.92, 1.0), strength=0.25)
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
    mp.inputs["Scale"].default_value = (0.35, 0.35, 0.35)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    vo = nt.nodes.new("ShaderNodeTexVoronoi")
    vo.feature = "DISTANCE_TO_EDGE"
    nt.links.new(mp.outputs[0], vo.inputs["Vector"])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.0, 0.09
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 1.0, 0.12
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
    sd.energy = 6.0
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
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Punchy"
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
