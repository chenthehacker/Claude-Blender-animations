"""Handheld (3DS era) toon style test: a noon beach with a pier, a chibi angler and palms.

Original assets, built from code. The look follows measurements taken from reference captures
(see STYLE_NOTES.md next to this file): lookup table lighting instead of PBR, hand painted low
resolution textures with bilinear filtering, binary alpha cutouts, soft blob shadows, no anti
aliasing, rendered at the 400x240 per eye resolution of the original top screen.

    /root/bpyenv/bin/python beach_style_test.py --check
    /root/bpyenv/bin/python beach_style_test.py --render out.png [--upscale 3] [--water day|lavender]
    /root/bpyenv/bin/python beach_style_test.py --save beach.blend

Coordinates: X east, Y north, Z up (metres, chibi scale). The camera looks north.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys

import bpy  # must come first: the pip bpy module exposes bmesh/mathutils after import
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
RES_X, RES_Y = 400, 240          # top screen, per eye (3dbrew Hardware page)
LENS, SENSOR = 30.0, 36.0        # horizontal sensor fit: about 62 x 40 degrees
CAM_POS = Vector((-3.0, -9.5, 3.0))
CAM_TARGET = Vector((-3.0, 2.0, 0.6))

# Noon sun, direction TO the sun: high (about 70 degrees) and slightly south, so the faces turned to the
# camera catch light. Estimate, not measured.
SUN = Vector((0.15, -0.32, 0.94)).normalized()


def hexc(h):
    """'#RRGGBB' to sRGB floats 0..1 (for painting textures, which are stored in sRGB)."""
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def lin(c):
    """sRGB floats to linear (for shader colour sockets)."""
    c = np.asarray(c, dtype=float)
    return tuple(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


# ---- Palette. M = measured from the reference captures, A = art choice (no daytime sample available).
P = {
    "sky_zenith": "#215DF6",     # M  day beach sky
    "sky_horizon": "#4A80F5",    # A  slightly lighter toward the horizon (measured skies are nearly flat)
    "sand": "#CFBC9A",           # M  day beach sand median
    "sand_hi": "#EBE5B8",        # M  day beach sand light 10%
    "sand_dark": "#B39A78",      # A  speckle
    "cliff": "#A68C8D",          # M  day cliff
    "cliff_dark": "#7E6667",     # A
    "grass": "#7DBB55",          # A
    "grass_dark": "#5C9A43",     # A
    "foliage_dark": "#0E4D3E",   # M  day foliage median
    "foliage_mid": "#2E7D4F",    # A
    "foliage_hi": "#62AD5C",     # A
    "palm_a": "#D09A96",         # M  palm trunk band (pink)
    "palm_b": "#FEDAA8",         # M  palm trunk band (cream)
    "skin": "#E2BF9B",           # M  skin median
    "skin_hi": "#FDF8CD",        # M  skin light 10%
    "wood": "#C8935F",           # A  day dock wood was not sampled; hue kept near the measured H 17
    "wood_dark": "#9A643E",      # A
    "wood_hi": "#E3B884",        # A
    "hair": "#7A5235",           # A
    "hair_hi": "#A8774C",        # A
    "shirt_a": "#3E8FC4",        # A
    "shirt_b": "#F4EEDC",        # A
    "shorts": "#4A4F8C",         # A
    "shoe": "#C9473E",           # A
    "eye": "#1E1A28",            # A
    "blush": "#F2A3A0",          # A
    "rod": "#D9B77A",            # A
    "rod_band": "#6E4A2E",       # A
    "bobber_red": "#E0473A",     # A
    "line": "#E6ECF2",           # A
    "cloud": "#FFFFFF",          # A
    "cloud_under": "#D5DBF2",    # A
}
WATER = {
    # M  day beach sea (k-means clusters of the day capture), foam from the day river capture
    "day": {"shallow": "#8DC2E7", "mid": "#6296C3", "deep": "#3A71AD", "caustic": "#B7DDF4", "foam": "#D1EBE9"},
    # M  violet sea of the rain capture (clusters), foam from the same capture
    "lavender": {"shallow": "#8C8BF5", "mid": "#606CCF", "deep": "#45569C", "caustic": "#A9A8F7", "foam": "#D8B6CB"},
}

# ---- Lookup table lighting (the PICA200 indexes LUTs with N.L; 3dbrew GPU/Fragment_Lighting).
# Measured on the white marlin belly, relative to its median: shadow (dark 10%) and highlight (light 10%).
#   shadow  #686794 / #BDAEC6 = 104/189, 103/174, 148/198 = 0.550, 0.592, 0.747
#   highlight #EFDFD5 / #BDAEC6 = 239/189, 223/174, 213/198 = 1.265, 1.282, 1.076
# These are sRGB ratios; the shader multiplies linear values, so each ratio is raised to 2.2.
LUT_SHADOW = tuple(r ** 2.2 for r in (0.550, 0.592, 0.747))      # 0.268, 0.316, 0.526
LUT_HIGH = tuple(r ** 2.2 for r in (1.265, 1.282, 1.076))        # 1.677, 1.727, 1.175
LUT_SCALE = max(LUT_HIGH)                                        # ramp colours must stay within 0..1
# Ramp input is N.L. Only faces turned away from the sun reach the full shadow tint; side faces stay near
# the painted colour, as the captured characters do.
LUT_STOPS = [(0.00, LUT_SHADOW), (0.08, LUT_SHADOW), (0.38, (1, 1, 1)), (0.965, (1, 1, 1)), (1.00, LUT_HIGH)]

# Characters: half strength shadow tint so faces read (art choice; the captured faces are nearly unshaded).
LUT_SHADOW_CHAR = tuple(1.0 - 0.5 * (1.0 - c) for c in LUT_SHADOW)
LUT_STOPS_CHAR = [(0.00, LUT_SHADOW_CHAR), (0.10, LUT_SHADOW_CHAR), (0.45, (1, 1, 1)), (0.965, (1, 1, 1)),
                  (1.00, LUT_HIGH)]

BLOB_SHADOW = 0.83 ** 2.2        # measured day blob shadow, neutral 0.83 multiply (sRGB) as linear: 0.664


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def shore_y(x):
    return 2.0 + 0.45 * math.sin(0.55 * x + 0.6) + 0.2 * math.sin(1.3 * x + 2.0)


def cliff_y(x):
    return 13.5 + 0.4 * math.sin(0.3 * x + 1.0)


DUNES = [(-12.0, 8.0, 0.45, 2.2), (-5.0, 9.5, 0.35, 2.0), (2.5, 8.5, 0.40, 2.4), (9.0, 9.8, 0.30, 1.8),
         (15.0, 8.2, 0.42, 2.5), (-18.0, 9.0, 0.35, 2.0)]


def terrain_h(x, y):
    s = y - shore_y(x)
    h = 0.075 * min(max(s, -5.0), 9.0)
    for xc, yc, a, sig in DUNES:
        h += a * math.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * sig * sig))
    h += 1.3 * float(smoothstep(cliff_y(x), cliff_y(x) + 1.0, y))
    return h


def collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def new_bm():
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    return bm, uv


def mesh_object(name, bm, col, mats, smooth=True, loc=(0, 0, 0)):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    if smooth:
        me.shade_smooth()
    else:
        me.shade_flat()
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    col.objects.link(ob)
    return ob


def frame_of(d):
    d = Vector(d).normalized()
    ref = Vector((0, 0, 1)) if abs(d.z) < 0.95 else Vector((1, 0, 0))
    u = d.cross(ref).normalized()
    return u, d.cross(u)


def bm_tube(bm, uv, pts, radii, segs, v0=0.0, v1=1.0, cap_end=True, cap_start=False, twist=0.0, mat=0):
    """Tube along a polyline with per loop UVs (u around, v along). Low side counts on purpose."""
    pts = [Vector(p) for p in pts]
    rings = []
    for k, p in enumerate(pts):
        d = (pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)])
        u, w = frame_of(d)
        r = radii[k]
        rings.append([bm.verts.new(p + (u * math.cos(twist + 2 * math.pi * i / segs) +
                                        w * math.sin(twist + 2 * math.pi * i / segs)) * r) for i in range(segs)])
    n = len(pts) - 1
    faces = []
    for k in range(n):
        va, vb = v0 + (v1 - v0) * k / n, v0 + (v1 - v0) * (k + 1) / n
        for i in range(segs):
            j = (i + 1) % segs
            f = bm.faces.new((rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i]))
            f.material_index = mat
            for loop, (uu, vv) in zip(f.loops, ((i / segs, va), ((i + 1) / segs, va), ((i + 1) / segs, vb), (i / segs, vb))):
                loop[uv].uv = (uu, vv)
            faces.append(f)
    for ring, use, rev in ((rings[-1], cap_end, False), (rings[0], cap_start, True)):
        if use:
            f = bm.faces.new(list(reversed(ring)) if rev else ring)
            f.material_index = mat
            c = sum((v.co for v in ring), Vector()) / len(ring)
            for loop in f.loops:
                o = loop.vert.co - c
                loop[uv].uv = (0.5 + 0.5 * o.x / max(radii), 0.5 + 0.5 * o.y / max(radii))
            faces.append(f)
    return faces


def bm_box_uv(bm, uv, center, size, uv_scale=(1.0, 1.0), uv_off=(0.0, 0.0), mat=0):
    """Box with planar UVs per face (grain runs along X on top and sides)."""
    m = Matrix.Translation(Vector(center)) @ Matrix.Diagonal((*size, 1.0))
    res = bmesh.ops.create_cube(bm, size=1.0, matrix=m)
    faces = list({f for v in res["verts"] for f in v.link_faces})
    for f in faces:
        f.material_index = mat
        nrm = f.normal if f.normal.length > 0 else Vector((0, 0, 1))
        for loop in f.loops:
            co = loop.vert.co
            if abs(nrm.z) > 0.5:
                a, b = co.x, co.y
            elif abs(nrm.y) > 0.5:
                a, b = co.x, co.z
            else:
                a, b = co.y, co.z
            loop[uv].uv = (a * uv_scale[0] + uv_off[0], b * uv_scale[1] + uv_off[1])
    return faces


def bm_sphere(bm, uv, center, radius, scale=(1, 1, 1), segs=12, rings=8, mat=0, rot=None):
    m = Matrix.Translation(Vector(center))
    if rot is not None:
        m = m @ rot.to_matrix().to_4x4()
    m = m @ Matrix.Diagonal((*scale, 1.0))
    res = bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=radius, matrix=m, calc_uvs=True)
    faces = list({f for v in res["verts"] for f in v.link_faces})
    for f in faces:
        f.material_index = mat
    return faces


def bm_disc(bm, uv, center, radius, segs=16, mat=0, sx=1.0, sy=1.0):
    c = bm.verts.new(Vector(center))
    ring = [bm.verts.new(Vector(center) + Vector((math.cos(2 * math.pi * i / segs) * radius * sx,
                                                   math.sin(2 * math.pi * i / segs) * radius * sy, 0)))
            for i in range(segs)]
    for i in range(segs):
        f = bm.faces.new((c, ring[i], ring[(i + 1) % segs]))
        f.material_index = mat
        for loop in f.loops:
            o = loop.vert.co - Vector(center)
            loop[uv].uv = (0.5 + 0.5 * o.x / (radius * sx), 0.5 + 0.5 * o.y / (radius * sy))


# --------------------------------------------------------------------------
# Hand painted textures (numpy, tileable, low resolution, sRGB)
# --------------------------------------------------------------------------
RNG = np.random.default_rng(7)


def vnoise(w, h, cx, cy, rng=RNG):
    """Tileable value noise: random grid cx x cy, smooth interpolated to w x h."""
    g = rng.random((cy, cx))
    ys, xs = np.linspace(0, cy, h, endpoint=False), np.linspace(0, cx, w, endpoint=False)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = smoothstep(0, 1, ys - y0)[:, None], smoothstep(0, 1, xs - x0)[None, :]
    y1, x1 = (y0 + 1) % cy, (x0 + 1) % cx
    a, b = g[y0][:, x0], g[y0][:, x1]
    c, d = g[y1][:, x0], g[y1][:, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(w, h, base=4, octaves=3, rng=RNG):
    out, amp, tot = np.zeros((h, w)), 1.0, 0.0
    for o in range(octaves):
        out += amp * vnoise(w, h, base * 2 ** o, base * 2 ** o, rng)
        tot += amp
        amp *= 0.5
    return out / tot


def ramp(t, cols):
    """Blend between hex colours over t in 0..1 (painterly gradient)."""
    t = np.clip(t, 0, 1)[..., None]
    cs = [hexc(c) for c in cols]
    n = len(cs) - 1
    idx = np.minimum((t * n).astype(int), n - 1)
    f = t * n - idx
    a = np.take(np.array(cs), idx[..., 0], axis=0)
    b = np.take(np.array(cs), idx[..., 0] + 1, axis=0)
    return a * (1 - f) + b * f


def make_image(name, rgb, alpha=None):
    h, w = rgb.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=alpha is not None)
    a = np.ones((h, w)) if alpha is None else alpha
    px = np.dstack([np.clip(rgb, 0, 1), np.clip(a, 0, 1)])[::-1]      # Blender rows run bottom up
    img.pixels.foreach_set(px.astype(np.float32).ravel())
    img.pack()
    return img


def tex_sand():
    w = h = 128
    n = fbm(w, h, 4, 3)
    col = ramp(0.35 + 0.5 * (n - 0.5), [P["sand_dark"], P["sand"], P["sand_hi"]])
    speck = RNG.random((h, w))
    col[speck > 0.985] = hexc(P["sand_hi"])
    col[speck < 0.012] = hexc(P["sand_dark"]) * 0.92
    return make_image("T_Sand", col)


def tex_grass():
    w = h = 64
    n = fbm(w, h, 4, 3)
    col = ramp(n, [P["grass_dark"], P["grass"], P["grass"]])
    tuft = RNG.random((h, w)) > 0.97
    col[tuft] = hexc(P["grass_dark"])
    return make_image("T_Grass", col)


def tex_rock():
    w, h = 64, 64
    u = np.linspace(0, 1, w, endpoint=False)[None, :]
    v = np.linspace(0, 1, h, endpoint=False)[:, None]
    streak = 0.5 + 0.5 * np.sin(2 * np.pi * (u * 6 + 0.6 * fbm(w, h, 2, 2)))
    n = fbm(w, h, 4, 2)
    t = 0.55 * streak + 0.45 * n
    col = ramp(t, [P["cliff_dark"], P["cliff"], P["cliff"]]) * (0.88 + 0.12 * v)   # painted: lighter at the lip
    return make_image("T_Rock", col)


def tex_wood():
    """Plank: grain along u with a couple of painted knots."""
    w, h = 128, 32
    u = np.linspace(0, 1, w, endpoint=False)[None, :]
    v = np.linspace(0, 1, h, endpoint=False)[:, None]
    warp = 0.35 * fbm(w, h, 3, 2)
    knot = np.zeros((h, w))
    for ku, kv in ((0.27, 0.45), (0.72, 0.62)):
        du = np.minimum(np.abs(u - ku), 1 - np.abs(u - ku)) * 4.0
        r = np.sqrt(du ** 2 + (v - kv) ** 2)
        knot += np.exp(-r * 9.0) * np.sin(r * 55.0)
    g = 0.5 + 0.5 * np.sin(2 * np.pi * (v * 5.0 + warp) + 3.0 * knot)
    t = 0.45 + 0.4 * (g - 0.5) + 0.25 * (fbm(w, h, 8, 1) - 0.5)
    col = ramp(t, [P["wood_dark"], P["wood"], P["wood_hi"]])
    return make_image("T_Wood", col)


def tex_post_top():
    w = h = 32
    y, x = np.mgrid[0:h, 0:w]
    r = np.hypot(x - w / 2 + 0.5, y - h / 2 + 0.5) / (w / 2)
    rings = 0.5 + 0.5 * np.sin(r * 22.0 + 0.8 * fbm(w, h, 3, 1))
    col = ramp(0.35 + 0.5 * rings * (1 - 0.4 * r), [P["wood_dark"], P["wood"], P["wood_hi"]])
    col[r > 0.86] = hexc(P["wood_dark"])
    return make_image("T_PostTop", col)


def tex_post_side():
    w, h = 32, 64
    u = np.linspace(0, 1, w, endpoint=False)[None, :]
    v = np.linspace(0, 1, h, endpoint=False)[:, None]
    g = 0.5 + 0.5 * np.sin(2 * np.pi * (u * 4 + 0.3 * fbm(w, h, 2, 2)))
    col = ramp(0.3 + 0.45 * g, [P["wood_dark"], P["wood"], P["wood_hi"]])
    wet = smoothstep(0.42, 0.30, v)            # painted waterline: lower part darker and cooler
    col = col * (1 - 0.45 * wet[..., None]) + wet[..., None] * np.array([0.0, 0.03, 0.05])
    return make_image("T_PostSide", col)


def tex_trunk():
    """Chevron bands (pink and cream), painted soft edges, darker at the base and under the crown."""
    w, h = 64, 128
    u = np.linspace(0, 1, w, endpoint=False)[None, :]
    v = np.linspace(0, 1, h, endpoint=False)[:, None]
    tri = np.abs(((u * 2) % 1.0) - 0.5)                 # two chevrons around the trunk
    phase = v * 9.0 + tri * 1.1
    band = smoothstep(0.38, 0.62, np.abs((phase % 1.0) - 0.5) * 2)
    col = ramp(band, [P["palm_a"], P["palm_b"]])
    shade = 0.82 + 0.18 * smoothstep(0.0, 0.35, v) - 0.10 * smoothstep(0.85, 1.0, v)
    return make_image("T_Trunk", col * shade[..., None])


def tex_frond():
    """Leaf card: rib along v, comb of leaflets with binary alpha (alpha test, no partial values)."""
    w, h = 64, 128
    u = np.linspace(0, 1, w)[None, :]
    v = np.linspace(0, 1, h)[:, None]
    du = np.abs(u - 0.5) * 2
    width = np.sin(np.pi * np.clip(v * 1.05, 0, 1)) ** 0.6
    saw = ((v * 16.0 + du * 1.6) % 1.0)
    alpha = (du < width * (0.55 + 0.45 * saw)).astype(float)
    alpha[(du < 0.06) & (v < 0.98)] = 1.0
    col = ramp(0.25 + 0.55 * v + 0.2 * (1 - du), [P["foliage_dark"], P["foliage_mid"], P["foliage_hi"]])
    col = col * (0.75 + 0.25 * (1 - saw[..., ]))[..., None]         # painted leaflet separation
    rib = np.broadcast_to(du < 0.05, (h, w))
    col[rib] = hexc("#9CC870")
    return make_image("T_Frond", col, alpha)


def tex_hair():
    w, h = 64, 32
    u = np.linspace(0, 1, w, endpoint=False)[None, :]
    strands = 0.5 + 0.5 * np.sin(2 * np.pi * u * 14 + 2.0 * fbm(w, h, 4, 1))
    v = np.linspace(0, 1, h, endpoint=False)[:, None]
    col = ramp(0.3 + 0.35 * strands + 0.35 * v, [P["hair"], P["hair"], P["hair_hi"]])
    return make_image("T_Hair", col)


def tex_shirt():
    w, h = 32, 32
    v = np.linspace(0, 1, h, endpoint=False)[:, None] * np.ones((1, w))
    stripe = smoothstep(0.42, 0.58, np.abs(((v * 4) % 1.0) - 0.5) * 2)
    col = ramp(stripe, [P["shirt_a"], P["shirt_b"]])
    return make_image("T_Shirt", col)


def tex_rod():
    w, h = 8, 64
    v = np.linspace(0, 1, h, endpoint=False)[:, None] * np.ones((1, w))
    band = ((v * 10) % 1.0) < 0.12
    col = np.where(band[..., None], hexc(P["rod_band"]), hexc(P["rod"]))
    return make_image("T_Rod", col)


def tex_bobber():
    w, h = 16, 16
    v = np.linspace(0, 1, h, endpoint=False)[:, None] * np.ones((1, w))
    col = np.where((v > 0.5)[..., None], hexc(P["bobber_red"]), hexc("#F7F3EA"))
    return make_image("T_Bobber", col)


def tex_caustic():
    """Tileable caustic web: bright where the two nearest cell points are nearly equidistant,
    domain warped so the lines wave, with soft painted edges."""
    w = h = 128
    k = 6
    rng = np.random.default_rng(5)
    pts = rng.random((k * k, 2))
    pts = (pts + np.stack(np.meshgrid(np.arange(k), np.arange(k)), -1).reshape(-1, 2)) / k
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    x = x + 0.22 * (fbm(w, h, 2, 3, rng) - 0.5)
    y = y + 0.22 * (fbm(w, h, 2, 3, rng) - 0.5)
    d = []
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            p = pts + np.array([ox, oy])
            d.append(np.hypot(x[..., None] - p[:, 0], y[..., None] - p[:, 1]))
    d = np.sort(np.concatenate(d, axis=-1), axis=-1)
    web = smoothstep(0.05, 0.0, d[..., 1] - d[..., 0]) * (0.6 + 0.4 * fbm(w, h, 4, 1, rng))
    return make_image("T_Caustic", np.dstack([web, web, web]))


def tex_cloud():
    """Soft cumulus card: overlapping round puffs (round in card space), flat shaded base."""
    w, h = 128, 48
    y, x = np.mgrid[0:h, 0:w]
    u, vb = x / (w - 1), 1.0 - y / (h - 1)          # vb: 0 at the bottom row of the card
    aspect = w / h
    a = np.zeros((h, w))
    for cx, cy, r in ((0.22, 0.30, 0.22), (0.36, 0.45, 0.32), (0.52, 0.52, 0.38), (0.68, 0.42, 0.30),
                      (0.80, 0.30, 0.20), (0.45, 0.28, 0.25), (0.62, 0.27, 0.24)):
        a = np.maximum(a, 1.0 - np.hypot((u - cx) * aspect, vb - cy) / r)
    a = smoothstep(0.0, 0.18, a) * smoothstep(0.10, 0.22, vb)
    shade = ramp(smoothstep(0.12, 0.6, vb), [P["cloud_under"], P["cloud"], P["cloud"]])
    return make_image("T_Cloud", shade, a)


# --------------------------------------------------------------------------
# Materials: unlit emission with lookup table lighting (no PBR, no lights, no ray tracing)
# --------------------------------------------------------------------------
def link(nt, a, b):
    nt.links.new(a, b)


def lut_group(name="LUT_Light", lut_stops=None):
    lut_stops = LUT_STOPS if lut_stops is None else lut_stops
    ng = bpy.data.node_groups.new(name, "ShaderNodeTree")
    ng.interface.new_socket("Color", in_out="INPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket("Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    geo = ng.nodes.new("ShaderNodeNewGeometry")
    dot = ng.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    link(ng, geo.outputs["Normal"], dot.inputs[0])
    dot.inputs[1].default_value = tuple(SUN)
    r = ng.nodes.new("ShaderNodeValToRGB")
    r.color_ramp.interpolation = "LINEAR"
    link(ng, dot.outputs["Value"], r.inputs[0])
    els = r.color_ramp.elements
    stops = [(p, tuple(c / LUT_SCALE for c in col)) for p, col in lut_stops]
    els[0].position, els[0].color = stops[0][0], (*stops[0][1], 1)
    els[1].position, els[1].color = stops[-1][0], (*stops[-1][1], 1)
    for p, c in stops[1:-1]:
        els.new(p).color = (*c, 1)
    mul = ng.nodes.new("ShaderNodeMix")
    mul.data_type, mul.blend_type = "RGBA", "MULTIPLY"
    mul.inputs[0].default_value = 1.0
    link(ng, gi.outputs[0], mul.inputs[6])
    link(ng, r.outputs["Color"], mul.inputs[7])
    sc = ng.nodes.new("ShaderNodeVectorMath")
    sc.operation = "SCALE"
    sc.inputs[3].default_value = LUT_SCALE
    link(ng, mul.outputs[2], sc.inputs[0])
    link(ng, sc.outputs["Vector"], go.inputs[0])
    return ng


LUT = None
LUT_CHAR = None


def material(name, color=None, image=None, lit=True, cutout=False, uv_scale=None, method=None, char=False):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    alpha = None
    if image is not None:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image, t.interpolation, t.extension = image, "Linear", "REPEAT"      # bilinear, like the captures
        if uv_scale is not None:
            tc = nt.nodes.new("ShaderNodeTexCoord")
            mp = nt.nodes.new("ShaderNodeMapping")
            mp.inputs["Scale"].default_value = (*uv_scale, 1)
            link(nt, tc.outputs["UV"], mp.inputs["Vector"])
            link(nt, mp.outputs["Vector"], t.inputs["Vector"])
        col = t.outputs["Color"]
        alpha = t.outputs["Alpha"]
    else:
        rgb = nt.nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = (*lin(hexc(color)), 1)
        col = rgb.outputs[0]
    if lit:
        g = nt.nodes.new("ShaderNodeGroup")
        g.node_tree = LUT_CHAR if char else LUT
        link(nt, col, g.inputs[0])
        col = g.outputs[0]
    em = nt.nodes.new("ShaderNodeEmission")
    link(nt, col, em.inputs["Color"])
    shader = em.outputs[0]
    if cutout:
        gt = nt.nodes.new("ShaderNodeMath")
        gt.operation = "GREATER_THAN"
        gt.inputs[1].default_value = 0.5            # alpha test: binary, never partial
        link(nt, alpha, gt.inputs[0])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mx = nt.nodes.new("ShaderNodeMixShader")
        link(nt, gt.outputs[0], mx.inputs[0])
        link(nt, tr.outputs[0], mx.inputs[1])
        link(nt, shader, mx.inputs[2])
        shader = mx.outputs[0]
        m.use_backface_culling = False
    link(nt, shader, out.inputs["Surface"])
    if method:
        m.surface_render_method = method
    return m


def blob_material(name, box=False, edge=0.35):
    """Soft blob shadow: a neutral multiply (Transparent BSDF tint) that fades out at the rim."""
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    link(nt, tc.outputs["UV"], sep.inputs[0])

    def centred(sock):
        a = nt.nodes.new("ShaderNodeMath")
        a.operation = "SUBTRACT"
        link(nt, sock, a.inputs[0])
        a.inputs[1].default_value = 0.5
        b = nt.nodes.new("ShaderNodeMath")
        b.operation = "ABSOLUTE"
        link(nt, a.outputs[0], b.inputs[0])
        c = nt.nodes.new("ShaderNodeMath")
        c.operation = "MULTIPLY"
        link(nt, b.outputs[0], c.inputs[0])
        c.inputs[1].default_value = 2.0
        return c.outputs[0]
    du, dv = centred(sep.outputs[0]), centred(sep.outputs[1])
    comb = nt.nodes.new("ShaderNodeMath")
    if box:
        comb.operation = "MAXIMUM"
        link(nt, du, comb.inputs[0])
        link(nt, dv, comb.inputs[1])
    else:
        sq = []
        for d in (du, dv):
            s = nt.nodes.new("ShaderNodeMath")
            s.operation = "MULTIPLY"
            link(nt, d, s.inputs[0])
            link(nt, d, s.inputs[1])
            sq.append(s.outputs[0])
        comb.operation = "ADD"
        link(nt, sq[0], comb.inputs[0])
        link(nt, sq[1], comb.inputs[1])
        root = nt.nodes.new("ShaderNodeMath")
        root.operation = "SQRT"
        link(nt, comb.outputs[0], root.inputs[0])
        comb = root
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    link(nt, comb.outputs[0], mr.inputs["Value"])
    mr.inputs["From Min"].default_value = 1.0 - edge
    mr.inputs["From Max"].default_value = 1.0
    mr.inputs["To Min"].default_value = BLOB_SHADOW
    mr.inputs["To Max"].default_value = 1.0
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    link(nt, mr.outputs["Result"], tr.inputs["Color"])
    link(nt, tr.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "BLENDED"
    return m


def cloud_material(img):
    m = bpy.data.materials.new("M_Cloud")
    try:
        m.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image, t.interpolation = img, "Linear"
    t.extension = "CLIP"
    em = nt.nodes.new("ShaderNodeEmission")
    link(nt, t.outputs["Color"], em.inputs["Color"])
    fac = nt.nodes.new("ShaderNodeMath")
    fac.operation = "MULTIPLY"
    link(nt, t.outputs["Alpha"], fac.inputs[0])
    fac.inputs[1].default_value = 0.75              # slightly cloudy: soft, partly see through puffs
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mx = nt.nodes.new("ShaderNodeMixShader")
    link(nt, fac.outputs[0], mx.inputs[0])
    link(nt, tr.outputs[0], mx.inputs[1])
    link(nt, em.outputs[0], mx.inputs[2])
    link(nt, mx.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "BLENDED"
    return m


def water_material(kind, caustic):
    """Flat lit water: depth colour from the shore distance attribute, caustic web, concentric shore foam."""
    W = WATER[kind]
    m = bpy.data.materials.new("M_Water")
    try:
        m.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "shore"
    d = at.outputs["Fac"]

    # depth colour ramp (metres from the shoreline)
    dr = nt.nodes.new("ShaderNodeMapRange")
    link(nt, d, dr.inputs["Value"])
    dr.inputs["From Min"].default_value, dr.inputs["From Max"].default_value = 0.0, 6.0
    cr = nt.nodes.new("ShaderNodeValToRGB")
    link(nt, dr.outputs["Result"], cr.inputs[0])
    els = cr.color_ramp.elements
    els[0].position, els[0].color = 0.0, (*lin(hexc(W["shallow"])), 1)
    els[1].position, els[1].color = 1.0, (*lin(hexc(W["deep"])), 1)
    els.new(0.25).color = (*lin(hexc(W["mid"])), 1)

    # caustic web, object space so it tiles over the whole sea
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.16, 0.30, 1)     # stretched along the shore
    link(nt, tc.outputs["Object"], mp.inputs["Vector"])
    ct = nt.nodes.new("ShaderNodeTexImage")
    ct.image, ct.interpolation = caustic, "Linear"
    link(nt, mp.outputs["Vector"], ct.inputs["Vector"])
    cfade = nt.nodes.new("ShaderNodeMapRange")
    link(nt, d, cfade.inputs["Value"])
    cfade.inputs["From Min"].default_value, cfade.inputs["From Max"].default_value = 0.0, 7.0
    cfade.inputs["To Min"].default_value, cfade.inputs["To Max"].default_value = 0.55, 0.22
    cm = nt.nodes.new("ShaderNodeMath")
    cm.operation = "MULTIPLY"
    link(nt, ct.outputs["Color"], cm.inputs[0])
    link(nt, cfade.outputs["Result"], cm.inputs[1])
    mix1 = nt.nodes.new("ShaderNodeMix")
    mix1.data_type = "RGBA"
    link(nt, cm.outputs[0], mix1.inputs[0])
    link(nt, cr.outputs["Color"], mix1.inputs[6])
    mix1.inputs[7].default_value = (*lin(hexc(W["caustic"])), 1)

    # concentric foam bands following the coast: |fract(d * k + noise) - 0.5| thin lines, fading offshore
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 0.45
    link(nt, tc.outputs["Object"], nz.inputs["Vector"])
    nzs = nt.nodes.new("ShaderNodeMath")
    nzs.operation = "MULTIPLY_ADD"
    link(nt, nz.outputs["Fac"], nzs.inputs[0])
    nzs.inputs[1].default_value = 0.9
    link(nt, d, nzs.inputs[2])
    k = nt.nodes.new("ShaderNodeMath")
    k.operation = "MULTIPLY"
    link(nt, nzs.outputs[0], k.inputs[0])
    k.inputs[1].default_value = 0.85
    fr = nt.nodes.new("ShaderNodeMath")
    fr.operation = "FRACT"
    link(nt, k.outputs[0], fr.inputs[0])
    band = nt.nodes.new("ShaderNodeMapRange")
    band.interpolation_type = "SMOOTHSTEP"
    link(nt, fr.outputs[0], band.inputs["Value"])
    band.inputs["From Min"].default_value, band.inputs["From Max"].default_value = 0.80, 0.93
    fade = nt.nodes.new("ShaderNodeMapRange")
    link(nt, d, fade.inputs["Value"])
    fade.inputs["From Min"].default_value, fade.inputs["From Max"].default_value = 0.0, 3.2
    fade.inputs["To Min"].default_value, fade.inputs["To Max"].default_value = 0.9, 0.0
    edge = nt.nodes.new("ShaderNodeMapRange")          # bright lip right at the waterline
    link(nt, d, edge.inputs["Value"])
    edge.inputs["From Min"].default_value, edge.inputs["From Max"].default_value = 0.05, 0.28
    edge.inputs["To Min"].default_value, edge.inputs["To Max"].default_value = 1.0, 0.0
    fm = nt.nodes.new("ShaderNodeMath")
    fm.operation = "MULTIPLY"
    link(nt, band.outputs["Result"], fm.inputs[0])
    link(nt, fade.outputs["Result"], fm.inputs[1])
    fx = nt.nodes.new("ShaderNodeMath")
    fx.operation = "MAXIMUM"
    link(nt, fm.outputs[0], fx.inputs[0])
    link(nt, edge.outputs["Result"], fx.inputs[1])
    mix2 = nt.nodes.new("ShaderNodeMix")
    mix2.data_type = "RGBA"
    link(nt, fx.outputs[0], mix2.inputs[0])
    link(nt, mix1.outputs[2], mix2.inputs[6])
    mix2.inputs[7].default_value = (*lin(hexc(W["foam"])), 1)
    # broad swell bands parallel to the coast (lighter and darker rows), as in the captured beaches
    sw = nt.nodes.new("ShaderNodeMath")
    sw.operation = "SINE"
    sm = nt.nodes.new("ShaderNodeMath")
    sm.operation = "MULTIPLY"
    link(nt, nzs.outputs[0], sm.inputs[0])
    sm.inputs[1].default_value = 2.6
    link(nt, sm.outputs[0], sw.inputs[0])
    swr = nt.nodes.new("ShaderNodeMapRange")
    link(nt, sw.outputs[0], swr.inputs["Value"])
    swr.inputs["From Min"].default_value, swr.inputs["From Max"].default_value = -1.0, 1.0
    swr.inputs["To Min"].default_value, swr.inputs["To Max"].default_value = 0.90, 1.07
    scl = nt.nodes.new("ShaderNodeVectorMath")
    scl.operation = "SCALE"
    link(nt, mix2.outputs[2], scl.inputs[0])
    link(nt, swr.outputs["Result"], scl.inputs[3])
    em = nt.nodes.new("ShaderNodeEmission")
    link(nt, scl.outputs["Vector"], em.inputs["Color"])
    link(nt, em.outputs[0], out.inputs["Surface"])
    return m


def sand_material(img):
    """Lit sand, darkened toward the waterline (wet attribute) as a painted, baked band."""
    m = material("M_Sand", image=img)
    nt = m.node_tree
    tex = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeTexImage")
    grp = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeGroup")
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "wet"
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type, mx.blend_type = "RGBA", "MULTIPLY"
    link(nt, at.outputs["Fac"], mx.inputs[0])
    link(nt, tex.outputs["Color"], mx.inputs[6])
    mx.inputs[7].default_value = (*lin(hexc("#B8A79A")), 1)       # A: wet sand multiplier, slightly cool
    link(nt, mx.outputs[2], grp.inputs[0])
    return m


# --------------------------------------------------------------------------
# Scene parts
# --------------------------------------------------------------------------
def build_terrain(col, M):
    xs = np.arange(-45.0, 45.01, 1.0)
    ys = np.concatenate([np.arange(-8.0, 1.0, 1.0), np.arange(1.0, 16.0, 0.5), np.arange(16.0, 47.0, 2.0)])
    bm, uv = new_bm()
    wet = bm.verts.layers.float.new("wet")
    grid = []
    for y in ys:
        row = []
        for x in xs:
            h = terrain_h(x, y)
            v = bm.verts.new((x, y, h))
            v[wet] = float(1.0 - smoothstep(0.02, 0.22, h))
            row.append(v)
        grid.append(row)
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            f = bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
            f.normal_update()
            c = f.calc_center_median()
            if c.y > cliff_y(c.x) + 0.85 and f.normal.z > 0.85:
                f.material_index = 1          # grass
            elif f.normal.z < 0.8 and c.y > 11.5:
                f.material_index = 2          # rock
            else:
                f.material_index = 0          # sand
            for loop in f.loops:
                co = loop.vert.co
                loop[uv].uv = (co.x * 0.3, (co.z if f.material_index == 2 else co.y) * 0.3)
    return mesh_object("Terrain", bm, col, [M["sand"], M["grass"], M["rock"]])


def build_water(col, M):
    xs = np.arange(-30.0, 30.01, 0.5)
    ys = np.arange(-14.0, 5.51, 0.5)
    bm, uv = new_bm()
    sh = bm.verts.layers.float.new("shore")
    grid = []
    for y in ys:
        row = []
        for x in xs:
            v = bm.verts.new((x, y, 0.0))
            v[sh] = float(shore_y(x) - y)
            row.append(v)
        grid.append(row)
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
    return mesh_object("Water", bm, col, [M["water"]], smooth=False)


PIER_X0, PIER_X1 = -6.15, -4.45
PIER_Y0, PIER_Y1 = -4.2, 4.0
DECK_Z = 0.62


def build_pier(col, M):
    rng = np.random.default_rng(3)
    bm, uv = new_bm()
    y = PIER_Y0
    while y < PIER_Y1 - 0.05:
        wpl = 0.30
        dz = rng.uniform(-0.012, 0.012)
        bm_box_uv(bm, uv, (0.5 * (PIER_X0 + PIER_X1), y + wpl / 2, DECK_Z - 0.045 + dz),
                  (PIER_X1 - PIER_X0 + rng.uniform(-0.04, 0.04), wpl - 0.035, 0.09),
                  uv_scale=(0.55, 1.6), uv_off=(rng.random(), rng.random()), mat=0)
        y += wpl
    for xb in (PIER_X0 + 0.25, PIER_X1 - 0.25):          # stringers under the planks
        bm_box_uv(bm, uv, (xb, 0.5 * (PIER_Y0 + PIER_Y1), DECK_Z - 0.15), (0.14, PIER_Y1 - PIER_Y0, 0.12),
                  uv_scale=(1.6, 0.55), mat=0)
    deck = mesh_object("PierDeck", bm, col, [M["wood"]], smooth=False)

    bm, uv = new_bm()
    posts = []
    for yy in np.linspace(PIER_Y0 + 0.1, PIER_Y1 - 0.4, 5):
        for xx in (PIER_X0 - 0.02, PIER_X1 + 0.02):
            base = min(terrain_h(xx, yy), 0.0) - 0.6
            top = DECK_Z + 0.16
            bm_tube(bm, uv, [(xx, yy, base), (xx, yy, top)], [0.13, 0.12], 8, cap_end=False, mat=0)
            bm_disc(bm, uv, (xx, yy, top), 0.12, segs=8, mat=1)
            posts.append((xx, yy))
    mesh_object("PierPosts", bm, col, [M["post_side"], M["post_top"]], smooth=True)
    return deck, posts


VILLAGER = Vector((-4.78, -3.15, DECK_Z))     # feet; faces east (+X)
BOBBER = Vector((-1.35, -2.55, 0.02))


def build_villager(col, M):
    """Original chibi angler: head about as tall as the body, 6 to 12 sided parts, painted clothes."""
    o = VILLAGER
    parts = {}
    # head and hair
    bm, uv = new_bm()
    head_c = o + Vector((0, 0, 0.74))
    bm_sphere(bm, uv, head_c, 0.27, scale=(0.97, 1.0, 0.93), segs=12, rings=8)
    bm_sphere(bm, uv, head_c + Vector((0.255, 0, -0.035)), 0.035, segs=6, rings=4)       # nose
    for side in (-1, 1):                                                              # ears
        bm_sphere(bm, uv, head_c + Vector((0.0, side * 0.255, -0.02)), 0.05, scale=(0.6, 0.5, 1.0), segs=6, rings=4)
    parts["head"] = mesh_object("Head", bm, col, [M["skin"]])

    bm, uv = new_bm()
    hres = bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=0.295,
                                     matrix=Matrix.Translation(head_c + Vector((-0.015, 0, 0.02))) @
                                     Matrix.Diagonal((1.0, 1.04, 0.98, 1.0)), calc_uvs=True)
    hfaces = list({f for v in hres["verts"] for f in v.link_faces})
    kill = []
    for f in hfaces:
        c = f.calc_center_median() - head_c
        keep = (c.z > 0.08) or (c.x < -0.06 and c.z > -0.20) or (c.x < 0.03 and c.z > 0.0 and abs(c.y) > 0.2)
        if not keep:
            kill.append(f)
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    for k, (yy, rz) in enumerate(((-0.14, 0.0), (-0.03, 0.0), (0.09, 0.0))):         # bangs: three low wedges
        bm_tube(bm, uv, [head_c + Vector((0.20, yy, 0.17)), head_c + Vector((0.255, yy + 0.02, 0.07))],
                [0.075, 0.02], 5, cap_end=False, cap_start=True)
    parts["hair"] = mesh_object("Hair", bm, col, [M["hair"]])

    bm, uv = new_bm()
    for side in (-1, 1):
        d = Vector((math.cos(math.radians(32)), side * math.sin(math.radians(32)), 0.0))
        p = head_c + Vector((d.x * 0.24, d.y * 0.245, 0.015))
        rot = Vector((1, 0, 0)).rotation_difference(d)
        bm_sphere(bm, uv, p, 0.05, scale=(0.45, 0.85, 1.3), segs=8, rings=5, rot=rot)
    parts["eyes"] = mesh_object("Eyes", bm, col, [M["eye"]])
    bm, uv = new_bm()
    for side in (-1, 1):
        d = Vector((math.cos(math.radians(44)), side * math.sin(math.radians(44)), 0))
        p = head_c + Vector((d.x * 0.235, d.y * 0.235, -0.07))
        rot = Vector((1, 0, 0)).rotation_difference(d)
        bm_sphere(bm, uv, p, 0.035, scale=(0.3, 1.0, 0.7), segs=6, rings=4, rot=rot)
    parts["blush"] = mesh_object("Blush", bm, col, [M["blush"]])

    # body (striped shirt), shorts, legs, shoes, arms, hands
    bm, uv = new_bm()
    bm_tube(bm, uv, [o + Vector((0, 0, 0.18)), o + Vector((0, 0, 0.47))], [0.17, 0.12], 8, v0=0.0, v1=1.0,
            cap_end=True, cap_start=True)
    for side in (-1, 1):
        sh = o + Vector((0.0, side * 0.13, 0.42))
        hand = o + Vector((0.24, side * 0.05, 0.33))
        bm_tube(bm, uv, [sh, hand], [0.055, 0.045], 6, cap_end=False)
    parts["body"] = mesh_object("Body", bm, col, [M["shirt"]])
    bm, uv = new_bm()
    bm_tube(bm, uv, [o + Vector((0, 0, 0.10)), o + Vector((0, 0, 0.20))], [0.165, 0.17], 8, cap_start=True)
    parts["shorts"] = mesh_object("Shorts", bm, col, [M["shorts"]])
    bm, uv = new_bm()
    for side in (-1, 1):
        bm_tube(bm, uv, [o + Vector((0, side * 0.07, 0.12)), o + Vector((0, side * 0.07, 0.04))], [0.05, 0.045], 6)
        bm_sphere(bm, uv, o + Vector((0.03, side * 0.07, 0.035)), 0.06, scale=(1.3, 0.85, 0.6), segs=8, rings=5)
    parts["legs"] = mesh_object("Shoes", bm, col, [M["shoe"]])
    bm, uv = new_bm()
    for side in (-1, 1):
        bm_sphere(bm, uv, o + Vector((0.25, side * 0.045, 0.33)), 0.05, segs=8, rings=5)
    parts["hands"] = mesh_object("Hands", bm, col, [M["skin"]])

    # rod, line, bobber
    grip = o + Vector((0.25, 0.0, 0.33))
    tip = o + Vector((1.55, 0.25, 1.62))
    bm, uv = new_bm()
    bm_tube(bm, uv, [grip + Vector((-0.12, 0, -0.08)), tip], [0.022, 0.009], 6)
    bm_tube(bm, uv, [grip + Vector((0.08, 0.04, 0.04)), grip + Vector((0.08, 0.10, 0.04))], [0.04, 0.04], 8,
            cap_start=True)                                                            # reel
    parts["rod"] = mesh_object("Rod", bm, col, [M["rod"]])
    bm, uv = new_bm()
    pts = []
    for k in range(7):
        t = k / 6
        p = tip.lerp(BOBBER + Vector((0, 0, 0.07)), t)
        p.z -= 0.35 * 4 * t * (1 - t)                                                 # gentle sag
        pts.append(p)
    bm_tube(bm, uv, pts, [0.011] * len(pts), 4, cap_end=False)
    parts["line"] = mesh_object("Line", bm, col, [M["line"]])
    bm, uv = new_bm()
    bm_sphere(bm, uv, BOBBER + Vector((0, 0, 0.04)), 0.075, scale=(1, 1, 1.15), segs=8, rings=6)
    parts["bobber"] = mesh_object("Bobber", bm, col, [M["bobber"]])
    # ripple ring around the bobber
    bm, uv = new_bm()
    ring_in, ring_out = [], []
    for i in range(16):
        a = 2 * math.pi * i / 16
        ring_in.append(bm.verts.new(BOBBER + Vector((math.cos(a) * 0.20, math.sin(a) * 0.20, 0.005))))
        ring_out.append(bm.verts.new(BOBBER + Vector((math.cos(a) * 0.27, math.sin(a) * 0.27, 0.005))))
    for i in range(16):
        j = (i + 1) % 16
        bm.faces.new((ring_in[i], ring_out[i], ring_out[j], ring_in[j]))
    parts["ripple"] = mesh_object("Ripple", bm, col, [M["foam"]], smooth=False)
    return parts


PALMS = [  # base x, y, height, lean direction (deg from +X), lean amount
    (-11.0, 8.6, 3.0, 200, 0.9),
    (-6.8, 10.2, 3.3, 160, 0.7),
    (1.2, 9.4, 3.1, 20, 0.9),
    (4.9, 8.0, 2.8, 330, 0.8),
    (11.5, 10.6, 3.4, 10, 0.6),
]


def build_palm(col, M, x, y, height, lean_deg, lean, rng):
    z0 = terrain_h(x, y) - 0.1
    ld = Vector((math.cos(math.radians(lean_deg)), math.sin(math.radians(lean_deg)), 0))
    n = 6
    pts = [Vector((x, y, z0)) + ld * (lean * (k / n) ** 2.2) + Vector((0, 0, height * k / n)) for k in range(n + 1)]
    bm, uv = new_bm()
    for k in range(n):                                       # stacked segments, each slightly flared at its base
        r0 = 0.20 - 0.012 * k
        bm_tube(bm, uv, [pts[k], pts[k + 1] + (pts[k + 1] - pts[k]) * 0.04], [r0 * 1.12, r0 * 0.9], 7,
                v0=k / n, v1=(k + 1) / n, cap_end=True, twist=0.4 * k)
    trunk = mesh_object("PalmTrunk", bm, col, [M["trunk"]])
    crown = pts[-1]
    bm, uv = new_bm()
    nf = 8
    for i in range(nf):
        th = 2 * math.pi * i / nf + rng.uniform(-0.2, 0.2)
        d = Vector((math.cos(th), math.sin(th), 0))
        side = Vector((-d.y, d.x, 0))
        L = rng.uniform(1.6, 2.0)
        pitch = rng.uniform(0.35, 0.6)
        droop = rng.uniform(1.1, 1.5)
        rows = []
        for k in range(5):
            t = k / 4
            c = crown + d * (L * t) + Vector((0, 0, L * (pitch * t - droop * 0.5 * t * t)))
            wdt = 0.42 * math.sin(math.pi * min(t * 1.05, 1.0)) ** 0.6 + 0.04
            rows.append([bm.verts.new(c - side * wdt + Vector((0, 0, -0.06 * wdt))),
                         bm.verts.new(c + Vector((0, 0, 0.05 * wdt))),
                         bm.verts.new(c + side * wdt + Vector((0, 0, -0.06 * wdt)))])
        for k in range(4):
            for col_i in range(2):
                f = bm.faces.new((rows[k][col_i], rows[k][col_i + 1], rows[k + 1][col_i + 1], rows[k + 1][col_i]))
                for loop in f.loops:
                    kk = next(a for a in range(5) if loop.vert in rows[a])
                    cc = rows[kk].index(loop.vert)
                    loop[uv].uv = (cc / 2.0, kk / 4.0)
    fronds = mesh_object("PalmFronds", bm, col, [M["frond"]])
    bm, uv = new_bm()
    for k in range(3):
        a = 2 * math.pi * k / 3
        bm_sphere(bm, uv, crown + Vector((math.cos(a) * 0.14, math.sin(a) * 0.14, -0.12)), 0.11, segs=8, rings=5)
    mesh_object("Coconuts", bm, col, [M["coconut"]])
    return trunk, fronds, crown


def build_bushes(col, M, rng):
    bm, uv = new_bm()
    for x in np.arange(-40, 41, 4.5):
        x += rng.uniform(-1.2, 1.2)
        y = cliff_y(x) + rng.uniform(4.0, 9.0)
        z = terrain_h(x, y)
        r = rng.uniform(0.7, 1.2)
        res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=r,
                                         matrix=Matrix.Translation((x, y, z + 0.55 * r)) @
                                         Matrix.Diagonal((1.2, 1.0, 0.85, 1.0)), calc_uvs=False)
        for v in res["verts"]:
            v.co += Vector(rng.uniform(-0.08, 0.08, 3))
    return mesh_object("Bushes", bm, col, [M["bush"]])


def build_shells(col, M):
    bm, uv = new_bm()
    for x, y, a in ((-1.8, 4.6, 0.3), (3.9, 3.7, 1.9)):
        z = terrain_h(x, y)
        res = bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=4, radius=0.13,
                                        matrix=Matrix.Translation((x, y, z)) @ Matrix.Rotation(a, 4, "Z") @
                                        Matrix.Diagonal((1.0, 0.85, 0.45, 1.0)), calc_uvs=False)
        bmesh.ops.delete(bm, geom=[v for v in res["verts"] if v.co.z < z - 0.001], context="VERTS")
    return mesh_object("Shells", bm, col, [M["shell"]])


def blob(col, mat, name, center, rx, ry=None):
    ry = rx if ry is None else ry
    bm, uv = new_bm()
    bm_disc(bm, uv, Vector(center), 1.0, segs=16, sx=rx, sy=ry)
    ob = mesh_object(name, bm, col, [mat], smooth=False)
    ob.visible_shadow = False
    return ob


def rect_blob(col, mat, name, x0, x1, y0, y1, z):
    bm, uv = new_bm()
    vs = [bm.verts.new(p) for p in ((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))]
    f = bm.faces.new(vs)
    for loop, c in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
        loop[uv].uv = c
    return mesh_object(name, bm, col, [mat], smooth=False)


def build_sky_and_clouds(col, M):
    w = bpy.data.worlds.new("Sky")
    bpy.context.scene.world = w
    try:
        w.use_nodes = True
    except (AttributeError, TypeError):
        pass
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    link(nt, tc.outputs["Generated"], sep.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    link(nt, sep.outputs[2], mr.inputs["Value"])
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.0, 0.12   # zenith colour by the frame top
    cr = nt.nodes.new("ShaderNodeValToRGB")
    link(nt, mr.outputs["Result"], cr.inputs[0])
    cr.color_ramp.elements[0].color = (*lin(hexc(P["sky_horizon"])), 1)
    cr.color_ramp.elements[1].color = (*lin(hexc(P["sky_zenith"])), 1)
    bg = nt.nodes.new("ShaderNodeBackground")
    link(nt, cr.outputs["Color"], bg.inputs["Color"])
    link(nt, bg.outputs[0], out.inputs["Surface"])

    for i, (x, z, s) in enumerate(((-48, 6.6, 1.0), (-14, 8.6, 0.8), (22, 7.2, 1.15), (58, 9.0, 0.9))):
        bm, uv = new_bm()
        wdt, hgt = 22 * s, 8.25 * s
        vs = [bm.verts.new(p) for p in ((x - wdt / 2, 100, z), (x + wdt / 2, 100, z),
                                         (x + wdt / 2, 100, z + hgt), (x - wdt / 2, 100, z + hgt))]
        f = bm.faces.new(vs)
        for loop, c in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
            loop[uv].uv = c
        mesh_object(f"Cloud{i}", bm, col, [M["cloud"]], smooth=False)


def build_camera(col):
    cd = bpy.data.cameras.new("Cam")
    cd.lens, cd.sensor_width, cd.sensor_fit = LENS, SENSOR, "HORIZONTAL"
    cd.clip_start, cd.clip_end = 0.1, 400.0
    cam = bpy.data.objects.new("Cam", cd)
    col.objects.link(cam)
    cam.location = CAM_POS
    d = (CAM_TARGET - CAM_POS).normalized()
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build(water_kind="day"):
    global LUT, LUT_CHAR
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    LUT = lut_group()
    LUT_CHAR = lut_group("LUT_Character", LUT_STOPS_CHAR)
    T = {f.__name__[4:]: f() for f in (tex_sand, tex_grass, tex_rock, tex_wood, tex_post_top, tex_post_side,
                                       tex_trunk, tex_frond, tex_hair, tex_shirt, tex_rod, tex_bobber,
                                       tex_caustic, tex_cloud)}
    M = {
        "sand": sand_material(T["sand"]),
        "grass": material("M_Grass", image=T["grass"]),
        "rock": material("M_Rock", image=T["rock"]),
        "wood": material("M_Wood", image=T["wood"]),
        "post_side": material("M_PostSide", image=T["post_side"]),
        "post_top": material("M_PostTop", image=T["post_top"]),
        "trunk": material("M_Trunk", image=T["trunk"]),
        "frond": material("M_Frond", image=T["frond"], cutout=True),
        "coconut": material("M_Coconut", color="#7A5A3A"),
        "bush": material("M_Bush", color=P["foliage_mid"]),
        "shell": material("M_Shell", color="#F4E4CF"),
        "skin": material("M_Skin", color=P["skin"], char=True),
        "hair": material("M_Hair", image=T["hair"], char=True),
        "eye": material("M_Eye", color=P["eye"], lit=False),
        "blush": material("M_Blush", color=P["blush"], lit=False),
        "shirt": material("M_Shirt", image=T["shirt"], char=True),
        "shorts": material("M_Shorts", color=P["shorts"], char=True),
        "shoe": material("M_Shoe", color=P["shoe"], char=True),
        "rod": material("M_Rod", image=T["rod"]),
        "line": material("M_Line", color=P["line"], lit=False),
        "bobber": material("M_Bobber", image=T["bobber"]),
        "foam": material("M_Foam", color=WATER[water_kind]["foam"], lit=False),
        "water": water_material(water_kind, T["caustic"]),
        "blob": blob_material("M_Blob"),
        "blob_box": blob_material("M_BlobBox", box=True, edge=0.25),
        "cloud": cloud_material(T["cloud"]),
    }
    world = collection("World")
    props = collection("Props")
    chars = collection("Characters")
    build_sky_and_clouds(world, M)
    build_terrain(world, M)
    build_water(world, M)
    deck, posts = build_pier(props, M)
    rng = np.random.default_rng(11)
    crowns = []
    for x, y, hgt, ld, lean in PALMS:
        _, _, crown = build_palm(props, M, x, y, hgt, ld, lean, rng)
        crowns.append(crown)
        # noon: the canopy shadow falls straight down under the crown
        blob(props, M["blob"], "PalmShadow", (crown.x, crown.y, terrain_h(crown.x, crown.y) + 0.03), 1.55, 1.35)
    build_bushes(world, M, rng)
    build_shells(props, M)
    parts = build_villager(chars, M)
    blob(chars, M["blob"], "VillagerShadow", (VILLAGER.x + 0.02, VILLAGER.y, DECK_Z + 0.02), 0.30, 0.26)
    rect_blob(props, M["blob_box"], "PierShadow", PIER_X0 - 0.15, PIER_X1 + 0.15, PIER_Y0 - 0.1,
              shore_y(0.5 * (PIER_X0 + PIER_X1)) + 0.4, 0.008)
    cam = build_camera(world)

    # render: EEVEE, one sample (no anti aliasing), no shadows or effects, Standard view so hex colours hold
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = RES_X, RES_Y, 100
    try:
        sc.render.filter_size = 0.0
    except Exception:
        sc.render.filter_size = 0.01
    ee = sc.eevee
    ee.taa_render_samples = 1
    for attr in ("use_shadows", "use_gtao", "use_bloom", "use_ssr", "use_raytracing", "use_volumetric_shadows"):
        if hasattr(ee, attr):
            try:
                setattr(ee, attr, False)
            except Exception:
                pass
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure, sc.view_settings.gamma = 0.0, 1.0
    sc.render.film_transparent = False
    try:
        sc.render.image_settings.media_type = "IMAGE"
    except Exception:
        pass
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    return {"cam": cam, "parts": parts, "posts": posts, "crowns": crowns}


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check(info):
    sc = bpy.context.scene
    bpy.context.view_layer.update()           # refresh matrix_world before projecting
    cam = info["cam"]
    ok = True
    ground = max(terrain_h(cam.location.x, cam.location.y), 0.0)
    print(f"camera height above ground/water: {cam.location.z - ground:.2f} m")
    ok &= cam.location.z - ground > 0.5
    subjects = {
        "villager head": VILLAGER + Vector((0, 0, 0.74)),
        "villager feet": VILLAGER,
        "rod tip": VILLAGER + Vector((1.55, 0.25, 1.62)),
        "bobber": BOBBER,
        "pier near end": Vector((0.5 * (PIER_X0 + PIER_X1), PIER_Y0, DECK_Z)),
        "pier land end": Vector((0.5 * (PIER_X0 + PIER_X1), PIER_Y1, DECK_Z)),
        "shoreline x=0": Vector((0, shore_y(0), 0)),
    }
    for i, c in enumerate(info["crowns"]):
        subjects[f"palm crown {i}"] = c
    for name, p in subjects.items():
        v = world_to_camera_view(sc, cam, p)
        px, py = v.x * RES_X, (1 - v.y) * RES_Y
        inside = 0 <= v.x <= 1 and 0 <= v.y <= 1 and v.z > 0
        print(f"  {name:16s} screen ({px:6.1f}, {py:6.1f}) px  {'in frame' if inside else 'OUT OF FRAME'}")
        if name in ("villager head", "villager feet", "rod tip", "bobber") and not inside:
            ok = False
    # character size on screen
    a = world_to_camera_view(sc, cam, VILLAGER)
    b = world_to_camera_view(sc, cam, VILLAGER + Vector((0, 0, 1.03)))
    print(f"villager height on screen: {(b.y - a.y) * RES_Y:.1f} px of {RES_Y}")
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in sc.objects if o.type == "MESH")
    for grp, names in (("villager", ("Head", "Hair", "Eyes", "Blush", "Body", "Shorts", "Shoes", "Hands", "Rod")),
                       ("one palm", ("PalmTrunk", "PalmFronds", "Coconuts"))):
        n = 0
        for o in sc.objects:
            if o.type == "MESH" and o.name.split(".")[0] in names:
                n += sum(len(p.vertices) - 2 for p in o.data.polygons)
        if grp == "one palm":
            n //= len(PALMS)
        print(f"{grp} triangles: {n}")
    print(f"scene triangles: {tris}")
    print("CHECK", "OK" if ok else "FAILED")
    return ok


def upscale(src, dst, k):
    ff = shutil.which("ffmpeg")
    if not ff:
        print("ffmpeg not found, skipping upscale")
        return
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-vf", f"scale=iw*{k}:ih*{k}:flags=neighbor", dst],
                   check=True)
    print("upscaled", dst)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--save")
    p.add_argument("--render")
    p.add_argument("--upscale", type=int, default=3)
    p.add_argument("--water", choices=sorted(WATER), default="day")
    a = p.parse_args(argv)
    info = build(a.water)
    if a.check:
        check(info)
    if a.save:
        path = os.path.abspath(a.save)
        if os.path.exists(path):
            sys.exit(f"refusing to overwrite {path}")
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=path)
        print("saved", path)
    if a.render:
        import time
        out = os.path.abspath(a.render)
        bpy.context.scene.render.filepath = out
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"rendered {out} in {time.time() - t:.1f}s")
        if a.upscale > 1:
            root, ext = os.path.splitext(out)
            upscale(out, f"{root}_x{a.upscale}{ext}", a.upscale)


if __name__ == "__main__":
    main()
