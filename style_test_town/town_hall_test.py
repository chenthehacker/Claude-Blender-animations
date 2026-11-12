"""Blueprint test scene 2: the plaza in front of a village hall, with the player and three villagers.

The design follows the series' design language (squat chunky buildings with soft bulging eaves, a central clock bay,
stone columns, plank walls, cobble plazas; round headed animal villagers with stub limbs), measured from retail frames
(S#2855 town hall exterior, S#3152 plaza, S#1436, S#1441), but the hall and the three villagers are original designs:
no game building, character, emblem or texture is copied.

Shares the style system of town_river_test.py: camera solve, rolling log bend, LUT lighting, character rim, multiply
shadow decals, painted textures with grain, haze, time of day grades.

    /root/bpyenv/bin/python town_hall_test.py --check --render renders/town_hall.png
    /root/bpyenv/bin/python town_hall_test.py --native --render renders/town_hall_native.png   (also writes _x3)

Units: P = player height = 1.0. X east, Y north, Z up; the camera looks north; the player stands at the origin.
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
sys.path.insert(0, HERE)
import town_river_test as R  # noqa: E402  (style system, camera, bend, materials, player, props)

R.terrain_h = lambda x, y: 0.0          # flat plaza: every shared prop builder reads the ground height from here

hexc, lin, smoothstep, link = R.hexc, R.lin, R.smoothstep, R.link
new_bm, mesh_object, bm_tube, bm_sphere = R.new_bm, R.mesh_object, R.bm_tube, R.bm_sphere
fbm, ramp, make_image, grain, mat = R.fbm, R.ramp, R.make_image, R.grain, R.mat
RNG = np.random.default_rng(31)

# --------------------------------------------------------------------------
# Camera in front of a building (16.16, measured on S#2855, S#1436, S#1441)
# --------------------------------------------------------------------------
# Feet row 0.90 and player share 0.33: S#2855 feet row 220 of 240, hat top 138; S#1436 and S#1441 feet 214 to 215, cap
# top 134 to 135 (by eye on 3x and 5x NEAREST crops). Object lean 18 deg: the hall's outer wall corners in S#2855
# drift 5.7 and 7.7 native px over the wall height (least squares on the strongest horizontal gradient per row); with
# pitch 35 and no lean this camera predicts 19.3 px, with a 17 deg lean 7.4 px, with 20 deg 5.8 px. Ground pitch, lens
# and world bend (R 13.28 P) are unchanged.
FEET_ROW, PLAYER_SHARE, LEAN_DEG = 0.90, 0.33, 18.0
R.OBJECT_LEAN = math.tan(math.radians(LEAN_DEG))
_VT = math.tan(math.radians(R.VFOV) / 2)


def project(p, h, d):
    """Row (from the top) and column of a world point for a camera at height h, distance d behind the player."""
    q = R.bend_point(Vector(p))
    a = math.radians(R.PITCH)
    fwd, up = Vector((0, math.cos(a), -math.sin(a))), Vector((0, math.sin(a), math.cos(a)))
    v = Vector((q.x, q.y + d, q.z - h))
    zc = v.dot(fwd)
    return 0.5 - 0.5 * (v.dot(up) / zc) / _VT, 0.5 + 0.5 * (q.x / zc) / (_VT * 5 / 3)


def solve_camera_lean():
    lo, hi = 0.5, 20.0
    for _ in range(60):
        d = 0.5 * (lo + hi)
        h = d * math.tan(math.radians(R.dep_for_row(FEET_ROW)))
        if FEET_ROW - project((0, 0, 1.0), h, d)[0] > PLAYER_SHARE:
            lo = d
        else:
            hi = d
    return h, d


R.CAM_H, R.CAM_D = solve_camera_lean()
R._NEAR = math.hypot(R.CAM_D, R.CAM_H)
R._FAR = R.tangent_distance(R.LOG_R, R.CAM_H, R.CAM_D)
R.HAZE_NEAR = R._NEAR + 0.4 * (R._FAR - R._NEAR)
R.HAZE_FAR = R._FAR

# --------------------------------------------------------------------------
# Palette (M = measured, E = design choice)
# --------------------------------------------------------------------------
HP = {
    "plaza_dirt": "#9D6223",        # M 16.10 NL plaza dirt (mortar)
    "cobble_grey": "#AAA9B9",       # M 16.10 NL cobble
    "cobble_haze": "#C6C3D6",       # M S#3152 far cobble band (near #ADB2C6, luminance x1.11, stays lavender grey)
    "plank": "#C89A55", "plank_dark": "#866936", "plank_light": "#DBB264",   # M S#2855 wall median #AB854A, p10, p90
    "stone": "#BCAEBB", "stone_dark": "#7F7383", "stone_light": "#E2D2D6",   # M S#2855 pillar #A08998 to #E6C0BD (lit)
    "roof": "#BB3F5B", "roof_dark": "#912B39", "roof_light": "#D5616C",      # M 16.7 NL roof p5, median, p95
    "trim": "#6B4A2A", "trim_light": "#8C6438",                              # E window frames (S#2855 #5A4829 family)
    "door": "#9A6234", "door_dark": "#5E3A1E",                               # E
    "glass": "#E9E4BC", "glass_dark": "#8C8A78",                             # M S#2855 window p90 #F1EEC0 (lit pane)
    "window_night": "#F4CF6A",                                               # M 16.5 NL night windows
    "clock": "#F6F0E2", "clock_rim": "#C9A24A", "clock_hand": "#3A2A20",     # E
    "metal": "#3C4A44", "lamp": "#FFF1C2", "lamp_night": "#FFE9A0",          # E (S#2855 lamp is dark green grey)
}
VILLAGERS = [
    # name, species, position, facing (deg, 0 = east, -90 = toward the camera), fur, fur light, accent, shirt a, b
    ("Cat", "cat", (1.3, -0.05), 205.0, "#F0A65C", "#FCEBCF", "#F59AA6", "#5BAE78", "#E4F3DE"),
    ("Dog", "dog", (-1.4, 0.4), -50.0, "#E0C08E", "#FAEED8", "#8C5A3A", "#D85A4E", "#F8E2D8"),
    ("Rabbit", "rabbit", (0.8, 0.72), -105.0, "#D6C6EC", "#F7F2FB", "#F3A9BC", "#EEC955", "#FFF4D0"),
]
PLAYER_FACING = -25.0              # turned toward the cat and the camera
HALL = {"front_y": 1.5, "w": 3.6, "d": 2.4, "eave": 1.32, "bay_w": 1.16, "bay_d": 0.42}


# --------------------------------------------------------------------------
# Soft geometry helpers (NL buildings: bevelled slabs, bulging eaves, 16.7)
# --------------------------------------------------------------------------
def rounded_box(bm, uv, center, size, r, nb=3, ni=2, rot=None, uvs=1.0):
    """Box with every edge rounded to radius r (a cube grid pushed onto the rounded box), smooth shaded.
    Planar UVs per face in P times uvs, v up on the sides."""
    h = [s / 2 for s in size]
    r = min(r, min(h) * 0.98)
    lim = [x - r for x in h]

    def coords(hh):
        band = list(np.linspace(-hh, -hh + r, nb + 1))
        inner = list(np.linspace(-hh + r, hh - r, ni + 1))[1:-1] if hh - r > 1e-6 else []
        return band + inner + [-c for c in reversed(band)]

    cs = [coords(x) for x in h]
    rm = rot.to_matrix() if hasattr(rot, "to_matrix") else (rot if rot is not None else Matrix.Identity(3))
    C = Vector(center)
    cache = {}

    def vert(p):
        key = tuple(round(c, 6) for c in p)
        if key not in cache:
            inner = Vector([max(-lim[i], min(lim[i], p[i])) for i in range(3)])
            d = Vector(p) - inner
            q = inner + d.normalized() * r if d.length > 1e-9 else Vector(p)
            cache[key] = (bm.verts.new(C + rm @ q), q)
        return cache[key]

    for a in range(3):
        b, c = [i for i in range(3) if i != a]
        axis = Vector([1.0 if i == a else 0.0 for i in range(3)])
        for s in (-1, 1):
            for i in range(len(cs[b]) - 1):
                for j in range(len(cs[c]) - 1):
                    quad = []
                    for ii, jj in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        p = [0.0, 0.0, 0.0]
                        p[a], p[b], p[c] = s * h[a], cs[b][ii], cs[c][jj]
                        quad.append(vert(p))
                    vs = list(dict.fromkeys(qq[0] for qq in quad))
                    if len(vs) < 3:
                        continue
                    try:
                        f = bm.faces.new(vs)
                    except ValueError:
                        continue
                    f.normal_update()
                    if f.normal.dot(rm @ (axis * s)) < 0:
                        f.normal_flip()
                    for lp in f.loops:
                        q = next(qq[1] for qq in quad if qq[0] == lp.vert)
                        if a == 2:
                            lp[uv].uv = ((q.x + h[0]) * uvs, (q.y + h[1]) * uvs)
                        else:
                            lp[uv].uv = ((q[b] + h[b]) * uvs, (q.z + h[2]) * uvs)


def arch_outline(cx, z0, w, hr, segs):
    """Rectangle w wide, hr tall, topped by a half circle of radius w/2 (x, z pairs, from bottom left)."""
    pts = [(cx - w / 2, z0), (cx + w / 2, z0)]
    for t in np.linspace(0, math.pi, segs + 1):
        pts.append((cx + w / 2 * math.cos(t), z0 + hr + w / 2 * math.sin(t)))
    return pts


def arch_slab(bm, uv, cx, y_front, depth, z0, w, hr, bevel=0.04, segs=12, uvs=1.0):
    """Arch topped slab facing -Y with a rounded front edge (flat front face kept sharp, smooth chamfer, sides)."""
    A = arch_outline(cx, z0 + bevel, w - 2 * bevel, hr - bevel, segs)
    B = arch_outline(cx, z0, w, hr, segs)
    ring_a = [bm.verts.new((x, y_front, z)) for x, z in A]
    ring_b = [bm.verts.new((x, y_front + bevel, z)) for x, z in B]
    ring_c = [bm.verts.new((x, y_front + depth, z)) for x, z in B]
    n = len(A)
    new_faces = []
    f = bm.faces.new(ring_a)
    new_faces.append(f)
    for lp in f.loops:
        lp[uv].uv = ((lp.vert.co.x - cx + w / 2) * uvs, (lp.vert.co.z - z0) * uvs)
    for e in f.edges:
        e.smooth = False
    per = 0.0
    for i in range(n):
        j = (i + 1) % n
        seg = math.dist(B[i], B[j])
        for r0, r1, v0, v1 in ((ring_a, ring_b, 0.0, bevel), (ring_b, ring_c, bevel, depth)):
            g = bm.faces.new((r0[i], r0[j], r1[j], r1[i]))
            new_faces.append(g)
            for lp, (uu, vv) in zip(g.loops, ((per, v0), (per + seg, v0), (per + seg, v1), (per, v1))):
                lp[uv].uv = (uu * uvs, vv * uvs)
        per += seg
    back = bm.faces.new(list(reversed(ring_c)))
    new_faces.append(back)
    bmesh.ops.recalc_face_normals(bm, faces=new_faces)


def torus_arc(bm, uv, center, U, V, R_, r, a0, a1, segs=24, sides=10, caps=True):
    """Tube of radius r bent along a circular arc of radius R_ in the plane (U, V) about center, from angle a0 to a1
    (a stable frame: the radial direction and the plane normal), smooth, optional round end caps."""
    C, U, V = Vector(center), Vector(U).normalized(), Vector(V).normalized()
    Nrm = U.cross(V)
    full = abs(a1 - a0 - 2 * math.pi) < 1e-6
    n = segs if full else segs + 1
    rings = []
    for i in range(n):
        t = a0 + (a1 - a0) * i / segs
        rad = U * math.cos(t) + V * math.sin(t)
        P = C + rad * R_
        rings.append([bm.verts.new(P + r * (rad * math.cos(ph) + Nrm * math.sin(ph)))
                      for ph in np.linspace(0, 2 * math.pi, sides, endpoint=False)])
    faces = []
    for i in range(segs):
        ra, rb = rings[i], rings[(i + 1) % n]
        for k in range(sides):
            f = bm.faces.new((ra[k], ra[(k + 1) % sides], rb[(k + 1) % sides], rb[k]))
            faces.append(f)
            for lp, q in zip(f.loops, ((i, k), (i, k + 1), (i + 1, k + 1), (i + 1, k))):
                lp[uv].uv = (q[0] / segs, q[1] / sides)
    if caps and not full:
        faces.append(bm.faces.new(rings[0]))
        faces.append(bm.faces.new(list(reversed(rings[-1]))))
    bmesh.ops.recalc_face_normals(bm, faces=faces)


def xform_new(bm, before, m):
    """Apply matrix m to the vertices created since the set `before`."""
    for v in bm.verts:
        if v not in before:
            v.co = m @ v.co


# --------------------------------------------------------------------------
# Painted textures: one density for the whole scene (about 64 to 80 texels per P, like the cobbles), bilinear,
# painted mottle and felt grain (section 6, 16.8). The user found the old hall and characters too clean next to the
# cobbles, so every surface gets the same painted treatment.
# --------------------------------------------------------------------------
def mottle(w, h, cycles=6, amp=0.07):
    return 1.0 + amp * 2 * (fbm(w, h, cycles, 3, RNG) - 0.5)


def tex_cobble():
    """Round cobbles (16.10: NL round cobbles, grey #AAA9B9) packed tight on a thin plaza dirt mortar (#9D6223):
    a wrapped Voronoi tiling clipped by discs, lighter toward each stone's top left, darker at its rim."""
    w = h = 160
    k = 13
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    g = np.stack(np.meshgrid(np.arange(k), np.arange(k)), -1).reshape(-1, 2)
    pts = (g + 0.5 + RNG.uniform(-0.28, 0.28, g.shape)) / k
    d1 = np.full((h, w), 9e9)
    d2 = np.full((h, w), 9e9)
    idx = np.zeros((h, w), int)
    cxs = np.zeros((h, w))
    cys = np.zeros((h, w))
    for i, (px, py) in enumerate(pts):
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                cx, cy = (px + ox) * w, (py + oy) * h
                d = np.hypot(xx - cx, yy - cy)
                closer = d < d1
                d2 = np.where(closer, d1, np.minimum(d2, d))
                idx = np.where(closer, i, idx)
                cxs = np.where(closer, cx, cxs)
                cys = np.where(closer, cy, cys)
                d1 = np.where(closer, d, d1)
    border = d2 - d1
    tone = RNG.uniform(0.88, 1.08, len(pts))
    warm = RNG.random(len(pts)) < 0.25
    base = np.where(warm[idx][..., None], hexc("#B8A68E"), hexc(HP["cobble_grey"])) * tone[idx][..., None]
    cell = w / k
    light = 1.0 + 0.16 * np.clip(1 - np.hypot(xx - cxs + 0.2 * cell, yy - cys + 0.2 * cell) / (0.55 * cell), 0, 1)
    rim = 0.72 + 0.28 * smoothstep(0.8, 5.0, border)
    stone = base * (light * rim)[..., None]
    mortar = hexc(HP["plaza_dirt"]) * (0.7 + 0.2 * fbm(w, h, 8, 2, RNG))[..., None]
    inside = np.minimum(border, 2.0 * (0.62 * cell - d1))
    t = smoothstep(0.5, 1.5, inside)[..., None]
    col = mortar * (1 - t) + stone * t
    return make_image("T_Cobble", grain(np.clip(col, 0, 1), 0.04))


def tex_planks():
    """Horizontal planks, 64 texels per P: 7 planks per P, each lighter at its top edge and shaded at its lower lip
    (painted, like the roof rows), dark grooves, knots, per plank tone and mottle."""
    w = h = 64
    yy, xx = np.mgrid[0:h, 0:w]
    v = (h - 1 - yy) / h * 7
    k = np.floor(v).astype(int)
    fv = v - k
    tone = RNG.uniform(0.9, 1.08, 8)[k % 8]
    t = 0.2 + 0.65 * fv
    col = ramp(t, [HP["plank_dark"], HP["plank"], HP["plank_light"]]) * tone[..., None]
    col[fv < 0.1] = hexc(HP["plank_dark"]) * 0.8
    for _ in range(5):
        kx, ky = RNG.integers(0, w), RNG.integers(0, h)
        d = np.hypot((xx - kx) / 2.2, (yy - ky) / 1.3)
        col = col * (1 - 0.25 * np.clip(1 - d, 0, 1))[..., None]
    col = col * mottle(w, h, 5, 0.06)[..., None]
    return make_image("T_Planks", grain(np.clip(col, 0, 1), 0.035))


def tex_stone():
    w = h = 64
    col = ramp(0.45 + 0.35 * (fbm(w, h, 5, 3, RNG) - 0.5), [HP["stone_dark"], HP["stone"], HP["stone_light"]])
    speck = RNG.random((h, w)) < 0.04
    col[speck] = col[speck] * 0.85
    return make_image("T_HStone", grain(np.clip(col * mottle(w, h, 8, 0.05)[..., None], 0, 1), 0.035))


def tex_roof():
    """Shingle rows along the slope (v), 5 rows per P: lighter top edge, darker lip (16.7), staggered tabs with
    rounded lower ends."""
    w, h = 64, 64
    yy, xx = np.mgrid[0:h, 0:w]
    rows = 5
    v = (h - 1 - yy) / h * rows
    r_idx = np.floor(v).astype(int)
    fv = v - r_idx
    tabs = 4
    u = xx / w * tabs + 0.5 * (r_idx % 2)
    fu = u % 1.0
    tvar = RNG.uniform(0.9, 1.08, 64)[(np.floor(u).astype(int) * 7 + r_idx * 3) % 64]
    t = 0.15 + 0.75 * fv
    col = ramp(t, [HP["roof_dark"], HP["roof"], HP["roof_light"]]) * tvar[..., None]
    round_end = fv < 0.18 * (1 - np.cos(2 * np.pi * fu)) / 2 + 0.06          # scalloped lower lip
    col[round_end] = hexc(HP["roof_dark"]) * 0.85
    col = col * mottle(w, h, 6, 0.05)[..., None]
    return make_image("T_Roof", grain(np.clip(col, 0, 1), 0.03))


def tex_wood(name, base, dark, light, w=32, h=64):
    yy, xx = np.mgrid[0:h, 0:w]
    streak = 0.5 + 0.5 * np.sin(2 * np.pi * (xx / w * 3 + 0.4 * fbm(w, h, 2, 2, RNG)))
    col = ramp(0.3 + 0.4 * streak, [dark, base, light]) * mottle(w, h, 5, 0.05)[..., None]
    return make_image(name, grain(np.clip(col, 0, 1), 0.035))


def tex_door():
    """Plank door with a round window (CF style arched doors with a glowing round window, 16.7 [R1])."""
    w, h = 32, 64
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx + 0.5) / w, 1 - (yy + 0.5) / h
    planks = 0.5 + 0.5 * np.cos(2 * np.pi * u * 4)
    col = ramp(0.25 + 0.5 * planks, [HP["door_dark"], HP["door"], HP["door"]]) * mottle(w, h, 4, 0.06)[..., None]
    win = np.hypot((u - 0.5) * 1.0, (v - 0.68) * 2.0) < 0.2
    rim = np.hypot((u - 0.5) * 1.0, (v - 0.68) * 2.0) < 0.25
    col[rim] = hexc(HP["trim"])
    col[win] = ramp(np.clip(v[win] * 1.4 - 0.5, 0, 1), [HP["glass_dark"], HP["glass"], "#FFFBE0"])
    knob = np.hypot((u - 0.82) * 1.0, (v - 0.4) * 2.0) < 0.05
    col[knob] = hexc(HP["clock_rim"])
    return make_image("T_Door", grain(np.clip(col, 0, 1), 0.03))


def tex_window(night=False):
    """Six panes (2 x 3) with a warm lit gradient (S#2855 panes run from a dark grey bottom to a pale cream top)."""
    w, h = 32, 48
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx + 0.5) / w, 1 - (yy + 0.5) / h
    if night:
        col = ramp(v, ["#E3B256", HP["window_night"], "#FFF0B8"])
    else:
        col = ramp(np.clip(v * 1.3 - 0.15, 0, 1), [HP["glass_dark"], HP["glass"], "#FFFBEA"])
        glint = np.abs((u - v * 0.6) - 0.2) < 0.05
        col[glint] = col[glint] * 0.6 + 0.4
    mull = (np.abs(u - 0.5) < 0.045) | (np.abs(v - 1 / 3) < 0.03) | (np.abs(v - 2 / 3) < 0.03)
    col[mull] = hexc(HP["trim"])
    return make_image("T_Window_night" if night else "T_Window", grain(np.clip(col, 0, 1), 0.02))


def tex_clock():
    w = h = 48
    yy, xx = (np.mgrid[0:h, 0:w] - (w - 1) / 2) / ((w - 1) / 2)
    r = np.hypot(xx, yy)
    col = ramp(np.clip(1 - r, 0, 1), ["#E6DCC8", HP["clock"], "#FFFFFF"])
    a = np.arctan2(yy, xx)
    ticks = (r > 0.64) & (r < 0.8) & (np.abs(((a / (np.pi / 6)) + 0.5) % 1 - 0.5) < 0.1)
    col[ticks] = hexc(HP["clock_hand"])
    hand1 = (np.abs(xx + 0.02) < 0.07) & (yy < 0.05) & (yy > -0.55)          # hour hand toward 12 (rows top down)
    hand2 = (np.abs(yy - xx * 0.55) < 0.07) & (xx > -0.05) & (xx < 0.62)      # minute hand toward about 4
    col[hand1 | hand2] = hexc(HP["clock_hand"])
    col[r < 0.1] = hexc(HP["clock_rim"])
    return make_image("T_Clock", grain(np.clip(col, 0, 1), 0.015))


def tex_fur(name, base, light):
    """Fur: soft mottle at about 64 texels per P plus felt grain (5.3), slightly darker toward the bottom rows."""
    w = h = 64
    n = fbm(w, h, 6, 3, RNG)
    col = ramp(0.3 + 0.22 * n, [base, base, light]) * mottle(w, h, 10, 0.04)[..., None]
    yy = np.mgrid[0:h, 0:w][0]
    col = col * (0.92 + 0.08 * (1 - yy / h))[..., None]
    return make_image(name, grain(np.clip(col, 0, 1), 0.04))


def tex_villager_face(fur, light, accent, species):
    """Head texture (u 0.5 = facing direction, v bottom to top) at about 110 texels per P (CF faces read 40 to 75
    texels across, section 6). Eyes: black discs with a white glint (5.3), centre 0.33 of the core head below its top
    (latitude about 20 deg), height 0.14 of the head, spacing 0.38 of its width; blush under the outer corners;
    mouth a local dark fill, never black."""
    w, h = 192, 96
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx + 0.5) / w, 1 - (yy + 0.5) / h
    n = fbm(w, h, 12, 3, RNG)
    col = ramp(0.3 + 0.22 * n, [fur, fur, light]) * mottle(w, h, 16, 0.035)[..., None]
    du = np.abs(u - 0.5)
    lower = smoothstep(0.5, 0.36, v) * smoothstep(0.16, 0.06, du)               # light muzzle and chin patch
    col = col * (1 - lower[..., None]) + hexc(light) * lower[..., None]
    if species == "cat":                                                     # forehead stripes (painted marking)
        for k in (-1, 0, 1):
            s = (np.abs(u - 0.5 - k * 0.022) < 0.007) & (v > 0.74) & (v < 0.86 - 0.04 * abs(k))
            col[s] = hexc(fur) * 0.72
    if species == "dog":                                                     # one eye patch
        patch = np.hypot((u - 0.44) / 0.05, (v - 0.62) / 0.11) < 1
        col[patch] = col[patch] * 0.8
    for sgn in (-1, 1):
        cu, cv = 0.5 + sgn * 0.062, 0.615
        e = ((u - cu) / 0.019) ** 2 + ((v - cv) / 0.05) ** 2
        col[e < 1] = hexc("#2A2230")
        gl = ((u - cu - 0.004) / 0.007) ** 2 + ((v - cv - 0.022) / 0.015) ** 2
        col[gl < 1] = hexc("#FFFFFF")
        b = np.hypot((u - (0.5 + sgn * 0.1)) / 0.03, (v - 0.5) / 0.035)
        kk = smoothstep(1.0, 0.2, b)[..., None] * 0.45
        col = col * (1 - kk) + hexc("#F08F7E") * kk
    mouth = (np.abs(v - 0.425 - 0.018 * (np.abs(u - 0.5) / 0.03) ** 2) < 0.007) & (np.abs(u - 0.5) < 0.03)
    col[mouth] = hexc(fur) * 0.45
    return make_image(f"T_Face_{species}", grain(np.clip(col, 0, 1), 0.025))


def tex_vshirt(name, a, b, pattern):
    """Shirt strip 32 x 128 (section 6: CF clothing 32 x 128): top half torso, bottom half sleeves, sleeve motifs
    larger (1.25 to 1.6x, section 6)."""
    w, h = 32, 128
    yy, xx = np.mgrid[0:h, 0:w]
    col = np.ones((h, w, 3)) * hexc(a)
    sl = yy >= 64
    if pattern == "stripes":
        col[((yy // 5) % 2 == 0) & ~sl] = hexc(b)
        col[((yy // 7) % 2 == 0) & sl] = hexc(b)
    elif pattern == "dots":
        col[(((xx % 8) - 3.5) ** 2 + ((yy % 8) - 3.5) ** 2 < 4) & ~sl] = hexc(b)
        col[(((xx % 12) - 5.5) ** 2 + ((yy % 10) - 4.5) ** 2 < 6) & sl] = hexc(b)
    else:                                                                       # gingham
        c1 = ((xx // 4) % 2 == 0)
        c2 = ((yy // 4) % 2 == 0)
        col[c1 ^ c2] = hexc(a) * 0.55 + hexc(b) * 0.45
        col[c1 & c2] = hexc(b)
    col[(yy % 64) < 3] = hexc(a) * 0.82                                         # collar and cuff band
    col = col * mottle(w, h, 6, 0.06)[..., None]
    return make_image(name, grain(np.clip(col, 0, 1), 0.04))


def paint_pass(img, scale=1.0, amp=0.06, grain_amt=0.035):
    """Give a shared texture the scene's painted treatment: resample to `scale`, add mottle and felt grain."""
    if scale != 1.0:
        img.scale(max(8, int(img.size[0] * scale)), max(8, int(img.size[1] * scale)))
    w, h = img.size
    px = np.array(img.pixels[:], dtype=float).reshape(h, w, 4)
    rgb = px[..., :3] * mottle(w, h, max(3, w // 12), amp)[..., None]
    px[..., :3] = grain(np.clip(rgb, 0, 1), grain_amt)
    img.pixels.foreach_set(px.astype(np.float32).ravel())
    img.pack()


# --------------------------------------------------------------------------
# Ground: cobble plaza in front of the hall, grass around it (plants only on grass)
# --------------------------------------------------------------------------
PLAZA = {"hx": 2.0, "y0": -2.2, "soft": 0.35}


def plaza_weight(x, y):
    y1 = HALL["front_y"] + HALL["d"] + 0.4
    hx = PLAZA["hx"] if y < HALL["front_y"] - 0.2 else HALL["w"] / 2 + 0.35
    dx = max(abs(x) - hx, 0.0)
    dy = max(PLAZA["y0"] - y, y - y1, 0.0)
    return float(smoothstep(PLAZA["soft"], 0.0, math.hypot(dx, dy)))


def build_ground(col, M):
    xs = np.concatenate([np.arange(-30, -6, 1.0), np.arange(-6, 6, 0.1), np.arange(6, 30.01, 1.0)])
    ys = np.concatenate([np.arange(-6, 10, 0.1), np.arange(10, 40.01, 0.6)])
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    uvw = bm.loops.layers.uv.new("UVWall")
    wl = bm.verts.layers.float.new("wall")
    ud = bm.verts.layers.float.new("under")
    grid = []
    for y in ys:
        row = []
        for x in xs:
            v = bm.verts.new((x, y, 0.0))
            v[wl] = plaza_weight(x, y)
            v[ud] = 0.0
            row.append(v)
        grid.append(row)
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            f = bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
            for lp in f.loops:
                c = lp.vert.co
                lp[uv].uv = (c.x * R.GRASS_UV, c.y * R.GRASS_UV)
                lp[uvw].uv = (c.x * 0.5, c.y * 0.5)            # cobble tile 2 P: 13 stones, about 0.15 P each
    return mesh_object("Terrain", bm, col, [M["terrain"]])


# --------------------------------------------------------------------------
# The village hall (original design in the series' language: squat, chunky, soft)
# --------------------------------------------------------------------------
def column(bm, uv, x, y, z0, z1, r):
    """Lathe column, 12 sides: a cushion base, a slightly swelling shaft, a rolled capital."""
    L = z1 - z0
    zs = [z0, z0 + 0.03, z0 + 0.07, z0 + 0.11, z0 + 0.5 * L, z1 - 0.11, z1 - 0.07, z1 - 0.03, z1]
    rs = [r * 1.38, r * 1.45, r * 1.3, r * 1.02, r * 1.06, r * 1.0, r * 1.3, r * 1.42, r * 1.32]
    bm_tube(bm, uv, [(x, y, z) for z in zs], rs, 12, cap_end=True, cap_start=True)


def build_hall(col, M):
    fy, W, D, E = HALL["front_y"], HALL["w"], HALL["d"], HALL["eave"]
    bw, bd = HALL["bay_w"], HALL["bay_d"]
    cy = fy + D / 2
    bf = fy - bd                                                  # bay front face
    foot = []
    # plinth and plank walls, soft edged
    bm, uv = new_bm()
    rounded_box(bm, uv, (0, cy, 0.09), (W + 0.16, D + 0.16, 0.18), 0.07)
    rounded_box(bm, uv, (0, fy - bd / 2 + 0.02, 0.09), (bw + 0.16, bd + 0.2, 0.18), 0.07)
    foot.append(mesh_object("HallPlinth", bm, col, [M["h_stone"]]))
    bm, uv = new_bm()
    rounded_box(bm, uv, (0, cy, 0.18 + (E - 0.18) / 2), (W, D, E - 0.18), 0.06, ni=4)
    mesh_object("HallWalls", bm, col, [M["h_planks"]])
    # corner posts (lathe, stone) at the four corners
    bm, uv = new_bm()
    for sx in (-1, 1):
        for yy in (fy, fy + D):
            column(bm, uv, sx * W / 2, yy, 0.16, E - 0.02, 0.085)
    mesh_object("CornerPosts", bm, col, [M["h_stone"]])
    # main gable roof along X: two thick rounded slabs (bulging eaves) and a ridge roll
    ov, rise = 0.3, 0.44
    half = D / 2 + ov
    th = math.atan2(rise, half)
    slope = math.hypot(half, rise)
    t = 0.16
    bm, uv = new_bm()
    for sgn in (-1, 1):                                                      # -1 front slope, +1 back slope
        rot = Matrix.Rotation(-sgn * th, 3, "X")
        mid_y = cy + sgn * half / 2
        mid_z = E - 0.06 + rise / 2
        nrm = rot @ Vector((0, 0, 1))
        c = Vector((0, mid_y, mid_z)) + nrm * (t / 2)
        if sgn > 0:
            rot = rot @ Matrix.Rotation(math.pi, 3, "Z")                     # shingle rows run down both slopes
        rounded_box(bm, uv, c, (W + 2 * ov, slope + 0.06, t), 0.075, ni=6, rot=rot, uvs=1.0)
    mesh_object("HallRoof", bm, col, [M["h_roof"]])
    ridge_z = E - 0.06 + rise + t * 0.85
    bm, uv = new_bm()
    bm_tube(bm, uv, [(-W / 2 - ov - 0.02, cy, ridge_z), (W / 2 + ov + 0.02, cy, ridge_z)], [0.09, 0.09], 12,
            cap_end=True, cap_start=True)
    mesh_object("RidgeRoll", bm, col, [M["h_roof_dark"]])
    bm, uv = new_bm()                                                        # gable end triangles (planks)
    for xe in (-W / 2 + 0.01, W / 2 - 0.01):
        vs = [bm.verts.new(p) for p in ((xe, fy, E - 0.05), (xe, fy + D, E - 0.05), (xe, cy, E - 0.05 + rise))]
        f = bm.faces.new(vs if xe > 0 else list(reversed(vs)))
        for lp in f.loops:
            lp[uv].uv = (lp.vert.co.y, lp.vert.co.z)
    mesh_object("HallGables", bm, col, [M["h_planks"]], smooth=False)
    # entrance bay with an arched clock gable
    bay_top = E + 0.02
    bm, uv = new_bm()
    rounded_box(bm, uv, (0, fy - bd / 2 + 0.05, 0.18 + (bay_top - 0.18) / 2), (bw, bd + 0.1, bay_top - 0.18), 0.06, ni=4)
    foot.append(mesh_object("HallBay", bm, col, [M["h_stone"]]))
    bm, uv = new_bm()
    arch_slab(bm, uv, 0.0, bf - 0.01, 0.3, bay_top - 0.02, bw, 0.12, bevel=0.05, segs=14, uvs=1.0)
    mesh_object("ClockGable", bm, col, [M["h_stone"]])
    bm, uv = new_bm()
    torus_arc(bm, uv, (0, bf - 0.03, bay_top + 0.1), (1, 0, 0), (0, 0, 1), bw / 2 + 0.02, 0.065, -0.2, math.pi + 0.2,
              segs=28, sides=10)
    mesh_object("GableRoll", bm, col, [M["h_roof"]])
    gable_top = bay_top + 0.1 + bw / 2 + 0.085
    cc = Vector((0, bf - 0.035, bay_top + 0.16))
    rr = 0.24
    bm, uv = new_bm()
    torus_arc(bm, uv, cc, (1, 0, 0), (0, 0, 1), rr, 0.04, 0.0, 2 * math.pi, segs=28, sides=8)
    mesh_object("ClockRim", bm, col, [M["h_clockrim"]])
    bm, uv = new_bm()
    rv = [bm.verts.new(cc + Vector((math.cos(a) * rr, -0.012, math.sin(a) * rr)))
          for a in np.linspace(0, 2 * math.pi, 25)[:-1]]
    ctr = bm.verts.new(cc + Vector((0, -0.012, 0)))
    for i in range(24):
        f = bm.faces.new((ctr, rv[i], rv[(i + 1) % 24]))
        f.normal_update()
        if f.normal.y > 0:
            f.normal_flip()
        for lp in f.loops:
            o = lp.vert.co - cc
            lp[uv].uv = (0.5 + o.x / (2 * rr), 0.5 + o.z / (2 * rr))
    mesh_object("ClockFace", bm, col, [M["h_clock"]], smooth=False)
    # columns flanking the door, door frame, arched door, steps
    bm, uv = new_bm()
    for sx in (-1, 1):
        column(bm, uv, sx * (bw / 2 - 0.13), bf - 0.1, 0.18, bay_top - 0.03, 0.075)
    mesh_object("DoorColumns", bm, col, [M["h_stone_light"]])
    bm, uv = new_bm()
    arch_slab(bm, uv, 0.0, bf - 0.03, 0.05, 0.18, 0.74, 0.62, bevel=0.035, segs=12, uvs=1.0)
    mesh_object("DoorFrame", bm, col, [M["h_trim"]])
    bm, uv = new_bm()
    A = arch_outline(0.0, 0.2, 0.6, 0.6, 12)
    vs = [bm.verts.new((x, bf - 0.04, z)) for x, z in A]
    f = bm.faces.new(vs)
    f.normal_update()
    if f.normal.y > 0:
        f.normal_flip()
    for lp in f.loops:
        lp[uv].uv = ((lp.vert.co.x + 0.3) / 0.6, (lp.vert.co.z - 0.2) / 0.9)
    mesh_object("Door", bm, col, [M["h_door"]], smooth=False)
    bm, uv = new_bm()
    rounded_box(bm, uv, (0, bf - 0.3, 0.06), (1.25, 0.5, 0.12), 0.05)
    rounded_box(bm, uv, (0, bf - 0.12, 0.16), (1.0, 0.3, 0.1), 0.045)
    foot.append(mesh_object("Steps", bm, col, [M["h_stone"]]))
    # windows: chunky wooden frames, six panes, flower boxes with blooms
    for sx in (-1, 1):
        wx = sx * (bw / 2 + (W / 2 - bw / 2) / 2 + 0.02)
        bm, uv = new_bm()
        rounded_box(bm, uv, (wx, fy - 0.02, 0.84), (0.72, 0.1, 0.72), 0.04)
        rounded_box(bm, uv, (wx, fy - 0.08, 1.24), (0.84, 0.16, 0.09), 0.04)   # lintel
        mesh_object("WinFrame", bm, col, [M["h_trim"]])
        bm, uv = new_bm()
        vs = [bm.verts.new(p) for p in ((wx - 0.28, fy - 0.075, 0.54), (wx + 0.28, fy - 0.075, 0.54),
                                        (wx + 0.28, fy - 0.075, 1.14), (wx - 0.28, fy - 0.075, 1.14))]
        f = bm.faces.new(vs)
        for lp, q in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
            lp[uv].uv = q
        mesh_object("WinGlass", bm, col, [M["h_glass"]], smooth=False)
        bm, uv = new_bm()
        rounded_box(bm, uv, (wx, fy - 0.17, 0.44), (0.78, 0.24, 0.17), 0.05)
        mesh_object("FlowerBox", bm, col, [M["h_box"]])
        bm, uv = new_bm()
        for j in range(6):
            bm_sphere(bm, uv, (wx - 0.3 + 0.12 * j, fy - 0.18 + 0.03 * (j % 2), 0.555), 0.058, segs=8, rings=5)
        mesh_object("BoxBlooms", bm, col, [M["h_bloom_a"] if sx < 0 else M["h_bloom_b"]])
        bm, uv = new_bm()
        for j in range(5):
            bm_sphere(bm, uv, (wx - 0.24 + 0.12 * j, fy - 0.12, 0.53), 0.06, scale=(1.2, 0.8, 0.6), segs=8, rings=4)
        mesh_object("BoxLeaves", bm, col, [M["foliage_core"]])
    # contact shadow around the base (sections 3 and 16.6)
    R.profile_shadow(col, "HallShadow", [(o, None) for o in foot], 0, cy - 0.3, 0.012, max(W, D) / 2 + 0.6, "contact")
    return gable_top


def build_lamp(col, M, x, y):
    """Chunky street lamp: rounded foot, lathe post, a six sided lantern with glowing panes and a dark cap."""
    bm, uv = new_bm()
    bm_tube(bm, uv, [(x, y, 0.0), (x, y, 0.05), (x, y, 0.12), (x, y, 0.2), (x, y, 0.95), (x, y, 1.0)],
            [0.12, 0.12, 0.08, 0.045, 0.04, 0.06], 8, cap_end=True, cap_start=True)
    bm_tube(bm, uv, [(x, y, 1.27), (x, y, 1.33), (x, y, 1.42), (x, y, 1.46)], [0.17, 0.15, 0.05, 0.03], 6,
            cap_end=True, cap_start=True, twist=math.pi / 6)
    bm_tube(bm, uv, [(x, y, 0.98), (x, y, 1.02)], [0.12, 0.12], 6, cap_end=True, cap_start=True, twist=math.pi / 6)
    mesh_object("LampMetal", bm, col, [M["h_metal"]])
    bm, uv = new_bm()
    bm_tube(bm, uv, [(x, y, 1.02), (x, y, 1.27)], [0.1, 0.13], 6, cap_end=True, cap_start=True, twist=math.pi / 6)
    mesh_object("LampGlass", bm, col, [M["h_lamp"]], smooth=False)
    R.blob(col, M["contact"], "LampShadow", (x, y, 0.008), 0.17, 0.15)


# --------------------------------------------------------------------------
# Villagers: original animals with NL standard animal proportions (sections 5.1, 5.2)
# --------------------------------------------------------------------------
def build_villager(col, M, name, species, pos, facing):
    """H = 1.0. Head with ears 0.50 of H, core head 0.40 (chin 0.50, top 0.90), head width over core head height
    1.44, leg gap about 0.11, stub arms reaching the hip, mitten hands; 12 to 16 segments per turn (5.2)."""
    yaw = math.radians(facing)
    Rz = Matrix.Rotation(yaw, 4, "Z")
    R3 = Rz.to_3x3()
    base = Vector((pos[0], pos[1], 0.0))

    def W(f, s, z):                                                # local: f forward, s left, z up
        return base + R3 @ Vector((f, s, z))

    fur, light, accent = M[f"v_{name}_fur"], M[f"v_{name}_light"], M[f"v_{name}_accent"]
    # head: a softly squared ellipsoid (superellipse exponent 2.4), UVs by longitude and latitude
    head_c = W(0.0, 0.0, 0.70)
    HX, HY, HZ = 0.25, 0.288, 0.20
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    res = bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=12, radius=1.0)
    nexp = 2.4
    for v in res["verts"]:
        d = v.co.normalized()
        k = (abs(d.x) ** nexp + abs(d.y) ** nexp + abs(d.z) ** nexp) ** (-1 / nexp)
        p = d * k
        if p.z > 0:
            p.z *= 0.92                                          # a little flatter on top
        v.co = head_c + R3 @ Vector((p.x * HX, p.y * HY, p.z * HZ))
    Ri = R3.inverted()
    for f in bm.faces:
        for lp in f.loops:
            dd = Ri @ (lp.vert.co - head_c)
            dd = Vector((dd.x / HX, dd.y / HY, dd.z / HZ)).normalized()
            lp[uv].uv = (0.5 + math.atan2(dd.y, dd.x) / (2 * math.pi), 0.5 + math.asin(max(-1, min(1, dd.z))) / math.pi)
        us = [lp[uv].uv.x for lp in f.loops]
        if max(us) - min(us) > 0.5:
            for lp in f.loops:
                if lp[uv].uv.x < 0.5:
                    lp[uv].uv.x += 1.0
    mesh_object(f"{name}Head", bm, col, [M[f"v_{name}_face"]])

    def surf(az, la, out=0.0):
        a, l = math.radians(az), math.radians(la)
        return head_c + R3 @ Vector(((HX + out) * math.cos(l) * math.cos(a), (HY + out) * math.cos(l) * math.sin(a),
                                     (HZ + out) * math.sin(l)))

    q = Rz.to_quaternion()
    # muzzle and nose
    bm, uv = new_bm()
    if species == "dog":
        bm_sphere(bm, uv, surf(0, -14, -0.02), 0.11, scale=(0.9, 1.0, 0.7), segs=14, rings=8, rot=q)
    else:
        for sgn in (-1, 1):                                       # two cheek puffs
            bm_sphere(bm, uv, surf(sgn * 13, -22, -0.035), 0.065, scale=(0.75, 1.0, 0.8), segs=12, rings=7, rot=q)
    mesh_object(f"{name}Muzzle", bm, col, [light])
    bm, uv = new_bm()
    if species == "dog":
        bm_sphere(bm, uv, surf(0, -8, 0.075), 0.042, scale=(0.8, 1.1, 0.75), segs=12, rings=7, rot=q)
        nose_m = M["v_nose_dark"]
    else:
        bm_sphere(bm, uv, surf(0, -14, 0.012), 0.024, scale=(0.7, 1.2, 0.8), segs=10, rings=6, rot=q)
        nose_m = accent
    mesh_object(f"{name}Nose", bm, col, [nose_m])
    # ears
    bm, uv = new_bm()
    bm2, uv2 = new_bm()
    for sgn in (-1, 1):
        if species == "cat":                                       # rounded cones, tilted out, inner ear painted
            b0 = surf(sgn * 42, 50, -0.03)
            for target, tuv, rad, hh, off in ((bm, uv, 0.095, 0.17, 0.0), (bm2, uv2, 0.055, 0.12, 0.03)):
                before = set(target.verts)
                bm_tube(target, tuv, [(0, 0, 0), (0, 0, hh * 0.6), (0, 0, hh)], [rad, rad * 0.55, rad * 0.12], 10,
                        cap_end=True, cap_start=True)
                m = (Matrix.Translation(b0) @ Rz @ Matrix.Rotation(math.radians(sgn * -18), 4, "X")
                     @ Matrix.Translation((off, 0, 0)) @ Matrix.Diagonal((0.55, 1.0, 1.0, 1.0)))
                xform_new(target, before, m)
        elif species == "dog":                                     # floppy ears hanging at the sides
            c = surf(sgn * 78, 18, 0.02) + Vector((0, 0, -0.07))
            rot = Vector((0, 0, 1)).rotation_difference(R3 @ Vector((0.05, sgn * 0.3, -1)).normalized())
            bm_sphere(bm, uv, c, 0.13, scale=(0.62, 0.32, 1.0), segs=12, rings=7, rot=rot)
        else:                                                      # rabbit: long ears, inner ear painted
            b0 = surf(sgn * 26, 62, -0.04)
            rot = Vector((0, 0, 1)).rotation_difference(R3 @ Vector((-0.08, sgn * 0.22, 1)).normalized())
            cc = b0 + rot @ Vector((0, 0, 0.19))
            bm_sphere(bm, uv, cc, 0.21, scale=(0.36, 0.26, 1.0), segs=12, rings=8, rot=rot)
            bm_sphere(bm2, uv2, cc + R3 @ Vector((0.045, 0, 0)), 0.165, scale=(0.24, 0.15, 0.82), segs=10, rings=6,
                      rot=rot)
    mesh_object(f"{name}Ears", bm, col, [accent if species == "dog" else fur])
    if len(bm2.verts):
        mesh_object(f"{name}EarsInner", bm2, col, [M["v_ear_inner"]])
    else:
        bm2.free()
    # torso (shirt): a soft bell shape; sleeves (shirt) then forearms (fur) to mitten hands at hip height
    bm, uv = new_bm()
    bm_tube(bm, uv, [W(0, 0, 0.11), W(0, 0, 0.14), W(0, 0, 0.26), W(0, 0, 0.42), W(0, 0, 0.53), W(0, 0, 0.56)],
            [0.13, 0.165, 0.17, 0.145, 0.1, 0.05], 14, v0=0.5, v1=1.0, cap_end=True, cap_start=True)
    for sgn in (-1, 1):
        bm_tube(bm, uv, [W(0.0, sgn * 0.12, 0.47), W(0.01, sgn * 0.175, 0.4)], [0.06, 0.056], 10, v0=0.0, v1=0.5,
                cap_start=True, cap_end=True)
    mesh_object(f"{name}Body", bm, col, [M[f"v_{name}_shirt"]])
    bm, uv = new_bm()
    for sgn in (-1, 1):
        bm_tube(bm, uv, [W(0.01, sgn * 0.17, 0.41), W(0.03, sgn * 0.205, 0.3)], [0.048, 0.044], 10, cap_end=True)
        bm_sphere(bm, uv, W(0.035, sgn * 0.21, 0.275), 0.055, scale=(1.0, 0.9, 1.05), segs=12, rings=7)
        bm_tube(bm, uv, [W(0, sgn * 0.075, 0.14), W(0, sgn * 0.075, 0.04)], [0.06, 0.058], 10)
        bm_sphere(bm, uv, W(0.03, sgn * 0.078, 0.035), 0.068, scale=(1.3, 0.95, 0.55), segs=12, rings=7)
    if species == "cat":
        bm_tube(bm, uv, [W(-0.16, 0, 0.18), W(-0.27, 0.04, 0.24), W(-0.31, 0.07, 0.36), W(-0.28, 0.08, 0.45)],
                [0.036, 0.033, 0.03, 0.026], 8, cap_end=True)
    elif species == "dog":
        bm_tube(bm, uv, [W(-0.16, 0, 0.2), W(-0.24, 0, 0.27), W(-0.26, 0, 0.33)], [0.045, 0.035, 0.02], 8, cap_end=True)
    else:
        bm_sphere(bm, uv, W(-0.18, 0, 0.17), 0.075, segs=12, rings=7)
    mesh_object(f"{name}Limbs", bm, col, [fur])
    dep = math.atan2(R.CAM_H, R.CAM_D)
    rx = R.BLOB_EXTENT / 2 * 1.12
    R.blob(col, M["blob"], f"{name}Shadow", (pos[0], pos[1], 0.006), rx, rx / (R.BLOB_ASPECT_ON_SCREEN * math.sin(dep)))
    return head_c


# --------------------------------------------------------------------------
# Haze on every object (user direction, 16.16): the same camera distance ramp as the ground. Retail evidence:
# low weeds near the horizon brighten like the ground (S#13257: #44964D near, #9DCE6D far band); a tree at the edge
# of a plaza shows none at its short distance (S#3152). The cobbles haze toward the measured far cobble #C6C3D6
# instead of the grass lime.
# --------------------------------------------------------------------------
def haze_node(nt, color_socket, haze_hex):
    cd = nt.nodes.new("ShaderNodeCameraData")
    hz = nt.nodes.new("ShaderNodeMapRange")
    hz.interpolation_type = "SMOOTHSTEP"
    link(nt, cd.outputs["View Distance"], hz.inputs["Value"])
    hz.inputs["From Min"].default_value, hz.inputs["From Max"].default_value = R.HAZE_NEAR, R.HAZE_FAR
    hz.inputs["To Min"].default_value, hz.inputs["To Max"].default_value = 0.0, R.HAZE_MAX
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    link(nt, hz.outputs["Result"], mx.inputs[0])
    link(nt, color_socket, mx.inputs[6])
    mx.inputs[7].default_value = (*lin(R.gcol(haze_hex)), 1)
    return mx.outputs[2]


def apply_object_haze(M):
    skip = {M[k].name for k in ("blob", "contact", "canopy", "cloud", "terrain", "water", "foam", "fish", "line")
            if k in M}
    n = 0
    for m in bpy.data.materials:
        if not m.node_tree or m.name in skip:
            continue
        nt = m.node_tree
        for em in [x for x in nt.nodes if x.bl_idname == "ShaderNodeEmission"]:
            if not em.inputs["Color"].is_linked:
                continue
            src = em.inputs["Color"].links[0].from_socket
            nt.links.remove(em.inputs["Color"].links[0])
            link(nt, haze_node(nt, src, R.PAL["haze"]), em.inputs["Color"])
            n += 1
    return n


def plaza_haze_colour(M):
    """Terrain: the haze colour follows the plaza weight (grass lime #8EC668, cobble #C6C3D6)."""
    nt = M["terrain"].node_tree
    target = lin(R.gcol(R.PAL["haze"]))
    hm = next(x for x in nt.nodes if x.bl_idname == "ShaderNodeMix" and x.data_type == "RGBA"
              and not x.inputs[7].is_linked and abs(x.inputs[7].default_value[0] - target[0]) < 1e-4
              and abs(x.inputs[7].default_value[1] - target[1]) < 1e-4)
    mr = next(x for x in nt.nodes if x.bl_idname == "ShaderNodeMapRange"
              and abs(x.inputs["From Min"].default_value - 0.42) < 1e-6)
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    link(nt, mr.outputs["Result"], mx.inputs[0])
    mx.inputs[6].default_value = (*target, 1)
    mx.inputs[7].default_value = (*lin(R.gcol(HP["cobble_haze"])), 1)
    link(nt, mx.outputs[2], hm.inputs[7])


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
TREES = [(-3.1, 2.4, 2.0), (3.2, 2.9, 2.1), (-4.2, 5.6, 2.1), (4.4, 6.2, 2.2), (-1.4, 7.2, 2.2), (1.9, 8.0, 2.1)]
FLOWER_BEDS = [(-2.75, 0.55), (2.7, 0.75), (-2.85, -0.75), (2.75, -0.6)]
BUSHES = [(-2.6, 1.6, 0.38), (2.65, 1.7, 0.4)]
LAMPS = [(-1.1, 0.95), (1.15, 0.98)]


def build(native=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rng = np.random.default_rng(5)
    T, M = R.make_assets()
    # the scene's single painted treatment on the shared character and plant textures
    for key, sc in (("face", 0.75), ("hair", 1.0), ("shirt", 1.0), ("skin", 1.0), ("shorts", 1.0), ("shoe", 1.0),
                    ("bark", 1.0), ("rock", 1.0)):
        paint_pass(T[key], scale=sc)
    T["wall"] = tex_cobble()                         # R.terrain_mat blends grass and "wall": here the cobble plaza
    M["terrain"] = R.terrain_mat(T)
    plaza_haze_colour(M)
    night = R.TIME == "night"
    M.update({
        "h_stone": mat("M_HStone", image=tex_stone()),
        "h_stone_light": mat("M_HStoneLight", image=R.tex_flat("T_HStoneL", [HP["stone"], HP["stone_light"], "#F2E6E8"],
                                                                0.04, 6)),
        "h_planks": mat("M_HPlanks", image=tex_planks()),
        "h_roof": mat("M_HRoof", image=tex_roof()),
        "h_roof_dark": mat("M_HRoofDark", image=R.tex_flat("T_RoofDark", [HP["roof_dark"], HP["roof"], HP["roof"]], 0.05)),
        "h_trim": mat("M_HTrim", image=tex_wood("T_Trim", HP["trim"], "#4A321C", HP["trim_light"])),
        "h_door": mat("M_HDoor", image=tex_door()),
        "h_glass": R.unlit("M_HGlassN", HP["window_night"]) if night else mat("M_HGlass", image=tex_window()),
        "h_box": mat("M_HBox", image=tex_wood("T_Box", "#8C5A34", "#5E3A1E", "#B07850")),
        "h_bloom_a": mat("M_HBloomA", image=R.tex_flat("T_BloomA", ["#D8405E", "#F06C8A", "#FFC0D0"], 0.05)),
        "h_bloom_b": mat("M_HBloomB", image=R.tex_flat("T_BloomB", ["#E8A21C", "#FFD24A", "#FFF0A8"], 0.05)),
        "h_clock": mat("M_HClock", image=tex_clock()),
        "h_clockrim": mat("M_HClockRim", image=R.tex_flat("T_ClockRim", ["#9C7A2C", HP["clock_rim"], "#E8CC7A"], 0.04)),
        "h_metal": mat("M_HMetal", image=R.tex_flat("T_Metal", ["#26302C", HP["metal"], "#56665E"], 0.04)),
        "h_lamp": R.unlit("M_HLampN", HP["lamp_night"]) if night else mat("M_HLamp", color=HP["lamp"]),
        "v_nose_dark": mat("M_VNoseDark", color="#4A3434", rim=1.0),
        "v_ear_inner": mat("M_VEarInner", image=R.tex_flat("T_EarInner", ["#E894A6", "#F6B4C2", "#FCD2DA"], 0.04),
                           rim=1.0),
    })
    for name, species, pos, facing, fur, light, accent, sa, sb in VILLAGERS:
        pattern = {"cat": "stripes", "dog": "gingham", "rabbit": "dots"}[species]
        M[f"v_{name}_face"] = mat(f"M_{name}Face", image=tex_villager_face(fur, light, accent, species), rim=1.0)
        M[f"v_{name}_fur"] = mat(f"M_{name}Fur", image=tex_fur(f"T_{name}Fur", fur, light), rim=1.0)
        M[f"v_{name}_light"] = mat(f"M_{name}Light", image=tex_fur(f"T_{name}Light", light, "#FFFFFF"), rim=1.0)
        M[f"v_{name}_accent"] = mat(f"M_{name}Accent", image=tex_fur(f"T_{name}Acc", accent, light), rim=1.0)
        M[f"v_{name}_shirt"] = mat(f"M_{name}Shirt", image=tex_vshirt(f"T_{name}Shirt", sa, sb, pattern), rim=1.0)
    world = bpy.data.collections.new("World")
    bpy.context.scene.collection.children.link(world)
    R.build_sky()
    build_ground(world, M)
    gable_top = build_hall(world, M)
    for x, y in LAMPS:
        build_lamp(world, M, x, y)
    for x, y, hgt in TREES:
        assert plaza_weight(x, y) == 0.0, ("tree on the plaza", x, y)
        R.build_tree(world, M, x, y, hgt, rng)
    for x, y in FLOWER_BEDS:
        assert plaza_weight(x, y) == 0.0, ("flowers on the plaza", x, y)
        R.build_flowers(world, M, x, y, rng)
    for x, y, r in BUSHES:
        assert plaza_weight(x, y) == 0.0, ("bush on the plaza", x, y)
        R.build_bush(world, M, x, y, r, rng)
    R.build_player(world, M, pos=(0.0, 0.0, 0.0), facing_deg=PLAYER_FACING, fishing=False)
    dep = math.atan2(R.CAM_H, R.CAM_D)
    rx = R.BLOB_EXTENT / 2
    R.blob(world, M["blob"], "PlayerShadow", (0.0, 0.0, 0.006), rx, rx / (R.BLOB_ASPECT_ON_SCREEN * math.sin(dep)))
    heads = {name: build_villager(world, M, name, species, pos, facing)
             for name, species, pos, facing, *_ in VILLAGERS}
    hazed = apply_object_haze(M)
    R.bend_world()
    cam = R.build_camera(world)
    R.finalize_render(native, world, M, cam)
    return {"cam": cam, "heads": heads, "gable_top": gable_top, "hazed": hazed}


def check(info):
    sc = bpy.context.scene
    bpy.context.view_layer.update()
    cam = info["cam"]

    def row_col(p):
        v = world_to_camera_view(sc, cam, R.bend_point(Vector(p)))
        return 1 - v.y, v.x

    print(f"camera: height {R.CAM_H:.2f} P, distance {R.CAM_D:.2f} P, bend R {R.LOG_R:.2f} P, object lean {LEAN_DEG} deg, "
          f"open ground horizon {R.horizon_row_for(R.LOG_R, R.CAM_H, R.CAM_D):.3f}; haze {R.HAZE_NEAR:.2f} to "
          f"{R.HAZE_FAR:.2f} P on {info['hazed']} object materials")
    for k, p in {"player feet": (0, 0, 0), "player hair top": (0, 0, 1.0)}.items():
        r, c = row_col(p)
        print(f"  {k:18s} row {r:.3f} column {c:.3f}")
    for name, species, pos, *_ in VILLAGERS:
        r0, c0 = row_col((pos[0], pos[1], 0.0))
        r1, _ = row_col((pos[0], pos[1], 1.0))
        print(f"  {name:18s} feet row {r0:.3f} column {c0:.3f}, height share {r0 - r1:.3f}")
    fy, E, bf = HALL["front_y"], HALL["eave"], HALL["front_y"] - HALL["bay_d"]
    for k, p in {"bay base": (0, bf, 0.0), "eave front": (HALL["w"] / 2, fy - 0.3, E), "clock centre": (0, bf, E + 0.18),
                 "gable top": (0, bf, info["gable_top"]), "hall left corner": (-HALL["w"] / 2, fy, 0),
                 "hall right corner": (HALL["w"] / 2, fy, 0)}.items():
        r, c = row_col(p)
        print(f"  {k:18s} row {r:.3f} column {c:.3f}{'' if 0 <= r <= 1 and 0 <= c <= 1 else '  OUT OF FRAME'}")
    print("  retail S#2855 (by eye): feet 0.917, hall base 0.854, eave 0.417, clock gable top 0.067, walls 0.15 to 0.83")
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in sc.objects if o.type == "MESH")
    vt = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in sc.objects
             if o.type == "MESH" and o.name.startswith("Cat") and not o.name.startswith("CatShadow"))
    print(f"triangles: one villager {vt}, scene {tris}")


def qa(png):
    im = bpy.data.images.load(png)
    w, h = im.size
    px = np.array(im.pixels[:]).reshape(h, w, 4)[::-1, :, :3]
    v = px.max(-1)
    s = np.where(v > 0, (v - px.min(-1)) / np.maximum(v, 1e-6), 0)
    print(f"QA envelope: neon {((s > 0.85) & (v > 0.85)).mean():.3%} (sky strip counts, 16.14), near black "
          f"{(v < 0.08).mean():.3%}, near white {((v > 0.92) & (s < 0.12)).mean():.3%}")
    sc = bpy.context.scene
    cam = sc.camera
    L = lambda a: 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    ob = bpy.data.objects["PlayerShadow"]
    tmp = os.path.join(os.path.dirname(png), "_qa_noblob.png")
    old = sc.render.filepath
    ob.hide_render = True
    sc.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    ob.hide_render = False
    sc.render.filepath = old
    im2 = bpy.data.images.load(tmp)
    px2 = np.array(im2.pixels[:]).reshape(h, w, 4)[::-1, :, :3]
    os.remove(tmp)
    ratio = L(px) / np.maximum(L(px2), 1e-4)
    q = world_to_camera_view(sc, cam, R.bend_point(Vector((0, 0, 0))))
    cxp, cyp = int(q.x * w), int((1 - q.y) * h)
    box = ratio[max(0, cyp - h // 20):cyp + h // 20, cxp - w // 25:cxp + w // 25]
    shaded = box[box < 0.97]
    if shaded.size:
        print(f"QA player blob on cobbles: median luminance ratio {np.median(shaded):.2f}, darkest decile "
              f"{np.percentile(shaded, 10):.2f} over {shaded.size} shaded pixels (section 3: 0.41 to 0.50)")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--render")
    p.add_argument("--native", action="store_true", help="400x240, one sample, then a 3x nearest upscale")
    p.add_argument("--time", choices=sorted(R.GRADE), default="day")
    a = p.parse_args(argv)
    R.TIME = a.time
    info = build(a.native)
    if a.check:
        check(info)
    if a.render:
        import time
        out = os.path.abspath(a.render)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        bpy.context.scene.render.filepath = out
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"rendered {out} in {time.time() - t:.1f}s")
        qa(out)
        if a.native and shutil.which("ffmpeg"):
            root, ext = os.path.splitext(out)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-vf",
                            "scale=iw*3:ih*3:flags=neighbor", f"{root}_x3{ext}"], check=True)


if __name__ == "__main__":
    main()
