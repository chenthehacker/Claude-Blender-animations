"""Blueprint test scene 2: the plaza in front of an original village hall, with the player and three villagers.

Original assets only (no game building, characters or logos are copied). Shares the style system of
town_river_test.py: the solved walking camera (16.1), rolling log bend, LUT lighting with the camera side headlight
(section 2), NL rim on characters, multiply shadow decals (blob for characters and posts, silhouette shaped for trees,
bushes and flowers), painted textures with grain, haze on the ground, time of day grades (16.5).

    /root/bpyenv/bin/python town_hall_test.py --check --render renders/town_hall.png
    /root/bpyenv/bin/python town_hall_test.py --native --render renders/town_hall_native.png   (also writes _x3)
    /root/bpyenv/bin/python town_hall_test.py --time dusk|night --render renders/town_hall_night.png

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

# Camera in front of a building. Retail frames in front of houses (S#1436, S#1441, by eye on 5x crops: cap top row 134 to
# 135 of 240, soles 214 to 215) put the feet at 0.89 of the height with the player 0.33 tall: the walking camera trails
# the player. Same pitch, lens and world bend (R stays 13.28 P); this moves the horizon to 0.250, close to the
# corpus value for that view (16.1: "in front of a house, ground ends 27 to 31%").
R.FEET_ROW, R.PLAYER_SHARE = 0.89, 0.33
R.CAM_H, R.CAM_D = R.solve_camera()
R._NEAR = math.hypot(R.CAM_D, R.CAM_H)
R._FAR = R.tangent_distance(R.LOG_R, R.CAM_H, R.CAM_D)
R.HAZE_NEAR = R._NEAR + 0.4 * (R._FAR - R._NEAR)
R.HAZE_FAR = R._FAR

hexc, lin, smoothstep, link = R.hexc, R.lin, R.smoothstep, R.link
new_bm, mesh_object, bm_tube, bm_sphere, bm_disc = R.new_bm, R.mesh_object, R.bm_tube, R.bm_sphere, R.bm_disc
fbm, ramp, make_image, grain, mat = R.fbm, R.ramp, R.make_image, R.grain, R.mat
box_uv = R.B.bm_box_uv
RNG = np.random.default_rng(31)

# --------------------------------------------------------------------------
# Palette (M = measured in the blueprint, E = design choice for an original asset)
# --------------------------------------------------------------------------
HP = {
    "plaza_dirt": "#9D6223",      # M 16.10 NL plaza dirt (mortar between cobbles)
    "cobble_grey": "#AAA9B9",     # M 16.10 NL cobble
    "cobble_warm": "#C2965C",     # E golden brown cobble (NL#940 "golden brown cobble", by eye)
    "plaster": "#E8D6B0", "plaster_shade": "#CDB48C",      # E (kept below the near white band after the LUT)
    "timber": "#7A4A2E", "timber_light": "#9A6440",        # E (wood family of section 7.3)
    "roof": "#BB3F5B", "roof_dark": "#8E2C45", "roof_light": "#D86A7E",   # M 16.7 red roof median, lip and edge E
    "stone": "#B8B2A6", "stone_dark": "#8E8779",           # E
    "door": "#A8693A", "door_dark": "#6E4024",             # E
    "glass": "#9CC8EE", "glass_hi": "#E8F4FF",             # E
    "window_night": "#F4CF6A",                             # M 16.5 NL night windows #F1C791 to #FCD45E
    "metal": "#3E5A4A", "lamp": "#FFF4C8", "lamp_night": "#FFE9A0",   # E post; M night lamp core family 16.10
    "clock": "#FBF6E8", "clock_rim": "#C9A24A",            # E
}
VILLAGERS = [
    # name, species, position, facing (deg, 0 = east, -90 = toward the camera), fur, fur light, accent, shirt a, b
    ("Cat", "cat", (1.45, 0.3), 200.0, "#E8A05A", "#FBE6C4", "#F2A0A8", "#4E9F6B", "#DDEFD9"),
    ("Dog", "dog", (-1.6, 0.75), -55.0, "#D9B98A", "#F6E8CF", "#8C5A3A", "#C8473F", "#F4D7CF"),
    ("Rabbit", "rabbit", (0.75, 1.15), -110.0, "#C9B6E4", "#F4EEF8", "#F2A7B8", "#E8C24A", "#FFF3CC"),
]
PLAYER_FACING = -20.0              # turned toward the cat and the camera
HALL = {"cx": 0.0, "front_y": 2.35, "w": 4.0, "d": 2.6, "wall_h": 1.5,
        "porch_w": 1.6, "porch_d": 0.5, "porch_h": 1.68}


# --------------------------------------------------------------------------
# Painted textures (numpy, sRGB, low resolution, bilinear, felt grain)
# --------------------------------------------------------------------------
def tex_cobble():
    """Round cobbles (16.10: NL round cobbles, grey #AAA9B9) packed tight on a thin plaza dirt mortar (#9D6223):
    a wrapped Voronoi tiling, mortar where the two nearest cell centres are almost equidistant, each stone lighter
    toward its top left and darker at its rim (painted, not lit)."""
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
    border = d2 - d1                                   # pixels to the cell edge (times about 2)
    tone = RNG.uniform(0.88, 1.08, len(pts))
    warm = RNG.random(len(pts)) < 0.25
    base = np.where(warm[idx][..., None], hexc("#B8A68E"), hexc(HP["cobble_grey"])) * tone[idx][..., None]
    cell = w / k
    light = 1.0 + 0.16 * np.clip(1 - np.hypot(xx - cxs + 0.2 * cell, yy - cys + 0.2 * cell) / (0.55 * cell), 0, 1)
    rim = 0.72 + 0.28 * smoothstep(0.8, 5.0, border)      # darker toward the rim: reads as a rounded stone
    stone = base * (light * rim)[..., None]
    mortar = hexc(HP["plaza_dirt"]) * (0.7 + 0.2 * fbm(w, h, 8, 2, RNG))[..., None]   # in shade between stones
    inside = np.minimum(border, 2.0 * (0.62 * cell - d1))   # Voronoi cell clipped by a disc: rounded corners
    t = smoothstep(0.5, 1.5, inside)[..., None]
    col = mortar * (1 - t) + stone * t
    return make_image("T_Cobble", grain(np.clip(col, 0, 1), 0.04))


def tex_wall():
    """Plaster with a painted timber frame: posts, rails and braces; baked darker skirt near the ground."""
    w, h = 128, 128
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = xx / w, 1 - yy / h                      # v = 0 at the bottom of the tile (the ground)
    col = ramp(0.45 + 0.4 * (fbm(w, h, 6, 2, RNG) - 0.5), [HP["plaster_shade"], HP["plaster"], HP["plaster"]])
    beam = np.zeros((h, w), bool)
    beam |= (u < 0.07) | (u > 0.93)               # posts at the tile edges
    beam |= (np.abs(v - 0.55) < 0.035)             # middle rail
    beam |= (v > 0.94)                             # top plate
    diag = np.abs((u - 0.07) / 0.86 - (0.55 - v) / 0.55)
    beam |= (diag < 0.05) & (v < 0.55) & (u > 0.07) & (u < 0.93)   # brace in the lower panel
    wood = ramp(0.4 + 0.4 * fbm(w, h, 16, 1, RNG), [HP["timber"], HP["timber"], HP["timber_light"]])
    col = np.where(beam[..., None], wood, col)
    col = col * (0.8 + 0.2 * smoothstep(0.0, 0.25, v))[..., None]    # painted ground grime and shade
    return make_image("T_HallWall", grain(col, 0.03))


def tex_roof():
    """Tile rows: each row lighter at its top edge, darker lip at the bottom (16.10 roof), slight per tile variation."""
    w, h = 128, 128
    yy, xx = np.mgrid[0:h, 0:w]
    rows = 8
    v = (1 - yy / h) * rows
    r_idx = np.floor(v)
    fv = v - r_idx
    tiles = 6
    u = xx / w * tiles + 0.5 * (r_idx % 2)
    tvar = np.take(RNG.uniform(0.9, 1.1, 200), (np.floor(u).astype(int) * 13 + r_idx.astype(int) * 7) % 200)
    t = 0.25 + 0.6 * fv
    col = ramp(t, [HP["roof_dark"], HP["roof"], HP["roof_light"]]) * tvar[..., None]
    gap = np.abs((u % 1.0) - 0.5) > 0.47
    col[gap] = col[gap] * 0.75
    return make_image("T_Roof", grain(np.clip(col, 0, 1), 0.03))


def tex_door():
    w, h = 64, 128
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = xx / w, 1 - yy / h
    planks = 0.5 + 0.5 * np.sin(2 * np.pi * (u * 5 + 0.15 * fbm(w, h, 2, 2, RNG)))
    col = ramp(0.35 + 0.35 * planks, [HP["door_dark"], HP["door"], HP["door"]])
    frame = (u < 0.06) | (u > 0.94) | (v > 0.95) | (np.abs(u - 0.5) < 0.025)
    col[frame] = hexc(HP["door_dark"])
    knob = np.hypot((u - 0.42) * 2, v - 0.48) < 0.035
    col[knob] = hexc(HP["clock_rim"])
    return make_image("T_Door", grain(col, 0.03))


def tex_glass(night=False):
    w, h = 32, 48
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = xx / w, 1 - yy / h
    if night:
        col = ramp(0.5 + 0.5 * v, ["#E8B85A", HP["window_night"], "#FFF0B8"])
    else:
        col = ramp(v, [HP["glass"], HP["glass"], HP["glass_hi"]])
        glint = np.abs((u - v * 0.7) - 0.15) < 0.06
        col[glint] = hexc(HP["glass_hi"])
    mull = (np.abs(u - 0.5) < 0.05) | (np.abs(v - 0.5) < 0.035)
    col[mull] = hexc("#FFFFFF")
    return make_image("T_Glass_night" if night else "T_Glass", col)


def tex_clock():
    w = h = 64
    yy, xx = (np.mgrid[0:h, 0:w] - 31.5) / 31.5
    r = np.hypot(xx, yy)
    col = np.ones((h, w, 3)) * hexc(HP["clock"])
    col[r > 0.86] = hexc(HP["clock_rim"])
    a = np.arctan2(yy, xx)
    ticks = (r > 0.68) & (r < 0.8) & (np.abs(((a / (np.pi / 6)) + 0.5) % 1 - 0.5) < 0.08)
    col[ticks] = hexc("#3A2A20")
    hand1 = (np.abs(xx) < 0.05) & (yy < 0) & (yy > -0.6)        # 12 o'clock hand (rows run top down)
    hand2 = (np.abs(yy + xx * 0.0) < 0.05) & (xx > 0) & (xx < 0.42)
    col[hand1 | hand2] = hexc("#3A2A20")
    return make_image("T_Clock", col)


def tex_fur(name, base, light, amount=0.035):
    return R.tex_flat(name, [base, base, light], amount, 4)


def tex_villager_face(fur, light, accent, species):
    """Face decal on the head UVs (u 0.5 = facing direction, v bottom to top), section 5.3: eyes are texture decals,
    eye centre 0.33 below the core head top (latitude about 20 deg up), eye height 0.14, spacing 0.38 of the width."""
    w, h = 256, 128
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = xx / w, 1 - yy / h
    col = np.ones((h, w, 3)) * hexc(fur)
    lower = smoothstep(0.50, 0.40, v) * smoothstep(0.17, 0.05, np.abs(u - 0.5))   # lighter muzzle and chin patch
    col = col * (1 - lower[..., None]) + hexc(light) * lower[..., None]
    for sgn in (-1, 1):
        cu, cv = 0.5 + sgn * 0.061, 0.61
        e = ((u - cu) / 0.020) ** 2 + ((v - cv) / 0.047) ** 2
        col[e < 1] = hexc("#1E1A28")
        g = ((u - cu + 0.006) / 0.007) ** 2 + ((v - cv - 0.02) / 0.014) ** 2
        col[g < 1] = 1.0
        b = np.hypot((u - (0.5 + sgn * 0.095)) / 0.035, (v - 0.50) / 0.04)
        k = smoothstep(1.0, 0.3, b)[..., None] * 0.4
        col = col * (1 - k) + hexc("#F4A3A0") * k
    mouth = (np.abs(v - 0.43 - 0.012 * np.cos((u - 0.5) * 60)) < 0.006) & (np.abs(u - 0.5) < 0.025)
    col[mouth] = hexc("#5A3A30")
    if species == "cat":                            # stripes on the forehead
        stripe = (v > 0.72) & (np.abs(((u - 0.5) * 40) % 1 - 0.5) < 0.18) & (np.abs(u - 0.5) < 0.06)
        col[stripe] = hexc(fur) * 0.78
    return make_image(f"T_Face_{species}", grain(col, 0.02))


def tex_vshirt(name, a, b, pattern):
    w, h = 32, 128
    yy, xx = np.mgrid[0:h, 0:w]
    col = np.ones((h, w, 3)) * hexc(a)
    if pattern == "stripes":
        col[(yy // 4) % 2 == 0] = hexc(b)
    elif pattern == "dots":
        col[((xx % 8 - 4) ** 2 + (yy % 8 - 4) ** 2) < 4] = hexc(b)
    else:                                            # plaid
        col[((xx // 4) % 2 == 0) | ((yy // 4) % 2 == 0)] = hexc(b) * 0.85 + hexc(a) * 0.15
        col[((xx // 4) % 2 == 0) & ((yy // 4) % 2 == 0)] = hexc(b)
    return make_image(name, grain(col, 0.02))


# --------------------------------------------------------------------------
# Ground: flat plaza blending softly into grass (R.terrain_mat: "wall" weight = cobble)
# --------------------------------------------------------------------------
def plaza_weight(x, y):
    """Rounded plaza in front of the hall, soft edged (NL boundaries blend, section 4.2)."""
    hx, y0, y1 = 2.7, -1.6, HALL["front_y"] + HALL["d"] + 1.0
    dx = max(abs(x) - hx, 0.0)
    dy = max(y0 - y, y - y1, 0.0)
    return float(smoothstep(0.6, 0.0, math.hypot(dx, dy)))


def build_ground(col, M):
    xs = np.concatenate([np.arange(-30, -8, 1.0), np.arange(-8, 8, 0.12), np.arange(8, 30.01, 1.0)])
    ys = np.concatenate([np.arange(-8, 14, 0.12), np.arange(14, 40.01, 0.6)])
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
# The village hall (original design)
# --------------------------------------------------------------------------
def build_hall(col, M):
    """Plinth, timber framed plaster walls, a red tiled gable roof, a projecting entrance porch with a clock, a double
    door on two stone steps, one window with a flower box each side. Sized so the facade fits the frame below the
    eaves, as houses do in retail frames (S#1436, S#1441: the roof runs off the top edge)."""
    cx, fy, W, D, H = HALL["cx"], HALL["front_y"], HALL["w"], HALL["d"], HALL["wall_h"]
    pw, pd, ph = HALL["porch_w"], HALL["porch_d"], HALL["porch_h"]
    cy = fy + D / 2
    pf = fy - pd                                        # porch front face
    objs = []
    bm, uv = new_bm()
    box_uv(bm, uv, (cx, cy, 0.1), (W + 0.12, D + 0.12, 0.2), uv_scale=(0.8, 0.8))
    objs.append(mesh_object("HallPlinth", bm, col, [M["h_stone"]], smooth=False))
    bm, uv = new_bm()
    wh = H - 0.2
    box_uv(bm, uv, (cx, cy, 0.2 + wh / 2), (W, D, wh), uv_scale=(1 / 1.15, 1 / wh))
    objs.append(mesh_object("HallWalls", bm, col, [M["h_wall"]], smooth=False))
    # main gable roof along X
    ov, ridge = 0.3, H + 1.0
    bm, uv = new_bm()
    x0, x1 = cx - W / 2 - ov, cx + W / 2 + ov
    yf, yb = fy - ov, fy + D + ov
    zf = H - 0.1
    for (ya, yr) in ((yf, cy), (yb, cy)):
        vs = [bm.verts.new(p) for p in ((x0, ya, zf), (x1, ya, zf), (x1, yr, ridge), (x0, yr, ridge))]
        f = bm.faces.new(vs if ya < cy else list(reversed(vs)))
        slope = math.hypot(yr - ya, ridge - zf)
        for lp in f.loops:
            q = lp.vert.co
            lp[uv].uv = ((q.x - x0) * 0.7, (0 if abs(q.z - zf) < 1e-6 else slope) * 0.85)
    bmesh.ops.solidify(bm, geom=list(bm.faces), thickness=0.07)
    objs.append(mesh_object("HallRoof", bm, col, [M["h_roof"]], smooth=False))
    bm, uv = new_bm()                                  # gable end triangles
    for xe in (cx - W / 2, cx + W / 2):
        vs = [bm.verts.new(p) for p in ((xe, fy, H - 0.1), (xe, fy + D, H - 0.1), (xe, cy, ridge - 0.05))]
        f = bm.faces.new(vs)
        for lp in f.loops:
            lp[uv].uv = (lp.vert.co.y * 0.6, lp.vert.co.z * 0.6)
    mesh_object("HallGables", bm, col, [M["h_plaster"]], smooth=False)
    # entrance porch with its own gable facing the plaza
    bm, uv = new_bm()
    box_uv(bm, uv, (cx, fy - pd / 2 + 0.02, ph / 2), (pw, pd + 0.04, ph), uv_scale=(1 / 1.15, 1 / ph))
    objs.append(mesh_object("HallPorch", bm, col, [M["h_wall"]], smooth=False))
    pr = ph + 0.55
    bm, uv = new_bm()
    py0, py1 = pf - 0.22, fy + 0.25
    for sgn in (-1, 1):
        xa = cx + sgn * (pw / 2 + 0.2)
        vs = [bm.verts.new(p) for p in ((xa, py0, ph - 0.12), (xa, py1, ph - 0.12), (cx, py1, pr), (cx, py0, pr))]
        f = bm.faces.new(vs if sgn < 0 else list(reversed(vs)))
        for lp in f.loops:
            q = lp.vert.co
            lp[uv].uv = ((q.y - py0) * 0.9, (0 if q.z < ph else 0.75))
    bmesh.ops.solidify(bm, geom=list(bm.faces), thickness=0.06)
    objs.append(mesh_object("PorchRoof", bm, col, [M["h_roof"]], smooth=False))
    bm, uv = new_bm()
    vs = [bm.verts.new(p) for p in ((cx - pw / 2, pf - 0.005, ph), (cx + pw / 2, pf - 0.005, ph),
                                    (cx, pf - 0.005, pr - 0.05))]
    f = bm.faces.new(vs)
    for lp in f.loops:
        lp[uv].uv = (lp.vert.co.x * 0.6, lp.vert.co.z * 0.6)
    mesh_object("PorchGable", bm, col, [M["h_plaster"]], smooth=False)
    # clock above the door
    bm, uv = new_bm()
    cc = Vector((cx, pf - 0.012, 1.36))
    rr = 0.16
    ring = [bm.verts.new(cc + Vector((math.cos(a) * rr, 0, math.sin(a) * rr)))
            for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
    ctr = bm.verts.new(cc)
    for i in range(24):
        f = bm.faces.new((ctr, ring[(i + 1) % 24], ring[i]))
        for lp in f.loops:
            o = lp.vert.co - cc
            lp[uv].uv = (0.5 + o.x / (2 * rr), 0.5 + o.z / (2 * rr))
    mesh_object("Clock", bm, col, [M["h_clock"]], smooth=False)
    # double door, frame and two steps
    bm, uv = new_bm()
    box_uv(bm, uv, (cx, pf - 0.02, 0.69), (0.92, 0.04, 0.9), uv_scale=(1, 1))
    mesh_object("DoorFrame", bm, col, [M["h_frame"]], smooth=False)
    bm, uv = new_bm()
    vs = [bm.verts.new(p) for p in ((cx - 0.38, pf - 0.045, 0.27), (cx + 0.38, pf - 0.045, 0.27),
                                    (cx + 0.38, pf - 0.045, 1.08), (cx - 0.38, pf - 0.045, 1.08))]
    f = bm.faces.new(vs)
    for lp, q in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
        lp[uv].uv = q
    mesh_object("Door", bm, col, [M["h_door"]], smooth=False)
    bm, uv = new_bm()
    box_uv(bm, uv, (cx, pf - 0.2, 0.07), (1.3, 0.4, 0.14), uv_scale=(0.9, 0.9))
    box_uv(bm, uv, (cx, pf - 0.1, 0.18), (1.1, 0.2, 0.1), uv_scale=(0.9, 0.9))
    objs.append(mesh_object("Steps", bm, col, [M["h_stone"]], smooth=False))
    # windows with white frames and flower boxes
    for sgn in (-1, 1):
        wx = cx + sgn * 1.32
        bm, uv = new_bm()
        box_uv(bm, uv, (wx, fy - 0.03, 0.9), (0.66, 0.08, 0.72), uv_scale=(1, 1))
        mesh_object("WinFrame", bm, col, [M["h_frame"]], smooth=False)
        bm, uv = new_bm()
        vs = [bm.verts.new(p) for p in ((wx - 0.26, fy - 0.075, 0.6), (wx + 0.26, fy - 0.075, 0.6),
                                        (wx + 0.26, fy - 0.075, 1.2), (wx - 0.26, fy - 0.075, 1.2))]
        f = bm.faces.new(vs)
        for lp, q in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
            lp[uv].uv = q
        mesh_object("WinGlass", bm, col, [M["h_glass"]], smooth=False)
        bm, uv = new_bm()
        box_uv(bm, uv, (wx, fy - 0.15, 0.5), (0.66, 0.2, 0.14), uv_scale=(1.2, 1.2))
        mesh_object("FlowerBox", bm, col, [M["h_box"]], smooth=False)
        bm, uv = new_bm()
        for j in range(5):
            bm_sphere(bm, uv, (wx - 0.24 + 0.12 * j, fy - 0.16, 0.6), 0.055, segs=8, rings=5)
        mesh_object("BoxFlowers", bm, col, [M["h_boxflower"]])
    # contact shadow around the base: footprint grown and softened (sections 3 and 16.6)
    R.profile_shadow(col, "HallShadow", [(o, None) for o in objs[:1]] + [(objs[3], None), (objs[5], None)],
                     cx, cy - 0.3, 0.012, max(W, D) / 2 + 0.8, "contact")


def build_lamp(col, M, x, y):
    bm, uv = new_bm()
    bm_tube(bm, uv, [(x, y, 0.0), (x, y, 0.1), (x, y, 1.12)], [0.065, 0.042, 0.033], 6, cap_end=True)
    bm_sphere(bm, uv, (x, y, 0.05), 0.075, scale=(1, 1, 0.6), segs=8, rings=4)
    box_uv(bm, uv, (x, y, 1.41), (0.24, 0.24, 0.04), uv_scale=(1, 1))
    bm_tube(bm, uv, [(x, y, 1.43), (x, y, 1.58)], [0.18, 0.02], 4, cap_start=True, twist=math.pi / 4)
    mesh_object("LampPost", bm, col, [M["h_metal"]])
    bm, uv = new_bm()
    box_uv(bm, uv, (x, y, 1.25), (0.18, 0.18, 0.3), uv_scale=(1, 1))
    mesh_object("LampGlass", bm, col, [M["h_lamp"]], smooth=False)
    R.blob(col, M["contact"], "LampShadow", (x, y, 0.01), 0.16, 0.14)


def build_noticeboard(col, M, x, y):
    bm, uv = new_bm()
    for sx in (-0.45, 0.45):
        box_uv(bm, uv, (x + sx, y, 0.6), (0.08, 0.08, 1.2), uv_scale=(1, 1))
    box_uv(bm, uv, (x, y, 1.0), (1.1, 0.07, 0.62), uv_scale=(1, 1))
    mesh_object("Board", bm, col, [M["h_door"]], smooth=False)
    bm, uv = new_bm()
    for k, (px, pz) in enumerate(((-0.3, 1.08), (0.05, 0.98), (0.3, 1.1), (-0.12, 0.86))):
        box_uv(bm, uv, (x + px, y - 0.04, pz), (0.24, 0.01, 0.2), uv_scale=(1, 1))
    mesh_object("Notes", bm, col, [M["h_note"]], smooth=False)
    R.blob(col, M["contact"], "BoardShadow", (x, y, 0.01), 0.62, 0.2)


# --------------------------------------------------------------------------
# Villagers: original animals with NL standard animal proportions (section 5.1)
# --------------------------------------------------------------------------
def build_villager(col, M, name, species, pos, facing, fur, light, accent):
    """H = 1.0. Head with ears 0.50 of the height; core head 0.40 (chin 0.50, top 0.90); head width over core head
    height 1.44; eyes 0.33 down the core head; leg gap 0.11; arms reach hip height; mitten hands (5.1, 5.2)."""
    yaw = math.radians(facing)
    Rz = Matrix.Rotation(yaw, 4, "Z")
    fwd = Vector((math.cos(yaw), math.sin(yaw), 0))
    side = Vector((-fwd.y, fwd.x, 0))
    base = Vector((pos[0], pos[1], 0.0))

    def at(f, s, z):
        return base + fwd * f + side * s + Vector((0, 0, z))

    head_c = at(0.0, 0.0, 0.70)
    HX, HY, HZ = 0.25, 0.288, 0.20
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    res = bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=1.0,
                                    matrix=Matrix.Translation(head_c) @ Rz @ Matrix.Diagonal((HX, HY, HZ, 1.0)))
    for f in {f for v in res["verts"] for f in v.link_faces}:
        for lp in f.loops:
            d = Rz.inverted() @ (lp.vert.co - head_c)
            d = Vector((d.x / HX, d.y / HY, d.z / HZ)).normalized()
            lp[uv].uv = (0.5 + math.atan2(d.y, d.x) / (2 * math.pi), 0.5 + math.asin(max(-1, min(1, d.z))) / math.pi)
        us = [lp[uv].uv.x for lp in f.loops]
        if max(us) - min(us) > 0.5:
            for lp in f.loops:
                if lp[uv].uv.x < 0.5:
                    lp[uv].uv.x += 1.0
    mesh_object(f"{name}Head", bm, col, [M[f"v_{name}_face"]])

    def surf(az, la, out=0.0):
        a, l = math.radians(az), math.radians(la)
        return head_c + Rz.to_3x3() @ Vector(((HX + out) * math.cos(l) * math.cos(a), (HY + out) * math.cos(l) * math.sin(a),
                                               (HZ + out) * math.sin(l)))

    # muzzle and nose by species
    bm, uv = new_bm()
    if species == "dog":
        bm_sphere(bm, uv, surf(0, -8, -0.03), 0.09, scale=(1.1, 1.0, 0.75), segs=10, rings=6, rot=Rz.to_quaternion())
    mesh_object(f"{name}Muzzle", bm, col, [M[f"v_{name}_light"]])
    bm, uv = new_bm()
    nose_at = surf(0, -2, 0.06) if species == "dog" else surf(0, -4, 0.005)
    bm_sphere(bm, uv, nose_at, 0.035 if species == "dog" else 0.022, scale=(0.8, 1.0, 0.75), segs=8, rings=5,
              rot=Rz.to_quaternion())
    mesh_object(f"{name}Nose", bm, col, [M["v_nose_dark"] if species == "dog" else M[f"v_{name}_accent"]])
    # ears by species
    bm, uv = new_bm()
    bm2, uv2 = new_bm()
    for sgn in (-1, 1):
        if species == "cat":
            b0 = surf(sgn * 38, 52, -0.02)
            tip = b0 + Rz.to_3x3() @ Vector((0.0, sgn * 0.05, 0.17))
            bm_tube(bm, uv, [b0, tip], [0.085, 0.005], 4, cap_start=True, twist=math.pi / 4)
            bm_tube(bm2, uv2, [b0 + Rz.to_3x3() @ Vector((0.03, 0, 0.01)), tip + Rz.to_3x3() @ Vector((0.03, 0, -0.03))],
                    [0.05, 0.004], 4, cap_start=True, twist=math.pi / 4)
        elif species == "dog":
            c = surf(sgn * 82, 12, 0.03) + Vector((0, 0, -0.08))
            rot = Vector((0, 0, 1)).rotation_difference(Rz.to_3x3() @ Vector((0, sgn * 0.25, -1)).normalized())
            bm_sphere(bm, uv, c, 0.12, scale=(0.55, 0.3, 1.0), segs=10, rings=6, rot=rot)
        else:                                          # rabbit: long upright ears
            b0 = surf(sgn * 28, 60, -0.03)
            c = b0 + Rz.to_3x3() @ Vector((0.0, sgn * 0.04, 0.2))
            rot = Vector((0, 0, 1)).rotation_difference(Rz.to_3x3() @ Vector((0, sgn * 0.18, 1)).normalized())
            bm_sphere(bm, uv, c, 0.21, scale=(0.32, 0.22, 1.0), segs=10, rings=6, rot=rot)
            bm_sphere(bm2, uv2, c + Rz.to_3x3() @ Vector((0.045, 0, 0)), 0.17, scale=(0.2, 0.12, 0.85), segs=8, rings=5,
                      rot=rot)
    ear_mat = M[f"v_{name}_accent"] if species == "dog" else M[f"v_{name}_fur"]
    mesh_object(f"{name}Ears", bm, col, [ear_mat])
    mesh_object(f"{name}EarsInner", bm2, col, [M["v_ear_inner"]])
    # body (shirt), arms, mitten hands, legs, tail
    bm, uv = new_bm()
    bm_tube(bm, uv, [at(0, 0, 0.16), at(0, 0, 0.32), at(0, 0, 0.51)], [0.19, 0.185, 0.13], 10, v0=0.75, v1=1.0,
            cap_end=True, cap_start=True)
    for sgn in (-1, 1):
        bm_sphere(bm, uv, at(0, sgn * 0.15, 0.45), 0.055, segs=8, rings=5)
        bm_tube(bm, uv, [at(0, sgn * 0.17, 0.45), at(0.02, sgn * 0.215, 0.36)], [0.052, 0.048], 6, v0=0.0, v1=0.24)
    mesh_object(f"{name}Body", bm, col, [M[f"v_{name}_shirt"]])
    bm, uv = new_bm()
    for sgn in (-1, 1):
        bm_tube(bm, uv, [at(0.02, sgn * 0.215, 0.37), at(0.05, sgn * 0.235, 0.25)], [0.042, 0.038], 6)
        bm_sphere(bm, uv, at(0.05, sgn * 0.235, 0.225), 0.05, segs=8, rings=5)
        bm_tube(bm, uv, [at(0, sgn * 0.085, 0.2), at(0, sgn * 0.085, 0.03)], [0.058, 0.055], 6)
        bm_sphere(bm, uv, at(0.03, sgn * 0.09, 0.028), 0.065, scale=(1.3, 0.9, 0.5), segs=8, rings=5)
    if species == "cat":
        bm_tube(bm, uv, [at(-0.18, 0, 0.16), at(-0.3, 0.05, 0.25), at(-0.33, 0.08, 0.42)], [0.035, 0.03, 0.025], 6)
    elif species == "dog":
        bm_tube(bm, uv, [at(-0.18, 0, 0.2), at(-0.27, 0, 0.3)], [0.04, 0.02], 6)
    else:
        bm_sphere(bm, uv, at(-0.2, 0, 0.18), 0.07, segs=8, rings=5)
    mesh_object(f"{name}Limbs", bm, col, [M[f"v_{name}_fur"]])
    # round blob shadow (characters keep round blobs, section 3 and 16.15)
    dep = math.atan2(R.CAM_H, R.CAM_D)
    rx = R.BLOB_EXTENT / 2 * 1.15
    R.blob(col, M["blob"], f"{name}Shadow", (pos[0], pos[1], 0.006), rx, rx / (R.BLOB_ASPECT_ON_SCREEN * math.sin(dep)))
    return head_c


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
TREES = [(-3.3, 3.0, 2.0), (3.5, 3.3, 2.1), (-4.4, 6.4, 2.1), (4.6, 7.0, 2.2), (-1.2, 8.6, 2.2)]
FLOWER_BEDS = [(-2.05, 1.9), (2.05, 1.95), (-2.75, 0.8), (2.6, 0.1)]
LAMPS = [(-1.1, 1.5), (1.15, 1.55)]


def build(native=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rng = np.random.default_rng(5)
    T, M = R.make_assets()
    T["wall"] = tex_cobble()                         # R.terrain_mat blends grass and "wall": here the cobble plaza
    M["terrain"] = R.terrain_mat(T)
    night = R.TIME == "night"
    M.update({
        "h_stone": mat("M_HStone", image=R.tex_flat("T_HStone", [HP["stone_dark"], HP["stone"], "#D2CCC0"], 0.05, 6)),
        "h_wall": mat("M_HWall", image=tex_wall()),
        "h_plaster": mat("M_HPlaster", image=R.tex_flat("T_Plaster", [HP["plaster_shade"], HP["plaster"], "#FFF6E4"])),
        "h_roof": mat("M_HRoof", image=tex_roof()),
        "h_door": mat("M_HDoor", image=tex_door()),
        "h_frame": mat("M_HFrame", image=R.tex_flat("T_Frame", ["#DCD6CA", "#EFEBE2", "#F6F3EC"], 0.02)),
        "h_glass": R.unlit("M_HGlassN", HP["window_night"]) if night else mat("M_HGlass", image=tex_glass()),
        "h_box": mat("M_HBox", image=R.tex_flat("T_Box", ["#7A4A2E", "#9A6440", "#B07850"], 0.04)),
        "h_boxflower": mat("M_HBoxFlower", image=R.tex_flat("T_BoxFlower", ["#E04A6A", "#F06C8A", "#FFC0D0"], 0.04)),
        "h_clock": mat("M_HClock", image=tex_clock()),
        "h_metal": mat("M_HMetal", image=R.tex_flat("T_Metal", ["#2A4034", HP["metal"], "#5C7A68"], 0.03)),
        "h_lamp": R.unlit("M_HLampN", HP["lamp_night"]) if night else mat("M_HLamp", color=HP["lamp"]),
        "h_note": mat("M_HNote", image=R.tex_flat("T_Note", ["#E8E0C8", "#FFF8E8", "#FFFFFF"], 0.02)),
        "v_nose_dark": mat("M_VNoseDark", color="#3A2A2A", rim=1.0),
        "v_ear_inner": mat("M_VEarInner", color="#F4B3C0", rim=1.0),
    })
    for name, species, pos, facing, fur, light, accent, sa, sb in VILLAGERS:
        pattern = {"cat": "stripes", "dog": "plaid", "rabbit": "dots"}[species]
        M[f"v_{name}_face"] = mat(f"M_{name}Face", image=tex_villager_face(fur, light, accent, species), rim=1.0)
        M[f"v_{name}_fur"] = mat(f"M_{name}Fur", image=tex_fur(f"T_{name}Fur", fur, light), rim=1.0)
        M[f"v_{name}_light"] = mat(f"M_{name}Light", image=tex_fur(f"T_{name}Light", light, "#FFFFFF"), rim=1.0)
        M[f"v_{name}_accent"] = mat(f"M_{name}Accent", color=accent, rim=1.0)
        M[f"v_{name}_shirt"] = mat(f"M_{name}Shirt", image=tex_vshirt(f"T_{name}Shirt", sa, sb, pattern), rim=1.0)
    world = bpy.data.collections.new("World")
    bpy.context.scene.collection.children.link(world)
    R.build_sky()
    build_ground(world, M)
    build_hall(world, M)
    for x, y in LAMPS:
        build_lamp(world, M, x, y)
    for x, y, hgt in TREES:
        R.build_tree(world, M, x, y, hgt, rng)
    for x, y in FLOWER_BEDS:
        R.build_flowers(world, M, x, y, rng)
    R.build_bush(world, M, -2.1, -0.2, 0.4, rng)         # foreground occluder (section 9 rule 4)
    head_c, _ = R.build_player(world, M, pos=(0.0, 0.0, 0.0), facing_deg=PLAYER_FACING, fishing=False)
    dep = math.atan2(R.CAM_H, R.CAM_D)
    rx = R.BLOB_EXTENT / 2
    R.blob(world, M["blob"], "PlayerShadow", (0.0, 0.0, 0.006), rx, rx / (R.BLOB_ASPECT_ON_SCREEN * math.sin(dep)))
    heads = {name: build_villager(world, M, name, species, pos, facing, fur, light, accent)
             for name, species, pos, facing, fur, light, accent, sa, sb in VILLAGERS}
    R.bend_world()
    cam = R.build_camera(world)
    R.finalize_render(native, world, M, cam)
    return {"cam": cam, "heads": heads}


def check(info):
    sc = bpy.context.scene
    bpy.context.view_layer.update()
    cam = info["cam"]
    def row_col(p):
        v = world_to_camera_view(sc, cam, R.bend_point(Vector(p)))
        return 1 - v.y, v.x
    print(f"camera: height {R.CAM_H:.2f} P, distance {R.CAM_D:.2f} P, bend R {R.LOG_R:.2f} P, open ground horizon "
          f"{R.horizon_row_for(R.LOG_R, R.CAM_H, R.CAM_D):.3f}")
    for k, p in {"player feet": (0, 0, 0), "player hair top": (0, 0, 1.0)}.items():
        r, c = row_col(p)
        print(f"  {k:16s} row {r:.3f} column {c:.3f}")
    for name, species, pos, *_ in VILLAGERS:
        r0, c0 = row_col((pos[0], pos[1], 0.0))
        r1, _ = row_col((pos[0], pos[1], 1.0))
        print(f"  {name:7s} feet row {r0:.3f} column {c0:.3f}, height share {r0 - r1:.3f}")
    fy, H, pf = HALL["front_y"], HALL["wall_h"], HALL["front_y"] - HALL["porch_d"]
    for k, p in {"door bottom": (0, pf, 0.27), "clock top": (0, pf, 1.52), "porch eave": (0, pf, HALL["porch_h"]),
                 "hall eave": (HALL["w"] / 2, fy - 0.3, H - 0.1), "hall left corner": (-HALL["w"] / 2, fy, 0),
                 "hall right corner": (HALL["w"] / 2, fy, 0), "window top": (1.32, fy, 1.26)}.items():
        r, c = row_col(p)
        print(f"  {k:16s} row {r:.3f} column {c:.3f}{'' if 0 <= r <= 1 and 0 <= c <= 1 else '  OUT OF FRAME'}")
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
    def sample(p):
        q = world_to_camera_view(sc, cam, R.bend_point(Vector(p)))
        x, y = int(q.x * w), int((1 - q.y) * h)
        return px[max(0, y - 1):y + 2, max(0, x - 1):x + 2].reshape(-1, 3).mean(0)
    # Blob darkness: render again with the player's blob hidden and divide, pixel by pixel, inside the blob core
    # (the cobble pattern makes single spot samples meaningless).
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
    box = ratio[cyp - h // 20:cyp + h // 20, cxp - w // 25:cxp + w // 25]
    shaded = box[box < 0.97]
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
