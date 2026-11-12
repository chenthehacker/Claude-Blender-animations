"""Blueprint test scene: a retail New Leaf look town river bank, built from style_blueprint/BLUEPRINT.md.

Original assets only. Every style number comes from the blueprint (section noted in comments). The camera is
solved from the measured framing targets, the rolling log radius is calibrated to the measured horizon row, and
the rendered PNG is checked against the blueprint QA list (section 12.2 item 12).

    /root/bpyenv/bin/python town_river_test.py --check
    /root/bpyenv/bin/python town_river_test.py --render renders/town_river.png      # 1200x720, clean
    /root/bpyenv/bin/python town_river_test.py --native --render renders/town_river_native.png

Units: P = player height = 1.0. X east, Y north, Z up. The camera looks north; the player stands at the origin.
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
sys.path.insert(0, os.path.join(HERE, "..", "style_test_beach"))
import beach_style_test as B  # noqa: E402  (shared mesh, texture and node helpers)

hexc, lin, smoothstep, link = B.hexc, B.lin, B.smoothstep, B.link
new_bm, mesh_object, bm_tube, bm_sphere, bm_disc = B.new_bm, B.mesh_object, B.bm_tube, B.bm_sphere, B.bm_disc
fbm, ramp, make_image, vnoise = B.fbm, B.ramp, B.make_image, B.vnoise

# --------------------------------------------------------------------------
# Blueprint constants (NL preset, section 12.1). M = measured, E = estimate or design choice.
# --------------------------------------------------------------------------
PITCH = 35.0                 # M section 1.1
VFOV = 28.0                  # M section 1.1 (25 to 31)
SENSOR_H = 24.0
FOCAL = 0.5 * SENSOR_H / math.tan(math.radians(VFOV) / 2)    # 48.1 mm, sensor fit VERTICAL
PLAYER_SHARE = 0.32          # M section 1.2 (31 to 34%)
FEET_ROW = 0.60              # M section 16.1: retail walking view feet row varies 44 to 63% (camera trails the player)
HORIZON_ROW = 0.144          # M section 16.1: retail walking view, ground starts at row 34 or 35 of 240 (H)
RES = (1200, 720)            # 5:3 game frame at 3x, clean (section 12.2 item 11)
NATIVE = (400, 240)
SAMPLES = 16

# Key light from the camera side ("headlight", section 2: brightest band central on round forms, falloff toward
# every silhouette edge, no left or right bias): along the reversed view direction, 35 deg above the ground.
SUN = Vector((0.0, -math.cos(math.radians(35.0)), math.sin(math.radians(35.0)))).normalized()
LUT_SHADOW_SRGB = (0.54, 0.56, 0.65)            # M NL rock, section 2
# The measured light end (1.46, 1.38, 1.20) is an UPPER BOUND that includes painted texture variation (section 2).
# Applied as lighting on top of painted textures it double counts and reads as shine, so the light term is kept
# small here: estimate, matched by eye to NL#949 and NL#951 (no hot spots on matte surfaces).
LUT_LIGHT_SRGB = (1.10, 1.08, 1.04)
RIM_TINT = "#C8B4E6"                            # E tint, gain M: +22% displayed, about 1.5x linear (section 2)
RIM_LINEAR_GAIN = 0.5
BLOB_W50, BLOB_EXTENT = 0.38, 0.43              # M section 3, in P
BLOB_ASPECT_ON_SCREEN = 1.35                    # M section 3
BLOB_TINT_LIN = (0.018, 0.028, 0.032)           # E dark blue grey blend target (section 3 audit: blue must rise)
# Core alpha solved so G falls to about 0.46 of the retail grass #5A964C (lin G 0.305 to 0.060 with target 0.028):
# a = (0.305 - 0.060) / (0.305 - 0.028) = 0.885. Blue is multiplied less, as measured (section 3).
BLOB_CORE_ALPHA = 0.88
# Object shadows, measured on NL#946 (E3 2010, the only frame with a near tree on open grass; M):
# canopy shadow about 0.65 of the canopy width, centred under it, soft irregular edge, luminance 0.77 to 0.84 of
# open grass near its rim and darker toward the trunk; ground under flower clumps 0.68; rock contact band dark.
# Same blend target as the player blob. Core alpha for a displayed ratio k on #5A964C (lin G 0.305):
# a = (0.305 - lin(150 k)) / (0.305 - 0.028): k 0.66 gives 0.65 (contact), k 0.75 gives 0.52 (canopy).
CONTACT_ALPHA, CANOPY_ALPHA = 0.72, 0.68      # tuned on the render to hit the measured ratios (blend mode)
# Multiply mode (default): shadows darken the ground and keep its painted texture (NL#946: the grass print runs
# through every shadow; a blend toward a flat colour reads as a sticker). Displayed (sRGB) core ratios, measured:
# player blob R = G about 0.46, blue multiplied less (0.62); flower ground 0.68; canopy darker inside, 0.77 to 0.84
# at the rim. Blue is lifted relative to R and G, as measured. A multiply cannot raise blue, so the lime "emote"
# grass keeps the blend mode.
SHADOW_MUL = {"blob": (0.54, 0.54, 0.68), "contact": (0.58, 0.58, 0.72), "canopy": (0.58, 0.60, 0.72)}
SHADOW_MODE = "multiply"
CANOPY_SHADOW_FRAC = 0.65

PAL = {
    "sky_top": "#2048EE", "sky_low": "#216EFC",       # M 16.2 walking strip: #2040F0 to #2050ED top, #2166FA to #2172FD
    # M retail stream capture supplied by the user (compressed 958x575 game area): near grass median #568F4F to
    # #5D9D49, p10 #137434 to #287D36, printed patches p90 #9DCA64, far ground #9FCE60 (haze). Medium confidence.
    "grass": "#0D7541", "grass_dark": "#026637", "grass_light": "#3E9A5C", "grass_print": "#005A3D",
    # M 16.3: spring and summer near grass (#026637 to #107442), darker print p10 #005A3D; round dot motif (by eye,
    # S#1341, 1246, 1351), dot colour E. Haze target between the measured far bands #94CC78 and #B5E36B.
    "haze": "#8EC668",                               # between the measured retail far ground #88BF68 and #94CC78 (16.3), suits the darker grass
    "cliff_lit": "#FFCEAE", "cliff_shade": "#BA8B8B",                   # M section 4.2
    "water": "#316BAF", "water_shallow": "#5AA2DD", "foam": "#ECF2F8",  # M 16.4: river #316BAF, shallow band, foam line
    "foliage": "#4A9C52", "foliage_dark": "#287046", "foliage_light": "#82D362",  # M section 7.3
    "wood": "#E4A560", "wood_dark": "#A1693B", "wood_light": "#FFD391",   # M section 7.3
    "skin": "#FDDBB2", "skin_shade": "#CE8F79",       # M skin: median of retail NL#941, 942, 944, 945 (shade 7.3)
    "rock": "#9992A2",                                                  # M section 4.2
    "hair": "#7A5450", "hair_light": "#9A706A",       # M hair median of retail NL#941 to 945 (light E)
    "nose": "#CF5846",                                # M retail NL#941, 945 nose nub
    "underwater": "#A0A5AB", "water_deep": "#3B6DAC", # M NL#957 submerged post just below the line, then water
    "shirt": "#3F8FD0", "shirt_light": "#8EC6F0", "shorts": "#2F3E8C", "shoe": "#F2F2F2",  # E
    "eye": "#1E1A28", "blush": "#E08A76",                               # E eye; M blush range section 5.3
    "bobber": "#E0473A", "rod": "#C88838", "rod_band": "#585888",       # E accent red; rod structure section 5.3
    "flower_a": "#FFF6E0", "flower_b": "#FFE15A", "flower_centre": "#F0A040",  # E pastel (keep red unique)
}

TIME = "day"
GRADE = {"day": (1.0, 1.0, 1.0), "dusk": (0.83, 0.58, 0.67), "night": (0.34, 0.31, 0.70)}   # M 16.5 displayed
WATER_BY_TIME = {"dusk": ("#3C5596", "#5A6FA8"), "night": ("#0E1D53", "#1B2C6E")}   # M 16.4 dusk sea, 16.10 night; shallow E
SKY_BY_TIME = {"dusk": ("#7D617B", "#C76153"), "night": ("#01215A", "#01215A")}            # M 16.2 (S#12510; night flat)


def gdisp():
    return np.array(GRADE[TIME])


def gcol(hexstr):
    """A colour constant as it appears under the time of day grade (displayed sRGB multiply)."""
    return np.clip(hexc(hexstr) * gdisp(), 0, 1)


GRASS_PRESS = {"grass": "#188050", "grass_dark": "#02613E", "grass_light": "#2A9A5E", "haze": "#94CC78"}  # M press
GRASS_EMOTE = {"grass": "#92DB10", "grass_dark": "#82CC09", "grass_light": "#A1E917", "haze": "#A1E917"}  # M NL#941 to 945

GRASS_Z, WATER_Z, BED_Z = 0.0, -0.42, -0.75


def bank_x(y):
    """West bank edge of the river (x of the lip). The river runs north, curving gently."""
    return 1.15 + 0.35 * math.sin(0.22 * y + 0.4)


RIVER_W = 3.2


def terrain_h(x, y):
    """Grass at 0, steep rounded walls down to the bed between the banks (section 4.2: tall walls, grass edge
    rounds over the drop)."""
    xa, xb = bank_x(y), bank_x(y) + RIVER_W
    inside = smoothstep(xa, xa + 0.22, x) * smoothstep(xb, xb - 0.22, x)
    return GRASS_Z + (BED_Z - GRASS_Z) * float(inside)


# --------------------------------------------------------------------------
# Camera solve and bend calibration (sections 1.1 to 1.3)
# --------------------------------------------------------------------------
def row_of(dep_deg):
    """Row from the top (0..1) of a ray dep_deg below horizontal, on the centre column."""
    return 0.5 + 0.5 * math.tan(math.radians(dep_deg - PITCH)) / math.tan(math.radians(VFOV) / 2)


def dep_for_row(row):
    return PITCH + math.degrees(math.atan((row - 0.5) * 2 * math.tan(math.radians(VFOV) / 2)))


def solve_camera():
    """Camera height h and ground distance d to the feet such that the feet sit on FEET_ROW and the hair top
    (z = 1) sits PLAYER_SHARE above it."""
    a_feet = math.radians(dep_for_row(FEET_ROW))
    head_row_target = FEET_ROW - PLAYER_SHARE
    lo, hi = 1.0, 20.0
    for _ in range(80):
        d = 0.5 * (lo + hi)
        h = d * math.tan(a_feet)
        head_row = row_of(math.degrees(math.atan((h - 1.0) / d)))
        if head_row < head_row_target:     # head too high on screen: player too big, move back
            lo = d
        else:
            hi = d
    return h, d


def horizon_row_for(R, h, d):
    """Topmost row reached by the bent ground (cylinder wrap from the feet, axis along X)."""
    best = 90.0
    for th in np.linspace(1e-4, math.pi / 2, 4000):
        y, z = R * math.sin(th), R * math.cos(th) - R
        best = min(best, math.degrees(math.atan2(h - z, y + d)))
    return row_of(best)


def calibrate_radius(h, d):
    lo, hi = 3.0, 200.0            # small R lowers the horizon row (more sky)
    for _ in range(60):
        R = 0.5 * (lo + hi)
        if horizon_row_for(R, h, d) > HORIZON_ROW:
            lo = R
        else:
            hi = R
    return 0.5 * (lo + hi)


CAM_H, CAM_D = solve_camera()
LOG_R = calibrate_radius(CAM_H, CAM_D)


def tangent_distance(R, h, d):
    """Camera distance to the point where the line of sight grazes the bent ground (the visible ground edge)."""
    best, bp = 90.0, None
    for th in np.linspace(1e-4, math.pi / 2, 4000):
        y, z = R * math.sin(th), R * math.cos(th) - R
        dep = math.degrees(math.atan2(h - z, y + d))
        if dep < best:
            best, bp = dep, (y, z)
    return math.hypot(bp[0] + d, h - bp[1])


def ground_distance_at_row(row, R, h, d):
    """Camera distance to the bent ground point seen on the centre column at a given row (0 = top)."""
    best = None
    for th in np.linspace(1e-4, math.pi / 2, 6000):
        y, z = R * math.sin(th), R * math.cos(th) - R
        r_ = row_of(math.degrees(math.atan2(h - z, y + d)))
        if best is None or abs(r_ - row) < best[0]:
            best = (abs(r_ - row), math.hypot(y + d, h - z))
        if r_ < HORIZON_ROW:
            break
    return best[1]


_FAR = tangent_distance(LOG_R, CAM_H, CAM_D)
# Haze: the ORIGINAL setting (user's choice): ground only, from halfway between the feet and the visible ground edge
# to the edge, mixed up to 0.8 toward #9FCE60. (The section 16.3 tuned variant reached a 2.12 far over near ratio.)
_NEAR = math.hypot(CAM_D, CAM_H)
HAZE_NEAR = _NEAR + 0.4 * (_FAR - _NEAR)          # original 0.5, starts slightly nearer (user: slightly more haze)
HAZE_FAR = _FAR
HAZE_MAX = 0.9                                         # original 0.8, nudged up slightly (user)


OBJECT_LEAN = 0.0      # shear back per unit height before the bend (0 here; the town hall scene measures 18 deg)


def bend_array(co):
    if OBJECT_LEAN:
        co[:, 1] = co[:, 1] + co[:, 2] * OBJECT_LEAN
    th = co[:, 1] / LOG_R
    r = LOG_R + co[:, 2]
    co[:, 1] = r * np.sin(th)
    co[:, 2] = r * np.cos(th) - LOG_R
    return co


def bend_point(p):
    return Vector(bend_array(np.array([[p.x, p.y, p.z]], dtype=float))[0])


def bend_world():
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            continue
        me = ob.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = bend_array(co.reshape(-1, 3) + np.array(ob.location)) - np.array(ob.location)
        me.vertices.foreach_set("co", co.ravel())
        me.update()


# --------------------------------------------------------------------------
# Textures (numpy, sRGB, low resolution, bilinear; NL felt grain on characters, section 5.3 and 12.2 item 10)
# --------------------------------------------------------------------------
RNG = np.random.default_rng(21)


def grain(img_rgb, amount=0.025):
    h, w = img_rgb.shape[:2]
    return np.clip(img_rgb + amount * (vnoise(w, h, w // 2, h // 2, RNG)[..., None] - 0.5) * 2, 0, 1)


def tex_grass():
    """Mottled base plus a printed lattice of soft lighter diamonds (retail capture: lighter yellow green patches on
    a jittered grid; press NL#949: a fine triangle print). 10 x 10 motifs per tile: one every 0.29 P."""
    w = h = 256
    n = 0.6 * fbm(w, h, 4, 3, RNG) + 0.4 * fbm(w, h, 20, 1, RNG)
    col = ramp(0.25 + 0.55 * n, [PAL["grass_dark"], PAL["grass"], PAL["grass"]])
    yy, xx = np.mgrid[0:h, 0:w]
    cell = 12
    nc = w // cell
    jit = RNG.uniform(-3.5, 3.5, (nc, nc, 2))
    print_mask = np.zeros((h, w))
    dark_mask = np.zeros((h, w))
    for i in range(nc):
        for j in range(nc):
            cx = (j * cell + (cell / 2 if i % 2 else 0) + cell / 2 + jit[i, j, 0]) % w
            cy = i * cell + cell / 2 + jit[i, j, 1]
            dx = np.minimum(np.abs(xx - cx), w - np.abs(xx - cx))
            dy = np.minimum(np.abs(yy - cy), h - np.abs(yy - cy))
            r = RNG.uniform(1.6, 2.6)
            d = np.sqrt(dx ** 2 + dy ** 2) / r       # round dot (16.3: spring and summer motif)
            print_mask = np.maximum(print_mask, smoothstep(1.2, 0.8, d) * RNG.uniform(0.45, 0.85))
            # small darker triangle beside it (fine print, press NL#949)
            tx, ty = (cx + cell / 2) % w, (cy + cell / 3) % h
            ddx = np.minimum(np.abs(xx - tx), w - np.abs(xx - tx))
            ddy = (yy - ty + h / 2) % h - h / 2
            tri = (ddy > -1.6) & (ddy < 1.6) & (ddx < (1.6 - ddy) * 0.9)
            dark_mask = np.maximum(dark_mask, tri * RNG.uniform(0.35, 0.6))
    light = hexc(PAL["grass_light"])
    dark = hexc(PAL["grass_print"])
    col = col * (1 - print_mask[..., None]) + light * print_mask[..., None]
    col = col * (1 - dark_mask[..., None]) + dark * dark_mask[..., None]
    fine = fbm(w, h, 40, 2, RNG) - 0.5                 # blade level mottle
    col = np.clip(col * (1 + 0.55 * fine[..., None]), 0, 1)
    col = grain(col, 0.10)
    return make_image("T_Grass", np.clip(col, 0, 1))


def tex_rock():
    w = h = 64
    n = 0.6 * fbm(w, h, 4, 3, RNG) + 0.4 * fbm(w, h, 16, 1, RNG)
    v = np.linspace(0, 1, h)[:, None]
    col = ramp(0.25 + 0.5 * n + 0.2 * (1 - v), ["#7E7590", PAL["rock"], "#C9C3D2"])
    yy, xx = np.mgrid[0:h, 0:w]
    for k in range(4):                                   # painted cracks: thin dark wandering lines
        x0, y0 = RNG.uniform(0, w), RNG.uniform(0, h)
        ang = RNG.uniform(0, math.pi)
        for t in np.linspace(0, 22, 60):
            ang += RNG.uniform(-0.25, 0.25)
            x0, y0 = (x0 + math.cos(ang) * 0.4) % w, (y0 + math.sin(ang) * 0.4) % h
            d = np.hypot(xx - x0, yy - y0)
            col = np.where((d < 0.9)[..., None], col * 0.72, col)
    return make_image("T_Rock", grain(col, 0.05))


def tex_leaves():
    """Leaf clump card: five serrated leaves fanning out, dark base to light tip, painted vein. The alpha is a
    smooth field so bilinear filtering plus the alpha test gives clean edges."""
    w = h = 96
    yy, xx = np.mgrid[0:h, 0:w] / (h - 1)
    col = np.zeros((h, w, 3))
    alpha = np.zeros((h, w))
    for k in range(5):
        ang = math.radians(-70 + 35 * k + RNG.uniform(-8, 8))
        dx, dy = math.sin(ang), -math.cos(ang)
        rx, ry = xx - 0.5, yy - 0.85
        along = rx * dx + ry * dy                    # 0 at the base, about 0.7 at the tip
        across = rx * dy - ry * dx
        L = 0.62 + RNG.uniform(-0.05, 0.05)
        t = np.clip(along / L, 0, 1)
        width = 0.15 * np.sin(np.pi * t) ** 0.8 * (0.85 + 0.15 * np.abs(np.sin(t * 30)))
        field = smoothstep(0.012, -0.012, np.abs(across) - width) * (along > 0) * (along < L)
        leafcol = ramp(0.15 + 0.75 * t + 0.1 * (RNG.random() - 0.5), [PAL["foliage_dark"], PAL["foliage"], PAL["foliage_light"]])
        vein = smoothstep(0.012, 0.004, np.abs(across)) * (t > 0.05) * (t < 0.9)
        leafcol = leafcol * (1 - 0.25 * vein[..., None])
        sel = field > alpha
        col[sel] = leafcol[sel]
        alpha = np.maximum(alpha, field)
    ALPHAS["leaves"] = alpha
    return make_image("T_Leaves", grain(col, 0.04), alpha)


ALPHAS = {}


def tex_flower(main, blotch):
    """Five rounded petals, two tone with a dark central blotch and a small yellow eye (generic garden flower)."""
    w = h = 64
    yy, xx = (np.mgrid[0:h, 0:w] - 31.5) / 31.5
    r = np.hypot(xx, yy)
    a = np.arctan2(yy, xx)
    petal_r = 0.62 + 0.30 * np.abs(np.cos(2.5 * a))
    alpha = smoothstep(0.03, -0.03, r - petal_r)
    t = np.clip(r / petal_r, 0, 1)
    col = ramp(t, [blotch, blotch, main, main])
    col = col * (0.85 + 0.15 * t)[..., None]
    eye = smoothstep(0.16, 0.10, r)
    col = col * (1 - eye[..., None]) + hexc("#F2C94C") * eye[..., None]
    lines = smoothstep(0.04, 0.0, np.abs(np.sin(5 * a))) * (t < 0.7) * (t > 0.2)
    col = col * (1 - 0.2 * lines[..., None])
    ALPHAS["flower"] = alpha
    return make_image(f"T_Flower_{main}", grain(col, 0.03), alpha)


def tex_flat(name, cols, amount=0.035, base=4):
    """Painted fill: gentle mottle plus felt grain (NL surfaces always carry texture, section 5.3)."""
    w = h = 32
    n = fbm(w, h, base, 2, RNG)
    return make_image(name, grain(ramp(0.3 + 0.4 * n, cols), amount))


def tex_wall():
    """Pale salmon wall with horizontal strata (section 4.2), lit at the lip."""
    w, h = 64, 64
    v = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
    strata = 0.5 + 0.5 * np.sin(2 * np.pi * (v * 5 + 0.15 * fbm(w, h, 3, 2, RNG)))
    t = 0.35 + 0.35 * strata + 0.3 * (1 - v)
    return make_image("T_Wall", ramp(t, [PAL["cliff_shade"], PAL["cliff_lit"], PAL["cliff_lit"]]))


def tex_caustic():
    """Cellular caustic web (section 4.3), soft edges."""
    w = h = 128
    k = 5
    rng = np.random.default_rng(9)
    pts = (rng.random((k * k, 2)) + np.stack(np.meshgrid(np.arange(k), np.arange(k)), -1).reshape(-1, 2)) / k
    y, x = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    x = x + 0.12 * (fbm(w, h, 2, 2, rng) - 0.5)
    y = y + 0.12 * (fbm(w, h, 2, 2, rng) - 0.5)
    d = []
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            p = pts + np.array([ox, oy])
            d.append(np.hypot(x[..., None] - p[:, 0], y[..., None] - p[:, 1]))
    d = np.sort(np.concatenate(d, -1), -1)
    web = smoothstep(0.05, 0.0, d[..., 1] - d[..., 0])
    return make_image("T_Caustic", np.dstack([web] * 3))


def tex_foliage():
    w = h = 64
    n = fbm(w, h, 4, 3, RNG)
    return make_image("T_Foliage", ramp(n, [PAL["foliage_dark"], PAL["foliage"], PAL["foliage_light"]]))


def tex_bark():
    """Painted bark (NL#946, 953 by eye): warm wood, vertical grain streaks that bend around one or two swirl knots
    (concentric rings, light centre), darker crevices. 96 x 192 so the knots stay readable on a fat trunk."""
    w, h = 96, 192
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    u, v = xx / w, yy / h
    knots = [(0.12, 0.62, 0.13, 0.08), (0.40, 0.30, 0.10, 0.06), (0.63, 0.70, 0.14, 0.08), (0.88, 0.40, 0.09, 0.06)]
    warp = np.zeros((h, w))
    ring = np.zeros((h, w))
    for cx, cy, rx, ry in knots:
        du = (u - cx + 0.5) % 1.0 - 0.5
        d = np.sqrt((du / rx) ** 2 + ((v - cy) / ry) ** 2)
        warp += np.exp(-d * 1.2) * np.sign(du) * 0.06
        inside = d < 1.6
        ring = np.where(inside, np.maximum(ring, (0.5 + 0.5 * np.cos(d * 9.0)) * smoothstep(1.6, 0.6, d)), ring)
    streak = 0.5 + 0.5 * np.sin(2 * np.pi * ((u + warp) * 9 + 0.6 * fbm(w, h, 3, 2, RNG)))
    crevice = smoothstep(0.12, 0.0, np.abs(((u + warp) * 9 + 0.6 * fbm(w, h, 3, 2, RNG)) % 1.0 - 0.5) - 0.38)
    t = 0.45 + 0.22 * (streak - 0.5) + 0.42 * (ring - 0.3) - 0.35 * crevice + 0.1 * (fbm(w, h, 8, 2, RNG) - 0.5)
    col = ramp(t, [PAL["wood_dark"], PAL["wood"], PAL["wood_light"]])
    return make_image("T_Bark", grain(col, 0.04))


def tex_face():
    """Head texture with the face as a decal (section 5.3): u around the head (0.5 = facing direction),
    v bottom to top. Eyes: black ovals with a white glint; blush under the outer corners; small mouth."""
    w, h = 256, 128
    col = np.ones((h, w, 3)) * hexc(PAL["skin"])
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = xx / w, 1 - yy / h
    for side in (-1, 1):
        cu, cv = 0.5 + side * 0.075, 0.47
        e = ((u - cu) / 0.022) ** 2 + ((v - cv) / 0.075) ** 2
        col[e < 1] = hexc(PAL["eye"])
        g = ((u - cu + 0.006) / 0.008) ** 2 + ((v - cv - 0.03) / 0.022) ** 2
        col[g < 1] = 1.0
        b = np.sqrt(((u - (0.5 + side * 0.115)) / 0.045) ** 2 + ((v - 0.36) / 0.05) ** 2)
        k = smoothstep(1.0, 0.3, b)[..., None] * 0.45                     # soft pink cheeks (retail captures)
        col = col * (1 - k) + hexc(PAL["blush"]) * k
    m = ((u - 0.5) / 0.022) ** 2 + ((v - 0.33) / 0.012) ** 2
    col[m < 1] = hexc("#B0503C")
    return make_image("T_Face", grain(col, 0.02))


def tex_hair():
    """Soft painted hair: low contrast strand mottle (no pole stripes) and a lighter sheen band (by eye)."""
    w, h = 64, 64
    n = 0.6 * fbm(w, h, 6, 2, RNG) + 0.4 * fbm(w, h, 24, 1, RNG)
    v = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
    sheen = np.exp(-((v - 0.72) / 0.08) ** 2)
    col = ramp(0.35 + 0.3 * n + 0.35 * sheen, [PAL["hair"], PAL["hair"], PAL["hair_light"]])
    return make_image("T_Hair", grain(col, 0.03))


def tex_shirt():
    """32x128 strip, front, back, sleeve, sleeve (section 6), 16 colour limit, simple original star motif."""
    w, h = 32, 128
    col = np.ones((h, w, 3)) * hexc(PAL["shirt"])
    yy, xx = np.mgrid[0:h, 0:w]
    for blk in (0, 1):                      # front and back: a pale star
        cy = blk * 32 + 16
        a = np.arctan2(yy - cy, xx - 16)
        r = np.hypot(yy - cy, xx - 16)
        star = r < 7 * (0.6 + 0.4 * np.cos(5 * a))
        col[star] = hexc(PAL["shirt_light"])
    stripe = ((yy >= 64) & ((yy // 4) % 2 == 0))
    col[stripe] = hexc(PAL["shirt_light"])
    return make_image("T_Shirt", grain(col, 0.02))


def tex_bobber():
    w, h = 16, 16
    v = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
    return make_image("T_Bobber", np.where((v < 0.5)[..., None], hexc(PAL["bobber"]), hexc("#F7F3EA")))


# --------------------------------------------------------------------------
# Materials: emission x LUT(N.L) (section 12.2 item 4), NL rim on characters (item 6)
# --------------------------------------------------------------------------
LUT = None


def lut_ramp_stops():
    sh = tuple(c ** 2.2 for c in LUT_SHADOW_SRGB)
    li = tuple(c ** 2.2 for c in LUT_LIGHT_SRGB)
    scale = max(li)
    stops = [(0.00, sh), (0.08, sh), (0.38, (1, 1, 1)), (0.965, (1, 1, 1)), (1.00, li)]
    return [(p, tuple(c / scale for c in col)) for p, col in stops], scale


def build_lut():
    ng = bpy.data.node_groups.new("LUT_NL", "ShaderNodeTree")
    ng.interface.new_socket("Color", in_out="INPUT", socket_type="NodeSocketColor")
    ng.interface.new_socket("Rim", in_out="INPUT", socket_type="NodeSocketFloat")
    ng.interface.new_socket("Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
    geo = ng.nodes.new("ShaderNodeNewGeometry")
    dot = ng.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    link(ng, geo.outputs["Normal"], dot.inputs[0])
    dot.inputs[1].default_value = tuple(SUN)
    r = ng.nodes.new("ShaderNodeValToRGB")
    r.color_ramp.interpolation = "LINEAR"          # smooth, never constant (section 2)
    link(ng, dot.outputs["Value"], r.inputs[0])
    stops, scale = lut_ramp_stops()
    els = r.color_ramp.elements
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
    sc.inputs[3].default_value = scale
    link(ng, mul.outputs[2], sc.inputs[0])
    # rim: Layer Weight facing, narrow band at the silhouette, lilac tinted gain (characters only, Rim input 1)
    lw = ng.nodes.new("ShaderNodeLayerWeight")
    lw.inputs["Blend"].default_value = 0.5
    mr = ng.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    link(ng, lw.outputs["Facing"], mr.inputs["Value"])
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.78, 0.95
    rm = ng.nodes.new("ShaderNodeMath")
    rm.operation = "MULTIPLY"
    link(ng, mr.outputs["Result"], rm.inputs[0])
    link(ng, gi.outputs[1], rm.inputs[1])
    tint = ng.nodes.new("ShaderNodeMix")
    tint.data_type, tint.blend_type = "RGBA", "MULTIPLY"
    tint.inputs[0].default_value = 1.0
    link(ng, sc.outputs["Vector"], tint.inputs[6])
    tint.inputs[7].default_value = (*(c * RIM_LINEAR_GAIN for c in lin(hexc(RIM_TINT))), 1)
    add = ng.nodes.new("ShaderNodeVectorMath")
    add.operation = "MULTIPLY_ADD"
    link(ng, tint.outputs[2], add.inputs[0])
    link(ng, rm.outputs[0], add.inputs[1])
    link(ng, sc.outputs["Vector"], add.inputs[2])
    # haze (16.3): distance mix toward the far ground colour; applies to everything lit (far trees too)
    cdn = ng.nodes.new("ShaderNodeCameraData")
    hzm = ng.nodes.new("ShaderNodeMapRange")
    hzm.interpolation_type = "SMOOTHSTEP"
    link(ng, cdn.outputs["View Distance"], hzm.inputs["Value"])
    hzm.inputs["From Min"].default_value, hzm.inputs["From Max"].default_value = HAZE_NEAR, HAZE_FAR
    # 16.3 measures haze on far GROUND and far water only; on objects it washed far trees out to flat lime, which the
    # captures do not show, so the lit group keeps the node but at zero (terrain and water carry the haze)
    hzm.inputs["To Min"].default_value, hzm.inputs["To Max"].default_value = 0.0, 0.0
    hmix = ng.nodes.new("ShaderNodeMix")
    hmix.data_type = "RGBA"
    link(ng, hzm.outputs["Result"], hmix.inputs[0])
    link(ng, add.outputs["Vector"], hmix.inputs[6])
    hmix.inputs[7].default_value = (*lin(hexc(PAL["haze"])), 1)
    gr = ng.nodes.new("ShaderNodeVectorMath")
    gr.operation = "MULTIPLY"
    link(ng, hmix.outputs[2], gr.inputs[0])
    gr.inputs[1].default_value = tuple(float(c) ** 2.2 for c in gdisp())
    link(ng, gr.outputs["Vector"], go.inputs[0])
    return ng


def mat(name, color=None, image=None, rim=0.0, cutout=False, uv_scale=None, uv_name=None):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    alpha = None
    if image is not None:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image, t.interpolation = image, "Linear"
        vec = None
        if uv_name:
            uvn = nt.nodes.new("ShaderNodeUVMap")
            uvn.uv_map = uv_name
            vec = uvn.outputs["UV"]
        if uv_scale:
            if vec is None:
                vec = nt.nodes.new("ShaderNodeTexCoord").outputs["UV"]
            mp = nt.nodes.new("ShaderNodeMapping")
            mp.inputs["Scale"].default_value = (*uv_scale, 1)
            link(nt, vec, mp.inputs["Vector"])
            vec = mp.outputs["Vector"]
        if vec is not None:
            link(nt, vec, t.inputs["Vector"])
        col, alpha = t.outputs["Color"], t.outputs["Alpha"]
    else:
        rgb = nt.nodes.new("ShaderNodeRGB")
        rgb.outputs[0].default_value = (*lin(hexc(color)), 1)
        col = rgb.outputs[0]
    g = nt.nodes.new("ShaderNodeGroup")
    g.node_tree = LUT
    link(nt, col, g.inputs[0])
    g.inputs[1].default_value = rim
    em = nt.nodes.new("ShaderNodeEmission")
    link(nt, g.outputs[0], em.inputs["Color"])
    shader = em.outputs[0]
    if cutout and alpha is not None:
        gt = nt.nodes.new("ShaderNodeMath")
        gt.operation = "GREATER_THAN"
        gt.inputs[1].default_value = 0.5
        link(nt, alpha, gt.inputs[0])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mx = nt.nodes.new("ShaderNodeMixShader")
        link(nt, gt.outputs[0], mx.inputs[0])
        link(nt, tr.outputs[0], mx.inputs[1])
        link(nt, shader, mx.inputs[2])
        shader = mx.outputs[0]
        m.surface_render_method = "DITHERED"
    link(nt, shader, out.inputs["Surface"])
    return m


def unlit(name, color):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*lin(gcol(color)), 1)
    link(nt, em.outputs[0], out.inputs["Surface"])
    return m


def shadow_mat(name, key, inner=0.2, wobble=0.0):
    """Multiply decal: Transparent BSDF tinted by mix(white, core multiply, profile). Ratios are converted from
    displayed (sRGB) to linear with the exact curve on a mid ground value, then raised as needed."""
    if SHADOW_MODE != "multiply":
        alpha = {"blob": BLOB_CORE_ALPHA, "contact": CONTACT_ALPHA, "canopy": CANOPY_ALPHA}[key]
        return blob_mat(name, alpha, BLOB_TINT_LIN, inner=max(inner, 0.45), wobble=wobble)
    mul_lin = tuple(k ** 2.2 for k in SHADOW_MUL[key])
    m = blob_mat(name, 1.0, (0, 0, 0), inner=inner, wobble=wobble)
    nt = m.node_tree
    mr = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeMapRange" and n.inputs["To Max"].default_value == 1.0)
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    for n in [n for n in nt.nodes if n.bl_idname in ("ShaderNodeBsdfTransparent", "ShaderNodeEmission", "ShaderNodeMixShader")]:
        nt.nodes.remove(n)
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    link(nt, mr.outputs["Result"], mx.inputs[0])
    mx.inputs[6].default_value = (1, 1, 1, 1)
    mx.inputs[7].default_value = (*mul_lin, 1)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    link(nt, mx.outputs[2], tr.inputs["Color"])
    link(nt, tr.outputs[0], out.inputs["Surface"])
    return m


def blob_mat(name, core_alpha, tint_lin, inner=0.76, wobble=0.0):
    """Section 3: blend toward a dark blue grey (a multiply cannot raise blue on lime grass). Alpha profile:
    full core, falling to 0 at the outer radius, 50% at about 0.88 of it (w50 / extent = 0.38 / 0.43)."""
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    link(nt, tc.outputs["UV"], sub.inputs[0])
    sub.inputs[1].default_value = (0.5, 0.5, 0.0)
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    link(nt, sub.outputs["Vector"], ln.inputs[0])
    dist = ln.outputs["Value"]
    if wobble:
        # irregular, leafy outline (NL#946 canopy and flower shadows are soft but not round)
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 4.0
        nz.inputs["Detail"].default_value = 2.0
        link(nt, tc.outputs["UV"], nz.inputs["Vector"])
        ad = nt.nodes.new("ShaderNodeMath")
        ad.operation = "MULTIPLY_ADD"
        link(nt, nz.outputs["Fac"], ad.inputs[0])
        ad.inputs[1].default_value = wobble
        link(nt, dist, ad.inputs[2])
        sb = nt.nodes.new("ShaderNodeMath")
        sb.operation = "SUBTRACT"
        link(nt, ad.outputs[0], sb.inputs[0])
        sb.inputs[1].default_value = 0.5 * wobble
        dist = sb.outputs[0]
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    link(nt, dist, mr.inputs["Value"])
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.5, 0.5 * inner
    mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.0, core_alpha
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*tint_lin, 1)
    mx = nt.nodes.new("ShaderNodeMixShader")
    link(nt, mr.outputs["Result"], mx.inputs[0])
    link(nt, tr.outputs[0], mx.inputs[1])
    link(nt, em.outputs[0], mx.inputs[2])
    link(nt, mx.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "BLENDED"
    return m


def terrain_mat(T):
    """Grass and wall blended softly through a per vertex weight plus noise (NL boundaries are soft, 4.2)."""
    m = bpy.data.materials.new("M_Terrain")
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")

    def tex(img, uvname):
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = uvname
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image, t.interpolation = img, "Linear"
        link(nt, uvn.outputs["UV"], t.inputs["Vector"])
        return t.outputs["Color"]
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "wall"
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 2.5
    link(nt, nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], nz.inputs["Vector"])
    ad = nt.nodes.new("ShaderNodeMath")
    ad.operation = "MULTIPLY_ADD"
    link(nt, nz.outputs["Fac"], ad.inputs[0])
    ad.inputs[1].default_value = 0.3
    link(nt, at.outputs["Fac"], ad.inputs[2])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    link(nt, ad.outputs[0], mr.inputs["Value"])
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = 0.42, 0.62
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    link(nt, mr.outputs["Result"], mx.inputs[0])
    link(nt, tex(T["grass"], "UVMap"), mx.inputs[6])
    link(nt, tex(T["wall"], "UVWall"), mx.inputs[7])
    g = nt.nodes.new("ShaderNodeGroup")
    g.node_tree = LUT
    link(nt, mx.outputs[2], g.inputs[0])
    # haze: far 40 to 50% of the visible ground drifts toward a light yellow green (section 2; the user's retail
    # capture shows the same drift, far ground #9FCE60), mixed up to 0.8 by camera distance
    cd = nt.nodes.new("ShaderNodeCameraData")
    hz = nt.nodes.new("ShaderNodeMapRange")
    hz.interpolation_type = "SMOOTHSTEP"
    link(nt, cd.outputs["View Distance"], hz.inputs["Value"])
    hz.inputs["From Min"].default_value, hz.inputs["From Max"].default_value = HAZE_NEAR, HAZE_FAR
    hz.inputs["To Min"].default_value, hz.inputs["To Max"].default_value = 0.0, HAZE_MAX
    hm = nt.nodes.new("ShaderNodeMix")
    hm.data_type = "RGBA"
    link(nt, hz.outputs["Result"], hm.inputs[0])
    link(nt, g.outputs[0], hm.inputs[6])
    hm.inputs[7].default_value = (*lin(gcol(PAL["haze"])), 1)

    # below the water line: first a pale blue grey cast, then the deep water colour by depth (NL#957)
    ua = nt.nodes.new("ShaderNodeAttribute")
    ua.attribute_name = "under"
    wet = nt.nodes.new("ShaderNodeMath")
    wet.operation = "GREATER_THAN"
    link(nt, ua.outputs["Fac"], wet.inputs[0])
    wet.inputs[1].default_value = 0.001
    c1 = nt.nodes.new("ShaderNodeMix")
    c1.data_type = "RGBA"
    k1 = nt.nodes.new("ShaderNodeMath")
    k1.operation = "MULTIPLY"
    link(nt, wet.outputs[0], k1.inputs[0])
    k1.inputs[1].default_value = 0.45
    link(nt, k1.outputs[0], c1.inputs[0])
    link(nt, hm.outputs[2], c1.inputs[6])
    c1.inputs[7].default_value = (*lin(gcol(PAL["underwater"])), 1)
    dp = nt.nodes.new("ShaderNodeMapRange")
    dp.interpolation_type = "SMOOTHSTEP"
    link(nt, ua.outputs["Fac"], dp.inputs["Value"])
    dp.inputs["From Min"].default_value, dp.inputs["From Max"].default_value = 0.0, UNDER_FADE_DEPTH
    c2 = nt.nodes.new("ShaderNodeMix")
    c2.data_type = "RGBA"
    link(nt, dp.outputs["Result"], c2.inputs[0])
    link(nt, c1.outputs[2], c2.inputs[6])
    c2.inputs[7].default_value = (*lin(gcol(PAL["water_deep"])), 1)
    em = nt.nodes.new("ShaderNodeEmission")
    link(nt, c2.outputs[2], em.inputs["Color"])
    link(nt, em.outputs[0], out.inputs["Surface"])
    return m


def water_mat(T):
    """Flat lit water: shallow colour near the walls, NL water mid colour, caustic web (section 4.3)."""
    m = bpy.data.materials.new("M_Water")
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "edge"
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    link(nt, at.outputs["Fac"], mx.inputs[0])
    wb, ws = WATER_BY_TIME.get(TIME, (None, None))
    mx.inputs[6].default_value = (*lin(hexc(wb) if wb else gcol(PAL["water"])), 1)
    mx.inputs[7].default_value = (*lin(hexc(ws) if ws else gcol(PAL["water_shallow"])), 1)
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.9, 0.65, 1)
    link(nt, nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], mp.inputs["Vector"])
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image, t.interpolation = T["caustic"], "Linear"
    link(nt, mp.outputs["Vector"], t.inputs["Vector"])
    cm = nt.nodes.new("ShaderNodeMath")
    cm.operation = "MULTIPLY"
    link(nt, t.outputs["Color"], cm.inputs[0])
    cm.inputs[1].default_value = WAVE_GAIN       # rivers: lines +28% at p95, +47% at p99 over the median (16.4)
    mx2 = nt.nodes.new("ShaderNodeMix")
    mx2.data_type = "RGBA"
    link(nt, cm.outputs[0], mx2.inputs[0])
    link(nt, mx.outputs[2], mx2.inputs[6])
    mx2.inputs[7].default_value = (*lin(hexc(ws) if ws else gcol(PAL["water_shallow"])), 1)
    wcd = nt.nodes.new("ShaderNodeCameraData")
    whz = nt.nodes.new("ShaderNodeMapRange")
    whz.interpolation_type = "SMOOTHSTEP"
    link(nt, wcd.outputs["View Distance"], whz.inputs["Value"])
    whz.inputs["From Min"].default_value, whz.inputs["From Max"].default_value = HAZE_NEAR, HAZE_FAR
    whz.inputs["To Min"].default_value, whz.inputs["To Max"].default_value = 0.0, 0.0   # original: no haze on water
    wmx = nt.nodes.new("ShaderNodeMix")
    wmx.data_type = "RGBA"
    link(nt, whz.outputs["Result"], wmx.inputs[0])
    link(nt, mx2.outputs[2], wmx.inputs[6])
    wmx.inputs[7].default_value = (*lin(hexc(WATER_BY_TIME[TIME][1]) if TIME in WATER_BY_TIME else gcol("#95B3F2")), 1)
    em = nt.nodes.new("ShaderNodeEmission")
    link(nt, wmx.outputs[2], em.inputs["Color"])
    # translucent surface (section 4.3: NL water shows posts and tyres below the line, NL#957, 967). What lies
    # underneath is darkened by depth in its own material (terrain "under" attribute), so the surface alpha can
    # stay low: more see through at the banks, denser in the middle.
    al = nt.nodes.new("ShaderNodeMapRange")
    link(nt, at.outputs["Fac"], al.inputs["Value"])
    al.inputs["To Min"].default_value, al.inputs["To Max"].default_value = WATER_ALPHA_MID, WATER_ALPHA_EDGE
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mxs = nt.nodes.new("ShaderNodeMixShader")
    link(nt, al.outputs["Result"], mxs.inputs[0])
    link(nt, tr.outputs[0], mxs.inputs[1])
    link(nt, em.outputs[0], mxs.inputs[2])
    link(nt, mxs.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "BLENDED"
    return m


WATER_ALPHA_EDGE, WATER_ALPHA_MID = 0.45, 0.70        # E tuned on the render
GRASS_UV = 0.55                                       # tiles per P: print every 0.085 P, tuned on the grain QA (16.8)
WAVE_GAIN = 0.85                                     # tuned on the render to the 16.4 river line ratio
UNDER_FADE_DEPTH = 0.25                              # E: NL#957 posts fade out within a short depth (about 0.2 P)


# --------------------------------------------------------------------------
# Scene parts
# --------------------------------------------------------------------------
def build_terrain(col, M):
    xs = np.concatenate([np.arange(-16, -3, 0.5), np.arange(-3, 7.5, 0.08), np.arange(7.5, 16.01, 0.5)])
    ys = np.concatenate([np.arange(-6, 12, 0.12), np.arange(12, 34.01, 0.5)])
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    uvw = bm.loops.layers.uv.new("UVWall")
    wl = bm.verts.layers.float.new("wall")
    ud = bm.verts.layers.float.new("under")
    grid = []
    for y in ys:
        row = []
        for x in xs:
            z = terrain_h(x, y)
            v = bm.verts.new((x, y, z))
            v[wl] = float(smoothstep(-0.03, -0.12, z))
            v[ud] = float(max(0.0, WATER_Z - z))
            row.append(v)
        grid.append(row)
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            f = bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
            for lp in f.loops:
                c = lp.vert.co
                lp[uv].uv = (c.x * GRASS_UV, c.y * GRASS_UV)
                lp[uvw].uv = (c.y * 0.5, c.z * 1.4)
    return mesh_object("Terrain", bm, col, [M["terrain"]])


def build_water(col, M):
    bm = bmesh.new()
    bm.loops.layers.uv.new("UVMap")
    ed = bm.verts.layers.float.new("edge")
    ys = np.arange(-6, 34.01, 0.25)
    rows = []
    for y in ys:
        xa, xb = bank_x(y) + 0.10, bank_x(y) + RIVER_W - 0.10
        r = []
        for k in range(17):
            x = xa + (xb - xa) * k / 16
            v = bm.verts.new((x, y, WATER_Z))
            dist = min(x - xa, xb - x)
            v[ed] = float(smoothstep(0.45, 0.0, dist))
            r.append(v)
        rows.append(r)
    for j in range(len(ys) - 1):
        for i in range(16):
            bm.faces.new((rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]))
    # origin far up river: EEVEE sorts blended objects by origin distance, so the water is drawn before the fish
    # silhouette, which then reads clearly through it
    o = Vector((2.8, 24.0, WATER_Z))
    for v in bm.verts:
        v.co -= o
    return mesh_object("Water", bm, col, [M["water"]], smooth=False, loc=o)


def leaf_cards(bm, uv, center, radius, n, size, rng, squash=0.9):
    """Leaf clump cards over a sphere (Fibonacci points), facing outward, randomly rolled. Their normals follow the
    sphere, so the LUT shades the canopy as a volume while the cut edges stay jagged."""
    c = Vector(center)
    ga = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        zz = 1 - 2 * (i + 0.5) / n
        rr = math.sqrt(max(0.0, 1 - zz * zz))
        nrm = Vector((rr * math.cos(ga * i), rr * math.sin(ga * i), zz))
        if nrm.z < -0.55:
            continue                                 # underside: hidden, skip
        p = c + Vector((nrm.x, nrm.y, nrm.z * squash)) * radius
        u, w = B.frame_of(nrm)
        roll = rng.uniform(0, 2 * math.pi)
        a1 = u * math.cos(roll) + w * math.sin(roll)
        a2 = nrm.cross(a1)
        sz = size * rng.uniform(0.85, 1.15)
        vs = [bm.verts.new(p + (a1 * sx + a2 * sy) * sz * 0.5 + nrm * 0.03 * sz)
              for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        f = bm.faces.new(vs)
        for lp, q in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
            lp[uv].uv = q


def raster_footprint(objs, cx, cy, half, res=160):
    """Project meshes straight down onto the ground and rasterise their footprint (row 0 = min y). Cards use their
    texture alpha (binary test at 0.5) so leaf and petal cut outs shape the shadow."""
    mask = np.zeros((res, res))
    for ob, alpha in objs:
        me = ob.data
        co = np.array([v.co[:] for v in me.vertices]) + np.array(ob.location)
        uvl = me.uv_layers.active if alpha is not None else None
        for poly in me.polygons:
            vi = list(poly.vertices)
            li = list(poly.loop_indices)
            for t in range(1, len(vi) - 1):
                k3 = (0, t, t + 1)
                P = co[[vi[k] for k in k3]]
                px_ = (P[:, 0] - (cx - half)) / (2 * half) * res
                py_ = (P[:, 1] - (cy - half)) / (2 * half) * res
                x0, x1 = max(0, int(np.floor(px_.min()))), min(res - 1, int(np.ceil(px_.max())))
                y0, y1 = max(0, int(np.floor(py_.min()))), min(res - 1, int(np.ceil(py_.max())))
                if x1 < x0 or y1 < y0:
                    continue
                d = (py_[1] - py_[2]) * (px_[0] - px_[2]) + (px_[2] - px_[1]) * (py_[0] - py_[2])
                if abs(d) < 1e-9:
                    continue
                gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
                l0 = ((py_[1] - py_[2]) * (gx - px_[2]) + (px_[2] - px_[1]) * (gy - py_[2])) / d
                l1 = ((py_[2] - py_[0]) * (gx - px_[2]) + (px_[0] - px_[2]) * (gy - py_[2])) / d
                l2 = 1 - l0 - l1
                inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
                if uvl is not None:
                    UV = np.array([uvl.data[li[k]].uv[:] for k in k3])
                    u = l0 * UV[0, 0] + l1 * UV[1, 0] + l2 * UV[2, 0]
                    v = l0 * UV[0, 1] + l1 * UV[1, 1] + l2 * UV[2, 1]
                    ah, aw = alpha.shape
                    ai = np.clip(((1 - v) * ah).astype(int), 0, ah - 1)       # numpy alpha rows run top down
                    aj = np.clip((u * aw).astype(int), 0, aw - 1)
                    inside &= alpha[ai, aj] > 0.5
                sub = mask[y0:y1 + 1, x0:x1 + 1]
                mask[y0:y1 + 1, x0:x1 + 1] = np.maximum(sub, inside)
    return mask


def soften(mask, grow=3, blur=3):
    """Grow the footprint a little (so it shows around the object from the high camera) and soften its edge."""
    m = mask.copy()
    for _ in range(grow):
        p = np.pad(m, 1)
        m = np.maximum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
    for _ in range(blur):
        p = np.pad(m, 1, mode="edge")
        m = sum(p[i:i + m.shape[0], j:j + m.shape[1]] for i in range(3) for j in range(3)) / 9
    return m


PROFILE_SHADOWS = []          # (name, centre x, centre y, half size, mask) for QA


def profile_shadow(col, name, objs, cx, cy, z, half, key, core=0.0):
    """Shadow decal shaped like the object's top down silhouette (trees, bushes, flowers: 'they tend to match the
    profile of the feature'). Multiply decal like the blobs (section 3 and 15), optionally darker toward the centre."""
    res = 160
    m = soften(raster_footprint(objs, cx, cy, half, res))
    if core:
        yy, xx = (np.mgrid[0:res, 0:res] + 0.5) / res * 2 - 1
        m = m * (1 - core + core * np.clip(1 - np.hypot(xx, yy), 0, 1))
    PROFILE_SHADOWS.append((name, cx, cy, half, m))
    img = make_image(f"T_{name}", np.ones((res, res, 3)), m[::-1])
    mat = bpy.data.materials.new(f"M_{name}")
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image, t.interpolation, t.extension = img, "Linear", "CLIP"
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    link(nt, t.outputs["Alpha"], mx.inputs[0])
    mx.inputs[6].default_value = (1, 1, 1, 1)
    mx.inputs[7].default_value = (*(c ** 2.2 for c in SHADOW_MUL[key]), 1)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    link(nt, mx.outputs[2], tr.inputs["Color"])
    link(nt, tr.outputs[0], out.inputs["Surface"])
    mat.surface_render_method = "BLENDED"
    # a 16 x 16 grid, not one quad: the rolling log bend would sink a large flat quad's middle under the ground
    bm, uv = new_bm()
    n = 16
    grid = [[bm.verts.new((cx - half + 2 * half * i / n, cy - half + 2 * half * j / n, z)) for i in range(n + 1)]
            for j in range(n + 1)]
    for j in range(n):
        for i in range(n):
            f = bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
            for lp, q in zip(f.loops, ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))):
                lp[uv].uv = (q[0] / n, q[1] / n)
    return mesh_object(name, bm, col, [mat], smooth=False)


def build_tree(col, M, x, y, height, rng):
    z0 = terrain_h(x, y)
    bm, uv = new_bm()
    # NL trees (NL#946, 953, by eye): fat flared trunk, short, canopy starting low and wide
    top = z0 + height * 0.42
    bm_tube(bm, uv, [(x, y, z0 - 0.1), (x, y, z0 + 0.10), (x, y, top)], [0.30, 0.19, 0.14], 9, cap_end=True)
    mesh_object("TreeTrunk", bm, col, [M["bark"]])
    lobes = []
    for k in range(4):
        a = 2 * math.pi * k / 4 + rng.uniform(0, 1)
        lobes.append(((x + 0.30 * math.cos(a), y + 0.30 * math.sin(a), top + 0.32 + 0.06 * k), 0.46))
    lobes.append(((x, y, top + 0.72), 0.40))
    bm, uv = new_bm()
    for c, r in lobes:
        bm_sphere(bm, uv, c, r * 0.82, scale=(1, 1, 0.9), segs=12, rings=7)        # dark core fills gaps
    core_ob = mesh_object("TreeCore", bm, col, [M["foliage_core"]])
    bm, uv = new_bm()
    for c, r in lobes:
        leaf_cards(bm, uv, c, r, 46, 0.36, rng)
    leaves_ob = mesh_object("TreeLeaves", bm, col, [M["leaves"]])
    # canopy shadow follows the canopy outline seen from above (leaf cut outs included), darker toward the trunk
    profile_shadow(col, "TreeShadow", [(core_ob, None), (leaves_ob, ALPHAS["leaves"])], x, y, z0 + 0.012, 1.1,
                   "canopy", core=0.35)
    blob(col, M["contact"], "TrunkContact", (x, y, z0 + 0.016), 0.40, 0.36)


def build_flowers(col, M, x, y, rng):
    """A clump about 0.45 P wide: a rosette of leaf cards with painted flower head cards on short stems."""
    z0 = terrain_h(x, y)
    bm, uv = new_bm()
    for k in range(9):                                   # rosette leaves, tilted up 25 deg
        a = 2 * math.pi * k / 9 + rng.uniform(-0.2, 0.2)
        d = Vector((math.cos(a), math.sin(a), 0))
        up = Vector((0, 0, 1))
        base = Vector((x, y, z0 + 0.01)) + d * 0.03
        tip_dir = (d * math.cos(0.45) + up * math.sin(0.45)).normalized()
        side = d.cross(up).normalized()
        L, W = 0.20 * rng.uniform(0.85, 1.1), 0.12
        vs = [bm.verts.new(base - side * W / 2), bm.verts.new(base + side * W / 2),
              bm.verts.new(base + side * W / 2 + tip_dir * L), bm.verts.new(base - side * W / 2 + tip_dir * L)]
        f = bm.faces.new(vs)
        for lp, q in zip(f.loops, ((0.3, 0.0), (0.7, 0.0), (0.95, 0.75), (0.05, 0.75))):
            lp[uv].uv = q
    parts = [(mesh_object("FlowerLeaves", bm, col, [M["leaves"]]), ALPHAS["leaves"])]
    for k in range(4):
        a = 2 * math.pi * k / 4 + rng.uniform(0, 0.5)
        px, py = x + 0.11 * math.cos(a), y + 0.09 * math.sin(a)
        hgt = 0.10 + 0.05 * rng.random()
        bm, uv = new_bm()
        bm_tube(bm, uv, [(px, py, z0), (px, py, z0 + hgt)], [0.010, 0.008], 4)
        mesh_object("FlowerStem", bm, col, [M["stem"]])
        bm, uv = new_bm()
        # head card faces up and toward the camera (south), tilted 40 deg from flat
        nrm = Vector((0, -math.sin(math.radians(40)), math.cos(math.radians(40))))
        a1 = Vector((1, 0, 0))
        a2 = nrm.cross(a1)
        c = Vector((px, py, z0 + hgt + 0.01))
        sz = 0.13
        vs = [bm.verts.new(c + (a1 * sx + a2 * sy) * sz * 0.5) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        f = bm.faces.new(vs)
        for lp, q in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
            lp[uv].uv = q
        parts.append((mesh_object("FlowerHead", bm, col, [M["flower_a"] if k % 2 else M["flower_b"]]), ALPHAS["flower"]))
    profile_shadow(col, "FlowerShadow", parts, x, y, z0 + 0.01, 0.38, "contact")


def build_rock(col, M, x, y, s):
    z0 = terrain_h(x, y)
    bm, uv = new_bm()
    res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=s,
                                     matrix=Matrix.Translation((x, y, z0 + 0.35 * s)) @
                                     Matrix.Diagonal((1.25, 1.0, 0.75, 1.0)), calc_uvs=True)
    rng = np.random.default_rng(4)
    for v in res["verts"]:
        v.co += Vector(rng.uniform(-0.05, 0.05, 3)) * s
    for f in bm.faces:
        for lp in f.loops:
            lp[uv].uv = ((lp.vert.co.x - x) * 2.0, (lp.vert.co.z - z0) * 2.0 + (lp.vert.co.y - y))
    mesh_object("Rock", bm, col, [M["rock"]], smooth=False)       # facets visible on props (section 2)
    blob(col, M["contact"], "RockShadow", (x, y - 0.02, z0 + 0.01), 1.75 * s, 1.45 * s)


def build_bush(col, M, x, y, r, rng):
    z0 = terrain_h(x, y)
    lobes = [((x + 0.45 * r * math.cos(a), y + 0.35 * r * math.sin(a), z0 + 0.55 * r), 0.6 * r)
             for a in (2 * math.pi * k / 5 for k in range(5))]
    lobes.append(((x, y, z0 + 0.9 * r), 0.6 * r))
    bm, uv = new_bm()
    for c, rr in lobes:
        bm_sphere(bm, uv, c, rr * 0.82, segs=12, rings=7)
    core_ob = mesh_object("BushCore", bm, col, [M["foliage_core"]])
    bm, uv = new_bm()
    for c, rr in lobes:
        leaf_cards(bm, uv, c, rr, 40, 0.30, rng)
    leaves_ob = mesh_object("BushLeaves", bm, col, [M["leaves"]])
    profile_shadow(col, "BushShadow", [(core_ob, None), (leaves_ob, ALPHAS["leaves"])], x, y, z0 + 0.012, 1.4 * r,
                   "canopy", core=0.35)


# Horizon tree line beyond the bend tangent so trunk bases sink (section 9 rule 3); 5 trees (rule 5: 4 to 6).
TREES = [(-3.6, 11.5, 2.0), (-1.4, 12.6, 2.1), (0.5, 11.4, 1.9), (5.6, 11.8, 2.0), (-5.6, 12.4, 2.1)]
NEAR_TREES = [(-2.75, 2.5, 1.8)]   # mid ground tree: its canopy shadow sits on visible ground
FLOWERS = [(-1.25, 1.35), (-2.0, -0.15), (0.75, 2.6)]   # rule 5: 2 to 5 clumps, 0.4 to 0.5 P wide

PLAYER = Vector((0.0, 0.0, 0.0))
FACING_DEG = -25.0            # east, turned 25 deg toward the camera so the face reads (design choice)


def build_player(col, M, pos=None, facing_deg=None, fishing=True):
    """Original NL proportioned player, H = 1.0: head fraction 0.46 (chin at 0.54), round forms 14 x 8, limbs
    6 sided, mitten hands, separate hair shell with a zigzag fringe (sections 5.1 to 5.3)."""
    PLAYER = Vector(pos) if pos is not None else globals()["PLAYER"]
    yaw = math.radians(FACING_DEG if facing_deg is None else facing_deg)
    R = Matrix.Rotation(yaw, 4, "Z")
    fwd = Vector((math.cos(yaw), math.sin(yaw), 0))
    side = Vector((-fwd.y, fwd.x, 0))

    def at(f, s, z):
        return PLAYER + fwd * f + side * s + Vector((0, 0, z))

    # head: wider than tall (by eye on retail NL#941 to 945), 16 x 10 (section 5.2: 12 to 16 segments)
    head_c = at(0, 0, 0.76)
    HX, HY, HZ = 0.215, 0.235, 0.205                     # depth, width, height radii

    def surf(az_deg, lat_deg, out=0.0):
        """Point on the head ellipsoid (az 0 = facing direction, positive toward the player's left)."""
        az, la = math.radians(az_deg), math.radians(lat_deg)
        loc = Vector(((HX + out) * math.cos(la) * math.cos(az), (HY + out) * math.cos(la) * math.sin(az),
                      (HZ + out) * math.sin(la)))
        return head_c + R.to_3x3() @ loc

    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    res = bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=1.0,
                                    matrix=Matrix.Translation(head_c) @ R @ Matrix.Diagonal((HX, HY, HZ, 1.0)))
    for f in {f for v in res["verts"] for f in v.link_faces}:
        for lp in f.loops:
            d = R.inverted() @ (lp.vert.co - head_c)
            d = Vector((d.x / HX, d.y / HY, d.z / HZ)).normalized()
            lp[uv].uv = (0.5 + math.atan2(d.y, d.x) / (2 * math.pi), 0.5 + math.asin(max(-1, min(1, d.z))) / math.pi)
        us = [lp[uv].uv.x for lp in f.loops]
        if max(us) - min(us) > 0.5:
            for lp in f.loops:
                if lp[uv].uv.x < 0.5:
                    lp[uv].uv.x += 1.0
    mesh_object("Head", bm, col, [M["face"]])

    # nose: a small red orange nub below the eyes, about 0.07 of the head width (retail NL#941, 945)
    bm, uv = new_bm()
    nc = surf(0, -15, 0.004)
    bm_sphere(bm, uv, nc, 0.020, scale=(0.75, 1.0, 0.85), segs=8, rings=5, rot=R.to_quaternion())
    mesh_object("Nose", bm, col, [M["nose"]])

    # ears: round skin discs at the sides, eye to cheek height, about 0.19 of the head width (retail captures)
    bm, uv = new_bm()
    for sgn in (-1, 1):
        c = surf(sgn * 92, -15, 0.006)
        n = (c - head_c).normalized()
        rot = Vector((1, 0, 0)).rotation_difference(n)
        bm_sphere(bm, uv, c, 0.050, scale=(0.6, 0.9, 1.0), segs=10, rings=6, rot=rot)
    mesh_object("Ears", bm, col, [M["skin"]])

    # hair: a shell 0.022 off the head; vertices below the hairline are pulled up onto it, so the zigzag fringe
    # (6 teeth across the forehead) is an exact edge, the sides stop at the ear tops, the back covers the nape.
    def hairline(az):
        a_ = abs(az)
        if a_ <= 52:
            phase = (az + 52) / (104 / 6)                 # 6 teeth
            tooth = abs((phase % 1.0) - 0.5) * 2          # 1 at tooth tips, 0 between
            return 12 + 9 * (1 - tooth)
        if a_ <= 70:                                       # temple: fringe height down to the ear top
            t = (a_ - 52) / 18
            return 16 * (1 - t) + 1 * t
        if a_ <= 120:
            return 1.0
        t = min(1.0, (a_ - 120) / 40)
        return 1 * (1 - t) - 38 * t

    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    off = 0.022
    hres = bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=14, radius=1.0,
                                     matrix=Matrix.Diagonal((1.0, 1.0, 1.0, 1.0)), calc_uvs=True)
    moved = set()
    for v in hres["verts"]:
        az = math.degrees(math.atan2(v.co.y, v.co.x))
        la = math.degrees(math.asin(max(-1, min(1, v.co.z))))
        hl = hairline(az)
        if la < hl:
            la = hl
            moved.add(v)
        v.co = surf(az, la, off) - head_c
        v.co = v.co + head_c
    kill = [f for f in bm.faces if all(v in moved for v in f.verts)]
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    mesh_object("Hair", bm, col, [M["hair"]])

    bm, uv = new_bm()                                    # torso (shirt), short sleeves
    bm_tube(bm, uv, [at(0, 0, 0.29), at(0, 0, 0.45), at(0, 0, 0.565)], [0.125, 0.115, 0.085], 8, v0=0.75, v1=1.0,
            cap_end=True, cap_start=True)
    for s in (-1, 1):
        bm_sphere(bm, uv, at(0, s * 0.10, 0.52), 0.045, segs=8, rings=5)          # shoulder
        if fishing:
            bm_tube(bm, uv, [at(0, s * 0.11, 0.52), at(0.07, s * 0.10, 0.46)], [0.045, 0.042], 6, v0=0.0, v1=0.24)
        else:                                            # arms relaxed at the sides
            bm_tube(bm, uv, [at(0, s * 0.12, 0.52), at(0.01, s * 0.145, 0.45)], [0.045, 0.042], 6, v0=0.0, v1=0.24)
    mesh_object("Body", bm, col, [M["shirt"]])
    bm, uv = new_bm()                                    # forearms (skin) to the mitten hands
    for s in (-1, 1):
        if fishing:
            bm_tube(bm, uv, [at(0.06, s * 0.10, 0.47), at(0.18, s * 0.045, 0.40)], [0.032, 0.028], 6)
        else:
            bm_tube(bm, uv, [at(0.01, s * 0.145, 0.46), at(0.02, s * 0.155, 0.34)], [0.032, 0.028], 6)
    mesh_object("Forearms", bm, col, [M["skin"]])
    bm, uv = new_bm()
    bm_tube(bm, uv, [at(0, 0, 0.22), at(0, 0, 0.32)], [0.125, 0.13], 8, cap_start=True)
    mesh_object("Shorts", bm, col, [M["shorts"]])
    bm, uv = new_bm()
    for s in (-1, 1):
        bm_tube(bm, uv, [at(0, s * 0.06, 0.24), at(0, s * 0.06, 0.05)], [0.04, 0.036], 6)
        bm_sphere(bm, uv, at(0, s * 0.06, 0.05), 0.045, segs=10, rings=6)
    mesh_object("Legs", bm, col, [M["skin"]])
    bm, uv = new_bm()
    for s in (-1, 1):
        bm_sphere(bm, uv, at(0.03, s * 0.06, 0.025), 0.055, scale=(1.35, 0.85, 0.55), segs=10, rings=6)
    mesh_object("Shoes", bm, col, [M["shoe"]])
    bm, uv = new_bm()
    for s in (-1, 1):
        bm_sphere(bm, uv, at(0.19, s * 0.04, 0.39) if fishing else at(0.02, s * 0.157, 0.32), 0.04, segs=10, rings=6)
    mesh_object("Hands", bm, col, [M["skin"]])

    if not fishing:                                      # hands at the sides, no rod
        return head_c, None
    grip = at(0.2, 0.0, 0.39)
    tip = grip + Vector((1.25, 0.25, 0.95))
    bm, uv = new_bm()
    bm_tube(bm, uv, [grip - Vector((0.1, 0.0, 0.07)), tip], [0.018, 0.007], 6, v0=0.0, v1=1.0)
    mesh_object("Rod", bm, col, [M["rod"]])
    bob = Vector((2.35, 0.55, WATER_Z + 0.02))
    # fish shadow: a soft teardrop a little below the surface, round head toward the bobber, pointed tail
    # (reference captures supplied by the user: length about 3x width, widest about 0.3 from the head, by eye)
    fc = Vector((2.05, 1.25, WATER_Z - 0.06))
    head_dir = (bob - fc)
    head_dir.z = 0
    head_dir.normalize()
    sdir = Vector((-head_dir.y, head_dir.x, 0))
    Lf = 0.55
    bm, uv = new_bm()
    vs = [bm.verts.new(c) for c in (fc + head_dir * Lf * 0.5 + sdir * Lf * 0.25, fc + head_dir * Lf * 0.5 - sdir * Lf * 0.25,
                                    fc - head_dir * Lf * 0.5 - sdir * Lf * 0.25, fc - head_dir * Lf * 0.5 + sdir * Lf * 0.25)]
    f = bm.faces.new([bm.verts.new(v.co - fc) for v in vs])
    for v in vs:
        bm.verts.remove(v)
    for lp, q in zip(f.loops, ((1, 1), (1, 0), (0, 0), (0, 1))):
        lp[uv].uv = q
    mesh_object("FishShadow", bm, col, [M["fish"]], smooth=False, loc=fc)
    bm, uv = new_bm()
    pts = []
    for k in range(7):
        t = k / 6
        p = tip.lerp(bob + Vector((0, 0, 0.06)), t)
        p.z -= 0.25 * 4 * t * (1 - t)
        pts.append(p)
    bm_tube(bm, uv, pts, [0.004] * 7, 4, cap_end=False)
    mesh_object("Line", bm, col, [M["line"]])
    bm, uv = new_bm()
    bm_sphere(bm, uv, bob + Vector((0, 0, 0.035)), 0.055, scale=(1, 1, 1.15), segs=10, rings=6)
    mesh_object("Bobber", bm, col, [M["bobber"]])
    bm, uv = new_bm()
    ring = [bm.verts.new(bob + Vector((math.cos(a) * r, math.sin(a) * r, 0.004)))
            for r in (0.15, 0.162) for a in np.linspace(0, 2 * math.pi, 20, endpoint=False)]
    for i in range(20):
        j = (i + 1) % 20
        bm.faces.new((ring[i], ring[20 + i], ring[20 + j], ring[j]))
    mesh_object("Ripple", bm, col, [M["foam"]], smooth=False)
    return head_c, bob


FISH_MUL = (0.50, 0.63, 0.68)    # M 16.4 retail plateau multiply (displayed); crisp 1 to 2 px edge


def tex_fish():
    """Fish shadow mask: u along the body (1 = head). Round head, widest at 0.3 from the head, one end tapering to a
    point; CRISP edge (16.4: 1 to 2 px at native resolution, no blur)."""
    w, h = 256, 80
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = xx / (w - 1), (yy / (h - 1) - 0.5) * 2
    t = 1 - u
    half = np.where(t < 0.3, np.sqrt(np.clip(1 - ((0.3 - t) / 0.3) ** 2, 0, 1)), np.clip((1 - t) / 0.7, 0, 1) ** 0.8)
    edge = half * 0.92 - np.abs(v)
    a = smoothstep(-0.012, 0.012, edge)
    return make_image("T_Fish", np.ones((h, w, 3)), a)


def fish_mat():
    """Multiply decal drawn after the water: Transparent BSDF tinted mix(white, multiply, mask)."""
    m = bpy.data.materials.new("M_Fish")
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image, t.interpolation, t.extension = tex_fish(), "Linear", "CLIP"
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    link(nt, t.outputs["Alpha"], mx.inputs[0])
    mx.inputs[6].default_value = (1, 1, 1, 1)
    mx.inputs[7].default_value = (*(c ** 2.2 for c in FISH_MUL), 1)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    link(nt, mx.outputs[2], tr.inputs["Color"])
    link(nt, tr.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "BLENDED"
    return m


def blob(col, m, name, center, rx, ry):
    bm, uv = new_bm()
    bm_disc(bm, uv, Vector(center), 1.0, segs=20, sx=rx, sy=ry)
    ob = mesh_object(name, bm, col, [m], smooth=False)
    return ob


def build_sky():
    w = bpy.data.worlds.new("Sky")
    bpy.context.scene.world = w
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    link(nt, tc.outputs["Window"], sep.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    link(nt, sep.outputs[1], mr.inputs["Value"])
    mr.inputs["From Min"].default_value = 1.0 - HORIZON_ROW - 0.02
    mr.inputs["From Max"].default_value = 1.0
    cr = nt.nodes.new("ShaderNodeValToRGB")
    link(nt, mr.outputs["Result"], cr.inputs[0])
    top, low = SKY_BY_TIME.get(TIME, (PAL["sky_top"], PAL["sky_low"]))
    cr.color_ramp.elements[0].color = (*lin(hexc(low)), 1)
    cr.color_ramp.elements[1].color = (*lin(hexc(top)), 1)
    bg = nt.nodes.new("ShaderNodeBackground")
    link(nt, cr.outputs["Color"], bg.inputs["Color"])
    link(nt, bg.outputs[0], out.inputs["Surface"])


def tex_cloud():
    """Thin soft wisps (retail NL#939: low contrast streaks, core #5790D9, brightest #6FA2E1, against #3044DF)."""
    w, h = 256, 64
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = xx / (w - 1), yy / (h - 1)
    a = np.zeros((h, w))
    for k in range(7):
        cx, cy = RNG.uniform(0.15, 0.85), RNG.uniform(0.35, 0.65)
        sx, sy = RNG.uniform(0.12, 0.28), RNG.uniform(0.08, 0.16)
        a = np.maximum(a, np.exp(-(((u - cx) / sx) ** 2 + ((v - cy) / sy) ** 2)))
    a = a * (0.75 + 0.25 * fbm(w, h, 8, 2, RNG))
    a = smoothstep(0.15, 0.85, a) * 0.9
    cols = {"dusk": ["#9C7A86", "#C99090", "#DDA495"]}.get(TIME, ["#5790D9", "#6FA2E1", "#8FB8EA"])
    col = ramp(a, cols)
    return make_image("T_Cloud", col, a)


def cloud_mat(img):
    m = bpy.data.materials.new("M_Cloud")
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image, t.interpolation, t.extension = img, "Linear", "CLIP"
    em = nt.nodes.new("ShaderNodeEmission")
    link(nt, t.outputs["Color"], em.inputs["Color"])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mx = nt.nodes.new("ShaderNodeMixShader")
    link(nt, t.outputs["Alpha"], mx.inputs[0])
    link(nt, tr.outputs[0], mx.inputs[1])
    link(nt, em.outputs[0], mx.inputs[2])
    link(nt, mx.outputs[0], out.inputs["Surface"])
    m.surface_render_method = "BLENDED"
    return m


# screen x, screen y (from the bottom), width as a share of the frame: inside the thin sky strip, between crowns
CLOUDS = [(0.08, 0.955, 0.30), (0.40, 0.975, 0.26), (0.74, 0.96, 0.34)]


def build_clouds(col, M, cam, dist=80.0):
    """Cloud cards along camera rays far beyond the bent ground, so they only show over the sky (not bent)."""
    sc = bpy.context.scene
    bpy.context.view_layer.update()
    mw = cam.matrix_world
    tr, br, bl, tl = [mw @ v for v in cam.data.view_frame(scene=sc)]
    org = mw.translation
    for i, (sx, sy, wf) in enumerate(CLOUDS):
        d = (bl.lerp(br, sx).lerp(tl.lerp(tr, sx), sy) - org).normalized()
        c = org + d * dist
        width = (tr - tl).length * dist / (tl - org).length * wf
        right = (tr - tl).normalized()
        up = right.cross(d).normalized()
        if up.z < 0:
            up = -up
        hw, hh = width / 2, width * 0.25 / 2
        bm, uv = new_bm()
        vs = [bm.verts.new(c + right * a * hw + up * b * hh) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        f = bm.faces.new(vs)
        for lp, q in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
            lp[uv].uv = q
        mesh_object(f"Cloud{i}", bm, col, [M["cloud"]], smooth=False)


def build_camera(col):
    cd = bpy.data.cameras.new("Cam")
    cd.lens, cd.sensor_height, cd.sensor_fit = FOCAL, SENSOR_H, "VERTICAL"
    cd.clip_start, cd.clip_end = 0.05, 300.0
    cam = bpy.data.objects.new("Cam", cd)
    col.objects.link(cam)
    cam.location = (PLAYER.x, PLAYER.y - CAM_D, CAM_H)
    cam.rotation_euler = (math.radians(90 - PITCH), 0.0, 0.0)
    bpy.context.scene.camera = cam
    return cam


def make_assets():
    """LUT group, painted textures and the material set shared by every test scene."""
    global LUT
    LUT = build_lut()
    T = {"grass": tex_grass(), "wall": tex_wall(), "caustic": tex_caustic(), "foliage": tex_foliage(),
         "bark": tex_bark(), "face": tex_face(), "hair": tex_hair(), "shirt": tex_shirt(), "bobber": tex_bobber(),
         "rock": tex_rock(), "leaves": tex_leaves(), "cloud": tex_cloud(),
         "flower_a": tex_flower("#F4F0FF", "#5B3C8C"), "flower_b": tex_flower("#FFE15A", "#7A4A1E"),
         "skin": tex_flat("T_Skin", [PAL["skin_shade"], PAL["skin"], PAL["skin"]], 0.02, 2),
         "shorts": tex_flat("T_Shorts", ["#26347A", PAL["shorts"], "#3B4C9E"]),
         "shoe": tex_flat("T_Shoe", ["#D8D8E0", PAL["shoe"], "#FFFFFF"], 0.02),
         "rod": tex_flat("T_Rod", ["#A86E2C", PAL["rod"], "#D8A050"], 0.03),
         "core": tex_flat("T_LeafCore", [PAL["foliage_dark"], PAL["foliage_dark"], PAL["foliage"]], 0.05, 6),
         "stem": tex_flat("T_Stem", [PAL["foliage_dark"], PAL["foliage"]], 0.03)}
    contact_alpha = 1.0 - 0.7     # trunk base patches: ratio about 0.7 (section 3), as a blend toward the tint
    M = {
        "terrain": terrain_mat(T), "water": water_mat(T),
        "foliage_core": mat("M_LeafCore", image=T["core"], uv_scale=(3, 3)),
        "leaves": mat("M_Leaves", image=T["leaves"], cutout=True),
        "bark": mat("M_Bark", image=T["bark"]),
        "stem": mat("M_Stem", image=T["stem"]),
        "flower_a": mat("M_FlowerA", image=T["flower_a"], cutout=True),
        "flower_b": mat("M_FlowerB", image=T["flower_b"], cutout=True),
        "rock": mat("M_Rock", image=T["rock"]),
        "face": mat("M_Face", image=T["face"], rim=1.0),
        "hair": mat("M_Hair", image=T["hair"], rim=1.0),
        "shirt": mat("M_Shirt", image=T["shirt"], rim=1.0),
        "shorts": mat("M_Shorts", image=T["shorts"], rim=1.0, uv_scale=(2, 2)),
        "skin": mat("M_Skin", image=T["skin"], rim=1.0),
        "shoe": mat("M_Shoe", image=T["shoe"], rim=1.0),
        "rod": mat("M_Rod", image=T["rod"]),
        "line": unlit("M_Line", "#F0F0F0"),
        "bobber": mat("M_Bobber", image=T["bobber"]),
        "foam": unlit("M_Foam", "#CFEFFF"),
        "nose": mat("M_Nose", image=tex_flat("T_Nose", ["#B84838", PAL["nose"], "#E06A55"], 0.02, 2), rim=1.0),
        "fish": fish_mat(),
        "blob": shadow_mat("M_Blob", "blob", inner=0.75),
        "contact": shadow_mat("M_Contact", "contact", inner=0.60, wobble=0.10),
        "canopy": shadow_mat("M_CanopyShadow", "canopy", inner=0.40, wobble=0.16),
        "cloud": cloud_mat(T["cloud"]),
    }
    return T, M


def finalize_render(native, world, M, cam):
    """Resolution, clouds and EEVEE settings shared by every test scene."""
    sc = bpy.context.scene
    rx_, ry_ = NATIVE if native else RES
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = rx_, ry_, 100
    if TIME != "night":
        build_clouds(world, M, cam)
    sc.render.engine = "BLENDER_EEVEE"
    ee = sc.eevee
    if native:
        sc.render.filter_size = 0.5
        ee.taa_render_samples = 1
    else:
        sc.render.filter_size = 1.5
        ee.taa_render_samples = SAMPLES
    for attr in ("use_shadows", "use_gtao", "use_bloom", "use_ssr", "use_raytracing"):
        if hasattr(ee, attr):
            try:
                setattr(ee, attr, False)
            except Exception:
                pass
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.render.image_settings.media_type = "IMAGE"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"


def build(native=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    rng = np.random.default_rng(3)
    T, M = make_assets()
    world = bpy.data.collections.new("World")
    sc.collection.children.link(world)
    build_sky()
    build_terrain(world, M)
    build_water(world, M)
    # horizon tree line (section 9 rule 3) and a few nearer trees on both sides of the river (4 to 6 trees)
    for x, y, hgt in TREES + NEAR_TREES:
        build_tree(world, M, x, y, hgt, rng)
    for x, y in FLOWERS:
        build_flowers(world, M, x, y, rng)
    build_rock(world, M, -1.55, 2.9, 0.26)
    build_bush(world, M, -1.55, -0.95, 0.40, rng)             # foreground occluder cutting the bottom edge
    head_c, bob = build_player(world, M)
    # player blob: on screen aspect 1.35 at this view (section 3): ground radius y / x = sin(dep) x 1.35 inverse
    dep = math.atan2(CAM_H, CAM_D)
    rx = BLOB_EXTENT / 2
    ry = rx * (1.0 / (BLOB_ASPECT_ON_SCREEN * math.sin(dep)))
    blob(world, M["blob"], "PlayerShadow", (PLAYER.x, PLAYER.y, 0.006), rx, ry)
    bend_world()
    cam = build_camera(world)
    finalize_render(native, world, M, cam)
    return {"cam": cam, "head": head_c, "bobber": bob}


# --------------------------------------------------------------------------
# Checks (geometry) and QA on the rendered PNG (section 12.2 item 12)
# --------------------------------------------------------------------------
def check(info):
    sc = bpy.context.scene
    bpy.context.view_layer.update()
    cam = info["cam"]
    print(f"camera: pitch {PITCH} deg, vFOV {VFOV} deg, focal {FOCAL:.1f} mm (24 mm sensor height)")
    print(f"camera solved: height {CAM_H:.2f} P, ground distance to feet {CAM_D:.2f} P "
          f"(blueprint outdoors 3.6 / 4.1)")
    print(f"bend radius calibrated to horizon row {HORIZON_ROW}: R = {LOG_R:.2f} P (blueprint 12 to 14 P)")
    pts = {"feet": PLAYER, "hair top": PLAYER + Vector((0, 0, 1.0)), "bobber": info["bobber"]}
    for k, p in pts.items():
        v = world_to_camera_view(sc, cam, bend_point(p))
        print(f"  {k:9s} row {1 - v.y:.3f}  column {v.x:.3f}")
    a = world_to_camera_view(sc, cam, bend_point(PLAYER))
    b = world_to_camera_view(sc, cam, bend_point(PLAYER + Vector((0, 0, 1.0))))
    print(f"player share {b.y - a.y:.3f} (target {PLAYER_SHARE}), feet row {1 - a.y:.3f} (target {FEET_ROW})")
    for x, y, hgt in TREES:
        top = bend_point(Vector((x, y, terrain_h(x, y) + hgt * 0.42 + 1.12)))
        base = bend_point(Vector((x, y, terrain_h(x, y))))
        vt, vb = world_to_camera_view(sc, cam, top), world_to_camera_view(sc, cam, base)
        print(f"  tree ({x}, {y}): crown top row {1 - vt.y:.3f}, base row {1 - vb.y:.3f}, column {vt.x:.3f}")
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in sc.objects if o.type == "MESH")
    pl = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in sc.objects
             if o.type == "MESH" and o.name.split(".")[0] in ("Head", "Hair", "Body", "Shorts", "Legs", "Shoes", "Hands",
                                                               "Nose", "Ears", "Forearms"))
    print(f"triangles: player {pl}, scene {tris}")


def qa(png, info):
    """Measure the rendered frame: colour envelope, horizon row at the centre column, blob core ratio."""
    im = bpy.data.images.load(png)
    w, h = im.size
    px = np.array(im.pixels[:]).reshape(h, w, 4)[::-1, :, :3]        # top row first, sRGB 0..1
    v = px.max(-1)
    s = np.where(v > 0, (v - px.min(-1)) / np.maximum(v, 1e-6), 0)
    neon = float(((s > 0.85) & (v > 0.85)).mean())
    nblack = float((v < 0.08).mean())
    nwhite = float(((v > 0.92) & (s < 0.12)).mean())
    print(f"QA envelope: neon {neon:.3%}, near black {nblack:.3%}, near white {nwhite:.3%} (each under 2%)")
    # haze (16.3, V2 method): mean luminance of ground pixels in a 20 row band just below the horizon over a band
    # at rows 170 to 236 of 240, left 45% of the columns (grass side, away from the river)
    Lp = 0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]
    gmask = (px[..., 1] > px[..., 0]) & (px[..., 1] > px[..., 2])
    cols = slice(0, int(w * 0.45))
    def band(r0, r1):
        ys = slice(int(r0 / 240 * h), int(r1 / 240 * h))
        m = gmask[ys, cols]
        return float(Lp[ys, cols][m].mean()) if m.any() else float("nan")
    hz = HORIZON_ROW * 240
    far, near = band(hz + 2, hz + 22), band(170, 236)
    print(f"QA haze: far band luminance / near band {far / near:.2f} (16.3 retail median 2.15, range 1.61 to 3.19)")
    # river lines (16.4): luminance p95 and p99 over the median on water pixels, right side of the frame
    wm = (px[..., 2] > px[..., 0] + 0.15) & (px[..., 2] > px[..., 1])
    wm[:, :int(w * 0.6)] = False
    wm[:int(h * 0.3)] = False
    if wm.any():
        lw = Lp[wm]
        print(f"QA river lines: p95 / median {np.percentile(lw, 95) / np.median(lw):.2f}, p99 / median "
              f"{np.percentile(lw, 99) / np.median(lw):.2f} (16.4 rivers 1.28 and 1.47)")
    # horizon: per column, last sky row from the top (sky = blue dominant and dark red channel)
    sky = (px[..., 2] > 0.6) & (px[..., 2] > px[..., 0] + 0.2) & (px[..., 2] > px[..., 1] + 0.1)   # sky or cloud
    rows = []
    for x in range(int(w * 0.40), int(w * 0.60)):
        col = sky[:, x]
        if col[0]:
            rows.append(np.argmin(col) / h if not col.all() else 1.0)
    if rows:
        print(f"QA horizon row (centre 20% of columns, sky columns only): median {np.median(rows):.3f} "
              f"(target {HORIZON_ROW}; 13 to 18% sky at the centre per section 1.2)")
    # blob core ratio: pixel just in front of the feet vs grass beside it
    sc = bpy.context.scene
    cam = info["cam"]
    pf = world_to_camera_view(sc, cam, bend_point(PLAYER + Vector((0.0, -0.17, 0.0))))
    pg = world_to_camera_view(sc, cam, bend_point(PLAYER + Vector((-0.75, -0.17, 0.0))))
    def sample(p):
        x, y = int(p.x * w), int((1 - p.y) * h)
        return px[max(0, y - 1):y + 2, max(0, x - 1):x + 2].reshape(-1, 3).mean(0)
    cf, cg = sample(pf), sample(pg)
    # object shadows (section 3): flower clump ground and near tree canopy shadow vs open grass beside them
    L = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    def south_point(name_, cx_, cy_):
        """First point south of the object (toward the camera) where its shadow mask exceeds 0.85."""
        for nm, sx, sy, half, m in PROFILE_SHADOWS:
            if nm.startswith(name_) and abs(sx - cx_) < 1e-6 and abs(sy - cy_) < 1e-6:
                res = m.shape[0]
                col_ = res // 2
                for row in range(0, res // 2):
                    if m[row, col_] > 0.85:
                        return (sx, sy - half + (row + 2.5) / res * 2 * half)
        return (cx_, cy_)
    fp = south_point("FlowerShadow", *FLOWERS[0])
    tp = south_point("TreeShadow", NEAR_TREES[0][0], NEAR_TREES[0][1])
    for name, at_, ref, target in (
            ("flower clump shadow (south edge)", fp, (fp[0], fp[1] - 0.45), "0.68"),
            ("tree canopy shadow (south edge)", tp, (tp[0] + 1.3, tp[1]), "0.77 to 0.84 near the rim, darker inside")):
        ps = world_to_camera_view(sc, cam, bend_point(Vector((at_[0], at_[1], 0.0))))
        pr = world_to_camera_view(sc, cam, bend_point(Vector((ref[0], ref[1], 0.0))))
        print(f"QA {name}: luminance ratio {L(sample(ps)) / max(L(sample(pr)), 1e-6):.2f} (NL#946 {target})")
    ratio = cf / np.maximum(cg, 1e-6)
    print(f"QA blob: shadow sRGB {tuple(int(c * 255) for c in cf)} vs grass {tuple(int(c * 255) for c in cg)}; "
          f"R, G ratios {ratio[0]:.2f}, {ratio[1]:.2f} (target about 0.45), B ratio {ratio[2]:.2f} "
          f"(blue multiplied less than R and G; above 1 only on lime grass)")


def haze_qa(png):
    im = bpy.data.images.load(png)
    w, h = im.size
    px = np.array(im.pixels[:]).reshape(h, w, 4)[::-1, :, :3]
    Lp = 0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]
    cols = slice(0, int(w * 0.45))
    hz = HORIZON_ROW * 240
    def band(r0, r1):
        return float(Lp[int(r0 / 240 * h):int(r1 / 240 * h), cols].mean())
    print(f"QA haze (ground only pass): far band {hz + 2:.0f} to {hz + 22:.0f} over near 170 to 236 = "
          f"{band(hz + 2, hz + 22) / band(170, 236):.2f} (16.3 retail median 2.15 fixed bands, 1.86 relative; range 1.61 to 3.19)")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--render")
    p.add_argument("--native", action="store_true", help="400x240, one sample, then a 3x nearest upscale")
    p.add_argument("--save")
    p.add_argument("--grass", choices=("retail", "press", "emote"), default="retail",
                   help="retail: user's retail stream capture; press: E3 build green; emote: lime emote crops")
    p.add_argument("--time", choices=sorted(GRADE), default="day", help="time of day grade and sky (16.2, 16.5)")
    a = p.parse_args(argv)
    global TIME
    TIME = a.time
    if a.grass == "emote":
        PAL.update(GRASS_EMOTE)
        global BLOB_CORE_ALPHA, SHADOW_MODE
        BLOB_CORE_ALPHA = 0.86
        SHADOW_MODE = "blend"
    if a.grass == "press":
        PAL.update(GRASS_PRESS)
        # press grass is teal green, not lime: solve the blend target so each channel falls to about 0.48 (sRGB)
        # of #188050 at the same core alpha: C = (target_lin - (1 - a) grass_lin) / a
        global BLOB_TINT_LIN
        BLOB_TINT_LIN = (0.0024, 0.020, 0.0094)
    info = build(a.native)
    if a.check:
        check(info)
    if a.save:
        path = os.path.abspath(a.save)
        if os.path.exists(path):
            sys.exit(f"refusing to overwrite {path}")
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=path)
    if a.render:
        import time
        out = os.path.abspath(a.render)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        bpy.context.scene.render.filepath = out
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"rendered {out} in {time.time() - t:.1f}s")
        qa(out, info)
        # haze QA on ground and water only (objects hidden), so far tree crowns do not dilute the band ratio
        keep = ("Terrain", "Water")
        hidden = [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.name.startswith(keep)]
        for o in hidden:
            o.hide_render = True
        gp = os.path.join(os.path.dirname(out), ".haze_ground_pass.png")
        bpy.context.scene.render.filepath = gp
        bpy.ops.render.render(write_still=True)
        haze_qa(gp)
        os.remove(gp)
        for o in hidden:
            o.hide_render = False
        if a.native and shutil.which("ffmpeg"):
            root, ext = os.path.splitext(out)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-vf",
                            "scale=iw*3:ih*3:flags=neighbor", f"{root}_x3{ext}"], check=True)


if __name__ == "__main__":
    main()
