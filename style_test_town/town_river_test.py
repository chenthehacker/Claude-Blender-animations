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
FEET_ROW = 0.675             # M section 1.2 (67 to 68% from the top)
HORIZON_ROW = 0.165          # M section 1.2 (16.5%): calibrates the bend radius
RES = (1200, 720)            # 5:3 game frame at 3x, clean (section 12.2 item 11)
NATIVE = (400, 240)
SAMPLES = 16

SUN = Vector((0.0, -0.34, 0.94)).normalized()   # E frontal key from the camera side (section 2)
LUT_SHADOW_SRGB = (0.54, 0.56, 0.65)            # M NL rock, section 2
LUT_LIGHT_SRGB = (1.46, 1.38, 1.20)             # M upper bound, section 2
RIM_TINT = "#C8B4E6"                            # E tint, gain M: +22% displayed, about 1.5x linear (section 2)
RIM_LINEAR_GAIN = 0.5
BLOB_W50, BLOB_EXTENT = 0.38, 0.43              # M section 3, in P
BLOB_ASPECT_ON_SCREEN = 1.35                    # M section 3
BLOB_TINT_LIN = (0.018, 0.028, 0.032)           # E dark blue grey blend target (section 3 audit: blue must rise)
BLOB_CORE_ALPHA = 0.86                          # derived below: R and G fall to about 0.45 of the lime grass

PAL = {
    "sky_top": "#2135E7", "sky_low": "#4078DA",                         # M retail NL#940, NL#939
    "grass": "#92DB10", "grass_dark": "#82CC09", "grass_light": "#A1E917",  # M retail NL#941 to 945
    "cliff_lit": "#FFCEAE", "cliff_shade": "#BA8B8B",                   # M section 4.2
    "water": "#3A75B4", "water_shallow": "#6BBEF0", "foam": "#FFFFFF",  # M section 4.3 (foam E)
    "foliage": "#4A9C52", "foliage_dark": "#287046", "foliage_light": "#82D362",  # M section 7.3
    "wood": "#E4A560", "wood_dark": "#A1693B", "wood_light": "#FFD391",   # M section 7.3
    "skin": "#FFD1A8", "skin_shade": "#CE8F79",                         # M section 7.3
    "rock": "#9992A2",                                                  # M section 4.2
    "hair": "#6B4A35", "hair_light": "#8E6648",                         # E (original character)
    "shirt": "#3F8FD0", "shirt_light": "#8EC6F0", "shorts": "#2F3E8C", "shoe": "#F2F2F2",  # E
    "eye": "#1E1A28", "blush": "#E08A76",                               # E eye; M blush range section 5.3
    "bobber": "#E0473A", "rod": "#C88838", "rod_band": "#585888",       # E accent red; rod structure section 5.3
    "flower_a": "#FFF6E0", "flower_b": "#FFE15A", "flower_centre": "#F0A040",  # E pastel (keep red unique)
}

GRASS_PRESS = {"grass": "#188050", "grass_dark": "#02613E", "grass_light": "#2A9A5E"}   # M press builds (light E)

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


def bend_array(co):
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
    w = h = 128
    n = 0.6 * fbm(w, h, 4, 3, RNG) + 0.4 * fbm(w, h, 16, 1, RNG)
    return make_image("T_Grass", ramp(n, [PAL["grass_dark"], PAL["grass"], PAL["grass_light"]]))


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
    w, h = 32, 64
    u = np.linspace(0, 1, w)[None, :] * np.ones((h, 1))
    g = 0.5 + 0.5 * np.sin(2 * np.pi * (u * 5 + 0.3 * fbm(w, h, 2, 2, RNG)))
    return make_image("T_Bark", ramp(0.25 + 0.5 * g, [PAL["wood_dark"], PAL["wood"], PAL["wood_light"]]))


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
        b = ((u - (0.5 + side * 0.11)) / 0.035) ** 2 + ((v - 0.37) / 0.035) ** 2
        col = np.where((b < 1)[..., None], col * 0.6 + hexc(PAL["blush"]) * 0.4, col)
    m = ((u - 0.5) / 0.022) ** 2 + ((v - 0.33) / 0.012) ** 2
    col[m < 1] = hexc("#B0503C")
    return make_image("T_Face", grain(col, 0.02))


def tex_hair():
    w, h = 64, 32
    u = np.linspace(0, 1, w)[None, :] * np.ones((h, 1))
    s = 0.5 + 0.5 * np.sin(2 * np.pi * u * 12 + 1.5 * fbm(w, h, 4, 1, RNG))
    return make_image("T_Hair", grain(ramp(0.3 + 0.5 * s, [PAL["hair"], PAL["hair_light"]]), 0.03))


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
    link(ng, add.outputs["Vector"], go.inputs[0])
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
    link(nt, em.outputs[0], out.inputs["Surface"])
    return m


def unlit(name, color):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*lin(hexc(color)), 1)
    link(nt, em.outputs[0], out.inputs["Surface"])
    return m


def blob_mat(name, core_alpha, tint_lin, inner=0.76):
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
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    link(nt, ln.outputs["Value"], mr.inputs["Value"])
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
    em = nt.nodes.new("ShaderNodeEmission")
    link(nt, g.outputs[0], em.inputs["Color"])
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
    mx.inputs[6].default_value = (*lin(hexc(PAL["water"])), 1)
    mx.inputs[7].default_value = (*lin(hexc(PAL["water_shallow"])), 1)
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.55, 0.40, 1)
    link(nt, nt.nodes.new("ShaderNodeTexCoord").outputs["Object"], mp.inputs["Vector"])
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image, t.interpolation = T["caustic"], "Linear"
    link(nt, mp.outputs["Vector"], t.inputs["Vector"])
    cm = nt.nodes.new("ShaderNodeMath")
    cm.operation = "MULTIPLY"
    link(nt, t.outputs["Color"], cm.inputs[0])
    cm.inputs[1].default_value = 0.35            # lines about +18% at p95 over the median (section 4.3)
    mx2 = nt.nodes.new("ShaderNodeMix")
    mx2.data_type = "RGBA"
    link(nt, cm.outputs[0], mx2.inputs[0])
    link(nt, mx.outputs[2], mx2.inputs[6])
    mx2.inputs[7].default_value = (*lin(hexc(PAL["water_shallow"])), 1)
    em = nt.nodes.new("ShaderNodeEmission")
    link(nt, mx2.outputs[2], em.inputs["Color"])
    link(nt, em.outputs[0], out.inputs["Surface"])
    return m


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
    grid = []
    for y in ys:
        row = []
        for x in xs:
            z = terrain_h(x, y)
            v = bm.verts.new((x, y, z))
            v[wl] = float(smoothstep(-0.03, -0.12, z))
            row.append(v)
        grid.append(row)
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            f = bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
            for lp in f.loops:
                c = lp.vert.co
                lp[uv].uv = (c.x * 0.35, c.y * 0.35)
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
    return mesh_object("Water", bm, col, [M["water"]], smooth=False)


def build_tree(col, M, x, y, height, rng):
    z0 = terrain_h(x, y)
    bm, uv = new_bm()
    bm_tube(bm, uv, [(x, y, z0 - 0.1), (x, y, z0 + height * 0.55)], [0.12, 0.08], 7, cap_end=True)
    mesh_object("TreeTrunk", bm, col, [M["bark"]])
    bm, uv = new_bm()
    top = z0 + height * 0.55
    for k in range(3):
        a = 2 * math.pi * k / 3 + rng.uniform(0, 1)
        c = (x + 0.18 * math.cos(a), y + 0.18 * math.sin(a), top + 0.28 + 0.08 * k)
        bm_sphere(bm, uv, c, 0.42 + 0.06 * rng.random(), scale=(1.05, 1.05, 0.9), segs=14, rings=8)
    bm_sphere(bm, uv, (x, y, top + 0.62), 0.36, segs=14, rings=8)
    mesh_object("TreeCrown", bm, col, [M["foliage"]])
    blob(col, M["contact"], "TreeShadow", (x, y, z0 + 0.012), 0.55, 0.55)


def build_flowers(col, M, x, y, rng):
    z0 = terrain_h(x, y)
    bm, uv = new_bm()
    bm2, uv2 = new_bm()
    bm3, uv3 = new_bm()
    for k in range(5):
        a = 2 * math.pi * k / 5 + rng.uniform(0, 0.6)
        px, py = x + 0.19 * math.cos(a), y + 0.16 * math.sin(a)
        hgt = 0.12 + 0.04 * rng.random()
        bm_tube(bm, uv, [(px, py, z0), (px, py, z0 + hgt)], [0.012, 0.010], 4)
        bm_sphere(bm, uv, (px + 0.03, py, z0 + 0.03), 0.04, scale=(1.4, 0.7, 0.3), segs=6, rings=3)
        for pk in range(5):
            pa = 2 * math.pi * pk / 5
            bm_sphere(bm2 if k % 2 else bm3, uv2 if k % 2 else uv3,
                      (px + 0.035 * math.cos(pa), py + 0.035 * math.sin(pa), z0 + hgt), 0.026,
                      scale=(1, 1, 0.45), segs=6, rings=3)
        bm_sphere(bm, uv, (px, py, z0 + hgt + 0.012), 0.018, segs=6, rings=3, mat=1)
    mesh_object("FlowerLeaves", bm, col, [M["leaf"], M["flower_c"]])
    mesh_object("FlowerPetalsA", bm2, col, [M["flower_a"]])
    mesh_object("FlowerPetalsB", bm3, col, [M["flower_b"]])
    blob(col, M["contact"], "FlowerShadow", (x, y, z0 + 0.01), 0.2, 0.2)


def build_rock(col, M, x, y, s):
    z0 = terrain_h(x, y)
    bm = bmesh.new()
    res = bmesh.ops.create_icosphere(bm, subdivisions=1, radius=s,
                                     matrix=Matrix.Translation((x, y, z0 + 0.35 * s)) @
                                     Matrix.Diagonal((1.25, 1.0, 0.75, 1.0)))
    rng = np.random.default_rng(4)
    for v in res["verts"]:
        v.co += Vector(rng.uniform(-0.05, 0.05, 3)) * s
    mesh_object("Rock", bm, col, [M["rock"]], smooth=False)       # facets visible on props (section 2)
    blob(col, M["contact"], "RockShadow", (x, y, z0 + 0.01), 1.3 * s, 1.1 * s)


def build_bush(col, M, x, y, r, rng):
    z0 = terrain_h(x, y)
    bm, uv = new_bm()
    for k in range(5):
        a = 2 * math.pi * k / 5
        bm_sphere(bm, uv, (x + 0.45 * r * math.cos(a), y + 0.35 * r * math.sin(a), z0 + 0.55 * r),
                  0.6 * r, segs=14, rings=8)
    bm_sphere(bm, uv, (x, y, z0 + 0.9 * r), 0.6 * r, segs=14, rings=8)
    mesh_object("Bush", bm, col, [M["foliage"]])


# Horizon tree line beyond the bend tangent so trunk bases sink (section 9 rule 3); 5 trees (rule 5: 4 to 6).
TREES = [(-3.6, 11.5, 2.0), (-1.4, 12.6, 2.1), (0.5, 11.4, 1.9), (5.6, 11.8, 2.0), (-5.6, 12.4, 2.1)]
FLOWERS = [(-1.25, 1.35), (-2.0, -0.15), (0.75, 2.6)]   # rule 5: 2 to 5 clumps, 0.4 to 0.5 P wide

PLAYER = Vector((0.0, 0.0, 0.0))
FACING_DEG = -25.0            # east, turned 25 deg toward the camera so the face reads (design choice)


def build_player(col, M):
    """Original NL proportioned player, H = 1.0: head fraction 0.46 (chin at 0.54), round forms 14 x 8, limbs
    6 sided, mitten hands, separate hair shell with a zigzag fringe (sections 5.1 to 5.3)."""
    yaw = math.radians(FACING_DEG)
    R = Matrix.Rotation(yaw, 4, "Z")
    fwd = Vector((math.cos(yaw), math.sin(yaw), 0))
    side = Vector((-fwd.y, fwd.x, 0))

    def at(f, s, z):
        return PLAYER + fwd * f + side * s + Vector((0, 0, z))

    head_c = at(0, 0, 0.75)
    # head with face decal: own UVs, u around (0.5 = facing direction), v bottom to top
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    res = bmesh.ops.create_uvsphere(bm, u_segments=14, v_segments=8, radius=0.21,
                                    matrix=Matrix.Translation(head_c) @ R @ Matrix.Diagonal((1.0, 1.08, 1.0, 1.0)))
    for f in {f for v in res["verts"] for f in v.link_faces}:
        for lp in f.loops:
            d = (R.inverted() @ (lp.vert.co - head_c)).normalized()
            u = 0.5 + math.atan2(d.y, d.x) / (2 * math.pi)
            lp[uv].uv = (u, 0.5 + math.asin(max(-1, min(1, d.z))) / math.pi)
        # fix the seam behind the head
        us = [lp[uv].uv.x for lp in f.loops]
        if max(us) - min(us) > 0.5:
            for lp in f.loops:
                if lp[uv].uv.x < 0.5:
                    lp[uv].uv.x += 1.0
    mesh_object("Head", bm, col, [M["face"]])

    bm, uv = new_bm()
    hres = bmesh.ops.create_uvsphere(bm, u_segments=14, v_segments=8, radius=0.225,
                                     matrix=Matrix.Translation(head_c + Vector((0, 0, 0.025))) @ R @
                                     Matrix.Diagonal((1.0, 1.1, 1.0, 1.0)), calc_uvs=True)
    kill = []
    for f in {f for v in hres["verts"] for f in v.link_faces}:
        c = R.inverted() @ (f.calc_center_median() - head_c)
        # fringe: keep alternating front faces lower, so the shell edge itself forms the zigzag (section 5.2)
        ang = math.atan2(c.y, c.x)
        tooth = (int((ang + math.pi) / (2 * math.pi) * 14) % 2) == 0
        front_ok = c.x > 0 and c.z > (0.03 if tooth else 0.09)
        if not (c.z > 0.09 or front_ok or (c.x < -0.05 and c.z > -0.16) or
                (c.x < 0.06 and c.z > -0.04 and abs(c.y) > 0.18)):
            kill.append(f)
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    mesh_object("Hair", bm, col, [M["hair"]])

    bm, uv = new_bm()                                    # torso (shirt strip front block), arms
    bm_tube(bm, uv, [at(0, 0, 0.30), at(0, 0, 0.55)], [0.13, 0.10], 8, v0=0.75, v1=1.0, cap_end=True, cap_start=True)
    for s in (-1, 1):
        bm_tube(bm, uv, [at(0, s * 0.11, 0.50), at(0.17, s * 0.05, 0.40)], [0.04, 0.035], 6, v0=0.0, v1=0.24)
    mesh_object("Body", bm, col, [M["shirt"]])
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
        bm_sphere(bm, uv, at(0.19, s * 0.04, 0.39), 0.04, segs=10, rings=6)
    mesh_object("Hands", bm, col, [M["skin"]])

    grip = at(0.2, 0.0, 0.39)
    tip = grip + Vector((1.25, 0.25, 0.95))
    bm, uv = new_bm()
    bm_tube(bm, uv, [grip - Vector((0.1, 0.0, 0.07)), tip], [0.018, 0.007], 6, v0=0.0, v1=1.0)
    mesh_object("Rod", bm, col, [M["rod"]])
    bob = Vector((2.35, 0.55, WATER_Z + 0.02))
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
    cr.color_ramp.elements[0].color = (*lin(hexc(PAL["sky_low"])), 1)
    cr.color_ramp.elements[1].color = (*lin(hexc(PAL["sky_top"])), 1)
    bg = nt.nodes.new("ShaderNodeBackground")
    link(nt, cr.outputs["Color"], bg.inputs["Color"])
    link(nt, bg.outputs[0], out.inputs["Surface"])


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


def build(native=False):
    global LUT
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    LUT = build_lut()
    rng = np.random.default_rng(3)
    T = {"grass": tex_grass(), "wall": tex_wall(), "caustic": tex_caustic(), "foliage": tex_foliage(),
         "bark": tex_bark(), "face": tex_face(), "hair": tex_hair(), "shirt": tex_shirt(), "bobber": tex_bobber()}
    contact_alpha = 1.0 - 0.7     # trunk base patches: ratio about 0.7 (section 3), as a blend toward the tint
    M = {
        "terrain": terrain_mat(T), "water": water_mat(T),
        "foliage": mat("M_Foliage", image=T["foliage"], uv_scale=(2, 2)),
        "bark": mat("M_Bark", image=T["bark"]),
        "leaf": mat("M_Leaf", color=PAL["foliage"]), "flower_c": mat("M_FlowerC", color=PAL["flower_centre"]),
        "flower_a": mat("M_FlowerA", color=PAL["flower_a"]), "flower_b": mat("M_FlowerB", color=PAL["flower_b"]),
        "rock": mat("M_Rock", color=PAL["rock"]),
        "face": mat("M_Face", image=T["face"], rim=1.0),
        "hair": mat("M_Hair", image=T["hair"], rim=1.0),
        "shirt": mat("M_Shirt", image=T["shirt"], rim=1.0),
        "shorts": mat("M_Shorts", color=PAL["shorts"], rim=1.0),
        "skin": mat("M_Skin", color=PAL["skin"], rim=1.0),
        "shoe": mat("M_Shoe", color=PAL["shoe"], rim=1.0),
        "rod": mat("M_Rod", color=PAL["rod"]),
        "line": unlit("M_Line", "#F0F0F0"),
        "bobber": mat("M_Bobber", image=T["bobber"]),
        "foam": unlit("M_Foam", "#CFEFFF"),
        "blob": blob_mat("M_Blob", BLOB_CORE_ALPHA, BLOB_TINT_LIN),
        "contact": blob_mat("M_Contact", contact_alpha * 1.6, BLOB_TINT_LIN, inner=0.35),
    }
    world = bpy.data.collections.new("World")
    sc.collection.children.link(world)
    build_sky()
    build_terrain(world, M)
    build_water(world, M)
    # horizon tree line (section 9 rule 3) and a few nearer trees on both sides of the river (4 to 6 trees)
    for x, y, hgt in TREES:
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

    rx_, ry_ = NATIVE if native else RES
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = rx_, ry_, 100
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
        top = bend_point(Vector((x, y, terrain_h(x, y) + hgt * 0.55 + 0.98)))
        base = bend_point(Vector((x, y, terrain_h(x, y))))
        vt, vb = world_to_camera_view(sc, cam, top), world_to_camera_view(sc, cam, base)
        print(f"  tree ({x}, {y}): crown top row {1 - vt.y:.3f}, base row {1 - vb.y:.3f}, column {vt.x:.3f}")
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in sc.objects if o.type == "MESH")
    pl = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in sc.objects
             if o.type == "MESH" and o.name.split(".")[0] in ("Head", "Hair", "Body", "Shorts", "Legs", "Shoes", "Hands"))
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
    # horizon: per column, last sky row from the top (sky = blue dominant and dark red channel)
    sky = (px[..., 2] > 0.6) & (px[..., 0] < 0.45) & (px[..., 2] > px[..., 1] + 0.1)
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
    ratio = cf / np.maximum(cg, 1e-6)
    print(f"QA blob: shadow sRGB {tuple(int(c * 255) for c in cf)} vs grass {tuple(int(c * 255) for c in cg)}; "
          f"R, G ratios {ratio[0]:.2f}, {ratio[1]:.2f} (target about 0.45), B ratio {ratio[2]:.2f} (must rise, above 1)")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--render")
    p.add_argument("--native", action="store_true", help="400x240, one sample, then a 3x nearest upscale")
    p.add_argument("--save")
    p.add_argument("--grass", choices=("retail", "press"), default="retail",
                   help="retail: lime from retail emote captures; press: E3 build green (section 4.2)")
    a = p.parse_args(argv)
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
        if a.native and shutil.which("ffmpeg"):
            root, ext = os.path.splitext(out)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-vf",
                            "scale=iw*3:ih*3:flags=neighbor", f"{root}_x3{ext}"], check=True)


if __name__ == "__main__":
    main()
