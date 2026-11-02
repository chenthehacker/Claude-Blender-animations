"""Original stylised low poly fish models, built from cited real anatomy (see SOURCES.md).

Design language (this repo's own): chunky toy proportions, enlarged eyes, softened versions of each
species' real colours. Default "ps1" style: boxy faceted bodies (superellipse cross sections, flat
shading), few hundred triangles per fish, unsmoothed fin outlines,
small nearest filtered textures in 15 bit colour with ordered dither, matte materials. "--style smooth"
gives the earlier soft, glossy, higher polygon version.
Anatomy (fin positions, fin shapes, barbels, cephalofoil, colour zones) follows the sources in SOURCES.md.
Every model is 1.0 Blender unit long (tail fork to snout or bill tip is about 1.0), facing +X, Z up.

Run with the bpy module (or stock Blender: blender -b -P stylised_fish.py -- ARGS):
  /root/bpyenv/bin/python stylised_fish.py --save stylised_fish.blend --glb glb \
      --render preview.png --res 480x300 [--style ps1|smooth] [--species carp,catfish]
The ps1 preview renders without anti aliasing at low resolution, then cuts to 15 bit colour with dither
and scales up by --upscale (nearest neighbour). Vertex snapping and affine texture wobble are render time
camera effects and belong in the scene that uses these models, not in the models.
"""
import argparse
import math
import os
import sys

import bpy  # noqa: I001  (bpy must load before bmesh in the pip module)
import bmesh
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
TEX_W, TEX_H = 1024, 512   # painting resolution
# Style switch. "ps1" (default): fewer rings and segments, faceted fin outlines, 256x128 textures cut to
# 15 bit colour with 4x4 ordered dither and nearest filtering, matte unlit-looking materials (no gloss).
# "smooth": the earlier soft toy version.
STYLE = {"name": "ps1"}
PS1 = {"ring_scale": 0.45, "min_rings": 7, "segs": 8, "fin_iters": 0, "sphere": (6, 4), "tube_segs": 4,
       "tex": (256, 128),
       "box_p": 3.4,        # cross section superellipse exponent floor (2 = round, higher = boxier)
       "flat_body": True}   # faceted body shading


def ps1():
    return STYLE["name"] == "ps1"
CHUNK = 1.32   # style: bodies deeper and wider than life (toy proportions)
EYE = 2.1      # style: eyes much larger than life


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def lin(c):
    """sRGB triple (0..1) to linear, for material colour sockets."""
    c = np.asarray(c, float)
    return tuple(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))


def prof(points, t):
    xs, ys = zip(*points)
    return float(np.interp(t, xs, ys))


def smoothstep(a, b, x):
    x = np.clip((x - a) / (b - a), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def chaikin(pts, iters):
    p = np.asarray(pts, float)
    for _ in range(iters):
        q = []
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]
            q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        p = np.array(q)
    return p


class Builder:
    """Accumulates one species into a single bmesh with material slots."""

    SLOTS = ("body", "fin", "fin_accent", "eye_iris", "eye_pupil", "dark")

    def __init__(self, sp):
        self.sp = sp
        self.bm = bmesh.new()
        self.uv = self.bm.loops.layers.uv.new("UVMap")

    def mat(self, slot):
        return self.SLOTS.index(slot)

    # ---------------- body ----------------
    def surf(self, t, th):
        """Body surface point at length t (0 tail .. 1 snout) and angle th (0 belly, pi top)."""
        sp = self.sp
        w = prof(sp["w"], t) * CHUNK
        hu = prof(sp["hu"], t) * CHUNK
        hl = prof(sp["hl"], t) * CHUNK
        cz = prof(sp["cz"], t)
        p = prof(sp.get("p", [(0, 2.0), (1, 2.0)]), t)
        if ps1():
            p = max(p, PS1["box_p"])
        s, c = math.sin(th), -math.cos(th)
        y = w * math.copysign(abs(s) ** (2 / p), s)
        h = hu if c > 0 else hl
        z = cz + h * math.copysign(abs(c) ** (2 / p), c)
        return Vector((t, y, z))

    def top(self, t):
        return self.surf(t, math.pi)

    def bottom(self, t):
        return self.surf(t, 0.0)

    def side_normal(self, t, th, d=1e-3):
        a = self.surf(t, th + d) - self.surf(t, th - d)
        b = self.surf(min(t + d, 1.0), th) - self.surf(max(t - d, 0.0), th)
        n = b.cross(a).normalized()
        centre = Vector((t, 0, prof(self.sp["cz"], t)))
        if n.dot(self.surf(t, th) - centre) < 0:
            n = -n
        return n

    def body(self):
        sp, bm, uv = self.sp, self.bm, self.uv
        R, N = sp.get("rings", 16), sp.get("segs", 12)
        if ps1():
            R, N = max(PS1["min_rings"], round(R * PS1["ring_scale"])), PS1["segs"]
        t0, t1 = 0.0, sp.get("t_end", 0.97)
        ts = t0 + (t1 - t0) * (1 - (1 - np.linspace(0, 1, R)) ** 1.15)
        rings = [[bm.verts.new(self.surf(t, 2 * math.pi * k / N)) for k in range(N)] for t in ts]
        faces = []

        def put(f, uvs):
            f.material_index = self.mat("body")
            f.smooth = not (ps1() and PS1["flat_body"])
            for loop, (u, v) in zip(f.loops, uvs):
                loop[uv].uv = (u, v)
            faces.append(f)

        for i in range(R - 1):
            for k in range(N):
                k2 = (k + 1) % N
                v1, v2 = k / N, (k + 1) / N
                f = bm.faces.new((rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]))
                put(f, [(ts[i], v1), (ts[i], v2), (ts[i + 1], v2), (ts[i + 1], v1)])
        tail = bm.verts.new((t0 - 0.004, 0, prof(sp["cz"], t0)))
        dz = sp.get("tip_dz", 0.0)
        tip = bm.verts.new((t1 + sp.get("tip_ext", 0.03), 0, prof(sp["cz"], t1) + dz))
        for k in range(N):
            k2 = (k + 1) % N
            v1, v2 = k / N, (k + 1) / N
            put(bm.faces.new((rings[0][k2], rings[0][k], tail)), [(t0, v2), (t0, v1), (0.0, (v1 + v2) / 2)])
            put(bm.faces.new((rings[-1][k], rings[-1][k2], tip)), [(t1, v1), (t1, v2), (1.0, (v1 + v2) / 2)])
        return faces

    # ---------------- parts ----------------
    def fin(self, pts2d, origin, ax_a, ax_b, slot="fin", thick=0.010, iters=2):
        bm = self.bm
        p2 = chaikin(pts2d, min(iters, PS1["fin_iters"]) if ps1() else iters)
        o, a, b = Vector(origin), Vector(ax_a).normalized(), Vector(ax_b).normalized()
        n = a.cross(b).normalized()
        front, back = [], []
        for x, y in p2:
            q = o + a * x + b * y
            front.append(bm.verts.new(q + n * thick / 2))
            back.append(bm.verts.new(q - n * thick / 2))
        faces = [bm.faces.new(front), bm.faces.new(list(reversed(back)))]
        m = len(p2)
        for i in range(m):
            j = (i + 1) % m
            faces.append(bm.faces.new((front[j], front[i], back[i], back[j])))
        for f in faces:
            f.material_index = self.mat(slot)
            f.smooth = False
            for loop in f.loops:
                loop[self.uv].uv = (0.5, 0.5)
        bmesh.ops.triangulate(self.bm, faces=faces[:2])

    def mirror_fin(self, pts2d, origin, ax_a, ax_b, **kw):
        """Paired fin: build on +Y, then mirror across XZ."""
        self.fin(pts2d, origin, ax_a, ax_b, **kw)
        o = Vector(origin)
        self.fin(pts2d, (o.x, -o.y, o.z), (ax_a[0], -ax_a[1], ax_a[2]), (ax_b[0], -ax_b[1], ax_b[2]), **kw)

    def sphere(self, centre, radius, slot, segs=12, rings=8, scale=(1, 1, 1), uv=(0.5, 0.5)):
        if ps1():
            segs, rings = PS1["sphere"]
        m = Matrix.LocRotScale(Vector(centre), None, Vector(scale) * radius)
        res = bmesh.ops.create_uvsphere(self.bm, u_segments=segs, v_segments=rings, radius=1.0, matrix=m)
        fs = {f for v in res["verts"] for f in v.link_faces}
        for f in fs:
            f.material_index = self.mat(slot)
            f.smooth = not ps1()
            for loop in f.loops:
                loop[self.uv].uv = uv
        return res["verts"]

    def eyes(self, t, th, r, at=None, normal=None):
        """Pair of eyes. Position on the body surface (t, th) unless an explicit point is given."""
        r *= EYE
        for side in (1, -1):
            if at is None:
                p = self.surf(t, th)
                n = self.side_normal(t, th)
            else:
                p, n = Vector(at), Vector(normal).normalized()
            if side < 0:
                p = Vector((p.x, -p.y, p.z))
                n = Vector((n.x, -n.y, n.z))
            self.sphere(p - n * 0.25 * r, r, "eye_iris")
            self.sphere(p + n * 0.42 * r, 0.62 * r, "eye_pupil", segs=10, rings=6)

    def tube(self, pts, r0, r1, slot="fin", segs=6, uv=(0.5, 0.5)):
        """Tapered tube along a polyline (barbels, bills)."""
        bm = self.bm
        if ps1():
            segs = min(segs, PS1["tube_segs"])
        pts = [Vector(p) for p in pts]
        prev = None
        n = len(pts)
        for i, p in enumerate(pts):
            d = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
            up = Vector((0, 0, 1)) if abs(d.z) < 0.9 else Vector((0, 1, 0))
            a = d.cross(up).normalized()
            b = d.cross(a).normalized()
            r = r0 + (r1 - r0) * i / (n - 1)
            ring = [bm.verts.new(p + (a * math.cos(2 * math.pi * k / segs) + b * math.sin(2 * math.pi * k / segs)) * r)
                    for k in range(segs)]
            if prev is not None:
                for k in range(segs):
                    k2 = (k + 1) % segs
                    f = bm.faces.new((prev[k], prev[k2], ring[k2], ring[k]))
                    f.material_index = self.mat(slot)
                    f.smooth = not ps1()
                    for loop in f.loops:
                        loop[self.uv].uv = uv
            prev = ring
        tipv = bm.verts.new(pts[-1] + (pts[-1] - pts[-2]).normalized() * r1)
        for k in range(segs):
            f = bm.faces.new((prev[k], prev[(k + 1) % segs], tipv))
            f.material_index = self.mat(slot)
            f.smooth = not ps1()
            for loop in f.loops:
                loop[self.uv].uv = uv

    def slab(self, outline_xy, z_of, thick_of, slot="body", iters=2):
        """Horizontal slab from a top view outline (cephalofoil). z_of/thick_of take (x, y)."""
        bm = self.bm
        p2 = chaikin(outline_xy, min(iters, PS1["fin_iters"]) if ps1() else iters)
        top, bot = [], []
        for x, y in p2:
            z, th = z_of(x, y), thick_of(x, y)
            top.append(bm.verts.new((x, y, z + th / 2)))
            bot.append(bm.verts.new((x, y, z - th / 2)))
        faces = [bm.faces.new(top), bm.faces.new(list(reversed(bot)))]
        m = len(p2)
        for i in range(m):
            j = (i + 1) % m
            faces.append(bm.faces.new((top[j], top[i], bot[i], bot[j])))
        for f in faces:
            f.material_index = self.mat(slot)
            f.smooth = not ps1()
            for loop in f.loops:
                above = loop.vert.co.z > z_of(loop.vert.co.x, loop.vert.co.y)
                loop[self.uv].uv = (0.9, 0.5 if above else 0.01)
        bmesh.ops.triangulate(self.bm, faces=faces[:2])

    def finish(self, name, mats):
        bm = self.bm
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        for s in self.SLOTS:
            me.materials.append(mats[s])
        ob = bpy.data.objects.new(name, me)
        return ob


# --------------------------------------------------------------------------
# Texture painting (sRGB values, UV: u along body tail 0 .. snout 1, v around: 0 belly, 0.5 top)
# --------------------------------------------------------------------------
def uv_grid():
    u = (np.arange(TEX_W) + 0.5) / TEX_W
    v = (np.arange(TEX_H) + 0.5) / TEX_H
    U, V = np.meshgrid(u, v)
    D = 1.0 - np.abs(2 * V - 1.0)   # 0 at belly, 1 at the top of the back
    S = np.where(V < 0.5, 1.0, -1.0)  # which flank
    return U, V, D, S


def countershade(back, belly, D, lo=0.35, hi=0.70, mid=None):
    k = smoothstep(lo, hi, D)[..., None]
    img = belly * (1 - k) + back * k
    if mid is not None:  # flank band colour between belly and back
        band = np.exp(-((D - (lo + hi) / 2) / 0.12) ** 2)[..., None] * 0.55
        img = img * (1 - band) + mid * band
    return img


def mouth_line(img, U, V, u0, v_mouth, width=0.012, colour=(0.18, 0.12, 0.12), curve=0.0):
    """Mouth gape drawn on the head, on both flanks."""
    for vm in (v_mouth, 1 - v_mouth):
        target = vm + curve * (U - u0) * (1 if vm < 0.5 else -1)
        m = (U > u0) & (np.abs(V - target) < width)
        img[m] = img[m] * 0.25 + np.array(colour) * 0.75
    return img


def paint_hammerhead(U, V, D, S):
    return countershade(hexc("8a8f87"), hexc("f1efe6"), D, 0.38, 0.62)


def paint_marlin(U, V, D, S):
    img = countershade(hexc("27466f"), hexc("e6eef2"), D, 0.30, 0.66, mid=hexc("9fb6c9"))
    # about 15 bars of pale cobalt dots along the flanks
    cobalt = hexc("8fc4ee")
    for i in range(15):
        uc = 0.25 + 0.48 * i / 14
        for j in range(6):
            dc = 0.40 + 0.075 * j
            for vc in (dc / 2, 1 - dc / 2):
                m = (((U - uc) * 2.2) ** 2 + (V - vc) ** 2) < 0.011 ** 2
                img[m] = cobalt
    return img


def paint_seabass(U, V, D, S):
    img = countershade(hexc("3f5a63"), hexc("eef1f0"), D, 0.34, 0.66, mid=hexc("b9c6c8"))
    head = smoothstep(0.80, 0.86, U)[..., None] * smoothstep(0.55, 0.75, D)[..., None]
    img = img * (1 - 0.35 * head) + hexc("2f4f57") * 0.35 * head
    return mouth_line(img, U, V, 0.93, 0.33, width=0.010, curve=-0.8)


def paint_catfish(U, V, D, S):
    img = countershade(hexc("5d6448"), hexc("e9e2cf"), D, 0.30, 0.62)
    rng = np.random.default_rng(4)
    mott = np.zeros_like(U)
    for _ in range(140):
        uc, vc = rng.uniform(0.02, 0.95), rng.uniform(0.18, 0.82)
        r = rng.uniform(0.012, 0.03)
        mott += rng.choice([-1, 1]) * np.exp(-(((U - uc) * 2.0) ** 2 + (V - vc) ** 2) / r ** 2)
    k = (np.clip(mott, -1, 1) * smoothstep(0.25, 0.5, D))[..., None]
    img = np.where(k > 0, img * (1 - 0.35 * k), img + (hexc("d8d2b8") - img) * (-k) * 0.45)
    return mouth_line(img, U, V, 0.94, 0.40, width=0.012)


def paint_carp(U, V, D, S):
    img = countershade(hexc("6b6a3c"), hexc("f2e7bf"), D, 0.30, 0.70, mid=hexc("c9a85a"))
    # large dark edged scales in a staggered grid, darker spot at each scale base, none on the head
    nu, nv = 26.0, 22.0
    gu, gv = U * nu, V * nv
    row = np.floor(gv)
    gu2 = gu + 0.5 * (row % 2)
    fu, fv = gu2 - np.floor(gu2), gv - row
    edge = np.maximum(np.abs(fu - 0.5) * 2, np.abs(fv - 0.5) * 2 * 0.8 + (fu - 0.5) * 0.4)
    scales = smoothstep(0.78, 0.95, edge) * smoothstep(0.88, 0.80, U) * smoothstep(0.08, 0.14, U)
    base = np.exp(-(((fu - 0.15) / 0.12) ** 2 + ((fv - 0.5) / 0.22) ** 2)) * 0.6
    dark = (np.clip(scales + base * smoothstep(0.88, 0.80, U) * smoothstep(0.08, 0.14, U), 0, 1)
            * (0.35 + 0.65 * smoothstep(0.15, 0.5, D)))[..., None]
    img = img * (1 - 0.45 * dark)
    return img


PAINT = {"hammerhead": paint_hammerhead, "blue_marlin": paint_marlin, "sea_bass": paint_seabass,
         "catfish": paint_catfish, "carp": paint_carp}


BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0 - 0.47


def quantise15(img):
    """Cut to 5 bits per channel with a 4x4 ordered dither (the classic 15 bit console look)."""
    h, w = img.shape[:2]
    d = np.tile(BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w][..., None]
    return np.clip(np.round(img * 31 + d), 0, 31) / 31


def ps1_texture(img):
    tw, th = PS1["tex"]
    fy, fx = img.shape[0] // th, img.shape[1] // tw
    small = img[:th * fy, :tw * fx].reshape(th, fy, tw, fx, 3).mean(axis=(1, 3))
    return quantise15(small)


def body_image(key):
    U, V, D, S = uv_grid()
    img = np.clip(PAINT[key](U, V, D, S), 0, 1)
    if ps1():
        img = ps1_texture(img)
    os.makedirs(os.path.join(HERE, "textures"), exist_ok=True)
    path = os.path.join(HERE, "textures", f"{key}_body.png")
    im = bpy.data.images.new(f"{key}_body", img.shape[1], img.shape[0], alpha=False)
    rgba = np.concatenate([img, np.ones(img.shape[:2] + (1,))], axis=2).astype(np.float32)
    im.pixels.foreach_set(rgba.ravel())
    im.filepath_raw = path
    im.file_format = "PNG"
    im.save()
    im.pack()
    return im


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
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])
    return m, nt, b


def body_material(key, img):
    m, nt, b = new_mat(f"{key}_body")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest" if ps1() else "Linear"
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    if ps1():
        b.inputs["Roughness"].default_value = 1.0
        b.inputs["Specular IOR Level"].default_value = 0.0
    else:
        b.inputs["Roughness"].default_value = 0.55
        b.inputs["Specular IOR Level"].default_value = 0.35
        b.inputs["Coat Weight"].default_value = 0.15
        b.inputs["Coat Roughness"].default_value = 0.35
    return m


def flat_material(name, srgb, rough=0.6, coat=0.0):
    m, nt, b = new_mat(name)
    b.inputs["Base Color"].default_value = (*lin(srgb), 1.0)
    if ps1():
        srgb = quantise15(np.asarray(srgb, float)[None, None, :])[0, 0]
        b.inputs["Base Color"].default_value = (*lin(srgb), 1.0)
        rough, coat = 1.0, 0.0
        b.inputs["Specular IOR Level"].default_value = 0.0
    b.inputs["Roughness"].default_value = rough
    b.inputs["Coat Weight"].default_value = coat
    m.diffuse_color = (*lin(srgb), 1.0)
    return m


def species_materials(key, sp):
    img = body_image(key)
    c = sp["colours"]
    return {
        "body": body_material(key, img),
        "fin": flat_material(f"{key}_fin", hexc(c["fin"])),
        "fin_accent": flat_material(f"{key}_fin_accent", hexc(c.get("fin_accent", c["fin"]))),
        "eye_iris": flat_material(f"{key}_iris", hexc(c.get("iris", "c9b27a")), rough=0.3, coat=0.6),
        "eye_pupil": flat_material(f"{key}_pupil", hexc("141416"), rough=0.08, coat=1.0),
        "dark": flat_material(f"{key}_dark", hexc(c.get("dark", "2a2422")), rough=0.5),
    }


# --------------------------------------------------------------------------
# Species. Positions in units of total length (tail 0 .. snout 1). Anatomy notes cite SOURCES.md.
# --------------------------------------------------------------------------
SPECIES = {}


def species(key):
    def deco(fn):
        SPECIES[key] = fn
        return fn
    return deco


@species("hammerhead")
def build_hammerhead():
    # Great hammerhead (Sphyrna mokarran). [H1] [H2]
    sp = {
        "rings": 18, "segs": 12, "t_end": 0.93, "tip_ext": 0.01,
        "w": [(0, 0.012), (0.2, 0.03), (0.45, 0.056), (0.65, 0.06), (0.8, 0.04), (0.88, 0.024), (0.93, 0.018)],
        "hu": [(0, 0.012), (0.2, 0.03), (0.45, 0.06), (0.62, 0.068), (0.82, 0.05), (0.93, 0.022)],
        "hl": [(0, 0.010), (0.2, 0.025), (0.45, 0.05), (0.62, 0.052), (0.82, 0.035), (0.93, 0.016)],
        "cz": [(0, 0.03), (0.3, 0.005), (0.6, 0.0), (1, 0.0)],
        "p": [(0, 2.0), (0.6, 2.1), (0.93, 2.6)],
        "colours": {"fin": "858a82", "fin_accent": "7c8179", "iris": "b9a46a"},
    }
    B = Builder(sp)
    B.body()
    # Cephalofoil: width about 23 to 27 percent of body length, nearly straight front margin with
    # a central (medial) notch and lateral indentations [H1]. Half width 0.125 = 25 percent.
    hw = 0.125
    xf = 0.985
    outline = [(xf - 0.012, 0.0), (xf, 0.018), (xf + 0.002, 0.06), (xf - 0.004, 0.085), (xf, 0.11),
               (xf - 0.006, hw), (xf - 0.03, hw + 0.002), (xf - 0.05, hw - 0.01), (xf - 0.06, 0.09),
               (xf - 0.085, 0.05), (xf - 0.11, 0.032), (xf - 0.11, -0.032), (xf - 0.085, -0.05),
               (xf - 0.06, -0.09), (xf - 0.05, -hw + 0.01), (xf - 0.03, -hw - 0.002), (xf - 0.006, -hw),
               (xf, -0.11), (xf - 0.004, -0.085), (xf + 0.002, -0.06), (xf, -0.018)]
    B.slab(outline, lambda x, y: 0.004 - 0.02 * (abs(y) / hw) ** 2,
           lambda x, y: 0.034 * (1 - 0.45 * abs(y) / hw), iters=1)
    # Eyes on the outer tips of the hammer
    B.eyes(0, 0, 0.011, at=(xf - 0.03, hw + 0.004, -0.012), normal=(0.2, 1.0, 0.25))
    # First dorsal: very tall, strongly falcate, origin over the pectoral insertions, free rear
    # tip ending in front of the pelvic fin origins [H1].
    x0 = 0.66
    zt = B.top(x0).z
    B.fin([(0.0, -0.02), (0.03, 0.06), (0.01, 0.17), (-0.04, 0.23), (-0.045, 0.20), (-0.06, 0.06),
           (-0.11, 0.012), (-0.11, -0.02)], (x0, 0, zt), (1, 0, 0), (0, 0, 1), iters=2, thick=0.012)
    # Second dorsal and anal fins relatively large, with deeply notched rear margins [H1]
    x2 = 0.30
    B.fin([(0.0, -0.015), (0.02, 0.03), (-0.005, 0.065), (-0.02, 0.03), (-0.06, 0.012), (-0.06, -0.015)],
          (x2, 0, B.top(x2).z), (1, 0, 0), (0, 0, 1), iters=1)
    xa = 0.31
    B.fin([(0.0, 0.015), (0.02, -0.03), (-0.005, -0.07), (-0.022, -0.032), (-0.06, -0.012), (-0.06, 0.015)],
          (xa, 0, B.bottom(xa).z), (1, 0, 0), (0, 0, 1), iters=1)
    # Caudal: long upper lobe (heterocercal shark tail; style choice, see SOURCES.md)
    B.fin([(0.03, 0.01), (-0.05, 0.08), (-0.13, 0.17), (-0.115, 0.16), (-0.07, 0.07), (-0.03, 0.025),
           (-0.035, -0.02), (-0.06, -0.075), (-0.035, -0.07), (0.03, -0.012)],
          (0.02, 0, 0.03), (1, 0, 0), (0, 0, 1), iters=2, thick=0.012)
    # Pectorals just behind the gills, pelvics in front of the anal fin
    xp = 0.70
    pp = B.surf(xp, math.pi * 0.28)
    B.mirror_fin([(0.0, 0.0), (-0.02, 0.13), (-0.06, 0.15), (-0.07, 0.02), (-0.04, -0.005)],
                 pp, (1, 0, 0), (0, 0.85, -0.5), thick=0.011)
    xv = 0.47
    pv = B.surf(xv, math.pi * 0.12)
    B.mirror_fin([(0.0, 0.0), (-0.015, 0.05), (-0.045, 0.055), (-0.05, 0.005)],
                 pv, (1, 0, 0), (0, 0.6, -0.8), thick=0.009)
    return B, sp


@species("blue_marlin")
def build_marlin():
    # Blue marlin (Makaira nigricans). [M1] [M2]
    sp = {
        "rings": 18, "segs": 12, "t_end": 0.80, "tip_ext": 0.025, "tip_dz": 0.0,
        "w": [(0, 0.012), (0.12, 0.02), (0.35, 0.05), (0.55, 0.066), (0.72, 0.058), (0.80, 0.035)],
        "hu": [(0, 0.012), (0.12, 0.025), (0.35, 0.065), (0.58, 0.08), (0.72, 0.065), (0.80, 0.03)],
        "hl": [(0, 0.010), (0.12, 0.022), (0.35, 0.06), (0.55, 0.072), (0.72, 0.055), (0.80, 0.028)],
        "cz": [(0, 0.0), (1, 0.0)],
        "colours": {"fin": "3b3a40", "fin_accent": "23395c", "iris": "3d4652"},
    }
    B = Builder(sp)
    B.body()
    # Bill: very stout, round in cross section [M1]. Length is a style choice.
    tip = 0.80 + 0.025
    B.tube([(tip - 0.02, 0, 0.004), (tip + 0.06, 0, 0.004), (tip + 0.13, 0, 0.003), (1.0, 0, 0.002)],
           0.017, 0.004, slot="body", segs=8, uv=(0.78, 0.45))
    B.eyes(0.74, math.pi * 0.62, 0.012)
    # First dorsal: long base, front lobe lower than the greatest body depth [M1]; dark blue [M2]
    x0 = 0.66
    B.fin([(0.0, -0.02), (0.02, 0.11), (-0.02, 0.12), (-0.06, 0.06), (-0.14, 0.04), (-0.26, 0.028),
           (-0.30, 0.012), (-0.30, -0.02)], (x0, 0, B.top(x0).z), (1, 0, 0), (0, 0, 1),
          slot="fin_accent", iters=2)
    # Small second dorsal and second anal near the tail [M1]
    for z_of, sgn in ((B.top, 1), (B.bottom, -1)):
        x = 0.15
        B.fin([(0.0, -0.01 * sgn), (0.005, 0.03 * sgn), (-0.02, 0.028 * sgn), (-0.03, -0.01 * sgn)],
              (x, 0, z_of(x).z), (1, 0, 0), (0, 0, 1), iters=1)
    xa = 0.40
    B.fin([(0.0, 0.015), (0.01, -0.07), (-0.03, -0.065), (-0.07, -0.02), (-0.08, 0.015)],
          (xa, 0, B.bottom(xa).z), (1, 0, 0), (0, 0, 1), iters=2)
    # Caudal: tall crescent (style choice, see SOURCES.md)
    B.fin([(0.04, 0.0), (0.0, 0.04), (-0.06, 0.15), (-0.085, 0.15), (-0.04, 0.05), (-0.03, 0.0),
           (-0.04, -0.05), (-0.085, -0.15), (-0.06, -0.15), (0.0, -0.04)],
          (0.02, 0, 0.0), (1, 0, 0), (0, 0, 1), iters=2, thick=0.012)
    # Pectorals long and narrow [M1], held out rigidly [M2]
    xp = 0.67
    pp = B.surf(xp, math.pi * 0.30)
    B.mirror_fin([(0.0, 0.0), (-0.06, 0.12), (-0.08, 0.13), (-0.04, 0.0)],
                 pp, (1, 0, 0), (0, 0.55, -0.83), thick=0.009)
    # Pelvics: slender (style)
    xv = 0.62
    pv = B.surf(xv, math.pi * 0.05)
    B.mirror_fin([(0.0, 0.0), (-0.07, 0.03), (-0.075, 0.02), (-0.01, -0.004)],
                 pv, (1, 0, 0), (0, 0.3, -0.95), thick=0.006, iters=1)
    return B, sp


@species("sea_bass")
def build_seabass():
    # Japanese sea bass (Lateolabrax japonicus). [S1] [S2] [S3]
    sp = {
        "rings": 16, "segs": 12, "t_end": 0.95, "tip_ext": 0.02, "tip_dz": -0.006,
        "w": [(0, 0.012), (0.12, 0.02), (0.4, 0.05), (0.6, 0.056), (0.8, 0.05), (0.95, 0.02)],
        "hu": [(0, 0.02), (0.12, 0.035), (0.4, 0.085), (0.6, 0.095), (0.8, 0.075), (0.95, 0.02)],
        "hl": [(0, 0.02), (0.12, 0.032), (0.4, 0.075), (0.6, 0.08), (0.8, 0.065), (0.95, 0.03)],
        "cz": [(0, 0.0), (0.8, 0.0), (0.95, -0.01)],
        "colours": {"fin": "6f8389", "fin_accent": "556b72", "iris": "d6cfa8"},
    }
    B = Builder(sp)
    B.body()
    # Large mouth with the lower jaw projecting beyond the upper [S1]
    tz = 0.95
    B.sphere((tz + 0.012, 0, -0.035 * CHUNK), 0.022, "body", segs=10, rings=6, scale=(1.6, 1.0, 0.45),
             uv=(0.9, 0.22))
    B.eyes(0.88, math.pi * 0.66, 0.012)
    # Spiny first dorsal (12 to 15 spines) then a separate soft second dorsal (12 to 14 rays) [S2]
    x0 = 0.68
    spikes = [(0.0, -0.02)]
    for i in range(7):
        x = -0.02 * i
        spikes += [(x - 0.004, 0.06 + 0.01 * math.sin(i / 6 * math.pi)), (x - 0.012, 0.035)]
    spikes += [(-0.15, 0.012), (-0.15, -0.02)]
    B.fin(spikes, (x0, 0, B.top(x0).z), (1, 0, 0), (0, 0, 1), slot="fin_accent", iters=1)
    x2 = 0.48
    B.fin([(0.0, -0.02), (-0.005, 0.07), (-0.04, 0.06), (-0.14, 0.03), (-0.15, -0.02)],
          (x2, 0, B.top(x2).z), (1, 0, 0), (0, 0, 1), iters=2)
    xa = 0.36
    B.fin([(0.0, 0.02), (-0.005, -0.06), (-0.04, -0.055), (-0.10, -0.02), (-0.10, 0.02)],
          (xa, 0, B.bottom(xa).z), (1, 0, 0), (0, 0, 1), iters=2)
    # Slightly forked tail [S1]
    B.fin([(0.03, 0.0), (0.0, 0.03), (-0.09, 0.105), (-0.11, 0.09), (-0.085, 0.01), (-0.085, -0.01),
           (-0.11, -0.09), (-0.09, -0.105), (0.0, -0.03)], (0.02, 0, 0.0), (1, 0, 0), (0, 0, 1),
          iters=2, thick=0.012)
    xp = 0.76
    B.mirror_fin([(0.0, 0.0), (-0.05, 0.06), (-0.085, 0.045), (-0.04, -0.005)],
                 B.surf(xp, math.pi * 0.38), (1, 0, 0), (0, 0.6, -0.8), thick=0.008)
    B.mirror_fin([(0.0, 0.0), (-0.05, 0.05), (-0.07, 0.035), (-0.03, -0.004)],
                 B.surf(0.70, math.pi * 0.06), (1, 0, 0), (0, 0.35, -0.94), thick=0.008)
    return B, sp


@species("catfish")
def build_catfish():
    # Amur catfish (Silurus asotus). [C1] [C2]
    sp = {
        "rings": 18, "segs": 12, "t_end": 0.96, "tip_ext": 0.012, "tip_dz": 0.0,
        "w": [(0, 0.01), (0.2, 0.035), (0.5, 0.055), (0.75, 0.07), (0.9, 0.075), (0.96, 0.05)],
        "hu": [(0, 0.03), (0.2, 0.05), (0.5, 0.066), (0.72, 0.062), (0.9, 0.042), (0.96, 0.022)],
        "hl": [(0, 0.03), (0.2, 0.05), (0.5, 0.06), (0.72, 0.052), (0.9, 0.032), (0.96, 0.018)],
        "cz": [(0, 0.0), (1, 0.0)],
        "p": [(0, 2.0), (0.7, 2.0), (0.96, 2.8)],
        "colours": {"fin": "75795c", "fin_accent": "6c7155", "iris": "a59a6a", "dark": "3a3a30"},
    }
    B = Builder(sp)
    B.body()
    B.eyes(0.905, math.pi * 0.70, 0.0085)
    # Barbels: two pairs; maxillary longer than the head, mandibular about 1/5 to 1/3 of it [C1]
    head = 0.18
    for s in (1, -1):
        a = B.surf(0.95, math.pi * 0.42)
        a = Vector((a.x, s * a.y, a.z))
        pts = [a]
        for i in range(1, 7):
            k = i / 6
            pts.append(a + Vector((-0.12 * k * head / 0.18, s * (0.06 + 0.10 * k), -0.03 * k * k)))
        B.tube(pts, 0.006, 0.0018, slot="dark")
        b = B.surf(0.95, math.pi * 0.15)
        b = Vector((b.x, s * b.y * 0.6, b.z))
        B.tube([b, b + Vector((-0.012, s * 0.02, -0.03)), b + Vector((-0.02, s * 0.028, -0.05))],
               0.003, 0.0012, slot="dark")
    # Small dorsal fin (about 4 soft rays) [C2]
    x0 = 0.68
    B.fin([(0.0, -0.015), (0.0, 0.045), (-0.02, 0.05), (-0.04, 0.02), (-0.04, -0.015)],
          (x0, 0, B.top(x0).z), (1, 0, 0), (0, 0, 1), iters=2)
    # Very long anal fin, joined to the caudal fin [C1] [C2]
    lower = [(0.62, B.bottom(0.62).z + 0.012), (0.60, B.bottom(0.60).z - 0.03)]
    for i in range(1, 9):
        x = 0.60 - i * 0.07
        lower.append((x, B.bottom(max(x, 0.0)).z - 0.028))
    tail = [(-0.035, -0.03), (-0.05, 0.0), (-0.035, 0.03), (0.0, B.top(0.0).z + 0.03),
            (0.06, B.top(0.06).z + 0.012), (0.08, B.top(0.08).z - 0.004)]
    inner = [(x, B.bottom(x).z + 0.01) for x in (0.06, 0.25, 0.45)]
    B.fin(lower + tail + [(0.06, 0.0)] + inner, (0, 0, 0), (1, 0, 0), (0, 0, 1), iters=2, thick=0.010)
    xp = 0.83
    B.mirror_fin([(0.0, 0.0), (-0.05, 0.05), (-0.075, 0.035), (-0.035, -0.004)],
                 B.surf(xp, math.pi * 0.25), (1, 0, 0), (0, 0.7, -0.7), thick=0.008)
    B.mirror_fin([(0.0, 0.0), (-0.03, 0.035), (-0.05, 0.02), (-0.02, -0.004)],
                 B.surf(0.62, math.pi * 0.08), (1, 0, 0), (0, 0.4, -0.9), thick=0.007)
    return B, sp


@species("carp")
def build_carp():
    # Common carp (Cyprinus carpio). [K1] [K2] [K3]
    sp = {
        "rings": 16, "segs": 12, "t_end": 0.95, "tip_ext": 0.015, "tip_dz": -0.025,
        "w": [(0, 0.014), (0.15, 0.03), (0.45, 0.07), (0.65, 0.075), (0.85, 0.06), (0.95, 0.03)],
        "hu": [(0, 0.024), (0.15, 0.045), (0.45, 0.12), (0.62, 0.13), (0.82, 0.09), (0.95, 0.03)],
        "hl": [(0, 0.022), (0.15, 0.04), (0.45, 0.085), (0.62, 0.09), (0.82, 0.07), (0.95, 0.03)],
        "cz": [(0, 0.0), (0.8, 0.0), (0.95, -0.02)],
        "colours": {"fin": "8a7448", "fin_accent": "a8644a", "iris": "d8b25a", "dark": "6d5a3c"},
    }
    B = Builder(sp)
    B.body()
    B.eyes(0.88, math.pi * 0.64, 0.0105)
    # Downturned mouth (tip_dz) with two pairs of barbels on the upper jaw [K1] [K3]
    for s in (1, -1):
        a = B.surf(0.95, math.pi * 0.33)
        a = Vector((a.x + 0.01, s * a.y * 0.9, a.z - 0.004))
        B.tube([a, a + Vector((-0.005, s * 0.012, -0.022)), a + Vector((-0.012, s * 0.016, -0.04))],
               0.0035, 0.0015, slot="dark")
        c = Vector((a.x + 0.012, s * a.y * 0.6, a.z + 0.006))
        B.tube([c, c + Vector((0.006, s * 0.008, -0.014))], 0.0025, 0.001, slot="dark")
    # Long dorsal fin with a stout serrated leading spine [K1] [K2]
    x0 = 0.62
    zt = B.top(x0).z
    B.fin([(0.0, -0.02), (0.012, 0.03), (-0.01, 0.095), (-0.04, 0.08), (-0.12, 0.05), (-0.25, 0.035),
           (-0.29, 0.02), (-0.30, -0.02)], (x0, 0, zt), (1, 0, 0), (0, 0, 1), iters=2)
    B.tube([(x0 + 0.004, 0, zt - 0.005), (x0 - 0.003, 0, zt + 0.05), (x0 - 0.012, 0, zt + 0.09)],
           0.006, 0.002, slot="dark", segs=5)
    # Anal fin with a serrated leading spine [K1] [K2]; lower fins often reddish [K3]
    xa = 0.33
    za = B.bottom(xa).z
    B.fin([(0.0, 0.02), (0.005, -0.06), (-0.025, -0.07), (-0.06, -0.03), (-0.07, 0.02)],
          (xa, 0, za), (1, 0, 0), (0, 0, 1), slot="fin_accent", iters=2)
    B.tube([(xa + 0.003, 0, za + 0.005), (xa + 0.004, 0, za - 0.035), (xa - 0.004, 0, za - 0.062)],
           0.004, 0.0015, slot="dark", segs=5)
    # Forked caudal [K2]
    B.fin([(0.03, 0.0), (0.0, 0.035), (-0.08, 0.13), (-0.11, 0.12), (-0.075, 0.02), (-0.075, -0.02),
           (-0.11, -0.12), (-0.08, -0.13), (0.0, -0.035)], (0.02, 0, 0.0), (1, 0, 0), (0, 0, 1),
          iters=2, thick=0.012)
    B.mirror_fin([(0.0, 0.0), (-0.05, 0.065), (-0.085, 0.05), (-0.04, -0.005)],
                 B.surf(0.78, math.pi * 0.22), (1, 0, 0), (0, 0.6, -0.8), slot="fin_accent", thick=0.008)
    B.mirror_fin([(0.0, 0.0), (-0.045, 0.055), (-0.07, 0.035), (-0.03, -0.004)],
                 B.surf(0.52, math.pi * 0.06), (1, 0, 0), (0, 0.35, -0.94), slot="fin_accent", thick=0.008)
    return B, sp


# --------------------------------------------------------------------------
# Scene, export, preview
# --------------------------------------------------------------------------
def build_all(keys):
    col = bpy.data.collections.new("StylisedFish")
    bpy.context.scene.collection.children.link(col)
    obs = {}
    for key in keys:
        B, sp = SPECIES[key]()
        mats = species_materials(key, sp)
        ob = B.finish(f"Fish_{key}", mats)
        ob["model_length_units"] = 1.0
        ob["sources"] = "assets/stylised_fish/SOURCES.md"
        col.objects.link(ob)
        obs[key] = ob
    return obs


def report(obs):
    print("\n=== MODEL REPORT ===")
    for key, ob in obs.items():
        me = ob.data
        xs = [v.co.x for v in me.vertices]
        ys = [v.co.y for v in me.vertices]
        zs = [v.co.z for v in me.vertices]
        print(f"{key:12s} verts {len(me.vertices):5d} tris {sum(len(p.vertices) - 2 for p in me.polygons):5d}  "
              f"x {min(xs):+.3f}..{max(xs):+.3f}  y {min(ys):+.3f}..{max(ys):+.3f}  z {min(zs):+.3f}..{max(zs):+.3f}")


def export_glb(obs, outdir):
    os.makedirs(outdir, exist_ok=True)
    for key, ob in obs.items():
        for o in bpy.context.scene.objects:
            o.select_set(False)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        path = os.path.join(outdir, f"{key}.glb")
        bpy.ops.export_scene.gltf(filepath=path, use_selection=True, export_format="GLB",
                                  export_apply=True, export_yup=True)
        print("exported", path)


def ps1_post(path, upscale):
    """Preview post: 15 bit colour with ordered dither, then a nearest neighbour upscale."""
    im = bpy.data.images.load(path)
    w, h = im.size
    px = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4)
    rgb = quantise15(px[..., :3])
    rgb = np.repeat(np.repeat(rgb, upscale, axis=0), upscale, axis=1)
    out = bpy.data.images.new("ps1_post", w * upscale, h * upscale, alpha=False)
    out.pixels.foreach_set(np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], axis=2).astype(np.float32).ravel())
    out.filepath_raw = path
    out.file_format = "PNG"
    out.save()


def preview_scene(obs, res, samples):
    sc = bpy.context.scene
    order = ["blue_marlin", "hammerhead", "sea_bass", "carp", "catfish"]
    order = [k for k in order if k in obs] + [k for k in obs if k not in order]
    cols = 2
    for i, key in enumerate(order):
        ob = obs[key]
        r, c = divmod(i, cols)
        ob.location = (c * 1.35 - 0.6 * (cols - 1) - 0.5, 0, -r * 0.62)
        ob.rotation_euler = (0, 0, math.radians(-32))
    rows = (len(order) + cols - 1) // cols
    cam_d = bpy.data.cameras.new("PreviewCam")
    cam_d.type = "ORTHO"
    cam_d.ortho_scale = max(2.9, rows * 0.62 * 1.6 * 1.05)
    cam = bpy.data.objects.new("PreviewCam", cam_d)
    sc.collection.objects.link(cam)
    cz = -(rows - 1) * 0.62 / 2
    cam.location = (0.0, -6.0, cz + 2.0)
    cam.rotation_euler = (math.radians(71.6), 0, 0)
    sc.camera = cam
    sun_d = bpy.data.lights.new("Key", "SUN")
    sun_d.energy = 3.8
    sun_d.angle = math.radians(12)
    sun = bpy.data.objects.new("Key", sun_d)
    sun.rotation_euler = (math.radians(50), 0, math.radians(-35))
    sc.collection.objects.link(sun)
    fill_d = bpy.data.lights.new("Fill", "SUN")
    fill_d.energy = 1.1
    fill = bpy.data.objects.new("Fill", fill_d)
    fill.rotation_euler = (math.radians(70), 0, math.radians(150))
    sc.collection.objects.link(fill)
    w = bpy.data.worlds.new("Soft")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*lin(hexc("dfe8ea")), 1.0)
    bg.inputs["Strength"].default_value = 0.7
    sc.world = w
    sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.taa_render_samples = samples
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "PNG"
    sc.render.film_transparent = False


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default=",".join(SPECIES))
    ap.add_argument("--save")
    ap.add_argument("--glb")
    ap.add_argument("--render")
    ap.add_argument("--res", default="1600x1000")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--style", choices=("ps1", "smooth"), default="ps1")
    ap.add_argument("--upscale", type=int, default=3, help="ps1 preview: nearest neighbour upscale factor")
    a = ap.parse_args(argv)
    STYLE["name"] = a.style
    bpy.ops.wm.read_factory_settings(use_empty=True)
    keys = [k for k in a.species.split(",") if k]
    obs = build_all(keys)
    report(obs)
    if a.glb:
        export_glb(obs, a.glb)
    if a.render or a.save:
        preview_scene(obs, tuple(int(x) for x in a.res.split("x")), a.samples)
    if a.save:
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save))
        print("saved", a.save)
    if a.render:
        bpy.context.scene.render.filepath = os.path.abspath(a.render)
        if ps1():
            sc = bpy.context.scene
            sc.eevee.taa_render_samples = 1   # no anti aliasing, like the hardware
            sc.render.filter_size = 0.0
        bpy.ops.render.render(write_still=True)
        if ps1():
            ps1_post(os.path.abspath(a.render), a.upscale)
        print("rendered", a.render)


if __name__ == "__main__":
    main()
