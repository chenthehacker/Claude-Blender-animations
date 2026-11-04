"""Grain comparison image: renders and reference captures side by side, each with a heat map of local surface
detail (|luminance minus its 3x3 mean|, the blueprint section 15 grain metric) and its median.

    python3 grain_compare.py out.png ours.png [ref1.png ...] [--labels "ours,NL#946,..."] [--size 400x240]

Every input is resized to the same size first (default 400x240, the native NL frame) so the numbers compare.
Reference captures are only read, never copied into the repository.
"""
import argparse

import numpy as np
from PIL import Image, ImageDraw


def grain_map(rgb):
    a = rgb.astype(float)
    L = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    p = np.pad(L, 1, mode="edge")
    mean = sum(p[i:i + L.shape[0], j:j + L.shape[1]] for i in range(3) for j in range(3)) / 9
    return np.abs(L - mean)


def colourise(g, vmax=8.0):
    """Dark blue (flat) to yellow (busy), so smooth regions stand out."""
    t = np.clip(g / vmax, 0, 1)[..., None]
    stops = np.array([[10, 12, 40], [70, 30, 120], [200, 60, 80], [250, 170, 40], [255, 245, 150]], float)
    x = t * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = x - i
    return (stops[i[..., 0]] * (1 - f) + stops[i[..., 0] + 1] * f).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("images", nargs="+")
    p.add_argument("--labels")
    p.add_argument("--size", default="400x240")
    a = p.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    labels = a.labels.split(",") if a.labels else [f"image {i}" for i in range(len(a.images))]
    cols = []
    for path, lab in zip(a.images, labels):
        im = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
        g = grain_map(np.asarray(im))
        med, p25, p75 = np.median(g), np.percentile(g, 25), np.percentile(g, 75)
        smooth = float((g < 0.5).mean())
        cols.append((im, Image.fromarray(colourise(g)), f"{lab}: grain median {med:.2f} (p25 {p25:.2f}, p75 {p75:.2f}), "
                                                        f"flat px {smooth:.0%}"))
        print(cols[-1][2])
    pad = 22
    sheet = Image.new("RGB", (len(cols) * (w + 8) + 8, 2 * h + 3 * pad), (24, 24, 28))
    dr = ImageDraw.Draw(sheet)
    for k, (im, gm, txt) in enumerate(cols):
        x = 8 + k * (w + 8)
        dr.text((x, 4), txt, fill=(240, 240, 240))
        sheet.paste(im, (x, pad))
        sheet.paste(gm, (x, pad + h + pad))
    dr.text((8, 2 * h + 2 * pad + 4), "heat map: dark blue = flat (no painted detail), yellow = busy; scale 0 to 8 luminance levels",
            fill=(200, 200, 200))
    sheet.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
