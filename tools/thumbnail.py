"""Shorts thumbnail in the channel style: clean plate + kicker + Playfair headline with one orange word.

    python3 tools/thumbnail.py PLATE.png FONTS_DIR OUT_BASENAME "KICKER" "Line one," "line two " "accent." \
        [--crop x0,y0,x1,y1] [--size1 104] [--size2 200]

Line two is drawn as two runs on one baseline: white text at size1, then the accent word in orange at size2.
The crop must be 9:16 (e.g. 900x1600 or 810x1440); it is scaled to 1080x1920. Writes OUT.png and OUT.jpg.
Check that each line fits: Playfair Bold Italic at 104 px is ~46 px per character.
"""
import argparse

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

p = argparse.ArgumentParser()
for n in ("plate", "fonts", "out", "kicker", "line1", "line2", "accent"):
    p.add_argument(n)
p.add_argument("--crop", default=None)
p.add_argument("--size1", type=int, default=104)
p.add_argument("--size2", type=int, default=200)
a = p.parse_args()
W, H = 1080, 1920
ORANGE, WHITE = (245, 178, 76), (255, 255, 255)

img = Image.open(a.plate).convert("RGB").resize((W, H), Image.LANCZOS)
if a.crop:
    img = img.crop(tuple(int(v) for v in a.crop.split(","))).resize((W, H), Image.LANCZOS)
img = ImageEnhance.Contrast(ImageEnhance.Color(img).enhance(1.1)).enhance(1.05).convert("RGBA")
scrim = Image.new("RGBA", (W, H))
d = ImageDraw.Draw(scrim)
for y in range(620):                                   # top scrim behind the headline
    d.line([(0, y), (W, y)], fill=(4, 18, 40, int(170 * (1 - y / 620) ** 1.6)))
for y in range(1446, H):                               # soft darkening toward the bottom
    d.line([(0, y), (W, y)], fill=(10, 20, 24, int(120 * ((y - 1446) / (H - 1446)) ** 1.3)))
img = Image.alpha_composite(img, scrim)
kick = ImageFont.truetype(f"{a.fonts}/Inter-SemiBold.otf", 36)
f1 = ImageFont.truetype(f"{a.fonts}/PlayfairDisplay-BoldItalic.ttf", a.size1)
f2 = ImageFont.truetype(f"{a.fonts}/PlayfairDisplay-BoldItalic.ttf", a.size2)


def draw(layer, shade):
    d = ImageDraw.Draw(layer)
    c = (lambda col: shade) if shade else (lambda col: col)
    x = (W - sum(d.textlength(ch, font=kick) + 9 for ch in a.kicker)) / 2
    for ch in a.kicker:                                 # letter-spaced kicker
        d.text((x, 92), ch, font=kick, fill=c(WHITE))
        x += d.textlength(ch, font=kick) + 9
    d.text(((W - d.textlength(a.line1, font=f1)) / 2, 150), a.line1, font=f1, fill=c(WHITE))
    base = 150 + int(a.size1 * 1.02) + f2.getmetrics()[0]
    w1, w2 = d.textlength(a.line2, font=f1), d.textlength(a.accent, font=f2)
    x = (W - w1 - w2) / 2
    d.text((x, base - f1.getmetrics()[0]), a.line2, font=f1, fill=c(WHITE))
    d.text((x + w1, base - f2.getmetrics()[0]), a.accent, font=f2, fill=c(ORANGE))


sh = Image.new("RGBA", (W, H))
draw(sh, (0, 0, 0, 150))
sh = sh.filter(ImageFilter.GaussianBlur(10))
shifted = Image.new("RGBA", (W, H))
shifted.paste(sh, (0, 6))
img = Image.alpha_composite(img, shifted)
fg = Image.new("RGBA", (W, H))
draw(fg, None)
img = Image.alpha_composite(img, fg).convert("RGB")
img.save(a.out + ".png", optimize=True)
img.save(a.out + ".jpg", quality=90, optimize=True, progressive=True)
print("saved", a.out)
