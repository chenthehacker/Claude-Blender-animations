"""Contact sheet of rendered test frames: python3 tools/contact_sheet.py FRAMES_DIR OUT.jpg FONT.otf [fps] [cols]"""
import glob
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

src, dst, font_path = sys.argv[1:4]
fps = float(sys.argv[4]) if len(sys.argv) > 4 else 24.0
cols = int(sys.argv[5]) if len(sys.argv) > 5 else 5
files = sorted(glob.glob(os.path.join(src, "frame_*.png")))
w, h, pad, cap = 360, 640, 12, 40
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * (h + cap) + pad), (20, 22, 26))
d = ImageDraw.Draw(sheet)
font = ImageFont.truetype(font_path, 24)
for i, fn in enumerate(files):
    f = int(re.findall(r"(\d+)", os.path.basename(fn))[-1])
    x, y = pad + (i % cols) * (w + pad), pad + (i // cols) * (h + cap)
    sheet.paste(Image.open(fn).convert("RGB").resize((w, h), Image.LANCZOS), (x, y))
    d.text((x, y + h + 6), f"frame {f}  ({(f - 1) / fps:.1f} s)", fill=(230, 230, 230), font=font)
sheet.save(dst, quality=90)
print("saved", dst)
