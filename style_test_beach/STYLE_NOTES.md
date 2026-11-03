# Handheld toon style test: notes and sources

Test scene only (no on screen facts). All assets are original and built in code: no game models, textures,
characters or logos are used. The style values come from pixel measurements of reference captures and
from public documentation of the PICA200 GPU.

## Run

    /root/bpyenv/bin/python beach_style_test.py --check
    /root/bpyenv/bin/python beach_style_test.py --render renders/beach_noon_day.png            # also writes _x3
    /root/bpyenv/bin/python beach_style_test.py --water lavender --render renders/beach_noon_lavender.png
    /root/bpyenv/bin/python beach_style_test.py --save beach.blend

EEVEE needs EGL in a fresh cloud container: `apt-get install -y libegl1 libegl-mesa0 libgl1-mesa-dri libgbm1`.
Measured render time here (CPU, llvmpipe): 3.9 to 7.2 s per 400x240 frame.

## Rendering rules taken from the analysis

* 400x240, one EEVEE sample, filter size 0: no anti aliasing, like the captures. Nearest neighbour 3x upscale.
* Unlit emission shaders. Lighting is a lookup table indexed by N.L (the PICA200 has no pixel shader; its
  fragment lighting indexes programmable LUTs with dot products: 3dbrew GPU/Fragment_Lighting).
* LUT values measured on a white marlin belly, relative to its median (sRGB ratios, raised to 2.2 for linear):
  shadow 104/189, 103/174, 148/198 = 0.550, 0.592, 0.747; highlight 239/189, 223/174, 213/198 = 1.265, 1.282, 1.076.
* Characters use a half strength shadow tint (art choice, so faces read).
* Blob shadows: neutral multiply 0.83 (measured on day sand, low confidence), directly under objects (noon).
* Textures 8 to 128 px, painted with numpy, bilinear filtering. Frond cards use alpha test (binary alpha).

## Palette provenance

Measured (M) in the script: day sky #215DF6, sand #CFBC9A and #EBE5B8, cliff #A68C8D, foliage #0E4D3E,
palm bands #D09A96 and #FEDAA8, skin #E2BF9B, day water clusters #8DC2E7, #6296C3, #3A71AD, foam #D1EBE9,
lavender water clusters (rain capture) #8C8BF5, #606CCF, #45569C, foam #D8B6CB.
Everything marked A in the script is an art choice because no daytime sample existed (wood, grass, clothes, hair).

Note: the measured noon water is blue (hue about 210). The violet water (hue about 230) was only measured in
rain, dusk and night captures, so `--water lavender` is a stylistic option, not a measured noon look.

## Sources

* 3dbrew, Hardware: https://www.3dbrew.org/wiki/Hardware
* 3dbrew, GPU Fragment Lighting: https://www.3dbrew.org/wiki/GPU/Fragment_Lighting
* 3dbrew, GPU Internal Registers: https://www.3dbrew.org/wiki/GPU/Internal_Registers
