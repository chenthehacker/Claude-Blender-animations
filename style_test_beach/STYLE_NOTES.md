# Handheld toon style test: notes and sources

Test scene only (no on screen facts). All assets are original and built in code: no game models, textures,
characters or logos are used. The style values come from pixel measurements of reference captures and
from public documentation of the PICA200 GPU.

## Run

    /root/bpyenv/bin/python beach_style_test.py --check
    /root/bpyenv/bin/python beach_style_test.py --render renders/beach_noon_day.png            # 1200x720, clean
    /root/bpyenv/bin/python beach_style_test.py --water lavender --render renders/beach_noon_lavender.png
    /root/bpyenv/bin/python beach_style_test.py --native --render renders/beach_noon_native.png  # 400x240 + _x3
    /root/bpyenv/bin/python beach_style_test.py --save beach.blend
    # camera overrides: --pitch 40 --dist 6.5 --lens 56 --radius 11

EEVEE needs EGL in a fresh cloud container: `apt-get install -y libegl1 libegl-mesa0 libgl1-mesa-dri libgbm1`.
Measured render time here (CPU, llvmpipe): 7.8 to 8.6 s per 1200x720 frame at 16 samples,
3.3 to 7.2 s per native 400x240 frame.

## Rendering rules taken from the analysis

* Default: 1200x720 (the 5:3 frame of the top screen at 3x), 16 samples, anti aliased. Models stay clean;
  only the textures are low resolution (bilinear, so they read soft, not blocky).
* `--native`: 400x240, one sample, filter size 0 (aliased, like the hardware), nearest neighbour 3x upscale.
* Camera: high follow camera facing north, 40 degrees down, 56 mm lens on a 36 mm sensor, 6.5 m from a focus
  point beside the player. These are estimates matched by eye to a town reference frame (player about a third
  of the frame height, little perspective distortion, tree tops cut by the top edge); no published values.
* Rolling log world: every mesh except the cloud cards is wrapped around a cylinder (axis east to west,
  radius 11 m) through the focus point, so the ground falls away and the sky shows at the top. The cylinder
  world is documented (80.lv, free camera mod); the radius is not, so 11 m is matched by eye.
* Terrain is one material: sand, cliff and grass blend through per vertex weights plus noise on a 0.33 m grid,
  so borders are soft painted curves (as in the reference, where path and grass blend softly).
* Unlit emission shaders. Lighting is a lookup table indexed by N.L (the PICA200 has no pixel shader; its
  fragment lighting indexes programmable LUTs with dot products: 3dbrew GPU/Fragment_Lighting).
* LUT values measured on a white marlin belly, relative to its median (sRGB ratios, raised to 2.2 for linear):
  shadow 104/189, 103/174, 148/198 = 0.550, 0.592, 0.747; highlight 239/189, 223/174, 213/198 = 1.265, 1.282, 1.076.
* Characters use a half strength shadow tint (art choice, so faces read).
* Blob shadows: neutral multiply 0.83 (measured on day sand, low confidence), directly under objects (noon).
* Textures 8 to 128 px (frond 128x256), painted with numpy, bilinear filtering. Frond cards use an alpha test on
  a smooth edge field, so cut edges stay clean at 1200x720.

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
* 80.lv, New Leaf's world is a cylinder: https://80.lv/articles/this-is-your-reminder-that-animal-crossing-new-leaf-s-world-is-a-cylinder
