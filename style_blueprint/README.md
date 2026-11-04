# Style blueprint: City Folk and New Leaf look, measured

`BLUEPRINT.md` is the master blueprint (version 3) for recreating the visual style of Animal Crossing: City Folk
(Wii) and New Leaf (3DS) in Blender with original assets. It cites images as GAME#INDEX; `image_index.csv` maps
every ID to its Nookipedia gallery section, caption and source URL. No images are stored in this repository.

How it was produced (November 2026):
* 1,864 images from the two Nookipedia gallery pages, plus 18 rasterised instruction booklet pages, analysed
  over every pixel (`section_metrics.txt` holds per section medians).
* Round 1: four analysts (scenes and camera, 3D models, textures and colour, 2D art and cohesion).
* Round 2: four independent verifiers re-measured every number with their own scripts and resolved three
  cross analyst conflicts (rim light, shadow side ratio, head share).
* Round 3: synthesis into one blueprint, then a final audit that re-measured 46 headline numbers
  (33 confirmed, 13 corrected; section 14 logs each one).

Paths such as `nook/...`, `synth/...` or `audit/...` in the blueprint refer to the session scratchpad where the
measurement scripts ran; they are not part of this repository.
