# Blender fishing shorts: how this repo works

Vertical (1080x1920, 24 fps) educational fishing animations, built entirely by Python scripts that drive
Blender 5.2 headless, with on-screen callouts, a synthesized soundtrack, a thumbnail and a cited caption.

## Environment

- Blender runs as the pip module `bpy` 5.2.2 in `/root/bpyenv` (with numpy, scipy). If missing, or EEVEE fails with
  "Couldn't open libEGL.so.1": `bash tools/setup_env.sh` (installs Mesa EGL too).
  Run scene scripts with `/root/bpyenv/bin/python script.py ...` (not `blender`). System `python3` has Pillow, no scipy.
- ffmpeg 6.1 is on PATH. EEVEE renders on the CPU here (llvmpipe): about 4 to 8 min per full res frame.
- Background Bash commands are capped at 2 h; long renders go in chained background batches.
- Kill renders by process name, never `pkill -f` (it matches its own shell):
  `ps -eo pid,comm,args | awk '$2 ~ /^python/ && /render/ {print $1}' | xargs -r kill`
- Most asset sites (polyhaven, ambientcg, sketchfab, printables, freesound) are blocked by the network policy;
  GitHub works (e.g. BabylonJS/Assets, KhronosGroup glTF samples). Record licences in `assets/CREDITS.md`.

## Projects

- `build1_offshore_fishing/` is the most complete template (copy it for a new video):
  `offshore_fishing.py` (scene, camera, HUD, checks), `offshore_audio.py` (soundtrack), `SOURCES.md`,
  `gpu_render/` (render_gpu.py + README for the user's GPU), `caption.txt`, `thumbnail.*`.
- `build1_saltwater_fishing/` uses imported glTF assets (rocks, snapper, barramundi).
- `build1_ocean_fishing/` is the first, low-poly version.
- `build_sockeye_masterclass/` river drift fishing (sockeye): imported Higgsfield/Tripo sockeye GLB, CesiumMan angler
  reposed with IK, over-under split shot (lens on the waterline, per-pixel fog via scene["split"]), Playfair serif
  captions with gold italic accents. Unattended preview render: `tools/preview_driver.sh` (detached, resumable).
- `build_tackle_box_masterclass/` gear breakdown (55 s): everything procedural (cantilever box, 8 lure types,
  baitfish); overlays live in a second scene "HUD" with a static camera, composited over "Main" (so camera depth
  of field never blurs text, and BLENDED overlay materials fade cleanly). Higgsfield text to speech voiceover in
  `voiceover/` (CDN downloads are blocked: the user uploads them); `tackle_box_audio.py --measure` writes
  `vo_timing.json`, which the scene reads to time cuts and captions.
- `build_record_bass_masterclass/` trophy largemouth (55 s): procedural reservoir (shelf, ledge, creek channel), flooded
  oak, bass with hinged jaw and throat, schools (trout, shad, bluegill), jointed swimbait, then a boat set (CesiumMan
  angler, bowed rod). Underwater look is all in materials (scene props "under"/"gloom" keyed per frame: fog group,
  caustics, depth tint). The strike runs on an action clock (`ACT`, an empty "ActionClock" read by GN) for a speed ramp.
  Pins use `best_anchor` (picks the candidate point that stays in the pin band longest). Voice: Higgsfield "Alexey"
  read "bass" as the instrument; the user swapped it for "largemouth" in vo2. Kokoro (local, Apache 2.0) works here as
  a stand-in voice (models from GitHub releases, `/root/ttsenv`), but the user found it not human enough.
- `tools/`: `render_frames.py` (test frames, timings, `--no-hud` thumbnail plates), `contact_sheet.py`,
  `thumbnail.py` (channel thumbnail style), `package_gpu.sh` (GPU zip), `setup_env.sh`.

Scene script CLI (offshore): `--check` (clearances, camera accel/turn, surface crossings vs wipes, overlay
overlaps/off-screen), `--save X.blend` (packs assets, render path `//gpu_render/frames/frame_`, no overwrite),
`--events events.json` (timing for the audio script), `--frames 1,200 --outdir d`, `--scale`, `--samples`.
Audio: `/root/bpyenv/bin/python offshore_audio.py events.json out.wav`, then FLAC into `gpu_render/`.

## Workflow that worked

1. Brief to beats: if requested beats overrun the duration, scale them proportionally and say so.
2. Build the scene script; run `--check` until clean (camera never inside geometry; every water-surface
   crossing and every cut hidden inside a wipe; no overlay overlaps).
3. Render 8 to 9 full res test frames at key beats from the saved .blend (`tools/render_frames.py`), in
   background batches; inspect every frame by eye (clipping, text overlap, blur, blown highlights, things
   off screen). Also project key subjects to screen space to confirm action happens in frame.
4. Fix, re-save, re-render only affected frames. Keep `events.json` unchanged or regenerate the soundtrack.
5. Estimate: mean measured seconds per frame x frame count. Over 10 h: package for the user's GPU
   (`tools/package_gpu.sh`), fill the README estimate, test `render_gpu.py --encode-only` on placeholder frames.
6. Thumbnail: clean plate (`render_frames.py --no-hud --no-blur --samples 64`) + `tools/thumbnail.py`.
7. Caption (`caption.txt`): hook line; sections of bullets; technique sections labelled "(angler technique,
   not research)"; safety or release practice; tagline; `Sources:` with full citations; a note that the film
   is simulated; asset credits (CC BY needs attribution). Under 5000 characters, no angle brackets
   (YouTube rejects them), no dashes or hyphens anywhere (user rule), every fact traceable to SOURCES.md.
   Every video gets this keyword led description (user rule): the opening sentence packs the main search terms,
   section headings reuse them naturally, a comment prompt, 3 to 5 hashtags at the end; also suggest a title
   (under 100 characters, main keyword first) in the reply.

## Blender 5.2 and EEVEE pitfalls (learned the hard way)

- Geometry Nodes modifier inputs: set via the interface identifier (`set_mod_input` helper), not `mod["Input_2"]`.
- RandomValue sockets by name ("Min", "Max", "ID", "Seed"); sky types `MULTIPLE_SCATTERING` etc.
- Sequencer: `ed.strips` (not `sequences`); set `image_settings.media_type` before choosing FFMPEG.
- Layered actions: fcurves live in `action.layers[].strips[].channelbags[].fcurves`.
- A world volume blocks distant light (black frame); fake god rays with emissive shafts instead.
- bmesh: store per-vertex data in `bm.verts.layers.float.new(name)` (becomes a mesh attribute); Python dicts
  keyed by BMVert do not survive.
- No per-object motion blur. HUD overlays parented to the camera must use DITHERED materials (BLENDED ones get
  smeared by the background's motion vectors). Moving overlays: key them still across the shutter
  (`key_still`: keys at f-0.3 and f+0.3, linear), so they stay sharp yet move between frames.
- Dithered partial alpha stipples: fade text by colour, keep alpha 1; key `hide_render` where alpha is 0
  (`hide_when_invisible`), or zero-alpha glyphs can still draw.
- Particles born/removed per frame get no motion blur: stretch spray along velocity in GN instead.
- Standard view transform clips the bright horizon sky to flat white: compress only camera rays
  (Light Path "Is Camera Ray"), preserving hue (scale by the brightest channel).
- Script must run in stock Blender too: no scipy in scene scripts (numpy only).
- glTF imports keep the Y-up to Z-up fix in the armature's rotation_quaternion: compose new rotations on top of it.
- Per-frame cost here is mostly scene sync and sun shadows, not pixels: 25% scale is barely faster than 50%.
  Each EEVEE process needs about 5 GB here, so run at most two in parallel (about 1.7x throughput).
- Underwater, hide everything above the surface (it is opaque from below); fewer, stronger shafts; cobbles cast no shadow.
- `bmesh.ops.extrude_face_region` keeps the original face: delete it, or an inset well renders as a solid block.
- Two-scene compositing: set every scene's `resolution_percentage`, and set the other scenes' `frame_current`
  before rendering a still (`render_frames.py` and `render_gpu.py` do both); a render of the HUD scene alone
  ignores `frame_current` unless you call `frame_set`.
- `tools/preview_driver.sh` takes N, SAMPLES, HEAVY_LO/HEAVY_HI from the environment (defaults: sockeye).
- Never wait on renders with `pgrep -f render_frames` inside the same shell line: it matches itself and never exits.
- Hugging Face, Microsoft and ElevenLabs hosts are blocked; GitHub release downloads and PyPI work.
- Text to speech engines often read "bass" (the fish) as the instrument: respell "bas" or use "largemouth".

## Rules

- Commit and push only when the user asks. Never put model names in commits or files.
- Cite sources for every on-screen fact (SOURCES.md); say what is simulated; never invent numbers.
- Report honest render estimates with the arithmetic shown.
