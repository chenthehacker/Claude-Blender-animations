# Rendering the record goliath grouper masterclass on your GPU

The cloud session that built this scene has no GPU. Its software renderer is far too slow for a full
quality 1057 frame render (see "Why this exists" below), so the final render is meant to run on your
computer. The preview.mp4 made in the cloud is a reduced quality draft (see the numbers below).

## What you need

1. Blender 5.2 or newer. The .blend was saved by Blender 5.2.2; older versions are not guaranteed to open it.
2. A GPU with current drivers. EEVEE always renders on the GPU, so there is no device to select.
3. About 6 GB of free disk space for the PNG frames, and 8 GB of GPU memory or more is comfortable.

## Folder layout

    record_goliath_masterclass.blend    the scene, with every texture and font packed inside
    gpu_render/render_gpu.py     joins the frames and the soundtrack into preview.mp4 (or renders, from a terminal)
    gpu_render/soundtrack.flac   the 44 s soundtrack with the voiceover (lossless)
    SOURCES.md                   where every fact in the on-screen text comes from
    assets/CREDITS.md            what is original (every model) and what is generated (the voice)
    fonts/OFL-*.txt              font licences

## Easiest: from the Blender window (no terminal)

1. Unzip the package. Keep the folder together; the steps below depend on its layout.
2. Start Blender, choose File > Open, and open record_goliath_masterclass.blend.
3. Choose Render > Render Animation (shortcut Ctrl+F12). A render window opens and shows the frame
   number and the time per frame; multiply that time by 1057 for the total. Frames are saved
   automatically into gpu_render/frames. You do not need to set an output folder.
   The file is set to 32 samples with motion blur on (Properties > Render tab).
4. To stop, press Esc in the render window. To continue later, open the file again and repeat step 3:
   finished frames are skipped.
5. When the last frame (1057) is done, click the Scripting tab at the top of the Blender window.
   In the Text Editor, choose Text > Open, select gpu_render/render_gpu.py, then click Run Script
   (the play arrow) or press Alt+P with the mouse over the text. Blender looks frozen for a few
   minutes, then shows "Saved ...preview.mp4": 1080x1920, 24 fps, H.264 with the AAC soundtrack.
   If some frames are missing, it says so instead and you go back to step 3.

## Alternative: from a terminal

Open a terminal in the folder that holds record_goliath_masterclass.blend and run:

    blender -b record_goliath_masterclass.blend -P gpu_render/render_gpu.py

On Windows, if `blender` is not on your PATH, use the full path, for example:

    "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b record_goliath_masterclass.blend -P gpu_render\render_gpu.py

The script prints the GPU Blender is using after the first frame (if it names llvmpipe, update your
graphics driver), then seconds per frame and minutes left. Ctrl+C stops it; running it again skips
finished frames. When all frames exist it writes preview.mp4. Options after a bare `--`:
`--samples 64`, `--start 1 --end 400`, `--no-encode`, `--encode-only`.

## Two scenes inside one file

The .blend holds two scenes: "Main" (the 3D) and "HUD" (the captions, badges and pins). The compositor lays
the HUD over the 3D render on every frame, so the depth of field never blurs the text. Render from the
"Main" scene (it opens on it); you do not need to touch "HUD".

## If your graphics driver times out

On a laptop with integrated graphics, very long frames can trip Windows' GPU timeout (the screen flickers,
the driver reports an error and Blender stops responding, or a frame comes out wrong). Before rendering:
lower Render Properties > Sampling > Render to 16, untick Raytracing, and set Shadows > Resolution to
50 %. Do this for the whole film, not part of it, so the quality does not change mid video. Check
any frame that is much smaller in file size than its neighbours before encoding.

## Why this exists

The cloud machine renders EEVEE on its processor (a software OpenGL driver, 4 cores, 16 GB). The test frames the
brief asked for, at full resolution, 32 samples, motion blur and depth of field on, two processes in parallel
(so each time below is per process while another frame renders alongside), took:

    frame    1: 210.0 s     (the water over the wreck from above, the loop join)
    frame  240: 402.5 s     (the goliath nosing out of the hull hole, barnacles, silt)
    frame  480: 124.2 s     (the rig on its turntable)
    frame  720: 254.5 s     (the strike, low on the sand looking up at the hole)
    frame 1057: 194.7 s     (the water over the wreck from above, the loop join)

    sum 1185.9 s / 5 frames = 237.2 s per frame per process (two processes running)
    two processes finish 2 frames every 237.2 s: 237.2 / 2 = 118.6 s per frame
    118.6 s x 1057 frames = 125,360 s = about 34.8 hours with two processes
    one process alone (two give about 1.7 times the throughput here): 34.8 h x 1.7 = about 59.2 hours

The five frames are a small sample, and the cross section and the boat shots were not among them, so the real
total could be higher. Either way it is far over the 10 hour limit, so the full quality render runs on your GPU
instead. How long it takes there depends on your card and driver; it cannot be predicted from these numbers. Read
the time per frame in the render window after the first few frames and multiply by 1057.

The cloud preview (preview_draft_12fps.mp4) was rendered from the same .blend at 50 percent resolution (540x960)
and 16 samples, motion blur and depth of field on, every second frame only (12 fps, each frame shown twice), then
scaled to 1080x1920 for playback, so it is softer, noisier and less smooth than the final. It covers only the first
14.1 s of the film (frames 1 to 339): the full draft would have taken about 6 more hours here, so it was stopped.
Within that span 39 of the 170 draft frames are missing (from frame 155 on), and each gap repeats the previous frame,
so parts of it play at about 6 fps. The soundtrack is cut at 14.1 s to match.
