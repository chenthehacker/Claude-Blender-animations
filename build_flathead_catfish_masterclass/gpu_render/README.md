# Rendering the flathead catfish masterclass on your GPU

The cloud session that built this scene has no GPU. Its software renderer is far too slow for a full
quality 1320 frame render (see "Why this exists" below), so the final render is meant to run on your
computer. The preview.mp4 made in the cloud is a reduced quality draft (see the numbers below).

## What you need

1. Blender 5.2 or newer. The .blend was saved by Blender 5.2.2; older versions are not guaranteed to open it.
2. A GPU with current drivers. EEVEE always renders on the GPU, so there is no device to select.
3. About 6 GB of free disk space for the PNG frames, and 8 GB of GPU memory or more is comfortable.

## Folder layout

    flathead_catfish_masterclass.blend    the scene, with every texture and font packed inside
    gpu_render/render_gpu.py     joins the frames and the soundtrack into preview.mp4 (or renders, from a terminal)
    gpu_render/soundtrack.flac   the 55 s soundtrack with the voiceover (lossless)
    SOURCES.md                   where every fact in the on-screen text comes from
    assets/CREDITS.md            what is original, what is third party (the angler body) and generated (the voice)
    fonts/OFL-*.txt              font licences

## Easiest: from the Blender window (no terminal)

1. Unzip the package. Keep the folder together; the steps below depend on its layout.
2. Start Blender, choose File > Open, and open flathead_catfish_masterclass.blend.
3. Choose Render > Render Animation (shortcut Ctrl+F12). A render window opens and shows the frame
   number and the time per frame; multiply that time by 1320 for the total. Frames are saved
   automatically into gpu_render/frames. You do not need to set an output folder.
   The file is set to 32 samples with motion blur on (Properties > Render tab).
4. To stop, press Esc in the render window. To continue later, open the file again and repeat step 3:
   finished frames are skipped.
5. When the last frame (1320) is done, click the Scripting tab at the top of the Blender window.
   In the Text Editor, choose Text > Open, select gpu_render/render_gpu.py, then click Run Script
   (the play arrow) or press Alt+P with the mouse over the text. Blender looks frozen for a few
   minutes, then shows "Saved ...preview.mp4": 1080x1920, 24 fps, H.264 with the AAC soundtrack.
   If some frames are missing, it says so instead and you go back to step 3.

## Alternative: from a terminal

Open a terminal in the folder that holds flathead_catfish_masterclass.blend and run:

    blender -b flathead_catfish_masterclass.blend -P gpu_render/render_gpu.py

On Windows, if `blender` is not on your PATH, use the full path, for example:

    "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b flathead_catfish_masterclass.blend -P gpu_render\render_gpu.py

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

The cloud machine renders EEVEE on its processor (a software OpenGL driver, 4 cores, 16 GB). Ten test
frames at full resolution, 32 samples, motion blur and depth of field on, rendered two at a time in
parallel processes, took:

    frame    1: 365.9 s     frame  700: 279.5 s
    frame  100: 344.2 s     frame  912: 308.1 s
    frame  264: 184.6 s     frame  930: 228.1 s
    frame  420: 276.4 s     frame 1000: 154.6 s
    frame  528: 253.4 s     frame 1320: 154.9 s

    sum 2549.7 s / 10 frames = 255.0 s per frame per process
    two processes in parallel: 255.0 s / 2 = 127.5 s per frame of throughput
    127.485 s x 1320 frames = 168,280 s = 46.7 hours

(These were measured on a scene build a few edits before the final one, with the same settings.)

That is far over the 10 hour limit, so the full quality render runs on your GPU instead. How long it takes
there depends on your card and driver; it cannot be predicted from these numbers. Read the time per frame
in the render window after the first few frames and multiply by 1320.

The cloud preview (preview.mp4) was rendered from the same .blend at 50 percent resolution (540x960) and 8
samples, motion blur and depth of field on, then scaled to 1080x1920 for playback, so it is softer and
noisier than the final.
