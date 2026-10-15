# Rendering the sockeye masterclass on your GPU

The cloud session that built this scene has no GPU. Its software renderer is far too slow for a full
quality 1800 frame render (see "Why this exists" below), so the final render is meant to run on your
computer. The preview.mp4 made in the cloud is a reduced quality draft (see the numbers below).

## What you need

1. Blender 5.2 or newer. The .blend was saved by Blender 5.2.2; older versions are not guaranteed to open it.
2. A GPU with current drivers. EEVEE always renders on the GPU, so there is no device to select.
3. About 6 GB of free disk space for the PNG frames, and 8 GB of GPU memory or more is comfortable.

## Folder layout

    sockeye_masterclass.blend    the scene, with every texture and font packed inside
    gpu_render/render_gpu.py     joins the frames and the soundtrack into preview.mp4 (or renders, from a terminal)
    gpu_render/soundtrack.flac   the 75 s soundtrack (lossless)
    SOURCES.md                   where every fact in the on-screen text comes from
    assets/CREDITS.md            licences and the attribution lines the CC BY 4.0 assets require
    fonts/OFL-*.txt              font licences

## Easiest: from the Blender window (no terminal)

1. Unzip the package. Keep the folder together; the steps below depend on its layout.
2. Start Blender, choose File > Open, and open sockeye_masterclass.blend.
3. Choose Render > Render Animation (shortcut Ctrl+F12). A render window opens and shows the frame
   number and the time per frame; multiply that time by 1800 for the total. Frames are saved
   automatically into gpu_render/frames. You do not need to set an output folder.
   The file is set to 32 samples with motion blur on (Properties > Render tab).
4. To stop, press Esc in the render window. To continue later, open the file again and repeat step 3:
   finished frames are skipped.
5. When the last frame (1800) is done, click the Scripting tab at the top of the Blender window.
   In the Text Editor, choose Text > Open, select gpu_render/render_gpu.py, then click Run Script
   (the play arrow) or press Alt+P with the mouse over the text. Blender looks frozen for a few
   minutes, then shows "Saved ...preview.mp4": 1080x1920, 24 fps, H.264 with the AAC soundtrack.
   If some frames are missing, it says so instead and you go back to step 3.

## Alternative: from a terminal

Open a terminal in the folder that holds sockeye_masterclass.blend and run:

    blender -b sockeye_masterclass.blend -P gpu_render/render_gpu.py

On Windows, if `blender` is not on your PATH, use the full path, for example:

    "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b sockeye_masterclass.blend -P gpu_render\render_gpu.py

The script prints the GPU Blender is using after the first frame (if it names llvmpipe, update your
graphics driver), then seconds per frame and minutes left. Ctrl+C stops it; running it again skips
finished frames. When all frames exist it writes preview.mp4. Options after a bare `--`:
`--samples 64`, `--start 1 --end 400`, `--no-encode`, `--encode-only`.

## Why this exists

The cloud machine renders EEVEE on its processor (a software OpenGL driver, 4 cores, 16 GB). The six test
frames at full resolution, 32 samples, motion blur on, rendered two at a time in parallel processes, took:

    frame    1: 235.5 s     frame 1080: 326.2 s
    frame  360: 311.5 s     frame 1440: 300.5 s
    frame  720: 241.2 s     frame 1800: 225.8 s

    sum 1640.7 s / 6 frames = 273.5 s per frame per process
    two processes in parallel: 273.5 s / 2 = 136.7 s per frame of throughput
    136.7 s x 1800 frames = 246,060 s = 68.4 hours (2.8 days)

That is far over the 10 hour limit, so the full quality render runs on your GPU instead. How long it takes
there depends on your card and driver; it cannot be predicted from these numbers. Read the time per frame
in the render window after the first few frames and multiply by 1800.

The cloud preview (preview.mp4) was rendered from the same .blend at 50 percent resolution (540x960), 4
samples, motion blur on, then scaled to 1080x1920 for playback, so it is softer and noisier than the final.
Its sun shadows and every other setting are the same as the final render.
