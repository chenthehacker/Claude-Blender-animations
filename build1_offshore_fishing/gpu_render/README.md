# Rendering the offshore trolling video on your GPU

The cloud session that built this scene has no GPU. Its software renderer is far too slow for 1800
frames (see "Why this exists" below), so the final render is meant to run on your computer.

## What you need

1. Blender 5.2 or newer. The .blend was saved by Blender 5.2.2; older versions are not guaranteed to open it.
2. A GPU with current drivers. EEVEE always renders on the GPU, so there is no device to select.
3. About 5 GB of free disk space for the PNG frames.

## Folder layout

    offshore_fishing.blend       the scene, with every texture and font packed inside
    gpu_render/render_gpu.py     joins the frames and the soundtrack into preview.mp4 (or renders, from a terminal)
    gpu_render/soundtrack.flac   the 75 s soundtrack (lossless)
    SOURCES.md                   where every fact in the on-screen text comes from
    assets/CREDITS.md            licences and the attribution line the CC BY 4.0 texture requires
    fonts/OFL-*.txt              font licences

## Easiest: from the Blender window (no terminal)

1. Unzip the package. Keep the folder together; the steps below depend on its layout.
2. Start Blender, choose File > Open, and open offshore_fishing.blend.
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

Open a terminal in the folder that holds offshore_fishing.blend and run:

    blender -b offshore_fishing.blend -P gpu_render/render_gpu.py

On Windows, if `blender` is not on your PATH, use the full path, for example:

    "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b offshore_fishing.blend -P gpu_render\render_gpu.py

The script prints the GPU Blender is using after the first frame (if it names llvmpipe, update your
graphics driver), then seconds per frame and minutes left. Ctrl+C stops it; running it again skips
finished frames. When all frames exist it writes preview.mp4. Options after a bare `--`:
`--samples 64`, `--start 1 --end 400`, `--no-encode`, `--encode-only`.

## Why this exists

The cloud machine renders EEVEE on its processor (a software OpenGL driver, 4 cores). Eight test frames
at full resolution, 32 samples, motion blur on, took:

    frame    1: 289.2 s     frame 1015: 497.5 s
    frame  200: 347.0 s     frame 1250: 333.9 s
    frame  450: 439.7 s     frame 1500: 276.5 s
    frame  760: 443.6 s     frame 1800: 306.7 s

    sum 2934.1 s / 8 frames = 366.8 s per frame
    366.8 s x 1800 frames = 660,240 s = 183.4 hours (7.6 days)

That is far over the 10 hour limit, so the final render runs on your GPU instead. How long it takes there
depends on your card and driver; it cannot be predicted from these numbers. Read the time per frame
in the render window after the first few frames and multiply by 1800.
