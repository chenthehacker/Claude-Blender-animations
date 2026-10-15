# Rendering the saltwater fishing video on your GPU

The cloud session that built this scene has no GPU. It renders EEVEE in software, and the measured time for
the full 1560 frame video there is far over 10 hours (see "Why this exists" below), so the final render is
meant to run on your computer.

## What you need

1. Blender 5.2 or newer. The .blend was saved by Blender 5.2.2; older versions are not guaranteed to open it.
2. A GPU with current drivers. EEVEE always renders on the GPU, so there is no device to select.
3. About 4 GB of free disk space for the PNG frames.

## Folder layout

    saltwater_fishing.blend      the scene, with every texture and font packed inside
    gpu_render/render_gpu.py     renders the frames, then joins them with the soundtrack
    gpu_render/soundtrack.flac   the 65 s soundtrack (lossless)
    assets/CREDITS.md            licenses and the attribution line the CC BY 4.0 assets require
    fonts/OFL-*.txt              font licenses

## Easiest: from the Blender window (no terminal)

1. Unzip saltwater_gpu_render.zip. Keep the folder together; the steps below depend on its layout.
2. Start Blender, choose File > Open, and open saltwater_fishing.blend from the unzipped folder.
3. Optional, less grain: in the Properties panel on the right, open the Render tab (camera back icon),
   then the Sampling panel, and change the Render samples from 16 to 32. That takes about twice as long.
4. Choose Render > Render Animation (shortcut Ctrl+F12). A render window opens and shows the frame
   number and the time per frame; multiply that time by 1560 for the total. Frames are saved
   automatically into gpu_render/frames. You do not need to set an output folder.
5. To stop, press Esc in the render window. To continue later, open the file again and repeat step 4:
   finished frames are skipped.
6. When the last frame (1560) is done, click the Scripting tab at the top of the Blender window.
   In the Text Editor (the large text panel), choose Text > Open, select gpu_render/render_gpu.py,
   then click the Run Script button (the play arrow) or press Alt+P with the mouse over the text.
   Blender looks frozen while it works (about 2.5 minutes on the cloud's 4 core CPU), then shows
   "Saved ...preview.mp4". The video is
   preview.mp4 in the unzipped folder: 1080x1920, 24 fps, H.264 with the AAC soundtrack.
   If some frames are missing, it says so instead and you go back to step 4.

## Alternative: from a terminal

Open a terminal in the folder that holds saltwater_fishing.blend and run:

    blender -b saltwater_fishing.blend -P gpu_render/render_gpu.py

On Windows, if `blender` is not on your PATH, use the full path, for example:

    "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" -b saltwater_fishing.blend -P gpu_render\render_gpu.py

On macOS:

    /Applications/Blender.app/Contents/MacOS/Blender -b saltwater_fishing.blend -P gpu_render/render_gpu.py

What happens:

1. After the first frame the script prints the GPU Blender is using. If it names llvmpipe or another
   software renderer, stop and update your graphics driver, because the render will be as slow as the cloud.
2. Each frame is written to gpu_render/frames and the script prints seconds per frame and minutes left.
3. When all 1560 frames exist, Blender joins them with gpu_render/soundtrack.flac into preview.mp4
   (H.264 video, AAC audio, 1080x1920, 24 fps). ffmpeg is not needed.

Stopping and resuming: press Ctrl+C at any time. Running the same command again skips finished frames.
Frames from the window route and the terminal route go to the same folder, so you can mix them.

Options (after a bare `--`):

    --samples 32         samples per pixel; the default here is 32, the window route uses the file's 16
    --start 1 --end 400  render only part of the range
    --no-encode          render frames only
    --encode-only        only build preview.mp4 from existing frames

Example: `blender -b saltwater_fishing.blend -P gpu_render/render_gpu.py -- --samples 64`

## Why this exists

Measured in the cloud session (Blender 5.2.2, EEVEE through Mesa llvmpipe software OpenGL on 4 CPU cores,
1080x1920, 16 samples, shadows and ray tracing on):

    test frames 1, 300, 700, 1100, 1560 took 129.4, 107.1, 116.1, 213.4, 100.1 s
    mean = 666.1 / 5 = 133.2 s per frame
    1560 frames x 133.2 s = 207,792 s = 57.7 hours

A second estimate from ten frames rendered at 40% size (mean 91.0 s) scaled by the measured full size to
40% time ratio (mean 1.62) gives 147 s per frame, or 63.7 hours. The default 32 samples would roughly
double either figure. All of these are far over the 10 hour limit, so the render was handed to your GPU.

How long it takes on your GPU depends on the card and driver and has not been measured. The script prints
seconds per frame and minutes left after every frame, so you will know within the first few frames.
