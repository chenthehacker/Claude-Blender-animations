"""
Render sockeye_masterclass.blend on this computer's GPU, then build preview.mp4 with the soundtrack.

EEVEE always draws on the GPU, so no device setting is needed. Run from the build_sockeye_masterclass folder:

    blender -b sockeye_masterclass.blend -P gpu_render/render_gpu.py
    blender -b sockeye_masterclass.blend -P gpu_render/render_gpu.py -- --samples 32 --start 1 --end 1800
    blender -b sockeye_masterclass.blend -P gpu_render/render_gpu.py -- --encode-only

From the Blender window instead: Render > Render Animation (the .blend already saves into gpu_render/frames and
skips finished frames), then open this file in the Scripting tab's Text Editor and press Run Script to build the video.

Frames go to gpu_render/frames/frame_0001.png ... Finished frames are skipped, so the command can be stopped
and run again. When every frame exists, the frames and
gpu_render/soundtrack.flac are joined into preview.mp4 (H.264 + AAC) inside Blender, so ffmpeg is not needed.
"""
import argparse
import os
import sys
import time

import bpy

# Paths follow the open .blend, so this also works when run from Blender's Text Editor.
HERE = os.path.join(os.path.dirname(bpy.data.filepath), "gpu_render") if bpy.data.filepath else \
    os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(HERE, "frames")
AUDIO = next((os.path.join(HERE, n) for n in ("soundtrack.flac", "soundtrack.wav")
              if os.path.exists(os.path.join(HERE, n))), None)
OUT = os.path.join(os.path.dirname(HERE), "preview.mp4")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=32, help="EEVEE samples per pixel (the .blend holds 32)")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--encode-only", action="store_true")
    p.add_argument("--no-encode", action="store_true")
    return p.parse_args(argv)


def frame_path(f):
    return os.path.join(FRAMES, f"frame_{f:04d}.png")


def report_gpu():
    try:
        import gpu
        print("GPU:", gpu.platform.vendor_get(), "|", gpu.platform.renderer_get(), "|", gpu.platform.backend_type_get())
        if "llvmpipe" in gpu.platform.renderer_get().lower():
            print("WARNING: Blender is drawing in software (llvmpipe), not on a GPU; this will be very slow.")
    except Exception as e:
        print("GPU: could not query the device", f"({e})")


def render(sc, args):
    os.makedirs(FRAMES, exist_ok=True)
    sc.eevee.taa_render_samples = args.samples
    sc.render.resolution_percentage = 100
    ims = sc.render.image_settings
    if hasattr(ims, "media_type"):
        ims.media_type = "IMAGE"
    ims.file_format = "PNG"
    ims.color_mode = "RGB"
    start = args.start or sc.frame_start
    end = args.end or sc.frame_end
    todo = [f for f in range(start, end + 1) if not os.path.exists(frame_path(f)) or os.path.getsize(frame_path(f)) == 0]
    print(f"frames {start}-{end}: {len(todo)} to render, {end - start + 1 - len(todo)} already done; {args.samples} samples")
    t0 = time.time()
    for n, f in enumerate(todo, 1):
        sc.frame_set(f)
        sc.render.filepath = frame_path(f)
        bpy.ops.render.render(write_still=True)
        if n == 1:
            report_gpu()                     # the GPU module is initialised by the first render
        dt = (time.time() - t0) / n
        print(f"frame {f} done | {dt:.1f} s/frame | about {dt * (len(todo) - n) / 60:.0f} min left", flush=True)


def encode(sc):
    missing = [f for f in range(sc.frame_start, sc.frame_end + 1)
               if not os.path.exists(frame_path(f)) or os.path.getsize(frame_path(f)) == 0]
    if missing:
        print(f"not encoding: {len(missing)} frames missing (first {missing[0]})")
        return False
    enc = bpy.data.scenes.new("Encode")
    enc.render.resolution_x, enc.render.resolution_y = sc.render.resolution_x, sc.render.resolution_y
    enc.render.resolution_percentage = 100
    enc.render.fps, enc.render.fps_base = sc.render.fps, sc.render.fps_base
    enc.frame_start, enc.frame_end = sc.frame_start, sc.frame_end
    enc.view_settings.view_transform = "Standard"     # frames are already display-referred
    ed = enc.sequence_editor_create()
    strips = ed.strips if hasattr(ed, "strips") else ed.sequences
    names = [os.path.basename(frame_path(f)) for f in range(sc.frame_start, sc.frame_end + 1)]
    img = strips.new_image("Frames", frame_path(sc.frame_start), 1, sc.frame_start)
    for nm in names[1:]:
        img.elements.append(nm)
    img.frame_final_duration = len(names)
    if AUDIO:
        strips.new_sound("Soundtrack", AUDIO, 2, sc.frame_start)
    else:
        print("note: gpu_render/soundtrack.flac not found, the video will be silent")
    ims = enc.render.image_settings
    if hasattr(ims, "media_type"):
        ims.media_type = "VIDEO"
    ims.file_format = "FFMPEG"
    ff = enc.render.ffmpeg
    ff.format, ff.codec = "MPEG4", "H264"
    ff.constant_rate_factor = "HIGH"
    ff.ffmpeg_preset = "GOOD"
    ff.audio_codec, ff.audio_bitrate, ff.audio_mixrate, ff.audio_channels = "AAC", 192, 48000, "STEREO"
    enc.render.use_sequencer = True
    enc.render.filepath = OUT
    bpy.ops.render.render(animation=True, scene=enc.name)
    bpy.data.scenes.remove(enc)
    print("wrote", OUT)
    return True


def popup(text):
    print(text)
    if bpy.app.background or not bpy.context.window_manager.windows:     # no screen to show it on
        return
    try:
        bpy.context.window_manager.popup_menu(lambda self, ctx: self.layout.label(text=text), title="render_gpu", icon="INFO")
    except Exception:
        pass


def main():
    args = parse_args()
    sc = bpy.context.scene
    if not bpy.app.background:
        # Run from the Blender window: frames come from Render > Render Animation, this only builds the video.
        ok = encode(sc)
        popup(f"Saved {OUT}" if ok else "Frames are missing: use Render > Render Animation first, then run this again.")
        return
    if not args.encode_only:
        render(sc, args)
    if not args.no_encode:
        encode(sc)


main()
