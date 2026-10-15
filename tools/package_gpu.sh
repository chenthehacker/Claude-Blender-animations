#!/bin/bash
# Zip a project for rendering on the user's GPU:  bash tools/package_gpu.sh build1_x_fishing x_fishing OUT_DIR
# Expects PROJECT/NAME.blend (packed, render path //gpu_render/frames/frame_), PROJECT/gpu_render/{render_gpu.py,
# README.md,soundtrack.flac}, PROJECT/SOURCES.md, PROJECT/assets/CREDITS.md (+ licence files), PROJECT/fonts/OFL-*.txt
set -e
P=$(realpath "$1"); NAME=$2; OUT=$(realpath "$3")
Z=$OUT/zip_$NAME/${NAME}_gpu_render
rm -rf "$OUT/zip_$NAME" && mkdir -p "$Z/gpu_render" "$Z/assets" "$Z/fonts"
cp "$P/$NAME.blend" "$P/SOURCES.md" "$Z/"
cp "$P/gpu_render/render_gpu.py" "$P/gpu_render/README.md" "$P"/gpu_render/soundtrack.* "$Z/gpu_render/"
(cd "$P/assets" && find . \( -name 'CREDITS.md' -o -name 'LICENSE*' \) -exec cp --parents {} "$Z/assets/" \;)
cp "$P"/fonts/OFL-*.txt "$Z/fonts/"
(cd "$OUT/zip_$NAME" && rm -f "$OUT/${NAME}_gpu_render.zip" && zip -q -r -X "$OUT/${NAME}_gpu_render.zip" "${NAME}_gpu_render")
ls -la "$OUT/${NAME}_gpu_render.zip"
