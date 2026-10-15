#!/bin/bash
# Cloud environment setup: Blender 5.2 as the Python module bpy, plus numpy/scipy (audio synthesis).
# Paste into the environment's Setup script (cloud environment menu > Edit), or run once by hand.
# FFmpeg and Pillow (system python3) come with the base image.
set -e
if [ ! -x /root/bpyenv/bin/python ] || ! /root/bpyenv/bin/python -c "import bpy" 2>/dev/null; then
    python3 -m venv /root/bpyenv
    /root/bpyenv/bin/pip install --quiet bpy==5.2.2 numpy scipy
fi
/root/bpyenv/bin/python -c "import bpy; print('Blender', bpy.app.version_string)"
