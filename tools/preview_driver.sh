#!/bin/bash
# Unattended preview render for a project: two Blender processes in parallel, odd frames first (a 12 fps
# draft is encoded halfway), then even frames, then the 24 fps preview with the soundtrack.
#   setsid nohup bash tools/preview_driver.sh build_sockeye_masterclass sockeye_masterclass > LOG 2>&1 &
# Resumable: finished frames are skipped, so running it again continues where it stopped.
set -u
P=$(realpath "$1"); NAME=$2
F=$P/preview_frames; mkdir -p "$F"
PY=/root/bpyenv/bin/python; R=$(dirname "$0")/render_frames.py
AUDIO=$P/gpu_render/soundtrack.flac
run_pass() {   # $1 = first frame (1 odd, 2 even): missing frames dealt out alternately, so both workers finish together
    local a="" b="" k=0
    for f in $(seq "$1" 2 1800); do
        [ -s "$F/frame_$(printf %04d "$f").png" ] && continue
        if [ $((k % 2)) = 0 ]; then a="$a,$f"; else b="$b,$f"; fi
        k=$((k + 1))
    done
    worker "${a#,}" &
    worker "${b#,}" &
    wait
}
worker() {   # $1 = comma list; a fresh Blender every 40 frames (memory grows per frame and got OOM killed)
    local list=($(echo "$1" | tr ',' ' ')) i
    for ((i = 0; i < ${#list[@]}; i += 40)); do
        local chunk=$(IFS=,; echo "${list[*]:i:40}")
        $PY "$R" "$P/$NAME.blend" "$F" "$chunk" --scale 50 --samples 4 --skip-existing
    done
}
encode() {   # $1 = output, $2 = 12 or 24
    if [ "$2" = 12 ]; then
        for f in $(seq 1 2 1799); do echo "file '$F/frame_$(printf %04d $f).png'"; echo "duration 0.0833333"; done > "$F/list12.txt"
        ffmpeg -loglevel error -y -f concat -safe 0 -i "$F/list12.txt" -i "$AUDIO" -map 0:v -map 1:a \
            -vf "scale=1080:1920:flags=lanczos,fps=24,format=yuv420p" -c:v libx264 -crf 18 -preset slow \
            -c:a aac -b:a 192k -shortest -movflags +faststart "$1"
    else
        ffmpeg -loglevel error -y -framerate 24 -i "$F/frame_%04d.png" -i "$AUDIO" -map 0:v -map 1:a \
            -vf "scale=1080:1920:flags=lanczos,format=yuv420p" -c:v libx264 -crf 18 -preset slow \
            -c:a aac -b:a 192k -shortest -movflags +faststart "$1"
    fi
    echo "$(date -u +%H:%M) wrote $1"
}
echo "$(date -u +%H:%M) pass 1 (odd frames)"; run_pass 1
n=$(ls "$F"/frame_*.png 2>/dev/null | wc -l); echo "$(date -u +%H:%M) frames on disk: $n"
odd=0; for f in $(seq 1 2 1799); do [ -s "$F/frame_$(printf %04d "$f").png" ] && odd=$((odd + 1)); done
if [ "$odd" -lt 900 ]; then echo "$(date -u +%H:%M) pass 1 incomplete ($odd of 900 odd frames): stopping"; exit 1; fi
[ ! -e "$P/preview_draft_12fps.mp4" ] && encode "$P/preview_draft_12fps.mp4" 12
echo "$(date -u +%H:%M) pass 2 (even frames)"; run_pass 2
n=$(ls "$F"/frame_*.png 2>/dev/null | wc -l); echo "$(date -u +%H:%M) frames on disk: $n"
if [ "$n" -ge 1800 ]; then encode "$P/preview.mp4" 24; echo "$(date -u +%H:%M) DONE"; else echo "$(date -u +%H:%M) INCOMPLETE: $n frames"; fi
