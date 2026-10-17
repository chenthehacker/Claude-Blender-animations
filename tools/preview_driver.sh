#!/bin/bash
# Unattended preview render for a project: two Blender processes in parallel, odd frames first (a 12 fps
# draft is encoded halfway), then even frames, then the 24 fps preview with the soundtrack.
#   setsid nohup bash tools/preview_driver.sh build_sockeye_masterclass sockeye_masterclass > LOG 2>&1 &
# Resumable: finished frames are skipped, so running it again continues where it stopped.
# Environment overrides (defaults are the sockeye build's): N (frame count), SAMPLES (default 4),
# HEAVY_LO/HEAVY_HI (memory hungry frames, all on worker A; set both to 0 for none).
set -u
N=${N:-1800}; SAMPLES=${SAMPLES:-4}; HEAVY_LO=${HEAVY_LO:-122}; HEAVY_HI=${HEAVY_HI:-485}
P=$(realpath "$1"); NAME=$2
F=$P/preview_frames; mkdir -p "$F"
PY=/root/bpyenv/bin/python; R=$(dirname "$0")/render_frames.py
AUDIO=$P/gpu_render/soundtrack.flac
run_pass() {   # $1 = first frame (1 odd, 2 even): missing frames dealt out alternately, so both workers finish together
    # Frames in HEAVY (memory hungry: two at once hit the memory limit) all go to worker A; the rest are
    # dealt to whichever worker has less estimated work, using measured seconds per frame per shot.
    local lists a b
    lists=$(python3 - "$F" "$1" "$N" "$HEAVY_LO" "$HEAVY_HI" <<'PYEOF'
import os, sys
F, first, n, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
heavy = range(lo, hi + 1) if hi > 0 else range(0)
sockeye = n == 1800
cost = (lambda f: 150 if f in heavy else 30 if f < 122 or 486 <= f < 1096 else 48 if f < 1262 else 115) if sockeye else \
    (lambda f: 2 if f in heavy else 1)
miss = [f for f in range(first, n + 1, 2) if not (os.path.exists(f"{F}/frame_{f:04d}.png") and os.path.getsize(f"{F}/frame_{f:04d}.png"))]
a = [f for f in miss if f in heavy]
b = []
la, lb = sum(map(cost, a)), 0
for f in (f for f in miss if f not in heavy):
    if la <= lb:
        a.append(f); la += cost(f)
    else:
        b.append(f); lb += cost(f)
print(",".join(map(str, sorted(a))))
print(",".join(map(str, sorted(b))))
PYEOF
)
    a=$(echo "$lists" | sed -n 1p); b=$(echo "$lists" | sed -n 2p)
    echo "$(date -u +%H:%M) worker A: $(echo "$a" | tr ',' '\n' | grep -c .) frames, worker B: $(echo "$b" | tr ',' '\n' | grep -c .) frames"
    [ -n "$a" ] && worker "$a" &
    [ -n "$b" ] && worker "$b" &
    wait
}
worker() {   # $1 = comma list; a fresh Blender every 40 frames (memory grows per frame and got OOM killed)
    local list=($(echo "$1" | tr ',' ' ')) i
    for ((i = 0; i < ${#list[@]}; i += 40)); do
        local chunk=$(IFS=,; echo "${list[*]:i:40}")
        $PY "$R" "$P/$NAME.blend" "$F" "$chunk" --scale 50 --samples "$SAMPLES" --skip-existing
    done
}
encode() {   # $1 = output, $2 = 12 or 24
    if [ "$2" = 12 ]; then
        for f in $(seq 1 2 $((N - 1))); do echo "file '$F/frame_$(printf %04d $f).png'"; echo "duration 0.0833333"; done > "$F/list12.txt"
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
odd=0; for f in $(seq 1 2 $((N - 1))); do [ -s "$F/frame_$(printf %04d "$f").png" ] && odd=$((odd + 1)); done
if [ "$odd" -lt $((N / 2)) ]; then echo "$(date -u +%H:%M) pass 1 incomplete ($odd of $((N / 2)) odd frames): stopping"; exit 1; fi
[ ! -e "$P/preview_draft_12fps.mp4" ] && encode "$P/preview_draft_12fps.mp4" 12
echo "$(date -u +%H:%M) pass 2 (even frames)"; run_pass 2
n=$(ls "$F"/frame_*.png 2>/dev/null | wc -l); echo "$(date -u +%H:%M) frames on disk: $n"
if [ "$n" -ge "$N" ]; then encode "$P/preview.mp4" 24; echo "$(date -u +%H:%M) DONE"; else echo "$(date -u +%H:%M) INCOMPLETE: $n frames"; fi
