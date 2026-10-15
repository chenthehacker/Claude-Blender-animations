"""Procedural soundtrack for the saltwater fishing breakdown.

Everything is synthesised here (no samples): surf and underwater ambience,
transition splashes, overlay pops, the cast, jig contacts, the strike and a
soft background music bed. Timing comes from the scene script:

    python saltwater_fishing.py --events events.json
    python saltwater_audio.py events.json soundtrack.wav [--video preview.mp4 --out preview_audio.mp4]

Needs numpy, scipy and ffmpeg (for loudness normalisation and muxing).
"""

import argparse
import json
import os
import subprocess
import tempfile
import wave

import numpy as np
from scipy import signal

SR = 48000
RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------
# DSP helpers
# --------------------------------------------------------------------------
def filt(x, kind, fc, order=2):
    sos = signal.butter(order, fc, btype=kind, fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def pink(n):
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    return signal.lfilter(b, a, RNG.standard_normal(n)) * 4.0


def brown(n):
    return signal.lfilter([1.0], [1.0, -0.997], RNG.standard_normal(n)) * 0.06


def tone(freq, dur, decay, partials=((1.0, 1.0),), attack=0.004):
    t = np.arange(int(dur * SR)) / SR
    out = sum(a * np.sin(2 * np.pi * freq * r * t) * np.exp(-t / (decay / r ** 0.5)) for r, a in partials)
    out *= np.minimum(1.0, t / attack)
    return out


def chirp(f0, f1, dur, decay):
    t = np.arange(int(dur * SR)) / SR
    f = f0 * (f1 / f0) ** (t / dur)
    phase = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(phase) * np.exp(-t / decay) * np.minimum(1.0, t / 0.002)


def noise_burst(dur, lo, hi, decay, attack=0.003):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = filt(RNG.standard_normal(n), "bandpass", [lo, hi])
    return x * np.exp(-t / decay) * np.minimum(1.0, t / attack)


def sweep(dur, f_lo, f_hi, rising=True, q=1.4):
    """Band-passed noise whose centre frequency glides (a whoosh)."""
    n = int(dur * SR)
    x = RNG.standard_normal(n)
    out = np.zeros(n)
    blocks = 24
    edges = np.linspace(0, n, blocks + 1).astype(int)
    for i in range(blocks):
        u = i / (blocks - 1)
        fc = f_lo * (f_hi / f_lo) ** (u if rising else 1 - u)
        lo, hi = fc / q, min(fc * q, SR / 2 - 100)
        seg = filt(x[max(0, edges[i] - 2000):edges[i + 1]], "bandpass", [lo, hi])
        out[edges[i]:edges[i + 1]] = seg[-(edges[i + 1] - edges[i]):]
    t = np.linspace(0, 1, n)
    return out * np.sin(np.pi * t) ** 1.5


def bubbles(dur, count, lo=350, hi=1400):
    out = np.zeros(int(dur * SR) + SR)
    for _ in range(count):
        f0 = RNG.uniform(lo, hi)
        c = chirp(f0, f0 * RNG.uniform(1.6, 2.4), RNG.uniform(0.03, 0.07), RNG.uniform(0.015, 0.03))
        i = int(RNG.uniform(0, dur) * SR)
        out[i:i + len(c)] += c * RNG.uniform(0.3, 1.0)
    return filt(out[:int(dur * SR)], "lowpass", 2500)


def place(buf, clip, t0, gain=1.0, pan=0.0):
    """Mix a mono clip into a stereo buffer at time t0 (seconds)."""
    i = int(round(t0 * SR))
    if i >= len(buf):
        return
    clip = clip[:len(buf) - max(i, 0)]
    if i < 0:
        clip, i = clip[-i:], 0
    lg, rg = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[i:i + len(clip), 0] += clip * gain * lg * 1.414
    buf[i:i + len(clip), 1] += clip * gain * rg * 1.414


def smooth_mask(per_frame, fps, n, ramp=0.12):
    """Per-frame 0/1 flags -> per-sample mask with short crossfades."""
    t = np.arange(n) / SR
    m = np.interp(t, np.arange(len(per_frame)) / fps, np.asarray(per_frame, dtype=float))
    k = int(ramp * SR)
    c = np.cumsum(np.concatenate([np.full(k // 2, m[0]), m, np.full(k - k // 2, m[-1])]))
    return (c[k:] - c[:-k])[:len(m)] / k


# --------------------------------------------------------------------------
# Music: soft F major loop (Fmaj7 - Am7 - Dm7 - Bbmaj7) at 92 BPM
# --------------------------------------------------------------------------
def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


CHORDS = [[53, 57, 60, 64], [57, 60, 64, 67], [50, 57, 60, 65], [58, 62, 65, 69]]
BASS = [41, 45, 38, 46]


def music(n, under, drums_ok):
    bpm = 92.0
    beat = 60.0 / bpm
    bar = 4 * beat
    out = np.zeros((n, 2))
    total = n / SR
    nbars = int(total / bar) + 1
    pad = np.zeros(n)
    for b in range(nbars):
        chord = CHORDS[b % 4]
        t0 = b * bar
        dur = bar + 1.2
        tt = np.arange(int(dur * SR)) / SR
        env = np.minimum(1.0, tt / 0.6) * np.minimum(1.0, np.maximum(0.0, (dur - tt) / 1.2))
        note = np.zeros(len(tt))
        for m in chord:
            for det in (-0.12, 0.0, 0.11):
                f = midi(m + det * 0.1)
                note += sum(np.sin(2 * np.pi * f * h * tt + RNG.uniform(0, 6.28)) / h ** 1.6 for h in (1, 2, 3, 4))
        i = int(t0 * SR)
        seg = (note * env)[:max(0, n - i)]
        pad[i:i + len(seg)] += seg
    pad = filt(pad, "lowpass", 1600) * 0.010
    out[:, 0] += pad
    out[:, 1] += np.roll(pad, int(0.011 * SR))

    # marimba-like arpeggio in eighth notes
    pattern = [0, 2, 1, 3, 2, 1, 3, 2]
    for b in range(nbars):
        chord = CHORDS[b % 4]
        for k, idx in enumerate(pattern):
            t0 = b * bar + k * beat / 2
            if t0 >= total:
                break
            f = midi(chord[idx] + 12)
            nt = tone(f, 0.9, 0.32, partials=((1.0, 1.0), (3.95, 0.22), (9.9, 0.05)))
            vel = 0.55 + 0.15 * (k % 2 == 0) + RNG.uniform(-0.05, 0.05)
            place(out, nt, t0, 0.045 * vel, pan=0.35 * np.sin(k * 0.9))

    # bass, soft kick and shaker (the beat drops out underwater)
    for b in range(nbars):
        for k in (0, 2):
            t0 = b * bar + k * beat
            if t0 >= total or b < 2:
                continue
            f = midi(BASS[b % 4])
            place(out, tone(f, 1.3, 0.55, partials=((1.0, 1.0), (2.0, 0.25))), t0, 0.09)
            if drums_ok(t0):
                place(out, chirp(110, 48, 0.28, 0.09), t0, 0.09)
        for k in range(8):
            t0 = b * bar + (k + 0.5) * beat / 2
            if t0 < total and b >= 2 and drums_ok(t0):
                place(out, filt(noise_burst(0.06, 5000, 12000, 0.02), "highpass", 4000), t0,
                      0.022 * (1.2 if k % 2 else 0.8), pan=0.4)

    # muffle the music while the camera is underwater
    muffled = filt(out, "lowpass", 650)
    u = under[:, None]
    out = out * (1 - u) + muffled * u * 1.1
    # fade in over the first beat, fade out over the last 2.5 s
    t = np.arange(n) / SR
    out *= (np.minimum(1.0, t / 0.6) * np.clip((total - t) / 2.5, 0, 1))[:, None]
    return out


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build(ev):
    fps = ev["fps"]
    total = ev["frames"] / fps
    n = int(round(total * SR))
    t = np.arange(n) / SR

    def ft(frame):
        return (frame - 1) / fps          # frame 1 is shown at t = 0

    under = smooth_mask(ev["under"], fps, n)
    above = 1.0 - under
    cam_z = np.interp(t, np.arange(ev["frames"]) / fps, np.asarray(ev["camera_z"]))
    out = np.zeros((n, 2))

    # Surf ambience: steady wash plus a crash every wave period, quieter from high up
    period = ev["wave_period_s"]
    wash = np.stack([filt(pink(n), "bandpass", [140, 3200]) for _ in range(2)], axis=1) * 0.05
    crash_env = np.zeros(n)
    for k in range(int(total / period) + 2):
        tc = k * period + 1.3
        rise = np.clip((t - (tc - 1.6)) / 1.6, 0, 1) ** 2
        fall = np.exp(-np.clip(t - tc, 0, None) / 1.4)
        crash_env += np.where(t < tc, rise, fall)
    crash = np.stack([filt(pink(n), "bandpass", [250, 6500]) for _ in range(2)], axis=1)
    surf = wash + crash * crash_env[:, None] * 0.07
    distance = 1.0 / (1.0 + np.maximum(cam_z, 0.0) / 22.0)
    out += surf * (above * (0.35 + 0.65 * distance))[:, None]

    # Underwater bed: low rumble, slow swell and sparse bubbles
    rumble = filt(filt(brown(n), "lowpass", 420), "highpass", 70) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.09 * t))
    bub = bubbles(total, int(total * 1.6), 300, 1100) * 0.08
    uw = (rumble + bub)
    out += np.stack([uw, np.roll(uw, 380)], axis=1) * under[:, None] * 0.55

    # Transitions: whoosh into a splash and bubbles (down), bubbles into a breach (up)
    for i, (a, b) in enumerate(ev["wipes"]):
        going_down = ev["under"][min(b + 4, ev["frames"] - 1)] == 1
        if going_down:
            place(out, sweep(0.45, 400, 3500, rising=True), ft(a - 9), 0.11)
            place(out, noise_burst(0.7, 900, 9000, 0.22), ft(a), 0.09)
            place(out, chirp(170, 55, 0.35, 0.12), ft(a), 0.13)
            place(out, bubbles(1.1, 45, 300, 1300), ft(a) + 0.05, 0.15)
        else:
            place(out, bubbles(0.9, 40, 400, 1500), ft(a - 9), 0.2)
            place(out, noise_burst(0.55, 1200, 10000, 0.16), ft(b), 0.13)
            place(out, sweep(0.5, 2500, 500, rising=False), ft(b) + 0.05, 0.07)

    # Overlay sounds
    for frame, kind, _label in ev["overlays"]:
        t0 = ft(frame)
        if kind == "badge":
            place(out, tone(880.0, 0.35, 0.09, partials=((1.0, 1.0), (2.0, 0.3))), t0, 0.07)
            place(out, tone(1318.5, 0.35, 0.10, partials=((1.0, 1.0), (2.0, 0.2))), t0 + 0.06, 0.06)
        elif kind == "pin":
            place(out, tone(1567.98, 0.2, 0.05, partials=((1.0, 1.0), (2.76, 0.15))), t0, 0.055, pan=0.2)
        elif kind == "headline":
            for j, f in enumerate((698.46, 880.0, 1046.5, 1318.5)):
                place(out, tone(f, 2.2, 0.7, partials=((1.0, 1.0), (2.76, 0.12), (5.4, 0.04))),
                      t0 + 0.07 * j, 0.03, pan=-0.3 + 0.2 * j)

    # The cast: backswing, forward whoosh, line zing in flight, plop on landing
    c = ev["cast"]
    place(out, sweep(0.55, 300, 900, rising=True), ft(c["backswing"]), 0.05)
    place(out, sweep(0.32, 700, 4200, rising=True), ft(c["forward"]), 0.12)
    fly = (c["land"] - c["release"]) / fps
    zt = np.arange(int(fly * SR)) / SR
    zing = np.sin(2 * np.pi * (2300 * zt - 400 * zt ** 2)) * np.exp(-zt / (fly * 0.6)) * np.minimum(1, zt / 0.05)
    place(out, zing, ft(c["release"]), 0.014)
    place(out, chirp(650, 140, 0.18, 0.06), ft(c["land"]), 0.14)
    place(out, noise_burst(0.35, 1500, 8000, 0.09), ft(c["land"]), 0.05)

    # Bounce jig: muffled thumps and sand hiss on each bottom contact
    for frame in ev["jig_contacts"]:
        place(out, chirp(150, 70, 0.18, 0.05), ft(frame), 0.28)
        place(out, filt(noise_burst(0.25, 300, 2500, 0.07), "lowpass", 2200), ft(frame), 0.16)

    # Strike, a small reward chime for "Fish on!", then reel clicks
    s = ev["strike"]
    place(out, chirp(90, 40, 0.4, 0.12), ft(s), 0.22)
    place(out, bubbles(0.8, 40, 300, 1200), ft(s), 0.17)
    for j, f in enumerate((1046.5, 1318.5, 1568.0)):
        place(out, tone(f, 0.8, 0.25, partials=((1.0, 1.0), (2.0, 0.2))), ft(s + 4) + 0.08 * j, 0.05)
    tk = ft(s + 6)
    while tk < ft(ev["reel_end"]):
        click = filt(noise_burst(0.012, 2000, 9000, 0.003), "lowpass", 3000)
        place(out, click, tk, 0.05 * RNG.uniform(0.7, 1.0), pan=0.25)
        tk += 1.0 / 21.0

    # Music bed under everything; drums sit out underwater
    under_frames = ev["under"]

    def drums_ok(tsec):
        f = min(len(under_frames) - 1, int(tsec * fps))
        lo, hi = max(0, f - 6), min(len(under_frames), f + 6)
        return not any(under_frames[lo:hi])
    out += music(n, under, drums_ok) * 1.35

    # Gentle master fades; remove sub-bass that small speakers cannot play
    out = filt(out, "highpass", 35)
    out *= (np.minimum(1.0, t / 0.05) * np.clip((total - t) / 0.6, 0, 1))[:, None]
    return out


def write_wav(path, data):
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


COMP = "acompressor=threshold=-20dB:ratio=2.5:attack=8:release=200:makeup=2"


def loudnorm(src, dst, target=-16.0, tp=-1.5):
    """Gentle compression, then two-pass EBU R128 normalisation with ffmpeg."""
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af",
                            f"{COMP},loudnorm=I={target}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    m = json.loads(probe[probe.rindex("{"):probe.rindex("}") + 1])
    af = (f"{COMP},loudnorm=I={target}:TP={tp}:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-af", af, "-ar", str(SR), dst], check=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events")
    ap.add_argument("wav")
    ap.add_argument("--video", default=None, help="silent video to add the soundtrack to")
    ap.add_argument("--out", default=None, help="output video with audio")
    args = ap.parse_args()
    with open(args.events) as fh:
        ev = json.load(fh)
    mix = build(ev)
    mix /= max(1e-6, np.abs(mix).max()) / 0.7
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw.wav")
        write_wav(raw, mix)
        m = loudnorm(raw, args.wav)
    print("loudness before normalisation:", m["input_i"], "LUFS, true peak", m["input_tp"], "dBTP")
    if args.video and args.out:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", args.video, "-i", args.wav, "-map", "0:v:0",
                        "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        "-movflags", "+faststart", args.out], check=True)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
