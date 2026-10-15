"""Procedural soundtrack for the offshore trolling breakdown.

Everything is synthesised here (no samples): open-ocean wind and hull slap, the twin diesels at trolling
speed (and their props underwater), bait splashes, the dive and the cut back to the cockpit, overlay pops,
the strike with the release clip snapping and the reel's drag screaming, the fight with the jump, and a
music bed. Timing comes from the scene script:

    python offshore_fishing.py --events events.json
    python offshore_audio.py events.json soundtrack.wav [--video preview.mp4 --out preview_audio.mp4]

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
# Music: D major loop (D - Bm - G - A) at 100 BPM, building towards the strike
# --------------------------------------------------------------------------
def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


CHORDS = [[50, 54, 57, 62], [47, 54, 57, 62], [43, 50, 55, 59], [45, 52, 57, 61]]
BASS = [38, 35, 31, 33]


def music(n, under, drums_ok, intensity):
    bpm = 100.0
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
                f = midi(m + 12 + det * 0.1)
                note += sum(np.sin(2 * np.pi * f * h * tt + RNG.uniform(0, 6.28)) / h ** 1.6 for h in (1, 2, 3, 4))
        i = int(t0 * SR)
        seg = (note * env)[:max(0, n - i)]
        pad[i:i + len(seg)] += seg
    pad = filt(pad, "lowpass", 1800) * 0.009
    out[:, 0] += pad
    out[:, 1] += np.roll(pad, int(0.011 * SR))
    pattern = [0, 2, 3, 1, 2, 3, 1, 2]
    for b in range(nbars):
        chord = CHORDS[b % 4]
        for k, idx in enumerate(pattern):
            t0 = b * bar + k * beat / 2
            if t0 >= total:
                break
            f = midi(chord[idx] + 24)
            nt = tone(f, 0.7, 0.22, partials=((1.0, 1.0), (2.0, 0.35), (3.0, 0.12)))
            vel = 0.55 + 0.15 * (k % 2 == 0) + RNG.uniform(-0.05, 0.05)
            place(out, nt, t0, 0.04 * vel * (0.7 + 0.5 * intensity(t0)), pan=0.35 * np.sin(k * 0.9))
    for b in range(nbars):
        for k in (0, 1.5, 2, 3):
            t0 = b * bar + k * beat
            if t0 >= total or b < 2:
                continue
            f = midi(BASS[b % 4])
            if k in (0, 2):
                place(out, tone(f, 1.0, 0.45, partials=((1.0, 1.0), (2.0, 0.3))), t0, 0.085)
            if drums_ok(t0) and k in (0, 2):
                place(out, chirp(115, 46, 0.26, 0.085), t0, 0.1 * (0.8 + 0.4 * intensity(t0)))
            if drums_ok(t0) and k in (1.5, 3) and intensity(t0) > 0.5:
                place(out, filt(noise_burst(0.16, 1200, 7000, 0.05), "highpass", 900), t0, 0.035)
        for k in range(8):
            t0 = b * bar + (k + 0.5) * beat / 2
            if t0 < total and b >= 2 and drums_ok(t0):
                place(out, filt(noise_burst(0.05, 6000, 13000, 0.018), "highpass", 5000), t0,
                      0.02 * (1.2 if k % 2 else 0.8), pan=0.4)
    muffled = filt(out, "lowpass", 600)
    u = under[:, None]
    out = out * (1 - u) + muffled * u * 1.15
    t = np.arange(n) / SR
    out *= (np.minimum(1.0, t / 0.6) * np.clip((total - t) / 2.5, 0, 1))[:, None]
    return out


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def scream(dur, rate0=520.0, rate1=380.0):
    """A reel's drag giving line: fast ratchet clicks through a resonant body."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    rate = rate0 + (rate1 - rate0) * t / dur
    phase = np.cumsum(rate) / SR
    clicks = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
    x = signal.lfilter([1.0], [1.0, -1.6 * np.cos(2 * np.pi * 3200 / SR), 0.81], clicks)
    x += signal.lfilter([1.0], [1.0, -1.7 * np.cos(2 * np.pi * 5100 / SR), 0.86], clicks) * 0.6
    return x / max(1e-6, np.abs(x).max())


def build(ev):
    fps = ev["fps"]
    total = ev["frames"] / fps
    n = int(round(total * SR))
    t = np.arange(n) / SR

    def ft(frame):
        return (frame - 1) / fps

    def per_frame(vals):
        return np.interp(t, np.arange(len(vals)) / fps, np.asarray(vals, dtype=float))

    under = smooth_mask(ev["under"], fps, n)
    above = 1.0 - under
    cam_z = per_frame(ev["camera_z"])
    speed = per_frame(ev["boat_speed"])
    near = 1.0 / (1.0 + per_frame(ev["cam_boat_dist"]) / 14.0)
    out = np.zeros((n, 2))

    # Open ocean: wind, small chop, hull slap; quieter from high up
    wind = np.stack([filt(pink(n), "bandpass", [180, 5000]) for _ in range(2)], axis=1) * 0.035
    gust = 0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * t) * np.sin(2 * np.pi * 0.023 * t + 1.0)
    slap_env = np.zeros(n)
    for k in range(int(total / 1.9) + 2):
        tc = k * 1.9 + RNG.uniform(-0.3, 0.3)
        slap_env += np.exp(-np.clip(t - tc, 0, None) / 0.25) * (t >= tc) * RNG.uniform(0.4, 1.0)
    slap = filt(pink(n), "bandpass", [150, 1400]) * slap_env * 0.05
    alt = 1.0 / (1.0 + np.maximum(cam_z, 0.0) / 30.0)
    out += (wind * gust[:, None] + np.stack([slap, np.roll(slap, 400)], axis=1) * near[:, None]) * \
        (above * (0.45 + 0.55 * alt))[:, None]

    # Twin diesels: harmonic drone that follows the throttle; louder near the boat
    f0 = 46.0 * (0.72 + 0.28 * speed / 3.1)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    drone = sum(a * np.sin(h * ph + RNG.uniform(0, 6.28)) for h, a in ((1, 0.5), (2, 1.0), (3, 0.7), (4, 0.45),
                                                                       (6, 0.25), (8, 0.12)))
    drone *= 1.0 + 0.18 * np.sin(ph * 0.125)                      # firing lumpiness
    rumble = filt(brown(n), "bandpass", [70, 380]) * 3.0
    engine = (drone * 0.05 + rumble * 0.04) * (0.4 + 0.6 * speed / 3.1)
    wake = filt(pink(n), "bandpass", [600, 7000]) * 0.03 * (speed / 3.1)
    above_mix = (engine + wake) * near * 1.6
    prop = filt(RNG.standard_normal(n), "bandpass", [700, 1600]) * (0.5 + 0.5 * np.sin(ph * 0.5) ** 2)
    under_mix = (filt(engine, "lowpass", 300) * 0.6 + prop * 0.01 * (speed / 3.1)) * (0.35 + 0.65 * near)
    mix = above_mix * above + under_mix * under
    out += np.stack([mix, np.roll(mix, 220)], axis=1)

    # Underwater bed: low swell and sparse bubbles
    rum = filt(filt(brown(n), "lowpass", 380), "highpass", 60) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.08 * t))
    bub = bubbles(total, int(total * 1.4), 300, 1100) * 0.07
    uw = rum * 0.55 + bub * 6.0                                  # keep the rumble under the music
    out += np.stack([uw, np.roll(uw, 380)], axis=1) * under[:, None] * 0.3

    # Baits popping back up through the surface
    for name, frames in ev["bait_pops"].items():
        for fr in frames:
            u = ev["under"][min(fr - 1, len(ev["under"]) - 1)]
            if u:
                place(out, bubbles(0.35, 10, 500, 1600), ft(fr), 0.07, pan=-0.3)
            else:
                place(out, chirp(700, 260, 0.12, 0.04), ft(fr), 0.035, pan=-0.3)

    # The dive (whoosh, splash, bubbles) and the cut from the strike to the cockpit (breach)
    for a, b in ev["wipes"]:
        if b < ev["cut"] - 10:
            place(out, sweep(0.45, 400, 3500, rising=True), ft(a - 9), 0.1)
            place(out, noise_burst(0.7, 900, 9000, 0.22), ft(a), 0.09)
            place(out, chirp(170, 55, 0.35, 0.12), ft(a), 0.12)
            place(out, bubbles(1.1, 45, 300, 1300), ft(a) + 0.05, 0.14)
        else:
            place(out, bubbles(0.8, 36, 400, 1500), ft(a - 8), 0.18)
            place(out, noise_burst(0.55, 1200, 10000, 0.16), ft(b), 0.12)
            place(out, sweep(0.5, 2500, 500, rising=False), ft(b) + 0.05, 0.06)

    # Overlay sounds
    for frame, kind, _label in ev["overlays"]:
        t0 = ft(frame)
        if kind == "badge":
            place(out, tone(880.0, 0.35, 0.09, partials=((1.0, 1.0), (2.0, 0.3))), t0, 0.065)
            place(out, tone(1318.5, 0.35, 0.10, partials=((1.0, 1.0), (2.0, 0.2))), t0 + 0.06, 0.055)
        elif kind == "pin":
            place(out, tone(1567.98, 0.2, 0.05, partials=((1.0, 1.0), (2.76, 0.15))), t0, 0.05, pan=0.2)
        elif kind == "headline":
            for j, f in enumerate((587.33, 739.99, 880.0, 1174.66)):
                place(out, tone(f, 2.2, 0.7, partials=((1.0, 1.0), (2.76, 0.12), (5.4, 0.04))),
                      t0 + 0.07 * j, 0.03, pan=-0.3 + 0.2 * j)
        elif kind == "gauge":
            for j in range(6):
                place(out, filt(noise_burst(0.015, 2500, 9000, 0.004), "lowpass", 5000), t0 - 0.9 + 0.15 * j, 0.04)

    # The strike: thump and bubbles, the release clip snapping, then the drag screams
    s = ev["strike"]
    place(out, chirp(95, 38, 0.45, 0.13), ft(s), 0.42)
    place(out, bubbles(0.9, 50, 300, 1300), ft(s), 0.24)
    snap = filt(noise_burst(0.03, 2500, 12000, 0.006), "highpass", 1800)
    place(out, snap, ft(s + 2), 0.16, pan=-0.4)
    run = (ev["cut"] + 200 - s) / fps
    sc = scream(run)
    env = np.clip(1 - np.arange(len(sc)) / len(sc), 0, 1) ** 0.6
    pulses = 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * np.arange(len(sc)) / SR) ** 2
    sc = sc * env * pulses
    i0 = int(ft(s + 3) * SR)
    seg = sc[:max(0, n - i0)]
    gain = np.where(under[i0:i0 + len(seg)] > 0.5, 0.02, 0.13)       # a faint, far-off whine underwater
    out[i0:i0 + len(seg), 0] += seg * gain * 0.8
    out[i0:i0 + len(seg), 1] += seg * gain

    # The fight: winding clicks while the angler pumps and reels, the jump and the thrashing
    t_take = ft(ev["angler_takes_rod"][1])
    tk = t_take
    while tk < total - 2.0:
        cyc = (tk - t_take) * 0.075 * fps / (2 * np.pi)
        if (cyc % 1.0) > 0.45:
            click = filt(noise_burst(0.012, 2000, 9000, 0.003), "lowpass", 3200)
            place(out, click, tk, 0.045 * RNG.uniform(0.7, 1.0), pan=-0.2)
        tk += 1.0 / 16.0
    j0, j1 = ev["jump"]
    place(out, noise_burst(0.9, 500, 9000, 0.25), ft(j0 + 2), 0.22)
    place(out, noise_burst(1.1, 300, 8000, 0.32), ft(j1 - 1), 0.27)
    place(out, chirp(220, 60, 0.4, 0.15), ft(j1 - 1), 0.1)
    for fr in range(1500, ev["frames"], 16):
        place(out, noise_burst(0.35, 600, 7000, 0.1), ft(fr), 0.045 * RNG.uniform(0.6, 1.0), pan=-0.25)

    # Music: builds from the rise of the school to the strike, eases off for the release
    def intensity(tsec):
        f = tsec * fps
        up = np.clip((f - 600) / 400, 0, 1)
        down = np.clip((f - 1450) / 300, 0, 1)
        return float(up * (1 - 0.6 * down))
    under_frames = ev["under"]

    def drums_ok(tsec):
        f = min(len(under_frames) - 1, int(tsec * fps))
        lo, hi = max(0, f - 6), min(len(under_frames), f + 6)
        return not any(under_frames[lo:hi])
    out += music(n, under, drums_ok, intensity) * 1.25

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
