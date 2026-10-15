"""Procedural soundtrack for the sockeye masterclass.

Everything is synthesised here (no samples): the river (rush above water, roar and bubbles below, louder
by the riffles), the slide wipes, overlay pops, the flip and plop of each drift, the take, the hook-set
swish and the drag giving line, the thrash and the leap, reel clicks in the fight, and a music bed that
punches in for the hook, dips under the rig breakdown, builds to the strike and resolves on the final card.
Timing comes from the scene script:

    python sockeye_masterclass.py --events events.json
    python sockeye_audio.py events.json soundtrack.wav [--video preview.mp4 --out preview_audio.mp4]

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


def splash(dur, big=1.0):
    body = noise_burst(dur, 300, 9000, dur * 0.35) * big
    c = chirp(240, 70, min(dur, 0.35), 0.12) * 0.4 * big
    body[:len(c)] += c
    return body


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
    s0, s1 = ev["split"]
    split = smooth_mask([1 if s0 <= f + 1 <= s1 else 0 for f in range(ev["frames"])], fps, n)
    under = np.clip(under + 0.5 * split, 0, 1)            # over-under: half above, half below
    above = 1.0 - under
    cam_z = per_frame(ev["camera_z"])
    rif = per_frame(ev["riffle"])
    out = np.zeros((n, 2))

    # The river above water: a broad rush, louder by the riffles, softer from high up
    rush = np.stack([filt(pink(n), "bandpass", [160, 7000]) for _ in range(2)], axis=1) * 0.04
    babble = np.stack([filt(pink(n), "bandpass", [900, 4500]) for _ in range(2)], axis=1)
    babble *= (0.6 + 0.4 * np.sin(2 * np.pi * 3.1 * t + np.sin(2 * np.pi * 0.7 * t)) ** 2)[:, None] * 0.02
    alt = 1.0 / (1.0 + np.maximum(cam_z, 0.0) / 12.0)
    lvl = (0.45 + 0.55 * rif) * (0.35 + 0.65 * alt)
    out += (rush + babble) * (above * lvl)[:, None]
    # Below: low roar and streams of bubbles, much louder in the white water of the hook
    rum = filt(filt(brown(n), "lowpass", 420), "highpass", 50) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.11 * t))
    bub = bubbles(total, int(total * 6), 300, 1400) * 0.07
    white = (1 - np.clip((t - 5.2) / 0.4, 0, 1))
    uw = rum * (0.5 + 0.9 * white) + bub * (4.0 + 10.0 * white)
    out += np.stack([uw, np.roll(uw, 380)], axis=1) * under[:, None] * 0.32

    # Slide wipes: a whoosh; crossing the surface adds a splash or a bubble burst
    for (a, b), cut in zip(ev["wipes"], ev["cuts"]):
        place(out, sweep(0.42, 300, 4200, rising=True), ft(a - 7), 0.11)
        before, after = ev["under"][max(0, cut - 3)], ev["under"][min(len(ev["under"]) - 1, cut + 2)]
        if before and not after:
            place(out, bubbles(0.7, 36, 400, 1500), ft(a - 5), 0.16)
            place(out, noise_burst(0.5, 1200, 10000, 0.14), ft(cut), 0.09)
        elif after and not before:
            place(out, splash(0.6, 0.9), ft(cut), 0.1)
            place(out, bubbles(1.0, 40, 300, 1300), ft(cut) + 0.05, 0.14)

    # Overlay sounds
    for frame, kind, _label in ev["overlays"]:
        t0 = ft(frame)
        if kind == "badge":
            place(out, tone(880.0, 0.35, 0.09, partials=((1.0, 1.0), (2.0, 0.3))), t0, 0.06)
            place(out, tone(1318.5, 0.35, 0.10, partials=((1.0, 1.0), (2.0, 0.2))), t0 + 0.06, 0.05)
        elif kind == "pin":
            place(out, tone(1567.98, 0.2, 0.05, partials=((1.0, 1.0), (2.76, 0.15))), t0, 0.045, pan=0.2)
        elif kind == "caption":
            place(out, filt(noise_burst(0.25, 2000, 9000, 0.06), "highpass", 1500), t0, 0.02, pan=-0.15)
        elif kind == "headline":
            for j, f in enumerate((587.33, 739.99, 880.0, 1174.66)):
                place(out, tone(f, 2.2, 0.7, partials=((1.0, 1.0), (2.76, 0.12), (5.4, 0.04))),
                      t0 + 0.07 * j, 0.03, pan=-0.3 + 0.2 * j)

    # Each drift: the flip lands with a plop above water, a plunk and bubbles below
    for fr in ev["drift_entries"]:
        if fr > ev["frames"]:
            continue
        u = ev["under"][fr - 1]
        if u:
            place(out, chirp(420, 180, 0.12, 0.05), ft(fr), 0.05, pan=-0.3)
            place(out, bubbles(0.5, 14, 500, 1600), ft(fr), 0.08, pan=-0.3)
        else:
            place(out, chirp(900, 300, 0.09, 0.03), ft(fr), 0.035, pan=-0.3)

    # The take, the hook-set and the drag
    s, hs = ev["strike"], ev["hookset"]
    place(out, chirp(95, 38, 0.4, 0.12), ft(s), 0.3)
    place(out, bubbles(0.7, 30, 300, 1300), ft(s), 0.16)
    place(out, sweep(0.28, 600, 5000, rising=True), ft(hs) - 0.12, 0.16)          # rod swish
    place(out, chirp(2400, 900, 0.18, 0.06), ft(hs) + 0.1, 0.05)                    # line coming tight
    sc = scream(5.5)
    env = np.clip(1 - np.arange(len(sc)) / len(sc), 0, 1) ** 0.7
    sc = sc * env * (0.55 + 0.45 * np.sin(2 * np.pi * 0.8 * np.arange(len(sc)) / SR) ** 2)
    i0 = int((ft(hs) + 0.25) * SR)
    seg = sc[:max(0, n - i0)]
    out[i0:i0 + len(seg), 0] += seg * 0.08
    out[i0:i0 + len(seg), 1] += seg * 0.1
    cut4 = ev["cuts"][3]
    for fr in range(cut4, ev["leap"][0], 7):                                        # thrashing at the surface
        place(out, splash(0.3, 1.0), ft(fr), 0.05 * RNG.uniform(0.6, 1.0), pan=-0.2)
    j0, j1 = ev["leap"]
    place(out, splash(0.8, 1.0), ft(j0 + 2), 0.2)
    place(out, splash(1.1, 1.3), ft(j1 - 1), 0.25)
    tk = ft(1400)
    while tk < ft(1700):                                                            # winding in
        if ((tk - ft(1400)) * 0.8) % 1.0 > 0.4:
            place(out, filt(noise_burst(0.012, 2000, 9000, 0.003), "lowpass", 3200), tk, 0.04 * RNG.uniform(0.7, 1.0), pan=-0.2)
        tk += 1.0 / 14.0

    # Music: punchy from the first frame (the hook), lighter under the rig, building to the strike
    def intensity(tsec):
        f = tsec * fps
        hook = 1.0 if f < ev["cuts"][0] else 0.0
        rig = np.clip((f - ev["cuts"][1]) / 60, 0, 1) * (1 - np.clip((f - 900) / 180, 0, 1))
        build_ = np.clip((f - 900) / 280, 0, 1)
        outro = np.clip((f - 1640) / 120, 0, 1)
        return float(max(hook, 0.65 - 0.4 * rig + 0.35 * build_ - 0.3 * outro))

    def drums_ok(tsec):
        f = tsec * fps
        return not (ev["cuts"][1] + 20 < f < 880)
    mus = music(n, np.zeros(n), drums_ok, intensity)
    out += mus * 1.3
    for j, f in enumerate((293.66, 369.99, 440.0, 587.33)):                       # final card chord
        place(out, tone(f, 3.5, 1.4, partials=((1.0, 1.0), (2.0, 0.4), (3.0, 0.15))), ft(1700) + 0.04 * j, 0.05)

    out = filt(out, "highpass", 35)
    out *= (np.minimum(1.0, t / 0.03) * np.clip((total - t) / 0.8, 0, 1))[:, None]
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
