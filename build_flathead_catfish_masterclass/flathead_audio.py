"""Soundtrack for the flathead catfish masterclass: voiceover, music bed and synthesised effects.

The voiceover clips are generated text to speech (voiceover/vo1..vo5, mp3 or wav; see SOURCES.md for the
script). Everything else is synthesised here: a low swell under the dolly zoom, the muffled hush of a stained
river, dusk crickets and flowing water over the aerial, a plunge at the dive, the bluegill's kicks, the strike
(a deep suction thump, bubbles), the surface break, the reel's drag and the rod under load, whooshes for the
wipes, overlay pops, a chime under the call to action, and a dark music bed that ducks under the voice and drops
out for the strike.

    python flathead_audio.py --measure                     # writes vo_timing.json from the clips
    python flathead_catfish_masterclass.py --events events.json    # (re-run after measuring: captions follow)
    python flathead_audio.py events.json soundtrack.wav [--video preview.mp4 --out preview_audio.mp4]

Needs numpy, scipy and ffmpeg.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import wave

import numpy as np
from scipy import signal

SR = 48000
RNG = np.random.default_rng(11)
HERE = os.path.dirname(os.path.abspath(__file__))
VO_DIR = os.path.join(HERE, "voiceover")


# --------------------------------------------------------------------------
# DSP helpers
# --------------------------------------------------------------------------
def filt(x, kind, fc, order=2):
    sos = signal.butter(order, fc, btype=kind, fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def tone(freq, dur, decay, partials=((1.0, 1.0),), attack=0.004):
    t = np.arange(int(dur * SR)) / SR
    out = sum(a * np.sin(2 * np.pi * freq * r * t) * np.exp(-t / (decay / r ** 0.5)) for r, a in partials)
    return out * np.minimum(1.0, t / attack)


def chirp(f0, f1, dur, decay):
    t = np.arange(int(dur * SR)) / SR
    f = f0 * (f1 / f0) ** (t / dur)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / decay) * np.minimum(1.0, t / 0.002)


def noise_burst(dur, lo, hi, decay, attack=0.002):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return filt(RNG.standard_normal(n), "bandpass", [lo, hi]) * np.exp(-t / decay) * np.minimum(1.0, t / attack)


def sweep(dur, f_lo, f_hi, rising=True, q=1.4):
    n = int(dur * SR)
    x = RNG.standard_normal(n)
    out = np.zeros(n)
    edges = np.linspace(0, n, 25).astype(int)
    for i in range(24):
        u = i / 23
        fc = f_lo * (f_hi / f_lo) ** (u if rising else 1 - u)
        seg = filt(x[max(0, edges[i] - 2000):edges[i + 1]], "bandpass", [fc / q, min(fc * q, SR / 2 - 100)])
        out[edges[i]:edges[i + 1]] = seg[-(edges[i + 1] - edges[i]):]
    return out * np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5


def place(buf, clip, t0, gain=1.0, pan=0.0):
    i = int(round(t0 * SR))
    if i >= len(buf):
        return
    clip = clip[:len(buf) - max(i, 0)]
    if i < 0:
        clip, i = clip[-i:], 0
    if clip.ndim == 2:
        buf[i:i + len(clip)] += clip * gain
        return
    lg, rg = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    buf[i:i + len(clip), 0] += clip * gain * lg * 1.414
    buf[i:i + len(clip), 1] += clip * gain * rg * 1.414


def bubbles(dur, count, lo=350, hi=1400):
    out = np.zeros(int(dur * SR) + SR)
    for _ in range(count):
        f0 = RNG.uniform(lo, hi)
        c = chirp(f0, f0 * RNG.uniform(1.6, 2.4), RNG.uniform(0.03, 0.07), RNG.uniform(0.015, 0.03))
        i = int(RNG.uniform(0, dur) * SR)
        out[i:i + len(c)] += c * RNG.uniform(0.3, 1.0)
    return filt(out[:int(dur * SR)], "lowpass", 2500)


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


def suction():
    """The take: a short inrush of water (reversed noise swell) closing in a deep, heavy thump."""
    rush = filt(RNG.standard_normal(int(0.22 * SR)), "bandpass", [150, 1800])
    rush *= np.linspace(0, 1, len(rush)) ** 2.5
    thump = chirp(95, 32, 0.7, 0.22) * 1.6
    thump[:int(0.3 * SR)] += noise_burst(0.3, 40, 400, 0.08) * 0.8
    out = np.zeros(len(rush) + len(thump))
    out[:len(rush)] += rush * 0.7
    out[len(rush) - int(0.01 * SR):len(rush) - int(0.01 * SR) + len(thump)] += thump
    return out


def creak(dur):
    """Graphite under load: a slow, low groan with fine grain."""
    t = np.arange(int(dur * SR)) / SR
    f = 70 + 25 * np.sin(2 * np.pi * 0.7 * t)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.35 * t) ** 2)
    grain = filt(RNG.standard_normal(len(t)), "bandpass", [600, 2400]) * 0.15
    return (x + grain) * np.clip(t / 0.3, 0, 1) * np.clip((dur - t) / 0.3, 0, 1)


# --------------------------------------------------------------------------
# Voiceover
# --------------------------------------------------------------------------
def vo_path(k):
    for ext in ("wav", "mp3", "m4a", "flac", "ogg"):
        p = os.path.join(VO_DIR, f"vo{k}.{ext}")
        if os.path.exists(p):
            return p
    return None


def decode(path):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.?:])\s+", text) if p.strip()]


def measure(events_text):
    """Clip lengths and sentence starts: estimated from characters inside the voiced span, then snapped to
    the nearest pause (a stretch quieter than -38 dB for at least 0.12 s)."""
    out = {"length": {}, "phrases": {}, "lead": {}}
    for k, text in events_text.items():
        p = vo_path(int(k))
        if p is None:
            print(f"vo{k}: no file in {VO_DIR}")
            continue
        x = decode(p)
        hop = int(0.01 * SR)
        rms = np.sqrt(np.convolve(x * x, np.ones(hop) / hop, mode="same"))[::hop]
        db = 20 * np.log10(rms + 1e-9)
        voiced = np.where(db > -38)[0]
        t0, t1 = voiced[0] * 0.01, voiced[-1] * 0.01
        quiet = db < -38
        pauses = []                    # pause end times (speech resumes)
        run = 0
        for i, q in enumerate(quiet):
            if q:
                run += 1
            else:
                if run * 0.01 >= 0.12 and i * 0.01 > t0 + 0.05:
                    pauses.append(i * 0.01)
                run = 0
        parts = sentences(text)
        w = [len(s) + 8 for s in parts]
        starts = []
        for i in range(len(parts)):
            est = t0 + (t1 - t0) * sum(w[:i]) / sum(w)
            if i and pauses:
                near = min(pauses, key=lambda t: abs(t - est))
                if abs(near - est) < 0.6:
                    est = near
            starts.append(round(est, 3))
        out["length"][k] = round(len(x) / SR, 3)
        out["phrases"][k] = starts
        out["lead"][k] = round(t0, 3)
        print(f"vo{k}: {len(x) / SR:.2f} s, voiced {t0:.2f}-{t1:.2f} s, sentences at {starts}")
    with open(os.path.join(HERE, "vo_timing.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote vo_timing.json")


# --------------------------------------------------------------------------
# Music: dark, cinematic D minor (Dm - Bb - F - C) at 90 BPM; energy(t) scales the drums, gate(t) drops them out
# --------------------------------------------------------------------------
def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


CHORDS = [[50, 53, 57, 62], [46, 50, 53, 58], [53, 57, 60, 65], [48, 52, 55, 60]]
BASS = [26, 22, 29, 24]


def music(n, energy, gate):
    bpm = 90.0
    beat = 60.0 / bpm
    bar = 4 * beat
    total = n / SR
    nbars = int(total / bar) + 1
    out = np.zeros((n, 2))
    pad = np.zeros(n)
    for b in range(nbars):
        tt = np.arange(int((bar + 1.2) * SR)) / SR
        env = np.minimum(1.0, tt / 0.8) * np.clip((bar + 1.2 - tt) / 1.2, 0, 1)
        note = np.zeros(len(tt))
        for m in CHORDS[b % 4]:
            for det in (-0.1, 0.0, 0.09):
                f = midi(m + det * 0.1)
                note += sum(np.sin(2 * np.pi * f * h * tt + RNG.uniform(0, 6.28)) / h ** 1.6 for h in (1, 2, 3, 4))
        i = int(b * bar * SR)
        seg = (note * env)[:max(0, n - i)]
        pad[i:i + len(seg)] += seg
    pad = filt(pad, "lowpass", 1600) * 0.006
    out[:, 0] += pad
    out[:, 1] += np.roll(pad, int(0.017 * SR))
    t = np.arange(n) / SR
    drone = np.sin(2 * np.pi * midi(26) * t) * 0.03 + np.sin(2 * np.pi * midi(38) * t + 1.0) * 0.012
    out += (drone * (0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t)))[:, None]
    for b in range(nbars):
        chord = CHORDS[b % 4]
        for k in range(8):                                   # soft piano-like arpeggio
            t0 = b * bar + k * beat / 2
            if t0 >= total:
                break
            f = midi(chord[[0, 2, 3, 1, 2, 3, 1, 3][k]] + 12)
            place(out, tone(f, 0.9, 0.3, partials=((1.0, 1.0), (2.0, 0.25), (3.0, 0.08))), t0,
                  0.022 * (0.5 + 0.6 * energy(t0)), pan=0.3 * np.sin(k * 1.1))
        for k in (0, 1, 2, 2.5, 3):
            t0 = b * bar + k * beat
            if t0 >= total:
                continue
            e, g = energy(t0), gate(t0)
            if k in (0, 2):
                place(out, tone(midi(BASS[b % 4]), 0.9, 0.35, partials=((1.0, 1.0), (2.0, 0.3))), t0, 0.09 * g)
                place(out, chirp(110, 38, 0.35, 0.1), t0, 0.14 * e * g)                      # low kick
            if k in (1, 3) and e > 0.5:
                place(out, filt(noise_burst(0.25, 900, 7000, 0.07), "highpass", 700), t0, 0.035 * e * g)
            if k == 2.5 and e > 0.8:
                place(out, chirp(110, 38, 0.3, 0.08), t0, 0.09 * e * g)
        for k in range(8):
            t0 = b * bar + (k + 0.5) * beat / 2
            if t0 < total and energy(t0) > 0.6:
                place(out, filt(noise_burst(0.04, 7000, 14000, 0.012), "highpass", 6000), t0,
                      0.012 * energy(t0) * gate(t0), pan=0.35)
    out *= (np.minimum(1.0, t / 0.05) * np.clip((total - t) / 1.5, 0, 1))[:, None]
    return out


# --------------------------------------------------------------------------
# Mix
# --------------------------------------------------------------------------
def build(ev):
    fps = ev["fps"]
    total = ev["frames"] / fps
    n = int(round(total * SR))
    t = np.arange(n) / SR

    def ft(frame):
        return (frame - 1) / fps
    fx = np.zeros((n, 2))
    surf = ft(ev["surface"])
    # the hook: a low boom, then a swell that rises under the dolly zoom
    place(fx, chirp(70, 30, 2.5, 0.9), 0.0, 0.25)
    place(fx, noise_burst(1.5, 30, 300, 0.5), 0.0, 0.15)
    place(fx, sweep(ft(ev["beats"][1]) - 0.3, 60, 900, rising=True, q=1.3), 0.1, 0.10)
    # the river under water: a muffled hush and distant bubbles whenever the lens is under
    hush = filt(RNG.standard_normal(n), "lowpass", 300) * 0.05
    hush *= (0.7 + 0.3 * np.sin(2 * np.pi * 0.11 * np.arange(n) / SR))
    under = np.zeros(n)
    for a_, b_ in ((0, ft(ev["aerial"][0])), (ft(ev["dive"]), surf)):
        under[int(a_ * SR):int(b_ * SR)] = 1.0
    under = np.convolve(under, np.ones(2400) / 2400, mode="same")
    fx[:, 0] += hush * under
    fx[:, 1] += np.roll(hush, 900) * under
    for k in range(14):
        t0 = RNG.uniform(ft(ev["dive"]), surf - 1.0) if k % 3 else RNG.uniform(0.3, ft(ev["aerial"][0]) - 1.0)
        place(fx, bubbles(0.8, 6, 180, 600), t0, 0.05, pan=RNG.uniform(-0.7, 0.7))

    def dusk(t0, t1, gain):              # above water at dusk: flowing water and crickets
        i0, i1 = int(t0 * SR), int(t1 * SR)
        tt = np.arange(i1 - i0) / SR
        flow = filt(RNG.standard_normal(i1 - i0), "bandpass", [250, 2200]) * 0.03
        chirps = np.sin(2 * np.pi * 4300 * tt) * (np.sin(2 * np.pi * 28 * tt) > 0.3) * (np.sin(2 * np.pi * 0.9 * tt) > 0.0)
        env = np.clip(tt / 0.3, 0, 1) * np.clip((tt[-1] - tt) / 0.3, 0, 1)
        sig = (flow + chirps * 0.006) * env * gain
        fx[i0:i1, 0] += sig
        fx[i0:i1, 1] += np.roll(sig, 600)
    dusk(ft(ev["aerial"][0]), ft(ev["aerial"][1]), 1.0)
    dusk(surf, total, 0.8)
    # the dive under the surface (hidden by the wipe), and the bluegill's kicks on the leader
    place(fx, splash(0.7, 0.8), ft(ev["dive"]) - 0.1, 0.12)
    place(fx, bubbles(1.2, 24, 250, 900), ft(ev["dive"]), 0.08)
    k0, k1 = ft(ev["kicks"][0]), ft(ev["kicks"][1])
    tt = k0 + 0.4
    while tt < k1 - 0.3:
        place(fx, filt(noise_burst(0.18, 200, 1400, 0.06), "lowpass", 1600), tt, 0.04, pan=RNG.uniform(-0.3, 0.3))
        tt += 2.2 if int(tt) % 2 else 1.1
    # wipes
    for a, _b in ev["wipes"]:
        place(fx, sweep(0.42, 300, 4500, rising=True), ft(a - 7), 0.10)
    # the strike (slow motion): the suction thump, then a burst of bubbles
    st = ft(ev["strike"])
    place(fx, suction(), st - 0.22, 0.6)
    place(fx, bubbles(2.2, 40, 250, 1100), st + 0.05, 0.12)
    # the surface: a splash at the wipe, then water lapping, the rod groaning and the drag screaming
    place(fx, splash(0.9, 1.0), surf - 0.1, 0.18)
    m0 = int(surf * SR)
    lap = filt(RNG.standard_normal(n - m0), "bandpass", [200, 1500]) * 0.03
    lap *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.4 * np.arange(n - m0) / SR) ** 2
    fx[m0:, 0] += lap
    fx[m0:, 1] += np.roll(lap, 700)
    place(fx, creak(total - surf), surf, 0.06)
    for k, (d, r0, r1) in enumerate(((2.4, 560, 420), (1.4, 480, 360), (2.0, 520, 380))):
        place(fx, scream(d, r0, r1), surf + 0.4 + k * 2.6, 0.05, pan=0.25)
    place(fx, filt(RNG.standard_normal(int((total - surf) * SR)), "bandpass", [3000, 8000]) * 0.01, surf, 1.0, pan=0.5)
    pb = ft(ev["pullback"][0])
    place(fx, sweep(0.9, 200, 3000, rising=False, q=1.5), pb - 0.1, 0.10)
    # overlay pops
    for frame, kind, _label in ev["overlays"]:
        t0 = ft(frame)
        if kind == "badge":
            place(fx, tone(587.33, 0.4, 0.1, partials=((1.0, 1.0), (2.0, 0.3))), t0, 0.05)
            place(fx, tone(880.0, 0.4, 0.11, partials=((1.0, 1.0), (2.0, 0.2))), t0 + 0.06, 0.04)
        elif kind == "pin":
            place(fx, tone(1174.66, 0.2, 0.05, partials=((1.0, 1.0), (2.76, 0.15))), t0, 0.03, pan=0.2)
        elif kind == "caption":
            place(fx, filt(noise_burst(0.2, 2000, 9000, 0.05), "highpass", 1500), t0, 0.014, pan=-0.15)
        elif kind == "headline":
            for j, f in enumerate((587.33, 698.46, 880.0)):
                place(fx, tone(f, 1.6, 0.5, partials=((1.0, 1.0), (2.76, 0.1))), t0 + 0.06 * j, 0.025)
        elif kind == "cta":
            for j, f in enumerate((587.33, 739.99, 880.0, 1174.66)):
                place(fx, tone(f, 2.0, 0.6, partials=((1.0, 1.0), (2.0, 0.3))), t0 + 0.05 * j, 0.03)

    # voiceover
    vo = np.zeros(n)
    for k, f0 in ev["vo_start"].items():
        p = vo_path(int(k))
        if p is None:
            print(f"vo{k}: missing, mixed without it")
            continue
        x = decode(p)
        x = x / max(1e-6, np.abs(x).max()) * 0.9
        i = int(round(ft(f0) * SR))
        seg = x[:max(0, n - i)]
        vo[i:i + len(seg)] += seg
    vo = filt(vo, "highpass", 80)
    # duck the music and effects under the voice (smoothed envelope)
    env = np.sqrt(np.convolve(vo * vo, np.ones(int(0.05 * SR)) / int(0.05 * SR), mode="same"))
    env = np.clip(env / 0.08, 0, 1)
    env = signal.lfilter([1 - 0.9995], [1, -0.9995], env)
    duck = 1.0 - 0.55 * np.clip(env * 1.6, 0, 1)

    def energy(tsec):
        f = tsec * fps
        b = ev["beats"]
        return float(0.3 if f < b[1] else 0.6 if f < b[2] else 0.75 if f < b[3] else 1.0)

    def gate(tsec):                       # the drums and bass drop out for the strike and come back at the surface
        f = tsec * fps
        return float(0.0 if ev["slowmo"][0] - 4 <= f < ev["surface"] else 1.0)
    mus = music(n, energy, gate)
    mix = (mus * 1.0 + fx * 0.9) * duck[:, None] + np.stack([vo, vo], axis=1) * 0.55
    mix = filt(mix, "highpass", 30)
    mix *= (np.minimum(1.0, t / 0.02) * np.clip((total - t) / 0.5, 0, 1))[:, None]
    return mix


def write_wav(path, data):
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


COMP = "acompressor=threshold=-18dB:ratio=2.5:attack=6:release=180:makeup=2"


def loudnorm(src, dst, target=-14.0, tp=-1.5):
    probe = subprocess.run(["ffmpeg", "-hide_banner", "-i", src, "-af",
                            f"{COMP},loudnorm=I={target}:TP={tp}:LRA=9:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    m = json.loads(probe[probe.rindex("{"):probe.rindex("}") + 1])
    af = (f"{COMP},loudnorm=I={target}:TP={tp}:LRA=9:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-af", af, "-ar", str(SR), dst], check=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="?")
    ap.add_argument("wav", nargs="?")
    ap.add_argument("--measure", action="store_true", help="measure the voiceover clips into vo_timing.json")
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.measure:
        src = args.events or os.path.join(HERE, "events.json")
        with open(src) as fh:
            measure(json.load(fh)["vo_text"])
        return
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
