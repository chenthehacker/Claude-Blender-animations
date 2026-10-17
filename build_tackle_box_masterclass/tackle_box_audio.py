"""Soundtrack for the tackle box masterclass: voiceover, music bed and synthesised effects.

The voiceover clips are generated text to speech (voiceover/vo1..vo5, mp3 or wav; see SOURCES.md for the
script). Everything else is synthesised here: plastic snaps as each lure lands in its cell, the trays
swinging out, whooshes for the junk and the wipes, overlay pops, the spinnerbait's blade whir and its
vibration throb, the zip of the downsizing, a chime under the call to action, and a music bed that ducks
under the voice.

    python tackle_box_audio.py --measure                     # writes vo_timing.json from the clips
    python tackle_box_masterclass.py --events events.json    # (re-run after measuring: captions follow)
    python tackle_box_audio.py events.json soundtrack.wav [--video preview.mp4 --out preview_audio.mp4]

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


def snap():
    """A plastic snap: a sharp click, a short hollow resonance and a soft thump."""
    click = noise_burst(0.03, 2500, 12000, 0.004)
    body = tone(RNG.uniform(1900, 2500), 0.08, 0.012, partials=((1.0, 1.0), (2.7, 0.3)))
    thump = chirp(180, 70, 0.08, 0.025)
    n = max(len(click), len(body), len(thump))
    out = np.zeros(n)
    out[:len(click)] += click
    out[:len(body)] += body * 0.5
    out[:len(thump)] += thump * 0.6
    return out


def tray_swing(dur=1.2):
    """Cantilever tray: a smooth plastic slide that ends in a clack."""
    slide = sweep(dur, 300, 1800, rising=True, q=2.0) * 0.6
    clack = np.zeros_like(slide)
    c = snap() * 1.4
    i = len(slide) - len(c) - int(0.04 * SR)
    clack[i:i + len(c)] += c
    return slide + clack


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
# Music: A minor groove (Am - F - C - G) at 104 BPM
# --------------------------------------------------------------------------
def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


CHORDS = [[57, 60, 64, 69], [53, 57, 60, 65], [48, 52, 55, 60], [55, 59, 62, 67]]
BASS = [33, 29, 36, 31]


def music(n, energy):
    bpm = 104.0
    beat = 60.0 / bpm
    bar = 4 * beat
    total = n / SR
    nbars = int(total / bar) + 1
    out = np.zeros((n, 2))
    pad = np.zeros(n)
    for b in range(nbars):
        tt = np.arange(int((bar + 1.0) * SR)) / SR
        env = np.minimum(1.0, tt / 0.4) * np.clip((bar + 1.0 - tt) / 1.0, 0, 1)
        note = np.zeros(len(tt))
        for m in CHORDS[b % 4]:
            for det in (-0.08, 0.0, 0.07):
                f = midi(m + det * 0.1)
                note += sum(np.sin(2 * np.pi * f * h * tt + RNG.uniform(0, 6.28)) / h ** 1.7 for h in (1, 2, 3))
        i = int(b * bar * SR)
        seg = (note * env)[:max(0, n - i)]
        pad[i:i + len(seg)] += seg
    pad = filt(pad, "lowpass", 2200) * 0.006
    out[:, 0] += pad
    out[:, 1] += np.roll(pad, int(0.013 * SR))
    for b in range(nbars):
        chord = CHORDS[b % 4]
        for k in range(8):                                   # plucked arpeggio
            t0 = b * bar + k * beat / 2
            if t0 >= total:
                break
            f = midi(chord[[0, 2, 3, 1, 2, 3, 1, 3][k]] + 12)
            place(out, tone(f, 0.5, 0.14, partials=((1.0, 1.0), (2.0, 0.3), (3.0, 0.1))), t0,
                  0.03 * (0.6 + 0.6 * energy(t0)), pan=0.3 * np.sin(k * 1.1))
        for k in (0, 1.5, 2, 2.75, 3.5):
            t0 = b * bar + k * beat
            if t0 >= total:
                continue
            e = energy(t0)
            if k in (0, 2):
                place(out, tone(midi(BASS[b % 4]), 0.6, 0.25, partials=((1.0, 1.0), (2.0, 0.25))), t0, 0.08)
                place(out, chirp(120, 44, 0.24, 0.07), t0, 0.11 * (0.6 + 0.6 * e))           # kick
            if k == 1.5 or k == 3.5:
                place(out, filt(noise_burst(0.18, 1500, 8000, 0.05), "highpass", 1200), t0, 0.04 * e)   # snare
            if k == 2.75 and e > 0.6:
                place(out, chirp(120, 44, 0.2, 0.06), t0, 0.07 * e)
        for k in range(8):
            t0 = b * bar + (k + 0.5) * beat / 2
            if t0 < total:
                place(out, filt(noise_burst(0.04, 7000, 14000, 0.012), "highpass", 6000), t0, 0.018 * (0.5 + e), pan=0.35)
    t = np.arange(n) / SR
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
    # the hook: junk whooshes out, trays swing, every keeper snaps into its cell
    for i, fr in enumerate(ev["junk"]):
        place(fx, sweep(0.35, 500, 5000, rising=True), ft(fr), 0.07, pan=0.5 if i % 2 == 0 else -0.5)
    for a, _b in ev["trays"]:
        place(fx, tray_swing(1.25), ft(a), 0.14)
    for i, fr in enumerate(ev["snaps"]):
        place(fx, snap(), ft(fr), 0.20, pan=0.4 * np.sin(i * 1.7))
    # wipes
    for a, _b in ev["wipes"]:
        place(fx, sweep(0.42, 300, 4500, rising=True), ft(a - 7), 0.10)
    # overlay pops
    for frame, kind, _label in ev["overlays"]:
        t0 = ft(frame)
        if kind == "badge":
            place(fx, tone(880.0, 0.35, 0.09, partials=((1.0, 1.0), (2.0, 0.3))), t0, 0.05)
            place(fx, tone(1318.5, 0.35, 0.10, partials=((1.0, 1.0), (2.0, 0.2))), t0 + 0.06, 0.04)
        elif kind == "pin":
            place(fx, tone(1567.98, 0.2, 0.05, partials=((1.0, 1.0), (2.76, 0.15))), t0, 0.035, pan=0.2)
        elif kind == "caption":
            place(fx, filt(noise_burst(0.2, 2000, 9000, 0.05), "highpass", 1500), t0, 0.015, pan=-0.15)
        elif kind == "headline":
            for j, f in enumerate((659.25, 783.99, 987.77)):
                place(fx, tone(f, 1.6, 0.5, partials=((1.0, 1.0), (2.76, 0.1))), t0 + 0.06 * j, 0.025)
        elif kind == "cta":
            for j, f in enumerate((880.0, 1108.73, 1318.51, 1760.0)):
                place(fx, tone(f, 2.0, 0.6, partials=((1.0, 1.0), (2.0, 0.3))), t0 + 0.05 * j, 0.03)
    # the spinnerbait: blade whir throughout set 3, a low throb while the vibration rings pulse
    a, b = ft(ev["spin"][0]), ft(ev["spin"][1])
    i0, i1 = int(a * SR), int(b * SR)
    tt = np.arange(i1 - i0) / SR
    whir = filt(RNG.standard_normal(len(tt)), "bandpass", [600, 3000]) * (0.55 + 0.45 * np.sin(2 * np.pi * 18 * tt) ** 2)
    env = np.clip(tt / 0.3, 0, 1) * np.clip((tt[-1] - tt) / 0.3, 0, 1)
    fx[i0:i1, 0] += whir * env * 0.012
    fx[i0:i1, 1] += whir * env * 0.010
    r0, r1 = ft(ev["rings"][0]), ft(ev["rings"][1])
    j0, j1 = int(r0 * SR), int(r1 * SR)
    tt = np.arange(j1 - j0) / SR
    throb = np.sin(2 * np.pi * 95 * tt) * (0.5 + 0.5 * np.sin(2 * np.pi * (24 / 14) * tt)) ** 2
    env = np.clip(tt / 0.25, 0, 1) * np.clip((tt[-1] - tt) / 0.25, 0, 1)
    fx[j0:j1] += (throb * env * 0.06)[:, None]
    # the downsizing zip
    place(fx, chirp(1400, 380, 0.45, 0.2), ft(ev["shrink"]), 0.06)
    place(fx, snap(), ft(ev["shrink"] + 12), 0.15)

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
        return float(1.0 if f < ev["cuts"][0] else 0.75 if f < ev["cuts"][3] else 0.9)
    mus = music(n, energy)
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
