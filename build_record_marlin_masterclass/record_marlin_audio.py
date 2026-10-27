"""Soundtrack for the record marlin masterclass: voiceover, music bed and synthesised effects, built to loop.

The voiceover clips are generated text to speech (voiceover/vo1..vo5; ElevenLabs, voice "Mark, Natural
Conversations"; see SOURCES.md for the script). Everything else is synthesised here: trade wind and chop on the
surface, the hush under water, a diesel rumble while the boat trolls, the fizz of the lures' smoke trails, a heavy
thump and a burst of white water at the strike, the reel screaming as line pours off, a click as the lever goes up
to strike, the splashes of the tail walk, whooshes for the whip pans, overlay pops, and a driving music bed in
E minor that ducks under the voice.

The film loops, so the soundtrack does too: the music is 24 bars that fit the film exactly, nothing fades in or out
at the ends, and anything still sounding past the end (a reverb tail, a ringing note) is wrapped onto the start.

    python record_marlin_audio.py --measure                         # writes vo_timing.json from the clips
    python record_marlin_masterclass.py --events events.json        # (re-run after measuring: captions follow)
    python record_marlin_audio.py events.json soundtrack.wav [--video preview.mp4 --out preview_audio.mp4]

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




def wind(n, gain):
    w = filt(RNG.standard_normal(n), "bandpass", [120, 1800]) * gain
    gust = 0.6 + 0.4 * np.sin(2 * np.pi * np.arange(n) / SR / 7.3) ** 2
    return w * gust


def swell_wash(n, period=6.1):
    """Waves breaking and hissing on a slow cycle."""
    t = np.arange(n) / SR
    hiss = filt(RNG.standard_normal(n), "bandpass", [400, 6000])
    env = np.clip(np.sin(2 * np.pi * t / period), 0, 1) ** 3
    return hiss * env * 0.05 + filt(RNG.standard_normal(n), "lowpass", 220) * 0.04


def diesel(n, rpm_hz=24.0):
    t = np.arange(n) / SR
    base = sum(np.sin(2 * np.pi * rpm_hz * h * t + h) / h for h in (1, 2, 3, 4, 6))
    return filt(base * (1 + 0.15 * np.sin(2 * np.pi * 1.7 * t)), "lowpass", 260) * 0.05


def drag_scream(dur, rate0=640.0, rate1=520.0):
    """Lever drag paying out line: a fast ratchet-like buzz on a slowly falling rate."""
    t = np.arange(int(dur * SR)) / SR
    rate = rate0 + (rate1 - rate0) * t / dur
    clicks = (np.sin(2 * np.pi * np.cumsum(rate) / SR) > 0.85).astype(float)
    return filt(clicks, "bandpass", [1800, 7000]) * np.minimum(1.0, t / 0.05) * np.clip((dur - t) / 0.2, 0, 1)


def crash(dur, big=1.0):
    t = np.arange(int(dur * SR)) / SR
    x = filt(RNG.standard_normal(len(t)), "bandpass", [150, 5000]) * np.exp(-t / (0.25 * big))
    return x + filt(RNG.standard_normal(len(t)), "lowpass", 180) * np.exp(-t / 0.15) * 0.8


def whoosh(dur=0.45, rising=True):
    return sweep(dur, 250, 6000, rising=rising, q=1.2)


# --------------------------------------------------------------------------
# Music: a driving E minor loop (Em, C, G, D) of 24 bars that fills the film exactly
# --------------------------------------------------------------------------
def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


CHORDS = [[52, 55, 59, 64], [48, 52, 55, 60], [55, 59, 62, 67], [50, 54, 57, 62]]
BASS = [28, 24, 31, 26]


def music(n_total, loop_n, energy):
    bars = 24
    bar = loop_n / SR / bars
    beat = bar / 4
    out = np.zeros((n_total, 2))
    pad = np.zeros(n_total)
    for b in range(bars):
        tt = np.arange(int((bar + 1.2) * SR)) / SR
        env = np.minimum(1.0, tt / 0.5) * np.clip((bar + 1.2 - tt) / 1.2, 0, 1)
        note = np.zeros(len(tt))
        for m in CHORDS[b % 4]:
            for det in (-0.08, 0.0, 0.07):
                f = midi(m + det * 0.1)
                note += sum(np.sin(2 * np.pi * f * h * tt + RNG.uniform(0, 6.28)) / h ** 1.5 for h in (1, 2, 3, 4))
        i = int(b * bar * SR)
        seg = (note * env)[:max(0, n_total - i)]
        pad[i:i + len(seg)] += seg
    pad = filt(pad, "lowpass", 1900) * 0.005
    out[:, 0] += pad
    out[:, 1] += np.roll(pad, int(0.017 * SR))
    t = np.arange(n_total) / SR
    drone = np.sin(2 * np.pi * midi(28) * t) * 0.026 + np.sin(2 * np.pi * midi(40) * t + 1.0) * 0.01
    out += drone[:, None]
    for b in range(bars):
        chord = CHORDS[b % 4]
        for k in range(8):                                   # pulsing eighth note ostinato
            t0 = b * bar + k * beat / 2
            e = energy(t0)
            f = midi(chord[[0, 2, 1, 3, 0, 2, 3, 1][k]] + 12)
            place(out, tone(f, 0.5, 0.16, partials=((1.0, 1.0), (2.0, 0.35), (3.0, 0.12))), t0, 0.018 * (0.4 + 0.8 * e),
                  pan=0.3 * np.sin(k * 1.3))
        for k in range(4):
            t0 = b * bar + k * beat
            e = energy(t0)
            place(out, tone(midi(BASS[b % 4]), 0.6, 0.3, partials=((1.0, 1.0), (2.0, 0.35))), t0, 0.07 * (0.5 + 0.5 * e))
            if k in (0, 2):
                place(out, chirp(115, 40, 0.35, 0.1), t0, 0.15 * e)                         # kick
            if k in (1, 3) and e > 0.45:
                place(out, filt(noise_burst(0.25, 900, 7000, 0.07), "highpass", 700), t0, 0.04 * e)     # snare
            for h in range(2):
                th = t0 + h * beat / 2 + beat / 4
                if e > 0.6:
                    place(out, filt(noise_burst(0.04, 7000, 14000, 0.012), "highpass", 6000), th, 0.012 * e, pan=0.35)
    return out


# --------------------------------------------------------------------------
# Mix
# --------------------------------------------------------------------------
def fizz(n):
    """The lures' smoke: a crackling bubble fizz."""
    x = filt(RNG.standard_normal(n), "bandpass", [1500, 9000])
    pops = (RNG.random(n) < 0.0016).astype(float)
    pops = filt(pops, "bandpass", [600, 4000]) * 6.0
    return x * 0.02 + pops * 0.03


def build(ev):
    fps = ev["fps"]
    total = ev["frames"] / fps
    n = int(round(total * SR))
    pad_n = int(4.0 * SR)
    N = n + pad_n
    t = np.arange(N) / SR

    def ft(frame):
        return (frame - 1) / fps
    wi, c1, c2, c3, c4 = ev["cuts"]
    c1a, c2b, c3b = ev["hard_cuts"]
    fx = np.zeros((N, 2))
    under = np.zeros(N)
    for a_, b_ in ev["under"]:
        under[int(ft(a_) * SR):int(ft(b_) * SR)] = 1.0
    above = 1.0 - under
    sm = np.ones(800) / 800
    above_s, under_s = np.convolve(above, sm, mode="same"), np.convolve(under, sm, mode="same")
    w = wind(N, 0.03) + swell_wash(N, period=4.3)
    fx[:, 0] += w * above_s
    fx[:, 1] += np.roll(w, 700) * above_s
    hush = filt(RNG.standard_normal(N), "lowpass", 300) * 0.06 * (0.75 + 0.25 * np.sin(2 * np.pi * 0.09 * t))
    fx[:, 0] += hush * under_s
    fx[:, 1] += np.roll(hush, 900) * under_s
    # the boat trolling: the spread from the air, the reel shot and the loop's overhead (quieter from above)
    for (a_, b_), g in (((0, ft(wi)), 0.6), ((ft(c2), ft(c2b)), 0.8), ((ft(c3), ft(c3b)), 1.4), ((ft(c4), total + 4.0), 0.6)):
        ia, ib = int(a_ * SR), int(min(N, b_ * SR))
        d = diesel(ib - ia, 25.0) * g
        fx[ia:ib, 0] += d
        fx[ia:ib, 1] += d
    # the lures' smoke under water
    for a_, b_ in ((ft(wi), ft(ev["hook_strike"])), (ft(c2b), ft(c3))):
        ia, ib = int(a_ * SR), int(b_ * SR)
        z = fizz(ib - ia)
        fx[ia:ib, 0] += z
        fx[ia:ib, 1] += np.roll(z, 300)
    # the strike: a deep thump and the white water burst
    s = ft(ev["hook_strike"])
    place(fx, chirp(80, 30, 0.7, 0.2), s - 0.02, 0.22)
    place(fx, filt(noise_burst(1.2, 200, 6000, 0.35), "lowpass", 5000), s, 0.12)
    # the reel screaming, the lever click, the scream easing as the drag comes up
    ia, ib = int(ft(c3) * SR), int(ft(c3b) * SR)
    scream = drag_scream((ib - ia) / SR, 900.0, 700.0)
    lk = ft(ev["lock"]) - ft(c3)
    env = 1.0 - 0.35 * np.clip((np.arange(ib - ia) / SR - lk) / 1.2, 0, 1)
    fx[ia:ib, 0] += scream * 0.07 * env
    fx[ia:ib, 1] += scream * 0.06 * env
    place(fx, filt(noise_burst(0.05, 2000, 9000, 0.01), "highpass", 1500), ft(ev["lock"]), 0.1)
    place(fx, tone(1800, 0.08, 0.02), ft(ev["lock"]) + 0.01, 0.04)
    # the tail walk: crashing splashes on each tail beat, the scream continuing underneath
    tw0, tw1 = ev["tail_walk"]
    tc = ft(tw0) - 0.05
    place(fx, crash(1.4, 1.6), tc, 0.16)
    while tc < ft(tw1) - 0.2:
        tc += 1.0 / 3.2
        place(fx, crash(0.6, 0.8), tc + RNG.uniform(-0.03, 0.03), 0.07, pan=RNG.uniform(-0.3, 0.3))
    ia, ib = int(ft(tw0) * SR), int(ft(tw1) * SR)
    sc2 = drag_scream((ib - ia) / SR, 760.0, 620.0)
    fx[ia:ib, 0] += sc2 * 0.03
    fx[ia:ib, 1] += sc2 * 0.03
    # whip pans and cuts
    for a_, b_ in ev["whips"]:
        place(fx, whoosh(max(0.3, (b_ - a_) / fps + 0.1), a_ < 10), ft(a_), 0.12)
    place(fx, splash_in(), ft(wi) - 0.02, 0.1)
    place(fx, splash_in(), ft(c2b) - 0.02, 0.08)
    for c in (c1, c2, c3):
        place(fx, filt(noise_burst(0.18, 80, 2500, 0.06), "lowpass", 3000), ft(c) - 0.02, 0.07)
        place(fx, chirp(90, 35, 0.4, 0.12), ft(c) - 0.01, 0.1)
    for c in (c1a, c2b, c3b) + tuple(ev.get("extra_cuts", [])):
        place(fx, filt(noise_burst(0.12, 300, 4000, 0.04), "lowpass", 4000), ft(c) - 0.02, 0.04)
    for frame, kind, _label in ev["overlays"]:
        t0 = ft(frame)
        if t0 < 0:
            continue
        if kind == "badge":
            place(fx, tone(659.26, 0.4, 0.1, partials=((1.0, 1.0), (2.0, 0.3))), t0, 0.05)
            place(fx, tone(987.77, 0.4, 0.11, partials=((1.0, 1.0), (2.0, 0.2))), t0 + 0.06, 0.04)
        elif kind == "pin":
            place(fx, tone(1318.5, 0.2, 0.05, partials=((1.0, 1.0), (2.76, 0.15))), t0, 0.03, pan=0.2)
        elif kind == "caption":
            place(fx, filt(noise_burst(0.2, 2000, 9000, 0.05), "highpass", 1500), t0, 0.014, pan=-0.15)
    vo = np.zeros(N)
    for k, f0 in ev["vo_start"].items():
        p = vo_path(int(k))
        if p is None:
            print(f"vo{k}: missing, mixed without it")
            continue
        x = decode(p)
        x = x / max(1e-6, np.abs(x).max()) * 0.9
        i = int(round(ft(f0) * SR))
        seg = x[:max(0, N - i)]
        vo[i:i + len(seg)] += seg
    vo = filt(vo, "highpass", 80)
    env = np.sqrt(np.convolve(vo * vo, np.ones(int(0.05 * SR)) / int(0.05 * SR), mode="same"))
    env = np.clip(env / 0.08, 0, 1)
    env = signal.lfilter([1 - 0.9995], [1, -0.9995], env)
    duck = 1.0 - 0.55 * np.clip(env * 1.6, 0, 1)
    strike = ev["hook_strike"]

    def energy(tsec):
        tsec = tsec % total
        f = tsec * fps + 1
        if f < strike:
            return 0.55
        if f < c1:
            return 0.85
        if f < c2:
            return 0.6
        if f < c3:
            return 0.75
        return 1.0 if f < c4 else 0.55
    mus = music(N, n, energy)
    mix = (mus * 1.0 + fx * 0.9) * duck[:, None] + np.stack([vo, vo], axis=1) * 0.55
    mix = filt(mix, "highpass", 30)
    loop = mix[:n].copy()
    loop[:pad_n] += mix[n:n + pad_n]
    return loop


def splash_in():
    t = np.arange(int(0.6 * SR)) / SR
    return filt(RNG.standard_normal(len(t)), "bandpass", [200, 4000]) * np.exp(-t / 0.12) + \
        filt(RNG.standard_normal(len(t)), "lowpass", 300) * np.exp(-t / 0.3) * 0.6


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
    return [p.strip() for p in re.split(r"(?<=[.?:;])\s+", text) if p.strip()]


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
        if os.path.exists(src):
            with open(src) as fh:
                measure(json.load(fh)["vo_text"])
        else:   # no events yet: read the voiceover text straight from the scene script
            import ast
            tree = ast.parse(open(os.path.join(HERE, "record_marlin_masterclass.py")).read())
            node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "VO_TEXT")
            measure({str(k): v for k, v in ast.literal_eval(node.value).items()})
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
