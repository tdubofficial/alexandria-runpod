#!/usr/bin/env python3
"""Alexandria batch renderer: instrumental loops on the engine -> seamless loops -> 5.1 -> Dolby Digital + DTS -> zip per genre.

Runs inside the pod (start.sh launches it when ALEX_BATCH names a file in batches/). Resumable: finished tracks are skipped.
Per track:
  1. Render with ACE-Step 1.5 (instrumental, requested BPM/key/meter, LM planning on).
  2. Measure the real tempo, pick a strong downbeat after an 8-bar lead-in, find the loop end that best matches the start,
     equal-power crossfade the continuation into the head (so the end flows into the start), then Fourier-resample the
     periodic loop to a whole number of codec frames (a change of ~0.01%, inaudible).
  3. Set loudness (static gain, so the loop stays seamless), upmix to 5.1 on three copies and keep the middle one
     (so the upmix itself is periodic), start the loop ~15 ms before the downbeat (codec start padding lands under the hit).
  4. Encode Dolby Digital (AC-3) and DTS, decode them back and measure the loop seam.
"""
import json, math, os, re, shutil, subprocess, sys, time, urllib.parse, urllib.request, zipfile
from pathlib import Path

import numpy as np

SR = 48000
FRAME = 1536                      # AC-3 frame (DTS frames are 512, which divides it)
APP = Path(os.environ.get("ALEX_APP", "/workspace/alexandria"))
ACE = os.environ.get("ACE_URL", "http://127.0.0.1:18011").rstrip("/")
UPMIX = "surround=chl_out=5.1,aformat=channel_layouts=5.1(side)"


def log(*a):
    print("BATCH", time.strftime("%H:%M:%S"), *a, flush=True)


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def http_json(url, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ff(args, inp=None):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args, input=inp, capture_output=True)
    if r.returncode:
        raise RuntimeError("ffmpeg: " + r.stderr.decode(errors="replace")[-300:])
    return r.stdout


def decode(path, ch=2):
    raw = ff(["-i", str(path), "-ar", str(SR), "-ac", str(ch), "-f", "f32le", "-"])
    return np.frombuffer(raw, dtype="<f4").reshape(-1, ch).astype(np.float32)


# ---------------------------------------------------------------- render
def render(t, attempt, out_dir):
    beats = t["time_signature"]
    bar = 60.0 / t["bpm"] * beats
    dur = min(600.0, (8 + t["loop_bars"] + 4) * bar + 25)
    body = {"prompt": t["prompt"], "lyrics": "[Instrumental]", "audio_duration": round(dur, 1), "audio_format": "flac",
            "batch_size": 1, "thinking": True, "vocal_language": "en", "task_type": "text2music",
            "time_signature": str(beats), "bpm": int(t["bpm"]), "key_scale": t["key"],
            "seed": 1000 * t["n"] + attempt, "use_random_seed": False}
    res = http_json(ACE + "/release_task", body)
    tid = ((res or {}).get("data") or {}).get("task_id")
    if not tid:
        raise RuntimeError(f"no task id: {str(res)[:200]}")
    t0 = time.time()
    while time.time() - t0 < 45 * 60:
        time.sleep(4)
        items = (http_json(ACE + "/query_result", {"task_id_list": [tid]}) or {}).get("data") or []
        it = next((i for i in items if i.get("task_id") == tid), None)
        if not it or it.get("status") == 0:
            continue
        result = it.get("result")
        result = json.loads(result) if isinstance(result, str) else result
        if it.get("status") == 2:
            raise RuntimeError(f"engine failure: {str(result)[:200]}")
        entries = result if isinstance(result, list) else [result]
        f = next((e.get("file") for e in entries if isinstance(e, dict) and e.get("file")), None)
        if not f:
            raise RuntimeError("engine returned no file")
        url = f if f.startswith("http") else ACE + f
        ext = os.path.splitext(urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("path", ["x.flac"])[0])[1] or ".flac"
        dst = out_dir / f"render_{attempt}{ext}"
        with urllib.request.urlopen(url, timeout=300) as r:
            dst.write_bytes(r.read())
        return dst, time.time() - t0, dur
    raise RuntimeError("render timed out")


# ---------------------------------------------------------------- analysis
def onset_env(mono, hop=512, n=2048, fmax=None):
    win = np.hanning(n).astype(np.float32)
    frames = (len(mono) - n) // hop
    out = np.zeros(frames, np.float32)
    kmax = n // 2 + 1 if fmax is None else int(fmax * n / SR) + 1
    prev = None
    for s in range(0, frames, 2048):                          # chunks keep memory small
        idx = np.arange(s, min(frames, s + 2048))
        seg = np.lib.stride_tricks.sliding_window_view(mono, n)[idx * hop] * win
        mag = np.log1p(100 * np.abs(np.fft.rfft(seg, axis=1))[:, 1:kmax])
        if prev is None:
            prev = mag[:1]
        d = np.diff(np.vstack([prev, mag]), axis=0)
        out[idx] = np.maximum(d, 0).sum(1)
        prev = mag[-1:]
    k = int(0.4 * SR / hop)                                    # remove slow trend
    trend = np.convolve(out, np.ones(k) / k, mode="same")
    return np.maximum(out - trend, 0)


def refine_tempo(env, bpm, fps):
    def ac(lag):
        L = int(lag); f = lag - L
        a = env[:-L - 2]; b = env[L:-2] * (1 - f) + env[L + 1:-1] * f
        return float(np.dot(a, b[:len(a)]))
    def comb(lag):
        return sum(ac(lag * k) for k in range(1, 9) if lag * k + 2 < len(env) - 8)
    p0 = fps * 60 / bpm
    lags = np.arange(p0 * 0.92, p0 * 1.08, 0.02)
    best = max(lags, key=comb)
    return 60 * fps / best


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = math.sqrt(float(np.dot(a, a) * np.dot(b, b))) or 1.0
    return float(np.dot(a, b)) / d


def fourier_resample(x, n_out):
    """Exact for periodic signals: keeps the loop seamless while changing its length slightly."""
    X = np.fft.rfft(x, axis=0)
    n_in = x.shape[0]
    Y = np.zeros((n_out // 2 + 1, x.shape[1]), dtype=X.dtype)
    m = min(X.shape[0], Y.shape[0])
    Y[:m] = X[:m]
    return (np.fft.irfft(Y, n=n_out, axis=0) * (n_out / n_in)).astype(np.float32)


def seam_jump(a):
    two = np.concatenate([a[-4096:], a[:4096]])
    d = np.abs(np.diff(two, axis=0)).sum(1)
    typ = np.percentile(np.abs(np.diff(a, axis=0)).sum(1), 99.9) or 1e-9
    return float(d[4090:4102].max() / typ)


def chroma_frames(mono, n=4096, hop=2048):
    """12-bin pitch-class energy per frame (80 Hz - 2 kHz spectral peaks)."""
    win = np.hanning(n).astype(np.float32)
    frames = (len(mono) - n) // hop
    k = np.arange(1, n // 2 + 1); f = k * SR / n
    sel = (f >= 80) & (f <= 2000)
    pc = (np.round(12 * np.log2(f[sel] / 440)).astype(int) + 69) % 12
    out = np.zeros((frames, 12), np.float32)
    for s in range(0, frames, 1024):
        idx = np.arange(s, min(frames, s + 1024))
        seg = np.lib.stride_tricks.sliding_window_view(mono, n)[idx * hop] * win
        mag = np.abs(np.fft.rfft(seg, axis=1))[:, 1:][:, sel]
        for c in range(12):
            out[idx, c] = mag[:, pc == c].sum(1)
    return out, SR / hop


def cos(a, b):
    a = a.ravel(); b = b.ravel()
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))


def make_loop(y, t):
    """Return (periodic stereo loop with a downbeat ~15 ms in, info dict).

    Searches start bars 8-15 and loop lengths of loop_bars +-12 bars, scoring how well the music right before and
    right after the end matches the music right before and after the start (harmony from chroma, rhythm from onsets).
    """
    hop = 512; fps = SR / hop
    mono = y.mean(1)
    env_all = onset_env(mono, hop)
    env_low = onset_env(mono, hop, fmax=250)
    chroma, cfps = chroma_frames(mono)
    bpm = refine_tempo(env_all + env_low, t["bpm"], fps)
    beat_s = 60 / bpm; bar_s = beat_s * t["time_signature"]
    lo, hi = int(8 * bar_s * fps), int(9 * bar_s * fps)
    g0 = (lo + int(np.argmax(env_low[lo:hi] + 0.5 * env_all[lo:hi]))) / fps        # a strong downbeat (seconds)
    total_s = len(mono) / SR
    W2 = 2 * bar_s

    def chroma_win(t0, t1):
        a, b = int(t0 * cfps), int(t1 * cfps)
        return chroma[max(0, a):max(a + 1, b)].mean(0)

    sm = np.hanning(7) / np.hanning(7).sum()                  # ~60 ms smoothing so near-misses still count
    lo_s, all_s = np.convolve(env_low, sm, "same"), np.convolve(env_all, sm, "same")

    def onset_win(t0, shift=0):
        a = int(t0 * fps) + shift; n = int(W2 * fps)
        return np.concatenate([lo_s[a:a + n], all_s[a:a + n]])

    q = int(beat_s * fps * 0.25)

    best = None
    for kb in range(0, 8):
        st = g0 + kb * bar_s
        ca, cb = chroma_win(st, st + W2), chroma_win(st - W2, st)
        oa = onset_win(st)
        for nb in range(max(16, t["loop_bars"] - 12), t["loop_bars"] + 13):
            en = st + nb * bar_s
            if en + W2 + 2 > total_s - 10 or nb * bar_s < 90:
                continue
            h_after = cos(ca, chroma_win(en, en + W2)); h_before = cos(cb, chroma_win(en - W2, en))
            rh = max(ncc(oa, onset_win(en, d)) for d in range(-q, q + 1, 2))
            score = 0.4 * h_after + 0.3 * h_before + 0.3 * rh - 0.002 * abs(nb - t["loop_bars"]) + (0.01 if nb % 4 == 0 else 0)
            if best is None or score > best[0]:
                best = (score, st, nb, h_after, h_before, rh)
    if best is None:
        raise RuntimeError("render too short for the loop")
    score, st, nb, h_after, h_before, rh = best
    # refine the end within half a beat on onsets, then to the sample on the waveform (+-12 ms)
    s0 = int(st * fps); L0 = nb * bar_s * fps; W = int(W2 * fps)
    ref = np.concatenate([env_low[s0:s0 + W], env_all[s0:s0 + W]])
    cands = range(int(s0 + L0 - beat_s * fps * 0.5), int(s0 + L0 + beat_s * fps * 0.5) + 1)
    rh_final, e0 = max((ncc(ref, np.concatenate([env_low[e:e + W], env_all[e:e + W]])), e) for e in cands)
    a0, b0 = s0 * hop, e0 * hop
    w = int(0.6 * SR); r = int(0.012 * SR)
    seg = mono[a0:a0 + w]
    k = max(range(-r, r + 1, 4), key=lambda k: ncc(seg, mono[b0 + k:b0 + k + w]))
    k = max(range(k - 4, k + 5), key=lambda k: ncc(seg, mono[b0 + k:b0 + k + w]))
    L = (b0 + k) - a0
    pre = int(0.015 * SR)                                      # codec start padding lands just before the start point
    # Any onset within the first bar can start the loop (the length L stays the same, so the seam match holds).
    # Pick the one with the quietest 15 ms before it relative to what follows: that is where decoders' start-up
    # padding is least audible.
    def mask_ratio(p):
        b = np.sqrt(np.mean(mono[p - pre:p] ** 2)) + 1e-9
        f = np.sqrt(np.mean(mono[p:p + int(0.08 * SR)] ** 2)) + 1e-9
        return b / f
    span = int(bar_s * SR); cands_p = [a0]
    e_ = env_all[s0:s0 + int(bar_s * fps)]
    if len(e_) > 3:
        thr = e_.mean() + e_.std()
        cands_p += [int((s0 + i) * hop) for i in range(1, len(e_) - 1) if e_[i] > thr and e_[i] >= e_[i - 1] and e_[i] > e_[i + 1]]
    refined = []
    for p in cands_p:                                          # snap each candidate to its sharpest rise (+-6 ms)
        win = mono[p - 288:p + 288]
        if len(win) < 576:
            continue
        rise = np.abs(np.diff(win)); q = p - 288 + int(np.argmax(np.convolve(rise, np.ones(48), "same")))
        if q + L + int(bar_s * SR) < len(mono) - 10 * SR:
            refined.append(q)
    a1 = min(refined or [a0], key=mask_ratio)
    masking = mask_ratio(a1)
    a = a1 - pre
    X = int(min(bar_s, 2.5) * SR)                              # one-bar equal-power crossfade, continuation -> head
    fin = np.sin(np.linspace(0, np.pi / 2, X, dtype=np.float32))[:, None]
    fout = np.cos(np.linspace(0, np.pi / 2, X, dtype=np.float32))[:, None]
    loop = y[a:a + L].copy()
    loop[:X] = y[a:a + X] * fin + y[a + L:a + L + X] * fout
    n_out = int(round(L / FRAME)) * FRAME
    loop = fourier_resample(loop, n_out)
    # QA: harmony across the seam of the finished loop vs ordinary 2-bar boundaries inside it
    lc, _ = chroma_frames(loop.mean(1))
    Wc = max(2, int(W2 * cfps))
    seam_h = cos(lc[-Wc:].mean(0), lc[:Wc].mean(0))
    inner = [cos(lc[i - Wc:i].mean(0), lc[i:i + Wc].mean(0)) for i in range(Wc, len(lc) - Wc, Wc)]
    info = {"measured_bpm": round(bpm, 2), "loop_start_s": round(a / SR, 3), "loop_seconds": round(n_out / SR, 3), "loop_bars_used": nb,
            "loop_samples": n_out, "match_score": round(score, 3), "harmony_after": round(h_after, 3), "harmony_before": round(h_before, 3),
            "rhythm_match": round(rh_final, 3), "rhythm_score": round(rh, 3), "seam_harmony": round(seam_h, 3), "inner_harmony_median": round(float(np.median(inner)), 3),
            "inner_harmony_p10": round(float(np.percentile(inner, 10)), 3), "length_change_pct": round(100 * (n_out - L) / L, 4),
            "start_mask_ratio": round(float(masking), 3), "cut_version": 4}
    return loop, info


# ---------------------------------------------------------------- loudness, 5.1, encode
def lufs(st):
    raw = np.tile(st, (2, 1)).astype("<f4").tobytes()
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                        "-af", "ebur128", "-f", "null", "-"], input=raw, capture_output=True)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr.decode(errors="replace"))
    return float(m[-1]) if m else -23.0


def upmix_periodic(st):
    n = st.shape[0]
    raw = ff(["-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-", "-af", UPMIX, "-f", "f32le", "-"],
             inp=np.tile(st, (3, 1)).astype("<f4").tobytes())
    up = np.frombuffer(raw, dtype="<f4").reshape(-1, 6)
    # the filter delays the signal; find the delay so the downbeat stays ~15 ms in
    ref = st[n // 2:n // 2 + SR].mean(1); dm = up[:, 0] + up[:, 1] + up[:, 2]
    seg = dm[n + n // 2:n + n // 2 + SR + 4096]
    xc = np.fft.irfft(np.fft.rfft(seg, 2 * len(seg)) * np.conj(np.fft.rfft(ref, 2 * len(seg))))
    d = int(np.argmax(xc[:4096]))
    return np.ascontiguousarray(up[n + d:2 * n + d]), d


CODEC_DELAY = {"ac3": 256, "dts": 512}          # samples; measured with impulses (ffmpeg ac3 / dca encoders at 48 kHz)


def encode(x51, path, codec, rate):
    """Encode one loop so the decoded file lines up sample-for-sample with the loop.

    Decoders output D samples of start-up padding, then the input. Feeding the loop rotated by D means decoded sample
    n (n >= D) equals loop sample n, so there is no time jump at the seam: only the first D samples (5.3 ms AC-3,
    10.7 ms DTS) are lost, and the loop starts ~15 ms before an onset, so that gap sits just ahead of a hit.
    Short fades into and out of the gap turn what would be a click into a soft breath.
    """
    D = CODEC_DELAY[codec]; L = x51.shape[0]; F = 144                  # 3 ms fades
    y = x51.copy()
    y[D:D + F] *= np.sin(np.linspace(0, np.pi / 2, F, dtype=np.float32))[:, None]
    y[L - F:] *= np.cos(np.linspace(0, np.pi / 2, F, dtype=np.float32))[:, None]
    y[:D] = 0.0                                                         # replaced by decoder padding anyway
    inp = np.roll(y, -D, axis=0)
    args = ["-f", "f32le", "-ar", str(SR), "-ac", "6", "-channel_layout", "5.1(side)", "-i", "-"]
    args += (["-c:a", "ac3", "-b:a", rate] if codec == "ac3" else ["-c:a", "dca", "-strict", "-2", "-b:a", rate])
    ff(args + [str(path)], inp=np.clip(inp, -1, 1).astype("<f4").tobytes())


def seam_error(decoded, ref, D):
    """How closely the decoded file matches the loop outside the start-up gap (1.0 = identical shape)."""
    a = decoded[D + 144:].ravel(); b = ref[D + 144:len(decoded)].ravel()
    return round(ncc(a, b), 4)


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,channels,channel_layout,bit_rate,sample_rate",
                        "-of", "json", str(path)], capture_output=True, text=True)
    s = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    return {k: s.get(k) for k in ("codec_name", "channels", "channel_layout", "bit_rate", "sample_rate")}


# ---------------------------------------------------------------- batch
def process(t, batch, out_root, work_root):
    gdir = out_root / t["genre"]; gdir.mkdir(parents=True, exist_ok=True)
    base = f"{t['n']:02d}_{slug(t['title'])}"
    rates = batch["formats"]
    f_ac3 = gdir / f"{base}_DolbyDigital5.1_{rates['ac3']}bps.ac3"
    f_dts = gdir / f"{base}_DTS5.1_{rates['dts']}bps.dts"
    meta_p = gdir / f"{base}.json"
    if f_ac3.exists() and f_dts.exists() and meta_p.exists():
        m = json.loads(meta_p.read_text())
        if m.get("cut_version", 1) >= 4:
            return m
    work = work_root / base; work.mkdir(parents=True, exist_ok=True)
    good = float(os.environ.get("ALEX_LOOP_GOOD", "0.85"))     # seam harmony this close to the track's own norm is accepted
    tried = []
    for attempt in range(3):
        try:
            existing = sorted(work.glob(f"render_{attempt}.*"))
            if existing:
                src, secs, dur = existing[0], 0, 0
                log(f"#{t['n']:02d} reusing render {attempt + 1}")
            else:
                src, secs, dur = render(t, attempt, work)
                log(f"#{t['n']:02d} rendered {dur:.0f}s of audio in {secs:.0f}s (attempt {attempt + 1})")
            y = decode(src)
            loop, info = make_loop(y, t)
            info["render_seconds"] = round(secs); info["attempt"] = attempt + 1
            ok = info["seam_harmony"] >= min(good, info["inner_harmony_p10"])
            tried.append((info["match_score"] + (1 if ok else 0), loop, info))
            log(f"#{t['n']:02d} attempt {attempt + 1}: seam harmony {info['seam_harmony']} (track p10 {info['inner_harmony_p10']}, "
                f"median {info['inner_harmony_median']}), rhythm {info['rhythm_match']}, {info['loop_bars_used']} bars")
            if ok:
                break
        except Exception as e:  # noqa: BLE001
            log(f"#{t['n']:02d} attempt {attempt + 1} failed: {str(e)[:200]}")
    if not tried:
        raise RuntimeError(f"{t['title']}: no usable render")
    _, loop, info = max(tried, key=lambda x: x[0])
    gain_db = batch.get("loudness_lufs", -16) - lufs(loop)
    peak = float(np.abs(loop).max()) or 1e-9
    gain_db = min(gain_db, 20 * math.log10(0.89 / peak))       # keep peaks under -1 dBFS
    loop *= 10 ** (gain_db / 20)
    x51, delay = upmix_periodic(loop)
    peak51 = float(np.abs(x51).max())
    if peak51 > 0.95:
        x51 *= 0.95 / peak51
    encode(x51, f_ac3, "ac3", rates["ac3"])
    encode(x51, f_dts, "dts", rates["dts"])
    checks = {}
    for name, f in (("ac3", f_ac3), ("dts", f_dts)):
        d = decode(f, 6)
        checks[name] = {**probe(f), "decoded_seconds": round(len(d) / SR, 3), "seam_jump": round(seam_jump(d), 2),
                        "gap_ms": round(CODEC_DELAY[name] / SR * 1000, 1), "match_to_master": seam_error(d, x51, CODEC_DELAY[name])}
    meta = {**{k: t[k] for k in ("n", "genre", "title", "kit", "bpm", "key", "time_signature", "loop_bars", "prompt")},
            **info, "pcm_seam_jump": round(seam_jump(x51), 2), "gain_db": round(gain_db, 2), "upmix_delay": delay,
            "files": {"ac3": f_ac3.name, "dts": f_dts.name}, "checks": checks}
    meta_p.write_text(json.dumps(meta, indent=1))
    log(f"#{t['n']:02d} {t['title']}: loop {info['loop_seconds']}s @ {info['measured_bpm']} BPM (attempt {info['attempt']}), seam harmony "
        f"{info['seam_harmony']} vs track median {info['inner_harmony_median']}, AC-3 seam x{checks['ac3']['seam_jump']}, DTS seam x{checks['dts']['seam_jump']}")
    return meta


def readme(gname, g, metas, batch):
    lines = [f"Alexandria Studio - {gname.replace('_', ' ')}", "=" * 60, "",
             "Instrumental seamless loops for event invitations, generated with Alexandria Studio on its ACE-Step 1.5 engine.",
             "", f"Style reference: {g['chart']} (sound only: tempo, groove, instruments, mix; no melodies or recordings were used).",
             f"Chart style notes: {g['style']}", "", "Tracks", "------"]
    for m in metas:
        lines.append(f"{m['n']:02d}  {m['title']}  ({m['kit']})")
        lines.append(f"    {m['measured_bpm']} BPM (asked {m['bpm']}), {m['key']}, {m['time_signature']}/4 time, loop {m['loop_seconds']} s ({m.get('loop_bars_used', m['loop_bars'])} bars)")
        lines.append(f"    Style prompt: {m['prompt']}")
    lines += ["", "Formats", "-------",
              f"- Dolby Digital (AC-3) 5.1, 48 kHz, {batch['formats']['ac3']}bps (.ac3)",
              f"- DTS 5.1, 48 kHz, {batch['formats']['dts']}bps (.dts). DTS 5.1 can't be encoded below about 670-768 kbps with the",
              "  encoders available (ffmpeg's DTS encoder and dcaenc), so 768 kbps, the standard DVD rate, is used.",
              "- 5.1 is upmixed from the engine's stereo render. Loudness is set to about -16 LUFS with peaks under -1 dBFS.",
              "", "Looping", "-------",
              "Each file is one full loop: set your player to repeat. The end crossfades into the start on the beat, and the",
              "5.1 upmix is periodic, so the audio itself has no break at the seam. Dolby Digital and DTS decoders add a few",
              "milliseconds of start-up padding (5.3 ms Dolby Digital, 10.7 ms DTS). Each file is aligned so that padding is the only",
              "difference from the master, it sits just before a note or drum hit, and it has 3 ms fades so it doesn't click.",
              "For sample-perfect looping in an editor or game engine, ask for the lossless 5.1 masters.",
              "", "Made with Alexandria Studio. Original compositions; no samples, melodies or lyrics from charted songs."]
    return "\n".join(lines) + "\n"


def stop_pod():
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("ALEX_RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY")
    if not pod or not key:
        return log("can't stop the pod: no pod id or API key")
    req = urllib.request.Request(f"https://api.runpod.io/v2/pods/{pod}/action", method="POST", data=json.dumps({"action": "stop"}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                          "User-Agent": "alexandria-batch/1.0 (+https://github.com/tdubofficial/alexandria-runpod)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            log(f"pod stop requested ({r.status})")
    except Exception as e:  # noqa: BLE001
        log(f"pod stop failed: {e}")


def main():
    batch = json.loads(Path(sys.argv[1]).read_text())
    out_root = APP / "batch_out" / batch["name"]; work_root = APP / "batch_work" / batch["name"]
    out_root.mkdir(parents=True, exist_ok=True)
    status_p = out_root / "status.json"
    status = {"name": batch["name"], "total": len(batch["tracks"]), "done": [], "failed": {}, "zips": [], "finished": False, "started": time.time()}
    for _ in range(60):                                        # wait for the engine
        try:
            urllib.request.urlopen(ACE + "/health", timeout=5); break
        except Exception:  # noqa: BLE001
            time.sleep(10)
    log(f"starting {batch['name']}: {len(batch['tracks'])} tracks -> {out_root}")
    metas = {}
    for t in batch["tracks"]:
        try:
            metas[t["n"]] = process(t, batch, out_root, work_root)
            status["done"].append(t["n"])
        except Exception as e:  # noqa: BLE001
            status["failed"][str(t["n"])] = str(e)[:300]
            log(f"#{t['n']:02d} FAILED: {e}")
        status_p.write_text(json.dumps(status, indent=1))
        # zip a genre as soon as all its tracks are done
        g = t["genre"]
        mine = [x for x in batch["tracks"] if x["genre"] == g]
        if all(x["n"] in metas or str(x["n"]) in status["failed"] for x in mine) and (mine[-1]["n"] == t["n"]):
            ms = [metas[x["n"]] for x in mine if x["n"] in metas]
            zp = out_root / f"Alexandria_Loops_{g}.zip"
            with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
                z.writestr(f"{g}/README.txt", readme(g, batch["genres"][g], ms, batch))
                z.writestr(f"{g}/tracklist.json", json.dumps(ms, indent=1))
                for m in ms:
                    for f in m["files"].values():
                        z.write(out_root / g / f, f"{g}/{f}")
            status["zips"].append(zp.name); status_p.write_text(json.dumps(status, indent=1))
            log(f"zipped {zp.name} ({zp.stat().st_size / 1e6:.1f} MB, {len(ms)} tracks)")
    status["finished"] = True; status["ended"] = time.time(); status_p.write_text(json.dumps(status, indent=1))
    log(f"FINISHED: {len(status['done'])} done, {len(status['failed'])} failed")
    hold = float(os.environ.get("ALEX_BATCH_HOLD_MIN", "120"))
    if os.environ.get("ALEX_BATCH_AUTOSTOP", "1") == "1":
        log(f"holding {hold:g} min for downloads, then stopping the pod")
        time.sleep(hold * 60)
        stop_pod()


if __name__ == "__main__":
    main()
