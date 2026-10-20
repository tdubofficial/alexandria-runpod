"""Alexandria extraction: listen to your own songs and pull out what training needs.

Run by alexandria_train.py in its own virtual environment (/workspace/alexandria/xvenv) so the engine's packages are
never touched. For each audio file in SONGS_DIR it writes OUT_DIR/<file name>.json:
  stems      - loudness share of drums / bass / other / vocals (Demucs htdemucs source separation)
  vocals     - how much of the song has a lead vocal, its pitch range, and the delivery rate (syllables per second / per beat)
  lyrics     - a transcription of the separated vocal (Whisper large-v3), grouped into sections; repeated sections are
               marked [Chorus]. Machine transcription of sung vocals: read and correct it before training.
  tempo      - beat tracking on the separated drums (and on the full mix as a cross-check)
  key        - Krumhansl-Schmuckler key from the bass + harmony stems, top 3 with scores
  tags       - AudioSet sound tags (Audio Spectrogram Transformer) for the full mix and for the instrumental stems:
               genres, instruments, vocal types
Progress goes to OUT_DIR/status.json. Usage: extract.py SONGS_DIR OUT_DIR
"""
import difflib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".opus"}
SONGS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
STATUS = OUT / "status.json"
state = {"phase": "loading models", "done": [], "failed": {}, "total": 0, "started": time.time(), "finished": False}


def save_state(**kw):
    state.update(kw, updated=time.time())
    tmp = STATUS.with_suffix(".tmp"); tmp.write_text(json.dumps(state, indent=1)); tmp.replace(STATUS)


def decode(path, sr, ch):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, ch).T.copy()


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-9))


def frame_rms(x, sr, hop_s=0.25):
    hop = int(sr * hop_s); n = len(x) // hop
    return np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)


MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def key_of(y, sr, librosa):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=4096).mean(1)
    scores = []
    for i in range(12):
        scores.append((float(np.corrcoef(np.roll(MAJOR, i), chroma)[0, 1]), f"{NAMES[i]} major"))
        scores.append((float(np.corrcoef(np.roll(MINOR, i), chroma)[0, 1]), f"{NAMES[i]} minor"))
    scores.sort(reverse=True)
    return [{"key": k, "score": round(s, 3)} for s, k in scores[:3]]


def tempo_of(y, sr, librosa):
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    bpm, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=512, start_bpm=100, units="time")
    bpm = float(np.atleast_1d(bpm)[0])
    ibi = np.diff(beats)
    steady = float(1 - np.std(ibi) / np.mean(ibi)) if len(ibi) > 8 else 0.0
    # per-window tempo, to spot tempo changes or double/half-time sections
    win = librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=512, aggregate=None, start_bpm=bpm)
    return {"bpm": round(bpm, 1), "beats": len(beats), "steadiness": round(steady, 3),
            "windowed_p10_p90": [round(float(np.percentile(win, 10)), 1), round(float(np.percentile(win, 90)), 1)]}


VOWELS = re.compile(r"[aeiouy]+")


def syllables(text):
    n = 0
    for w in re.findall(r"[a-z']+", text.lower()):
        k = len(VOWELS.findall(w))
        if w.endswith("e") and k > 1 and not w.endswith(("le", "ee")):
            k -= 1
        n += max(1, k)
    return n


JUNK = re.compile(r"^\W*(thank you( for watching)?|you|bye|subtitles? by.*|\[music\]|music|♪+|\.+)\W*$", re.I)


def norm(s):
    return re.sub(r"[^a-z ]", "", s.lower())


def collapse_loops(text):
    """Whisper sometimes loops on a syllable ("la la la ..." x200). Keep at most 8 repeats of any 1-3 word pattern."""
    words = text.split()
    if len(words) < 12:
        return text
    out = []
    for w in words:
        out.append(w)
        for n in (1, 2, 3):
            if len(out) >= n * 9 and all(out[-k - 1].lower().strip(",.!?") == out[-k - 1 - n].lower().strip(",.!?")
                                          for k in range(n * 8)):
                out.pop(); break
    return " ".join(out)


def sections(chunks, duration):
    """Lines that come back later in the song are the hook/chorus; runs of other lines are verses.
    A new section also starts after a vocal gap of 2.2 s or more."""
    texts = [norm(c["text"]) for c in chunks]
    rep = []
    for i, t in enumerate(texts):
        rep.append(len(t) > 8 and any(abs(i - j) > 3 and difflib.SequenceMatcher(None, t, texts[j]).ratio() > 0.72
                                      for j in range(len(texts))))
    runs, cur = [], []
    for i, c in enumerate(chunks):
        brk = cur and (rep[i] != cur[-1][1] or c["start"] - cur[-1][0]["end"] > 2.2 or len(cur) >= 16)
        if brk:
            runs.append(cur); cur = []
        cur.append((c, rep[i]))
    if cur:
        runs.append(cur)
    # a lone repeated line inside verse material stays in the verse
    merged = []
    for r in runs:
        is_hook = r[0][1] and len(r) >= 2
        if merged and not is_hook and not merged[-1][0] and r[0][0]["start"] - merged[-1][1][-1]["end"] <= 2.2 and len(merged[-1][1]) + len(r) <= 16:
            merged[-1][1].extend(x[0] for x in r)
        else:
            merged.append((is_hook, [x[0] for x in r]))
    out, verse = [], 0
    if chunks and chunks[0]["start"] > 8:
        out.append({"tag": "Intro", "start": 0.0, "end": round(chunks[0]["start"], 1), "lines": []})
    for hook, lines in merged:
        if hook:
            tag = "Chorus"
        else:
            verse += 1; tag = f"Verse {verse}"
        out.append({"tag": tag, "start": round(lines[0]["start"], 1), "end": round(lines[-1]["end"], 1), "lines": [c["text"] for c in lines]})
    if chunks and duration - chunks[-1]["end"] > 10:
        out.append({"tag": "Outro", "start": round(chunks[-1]["end"], 1), "end": round(duration, 1), "lines": []})
    return out


def lyrics_text(secs):
    blocks = []
    for s in secs:
        blocks.append(f"[{s['tag']}]" + ("\n" + "\n".join(s["lines"]) if s["lines"] else ""))
    return "\n\n".join(blocks)


# AudioSet labels worth reporting: genres, instruments, voices (everything else - insects, rooms, cap guns - is stem residue)
MUSIC_LABELS = set("""Singing|Male singing|Female singing|Child singing|Choir|Rapping|Humming|Whistling|Yodeling|Chant|Beatboxing|
Synthetic singing|Speech|Male speech, man speaking|Female speech, woman speaking|Narration, monologue|Musical instrument|
Plucked string instrument|Guitar|Electric guitar|Bass guitar|Acoustic guitar|Steel guitar, slide guitar|Tapping (guitar technique)|
Strum|Banjo|Sitar|Mandolin|Zither|Ukulele|Keyboard (musical)|Piano|Electric piano|Organ|Electronic organ|Hammond organ|Synthesizer|
Sampler|Harpsichord|Percussion|Drum kit|Drum machine|Drum|Snare drum|Rimshot|Drum roll|Bass drum|Timpani|Tabla|Cymbal|Hi-hat|
Wood block|Tambourine|Rattle (instrument)|Maraca|Gong|Tubular bells|Mallet percussion|Marimba, xylophone|Glockenspiel|Vibraphone|
Steelpan|Orchestra|Brass instrument|French horn|Trumpet|Trombone|Bowed string instrument|String section|Violin, fiddle|Pizzicato|
Cello|Double bass|Wind instrument, woodwind instrument|Flute|Saxophone|Clarinet|Harp|Bell|Church bell|Jingle bell|Bicycle bell|
Chime|Wind chime|Harmonica|Accordion|Bagpipes|Didgeridoo|Shofar|Theremin|Singing bowl|Scratching (performance technique)|
Pop music|Hip hop music|Grime music|Trap music|Rock music|Heavy metal|Punk rock|Grunge|Progressive rock|Rock and roll|
Psychedelic rock|Rhythm and blues|Soul music|Reggae|Dub|Country|Swing music|Bluegrass|Funk|Folk music|Middle Eastern music|Jazz|
Disco|Classical music|Opera|Electronic music|House music|Techno|Dubstep|Drum and bass|Electronica|Electronic dance music|
Ambient music|Trance music|Music of Latin America|Salsa music|Flamenco|Blues|Music for children|New-age music|Vocal music|
A capella|Music of Africa|Afrobeat|Christian music|Gospel music|Music of Asia|Carnatic music|Music of Bollywood|Ska|
Traditional music|Independent music|Song|Background music|Theme music|Jingle (music)|Soundtrack music|Lullaby|
Video game music|Christmas music|Dance music|Wedding music|Happy music|Funny music|Sad music|Tender music|Exciting music|
Angry music|Scary music|Bass (instrument role)|Lead (instrument role)|Clapping|Finger snapping|Cheering|Crowd|Reverberation|
Echo|Distortion|Sine wave""".replace("\n", "").split("|"))


def tags_of(clf, y16, sr=16000, win=10, every=15):
    if len(y16) < sr * 2:
        return []
    acc, n = {}, 0
    starts = list(range(0, max(1, int(len(y16) / sr) - win + 1), every)) or [0]
    for t in starts:
        seg = y16[t * sr:(t + win) * sr]
        for r in clf(seg, top_k=200):
            acc[r["label"]] = acc.get(r["label"], 0) + r["score"]
        n += 1
    ranked = [(k, v / n) for k, v in sorted(acc.items(), key=lambda kv: -kv[1]) if k in MUSIC_LABELS]
    return [{"label": k, "score": round(v, 4)} for k, v in ranked[:30]]


def main():
    files = sorted(p for p in SONGS.iterdir() if p.suffix.lower() in AUDIO_EXT)
    save_state(total=len(files))
    import torch
    import librosa
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from transformers import pipeline

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sep = get_model("htdemucs").to(dev).eval()
    asr = pipeline("automatic-speech-recognition", model=os.environ.get("ALEX_WHISPER_MODEL", "openai/whisper-large-v3"), device=dev,
                   torch_dtype=torch.float16 if dev == "cuda" else torch.float32)
    clf = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593", device=dev)

    for f in files:
        out_f = OUT / (f.name + ".json")
        if out_f.exists():
            state["done"].append(f.name); continue
        save_state(phase=f"working on {f.name}")
        try:
            t0 = time.time()
            wav = decode(f, sep.samplerate, 2)
            dur = wav.shape[1] / sep.samplerate
            x = torch.from_numpy(wav)
            ref = x.mean(0)
            mu, sd = ref.mean(), ref.std() + 1e-8
            with torch.no_grad():
                st = apply_model(sep, ((x - mu) / sd)[None].to(dev), split=True, overlap=0.25, progress=False)[0]
            st = (st * sd + mu).cpu().numpy()
            stems = {name: st[i].mean(0) for i, name in enumerate(sep.sources)}
            energy = {k: float(np.mean(v ** 2)) for k, v in stems.items()}
            tot = sum(energy.values()) or 1
            sr = sep.samplerate

            # vocal activity
            vr = frame_rms(stems["vocals"], sr)
            mr = frame_rms(wav.mean(0), sr)
            active = (db(vr) > db(mr) - 14) & (db(vr) > -45)
            presence = float(active.mean())

            # transcription of the separated vocal
            lyr, secs, chunks = "", [], []
            v16 = librosa.resample(stems["vocals"], orig_sr=sr, target_sr=16000)
            if presence > 0.08:
                res = asr(v16.astype(np.float32), chunk_length_s=30, batch_size=8, return_timestamps=True,
                          generate_kwargs={"task": "transcribe"})
                for c in res.get("chunks", []):
                    a, b = c["timestamp"]
                    b = b if b is not None else dur
                    raw = c["text"].strip()
                    txt = collapse_loops(raw)
                    if len(raw.split()) > 24 and len(txt.split()) < len(raw.split()) / 3:
                        txt = collapse_loops(" ".join(txt.split()[:8]))          # a loop: keep a short ad-lib
                    if not txt or JUNK.match(txt) or a is None:
                        continue
                    i0, i1 = int(a / 0.25), max(int(a / 0.25) + 1, int(b / 0.25))
                    if active[i0:i1].mean() < 0.25:                      # no vocal under it: a hallucination
                        continue
                    chunks.append({"start": float(a), "end": float(b), "text": txt})
                secs = sections(chunks, dur)
                lyr = lyrics_text(secs)

            # delivery rate
            sung = sum(c["end"] - c["start"] for c in chunks) or 0
            syl = sum(syllables(c["text"]) for c in chunks)

            drums = librosa.resample(stems["drums"], orig_sr=sr, target_sr=22050)
            mix22 = librosa.resample(wav.mean(0), orig_sr=sr, target_sr=22050)
            tempo_d = tempo_of(drums, 22050, librosa) if energy["drums"] / tot > 0.03 else None
            tempo_m = tempo_of(mix22, 22050, librosa)
            bpm = (tempo_d or tempo_m)["bpm"]
            harm = librosa.resample(stems["bass"] + stems["other"], orig_sr=sr, target_sr=22050)
            keys = key_of(harm, 22050, librosa)

            # vocal register
            f0_med = f0_lo = f0_hi = None
            if presence > 0.08:
                f0 = librosa.yin(v16, fmin=65, fmax=1000, sr=16000, frame_length=2048, hop_length=512)
                fr = frame_rms(v16, 16000, 512 / 16000)[: len(f0)]
                voiced = f0[: len(fr)][db(fr) > db(fr).max() - 30]
                if len(voiced) > 50:
                    f0_med, f0_lo, f0_hi = (round(float(np.percentile(voiced, p)), 1) for p in (50, 10, 90))

            inst16 = librosa.resample(stems["drums"] + stems["bass"] + stems["other"], orig_sr=sr, target_sr=16000)
            mix16 = librosa.resample(wav.mean(0), orig_sr=sr, target_sr=16000)
            result = {
                "file": f.name, "duration": round(dur, 1),
                "stems": {k: round(v / tot, 3) for k, v in energy.items()},
                "vocals": {"presence": round(presence, 3), "first_vocal": round(chunks[0]["start"], 1) if chunks else None,
                           "pitch_hz_median": f0_med, "pitch_hz_p10_p90": [f0_lo, f0_hi],
                           "syllables_per_second": round(syl / sung, 2) if sung > 5 else None,
                           "syllables_per_beat": round(syl / sung * 60 / bpm, 2) if sung > 5 and bpm else None},
                "lyrics": lyr, "sections": secs,
                "tempo": {"drums": tempo_d, "mix": tempo_m},
                "key": keys,
                "tags": {"mix": tags_of(clf, mix16), "instrumental": tags_of(clf, inst16)},
                "seconds": round(time.time() - t0, 1),
            }
            out_f.write_text(json.dumps(result, indent=1))
            state["done"].append(f.name)
        except Exception as e:  # noqa: BLE001
            state["failed"][f.name] = f"{e.__class__.__name__}: {e}"[:400]
            traceback.print_exc()
        save_state()
    save_state(phase="finished", finished=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        save_state(phase=f"error: {e.__class__.__name__}: {e}"[:400], finished=True)
