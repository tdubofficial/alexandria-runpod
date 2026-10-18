#!/usr/bin/env python3
"""
Alexandria Bridge — connects the Alexandria Music Design Studio to local music models.

Standard library only (Python 3.9+). Serves the studio page (alexandria.html placed next
to this file) and a small JSON API on the same address, so the browser never needs
cross-origin access.

    python alexandria_bridge.py                      # finds ACE-Step 1.5 on http://127.0.0.1:8001 by itself
    python alexandria_bridge.py --yue-dir ~/YuE
    python alexandria_bridge.py --ace-version 1 --ace-api http://127.0.0.1:8000
    python alexandria_bridge.py --custom-cmd "python my_engine.py --tags {tags_file} --lyrics {lyrics_file} --out {out_dir}"

On Windows you can double-click start_alexandria.bat instead. The studio opens at http://127.0.0.1:7870

API
  GET  /health              -> {"ok": true, "engines": {"yue": bool, "ace": bool, "custom": bool}}
  POST /render              -> {"job_id": "..."}      body: job JSON sent by the studio
  GET  /jobs/<id>           -> {"status", "progress", "message", "elapsed"}
  GET  /jobs/<id>/audio     -> audio bytes (wav / mp3 / flac)

Engines
  yue     Runs YuE's inference/infer.py (two-stage LM). Uses the spec's short YuE tags and
          lower-cased section lyrics; run_n_segments = number of lyric sections.
  ace     ACE-Step 1.5 (default): start its API server first (start_api_server.bat on
          Windows, or `uv run acestep-api`), which listens on http://127.0.0.1:8001. The
          bridge submits to /release_task, polls /query_result and downloads /v1/audio.
          ACE-Step v1 (--ace-version 1): calls infer-api.py's POST /generate (port 8000).
          Both use the full style prompt, the tagged lyrics and the bar-count duration.
  custom  Any command. Placeholders: {tags_file} {style_file} {lyrics_file} {out_dir}
          {duration} {seed}. The newest audio file written to {out_dir} is returned.
"""
import argparse
try:
    import alexandria_train          # training (engine); optional so a bridge copied on its own still runs
except ImportError:
    alexandria_train = None
import glob
import hashlib
import hmac
import html
import secrets
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STORE_NAMES = {"specs", "profiles", "presets", "genres", "lyrics", "tracks"}

# Export formats: key -> (extension, mime, ffmpeg output arguments). Surround versions are upmixed from the stereo render.
UPMIX = ["-af", "surround=chl_out=5.1:lfe_low=120,aresample=48000"]
EXPORT_FORMATS = {
    "wav48_24":  (".wav",  "audio/wav",  ["-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le"]),
    "aiff48_24": (".aiff", "audio/aiff", ["-ar", "48000", "-ac", "2", "-c:a", "pcm_s24be", "-f", "aiff"]),
    "wav44_24":  (".wav",  "audio/wav",  ["-ar", "44100", "-ac", "2", "-c:a", "pcm_s24le"]),
    "wav51":     (".wav",  "audio/wav",  UPMIX + ["-c:a", "pcm_s24le"]),
    "cd":        (".wav",  "audio/wav",  ["-af", "aresample=44100:dither_method=triangular", "-ac", "2", "-c:a", "pcm_s16le"]),
    "flac":      (".flac", "audio/flac", ["-ac", "2", "-c:a", "flac", "-compression_level", "8"]),
    "alac":      (".m4a",  "audio/mp4",  ["-ac", "2", "-c:a", "alac"]),
    "mp3":       (".mp3",  "audio/mpeg", ["-ar", "44100", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "320k"]),
    "aac":       (".m4a",  "audio/mp4",  ["-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "256k"]),
    "ogg":       (".ogg",  "audio/ogg",  ["-ar", "48000", "-ac", "2", "-c:a", "libvorbis", "-q:a", "6"]),
    "opus":      (".opus", "audio/ogg",  ["-ar", "48000", "-ac", "2", "-c:a", "libopus", "-b:a", "160k"]),
    "ac3_20":    (".ac3",  "audio/ac3",  ["-ar", "48000", "-ac", "2", "-c:a", "ac3", "-b:a", "448k"]),
    "ac3_51":    (".ac3",  "audio/ac3",  UPMIX + ["-c:a", "ac3", "-b:a", "640k"]),
    "eac3_51":   (".ec3",  "audio/eac3", UPMIX + ["-c:a", "eac3", "-b:a", "768k", "-f", "eac3"]),
    "truehd_51": (".mkv",  "audio/x-matroska", UPMIX + ["-c:a", "truehd", "-strict", "-2"]),
    "dts_51":    (".dts",  "audio/vnd.dts", UPMIX + ["-c:a", "dca", "-strict", "-2", "-b:a", "1509k"]),
    "mp4_aac":   (".mp4",  "audio/mp4",  ["-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart"]),
    "mkv_ac3":   (".mkv",  "audio/x-matroska", UPMIX + ["-c:a", "ac3", "-b:a", "640k"]),
}


# ---------- AI writer: open model on this machine first (Ollama), Claude API as the fallback ----------
LLM_JOBS = {}
LLM_LOCK = threading.Lock()


def ollama_url():
    return os.environ.get("ALEX_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def local_model():
    return os.environ.get("ALEX_LLM_MODEL", "qwen3.6:27b")


def claude_key():
    k = os.environ.get("ALEX_ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY") or ""
    return "" if k.startswith("{{") else k   # an unresolved RunPod secret placeholder means "not set"


def llm_status():
    st = {"local": {"up": False, "model": local_model(), "ready": False}, "claude": {"configured": bool(claude_key()),
          "model": os.environ.get("ALEX_CLAUDE_MODEL", "claude-sonnet-5-5")}}
    try:
        tags = http_json(ollama_url() + "/api/tags", timeout=2) or {}
        st["local"]["up"] = True
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        st["local"]["ready"] = local_model() in names or (local_model() + ":latest") in names
    except Exception:
        pass
    st["available"] = st["local"]["ready"] or st["claude"]["configured"]
    return st


def _strip_json(text):
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith(("{", "[")):
        i = min([x for x in (t.find("{"), t.find("[")) if x >= 0] or [0])
        t = t[i:]
    return t


def ask_local(prompt, schema, want_json, max_tokens, temperature, system=""):
    # The fixed system prompt goes first so Ollama can reuse its processed prefix between requests.
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": local_model(), "messages": msgs, "stream": False, "think": False,
            "options": {"temperature": temperature, "num_ctx": 16384, "num_predict": max_tokens}}
    if schema:
        body["format"] = schema
    elif want_json:
        body["format"] = "json"
    res = http_json(ollama_url() + "/api/chat", body, timeout=900)
    return ((res or {}).get("message") or {}).get("content", "")


def ask_claude(prompt, want_json, max_tokens, temperature, system=""):
    key = claude_key()
    if not key:
        raise RuntimeError("Claude fallback isn't set up (no Anthropic API key on this machine).")
    # Current Claude models: no temperature (rejected as deprecated); thinking is adaptive and counts toward
    # max_tokens, so leave generous room and ask for low effort (a spec is structured writing, not a puzzle).
    body = {"model": os.environ.get("ALEX_CLAUDE_MODEL", "claude-sonnet-5-5"),
            "max_tokens": max(int(max_tokens), 16000 if want_json else 4000),
            "output_config": {"effort": os.environ.get("ALEX_CLAUDE_EFFORT", "low")},
            "messages": [{"role": "user", "content": prompt + ("\n\nReturn only the JSON object, no commentary." if want_json else "")}]}
    if system:
        # Prompt caching: the system prompt is identical on every spec request, so mark it as a cache breakpoint.
        body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    def call(b):
        req = urllib.request.Request(os.environ.get("ALEX_ANTHROPIC_URL", "https://api.anthropic.com").rstrip("/") + "/v1/messages",
                                     data=json.dumps(b).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01",
                                              "User-Agent": "alexandria-bridge/1.5"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode())
    try:
        try:
            res = call(body)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 400 and ("output_config" in msg or "effort" in msg):   # model without effort control
                body.pop("output_config", None)
                res = call(body)
            else:
                raise RuntimeError(f"Claude API returned {e.code}: {msg[:200]}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Claude API returned {e.code}: {e.read().decode(errors='replace')[:200]}")
    u = res.get("usage") or {}
    usage = {k: u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")}
    print(f"Claude usage: {usage['cache_read_input_tokens']} cached read, {usage['cache_creation_input_tokens']} cache write, "
          f"{usage['input_tokens']} uncached in, {usage['output_tokens']} out, stop={res.get('stop_reason')}", flush=True)
    if res.get("stop_reason") == "max_tokens":
        raise RuntimeError("Claude's reply was cut off at the length limit")
    return "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text"), usage


def llm_worker(job_id, req):
    prompt = str(req.get("prompt") or "")
    system = str(req.get("system") or "")
    schema = req.get("schema") if isinstance(req.get("schema"), dict) else None
    want_json = bool(req.get("json") or schema)
    max_tokens = int(req.get("max_tokens") or (4096 if want_json else 1024))
    temperature = float(req.get("temperature") if req.get("temperature") is not None else 0.7)
    prefer = req.get("prefer") or "local"
    order = ["claude", "local"] if prefer == "claude" else ["local", "claude"]
    errors = []
    for provider in order:
        try:
            if provider == "local":
                st = llm_status()["local"]
                if not st["ready"]:
                    raise RuntimeError("the open model isn't downloaded yet" if st["up"] else "the open model server isn't running")
                with LLM_LOCK:
                    LLM_JOBS[job_id].update(status="running", provider="local", message=f"Writing with {local_model()}…")
                text = ask_local(prompt, schema, want_json, max_tokens, temperature, system)
                model, usage = local_model(), None
            else:
                with LLM_LOCK:
                    LLM_JOBS[job_id].update(status="running", provider="claude", message="Writing with Claude…")
                text, usage = ask_claude(prompt, want_json, max_tokens, temperature, system)
                model = os.environ.get("ALEX_CLAUDE_MODEL", "claude-sonnet-5-5")
            if want_json:  # must parse and carry the required keys, or try the next provider
                try:
                    val = json.loads(_strip_json(text))
                except ValueError:
                    raise RuntimeError("its reply wasn't valid JSON")
                missing = [k for k in (schema or {}).get("required", []) if not isinstance(val, dict) or k not in val]
                if missing:
                    raise RuntimeError("its reply was missing " + ", ".join(missing[:4]))
                text = _strip_json(text)
            with LLM_LOCK:
                LLM_JOBS[job_id].update(status="done", text=text, provider=provider, model=model, fallback_reasons=errors, usage=usage)
            return
        except Exception as e:  # noqa: BLE001
            errors.append(f"{provider}: {str(e)[:200]}")
    with LLM_LOCK:
        LLM_JOBS[job_id].update(status="error", message="No AI writer could answer. " + " | ".join(errors))


def ffmpeg_path():
    import shutil
    return shutil.which("ffmpeg")


def export_audio(src_bytes, src_name, fmt):
    """Convert one audio file to an export format with ffmpeg; returns (bytes, ext, mime)."""
    import tempfile
    ext, mime, args = EXPORT_FORMATS[fmt]
    ff = ffmpeg_path()
    if not ff:
        raise RuntimeError("ffmpeg isn't installed on this machine, so only WAV and AIFF exports are available here.")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td, "in" + (Path(src_name).suffix.lower() or ".wav"))
        dst = Path(td, "out" + ext)
        src.write_bytes(src_bytes)
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-map_metadata", "-1"] + args + [str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not dst.exists():
            raise RuntimeError("Conversion failed: " + (r.stderr.strip().splitlines() or ["unknown error"])[-1][:300])
        return dst.read_bytes(), ext, mime
VOICE_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
STORE_LOCK = threading.Lock()


def data_dir(name):
    d = Path(ARGS.out).parent / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def voice_path(voice_id):
    if not re.fullmatch(r"v_[a-f0-9]{12}", voice_id or ""):
        return None
    hits = [p for p in data_dir("voices").glob(voice_id + ".*") if p.suffix.lower() in VOICE_EXT]
    return hits[0] if hits else None
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg")
MIME = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg"}

ARGS = None
TRAINER = None          # alexandria_train.Trainer, set in main()
AUTH = None            # {"salt": bytes, "hash": bytes, "iterations": int} when a password is set
SESSIONS = {}          # token -> expiry timestamp
FAILS = {}             # client ip -> [timestamps of failed logins]
AUTH_LOCK = threading.Lock()
SESSION_SECONDS = 12 * 3600
LAST_ACTIVITY = time.time()   # idle auto-stop clock; any signed-in request or running job resets it
IDLE_STATE = {"stopping": False, "error": None}
JOBS = {}
JOBS_LOCK = threading.Lock()
RENDER_LOCK = threading.Lock()  # one render at a time: these models want the whole GPU


def set_job(job_id, **kw):
    with JOBS_LOCK:
        JOBS[job_id].update(kw)


def newest_audio(folder):
    files = [p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if p.lower().endswith(AUDIO_EXT)]
    if not files:
        return None
    mixed = [p for p in files if "mix" in os.path.basename(p).lower()]
    return max(mixed or files, key=os.path.getmtime)


def run_process(job_id, cmd, cwd, expected_seconds, label):
    """Run a command, stream its output into the job log, estimate progress by elapsed time."""
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            shell=isinstance(cmd, str))
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail = (tail + [line])[-40:]
        elapsed = time.time() - start
        pct = min(95.0, 100.0 * elapsed / max(1.0, expected_seconds))
        m = re.search(r"(\d{1,3})%\|", line)  # tqdm bars, when present
        set_job(job_id, progress=pct, message=f"{label}… {line[:140]}" if m else f"{label}…", log_tail=tail[-8:])
    code = proc.wait()
    if code != 0:
        oom = any("out of memory" in l.lower() for l in tail)
        raise RuntimeError("Out of memory on the GPU. Try fewer segments or a lower stage-2 batch size." if oom
                           else f"{label} exited with code {code}: {' | '.join(tail[-3:])[:400]}")


def render_yue(job_id, job, out_dir):
    if not ARGS.yue_dir:
        raise RuntimeError("YuE isn't set up. Restart the bridge with --yue-dir /path/to/YuE")
    tags = Path(out_dir, "genre.txt")
    lyrics = Path(out_dir, "lyrics.txt")
    tags.write_text((job.get("yue_tags") or "").strip() + "\n", encoding="utf-8")
    lyrics.write_text((job.get("yue_lyrics") or job.get("lyrics") or "").strip() + "\n", encoding="utf-8")
    segments = int(job.get("yue_segments") or 2)
    if ARGS.yue_max_segments:
        segments = min(segments, ARGS.yue_max_segments)
    cmd = [ARGS.python, "infer.py",
           "--cuda_idx", str(ARGS.cuda_idx),
           "--stage1_model", ARGS.yue_stage1,
           "--stage2_model", ARGS.yue_stage2,
           "--genre_txt", str(tags),
           "--lyrics_txt", str(lyrics),
           "--run_n_segments", str(segments),
           "--stage2_batch_size", str(ARGS.yue_stage2_batch),
           "--output_dir", str(Path(out_dir, "yue")),
           "--max_new_tokens", str(ARGS.yue_max_new_tokens),
           "--repetition_penalty", "1.1"]
    # Rough wall-clock guide for the progress bar only; real time depends on your GPU.
    run_process(job_id, cmd, Path(ARGS.yue_dir, "inference"), expected_seconds=240 * segments, label="YuE generating")


def http_json(url, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "null")


def ace_api_url():
    """ACE-Step API base URL: the --ace-api flag, else ACE-Step 1.5's default if it answers."""
    if ARGS.ace_api:
        return ARGS.ace_api.rstrip("/")
    if ARGS.ace_version == "1.5":
        try:
            urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=1.5).read()
            return "http://127.0.0.1:8001"
        except Exception:
            return None
    return None


def ace_alive():
    base = ace_api_url()
    if not base:
        return False
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as r:
            return r.status == 200 and "nginx" not in (r.headers.get("Server") or "").lower()
    except Exception:
        return False


def parse_key(text):
    m = re.match(r"\s*([A-G])([#b\u266f\u266d]?)\s*(major|minor|maj|min|m)?\b", text or "", re.I)
    if not m:
        return ""
    acc = {"\u266f": "#", "\u266d": "b"}.get(m.group(2), m.group(2))
    mode = (m.group(3) or "major").lower()
    return f"{m.group(1)}{acc} {'Minor' if mode.startswith('m') and mode != 'maj' and mode != 'major' else 'Major'}"


def render_ace15(job_id, job, out_dir, base):
    seed = job.get("seed")
    body = {
        "prompt": job.get("style_prompt") or "",
        "lyrics": job.get("lyrics") or "",
        "audio_duration": max(10.0, min(600.0, float(job.get("duration") or 200))),
        "audio_format": ARGS.ace_format,
        "batch_size": 1,
        "thinking": ARGS.ace_thinking,
        "vocal_language": "en",
        "task_type": "text2music",
        "time_signature": "4",
    }
    if job.get("bpm"):
        body["bpm"] = max(30, min(300, int(job["bpm"])))
    key = parse_key(job.get("key"))
    if key:
        body["key_scale"] = key
    if ARGS.ace_steps15:
        body["inference_steps"] = ARGS.ace_steps15
    if seed is not None:
        body.update(seed=int(seed), use_random_seed=False)
    ref = voice_path(job.get("reference_voice_id"))
    if ref:
        body["reference_audio_path"] = str(ref.resolve())   # ACE-Step "style transfer" reference
        set_job(job_id, message="ACE-Step generating with the voice reference…")
    try:
        res = http_json(base + "/release_task", body, timeout=120)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ACE-Step refused the job ({e.code}): {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Can't reach ACE-Step at {base} ({e.reason}). Is its API server running?")
    task_id = ((res or {}).get("data") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"ACE-Step didn't return a task id: {str(res)[:300]}")
    start = time.time()
    while True:
        time.sleep(2)
        pct = min(95.0, 100.0 * (time.time() - start) / ARGS.ace_expected_seconds)
        set_job(job_id, progress=pct, message="ACE-Step generating…")
        items = (http_json(base + "/query_result", {"task_id_list": [task_id]}) or {}).get("data") or []
        item = next((i for i in items if i.get("task_id") == task_id), items[0] if items else None)
        if not item or item.get("status") == 0:
            continue
        result = item.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = None
        if item.get("status") == 2:
            raise RuntimeError(f"ACE-Step reported a failure: {str(result or item)[:300]}")
        entries = result if isinstance(result, list) else [result] if isinstance(result, dict) else []
        file_url = next((e.get("file") for e in entries if isinstance(e, dict) and e.get("file")), None)
        if not file_url:
            raise RuntimeError(f"ACE-Step finished but returned no audio file: {str(result)[:300]}")
        url = file_url if file_url.startswith("http") else base + file_url
        ext = os.path.splitext(urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("path", ["x.wav"])[0])[1] or ".wav"
        with urllib.request.urlopen(url, timeout=300) as r:
            Path(out_dir, "ace_step" + ext).write_bytes(r.read())
        (Path(out_dir) / "ace_result.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return


def render_ace(job_id, job, out_dir):
    base = ace_api_url()
    if ARGS.ace_version == "1.5":
        if not base:
            raise RuntimeError("ACE-Step isn't running. Start ACE-Step 1.5's API server (start_api_server.bat), wait until it says it's ready, then render again.")
        return render_ace15(job_id, job, out_dir, base)
    if not ARGS.ace_api:
        raise RuntimeError("ACE-Step isn't set up. Start ACE-Step's infer-api.py server, then restart the bridge with --ace-api http://127.0.0.1:8000")
    out_path = str(Path(out_dir, "ace_step.wav").resolve())
    seed = job.get("seed")
    body = {
        "checkpoint_path": ARGS.ace_checkpoint,
        "bf16": not ARGS.ace_fp32,
        "torch_compile": False,
        "device_id": ARGS.cuda_idx,
        "output_path": out_path,
        "audio_duration": float(job.get("duration") or 200),
        "prompt": job.get("style_prompt") or "",
        "lyrics": job.get("lyrics") or "",
        # Defaults below follow ACE-Step v1's text-to-music settings; adjust to taste.
        "infer_step": ARGS.ace_steps,
        "guidance_scale": 15.0,
        "scheduler_type": "euler",
        "cfg_type": "apg",
        "omega_scale": 10.0,
        "actual_seeds": [int(seed)] if seed is not None else [int(time.time()) % 2**31],
        "guidance_interval": 0.5,
        "guidance_interval_decay": 0.0,
        "min_guidance_scale": 3.0,
        "use_erg_tag": True,
        "use_erg_lyric": True,
        "use_erg_diffusion": True,
        "oss_steps": [],
        "guidance_scale_text": 0.0,
        "guidance_scale_lyric": 0.0,
    }
    done = threading.Event()
    start = time.time()

    def pulse():
        while not done.is_set():
            pct = min(95.0, 100.0 * (time.time() - start) / ARGS.ace_expected_seconds)
            set_job(job_id, progress=pct, message="ACE-Step diffusing…")
            time.sleep(1)

    threading.Thread(target=pulse, daemon=True).start()
    try:
        req = urllib.request.Request(ARGS.ace_api.rstrip("/") + "/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60 * 60) as r:
            res = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ACE-Step returned {e.code}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Can't reach ACE-Step at {ARGS.ace_api} ({e.reason}). Is infer-api.py running?")
    finally:
        done.set()
    if res.get("status") != "success":
        raise RuntimeError(res.get("message") or "ACE-Step reported a failure")


def render_custom(job_id, job, out_dir):
    if not ARGS.custom_cmd:
        raise RuntimeError("No custom command set. Restart the bridge with --custom-cmd \"...\"")
    files = {
        "tags_file": Path(out_dir, "tags.txt"),
        "style_file": Path(out_dir, "style.txt"),
        "lyrics_file": Path(out_dir, "lyrics.txt"),
    }
    files["tags_file"].write_text(job.get("yue_tags") or "", encoding="utf-8")
    files["style_file"].write_text(job.get("style_prompt") or "", encoding="utf-8")
    files["lyrics_file"].write_text(job.get("lyrics") or "", encoding="utf-8")
    values = {k: shlex.quote(str(v)) for k, v in files.items()}
    values.update(out_dir=shlex.quote(str(out_dir)), duration=str(job.get("duration") or 200),
                  seed=str(job.get("seed") if job.get("seed") is not None else ""))
    run_process(job_id, ARGS.custom_cmd.format(**values), HERE, expected_seconds=ARGS.custom_expected_seconds,
                label="Engine running")


def worker(job_id, job):
    out_dir = Path(ARGS.out, time.strftime("%Y%m%d-%H%M%S") + "_" + re.sub(r"[^\w\-]+", "_", job.get("title") or "track")[:40])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    set_job(job_id, status="queued", message="Waiting for the GPU…", out_dir=str(out_dir))
    with RENDER_LOCK:
        set_job(job_id, status="running", started=time.time(), message="Starting engine…")
        try:
            {"yue": render_yue, "ace": render_ace, "custom": render_custom}[job.get("engine", "yue")](job_id, job, out_dir)
            audio = newest_audio(str(out_dir))
            if not audio:
                raise RuntimeError("The engine finished but wrote no audio file.")
            set_job(job_id, status="done", progress=100, audio=audio, message=f"Saved {audio}")
        except Exception as e:  # noqa: BLE001 — surface every failure to the studio
            set_job(job_id, status="error", message=str(e))


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alexandria — Sign in</title><style>
:root{color-scheme:dark}body{margin:0;min-height:100vh;display:grid;place-items:center;background:#000;color:#fff;
font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",system-ui,sans-serif;padding:16px}
form{width:min(380px,100%);background:#1C1C1E;border:1px solid #38383A;border-radius:18px;padding:28px;display:flex;flex-direction:column;gap:14px}
h1{margin:0;font-size:24px}p{margin:0;color:#8E8E93;font-size:14px}label{font-size:13px;color:rgba(235,235,245,.62)}
input{min-height:44px;border-radius:12px;border:1px solid #38383A;background:#000;color:#fff;padding:10px 14px;font-size:16px;font-family:ui-monospace,Menlo,Consolas,monospace}
input:focus{outline:none;border-color:#007AFF}button{min-height:44px;border:0;border-radius:14px;background:#007AFF;color:#fff;font-size:15px;font-weight:600;cursor:pointer}
.err{color:#FF6961;font-size:14px}</style></head><body><form method="post" action="/login">
<h1>Alexandria</h1><p>Music Design Studio — sign in to this engine.</p>{error}
<label for="pw">Password</label><input id="pw" name="password" type="password" autocomplete="current-password" autofocus required>
<button type="submit">Sign in</button></form></body></html>"""


def client_ip(handler):
    return (handler.headers.get("CF-Connecting-IP") or (handler.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
            or handler.client_address[0])


def touch():
    global LAST_ACTIVITY
    LAST_ACTIVITY = time.time()


def busy():
    if TRAINER is not None and TRAINER.running:
        return True
    with JOBS_LOCK:
        return any(j.get("status") in ("queued", "running") for j in JOBS.values())


def runpod_key():
    return os.environ.get("ALEX_RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY") or ""


def idle_seconds_left():
    if not ARGS.idle_stop_minutes:
        return None
    return max(0, int(ARGS.idle_stop_minutes * 60 - (time.time() - LAST_ACTIVITY)))


def stop_this_pod():
    pod_id, key = os.environ.get("RUNPOD_POD_ID"), runpod_key()
    if not pod_id or not key:
        IDLE_STATE["error"] = "Auto-stop needs RUNPOD_POD_ID and a RunPod API key (ALEX_RUNPOD_API_KEY)."
        print(IDLE_STATE["error"], flush=True)
        return False
    req = urllib.request.Request(f"https://api.runpod.io/v2/pods/{pod_id}/action", method="POST",
                                 data=json.dumps({"action": "stop"}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                          "User-Agent": "alexandria-bridge/1.1 (+https://github.com/tdubofficial/alexandria-runpod)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"Auto-stop: RunPod accepted the stop ({r.status}).", flush=True)
            return True
    except urllib.error.HTTPError as e:
        IDLE_STATE["error"] = f"Auto-stop failed: RunPod returned {e.code} {e.read().decode(errors='replace')[:200]}"
    except Exception as e:  # noqa: BLE001
        IDLE_STATE["error"] = f"Auto-stop failed: {e}"
    print(IDLE_STATE["error"], flush=True)
    return False


def idle_watch():
    limit = ARGS.idle_stop_minutes * 60
    print(f"Auto-stop: on, after {ARGS.idle_stop_minutes:g} idle minutes "
          f"(API key {'found' if runpod_key() else 'MISSING'}, pod {os.environ.get('RUNPOD_POD_ID') or 'unknown'}).", flush=True)
    while True:
        time.sleep(20)
        if busy():
            touch()
            continue
        if time.time() - LAST_ACTIVITY >= limit and not IDLE_STATE["stopping"]:
            print(f"Auto-stop: idle for {ARGS.idle_stop_minutes:g} minutes; stopping pod.", flush=True)
            IDLE_STATE["stopping"] = True
            if not stop_this_pod():
                IDLE_STATE["stopping"] = False
                touch()  # try again after another full idle period


def check_password(pw):
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), AUTH["salt"], AUTH["iterations"])
    return hmac.compare_digest(h, AUTH["hash"])


class Handler(BaseHTTPRequestHandler):
    server_version = "AlexandriaBridge/1.1"

    # ---------- auth ----------
    def session_token(self):
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "alx_session":
                return v
        return None

    def authed(self):
        if AUTH is None:
            if not self.path.startswith("/health?passive"):
                touch()
            return True
        tok = self.session_token()
        with AUTH_LOCK:
            exp = SESSIONS.get(tok) if tok else None
            if exp and exp > time.time():
                if not self.path.startswith("/health?passive"):
                    touch()
                return True
            if tok in SESSIONS:
                del SESSIONS[tok]
        return False

    def deny(self, path):
        if path in ("/", "/index.html", "/alexandria.html"):
            self.send_response(303)
            self.send_header("Location", "/login")
            self.end_headers()
        else:
            self.send_json({"error": "Sign in first: open the studio address and enter your password."}, 401)

    def login_page(self, error="", code=200):
        data = LOGIN_PAGE.replace("{error}", f'<div class="err">{html.escape(error)}</div>' if error else "").encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_login(self):
        ip = client_ip(self)
        now = time.time()
        with AUTH_LOCK:
            recent = [t for t in FAILS.get(ip, []) if now - t < 600]
            FAILS[ip] = recent
        if len(recent) >= 5:
            return self.login_page("Too many wrong passwords. Wait 10 minutes, then try again.", 429)
        length = min(int(self.headers.get("Content-Length") or 0), 4096)
        form = urllib.parse.parse_qs(self.rfile.read(length).decode(errors="replace"))
        pw = (form.get("password") or [""])[0]
        if not pw or not check_password(pw):
            with AUTH_LOCK:
                FAILS.setdefault(ip, []).append(now)
            time.sleep(1.0)
            return self.login_page("That password isn't right.", 401)
        tok = secrets.token_urlsafe(32)
        with AUTH_LOCK:
            SESSIONS[tok] = now + SESSION_SECONDS
            FAILS.pop(ip, None)
        secure = "; Secure" if self.headers.get("X-Forwarded-Proto", "").lower() == "https" else ""
        self.send_response(303)
        self.send_header("Set-Cookie", f"alx_session={tok}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_SECONDS}{secure}")
        self.send_header("Location", "/")
        self.end_headers()

    def do_logout(self):
        tok = self.session_token()
        with AUTH_LOCK:
            SESSIONS.pop(tok, None)
        self.send_response(303)
        self.send_header("Set-Cookie", "alx_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
        self.send_header("Location", "/login")
        self.end_headers()

    def log_message(self, fmt, *args):
        if ARGS.verbose:
            super().log_message(fmt, *args)

    def cors(self):
        if AUTH is not None:
            return  # password mode: same-origin only
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def batch_files(self, rel):
        """Batch results, for fetching without a browser session. Enabled only while ALEX_BATCH_TOKEN (24+ chars) is set;
        the token must arrive in the X-Alex-Batch-Token header. Anything else gets a plain 404."""
        token = os.environ.get("ALEX_BATCH_TOKEN", "")
        given = self.headers.get("X-Alex-Batch-Token", "")
        if len(token) < 24 or not hmac.compare_digest(token.encode(), given.encode()):
            return self.send_json({"error": "not found"}, 404)
        root = (Path(ARGS.out).parent / "batch_out").resolve()
        target = (root / rel).resolve()
        if root not in target.parents and target != root:
            return self.send_json({"error": "not found"}, 404)
        if target.is_dir():
            files = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "mtime": int(p.stat().st_mtime)}
                     for p in sorted(target.rglob("*")) if p.is_file()]
            return self.send_json({"files": files})
        if not target.is_file():
            return self.send_json({"error": "not found"}, 404)
        size = target.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "application/zip" if target.suffix == ".zip" else "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(target, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 1 << 20)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/login":
            return self.login_page() if AUTH is not None else self.deny("/")
        if path == "/logout":
            return self.do_logout()
        if path.startswith("/batch-files/"):
            return self.batch_files(urllib.parse.unquote(path[len("/batch-files/"):]))
        if not self.authed():
            return self.deny(path)
        if path.startswith("/train/"):
            return TRAINER.handle(self, "GET", path) if TRAINER else self.send_json({"error": "Training isn't available on this bridge"}, 404)
        if path in ("/", "/index.html", "/alexandria.html"):
            page = HERE / "alexandria.html"
            if not page.exists():
                return self.send_json({"error": "Put alexandria.html (Save offline copy in the studio) next to alexandria_bridge.py"}, 404)
            data = page.read_bytes()
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        if path == "/health":
            return self.send_json({"ok": True, "version": "1.6", "store": True,
                                   "idle": {"enabled": bool(ARGS.idle_stop_minutes), "seconds_left": idle_seconds_left(),
                                            "minutes": ARGS.idle_stop_minutes, "key": bool(runpod_key()),
                                            "error": IDLE_STATE["error"]},
                                   "engines": {"yue": bool(ARGS.yue_dir), "ace": ace_alive(), "custom": bool(ARGS.custom_cmd)}})
        if path == "/llm/status":
            return self.send_json(llm_status())
        m = re.fullmatch(r"/llm/([a-f0-9]{12})", path)
        if m:
            with LLM_LOCK:
                job = dict(LLM_JOBS.get(m.group(1)) or {})
            return self.send_json(job or {"status": "error", "message": "No such writing job"}, 200 if job else 404)
        if path == "/formats":
            return self.send_json({"ffmpeg": bool(ffmpeg_path()), "formats": list(EXPORT_FORMATS)})
        m = re.fullmatch(r"/store/([a-z]+)", path)
        if m and m.group(1) in STORE_NAMES:
            f = data_dir("store") / f"{m.group(1)}.json"
            with STORE_LOCK:
                data = f.read_bytes() if f.exists() else b"[]"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        m = re.fullmatch(r"/voices/(v_[a-f0-9]{12})", path)
        if m:
            vp = voice_path(m.group(1))
            if not vp:
                return self.send_json({"error": "No such voice sample"}, 404)
            data = vp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", {".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4",
                                              ".aac": "audio/aac", ".webm": "audio/webm"}.get(vp.suffix.lower(), "audio/wav"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        m = re.fullmatch(r"/logs/(bootstrap|ace-step|bridge|ollama|ollama-pull)", path)
        if m:
            f = Path(ARGS.out).parent / "logs" / f"{m.group(1)}.log"
            text = "\n".join(f.read_text(errors="replace").splitlines()[-300:]) if f.exists() else f"(no {f.name} yet)"
            data = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)
        m = re.fullmatch(r"/jobs/([\w\-]+)(/audio)?", path)
        if m:
            with JOBS_LOCK:
                job = dict(JOBS.get(m.group(1)) or {})
            if not job:
                return self.send_json({"error": "No such job"}, 404)
            if m.group(2):
                if job.get("status") != "done":
                    return self.send_json({"error": "Audio not ready"}, 409)
                data = Path(job["audio"]).read_bytes()
                self.send_response(200)
                self.cors()
                self.send_header("Content-Type", MIME.get(Path(job["audio"]).suffix.lower(), "application/octet-stream"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            elapsed = time.time() - job["started"] if job.get("started") else 0
            return self.send_json({k: job.get(k) for k in ("status", "progress", "message", "log_tail")} | {"elapsed": elapsed})
        self.send_json({"error": "Not found"}, 404)

    def do_PUT(self):
        if not self.authed():
            return self.deny(self.path)
        if self.path.split("?")[0].startswith("/train/"):
            return TRAINER.handle(self, "PUT", self.path.split("?")[0]) if TRAINER else self.send_json({"error": "Training isn't available on this bridge"}, 404)
        m = re.fullmatch(r"/store/([a-z]+)", self.path.split("?")[0])
        if not m or m.group(1) not in STORE_NAMES:
            return self.send_json({"error": "Not found"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        if length > 40 * 1024 * 1024:
            return self.send_json({"error": "Too large"}, 413)
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw or b"[]")
            assert isinstance(data, list)
        except Exception:
            return self.send_json({"error": "Body must be a JSON list"}, 400)
        f = data_dir("store") / f"{m.group(1)}.json"
        with STORE_LOCK:
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(f)
        return self.send_json({"ok": True, "count": len(data)})

    def do_DELETE(self):
        if not self.authed():
            return self.deny(self.path)
        if self.path.split("?")[0].startswith("/train/"):
            return TRAINER.handle(self, "DELETE", self.path.split("?")[0]) if TRAINER else self.send_json({"error": "Training isn't available on this bridge"}, 404)
        m = re.fullmatch(r"/voices/(v_[a-f0-9]{12})", self.path.split("?")[0])
        vp = voice_path(m.group(1)) if m else None
        if not vp:
            return self.send_json({"error": "No such voice sample"}, 404)
        vp.unlink()
        return self.send_json({"ok": True})

    def do_POST(self):
        if self.path.split("?")[0] == "/login" and AUTH is not None:
            return self.do_login()
        if not self.authed():
            return self.deny(self.path)
        if self.path.split("?")[0].startswith("/train/"):
            return TRAINER.handle(self, "POST", self.path.split("?")[0]) if TRAINER else self.send_json({"error": "Training isn't available on this bridge"}, 404)
        if self.path.split("?")[0] == "/llm":
            try:
                req = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 2 * 1024 * 1024)) or b"{}")
            except json.JSONDecodeError:
                return self.send_json({"error": "Body must be JSON"}, 400)
            job_id = uuid.uuid4().hex[:12]
            with LLM_LOCK:
                LLM_JOBS[job_id] = {"status": "queued", "message": "Queued", "started": time.time()}
                for old in [k for k, v in LLM_JOBS.items() if time.time() - v.get("started", 0) > 3600]:
                    LLM_JOBS.pop(old, None)
            threading.Thread(target=llm_worker, args=(job_id, req), daemon=True).start()
            return self.send_json({"id": job_id})
        if self.path.split("?")[0] == "/export":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            fmt = (q.get("fmt") or [""])[0]
            if fmt not in EXPORT_FORMATS:
                return self.send_json({"error": "Unknown export format"}, 400)
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= 400 * 1024 * 1024:
                return self.send_json({"error": "Audio must be under 400 MB"}, 413)
            body = self.rfile.read(length)
            try:
                data, ext, mime = export_audio(body, urllib.parse.unquote(self.headers.get("X-Filename") or "audio.wav"), fmt)
            except subprocess.TimeoutExpired:
                return self.send_json({"error": "Conversion took too long"}, 504)
            except Exception as e:  # noqa: BLE001
                return self.send_json({"error": str(e)}, 500)
            name = re.sub(r"[^\w\- ]+", "", (q.get("name") or ["Alexandria"])[0]).strip() or "Alexandria"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'attachment; filename="{name}{ext}"')
            self.send_header("X-Extension", ext)
            self.end_headers()
            return self.wfile.write(data)
        if self.path.split("?")[0] == "/voices":
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= 30 * 1024 * 1024:
                return self.send_json({"error": "Voice samples must be under 30 MB"}, 413)
            ext = Path(urllib.parse.unquote(self.headers.get("X-Filename") or "sample.wav")).suffix.lower()
            ext = ext if ext in VOICE_EXT else ".wav"
            vid = "v_" + uuid.uuid4().hex[:12]
            (data_dir("voices") / (vid + ext)).write_bytes(self.rfile.read(length))
            return self.send_json({"id": vid, "bytes": length})
        if self.path.split("?")[0] != "/render":
            return self.send_json({"error": "Not found"}, 404)
        try:
            job = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except json.JSONDecodeError:
            return self.send_json({"error": "Body must be JSON"}, 400)
        if job.get("engine") not in ("yue", "ace", "custom"):
            return self.send_json({"error": "engine must be yue, ace or custom"}, 400)
        if job.get("engine") == "yue" and not ARGS.yue_dir:
            return self.send_json({"error": "YuE isn't set up on this computer. Pick ACE-Step (Local) as the engine, or restart the bridge with --yue-dir"}, 400)
        if len(job.get("style_prompt") or "") > 2000:
            return self.send_json({"error": "Style prompt is over 2000 characters"}, 400)
        job_id = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "queued", "progress": 0, "message": "Queued"}
        threading.Thread(target=worker, args=(job_id, job), daemon=True).start()
        self.send_json({"job_id": job_id})


def main():
    global ARGS
    p = argparse.ArgumentParser(description="Alexandria Bridge: local engine server for the Music Design Studio")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (keep 127.0.0.1 unless you mean to expose it)")
    p.add_argument("--port", type=int, default=7870)
    p.add_argument("--out", default=str(HERE / "renders"), help="Where renders and job files are written")
    p.add_argument("--python", default="python", help="Python used to run YuE (e.g. your conda env's python)")
    p.add_argument("--cuda-idx", type=int, default=0)
    p.add_argument("--yue-dir", help="Path to the cloned YuE repo")
    p.add_argument("--yue-stage1", default="m-a-p/YuE-s1-7B-anneal-en-cot")
    p.add_argument("--yue-stage2", default="m-a-p/YuE-s2-1B-general")
    p.add_argument("--yue-stage2-batch", type=int, default=4)
    p.add_argument("--yue-max-new-tokens", type=int, default=3000)
    p.add_argument("--yue-max-segments", type=int, default=0, help="Cap sections per render (0 = all)")
    p.add_argument("--ace-api", help="ACE-Step API base URL (1.5 default: auto-detect http://127.0.0.1:8001; v1: http://127.0.0.1:8000)")
    p.add_argument("--ace-version", choices=["1.5", "1"], default="1.5")
    p.add_argument("--ace-thinking", action="store_true", help="ACE-Step 1.5: let its language model plan the song (needs an LM loaded)")
    p.add_argument("--ace-format", default="flac", choices=["flac", "wav", "mp3"], help="ACE-Step 1.5 output format")
    p.add_argument("--ace-steps15", type=int, default=0, help="ACE-Step 1.5 inference steps (0 = server default)")
    p.add_argument("--no-browser", action="store_true", help="Don't open the studio in your browser on start")
    p.add_argument("--ace-checkpoint", default=str(Path.home() / ".cache/ace-step/checkpoints"))
    p.add_argument("--ace-steps", type=int, default=60)
    p.add_argument("--ace-fp32", action="store_true", help="Disable bf16 (e.g. on macOS)")
    p.add_argument("--ace-expected-seconds", type=float, default=90, help="Progress-bar estimate only")
    p.add_argument("--custom-cmd", help="Custom engine command template")
    p.add_argument("--custom-expected-seconds", type=float, default=180, help="Progress-bar estimate only")
    p.add_argument("--auth-file", default=str(HERE / "alexandria_auth.json"),
                   help="Password hash file; when present, every page and API call needs a sign-in")
    p.add_argument("--set-password", action="store_true", help="Prompt for a new password, save its hash, and exit")
    p.add_argument("--idle-stop-minutes", type=float, default=0,
                   help="Stop this RunPod pod after this many minutes with no signed-in activity and no render (0 = off)")
    p.add_argument("--verbose", action="store_true")
    ARGS = p.parse_args()
    global AUTH
    if ARGS.set_password:
        import getpass
        pw = getpass.getpass("New password (12+ characters): ")
        if len(pw) < 12 or pw != getpass.getpass("Repeat it: "):
            raise SystemExit("Passwords didn't match or were shorter than 12 characters. Nothing changed.")
        salt = secrets.token_bytes(16)
        Path(ARGS.auth_file).write_text(json.dumps({"algorithm": "pbkdf2_sha256", "iterations": 600000, "salt": salt.hex(),
            "hash": hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 600000).hex()}, indent=2))
        raise SystemExit(f"Password saved to {ARGS.auth_file}. Restart the bridge to use it.")
    if Path(ARGS.auth_file).exists():
        a = json.loads(Path(ARGS.auth_file).read_text())
        AUTH = {"salt": bytes.fromhex(a["salt"]), "hash": bytes.fromhex(a["hash"]), "iterations": int(a["iterations"])}
    elif ARGS.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("Refusing to listen on a public address without a password. Run with --set-password first.")
    if ARGS.yue_dir:
        ARGS.yue_dir = os.path.expanduser(ARGS.yue_dir)
        if not Path(ARGS.yue_dir, "inference", "infer.py").exists():
            raise SystemExit(f"Can't find inference/infer.py under {ARGS.yue_dir}")
    ARGS.ace_checkpoint = os.path.expanduser(ARGS.ace_checkpoint)
    Path(ARGS.out).mkdir(parents=True, exist_ok=True)
    ace = ace_api_url()
    engines = [n for n, on in (("YuE", ARGS.yue_dir), (f"ACE-Step {ARGS.ace_version} at {ace}", ace), ("custom", ARGS.custom_cmd)) if on]
    print(f"Sign-in: {'password required' if AUTH else 'off (local only)'}")
    print(f"Alexandria Bridge on http://{ARGS.host}:{ARGS.port}  engines: {', '.join(engines) or 'none found yet'}")
    ls_ = llm_status()
    print(f"AI writer: open model {local_model()} {'ready' if ls_['local']['ready'] else ('server up, model not downloaded yet' if ls_['local']['up'] else 'not running')}; "
          f"Claude fallback {'configured' if ls_['claude']['configured'] else 'not configured'}")
    if not engines:
        print("  No music engine is running. Start ACE-Step 1.5's API server (start_api_server.bat);")
        print("  the bridge will find it on http://127.0.0.1:8001 without a restart.")
    print(f"Renders: {ARGS.out}")
    global TRAINER
    if alexandria_train is None:
        print("Training: off (alexandria_train.py isn't next to the bridge)")
    else:
      TRAINER = alexandria_train.Trainer(os.environ.get("ALEX_ACE_DIR", "/workspace/ACE-Step-1.5"), ace_api_url, RENDER_LOCK,
                                       log=lambda m: print(m, flush=True))
      print(f"Training sets: {TRAINER.root}")
      threading.Thread(target=TRAINER.restore_active, daemon=True).start()
    if ARGS.idle_stop_minutes:
        threading.Thread(target=idle_watch, daemon=True).start()
    if not ARGS.no_browser and (HERE / "alexandria.html").exists():
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{ARGS.port}/")).start()
    ThreadingHTTPServer((ARGS.host, ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
