"""Alexandria training: upload your own songs, label them, and train an ACE-Step 1.5 adapter (LoKr or LoRA).

Used by alexandria_bridge.py (routes under /train, behind the studio's sign-in). Everything lives under the engine's
own folder because ACE-Step only reads paths inside its working directory:
  <ACE>/datasets/alexandria/<set>/songs/     audio + <name>.json (caption, bpm, keyscale...) + <name>.lyrics.txt
  <ACE>/datasets/alexandria/<set>/tensors/   preprocessed training tensors
  <ACE>/datasets/alexandria/<set>/run/       training checkpoints
  <ACE>/adapters/alexandria/<adapter>/       exported adapter + meta.json; active.json says which one renders use
Pipeline (ACE-Step API): /v1/dataset/scan -> auto_label_async (only songs with no caption) -> preprocess_async ->
/v1/training/start_lokr or /v1/training/start -> /v1/training/status -> /v1/training/export -> /v1/lora/load.
"""
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".opus"}      # what ACE-Step's dataset scanner reads
SIDE_EXT = {".txt", ".json", ".csv"}
MAX_UPLOAD = 4 * 1024 ** 3                                    # one file (a zip of an album) up to 4 GB
MAX_CHUNK = 16 * 1024 * 1024
PRESETS = {                                                   # epochs; ACE-Step's own defaults are 500 (LoKr) / suggested 500-800
    "lokr": {"quick": 300, "standard": 500, "thorough": 800},
    "lora": {"quick": 200, "standard": 500, "thorough": 800},
}


# Public adapters with permissive licenses, checked 2026-10-18 on Hugging Face (fixed list: nothing else is downloaded).
STARTERS = {
    "refine-redmond": {
        "title": "Refine Redmond (general musicality)", "repo": "artificialguybr/AceStep_Refine_Redmond", "license": "MIT",
        "files": {"standard/AceStep_Refine_Redmond_standard.safetensors": "adapter_model.safetensors",
                  "standard/adapter_config.json": "adapter_config.json"},
        "about": "DPO-refined LoRA for ACE-Step 1.5 turbo aimed at musicality and arrangement coherence (author reports ~70% blind A/B "
                 "preference). Training data not described by the author.", "scale": 0.7},
    "dtf-instrumental": {
        "title": "Self-made instrumentals (rock, ambient, guitar, piano, synth)", "repo": "DisturbingTheField/ACE-Step-v1.5-raspy-vocal-and-instrumental-5-LoRAs",
        "license": "MIT", "files": {"instrumental_adapter_model.safetensors": "adapter_model.safetensors", "adapter_config.json": "adapter_config.json"},
        "about": "Trained on 21 instrumental tracks the author made themselves. Works best as a light filter (strength 0.2-0.7).", "scale": 0.4},
    "dtf-acoustic-guitar": {
        "title": "Fingerstyle acoustic guitar", "repo": "DisturbingTheField/ACE-Step-v1.5-acoustic-guitar-and-a-merge-LoRA", "license": "MIT",
        "files": {"adapter_model.safetensors": "adapter_model.safetensors", "adapter_config.json": "adapter_config.json"},
        "about": "Trained on 6 recordings of the author's own fingerstyle guitar. Strength 0.2-0.7; can make songs calmer and darker.", "scale": 0.4},
}


def slug(s, n=40):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:n] or "set"


def safe_name(name):
    base = Path(urllib.parse.unquote(str(name))).name
    base = re.sub(r"[^\w .,\-()&'+\[\]]", "_", base).strip(" .")
    return base[:150]


class Trainer:
    def __init__(self, ace_dir, ace_url, render_lock, log=print):
        self.ace_dir = Path(ace_dir)
        self.root = self.ace_dir / "datasets" / "alexandria"
        self.adapters = self.ace_dir / "adapters" / "alexandria"
        self.ace_url = ace_url                                 # callable -> base URL or None
        self.render_lock = render_lock
        self.log = log
        self.lock = threading.Lock()
        self.job = {"running": False, "phase": "idle", "message": "", "set": None}
        self.installing = {}                                   # starter key -> "downloading" | "error: ..."
        self.extracting = {}                                   # set id -> "installing tools" | "running" | "error: ..."

    # ---------------------------------------------------------------- listen to the songs (extract.py)
    XVENV = Path(os.environ.get("ALEX_XVENV", "/workspace/alexandria/xvenv"))
    XPKGS = ["numpy==1.26.4", "torch==2.5.1", "torchaudio==2.5.1", "demucs==4.0.1", "transformers==4.46.3",
             "librosa==0.10.2.post1", "soundfile"]

    def _xvenv(self):
        """Tools for extraction live in their own venv on the volume, so the engine's packages are never changed."""
        py = self.XVENV / "bin" / "python"
        ready = self.XVENV / ".ready"
        if ready.exists() and ready.read_text() == " ".join(self.XPKGS):
            return py
        uv = shutil.which("uv") or "/workspace/.uv/bin/uv"
        logf = open(self.XVENV.parent / "logs" / "extract-install.log", "a")
        if not py.exists():
            subprocess.run([uv, "venv", str(self.XVENV), "--python", "3.11"], check=True, stdout=logf, stderr=logf)
        subprocess.run([uv, "pip", "install", "--python", str(py), *self.XPKGS], check=True, stdout=logf, stderr=logf)
        ready.write_text(" ".join(self.XPKGS))
        return py

    def extract_start(self, sid):
        d = self.set_dir(sid)
        if not (d / "songs").exists():
            raise FileNotFoundError("No such training set")
        if self.job.get("running"):
            raise RuntimeError("Training is running; extract after it finishes")
        if self.extracting.get(sid) in ("installing tools", "running"):
            return self.extracting[sid]
        self.extracting[sid] = "installing tools"

        def go():
            try:
                py = self._xvenv()
                self.extracting[sid] = "running"
                logf = open(self.XVENV.parent / "logs" / f"extract-{sid}.log", "a")
                subprocess.run([str(py), str(Path(__file__).with_name("extract.py")), str(d / "songs"), str(d / "extract")],
                               check=True, stdout=logf, stderr=logf, cwd=str(d),
                               env={**os.environ, "TORCH_HOME": os.environ.get("TORCH_HOME", "/workspace/hf-cache/torch")})
                self.extracting[sid] = "finished"
            except Exception as e:  # noqa: BLE001
                self.extracting[sid] = f"error: {str(e)[:300]}"
                self.log(f"extract {sid} failed: {e}")
        threading.Thread(target=go, daemon=True).start()
        return self.extracting[sid]

    def extract_results(self, sid):
        x = self.set_dir(sid) / "extract"
        res = {"state": self.extracting.get(sid, "idle"), "status": self.read_json(x / "status.json", {}), "songs": {}}
        if x.exists():
            for f in sorted(x.glob("*.json")):
                if f.name != "status.json":
                    res["songs"][f.name[:-5]] = self.read_json(f, {})
        return res

    # ---------------------------------------------------------------- engine calls
    def ace(self, path, body=None, method=None, timeout=120):
        base = self.ace_url()
        if not base:
            raise RuntimeError("The music engine isn't running yet.")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=data, method=method or ("POST" if data is not None else "GET"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                res = json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("detail") or detail
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"Engine said: {str(detail)[:300]}")
        if isinstance(res, dict) and "data" in res and "code" in res:      # ACE-Step's response envelope
            if res.get("code", 200) not in (200, None) or res.get("error"):
                raise RuntimeError(f"Engine said: {str(res.get('error') or res)[:300]}")
            return res.get("data") or {}
        return res

    @property
    def running(self):
        return bool(self.job.get("running")) or any(v in ("installing tools", "running") for v in self.extracting.values())

    def set_job(self, **kw):
        with self.lock:
            self.job.update(kw, updated=time.time())

    # ---------------------------------------------------------------- sets and files
    def set_dir(self, sid):
        if not re.fullmatch(r"[a-z0-9][a-z0-9\-]{0,39}", sid or ""):
            raise ValueError("bad set id")
        return self.root / sid

    def list_sets(self):
        out = []
        if self.root.exists():
            for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
                meta = self.read_json(d / "set.json", {})
                songs_dir = d / "songs"
                songs = []
                if songs_dir.exists():
                    for f in sorted(songs_dir.iterdir()):
                        if f.suffix.lower() in AUDIO_EXT:
                            stem = f.with_suffix("")
                            js = self.read_json(Path(str(stem) + ".json"), {})
                            lyr = Path(str(stem) + ".lyrics.txt")
                            songs.append({"file": f.name, "bytes": f.stat().st_size, "caption": js.get("caption", ""),
                                          "bpm": js.get("bpm"), "keyscale": js.get("keyscale", ""),
                                          "labeled": bool(js.get("caption")) or Path(str(stem) + ".caption.txt").exists(),
                                          "lyrics": lyr.exists()})
                    partial = [{"file": f.name[:-5], "bytes": f.stat().st_size} for f in songs_dir.glob("*.part")]
                else:
                    partial = []
                out.append({"id": d.name, "name": meta.get("name", d.name), "created": meta.get("created"),
                            "songs": songs, "partial": partial, "bytes": sum(s["bytes"] for s in songs)})
        return out

    def create_set(self, name):
        sid = slug(name)
        d = self.root / sid
        n = 2
        while d.exists():
            sid = f"{slug(name, 36)}-{n}"; d = self.root / sid; n += 1
        (d / "songs").mkdir(parents=True)
        (d / "set.json").write_text(json.dumps({"name": str(name)[:80], "created": time.time()}))
        return sid

    @staticmethod
    def read_json(p, default):
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return default

    def upload_chunk(self, sid, name, offset, total, body_reader, length):
        songs = self.set_dir(sid) / "songs"
        if not songs.exists():
            raise FileNotFoundError("No such training set")
        fname = safe_name(name)
        ext = Path(fname).suffix.lower()
        if ext not in AUDIO_EXT | SIDE_EXT | {".zip"}:
            raise ValueError(f"{ext or 'that file type'} isn't supported. Use MP3, WAV, FLAC, OGG, Opus, a .zip, or lyrics .txt files.")
        if not 0 < total <= MAX_UPLOAD or length > MAX_CHUNK:
            raise ValueError("File or chunk too large")
        part = songs / (fname + ".part")
        have = part.stat().st_size if part.exists() else 0
        if offset != have:
            return {"bytes": have, "resume": True}
        with open(part, "ab") as f:
            left = length
            while left:
                b = body_reader(min(left, 1 << 20))
                if not b:
                    break
                f.write(b); left -= len(b)
        have = part.stat().st_size
        if have < total:
            return {"bytes": have}
        final = songs / fname
        part.replace(final)
        if ext == ".zip":
            added = self.extract_zip(final, songs)
            final.unlink()
            return {"bytes": have, "done": True, "extracted": added}
        return {"bytes": have, "done": True}

    def upload_size(self, sid, name):
        songs = self.set_dir(sid) / "songs"
        fname = safe_name(name)
        part, final = songs / (fname + ".part"), songs / fname
        if final.exists():
            return {"bytes": final.stat().st_size, "done": True}
        return {"bytes": part.stat().st_size if part.exists() else 0}

    @staticmethod
    def extract_zip(zpath, dest):
        added = []
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                if info.is_dir() or "__MACOSX" in info.filename or Path(info.filename).name.startswith("."):
                    continue
                name = safe_name(Path(info.filename).name)
                ext = Path(name).suffix.lower()
                if ext not in AUDIO_EXT | SIDE_EXT or info.file_size > MAX_UPLOAD or (ext in AUDIO_EXT and info.file_size < 20000):
                    continue                                                       # skip junk and tiny "audio"
                with z.open(info) as src, open(dest / name, "wb") as dst:          # flattened, never outside dest
                    shutil.copyfileobj(src, dst, 1 << 20)
                added.append(name)
        return added

    def delete_song(self, sid, name):
        songs = self.set_dir(sid) / "songs"
        stem = Path(safe_name(name)).stem
        for f in list(songs.iterdir()):
            if f.name == safe_name(name) or f.name.startswith(stem + "."):
                f.unlink()

    def write_labels(self, sid, labels):
        """labels: {audio file name: {caption, bpm, keyscale, timesignature, language, lyrics}} -> sidecar files."""
        songs = self.set_dir(sid) / "songs"
        n = 0
        for fname, lab in (labels or {}).items():
            audio = songs / safe_name(fname)
            if not audio.exists():
                continue
            stem = str(audio.with_suffix(""))
            meta = {k: lab[k] for k in ("caption", "bpm", "keyscale", "timesignature", "language") if lab.get(k) not in (None, "")}
            if meta.get("bpm"):
                meta["bpm"] = int(round(float(meta["bpm"])))
            if meta:
                Path(stem + ".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            lyr = str(lab.get("lyrics") or "").strip()
            if lyr and lyr.lower() not in ("[instrumental]", "[inst]"):
                Path(stem + ".lyrics.txt").write_text(lyr, encoding="utf-8")
            n += 1
        return n

    def delete_set(self, sid):
        if self.running and self.job.get("set") == sid:
            raise RuntimeError("This set is training right now.")
        shutil.rmtree(self.set_dir(sid), ignore_errors=True)

    # ---------------------------------------------------------------- adapters
    def list_adapters(self):
        out = []
        if self.adapters.exists():
            for d in sorted(p for p in self.adapters.iterdir() if p.is_dir()):
                m = self.read_json(d / "meta.json", {})
                out.append({"name": d.name, **m})
        return out

    def adapter_path(self, name):
        d = self.adapters / slug(name, 60)
        lokr = d / "lokr_weights.safetensors"
        return lokr if lokr.exists() else d

    def active(self):
        return self.read_json(self.adapters / "active.json", {"name": None, "scale": 1.0})

    def use_adapter(self, name, scale=1.0):
        scale = max(0.0, min(1.0, float(scale)))
        if not name:
            try:
                self.ace("/v1/lora/unload", {})
            except RuntimeError:
                pass
            self.adapters.mkdir(parents=True, exist_ok=True)
            (self.adapters / "active.json").write_text(json.dumps({"name": None, "scale": scale}))
            return {"name": None}
        p = self.adapter_path(name)
        if not p.exists():
            raise FileNotFoundError("No such adapter")
        cur = self.active()
        if cur.get("name") != slug(name, 60) or not self.engine_has_adapter():
            try:
                self.ace("/v1/lora/unload", {})
            except RuntimeError:
                pass
            self.ace("/v1/lora/load", {"lora_path": str(p)}, timeout=600)
        self.ace("/v1/lora/scale", {"scale": scale})
        self.adapters.mkdir(parents=True, exist_ok=True)
        (self.adapters / "active.json").write_text(json.dumps({"name": slug(name, 60), "scale": scale}))
        return {"name": slug(name, 60), "scale": scale}

    def engine_has_adapter(self):
        try:
            return bool(self.ace("/v1/lora/status").get("lora_loaded"))
        except Exception:  # noqa: BLE001
            return False

    def restore_active(self):
        """After the engine (re)starts, put the chosen adapter back."""
        a = self.active()
        if not a.get("name"):
            return
        for _ in range(120):
            if self.ace_url():
                try:
                    self.ace("/v1/lora/load", {"lora_path": str(self.adapter_path(a["name"]))}, timeout=600)
                    self.ace("/v1/lora/scale", {"scale": a.get("scale", 1.0)})
                    self.log(f"Adapter '{a['name']}' loaded for renders (strength {a.get('scale', 1.0):.2f}).")
                    return
                except Exception as e:  # noqa: BLE001
                    self.log(f"Couldn't load adapter '{a['name']}' yet: {e}")
            time.sleep(15)

    def install_starter(self, key):
        st = STARTERS.get(key)
        if not st:
            raise ValueError("Unknown starter adapter")
        dest = self.adapters / slug(key, 60)
        tmp = self.adapters / (slug(key, 60) + ".downloading")
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        for src, name in st["files"].items():
            url = f"https://huggingface.co/{st['repo']}/resolve/main/{urllib.parse.quote(src)}"
            req = urllib.request.Request(url, headers={"User-Agent": "alexandria-bridge/1.6"})
            with urllib.request.urlopen(req, timeout=600) as r, open(tmp / name, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        (tmp / "meta.json").write_text(json.dumps({"starter": key, "title": st["title"], "repo": st["repo"], "license": st["license"],
                                                   "about": st["about"], "suggested_scale": st["scale"], "created": time.time(),
                                                   "source": f"https://huggingface.co/{st['repo']}"}, indent=1))
        shutil.rmtree(dest, ignore_errors=True); tmp.replace(dest)
        return {"name": dest.name}

    def delete_adapter(self, name):
        if self.active().get("name") == slug(name, 60):
            self.use_adapter(None)
        shutil.rmtree(self.adapters / slug(name, 60), ignore_errors=True)

    # ---------------------------------------------------------------- training
    def start(self, sid, method="lokr", preset="standard", name=None):
        if self.running:
            raise RuntimeError("A training run is already in progress.")
        if method not in PRESETS or preset not in PRESETS[method]:
            raise ValueError("Unknown method or preset")
        d = self.set_dir(sid)
        songs = [f for f in (d / "songs").iterdir() if f.suffix.lower() in AUDIO_EXT] if (d / "songs").exists() else []
        if not songs:
            raise ValueError("Upload some songs first.")
        adapter = slug(name or f"{sid}-{method}-{time.strftime('%Y%m%d-%H%M')}", 60)
        self.set_job(running=True, set=sid, method=method, preset=preset, adapter=adapter, phase="starting",
                     message="Starting…", started=time.time(), error=None, epoch=0, epochs=PRESETS[method][preset],
                     loss=None, eta=None, songs=len(songs))
        threading.Thread(target=self._run, args=(sid, method, PRESETS[method][preset], adapter), daemon=True).start()
        return self.job

    def stop(self):
        try:
            self.ace("/v1/training/stop", {})
        except RuntimeError:
            pass
        self.set_job(stop_requested=True, message="Stopping after the current step…")

    def _poll(self, path, done, label, every=5, limit=6 * 3600):
        t0 = time.time()
        while time.time() - t0 < limit:
            st = self.ace(path)
            if done(st):
                return st
            self.set_job(message=label(st))
            time.sleep(every)
        raise RuntimeError(f"Timed out: {path}")

    def _run(self, sid, method, epochs, adapter):
        d = self.set_dir(sid)
        songs_dir, tensors, run = d / "songs", d / "tensors", d / "run"
        restore = self.active()
        try:
            with self.render_lock:                                   # renders wait while the GPU trains
                if restore.get("name") or self.engine_has_adapter():  # train on the plain model
                    try:
                        self.ace("/v1/lora/unload", {})
                    except RuntimeError:
                        pass
                self.set_job(phase="scanning", message="Reading your songs…")
                scan = self.ace("/v1/dataset/scan", {"audio_dir": str(songs_dir), "dataset_name": sid, "all_instrumental": True})
                samples = scan.get("samples") or []
                unlabeled = sum(1 for s in samples if not s.get("labeled"))
                self.set_job(songs=len(samples), unlabeled=unlabeled)
                if unlabeled:
                    self.set_job(phase="labeling", message=f"Describing {unlabeled} song(s) that have no caption…")
                    t = self.ace("/v1/dataset/auto_label_async", {"only_unlabeled": True, "skip_metas": False})
                    tid = t.get("task_id")
                    if tid and t.get("total", 1):
                        st = self._poll(f"/v1/dataset/auto_label_status/{tid}", lambda s: s.get("status") in ("completed", "failed"),
                                        lambda s: f"Labeling {s.get('current', 0)}/{s.get('total', '?')}…")
                        if st.get("status") == "failed":
                            raise RuntimeError("Auto-labeling failed: " + str(st.get("error"))[:200])
                if tensors.exists():
                    shutil.rmtree(tensors)
                self.set_job(phase="preprocessing", message="Preparing audio for training…")
                t = self.ace("/v1/dataset/preprocess_async", {"output_dir": str(tensors), "skip_existing": False})
                tid = t.get("task_id")
                if not t.get("total"):
                    raise RuntimeError(t.get("message") or "No labeled songs to train on.")
                st = self._poll(f"/v1/dataset/preprocess_status/{tid}", lambda s: s.get("status") in ("completed", "failed"),
                                lambda s: f"Preparing {s.get('current', 0)}/{s.get('total', '?')}…")
                if st.get("status") == "failed":
                    raise RuntimeError("Preprocessing failed: " + str(st.get("error"))[:200])
                if run.exists():
                    shutil.rmtree(run)
                self.set_job(phase="training", message="Training…")
                common = {"tensor_dir": str(tensors), "train_epochs": epochs, "save_every_n_epochs": max(25, epochs // 5)}
                if method == "lokr":
                    self.ace("/v1/training/start_lokr", {**common, "output_dir": str(run)})
                else:
                    self.ace("/v1/training/start", {**common, "lora_output_dir": str(run)})
                time.sleep(3)

                def tlabel(s):
                    self.set_job(epoch=s.get("current_epoch", 0), loss=s.get("current_loss"), eta=s.get("estimated_time_remaining"))
                    eta = s.get("estimated_time_remaining") or 0
                    return f"Epoch {s.get('current_epoch', 0)}/{epochs}" + (f", about {int(eta // 60)} min left" if eta else "")
                st = self._poll("/v1/training/status", lambda s: not s.get("is_training"), tlabel, every=10)
                if st.get("error"):
                    raise RuntimeError("Training failed: " + str(st.get("error"))[:300])
                self.set_job(phase="saving", message="Saving the adapter…")
                dest = self.adapters / adapter
                self.adapters.mkdir(parents=True, exist_ok=True)
                self.ace("/v1/training/export", {"export_path": str(dest), "lora_output_dir": str(run)})
                losses = [h.get("loss") for h in (st.get("loss_history") or []) if h.get("loss") is not None]
                meta = {"set": sid, "set_name": self.read_json(d / "set.json", {}).get("name", sid), "method": method,
                        "epochs": epochs, "epochs_done": st.get("current_epoch"), "songs": len(samples),
                        "created": time.time(), "final_loss": losses[-1] if losses else None,
                        "stopped_early": bool(self.job.get("stop_requested"))}
                (dest / "meta.json").write_text(json.dumps(meta, indent=1))
            self.set_job(running=False, phase="done", message=f"Adapter '{adapter}' is ready. Turn it on under Adapters for renders.",
                         finished=time.time())
            self.log(f"Training finished: adapter {adapter}")
        except Exception as e:  # noqa: BLE001
            self.set_job(running=False, phase="error", error=str(e)[:400], message=str(e)[:400], finished=time.time())
            self.log(f"Training failed: {e}")
        finally:
            if restore.get("name"):                                  # put back the adapter renders were using
                try:
                    self.use_adapter(restore["name"], restore.get("scale", 1.0))
                except Exception:  # noqa: BLE001
                    pass

    # ---------------------------------------------------------------- HTTP routes (bridge has already checked sign-in)
    def handle(self, h, method, path):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(h.path).query)
        parts = [urllib.parse.unquote(p) for p in path.split("/")[2:]]   # after /train
        try:
            if method == "GET" and parts == ["status"]:
                return h.send_json({"sets": self.list_sets(), "job": self.job, "adapters": self.list_adapters(),
                                    "active": self.active(), "engine": bool(self.ace_url()), "presets": PRESETS, "installing": self.installing,
                                    "starters": {k: {kk: v[kk] for kk in ("title", "repo", "license", "about", "scale")} for k, v in STARTERS.items()}})
            if method == "POST" and parts == ["sets"]:
                body = self._body(h)
                return h.send_json({"id": self.create_set(body.get("name") or "My songs")})
            if method == "DELETE" and len(parts) == 2 and parts[0] == "sets":
                self.delete_set(parts[1]); return h.send_json({"ok": True})
            if parts[:1] == ["upload"] and len(parts) == 3:
                if method == "GET":
                    return h.send_json(self.upload_size(parts[1], parts[2]))
                if method == "PUT":
                    length = int(h.headers.get("Content-Length") or 0)
                    res = self.upload_chunk(parts[1], parts[2], int((q.get("offset") or ["0"])[0]), int((q.get("total") or ["0"])[0]),
                                            h.rfile.read, length)
                    if res.get("resume"):
                        return h.send_json(res, 409)
                    return h.send_json(res)
            if method == "DELETE" and len(parts) == 3 and parts[0] == "file":
                self.delete_song(parts[1], parts[2]); return h.send_json({"ok": True})
            if method == "POST" and len(parts) == 2 and parts[0] == "labels":
                return h.send_json({"written": self.write_labels(parts[1], self._body(h).get("songs"))})
            if len(parts) == 2 and parts[0] == "extract":
                if method == "POST":
                    return h.send_json({"state": self.extract_start(parts[1])})
                if method == "GET":
                    return h.send_json(self.extract_results(parts[1]))
            if method == "POST" and len(parts) == 2 and parts[0] == "start":
                b = self._body(h)
                return h.send_json({"job": self.start(parts[1], b.get("method", "lokr"), b.get("preset", "standard"), b.get("name"))})
            if method == "POST" and parts == ["stop"]:
                self.stop(); return h.send_json({"ok": True})
            if method == "POST" and len(parts) == 2 and parts[0] == "starter":
                key = parts[1]
                if key not in STARTERS:
                    raise ValueError("Unknown starter adapter")
                if self.installing.get(key) != "downloading":         # downloads outlast the proxy's 100 s, so run in the background
                    self.installing[key] = "downloading"
                    def go():
                        try:
                            self.install_starter(key); self.installing.pop(key, None)
                        except Exception as e:  # noqa: BLE001
                            self.installing[key] = f"error: {str(e)[:200]}"
                    threading.Thread(target=go, daemon=True).start()
                return h.send_json({"installing": key})
            if method == "POST" and parts == ["adapter"]:
                b = self._body(h)
                return h.send_json(self.use_adapter(b.get("name"), b.get("scale", 1.0)))
            if method == "DELETE" and len(parts) == 2 and parts[0] == "adapter":
                self.delete_adapter(parts[1]); return h.send_json({"ok": True})
            return h.send_json({"error": "Not found"}, 404)
        except FileNotFoundError as e:
            return h.send_json({"error": str(e)}, 404)
        except ValueError as e:
            return h.send_json({"error": str(e)}, 400)
        except RuntimeError as e:
            return h.send_json({"error": str(e)}, 409)

    @staticmethod
    def _body(h):
        n = min(int(h.headers.get("Content-Length") or 0), 8 * 1024 * 1024)
        try:
            return json.loads(h.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            raise ValueError("Body must be JSON")
