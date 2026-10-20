# Alexandria on RunPod

Runs the Alexandria Music Design Studio and the ACE-Step 1.5 music engine on a RunPod GPU pod.
You open the studio in a browser; nothing is installed on your own computer.

**No secrets live in this repo.** The studio password is stored only as a salted hash
(PBKDF2-SHA256, 600,000 rounds) in the RunPod secret `alexandria-auth`, which the pod reads at boot.

## How it starts

The pod's start command downloads `bootstrap.sh` from this repo and runs it in the background, then hands
off to the image's normal start script. `bootstrap.sh`:

1. downloads the latest copy of this repo,
2. writes the password hash from the `ALEX_AUTH_JSON` environment variable (set to `{{ RUNPOD_SECRET_alexandria-auth }}`),
3. runs `setup.sh` — installs or updates ACE-Step 1.5 on the network volume at `/workspace`, picks the planner size from the GPU's memory,
4. runs `start.sh` — starts ACE-Step's API privately on `127.0.0.1:8001` and the password-protected studio on port `7870`.

Open `https://<pod-id>-7870.proxy.runpod.net` and sign in. The first boot downloads ACE-Step's models, so it takes a while.
Progress: `/workspace/alexandria/logs/bootstrap.log`, `ace-step.log`, `bridge.log`. While signed in, read them in the browser at `/logs/bootstrap`, `/logs/ace-step` and `/logs/bridge`.

ACE-Step's API runs on `127.0.0.1:18011` because RunPod images use nginx on port 8001. The first render after a fresh volume downloads ACE-Step's models (about 18 minutes on the first test); later renders skip that.

To ship an update, push to `main` and restart the pod.

## Files

| File | What it does |
|---|---|
| `bootstrap.sh` | Pod start hook: fetch code, write the password hash, run setup and start |
| `setup.sh` | Install/update ACE-Step 1.5 and the studio onto `/workspace` (safe to re-run) |
| `start.sh` | Start ACE-Step's API (private) and the studio on port 7870 |
| `alexandria_bridge.py` | Studio server: sign-in, job queue, talks to ACE-Step |
| `alexandria.html` | The studio |
| `alexandria_train.py` | Training sets, uploads, labels, extraction and adapters |
| `extract.py` | Listens to training songs: stems, lyrics, tempo, key, tags (runs in its own venv) |

## Your library on the engine

When you use the studio from the engine, profiles, presets, custom genres and saved specs are stored on the
network volume (`/workspace/alexandria/store/`), so they survive restarts and work from any browser. Vocal
samples kept as voice references are stored in `/workspace/alexandria/voices/`, and renders with that profile
pass the sample to ACE-Step as `reference_audio_path` (ACE-Step calls this style transfer: it steers the sound,
it is not an exact voice clone). Move profiles between the claude.ai studio and the engine with Export and Import.

## AI writer (Design Spec, Smart Trim, AI wording)

On the engine, the writing features run on an open model on the same GPU: `start.sh` installs
[Ollama](https://ollama.com) to `/workspace/ollama` and downloads `qwen3.6:27b` (about 19 GB) on the first
boot, in the background. The studio says "AI writer not ready yet" until the download finishes
(watch `/logs/ollama-pull`). Set `ALEX_LLM_MODEL` to use another Ollama model, or `ALEX_LLM_DISABLE=1` to skip it.

Claude is the fallback. If the open model isn't ready, fails, or returns an incomplete spec, the bridge asks
Claude instead, when a key is set: create a RunPod secret named `anthropic-api-key` holding an Anthropic API
key and set the pod variable `ALEX_ANTHROPIC_KEY={{ RUNPOD_SECRET_anthropic-api-key }}`. Each result says which
model wrote it. `ALEX_CLAUDE_MODEL` picks the Claude model (default `claude-sonnet-5-5`).

Spec requests use prompt caching on the Claude API: the fixed studio instructions (about 1,000 tokens) go in
the system prompt marked `cache_control: ephemeral`, and the brief, settings and lyric examples follow in the
user message. A second spec within 5 minutes reads those instructions from the cache. `bridge.log` prints
each Claude call's cached-read, cache-write and uncached token counts. The open model gets the same fixed
system prompt first, so Ollama can reuse it too.

## Your lyrics as style examples

The Library's **My lyrics** card stores your own songs (.txt, .md or .lrc files, or pasted). A first line
like `# Title`, or a short first line followed by a blank line or a section tag, becomes the title. When the
writer designs a spec, it gets up to three of your songs closest to the brief and is told to write new lines in
your voice without reusing yours. Turn it off with **Use my lyrics as style examples**. On the engine the
lyrics are stored in `/workspace/alexandria/store/`.

Each song can also carry its BPM, key, duration, mood/target and the style prompt you used. Paste or upload a
song sheet (`SONG 1: "Title"`, `BPM: 102 | Key: F# Major | Duration: ~3:15 | Target: Upbeat/Hopeful`, a
`Style:` block, a `Lyrics:` block; several songs per file is fine) and the fields fill in. Uploading a song with
a title already in the library updates it. Artist names written as "Name-like", "Name-inspired", "Name-type" or
"Name-style" stay in your library but are removed before the writer sees the example.

## My Sound (your finished tracks)

The **My Sound** page measures finished tracks you produced: tempo, groove and swing, key, integrated loudness
(BS.1770 LUFS), peak-to-loudness ratio, low-end weight, brightness, stereo width, harmonic density and the energy
arc. The audio is decoded and measured in the browser and then discarded; only the numbers and the style cues
are saved (`/workspace/alexandria/store/tracks.json` on the engine). The cues can be edited, given notes about
what a meter can't hear (genre, instruments, vocals) or worded by the AI writer. A profile across all tracks,
plus the two tracks closest to the brief, goes to the writer with every spec. Export and Import move the
library as JSON. A track with the same title as a song in My lyrics fills that song's missing BPM, key and length.

Checked against synthetic tracks with known values: tempo within 1 BPM, loudness within 0.3 LU of ffmpeg's
EBU R128 meter, swing within 0.02, key correct except for relative major/minor pairs (both are shown when close).
The descriptor wording uses fixed thresholds and is an estimate.

## Train the engine on your music

My Sound -> **Train the engine on my music** (engine studio only):

1. Create a training set and tick the rights box (your recordings; anyone whose voice is on them agreed).
2. **Add songs**: MP3, WAV, FLAC, OGG or Opus (FLAC is best), lyrics as `Song Title.lyrics.txt`, or one `.zip`.
   Uploads go in 8 MB pieces and resume where they stopped if the connection drops. Each audio file is also measured
   in the browser and added to My Sound.
3. Labels are written for the engine from My Sound (caption, BPM, key) and My lyrics (lyrics), matched by title.
   Songs without a label are described by ACE-Step's own auto-labeler when training starts.
4. **Train on this set**: Fast (LoKr, recommended) or Standard (LoRA), Quick/Standard/Thorough (epochs 300/500/800 for
   LoKr; 200/500/800 for LoRA). The bridge drives ACE-Step's API: dataset scan, auto-label, preprocess, train, export.
   Renders wait while training runs; idle auto-stop treats training as activity.
5. **Listen: lyrics and instruments** (optional): the engine listens to every song in the set. Demucs (htdemucs) splits each
   one into drums, bass, other and vocals; Whisper large-v3 transcribes the separated vocal; tempo is beat-tracked on the drum
   stem and key is read from the bass and harmony stems; an AudioSet tagger (Audio Spectrogram Transformer) names genres,
   instruments and voice types; vocal presence, pitch range and syllables per beat describe the delivery. Results are in
   `<set>/extract/<song>.json`. Songs you have no notes or lyrics for are labeled from this. The transcription is a machine
   draft (sung vocals are hard to transcribe), so lyrics in My lyrics always win. The tools install once into their own
   environment, `/workspace/alexandria/xvenv` (about 6 GB with models), so the engine's packages are untouched.
   Labels prefer what you wrote: BPM and key on a song sheet in My lyrics beat the meter's estimate.
6. **Adapters for renders**: pick one (or none) and a strength; the choice is reloaded after the pod restarts.

Files live under the engine's folder because ACE-Step only reads paths inside its working directory:
`/workspace/ACE-Step-1.5/datasets/alexandria/<set>/` (songs, tensors, run) and
`/workspace/ACE-Step-1.5/adapters/alexandria/<adapter>/`.

**Starter adapters** (free, MIT, from Hugging Face, installed on request from a fixed list):
- `artificialguybr/AceStep_Refine_Redmond` - general musicality refinement for the turbo model (training data not described).
- `DisturbingTheField/ACE-Step-v1.5-raspy-vocal-and-instrumental-5-LoRAs` (instrumental adapter) - trained on the author's own tracks.
- `DisturbingTheField/ACE-Step-v1.5-acoustic-guitar-and-a-merge-LoRA` - trained on the author's own guitar recordings.

Scripted imports: while the pod variable `ALEX_BATCH_TOKEN` (24+ characters) is set, the `/train/*`, `/store/*` and
`/batch-files/*` routes also accept that token in an `X-Alex-Batch-Token` header. Remove the variable when you're done.

## Export formats

The player's **Export** button converts the current track on the engine with ffmpeg: WAV and AIFF masters
(24-bit/48 kHz, 24-bit/44.1 kHz), FLAC, ALAC, MP3 320, AAC 256, Ogg Vorbis, Opus, CD Red Book WAV
(16-bit/44.1 kHz, dithered), Dolby Digital 2.0 and 5.1, Dolby Digital Plus 5.1, Dolby TrueHD 5.1 (MKV),
DTS 5.1, six-channel WAV, MP4 (AAC) and MKV (Dolby Digital 5.1). Surround versions are upmixed from the
stereo render. The Dolby and DTS files come from ffmpeg's open-source encoders (TrueHD and DTS are marked
experimental there), not certified encoders. Without an engine, the studio builds the WAV, AIFF and CD
masters in the browser.

## Idle auto-stop

The studio stops its own pod after 30 minutes with no signed-in activity and no song rendering
(set `ALEX_IDLE_STOP_MINUTES` on the pod to change it, `0` to turn it off). The studio shows a countdown in
the last 5 minutes with a **Keep working** button. Stopping keeps the network volume; start the pod again
from RunPod when you need it.

The pod stops itself through RunPod's API (`POST https://api.runpod.io/v2/pods/{id}/action`), so it needs a
RunPod API key in `ALEX_RUNPOD_API_KEY` (set it to `{{ RUNPOD_SECRET_alexandria-runpod-key }}`) unless
RunPod provides `RUNPOD_API_KEY` inside the pod. `/health` reports `idle.key` so you can see which.

## Security

- RunPod's port links are public addresses. Every page and API call requires the password; after 5 wrong tries from one address, sign-in pauses for 10 minutes. Sessions last 12 hours.
- ACE-Step's own API listens only inside the pod.
- The bridge refuses to listen publicly without a password hash.
- Change the password: run `python3 alexandria_bridge.py --set-password` anywhere, then put the new `alexandria_auth.json` contents into the `alexandria-auth` secret and restart the pod.

## Tuning

- Engine settings: `/workspace/ACE-Step-1.5/.env` (delete its first comment line to keep your own edits across boots).
- Renders: `/workspace/alexandria/renders/` (one folder per render with the job and the audio).
