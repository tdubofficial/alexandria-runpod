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

## Security

- RunPod's port links are public addresses. Every page and API call requires the password; after 5 wrong tries from one address, sign-in pauses for 10 minutes. Sessions last 12 hours.
- ACE-Step's own API listens only inside the pod.
- The bridge refuses to listen publicly without a password hash.
- Change the password: run `python3 alexandria_bridge.py --set-password` anywhere, then put the new `alexandria_auth.json` contents into the `alexandria-auth` secret and restart the pod.

## Tuning

- Engine settings: `/workspace/ACE-Step-1.5/.env` (delete its first comment line to keep your own edits across boots).
- Renders: `/workspace/alexandria/renders/` (one folder per render with the job and the audio).
