#!/usr/bin/env python3
"""Alexandria engine self-test. Runs inside the pod when ALEX_SELFTEST=1 and prints to the pod log.

Checks: disk space, GPU memory, the open model (download, spec quality, speed), the Claude fallback
(with prompt-cache usage), and whether ACE-Step can render while the open model is loaded.
Uses the studio's real spec instructions (SPEC_SYSTEM in alexandria.html) and the bridge's own model calls.
"""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import alexandria_bridge as b  # noqa: E402

T0 = time.time()
WS = "/workspace" if os.path.isdir("/workspace") else "/"


def say(*a):
    print(f"SELFTEST [{int(time.time() - T0):>5}s]", *a, flush=True)


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"({e})"


def gpu():
    return sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")


def system_prompt():
    html = (HERE / "alexandria.html").read_text(encoding="utf-8")
    m = re.search(r"const SPEC_SYSTEM = `(.*?)`;", html, re.S)
    if not m:
        raise SystemExit("SPEC_SYSTEM not found in alexandria.html")
    return m.group(1)


REQUIRED = ["title", "concept", "genre", "bpm", "key", "structure", "lyrics", "style_pre_vocal", "vocal_descriptors",
            "style_post_vocal", "beat_blueprint", "yue_tags", "levers"]
S, N = {"type": "string"}, {"type": "number"}
SCHEMA = {"type": "object", "required": REQUIRED + ["directions", "time_feel", "palette", "groove", "arrangement", "mix", "weirdness", "style_influence"],
          "properties": {**{k: S for k in ("title", "concept", "genre", "key", "time_feel", "palette", "groove", "arrangement", "mix", "lyrics",
                                           "style_pre_vocal", "vocal_descriptors", "style_post_vocal", "yue_tags")},
                         "bpm": N, "weirdness": N, "style_influence": N, "levers": {"type": "array", "items": S},
                         "directions": {"type": "array", "items": {"type": "object", "required": ["name", "summary"], "properties": {"name": S, "summary": S}}},
                         "beat_blueprint": {"type": "object", "required": ["kick", "snare", "hats", "low_end", "percussion", "dropout"],
                                            "properties": {k: S for k in ("kick", "snare", "hats", "low_end", "percussion", "dropout")}},
                         "structure": {"type": "array", "items": {"type": "object", "required": ["section", "bars", "energy", "notes"],
                                                                  "properties": {"section": S, "bars": N, "energy": N, "notes": S}}}}}
SAMPLE_LYRIC = """--- "Porch Light" ---
[Verse 1]
Mama left the porch light burning like a promise on a nail,
every moth that hit the window carried home another tale.
[Chorus]
Leave it on, leave it on, I'm still a long way from the door,
leave it on, leave it on, I ain't the boy I was before."""


def user_prompt(brief, budget=900, with_lyrics=False):
    p = (f"BRIEF: {brief}\nWORKING TITLE: (invent one)\nCATEGORY: Song\nSTRUCTURE: Choose the section form that fits the brief.\n"
         "VOCAL: Write your own vocal descriptors in vocal_descriptors (delivery, register, phrasing, harmonies).\n"
         f"CHARACTER BUDGET: style_pre_vocal + vocal_descriptors + style_post_vocal must total at most {budget} characters including spaces.\n")
    if with_lyrics:
        p += ("\nTHE ARTIST'S OWN PAST LYRICS (study and match this writer's voice: vocabulary, imagery, rhyme habits, meter, "
              "point of view, emotional register. Never copy or reuse their lines; write new ones in their voice):\n"
              + SAMPLE_LYRIC + "\n--- end of examples ---\n")
    return p


def check(label, text, secs, budget=900):
    try:
        v = json.loads(b._strip_json(text))
    except ValueError as e:
        say(f"{label}: FAIL not JSON ({e}); first 200 chars: {text[:200]!r}")
        return False
    missing = [k for k in REQUIRED if k not in v]
    bpm = float(v.get("bpm") or 0)
    bars = [int(s.get("bars") or 0) for s in v.get("structure") or []]
    runtime = sum(bars) * 240 / bpm if bpm else 0
    style = ", ".join(x for x in (v.get("style_pre_vocal"), v.get("vocal_descriptors"), v.get("style_post_vocal")) if x)
    lyr = v.get("lyrics") or ""
    tag_issues = [w for w, bad in (("starts with [Intro]", lyr.lstrip().lower().startswith("[intro")), ("uses [Hook]", "[hook" in lyr.lower())) if bad]
    ok = not missing and 185 <= runtime <= 215 and len(style) <= budget and not tag_issues
    say(f"{label}: {'PASS' if ok else 'CHECK'} in {secs:.0f}s | title={v.get('title')!r} genre={v.get('genre')!r} bpm={bpm:g} key={v.get('key')!r}")
    say(f"{label}:   sections={len(bars)} bars={bars} runtime={runtime:.0f}s (target 185-215) style={len(style)}/{budget} chars"
        + (f" missing={missing}" if missing else "") + (f" tags: {', '.join(tag_issues)}" if tag_issues else ""))
    say(f"{label}:   style: {style[:400]}")
    for line in [l for l in lyr.splitlines() if l.strip()][:8]:
        say(f"{label}:   | {line}")
    return ok


def ace_render(seconds=30):
    base = os.environ.get("ACE_URL", "http://127.0.0.1:18011")
    body = {"prompt": "warm Afro-R&B, 98 BPM, log drum, Rhodes, breathy alto", "lyrics": "[Verse 1]\nLamp on the stair, salt on the glass\n[Chorus]\nI keep the light on",
            "audio_duration": float(seconds), "audio_format": "flac", "batch_size": 1, "thinking": True, "vocal_language": "en",
            "task_type": "text2music", "time_signature": "4", "bpm": 98}
    t = time.time()
    res = b.http_json(base + "/release_task", body, timeout=120)
    tid = ((res or {}).get("data") or {}).get("task_id")
    peak = gpu()
    if not tid:
        return f"FAIL no task id: {str(res)[:200]}"
    while time.time() - t < 900:
        time.sleep(3)
        peak = max(peak, gpu())
        items = (b.http_json(base + "/query_result", {"task_id_list": [tid]}) or {}).get("data") or []
        st = items[0].get("status") if items else 0
        if st == 1:
            return f"PASS rendered {seconds}s of audio in {time.time() - t:.0f}s; GPU during render: {peak}"
        if st == 2:
            return f"FAIL ACE-Step reported a failure: {str(items[0])[:300]}"
    return "FAIL timed out after 15 minutes"


def main():
    say("=== Alexandria self-test ===")
    du = shutil.disk_usage(WS)
    say(f"volume: {du.used / 1e9:.0f} GB used, {du.free / 1e9:.0f} GB free of {du.total / 1e9:.0f} GB")
    say(f"GPU memory at start: {gpu()}")
    say("folder sizes:", sh("du -sh /workspace/ollama /workspace/hf-cache /workspace/ACE-Step-1.5 /workspace/.uv 2>/dev/null | tr '\\n' ' '"))
    sysprompt = system_prompt()
    say(f"spec system prompt: {len(sysprompt)} chars")

    # wait for the open model
    deadline, last = time.time() + 60 * 60, 0
    while True:
        st = b.llm_status()
        if st["local"]["ready"]:
            say(f"open model {st['local']['model']} ready")
            break
        if time.time() > deadline:
            say("FAIL open model not ready after 60 minutes;", sh("tail -c 300 /workspace/alexandria/logs/ollama-pull.log"))
            break
        if time.time() - last > 120:
            last = time.time()
            pull = sh("tail -c 400 /workspace/alexandria/logs/ollama-pull.log | tr '\\r' '\\n' | grep -v '^$' | tail -1")
            say(f"waiting for the open model (server up={st['local']['up']}): {pull[-160:]}")
        time.sleep(20)

    briefs = [("local-1", "late-night Afro-R&B about a lighthouse keeper waiting on a ship that's months overdue; hopeful, warm, slow-burning", False),
              ("local-2", "gospel-trap anthem about coming home after years away; triumphant choir on the chorus", True)]
    if b.llm_status()["local"]["ready"]:
        for label, brief, lyr in briefs:
            t = time.time()
            try:
                text = b.ask_local(user_prompt(brief, with_lyrics=lyr), SCHEMA, True, 6000, 0.7, sysprompt)
                check(label, text, time.time() - t)
            except Exception as e:  # noqa: BLE001
                say(f"{label}: FAIL {e}")
            say(f"GPU memory after {label}: {gpu()}")

        say("ACE-Step render with the open model still loaded:")
        try:
            say("ace:", ace_render(30))
        except Exception as e:  # noqa: BLE001
            say(f"ace: FAIL {e}")
        say(f"GPU memory after render: {gpu()}")

    if b.claude_key():
        for label, brief, lyr in [("claude-1", briefs[0][1], False), ("claude-2", briefs[1][1], True)]:
            t = time.time()
            try:
                text, usage = b.ask_claude(user_prompt(brief, with_lyrics=lyr), True, 6000, 0.7, sysprompt)
                check(label, text, time.time() - t)
                say(f"{label}: cache read={usage['cache_read_input_tokens']} cache write={usage['cache_creation_input_tokens']} "
                    f"uncached in={usage['input_tokens']} out={usage['output_tokens']}")
            except Exception as e:  # noqa: BLE001
                say(f"{label}: FAIL {e}")
    else:
        say("claude: SKIP no Anthropic key on this pod")
    du = shutil.disk_usage(WS)
    say(f"volume at end: {du.free / 1e9:.0f} GB free")
    say("=== self-test finished ===")


if __name__ == "__main__":
    main()
