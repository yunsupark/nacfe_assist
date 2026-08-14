#!/usr/bin/env python3
"""Standard ingest path for NACFE podcast episodes ("Freight Efficiency with Mike Roeth and
Friends"): reformat YouTube's auto-generated captions into our house transcript style, rather
than transcribing the raw audio directly.

Why: direct audio transcription (ingest_youtube.py --audio-file) turned out to be badly prone
to a degenerate repetition loop on long, unstructured, visual-free audio -- comparing this
corpus's 140 video-ingested sources (avg 1.6% duplicated content, real slides/visual content
seem to anchor the model) against the first pass of 139 podcast episodes (many with 40-90%+
duplicated content, no visual anchor at all) confirmed this is a structural weak point of
long-form audio-only generation, not bad luck on a few files. Reformatting real, ground-truth
YouTube captions as a bounded *text* task sidesteps the failure mode entirely: a controlled
test batch came back at 0% duplication across the board.

Usage:
    python ingest_podcast_captions.py <source-id> [<source-id> ...]
    python ingest_podcast_captions.py --ids-file some_ids.json

Looks up video_id/title for each source-id in podcast_episodes.json. Requires GEMINI_INGEST_API
or GEMINI_API set (environment or .env).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from google import genai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_youtube import call_gemini, transcript_quality_issues, timestamp_to_seconds, TIMESTAMP_ANCHOR_RE  # noqa: E402
from parse_vtt import parse_vtt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = REPO_ROOT / "corpus" / "sources"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "reformat_captions.txt"
EPISODES_PATH = Path(__file__).resolve().parent / "podcast_episodes.json"

MAX_ATTEMPTS = 3


def fetch_captions(video_id, workdir):
    out_template = str(Path(workdir) / "cap")
    subprocess.run(
        ["yt-dlp", "--write-auto-sub", "--sub-lang", "en", "--skip-download",
         "--sub-format", "vtt", "-o", out_template, "--", video_id],
        capture_output=True, text=True, check=True,
    )
    vtt_files = list(Path(workdir).glob("cap*.vtt"))
    if not vtt_files:
        raise RuntimeError(f"no caption file produced for {video_id}")
    return vtt_files[0]


def ingest_one(client, base_prompt, source_id, video_id, title, force=False):
    out_path = SOURCES_DIR / f"{source_id}.md"
    print(f"\n=== {source_id} <- {video_id} ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        vtt_path = fetch_captions(video_id, tmpdir)
        segments = parse_vtt(str(vtt_path))
        if not segments:
            print("  NO CAPTION SEGMENTS PARSED, skipping", file=sys.stderr)
            return

    raw_lines = []
    for s, t in segments:
        mins, secs = divmod(int(s), 60)
        raw_lines.append(f"[{mins:02d}:{secs:02d}] {t}")
    raw_captions = "\n".join(raw_lines)
    last_caption_seconds = segments[-1][0]
    print(f"  parsed {len(segments)} caption segments, last at {last_caption_seconds:.0f}s")

    prompt = base_prompt.format(title=title, captions=raw_captions)

    best_text, best_issues, best_coverage = None, None, -1.0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = call_gemini(client, "gemini-3.5-flash-lite", prompt, None)
        text = response.text
        issues = transcript_quality_issues(text)
        timestamps = TIMESTAMP_ANCHOR_RE.findall(text)
        last_seconds = timestamp_to_seconds(timestamps[-1]) if timestamps else 0
        coverage = (last_seconds or 0) / last_caption_seconds if last_caption_seconds else 1.0
        print(f"  attempt {attempt}: {len(text)} chars, coverage={coverage:.0%}, issues={issues if issues else 'none'}")

        is_better = (
            best_issues is None
            or len(issues) < len(best_issues)
            or (len(issues) == len(best_issues) and coverage > best_coverage)
        )
        if is_better:
            best_text, best_issues, best_coverage = text, issues, coverage
        if not issues and coverage >= 0.85:
            break

    note = f"reformatted from YouTube auto-captions (standard podcast ingest path), source video {video_id}"
    if best_issues or best_coverage < 0.85:
        summary = ", ".join(best_issues) if best_issues else f"coverage only {best_coverage:.0%}"
        print(f"  WARNING: best attempt still has issues ({summary})", file=sys.stderr)
        note += f" [WARNING: quality issues after caption-based reformat: {summary} -- needs manual review]"

    if out_path.exists() and not force:
        existing_issues = transcript_quality_issues(out_path.read_text())
        if len(existing_issues) <= len(best_issues):
            print(f"  existing file has {len(existing_issues)} issue(s), new best has {len(best_issues)} -- keeping existing", file=sys.stderr)
            return

    header = f"# {title}\n\n<!-- source: {note} -->\n\n"
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + best_text.strip() + "\n")
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", help="source-id(s) to ingest")
    parser.add_argument("--ids-file", default=None, help="JSON file containing a list of source-ids")
    args = parser.parse_args()

    ids = list(args.ids)
    if args.ids_file:
        ids.extend(json.load(open(args.ids_file)))
    if not ids:
        print("no source-ids given (pass as args or --ids-file)", file=sys.stderr)
        sys.exit(1)

    episodes = {e["id"]: e for e in json.load(open(EPISODES_PATH))}
    unknown = [i for i in ids if i not in episodes]
    if unknown:
        print(f"unknown source-id(s) not in {EPISODES_PATH}: {unknown}", file=sys.stderr)
        sys.exit(1)

    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.getenv("GEMINI_INGEST_API") or os.getenv("GEMINI_API")
    if not api_key:
        print("Neither GEMINI_INGEST_API nor GEMINI_API is set.", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)
    base_prompt = PROMPT_PATH.read_text()

    for sid in ids:
        e = episodes[sid]
        ingest_one(client, base_prompt, sid, e["video_id"], e["title"])


if __name__ == "__main__":
    main()
