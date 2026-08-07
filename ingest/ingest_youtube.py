#!/usr/bin/env python3
"""Ingest a NACFE YouTube video into corpus/sources/<id>.md.

Per SPEC.md 2.2: pass the YouTube URL directly to Gemini as a video input, asking for a
transcript with [MM:SS] timestamps every ~30 seconds, speaker labels where distinguishable,
and descriptions of on-screen slides/charts (Bootcamp recordings carry most of their data
on slides -- a pure audio transcript throws that away). This is the real pipeline that
ingest_video.py's docstring describes but doesn't implement (it only handles a manually
copy-pasted caption-only transcript, a strictly weaker fallback input).

Usage:
    python ingest_youtube.py <youtube-url-or-video-id> --id <source-id> --title "<video title>"

Requires GEMINI_API set in the environment or a .env file discoverable from cwd.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "transcribe_video.txt"
SOURCES_DIR = REPO_ROOT / "corpus" / "sources"
LOG_PATH = Path(__file__).resolve().parent / "ingest_log.jsonl"

DEFAULT_MODEL = "gemini-3.5-flash-lite"
# See ingest_pdf.py for why this exists: without it, a stalled connection hangs forever
# instead of raising, bypassing retry entirely. Video calls run longer than PDF calls, so
# the budget is more generous.
REQUEST_TIMEOUT_MS = 600_000
RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+)s'")
RATE_LIMIT_MAX_WAIT = 65

TIMESTAMP_ANCHOR_RE = re.compile(r"<!--\s*timestamp:\s*([\d:]+)\s*-->")


def normalize_youtube_url(url_or_id):
    """Accept a bare video ID, youtu.be link, or full watch URL; return a canonical watch URL."""
    if url_or_id.startswith("http://") or url_or_id.startswith("https://"):
        return url_or_id
    return f"https://www.youtube.com/watch?v={url_or_id}"


def upload_and_wait(client, audio_path):
    """Upload a local audio file via the Files API and block until it's ACTIVE.

    Per SPEC.md 2.2: "For non-YouTube audio, upload via the Files API and use the same
    prompt." Also useful as a fallback for a YouTube video that hits a persistent
    RECITATION block -- audio-only input carries none of the on-screen visual content
    that seems to trigger it, at the cost of losing slide/chart descriptions.
    """
    uploaded = client.files.upload(file=str(audio_path))
    while uploaded.state.name == "PROCESSING":
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"file upload ended in state {uploaded.state.name}: {uploaded}")
    return uploaded


def call_gemini(client, model, prompt, media_part, retries=6):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[media_part, prompt],
            )
            if response.text is None:
                # See ingest_pdf.py -- an empty response with no exception is usually a
                # content-based block (recitation/safety), not a transient failure. Surface
                # the reason and retry; a different sampling draw sometimes gets through.
                finish_reason = None
                if response.candidates:
                    finish_reason = getattr(response.candidates[0], "finish_reason", None)
                raise ValueError(f"empty response from model (finish_reason={finish_reason})")
            return response
        except Exception as e:  # noqa: BLE001 - real network/API errors, log and retry
            last_err = e
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            m = RETRY_DELAY_RE.search(str(e))
            if m:
                wait = min(int(m.group(1)) + 2, RATE_LIMIT_MAX_WAIT)
            elif is_rate_limit:
                wait = min(15 * attempt, RATE_LIMIT_MAX_WAIT)
            else:
                wait = min(2 ** attempt, 30)
            reason = "rate limit" if is_rate_limit else "error"
            print(f"  {reason} (attempt {attempt}/{retries}): {e}. Waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise last_err


def log_call(source_id, source_ref, model, usage):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "youtube_url": source_ref,
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def ingest(url_or_id, source_id, title, model, force=False, audio_file=None):
    out_path = SOURCES_DIR / f"{source_id}.md"
    if out_path.exists() and not force:
        print(f"{out_path} already exists. Pass --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("GEMINI_API")
    if not api_key:
        print("GEMINI_API not set (checked environment and .env).", file=sys.stderr)
        sys.exit(1)

    youtube_url = normalize_youtube_url(url_or_id)
    base_prompt = PROMPT_PATH.read_text()
    prompt = base_prompt.format(title=title)

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))

    if audio_file:
        print(f"{youtube_url}: uploading {audio_file} and requesting transcript (audio-only fallback), model={model}")
        uploaded = upload_and_wait(client, audio_file)
        media_part = uploaded
        source_ref = youtube_url
        note = f"audio-only fallback (video RECITATION-blocked), downloaded from {youtube_url}"
    else:
        print(f"{youtube_url}: requesting transcript, model={model}")
        media_part = types.Part(file_data=types.FileData(file_uri=youtube_url))
        source_ref = youtube_url
        note = f"YouTube video, {youtube_url}"

    response = call_gemini(client, model, prompt, media_part)
    entry = log_call(source_id, source_ref, model, response.usage_metadata)
    print(f"  tokens: prompt={entry['prompt_tokens']} output={entry['output_tokens']} total={entry['total_tokens']}")

    n_segments = len(TIMESTAMP_ANCHOR_RE.findall(response.text))
    print(f"  {n_segments} timestamp segments found")

    header = f"# {title}\n\n<!-- source: {note} -->\n\n"
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + response.text.strip() + "\n")
    print(f"  wrote {out_path}")
    return out_path, entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url_or_id", help="YouTube video ID or full URL")
    parser.add_argument("--id", required=True, dest="source_id")
    parser.add_argument("--title", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--audio-file",
        default=None,
        help=(
            "Path to a locally-downloaded audio file to upload via the Files API instead of "
            "referencing the YouTube URL directly. Fallback for a video that hits a "
            "persistent RECITATION block, or for non-YouTube audio per SPEC.md 2.2."
        ),
    )
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    ingest(args.url_or_id, args.source_id, args.title, args.model, force=args.force, audio_file=args.audio_file)


if __name__ == "__main__":
    main()
