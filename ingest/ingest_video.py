#!/usr/bin/env python3
"""Ingest a NACFE video/podcast into corpus/sources/<id>.md.

Per SPEC.md 2.2: pass the YouTube URL directly to Gemini as a video input, asking for a
transcript with [MM:SS] timestamps every ~30 seconds, speaker labels where distinguishable,
and descriptions of on-screen slides/charts (Bootcamp recordings carry most of their data
on slides -- a pure audio transcript throws that away).

This repo doesn't yet have a case of ingesting directly from a YouTube URL. What it has is
a manually copy-pasted auto-caption transcript (raw text, MM:SS markers, no speaker labels,
no slide descriptions) saved as a PDF. That's a strictly weaker input than the real pipeline
will have -- there is no slide/chart information to recover here, only text cleanup and
speaker attribution from context. This script handles that case: it does not call Gemini's
video understanding API (there is no video/audio file to give it), it asks a text model to
clean up and attribute the existing raw transcript.

Usage:
    python ingest_video.py <path/to/transcript.pdf> --id <source-id> --title "<video title>"
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = REPO_ROOT / "corpus" / "sources"
DEFAULT_MODEL = "gemini-3.5-flash"
# See ingest_pdf.py for why this exists: without it, a stalled connection hangs forever
# instead of raising, bypassing retry entirely.
REQUEST_TIMEOUT_MS = 180_000
RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+)s'")
RATE_LIMIT_MAX_WAIT = 65

CLEANUP_PROMPT = """You are cleaning up a raw auto-generated caption transcript of a NACFE
webinar/video, "{title}", for use as the source of record in a public question-answering
system.

The raw transcript below has run-on, uncapitalized, unpunctuated text with timestamp
markers (MM:SS or H:MM:SS) scattered through it marking where each caption chunk ended.

Rules:
1. Restore normal capitalization and punctuation. Do not change, add, remove, or
   paraphrase any word choice, fact, or claim -- fix only mechanics (case, punctuation,
   obvious transcription typos of proper nouns you can identify from context).
2. Insert a timestamp marker in the form [MM:SS] at the start of each paragraph/speaker
   turn, roughly every 20-40 seconds of content, using the nearest timestamp already
   present in the raw transcript. Do not invent timestamps not derivable from the source.
3. Attribute speaker turns by name where the transcript itself identifies who is talking
   (e.g. an introduction like "I'd like to turn it over to Mike" or "thanks Rachel, this is
   Dan Deppeler from Paper Transport"). Format as "**Name:** ..." Use "**Speaker (unclear):**"
   if a turn's speaker cannot be determined from context. Never guess an identity you
   cannot support from the transcript's own introductions.
4. Never add a fact that is not in the transcript. Never resolve a question the transcript
   leaves open. If a passage is garbled beyond repair, write [inaudible] rather than
   guessing.
5. This is a caption-only transcript -- no slide or on-screen content is available. Do not
   invent descriptions of slides, charts, or visuals.

RAW TRANSCRIPT:
{transcript}
"""


def extract_text(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() for page in reader.pages)
    return path.read_text()


def ingest(transcript_path, source_id, title, model, force=False):
    out_path = SOURCES_DIR / f"{source_id}.md"
    if out_path.exists() and not force:
        print(f"{out_path} already exists. Pass --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("GEMINI_INGEST_API") or os.getenv("GEMINI_API")
    if not api_key:
        print("Neither GEMINI_INGEST_API nor GEMINI_API is set (checked environment and .env).", file=sys.stderr)
        sys.exit(1)

    raw_text = extract_text(transcript_path)
    print(f"{Path(transcript_path).name}: {len(raw_text)} raw chars")

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    prompt = CLEANUP_PROMPT.format(title=title, transcript=raw_text)

    last_err = None
    response = None
    for attempt in range(1, 7):
        try:
            response = client.models.generate_content(model=model, contents=[prompt])
            break
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
            print(f"  {reason} (attempt {attempt}/6): {e}. Waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
    if response is None:
        raise last_err

    usage = response.usage_metadata
    print(f"  tokens: prompt={usage.prompt_token_count} output={usage.candidates_token_count} total={usage.total_token_count}")

    header = f"# {title}\n\n<!-- source: video transcript, cleaned from copy-pasted auto-captions -->\n\n"
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + response.text.strip() + "\n")
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript_path")
    parser.add_argument("--id", required=True, dest="source_id")
    parser.add_argument("--title", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    ingest(args.transcript_path, args.source_id, args.title, args.model, force=args.force)


if __name__ == "__main__":
    main()
