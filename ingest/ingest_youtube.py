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

# Long-form audio-only transcription is stochastic: the same audio + prompt can produce a
# clean full-length transcript on one sampling draw and, on another, a degenerate repetition
# loop that eats the entire 65536-token output ceiling for ~1 timestamp segment, or a
# transcript that just stops early. call_gemini()'s retry loop only catches exceptions, so a
# "successful" call can still silently return an unusably short transcript. Coverage-check
# against the real audio duration and retry with a fresh draw when it's too short.
COVERAGE_RETRY_THRESHOLD = 0.85
MAX_COVERAGE_ATTEMPTS = 3

# Raw timestamp-coverage math turned out not to be a reliable corruption signal on its own:
# the model sometimes drifts into an undocumented MM:SS:mmm timestamp format (misread as
# H:MM:SS, producing nonsense coverage%) while the actual content is fine, and conversely can
# produce a transcript whose *last* timestamp looks complete while the middle is a degenerate
# repetition loop. These catch the real failure modes coverage% misses or falsely flags.
OUTRO_RE = re.compile(r"freight efficiency with nacfe", re.IGNORECASE)
# Real generation loops repeat a short phrase dozens+ times; normal human stuttering
# ("the the the", "that that that") rarely exceeds ~5-6 repeats, so a much higher bound
# avoids flagging genuine, prompt-requested verbatim disfluency as corruption.
WITHIN_LINE_REPEAT_RE = re.compile(r"\b(\S+(?:\s+\S+){0,2})\b(?:\s+\1\b){14,}", re.IGNORECASE)


def repeated_block(lines, min_block=3, max_block=30, min_repeats=3):
    """Detect a run of min_block..max_block consecutive lines that repeats verbatim
    min_repeats+ times back to back -- a degenerate generation loop that cycles through a
    whole paragraph (often with incrementing fabricated timestamps between cycles) rather
    than looping a single word or line. Single-line/within-line repeat checks miss this
    entirely since no two *consecutive* lines are identical, only the whole block recurs.
    """
    n = len(lines)
    for block_len in range(min_block, max_block + 1):
        i = 0
        while i + block_len * min_repeats <= n:
            block = lines[i:i + block_len]
            repeats = 1
            j = i + block_len
            while j + block_len <= n and lines[j:j + block_len] == block:
                repeats += 1
                j += block_len
            if repeats >= min_repeats:
                return f"block_len={block_len} repeats={repeats}: {block[0][:50]!r}"
            i += 1
    return None


def nonconsecutive_duplicate_fraction(lines, min_len=40):
    """Fraction of the transcript's substantial lines that are exact duplicates of an earlier
    line *anywhere* in the file, not just immediately adjacent. A real, continuous conversation
    essentially never produces the same 40+ character sentence twice verbatim; this catches
    windowed/overlapping-regeneration artifacts where a chunk of content reappears at a
    non-adjacent point, which the consecutive dup_lines/repeated_block checks structurally
    cannot see (no two *consecutive* lines are identical in that failure mode).
    """
    from collections import Counter
    substantial = [l for l in lines if len(l) >= min_len and "freight efficiency with nacfe" not in l.lower()]
    if not substantial:
        return 0.0
    counts = Counter(substantial)
    wasted = sum(c - 1 for c in counts.values() if c >= 2)
    return wasted / len(lines) if lines else 0.0


def transcript_quality_issues(text, dup_fraction_threshold=0.15):
    """Content-based corruption signals: exact-repeated lines, a repeated multi-line block
    (a paragraph-scale generation loop), a phrase looping many times within one line, a
    meaningful fraction of non-adjacent duplicate sentences, or a transcript that neither
    reaches the show's real outro nor ends on a complete sentence (a sign of a truncated/cut-off
    response). Returns a list of issue strings; empty means no problems found.
    """
    lines = [l for l in text.split("\n") if l.strip() and not l.strip().startswith("<!--")]
    if not lines:
        return ["no content"]
    issues = []
    dup_lines = sum(1 for i in range(1, len(lines)) if lines[i] == lines[i - 1] and len(lines[i]) > 10)
    if dup_lines > 3:
        issues.append(f"dup_lines={dup_lines}")
    block_issue = repeated_block(lines)
    if block_issue:
        issues.append(f"repeated_block={block_issue}")
    within_match = next((m.group(0)[:60] for l in lines if (m := WITHIN_LINE_REPEAT_RE.search(l))), None)
    if within_match:
        issues.append(f"within_line_repeat={within_match!r}")
    dup_frac = nonconsecutive_duplicate_fraction(lines)
    if dup_frac >= dup_fraction_threshold:
        issues.append(f"nonconsecutive_duplicate_fraction={dup_frac:.0%}")
    has_outro = bool(OUTRO_RE.search(text[-500:]))
    ends_clean = lines[-1].strip().endswith((".", "!", "?", '"', ")"))
    if not has_outro and not ends_clean:
        issues.append("no_outro_no_terminal_punct")
    return issues


def normalize_youtube_url(url_or_id):
    """Accept a bare video ID, youtu.be link, or full watch URL; return a canonical watch URL."""
    if url_or_id.startswith("http://") or url_or_id.startswith("https://"):
        return url_or_id
    return f"https://www.youtube.com/watch?v={url_or_id}"


def timestamp_to_seconds(ts, expected_duration=None):
    """Parse MM:SS or H:MM:SS (the two forms the transcribe_video.txt prompt uses)."""
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 3:
        h, m, s = parts
        as_hms = h * 3600 + m * 60 + s
        # Long-form transcription sometimes drifts into an undocumented MM:SS:mmm format
        # instead of the prompted H:MM:SS once past a certain point -- both are 3-colon
        # triples, so "26:30:00" is ambiguous between 26h30m0s and 26m30s+000ms. When we
        # know the real duration, prefer whichever reading is actually plausible for it.
        as_min_sec_ms = h * 60 + m
        if expected_duration and as_hms > expected_duration * 1.5 and as_min_sec_ms <= expected_duration * 1.5:
            return as_min_sec_ms
        return as_hms
    return None


def audio_duration_seconds(path):
    """Real duration of a local audio file, or None if it can't be determined."""
    try:
        from mutagen import File as MutagenFile

        f = MutagenFile(path)
        if f is not None and f.info is not None:
            return f.info.length
    except Exception:
        pass
    return None


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
    """media_part may be None for a text-only prompt (e.g. reformatting existing text rather
    than transcribing audio/video)."""
    contents = [media_part, prompt] if media_part is not None else [prompt]
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
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

    api_key = os.getenv("GEMINI_INGEST_API") or os.getenv("GEMINI_API")
    if not api_key:
        print("Neither GEMINI_INGEST_API nor GEMINI_API is set (checked environment and .env).", file=sys.stderr)
        sys.exit(1)

    youtube_url = normalize_youtube_url(url_or_id)
    base_prompt = PROMPT_PATH.read_text()
    prompt = base_prompt.format(title=title)

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))

    expected_duration = None
    if audio_file:
        print(f"{youtube_url}: uploading {audio_file} and requesting transcript (audio-only fallback), model={model}")
        uploaded = upload_and_wait(client, audio_file)
        media_part = uploaded
        source_ref = youtube_url
        note = f"audio-only ingest (Files API), downloaded from {youtube_url}"
        expected_duration = audio_duration_seconds(audio_file)
    else:
        print(f"{youtube_url}: requesting transcript, model={model}")
        media_part = types.Part(file_data=types.FileData(file_uri=youtube_url))
        source_ref = youtube_url
        note = f"YouTube video, {youtube_url}"

    attempts = MAX_COVERAGE_ATTEMPTS if expected_duration else 1
    best_response, best_entry, best_coverage, best_issues = None, None, -1.0, None
    for attempt in range(1, attempts + 1):
        response = call_gemini(client, model, prompt, media_part)
        entry = log_call(source_id, source_ref, model, response.usage_metadata)
        print(f"  tokens: prompt={entry['prompt_tokens']} output={entry['output_tokens']} total={entry['total_tokens']}")

        timestamps = TIMESTAMP_ANCHOR_RE.findall(response.text)
        print(f"  {len(timestamps)} timestamp segments found")

        if not expected_duration:
            best_response, best_entry = response, entry
            break

        last_seconds = timestamp_to_seconds(timestamps[-1], expected_duration) if timestamps else 0
        coverage = (last_seconds or 0) / expected_duration
        issues = transcript_quality_issues(response.text)
        print(f"  coverage: {coverage:.0%} (last timestamp {timestamps[-1] if timestamps else 'none'} of {expected_duration:.0f}s)")
        if issues:
            print(f"  quality issues: {issues}")

        # Rank primarily by fewest content-quality issues (coverage% alone is an unreliable
        # signal -- it's fooled by undocumented timestamp-format drift and can't see a
        # mid-transcript repetition loop that still reaches a plausible-looking last timestamp).
        is_better = (
            best_issues is None
            or len(issues) < len(best_issues)
            or (len(issues) == len(best_issues) and coverage > best_coverage)
        )
        if is_better:
            best_response, best_entry, best_coverage, best_issues = response, entry, coverage, issues
        if not issues and coverage >= COVERAGE_RETRY_THRESHOLD:
            break
        if attempt < attempts:
            print(f"  not clean yet, retrying with a fresh draw (attempt {attempt + 1}/{attempts})...")

    response, entry = best_response, best_entry
    if expected_duration and (best_issues or best_coverage < COVERAGE_RETRY_THRESHOLD):
        summary = ", ".join(best_issues) if best_issues else f"coverage only {best_coverage:.0%}"
        print(f"  WARNING: best of {attempts} attempts still has issues ({summary}) -- flagging for manual review", file=sys.stderr)
        note += f" [WARNING: transcript quality issues ({summary}) after {attempts} attempts -- needs manual review]"

    if out_path.exists():
        existing_issues = transcript_quality_issues(out_path.read_text())
        new_issues = best_issues if best_issues is not None else transcript_quality_issues(response.text)
        if len(existing_issues) <= len(new_issues):
            print(f"  existing file has {len(existing_issues)} quality issue(s) {existing_issues}, this run's best has {len(new_issues)} {new_issues} -- keeping existing file, not overwriting", file=sys.stderr)
            return out_path, entry

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
