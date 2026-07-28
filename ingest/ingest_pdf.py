#!/usr/bin/env python3
"""Ingest a NACFE PDF report into corpus/sources/<id>.md.

Per SPEC.md 2.1: send the whole PDF to Gemini in one call under ~60 pages;
above that, use 40-page windows with 2 pages of overlap so tables and figures
that straddle a window boundary still get full context in at least one call.

Usage:
    python ingest_pdf.py <path/to/report.pdf> --id <source-id>

Requires GEMINI_API set in the environment or in a .env file discoverable
from the current working directory (see python-dotenv's find_dotenv).
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
from pypdf import PdfReader, PdfWriter

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "transcribe_page.txt"
SOURCES_DIR = REPO_ROOT / "corpus" / "sources"
LOG_PATH = Path(__file__).resolve().parent / "ingest_log.jsonl"

DEFAULT_MODEL = "gemini-3.5-flash-lite"
SINGLE_CALL_PAGE_LIMIT = 60
WINDOW_SIZE = 40
WINDOW_OVERLAP = 2

PAGE_ANCHOR_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*(\(unnumbered\))?\s*-->")


def page_windows(total_pages, window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP):
    """Yield (start, end) 1-indexed inclusive page ranges covering the document."""
    if total_pages <= SINGLE_CALL_PAGE_LIMIT:
        yield (1, total_pages)
        return
    start = 1
    while True:
        end = min(start + window_size - 1, total_pages)
        yield (start, end)
        if end == total_pages:
            return
        start = end - overlap + 1


def extract_window_bytes(reader, start, end):
    """Return PDF bytes for 1-indexed inclusive page range [start, end]."""
    writer = PdfWriter()
    for i in range(start - 1, end):
        writer.add_page(reader.pages[i])
    from io import BytesIO

    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def build_prompt(base_prompt, start, end, total_pages, windowed):
    if not windowed:
        return base_prompt
    return (
        base_prompt
        + f"\n\nThis is a partial excerpt of a larger {total_pages}-page document. "
        f"It contains pages {start} through {end} of that document. If a page has no "
        f"printed page number, use its position in the *full* document (i.e. starting "
        f"the count for this excerpt's first page at {start}), not its position within "
        f"this excerpt."
    )


def call_gemini(client, model, prompt, pdf_bytes, retries=3):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    prompt,
                ],
            )
            return response
        except Exception as e:  # noqa: BLE001 - real network/API errors, log and retry
            last_err = e
            wait = 2 ** attempt
            print(f"  call failed (attempt {attempt}/{retries}): {e}. Retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise last_err


def split_into_page_blocks(md_text):
    """Split transcribed markdown into (page_num, unnumbered, block_text) tuples."""
    matches = list(PAGE_ANCHOR_RE.finditer(md_text))
    blocks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        page_num = int(m.group(1))
        unnumbered = bool(m.group(2))
        blocks.append((page_num, unnumbered, md_text[start:end].strip()))
    return blocks


def log_call(source_id, start, end, model, usage, window_idx):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "window": window_idx,
        "pages": f"{start}-{end}",
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def ingest(pdf_path, source_id, model, published=None, force=False):
    pdf_path = Path(pdf_path)
    out_path = SOURCES_DIR / f"{source_id}.md"
    if out_path.exists() and not force:
        print(f"{out_path} already exists. Pass --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("GEMINI_API")
    if not api_key:
        print("GEMINI_API not set (checked environment and .env).", file=sys.stderr)
        sys.exit(1)

    base_prompt = PROMPT_PATH.read_text()
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    windowed = total_pages > SINGLE_CALL_PAGE_LIMIT

    client = genai.Client(api_key=api_key)

    windows = list(page_windows(total_pages))
    print(f"{pdf_path.name}: {total_pages} pages -> {len(windows)} call(s), model={model}")

    all_window_blocks = []
    call_log = []
    for idx, (start, end) in enumerate(windows):
        print(f"  window {idx + 1}/{len(windows)}: pages {start}-{end}")
        pdf_bytes = extract_window_bytes(reader, start, end)
        prompt = build_prompt(base_prompt, start, end, total_pages, windowed)
        response = call_gemini(client, model, prompt, pdf_bytes)
        entry = log_call(source_id, start, end, model, response.usage_metadata, idx)
        call_log.append(entry)
        print(f"    tokens: prompt={entry['prompt_tokens']} output={entry['output_tokens']} total={entry['total_tokens']}")
        all_window_blocks.append(split_into_page_blocks(response.text))

    merged = {}
    order = []
    overwritten = 0
    for window_idx, blocks in enumerate(all_window_blocks):
        for page_num, unnumbered, text in blocks:
            key = page_num if not unnumbered else f"u{window_idx}_{page_num}"
            if key in merged:
                overwritten += 1
            else:
                order.append(key)
            merged[key] = text  # later window's version wins on overlap

    final_md = "\n\n".join(merged[k] for k in order) + "\n"

    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(final_md)

    total_tokens = sum(e["total_tokens"] or 0 for e in call_log)
    print(f"  wrote {out_path} ({len(order)} pages, {overwritten} overlap pages resolved, {total_tokens} total tokens)")
    return out_path, call_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", help="Path to the source PDF")
    parser.add_argument("--id", required=True, dest="source_id", help="Source id, e.g. run-on-less-messy-middle-blueprint-2025")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true", help="Overwrite existing corpus/sources/<id>.md")
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    ingest(args.pdf_path, args.source_id, args.model, force=args.force)


if __name__ == "__main__":
    main()
