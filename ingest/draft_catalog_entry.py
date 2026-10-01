#!/usr/bin/env python3
"""Drafts a first-pass catalog entry for a freshly-ingested source, for a human to review.

Run after ingest_youtube.py / ingest_pdf.py has already produced corpus/sources/<id>.md.
Writes corpus/pending/<id>.json -- deliberately NOT corpus/catalog.json -- so a draft can never
be mistaken for a reviewed, live entry. The admin console's "Add source" flow (see
.github/workflows/ingest-source.yml) runs this as its second step and opens a PR containing
both files; a human moves the reviewed fields into corpus/catalog.json by hand and deletes the
pending file as part of that PR review, same manual step as every other entry in this catalog
has always gone through (see SPEC.md 3: "supersedes/superseded_by/status must be curated by a
human, not inferred by the model" -- this script doesn't even ask the model to guess those).

Usage:
    python draft_catalog_entry.py <source-id>

Requires GEMINI_INGEST_API or GEMINI_API set (environment or .env), same as the other ingest
scripts.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from google import genai

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_youtube import call_gemini, log_call  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = REPO_ROOT / "corpus" / "sources"
PENDING_DIR = REPO_ROOT / "corpus" / "pending"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "draft_catalog_entry.txt"

DEFAULT_MODEL = "gemini-3.6-flash"
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Fields this script never proposes -- see the module docstring and SPEC.md 3. Left null/empty
# in the draft so a reviewer is never tempted to rubber-stamp a guessed value for exactly the
# fields that most need their own judgment.
HUMAN_ONLY_FIELDS = {"status": "current", "supersedes": None, "superseded_by": None}


def draft(client, model, source_id, document_text):
    prompt = PROMPT_PATH.read_text().replace("{{document}}", document_text)
    response = call_gemini(client, model, prompt, media_part=None)
    log_call(source_id, source_id, model, response.usage_metadata)
    cleaned = JSON_FENCE_RE.sub("", response.text.strip())
    fields = json.loads(cleaned)
    return {
        "id": source_id,
        **fields,
        **HUMAN_ONLY_FIELDS,
        "_draft": True,
        "_draft_model": model,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_id")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    import os

    api_key = os.getenv("GEMINI_INGEST_API") or os.getenv("GEMINI_API")
    if not api_key:
        print("Neither GEMINI_INGEST_API nor GEMINI_API is set (checked environment and .env).", file=sys.stderr)
        sys.exit(1)

    source_path = SOURCES_DIR / f"{args.source_id}.md"
    if not source_path.exists():
        print(f"{source_path} does not exist -- run the ingest script first.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    entry = draft(client, args.model, args.source_id, source_path.read_text())

    PENDING_DIR.mkdir(exist_ok=True)
    out_path = PENDING_DIR / f"{args.source_id}.json"
    out_path.write_text(json.dumps(entry, indent=2) + "\n")
    print(f"wrote {out_path} -- AI-drafted, review before moving into corpus/catalog.json")


if __name__ == "__main__":
    main()
