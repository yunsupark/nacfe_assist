#!/usr/bin/env python3
"""Run the two-stage query loop (route -> answer) against eval/questions.jsonl.

Stage 1 (routing): full catalog + question -> cheap model -> selected source ids.
Stage 2 (answering): full markdown of selected sources + question -> stronger model -> answer.

Per SPEC.md sections 4, 5, and 8 step 7 ("build the two-stage query loop as a local
script, run the eval, iterate on prompts" -- before any Worker/cache/widget work).

Usage:
    python run_eval.py [--questions eval/questions.jsonl] [--limit N] [--only-bucket prose]
    python run_eval.py --fresh   # ignore any existing results, re-run everything

Requires GEMINI_API set in the environment or a .env file discoverable from cwd.
Writes raw results to eval/results/two_stage_raw.jsonl.

Resumable by default: if that file already exists, questions with a prior valid answer
(not an error, not a daily-quota skip) are reused rather than re-queried, so a free-tier
per-day request cap doesn't force re-spending quota on questions that already succeeded.
Pass --fresh to ignore prior results and re-run everything from scratch.

Per-day quota errors (distinct from per-minute rate limits, which retry with backoff as
normal) are NOT retried -- once one model's daily cap is hit, every subsequent call to that
model would fail identically, so remaining questions needing that model are marked
"skipped: daily quota exhausted" rather than burning retries proving that again and again.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import os
from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "corpus" / "catalog.json"
SOURCES_DIR = REPO_ROOT / "corpus" / "sources"
ROUTE_PROMPT_PATH = REPO_ROOT / "worker" / "prompts" / "route.txt"
ANSWER_PROMPT_PATH = REPO_ROOT / "worker" / "prompts" / "answer.txt"
DEFAULT_QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

ROUTE_MODEL = "gemini-3.5-flash-lite"
ANSWER_MODEL = "gemini-3.6-flash"  # gemini-3.5-flash's free-tier daily quota got stuck (not
# resetting on the documented midnight-PT schedule) -- confirmed both gemini-3.5-flash-lite
# and gemini-3.6-flash work fine on the same key while 3.5-flash stays exhausted, so this
# switches the answering stage off the stuck model rather than keep waiting on it.
CURRENT_YEAR = "2026"
# See ingest_pdf.py for why this exists: without it, a stalled connection hangs forever
# instead of raising, bypassing retry entirely.
REQUEST_TIMEOUT_MS = 180_000

JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+)s'")
RATE_LIMIT_MAX_RETRIES = 6
RATE_LIMIT_MAX_WAIT = 65  # free-tier per-minute quotas; don't wait past that pointlessly


class DailyQuotaExhausted(Exception):
    """A per-day (not per-minute) quota was hit -- retrying won't help until tomorrow."""


def load_questions(path):
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_prior_results(path):
    """question text -> prior record, for records worth reusing (valid answer or legit no-op)."""
    if not path.exists():
        return {}
    prior = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if is_reusable(r):
                prior[r["q"]] = r
    return prior


def is_reusable(record):
    a = record.get("answer")
    if not a:
        return False
    if a.startswith("[ERROR") or a.startswith("[SKIPPED"):
        return False
    return True  # includes legit "(router flagged out_of_scope...)" / "no sources" answers


def call_model(client, model, prompt, retries=RATE_LIMIT_MAX_RETRIES):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return client.models.generate_content(model=model, contents=[prompt])
        except Exception as e:  # noqa: BLE001 - real network/API errors, log and retry
            last_err = e
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            is_daily = "PerDay" in str(e) or "GenerateRequestsPerDayPerProjectPerModel" in str(e)
            if is_daily:
                # Retrying a per-day cap is pointless within the same day -- fail fast so the
                # caller can skip the rest of this model's work instead of burning retries.
                print(f"    daily quota exhausted for {model}, not retrying: {e}", file=sys.stderr)
                raise DailyQuotaExhausted(str(e)) from e
            m = RETRY_DELAY_RE.search(str(e))
            if m:
                wait = min(int(m.group(1)) + 2, RATE_LIMIT_MAX_WAIT)  # server-suggested + buffer
            elif is_rate_limit:
                wait = min(15 * attempt, RATE_LIMIT_MAX_WAIT)  # per-minute quota
            else:
                wait = min(2 ** attempt, 30)
            reason = "rate limit" if is_rate_limit else "error"
            print(f"    {reason} (attempt {attempt}/{retries}): {e}. Waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise last_err


def parse_route_response(text):
    cleaned = JSON_FENCE_RE.sub("", text.strip())
    return json.loads(cleaned)


def route(client, catalog_json, question):
    prompt = (
        ROUTE_PROMPT_PATH.read_text()
        .replace("{{catalog}}", catalog_json)
        .replace("{{question}}", question)
    )
    response = call_model(client, ROUTE_MODEL, prompt)
    parsed = parse_route_response(response.text)
    return parsed, response.usage_metadata


def build_documents_block(selected_ids, catalog_by_id):
    parts = []
    for source_id in selected_ids:
        entry = catalog_by_id.get(source_id)
        title = entry["title"] if entry else source_id
        md_path = SOURCES_DIR / f"{source_id}.md"
        if not md_path.exists():
            print(f"    warning: {md_path} not found, skipping", file=sys.stderr)
            continue
        text = md_path.read_text()
        parts.append(f"=== SOURCE: {title} [{source_id}] ===\n{text}")
    return "\n\n".join(parts)


def answer(client, documents_block, question):
    prompt = (
        ANSWER_PROMPT_PATH.read_text()
        .replace("{{current_year}}", CURRENT_YEAR)
        .replace("{{documents}}", documents_block)
        .replace("{{question}}", question)
    )
    response = call_model(client, ANSWER_MODEL, prompt)
    return response.text, response.usage_metadata


def usage_dict(usage):
    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    }


def run(questions_path, limit=None, only_bucket=None, fresh=False):
    api_key = os.getenv("GEMINI_API")
    if not api_key:
        print("GEMINI_API not set (checked environment and .env).", file=sys.stderr)
        sys.exit(1)

    catalog = json.loads(CATALOG_PATH.read_text())
    catalog_by_id = {entry["id"]: entry for entry in catalog}
    catalog_json = json.dumps(catalog, indent=2)

    questions = load_questions(questions_path)
    if only_bucket:
        questions = [q for q in questions if q["bucket"] == only_bucket]
    if limit:
        questions = questions[:limit]

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "two_stage_raw.jsonl"
    prior = {} if fresh else load_prior_results(out_path)
    if prior:
        print(f"resuming: reusing {len(prior)} prior valid answer(s) from {out_path}")

    answer_model_exhausted = False
    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] [{q['bucket']}] {q['q'][:80]}...")

        if q["q"] in prior:
            print("    reusing prior result")
            results.append(prior[q["q"]])
            continue

        try:
            route_result, route_usage = route(client, catalog_json, q["q"])
        except Exception as e:  # noqa: BLE001
            print(f"    routing failed: {e}", file=sys.stderr)
            route_result, route_usage = {"selected": [], "out_of_scope": None, "recency_warning": None, "error": str(e)}, None

        selected_ids = [s["id"] for s in route_result.get("selected", [])]
        print(f"    routed to: {selected_ids or '(none)'}" + (" [out_of_scope]" if route_result.get("out_of_scope") else ""))

        answer_text, answer_usage = None, None
        if selected_ids and not route_result.get("out_of_scope"):
            if answer_model_exhausted:
                answer_text = f"[SKIPPED: {ANSWER_MODEL} daily quota already exhausted this run]"
            else:
                documents_block = build_documents_block(selected_ids, catalog_by_id)
                try:
                    answer_text, answer_usage = answer(client, documents_block, q["q"])
                except DailyQuotaExhausted as e:
                    answer_model_exhausted = True
                    answer_text = f"[SKIPPED: {ANSWER_MODEL} daily quota exhausted: {e}]"
                except Exception as e:  # noqa: BLE001
                    print(f"    answering failed: {e}", file=sys.stderr)
                    answer_text = f"[ERROR: {e}]"
        elif route_result.get("out_of_scope"):
            answer_text = "(router flagged out_of_scope; no answer call made)"
        else:
            answer_text = "(router selected no sources; no answer call made)"

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "q": q["q"],
            "bucket": q["bucket"],
            "expected": q["expect"],
            "expected_source": q.get("source"),
            "expected_loc": q.get("loc"),
            "route_result": route_result,
            "route_usage": usage_dict(route_usage) if route_usage else None,
            "answer": answer_text,
            "answer_usage": usage_dict(answer_usage) if answer_usage else None,
        }
        results.append(record)

        with open(out_path, "w") as out_f:
            for r in results:
                out_f.write(json.dumps(r) + "\n")

    remaining = sum(1 for r in results if not is_reusable(r))
    print(f"\nwrote {out_path} ({len(results) - remaining}/{len(results)} answered, {remaining} still missing)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-bucket", default=None)
    parser.add_argument("--fresh", action="store_true", help="Ignore prior results, re-run everything")
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    run(args.questions, limit=args.limit, only_bucket=args.only_bucket, fresh=args.fresh)


if __name__ == "__main__":
    main()
