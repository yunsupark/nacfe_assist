#!/usr/bin/env python3
"""Run the two-stage query loop (route -> answer) against eval/questions.jsonl.

Stage 1 (routing): full catalog + question -> cheap model -> selected source ids.
Stage 2 (answering): full markdown of selected sources + question -> stronger model -> answer.

Per SPEC.md sections 4, 5, and 8 step 7 ("build the two-stage query loop as a local
script, run the eval, iterate on prompts" -- before any Worker/cache/widget work).

Usage:
    python run_eval.py [--questions eval/questions.jsonl] [--limit N] [--only-bucket prose]
    python run_eval.py --fresh   # ignore any existing results, re-run everything

Keys: uses GEMINI_API_FREE (free tier) when it is set, otherwise GEMINI_API (paid).
Pass --paid to force the paid key even when a free key is present. There is deliberately
no silent free -> paid fallback here, unlike the Worker: an eval run that quietly starts
billing after the free quota runs out is the opposite of what a free-tier run is for.
Set one of them in the environment or a .env file discoverable from cwd.
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

# The Worker strips these five fields before showing the catalog to the router, and
# serializes compactly rather than pretty-printed (see ROUTER_CATALOG_JSON in
# worker/src/index.ts). They are serving-side bookkeeping the router never reasons over.
# Mirrored here so the eval measures the routing payload that actually ships -- this script
# is a parallel implementation of the Worker, and every place the two drift is a place the
# eval stops predicting production.
ROUTER_DROPPED_FIELDS = ("url", "media", "token_count", "ingested", "ingest_model")

JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
RETRY_DELAY_RE = re.compile(r"'retryDelay':\s*'(\d+)s'")
RATE_LIMIT_MAX_RETRIES = 6
RATE_LIMIT_MAX_WAIT = 65  # free-tier per-minute quotas; don't wait past that pointlessly


class DailyQuotaExhausted(Exception):
    """A per-day (not per-minute) quota was hit -- retrying won't help until tomorrow."""


def build_router_catalog_json(catalog):
    """Serialize the catalog byte-for-byte the way the Worker does.

    `separators` and `ensure_ascii=False` are what make this match JavaScript's
    JSON.stringify: Python would otherwise emit ", "/": " separators and \\uXXXX-escape every
    non-ASCII character, so the router would see a different (and larger) payload than
    production sends.
    """
    projected = [
        {k: v for k, v in entry.items() if k not in ROUTER_DROPPED_FIELDS} for entry in catalog
    ]
    return json.dumps(projected, separators=(",", ":"), ensure_ascii=False)


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


def has_valid_route(record):
    """Did stage 1 succeed for this record, whatever happened to stage 2?

    The two stages have separate free-tier quotas and fail independently -- the answering
    model's per-day request cap (20) runs out long before the routing model's, so a run
    typically ends with routing done for every question and answers missing for most of
    them. Without this, resume treats such a record as worthless and re-runs both stages,
    re-spending ~150K routing tokens per question to reach an answer call that was the only
    thing actually missing.
    """
    route = record.get("route_result")
    if not isinstance(route, dict) or route.get("error"):
        return False
    return isinstance(route.get("selected"), list)


def load_prior_routes(path):
    """question text -> prior record, for any record whose routing stage succeeded."""
    if not path.exists():
        return {}
    routes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if has_valid_route(r):
                routes[r["q"]] = r
    return routes


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
        # Title only, matching answer() in worker/src/index.ts -- the id in this header was
        # being cited instead of the title. Keep the two in sync.
        parts.append(f"=== SOURCE: {title} ===\n{text}")
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


def resolve_api_key(force_paid=False):
    """Free key by default, paid only when asked for (or when no free key exists)."""
    free_key = os.getenv("GEMINI_API_FREE")
    paid_key = os.getenv("GEMINI_API")

    if not force_paid and free_key:
        return free_key, "GEMINI_API_FREE (free tier)"
    if paid_key:
        if not force_paid and not free_key:
            print("note: GEMINI_API_FREE not set, falling back to the paid GEMINI_API key.")
        return paid_key, "GEMINI_API (paid)"
    if free_key:
        return free_key, "GEMINI_API_FREE (free tier)"

    print("Neither GEMINI_API_FREE nor GEMINI_API is set (checked environment and .env).",
          file=sys.stderr)
    sys.exit(1)


def run(questions_path, limit=None, only_bucket=None, fresh=False, paid=False, routing_only=False):
    api_key, key_label = resolve_api_key(force_paid=paid)
    print(f"using {key_label}")

    catalog = json.loads(CATALOG_PATH.read_text())
    catalog_by_id = {entry["id"]: entry for entry in catalog}
    catalog_json = build_router_catalog_json(catalog)
    print(f"router catalog: {len(catalog)} entries, {len(catalog_json):,} chars")

    all_questions = load_questions(questions_path)
    questions = all_questions
    if only_bucket:
        questions = [q for q in questions if q["bucket"] == only_bucket]
    if limit:
        questions = questions[:limit]
    if len(questions) != len(all_questions):
        print(f"running {len(questions)} of {len(all_questions)} questions "
              f"(results for the other {len(all_questions) - len(questions)} are preserved)")

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # A routing-only run writes to its own file and always re-routes. It exists to validate a
    # route.txt change cheaply: stage 1 runs on the routing model, which free tier throttles
    # by tokens-per-minute but did not cap by requests-per-day in practice, so a full 52-question
    # routing pass costs nothing and takes about an hour. Keeping it out of two_stage_raw.jsonl
    # matters -- re-routing invalidates the answers already scored against the old selections.
    out_path = RESULTS_DIR / ("two_stage_routing_only.jsonl" if routing_only else "two_stage_raw.jsonl")
    if routing_only:
        fresh = True
        print("routing-only: stage 2 skipped; writing to", out_path.name)
    prior = {} if fresh else load_prior_results(out_path)
    prior_routes = {} if fresh else load_prior_routes(out_path)
    if prior:
        print(f"resuming: reusing {len(prior)} prior valid answer(s) from {out_path}")
    reroutable = sum(1 for q in questions if q["q"] not in prior and q["q"] in prior_routes)
    if reroutable:
        print(f"resuming: reusing {reroutable} prior routing result(s); only the answer "
              f"stage will be re-run for those")

    # Records for questions this run isn't touching (--limit / --only-bucket), so a narrowed
    # run updates the results file in place instead of truncating it to just its own subset.
    running = {q["q"] for q in questions}
    by_question = {} if fresh else {k: v for k, v in prior.items() if k not in running}

    # Canonical order is questions.jsonl's, with any rows whose question is no longer in that
    # file kept at the end rather than silently dropped -- a reworded question leaves its old
    # answer stranded here, and losing it without a word would hide that it happened.
    canonical = [q["q"] for q in all_questions]
    orphans = [k for k in prior if k not in set(canonical)]
    if orphans and not fresh:
        print(f"note: {len(orphans)} result row(s) are for questions no longer in "
              f"{Path(questions_path).name}; keeping them, but they are not scored against "
              f"the current set:")
        for k in orphans:
            print(f"  - {k[:100]}...")
    order = canonical + orphans

    def flush():
        with open(out_path, "w") as out_f:
            for question_text in order:
                if question_text in by_question:
                    out_f.write(json.dumps(by_question[question_text]) + "\n")

    route_model_exhausted = False
    answer_model_exhausted = False
    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] [{q['bucket']}] {q['q'][:80]}...")

        if q["q"] in prior:
            print("    reusing prior result")
            results.append(prior[q["q"]])
            by_question[q["q"]] = prior[q["q"]]
            flush()
            continue

        prior_route = prior_routes.get(q["q"]) if not fresh else None
        if prior_route is not None:
            # Stage 1 already succeeded for this question on an earlier run; only the answer
            # is missing. Carry the original usage figures on the record so it still
            # describes what that question's full pipeline cost.
            route_result = prior_route["route_result"]
            route_usage = None
            print("    reusing prior routing")
        elif route_model_exhausted:
            # Free tier caps requests per day, not just per minute. Once routing has hit that
            # cap every remaining question would fail identically, so record them as skipped
            # (which is_valid_result treats as not-reusable) and let a later run pick them up.
            route_result, route_usage = {
                "selected": [], "out_of_scope": None, "recency_warning": None,
                "error": f"{ROUTE_MODEL} daily quota already exhausted this run",
            }, None
        else:
            try:
                route_result, route_usage = route(client, catalog_json, q["q"])
            except DailyQuotaExhausted as e:
                route_model_exhausted = True
                print(f"    routing daily quota exhausted; skipping the rest of this run", file=sys.stderr)
                route_result, route_usage = {
                    "selected": [], "out_of_scope": None, "recency_warning": None,
                    "error": f"{ROUTE_MODEL} daily quota exhausted: {e}",
                }, None
            except Exception as e:  # noqa: BLE001
                print(f"    routing failed: {e}", file=sys.stderr)
                route_result, route_usage = {"selected": [], "out_of_scope": None, "recency_warning": None, "error": str(e)}, None

        selected_ids = [s["id"] for s in route_result.get("selected", [])]
        print(f"    routed to: {selected_ids or '(none)'}" + (" [out_of_scope]" if route_result.get("out_of_scope") else ""))

        answer_text, answer_usage = None, None
        if routing_only:
            answer_text = "(routing-only run; no answer call made)"
        elif selected_ids and not route_result.get("out_of_scope"):
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
        elif route_result.get("error"):
            answer_text = f"[SKIPPED: routing unavailable: {route_result['error']}]"
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
            "route_usage": (
                usage_dict(route_usage) if route_usage
                else (prior_route.get("route_usage") if prior_route else None)
            ),
            "answer": answer_text,
            "answer_usage": usage_dict(answer_usage) if answer_usage else None,
        }
        results.append(record)
        by_question[q["q"]] = record
        flush()

    remaining = sum(1 for r in results if not is_reusable(r))
    print(f"\nwrote {out_path} ({len(results) - remaining}/{len(results)} answered this run, "
          f"{remaining} still missing; {len(by_question)} row(s) in file)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-bucket", default=None)
    parser.add_argument("--fresh", action="store_true", help="Ignore prior results, re-run everything")
    parser.add_argument("--paid", action="store_true",
                        help="Force the paid GEMINI_API key even if GEMINI_API_FREE is set")
    parser.add_argument("--routing-only", action="store_true",
                        help="Run stage 1 only, into eval/results/two_stage_routing_only.jsonl. "
                             "Cheap way to validate a route.txt change without spending the "
                             "answering model's daily quota or disturbing scored answers.")
    args = parser.parse_args()

    load_dotenv(find_dotenv(usecwd=True))
    run(args.questions, limit=args.limit, only_bucket=args.only_bucket, fresh=args.fresh,
        paid=args.paid, routing_only=args.routing_only)


if __name__ == "__main__":
    main()
