#!/usr/bin/env python3
"""Is the eval runnable on the free-tier key right now?

Both stages have to work on GEMINI_API_FREE for `run_eval.py` (which deliberately has no
free -> paid fallback) to complete a run. Free-tier availability is per model: measured
2026-09-04, the routing model returned 503 "high demand" on every free-key attempt while the
answering model on the same key, and both models on the paid key, were fine.

Costs a few dozen tokens per model. Exits 0 when both stages are free-tier runnable, 1 when
not, so it can drive a wait loop:

    until python3 eval/check_free_tier.py; do sleep 900; done && \
        python3 eval/run_eval.py --fresh
"""
import os
import sys

from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import ANSWER_MODEL, ROUTE_MODEL  # noqa: E402  keep the models in one place

PROBE = "Reply with the single word: ok"


def probe(client, model):
    try:
        client.models.generate_content(model=model, contents=[PROBE])
        return True, "available"
    except Exception as e:  # noqa: BLE001 - any failure means "not usable right now"
        text = str(e)
        for code, label in (("429", "rate limited / quota"), ("503", "unavailable (high demand)"),
                            ("500", "server error")):
            if code in text:
                return False, f"{code} {label}"
        return False, text.split("\n")[0][:80]


def main():
    load_dotenv(find_dotenv(usecwd=True))
    key = os.getenv("GEMINI_API_FREE")
    if not key:
        print("GEMINI_API_FREE is not set; nothing to check.", file=sys.stderr)
        return 1

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60_000))
    results = {}
    for stage, model in (("routing", ROUTE_MODEL), ("answering", ANSWER_MODEL)):
        ok, detail = probe(client, model)
        results[stage] = ok
        print(f"  {'OK  ' if ok else 'DOWN'}  {stage:10s} {model:24s} {detail}")

    if all(results.values()):
        print("\nfree tier is ready: python3 eval/run_eval.py --fresh")
        return 0
    print("\nnot runnable on the free key yet.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
