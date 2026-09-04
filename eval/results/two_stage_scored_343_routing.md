# Two-Stage Query Loop — Routing Scored (343-source catalog, free tier)

Run 2026-09-04 on `GEMINI_API_FREE`, after the security-review changes: the `<question>`
injection guard in `route.txt`/`answer.txt`, and the router catalog aligned to what the
Worker actually sends (compact JSON, `url`/`media`/`token_count`/`ingested`/`ingest_model`
stripped — byte-identical to `ROUTER_CATALOG_JSON` in `worker/src/index.ts`).

**This is the first eval against the 343-source catalog.** The headline 96% routing / 92%
correct figures in `two_stage_scored_204_source.md` were measured on the 204-entry catalog
(run 2026-08-09); the corpus grew to 343 on 2026-08-14 (`951e16c`) and the eval was never
re-run. So the numbers below are not directly comparable — catalog size changed as well as
the prompt.

## Routing accuracy: 48/52 (92%)

Same rubric as the 204-source run: the question's known source appears in
`route_result.selected`, or — where no correct source exists in the corpus — the selection
is empty.

| Bucket | Correct | Total |
|---|---|---|
| prose | 24 | 24 |
| table | 6 | 6 |
| figure | 7 | 7 |
| recency | 4 | 5 |
| synthesis | 4 | 6 |
| out_of_scope | 3 | 4 |
| **Total** | **48** | **52** |

204-source baseline: 50/52 (96%). Two questions' difference on n=52 is inside the noise, and
the catalog grew 68% between the runs, so this does not isolate a prompt regression.

### The four misses

All four are near-miss topical confusion — the router picked a plausible sibling report from
the same programme, not a refusal or an off-topic pick. Nothing suggests the injection guard
caused a wrong rejection.

1. `[out_of_scope]` Tesla Semi comparative performance — expected
   `run-on-less-messy-middle-blueprint-2025`, selected nothing.
2. `[synthesis]` Last-mile electric TCO — expected `run-on-less-electric-depot-2024`,
   selected `confidence-report-vans-step-vans-2022`, `run-on-less-electric-2021`.
3. `[recency]` Longest Class 8 BEV range — expected
   `run-on-less-messy-middle-blueprint-2025`, selected the two 2024 DEPOT reports. This is
   the superlative-across-campaigns case `route.txt` already has a rule for; the rule did
   not fire against the larger catalog.
4. `[synthesis]` WM / Geotab fleet tracking — expected `messy-middle-bootcamp-session-2-2025`,
   selected the two campaign-wide Messy Middle reports instead of the session-specific one.
   Also a case `route.txt` has an explicit rule for (prefer the narrower companion source).

Both 3 and 4 are existing `route.txt` rules failing to fire at 343 sources rather than new
failure modes — worth a prompt iteration, not a redesign.

## Answer quality: NOT SCORED (13 of 52 answered)

Free tier caps `gemini-3.6-flash` at **20 generate requests per day**
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). The run consumed that budget after 13
answers (plus 3 questions the router correctly sent out-of-scope, which make no answer call),
and the remaining 36 were recorded as `[SKIPPED: ... daily quota exhausted]`. Routing
completed for all 52, so nothing needs re-routing.

To finish the answer stage: re-run `python3 eval/run_eval.py` (no `--fresh`) on subsequent
days — resume reuses the 52 routing results and the 13 answers, spending the daily budget
only on what is missing — or run once on the paid key.

## Measured free-tier limits (2026-09-04)

| Limit | Model | Value |
|---|---|---|
| Input tokens per minute | `gemini-3.5-flash-lite` (routing) | 250,000 |
| Requests per day | `gemini-3.6-flash` (answering) | 20 |

At 150,778 tokens per routing call, one query consumes ~60% of the per-minute token budget,
capping free tier at roughly 1.5 routing calls a minute. The full 52-question routing pass
took 51 minutes, almost entirely in backoff.

## Catalog size

Measured with the tokenizer, not estimated from bytes:

| Catalog variant | Bytes | Tokens |
|---|---|---|
| pretty-printed, all fields (before) | 781,631 | 224,206 |
| pretty-printed, projected | 702,575 | 189,295 |
| compact, all fields | 690,791 | 175,540 |
| **compact, projected (now shipping)** | **629,228** | **150,778** |

The alignment cut the routing payload 32.8%. It is still over SPEC.md §3's stated ceiling
("should land around 40–60K tokens ... if it exceeds ~100K, tighten the abstracts"), which
remains the outstanding work — `abstract` and `key_findings` are ~60% of the payload.
