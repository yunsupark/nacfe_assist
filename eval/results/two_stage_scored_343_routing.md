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

## Answer quality: 27 correct / 5 partial / 0 wrong / 0 hallucinated (32 of 52 answered)

Free tier caps `gemini-3.6-flash` at **20 generate requests per day**
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), so the set is being completed across
days. 35 of 52 questions are resolved: 32 answered, plus 3 the router correctly sent
out-of-scope (which make no answer call). 17 remain.

| | Count |
|---|---|
| Correct | 27 |
| Partial | 5 |
| Wrong | 0 |
| **Hallucinated** | **0** |

84% correct of those answered, against 92% for the 204-source run — but that run scored all
52, and this is 32 of 52, so the two are not yet comparable. What is comparable is the
hallucination count, which stays at zero: **every one of the five partials under-claims
rather than invents.**

### The five partials

1. **Demonstration date range** — said the report "does not state an explicit start and end
   date range", gave "18-day, three-week, concluded September 2025" instead of
   September 8–25. Other answers cite the Sept 8–26 chart window, so the figure was in the
   context and was not committed to.
2. **Weight chart peak date** — peak ~80,000 lbs correct; gave the chart's full window
   rather than pinning September 11–13.
3. **Elevation chart** — "over 8,000 feet" against an expected 8,500–8,800 ft, and no date.
4. **2023 powertrain-decision framework vs 2025 results** — answered only the 2025 half;
   never retrieved the 2023 white paper's Figure 15 sequence (BEV → hydrogen → natural
   gas/hybrid → efficient diesel). A cross-vintage question of exactly the kind `route.txt`
   has a rule for.
5. **Guidance Report counts, 2019 vs mid-2025** — got the 2019 count (three) right, then
   said "the provided sources do not contain data for mid-2025". Correct given what routing
   selected; the mid-2025 figure lives in a source the router did not pick.

Partials 4 and 5 are routing shortfalls surfacing as answer gaps, not answer-stage failures:
the answer stage correctly refused to invent what it had not been given.

### Routing accuracy understates end-to-end quality

Two of the four routing "misses" produced **correct answers anyway** — the router chose a
different source that still contained the fact:

- *Longest Class 8 BEV range* — scored a miss (picked the 2024 DEPOT reports over the 2025
  blueprint) but returned the right answer, 410 miles on a single charge and 1,076 miles in
  24 hours, which is where that fact actually lives.
- *WM tracked with Geotab?* — scored a miss, answered "No" correctly with the right
  supporting reports.

So 48/52 routing is a floor on end-to-end correctness, not a ceiling on it.

### Defect found: internal source ids leaking into citations

**5 of 32 answers (16%) cite the internal catalog id instead of the report title**, e.g.

> …a total of 901 people registered [`run-on-less-messy-middle-blueprint-2025`, p. 45]

A reader on nacfe.org would see raw slugs. `answer()` in `worker/src/index.ts` builds each
document header as `=== SOURCE: {title} [{id}] ===`, and `answer.txt` says only "Cite every
claim. PDFs: [Title, p. 14]" without ruling the id out. Notably these five never mix the two
forms — an answer commits to the id or to the title for its whole response, so the model is
picking a convention per answer rather than slipping occasionally.

All 343 catalog titles are unique, so removing `[{id}]` from the header is safe. Not applied
during this run: `run_eval.py` re-reads `answer.txt` on every call, so changing it mid-set
would leave the run scored against two different prompts — the same methodological problem
already noted in `two_stage_scored.md`, where answers 1–23 and 24–30 used different models.

## Finishing the set

`python3 eval/run_eval.py` (no `--fresh`) on a later day. Resume reuses the 52 routing
results and the 32 answers, spending the day's 20-request budget only on the 17 missing —
one more day completes it. Or `--paid` to finish in a single pass.

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
