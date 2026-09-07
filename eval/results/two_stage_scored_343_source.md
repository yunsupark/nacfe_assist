# Two-Stage Query Loop — Scored (52/52, 343-source catalog, free tier)

Run 2026-09-04 to 2026-09-07 on `GEMINI_API_FREE`, after the security-review changes: the
`<question>` injection guard in `route.txt`/`answer.txt`, and the router catalog aligned to
what the Worker actually sends (compact JSON, `url`/`media`/`token_count`/`ingested`/
`ingest_model` stripped — byte-identical to `ROUTER_CATALOG_JSON` in `worker/src/index.ts`).

**This is the first eval against the 343-source catalog.** The widely-quoted 96% routing /
92% correct figures in `two_stage_scored_204_source.md` were measured on the 204-entry
catalog (2026-08-09); the corpus grew to 343 on 2026-08-14 (`951e16c`) and the eval was never
re-run, so the deployed configuration had no eval behind it until now. Catalog size and
prompt both changed, so the two runs are not a controlled comparison.

Spread across three days because free tier caps the answering model at 20 requests/day.
Whole set on one prompt version — no mid-run prompt or model changes.

## Routing accuracy: 48/52 (92%)

Same rubric as the 204-source run: the question's known source appears in
`route_result.selected`, or — where no correct source exists in the corpus — the selection is
empty.

| Bucket | Correct | Total |
|---|---|---|
| prose | 24 | 24 |
| table | 6 | 6 |
| figure | 7 | 7 |
| recency | 4 | 5 |
| synthesis | 4 | 6 |
| out_of_scope | 3 | 4 |
| **Total** | **48** | **52** |

204-source baseline: 50/52 (96%).

### Routing accuracy understates end-to-end quality

Two of the four routing "misses" produced **correct answers anyway** — the router chose a
different source that still contained the fact:

- *Longest Class 8 BEV range* — scored a miss (picked the 2024 DEPOT reports over the 2025
  blueprint) but returned the right answer: 410 miles on a single charge, 1,076 miles in 24
  hours, which is where that fact actually lives.
- *WM tracked with Geotab?* — scored a miss, answered "No" correctly with sound supporting
  reports.

So 48/52 is a floor on end-to-end correctness, not a ceiling.

The other two misses are existing `route.txt` rules failing to fire at 343 sources rather
than new failure modes: the superlative-across-campaigns rule, and the prefer-the-narrower-
companion-source rule. Both are prompt iterations, not redesigns.

## Answer quality: 43 correct / 6 partial / 0 wrong / 0 hallucinated

49 questions produced answers; 3 more were correctly routed out of scope and make no answer
call, which is the right behaviour. All 52 resolved.

| | Count |
|---|---|
| Correct | 43 |
| Partial | 6 |
| Wrong | 0 |
| **Hallucinated** | **0** |

88% correct (43/49 answered, or 46/52 counting the three correct refusals), against 92% on
the 204-source run. **Zero hallucinations, unchanged** — and every partial under-claims
rather than invents, which is the failure direction SPEC.md §0 asks for
("grounding over coverage").

### The six partials

1. **Demonstration date range** — said the report "does not state an explicit start and end
   date range", gave "18-day, three-week, concluded September 2025" instead of
   September 8–25. Other answers cite the Sept 8–26 chart window, so the figure was in
   context and simply was not committed to.
2. **Weight chart peak date** — peak ~80,000 lbs correct; gave the chart's full window
   instead of pinning September 11–13.
3. **Elevation chart** — "over 8,000 feet" against an expected 8,500–8,800 ft, and no date.
4. **2023 powertrain-decision framework vs 2025 results** — answered only the 2025 half,
   never retrieving the 2023 white paper's Figure 15 sequence.
5. **Guidance Report counts, 2019 vs mid-2025** — 2019 count (three) right, then correctly
   said the selected sources hold no mid-2025 figure.
6. **Windrose fleet identification** — declined to name JoyRide Logistics as the Windrose
   operator. Notably this is the question reworded since the 204-source run, so it had never
   been evaluated in its current form.

4 and 5 are routing shortfalls surfacing as answer gaps: the answer stage correctly refused
to invent what it had not been given. 1–3 are figure-reading questions where the model
declined to commit to a value it demonstrably had.

## Fixed during this run: internal source ids leaking into citations

5 of the 49 answers (10%) cited the internal catalog id instead of the report title:

> …a total of 901 people registered [`run-on-less-messy-middle-blueprint-2025`, p. 45]

Readers on nacfe.org would have seen raw slugs. `answer()` built each document header as
`=== SOURCE: {title} [{id}] ===` and `answer.txt` never ruled the id out. Each affected
answer committed to one convention throughout rather than mixing, so the model was choosing
a citation style per response.

Fixed by dropping the id from the header in **both** implementations (`worker/src/index.ts`
and `eval/run_eval.py` — verified byte-identical afterwards) and stating the rule explicitly
in `answer.txt`. All 343 titles are unique, so the title alone identifies a source. Verified
live on two of the five affected questions: both now cite "How NACFE Helped Bring Clarity to
the Messy Middle" with the figures unchanged.

Applied only after the full 52 were scored, so this run is not split across two prompts.

## Measured free-tier limits (2026-09-04 — 09-07)

| Limit | Model | Value |
|---|---|---|
| Input tokens per minute | `gemini-3.5-flash-lite` (routing) | 250,000 |
| Requests per day | `gemini-3.6-flash` (answering) | 20 |

At 150,778 tokens per routing call, one query consumes ~60% of the per-minute token budget,
capping free tier near 1.5 routing calls a minute. The 52-question routing pass took 51
minutes, almost entirely in backoff. `gemini-3.5-flash-lite` was also observed returning
503 on 5/5 free-key attempts while succeeding on the paid key — free-tier availability is
per model, which is what `gemini.ts`'s per-model skip flag now handles.

## Catalog size

Measured with the tokenizer, not estimated from bytes:

| Catalog variant | Bytes | Tokens |
|---|---|---|
| pretty-printed, all fields (before) | 781,631 | 224,206 |
| pretty-printed, projected | 702,575 | 189,295 |
| compact, all fields | 690,791 | 175,540 |
| **compact, projected (now shipping)** | **629,228** | **150,778** |

The alignment cut the routing payload 32.8%. Still over SPEC.md §3's ceiling ("around 40–60K
tokens ... if it exceeds ~100K, tighten the abstracts"), which remains the outstanding work —
`abstract` and `key_findings` are ~60% of the payload. At the §3 target, free tier would fit
4–6 routing calls a minute instead of 1.5.
