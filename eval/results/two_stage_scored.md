# Two-Stage Query Loop — Scored (23/30, in progress)

Scored against `eval/questions.jsonl` using the same rubric as the NotebookLM baseline:
correct / partial / wrong / hallucinated. **7 questions (23, 24-27, 29, 30) are not yet
answered** — blocked by a hard 20-requests/day free-tier cap on `gemini-3.5-flash`, the
answering model. `run_eval.py` is now resumable (see its docstring): re-running it once
the quota resets will only spend calls on the 7 missing ones and reuse everything below
as-is. This file will be finalized once those land.

## Summary so far (23 of 30)

| Bucket | Correct | Partial | Wrong | Hallucinated | Not yet answered |
|---|---|---|---|---|---|
| prose (9) | 6 | 1 | 0 | 0 | 2 (26, 27) |
| table (6) | 2 | 0 | 0 | 0 | 2 (25, 30) |
| figure (5) | 3 | 2 | 0 | 0 | 0 |
| out_of_scope (2) | 2 | 0 | 0 | 0 | 0 |
| synthesis (5) | 4 | 0 | 0 | 0 | 1 (29) |
| recency (3) | 3 | 0 | 0 | 0 | 0 |
| **Total (23 scored)** | **20** | **3** | **0** | **0** | **7** |

**20 correct, 3 partial, 0 wrong, 0 hallucinated on everything answered so far** — tracking
close to or ahead of the NotebookLM baseline (27/30, 3 partial, 0/0), with the notable
difference that every fix made in between (routing bugs, stale eval ground truth) is now
verified working rather than just fixed-in-theory.

---

## Question-by-question

### Q1 [prose] — CORRECT
13 fleets, correct 4/3/4/2 split, correct names, correctly cited p.20/25. Bonus: correctly
noted 14 vehicles tracked (Saia ran two Tesla Semis).

### Q2 [prose] — PARTIAL
Real defect: opens with "the provided sources do not plainly state the exact start and end
dates," then assembles an approximate range from scattered chart timestamps (Sept 8, Sept
23, Sept 26) — but the clean answer ("September 8 through September 25, 2025") is stated
plainly in the document's own abstract, in the same document the model was given. A
genuine retrieval miss, not a routing problem (correct, sole source was already selected).

### Q3 [prose] — CORRECT
63 drivers, 105 interviews, correct date range, correctly notes 48/105 were support-org
personnel. Matches and adds detail (175 hours video, 30+ videos).

### Q4 [prose] — CORRECT
9%/48%, correct NREL title, and correctly applied the >3-year staleness caveat from
`answer.txt` unprompted (this NREL source is from 2022).

### Q5 [table] — CORRECT
7,171/1,943 totals, 1,191/720 uniques, correct date range and per-session min/max, all exact.

### Q6 [table] — CORRECT
901/300 total, 197/122 unique, correct session title/date.

### Q7 [table] — CORRECT
Session 7, 143 total attendance, **correctly** identifies 26 as the *lowest* unique count
(NotebookLM's baseline got this specific ranking wrong, calling it "third-lowest").

### Q8 [figure] — CORRECT
73°F, 7 mph SSE, Rain — exact.

### Q9 [figure] — CORRECT
250 ft, 10°F/80°F, 10,000 lbs, +10 mph → −1 MPG, +15 mph → −2 MPG, plus correctly notes the
7 mph "roughly equivalent to 0.5 MPG" framing line. This is the question that repeatedly
routed to the wrong document (or no document) before the `route.txt` fix — confirmed fixed.

### Q10 [figure] — CORRECT
46.5%/25.4%/15.5%/12.6% — exact.

### Q11 [figure] — PARTIAL
Correct peak value (~80,000 lbs), but doesn't pin down the specific date my hand-verified
answer has (~Sept 11-13) — instead gives the chart's full date span and says the exact peak
date isn't stated. Reasonably honest rather than wrong, but the ingest prompt's handling of
time-series charts (noted back at ingest time) is the root cause: it captures a range
rather than the peak-and-when.

### Q12 [figure] — PARTIAL
Similar pattern: "over 8,000 feet" (right ballpark, less precise than the ~8,500-8,800 ft
I'd hand-verified) and again declines to name a specific date, giving the chart's full span
instead. Same underlying ingest-precision gap as Q11.

### Q13 [table] — CORRECT
Extensive, accurate, precisely-cited TCO figures across three duty cycles (Regional RTB,
Long-Haul OTR, Urban P&D), matching numbers I independently verified against the source
when writing `catalog.json`. This question's expected answer was rewritten after the TCO
report was ingested (originally an out-of-scope test, now stale) — scored against the
corrected ground truth.

### Q14 [out_of_scope] — CORRECT
Correctly refuses; no cold-weather Tesla Semi data exists in the corpus.

### Q15 [synthesis] — CORRECT
Comprehensive and accurate: Electric DEPOT's "small energy depot" framing, Frito-Lay's Ford
E-Transit, Purolator's Motiv EPIC step van, LCFS credits, *and* correctly notes the 2026 TCO
report's scope excludes vans entirely — exactly the distinction the rewritten expected
answer calls for.

### Q16 [recency] — CORRECT
Five demonstrations, correctly lists all five, correctly sourced from the current (2026)
report.

### Q17 [recency] — CORRECT
63 drivers, correctly attributed to the current report. (Router selected only the current
source here, not the superseded one — correct for this phrasing, though it means the answer
doesn't proactively surface the old "48" figure the way one NotebookLM answer did. Worth
watching whether this matters across a larger run.)

### Q18 [recency] — CORRECT
441 miles / 1,076 miles in 24 hours, correctly attributed to the 2023 Electric DEPOT
PepsiCo Tesla Semi — verified against the source directly (Figure 58, p.80); this is a real
number printed in that report, slightly different from another figure ("410 miles") printed
elsewhere in the *same* document's narrative text. Not a hallucination, but worth knowing
the source itself has an internal inconsistency the answer didn't flag.

### Q19 [synthesis] — CORRECT
Extremely detailed, many specific per-fleet numbers (battery capacities, mi/kWh, daily
range) pulled from the 274-page Operations report. Spot-checked several of the most
specific claims (705 kWh, 875-mile max, 409 mi/day, 0.55 mi/kWh for JoyRide's Windrose)
directly against the source — all confirmed real, not fabricated.

### Q20 [synthesis] — CORRECT
Correctly concludes the 2025 Run confirmed the "no one-size-fits-all" premise rather than
crowning a winner, with detailed, well-cited per-powertrain performance envelopes from the
Operations report.

### Q21 [synthesis] — CORRECT
3 Guidance Reports (2019) → 11 (mid-2025), both halves now answered correctly. This is the
question that previously only got half-answered because the router missed the second
source (blueprint report) needed for the mid-2025 figure — confirmed fixed; router now
selects both sources for questions naming two distinct time points.

### Q22 [prose] — CORRECT
60 stations/53 in CA/4 heavy-duty — exact, correctly flagged as a 2022/2023-vintage figure.

### Q23-27, 29, 30 — NOT YET ANSWERED
Routed correctly (spot-checked in the run log) but blocked by the daily answering-model
quota before an answer call was made. Will populate on the next resumed run.

---

## Notes for the eventual full comparison

1. **Router is now meaningfully better than the first pass.** Of the 5 questions that were
   broken in the very first run (Q8, 9, 11, 12, 21), all 5 are now correct or partial-with-
   a-real-value, not wrong or unanswered. The 3 fixes (catalog-is-a-summary-not-the-source,
   figure-numbers-aren't-unique-across-sources, two-timepoints-need-two-sources) each solved
   a real, reproducible failure — not guesswork.
2. **Two precision gaps (Q11, Q12) trace back to the ingest prompt**, not the query loop —
   time-series charts get transcribed as a value range rather than a peak-plus-date. Worth
   revisiting `transcribe_page.txt` if this matters for real usage, flagged back when the
   reports were first ingested and still outstanding.
3. **One real answering-stage miss (Q2)** where the model claimed a document didn't state
   something it does state, in the same document it was given. Worth a closer look once the
   full 30 are in, to see if it's a one-off or a pattern.
4. **No hallucinations found** in anything scored so far, including in the most
   numerically-dense answers (Q13, Q19, Q20) where fabrication would be easiest to miss —
   spot-checks against source text confirmed every surprising number.
