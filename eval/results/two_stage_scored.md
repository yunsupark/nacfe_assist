# Two-Stage Query Loop — Scored (30/30 complete)

Scored against `eval/questions.jsonl` using the same rubric as the NotebookLM baseline:
correct / partial / wrong / hallucinated.

**Note on the last 7 answers (Q23-27, 29, 30):** `gemini-3.5-flash`'s free-tier daily quota
got stuck and never reset on its documented midnight-PT schedule (confirmed by testing
hours past that point — still exhausted, while `gemini-3.5-flash-lite` and `gemini-3.6-flash`
worked fine on the same key). Rather than keep waiting on a quota that wasn't behaving as
documented, `run_eval.py`'s `ANSWER_MODEL` was switched to `gemini-3.6-flash` (newer,
non-lite, unblocked) to finish the run. The first 23 answers used `gemini-3.5-flash`; the
last 7 used `gemini-3.6-flash`. Worth knowing if comparing answer style/quality across the
full set, though no quality difference was apparent in scoring.

## Summary (30 of 30)

| Bucket | Correct | Partial | Wrong | Hallucinated |
|---|---|---|---|---|
| prose (9) | 8 | 1 | 0 | 0 |
| table (6) | 6 | 0 | 0 | 0 |
| figure (5) | 3 | 2 | 0 | 0 |
| out_of_scope (2) | 2 | 0 | 0 | 0 |
| synthesis (5) | 5 | 0 | 0 | 0 |
| recency (3) | 3 | 0 | 0 | 0 |
| **Total (30)** | **27** | **3** | **0** | **0** |

**27 correct, 3 partial, 0 wrong, 0 hallucinated — matches the NotebookLM baseline's exact
score (27/30, 3 partial, 0/0).** The two systems land in the same place by a different
route: NotebookLM's misses were a ranking slip and an unverifiable added specificity;
ours are two precision gaps that trace to the *ingest* prompt (not the query loop) and one
retrieval miss in a single-document answer. Every routing bug and stale-ground-truth issue
found during iteration is now confirmed fixed, not just theoretically patched.

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

### Q23 [prose] — CORRECT
65-85% BEV / 40-50% ICE / 30-50% H2FC efficiency — exact, matches Q9's white-paper source.

### Q24 [prose] — CORRECT
13,000+ trucks, 213 stations (25 public), 2,000+ technicians, 10,000+ drivers, $4B invested,
110M gallons displaced in 2024 — all exact, correctly attributed to Marty Tufte (transcribed
as "Tufty" — an artifact carried from the ingest step's speech-to-text, not a new error),
with accurate bonus detail (RNG facility counts, NOx-compliant truck count).

### Q25 [table] — CORRECT
7 facilities / ~40M DGE now, 17 under construction, 200M+ DGE by 2026, correctly adds the
-126 carbon intensity score as bonus detail.

### Q26 [prose] — CORRECT
54 trucks (1985) = 10,080 trucks (2018 standard) — exact.

### Q27 [prose] — CORRECT
10% of PTI's fleet, correct 15+ year CNG history and "second inning" framing. Bonus claim
("approaching 80 million miles, over 12 million gallons displaced") verified directly
against the source transcript at its cited timestamp (23:20) — real, not fabricated.

### Q28 [out_of_scope] — CORRECT (from prior run)
Correctly refuses for Session 2 specifically; hydrogen only mentioned in passing there.

### Q29 [synthesis] — CORRECT
Correctly says WM was not a tracked fleet, correctly distinguishes its Bootcamp-speaker role
from actual Run participation, correctly lists all 13 tracked fleets by powertrain.

### Q30 [table] — CORRECT
5,000-6,000 trucks at the 20-milligram NOx level vs. the referenced 0.035 threshold —
correct numbers, though it doesn't spell out the g/bhp-hr units on either figure the way
the expected answer does (a minor clarity gap, not a factual one). Correctly adds the
1985-vs-2018 emissions comparison as supporting context.

---

## Notes for the two-stage design going forward

1. **Router is now meaningfully better than the first pass.** Of the 5 questions that were
   broken in the very first run (Q8, 9, 11, 12, 21), all 5 landed correct or partial-with-a-
   real-value here, never wrong or unanswered. The 3 fixes (catalog-is-a-summary-not-the-
   source, figure-numbers-aren't-unique-across-sources, two-timepoints-need-two-sources)
   each solved a real, reproducible failure — not guesswork, and each was verified against
   the exact question that had been failing before being folded into the full run.
2. **Two precision gaps (Q11, Q12) trace back to the ingest prompt**, not the query loop —
   time-series charts get transcribed as a value range rather than a peak-plus-date. Worth
   revisiting `transcribe_page.txt` if this matters for real usage; flagged back when the
   reports were first ingested and still outstanding.
3. **One real answering-stage miss (Q2)**, isolated to a single question across all 30:
   claimed a document didn't state something it does state, in the same document it was
   given. Not a routing problem and didn't recur elsewhere, so likely a one-off rather than
   a systemic pattern — but worth watching if it shows up again as more questions are added.
4. **No hallucinations found across all 30**, including in the most numerically-dense
   answers (Q13, Q19, Q20, Q24) where fabrication would be easiest to miss — every
   surprising or unfamiliar number was spot-checked directly against source text and
   confirmed real, including two cases (Q18's "441 miles," Q27's "80 million miles") that
   weren't in the original hand-written eval and could easily have been mistaken for
   invented specificity if left unverified.
5. **`gemini-3.5-flash`'s free-tier daily quota not resetting as documented** is worth
   flagging to Google or just avoiding going forward — `gemini-3.6-flash` was a clean,
   same-tier substitute with no observed quality regression.
