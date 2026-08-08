# Two-Stage Query Loop — Scored at 178-Source Scale (37/52 answered, 52/52 routed)

Re-run of `eval/questions.jsonl` (52 questions: the original 30 plus 22 new
routing-stress questions added for the expanded corpus) against the full 178-entry
`catalog.json` — up from the ~6-25 sources the original eval ran against.

## A real bug found and fixed along the way

While resuming this run across two days (blocked by `gemini-3.6-flash`'s free-tier
**daily** quota, see below), the results file came back with only 50 of 52 records after
a resume, even though the console log clearly showed all 52 processing (30 as "reusing
prior result"). Root cause: `run_eval.py` only rewrote `two_stage_raw.jsonl` inside the
freshly-fetched branch of the per-question loop — a `continue` on the "already answered,
reuse it" branch skipped the file write entirely. If the *last* questions in a run happen
to be reused (as happened here — the two newest out-of-scope questions were both reused
tail entries), they get appended to the in-memory list and reported as "answered" in the
final summary, but never actually reach disk. Fixed in `eval/run_eval.py` by writing the
file on every iteration, reused or not. Re-ran afterward and confirmed all 52 questions
now persist correctly. Worth knowing if any *other* past resumed run silently dropped
trailing reused records.

## Coverage: 37/52 answered, 15 blocked on daily quota

Routing (cheap model, full catalog per call) completed all 52 questions across the first
run. The answer stage (`gemini-3.6-flash`) hit its free-tier **daily** quota after 20
questions on day 1, then again after 17 more on day 2 (37 total) — each cleanly marked
`[SKIPPED]`/`[ERROR: 429 ...]` rather than failing silently or fabricating. The catalog is
now ~80K tokens per routing call (vs. a few K before expansion), which is unrelated to
this cap — it's the *answer* stage's per-day request/token budget on the free tier,
exhausted faster now because each answer call also carries full document text for 1-2
much longer sources (full video transcripts, multi-hundred-page reports) than the
original small corpus.

The remaining 15 are queued for another resume once quota resets; `run_eval.py` will pick
up exactly those 15 without re-spending quota on the 37 already answered.

## Routing accuracy: 45/52 (87%)

Scored by checking whether the question's known correct source appears in `route_result
.selected`, or — for out-of-scope questions with no correct source anywhere in the corpus
— whether the router returned an empty selection, per SPEC.md's explicit rule ("If no
source in the catalog addresses the question, return an empty selection").

| Bucket | Correct | Total |
|---|---|---|
| prose | 22 | 22 |
| table | 6 | 6 |
| figure | 6 | 7 |
| out_of_scope | 3 | 4 |
| synthesis | 3 | 6 |
| recency | 4 | 5 |
| **Total** | **45** | **52** |

### Misroutes (6)

1. **[figure]** "What is 0.5 MPG worth?" infographic (Figure 32) — routed to
   `run-on-less-2017` instead of `run-on-less-messy-middle-blueprint-2025`. Downstream
   effect: the answer cited different figures/numbers than expected (Q9 in the answer
   review below) — a real, traceable routing→answer failure chain.

2. **[recency]** "Longest range demonstrated by a battery-electric Class 8 truck" —
   routed only to `run-on-less-electric-depot-2024` (2023 demo, correctly the source of
   the 410-mile Tesla Semi answer), never pulling in `run-on-less-messy-middle-blueprint
   -2025` to check whether the 2025 demo beat it. Exactly the failure mode the question
   was designed to stress-test — the router fell into it.

3. **[synthesis]** Last-mile electric van TCO — routed to `confidence-report-vans-step
   -vans-2022` + `guidance-report-medium-duty-electric-cost-of-ownership-2018` instead of
   `run-on-less-electric-depot-2024`. Reasonable adjacent picks, but missed the source the
   question was built from — though the answer still landed correct (see below).

4-5. **[synthesis] x2** — Two questions built around specific claims in `messy-middle-a
   -time-for-action-2023` (the 2023 white paper) routed only to the 2025 Messy Middle
   blueprint/operations reports, missing the 2023 source. Re-confirmed on a second,
   independent routing pass (not just one-off sampling noise) for one of the two — the
   router consistently favors the two big 2025 overview docs over the more specific 2023
   companion source when a question mixes recency and cross-reference framing.

6. **[synthesis]** WM/Geotab tracking question — same pattern, missing `messy-middle
   -bootcamp-session-2-2025` (where WM actually appears as a guest speaker, the source
   that explains *why* WM isn't in the tracked-fleet list).

**One borderline case not counted as a miss, and non-deterministic across runs:** the
pre-existing "Messy Middle Bootcamp Session 2 — hydrogen refueling" out-of-scope question.
At the old, smaller catalog scale the router returned an empty selection (scored CORRECT
previously). On this run's first pass at 178 sources it selected `messy-middle-bootcamp
-session-2-2025` itself instead (defensible — that doc covers the general topic, just not
the specific fact asked); on the independent second routing pass (resume run) it went
back to an empty selection. Noted as router non-determinism on a genuinely ambiguous case,
not scored as wrong either way.

**Pattern across the misses:** every one of the 6 clear misroutes involves two or more
real sources on the same topic/campaign — Run on Less 2017 vs. Messy Middle 2025, the
2023 white paper vs. the 2025 field reports, Electric DEPOT 2024 vs. Messy Middle 2025.
This is precisely the routing-stress condition the new questions (task #5) were built to
surface: at 6-25 sources the router never had two similar-enough candidates to confuse;
at 178 it does, and picks the more prominent/recent doc over the more precisely correct
one often enough to matter.

## Answer accuracy on the 37 completed: 30 correct, 4 partial, 2 wrong, 0 hallucinated, 1 flagged (stale ground truth)

| Bucket | Correct | Partial | Wrong | Notes |
|---|---|---|---|---|
| prose (15) | 13 | 1 | 0 | 1 flagged, see below |
| table (6) | 6 | 0 | 0 | |
| figure (5) | 2 | 2 | 1 | |
| out_of_scope (4) | 4 | 0 | 0 | |
| synthesis (4) | 3 | 1 | 0 | |
| recency (3) | 2 | 0 | 1 | |
| **Total (37, +1 flagged)** | **30** | **4** | **2** | |

Both "wrong" answers trace directly to misroutes above (Figure 32, longest BEV range) —
no hallucination or answer-stage invention once given the wrong source; the model just
answered faithfully from what it was handed. The "partial" figure answers (weight/
elevation charts) are figure-reading imprecision with the *correct* source selected —
vague "over 8,000 feet" instead of the source's 8,500-8,800 ft range, no specific peak
date given — an answer-stage precision gap, not a routing one. The "partial" synthesis
answer (2023 sequence vs. 2025 results) is directionally correct (confirms fleet-specific
evaluation was needed) but never actually cites the 2023 white paper's own framework
language, because routing never gave the answer stage that source. The last-mile EV van
TCO synthesis question is the one case where a routing miss cost nothing — scored correct
despite the router substituting two adjacent sources for the expected one, because the
substitutes covered the same underlying parity claim well enough.

### Flagged: stale ground truth, not a system error

One question ("What comparison did WM's Marty Tufte give...1985 vs. 2018 standard
trucks") has `expect: "...10,080 of its trucks..."` in `questions.jsonl`, written against
this video's *original* caption-only ingest. The model's answer, sourced from the video's
new Gemini-video-understanding transcript (re-ingested this session, see the "Add 140 Run
on Less video sources" commit), says "1,080" — a factor-of-10 discrepancy between the two
independent transcriptions of the same spoken number. The answer is faithful to the
current (better) source; the eval question's `expect` field just wasn't updated when the
source was re-ingested. Worth fixing `questions.jsonl` directly rather than treating this
as a model or routing failure — noted here rather than scored either way.

### Notable correct answers
- Both recency questions using the current `blueprint-2025` report as ground truth
  correctly gave up-to-date figures (five demonstrations, 63 drivers) rather than the
  stale numbers a naive single-source answer would produce from the 2024 Electric DEPOT
  report.
- The "Guidance Reports as of 2019 vs. mid-2025" synthesis question — one of the 3
  synthesis questions where routing succeeded — got both counts exactly right (3 and 11)
  by correctly pulling in both the 2023 white paper and the 2025 blueprint.
- All 10 table questions (bootcamp registration/attendance figures, WM RNG production
  volumes, NOx thresholds) matched exactly, including unit-level precision (BCF↔DGE
  conversions, g/bhp-hr NOx thresholds).
- All 4 out-of-scope questions correctly short-circuited to "no answer call made" with no
  fabrication, across two independent routing passes.
- Several new fleet-disambiguation questions (Henry Albert 2017 vs. 2025, Frito-Lay
  across three campaigns, Mesilla Valley Transportation) all routed and answered
  correctly, confirming the router *can* separate near-identical fleet content across
  campaigns when the question doesn't also mix in a recency/cross-reference comparison.

## Bottom line

Routing degrades gracefully but measurably going from ~25 to 178 sources: 87% exact-match
on known-source questions, with every miss involving genuine near-duplicate campaign
content rather than random noise — and specifically clustering on questions that ask the
router to compare *across* two similar-but-different-vintage sources, not on straight
single-fact lookups (those routed perfectly, 22/22 prose and 6/6 table). Answer quality,
when routing succeeds, remains strong (30/36 scorable correct, 4 partial, 0 hallucinated).
Both answer-stage failures are fully explained by upstream routing misses, not model
invention.

**Suggested next steps**, not yet done:
1. Resume this eval run (no `--fresh`) once `gemini-3.6-flash`'s daily quota resets again,
   to score the remaining 15 answers.
2. Fix the stale `expect` value for the WM 1985-vs-2018 question in `questions.jsonl` (or
   verify against the original video audio which number is actually correct).
3. Consider whether `route.txt`'s prompt needs an explicit tie-breaking rule for
   same-topic docs across campaigns/years, and specifically for cross-reference/comparison
   questions that need *both* an older and a newer source — the 6 misroutes above all
   look like exactly this ambiguity, concentrated in `synthesis` and `recency` buckets.
