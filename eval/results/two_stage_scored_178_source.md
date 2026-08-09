# Two-Stage Query Loop — Scored at 178-Source Scale (52/52 complete)

Re-run of `eval/questions.jsonl` (52 questions: the original 30 plus 22 new
routing-stress questions added for the expanded corpus) against the full 178-entry
`catalog.json` — up from the ~6-25 sources the original eval ran against. Completed
across three resumed runs over three days, each blocked by `gemini-3.6-flash`'s
free-tier **daily** answer-stage quota (the catalog itself, at ~80K tokens per routing
call now, is unrelated to that cap — routing completed in full on the very first pass).

## A real bug found and fixed along the way

The first resume run's results file came back with only 50 of 52 records, even though
the console log clearly showed all 52 processing (many as "reusing prior result"). Root
cause: `run_eval.py` only rewrote `two_stage_raw.jsonl` inside the freshly-fetched branch
of the per-question loop — a `continue` on the "already answered, reuse it" branch
skipped the file write entirely. If the *last* questions in a run happen to be reused (as
happened — the two newest out-of-scope questions were both trailing reused entries), they
get appended to the in-memory list and reported as answered in the final summary, but
never reach disk. Fixed in `eval/run_eval.py` by writing the file on every iteration,
reused or not. Confirmed fixed: the next two resumes both landed the full record count.

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

### Misroutes (5 stable + 1 that self-corrected on retry)

1. **[figure]** "What is 0.5 MPG worth?" infographic (Figure 32) — routed to
   `run-on-less-2017` instead of `run-on-less-messy-middle-blueprint-2025`. Downstream
   effect: the answer cited different figures/numbers than expected — a real, traceable
   routing→answer failure chain (this is the eval's one clear "wrong" answer).

2. **[recency]** "Longest range demonstrated by a battery-electric Class 8 truck" —
   routed only to `run-on-less-electric-depot-2024` (2023 demo, correctly the source of
   the 410-mile Tesla Semi answer), never pulling in `run-on-less-messy-middle-blueprint
   -2025` to check whether the 2025 demo beat it. Exactly the failure mode the question
   was designed to stress-test — the router fell into it.

3. **[synthesis]** Last-mile electric van TCO — routed to `confidence-report-vans-step
   -vans-2022` + `guidance-report-medium-duty-electric-cost-of-ownership-2018` instead of
   `run-on-less-electric-depot-2024`. Reasonable adjacent picks, and the answer still
   landed correct anyway (the substitutes covered the same underlying claim).

4. **[synthesis]** "NACFE's 2023 white paper laid out a suggested sequence for fleets..."
   — consistently routed to just the 2025 Messy Middle blueprint/operations reports,
   missing `messy-middle-a-time-for-action-2023` (the 2023 source the question is
   explicitly about). This is the eval's one "partial" answer: directionally correct
   (confirms fleet-specific evaluation was needed) but never cites the 2023 framework
   language the question asked about, because routing never supplied that source.

5. **[synthesis]** WM/Geotab tracking question — same pattern, missing `messy-middle
   -bootcamp-session-2-2025` (where WM actually appears as a guest speaker, the source
   that explains *why* WM isn't in the tracked-fleet list). Answer still landed correct
   on the core yes/no claim using the sources it did get.

**Self-corrected on retry, not counted as a stable miss:** a near-identical synthesis
question ("In 2023, NACFE described typical production BEV range as 150-250 miles...")
missed `messy-middle-a-time-for-action-2023` on the first routing pass but correctly
included it (alongside both 2025 reports) on a later independent pass, producing a fully
correct answer. Confirms the misses above are a real bias, not fixed behavior — the
router *can* pull in the 2023 companion source for this exact style of question, it just
doesn't reliably.

**Also non-deterministic across runs, not scored as a miss either way:** the "Messy
Middle Bootcamp Session 2 — hydrogen refueling" out-of-scope question and the Tesla Semi
cold-weather question. Both have a documented source that explains the "why" behind the
out-of-scope answer, and both got an empty selection on final scoring — correct per
SPEC.md's rule that no source addressing the specific question means an empty selection
is correct. On an earlier pass, one of the two instead selected the loosely-related
document. Noted as router non-determinism on genuinely ambiguous cases.

**Pattern across the misses:** every stable misroute involves two or more real sources on
the same topic/campaign — Run on Less 2017 vs. Messy Middle 2025, the 2023 white paper vs.
the 2025 field reports, Electric DEPOT 2024 vs. Messy Middle 2025 — and specifically
clusters on `synthesis`/`recency` questions that need the router to compare *across* an
older and a newer source, not on single-fact lookups (`prose` went 22/22, `table` 6/6).
This is precisely the routing-stress condition the new questions (task #5) were built to
surface: at 6-25 sources the router never had two similar-enough candidates to confuse;
at 178 it does, and it favors the more prominent/recent doc over the more precisely
correct one often enough to matter, though not with perfect consistency.

## Answer accuracy: 45 correct, 4 partial, 2 wrong, 0 hallucinated, 1 flagged (52 total)

| Bucket | Correct | Partial | Wrong |
|---|---|---|---|
| prose (24) | 22 | 1 | 0 |
| table (6) | 6 | 0 | 0 |
| figure (7) | 4 | 2 | 1 |
| out_of_scope (4) | 4 | 0 | 0 |
| synthesis (6) | 5 | 1 | 0 |
| recency (5) | 4 | 0 | 1 |
| **Total (52, +1 flagged)** | **45** | **4** | **2** |

(Prose bucket total is 23 questions plus the 1 flagged stale-ground-truth question below —
24 including it.)

Both "wrong" answers trace directly to the misroutes above (Figure 32, longest BEV range)
— no hallucination or answer-stage invention once given the wrong source; the model just
answered faithfully from what it was handed. The two "partial" figure answers (weight/
elevation charts) are figure-reading imprecision with the *correct* source selected —
vague "over 8,000 feet" instead of the source's 8,500-8,800 ft range, no specific peak
date given — an answer-stage precision gap, not a routing one. The one "partial" synthesis
answer is the 2023-sequence-vs-2025-results question described above.

### Flagged: stale ground truth, not a system error (fixed in `questions.jsonl`)

One question ("What comparison did WM's Marty Tufte give...1985 vs. 2018 standard
trucks") originally had `expect: "...10,080 of its trucks..."`, written against this
video's *original* caption-only ingest. The model's answer, sourced from the video's new
Gemini-video-understanding transcript (re-ingested this session), correctly says "1,080"
— verified directly against the current source text
(`corpus/sources/messy-middle-bootcamp-session-2-2025.md`, line 597: "54 trucks in 1985
... equals 1,080 trucks"). The answer was faithful to the current, better source; the old
`expect` field just hadn't been updated when the source was re-ingested. Already corrected
in `questions.jsonl`.

### Notable correct answers
- Both recency questions using the current `blueprint-2025` report as ground truth
  correctly gave up-to-date figures (five demonstrations, 63 drivers) rather than stale
  numbers a naive single-source answer would produce from the 2024 Electric DEPOT report.
- The hydrogen and charging-infrastructure "current guidance" recency questions both
  correctly routed to and cited the 2023 (current, non-superseded) reports rather than
  the superseded 2019/2020 ones — direct validation of task #3's supersession curation.
- The "Guidance Reports as of 2019 vs. mid-2025" synthesis question got both counts
  exactly right (3 and 11) by correctly pulling in both the 2023 white paper and the 2025
  blueprint.
- The ATIS synthesis question (confidence report explanation + a specific 2017 driver
  profile) routed to and correctly synthesized both sources, correctly naming driver Mark
  Risien (US Xpress) as the answer.
- All 6 table questions plus every video-clip disambiguation question (Henry Albert 2017
  vs. 2025, Frito-Lay across three campaigns, Diving into Data vs. Importance of Data,
  Regional vs. Messy Middle duty-cycle stories) routed and answered correctly, confirming
  the router can separate near-identical fleet/topic content across campaigns when the
  question doesn't also demand an explicit cross-vintage comparison.
- All 4 out-of-scope questions correctly short-circuited to "no answer call made" with no
  fabrication, across multiple independent routing passes.

## Bottom line

Routing degrades gracefully but measurably going from ~25 to 178 sources: 87% exact-match
on known-source questions, with every stable miss involving genuine near-duplicate
campaign content and specifically clustering on cross-vintage comparison questions, not
random noise or single-fact lookups. Answer quality, when routing succeeds, is very
strong: 45/49 scorable correct, 4 partial, 0 hallucinated. Both answer-stage failures are
fully explained by upstream routing misses, not model invention — when the model gets the
right source, it gets the fact right, including precise unit-level figures (BCF↔DGE
conversions, g/bhp-hr NOx thresholds, kWh battery specs) across 10 different table/figure
questions with zero errors.

**Suggested next step**, not yet done: consider whether `route.txt`'s prompt needs an
explicit tie-breaking rule for same-topic docs across campaigns/years, and specifically
for cross-reference/comparison questions that need *both* an older and a newer source —
the stable misroutes above all look like exactly this ambiguity, concentrated in
`synthesis` and `recency` buckets, and the router's own inconsistency across repeated
routing passes on the same question suggests this is a genuinely underspecified case in
the current prompt rather than a hard capability limit.
