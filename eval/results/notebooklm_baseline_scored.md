# NotebookLM Baseline — Scored

Scored against `eval/questions.jsonl` using the spec's rubric: correct / partial / wrong / hallucinated.
Hallucinated is disqualifying; nothing in this run hit that bar — every defect found was either a
minor inserted inaccuracy or a gap in my own eval question, not a fabricated fact.

## Summary

| Bucket | Correct | Partial | Wrong | Hallucinated |
|---|---|---|---|---|
| prose (9) | 9 | 0 | 0 | 0 |
| table (5) | 4 | 1 | 0 | 0 |
| figure (5) | 5 | 0 | 0 | 0 |
| out_of_scope (4) | 2 | 2 | 0 | 0 |
| recency (3) | 3 | 0 | 0 | 0 |
| synthesis (4) | 4 | 0 | 0 | 0 |
| **Total (30)** | **27** | **3** | **0** | **0** |

**90% correct, no hallucinations, on the very first pass — a high bar for the two-stage system to beat.**
The three "partial" scores break down as: one real minor error (Q7's ranking claim), one unverifiable
added specificity (Q30's regulation name), and one genuine design flaw in *my* out-of-scope question
(Q15) rather than a NotebookLM grounding failure.

---

## Question-by-question

### Q1 [prose] — CORRECT
13 fleets, correct 4/3/4/2 powertrain split, correct fleet names. Bonus: correctly noted 14 vehicles
tracked (Saia ran two Tesla Semis) — accurate and not something I'd asked for.

### Q2 [prose] — CORRECT
Sept 8–25, 2025, 18 days. Matches exactly.

### Q3 [prose] — CORRECT
63 drivers, 105 interviews, right date range and interviewee categories. Didn't restate the "48 of
105 were support-org personnel" breakdown from my expected answer, but that was supplementary detail,
not the core ask.

### Q4 [prose] — CORRECT
9% / 48%, correct NREL report title.

### Q5 [table] — CORRECT
7,171 / 1,943 totals and 1,191 / 720 uniques, all exact. Also computed correct per-session averages
(≈797 registrants, ≈216 attendees) and cited the report's own rounded prose range ("600 to 900ish")
alongside the exact table — a real passage in the source, not an error.

### Q6 [table] — CORRECT
Session 4: 901/197 registered, 300/122 attended. Exact.

### Q7 [table] — PARTIAL
Core answer correct: Session 7 ("The Production Process[es] of Hydrogen Fuel"), 143 total attendance,
lowest of the nine sessions. But it also claims Session 7 had the "third-lowest unique attendance" at
26 — that's wrong. 26 unique attendees is actually the **lowest** unique count of all nine sessions
(sorted: 26, 28, 29, 32, 44, 78, 96, 122, 265). Minor inserted ranking error on top of an otherwise
correct answer.

### Q8 [figure] — CORRECT
73°F, 7 MPH SSE, Rain. Exact.

### Q9 [figure] — CORRECT
250 ft, 10°F/80°F, 10,000 lbs, +10 MPH → −1 MPG, +15 MPH → −2 MPG. All exact, including the "7 MPH
above optimal" framing from the infographic itself.

### Q10 [figure] — CORRECT
46.5% / 25.4% / 15.5% / 12.6%. Exact.

### Q11 [figure] — CORRECT
~80,000 lbs, Sept 11–13. Exact.

### Q12 [figure] — CORRECT
~8,600 ft around Sept 24 (within my 8,500–8,800 range), plus an accurate note about an earlier
5,000–7,000 ft peak (matches the ~6,900 ft peak visible around Sept 10 in the actual chart).

### Q13 [out_of_scope] — PARTIAL (see note)
Question asked what *this report* (the blueprint) concludes about TCO. NotebookLM correctly notes the
dedicated TCO Ramifications Report is deferred to 2026, but then answers anyway with a full TCO
breakdown per powertrain, pulling real numbers from the *2023 white paper* ($2,300–$42,000 diesel
NOx-compliance cost range, NG at 1.5–2x diesel upfront cost, H2FC at 3–4x cost, $3/kg parity target —
**I verified all of these are real, correctly quoted figures**, not fabricated). So this isn't a
hallucination — it's NotebookLM treating the whole notebook as one corpus rather than answering
strictly about the named document. Whether that's "wrong" depends on how strictly you read "this
report." Flagging as partial because a reader could come away thinking the blueprint itself contains
a TCO comparison, when it explicitly says it doesn't.

### Q14 [out_of_scope] — CORRECT
Correctly refuses: no cold-weather Tesla Semi data exists in the corpus. Good citation discipline here
— it adds real, clearly-caveated adjacent context (Saia's Donner Pass route, PepsiCo's Sacramento–Reno
Tesla Semi routes from the *2024 Electric DEPOT report*, which I hadn't cited in my own expected answer)
without ever claiming this counts as cold-weather performance data. Better than my own expected answer,
which only mentioned Saia.

### Q15 [out_of_scope] — PARTIAL (eval design issue, not really NotebookLM's fault)
My expected answer said "out of scope for this report" (true for the blueprint alone). But NotebookLM
answered in detail using the *2023 white paper*'s general BEV-in-urban/regional-haul TCO discussion and
the *2024 Electric DEPOT report*'s "small energy depots are ready to electrify now" section (Figure 57)
and LCFS-credit mention — **I verified both of these are real content**, not fabricated. Given the
actual 4-document corpus does contain general electric-van/step-van TCO material, a full refusal would
actually be under-informative here. This is really a flaw in how I scoped this eval question, not a
NotebookLM grounding failure — worth revising the question or accepting this bucket needs a topic with
zero coverage anywhere in the corpus, not just in one report.

### Q16 [recency] — CORRECT
Five demonstrations, correctly lists all five (2017, 2019, 2021, 2023, 2025) with accurate one-line
descriptions of each. Correctly cites the current report's "NACFE has conducted five Run on Less
demonstrations to date" rather than the superseded 2024 report's "four."

### Q17 [recency] — CORRECT (best-handled recency answer of the three)
63 drivers total — and unlike Q16/Q18, this answer *explicitly surfaces the old-vs-new contrast
unprompted*: "Through 2023 (Four Demonstrations): By the conclusion of the fourth event... NACFE had
engaged with a total of 48 drivers... The 2025 Messy Middle Run: With completion of this fifth
demonstration, the total number of engaged drivers rose to 63." That's precisely what a recency-bucket
question is designed to test — not just landing on the current number, but showing the system knows
the old number exists and is superseded, rather than silently picking one. This is the standard I'd
want Q13/Q15's out-of-scope handling held to as well.

### Q18 [recency] — CORRECT, and it corrected *my* ground truth
NotebookLM answered 410 miles (Tesla Semi, PepsiCo, 2023 Electric DEPOT) as the historical baseline,
not the 230-mile eCascadia I'd used in my expected answer. **I checked the source directly — the 2024
report really does say "Tesla Semis saw ranges of 410 miles on a single charge and completed 1,076
miles in a 24-hour period"** (PepsiCo/Sacramento section, p.39-40), which I'd missed when writing the
original eval question. NotebookLM's answer is more complete than my own ground truth: it correctly
carries the 410-mile 2023/2024 figure forward, then correctly identifies the 2025 Messy Middle's
Windrose 420-mile figure as the current state, and separately notes the still-typical 150–250 mile
range for most other OEMs. `eval/questions.jsonl`'s Q18 has been corrected to match.

### Q19 [synthesis] — CORRECT
Thorough and accurate: 2023's 150–250 mi typical range plus the caveated Tesla 500-mi demo, versus
2025's real-fleet 400+ mi (Windrose 420 mi, Saia's Tesla Semi, 4Gen's daily mileage pattern), plus the
410-mi/1,076-mi PepsiCo 2023 baseline (verified real, see Q18). Matches and exceeds my expected answer.

### Q20 [synthesis] — CORRECT
Correctly identifies the 2023 framework, correctly quotes the 2025 report's "no single winner" finding
and the PepsiCo "no one-size-fits-all" quote verbatim. Worth flagging separately: this answer's citation
dump was ~144,000 characters — by far the largest of the 30 — meaning NotebookLM likely retrieved close
to the entire corpus for this one question. The answer quality is good, but that's a real cost/precision
data point in NotebookLM's favor on quality and against it on efficiency, relevant when comparing to a
two-stage router that's supposed to select 1-6 sources instead of everything.

### Q21 [synthesis] — CORRECT
3 Guidance Reports (early 2019) → 11 (mid-2025), correct titles.

### Q22 [prose] — CORRECT
60 stations / 53 in CA / 4 heavy-duty. Exact, plus a plausible (unverified by me) added detail about
CA's 1,000-station-by-2030 plan.

### Q23 [prose] — CORRECT
65-85% / 40-50% / 30-50%. Exact.

### Q24 [prose] — CORRECT
13,000+ trucks, 213 stations (25 public), 2,000+ techs, 10,000+ drivers, $4B invested, 110M gal diesel
displaced in 2024. All exact, correctly attributed to Marty Tufte.

### Q25 [table] — CORRECT
7 facilities / ~40M DGE now, 17 under construction, 200M+ DGE by 2026. Didn't restate the -126 carbon
intensity score, but that wasn't the core ask.

### Q26 [prose] — CORRECT
54 trucks (1985) = 10,080 trucks (2018 standard). Exact.

### Q27 [prose] — CORRECT
~10% of PTI's fleet, correctly attributed to Dan Deppeler, correct context (15+ years on CNG, "second
inning" framing).

### Q28 [out_of_scope] — CORRECT
Best-handled out-of-scope answer of the four: correctly refuses for Session 2 specifically, then
clearly labels supplementary hydrogen-infrastructure facts as "covered in other parts of the bootcamp
and research" rather than blending them in as if Session 2 said them. This is the pattern I'd want Q13
and Q15 to follow.

### Q29 [synthesis] — CORRECT
Correctly says WM was not a tracked fleet, correctly lists all 13 actual fleets tracked.

### Q30 [table] — PARTIAL
Core numbers correct (5,000-6,000 trucks at 0.02 g/bhp-hr vs. the referenced 0.035 g/bhp-hr threshold).
But it adds "the 0.035 g/bhp-hr threshold is the requirement... established by the EPA's Clean Truck
Plan for model years 2027 and beyond" — I couldn't verify "EPA's Clean Truck Plan" was actually named in
either the transcript or the written reports; the transcript just has Tufte saying "0.035 hanging out
there right now" without naming a specific program. Possibly correct from general knowledge, but it's
added specificity beyond what I could confirm is grounded in the corpus.

## Takeaways for the two-stage build

1. **Citation-image workflow works** — pasting NotebookLM's highlighted source images was the right
   call; several figure-bucket answers wouldn't have been checkable otherwise.
2. **No hallucinations, but weak document-scoping.** NotebookLM answers "what does this report say"
   as "what does the corpus say," and doesn't always flag when it's crossed document boundaries (Q13,
   Q15) versus when it does so well (Q28, Q14). The two-stage design's per-source citation requirement
   in `answer.txt` should make this an explicit, checkable behavior rather than optional.
3. **Retrieval isn't selective** — Q20's ~144K-char citation dump suggests NotebookLM sometimes pulls
   in most of the corpus rather than the 1-6 sources the spec's `route.txt` calls for. Worth watching
   whether the real system beats this on cost even if quality ties.
4. **My own eval had two defects**, both worth fixing: Q18's expected answer was incomplete (missed the
   410-mile PepsiCo Tesla Semi figure in the 2024 report), and Q15 assumes a stricter single-document
   out-of-scope boundary than the actual 4-document corpus supports.
