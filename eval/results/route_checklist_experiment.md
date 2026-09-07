# Rejected: a pre-commit checklist in route.txt

Tested 2026-09-07 against the 343-source catalog with `run_eval.py --routing-only`, 52
questions, free tier. **Reverted — no benefit, and a likely end-to-end regression.**

## Hypothesis

The eval's four routing misses and two of its answer partials looked like rules `route.txt`
already stated in prose (two-anchor questions needing a source each, prefer the narrower
companion source, don't over-refuse, prefer the recent specific source for cost questions)
failing to fire once the corpus reached 343 entries. Adding a short four-point checklist to
run before finalizing should make those rules bite.

## Result: 48/52 → 48/52

| Bucket | Before | After | |
|---|---|---|---|
| prose | 24/24 | 24/24 | |
| table | 6/6 | 6/6 | |
| figure | 7/7 | **6/7** | regressed |
| recency | 4/5 | **5/5** | improved |
| synthesis | 4/6 | 4/6 | |
| out_of_scope | 3/4 | 3/4 | |
| **Total** | **48/52** | **48/52** | net zero |

One fixed, one broken. On n=52 a single swap is inside the noise, so this is a null result
rather than a measured tie.

## Why it was reverted rather than kept as neutral

Routing accuracy is a proxy. Checked against what the two swapped questions actually
answered in the scored run, the trade is negative:

- **Fixed — "longest range demonstrated by a battery-electric Class 8 truck."** Scored a
  routing miss before (it picked the two 2024 DEPOT reports over the 2025 blueprint) but
  **answered correctly anyway**, because that fact lives in the DEPOT reports. Correcting
  this routing miss buys nothing end to end.
- **Broken — "What is 0.5 MPG worth?" (Figure 32).** Before, it selected three sources
  including `run-on-less-messy-middle-blueprint-2025`, which is where Figure 32 actually is,
  and answered correctly. After, it selects one source, `run-on-less-2017`, losing the
  document that holds the figure. That answer would very likely become wrong or a refusal.

So the checklist trades a harmless routing miss for a harmful one.

The mechanism is visible in the selection counts: average sources selected fell from 1.19 to
1.17. The checklist reads as a final pruning step and makes the router *more* selective,
which runs directly against the existing figure rule — "select more than one rather than
picking a single guess" — that was carrying the figure bucket at 7/7.

It also cost ~1.8K tokens on every routing call, for no measured gain.

## What this says about the four misses

They are not a prompt-obedience problem that more instruction fixes. Two of the four never
mattered end to end. Any future attempt should be measured on answer correctness, not routing
accuracy, and should avoid anything that encourages narrowing the selection — the figure
bucket depends on breadth.

Re-run with: `python3 eval/run_eval.py --routing-only` (free, ~1 hour, writes to its own file).
