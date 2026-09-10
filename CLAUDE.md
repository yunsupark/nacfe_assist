# NACFE Knowledge Base — working notes

Read this before changing anything. It records what is deployed and which properties were
established by measurement, so they are not undone by accident.

## Where the code is

`main`, in the repo root. The two worktrees under `.claude/worktrees/` are historical: their
branches are ancestors of `main` and contain nothing `main` lacks. Do not deploy or edit from
them — `wrangler` will run there quite happily against a stale tree.

Live: `https://nacfe-assist.nacfe.workers.dev`. Verify any deploy with
`cd worker && ./preflight.sh https://nacfe-assist.nacfe.workers.dev` — it checks the things
that fail silently (missing secrets, unprotected `/query`, an unsubstituted sitekey, a dead
impressions beacon).

## Invariants — breaking these has cost real money or real credibility

**Turnstile runs before anything that spends.** Verification sits ahead of the rate limiter,
the budget reservation and every model call in `handleQuery`. `/query` costs ~$0.076 uncached
and CORS is `*`, so a rejected request must cost nothing but the siteverify round trip.
`TURNSTILE_HOSTNAMES` must never contain `localhost` in production — an attacker would mint
tokens on their own local page. Consequence: `/query` can only be exercised from `nacfe.org`.

**The monthly ceiling is denominated in dollars, not tokens** (`MONTHLY_COST_CEILING_USD`,
`worker/src/pricing.ts`). It was previously 20,000,000 tokens, which sounds generous and is
about **$8** — roughly 100 questions, six times under the $50 target in SPEC §0. That
under-sizing is a reason to fix the unit, not to remove the cap: input and output prices differ
by up to 12.5x across the two stages, so a token count is a poor proxy for spend. `$50` buys
~650 uncached questions; cache hits are free. `/status`, the widget's budget banner and the
`Budget` durable object all depend on this existing.

**Prompt templates use `fillTemplate`, never `String.replace`.** With a string replacement,
`$&`, `` $` `` and `$'` in the *replacement* are substitution patterns. Six corpus sources
contain `$'`; a question containing `` $` `` duplicated the entire ~150K-token catalog into the
prompt. This shipped once and the Python eval could not reproduce it, because Python's
`str.replace` has no `$` semantics.

**`eval/run_eval.py` must mirror the Worker.** It is a parallel implementation, so anywhere the
two drift is somewhere the eval stops predicting production. The router catalog is built
identically and verified byte-identical (`separators=(",", ":")` plus `ensure_ascii=False` to
match `JSON.stringify`). Validate a prompt change with `python3 eval/run_eval.py --routing-only`
— free tier, ~1 hour, writes to its own results file.

**Rate limit and budget live in durable objects, not KV.** KV is eventually consistent and
edge-cached; as read-modify-write counters neither limit actually held.

**Logging must never break serving.** `logQuery` is awaited in the request path; a D1 failure
returns null rather than throwing. Failed queries are logged too (`queries.error`) — without
that a reported timeout leaves nothing to diagnose.

**Only the Gemini calls get the request's abort signal.** R2 gets and D1 writes take no
`AbortSignal`, so both are bounded with `withTimeout`. The free-tier attempt has its own 35s
budget: a stalled free call used to consume the whole request deadline and starve the paid
fallback.

## Data

Migrations 0001–0009 are applied to the remote D1. `schema.sql` matches production; several
columns lived there undeclared for months, so keep the two in step.

`visitor_id` is an HMAC over (month, IP) under `VISITOR_SALT` — the month is inside the hashed
message, so ids rotate by themselves and are unlinkable across months. Within a month
`COUNT(DISTINCT visitor_id)` is a true unique count; **never sum it across months**. The raw IP
is never stored; with `VISITOR_SALT` unset nothing is derived. Reports filter
`LENGTH(visitor_id) = 16` to exclude the legacy persistent UUIDs (now nulled).

Feedback asks helpfulness (`yes`/`partly`/`no`), not accuracy. Readers cannot grade the
correctness of an answer they asked for; the expert eval measures that. The Worker still
accepts the legacy accuracy words because `widget.js` is served with `max-age=3600`, so for an
hour after every deploy browsers run the previous copy.

Useful queries live in `worker/reports.sql`.

## Secrets

`GEMINI_API`, `GEMINI_API_FREE`, `FEEDBACK_SECRET`, `TURNSTILE_SECRET`, `VISITOR_SALT`. All
set. Secrets take effect immediately without a redeploy. Do not rotate `VISITOR_SALT`
mid-month — every visitor would be counted twice.

## Measured numbers, so they are not re-derived by guess

Uncached query: ~$0.076 (routing $0.046 + answering $0.030). Router catalog 150,778 tokens.
Latency median 21s, p75 33s, p90 46s. Free tier: 250,000 input tokens/minute on the routing
model, 20 requests/day on the answering model. Eval at 343 sources: routing 48/52, answers 43
correct / 6 partial / 0 wrong / 0 hallucinated.

Catalog size is a scale problem, not a verbosity one — abstracts average 77 words against
SPEC §3's 300-word target, so there is nothing safe to trim.

## Open decisions

Whether 90s is the right request ceiling (a recent real query took 46s). Whether to keep the
`events` impressions table now that GA covers reach — the sponsor-click half is not something
GA gives you. Sponsor slot ships dormant; set `SPONSOR_NAME`/`SPONSOR_TAGLINE`/`SPONSOR_URL`.
