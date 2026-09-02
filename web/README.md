# Embedding the NACFE Knowledge Base widget

`widget.js` is a single, dependency-free file. It injects its own CSS and DOM into a
mount point and talks to the deployed Cloudflare Worker over `fetch()`. No build step,
no framework, nothing else to load.

## Minimal embed

The Worker itself now serves `widget.js` (added via a `GET /widget.js` route — see
`build_corpus.mjs`, which bundles this file into the Worker at build time same as the
prompts, and `worker/src/index.ts`'s route handler). So the actual, currently-live embed
snippet is just:

```html
<div id="nacfe-assist"></div>
<script src="https://nacfe-assist.nacfe.workers.dev/widget.js" defer></script>
```

One deployed artifact, one URL, nothing else to host. `data-api` is optional — the
widget's default already points at that same Worker's `/query` endpoint, so you only
need it if you ever point the widget at a different deployment (e.g. testing against a
preview/staging Worker):

```html
<script src="https://nacfe-assist.nacfe.workers.dev/widget.js"
        data-api="https://some-other-deployment.workers.dev/query" defer></script>
```

- The `<div id="nacfe-assist">` is where the widget renders. If you omit it, the script
  creates one right after itself.
- The widget is self-contained under `#nacfe-assist-root`; it won't collide with the
  host page's CSS or JS. Safe to drop into a WordPress page/post body, a template
  include, or a static HTML page.

## What actually needs to happen to put this on nacfe.org

Two separate pieces, each a real decision for you or NACFE's web team, not something
resolvable from inside this repo:

**1. Get the `<div>` + `<script>` snippet onto an actual NACFE page.** This requires
   whoever manages nacfe.org's content (WordPress admin, page builder, or raw HTML
   access) to paste the two-line embed above into a page or template. I don't have
   access to NACFE's CMS from this environment — this step has to happen on their side.

**2. (Recommended, not required) Put the API behind a NACFE-branded custom domain.**
   Right now the Worker answers at `nacfe-assist.nacfe.workers.dev` — a Cloudflare
   subdomain, not NACFE's. It works fine as-is (CORS is already open to any origin), but
   a URL like `assist.nacfe.org` or `api.nacfe.org` reads better and survives if the
   Workers.dev URL ever needs to change. This requires NACFE's domain to be on
   Cloudflare (or delegate a subdomain to it):
   ```bash
   npx wrangler deploy --route "assist.nacfe.org/*"
   ```
   then add a DNS record for that subdomain in NACFE's Cloudflare zone (or add
   `nacfe.org` as a zone if it isn't on Cloudflare yet). If NACFE's site is *not* on
   Cloudflare at all, skip this — the `workers.dev` URL works fine indefinitely, it's
   just not branded.

## Deploy checklist (new steps from the security review)

Three one-time steps before the next `npm run deploy`, on top of the existing secrets:

```bash
cd worker

# 1. Signs the per-answer feedback token. Without it /feedback stays disabled
#    (GET /health reports "feedback": false) rather than accepting ratings against
#    guessable sequential query ids.
openssl rand -hex 32 | npx wrangler secret put FEEDBACK_SECRET

# 2. One rating per served answer, and dedupe any existing duplicates.
npx wrangler d1 execute nacfe-assist --remote --file=migrations/0001_feedback_unique_query_id.sql
```

3. The rate limit and the monthly token ceiling are now Durable Objects (`RateLimiter`,
   `Budget` in `src/counters.ts`) rather than KV counters, which were eventually consistent
   and so didn't actually hold under concurrency. `wrangler deploy` applies the `v1`
   migration in `wrangler.toml` automatically; Durable Objects must be enabled on the
   account for the deploy to succeed.

After deploying, `curl https://nacfe-assist.nacfe.workers.dev/health` should report
`"feedback": true`.

## Before sending real public traffic

The deployed Worker currently runs on a **free-tier Gemini API key** — the same one
used throughout this project's development and eval runs. Free tier limits (per Google,
subject to change): a few requests per minute and a capped number of requests per day,
shared across every call this key makes, ingest scripts included. A public embed on
nacfe.org would very likely exhaust this within the first real day of traffic, at which
point the Worker's `MONTHLY_TOKEN_CEILING` degrade-to-cache-only behavior (SPEC.md §7)
kicks in for the wrong reason — not because the intended monthly budget was hit, but
because the underlying key choked on rate, not spend.

**Fix, before going live:** put the project on a paid Gemini API tier (a billing account
attached to the same Google Cloud project). At Flash-Lite/Flash pricing this is the
<$50/month target SPEC.md §0 already budgets for — the free tier was only ever meant to
keep ingest and eval costs at zero during development, not to serve production traffic.
Set the new paid key with:
```bash
cd worker && npx wrangler secret put GEMINI_API
```

## Local testing

```bash
cd web && python3 -m http.server 8420
```
then open `http://localhost:8420/demo.html`. `demo.html` is the reference embed — it's
the exact two-line snippet above, nothing more, so what you see there is what a
publisher embedding this will get.
