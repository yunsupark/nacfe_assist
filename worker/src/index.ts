// NACFE Knowledge Base query API. Two-stage loop per SPEC.md 4/5: cheap-model routing over
// the full catalog, then a stronger model answering from the selected sources' full text.
// See eval/run_eval.py for the local-script version this was ported from, and
// eval/results/two_stage_scored.md for the eval this design is validated against.
import { CATALOG, ROUTE_PROMPT, ANSWER_PROMPT, WIDGET_JS } from "./corpus_data";
import { generateContentWithFallback, stripJsonFence, GeminiError, GeminiEmptyResponse } from "./gemini";
import { fillTemplate } from "./prompt";
import type { CatalogEntry } from "./catalog_types";

export { RateLimiter, Budget } from "./counters";

export interface Env {
  GEMINI_API: string;
  GEMINI_API_FREE: string;
  FEEDBACK_SECRET: string;
  CACHE: KVNamespace;
  DB: D1Database;
  SOURCES_BUCKET: R2Bucket;
  RATE_LIMITER: DurableObjectNamespace<import("./counters").RateLimiter>;
  BUDGET: DurableObjectNamespace<import("./counters").Budget>;
  MONTHLY_TOKEN_CEILING: string;
  ROUTE_MODEL: string;
  ANSWER_MODEL: string;
  RATE_LIMIT_PER_IP_PER_HOUR: string;
}

interface RouteResult {
  selected: Array<{ id: string; why: string }>;
  out_of_scope: boolean;
  recency_warning: string | null;
}

/** Hard cap on how many sources the answer stage will ever open. route.txt asks the model for
 * "between 1 and 6", but a prompt instruction is not an enforcement mechanism: a confused or
 * injected routing response naming all 343 sources would otherwise have the answer stage
 * fetch every one of them and build a multi-million-token prompt. */
const MAX_SELECTED_SOURCES = 6;

/** How long a single query may occupy the pipeline before we give up, in ms. Bounds a hung
 * upstream so the widget doesn't spin forever and the Worker doesn't bill for a stalled call. */
const QUERY_TIMEOUT_MS = 90_000;

const CATALOG_BY_ID = new Map(CATALOG.map((c) => [c.id, c]));

/**
 * The projection of the catalog the router actually sees, serialized once at module load.
 *
 * Two savings, neither of which changes what the router can reason about. Compact rather than
 * pretty-printed JSON (`null, 2` was costing ~90KB of pure indentation on every uncached
 * query), and only the fields route.txt actually reasons over -- `url`, `media`,
 * `token_count`, `ingested` and `ingest_model` are serving-side bookkeeping the router has no
 * use for (SPEC.md 3: "abstract and key_findings are what the router sees; everything else is
 * filters and metadata"). `url` in particular is re-attached server-side by enrichSources, so
 * sending it to the model was pure waste.
 *
 * This is a mitigation, not the fix. SPEC.md 3 sets the real budget -- "should land around
 * 40-60K tokens ... if it exceeds ~100K, tighten the abstracts" -- and the catalog is
 * currently ~170K tokens, over that ceiling before any trimming here. Tightening abstracts
 * and key_findings (together ~60% of the payload) is still owed.
 */
type RouterCatalogEntry = Omit<
  CatalogEntry,
  "url" | "media" | "token_count" | "ingested" | "ingest_model"
>;
const ROUTER_CATALOG_JSON = JSON.stringify(
  CATALOG.map((c): RouterCatalogEntry => {
    const { url, media, token_count, ingested, ingest_model, ...rest } = c;
    return rest;
  }),
);

/** Rough token estimate for the fixed part of a routing call, used only to reserve budget
 * before the call is made (reconciled against real usage afterwards). ~4 chars/token. */
const ROUTE_TOKEN_ESTIMATE = Math.ceil((ROUTE_PROMPT.length + ROUTER_CATALOG_JSON.length) / 4);
/** Nominal allowance for the answer stage on top of routing, for the same reservation. */
const ANSWER_TOKEN_ESTIMATE = 60_000;

function normalizeQuestion(q: string): string {
  return q.trim().toLowerCase().replace(/\s+/g, " ").replace(/[?.!]+$/, "");
}

/**
 * Strip the delimiters the prompts use to fence off untrusted input, plus control characters,
 * so a question can't forge the end of its own <question> block and have the rest of its text
 * read as prompt instructions. Tab, newline and carriage return are kept -- a pasted
 * multi-line question is legitimate, and dropping its line breaks would run the words
 * together. Everything else becomes a space rather than being deleted, so removing a
 * character can't join two words either.
 */
function sanitizeQuestion(q: string): string {
  return q
    .replace(/<\/?question>/gi, " ")
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, " ")
    .trim();
}

async function sha256Hex(input: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(input));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Validate the routing model's JSON before trusting any of it, and reduce `selected` to ids
 * that actually exist in the catalog, capped at MAX_SELECTED_SOURCES. The ids flow straight
 * into R2 object keys in answer(), so an unvalidated id is an arbitrary-key read against the
 * bucket; and an uncapped list is an unbounded prompt. run_eval.py was already defensive here
 * (`route_result.get("selected", [])`); the TypeScript port dropped that and would throw on a
 * response missing the field.
 */
function validateRouteResult(parsed: unknown): RouteResult {
  const obj = (parsed ?? {}) as Partial<RouteResult>;
  const rawSelected = Array.isArray(obj.selected) ? obj.selected : [];

  const seen = new Set<string>();
  const selected: Array<{ id: string; why: string }> = [];
  for (const entry of rawSelected) {
    const id = typeof entry?.id === "string" ? entry.id : null;
    if (!id || seen.has(id) || !CATALOG_BY_ID.has(id)) continue;
    seen.add(id);
    selected.push({ id, why: typeof entry.why === "string" ? entry.why.slice(0, 500) : "" });
    if (selected.length >= MAX_SELECTED_SOURCES) break;
  }

  const warning = obj.recency_warning;
  return {
    selected,
    out_of_scope: obj.out_of_scope === true,
    recency_warning: typeof warning === "string" && warning.trim() ? warning.slice(0, 500) : null,
  };
}

/** Attach each selected source's catalog url (null if the source has none) so the widget can
 * render a real link instead of plain text. */
function enrichSources(
  selected: Array<{ id: string; why: string }>,
): Array<{ id: string; why: string; url: string | null }> {
  return selected.map((s) => ({ ...s, url: CATALOG_BY_ID.get(s.id)?.url ?? null }));
}

function corsHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return {
    // the embeddable widget (SPEC.md /web/) is meant to be embedded cross-origin
    "access-control-allow-origin": "*",
    ...extra,
  };
}

function jsonResponse(body: unknown, status = 200, extraHeaders: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: corsHeaders({ "content-type": "application/json", ...extraHeaders }),
  });
}

/**
 * A response that streams newline-delimited lines as the query pipeline actually progresses,
 * so the widget's loading text is driven by real backend events instead of a client-side
 * guess at timing. Two line kinds: "STAGE:<name>" (progress marker, zero or more) and
 * "DATA:<json>" (exactly one, terminal -- either the real result or {"error": "..."}).
 *
 * The status code is necessarily fixed at 200 for this response, since headers are already
 * committed before we know whether route()/answer() will succeed -- errors are signaled
 * in-band via a DATA line with an "error" field instead of an HTTP status, same tradeoff any
 * streaming API (SSE, GraphQL subscriptions, etc.) makes. `run` does the real work and must
 * be kept alive past the synchronous return via ctx.waitUntil, per the Workers streaming
 * pattern (returning a Response wrapping a stream while the producer side keeps writing).
 */
function lineStreamResponse(
  ctx: ExecutionContext,
  run: (write: (line: string) => Promise<void>) => Promise<void>,
): Response {
  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const encoder = new TextEncoder();
  const write = (line: string) => writer.write(encoder.encode(line + "\n"));

  ctx.waitUntil(
    (async () => {
      try {
        await run(write);
      } catch (err) {
        console.error(err);
        try {
          await write(`DATA:${JSON.stringify({ error: "internal error" })}`);
        } catch {
          // writer already errored/closed -- nothing more we can do
        }
      } finally {
        try {
          await writer.close();
        } catch {
          // already closed
        }
      }
    })(),
  );

  return new Response(readable, {
    headers: corsHeaders({
      "content-type": "text/plain; charset=utf-8",
      "cache-control": "no-store",
    }),
  });
}

/** Salted so the stored value isn't a bare SHA-256 of an IP, which is brute-forceable across
 * the whole IPv4 space in seconds. This is a Durable Object addressing key only -- it is
 * never written to D1 (see schema.sql). */
const IP_HASH_SALT = "nacfe-assist:v1:";

async function hashIp(ip: string): Promise<string> {
  return sha256Hex(IP_HASH_SALT + ip);
}

/**
 * Atomic per-IP rate limit. `scope` keeps separate budgets for separate endpoints so feedback
 * spam can't consume a reader's question allowance (and vice versa).
 */
async function checkRateLimit(env: Env, ip: string, scope: string, limit: number): Promise<boolean> {
  const ipHash = await hashIp(ip);
  const stub = env.RATE_LIMITER.get(env.RATE_LIMITER.idFromName(`${scope}:${ipHash}`));
  const hourBucket = String(Math.floor(Date.now() / 3_600_000));
  const { allowed } = await stub.checkAndIncrement(hourBucket, limit);
  return allowed;
}

function budgetStub(env: Env) {
  return env.BUDGET.get(env.BUDGET.idFromName("global"));
}

function currentPeriod(): string {
  return new Date().toISOString().slice(0, 7); // YYYY-MM
}

async function route(
  env: Env,
  question: string,
  signal: AbortSignal,
): Promise<{ result: RouteResult; tokens: number }> {
  const prompt = fillTemplate(ROUTE_PROMPT, {
    catalog: ROUTER_CATALOG_JSON,
    question,
  });
  const { text, usage } = await generateContentWithFallback(
    env.CACHE,
    env.GEMINI_API_FREE,
    env.GEMINI_API,
    env.ROUTE_MODEL,
    prompt,
    signal,
  );
  let parsed: unknown;
  try {
    parsed = JSON.parse(stripJsonFence(text));
  } catch {
    console.error(`route: model did not return parseable JSON: ${text.slice(0, 400)}`);
    parsed = null;
  }
  return { result: validateRouteResult(parsed), tokens: usage.totalTokens };
}

async function fetchSource(env: Env, id: string): Promise<string | null> {
  const obj = await env.SOURCES_BUCKET.get(`${id}.md`);
  if (!obj) {
    console.error(`source not found in R2: ${id}.md`);
    return null;
  }
  return obj.text();
}

async function answer(
  env: Env,
  question: string,
  selectedIds: string[],
  signal: AbortSignal,
): Promise<{ text: string; tokens: number; complete: boolean }> {
  // Fetch the (at most MAX_SELECTED_SOURCES) selected sources from R2 in parallel --
  // sequential round-trips would otherwise stack their latency on top of each other for no
  // reason. Ids are catalog-validated upstream, so these are never attacker-chosen keys.
  const bodies = await Promise.all(selectedIds.map((id) => fetchSource(env, id)));
  const documents = selectedIds
    .map((id, i) => {
      const entry = CATALOG_BY_ID.get(id);
      const title = entry?.title ?? id;
      const body = bodies[i];
      if (!body) return null;
      return `=== SOURCE: ${title} [${id}] ===\n${body}`;
    })
    .filter((d): d is string => d !== null)
    .join("\n\n");

  const prompt = fillTemplate(ANSWER_PROMPT, {
    current_year: String(new Date().getFullYear()),
    documents,
    question,
  });
  const { text, usage, finishReason } = await generateContentWithFallback(
    env.CACHE,
    env.GEMINI_API_FREE,
    env.GEMINI_API,
    env.ANSWER_MODEL,
    prompt,
    signal,
  );
  if (finishReason !== "STOP") {
    console.error(`answer: incomplete generation, finishReason=${finishReason}`);
  }
  return { text, tokens: usage.totalTokens, complete: finishReason === "STOP" };
}

async function logQuery(
  env: Env,
  fields: {
    question: string;
    normalizedQuestion: string;
    cacheHit: boolean;
    outOfScope: boolean | null;
    selectedSources: string[] | null;
    recencyWarning: string | null;
    routeTokens: number | null;
    answerTokens: number | null;
    totalTokens: number | null;
    latencyMs: number;
    degradedCacheOnly: boolean;
  },
): Promise<number> {
  const result = await env.DB.prepare(
    `INSERT INTO queries
      (timestamp, question, normalized_question, cache_hit, out_of_scope, selected_sources,
       recency_warning, route_tokens, answer_tokens, total_tokens, latency_ms, degraded_cache_only)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      new Date().toISOString(),
      fields.question,
      fields.normalizedQuestion,
      fields.cacheHit ? 1 : 0,
      fields.outOfScope === null ? null : fields.outOfScope ? 1 : 0,
      fields.selectedSources ? JSON.stringify(fields.selectedSources) : null,
      fields.recencyWarning,
      fields.routeTokens,
      fields.answerTokens,
      fields.totalTokens,
      fields.latencyMs,
      fields.degradedCacheOnly ? 1 : 0,
    )
    .run();
  return result.meta.last_row_id;
}

/**
 * A per-answer token the widget must present to rate that answer.
 *
 * `query_id` is a sequential integer, so without this anyone can enumerate ids and mass-submit
 * ratings against answers they were never served -- poisoning the one signal being used to
 * judge answer quality. The token is an HMAC over the id, so it can be verified without
 * storing anything extra.
 */
async function feedbackToken(secret: string, queryId: number): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(`q:${queryId}`));
  return [...new Uint8Array(sig).slice(0, 16)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Constant-time string compare, so token verification doesn't leak a byte at a time. */
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

async function handleQuery(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  const start = Date.now();
  let body: { question?: string };
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "invalid JSON body" }, 400);
  }
  const question = sanitizeQuestion((body.question ?? "").trim());
  if (!question) return jsonResponse({ error: "missing 'question'" }, 400);
  if (question.length > 1000) return jsonResponse({ error: "question too long" }, 400);

  const ip = request.headers.get("cf-connecting-ip") ?? "unknown";
  const perHour = parseInt(env.RATE_LIMIT_PER_IP_PER_HOUR, 10) || 20;
  if (!(await checkRateLimit(env, ip, "query", perHour))) {
    return jsonResponse({ error: "rate limit exceeded, try again later" }, 429, {
      "retry-after": "3600",
    });
  }

  const normalized = normalizeQuestion(question);
  // Hashed rather than raw: KV keys cap at 512 bytes and questions are allowed up to 1000
  // characters, so a long question used to throw here -- outside any try/catch, surfacing as
  // a bare 500 with no CORS headers.
  const cacheKey = `answer:${await sha256Hex(normalized)}`;
  const cached = await env.CACHE.get(cacheKey, "json");
  if (cached) {
    return lineStreamResponse(ctx, async (write) => {
      const queryId = await logQuery(env, {
        question,
        normalizedQuestion: normalized,
        cacheHit: true,
        outOfScope: null,
        selectedSources: null,
        recencyWarning: null,
        routeTokens: null,
        answerTokens: null,
        totalTokens: null,
        latencyMs: Date.now() - start,
        degradedCacheOnly: false,
      });
      const token = env.FEEDBACK_SECRET ? await feedbackToken(env.FEEDBACK_SECRET, queryId) : null;
      await write(
        `DATA:${JSON.stringify({ ...(cached as object), cached: true, query_id: queryId, feedback_token: token })}`,
      );
    });
  }

  // Reserve budget up front rather than checking-then-spending: the check and the spend used
  // to be separate eventually-consistent KV operations, so concurrent requests all read the
  // same pre-spend total and sailed past the ceiling together. The reservation is reconciled
  // against real usage once the request finishes (including on the error paths, where the
  // old code silently dropped the routing tokens it had already spent).
  const period = currentPeriod();
  const ceiling = parseInt(env.MONTHLY_TOKEN_CEILING, 10);
  const budget = budgetStub(env);
  const reservation = ROUTE_TOKEN_ESTIMATE + ANSWER_TOKEN_ESTIMATE;
  const { allowed: budgetAllowed } = await budget.reserve(period, ceiling, reservation);
  if (!budgetAllowed) {
    // Degrade to cache-only per SPEC.md 7 -- Gemini won't stop on its own, we have to.
    ctx.waitUntil(
      logQuery(env, {
        question,
        normalizedQuestion: normalized,
        cacheHit: false,
        outOfScope: null,
        selectedSources: null,
        recencyWarning: null,
        routeTokens: null,
        answerTokens: null,
        totalTokens: null,
        latencyMs: Date.now() - start,
        degradedCacheOnly: true,
      }),
    );
    return jsonResponse(
      {
        answer:
          "This tool has reached its monthly usage budget and is temporarily answering only from previously cached questions. Please try again next month, or rephrase your question in case a similar one was already answered.",
        degraded: true,
      },
      503,
      { "retry-after": "86400" },
    );
  }

  // Bound the pipeline, and abandon it entirely if the reader navigates away -- otherwise a
  // fire-and-forget request still bills for a full two-stage generation nobody will read.
  const signal = AbortSignal.any([AbortSignal.timeout(QUERY_TIMEOUT_MS), request.signal]);

  return lineStreamResponse(ctx, async (write) => {
    let spent = 0;
    try {
      const { result: routeResult, tokens: routeTokens } = await route(env, question, signal);
      spent += routeTokens;
      const selectedIds = routeResult.selected.map((s) => s.id);

      if (selectedIds.length === 0 || routeResult.out_of_scope) {
        const payload = {
          answer:
            "NACFE hasn't published research that addresses this question. This is a correct answer, not a limitation of this tool -- see nacfe.org for the full library of what NACFE has studied.",
          selected_sources: [],
          out_of_scope: true,
          recency_warning: routeResult.recency_warning,
        };
        ctx.waitUntil(env.CACHE.put(cacheKey, JSON.stringify(payload), { expirationTtl: 3600 * 24 * 30 }));
        const queryId = await logQuery(env, {
          question,
          normalizedQuestion: normalized,
          cacheHit: false,
          outOfScope: true,
          selectedSources: [],
          recencyWarning: routeResult.recency_warning,
          routeTokens,
          answerTokens: null,
          totalTokens: routeTokens,
          latencyMs: Date.now() - start,
          degradedCacheOnly: false,
        });
        const token = env.FEEDBACK_SECRET ? await feedbackToken(env.FEEDBACK_SECRET, queryId) : null;
        await write(
          `DATA:${JSON.stringify({ ...payload, cached: false, query_id: queryId, feedback_token: token })}`,
        );
        return;
      }

      // Real signal, not a guessed delay: the widget switches its loading text here, exactly
      // when routing has actually finished and the (much longer) answer-generation call
      // actually starts -- there's no further mid-call signal available since reading the
      // documents and writing the response happen inside one continuous Gemini generation.
      await write("STAGE:reading");

      const {
        text: answerText,
        tokens: answerTokens,
        complete,
      } = await answer(env, question, selectedIds, signal);
      spent += answerTokens;

      const payload = {
        answer: answerText,
        selected_sources: enrichSources(routeResult.selected),
        out_of_scope: false,
        recency_warning: routeResult.recency_warning,
      };
      // Only cache a generation the model actually finished. A MAX_TOKENS truncation or a
      // safety stop otherwise gets pinned for 30 days and served to everyone who asks this
      // question again.
      if (complete) {
        ctx.waitUntil(env.CACHE.put(cacheKey, JSON.stringify(payload), { expirationTtl: 3600 * 24 * 30 }));
      }
      const queryId = await logQuery(env, {
        question,
        normalizedQuestion: normalized,
        cacheHit: false,
        outOfScope: false,
        selectedSources: selectedIds,
        recencyWarning: routeResult.recency_warning,
        routeTokens,
        answerTokens,
        totalTokens: routeTokens + answerTokens,
        latencyMs: Date.now() - start,
        degradedCacheOnly: false,
      });
      const token = env.FEEDBACK_SECRET ? await feedbackToken(env.FEEDBACK_SECRET, queryId) : null;
      await write(
        `DATA:${JSON.stringify({ ...payload, cached: false, query_id: queryId, feedback_token: token })}`,
      );
    } catch (err) {
      if (err instanceof GeminiError && (err.isRateLimit || err.isDailyQuota || err.isServerError)) {
        await write(`DATA:${JSON.stringify({ error: "upstream model is temporarily unavailable, try again shortly" })}`);
        return;
      }
      if (err instanceof GeminiEmptyResponse) {
        console.error(err);
        await write(
          `DATA:${JSON.stringify({ error: "the model could not produce an answer for that question" })}`,
        );
        return;
      }
      if (err instanceof Error && (err.name === "AbortError" || err.name === "TimeoutError")) {
        console.error(`query aborted: ${err.name}`);
        await write(`DATA:${JSON.stringify({ error: "the request timed out, try again shortly" })}`);
        return;
      }
      throw err; // handled generically (logged + {"error":"internal error"}) by lineStreamResponse
    } finally {
      // Settle the reservation against what was really spent -- refunding the unused portion,
      // or booking the overrun. Runs on every exit path, so tokens burned by a request that
      // then failed are still counted against the ceiling.
      ctx.waitUntil(budget.add(period, spent - reservation).then(() => undefined));
    }
  });
}

const VALID_RATINGS = ["correct", "partial", "wrong"];
/** Generous next to the query limit -- a reader rates at most once per answer -- but finite,
 * where this endpoint previously had no limit at all. */
const FEEDBACK_RATE_LIMIT_PER_HOUR = 60;

async function handleFeedback(request: Request, env: Env): Promise<Response> {
  if (!env.FEEDBACK_SECRET) {
    console.error("handleFeedback: FEEDBACK_SECRET is not set; feedback is disabled");
    return jsonResponse({ error: "feedback is not configured" }, 503);
  }

  let body: { query_id?: number; rating?: string; token?: string };
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "invalid JSON body" }, 400);
  }

  const ip = request.headers.get("cf-connecting-ip") ?? "unknown";
  if (!(await checkRateLimit(env, ip, "feedback", FEEDBACK_RATE_LIMIT_PER_HOUR))) {
    return jsonResponse({ error: "rate limit exceeded, try again later" }, 429, {
      "retry-after": "3600",
    });
  }

  const queryId = body.query_id;
  const rating = body.rating;
  if (!Number.isInteger(queryId) || (queryId as number) <= 0) {
    return jsonResponse({ error: "missing or invalid 'query_id'" }, 400);
  }
  if (!rating || !VALID_RATINGS.includes(rating)) {
    return jsonResponse({ error: `'rating' must be one of: ${VALID_RATINGS.join(", ")}` }, 400);
  }
  const expected = await feedbackToken(env.FEEDBACK_SECRET, queryId as number);
  if (typeof body.token !== "string" || !timingSafeEqual(body.token, expected)) {
    return jsonResponse({ error: "missing or invalid 'token'" }, 403);
  }

  try {
    // One rating per served answer: each serve gets its own queries row, so a second rating
    // for the same row is the same reader changing their mind, not a second opinion.
    await env.DB.prepare(
      `INSERT INTO feedback (query_id, rating, timestamp) VALUES (?, ?, ?)
       ON CONFLICT (query_id) DO UPDATE SET rating = excluded.rating, timestamp = excluded.timestamp`,
    )
      .bind(queryId, rating, new Date().toISOString())
      .run();
  } catch (err) {
    // Most likely an unknown query_id hitting the foreign key. Previously this escaped as a
    // bare 500 with no CORS headers, which the widget could only report as a network failure.
    console.error(err);
    return jsonResponse({ error: "could not record feedback" }, 400);
  }

  return jsonResponse({ ok: true });
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: corsHeaders({
          "access-control-allow-methods": "POST, GET, OPTIONS",
          "access-control-allow-headers": "content-type",
          "access-control-max-age": "86400",
        }),
      });
    }

    if (url.pathname === "/query" && request.method === "POST") {
      return handleQuery(request, env, ctx);
    }

    if (url.pathname === "/feedback" && request.method === "POST") {
      return handleFeedback(request, env);
    }

    if (url.pathname === "/health") {
      // `feedback` surfaces whether FEEDBACK_SECRET actually made it into the deployment --
      // without it the widget's rating row goes quietly missing, which is easy not to notice.
      return jsonResponse({ ok: true, sources: CATALOG.length, feedback: Boolean(env.FEEDBACK_SECRET) });
    }

    if (url.pathname === "/widget.js" && request.method === "GET") {
      return new Response(WIDGET_JS, {
        headers: corsHeaders({
          "content-type": "application/javascript; charset=utf-8",
          "cache-control": "public, max-age=3600", // an hour: cheap to bust by redeploying, not so long a fix takes all day to land
        }),
      });
    }

    return jsonResponse({ error: "not found" }, 404);
  },
};
