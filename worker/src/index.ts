// NACFE Knowledge Base query API. Two-stage loop per SPEC.md 4/5: cheap-model routing over
// the full catalog, then a stronger model answering from the selected sources' full text.
// See eval/run_eval.py for the local-script version this was ported from, and
// eval/results/two_stage_scored.md for the eval this design is validated against.
import { CATALOG, ROUTE_PROMPT, ANSWER_PROMPT, WIDGET_JS } from "./corpus_data";
import { generateContentWithFallback, stripJsonFence, GeminiError } from "./gemini";

export interface Env {
  GEMINI_API: string;
  GEMINI_API_FREE: string;
  CACHE: KVNamespace;
  DB: D1Database;
  SOURCES_BUCKET: R2Bucket;
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

function normalizeQuestion(q: string): string {
  return q.trim().toLowerCase().replace(/\s+/g, " ").replace(/[?.!]+$/, "");
}

/** Attach each selected source's catalog url (null if the source has none) so the widget can
 * render a real link instead of plain text. */
function enrichSources(
  selected: Array<{ id: string; why: string }>,
): Array<{ id: string; why: string; url: string | null }> {
  const catalogById = new Map(CATALOG.map((c) => [c.id, c]));
  return selected.map((s) => ({ ...s, url: catalogById.get(s.id)?.url ?? null }));
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json",
      "access-control-allow-origin": "*", // the embeddable widget (SPEC.md /web/) is meant to be embedded cross-origin
    },
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
    headers: {
      "content-type": "text/plain; charset=utf-8",
      "access-control-allow-origin": "*",
      "cache-control": "no-store",
    },
  });
}

async function hashIp(ip: string): Promise<string> {
  const data = new TextEncoder().encode(ip);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function checkRateLimit(env: Env, ip: string): Promise<boolean> {
  const ipHash = await hashIp(ip);
  const hourBucket = Math.floor(Date.now() / 3_600_000);
  const key = `ratelimit:${ipHash}:${hourBucket}`;
  const limit = parseInt(env.RATE_LIMIT_PER_IP_PER_HOUR, 10);
  const current = parseInt((await env.CACHE.get(key)) ?? "0", 10);
  if (current >= limit) return false;
  await env.CACHE.put(key, String(current + 1), { expirationTtl: 3600 });
  return true;
}

async function getMonthlyTokenUsage(env: Env): Promise<{ key: string; used: number }> {
  const month = new Date().toISOString().slice(0, 7); // YYYY-MM
  const key = `budget:${month}`;
  const used = parseInt((await env.CACHE.get(key)) ?? "0", 10);
  return { key, used };
}

async function addMonthlyTokenUsage(env: Env, key: string, tokens: number): Promise<void> {
  const current = parseInt((await env.CACHE.get(key)) ?? "0", 10);
  // ~40 days covers a full calendar month regardless of when in the month this write lands.
  await env.CACHE.put(key, String(current + tokens), { expirationTtl: 3600 * 24 * 40 });
}

async function route(env: Env, question: string): Promise<{ result: RouteResult; tokens: number }> {
  const prompt = ROUTE_PROMPT.replace("{{catalog}}", JSON.stringify(CATALOG, null, 2)).replace(
    "{{question}}",
    question,
  );
  const { text, usage } = await generateContentWithFallback(
    env.CACHE,
    env.GEMINI_API_FREE,
    env.GEMINI_API,
    env.ROUTE_MODEL,
    prompt,
  );
  const parsed = JSON.parse(stripJsonFence(text)) as RouteResult;
  return { result: parsed, tokens: usage.totalTokens };
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
): Promise<{ text: string; tokens: number }> {
  const catalogById = new Map(CATALOG.map((c) => [c.id, c]));
  // Fetch the (at most 6) selected sources from R2 in parallel -- sequential round-trips
  // would otherwise stack their latency on top of each other for no reason.
  const bodies = await Promise.all(selectedIds.map((id) => fetchSource(env, id)));
  const documents = selectedIds
    .map((id, i) => {
      const entry = catalogById.get(id);
      const title = entry?.title ?? id;
      const body = bodies[i];
      if (!body) return null;
      return `=== SOURCE: ${title} [${id}] ===\n${body}`;
    })
    .filter((d): d is string => d !== null)
    .join("\n\n");

  const currentYear = String(new Date().getFullYear());
  const prompt = ANSWER_PROMPT.replace("{{current_year}}", currentYear)
    .replace("{{documents}}", documents)
    .replace("{{question}}", question);
  const { text, usage } = await generateContentWithFallback(
    env.CACHE,
    env.GEMINI_API_FREE,
    env.GEMINI_API,
    env.ANSWER_MODEL,
    prompt,
  );
  return { text, tokens: usage.totalTokens };
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

async function handleQuery(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  const start = Date.now();
  let body: { question?: string };
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "invalid JSON body" }, 400);
  }
  const question = (body.question ?? "").trim();
  if (!question) return jsonResponse({ error: "missing 'question'" }, 400);
  if (question.length > 1000) return jsonResponse({ error: "question too long" }, 400);

  const ip = request.headers.get("cf-connecting-ip") ?? "unknown";
  const allowed = await checkRateLimit(env, ip);
  if (!allowed) return jsonResponse({ error: "rate limit exceeded, try again later" }, 429);

  const normalized = normalizeQuestion(question);
  const cacheKey = `answer:${normalized}`;
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
      await write(`DATA:${JSON.stringify({ ...(cached as object), cached: true, query_id: queryId })}`);
    });
  }

  const { key: budgetKey, used: monthlyUsed } = await getMonthlyTokenUsage(env);
  const ceiling = parseInt(env.MONTHLY_TOKEN_CEILING, 10);
  if (monthlyUsed >= ceiling) {
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
    );
  }

  return lineStreamResponse(ctx, async (write) => {
    try {
      const { result: routeResult, tokens: routeTokens } = await route(env, question);
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
        ctx.waitUntil(addMonthlyTokenUsage(env, budgetKey, routeTokens));
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
        await write(`DATA:${JSON.stringify({ ...payload, cached: false, query_id: queryId })}`);
        return;
      }

      // Real signal, not a guessed delay: the widget switches its loading text here, exactly
      // when routing has actually finished and the (much longer) answer-generation call
      // actually starts -- there's no further mid-call signal available since reading the
      // documents and writing the response happen inside one continuous Gemini generation.
      await write("STAGE:reading");

      const { text: answerText, tokens: answerTokens } = await answer(env, question, selectedIds);
      const totalTokens = routeTokens + answerTokens;

      const payload = {
        answer: answerText,
        selected_sources: enrichSources(routeResult.selected),
        out_of_scope: false,
        recency_warning: routeResult.recency_warning,
      };
      ctx.waitUntil(env.CACHE.put(cacheKey, JSON.stringify(payload), { expirationTtl: 3600 * 24 * 30 }));
      ctx.waitUntil(addMonthlyTokenUsage(env, budgetKey, totalTokens));
      const queryId = await logQuery(env, {
        question,
        normalizedQuestion: normalized,
        cacheHit: false,
        outOfScope: false,
        selectedSources: selectedIds,
        recencyWarning: routeResult.recency_warning,
        routeTokens,
        answerTokens,
        totalTokens,
        latencyMs: Date.now() - start,
        degradedCacheOnly: false,
      });
      await write(`DATA:${JSON.stringify({ ...payload, cached: false, query_id: queryId })}`);
    } catch (err) {
      if (err instanceof GeminiError && (err.isRateLimit || err.isDailyQuota || err.isServerError)) {
        await write(`DATA:${JSON.stringify({ error: "upstream model is temporarily unavailable, try again shortly" })}`);
        return;
      }
      throw err; // handled generically (logged + {"error":"internal error"}) by lineStreamResponse
    }
  });
}

const VALID_RATINGS = ["correct", "partial", "wrong"];

async function handleFeedback(request: Request, env: Env): Promise<Response> {
  let body: { query_id?: number; rating?: string };
  try {
    body = await request.json();
  } catch {
    return jsonResponse({ error: "invalid JSON body" }, 400);
  }
  const queryId = body.query_id;
  const rating = body.rating;
  if (!Number.isInteger(queryId) || (queryId as number) <= 0) {
    return jsonResponse({ error: "missing or invalid 'query_id'" }, 400);
  }
  if (!rating || !VALID_RATINGS.includes(rating)) {
    return jsonResponse({ error: `'rating' must be one of: ${VALID_RATINGS.join(", ")}` }, 400);
  }

  await env.DB.prepare(`INSERT INTO feedback (query_id, rating, timestamp) VALUES (?, ?, ?)`)
    .bind(queryId, rating, new Date().toISOString())
    .run();

  return jsonResponse({ ok: true });
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "POST, OPTIONS",
          "access-control-allow-headers": "content-type",
        },
      });
    }

    if (url.pathname === "/query" && request.method === "POST") {
      return handleQuery(request, env, ctx);
    }

    if (url.pathname === "/feedback" && request.method === "POST") {
      return handleFeedback(request, env);
    }

    if (url.pathname === "/health") {
      return jsonResponse({ ok: true, sources: CATALOG.length });
    }

    if (url.pathname === "/widget.js" && request.method === "GET") {
      return new Response(WIDGET_JS, {
        headers: {
          "content-type": "application/javascript; charset=utf-8",
          "cache-control": "public, max-age=3600", // an hour: cheap to bust by redeploying, not so long a fix takes all day to land
          "access-control-allow-origin": "*",
        },
      });
    }

    return jsonResponse({ error: "not found" }, 404);
  },
};
