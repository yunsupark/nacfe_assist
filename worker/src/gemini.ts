// Minimal Gemini REST client. Workers have native fetch and no reliable Node runtime for
// the full google-genai SDK, so this calls the REST API directly rather than depending on
// a Node-oriented SDK -- see eval/run_eval.py for the equivalent local-script version using
// that SDK, and ingest/ingest_pdf.py / ingest_video.py for the ingest-side equivalent.

export interface GeminiUsage {
  promptTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export interface GeminiResult {
  text: string;
  usage: GeminiUsage;
  /** Gemini's own reason for stopping. Anything other than "STOP" means the text is not a
   * complete answer (MAX_TOKENS truncation, a safety block, a recitation block, ...) and
   * must not be cached and served as one. */
  finishReason: string;
}

export class GeminiError extends Error {
  constructor(
    message: string,
    public status: number,
    public isRateLimit: boolean,
    public isDailyQuota: boolean,
    // 5xx: a transient failure on Google's side (this project has repeatedly seen "high
    // demand" 503s and occasional 500s from Gemini, on both free and paid keys, throughout
    // ingest and now live queries) -- distinct from a rate limit, but just as worth retrying
    // or falling back for rather than failing the request outright.
    public isServerError: boolean,
  ) {
    super(message);
  }
}

/** The HTTP call succeeded but the response carries no usable answer -- blocked by a safety
 * filter, an empty candidate list, or a candidate with no text parts. Distinct from
 * GeminiError (a transport/status failure) because it must never be retried blindly or
 * cached, but it also isn't an outage. */
export class GeminiEmptyResponse extends Error {
  constructor(public reason: string) {
    super(`Gemini returned no usable text (${reason})`);
  }
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function generateContentOnce(
  apiKey: string,
  model: string,
  prompt: string,
  signal?: AbortSignal,
): Promise<GeminiResult> {
  // The key goes in a header, not the query string: a URL carrying the key ends up in
  // proxy/CDN access logs and in any error text that echoes the request URL.
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json", "x-goog-api-key": apiKey },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
    }),
    signal,
  });

  if (!response.ok) {
    const bodyText = await response.text();
    const isRateLimit = response.status === 429;
    const isDailyQuota = isRateLimit && /PerDay/.test(bodyText);
    const isServerError = response.status >= 500;
    throw new GeminiError(
      `Gemini ${model} call failed (${response.status}): ${bodyText}`,
      response.status,
      isRateLimit,
      isDailyQuota,
      isServerError,
    );
  }

  const data = (await response.json()) as {
    candidates?: Array<{
      content?: { parts?: Array<{ text?: string; thought?: boolean }> };
      finishReason?: string;
    }>;
    promptFeedback?: { blockReason?: string };
    usageMetadata?: {
      promptTokenCount?: number;
      candidatesTokenCount?: number;
      totalTokenCount?: number;
    };
  };

  const usage: GeminiUsage = {
    promptTokens: data.usageMetadata?.promptTokenCount ?? 0,
    outputTokens: data.usageMetadata?.candidatesTokenCount ?? 0,
    totalTokens: data.usageMetadata?.totalTokenCount ?? 0,
  };

  if (data.promptFeedback?.blockReason) {
    throw new GeminiEmptyResponse(`prompt blocked: ${data.promptFeedback.blockReason}`);
  }

  const candidate = data.candidates?.[0];
  if (!candidate) throw new GeminiEmptyResponse("no candidates returned");

  // Join every non-thought part. Reading only parts[0] silently drops the answer whenever the
  // model emits a thought part first (thinking models) or splits its output across parts.
  const text = (candidate.content?.parts ?? [])
    .filter((p) => p.thought !== true && typeof p.text === "string")
    .map((p) => p.text as string)
    .join("");

  const finishReason = candidate.finishReason ?? "STOP";
  if (!text.trim()) throw new GeminiEmptyResponse(`empty text (finishReason=${finishReason})`);

  return { text, usage, finishReason };
}

/** One same-key retry on a transient 5xx before giving up -- a single short-lived "high
 * demand" blip (the most common real-world failure this project has seen) often clears on
 * an immediate retry, without even needing to burn the fallback key. */
export async function generateContent(
  apiKey: string,
  model: string,
  prompt: string,
  signal?: AbortSignal,
): Promise<GeminiResult> {
  try {
    return await generateContentOnce(apiKey, model, prompt, signal);
  } catch (err) {
    if (!(err instanceof GeminiError) || !err.isServerError) throw err;
    // Jittered so a burst of requests hitting the same blip doesn't retry in lockstep.
    await sleep(400 + Math.floor(Math.random() * 400));
    if (signal?.aborted) throw err;
    return generateContentOnce(apiKey, model, prompt, signal);
  }
}

/** Strip a ```json ... ``` fence if the model wrapped its JSON response in one. */
export function stripJsonFence(text: string): string {
  return text.trim().replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "");
}

/**
 * Try the free-tier key first, falling back to the paid key when free tier can't serve the
 * call. A KV flag remembers "don't bother with free tier for this model right now" so the
 * rest of the affected window skips straight to paid instead of each request re-proving the
 * same failure first.
 *
 * The flag is keyed per model, not globally: quotas and capacity are per model per project.
 * This project has already seen one model's free-tier daily quota stick while others on the
 * same key kept working (see ROUTE_MODEL in eval/run_eval.py), so disabling free tier
 * wholesale because one model ran out would give up free capacity that is still there.
 *
 * Two reasons to set it, with very different lifetimes:
 *
 *  - "daily-quota": free tier's own daily cap. 24h, because it's a flat TTL from the moment
 *    we detect exhaustion -- free tier's daily quota has been observed (see
 *    eval/results/two_stage_scored.md) not to reset exactly on schedule, so that is more
 *    robust than computing time-until-next-midnight-PT.
 *  - "outage": a 5xx that survived generateContent's own same-key retry, i.e. the model is
 *    unavailable on the free tier specifically. Measured 2026-09-04: the routing model
 *    returned 503 "high demand" on 5/5 free-key attempts while the identical call succeeded
 *    on the paid key. Without this flag every uncached query pays two failed free-tier
 *    requests plus the retry sleep (~600ms) before falling through to paid, on every query,
 *    for as long as the outage lasts. Short TTL, since unlike a quota this is expected to
 *    clear on its own and we want to start using free tier again promptly when it does.
 *
 * A plain per-minute rate limit sets no flag at all: it falls back to paid for just this one
 * call, since free tier should recover within the minute and marking the model unusable over
 * a transient burst would be overly aggressive.
 */
const freeSkipKey = (model: string) => `gemini_free_skip:${model}`;
const FREE_TIER_QUOTA_TTL_SECONDS = 3600 * 24;
const FREE_TIER_OUTAGE_TTL_SECONDS = 600;
/**
 * How long a single free-tier attempt may run before it is abandoned in favour of the paid key.
 *
 * The free attempt used to be handed the caller's whole deadline. A free call that STALLS --
 * as opposed to returning 503 promptly, which is the failure this project usually sees -- then
 * consumes the entire request budget, and the paid fallback never runs at all. The request
 * fails with a timeout even though the paid key would have answered in ~20s. That is the
 * likely shape of the timeout observed on 2026-09-09, whose retry succeeded in 31.5s once the
 * skip flag had already routed it straight to paid.
 *
 * 35s is generous against observed healthy latency (routing ~8-20s, answering ~10-30s) so it
 * only trips on a genuine stall, and it still leaves the majority of a 90s request budget for
 * the paid attempt that follows.
 */
const FREE_TIER_ATTEMPT_TIMEOUT_MS = 35_000;

export async function generateContentWithFallback(
  cache: KVNamespace,
  freeApiKey: string,
  paidApiKey: string,
  model: string,
  prompt: string,
  signal?: AbortSignal,
): Promise<GeminiResult> {
  const skipKey = freeSkipKey(model);
  const skip = freeApiKey ? await cache.get(skipKey) : "no-free-key";
  if (!skip) {
    // The free attempt runs against its own shorter deadline, so a stalled free call cannot
    // spend the caller's entire budget and starve the paid fallback.
    const freeDeadline = AbortSignal.timeout(FREE_TIER_ATTEMPT_TIMEOUT_MS);
    const freeSignal = signal ? AbortSignal.any([signal, freeDeadline]) : freeDeadline;
    try {
      return await generateContent(freeApiKey, model, prompt, freeSignal);
    } catch (err) {
      // The caller's own deadline expired or the client went away: this is not a free-tier
      // problem and retrying on the paid key would only bill for an answer nobody waits for.
      if (signal?.aborted) throw err;

      if (freeDeadline.aborted) {
        console.error(
          `free tier stalled for ${model} (no response in ${FREE_TIER_ATTEMPT_TIMEOUT_MS}ms); ` +
            `abandoning it for ${FREE_TIER_OUTAGE_TTL_SECONDS}s and using the paid key`,
        );
        await cache.put(skipKey, "stalled", { expirationTtl: FREE_TIER_OUTAGE_TTL_SECONDS });
      } else if (!(err instanceof GeminiError) || !(err.isRateLimit || err.isServerError)) {
        throw err;
      } else if (err.isDailyQuota) {
        await cache.put(skipKey, "daily-quota", { expirationTtl: FREE_TIER_QUOTA_TTL_SECONDS });
      } else if (err.isServerError) {
        console.error(`free tier unavailable for ${model} (${err.status}); skipping it for ${FREE_TIER_OUTAGE_TTL_SECONDS}s`);
        await cache.put(skipKey, "outage", { expirationTtl: FREE_TIER_OUTAGE_TTL_SECONDS });
      }
      // fall through to the paid key below
    }
  }
  return generateContent(paidApiKey, model, prompt, signal);
}
