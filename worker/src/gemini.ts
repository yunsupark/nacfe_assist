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
 * Try the free-tier key first, falling back to the paid key on quota exhaustion. A KV flag
 * remembers "free tier is out for today" once we see an actual daily-quota error, so the rest
 * of the day's requests skip straight to paid instead of each eating a failed free-tier call
 * first -- free tier's own daily quota has been observed (see eval/results/two_stage_scored.md)
 * to not always reset exactly on schedule, so a flat 24h TTL from the moment we detect
 * exhaustion is more robust than trying to compute time-until-next-midnight-PT.
 *
 * A plain (non-daily) rate limit -- a per-minute burst -- does NOT set that flag: it falls
 * back to paid for just this one call, since free tier should recover within the minute and
 * marking the whole day exhausted over a transient burst would be overly aggressive.
 */
const FREE_TIER_EXHAUSTED_KEY = "gemini_free_tier_exhausted";
const FREE_TIER_EXHAUSTED_TTL_SECONDS = 3600 * 24;

export async function generateContentWithFallback(
  cache: KVNamespace,
  freeApiKey: string,
  paidApiKey: string,
  model: string,
  prompt: string,
  signal?: AbortSignal,
): Promise<GeminiResult> {
  const exhausted = await cache.get(FREE_TIER_EXHAUSTED_KEY);
  if (!exhausted && freeApiKey) {
    try {
      return await generateContent(freeApiKey, model, prompt, signal);
    } catch (err) {
      if (!(err instanceof GeminiError) || !(err.isRateLimit || err.isServerError)) throw err;
      if (err.isDailyQuota) {
        await cache.put(FREE_TIER_EXHAUSTED_KEY, "1", { expirationTtl: FREE_TIER_EXHAUSTED_TTL_SECONDS });
      }
      // fall through to the paid key below -- for a rate limit (daily or per-minute) or a
      // transient server error that survived generateContent's own same-key retry
    }
  }
  return generateContent(paidApiKey, model, prompt, signal);
}
