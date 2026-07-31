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
}

export class GeminiError extends Error {
  constructor(
    message: string,
    public status: number,
    public isRateLimit: boolean,
    public isDailyQuota: boolean,
  ) {
    super(message);
  }
}

export async function generateContent(
  apiKey: string,
  model: string,
  prompt: string,
): Promise<GeminiResult> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
    }),
  });

  if (!response.ok) {
    const bodyText = await response.text();
    const isRateLimit = response.status === 429;
    const isDailyQuota = isRateLimit && /PerDay/.test(bodyText);
    throw new GeminiError(
      `Gemini ${model} call failed (${response.status}): ${bodyText}`,
      response.status,
      isRateLimit,
      isDailyQuota,
    );
  }

  const data = (await response.json()) as {
    candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
    usageMetadata?: {
      promptTokenCount?: number;
      candidatesTokenCount?: number;
      totalTokenCount?: number;
    };
  };

  const text = data.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  const usage: GeminiUsage = {
    promptTokens: data.usageMetadata?.promptTokenCount ?? 0,
    outputTokens: data.usageMetadata?.candidatesTokenCount ?? 0,
    totalTokens: data.usageMetadata?.totalTokenCount ?? 0,
  };

  return { text, usage };
}

/** Strip a ```json ... ``` fence if the model wrapped its JSON response in one. */
export function stripJsonFence(text: string): string {
  return text.trim().replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "");
}
