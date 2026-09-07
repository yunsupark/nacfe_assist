// Gemini list prices, and the conversion from token usage to money.
//
// The monthly ceiling used to be denominated in tokens, which is a poor proxy for spend: the
// prices below span 12.5x between the routing model's input and the answering model's output,
// so the same token count can differ several-fold in cost depending on where it was spent.
// The ceiling is a cost control, so it is denominated in cost.
//
// Kept as a code constant rather than env vars: prices are reviewable in diffs, change rarely,
// and a change should be a deliberate deploy rather than a silent var edit. Source:
// https://ai.google.dev/gemini-api/docs/pricing
//
// Money is carried as integer micro-USD (1e-6 USD) everywhere, never floats. A month's
// spend at any plausible volume stays far inside Number.MAX_SAFE_INTEGER, and integers mean
// the running total can't drift the way repeated float addition does.

export interface ModelPrice {
  /** micro-USD per input token */
  inputPerToken: number;
  /** micro-USD per output token */
  outputPerToken: number;
}

const perMillion = (usd: number): number => usd; // $/1M tokens == micro-USD per token

export const MODEL_PRICING: Record<string, ModelPrice> = {
  // $0.30 in / $2.50 out per 1M tokens
  "gemini-3.5-flash-lite": { inputPerToken: perMillion(0.3), outputPerToken: perMillion(2.5) },
  // $0.75 in / $3.75 out per 1M tokens THROUGH 2026-12-31. On 2027-01-01 these double, to
  // $1.50 / $7.50 -- per-query cost rises roughly 40%. Update this entry then, and revisit
  // MONTHLY_COST_CEILING_USD, which buys ~30% fewer questions at the new prices.
  "gemini-3.6-flash": { inputPerToken: perMillion(0.75), outputPerToken: perMillion(3.75) },
};

/** The most expensive rates we know about, used when a model isn't in the table. */
const FALLBACK_PRICE: ModelPrice = Object.values(MODEL_PRICING).reduce(
  (worst, p) => ({
    inputPerToken: Math.max(worst.inputPerToken, p.inputPerToken),
    outputPerToken: Math.max(worst.outputPerToken, p.outputPerToken),
  }),
  { inputPerToken: 0, outputPerToken: 0 },
);

/**
 * Cost of one model call, in integer micro-USD, rounded up.
 *
 * An unknown model (someone changed ROUTE_MODEL/ANSWER_MODEL without touching this table)
 * bills at the most expensive known rate rather than zero. Charging nothing for a model we
 * don't recognise would silently disable the ceiling, which is the one failure this control
 * exists to prevent.
 */
export function costMicroUsd(model: string, promptTokens: number, outputTokens: number): number {
  const price = MODEL_PRICING[model];
  if (!price) {
    console.error(`pricing: no entry for model "${model}"; billing at the highest known rate`);
  }
  const { inputPerToken, outputPerToken } = price ?? FALLBACK_PRICE;
  return Math.ceil(promptTokens * inputPerToken + outputTokens * outputPerToken);
}

export const microUsdToUsd = (micro: number): number => micro / 1_000_000;

/** Format micro-USD as a plain dollar string for logs and operator-facing output. */
export const formatUsd = (micro: number): string => `$${microUsdToUsd(micro).toFixed(2)}`;
