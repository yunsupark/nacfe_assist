// Atomic counters for the two things that stand between a public URL and an unbounded
// Gemini bill: the per-IP rate limit and the monthly token ceiling.
//
// Both were previously KV read-modify-write (`get` -> `+1` -> `put`). KV is eventually
// consistent and edge-cached for up to ~60s, so concurrent requests all read the same stale
// value and requests landing in different Cloudflare PoPs keep independent counters --
// making both limits advisory at best. A Durable Object is single-threaded per instance with
// strongly consistent storage, so check-and-increment is genuinely atomic.
//
// Addressing: one RateLimiter instance per hashed IP (so limits don't serialize against each
// other), one Budget instance globally (it has to be a single shared total).
import { DurableObject } from "cloudflare:workers";

interface Window {
  bucket: string;
  count: number;
}

export class RateLimiter extends DurableObject {
  /**
   * Atomically increment this IP's counter for `bucket` (an hour stamp) and report whether
   * the request is allowed. A bucket change resets the count, so only one record is ever
   * stored per IP -- no key accumulation to clean up.
   */
  async checkAndIncrement(bucket: string, limit: number): Promise<{ allowed: boolean; count: number }> {
    const stored = (await this.ctx.storage.get<Window>("w")) ?? { bucket, count: 0 };
    const window: Window = stored.bucket === bucket ? stored : { bucket, count: 0 };

    if (window.count >= limit) return { allowed: false, count: window.count };

    window.count += 1;
    await this.ctx.storage.put("w", window);
    // Tidy up an idle IP's record rather than leaving it resident forever.
    await this.ctx.storage.setAlarm(Date.now() + 2 * 3600 * 1000);
    return { allowed: true, count: window.count };
  }

  async alarm(): Promise<void> {
    await this.ctx.storage.deleteAll();
  }
}

export class Budget extends DurableObject {
  /** Tokens spent so far in `period` (a YYYY-MM stamp). A period change resets the total. */
  async used(period: string): Promise<number> {
    const stored = await this.ctx.storage.get<Window>("w");
    return stored && stored.bucket === period ? stored.count : 0;
  }

  /** Atomically add `delta` tokens to `period`'s total and return the new total. */
  async add(period: string, delta: number): Promise<number> {
    const stored = await this.ctx.storage.get<Window>("w");
    const window: Window =
      stored && stored.bucket === period ? stored : { bucket: period, count: 0 };
    window.count += delta;
    await this.ctx.storage.put("w", window);
    return window.count;
  }

  /**
   * Reserve budget for a request before spending it: allows the request only if the period
   * is still under `ceiling`, and immediately books an `estimate` so concurrent requests
   * can't all slip through on the same pre-spend reading. The caller reconciles the estimate
   * against actual usage once the request finishes.
   */
  async reserve(
    period: string,
    ceiling: number,
    estimate: number,
  ): Promise<{ allowed: boolean; used: number }> {
    const current = await this.used(period);
    if (current >= ceiling) return { allowed: false, used: current };
    const used = await this.add(period, estimate);
    return { allowed: true, used };
  }
}
