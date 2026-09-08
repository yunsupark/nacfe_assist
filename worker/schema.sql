-- Per SPEC.md 7: "D1 logs every question, selected sources, token counts, and latency.
-- What the public asks NACFE is itself a research finding -- treat the log as an output,
-- not telemetry." No raw IP is stored here (only used transiently in KV for rate
-- limiting) -- what the public asked is the research asset, not who asked it.

CREATE TABLE IF NOT EXISTS queries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,               -- ISO 8601
  question TEXT NOT NULL,
  normalized_question TEXT NOT NULL,
  cache_hit INTEGER NOT NULL,            -- 0/1
  out_of_scope INTEGER,                  -- 0/1, null if cache hit or degraded
  selected_sources TEXT,                 -- JSON array of source ids, null if cache hit
  recency_warning TEXT,
  route_tokens INTEGER,
  answer_tokens INTEGER,
  total_tokens INTEGER,
  latency_ms INTEGER NOT NULL,
  degraded_cache_only INTEGER NOT NULL DEFAULT 0, -- 1 if served under the monthly ceiling
  -- What this query actually cost, in integer micro-USD (1e-6 USD). The monthly ceiling is
  -- denominated in cost rather than tokens (see worker/src/pricing.ts), so this is what the
  -- ceiling is actually counting, and it makes spend auditable per question. 0 for cache hits.
  cost_micro_usd INTEGER
);

CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries (timestamp);
CREATE INDEX IF NOT EXISTS idx_queries_normalized_question ON queries (normalized_question);

-- Widget feedback, one row per rating a reader submits. Linked to the specific queries row
-- that was actually served (not just the question text) since the same question can be
-- answered differently over time and cache hits get their own fresh queries row per serve.
-- Anonymous like the queries table itself -- no IP or user identity, just the rating.
--
-- query_id is UNIQUE: each serve gets its own queries row, so a second rating for the same
-- row is one reader changing their mind, not a second opinion. The Worker upserts. Together
-- with the HMAC feedback token it requires (see feedbackToken in src/index.ts) this stops
-- anyone from enumerating sequential query ids and mass-submitting ratings against answers
-- they were never served.
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query_id INTEGER NOT NULL REFERENCES queries (id),
  rating TEXT NOT NULL CHECK (rating IN ('yes', 'partly', 'no')),  -- helpfulness, not accuracy
  timestamp TEXT NOT NULL  -- ISO 8601
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback (query_id);

-- Widget-level events: impressions (widget loads) and sponsor-link clicks. See
-- migrations/0003_events.sql for why this is separate from the query log, and why it
-- deliberately carries no visitor identifier.

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,             -- ISO 8601
  day TEXT NOT NULL,                   -- YYYY-MM-DD, denormalized so daily rollups don't scan
  type TEXT NOT NULL CHECK (type IN ('impression', 'sponsor_click')),
  page_url TEXT,                       -- embedding page, scheme+host+path only (no query string)
  country TEXT                         -- 2-letter code from Cloudflare, or null
);

CREATE INDEX IF NOT EXISTS idx_events_day_type ON events (day, type);
CREATE INDEX IF NOT EXISTS idx_events_type_timestamp ON events (type, timestamp);
