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
  degraded_cache_only INTEGER NOT NULL DEFAULT 0  -- 1 if served under the monthly ceiling
);

CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries (timestamp);
CREATE INDEX IF NOT EXISTS idx_queries_normalized_question ON queries (normalized_question);
