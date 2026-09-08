-- Widget-level events, separate from the query log.
--
-- The queries table records research questions. This records the things a sponsor asks about
-- and questions cannot answer: how many people SAW the widget (impressions), and how many
-- clicked the sponsor link. Impressions are the larger number by far -- most readers never
-- type anything -- so reporting reach from question counts would understate it badly.
--
-- Deliberately carries no visitor identifier. Country and page are coarse and non-identifying,
-- and keeping this table free of per-person ids means unique-visitor counting stays a separate
-- decision rather than something this table quietly presupposes.
--
--   npx wrangler d1 execute nacfe-assist --remote --file=migrations/0003_events.sql

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
