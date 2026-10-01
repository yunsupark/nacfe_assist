-- Adds 'widget_expand' to the events.type CHECK constraint.
--
-- The compact corner-launcher embed (WordPress demo) shows a small teaser card by default and
-- opens the full Q&A panel on click. That open action is worth counting on its own: it is the
-- point between "saw the teaser" (already counted as an impression) and "asked a question" --
-- without it there is no way to tell whether a low question count means the teaser isn't
-- getting clicked, or people open the panel and then don't type anything.
--
-- SQLite cannot alter a CHECK constraint in place, so the table is rebuilt, same approach as
-- migrations/0005_feedback_helpfulness.sql. All existing rows carry only the two prior types and
-- are preserved unchanged.
--
--   npx wrangler d1 execute nacfe-assist --remote --file=migrations/0010_widget_expand_event.sql

CREATE TABLE events_new (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  day TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('impression', 'sponsor_click', 'widget_expand')),
  page_url TEXT,
  country TEXT
);

INSERT INTO events_new (id, timestamp, day, type, page_url, country)
SELECT id, timestamp, day, type, page_url, country FROM events;

DROP TABLE events;
ALTER TABLE events_new RENAME TO events;

CREATE INDEX IF NOT EXISTS idx_events_day_type ON events (day, type);
CREATE INDEX IF NOT EXISTS idx_events_type_timestamp ON events (type, timestamp);
