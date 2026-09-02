-- Migration for databases created before the security review.
--
-- Makes idx_feedback_query_id UNIQUE so the Worker's upsert (ON CONFLICT (query_id) DO
-- UPDATE) has a conflict target, giving one rating per served answer instead of unlimited
-- ratings per id. Deduplicates any existing rows first, keeping the most recent rating for
-- each query_id -- SQLite cannot build a unique index over duplicates.
--
-- Apply with:
--   npx wrangler d1 execute nacfe-assist --remote --file=migrations/0001_feedback_unique_query_id.sql

DELETE FROM feedback
WHERE id NOT IN (SELECT MAX(id) FROM feedback GROUP BY query_id);

DROP INDEX IF EXISTS idx_feedback_query_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback (query_id);
