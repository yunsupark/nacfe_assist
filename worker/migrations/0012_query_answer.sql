-- Stores the answer text that was served for each query, so the admin console can show what a
-- reader was actually told when they rated it "partly" or "no" (feedback.query_id -> queries.id).
-- Nullable: rows logged before this migration, degraded (budget-limited) and failed requests
-- have no answer. Run this BEFORE deploying the Worker that writes the column; against an
-- unmigrated database every query-log insert would fail (see logQuery's comment).
ALTER TABLE queries ADD COLUMN answer TEXT;
