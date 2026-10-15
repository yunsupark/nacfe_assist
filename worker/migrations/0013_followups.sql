-- How many earlier questions in the same conversation preceded this one (0 = a fresh question,
-- 1 = the first follow-up, ...), as counted by the widget; falls back to the number of prior turns
-- sent when an older widget doesn't report it. Nullable: rows before this migration have no value.
-- Response time needs no new column: queries.latency_ms already records it.
-- Run this BEFORE deploying the Worker that writes the column (see logQuery's comment).
ALTER TABLE queries ADD COLUMN followups INTEGER;
