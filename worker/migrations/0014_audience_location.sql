-- Who asked and roughly where from.
--   audience: 'internal' when the question carried a valid NACFE staff pass (see verifyStaffPass in
--             src/index.ts), otherwise 'public'. NULL on rows from before this migration; the
--             admin console treats NULL as 'public'.
--   city, region: Cloudflare's approximate, IP-based location for the request (country was already
--             logged). Coarse by design: no coordinates, no postal code.
-- Run this BEFORE deploying the Worker that writes these columns (see logQuery's comment).
ALTER TABLE queries ADD COLUMN audience TEXT;
ALTER TABLE queries ADD COLUMN city TEXT;
ALTER TABLE queries ADD COLUMN region TEXT;
CREATE INDEX IF NOT EXISTS idx_queries_audience_timestamp ON queries (audience, timestamp);
