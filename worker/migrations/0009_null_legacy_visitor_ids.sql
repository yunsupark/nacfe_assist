-- Clears the persistent per-browser visitor ids written by the previous tracking code.
--
-- Eight rows (2026-08-22 to 2026-09-03) carry 36- and 17-character UUIDs generated in the
-- browser and kept in localStorage. They are durable device identifiers: the same value
-- follows a person across months and across questions, which is the property the current
-- month-scoped HMAC scheme exists to avoid. Keeping them would mean the database holds
-- exactly the kind of identifier the project has decided not to collect, and the two schemes
-- cannot be counted together -- a report over both silently mixed the definitions and read
-- 2 unique visitors for 2026-09 when only one row used the current scheme.
--
-- Only the identifier is removed. The questions, timings, costs, country and page_url on
-- those rows are research data and stay. Backed up before running.
--
--   npx wrangler d1 execute nacfe-assist --remote --file=migrations/0009_null_legacy_visitor_ids.sql

UPDATE queries SET visitor_id = NULL WHERE visitor_id IS NOT NULL AND LENGTH(visitor_id) <> 16;
