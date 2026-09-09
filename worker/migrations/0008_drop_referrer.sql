-- Drops queries.referrer.
--
-- Referral sources for nacfe.org come from Google Analytics on the site itself, which sees the
-- whole visit rather than only the moment someone asks a question. Keeping a second, worse
-- copy here has no use, and referrer strings are the most likely of the request-context fields
-- to carry a search query or a private URL -- so the column is removed rather than left unused.
--
-- visitor_id, country and page_url are kept and are now populated; see schema.sql.
--
--   npx wrangler d1 execute nacfe-assist --remote --file=migrations/0008_drop_referrer.sql

ALTER TABLE queries DROP COLUMN referrer;
