-- Adds per-query cost to the log, for databases created before the ceiling became
-- dollar-denominated. Existing rows keep NULL: their cost was never recorded, and
-- backfilling from token counts would invent precision that was not measured.
--
--   npx wrangler d1 execute nacfe-assist --remote --file=migrations/0002_query_cost.sql

ALTER TABLE queries ADD COLUMN cost_micro_usd INTEGER;
