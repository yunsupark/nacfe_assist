-- Records why a query failed.
--
-- Until now a failed query wrote no row at all: the error branches streamed a message to the
-- reader and returned without logging. A timeout reported on 2026-09-09 left no trace in the
-- database, so the failure rate was unmeasurable and the cause had to be reconstructed from
-- live tailing after the fact.
--
--   npx wrangler d1 execute nacfe-assist --remote --file=migrations/0006_query_error.sql

ALTER TABLE queries ADD COLUMN error TEXT;
