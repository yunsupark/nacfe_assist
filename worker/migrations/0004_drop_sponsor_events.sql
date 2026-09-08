-- Drops sponsor_events, superseded by the events table from migration 0003.
--
-- sponsor_events predates this branch and was never referenced by any code in the repo. It
-- held one row, a QA test impression from 2026-08-21 ("Test Sponsor QA"). It could not serve
-- the current plan in any case: sponsor_name is NOT NULL, so it cannot record an impression
-- while running without a sponsor, which is the phase the widget is entering.
--
-- events replaces it and additionally records country, denormalizes the day for cheap
-- rollups, and stores no visitor identifier.
--
--   npx wrangler d1 execute nacfe-assist --remote --file=migrations/0004_drop_sponsor_events.sql

DROP TABLE IF EXISTS sponsor_events;
