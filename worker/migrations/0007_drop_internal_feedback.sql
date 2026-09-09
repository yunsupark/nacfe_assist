-- Removes internal pre-launch feedback.
--
-- Rows 1-14 (2026-08-14 to 2026-08-19) are internal testers answering the old question, "Was
-- this answer accurate? Correct / Partially correct / Wrong". Migration 0005 mapped them
-- ordinally onto the new helpfulness vocabulary so nothing was lost at the time, but they are
-- not the same judgment: an internal reviewer grading correctness against the sources is
-- measuring something the public question does not ask. Pooling them would put a 93% "yes"
-- rate from staff underneath the first real readers and quietly flatter the metric.
--
-- The 2026-09-09 row is kept: it is a genuine post-launch rating given through the deployed
-- helpfulness prompt.
--
-- Backed up before deletion; correctness continues to be measured by the expert eval.
--
--   npx wrangler d1 execute nacfe-assist --remote --file=migrations/0007_drop_internal_feedback.sql

DELETE FROM feedback WHERE timestamp < '2026-09-01';
