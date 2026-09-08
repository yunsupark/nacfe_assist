-- Re-vocabularies feedback from accuracy to helpfulness.
--
-- "Was this answer accurate? Correct / Partially correct / Wrong" is the right question for
-- internal QA and the wrong one for the public. A reader asks because they do not know the
-- answer, so asking them to grade correctness collects confident-sounding noise from people
-- with no way to judge -- and "Wrong" on a NACFE-branded page frames the research itself as
-- unreliable. Correctness is already measured properly by the expert eval in eval/.
--
-- Public feedback now asks "Was this helpful?" -> yes / partly / no, which a reader can
-- actually answer, and which stays meaningful when the correct answer is "NACFE hasn't
-- studied that" (a "did this answer your question?" framing would score that a failure even
-- though the tool did exactly the right thing).
--
-- SQLite cannot alter a CHECK constraint in place, so the table is rebuilt. Existing rows are
-- preserved and mapped ordinally (correct->yes, partial->partly, wrong->no). They are NOT
-- equivalent judgments: every row predating this migration is an internal tester grading
-- accuracy, not a reader reporting helpfulness. Filter on timestamp before mixing the two
-- eras, or delete the pre-launch rows outright -- they are internal test data.
--
--   npx wrangler d1 execute nacfe-assist --remote --file=migrations/0005_feedback_helpfulness.sql

CREATE TABLE feedback_new (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query_id INTEGER NOT NULL REFERENCES queries (id),
  rating TEXT NOT NULL CHECK (rating IN ('yes', 'partly', 'no')),
  timestamp TEXT NOT NULL
);

INSERT INTO feedback_new (id, query_id, rating, timestamp)
SELECT id, query_id,
       CASE rating WHEN 'correct' THEN 'yes' WHEN 'partial' THEN 'partly' ELSE 'no' END,
       timestamp
FROM feedback;

DROP TABLE feedback;
ALTER TABLE feedback_new RENAME TO feedback;
CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback (query_id);
