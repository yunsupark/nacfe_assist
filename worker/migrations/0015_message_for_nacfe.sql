-- Messages meant for NACFE's people rather than research questions (a hacked social account, a broken
-- link, "how do I reach someone"). The router flags them (message_for_nacfe in its JSON); the
-- widget answers with a pointer to NACFE's contact page, and the admin console's Messages tab
-- lists them. NULL on rows from before this migration, which counts as "not a message".
-- Run this BEFORE deploying the Worker that writes the column (see logQuery's comment).
ALTER TABLE queries ADD COLUMN message_for_nacfe INTEGER;
CREATE INDEX IF NOT EXISTS idx_queries_message ON queries (message_for_nacfe, timestamp);
