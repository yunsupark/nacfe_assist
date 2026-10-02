-- Backs the admin console (see web/admin.html and the /admin* routes in worker/src/index.ts).
-- Three tables: a generic admin-editable config store (currently just the four sponsor fields,
-- which this moves off wrangler.toml vars so changes take effect without a deploy), a source
-- exclusion list for instant/reversible "hide from routing", and an append-only admin action
-- log.
--
--   npx wrangler d1 execute nacfe-assist --remote --file=migrations/0011_admin_console.sql

-- Generic key/value store for admin-editable config. updated_by is the admin's email from
-- their session token, an audit trail only -- the Worker never re-derives permission from it.
-- catalog_epoch is bumped on every hide/unhide (see bumpCatalogEpoch in src/index.ts) so the
-- 30-day answer cache is invalidated along with routing -- without it, a source hidden for
-- being wrong could still be cited from a cached answer for up to a month.
CREATE TABLE IF NOT EXISTS admin_config (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  updated_by TEXT
);

INSERT OR IGNORE INTO admin_config (key, value, updated_at, updated_by) VALUES
  ('sponsor_name', '', CURRENT_TIMESTAMP, NULL),
  ('sponsor_tagline', '', CURRENT_TIMESTAMP, NULL),
  ('sponsor_url', '', CURRENT_TIMESTAMP, NULL),
  ('sponsor_logo_url', '', CURRENT_TIMESTAMP, NULL),
  ('catalog_epoch', '0', CURRENT_TIMESTAMP, NULL);

-- Sources excluded from routing. No FK -- the 343-entry catalog lives in corpus_data.ts
-- (bundled at build time from corpus/catalog.json), not D1, so source_id is just a string
-- route() checks membership against before building the router prompt. Hiding stops FUTURE
-- routing selections only (and, via catalog_epoch above, invalidates the answer cache); it
-- does not touch the R2 document text or the git-tracked catalog itself -- for a permanent,
-- git-visible removal, see the retire-source.yml GitHub Actions workflow instead.
CREATE TABLE IF NOT EXISTS hidden_sources (
  source_id TEXT PRIMARY KEY,
  hidden_at TEXT NOT NULL,
  hidden_by TEXT
);

-- Append-only admin action log. Multiple WordPress admins can use this console, and
-- hidden_sources/admin_config only ever carry the LATEST actor -- "who hid this, and when did
-- it come back" needs real history, not just a last-value-wins column. Mirrors the existing
-- events table's append-only, no-FK shape (see migrations/0003_events.sql).
CREATE TABLE IF NOT EXISTS admin_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,
  actor_email TEXT NOT NULL,
  action TEXT NOT NULL,         -- 'hide' | 'unhide' | 'sponsor_update' | 'ingest_requested' | 'retire_requested'
  target TEXT                   -- source id, or the source URL for ingest_requested, or null for sponsor_update
);
CREATE INDEX IF NOT EXISTS idx_admin_log_timestamp ON admin_log (timestamp);
