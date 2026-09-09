-- Sponsor and usage reporting. Run one at a time:
--   npx wrangler d1 execute nacfe-assist --remote --command "<paste a query>"
--
-- Impressions come from the events table; questions and topics from the query log. Reach and
-- questions are different numbers and should never be conflated in a sponsor report --
-- impressions are far larger, because most readers who see the widget never type anything.

-- === Reach, by month =======================================================================
SELECT substr(day, 1, 7) AS month,
       SUM(type = 'impression')    AS impressions,
       SUM(type = 'sponsor_click') AS sponsor_clicks,
       ROUND(100.0 * SUM(type = 'sponsor_click') / NULLIF(SUM(type = 'impression'), 0), 2) AS ctr_pct
FROM events GROUP BY month ORDER BY month DESC;

-- === Which embedding pages drive the widget ================================================
SELECT page_url,
       SUM(type = 'impression')    AS impressions,
       SUM(type = 'sponsor_click') AS clicks
FROM events WHERE page_url IS NOT NULL GROUP BY page_url ORDER BY impressions DESC LIMIT 20;

-- === Monthly unique visitors and questions ================================================
-- visitor_id is month-scoped by construction, so DISTINCT within a month is a true unique
-- count. Never sum it across months -- the ids are unlinkable between them by design, so a
-- returning visitor is a different id in each and the total would be meaningless.
-- LENGTH(visitor_id) = 16 selects only the month-scoped HMAC scheme. The database also holds
-- persistent per-browser UUIDs (36 and 17 characters) written by the previous tracking code
-- up to 2026-09-03. Those are a different thing: durable across months, so counting them
-- alongside these would mix two incompatible definitions of "unique visitor" and silently
-- inflate the figure -- it reported 2 for 2026-09 when only 1 row used this scheme.
SELECT substr(timestamp, 1, 7)     AS month,
       COUNT(DISTINCT visitor_id)  AS unique_visitors,
       COUNT(*)                    AS questions,
       ROUND(1.0 * COUNT(*) / NULLIF(COUNT(DISTINCT visitor_id), 0), 1) AS questions_per_visitor
FROM queries WHERE LENGTH(visitor_id) = 16 GROUP BY month ORDER BY month DESC;

-- === Which NACFE pages the questions come from ============================================
SELECT page_url, COUNT(*) AS questions,
       COUNT(DISTINCT CASE WHEN LENGTH(visitor_id) = 16 THEN visitor_id END) AS unique_visitors
FROM queries WHERE page_url IS NOT NULL GROUP BY page_url ORDER BY questions DESC LIMIT 20;

-- === Failures =============================================================================
SELECT substr(timestamp,1,10) AS day, error, COUNT(*) n
FROM queries WHERE error IS NOT NULL GROUP BY day, error ORDER BY day DESC;

-- === Geography =============================================================================
SELECT country, COUNT(*) AS questions,
       COUNT(DISTINCT CASE WHEN LENGTH(visitor_id) = 16 THEN visitor_id END) AS unique_visitors
FROM queries WHERE country IS NOT NULL GROUP BY country ORDER BY questions DESC;

-- === Engagement: what share of impressions became questions ================================
-- Run both and divide; a single query would need a join across tables with different grain.
SELECT substr(day, 1, 7) AS month, COUNT(*) AS impressions
FROM events WHERE type = 'impression' GROUP BY month ORDER BY month DESC;
SELECT substr(timestamp, 1, 7) AS month, COUNT(*) AS questions
FROM queries GROUP BY month ORDER BY month DESC;

-- === Spend against the monthly ceiling =====================================================
SELECT substr(timestamp, 1, 7) AS month,
       COUNT(*)                                   AS questions,
       SUM(cache_hit)                             AS cache_hits,
       ROUND(SUM(COALESCE(cost_micro_usd, 0)) / 1000000.0, 2) AS usd
FROM queries GROUP BY month ORDER BY month DESC;

-- === Demand: which reports the industry is actually pulling on =============================
-- selected_sources is a JSON array of source ids; json_each expands it.
SELECT j.value AS source_id, COUNT(*) AS times_consulted
FROM queries, json_each(queries.selected_sources) j
WHERE queries.selected_sources IS NOT NULL
GROUP BY source_id ORDER BY times_consulted DESC LIMIT 20;

-- === Gaps: questions NACFE could not answer ================================================
-- The research agenda, and the most sponsorable finding here: demand with no report behind it.
SELECT question, timestamp FROM queries
WHERE out_of_scope = 1 ORDER BY timestamp DESC LIMIT 50;

-- === Answer quality, from reader ratings ===================================================
SELECT rating, COUNT(*) AS n FROM feedback GROUP BY rating ORDER BY n DESC;
