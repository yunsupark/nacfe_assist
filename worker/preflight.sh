#!/usr/bin/env bash
# Post-deploy preflight: asks a deployed Worker whether the things that are easy to forget
# actually landed. Each of these fails silently in production if missed -- a missing
# FEEDBACK_SECRET just makes the rating row vanish, a missing TURNSTILE_SECRET just leaves the
# query endpoint open -- so check rather than assume.
#
#   ./preflight.sh https://nacfe-assist.nacfe.workers.dev
set -euo pipefail
BASE="${1:-https://nacfe-assist.nacfe.workers.dev}"
fail=0
pass() { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
warn() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }

echo "preflight against $BASE"
health="$(curl -fsS --max-time 15 "$BASE/health")" || { echo "  FAIL  /health unreachable"; exit 1; }
echo "  /health -> $health"

jqf() { printf '%s' "$health" | python3 -c "import json,sys;print(json.load(sys.stdin).get('$1'))"; }

[ "$(jqf ok)" = "True" ]        && pass "worker responding"           || warn "worker not ok"
[ "$(jqf sources)" -gt 0 ] 2>/dev/null && pass "catalog loaded ($(jqf sources) sources)" || warn "catalog empty"
[ "$(jqf feedback)" = "True" ]  && pass "FEEDBACK_SECRET set"          || warn "FEEDBACK_SECRET missing -- rating row will not appear"
case "$(jqf turnstile)" in
  on)            pass "Turnstile enforced" ;;
  misconfigured) warn "Turnstile MISCONFIGURED -- TURNSTILE_SECRET set without TURNSTILE_HOSTNAMES; every query is being rejected" ;;
  *)             warn "Turnstile off -- /query is unprotected" ;;
esac

status_json="$(curl -fsS --max-time 15 "$BASE/status" 2>/dev/null)" || status_json=""
if [ -z "$status_json" ]; then
  warn "/status unreachable -- the widget cannot warn readers before the budget runs out"
else
  echo "  /status -> $status_json"
  if printf '%s' "$status_json" | grep -q '"degraded":true'; then
    warn "budget ceiling REACHED -- only cached questions are being answered"
  else
    pass "within monthly budget"
  fi
fi

# The impressions beacon is the reach number a sponsor is quoted; silent failure means
# under-reporting, so confirm the endpoint exists rather than assuming it deployed.
ev_code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 -X POST "$BASE/event" \
  -H 'content-type: application/json' --data '{"type":"impression"}' 2>/dev/null)" || ev_code=000
[ "$ev_code" = "204" ] && pass "/event accepts impressions" \
  || warn "/event returned $ev_code (expected 204) -- impressions are not being counted"

# Fetch widget.js once, and treat a failed fetch as a failure of every check that reads it.
# Grepping the empty output of a failed curl reports "placeholder absent", which is a pass for
# the wrong reason -- the exact silent-pass shape this preflight exists to catch.
# Headers come from the GET itself. A separate HEAD would be a different request that the
# Worker may route differently -- as it did: HEAD /widget.js used to 404 while GET succeeded.
hdr_file="$(mktemp)"; trap 'rm -f "$hdr_file"' EXIT
widget_body="$(curl -fsS --max-time 15 -D "$hdr_file" "$BASE/widget.js" 2>/dev/null)" || widget_body=""
widget_headers="$(cat "$hdr_file")"

if [ -z "$widget_body" ]; then
  warn "GET /widget.js failed -- the deployment does not serve the widget"
  warn "  (skipping CORS and sitekey checks; they cannot pass without a body)"
else
  printf '%s' "$widget_headers" | grep -qi '^access-control-allow-origin: \*' \
    && pass "widget.js serves with CORS" || warn "widget.js missing access-control-allow-origin"
  if printf '%s' "$widget_body" | grep -q '__TURNSTILE_SITEKEY__'; then
    warn "widget.js still contains the sitekey placeholder -- TURNSTILE_SITEKEY is unset"
  else
    pass "widget.js sitekey substituted"
  fi
fi

echo
[ "$fail" -eq 0 ] && echo "preflight passed" || { echo "preflight found problems (see FAIL above)"; exit 1; }
