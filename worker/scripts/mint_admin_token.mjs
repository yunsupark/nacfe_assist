#!/usr/bin/env node
// Mints an admin handoff token for local testing, without needing the real WordPress plugin.
// Not deployed, not imported by anything under src/ -- keeping token-minting code entirely
// outside the deployed Worker bundle is itself a security property worth preserving (a route
// that could ever mint a valid admin token has no business existing in production).
//
// Replicates the exact scheme verifyAdminToken() in src/index.ts expects:
//   <base64url(JSON payload)>.<hex HMAC-SHA256 signature over the base64url string>
//
// Usage:
//   ADMIN_TOKEN_SECRET=devsecret node scripts/mint_admin_token.mjs you@example.com
//   ADMIN_TOKEN_SECRET=devsecret node scripts/mint_admin_token.mjs you@example.com --iat <unix-seconds>
//
// Prints a ready-to-open URL for a local `wrangler dev` server (default localhost:8787).
import { createHmac, randomBytes } from "node:crypto";

const args = process.argv.slice(2);
const email = args[0];
if (!email) {
  console.error("usage: node scripts/mint_admin_token.mjs <email> [--iat <unix-seconds>] [--base-url <url>]");
  process.exit(1);
}

function flagValue(name, fallback) {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
}

const secret = process.env.ADMIN_TOKEN_SECRET;
if (!secret) {
  console.error("set ADMIN_TOKEN_SECRET in the environment (must match .dev.vars for wrangler dev)");
  process.exit(1);
}

const iat = parseInt(flagValue("--iat", String(Math.floor(Date.now() / 1000))), 10);
const baseUrl = flagValue("--base-url", "http://localhost:8787");
const jti = randomBytes(16).toString("hex");

function base64Url(buf) {
  return buf.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

const payload = JSON.stringify({ email, iat, jti });
const b64 = base64Url(Buffer.from(payload, "utf-8"));
const sig = createHmac("sha256", secret).update(b64).digest("hex");
const token = `${b64}.${sig}`;

console.log(`payload: ${payload}`);
console.log(`token:   ${token}`);
console.log(`url:     ${baseUrl}/admin/login?token=${encodeURIComponent(token)}`);
