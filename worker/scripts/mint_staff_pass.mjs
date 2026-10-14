#!/usr/bin/env node
// Dev-only: mints a staff pass like the WordPress launcher plugin does, for testing the Worker's
// audience tagging locally. Not deployed, not imported by src/.
//
//   ADMIN_TOKEN_SECRET=devsecret node scripts/mint_staff_pass.mjs [--iat <unix-seconds>] [--typ <typ>]
//
// Prints the token. --iat backdates it (to test the one-hour expiry); --typ overrides the type
// (to test that a token of another type is rejected).
import { createHmac } from "node:crypto";

const args = process.argv.slice(2);
function flagValue(name, fallback) {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
}
const secret = process.env.ADMIN_TOKEN_SECRET;
if (!secret) {
  console.error("ADMIN_TOKEN_SECRET must be set (same value as the Worker secret / wp-config constant)");
  process.exit(1);
}
const payload = JSON.stringify({
  typ: flagValue("--typ", "staff"),
  iat: Number(flagValue("--iat", Math.floor(Date.now() / 1000))),
});
const b64 = Buffer.from(payload).toString("base64url");
console.log(`${b64}.${createHmac("sha256", secret).update(b64).digest("hex")}`);
