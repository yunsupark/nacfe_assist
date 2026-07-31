#!/usr/bin/env node
// Uploads corpus/sources/*.md to the R2 bucket the Worker fetches selected documents from at
// request time (see src/index.ts's answer()). Run after any ingest that adds/changes a
// source, and before deploy if sources changed. R2 write cost is ~$4.50/million ops -- at
// even a few hundred documents this is fractions of a cent, so no incremental-upload logic
// here; it just re-uploads everything each run.
//
// Usage:
//   node upload_corpus.mjs --local   (uploads to the local R2 emulation used by `wrangler dev`)
//   node upload_corpus.mjs --remote  (uploads to the real bucket -- needs `wrangler login`)
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "..");
const SOURCES_DIR = join(REPO_ROOT, "corpus", "sources");
const BUCKET_NAME = "nacfe-assist-sources";

const mode = process.argv.includes("--remote") ? "--remote" : "--local";
if (mode === "--local") {
  console.log("upload_corpus: uploading to LOCAL R2 emulation (pass --remote for the real bucket)");
}

const files = readdirSync(SOURCES_DIR).filter((f) => f.endsWith(".md"));
for (const file of files) {
  const path = join(SOURCES_DIR, file);
  execFileSync(
    "npx",
    ["wrangler", "r2", "object", "put", `${BUCKET_NAME}/${file}`, `--file=${path}`, mode, "--content-type=text/markdown"],
    { stdio: "inherit", cwd: __dirname },
  );
}
console.log(`upload_corpus: uploaded ${files.length} source file(s) to ${BUCKET_NAME} (${mode})`);
