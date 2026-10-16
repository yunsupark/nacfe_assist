#!/usr/bin/env node
// Bundles ../corpus/catalog.json and worker/prompts/*.txt into src/corpus_data.ts so the
// Worker can serve them without a round-trip. Workers don't have filesystem access at
// runtime, so this runs at build time (see package.json's "dev"/"deploy" scripts) and the
// generated file is gitignored -- it's derived data, regenerate it, don't hand-edit it.
//
// The full document TEXT (corpus/sources/*.md) is NOT bundled here -- see upload_corpus.mjs.
// Bundling all document text directly into the JS bundle was the original approach and broke
// down at real scale: Workers cap script size at 3 MiB (free) / 10 MiB (paid) compressed, and
// at the corpus's actual measured per-document average (~61KB gzip), that ceiling arrives
// around ~48 documents on the free tier -- well before the ~170-230 document target. The
// catalog stays bundled because it's small by design (SPEC.md 3 targets ~40-60K tokens for
// the whole thing) and every query needs it for routing regardless of which sources get
// selected; individual documents only get fetched for the 1-6 sources actually selected per
// query, so fetching them from R2 at request time doesn't cost anything extra per query that
// bundling would have saved.
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "..");
const CATALOG_PATH = join(REPO_ROOT, "corpus", "catalog.json");
const SOURCES_DIR = join(REPO_ROOT, "corpus", "sources");
const PROMPTS_DIR = join(__dirname, "prompts");
const WIDGET_PATH = join(REPO_ROOT, "web", "widget.js");
const ADMIN_PATH = join(REPO_ROOT, "web", "admin.html");
const OUT_PATH = join(__dirname, "src", "corpus_data.ts");

const catalog = JSON.parse(readFileSync(CATALOG_PATH, "utf-8"));
const routePrompt = readFileSync(join(PROMPTS_DIR, "route.txt"), "utf-8");
const answerPrompt = readFileSync(join(PROMPTS_DIR, "answer.txt"), "utf-8");
const widgetJs = readFileSync(WIDGET_PATH, "utf-8");
const adminHtml = readFileSync(ADMIN_PATH, "utf-8");

const sourceFiles = new Set(
  readdirSync(SOURCES_DIR)
    .filter((f) => f.endsWith(".md"))
    .map((f) => f.slice(0, -3)),
);
const missing = catalog.map((c) => c.id).filter((id) => !sourceFiles.has(id));
if (missing.length) {
  console.error(`build_corpus: catalog references sources with no .md file: ${missing.join(", ")}`);
  process.exit(1);
}

// A retired source leaving a dangling supersedes/superseded_by pointer is a silent data-
// integrity bug -- nothing else catches it, since those fields are only ever read by a human
// browsing the admin console's source list. Catch it here, the same way a missing .md fails
// the build above, rather than shipping a reference to an entry that no longer exists.
const catalogIds = new Set(catalog.map((c) => c.id));
const dangling = [];
for (const c of catalog) {
  // supersedes is string[] | null (a report can supersede more than one earlier one);
  // superseded_by is string | null (one entry can only be superseded by a single newer one).
  for (const target of c.supersedes ?? []) {
    if (!catalogIds.has(target)) dangling.push(`${c.id} supersedes missing "${target}"`);
  }
  if (c.superseded_by && !catalogIds.has(c.superseded_by)) {
    dangling.push(`${c.id} superseded_by missing "${c.superseded_by}"`);
  }
}
if (dangling.length) {
  console.error(`build_corpus: dangling supersession references:\n  ${dangling.join("\n  ")}`);
  process.exit(1);
}

// Fingerprint of everything that determines an answer's content: the catalog, both prompts, and
// the source documents themselves. It is folded into the answer-cache key, so any deploy that
// changes one of them -- a source retired, ingested or re-ingested, a routing/answer prompt edit --
// stops serving older cached answers by itself, with nothing to remember at retire time. Widget
// and console edits don't change it (they don't affect answers), so they keep the cache warm.
//
// Two choices here are deliberate, so please don't "tighten" them:
//  * The FULL catalog is hashed, not just the fields the router sees. That means a change to
//    ingested/ingest_model also busts the cache, which is the only thing that catches a re-ingest
//    that updates those fields.
//  * The document text is hashed too. Answers are generated from the .md files, which move
//    independently of the catalog: a re-ingest rewrites transcripts and can leave catalog.json
//    byte-identical (commit e9cd1be re-ingested 130 videos without touching it).
const versionHash = createHash("sha256")
  .update(JSON.stringify(catalog))
  .update("\0")
  .update(routePrompt)
  .update("\0")
  .update(answerPrompt);
for (const f of [...sourceFiles].sort()) {
  versionHash.update("\0").update(f).update("\0").update(readFileSync(join(SOURCES_DIR, `${f}.md`)));
}
const CONTENT_VERSION = versionHash.digest("hex").slice(0, 12);

const out = `// GENERATED by build_corpus.mjs from corpus/catalog.json, worker/prompts/*.txt,
// web/widget.js, and web/admin.html. Do not hand-edit -- edit those and re-run
// \`npm run build:corpus\`.
// Document TEXT is NOT here -- it's in R2, uploaded via upload_corpus.mjs / npm run upload:corpus.
import type { CatalogEntry } from "./catalog_types";

export const CATALOG: CatalogEntry[] = ${JSON.stringify(catalog, null, 2)};

export const CONTENT_VERSION = ${JSON.stringify(CONTENT_VERSION)};

export const ROUTE_PROMPT = ${JSON.stringify(routePrompt)};

export const ANSWER_PROMPT = ${JSON.stringify(answerPrompt)};

export const WIDGET_JS = ${JSON.stringify(widgetJs)};

export const ADMIN_HTML = ${JSON.stringify(adminHtml)};
`;

writeFileSync(OUT_PATH, out);
console.log(`build_corpus: wrote ${OUT_PATH} (${catalog.length} catalog entries, content version ${CONTENT_VERSION})`);
