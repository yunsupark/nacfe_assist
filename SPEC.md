# NACFE Knowledge Base — Build Spec

Public-facing Q&A over NACFE's report, video, podcast, and article library.
Two-stage architecture: catalog-level routing, then full-document answering.

---

## 0. Design constraints

- **Low cost is a hard requirement.** Target: <$100 one-time ingest, <$50/month running.
- **Grounding over coverage.** "NACFE hasn't studied that" is a correct and acceptable answer. Answering from the model's general knowledge is a failure, even when the answer happens to be right.
- **The corpus is the asset.** Ingest output is plain markdown in git. Any model or vendor downstream is replaceable; the markdown is not.
- **Recency is safety-critical.** NACFE republishes Confidence Reports and runs annual studies. Quoting a superseded figure as current is the worst available failure.

---

## 1. Repo layout

```
/corpus/
  /sources/<id>.md          # ingested markdown, one per source item
  catalog.json              # array of source records (schema §3)
/ingest/
  ingest_pdf.py
  ingest_video.py
  ingest_article.py
  build_catalog.py
  prompts/
    transcribe_page.txt
    build_abstract.txt
/worker/                    # Cloudflare Worker (query API)
  src/index.ts
  prompts/
    route.txt
    answer.txt
/eval/
  questions.jsonl           # the golden set (§6)
  run_eval.py
  results/
/web/                       # embeddable widget
SPEC.md
```

Source `id` convention: `<program>-<topic>-<year>`, e.g. `run-on-less-electric-depot-2023`,
`confidence-report-tractor-trailer-aero-2021`. Stable forever — it appears in citations.

---

## 2. Ingest

### 2.1 PDFs

Send the **whole PDF** to Gemini in one call when under ~60 pages — full-document context
dramatically improves table continuation, acronym resolution, and figure references.
Above that, use 40-page windows with 2 pages of overlap.

Output is markdown with a page anchor before every page:

```
<!-- page: 14 -->
```

Citations resolve against these anchors, so they must be present and correct on every page.

### 2.2 Videos and podcasts

Pass the YouTube URL directly to the Gemini API as a video input. Ask for a transcript with
`[MM:SS]` timestamps every ~30 seconds and speaker labels where distinguishable. Also ask for
descriptions of on-screen slides and charts — for Bootcamp recordings the slides carry most of
the data, and a pure audio transcript throws it away.

For non-YouTube audio, upload via the Files API and use the same prompt.

### 2.3 Articles

Fetch, convert to markdown, store. Record the retrieval date; third-party URLs rot.

### 2.4 Page transcription prompt (`prompts/transcribe_page.txt`)

```
You are transcribing pages of a technical report from the North American Council for
Freight Efficiency (NACFE) into markdown, for use as the source of record in a public
question-answering system.

Transcribe. Do not summarize, condense, interpret, or editorialize.

Rules:

1. Begin each page with an anchor comment on its own line: <!-- page: N -->
   Use the page number printed on the page. If none is printed, use the sequential
   position in the PDF and note this: <!-- page: N (unnumbered) -->

2. Transcribe all body text faithfully. Preserve heading hierarchy using markdown
   headings. Preserve bulleted and numbered lists.

3. Convert every table to a markdown table. Include every row and every cell. Put units
   in the column headers where the table shows them. If a table continues from the prior
   page, continue it rather than restarting, and note: <!-- table continues -->

4. For every chart, graph, diagram, or figure, emit a block in this form:

   > **Figure N — <caption as printed>**
   > Type: <bar chart / line chart / scatter / schematic / photograph / ...>
   > Axes: <x axis label and units> vs <y axis label and units>
   > Series: <names of each series or category>
   > Values: <every data value legibly readable from the figure, as label: value pairs>
   > Notes: <legend text, annotations, callouts, source notes printed on the figure>

   If a value must be estimated from the plot rather than read from a printed label,
   prefix it with "approx." Never invent a value you cannot see. If the figure is
   decorative or a photograph with no data, say so in one line and stop.

5. Transcribe footnotes, source citations, and figure source lines. They matter for
   provenance.

6. For cover pages, tables of contents, acknowledgements, and blank pages, emit the page
   anchor and a single line describing what the page is. Do not transcribe them further.

7. Never add a fact that is not on the page. Never resolve a question the page leaves
   open. If text is illegible, write [illegible] rather than guessing.
```

### 2.5 Cost note

At Flash-Lite rates a ~50-page report costs single-digit cents. Log token counts per call
from the first run so the projected full-corpus cost is a measurement, not an estimate.

---

## 3. Catalog schema (`catalog.json`)

One record per source item. The `abstract` and `key_findings` fields are what the router
sees; everything else is filters and metadata.

```json
{
  "id": "run-on-less-electric-depot-2023",
  "title": "Run on Less – Electric DEPOT",
  "type": "run_on_less",
  "published": "2023-09",
  "url": "https://nacfe.org/...",
  "media": { "kind": "pdf", "pages": 72 },

  "topics": ["battery-electric", "depot-charging", "infrastructure"],
  "vehicle_classes": [8],
  "duty_cycles": ["regional-haul", "drayage"],
  "fleets_studied": ["..."],

  "supersedes": ["run-on-less-electric-2021"],
  "superseded_by": null,
  "status": "current",

  "abstract": "300 words, factual, written to help a router decide relevance. States what
    was studied, over what period, with what method, and what the headline findings were.
    No marketing language.",

  "key_findings": [
    "Each finding as a standalone sentence including its numbers and units."
  ],
  "data_available": [
    "Daily energy consumption by fleet",
    "Charging session duration distribution"
  ],

  "token_count": 28400,
  "ingested": "2026-07-24",
  "ingest_model": "gemini-flash-lite-..."
}
```

`type` values: `guidance_report`, `confidence_report`, `run_on_less`, `bootcamp`,
`podcast`, `video`, `article`, `thought_leadership`.

`supersedes` / `superseded_by` / `status` must be curated by a human, not inferred by the
model. This is a half-day of work across 100 items and it is the highest-leverage half-day
in the project.

The full catalog (abstracts + key findings only) should land around 40–60K tokens. If it
exceeds ~100K, tighten the abstracts rather than switching to vector search.

---

## 4. Query — Stage 1: routing (`prompts/route.txt`)

Input: full catalog (cached) + user question. Cheap model.

```
You are selecting sources from NACFE's research library to answer a question.

Below is the complete catalog of NACFE's published work. Select the sources that contain
information needed to answer the question. Select between 1 and 6. Fewer is better.

Rules:
- Prefer sources marked "status": "current" over ones they supersede. Include a superseded
  source only when the question is explicitly historical or about how findings changed.
- If several sources cover the topic across years, include the most recent plus at most
  one earlier one for trend context.
- If no source in the catalog addresses the question, return an empty selection. Do not
  stretch to find something adjacent. An empty selection is a correct answer.

Return JSON only:
{
  "selected": [{"id": "...", "why": "one sentence"}],
  "out_of_scope": false,
  "recency_warning": "set if the newest relevant source is more than 3 years old, else null"
}

CATALOG:
{{catalog}}

QUESTION:
{{question}}
```

---

## 5. Query — Stage 2: answering (`prompts/answer.txt`)

Input: full markdown of the selected sources + the question. Stronger model.

```
Answer the question using only the NACFE sources provided below.

Grounding rules:
- Every factual claim must come from the provided sources. You have general knowledge about
  trucking; do not use it. If the sources don't answer the question, say so plainly and
  state what NACFE has published that is closest.
- Cite every claim. PDFs: [Title, p. 14]. Video/audio: [Title, 12:30].
- Give numbers with their units and the conditions they were measured under. A fuel economy
  figure without its duty cycle is misleading.
- If sources disagree, say so and give the publication date of each. Do not silently prefer
  one.
- If the most recent source on this topic predates {{current_year}} by more than three years,
  note that the finding may not reflect current technology.
- Do not recommend products or vendors. Report what NACFE measured.

Write for an informed general reader: a fleet manager, journalist, or policy staffer, not a
drivetrain engineer. Lead with the direct answer. Two to four paragraphs unless the question
needs more.

SOURCES:
{{documents}}

QUESTION:
{{question}}
```

---

## 6. Evaluation — write this before writing code

Thirty questions minimum, with answers you have verified by hand against the source
documents. Store as `eval/questions.jsonl`:

```json
{"q": "...", "expect": "...", "source": "run-on-less-electric-depot-2023", "loc": "p.31", "bucket": "table"}
```

Buckets, roughly six questions each:

| Bucket | Tests | Why it's there |
|---|---|---|
| `prose` | Answer is in body text | Baseline |
| `table` | Answer exists **only** in a table | The old pipeline's failure |
| `figure` | Answer exists **only** in a chart | The other old failure |
| `synthesis` | Requires 2+ documents | Tests the router, not retrieval |
| `out_of_scope` | NACFE has never studied this | Tests refusal — the credibility failure |
| `recency` | Naive answer comes from a superseded report | Tests supersession handling |

Score each response: **correct / partial / wrong / hallucinated**. Track `hallucinated`
separately and treat it as disqualifying — a public tool under a nonprofit's name can be
incomplete but cannot be confidently wrong.

**Run the eval against NotebookLM first.** That gives you a baseline number for the quality
you already judged acceptable, and tells you whether the build is beating it or just
matching it at higher effort.

Re-run on every prompt change. This is the whole reason the eval exists.

---

## 7. Serving

- **Cloudflare Worker** holds the API key, runs both stages, returns answer + citations.
- **KV cache** keyed on the normalized question. Public tools see heavy repetition; expect
  a high hit rate and budget accordingly.
- **D1** logs every question, selected sources, token counts, and latency. What the public
  asks NACFE is itself a research finding — treat the log as an output, not telemetry.
- **Rate limit** per IP. Add Turnstile if abused.
- **Hard monthly token ceiling** in KV. Gemini will not stop on its own; degrade to
  cache-only with a notice when the ceiling is hit.

---

## 8. Build order

1. Write the eval questions. By hand. Against the PDFs.
2. Baseline them against NotebookLM. Record the score.
3. Ingest 5 reports chosen to include the ugliest tables and charts in the library.
4. Inspect the markdown yourself, page by page. Fix the transcription prompt. Repeat.
5. Hand-write catalog entries for those 5. Confirm the schema survives contact.
6. Ingest the rest. Curate supersession.
7. Build the two-stage query loop as a local script. Run the eval. Iterate on prompts.
8. Only then: Worker, cache, widget.

Step 4 is where the project is won or lost, and it is the step most likely to be skipped.
