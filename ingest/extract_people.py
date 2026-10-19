"""Add `authors` and `speakers` to corpus/catalog.json for a chosen set of NACFE people.

The router only reads the catalog, and the catalog had no people information, so a question
about a staffer could not find the reports they wrote or the videos they appeared in. This
script fills two optional fields, restricted to the people named on the command line:

  authors   -- from a report PDF's "STUDY TEAM" block (the lead author is marked "(lead author)").
               The ingested .md files don't carry that block (transcribe_page.txt rule 6 collapses
               author pages to one line), so it is read from the original PDFs, matched to catalog
               entries by comparing body text -- never by filename guesses.
  speakers  -- from the "**Name:**" speaker labels in video / podcast / bootcamp transcripts.

Reports whose PDF has no recognisable study-team block, or whose PDF can't be matched confidently,
are listed and left without `authors` rather than guessed.

Usage:
    python ingest/extract_people.py --reports-dir /path/to/reports \
        --people "Mike Roeth,Dave Schaller|David Schaller,Rick Mihelic,Yunsu Park" [--write]

Without --write it only prints what it would change.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "corpus" / "catalog.json"
SOURCES_DIR = REPO_ROOT / "corpus" / "sources"
OVERRIDES_PATH = Path(__file__).resolve().parent / "people_overrides.json"

# Where a credits block starts, and what ends it. Layouts vary: a "STUDY TEAM" heading on its own
# line (recent reports) or inline ("Acknowledgements: Study Team: Name, Title Name, Title ...",
# older ones), so the first pages are flattened to one line of text before searching.
TEAM_ANCHOR_RE = re.compile(
    r"(?i)(?:\bnacfe\s+)?\b(?:study|report|project)\s+team\s*:?|\bauthors?\b(?:\s*&\s*acknowledg\w+)?\s*:?"
)
TEAM_END_RE = re.compile(
    r"(?i)\b(?:study\s+editors?|editors?:|peer\s+review\w*|contacts?\b|nacfe\s+technical\s+advisory|"
    r"technical\s+advisory|table\s+of\s+(?:contents|figures)|electrada\s+(?:study|report)\s+team|"
    r"acknowledg\w+|about\s+us|fiscal\s+support|sponsors?\b)"
)
BLOCK_MAX_CHARS = 1800
PAGES_FOR_CREDITS = 10   # credits are always on the first few pages
PAGES_FOR_MATCHING = range(7, 30)  # body text used to identify which PDF is which entry

SPEAKER_LABEL_RE = re.compile(r"\*\*([A-Z][A-Za-z.'\- ]{2,40}):\*\*")


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def pdf_text(path, pages):
    from pypdf import PdfReader
    import logging
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path))
    out = []
    for i in pages:
        if i >= len(reader.pages):
            break
        try:
            out.append(reader.pages[i].extract_text() or "")
        except Exception:
            out.append("")
    return "\n".join(out)


def shingles(text, size=6):
    """Set of `size`-word runs, lowercased, punctuation dropped -- robust to line-wrapping and
    to the small reformatting a transcription introduces."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + size]) for i in range(0, max(0, len(words) - size + 1))}


def match_pdfs(reports_dir, entries):
    """Map entry id -> pdf path: the share of a PDF's body-text word runs found in an entry's .md."""
    md_shingles = {
        e["id"]: shingles((SOURCES_DIR / f"{e['id']}.md").read_text(errors="ignore")) for e in entries
    }
    best_for_pdf = {}
    for pdf in sorted(Path(reports_dir).glob("*.pdf")):
        try:
            sh = shingles(pdf_text(pdf, PAGES_FOR_MATCHING))
        except Exception:
            continue
        if len(sh) < 200:
            continue
        scored = sorted(((len(sh & ms) / len(sh), eid) for eid, ms in md_shingles.items()), reverse=True)
        top, runner = scored[0], scored[1][0] if len(scored) > 1 else 0
        # Confident: a large share found, and clearly ahead of the next-best entry.
        if top[0] >= 0.25 and top[0] - runner >= 0.15:
            best_for_pdf[pdf.name] = (top[1], top[0])
    matches = {}
    # One entry can have several PDFs (e.g. a compressed copy); keep the best-scoring one.
    for name, (eid, score) in best_for_pdf.items():
        if eid not in matches or score > matches[eid][1]:
            matches[eid] = (name, score)
    return {eid: Path(reports_dir) / name for eid, (name, _) in matches.items()}


def credits_for(pdf_path, people):
    """Returns (saw_a_credits_block, authors). `people` maps canonical name -> alias list.
    The first credits block that names any of the people wins; a block that names none (another
    organisation's team, say) is skipped."""
    flat = re.sub(r"\s+", " ", pdf_text(pdf_path, range(0, PAGES_FOR_CREDITS)))
    saw_block = False
    for m in TEAM_ANCHOR_RE.finditer(flat):
        rest = flat[m.end(): m.end() + BLOCK_MAX_CHARS]
        end = TEAM_END_RE.search(rest)
        block = rest[: end.start()] if end else rest
        if len(block.strip()) < 25:
            continue
        saw_block = True
        found = []
        for canonical, aliases in people.items():
            hit = next((a for a in aliases if a in block), None)
            if hit:
                lead = re.search(re.escape(hit) + r",?\s*(?:co-)?lead authors?", block, re.I)
                found.append(f"{canonical} (lead author)" if lead else canonical)
        if found:
            return True, found
    return saw_block, []


def speakers_for(entry, people):
    f = SOURCES_DIR / f"{entry['id']}.md"
    labels = set(SPEAKER_LABEL_RE.findall(f.read_text(errors="ignore")))
    return [canonical for canonical, aliases in people.items() if any(a in labels for a in aliases)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--people", required=True, help="comma-separated names as written in the documents; alternate spellings joined with |")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    # "Dave Schaller|David Schaller" -> canonical name first, then other spellings seen in documents.
    people = {}
    for item in args.people.split(","):
        names = [n.strip() for n in item.split("|") if n.strip()]
        if names:
            people[names[0]] = names

    catalog = json.loads(CATALOG_PATH.read_text())
    docs = [e for e in catalog if e["media"]["kind"] == "pdf"]
    pdf_for = match_pdfs(args.reports_dir, docs)

    # Hand-curated overrides win over the automatic match (see people_overrides.json).
    overrides = json.loads(OVERRIDES_PATH.read_text()) if OVERRIDES_PATH.exists() else {}
    ids = {e["id"] for e in catalog}
    for eid, name in overrides.get("pdf_for", {}).items():
        path = Path(args.reports_dir) / name
        if eid not in ids or not path.exists():
            sys.exit(f"people_overrides.json: pdf_for entry {eid!r} -> {name!r}: unknown id or missing file")
        pdf_for[eid] = path
    manual_authors = {}
    for eid, names in overrides.get("authors", {}).items():
        if eid not in ids:
            sys.exit(f"people_overrides.json: authors entry {eid!r} is not a catalog id")
        kept = [n for n in names if n in people]
        if kept:
            manual_authors[eid] = kept

    report, changed = [], 0
    for e in catalog:
        e.pop("authors", None)
        e.pop("speakers", None)
        if e["id"] in manual_authors:
            e["authors"] = manual_authors[e["id"]]
        elif e["media"]["kind"] == "pdf":
            pdf = pdf_for.get(e["id"])
            if pdf is None:
                report.append(f"NO CONFIDENT PDF MATCH   {e['id']}")
                continue
            saw_block, authors = credits_for(pdf, people)
            if not saw_block:
                report.append(f"NO CREDITS BLOCK FOUND   {e['id']}  ({pdf.name})")
                continue
            if authors:
                e["authors"] = authors
        else:
            speakers = speakers_for(e, people)
            if speakers:
                e["speakers"] = speakers
        if "authors" in e or "speakers" in e:
            changed += 1

    # Place the new fields right after "published" so they sit next to the title for the router.
    ordered = []
    for e in catalog:
        new = {}
        for k, v in e.items():
            if k in ("authors", "speakers"):
                continue
            new[k] = v
            if k == "published":
                if "authors" in e:
                    new["authors"] = e["authors"]
                if "speakers" in e:
                    new["speakers"] = e["speakers"]
        ordered.append(new)

    print(f"people: {list(people)}")
    print(f"PDFs confidently matched: {len(pdf_for)} of {len(docs)} document entries")
    print(f"entries with authors or speakers set: {changed}")
    for line in report:
        print("  " + line)
    if args.write:
        CATALOG_PATH.write_text(json.dumps(ordered, indent=2, ensure_ascii=False))
        print(f"wrote {CATALOG_PATH}")
    else:
        print("(dry run; pass --write to update the catalog)")


if __name__ == "__main__":
    sys.exit(main())
