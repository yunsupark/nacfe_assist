"""Add `authors` and `speakers` to corpus/catalog.json from the study-team roster.

The router only reads the catalog, and the catalog had no people information, so a question
about a person could not find the reports they wrote or the recordings they spoke in. This script
fills two optional fields:

  authors   -- people listed on a report's study-team page (lead author marked "(lead author)").
               The ingested .md files don't carry that page (transcribe_page.txt rule 6 collapses
               author pages to one line), so the names come from ingest/people_study_team.json,
               built from the original PDFs by build_study_team_roster.py. Editors, reviewers and
               advisory-committee members are not included. Hand-curated additions and PDF pairings
               live in ingest/people_overrides.json.
  speakers  -- anyone from that roster who has a "**Name:**" speaker label in a video / bootcamp /
               podcast transcript.

Reports with no roster entry (no credits page found, or the PDF could not be matched) are left
without `authors` rather than guessed. This script does not call a model or read PDFs.

Usage:
    python ingest/extract_people.py [--people "Mike Roeth,Dave Schaller"] [--write]

--people limits the run to a subset of the roster (canonical names). Without --write it only
prints what it would change.

This module also holds the PDF helpers (pdf_text, match_pdfs) that build_study_team_roster.py uses.
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
ROSTER_PATH = Path(__file__).resolve().parent / "people_study_team.json"
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


def speakers_for(entry, people):
    f = SOURCES_DIR / f"{entry['id']}.md"
    labels = set(SPEAKER_LABEL_RE.findall(f.read_text(errors="ignore")))
    return [canonical for canonical, spellings in people.items() if any(s in labels for s in spellings)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", help="comma-separated canonical names to limit the run to a subset of the roster")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    roster = json.loads(ROSTER_PATH.read_text())
    people = roster["people"]  # canonical name -> spellings seen in documents
    if args.people:
        wanted = {p.strip() for p in args.people.split(",") if p.strip()}
        unknown = wanted - set(people)
        if unknown:
            sys.exit(f"not in the roster: {sorted(unknown)}")
        people = {c: s for c, s in people.items() if c in wanted}

    catalog = json.loads(CATALOG_PATH.read_text())
    ids = {e["id"] for e in catalog}
    overrides = json.loads(OVERRIDES_PATH.read_text()) if OVERRIDES_PATH.exists() else {}
    manual_authors = {}
    for eid, names in overrides.get("authors", {}).items():
        if eid not in ids:
            sys.exit(f"people_overrides.json: authors entry {eid!r} is not a catalog id")
        if args.people:  # a subset run only keeps the subset's names
            names = [n for n in names if n in people]
        else:  # a byline can name someone who is on no study-team page; they join the people list
            for n in names:
                people.setdefault(n, [n])
        if names:
            manual_authors[eid] = names

    report, changed = [], 0
    for e in catalog:
        e.pop("authors", None)
        e.pop("speakers", None)
        if e["id"] in manual_authors:
            e["authors"] = manual_authors[e["id"]]
        elif e["media"]["kind"] == "pdf":
            entry_people = roster["entries"].get(e["id"])
            if entry_people is None:
                report.append(f"NO ROSTER ENTRY (no credits page / PDF unmatched)   {e['id']}")
            else:
                authors = [
                    f"{p['name']} (lead author)" if p.get("lead") else p["name"]
                    for p in entry_people
                    if p["name"] in people
                ]
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

    print(f"people in this run: {len(people)}")
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
