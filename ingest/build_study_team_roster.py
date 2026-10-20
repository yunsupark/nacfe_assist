"""Build ingest/people_study_team.json: everyone named on a NACFE report's study-team page.

Credits pages come in many layouts (a "STUDY TEAM" heading on its own line, inline
"Acknowledgements: Study Team: Name, Title Name, Title ...", "Original Study Team" plus an
"Update Team", ...), so the page text is handed to a model to list the people, and every name it
returns is then checked to appear verbatim in that page's text, so nothing can be invented. The
result is a reviewable data file; `extract_people.py` turns it into catalog `authors`/`speakers`
without calling a model.

Included: people listed under a study team / report team / update team / project team / authors
heading. Excluded: study editors, peer reviewers, technical advisory committee members, contacts,
sponsors -- and anyone on a team list whose stated role is editor or designer.

Usage:
    python ingest/build_study_team_roster.py --reports-dir /path/to/reports [--model gemini-3.6-flash]

Needs GEMINI_API (or GEMINI_API_FREE) in the environment or a .env file. Costs a few cents.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_people as ep  # noqa: E402

ROSTER_PATH = Path(__file__).resolve().parent / "people_study_team.json"
DEFAULT_MODEL = "gemini-3.6-flash"
CREDITS_PAGE_RE = re.compile(r"(?i)\b(?:study|report|project|update)\s+team\b|\bauthors?\s*&\s*acknowledg")

# First names that are the same person under two spellings. Only used to merge a person's variants
# (same surname + equivalent first name); every merge is printed for review.
NICKNAMES = {
    "mike": "michael", "dave": "david", "rick": "richard", "rich": "richard", "dick": "richard",
    "jim": "james", "jimmy": "james", "bill": "william", "will": "william", "bob": "robert",
    "rob": "robert", "tom": "thomas", "steve": "steven", "stephen": "steven", "dan": "daniel",
    "chris": "christopher", "tony": "anthony", "ken": "kenneth", "jeff": "jeffrey", "joe": "joseph",
    "matt": "matthew", "greg": "gregory", "don": "donald", "ron": "ronald", "ed": "edward",
    "tim": "timothy", "jon": "jonathan", "nick": "nicholas", "pat": "patrick", "andy": "andrew",
    "drew": "andrew", "ben": "benjamin", "sam": "samuel", "alex": "alexander", "jen": "jennifer",
    "jenny": "jennifer", "jess": "jessica", "kim": "kimberly", "liz": "elizabeth",
    "beth": "elizabeth", "sue": "susan", "cathy": "catherine", "kate": "katherine", "katie": "katherine",
}

PROMPT = """Below is the text of the acknowledgements / credits page of a NACFE (North American Council
for Freight Efficiency) report.

List every person named as a member of the report's study team, report team, update team,
project team, or as an author -- including an "Original Study Team" and an "Update Team" if both
appear. For each give the name exactly as written, the role / title / organisation written next
to it (empty string if none), the heading it was listed under, and whether it is marked as the
lead author (a phrase like "lead author" next to the name).

Do NOT include anyone listed only under: study editors / editors, peer reviewers, the technical
advisory committee, contacts, sponsors, or acknowledgements of interviewees. Do not include a
person twice. If the page names no study team, return an empty list.

Return JSON only: {{"people": [{{"name": "...", "role": "...", "heading": "...", "lead": false}}]}}

PAGE TEXT:
{text}
"""


CREDIT_CUES = [
    CREDITS_PAGE_RE,
    re.compile(r"(?i)acknowledg"),
    re.compile(r"(?i)study editors?|\beditors?:"),
    re.compile(r"(?i)technical advisory"),
    re.compile(r"(?i)\bcontacts?\b"),
    re.compile(r"(?i)peer review"),
    re.compile(r"(?i)fiscal support|sponsors?\b"),
]


def credits_text(pdf_path):
    """Text of the front-matter page that looks most like a credits page: one that carries a team
    heading AND other credits cues (acknowledgements, editors, advisory committee, contacts...).
    A page that merely says "the study team engaged ..." in prose scores one cue and is skipped."""
    from pypdf import PdfReader
    import logging
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(pdf_path))
    best, best_score = None, 1
    for i in range(min(10, len(reader.pages))):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:
            continue
        if not CREDITS_PAGE_RE.search(text):
            continue
        score = sum(1 for c in CREDIT_CUES if c.search(text))
        if score > best_score:
            best, best_score = text, score
    return re.sub(r"[ \t]+", " ", best).strip()[:6000] if best else None


def norm_ws(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def ask(client, model, text):
    from google.genai import types
    last = None
    for attempt in range(1, 6):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[PROMPT.format(text=text)],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
            )
            return json.loads(resp.text)
        except Exception as e:  # noqa: BLE001 - network / parse errors: back off and retry
            last = e
            time.sleep(4 * attempt)
    raise RuntimeError(f"model call failed: {last}")


def surname_first(name):
    parts = name.split()
    return parts[-1].lower(), parts[0].lower()


def merge_variants(all_names):
    """Group spellings of one person: same surname and first names that are equal or nicknames of
    each other. Returns {canonical: [variants]} choosing the most frequent spelling as canonical."""
    groups = {}
    for name, count in all_names.items():
        last, first = surname_first(name)
        key = (last, NICKNAMES.get(first, first))
        groups.setdefault(key, []).append((count, name))
    people = {}
    for members in groups.values():
        members.sort(key=lambda m: (-m[0], m[1]))
        people[members[0][1]] = [m[1] for m in members]
    return people


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    from dotenv import find_dotenv, load_dotenv
    from google import genai
    load_dotenv(find_dotenv(usecwd=True))
    key = os.getenv("GEMINI_API") or os.getenv("GEMINI_API_FREE")
    if not key:
        sys.exit("Set GEMINI_API (or GEMINI_API_FREE) in the environment or a .env file.")
    client = genai.Client(api_key=key)

    catalog = json.loads(ep.CATALOG_PATH.read_text())
    docs = [e for e in catalog if e["media"]["kind"] == "pdf"]
    pdf_for = ep.match_pdfs(args.reports_dir, docs)
    overrides = json.loads(ep.OVERRIDES_PATH.read_text()) if ep.OVERRIDES_PATH.exists() else {}
    for eid, name in overrides.get("pdf_for", {}).items():
        pdf_for[eid] = Path(args.reports_dir) / name

    raw, no_page, hallucinated = {}, [], []
    for e in docs:
        pdf = pdf_for.get(e["id"])
        if pdf is None:
            continue
        text = credits_text(pdf)
        if text is None:
            no_page.append(e["id"])
            continue
        flat = norm_ws(text)
        result = ask(client, args.model, text)
        kept = []
        for p in result.get("people", []):
            name = re.sub(r"\s+", " ", str(p.get("name", ""))).strip()
            if len(name.split()) < 2:
                continue
            if norm_ws(name) not in flat:
                hallucinated.append((e["id"], name))
                continue
            role = str(p.get("role", "")).strip()
            # An editor or designer listed among the team is not an author.
            if re.search(r"(?i)\b(editors?|designers?|design director)\b", role):
                continue
            if any(k["name"] == name for k in kept):
                continue
            kept.append({"name": name, "role": role, "heading": str(p.get("heading", "")).strip(),
                         "lead": bool(p.get("lead")) or bool(re.search(r"(?i)lead author", role))})
        raw[e["id"]] = kept
        print(f"{e['id'][:58]:58s} {len(kept):2d} people", flush=True)

    counts = {}
    for plist in raw.values():
        for p in plist:
            counts[p["name"]] = counts.get(p["name"], 0) + 1
    people_map = merge_variants(counts)
    canon = {v: c for c, vs in people_map.items() for v in vs}
    entries = {}
    for eid, plist in raw.items():
        seen, uniq = set(), []
        for p in plist:
            c = canon[p["name"]]
            if c in seen:  # same person under two spellings in one entry: keep the first
                continue
            seen.add(c)
            uniq.append({**p, "name": c, "as_written": p["name"]})
        entries[eid] = uniq

    out = {
        "_about": "Generated by ingest/build_study_team_roster.py from each report's credits page; "
                  "review before use. people: canonical name -> spellings seen. entries: catalog id -> "
                  "people on the study / report / update team (editors, reviewers, advisory committee excluded).",
        "people": dict(sorted(people_map.items())),
        "entries": entries,
    }
    ROSTER_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    merged = {c: vs for c, vs in people_map.items() if len(vs) > 1}
    print(f"\nwrote {ROSTER_PATH}")
    print(f"{len(entries)} entries with a credits page, {sum(1 for p in entries.values() if p)} with at least one person")
    print(f"{len(people_map)} distinct people; spelling variants merged: {merged}")
    print(f"credits page not found for: {no_page}")
    print(f"names the model returned that are NOT in the page text (dropped): {hallucinated}")


if __name__ == "__main__":
    main()
