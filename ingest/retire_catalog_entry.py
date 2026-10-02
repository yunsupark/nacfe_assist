#!/usr/bin/env python3
"""Removes one entry from corpus/catalog.json by id. Used by
.github/workflows/retire-source.yml (admin console "Retire" action) -- kept as its own small
script rather than inlined into the workflow YAML, since a heredoc there would inherit the
YAML block's indentation and break Python's indentation-sensitive parsing.

Usage:
    python retire_catalog_entry.py <source-id>
"""
import json
import sys
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent / "corpus" / "catalog.json"


def main():
    if len(sys.argv) != 2:
        print("usage: python retire_catalog_entry.py <source-id>", file=sys.stderr)
        sys.exit(1)
    source_id = sys.argv[1]
    catalog = json.loads(CATALOG_PATH.read_text())
    remaining = [c for c in catalog if c["id"] != source_id]
    if len(remaining) == len(catalog):
        print(f"no catalog entry with id {source_id!r} -- nothing removed", file=sys.stderr)
        sys.exit(1)
    # ensure_ascii=False: the catalog's real titles carry en-dashes and curly quotes, and
    # json.dumps's default (ensure_ascii=True) would escape every one of them into \uXXXX --
    # rewriting ~740 lines that didn't actually change and burying the one real removal in
    # noise a PR reviewer can't see through.
    CATALOG_PATH.write_text(json.dumps(remaining, indent=2, ensure_ascii=False) + "\n")
    print(f"removed {source_id!r} from {CATALOG_PATH}")


if __name__ == "__main__":
    main()
