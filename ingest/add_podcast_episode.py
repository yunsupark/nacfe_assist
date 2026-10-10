#!/usr/bin/env python3
"""Adds one episode to ingest/podcast_episodes.json (newest first), the lookup table that
ingest_podcast_captions.py reads video_id/title from. Used by
.github/workflows/ingest-source.yml (admin console, "Podcast episode" option) so a new episode
can be ingested through the standard caption path without hand-editing the JSON.

Usage:
    python add_podcast_episode.py --id podcast-ep142-jane-doe-2026 --video-id <youtube-id> \
        --title "Ep. 142: Jane Doe – Example Co" --episode 142 --guest "Jane Doe" \
        --org "Example Co" --published 2026-10-06 --duration 1500
"""
import argparse
import json
import sys
from pathlib import Path

EPISODES_PATH = Path(__file__).resolve().parent / "podcast_episodes.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, dest="source_id")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--guest", required=True)
    parser.add_argument("--org", required=True)
    parser.add_argument("--published", required=True, help="YYYY-MM-DD")
    parser.add_argument("--duration", required=True, type=int, help="seconds")
    args = parser.parse_args()

    raw = EPISODES_PATH.read_text()
    episodes = json.loads(raw)
    for e in episodes:
        if e["id"] == args.source_id or e["video_id"] == args.video_id or e["episode_num"] == args.episode:
            print(
                f"already in {EPISODES_PATH.name}: id={e['id']} video_id={e['video_id']} episode={e['episode_num']}",
                file=sys.stderr,
            )
            sys.exit(1)

    entry = {
        "id": args.source_id,
        "video_id": args.video_id,
        "title": args.title,
        "episode_num": args.episode,
        "guest": args.guest,
        "org": args.org,
        "published": args.published,
        "duration": args.duration,
    }
    # Match the file's existing escaping so the diff is just the new entry (the real titles
    # carry en-dashes, which the file stores as –).
    ensure_ascii = "\\u2013" in raw
    EPISODES_PATH.write_text(json.dumps([entry] + episodes, indent=2, ensure_ascii=ensure_ascii))
    print(f"added {args.source_id} to {EPISODES_PATH}")


if __name__ == "__main__":
    main()
