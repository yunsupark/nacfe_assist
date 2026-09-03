#!/usr/bin/env python3
"""Parse a YouTube auto-caption VTT (rolling word-by-word cascade format) into a clean,
deduplicated list of (start_seconds, text) segments. Auto-captions repeat growing text across
consecutive cues; this extracts only the newly-appended words in each cue.
"""
import re
import sys

TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})")
TAG_RE = re.compile(r"<[^>]+>")


def vtt_time_to_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def strip_tags(text):
    text = TAG_RE.sub("", text)
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    return " ".join(text.split())


def parse_vtt(path):
    cues = []
    start = None
    text_lines = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            m = TIME_RE.match(line)
            if m:
                if start is not None and text_lines:
                    cues.append((start, strip_tags(" ".join(text_lines))))
                start = vtt_time_to_seconds(*m.groups()[:4])
                text_lines = []
            elif line.strip() == "":
                if start is not None and text_lines:
                    cues.append((start, strip_tags(" ".join(text_lines))))
                start = None
                text_lines = []
            elif line.strip() in ("WEBVTT",) or line.startswith("Kind:") or line.startswith("Language:"):
                continue
            else:
                text_lines.append(line)
    if start is not None and text_lines:
        cues.append((start, strip_tags(" ".join(text_lines))))

    # dedupe empty
    cues = [(s, t) for s, t in cues if t]

    # Cascade cleanup: each cue's text is often a growing prefix of the next. Walk through,
    # and whenever the current cue's text starts with the previous kept text, emit only the
    # new suffix at the current cue's timestamp; otherwise emit the whole thing.
    segments = []
    prev_text = ""
    for start, text in cues:
        if text == prev_text:
            continue
        if prev_text and text.startswith(prev_text):
            new_part = text[len(prev_text):].strip()
            if new_part:
                segments.append((start, new_part))
        else:
            segments.append((start, text))
        prev_text = text

    return segments


if __name__ == "__main__":
    segs = parse_vtt(sys.argv[1])
    for start, text in segs:
        mins, secs = divmod(int(start), 60)
        print(f"[{mins:02d}:{secs:02d}] {text}")
