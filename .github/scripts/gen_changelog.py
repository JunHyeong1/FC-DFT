#!/usr/bin/env python3
"""Regenerate CHANGELOG.md from git history.

Every non-merge commit between two release tags is listed. Commits are sorted
into sections by conventional-commit prefix when present, otherwise by keyword
heuristics on the subject line; anything that matches nothing lands in "Other",
so no commit is silently dropped.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/JunHyeong1/FC-DFT"

# Checked in order; first match wins.
SECTIONS = [
    ("Performance", re.compile(
        r"^(perf|refactor)[(:]|"
        r"\b(perf|performance|optimi[sz]|speed|faster|efficien|memory|"
        r"accelerat|overhead)", re.I)),
    ("Fixes", re.compile(
        r"^(fix|bugfix|hotfix)[(:]|"
        r"\b(fix|fixes|fixed|bug|patch|correct|resolve|debug|"
        r"workaround|regression|deprecat)", re.I)),
    ("Features", re.compile(
        r"^(feat|feature)[(:]|"
        r"\b(add|adds|added|new|implement|support|introduce|enable|"
        r"compatib)", re.I)),
]
OTHER = "Other"

MERGE_SUBJECT = re.compile(r"^Merge (pull request|branch|remote-tracking)", re.I)


def git(*args):
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def tags():
    """Release tags, newest first."""
    out = git("tag", "--sort=-creatordate")
    return [t for t in out.splitlines() if t]


def commits(rev_range):
    """(sha, subject) for non-merge commits in rev_range, newest first."""
    out = git("log", "--no-merges", "--format=%H%x1f%s", rev_range)
    result = []
    for line in out.splitlines():
        if not line:
            continue
        sha, _, subject = line.partition("\x1f")
        subject = subject.strip()
        if not subject or MERGE_SUBJECT.match(subject):
            continue
        result.append((sha, subject))
    return result


def classify(subject):
    for name, pattern in SECTIONS:
        if pattern.search(subject):
            return name
    return OTHER


def strip_prefix(subject):
    """Drop a conventional-commit prefix so entries read uniformly."""
    return re.sub(r"^\w+(\([^)]*\))?!?:\s*", "", subject)


def render_release(tag, date, entries):
    lines = [f"## [{tag}]({REPO_URL}/releases/tag/{tag}) - {date}", ""]
    buckets = {}
    for sha, subject in entries:
        buckets.setdefault(classify(subject), []).append((sha, subject))
    for name in [s[0] for s in SECTIONS] + [OTHER]:
        if name not in buckets:
            continue
        lines.append(f"### {name}")
        lines.append("")
        for sha, subject in buckets[name]:
            short = sha[:7]
            lines.append(
                f"- {strip_prefix(subject)} "
                f"([{short}]({REPO_URL}/commit/{sha}))"
            )
        lines.append("")
    return lines


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "CHANGELOG.md")
    all_tags = tags()
    if not all_tags:
        print("no tags found; nothing to generate", file=sys.stderr)
        return 1

    out = [
        "# Changelog",
        "",
        "Generated automatically on each release; do not edit by hand.",
        "",
    ]
    for i, tag in enumerate(all_tags):
        previous = all_tags[i + 1] if i + 1 < len(all_tags) else None
        rev_range = f"{previous}..{tag}" if previous else tag
        entries = commits(rev_range)
        if not entries:
            continue
        date = git("log", "-1", "--format=%ad", "--date=short", tag)
        out += render_release(tag, date, entries)

    path.write_text("\n".join(out).rstrip() + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
