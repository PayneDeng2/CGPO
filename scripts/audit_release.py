#!/usr/bin/env python
"""Conservative local scan for common anonymity and release mistakes."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".jsonl", ".sh"}
PATTERNS = {
    "absolute local path": re.compile(r"/(home|Users|data)/[^\s'\"]+"),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "non-anonymous GitHub URL": re.compile(r"https?://(?:www\.)?github\.com/", re.I),
    "secret-like token": re.compile(r"(?:hf_|ghp_|github_pat_)[A-Za-z0-9_\-]{12,}"),
}


def main() -> None:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: {label}")

    remote = subprocess.run(
        ["git", "remote", "-v"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if remote:
        findings.append("git remote is configured; verify that it is anonymous")

    if findings:
        print("Release audit found items requiring review:")
        for finding in findings:
            print(f"- {finding}")
        raise SystemExit(1)
    print("Release audit passed: no common identity or credential patterns found.")


if __name__ == "__main__":
    main()

