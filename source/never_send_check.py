"""Never-send gate: fail if the source calls a Gmail send endpoint or asks for a wider Gmail scope.

Usage: never_send_check.py [PATH ...]   (default: the repo's src/ and scripts/)

The default covers both services: the main app and the draft approver (`approver.py`,
`approvals.py`, `approver_api.py`, `GoogleDraftWriter`), whose compose token is the one that
could send (#19, #25).

Deterministic and stdlib-only, so it runs before dependencies are installed. It complements the
runtime test (the fake inbox fails on any send attempt); it doesn't replace it. Tests are not
scanned by default because they name these patterns on purpose.
"""

from __future__ import annotations

import base64
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (REPO / "src", REPO / "scripts")
SUFFIXES = {".py", ".html", ".js", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".txt"}
ALLOWED_SCOPES = {"gmail.readonly", "gmail.compose"}

SEND_PATTERNS = (
    (re.compile(r"\.send\s*\("), "a .send(...) call"),
    (re.compile(r"\b(?:users\.)?(?:messages|drafts)\.send\b"), "a Gmail send method"),
    (re.compile(r"/(?:messages|drafts)/send\b"), "a Gmail send endpoint"),
    # The scope is the bare origin; a web link to a thread (mail.google.com/mail/...) is not it.
    (re.compile(r"mail\.google\.com(?!/mail/)"), "the full-mailbox scope"),
)
SCOPE = re.compile(r"googleapis\.com/auth/([\w.]+)")

# Third-party files served as they were published, pinned by hash: htmx's XMLHttpRequest
# `.send(` is how it posts to our own app. A changed file is scanned like any other.
VENDORED = {
    REPO / "src/langchain_email/static/htmx-2.0.11.min.js": (
        "2OatzQy1H+Zd/IIrjr1TcuDGqLXeHhbooAyJY1KdQMKnr4LZ22k31GBLdYKHmVjg"
    ),
}


def _is_pinned_vendor_file(path: Path) -> bool:
    expected = VENDORED.get(path.resolve())
    if expected is None:
        return False
    digest = hashlib.sha384(path.read_bytes()).digest()
    return base64.b64encode(digest).decode() == expected


def violations(path: Path) -> list[str]:
    if _is_pinned_vendor_file(path):
        return []
    found = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, what in SEND_PATTERNS:
            if pattern.search(line):
                found.append(f"{path}:{lineno}: {what}: {line.strip()}")
        for scope in SCOPE.findall(line):
            if scope not in ALLOWED_SCOPES:
                found.append(f"{path}:{lineno}: Gmail scope {scope!r} is not allowed")
    return found


def files_under(paths: list[Path]) -> list[Path]:
    me = Path(__file__).resolve()
    files = []
    for root in paths:
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        files += [
            f for f in candidates if f.is_file() and f.suffix in SUFFIXES and f.resolve() != me
        ]
    return files


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or [p for p in DEFAULT_PATHS if p.exists()]
    found = [v for f in files_under(paths) for v in violations(f)]
    for v in found:
        print(v)
    if found:
        print(f"never-send check failed: {len(found)} problem(s). The app must never send email.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
