"""Only the approver's own page can approve or reject (#51).

A page from another origin (a compromised workspace, any site the owner visits) can make the
owner's browser POST to the approver, though it can't read the answer. Three checks stop that:

- **Same origin.** A POST to the page service must carry an `Origin` that is the approver's own
  (`[approver] public_url`, or localhost / 127.0.0.1 on `port` when it is unset), and a
  `Sec-Fetch-Site`, when sent, of `same-origin`. A missing `Origin` is refused too.
- **The card's token.** Approve and Reject carry an HMAC-SHA256 of the request id, keyed by a
  secret only the approver holds (`[approver] page_key_file`, 0600, made on first start), and
  the route checks it in constant time.
- **No framing.** Every page response forbids framing.

Anything refused is a 403 and changes nothing. The submit API (`api_port`) is not affected.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Request, Response
from fastapi.responses import PlainTextResponse

from langchain_email.config import ApproverSettings

TOKEN_HEADER = "X-Approval-Token"  # noqa: S105 - a header name, not a secret
UNSAFE = frozenset({"POST", "PUT", "PATCH", "DELETE"})
DEFAULT_PORTS = {"http": 80, "https": 443}
NO_FRAMING = {
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Frame-Options": "DENY",
}


def origin(url: str) -> str | None:
    """`scheme://host[:port]` of a URL, lower-cased and without a default port; None if it has
    no http(s) scheme or host (such as the `null` Origin of a sandboxed page)."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname:
        return None
    try:
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    if port is None or port == DEFAULT_PORTS[scheme]:
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def page_origins(approver: ApproverSettings) -> frozenset[str]:
    """The origins the approver's page is opened from: its public URL, or, when that is unset,
    the loopback names on its port (the local and test setup)."""
    if approver.public_url:
        own = origin(approver.public_url)
        if own is None:
            raise ValueError(f"[approver] public_url is not an http(s) URL: {approver.public_url}")
        return frozenset({own})
    port = approver.port
    return frozenset(f"http://{host}:{port}" for host in ("localhost", "127.0.0.1", "[::1]"))


def same_origin(request: Request, allowed: frozenset[str]) -> bool:
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None and fetch_site != "same-origin":
        return False
    sent = request.headers.get("origin")
    return sent is not None and origin(sent) in allowed


def guard(
    approver: ApproverSettings,
) -> Callable[[Request, Callable[[Request], Awaitable[Response]]], Awaitable[Response]]:
    """The page app's middleware: refuse an unsafe request from elsewhere, forbid framing."""
    allowed = page_origins(approver)

    async def middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method in UNSAFE and not same_origin(request, allowed):
            response: Response = PlainTextResponse(
                "Refused: Approve and Reject work only from the approver's own page.",
                status_code=403,
            )
        else:
            response = await call_next(request)
        response.headers.update(NO_FRAMING)
        return response

    return middleware


class PageKey:
    """The approver's secret for the cards' tokens. Never logged or shown."""

    def __init__(self, key: bytes) -> None:
        self._key = key

    @classmethod
    def load(cls, path: Path) -> PageKey:
        """Read the key file, making it (0600, atomically) if it doesn't exist yet."""
        if not path.exists():
            _create(path)
        key = bytes.fromhex(path.read_text().strip())
        if len(key) < 32:
            raise ValueError(f"the approver's page key in {path} is too short; delete it to remake")
        return cls(key)

    def token(self, request_id: str) -> str:
        return hmac.new(self._key, request_id.encode(), hashlib.sha256).hexdigest()

    def valid(self, request_id: str, token: str | None) -> bool:
        return token is not None and hmac.compare_digest(
            self.token(request_id).encode(), token.encode()
        )


def _create(path: Path) -> None:
    """Write a new key to a private temp file, then link it into place, so a reader never sees
    a partial file and two starts at once keep the same key."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32) + "\n")
            f.flush()
            os.fsync(f.fileno())
        with contextlib.suppress(FileExistsError):
            os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
