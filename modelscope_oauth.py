"""Direct ModelScope OAuth 2.0 support for the RSMeM Studio.

The visitor signs in with their own ModelScope account and the Studio uses the
resulting access token to call ModelScope API-Inference with the visitor's own
quota. Tokens live only in this process's memory, keyed by a random browser
session id stored in a cookie; they are never written to disk, logs, state
dicts, or the client page.

Endpoints and the `api-inference` scope follow the official ModelScope OAuth
guide (modelscope.cn/learn/6670) and the PaperSeek Studio reference
implementation. The OAuth client id/secret must be provided by the Studio
owner through environment variables — never committed to the repository.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from urllib.parse import urlencode

import httpx

AUTHORIZE_URL = "https://www.modelscope.cn/oauth/authorize"
TOKEN_URL = "https://www.modelscope.cn/oauth/token"
USERINFO_URL = "https://www.modelscope.cn/oauth/userinfo"
INFERENCE_BASE_URL = "https://api-inference.modelscope.cn/v1"
OAUTH_SCOPES = "openid profile api-inference"

SESSION_COOKIE = "rsmem_sid"
NONCE_TTL_SECONDS = 15 * 60
SESSION_TTL_SECONDS = 12 * 60 * 60


def oauth_client_id() -> str:
    return os.getenv("OAUTH_CLIENT_ID", "").strip()


def oauth_client_secret() -> str:
    return os.getenv("OAUTH_CLIENT_SECRET", "").strip()


def oauth_configured() -> bool:
    return bool(oauth_client_id() and oauth_client_secret())


def resolve_redirect_uri(forwarded_host: str | None, host: str | None) -> str:
    """Redirect URI must use the Studio's ms.show host, not modelscope.cn.

    The Studio page on modelscope.cn embeds the app in an iframe, so the
    registered callback has to target the real app host. Behind the platform
    reverse proxy the original host arrives in ``x-forwarded-host``.
    """
    explicit = os.getenv("OAUTH_REDIRECT_URI", "").strip()
    if explicit:
        return explicit
    chosen = (forwarded_host or host or "").split(",")[0].strip()
    if not chosen:
        raise ValueError("Cannot determine the OAuth redirect host.")
    return f"https://{chosen}/oauth/callback"


def build_authorize_url(client_id: str, redirect_uri: str, state: str) -> str:
    query = urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": OAUTH_SCOPES,
            "state": state,
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


def exchange_code(
    code: str, client_id: str, client_secret: str, redirect_uri: str
) -> dict:
    """Exchange an authorization code for tokens. Never log the response."""
    response = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        },
        headers={"Accept": "application/json"},
        timeout=20.0,
    )
    if response.status_code >= 400:
        raise ValueError(f"ModelScope token endpoint returned HTTP {response.status_code}.")
    payload = response.json()
    if not isinstance(payload, dict) or not payload.get("access_token"):
        raise ValueError("ModelScope token response did not include an access token.")
    return payload


def fetch_userinfo(access_token: str) -> dict:
    try:
        response = httpx.get(
            USERINFO_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            timeout=12.0,
        )
    except httpx.HTTPError:
        return {}
    if response.status_code >= 400:
        return {}
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def display_name(userinfo: dict) -> str:
    for key in ("preferred_username", "username", "name", "nickname", "sub"):
        value = str(userinfo.get(key, "") or "").strip()
        if value:
            return value
    return "ModelScope user"


class OAuthSessionStore:
    """In-memory, process-local store linking browser sessions to tokens.

    A login nonce carries the browser session id through the new-tab OAuth
    flow (the sign-in tab cannot read the iframe's partitioned cookie, so the
    link embeds a one-time nonce instead). The nonce doubles as the OAuth
    ``state`` value and is consumed exactly once at the callback.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nonces: dict[str, tuple[str, float]] = {}
        self._sessions: dict[str, dict] = {}

    def _prune(self, now: float) -> None:
        expired_nonces = [k for k, (_, exp) in self._nonces.items() if exp <= now]
        for key in expired_nonces:
            del self._nonces[key]
        expired_sessions = [
            k for k, v in self._sessions.items() if v.get("expires_at", 0) <= now
        ]
        for key in expired_sessions:
            del self._sessions[key]

    def mint_login_nonce(self, session_id: str) -> str:
        nonce = secrets.token_urlsafe(24)
        now = time.time()
        with self._lock:
            self._prune(now)
            self._nonces[nonce] = (session_id, now + NONCE_TTL_SECONDS)
        return nonce

    def nonce_valid(self, nonce: str) -> bool:
        now = time.time()
        with self._lock:
            entry = self._nonces.get(nonce)
        return bool(entry and entry[1] > now)

    def consume_nonce(self, nonce: str) -> str | None:
        now = time.time()
        with self._lock:
            self._prune(now)
            entry = self._nonces.pop(nonce, None)
        if not entry:
            return None
        session_id, expires_at = entry
        return session_id if expires_at > now else None

    def store_token(self, session_id: str, token_payload: dict, userinfo: dict) -> None:
        now = time.time()
        lifetime = float(token_payload.get("expires_in") or SESSION_TTL_SECONDS)
        with self._lock:
            self._sessions[session_id] = {
                "access_token": str(token_payload["access_token"]),
                "expires_at": now + min(lifetime, SESSION_TTL_SECONDS),
                "user": display_name(userinfo),
            }

    def get_session(self, session_id: str | None) -> dict | None:
        if not session_id:
            return None
        now = time.time()
        with self._lock:
            self._prune(now)
            entry = self._sessions.get(session_id)
            return dict(entry) if entry else None

    def get_token(self, session_id: str | None) -> str | None:
        entry = self.get_session(session_id)
        return entry["access_token"] if entry else None

    def clear(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._lock:
            self._sessions.pop(session_id, None)


SESSION_STORE = OAuthSessionStore()
