"""Direct ModelScope OAuth 2.0 support for the RSMeM Studio.

The visitor signs in with their own ModelScope account and the Studio uses the
resulting access token to call ModelScope API-Inference with the visitor's own
quota. Tokens live only in this process's memory, keyed by a random browser
session id stored in a cookie; they are never written to disk, logs, state
dicts, or the client page.

Configuration comes from the Studio's built-in OAuth support: once the owner
enables it, ModelScope creates the OAuth application and injects
``OAUTH_CLIENT_ID``, ``OAUTH_CLIENT_SECRET``, ``OAUTH_SCOPES``,
``OPENID_PROVIDER_URL`` and ``STUDIO_HOST`` into the runtime. Nothing here may
be committed to the repository.
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

# The Studio page is embedded in an iframe on modelscope.cn, so a forwarded
# host under this domain is the platform shell rather than the Studio's own
# origin. Authorizing against it is what triggers ExternalRedirectForbidden.
PLATFORM_HOST_SUFFIXES = ("modelscope.cn", "modelscope.ai")

SESSION_COOKIE = "rsmem_sid"
NONCE_TTL_SECONDS = 15 * 60
SESSION_TTL_SECONDS = 12 * 60 * 60


def oauth_client_id() -> str:
    return os.getenv("OAUTH_CLIENT_ID", "").strip()


def oauth_client_secret() -> str:
    return os.getenv("OAUTH_CLIENT_SECRET", "").strip()


def oauth_configured() -> bool:
    return bool(oauth_client_id() and oauth_client_secret())


def _normalize_host(value: str | None) -> str:
    """Reduce a configured or forwarded host value to a bare ``host[:port]``."""
    chosen = (value or "").split(",")[0].strip()
    if not chosen:
        return ""
    for scheme in ("https://", "http://"):
        if chosen.lower().startswith(scheme):
            chosen = chosen[len(scheme) :]
            break
    return chosen.strip("/").strip()


def _is_platform_host(host: str) -> bool:
    bare = host.split(":")[0].lower().rstrip(".")
    return any(
        bare == suffix or bare.endswith(f".{suffix}")
        for suffix in PLATFORM_HOST_SUFFIXES
    )


def studio_host() -> str:
    """The Studio's own access domain, as injected by ModelScope.

    ``STUDIO_HOST`` is the documented general variable; ``SPACE_HOST`` is the
    Gradio-compatibility alias the platform injects alongside it. Both are
    documented as carrying no scheme, but tolerate one anyway.
    """
    for name in ("STUDIO_HOST", "SPACE_HOST"):
        host = _normalize_host(os.getenv(name))
        if host:
            return host
    return ""


def resolve_redirect_uri(forwarded_host: str | None, host: str | None) -> str:
    """Redirect URI must target the Studio's own domain.

    ModelScope rejects an authorization request whose ``redirect_uri`` points
    anywhere other than the Studio's access domain with
    ``ExternalRedirectForbidden``. Prefer the platform-injected host over the
    proxy headers, which report the modelscope.cn shell instead.
    """
    explicit = os.getenv("OAUTH_REDIRECT_URI", "").strip()
    if explicit:
        return explicit
    chosen = studio_host()
    if not chosen:
        for candidate in (forwarded_host, host):
            normalized = _normalize_host(candidate)
            if normalized and not _is_platform_host(normalized):
                chosen = normalized
                break
    if not chosen:
        raise ValueError("Cannot determine the OAuth redirect host.")
    return f"https://{chosen}/oauth/callback"


def oauth_scopes() -> str:
    """Requested scopes, honouring what the owner ticked in Studio settings.

    ``openid`` is mandatory for the authorization code flow, so it is added
    back even if the injected value omits it.
    """
    configured = (os.getenv("OAUTH_SCOPES", "") or "").split()
    scopes = configured or OAUTH_SCOPES.split()
    if "openid" not in scopes:
        scopes = ["openid", *scopes]
    return " ".join(scopes)


_ENDPOINT_DEFAULTS = {
    "authorization_endpoint": AUTHORIZE_URL,
    "token_endpoint": TOKEN_URL,
    "userinfo_endpoint": USERINFO_URL,
}
_endpoint_lock = threading.Lock()
_endpoint_cache: dict[str, str] | None = None


def discover_endpoints() -> dict[str, str]:
    """Resolve the OIDC endpoints, falling back to the published constants.

    ``OPENID_PROVIDER_URL`` is injected by the Studio runtime. Discovery is
    attempted once per process; any failure keeps the hard-coded endpoints so
    that sign-in never breaks because metadata was briefly unreachable.
    """
    global _endpoint_cache
    with _endpoint_lock:
        if _endpoint_cache is not None:
            return dict(_endpoint_cache)
        provider = (os.getenv("OPENID_PROVIDER_URL", "") or "").strip().rstrip("/")
        endpoints = dict(_ENDPOINT_DEFAULTS)
        if provider:
            try:
                response = httpx.get(
                    f"{provider}/.well-known/openid-configuration",
                    headers={"Accept": "application/json"},
                    timeout=10.0,
                )
                if response.status_code < 400:
                    payload = response.json()
                    if isinstance(payload, dict):
                        for key in _ENDPOINT_DEFAULTS:
                            value = str(payload.get(key, "") or "").strip()
                            if value:
                                endpoints[key] = value
            except (httpx.HTTPError, ValueError):
                pass
        _endpoint_cache = endpoints
        return dict(endpoints)


def reset_endpoint_cache() -> None:
    """Drop the discovery cache; used by tests and after a config change."""
    global _endpoint_cache
    with _endpoint_lock:
        _endpoint_cache = None


def build_authorize_url(client_id: str, redirect_uri: str, state: str) -> str:
    query = urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": oauth_scopes(),
            "state": state,
        }
    )
    return f"{discover_endpoints()['authorization_endpoint']}?{query}"


def exchange_code(
    code: str, client_id: str, client_secret: str, redirect_uri: str
) -> dict:
    """Exchange an authorization code for tokens. Never log the response."""
    response = httpx.post(
        discover_endpoints()["token_endpoint"],
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
            discover_endpoints()["userinfo_endpoint"],
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
