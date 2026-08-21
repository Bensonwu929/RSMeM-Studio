"""Tests for the dual connection modes and the direct ModelScope OAuth flow."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import gradio as gr

import app
import modelscope_oauth as msoauth


class FakeRequest:
    def __init__(self, cookies=None):
        self.cookies = dict(cookies or {})


OAUTH_ENV = {"OAUTH_CLIENT_ID": "test-client-id", "OAUTH_CLIENT_SECRET": "test-client-secret"}


class OAuthSessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = msoauth.OAuthSessionStore()

    def test_nonce_links_browser_session_and_is_single_use(self):
        nonce = self.store.mint_login_nonce("sid-1")
        self.assertTrue(self.store.nonce_valid(nonce))
        self.assertEqual(self.store.consume_nonce(nonce), "sid-1")
        self.assertIsNone(self.store.consume_nonce(nonce))
        self.assertFalse(self.store.nonce_valid(nonce))

    def test_expired_nonce_is_rejected(self):
        nonce = self.store.mint_login_nonce("sid-1")
        with patch.object(msoauth.time, "time", return_value=msoauth.time.time() + msoauth.NONCE_TTL_SECONDS + 1):
            self.assertIsNone(self.store.consume_nonce(nonce))

    def test_token_roundtrip_and_sign_out(self):
        self.store.store_token(
            "sid-1", {"access_token": "tok", "expires_in": 3600},
            {"preferred_username": "alice"},
        )
        session = self.store.get_session("sid-1")
        self.assertEqual(session["user"], "alice")
        self.assertEqual(self.store.get_token("sid-1"), "tok")
        self.store.clear("sid-1")
        self.assertIsNone(self.store.get_token("sid-1"))

    def test_expired_token_requires_new_sign_in(self):
        self.store.store_token("sid-1", {"access_token": "tok", "expires_in": 60}, {})
        with patch.object(msoauth.time, "time", return_value=msoauth.time.time() + 120):
            self.assertIsNone(self.store.get_token("sid-1"))


class OAuthUrlTests(unittest.TestCase):
    def test_authorize_url_carries_required_oauth_params(self):
        url = msoauth.build_authorize_url(
            "cid", "https://wbx929-rsmem.ms.show/oauth/callback", "state-1"
        )
        self.assertTrue(url.startswith(msoauth.AUTHORIZE_URL + "?"))
        params = parse_qs(urlparse(url).query)
        self.assertEqual(params["client_id"], ["cid"])
        self.assertEqual(params["response_type"], ["code"])
        self.assertEqual(params["state"], ["state-1"])
        self.assertEqual(params["scope"], ["openid profile api-inference"])
        self.assertEqual(
            params["redirect_uri"], ["https://wbx929-rsmem.ms.show/oauth/callback"]
        )

    def test_redirect_uri_prefers_forwarded_host_over_proxy_host(self):
        with patch.dict(os.environ, {"OAUTH_REDIRECT_URI": ""}):
            uri = msoauth.resolve_redirect_uri("wbx929-rsmem.ms.show", "internal:7860")
        self.assertEqual(uri, "https://wbx929-rsmem.ms.show/oauth/callback")


class ConnectionResolutionTests(unittest.TestCase):
    def tearDown(self):
        msoauth.SESSION_STORE.clear("sid-test")

    def test_modelscope_mode_without_sign_in_blocks_execution(self):
        with patch.dict(os.environ, OAUTH_ENV):
            with self.assertRaises(gr.Error) as ctx:
                app._resolve_connection(
                    app.MODE_MODELSCOPE, "", "", "", "", FakeRequest()
                )
        self.assertIn("Sign in with ModelScope", str(ctx.exception))

    def test_unconfigured_modelscope_mode_blocks_execution(self):
        with patch.dict(os.environ, {"OAUTH_CLIENT_ID": "", "OAUTH_CLIENT_SECRET": ""}):
            with self.assertRaises(gr.Error) as ctx:
                app._resolve_connection(
                    app.MODE_MODELSCOPE, "", "", "", "", FakeRequest()
                )
        self.assertIn("not configured", str(ctx.exception))

    def test_modelscope_mode_uses_visitor_token_and_quota_endpoint(self):
        msoauth.SESSION_STORE.store_token(
            "sid-test", {"access_token": "visitor-token", "expires_in": 600}, {}
        )
        with patch.dict(os.environ, OAUTH_ENV):
            base_url, model, api_key = app._resolve_connection(
                app.MODE_MODELSCOPE, "  ", "", "", "pasted-key-ignored",
                FakeRequest({msoauth.SESSION_COOKIE: "sid-test"}),
            )
        self.assertEqual(base_url, msoauth.INFERENCE_BASE_URL)
        self.assertEqual(model, app.MODELSCOPE_DEFAULT_MODEL)
        self.assertEqual(api_key, "visitor-token")

    def test_openai_mode_never_uses_the_modelscope_session(self):
        msoauth.SESSION_STORE.store_token(
            "sid-test", {"access_token": "visitor-token", "expires_in": 600}, {}
        )
        request = FakeRequest({msoauth.SESSION_COOKIE: "sid-test"})
        with patch.dict(os.environ, {"RSMEM_API_KEY": ""}):
            base_url, model, api_key = app._resolve_connection(
                app.MODE_OPENAI, "", "https://example.com/v1", "m1", "user-key", request
            )
            self.assertEqual((base_url, model, api_key), ("https://example.com/v1", "m1", "user-key"))
            with self.assertRaises(gr.Error):
                app._resolve_connection(app.MODE_OPENAI, "", "", "", "", request)

    def test_switching_to_modelscope_clears_the_pasted_key(self):
        ms_pane, oa_pane, key_update = app.switch_connection_mode(app.MODE_MODELSCOPE)
        self.assertTrue(ms_pane["visible"])
        self.assertFalse(oa_pane["visible"])
        self.assertEqual(key_update["value"], "")
        ms_pane, oa_pane, _ = app.switch_connection_mode(app.MODE_OPENAI)
        self.assertFalse(ms_pane["visible"])
        self.assertTrue(oa_pane["visible"])


class SessionStatusTests(unittest.TestCase):
    def tearDown(self):
        msoauth.SESSION_STORE.clear("sid-status")

    def test_disconnected_status_offers_sign_in_link_with_nonce(self):
        with patch.dict(os.environ, OAUTH_ENV):
            html, signed_in = app._ms_status_html(
                FakeRequest({msoauth.SESSION_COOKIE: "sid-status"})
            )
        self.assertFalse(signed_in)
        self.assertIn("Sign in with ModelScope", html)
        self.assertIn("oauth/login?nonce=", html)

    def test_connected_status_shows_user_but_never_the_token(self):
        msoauth.SESSION_STORE.store_token(
            "sid-status", {"access_token": "secret-token", "expires_in": 600},
            {"preferred_username": "alice"},
        )
        with patch.dict(os.environ, OAUTH_ENV):
            html, signed_in = app._ms_status_html(
                FakeRequest({msoauth.SESSION_COOKIE: "sid-status"})
            )
        self.assertTrue(signed_in)
        self.assertIn("alice", html)
        self.assertNotIn("secret-token", html)


class OAuthRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        cls.client = TestClient(app.create_app())

    def test_deployed_host_cookie_is_partitioned_for_the_iframe(self):
        attributes = app._session_cookie_attributes("wbx929-rsmem.ms.show")
        self.assertIn("SameSite=None", attributes)
        self.assertIn("Secure", attributes)
        self.assertIn("Partitioned", attributes)
        local = app._session_cookie_attributes("127.0.0.1:7861")
        self.assertIn("SameSite=Lax", local)
        self.assertNotIn("Partitioned", local)

    def test_root_response_sets_the_session_cookie(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(msoauth.SESSION_COOKIE, response.cookies)

    def test_config_response_sets_the_session_cookie_for_the_platform_shell(self):
        self.client.cookies.clear()
        response = self.client.get("/config")
        self.assertEqual(response.status_code, 200)
        self.assertIn(msoauth.SESSION_COOKIE, response.cookies)

    def test_login_without_configuration_returns_503(self):
        with patch.dict(os.environ, {"OAUTH_CLIENT_ID": "", "OAUTH_CLIENT_SECRET": ""}):
            response = self.client.get("/oauth/login", params={"nonce": "x"})
        self.assertEqual(response.status_code, 503)

    def test_login_with_stale_nonce_is_rejected(self):
        with patch.dict(os.environ, OAUTH_ENV):
            response = self.client.get("/oauth/login", params={"nonce": "stale"})
        self.assertEqual(response.status_code, 400)

    def test_login_redirects_to_modelscope_authorize(self):
        nonce = msoauth.SESSION_STORE.mint_login_nonce("sid-route")
        with patch.dict(os.environ, OAUTH_ENV):
            response = self.client.get(
                "/oauth/login", params={"nonce": nonce}, follow_redirects=False
            )
        self.assertIn(response.status_code, (302, 307))
        self.assertTrue(response.headers["location"].startswith(msoauth.AUTHORIZE_URL))
        self.assertIn("api-inference", response.headers["location"])

    def test_callback_with_unknown_state_is_rejected(self):
        with patch.dict(os.environ, OAUTH_ENV):
            response = self.client.get(
                "/oauth/callback", params={"code": "abc", "state": "unknown"}
            )
        self.assertEqual(response.status_code, 400)

    def test_callback_stores_the_token_server_side_only(self):
        nonce = msoauth.SESSION_STORE.mint_login_nonce("sid-cb")
        with patch.dict(os.environ, OAUTH_ENV), patch.object(
            msoauth, "exchange_code",
            return_value={"access_token": "cb-token", "expires_in": 600},
        ), patch.object(
            msoauth, "fetch_userinfo", return_value={"preferred_username": "alice"}
        ):
            response = self.client.get(
                "/oauth/callback", params={"code": "abc", "state": nonce}
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("alice", response.text)
        self.assertNotIn("cb-token", response.text)
        self.assertEqual(msoauth.SESSION_STORE.get_token("sid-cb"), "cb-token")
        msoauth.SESSION_STORE.clear("sid-cb")


if __name__ == "__main__":
    unittest.main()
