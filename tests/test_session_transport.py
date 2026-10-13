"""Homepage/session controls stay usable when the hosted queue is unavailable."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import app
import modelscope_oauth as msoauth


@pytest.fixture
def direct_client(monkeypatch):
    monkeypatch.setenv("OAUTH_CLIENT_ID", "test-client")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("STUDIO_HOST", "wbx929-rsmem.ms.show")
    monkeypatch.setattr(msoauth, "SESSION_STORE", msoauth.OAuthSessionStore())
    monkeypatch.setattr(app.run_storage, "sweep_local_runs", lambda: 0)
    monkeypatch.setattr(app.run_storage, "sweep_downloads", lambda: 0)
    server = app.create_app()
    # Match the hosted app: queued functions reject the direct POST route.
    monkeypatch.setattr(app.demo, "api_open", False)
    queue_push = AsyncMock(side_effect=AssertionError("Session UI used the queue"))
    monkeypatch.setattr(app.demo._queue, "push", queue_push)
    with TestClient(server) as client:
        yield client
    queue_push.assert_not_called()


def post_event(client, event, data):
    response = client.post(
        f"/gradio_api/run/{event['api_name']}",
        json={"data": data, "fn_index": event["id"], "session_hash": "session-test"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")
    return response.json()["data"]


def events_for(handler):
    return [
        event for event in app.demo.get_config_file()["dependencies"]
        if app.demo.fns[event["id"]].fn is handler
    ]


def test_homepage_initializes_and_restores_login_without_queue_or_cookies(direct_client):
    config = direct_client.get("/config").json()
    load_event = next(
        event for event in config["dependencies"]
        if any(target[1] == "load" for target in event["targets"])
        and app.demo.fns[event["id"]].fn is app.refresh_ms_session
    )
    assert load_event["queue"] is False
    # The ModelScope shell can drop Set-Cookie. BrowserState must still work.
    direct_client.cookies.clear()
    status, signout, sid = post_event(direct_client, load_event, [""])
    assert "not connected" in status
    assert signout["visible"] is False
    assert sid

    msoauth.SESSION_STORE.store_token(
        sid, {"access_token": "test-token", "expires_in": 600},
        {"preferred_username": "alice"},
    )
    status, signout, restored_sid = post_event(direct_client, load_event, [sid])
    assert restored_sid == sid
    assert "alice" in status
    assert signout["visible"] is True
    assert "test-token" not in status


def test_refresh_and_signout_return_json_without_queue(direct_client):
    refresh = next(
        event for event in events_for(app.refresh_ms_session)
        if any(target[1] == "click" for target in event["targets"])
    )
    signout = events_for(app.sign_out_ms)[0]
    assert refresh["queue"] is False
    assert signout["queue"] is False
    sid = "existing-browser-session"
    msoauth.SESSION_STORE.store_token(
        sid, {"access_token": "test-token", "expires_in": 600},
        {"preferred_username": "alice"},
    )
    status, _, _ = post_event(direct_client, refresh, [sid])
    assert "alice" in status
    status, button, returned_sid = post_event(direct_client, signout, [sid])
    assert "not connected" in status
    assert button["visible"] is False
    assert returned_sid == sid
    assert msoauth.SESSION_STORE.get_token(sid) is None


def test_mode_switch_clears_pasted_key_and_model_execution_stays_queued(direct_client):
    event = events_for(app.switch_connection_mode)[0]
    assert event["queue"] is False
    ms_pane, api_pane, api_key = post_event(direct_client, event, [app.MODE_MODELSCOPE])
    assert ms_pane["visible"] is True
    assert api_pane["visible"] is False
    assert api_key["value"] == ""
    ms_pane, api_pane, _ = post_event(direct_client, event, [app.MODE_OPENAI])
    assert ms_pane["visible"] is False
    assert api_pane["visible"] is True

    for handler in (
        app.start_workflow_ui,
        app.request_custom_reflections_ui,
        app.confirm_reflection_and_retry_ui,
    ):
        workflow_event = events_for(handler)[0]
        assert workflow_event["queue"] is True
