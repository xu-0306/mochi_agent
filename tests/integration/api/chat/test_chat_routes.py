"""Chat API route integration tests."""

from __future__ import annotations

from ._support import *  # noqa: F401,F403


def test_chat_route_returns_bounded_response_with_serialized_events() -> None:
    """`POST /v1/chat` 應收斂事件流並回傳 final answer/trajectory。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"message": "現在幾點？", "session_id": "session-42"},
        )

    assert response.status_code == 200
    assert engine.chat_calls == [("現在幾點？", "session-42")]
    payload = response.json()
    assert payload["turn_id"]
    payload["turn_id"] = "turn-id"
    assert payload == {
        "type": "chat_response",
        "session_id": "session-42",
        "turn_id": "turn-id",
        "final_answer": "已收到：現在幾點？",
        "trajectory_id": "traj-123",
        "events": [
            {"type": "thinking", "content": "分析中", "metadata": {}},
            {
                "type": "tool_call_request",
                "call_id": "call-1",
                "tool_name": "clock",
                "arguments": {"timezone": "Asia/Taipei"},
            },
            {
                "type": "tool_call_result",
                "call_id": "call-1",
                "tool_name": "clock",
                "result": {"now": "2026-04-27T09:30:00+00:00"},
                "error": None,
                "metadata": {},
            },
            {
                "type": "final_answer",
                "content": "已收到：現在幾點？",
                "trajectory_id": "traj-123",
                "input_tokens": 128,
                "output_tokens": 32,
                "generation_time_ms": 250.0,
                "finish_reason": "stop",
            },
        ],
    }

def test_chat_route_applies_selected_available_model_before_chat() -> None:
    """`POST /v1/chat` 帶模型 id 時應先切換到該模型再執行對話。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": "openai_compat:https://api.example.com/v1:gpt-test",
                        "provider": "openai_compat",
                        "model": "gpt-test",
                        "model_spec": "https://api.example.com/v1",
                        "base_url": "https://api.example.com/v1",
                    }
                ]
            },
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-secret-value",
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={
                "message": "hello",
                "session_id": "session-model",
                "model": "openai_compat:https://api.example.com/v1:gpt-test",
            },
        )

    assert response.status_code == 200
    assert engine.openai_switch_calls == [
        ("https://api.example.com/v1", "gpt-test", "sk-secret-value", "openai_compat")
    ]
    assert engine.chat_calls == [("hello", "session-model")]
    assert "sk-secret-value" not in response.text

def test_chat_route_does_not_reswitch_current_model() -> None:
    """聊天頁每輪都帶目前模型 id 時，後端不應重複切換已 active 的模型。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "ollama": {"base_url": "http://localhost:11434"},
            "model_setup": {
                "configured_models": [
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                    }
                ]
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={
                "message": "hello",
                "session_id": "session-model",
                "model": "ollama:qwen2.5",
            },
        )

    assert response.status_code == 200
    assert engine.ollama_switch_calls == []
    assert engine.chat_calls == [("hello", "session-model")]

def test_chat_route_persists_turn_events_and_sessions_route_returns_them(tmp_path: Path) -> None:
    """`POST /v1/chat` 後應將 replay `turn_event` 寫入 session JSONL。"""
    sessions_dir = tmp_path / "sessions"
    app, engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)

    with TestClient(app) as client:
        post_response = client.post(
            "/v1/chat",
            json={"message": "現在幾點？", "session_id": "session-42"},
        )
        get_response = client.get("/v1/sessions/session-42")

    assert post_response.status_code == 200
    assert engine.chat_calls == [("現在幾點？", "session-42")]

    store_events = asyncio.run(SessionStore(sessions_dir).load_session("session-42"))
    assert [event["type"] for event in store_events] == ["turn_event"] * 4
    assert [event["phase"] for event in store_events] == [
        "thinking",
        "tool_call_request",
        "tool_call_result",
        "final_answer",
    ]
    assert [event["seq"] for event in store_events] == [1, 2, 3, 4]
    assert all(event["schema_version"] == 1 for event in store_events)
    assert len({event["event_id"] for event in store_events}) == 4
    assert len({event["turn_id"] for event in store_events}) == 1
    assert store_events[0]["payload"] == {"type": "thinking", "content": "分析中", "metadata": {}}
    assert store_events[-1]["payload"] == {
        "type": "final_answer",
        "content": "已收到：現在幾點？",
        "trajectory_id": "traj-123",
        "input_tokens": 128,
        "output_tokens": 32,
        "generation_time_ms": 250.0,
        "finish_reason": "stop",
    }

    assert get_response.status_code == 200
    assert get_response.json()["events"] == store_events

def test_response_language_addendum_tracks_traditional_chinese_messages() -> None:
    addendum = _build_response_language_prompt_addendum(
        "same_as_user",
        "幫我查詢 ESG 相關 LLM 微調資訊，方法等",
    )

    assert addendum is not None
    assert "Reply in the same language as the user's latest message" in addendum
    assert "Traditional Chinese" in addendum
    assert "current user message is in Traditional Chinese" in addendum

def test_response_language_addendum_tracks_japanese_messages_without_chinese_bias() -> None:
    addendum = _build_response_language_prompt_addendum(
        "same_as_user",
        "ハイ",
    )

    assert addendum is not None
    assert "Reply in the same language as the user's latest message" in addendum
    assert "The current user message is in Japanese. Reply in Japanese." in addendum
    assert "Traditional Chinese" not in addendum

def test_response_language_addendum_tracks_latin_script_messages_without_language_switch() -> None:
    addendum = _build_response_language_prompt_addendum(
        "same_as_user",
        "hi there",
    )

    assert addendum is not None
    assert "Reply in the same language as the user's latest message" in addendum
    assert "The current user message is written in a Latin-script language." in addendum
    assert "Traditional Chinese" not in addendum

def test_response_language_addendum_respects_explicit_language_preference() -> None:
    addendum = _build_response_language_prompt_addendum(
        "en-US",
        "請用中文回覆這段測試",
    )

    assert addendum is not None
    assert "Default response language: en-US." in addendum
    assert "Keep using that language unless the user explicitly requests another language." in addendum

def test_merge_prompt_addenda_preserves_existing_invocation_context() -> None:
    merged = _merge_prompt_addenda(
        "Language Policy:\n- Reply in Traditional Chinese.",
        "Goal context:\n- Active goal is blocked.",
    )

    assert merged == (
        "Language Policy:\n- Reply in Traditional Chinese.\n\n"
        "Goal context:\n- Active goal is blocked."
    )
