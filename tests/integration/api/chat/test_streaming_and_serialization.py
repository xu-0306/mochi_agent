"""Chat streaming and serialization integration tests."""

from __future__ import annotations

from ._support import *  # noqa: F401,F403


def test_chat_stream_route_returns_sse_events_incrementally() -> None:
    """`POST /v1/chat/stream` 應以 SSE 逐筆送出 serialized chat events。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={"message": "現在幾點？", "session_id": "session-42"},
        ) as response:
            chunks = [
                line.removeprefix("data: ")
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]
            session_id = response.headers["x-session-id"]
            cache_control = response.headers["cache-control"]
            content_type = response.headers["content-type"]

    assert response.status_code == 200
    assert engine.chat_calls == [("現在幾點？", "session-42")]
    assert session_id == "session-42"
    assert cache_control == "no-cache"
    assert content_type.startswith("text/event-stream")

    events = [json.loads(chunk) for chunk in chunks]
    turn_ids = {event["turn_id"] for event in events}
    assert len(turn_ids) == 1
    turn_id = next(iter(turn_ids))
    assert turn_id
    assert events == [
        {"type": "thinking", "content": "分析中", "metadata": {}, "turn_id": turn_id},
        {
            "type": "tool_call_request",
            "call_id": "call-1",
            "tool_name": "clock",
            "arguments": {"timezone": "Asia/Taipei"},
            "turn_id": turn_id,
        },
            {
                "type": "tool_call_result",
                "call_id": "call-1",
                "tool_name": "clock",
                "result": {"now": "2026-04-27T09:30:00+00:00"},
                "error": None,
                "metadata": {},
                "turn_id": turn_id,
            },
        {
            "type": "final_answer",
            "content": "已收到：現在幾點？",
            "trajectory_id": "traj-123",
            "input_tokens": 128,
            "output_tokens": 32,
            "generation_time_ms": 250.0,
            "finish_reason": "stop",
            "turn_id": turn_id,
        },
    ]

def test_stream_chat_events_close_underlying_stream_on_generator_close(tmp_path: Path) -> None:
    closed = threading.Event()
    sessions_dir = tmp_path / "sessions"

    class _DisconnectEngine(_FakeEngine):
        async def chat(
            self,
            message: str,
            session_id: str | None = None,
            inference_overrides: dict[str, Any] | None = None,
            project_id: str | None = None,
            workspace_dir: str | None = None,
            selected_skill_ids: list[str] | None = None,
            attachments: list[AttachmentRef] | None = None,
        ) -> AsyncIterator[object]:
            _ = (inference_overrides, project_id, workspace_dir, selected_skill_ids, attachments)
            self.chat_calls.append((message, session_id))
            try:
                yield ThinkingEvent(content="stream-open")
                await asyncio.Future()
            finally:
                closed.set()

    app, engine = _build_app(engine=_DisconnectEngine())
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)
    app.state.runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(sessions_dir / "runtime-disconnect.db"),
    )

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    request = StarletteRequest(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/stream",
            "headers": [],
            "app": app,
        },
        _receive,
    )

    async def _run() -> None:
        stream = engine.chat("disconnect me", session_id="session-disconnect")
        wrapped = _stream_chat_events(
            request,
            "session-disconnect",
            stream,
            fallback_turn_id="turn-disconnect",
        )
        first = await anext(wrapped)
        assert first["type"] == "thinking"
        await wrapped.aclose()

    asyncio.run(_run())

    assert engine.chat_calls == [("disconnect me", "session-disconnect")]
    assert closed.wait(timeout=1.0)

def test_chat_stream_route_persists_fallback_turn_events(tmp_path: Path) -> None:
    """stream route 對未自行持久化 turn replay 的 engine 應補寫 session turn_event。"""
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
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={"message": "現在幾點？", "session_id": "session-stream"},
        ) as response:
            _ = list(response.iter_lines())
        get_response = client.get("/v1/sessions/session-stream")

    assert response.status_code == 200
    assert engine.chat_calls == [("現在幾點？", "session-stream")]

    store_events = asyncio.run(SessionStore(sessions_dir).load_session("session-stream"))
    assert [event["type"] for event in store_events] == ["turn_event"] * 4
    assert [event["phase"] for event in store_events] == [
        "thinking",
        "tool_call_request",
        "tool_call_result",
        "final_answer",
    ]
    assert [event["seq"] for event in store_events] == [1, 2, 3, 4]
    assert len({event["turn_id"] for event in store_events}) == 1
    assert store_events[0]["payload"]["turn_id"] == store_events[0]["turn_id"]
    assert store_events[-1]["payload"] == {
        "type": "final_answer",
        "content": "已收到：現在幾點？",
        "trajectory_id": "traj-123",
        "input_tokens": 128,
        "output_tokens": 32,
        "generation_time_ms": 250.0,
        "finish_reason": "stop",
        "turn_id": store_events[-1]["turn_id"],
    }

    assert get_response.status_code == 200
    assert get_response.json()["events"] == store_events
