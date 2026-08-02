"""Chat streaming and serialization integration tests."""

from __future__ import annotations

from mochi.backends.base import BackendRequestError

from ._support import (
    Any,
    AsyncIterator,
    AttachmentRef,
    FinalAnswerEvent,
    MochiConfig,
    Path,
    RuntimeService,
    RuntimeStore,
    SessionStore,
    StarletteRequest,
    TestClient,
    ThinkingEvent,
    _build_app,
    _FakeEngine,
    _stream_chat_events,
    asyncio,
    json,
    threading,
)


def test_chat_stream_route_returns_sse_events_incrementally() -> None:
    """`POST /v1/chat/stream` 應以 SSE 逐筆送出 serialized chat events。"""
    app, engine = _build_app()

    with TestClient(app) as client, client.stream(
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


def test_stream_chat_events_classifies_provider_outage_without_exposing_raw_error(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    app, _ = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)
    app.state.runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(sessions_dir / "runtime-provider-outage.db"),
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

    async def _provider_outage() -> AsyncIterator[object]:
        raise BackendRequestError(
            "sensitive upstream detail",
            metadata={
                "backend_name": "openai_compat",
                "status_code": 503,
                "model": "gpt-5.6-luna",
            },
        )
        yield object()  # pragma: no cover

    async def _run() -> list[dict[str, Any]]:
        return [
            event
            async for event in _stream_chat_events(
                request,
                "session-provider-outage",
                _provider_outage(),
                fallback_turn_id="turn-provider-outage",
            )
        ]

    events = asyncio.run(_run())

    assert events == [
        {
            "type": "error",
            "error": (
                "The configured model is currently unavailable from its provider. "
                "No tools were run. Retry later or select an available model."
            ),
            "code": "MODEL_PROVIDER_UNAVAILABLE",
            "metadata": {
                "classification": "provider_unavailable",
                "retryable": True,
                "backend_name": "openai_compat",
                "status_code": 503,
            },
            "turn_id": "turn-provider-outage",
        }
    ]
    assert "sensitive upstream detail" not in str(events)


def test_stream_chat_events_classifies_provider_connect_failure(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    app, _ = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)
    app.state.runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(sessions_dir / "runtime-provider-connect-failure.db"),
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

    async def _provider_connect_failure() -> AsyncIterator[object]:
        raise BackendRequestError(
            "All connection attempts failed",
            metadata={
                "backend_name": "openai_compat",
                "model": "gpt-5.6-luna",
            },
        )
        yield object()  # pragma: no cover

    async def _run() -> list[dict[str, Any]]:
        return [
            event
            async for event in _stream_chat_events(
                request,
                "session-provider-connect-failure",
                _provider_connect_failure(),
                fallback_turn_id="turn-provider-connect-failure",
            )
        ]

    events = asyncio.run(_run())

    assert events == [
        {
            "type": "error",
            "error": (
                "The configured model is currently unavailable from its provider. "
                "No tools were run. Retry later or select an available model."
            ),
            "code": "MODEL_PROVIDER_UNAVAILABLE",
            "metadata": {
                "classification": "provider_unavailable",
                "retryable": True,
                "backend_name": "openai_compat",
                "status_code": None,
            },
            "turn_id": "turn-provider-connect-failure",
        }
    ]


def test_stream_chat_events_exposes_safe_provider_access_denied_diagnostics(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    app, _ = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)
    app.state.runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(sessions_dir / "runtime-provider-access-denied.db"),
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

    secret_detail = "upstream response may contain credential-adjacent diagnostics"

    async def _provider_access_denied() -> AsyncIterator[object]:
        raise BackendRequestError(
            secret_detail,
            metadata={
                "backend_name": "openai_compat",
                "status_code": 403,
                "model": "gpt-5.4",
                "response_text": secret_detail,
            },
        )
        yield object()  # pragma: no cover

    async def _run() -> list[dict[str, Any]]:
        return [
            event
            async for event in _stream_chat_events(
                request,
                "session-provider-access-denied",
                _provider_access_denied(),
                fallback_turn_id="turn-provider-access-denied",
            )
        ]

    events = asyncio.run(_run())

    assert events == [
        {
            "type": "error",
            "error": (
                "The model provider denied the request (HTTP 403 Forbidden). "
                "No tools were run. Check the API key's model/provider permissions "
                "or select another model."
            ),
            "code": "MODEL_PROVIDER_ACCESS_DENIED",
            "metadata": {
                "classification": "provider_access_denied",
                "retryable": False,
                "backend_name": "openai_compat",
                "status_code": 403,
                "model": "gpt-5.4",
            },
            "turn_id": "turn-provider-access-denied",
        }
    ]
    assert secret_detail not in str(events)


def test_stream_chat_events_redacts_unclassified_backend_failure(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    app, _ = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)
    app.state.runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(sessions_dir / "runtime-backend-failure.db"),
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

    secret_detail = "opaque-sensitive-detail-42"

    async def _backend_failure() -> AsyncIterator[object]:
        raise BackendRequestError(secret_detail)
        yield object()  # pragma: no cover

    async def _run() -> list[dict[str, Any]]:
        return [
            event
            async for event in _stream_chat_events(
                request,
                "session-backend-failure",
                _backend_failure(),
                fallback_turn_id="turn-backend-failure",
            )
        ]

    events = asyncio.run(_run())

    assert events == [
        {
            "type": "error",
            "error": (
                "The configured model request failed. No tools were run. "
                "Retry later or select another model."
            ),
            "code": "MODEL_REQUEST_FAILED",
            "metadata": {
                "classification": "unclassified_backend_failure",
                "retryable": False,
            },
            "turn_id": "turn-backend-failure",
        }
    ]
    assert secret_detail not in str(events)


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
    persisted_turn_id = store_events[0]["turn_id"]
    assert [event["event_id"] for event in store_events] == [
        f"{persisted_turn_id}:1",
        f"{persisted_turn_id}:2",
        f"{persisted_turn_id}:3",
        f"{persisted_turn_id}:4",
    ]


def test_chat_stream_route_does_not_duplicate_engine_replay_events(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir)

    class _PersistingEngine(_FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.store = store

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
            turn_id: str | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[object]:
            del kwargs
            assert session_id is not None
            assert turn_id is not None
            self.chat_calls.append((message, session_id))
            for seq, event in enumerate(
                (
                    ThinkingEvent(content="engine persisted"),
                    FinalAnswerEvent(content="done", trajectory_id="traj-persisted"),
                ),
                start=1,
            ):
                event.turn_id = turn_id
                event._session_replay_persisted = True  # type: ignore[attr-defined]
                event._session_replay_event_id = f"{turn_id}:{seq}"  # type: ignore[attr-defined]
                event._session_replay_seq = seq  # type: ignore[attr-defined]
                await self.store.save_event(
                    session_id,
                    {
                        "type": "turn_event",
                        "schema_version": 1,
                        "turn_id": turn_id,
                        "event_id": f"{turn_id}:{seq}",
                        "seq": seq,
                        "phase": event.type,
                        "timestamp": "2026-07-31T00:00:00+00:00",
                        "payload": {"type": event.type, "content": getattr(event, "content", "")},
                    },
                )
                yield event

    engine = _PersistingEngine()
    app, _ = _build_app(engine=engine)
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = store

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={"message": "hello", "session_id": "session-engine-authority"},
        ) as response:
            _ = list(response.iter_lines())

    assert response.status_code == 200
    store_events = asyncio.run(store.load_session("session-engine-authority"))
    assert [event["phase"] for event in store_events] == ["thinking", "final_answer"]
    assert [event["event_id"] for event in store_events] == [
        f"{store_events[0]['turn_id']}:1",
        f"{store_events[0]['turn_id']}:2",
    ]


def test_chat_stream_route_appends_only_unpersisted_error_after_engine_failure(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir)

    class _PartiallyPersistingEngine(_FakeEngine):
        def __init__(self) -> None:
            super().__init__()
            self.store = store

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
            turn_id: str | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[object]:
            del kwargs
            assert session_id is not None
            assert turn_id is not None
            self.chat_calls.append((message, session_id))
            event = ThinkingEvent(content="before failure")
            event.turn_id = turn_id
            event._session_replay_persisted = True  # type: ignore[attr-defined]
            event._session_replay_event_id = f"{turn_id}:1"  # type: ignore[attr-defined]
            event._session_replay_seq = 1  # type: ignore[attr-defined]
            await self.store.save_event(
                session_id,
                {
                    "type": "turn_event",
                    "schema_version": 1,
                    "turn_id": turn_id,
                    "event_id": f"{turn_id}:1",
                    "seq": 1,
                    "phase": "thinking",
                    "timestamp": "2026-07-31T00:00:00+00:00",
                    "payload": {"type": "thinking", "content": event.content},
                },
            )
            yield event
            raise BackendRequestError(
                "provider failure",
                metadata={"backend_name": "openai_compat", "status_code": 503},
            )

    engine = _PartiallyPersistingEngine()
    app, _ = _build_app(engine=engine)
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = store

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={"message": "hello", "session_id": "session-partial-engine"},
        ) as response:
            _ = list(response.iter_lines())

    assert response.status_code == 200
    store_events = asyncio.run(store.load_session("session-partial-engine"))
    assert [event["phase"] for event in store_events] == ["thinking", "error"]
    assert [event["seq"] for event in store_events] == [1, 2]
    assert [event["event_id"] for event in store_events] == [
        f"{store_events[0]['turn_id']}:1",
        f"{store_events[0]['turn_id']}:2",
    ]
