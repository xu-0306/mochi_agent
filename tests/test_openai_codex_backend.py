"""OpenAICodexBackend auth-refresh behavior tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from mochi.auth.models import OpenAICodexAuthProfile
from mochi.backends.base import BackendRequestError
from mochi.backends.openai_codex import OpenAICodexBackend
from mochi.backends.types import Message


def _mock_response(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


class _MockStreamContext:
    def __init__(
        self,
        lines: list[str] | None = None,
        *,
        raise_error: Exception | None = None,
    ) -> None:
        self._lines = lines or []
        self._raise_error = raise_error

    async def __aenter__(self) -> _MockStreamContext:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def raise_for_status(self) -> None:
        if self._raise_error is not None:
            raise self._raise_error

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


@pytest.mark.asyncio
async def test_codex_backend_refreshes_token_before_nonstream_request(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = OpenAICodexBackend(
        base_url="https://chatgpt.com/backend-api",
        model="gpt-5.4",
        access_token="stale-token",
        auth_profile_id="openai_codex:default",
        workspace_dir="H:/_python/agent_mochi",
    )

    class _FakeService:
        def resolve_access_token(self, profile_id: str | None = None) -> str:
            assert profile_id == "openai_codex:default"
            return "fresh-token"

    monkeypatch.setattr(backend, "_auth_service", lambda: _FakeService())
    mock_resp = _mock_response(
        {
            "model": "gpt-5.4",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
    )

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp) as post:
            result = await backend.generate(messages=[Message(role="user", content="hi")], stream=False)
    finally:
        await backend.close()

    assert result.content == "ok"
    assert post.await_args.kwargs["headers"]["Authorization"] == "Bearer fresh-token"


@pytest.mark.asyncio
async def test_codex_backend_retries_nonstream_request_after_401_with_refreshed_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenAICodexBackend(
        base_url="https://chatgpt.com/backend-api",
        model="gpt-5.4",
        access_token="stale-token",
        auth_profile_id="openai_codex:default",
        workspace_dir="H:/_python/agent_mochi",
    )

    request = httpx.Request("POST", "https://chatgpt.com/backend-api")
    unauthorized = httpx.HTTPStatusError(
        "unauthorized",
        request=request,
        response=httpx.Response(401, request=request, text="expired"),
    )

    class _FakeService:
        def resolve_access_token(self, profile_id: str | None = None) -> str:
            assert profile_id == "openai_codex:default"
            return "stale-token"

        def refresh_access_token(
            self,
            profile_id: str | None = None,
            *,
            force: bool = True,
        ) -> OpenAICodexAuthProfile:
            assert profile_id == "openai_codex:default"
            assert force is True
            return OpenAICodexAuthProfile(
                profile_id="openai_codex:default",
                access_token=SecretStr("fresh-token"),
                refresh_token=SecretStr("refresh-token"),
                expires_at=4_100_000_000,
            )

    monkeypatch.setattr(backend, "_auth_service", lambda: _FakeService())
    mock_resp = _mock_response(
        {
            "model": "gpt-5.4",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[unauthorized, mock_resp],
        ) as post:
            result = await backend.generate(messages=[Message(role="user", content="hi")], stream=False)
    finally:
        await backend.close()

    assert result.content == "ok"
    assert post.await_count == 2
    assert post.await_args_list[0].kwargs["headers"]["Authorization"] == "Bearer stale-token"
    assert post.await_args_list[1].kwargs["headers"]["Authorization"] == "Bearer fresh-token"


@pytest.mark.asyncio
async def test_codex_backend_raises_auth_specific_error_when_refresh_fails_after_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenAICodexBackend(
        base_url="https://chatgpt.com/backend-api",
        model="gpt-5.4",
        access_token="stale-token",
        auth_profile_id="openai_codex:default",
        workspace_dir="H:/_python/agent_mochi",
    )

    request = httpx.Request("POST", "https://chatgpt.com/backend-api")
    unauthorized = httpx.HTTPStatusError(
        "unauthorized",
        request=request,
        response=httpx.Response(401, request=request, text="expired"),
    )
    recorded_failures: list[tuple[str | None, str, bool]] = []

    class _FakeService:
        def resolve_access_token(self, profile_id: str | None = None) -> str:
            return "stale-token"

        def refresh_access_token(
            self,
            profile_id: str | None = None,
            *,
            force: bool = True,
        ) -> OpenAICodexAuthProfile:
            raise RuntimeError("invalid_grant")

        def record_auth_failure(
            self,
            profile_id: str | None,
            message: str,
            expire_now: bool = False,
        ) -> None:
            recorded_failures.append((profile_id, message, expire_now))

        def status_for_profile(self, profile_id: str | None = None) -> str:
            return "refresh_failed"

    monkeypatch.setattr(backend, "_auth_service", lambda: _FakeService())

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=unauthorized,
        ):
            with pytest.raises(BackendRequestError) as exc_info:
                await backend.generate(messages=[Message(role="user", content="hi")], stream=False)
    finally:
        await backend.close()

    exc = exc_info.value
    assert "invalid_grant" in str(exc)
    assert exc.metadata["backend_name"] == "openai_codex"
    assert exc.metadata["auth_status"] == "refresh_failed"
    assert recorded_failures == [
        ("openai_codex:default", "invalid_grant", True)
    ]


@pytest.mark.asyncio
async def test_codex_backend_retries_stream_request_after_401_with_refreshed_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenAICodexBackend(
        base_url="https://chatgpt.com/backend-api",
        model="gpt-5.4",
        access_token="stale-token",
        auth_profile_id="openai_codex:default",
        workspace_dir="H:/_python/agent_mochi",
    )

    request = httpx.Request("POST", "https://chatgpt.com/backend-api")
    unauthorized = httpx.HTTPStatusError(
        "unauthorized",
        request=request,
        response=httpx.Response(401, request=request, text="expired"),
    )

    class _FakeService:
        def resolve_access_token(self, profile_id: str | None = None) -> str:
            return "stale-token"

        def refresh_access_token(
            self,
            profile_id: str | None = None,
            *,
            force: bool = True,
        ) -> OpenAICodexAuthProfile:
            return OpenAICodexAuthProfile(
                profile_id="openai_codex:default",
                access_token=SecretStr("fresh-token"),
                refresh_token=SecretStr("refresh-token"),
                expires_at=4_100_000_000,
            )

    monkeypatch.setattr(backend, "_auth_service", lambda: _FakeService())
    stream_success = _MockStreamContext(
        [
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        ]
    )

    try:
        with patch.object(
            backend._client,
            "stream",
            side_effect=[
                _MockStreamContext(raise_error=unauthorized),
                stream_success,
            ],
        ) as stream:
            stream_iter = await backend.generate(messages=[Message(role="user", content="hi")], stream=True)
            chunks = [chunk async for chunk in stream_iter]
    finally:
        await backend.close()

    assert stream.call_count == 2
    assert chunks[0].delta == "ok"
    assert chunks[-1].is_final is True
