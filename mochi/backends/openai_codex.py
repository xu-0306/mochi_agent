"""OpenAI Codex backend using ChatGPT OAuth tokens."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast

from mochi.auth.openai_codex import OPENAI_CODEX_DEFAULT_BASE_URL, OpenAICodexAuthService
from mochi.backends.base import BackendRequestError
from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk, ToolSchema


class OpenAICodexBackend(OpenAICompatBackend):
    """Thin transport wrapper for OpenAI Codex responses."""

    def __init__(
        self,
        *,
        model: str,
        access_token: str,
        base_url: str = OPENAI_CODEX_DEFAULT_BASE_URL,
        timeout: float = 120.0,
        auth_profile_id: str | None = None,
        workspace_dir: str | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=access_token,
            timeout=timeout,
            provider="openai_codex",
        )
        self._auth_profile_id = auth_profile_id
        self._workspace_dir = workspace_dir

    def get_model_info(self) -> ModelInfo:
        info = super().get_model_info()
        metadata = dict(info.metadata)
        metadata["auth_profile_id"] = self._auth_profile_id
        metadata["workspace_dir"] = self._workspace_dir
        return ModelInfo(
            name=info.name,
            provider="openai_codex",
            backend_type="openai_codex",
            context_length=info.context_length,
            supports_tool_calling=info.supports_tool_calling,
            metadata=metadata,
        )

    async def health_check(self) -> bool:
        return bool(self.api_key.strip() and self.base_url.strip() and self.model.strip())

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        reasoning_effort: str | None = None,
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        try:
            await self._prepare_authenticated_request(force_refresh=False)
        except RuntimeError as exc:
            raise await self._raise_auth_specific_error(
                BackendRequestError(str(exc), metadata={"backend_name": "openai_codex"}),
                str(exc),
            )

        if stream:
            return self._stream_generate_with_auth_retry(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                reasoning_effort=reasoning_effort,
            )

        return await self._generate_once_with_auth_retry(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            reasoning_effort=reasoning_effort,
        )

    def _auth_service(self) -> OpenAICodexAuthService | None:
        if not self._workspace_dir:
            return None
        return OpenAICodexAuthService(self._workspace_dir)

    async def _prepare_authenticated_request(self, *, force_refresh: bool) -> None:
        service = self._auth_service()
        if service is None or self._auth_profile_id is None:
            return
        if force_refresh:
            profile = await asyncio.to_thread(
                lambda: service.refresh_access_token(self._auth_profile_id, force=True)
            )
            self.api_key = profile.access_token.get_secret_value()
            return
        self.api_key = await asyncio.to_thread(
            lambda: service.resolve_access_token(self._auth_profile_id)
        )

    async def _generate_once_with_auth_retry(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int,
        top_p: float,
        min_p: float,
        top_k: int,
        frequency_penalty: float,
        presence_penalty: float,
        repeat_penalty: float,
        reasoning_effort: str | None,
    ) -> GenerationResult:
        try:
            result = await super().generate(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repeat_penalty=repeat_penalty,
                reasoning_effort=reasoning_effort,
                stream=False,
            )
            return cast(GenerationResult, result)
        except BackendRequestError as exc:
            if not await self._should_retry_auth_failure(exc):
                raise
            try:
                await self._prepare_authenticated_request(force_refresh=True)
                result = await super().generate(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    min_p=min_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    repeat_penalty=repeat_penalty,
                    reasoning_effort=reasoning_effort,
                    stream=False,
                )
                return cast(GenerationResult, result)
            except RuntimeError as refresh_exc:
                raise await self._raise_auth_specific_error(
                    exc,
                    str(refresh_exc),
                )
            except BackendRequestError as retry_exc:
                raise await self._raise_auth_specific_error(
                    retry_exc,
                    "OpenAI Codex request was rejected after token refresh.",
                )

    async def _stream_generate_with_auth_retry(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int,
        top_p: float,
        min_p: float,
        top_k: int,
        frequency_penalty: float,
        presence_penalty: float,
        repeat_penalty: float,
        reasoning_effort: str | None,
    ) -> AsyncIterator[StreamChunk]:
        first_failure: BackendRequestError | None = None
        for attempt in range(2):
            try:
                if attempt == 1:
                    if first_failure is None:
                        break
                    await self._prepare_authenticated_request(force_refresh=True)
                stream_iter = await super().generate(
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    min_p=min_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    repeat_penalty=repeat_penalty,
                    reasoning_effort=reasoning_effort,
                    stream=True,
                )
                async for chunk in cast(AsyncIterator[StreamChunk], stream_iter):
                    yield chunk
                return
            except BackendRequestError as exc:
                if attempt == 0 and await self._should_retry_auth_failure(exc):
                    first_failure = exc
                    continue
                raise await self._raise_auth_specific_error(
                    exc,
                    "OpenAI Codex streaming request was rejected after token refresh.",
                )
            except RuntimeError as refresh_exc:
                if first_failure is not None:
                    raise await self._raise_auth_specific_error(first_failure, str(refresh_exc))
                raise

    async def _should_retry_auth_failure(self, exc: BackendRequestError) -> bool:
        return (
            self._auth_profile_id is not None
            and exc.metadata.get("status_code") == 401
        )

    async def _raise_auth_specific_error(
        self,
        exc: BackendRequestError,
        message: str,
    ) -> BackendRequestError:
        service = self._auth_service()
        auth_status = "unknown"
        if service is not None and self._auth_profile_id is not None:
            await asyncio.to_thread(
                lambda: service.record_auth_failure(
                    self._auth_profile_id,
                    message,
                    expire_now=True,
                )
            )
            auth_status = await asyncio.to_thread(
                lambda: service.status_for_profile(self._auth_profile_id)
            )
        metadata = dict(exc.metadata)
        metadata.update(
            {
                "backend_name": "openai_codex",
                "auth_profile_id": self._auth_profile_id,
                "auth_status": auth_status,
            }
        )
        return BackendRequestError(message, metadata=metadata)
