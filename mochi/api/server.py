"""FastAPI 主伺服器。"""

from __future__ import annotations

import base64
import inspect
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import Body, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

from mochi.voice.capabilities import get_voice_capabilities
from mochi.voice.ws_bridge import VoiceWebSocketBridge


def create_app() -> FastAPI:
    """建立 API 應用。"""

    class DiscordVoiceJoinRequest(BaseModel):
        """Discord voice join 控制請求。"""

        guild_id: int = Field(..., gt=0)
        channel_id: int = Field(..., gt=0)

    class DiscordVoiceGuildRequest(BaseModel):
        """Discord voice guild 控制請求。"""

        guild_id: int = Field(..., gt=0)

    class DiscordVoiceChunkRequest(BaseModel):
        """Discord voice chunk ingest 請求。"""

        guild_id: int = Field(..., gt=0)
        chunk_base64: str = Field(..., min_length=1)
        speaker_id: str | None = None
        auto_end: bool = True

    class VoicePrepareRequest(BaseModel):
        """預載 WebGUI 語音 runtime 的請求。"""

        session_id: str | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """管理 app 生命週期資源。"""
        try:
            yield
        finally:
            await _shutdown_engine(app)

    app = FastAPI(
        title="Mochi API",
        description="Mochi AI Agent REST API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = None
    app.state.config_path = None
    app.state.engine = None
    app.state.engine_factory = None
    app.state.config_factory = None
    app.state.channel_manager = None
    app.state.session_store = None
    app.state.project_store = None
    app.state.skill_library = None
    app.state.vllm_runtime_manager = None
    app.state.voice_bridge_diagnostics = _create_voice_bridge_diagnostics_state()
    app.state.runtime_service = None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_resolve_initial_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from mochi.api.routes import (
        agent_runs_router,
        approvals_router,
        chat_router,
        file_ops_router,
        filesystem_router,
        model_auth_router,
        models_router,
        projects_router,
        sessions_router,
        settings_router,
        skills_router,
        tasks_router,
        voice_router,
        workspace_router,
    )

    app.include_router(chat_router)
    app.include_router(tasks_router)
    app.include_router(agent_runs_router)
    app.include_router(approvals_router)
    app.include_router(model_auth_router)
    app.include_router(models_router)
    app.include_router(skills_router)
    app.include_router(projects_router)
    app.include_router(sessions_router)
    app.include_router(settings_router)
    app.include_router(voice_router)
    app.include_router(filesystem_router)
    app.include_router(file_ops_router)
    app.include_router(workspace_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """健康檢查端點。"""
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/v1/voice/capabilities")
    async def voice_capabilities() -> dict[str, Any]:
        """回傳 `/v1/voice` 的共享能力描述。"""
        return get_voice_capabilities()

    @app.get("/v1/voice/status")
    async def voice_status() -> dict[str, Any]:
        """回傳共享 voice runtime 狀態。"""
        engine = await _get_or_create_engine(app)
        get_status = getattr(engine, "get_voice_runtime_status", None)
        if callable(get_status):
            payload = await _maybe_await(_call_with_supported_kwargs(get_status))
            if isinstance(payload, dict):
                return _attach_voice_bridge_diagnostics(app, payload)
        return _attach_voice_bridge_diagnostics(app, {
            "type": "voice_runtime_status",
            "phase": "bounded",
            "loaded": False,
            "error": "Engine does not provide get_voice_runtime_status().",
        })

    @app.post("/v1/voice/prepare")
    async def prepare_voice_runtime(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        """預先載入 WebGUI 語音所需 runtime，讓開始錄音可直接進入待機。"""
        engine = await _get_or_create_engine(app)
        session_id = payload.get("session_id") if isinstance(payload, dict) else None
        if session_id is not None:
            session_id = str(session_id)
        prepare_runtime = getattr(engine, "prepare_voice_runtime", None)
        if callable(prepare_runtime):
            prepared = await _maybe_await(
                _call_with_supported_kwargs(prepare_runtime, session_id=session_id)
            )
            if isinstance(prepared, dict):
                return _attach_voice_bridge_diagnostics(app, prepared)

        get_status = getattr(engine, "get_voice_runtime_status", None)
        if callable(get_status):
            status = await _maybe_await(_call_with_supported_kwargs(get_status))
            if isinstance(status, dict):
                return _attach_voice_bridge_diagnostics(app, status)

        return _attach_voice_bridge_diagnostics(app, {
            "type": "voice_runtime_status",
            "phase": "bounded",
            "loaded": False,
            "error": "Engine does not provide prepare_voice_runtime().",
        })

    @app.get("/v1/channels")
    async def channels_status() -> dict[str, Any]:
        """回傳 Phase 4.5 channel bounded status，不啟動任何 bot。"""
        config = await _get_config(app)
        manager = cast(Any | None, getattr(app.state, "channel_manager", None))
        registered = set(manager.list_channels()) if manager is not None else set()
        running_channels = (
            set(manager.running_channels())
            if manager is not None and callable(getattr(manager, "running_channels", None))
            else set()
        )

        return {
            "type": "channels_status",
            "phase": "bounded",
            "supported_channels": ["discord", "telegram"],
            "channels": {
                "discord": _channel_status(
                    config=config,
                    name="discord",
                    registered=registered,
                    running="discord" in running_channels,
                    manager=manager,
                ),
                "telegram": _channel_status(
                    config=config,
                    name="telegram",
                    registered=registered,
                    running="telegram" in running_channels,
                    manager=manager,
                ),
            },
        }

    @app.post("/v1/channels/start")
    async def start_all_channels() -> dict[str, Any]:
        """啟動所有已註冊頻道。"""
        manager = _require_channel_manager(app)
        await _maybe_await(manager.start_all())
        return {
            "type": "channels_control",
            "action": "start",
            "scope": "all",
            "running_channels": _safe_running_channels(manager),
        }

    @app.post("/v1/channels/stop")
    async def stop_all_channels() -> dict[str, Any]:
        """停止所有已註冊頻道。"""
        manager = _require_channel_manager(app)
        await _maybe_await(manager.stop_all())
        return {
            "type": "channels_control",
            "action": "stop",
            "scope": "all",
            "running_channels": _safe_running_channels(manager),
        }

    @app.post("/v1/channels/{channel_name}/start")
    async def start_channel(channel_name: str) -> dict[str, Any]:
        """啟動單一已註冊頻道。"""
        try:
            manager = await _ensure_channel_manager(app)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if channel_name not in set(_safe_registered_channels(manager)):
            raise HTTPException(status_code=404, detail="Channel not registered")
        start_one = getattr(manager, "start_channel", None)
        if not callable(start_one):
            raise HTTPException(
                status_code=501,
                detail="Channel manager does not support start_channel",
            )
        await _maybe_await(start_one(channel_name))
        return {
            "type": "channels_control",
            "action": "start",
            "scope": "single",
            "channel": channel_name,
            "running": channel_name in set(_safe_running_channels(manager)),
            "running_channels": _safe_running_channels(manager),
        }

    @app.post("/v1/channels/{channel_name}/stop")
    async def stop_channel(channel_name: str) -> dict[str, Any]:
        """停止單一已註冊頻道。"""
        manager = _require_channel_manager(app)
        if channel_name not in set(_safe_registered_channels(manager)):
            raise HTTPException(status_code=404, detail="Channel not registered")
        stop_one = getattr(manager, "stop_channel", None)
        if not callable(stop_one):
            raise HTTPException(
                status_code=501,
                detail="Channel manager does not support stop_channel",
            )
        await _maybe_await(stop_one(channel_name))
        return {
            "type": "channels_control",
            "action": "stop",
            "scope": "single",
            "channel": channel_name,
            "running": channel_name in set(_safe_running_channels(manager)),
            "running_channels": _safe_running_channels(manager),
        }

    @app.post("/v1/channels/discord/voice/join")
    async def discord_voice_join(request: DiscordVoiceJoinRequest) -> dict[str, Any]:
        """要求 Discord bot 加入指定語音頻道。"""
        manager = _require_channel_manager(app)
        channel = _require_registered_channel(manager, "discord")
        join_voice = getattr(channel, "join_voice_channel", None)
        if not callable(join_voice):
            raise HTTPException(
                status_code=501,
                detail="Discord adapter does not support join_voice_channel",
            )
        try:
            room = await _maybe_await(join_voice(request.guild_id, request.channel_id))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "type": "discord_voice_control",
            "action": "join",
            "guild_id": request.guild_id,
            "channel_id": request.channel_id,
            "room": room,
        }

    @app.post("/v1/channels/discord/voice/leave")
    async def discord_voice_leave(request: DiscordVoiceGuildRequest) -> dict[str, Any]:
        """要求 Discord bot 離開指定 guild 的語音房。"""
        manager = _require_channel_manager(app)
        channel = _require_registered_channel(manager, "discord")
        leave_voice = getattr(channel, "leave_voice_channel", None)
        if not callable(leave_voice):
            raise HTTPException(
                status_code=501,
                detail="Discord adapter does not support leave_voice_channel",
            )
        try:
            left = bool(await _maybe_await(leave_voice(request.guild_id)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "type": "discord_voice_control",
            "action": "leave",
            "guild_id": request.guild_id,
            "left": left,
        }

    @app.post("/v1/channels/discord/voice/interrupt")
    async def discord_voice_interrupt(request: DiscordVoiceGuildRequest) -> dict[str, Any]:
        """中斷指定 guild 的 Discord 語音播放。"""
        manager = _require_channel_manager(app)
        channel = _require_registered_channel(manager, "discord")
        interrupt_voice = getattr(channel, "interrupt_voice_playback", None)
        if not callable(interrupt_voice):
            raise HTTPException(
                status_code=501,
                detail="Discord adapter does not support interrupt_voice_playback",
            )
        try:
            interrupted = bool(await _maybe_await(interrupt_voice(request.guild_id)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "type": "discord_voice_control",
            "action": "interrupt",
            "guild_id": request.guild_id,
            "interrupted": interrupted,
        }

    @app.post("/v1/channels/discord/voice/end-turn")
    async def discord_voice_end_turn(request: DiscordVoiceGuildRequest) -> dict[str, Any]:
        """要求 Discord runtime 結束目前 buffered voice turn。"""
        manager = _require_channel_manager(app)
        channel = _require_registered_channel(manager, "discord")
        end_turn = getattr(channel, "end_voice_turn", None)
        if not callable(end_turn):
            raise HTTPException(
                status_code=501,
                detail="Discord adapter does not support end_voice_turn",
            )
        try:
            started = bool(await _maybe_await(end_turn(request.guild_id)))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "type": "discord_voice_control",
            "action": "end_turn",
            "guild_id": request.guild_id,
            "started": started,
        }

    @app.post("/v1/channels/discord/voice/ingest")
    async def discord_voice_ingest(request: DiscordVoiceChunkRequest) -> dict[str, Any]:
        """將 Discord voice chunk 送入 runtime。"""
        manager = _require_channel_manager(app)
        channel = _require_registered_channel(manager, "discord")
        ingest = getattr(channel, "ingest_voice_audio_chunk", None)
        if not callable(ingest):
            raise HTTPException(
                status_code=501,
                detail="Discord adapter does not support ingest_voice_audio_chunk",
            )
        try:
            chunk = base64.b64decode(request.chunk_base64, validate=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio chunk: {exc}") from exc
        try:
            payload = await _maybe_await(ingest(
                request.guild_id,
                chunk=chunk,
                speaker_id=request.speaker_id,
                auto_end=request.auto_end,
            ))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "type": "discord_voice_control",
            "action": "ingest",
            "guild_id": request.guild_id,
            "payload": payload,
        }

    @app.websocket("/v1/voice")
    async def voice_websocket(websocket: WebSocket) -> None:
        """Phase 4 最小語音 websocket 端點。"""
        session_id = websocket.query_params.get("session_id")
        idle_timeout_seconds = _parse_voice_idle_timeout_seconds(
            websocket.query_params.get("idle_timeout_seconds"),
        )
        engine = await _get_or_create_engine(app)
        bridge = VoiceWebSocketBridge(
            engine=engine,
            session_id=session_id,
            auto_end_idle_timeout_seconds=idle_timeout_seconds,
        )
        try:
            await bridge.serve(websocket)
        finally:
            _merge_voice_bridge_diagnostics(app, bridge.get_diagnostics())
            release = getattr(engine, "release_voice_session", None)
            if callable(release):
                await _maybe_await(_call_with_supported_kwargs(release, session_id=session_id))

    return app


async def _get_or_create_engine(app: FastAPI) -> Any:
    """取得可供 API 共用的 AgentEngine。"""
    existing = cast(Any, app.state.engine)
    if existing is not None:
        return existing

    factory = cast(Callable[[], Any] | Callable[[FastAPI], Any] | None, app.state.engine_factory)
    if factory is not None:
        engine = await _maybe_await(_call_with_supported_kwargs(factory, app=app))
        app.state.engine = engine
        return engine

    from mochi.agents.engine import AgentEngine

    engine = AgentEngine(
        await _get_config(app),
        vllm_runtime_manager=_get_or_create_vllm_runtime_manager(app),
    )
    app.state.engine = engine
    return engine


def _get_or_create_vllm_runtime_manager(app: FastAPI) -> Any:
    manager = getattr(app.state, "vllm_runtime_manager", None)
    if manager is not None:
        return manager

    from mochi.backends.vllm_runtime import ManagedVLLMRuntimeManager

    manager = ManagedVLLMRuntimeManager()
    app.state.vllm_runtime_manager = manager
    return manager


async def _build_channel_manager_for_app(app: FastAPI) -> Any:
    """依目前 app config/engine 建立新的 channel manager。"""
    from mochi.channels.manager import build_channel_manager

    config = await _get_config(app)
    engine = await _get_or_create_engine(app)
    config_path = getattr(app.state, "config_path", None)
    persist_config_updates = (
        config_path is not None or getattr(app.state, "config_factory", None) is None
    )
    return build_channel_manager(
        config,
        engine,
        config_path=config_path,
        persist_config_updates=persist_config_updates,
    )


async def _ensure_channel_manager(app: FastAPI) -> Any:
    """確保 app 擁有可用的 channel manager，但不自動啟動頻道。"""
    existing = cast(Any | None, getattr(app.state, "channel_manager", None))
    if existing is not None:
        return existing

    manager = await _build_channel_manager_for_app(app)
    app.state.channel_manager = manager
    return manager


async def _rebuild_channel_manager(app: FastAPI) -> Any:
    """以最新 config/engine 重建 channel manager，保留先前 running set。"""
    previous = cast(Any | None, getattr(app.state, "channel_manager", None))
    previously_running = set(_safe_running_channels(previous)) if previous is not None else set()

    if previous is not None:
        stop_all = getattr(previous, "stop_all", None)
        if callable(stop_all):
            await _maybe_await(stop_all())

    manager = await _build_channel_manager_for_app(app)
    app.state.channel_manager = manager

    for name in sorted(previously_running):
        if name not in set(_safe_registered_channels(manager)):
            continue
        start_one = getattr(manager, "start_channel", None)
        if callable(start_one):
            await _maybe_await(start_one(name))

    return manager


async def _get_config(app: FastAPI) -> Any:
    """取得 API status endpoint 使用的設定物件。"""
    existing = cast(Any | None, getattr(app.state, "config", None))
    if existing is not None:
        return existing

    factory = cast(Callable[[], Any] | Callable[[FastAPI], Any] | None, app.state.config_factory)
    if factory is not None:
        return await _maybe_await(_call_with_supported_kwargs(factory, app=app))

    from mochi.config.manager import load_config

    return load_config()


def _channel_status(
    *,
    config: Any,
    name: str,
    registered: set[str],
    running: bool,
    manager: Any | None = None,
) -> dict[str, Any]:
    """建立單一 channel 的非敏感狀態。"""
    channel_config = getattr(getattr(config, "channels", None), name, None)
    payload = {
        "supported": True,
        "enabled": bool(getattr(channel_config, "enabled", False)),
        "registered": name in registered,
        "running": running and name in registered,
    }
    if name == "discord":
        payload.update(
            {
                "text_enabled": bool(getattr(channel_config, "text_enabled", False)),
                "voice_enabled": bool(getattr(channel_config, "voice_enabled", False)),
                "bot_token_configured": getattr(channel_config, "bot_token", None) is not None,
                "allowed_guild_ids": list(getattr(channel_config, "allowed_guild_ids", [])),
                "allowed_channel_ids": list(getattr(channel_config, "allowed_channel_ids", [])),
                "allowed_voice_channel_ids": list(
                    getattr(channel_config, "allowed_voice_channel_ids", [])
                ),
                "allowed_user_ids": list(getattr(channel_config, "allowed_user_ids", [])),
                "rate_limit_per_user": int(getattr(channel_config, "rate_limit_per_user", 0) or 0),
                "message_mode": getattr(channel_config, "message_mode", None),
                "auto_join_policy": getattr(channel_config, "auto_join_policy", None),
                "voice_auto_reply": bool(getattr(channel_config, "voice_auto_reply", False)),
                "voice_stt_enabled": bool(getattr(channel_config, "voice_stt_enabled", False)),
                "voice_tts_enabled": bool(getattr(channel_config, "voice_tts_enabled", False)),
            }
        )
    elif name == "telegram":
        payload.update(
            {
                "allowed_chat_ids": list(getattr(channel_config, "allowed_chat_ids", [])),
                "allowed_user_ids": list(getattr(channel_config, "allowed_user_ids", [])),
                "rate_limit_per_user": int(getattr(channel_config, "rate_limit_per_user", 0) or 0),
                "bot_token_configured": getattr(channel_config, "bot_token", None) is not None,
            }
        )

    runtime_details = _get_channel_runtime_status(manager=manager, name=name)
    if runtime_details:
        payload.update(runtime_details)

    return payload


def _get_channel_runtime_status(*, manager: Any | None, name: str) -> dict[str, Any]:
    """從已註冊 channel adapter 取得額外 runtime 狀態。"""
    if manager is None:
        return {}
    get_channel = getattr(manager, "get", None)
    if not callable(get_channel):
        return {}
    channel = get_channel(name)
    if channel is None:
        return {}
    get_status = getattr(channel, "get_runtime_status", None)
    if not callable(get_status):
        return {}
    payload = get_status()
    return payload if isinstance(payload, dict) else {}


def _require_channel_manager(app: FastAPI) -> Any:
    """取得 channel manager，若未注入則回傳 503。"""
    manager = cast(Any | None, getattr(app.state, "channel_manager", None))
    if manager is None:
        raise HTTPException(status_code=503, detail="Channel manager is not configured")
    return manager


def _require_registered_channel(manager: Any, name: str) -> Any:
    """取得指定已註冊 channel adapter。"""
    get_channel = getattr(manager, "get", None)
    if not callable(get_channel):
        raise HTTPException(status_code=503, detail="Channel manager does not support lookup")
    channel = get_channel(name)
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel not registered")
    return channel


def _safe_registered_channels(manager: Any) -> list[str]:
    """安全取得已註冊頻道列表。"""
    list_channels = getattr(manager, "list_channels", None)
    if not callable(list_channels):
        return []
    value = list_channels()
    if isinstance(value, list):
        return [name for name in value if isinstance(name, str)]
    return []


def _safe_running_channels(manager: Any) -> list[str]:
    """安全取得執行中頻道列表。"""
    running_channels = getattr(manager, "running_channels", None)
    if callable(running_channels):
        value = running_channels()
        if isinstance(value, list):
            return [name for name in value if isinstance(name, str)]

    # backward-compatible fallback for older manager implementation
    is_running = bool(getattr(manager, "_running", False))
    return _safe_registered_channels(manager) if is_running else []


async def _shutdown_engine(app: FastAPI) -> None:
    """關閉時釋放 engine 資源。"""
    manager = cast(Any | None, getattr(app.state, "channel_manager", None))
    if manager is not None:
        stop_all = getattr(manager, "stop_all", None)
        if callable(stop_all):
            await _maybe_await(stop_all())
        app.state.channel_manager = None

    runtime_service = cast(Any | None, getattr(app.state, "runtime_service", None))
    if runtime_service is not None:
        close_runtime_service = getattr(runtime_service, "close", None)
        if callable(close_runtime_service):
            await _maybe_await(close_runtime_service())
    app.state.runtime_service = None

    engine = cast(Any, app.state.engine)
    if engine is not None:
        close = getattr(engine, "close", None)
        if callable(close):
            await _maybe_await(close())
    app.state.engine = None

    vllm_manager = cast(Any | None, getattr(app.state, "vllm_runtime_manager", None))
    if vllm_manager is not None:
        stop = getattr(vllm_manager, "stop", None)
        if callable(stop):
            await _maybe_await(stop())
    app.state.vllm_runtime_manager = None


def _create_voice_bridge_diagnostics_state() -> dict[str, Any]:
    """建立 `/v1/voice` bridge 最小診斷累積器。"""
    return {
        "preview_append_failures": 0,
        "preview_flush_failures": 0,
        "preview_degraded_turns": 0,
        "last_preview_failure": None,
    }


def _merge_voice_bridge_diagnostics(app: FastAPI, diagnostics: dict[str, Any] | None) -> None:
    """將單條 websocket bridge 診斷累積到 app state。"""
    if not diagnostics:
        return

    state = cast(dict[str, Any] | None, getattr(app.state, "voice_bridge_diagnostics", None))
    if not isinstance(state, dict):
        state = _create_voice_bridge_diagnostics_state()
        app.state.voice_bridge_diagnostics = state

    for key in (
        "preview_append_failures",
        "preview_flush_failures",
        "preview_degraded_turns",
    ):
        value = diagnostics.get(key)
        if isinstance(value, int):
            state[key] = int(state.get(key, 0)) + value

    last_failure = diagnostics.get("last_preview_failure")
    if isinstance(last_failure, dict):
        state["last_preview_failure"] = dict(last_failure)


def _attach_voice_bridge_diagnostics(app: FastAPI, payload: dict[str, Any]) -> dict[str, Any]:
    """以 additive 欄位附加 bridge 診斷，不破壞既有 status surface。"""
    enriched = dict(payload)
    state = cast(dict[str, Any] | None, getattr(app.state, "voice_bridge_diagnostics", None))
    if isinstance(state, dict):
        enriched["bridge_diagnostics"] = {
            "preview_append_failures": int(state.get("preview_append_failures", 0)),
            "preview_flush_failures": int(state.get("preview_flush_failures", 0)),
            "preview_degraded_turns": int(state.get("preview_degraded_turns", 0)),
            "last_preview_failure": (
                dict(last_failure)
                if isinstance(last_failure := state.get("last_preview_failure"), dict)
                else None
            ),
        }
    return enriched


async def _maybe_await(value: Any | Awaitable[Any]) -> Any:
    """接受 sync/async 回傳值。"""
    if inspect.isawaitable(value):
        return await cast(Awaitable[Any], value)
    return value


def _call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """只傳入目標函式支援的 keyword 參數。"""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    accepted_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(*args, **accepted_kwargs)


def _parse_voice_idle_timeout_seconds(raw_value: str | None) -> float:
    """解析 `/v1/voice` 的可選 idle timeout 秒數。"""
    if raw_value is None:
        return VoiceWebSocketBridge._DEFAULT_AUTO_END_IDLE_TIMEOUT_SECONDS
    try:
        parsed = float(raw_value)
    except ValueError:
        return VoiceWebSocketBridge._DEFAULT_AUTO_END_IDLE_TIMEOUT_SECONDS
    return VoiceWebSocketBridge._normalize_idle_timeout(parsed)


def _resolve_initial_cors_origins() -> list[str]:
    """解析 app 啟動時的 CORS origins（可由設定檔或環境變數覆蓋）。"""
    from mochi.config.manager import load_config, read_env_cors_origins

    # Production deployments should set MOCHI_WEB_CORS_ORIGINS explicitly.
    # This lets app bootstrap keep the configured origins even if full config
    # validation is temporarily broken by an unrelated setting.
    env_origins = read_env_cors_origins()

    try:
        config = load_config()
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("Failed to load config for CORS setup: %s", exc)
        if env_origins is not None:
            return env_origins
        return ["http://localhost:3000"]

    origins = [
        origin.strip()
        for origin in config.web.cors_origins
        if isinstance(origin, str) and origin.strip()
    ]
    if "http://localhost:3000" in origins and "http://127.0.0.1:3000" not in origins:
        origins.append("http://127.0.0.1:3000")
    if "http://127.0.0.1:3000" in origins and "http://localhost:3000" not in origins:
        origins.append("http://localhost:3000")
    if origins:
        return origins
    return ["http://localhost:3000", "http://127.0.0.1:3000"]


app = create_app()
