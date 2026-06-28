"""Mochi CLI entry point."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shlex
import sys
from collections import deque
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mochi import __version__
from mochi.voice.audio_io import (
    BaseAudioIO,
    create_default_audio_io,
    read_audio_file_as_pcm16,
    write_audio_file_from_pcm16,
)
from mochi.voice.events import VoiceStageEvent

app = typer.Typer(
    name="mochi",
    help="Mochi - lightweight voice-first self-learning AI agent",
    add_completion=False,
)
model_app = typer.Typer(help="Model management")
app.add_typer(model_app, name="model")
channels_app = typer.Typer(help="Discord / Telegram channel management")
app.add_typer(channels_app, name="channels")
skills_app = typer.Typer(help="Skill library management")
app.add_typer(skills_app, name="skills")
console = Console()
DEFAULT_TUI_SESSION_ID = "default"
DEFAULT_TUI_MAX_TURNS = 100


def _health_status_markup(is_ready: bool) -> str:
    """將 health_check 結果轉為 CLI 標記字串。"""
    return "[green]ready[/green]" if is_ready else "[red]unavailable[/red]"


def _describe_backend_issue(
    backend_type: str | None,
    metadata: dict | None,
    error: str | None,
) -> str | None:
    """將底層錯誤轉為較易讀的診斷說明。"""
    if error:
        return error

    meta = metadata or {}
    if meta.get("dependency_ready") is False:
        return "runtime dependency is not ready"

    if backend_type == "gguf" and meta.get("model_path"):
        return f"model file is not available: {meta['model_path']}"

    if backend_type == "safetensors" and meta.get("model_dir"):
        return f"model directory is not available: {meta['model_dir']}"

    if backend_type == "ollama":
        return "ollama backend is not reachable"

    if backend_type == "openai_compat":
        return "OpenAI-compatible endpoint is not reachable"

    return None


def _describe_voice_runtime_event(event: object) -> tuple[str, str] | None:
    """將 voice runtime 狀態事件轉為通用 CLI 顯示文字。"""
    if isinstance(event, VoiceStageEvent):
        return "Stage", event.stage

    if isinstance(event, dict):
        event_type = str(event.get("type", ""))
        if event_type == "voice_stage":
            return "Stage", str(event.get("stage", ""))
        if event_type == "vad_state":
            return "VAD", str(event.get("state", ""))

    event_type = getattr(event, "type", None)
    if event_type == "voice_stage":
        return "Stage", str(getattr(event, "stage", ""))
    if event_type == "vad_state":
        return "VAD", str(getattr(event, "state", ""))

    return None


def _find_audio_runtime_diagnostics_helper(audio_io: object) -> tuple[str | None, object | None]:
    """尋找 audio_io 提供的 runtime/device 診斷 helper。"""
    helper_names = (
        "runtime_diagnostics",
        "get_runtime_diagnostics",
        "diagnose_runtime",
        "diagnostics",
        "device_diagnostics",
        "get_device_diagnostics",
    )
    for name in helper_names:
        helper = getattr(audio_io, name, None)
        if callable(helper):
            return name, helper
    return None, None


async def _inspect_configured_model(
    model_spec: str,
    ollama_base_url: str,
) -> tuple[object | None, bool, str | None]:
    """解析目前 configured model，並回傳模型資訊與 readiness。"""
    from mochi.backends.router import BackendRouter

    router = BackendRouter(ollama_base_url=ollama_base_url)
    try:
        backend = await router.load(model_spec)
    except Exception as exc:
        return None, False, str(exc)

    try:
        info = backend.get_model_info()
        try:
            is_ready = await backend.health_check()
            error: str | None = None
        except Exception as exc:
            is_ready = False
            error = str(exc)
        return info, is_ready, error
    finally:
        await backend.close()


def _default_skills_db_path() -> Path:
    """依目前設定推導預設技能庫 SQLite 路徑。"""
    from mochi.config.manager import load_config
    from mochi.learning.skill_library_factory import resolve_skills_db_path

    cfg = load_config()
    return resolve_skills_db_path(
        skills_dir=cfg.skills_dir,
    )


def _default_skills_dir() -> Path:
    """依目前設定推導預設技能目錄。"""
    from mochi.config.manager import load_config

    cfg = load_config()
    return Path(os.path.expanduser(cfg.skills_dir))


def _auto_sync_filesystem_skills_enabled() -> bool:
    """依目前設定判斷 CLI 是否自動同步 filesystem skills。"""
    from mochi.config.manager import load_config

    cfg = load_config()
    return cfg.learning.auto_sync_filesystem_skills


def _resolve_skills_db_path(db_path: str | None) -> Path:
    """解析 CLI 指定或設定推導的技能庫路徑。"""
    if db_path:
        return Path(os.path.expanduser(db_path))
    return _default_skills_db_path()


def _print_channels_setup_guide(platform: str) -> None:
    """輸出頻道平台的 setup guide。"""
    normalized = platform.strip().lower()
    if normalized != "discord":
        console.print(f"[yellow]Unsupported channel guide: {platform}[/yellow]")
        console.print("[dim]Currently available: discord[/dim]")
        raise typer.Exit(code=1)

    lines = [
        "[bold]Discord Setup Guide[/bold]",
        "",
        "1. Create a Discord application and bot in the Discord Developer Portal.",
        "2. Enable the Message Content Intent in the Bot settings.",
        "3. Invite the bot with scopes: bot, applications.commands.",
        "4. Grant permissions: View Channels, Send Messages, Read Message History, Use Application Commands, Connect, Speak.",
        "5. Install channels dependencies:",
        "   uv sync --extra channels --active",
        "6. Set your bot token in the environment:",
        "   PowerShell: $env:DISCORD_BOT_TOKEN=\"your-token\"",
        "7. Create a Mochi user config or a local config file with Discord enabled.",
        "8. Start the bot:",
        "   uv run mochi channels run",
        "",
        "[bold]Minimal config example[/bold]",
        "channels:",
        "  discord:",
        "    enabled: true",
        "    text_enabled: true",
        "    voice_enabled: true",
        "    bot_token: null",
        "    allowed_guild_ids: []",
        "    allowed_channel_ids: []",
        "    allowed_voice_channel_ids: []",
        "    allowed_user_ids: []",
        "    rate_limit_per_user: 10",
        "    message_mode: \"mentions_only\"",
        "    auto_join_policy: \"manual_only\"",
        "    voice_auto_reply: true",
        "    voice_stt_enabled: true",
        "    voice_tts_enabled: true",
        "voice:",
        "  enabled: true",
        "  stt_backend: \"faster-whisper\"",
        "  tts_backend: \"kokoro-tts\"",
        "  sample_rate: 16000",
        "  channels: 1",
        "",
        "[bold]Notes[/bold]",
        "- WebGUI can currently show Discord status and guidance, but it does not yet accept bot tokens directly.",
        "- For safety, prefer keeping the token in DISCORD_BOT_TOKEN instead of writing it into a tracked config file.",
    ]
    for line in lines:
        console.print(line)


def _skill_value(skill: object, key: str, default: object = "") -> object:
    """從 Skill 物件或 dict 取得欄位值。"""
    if isinstance(skill, dict):
        return skill.get(key, default)
    return getattr(skill, key, default)


def _skill_to_jsonable(skill: object) -> object:
    """將 Skill 物件轉為 JSON 可序列化資料。"""
    if isinstance(skill, dict):
        return skill
    if hasattr(skill, "model_dump"):
        return skill.model_dump()
    if hasattr(skill, "__dict__"):
        return dict(skill.__dict__)
    return skill


def _format_json_payload(payload: object) -> str:
    """格式化 SkillLibrary export payload。"""
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return payload
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_skill_to_jsonable)


@app.command()
def version() -> None:
    """Show version information."""
    console.print(Panel(
        Text(f"Mochi v{__version__}", justify="center", style="bold cyan"),
        subtitle="lightweight voice-first self-learning AI agent",
    ))


@app.callback(invoke_without_command=True)
def root_callback(ctx: typer.Context) -> None:
    """處理 root CLI 入口；無子命令時啟動文字 TUI。"""
    if ctx.invoked_subcommand is not None:
        return
    asyncio.run(
        _chat_tui_async(
            model=None,
            config_path=None,
            session_id=DEFAULT_TUI_SESSION_ID,
            max_turns=DEFAULT_TUI_MAX_TURNS,
        )
    )


@app.command()
def chat(
    text: Annotated[str, typer.Argument(help="Message to send to the agent")],
    model: Annotated[str, typer.Option("--model", "-m", help="Model spec")] = "",
    config_path: Annotated[str, typer.Option("--config", "-c", help="Config file path")] = "",
    session_id: Annotated[
        str,
        typer.Option("--session-id", help="Text chat session ID"),
    ] = DEFAULT_TUI_SESSION_ID,
) -> None:
    """Send one message and print the agent response."""
    asyncio.run(
        _chat_async_terminal(
            text,
            model or None,
            config_path or None,
            session_id or DEFAULT_TUI_SESSION_ID,
        )
    )


@app.command("tui")
def chat_tui(
    model: Annotated[str, typer.Option("--model", "-m", help="Model spec")] = "",
    config_path: Annotated[str, typer.Option("--config", "-c", help="Config file path")] = "",
    session_id: Annotated[
        str,
        typer.Option("--session-id", help="Initial text chat session ID"),
    ] = DEFAULT_TUI_SESSION_ID,
    max_turns: Annotated[
        int,
        typer.Option("--max-turns", help="Maximum user turns in this interactive session"),
    ] = DEFAULT_TUI_MAX_TURNS,
) -> None:
    """Start bounded interactive text TUI chat."""
    asyncio.run(
        _chat_tui_async(
            model=model or None,
            config_path=config_path or None,
            session_id=session_id or DEFAULT_TUI_SESSION_ID,
            max_turns=max_turns,
        )
    )


@app.command()
def doctor() -> None:
    """Run system diagnostics for Ollama and the local environment."""
    asyncio.run(_doctor_async())


@app.command()
def voice(
    config_path: Annotated[str, typer.Option("--config", "-c", help="Config file path")] = "",
    session_id: Annotated[str, typer.Option("--session-id", help="Voice chat session ID")] = "voice-cli",
    max_record_seconds: Annotated[
        float,
        typer.Option("--max-record-seconds", help="Maximum recording length per turn in seconds"),
    ] = 6.0,
    playback: Annotated[
        bool,
        typer.Option("--playback/--no-playback", help="Play the TTS result"),
    ] = True,
    input_audio: Annotated[
        str,
        typer.Option("--input-audio", help="Input PCM16 or WAV audio file. Skips recording"),
    ] = "",
    output_audio: Annotated[
        str,
        typer.Option(
            "--output-audio",
            help="Output TTS audio file. .wav writes WAV; other extensions keep PCM16",
        ),
    ] = "",
    continuous: Annotated[
        bool,
        typer.Option("--continuous/--single", help="Enable local continuous voice mode"),
    ] = False,
    chunk_seconds: Annotated[
        float,
        typer.Option("--chunk-seconds", help="Recording chunk length in continuous mode, in seconds"),
    ] = 0.25,
    max_turns: Annotated[
        int,
        typer.Option(
            "--max-turns",
            help="Maximum turns to process in continuous mode. 0 means until the recording limit",
        ),
    ] = 0,
) -> None:
    """Run voice mode for a single turn or local continuous turns."""
    asyncio.run(
        _voice_async(
            config_path=config_path or None,
            session_id=session_id or None,
            max_record_seconds=max_record_seconds,
            playback=playback,
            input_audio=input_audio or None,
            output_audio=output_audio or None,
            continuous=continuous,
            chunk_seconds=chunk_seconds,
            max_turns=max_turns,
        )
    )


@model_app.command("list")
def model_list(
    config_path: Annotated[str, typer.Option("--config", "-c", help="Config file path")] = "",
) -> None:
    """List the configured model and supported model spec formats."""
    from mochi.config.manager import load_config

    cfg = load_config(config_path or None)
    console.print("[bold]Current Model Configuration[/bold]")
    console.print(f"  configured: [cyan]{cfg.model}[/cyan]\n")
    console.print("[bold]Supported Model Spec Formats[/bold]")
    console.print("  ollama:<model>")
    console.print("  /path/to/model.gguf")
    console.print("  /path/to/model_dir/")
    console.print("  http://host/v1")


@model_app.command("info")
def model_info(
    config_path: Annotated[str, typer.Option("--config", "-c", help="Config file path")] = "",
) -> None:
    """Show information about the configured model."""
    asyncio.run(_model_info_async(config_path or None))


@model_app.command("switch")
def model_switch(
    model_spec: Annotated[str, typer.Argument(help="New model spec")],
    config_path: Annotated[str, typer.Option("--config", "-c", help="Config file path")] = "",
) -> None:
    """Load and switch to a model for this run."""
    asyncio.run(_model_switch_async(model_spec, config_path or None))


@channels_app.command("run")
def channels_run(
    config_path: Annotated[str, typer.Option("--config", "-c", help="Config file path")] = "",
) -> None:
    """Start enabled Discord / Telegram bots."""
    asyncio.run(_channels_run_async(config_path or None))


@channels_app.command("guide")
def channels_guide(
    platform: Annotated[str, typer.Argument(help="Platform name: discord or telegram")] = "discord",
) -> None:
    """Show setup guide for a channel platform."""
    _print_channels_setup_guide(platform)


@channels_app.command("voice-settings")
def channels_voice_settings(
    config_path: Annotated[str, typer.Option("--config", "-c", help="Config file path")] = "",
    tts_voice: Annotated[
        str,
        typer.Option("--tts-voice", help="Discord voice reply TTS voice id"),
    ] = "",
    session_mode: Annotated[
        str,
        typer.Option(
            "--session-mode",
            help="Discord voice session mode: voice_room/shared/per_guild/per_channel",
        ),
    ] = "",
    reply_model_mode: Annotated[
        str,
        typer.Option("--reply-model-mode", help="Discord voice reply model mode: agent-default/fixed"),
    ] = "",
    reply_model: Annotated[
        str,
        typer.Option("--reply-model", help="Discord voice fixed reply model id"),
    ] = "",
) -> None:
    """Update shared Discord voice conversation settings."""
    asyncio.run(
        _channels_voice_settings_async(
            config_path=config_path or None,
            tts_voice=tts_voice or None,
            session_mode=session_mode or None,
            reply_model_mode=reply_model_mode or None,
            reply_model=reply_model or None,
        )
    )


@skills_app.command("list")
def skills_list(
    db_path: Annotated[str, typer.Option("--db", help="Skill library SQLite path")] = "",
) -> None:
    """List learned skills."""
    asyncio.run(_skills_list_async(db_path or None))


@skills_app.command("show")
def skills_show(
    skill_id: Annotated[str, typer.Argument(help="Skill ID")],
    db_path: Annotated[str, typer.Option("--db", help="Skill library SQLite path")] = "",
) -> None:
    """Show skill details."""
    asyncio.run(_skills_show_async(skill_id, db_path or None))


@skills_app.command("delete")
def skills_delete(
    skill_id: Annotated[str, typer.Argument(help="Skill ID")],
    db_path: Annotated[str, typer.Option("--db", help="Skill library SQLite path")] = "",
) -> None:
    """Delete a skill."""
    asyncio.run(_skills_delete_async(skill_id, db_path or None))


@skills_app.command("export")
def skills_export(
    db_path: Annotated[str, typer.Option("--db", help="Skill library SQLite path")] = "",
    output_path: Annotated[
        str,
        typer.Option("--output", "-o", help="Output JSON file path. Omit to print to stdout"),
    ] = "",
) -> None:
    """Export the skill library as JSON."""
    asyncio.run(_skills_export_async(db_path or None, output_path or None))


async def _chat_async(
    text: str,
    model: str | None,
    config_path: str | None,
    session_id: str,
) -> None:
    """非同步執行對話。"""
    from mochi.agents.engine import AgentEngine
    from mochi.agents.events import ErrorEvent, FinalAnswerEvent, TextChunkEvent
    from mochi.config.manager import load_config

    cfg = load_config(config_path)
    if model:
        cfg.model = model

    engine = AgentEngine(cfg)
    runtime_service = None
    try:
        await engine.initialize()
    except Exception as exc:
        console.print(f"[red]Initialization failed: {exc}[/red]")
        sys.exit(1)

    console.print(f"[dim]Model: {cfg.model}[/dim]")
    console.print()

    full_reply = ""
    try:
        async for event in engine.chat(text):
            if isinstance(event, TextChunkEvent):
                console.print(event.content, end="", highlight=False)
                full_reply += event.content
            elif isinstance(event, FinalAnswerEvent) and not full_reply:
                console.print(event.content, highlight=False)
            elif isinstance(event, ErrorEvent):
                console.print(f"\n[red]Error: {event.message}[/red]")
                break
    finally:
        if runtime_service is not None:
            await runtime_service.close()
        await engine.close()

    console.print()


async def _chat_async_terminal(
    text: str,
    model: str | None,
    config_path: str | None,
    session_id: str,
) -> None:
    """Send one text turn through the shared terminal goal-aware flow."""
    from mochi.agents.engine import AgentEngine
    from mochi.agents.events import ErrorEvent, FinalAnswerEvent, TextChunkEvent
    from mochi.config import defaults
    from mochi.config.manager import load_config
    from mochi.config.schema import SecurityConfig
    from mochi.sessions.store import SessionStore

    cfg = load_config(config_path)
    if model:
        cfg.model = model

    engine = AgentEngine(cfg)
    runtime_service: object | None = None
    current_session = session_id.strip() or DEFAULT_TUI_SESSION_ID

    def _sessions_dir() -> str:
        return str(getattr(cfg, "sessions_dir", defaults.default_sessions_dir()))

    def _ensure_security_config() -> SecurityConfig:
        current = getattr(cfg, "security", None)
        if isinstance(current, SecurityConfig):
            return current
        payload = {}
        if current is not None and hasattr(current, "__dict__"):
            payload = dict(current.__dict__)
        normalized = SecurityConfig.model_validate(payload)
        setattr(cfg, "security", normalized)
        return normalized

    session_store = SessionStore(sessions_dir=_sessions_dir())

    async def _build_runtime_service() -> object:
        _ensure_security_config()
        return await _create_tui_runtime_service(
            engine=engine,
            config=cfg,
            config_path=config_path,
        )

    async def _ensure_runtime_service() -> object:
        nonlocal runtime_service
        if runtime_service is None:
            runtime_service = await _build_runtime_service()
        return runtime_service

    try:
        await engine.initialize()
    except Exception as exc:
        console.print(f"[red]Initialization failed: {exc}[/red]")
        sys.exit(1)

    console.print(f"[dim]Model: {cfg.model}[/dim]")
    console.print(f"[dim]Session: {current_session}[/dim]")
    console.print()

    full_reply = ""
    try:
        routing_result = await _handle_terminal_goal_input(
            text=text,
            session_id=current_session,
            current_model=cfg.model,
            autonomy_mode=_ensure_security_config().autonomy_mode,
            session_store=session_store,
            ensure_runtime_service=_ensure_runtime_service,
            intent_invoker=engine,
        )
        if routing_result["handled"]:
            return

        chat_text = routing_result["chat_text"] or text
        async for event in engine.chat(chat_text, session_id=current_session):
            if isinstance(event, TextChunkEvent):
                console.print(event.content, end="", highlight=False)
                full_reply += event.content
            elif isinstance(event, FinalAnswerEvent) and not full_reply:
                console.print(event.content, highlight=False)
            elif isinstance(event, ErrorEvent):
                console.print(f"\n[red]Error: {event.message}[/red]")
                break
    finally:
        if runtime_service is not None:
            await runtime_service.close()
        await engine.close()

    console.print()


def _print_tui_help() -> None:
    """輸出互動式 TUI 可用 slash 指令。"""
    console.print("[bold]Slash Commands[/bold]")
    console.print("  /help                Show this help")
    console.print("  /exit                Exit interactive mode")
    console.print("  /clear               Clear the current session history")
    console.print("  /model               Show current model")
    console.print("  /model <spec>        Switch model for this session")
    console.print("  /session             Show current session id")
    console.print("  /session <id>        Switch session id")
    console.print("  /goal <request>      Prepare a long-running single-agent goal")
    console.print("  /goal                Show the active goal summary or goal help")
    console.print("  /goal status         Show the active goal status")
    console.print("  /goal pause          Pause the active goal")
    console.print("  /goal resume         Resume the active goal")
    console.print("  /goal stop           Stop the active goal or clear a pending proposal")
    console.print("  /workflow <request>  Prepare a workflow goal")
    console.print("  /chat <request>      Send one turn outside the active goal lane")
    console.print("  /approvals           Show approval requests")
    console.print("  /approve <id>        Approve one request once")
    console.print("  /approve-save <id>   Approve and save the suggested exec rule")
    console.print("  /reject <id>         Reject one approval request")
    console.print("  /exec-read <id>      Read the approval-bound exec session")
    console.print("  /exec-stop <id>      Stop the approval-bound exec session")
    console.print("  /safety              Show current safety mode")
    console.print("  /safety <mode>       Set safety mode")
    console.print("  /tools               Show web search / fetch tool settings")
    console.print("  /tools search-engine <engine>")
    console.print("                       Set primary web search engine")
    console.print("  /tools fallback <engine...>")
    console.print("                       Set fallback web search engines")
    console.print("  /tools fetch-extractor <extractor>")
    console.print("                       Set web fetch extractor")
    console.print("  /tools key-status    Show web search provider key status")
    console.print("  /tools key <provider>")
    console.print("                       Set one provider API key using a masked prompt")
    console.print("  /tools key-clear <provider>")
    console.print("                       Clear one provider API key")
    console.print("  channels guide       Run `mochi channels guide discord` for Discord setup help")
    console.print()


def _parse_tui_slash_command(text: str) -> tuple[str, list[str]]:
    """解析 slash command 與參數。"""
    raw = text.strip()
    if not raw.startswith("/"):
        return "", []
    try:
        tokens = shlex.split(raw[1:])
    except ValueError:
        body = raw[1:].strip()
        return body.lower(), []
    if not tokens:
        return "", []
    command = tokens[0].strip().lower()
    args = [token.strip() for token in tokens[1:]]
    return command, args


def _session_workflow_state(events: list[dict[str, object]]) -> dict[str, object] | None:
    for event in reversed(events):
        if event.get("type") != "session_meta":
            continue
        if event.get("event") != "workflow_state_updated":
            continue
        workflow = event.get("workflow")
        if isinstance(workflow, dict):
            return dict(workflow)
    return None


def _session_goal_state(events: list[dict[str, object]]) -> dict[str, object] | None:
    for event in reversed(events):
        if event.get("type") != "session_meta":
            continue
        if event.get("event") != "goal_state_updated":
            continue
        goal = event.get("goal")
        if isinstance(goal, dict):
            return dict(goal)
    return None


def _session_project_id(events: list[dict[str, object]]) -> str | None:
    for event in reversed(events):
        if event.get("type") != "session_meta":
            continue
        if event.get("event") != "project_assigned":
            continue
        project_id = event.get("project_id")
        if isinstance(project_id, str) and project_id.strip():
            return project_id.strip()
        return None
    return None


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return None


def _get_goal_attempt_run_id(goal: dict[str, object]) -> str | None:
    current_attempt_id = _string_or_none(goal.get("current_attempt_id"))
    attempts = goal.get("attempts")
    if not isinstance(attempts, list):
        return None
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        attempt_id = _string_or_none(attempt.get("attempt_id")) or _string_or_none(attempt.get("id"))
        if current_attempt_id is not None and attempt_id == current_attempt_id:
            return _string_or_none(attempt.get("agent_run_id"))
    for attempt in reversed(attempts):
        if isinstance(attempt, dict):
            run_id = _string_or_none(attempt.get("agent_run_id"))
            if run_id is not None:
                return run_id
    return None


def _print_goal_card(goal_card: dict[str, object]) -> None:
    from mochi.goal_proposal_copy import (
        build_goal_card_chrome_copy,
        build_goal_card_execution_mode_label,
        build_goal_card_kind_label,
        build_goal_card_status_label,
        build_goal_proposal_system_cta_copy,
    )

    label = _string_or_none(goal_card.get("label")) or "Goal"
    objective = _string_or_none(goal_card.get("objective")) or ""
    execution_mode = _string_or_none(goal_card.get("executionMode")) or "workflow"
    copy_source = (
        _string_or_none(goal_card.get("copySource"))
        or objective
        or _string_or_none(goal_card.get("roleSummary"))
        or _string_or_none(goal_card.get("runtimeMode"))
        or label
    )
    kind = _string_or_none(goal_card.get("kind")) or "proposal"
    status = _string_or_none(goal_card.get("status"))
    protocol_id = _string_or_none(goal_card.get("protocolId"))
    models = goal_card.get("models") if isinstance(goal_card.get("models"), list) else []
    role_summary = _string_or_none(goal_card.get("roleSummary"))
    runtime_mode = _string_or_none(goal_card.get("runtimeMode"))
    risk_note = _string_or_none(goal_card.get("riskNote"))
    superseded = bool(goal_card.get("superseded"))
    chrome_copy = build_goal_card_chrome_copy(user_message=copy_source)
    localized_kind_label = build_goal_card_kind_label(
        user_message=copy_source,
        kind=kind,  # type: ignore[arg-type]
    )
    localized_execution_mode = build_goal_card_execution_mode_label(
        user_message=copy_source,
        execution_mode=execution_mode,
    )
    localized_status = build_goal_card_status_label(
        user_message=copy_source,
        status=status,
    )

    console.print(f"[bold]{label}[/bold]")
    if objective:
        console.print(f"  {chrome_copy.objective_label}: {objective}", highlight=False)
    console.print(f"  {chrome_copy.execution_label}: {localized_execution_mode}", highlight=False)
    console.print(f"  {localized_kind_label}", highlight=False)
    if localized_status:
        console.print(f"  {chrome_copy.goal_status_label}: {localized_status}", highlight=False)
    if protocol_id:
        console.print(f"  {chrome_copy.protocol_label}: {protocol_id}", highlight=False)
    if models:
        console.print(f"  {chrome_copy.models_label}: {', '.join(str(item) for item in models)}", highlight=False)
    if role_summary:
        console.print(f"  {chrome_copy.role_summary_label}: {role_summary}", highlight=False)
    if runtime_mode:
        console.print(f"  {chrome_copy.runtime_label}: {runtime_mode}", highlight=False)
    if risk_note:
        console.print(f"  {chrome_copy.risk_note_label}: {risk_note}", highlight=False)
    if not superseded and kind in {"proposal", "revised_proposal"}:
        cta_copy = build_goal_proposal_system_cta_copy(
            user_message=copy_source
        )
        console.print(f"  {cta_copy.title}:", highlight=False)
        console.print(
            f"    {cta_copy.launch_label}: {cta_copy.launch_body}",
            highlight=False,
        )
        console.print(
            f"    {cta_copy.revise_label}: {cta_copy.revise_body}",
            highlight=False,
        )
        console.print(
            f"    {cta_copy.chat_label}: {cta_copy.chat_body}",
            highlight=False,
        )


async def _append_terminal_goal_conversation(
    *,
    session_store: object,
    session_id: str,
    user_content: str,
    assistant_content: str,
    goal_card: dict[str, object] | None = None,
) -> None:
    now = datetime.now(tz=UTC).isoformat()
    await session_store.save_event(
        session_id,
        {
            "type": "message",
            "role": "user",
            "content": user_content,
            "timestamp": now,
        },
    )
    assistant_event: dict[str, object] = {
        "type": "message",
        "role": "assistant",
        "content": assistant_content,
        "timestamp": now,
    }
    if goal_card is not None:
        assistant_event["goal_card"] = goal_card
    await session_store.save_event(session_id, assistant_event)
    console.print(f"[green]Mochi[/green] {assistant_content}", highlight=False)
    if goal_card is not None:
        _print_goal_card(goal_card)


async def _persist_terminal_goal_state(
    *,
    session_store: object,
    session_id: str,
    goal_state: dict[str, object],
) -> None:
    await session_store.save_event(
        session_id,
        {
            "type": "session_meta",
            "event": "goal_state_updated",
            "session_id": session_id,
            "goal": goal_state,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
    )


async def _persist_terminal_workflow_state(
    *,
    session_store: object,
    session_id: str,
    workflow_state: dict[str, object],
) -> None:
    await session_store.save_event(
        session_id,
        {
            "type": "session_meta",
            "event": "workflow_state_updated",
            "session_id": session_id,
            "workflow": workflow_state,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
    )


async def _sync_terminal_workflow_state_for_goal(
    *,
    session_store: object,
    session_id: str,
    execution_mode: str,
    interaction_mode: str | None = None,
    execution_topology: str | None = None,
    goal_status: str | None = None,
    run_id: str | None = None,
) -> None:
    from mochi.terminal_goal_helpers import is_goal_terminal_status

    events = await session_store.load_session(session_id)
    base_workflow = _session_workflow_state(events) or {}
    goal_terminal = is_goal_terminal_status(goal_status)
    next_run_id = run_id if not goal_terminal else None
    workflow_enabled = (
        bool(next_run_id)
        and not goal_terminal
        and goal_status != "paused"
        and (
            interaction_mode == "workflow"
            or execution_topology == "multi_agent"
            or execution_mode == "workflow"
        )
    )
    next_state = dict(base_workflow)
    next_state["enabled"] = workflow_enabled
    next_state["bound_run_id"] = next_run_id
    if next_run_id and next_run_id == _string_or_none(base_workflow.get("bound_run_id")):
        next_state["synced_run_event_count"] = base_workflow.get("synced_run_event_count", 0)
    else:
        next_state["synced_run_event_count"] = 0
    await _persist_terminal_workflow_state(
        session_store=session_store,
        session_id=session_id,
        workflow_state=next_state,
    )


async def _show_terminal_goal_summary(
    *,
    session_store: object,
    session_id: str,
    prefix: str | None = None,
) -> None:
    from mochi.goal_proposal_copy import build_goal_card_chrome_copy
    from mochi.terminal_goal_helpers import (
        goal_card_from_summary,
        normalize_goal_session_state,
    )

    events = await session_store.load_session(session_id)
    goal_state = normalize_goal_session_state(_session_goal_state(events))
    pending_proposal = goal_state.get("pending_proposal")
    last_goal_summary = goal_state.get("last_goal_summary")
    active_goal_id = goal_state.get("active_goal_id")
    active_goal_status = goal_state.get("active_goal_status")

    if prefix:
        console.print(prefix, highlight=False)

    if isinstance(pending_proposal, dict):
        copy = build_goal_card_chrome_copy(
            user_message=str(pending_proposal.get("objective") or ""),
        )
        console.print(f"[dim]{copy.pending_summary_intro}[/dim]")
        _print_goal_card(
            goal_card_from_summary(
                pending_proposal,
                kind="revised_proposal" if int(pending_proposal.get("revision_index", 0) or 0) > 0 else "proposal",
                copy_source=str(pending_proposal.get("objective") or ""),
            )
        )
        return

    if isinstance(last_goal_summary, dict):
        copy = build_goal_card_chrome_copy(
            user_message=str(last_goal_summary.get("objective") or ""),
        )
        console.print(
            f"[dim]{copy.active_summary_intro}[/dim]"
            if active_goal_id
            else f"[dim]{copy.recent_summary_intro}[/dim]"
        )
        _print_goal_card(
            goal_card_from_summary(
                last_goal_summary,
                kind="started",
                label=copy.active_goal_label if active_goal_id else copy.most_recent_goal_label,
                copy_source=str(last_goal_summary.get("objective") or ""),
                goal_id=last_goal_summary.get("goal_id"),
                status=active_goal_status or last_goal_summary.get("status"),
            )
        )


async def _handle_terminal_goal_input(
    *,
    text: str,
    session_id: str,
    current_model: str,
    autonomy_mode: str | None,
    session_store: object,
    ensure_runtime_service: object,
    intent_invoker: object,
) -> dict[str, object]:
    from mochi.runtime.models import AgentRunMessageRequest, GoalCreateRequest
    from mochi.goal_intent import classify_goal_proposal_follow_up_intent
    from mochi.goal_proposal_copy import (
        build_goal_card_chrome_copy,
        build_goal_command_help_message,
        build_goal_follow_up_message,
        build_goal_lifecycle_message,
        build_goal_proposal_assistant_copy_fallback,
        generate_goal_proposal_assistant_copy,
    )
    from mochi.terminal_goal_helpers import (
        build_goal_proposal_state,
        build_goal_summary_from_goal,
        goal_card_from_summary,
        is_goal_terminal_status,
        normalize_goal_session_state,
        resolve_goal_continuation_decision,
        resolve_goal_workflow_routing,
    )

    def _proposal_revision_index(proposal: dict[str, object]) -> int:
        try:
            return max(0, int(proposal.get("revision_index", 0) or 0))
        except (TypeError, ValueError):
            return 0

    async def _resolve_goal_proposal_assistant_content(
        *,
        user_message: str,
        proposal: dict[str, object],
    ) -> tuple[str, str]:
        try:
            result = await generate_goal_proposal_assistant_copy(
                intent_invoker,  # type: ignore[arg-type]
                user_message=user_message.strip()
                or str(proposal.get("objective") or ""),
                proposal_objective=str(proposal.get("objective") or ""),
                execution_mode=str(proposal.get("execution_mode") or "single_agent"),
                protocol_selection=_string_or_none(proposal.get("protocol_selection")),
                role_summary=_string_or_none(proposal.get("role_summary")),
                runtime_mode=_string_or_none(proposal.get("runtime_mode")),
                revision_index=_proposal_revision_index(proposal),
            )
            if result.explanation.strip():
                return result.explanation, result.source
        except Exception:
            pass
        return (
            build_goal_proposal_assistant_copy_fallback(
                user_message=user_message.strip()
                or str(proposal.get("objective") or ""),
                proposal_objective=str(proposal.get("objective") or ""),
                execution_mode=str(proposal.get("execution_mode") or "single_agent"),
                protocol_selection=_string_or_none(proposal.get("protocol_selection")),
                revision_index=_proposal_revision_index(proposal),
            ),
            "fallback",
        )

    def _pending_goal_proposal_assistant_content(proposal: dict[str, object]) -> str:
        explanation = _string_or_none(proposal.get("assistant_explanation"))
        if explanation is not None:
            return explanation
        return build_goal_proposal_assistant_copy_fallback(
            user_message=str(proposal.get("objective") or ""),
            proposal_objective=str(proposal.get("objective") or ""),
            execution_mode=str(proposal.get("execution_mode") or "single_agent"),
            protocol_selection=_string_or_none(proposal.get("protocol_selection")),
            revision_index=_proposal_revision_index(proposal),
        )

    events = await session_store.load_session(session_id)
    base_goal_state = normalize_goal_session_state(_session_goal_state(events))
    routing_decision = resolve_goal_workflow_routing(
        text=text,
        has_pending_proposal=base_goal_state["pending_proposal"] is not None,
        has_active_goal=(
            base_goal_state["active_goal_id"] is not None
            and not is_goal_terminal_status(base_goal_state["active_goal_status"])
        ),
    )

    mode_command = routing_decision.mode_command
    if mode_command is not None and mode_command.mode == "chat":
        if not mode_command.content:
            console.print("[yellow]Usage: /chat <request>[/yellow]")
            return {"handled": True, "chat_text": None}
        return {"handled": False, "chat_text": routing_decision.request_text}

    if mode_command is not None and mode_command.mode == "workflow" and not mode_command.content:
        console.print("[yellow]Usage: /workflow <request>[/yellow]")
        return {"handled": True, "chat_text": None}

    if not routing_decision.should_handle_goal_workflow_routing:
        return {"handled": False, "chat_text": None}

    pending_proposal = base_goal_state["pending_proposal"]
    latest_goal_summary = base_goal_state["last_goal_summary"]
    active_goal_id = base_goal_state["active_goal_id"]
    goal_command = routing_decision.goal_command
    confirmation_requested = routing_decision.confirmation_requested
    proposal_revision_requested = routing_decision.proposal_revision_requested

    if routing_decision.pending_proposal_follow_up_requested and isinstance(pending_proposal, dict):
        try:
            intent_result = await classify_goal_proposal_follow_up_intent(
                intent_invoker,  # type: ignore[arg-type]
                user_message=routing_decision.request_text,
                proposal_objective=str(pending_proposal.get("objective") or ""),
                execution_mode=str(pending_proposal.get("execution_mode") or "single_agent"),
            )
        except Exception as exc:
            intent_result = SimpleNamespace(
                intent="ambiguous",
                confidence=None,
                rationale=str(exc),
            )

        if intent_result.intent == "confirm_start":
            confirmation_requested = True
        elif intent_result.intent == "revise_proposal":
            proposal_revision_requested = True
        elif intent_result.intent == "exit_goal_lane":
            return {"handled": False, "chat_text": text}
        else:
            await _append_terminal_goal_conversation(
                session_store=session_store,
                session_id=session_id,
                user_content=text.strip(),
                assistant_content=_pending_goal_proposal_assistant_content(pending_proposal),
                goal_card=goal_card_from_summary(
                    pending_proposal,
                    kind="revised_proposal" if pending_proposal["revision_index"] > 0 else "proposal",
                ),
            )
            return {"handled": True, "chat_text": None}

    if goal_command is not None and goal_command.action == "help":
        if pending_proposal is not None:
            pending_card = goal_card_from_summary(
                pending_proposal,
                kind="revised_proposal" if pending_proposal["revision_index"] > 0 else "proposal",
                copy_source=goal_command.raw,
            )
            await _append_terminal_goal_conversation(
                session_store=session_store,
                session_id=session_id,
                user_content=goal_command.raw,
                assistant_content=_pending_goal_proposal_assistant_content(pending_proposal),
                goal_card=pending_card,
            )
            return {"handled": True, "chat_text": None}
        if latest_goal_summary is not None:
            card_copy = build_goal_card_chrome_copy(user_message=goal_command.raw)
            summary_card = goal_card_from_summary(
                latest_goal_summary,
                kind="started",
                label=card_copy.goal_summary_label if active_goal_id else card_copy.most_recent_goal_label,
                copy_source=goal_command.raw,
                goal_id=latest_goal_summary.get("goal_id"),
                status=base_goal_state["active_goal_status"] or latest_goal_summary.get("status"),
            )
            await _append_terminal_goal_conversation(
                session_store=session_store,
                session_id=session_id,
                user_content=goal_command.raw,
                assistant_content=(
                    build_goal_lifecycle_message(
                        user_message=(
                            (latest_goal_summary.get("objective") or goal_command.raw)
                            if isinstance(latest_goal_summary, dict)
                            else goal_command.raw
                        ),
                        kind="goal_manage_hint",
                    )
                    if active_goal_id
                    else build_goal_lifecycle_message(
                        user_message=goal_command.raw,
                        kind="no_active_goal",
                    )
                ),
                goal_card=summary_card,
            )
            return {"handled": True, "chat_text": None}
        await _append_terminal_goal_conversation(
            session_store=session_store,
            session_id=session_id,
            user_content=goal_command.raw,
            assistant_content=build_goal_command_help_message(
                user_message=goal_command.raw,
            ),
        )
        return {"handled": True, "chat_text": None}

    if (
        (
            goal_command is not None
            and goal_command.action == "proposal"
        )
        or routing_decision.workflow_proposal_requested
        or routing_decision.natural_language_goal_requested
    ) and active_goal_id and not is_goal_terminal_status(base_goal_state["active_goal_status"]):
        await _append_terminal_goal_conversation(
            session_store=session_store,
            session_id=session_id,
            user_content=goal_command.content if goal_command is not None and goal_command.action == "proposal" else routing_decision.request_text,
            assistant_content=build_goal_follow_up_message(
                user_message=(
                    goal_command.content
                    if goal_command is not None and goal_command.action == "proposal"
                    else routing_decision.request_text
                ),
                kind="active_goal_exists",
            ),
        )
        return {"handled": True, "chat_text": None}

    if (
        (goal_command is not None and goal_command.action == "proposal")
        or routing_decision.workflow_proposal_requested
        or routing_decision.natural_language_goal_requested
        or proposal_revision_requested
    ):
        revision_source_text = (
            routing_decision.request_text
            if proposal_revision_requested
            else goal_command.content
            if goal_command is not None and goal_command.action == "proposal"
            else routing_decision.request_text
        )
        explicit_execution_mode = (
            "workflow"
            if routing_decision.workflow_proposal_requested
            else "single_agent"
            if goal_command is not None and goal_command.action == "proposal"
            else base_goal_state["execution_mode"] or "single_agent"
        )
        proposal_objective = (
            pending_proposal.get("objective")
            if proposal_revision_requested and isinstance(pending_proposal, dict)
            else goal_command.content
            if goal_command is not None and goal_command.action == "proposal"
            else routing_decision.request_text
        )
        next_proposal = build_goal_proposal_state(
            proposal_objective,
            explicit_execution_mode,
            current_model=current_model,
            autonomy_mode=autonomy_mode,
            previous=pending_proposal,
            revision_text=revision_source_text if pending_proposal is not None else None,
        )
        proposal_user_content = revision_source_text
        proposal_explanation_source = revision_source_text or proposal_objective
        (
            proposal_assistant_content,
            proposal_assistant_content_source,
        ) = await _resolve_goal_proposal_assistant_content(
            user_message=proposal_explanation_source,
            proposal=next_proposal,
        )
        next_proposal["assistant_explanation"] = proposal_assistant_content
        next_proposal["assistant_explanation_source"] = proposal_assistant_content_source
        next_goal_state = {
            "active_goal_id": None,
            "active_goal_status": None,
            "execution_mode": next_proposal["execution_mode"],
            "interaction_mode": next_proposal["interaction_mode"],
            "execution_topology": next_proposal["execution_topology"],
            "bound_run_id": next_proposal["bound_run_id"],
            "protocol_selection": next_proposal["protocol_selection"],
            "selection_rationale": next_proposal["selection_rationale"],
            "default_route": "workflow" if next_proposal["execution_mode"] == "workflow" else "goal",
            "last_goal_summary": latest_goal_summary,
            "pending_proposal": next_proposal,
        }
        await _persist_terminal_goal_state(
            session_store=session_store,
            session_id=session_id,
            goal_state=next_goal_state,
        )
        await _sync_terminal_workflow_state_for_goal(
            session_store=session_store,
            session_id=session_id,
            execution_mode=next_proposal["execution_mode"],
            interaction_mode=next_proposal.get("interaction_mode"),
            execution_topology=next_proposal.get("execution_topology"),
        )
        await _append_terminal_goal_conversation(
            session_store=session_store,
            session_id=session_id,
            user_content=proposal_user_content,
            assistant_content=proposal_assistant_content,
            goal_card=goal_card_from_summary(
                next_proposal,
                kind="revised_proposal" if pending_proposal is not None or proposal_revision_requested else "proposal",
                copy_source=proposal_user_content,
            ),
        )
        return {"handled": True, "chat_text": None}

    if confirmation_requested and pending_proposal is not None:
        runtime_service = await ensure_runtime_service()
        created_goal = await runtime_service.create_goal(
            GoalCreateRequest(
                objective=pending_proposal["objective"],
                execution_mode=pending_proposal["execution_mode"],
                interaction_mode=pending_proposal.get("interaction_mode"),
                execution_topology=pending_proposal.get("execution_topology"),
                protocol_id=pending_proposal.get("protocol_id"),
                bound_run_id=pending_proposal.get("bound_run_id"),
                protocol_selection=pending_proposal.get("protocol_selection"),
                selection_rationale=pending_proposal.get("selection_rationale"),
                topic=pending_proposal["objective"],
                project_id=_session_project_id(events),
                summary={
                    "operator_message": pending_proposal["objective"],
                    "source_session_id": session_id,
                },
                metadata={
                    "channel": "terminal_goal",
                    "source_session_id": session_id,
                    "pending_proposal_id": pending_proposal["proposal_id"],
                },
            )
        )
        started_goal = await runtime_service.start_goal(created_goal["goal_id"])
        started_summary = build_goal_summary_from_goal(started_goal, pending_proposal)
        started_run_id = _get_goal_attempt_run_id(started_goal)
        await _sync_terminal_workflow_state_for_goal(
            session_store=session_store,
            session_id=session_id,
            execution_mode=started_goal["execution_mode"],
            interaction_mode=_string_or_none(started_goal.get("interaction_mode")),
            execution_topology=_string_or_none(started_goal.get("execution_topology")),
            goal_status=started_goal.get("status"),
            run_id=started_run_id,
        )
        await _persist_terminal_goal_state(
            session_store=session_store,
            session_id=session_id,
            goal_state={
                "active_goal_id": started_goal["goal_id"],
                "active_goal_status": started_goal.get("status"),
                "execution_mode": started_goal["execution_mode"],
                "interaction_mode": started_summary.get("interaction_mode"),
                "execution_topology": started_summary.get("execution_topology"),
                "bound_run_id": started_summary.get("bound_run_id"),
                "protocol_selection": started_summary.get("protocol_selection"),
                "selection_rationale": started_summary.get("selection_rationale"),
                "default_route": "workflow" if started_goal["execution_mode"] == "workflow" else "goal",
                "last_goal_summary": started_summary,
                "pending_proposal": None,
            },
        )
        await _append_terminal_goal_conversation(
            session_store=session_store,
            session_id=session_id,
            user_content=text.strip(),
            assistant_content=build_goal_lifecycle_message(
                user_message=str(pending_proposal.get("objective") or "") or text.strip(),
                kind="goal_started",
            ),
            goal_card=goal_card_from_summary(
                started_summary,
                kind="started",
                copy_source=text.strip(),
                goal_id=started_goal["goal_id"],
                status=started_goal.get("status"),
            ),
        )
        return {"handled": True, "chat_text": None}

    if active_goal_id is None:
        if goal_command is not None and goal_command.action == "status" and pending_proposal is not None:
            await _append_terminal_goal_conversation(
                session_store=session_store,
                session_id=session_id,
                user_content=goal_command.raw,
                assistant_content=_pending_goal_proposal_assistant_content(pending_proposal),
                goal_card=goal_card_from_summary(
                    pending_proposal,
                    kind="revised_proposal" if pending_proposal["revision_index"] > 0 else "proposal",
                    copy_source=goal_command.raw,
                ),
            )
            return {"handled": True, "chat_text": None}
        if goal_command is not None and goal_command.action == "stop" and pending_proposal is not None:
            await _persist_terminal_goal_state(
                session_store=session_store,
                session_id=session_id,
                goal_state={
                    "active_goal_id": None,
                    "active_goal_status": None,
                    "execution_mode": latest_goal_summary.get("execution_mode") if isinstance(latest_goal_summary, dict) else None,
                    "interaction_mode": latest_goal_summary.get("interaction_mode") if isinstance(latest_goal_summary, dict) else None,
                    "execution_topology": latest_goal_summary.get("execution_topology") if isinstance(latest_goal_summary, dict) else None,
                    "bound_run_id": latest_goal_summary.get("bound_run_id") if isinstance(latest_goal_summary, dict) else None,
                    "protocol_selection": latest_goal_summary.get("protocol_selection") if isinstance(latest_goal_summary, dict) else None,
                    "selection_rationale": latest_goal_summary.get("selection_rationale") if isinstance(latest_goal_summary, dict) else None,
                    "default_route": "chat",
                    "last_goal_summary": latest_goal_summary,
                    "pending_proposal": None,
                },
            )
            await _append_terminal_goal_conversation(
                session_store=session_store,
                session_id=session_id,
                user_content=goal_command.raw,
                assistant_content=build_goal_lifecycle_message(
                    user_message=str(pending_proposal.get("objective") or "") or goal_command.raw,
                    kind="pending_cleared",
                ),
            )
            return {"handled": True, "chat_text": None}
        await _append_terminal_goal_conversation(
            session_store=session_store,
            session_id=session_id,
            user_content=(goal_command.raw if goal_command is not None else text.strip()),
            assistant_content=build_goal_lifecycle_message(
                user_message=(
                    str(latest_goal_summary.get("objective") or "")
                    if isinstance(latest_goal_summary, dict)
                    else ""
                )
                or (
                    goal_command.raw
                    if goal_command is not None
                    else text.strip()
                ),
                kind="no_active_goal",
            ),
        )
        return {"handled": True, "chat_text": None}

    runtime_service = await ensure_runtime_service()

    if routing_decision.active_goal_follow_up_requested:
        health = await runtime_service.get_goal_health(active_goal_id)
        continuation = resolve_goal_continuation_decision(health or {})

        if continuation.action in {"manual_resolution_required", "blocked"}:
            active_goal = await runtime_service.get_goal(active_goal_id)
            active_goal_summary = build_goal_summary_from_goal(active_goal or {}, latest_goal_summary)
            active_goal_status = _string_or_none((active_goal or {}).get("status"))
            card_copy = build_goal_card_chrome_copy(
                user_message=routing_decision.request_text,
            )
            await _persist_terminal_goal_state(
                session_store=session_store,
                session_id=session_id,
                goal_state={
                    "active_goal_id": None if is_goal_terminal_status(active_goal_status) else (active_goal or {}).get("goal_id"),
                    "active_goal_status": active_goal_status,
                    "execution_mode": (active_goal or {}).get("execution_mode"),
                    "interaction_mode": active_goal_summary.get("interaction_mode"),
                    "execution_topology": active_goal_summary.get("execution_topology"),
                    "bound_run_id": active_goal_summary.get("bound_run_id"),
                    "protocol_selection": active_goal_summary.get("protocol_selection"),
                    "selection_rationale": active_goal_summary.get("selection_rationale"),
                    "default_route": (
                        "chat"
                        if is_goal_terminal_status(active_goal_status)
                        else "workflow"
                        if (active_goal or {}).get("execution_mode") == "workflow"
                        else "goal"
                    ),
                    "last_goal_summary": active_goal_summary,
                    "pending_proposal": None,
                },
            )
            approval_state = health.get("approval_state") if isinstance(health, dict) else {}
            approval_tool_names = (
                [
                    str(item).strip()
                    for item in (approval_state.get("tool_names") or [])
                    if str(item).strip()
                ]
                if isinstance(approval_state, dict)
                else []
            )
            assistant_content = build_goal_follow_up_message(
                user_message=routing_decision.request_text,
                kind=(
                    "manual_resolution_required"
                    if continuation.action == "manual_resolution_required"
                    else "blocked"
                ),
                summary=continuation.summary,
                approval_count=len(continuation.approval_ids),
                tool_names=approval_tool_names,
            )
            await _append_terminal_goal_conversation(
                session_store=session_store,
                session_id=session_id,
                user_content=routing_decision.request_text,
                assistant_content=assistant_content,
                goal_card=goal_card_from_summary(
                    active_goal_summary,
                    kind="started",
                    label=card_copy.goal_blocked_label,
                    copy_source=routing_decision.request_text,
                    goal_id=(active_goal or {}).get("goal_id"),
                    status=active_goal_status,
                ),
            )
            return {"handled": True, "chat_text": None}

        active_goal = (
            await runtime_service.refresh_goal(active_goal_id)
            if continuation.action == "refresh_then_forward"
            else await runtime_service.resume_goal(active_goal_id)
            if continuation.action == "resume_then_forward"
            else await runtime_service.get_goal(active_goal_id)
        )
        active_goal_summary = build_goal_summary_from_goal(active_goal or {}, latest_goal_summary)
        active_goal_status = _string_or_none((active_goal or {}).get("status"))
        active_run_id = _get_goal_attempt_run_id(active_goal or {})

        if active_run_id is None:
            await _persist_terminal_goal_state(
                session_store=session_store,
                session_id=session_id,
                goal_state={
                    "active_goal_id": None if is_goal_terminal_status(active_goal_status) else (active_goal or {}).get("goal_id"),
                    "active_goal_status": active_goal_status,
                    "execution_mode": (active_goal or {}).get("execution_mode"),
                    "interaction_mode": active_goal_summary.get("interaction_mode"),
                    "execution_topology": active_goal_summary.get("execution_topology"),
                    "bound_run_id": active_goal_summary.get("bound_run_id"),
                    "protocol_selection": active_goal_summary.get("protocol_selection"),
                    "selection_rationale": active_goal_summary.get("selection_rationale"),
                    "default_route": (
                        "chat"
                        if is_goal_terminal_status(active_goal_status)
                        else "workflow"
                        if (active_goal or {}).get("execution_mode") == "workflow"
                        else "goal"
                    ),
                    "last_goal_summary": active_goal_summary,
                    "pending_proposal": None,
                },
            )
            await _append_terminal_goal_conversation(
                session_store=session_store,
                session_id=session_id,
                user_content=routing_decision.request_text,
                assistant_content=build_goal_follow_up_message(
                    user_message=routing_decision.request_text,
                    kind="no_live_attempt",
                ),
                goal_card=goal_card_from_summary(
                    active_goal_summary,
                    kind="started",
                    label=build_goal_card_chrome_copy(
                        user_message=routing_decision.request_text,
                    ).goal_status_label,
                    copy_source=routing_decision.request_text,
                    goal_id=(active_goal or {}).get("goal_id"),
                    status=active_goal_status,
                ),
            )
            return {"handled": True, "chat_text": None}

        await runtime_service.append_agent_run_message(
            active_run_id,
            AgentRunMessageRequest(
                role="operator",
                content=routing_decision.request_text,
                project_id=(active_goal or {}).get("project_id"),
                workspace_dir=(active_goal or {}).get("workspace_dir"),
                metadata={
                    "channel": "terminal_goal",
                    "goal_id": (active_goal or {}).get("goal_id"),
                    "source_session_id": session_id,
                },
            ),
        )
        await _sync_terminal_workflow_state_for_goal(
            session_store=session_store,
            session_id=session_id,
            execution_mode=(active_goal or {}).get("execution_mode") or "workflow",
            interaction_mode=active_goal_summary.get("interaction_mode"),
            execution_topology=active_goal_summary.get("execution_topology"),
            goal_status=active_goal_status,
            run_id=active_run_id,
        )
        await _persist_terminal_goal_state(
            session_store=session_store,
            session_id=session_id,
            goal_state={
                "active_goal_id": None if is_goal_terminal_status(active_goal_status) else (active_goal or {}).get("goal_id"),
                "active_goal_status": active_goal_status,
                "execution_mode": (active_goal or {}).get("execution_mode"),
                "interaction_mode": active_goal_summary.get("interaction_mode"),
                "execution_topology": active_goal_summary.get("execution_topology"),
                "bound_run_id": active_goal_summary.get("bound_run_id"),
                "protocol_selection": active_goal_summary.get("protocol_selection"),
                "selection_rationale": active_goal_summary.get("selection_rationale"),
                "default_route": (
                    "chat"
                    if is_goal_terminal_status(active_goal_status)
                    else "workflow"
                    if (active_goal or {}).get("execution_mode") == "workflow"
                    else "goal"
                ),
                "last_goal_summary": active_goal_summary,
                "pending_proposal": None,
            },
        )
        await _append_terminal_goal_conversation(
            session_store=session_store,
            session_id=session_id,
            user_content=routing_decision.request_text,
            assistant_content=build_goal_follow_up_message(
                user_message=routing_decision.request_text,
                kind=(
                    "refreshed_forwarded"
                    if continuation.action == "refresh_then_forward"
                    else "resumed_forwarded"
                    if continuation.action == "resume_then_forward"
                    else "forwarded"
                ),
            ),
            goal_card=goal_card_from_summary(
                active_goal_summary,
                kind="started",
                label=build_goal_card_chrome_copy(
                    user_message=routing_decision.request_text,
                ).goal_updated_label,
                copy_source=routing_decision.request_text,
                goal_id=(active_goal or {}).get("goal_id"),
                status=active_goal_status,
            ),
        )
        return {"handled": True, "chat_text": None}

    next_goal = (
        await runtime_service.get_goal(active_goal_id)
        if goal_command is not None and goal_command.action == "status"
        else await runtime_service.pause_goal(active_goal_id)
        if goal_command is not None and goal_command.action == "pause"
        else await runtime_service.resume_goal(active_goal_id)
        if goal_command is not None and goal_command.action == "resume"
        else await runtime_service.cancel_goal(active_goal_id)
    )
    next_goal_summary = build_goal_summary_from_goal(next_goal or {}, latest_goal_summary)
    next_run_id = _get_goal_attempt_run_id(next_goal or {})
    next_goal_status = _string_or_none((next_goal or {}).get("status"))
    next_goal_terminal = is_goal_terminal_status(next_goal_status)
    await _sync_terminal_workflow_state_for_goal(
        session_store=session_store,
        session_id=session_id,
        execution_mode=(next_goal or {}).get("execution_mode") or "workflow",
        interaction_mode=next_goal_summary.get("interaction_mode"),
        execution_topology=next_goal_summary.get("execution_topology"),
        goal_status=next_goal_status,
        run_id=next_run_id,
    )
    await _persist_terminal_goal_state(
        session_store=session_store,
        session_id=session_id,
        goal_state={
            "active_goal_id": None if next_goal_terminal else (next_goal or {}).get("goal_id"),
            "active_goal_status": next_goal_status,
            "execution_mode": (next_goal or {}).get("execution_mode"),
            "interaction_mode": next_goal_summary.get("interaction_mode"),
            "execution_topology": next_goal_summary.get("execution_topology"),
            "bound_run_id": next_goal_summary.get("bound_run_id"),
            "protocol_selection": next_goal_summary.get("protocol_selection"),
            "selection_rationale": next_goal_summary.get("selection_rationale"),
            "default_route": (
                "chat"
                if next_goal_terminal
                else "workflow"
                if (next_goal or {}).get("execution_mode") == "workflow"
                else "goal"
            ),
            "last_goal_summary": next_goal_summary,
            "pending_proposal": None,
        },
    )
    lifecycle_copy = build_goal_card_chrome_copy(
        user_message=goal_command.raw if goal_command is not None else text.strip(),
    )
    lifecycle_label = (
        lifecycle_copy.goal_status_label
        if goal_command is not None and goal_command.action == "status"
        else lifecycle_copy.goal_paused_label
        if goal_command is not None and goal_command.action == "pause"
        else lifecycle_copy.goal_resumed_label
        if goal_command is not None and goal_command.action == "resume"
        else lifecycle_copy.goal_stopped_label
    )
    lifecycle_content = _string_or_none((next_goal or {}).get("latest_error"))
    if lifecycle_content is None:
        if goal_command is not None and goal_command.action == "status":
            lifecycle_content = build_goal_lifecycle_message(
                user_message=str(next_goal_summary.get("objective") or "") or goal_command.raw,
                kind="status_fetched",
            )
        elif goal_command is not None and goal_command.action == "pause":
            lifecycle_content = build_goal_lifecycle_message(
                user_message=str(next_goal_summary.get("objective") or "") or goal_command.raw,
                kind="goal_paused",
            )
        elif goal_command is not None and goal_command.action == "resume":
            lifecycle_content = build_goal_lifecycle_message(
                user_message=str(next_goal_summary.get("objective") or "") or goal_command.raw,
                kind="goal_resumed",
            )
        else:
            lifecycle_content = build_goal_lifecycle_message(
                user_message=str(next_goal_summary.get("objective") or "")
                or (goal_command.raw if goal_command is not None else text.strip()),
                kind="goal_stopped",
            )
    await _append_terminal_goal_conversation(
        session_store=session_store,
        session_id=session_id,
        user_content=goal_command.raw if goal_command is not None else text.strip(),
        assistant_content=lifecycle_content,
        goal_card=goal_card_from_summary(
            next_goal_summary,
            kind="started",
            label=lifecycle_label,
            copy_source=goal_command.raw if goal_command is not None else text.strip(),
            goal_id=(next_goal or {}).get("goal_id"),
            status=next_goal_status,
        ),
    )
    return {"handled": True, "chat_text": None}


async def _create_tui_runtime_service(
    *,
    engine: object,
    config: object,
    config_path: str | None,
) -> object:
    from mochi.config import defaults
    from mochi.config.schema import SecurityConfig
    from mochi.runtime.service import RuntimeService
    from mochi.runtime.store import RuntimeStore

    sessions_dir = str(getattr(config, "sessions_dir", defaults.default_sessions_dir()))
    store = RuntimeStore(Path(sessions_dir) / "runtime.db")
    await store.initialize()
    service = RuntimeService(engine=engine, store=store)
    security = getattr(config, "security", None)
    if not isinstance(security, SecurityConfig):
        payload = dict(security.__dict__) if security is not None and hasattr(security, "__dict__") else {}
        security = SecurityConfig.model_validate(payload)
        setattr(config, "security", security)
    service.update_security_config(security)
    service.bind_app_config(config=config, config_path=config_path)
    service.set_runtime_tasks_root(Path(sessions_dir) / "runtime-tasks")
    await service.start()
    return service


async def _chat_tui_async(
    *,
    model: str | None,
    config_path: str | None,
    session_id: str,
    max_turns: int,
) -> None:
    """啟動有界互動式文字聊天。"""
    from mochi.agents.engine import AgentEngine
    from mochi.agents.events import (
        ErrorEvent,
        FinalAnswerEvent,
        TextChunkEvent,
        ToolCallResultEvent,
    )
    from mochi.config.manager import load_config, save_config
    from mochi.config.schema import SecurityConfig
    from mochi.security.policy import autonomy_mode_defaults
    from mochi.sessions.store import SessionStore
    from mochi.tools.web_search_providers import (
        iter_web_search_provider_specs,
        normalize_web_search_provider,
        provider_key_config_field,
        supported_web_search_provider_names,
    )

    if max_turns <= 0:
        console.print("[red]max_turns must be greater than 0.[/red]")
        sys.exit(1)

    cfg = load_config(config_path)
    if model:
        cfg.model = model

    engine = AgentEngine(cfg)
    runtime_service: object | None = None
    current_session = session_id.strip() or DEFAULT_TUI_SESSION_ID
    from mochi.config import defaults

    def _sessions_dir() -> str:
        return str(getattr(cfg, "sessions_dir", defaults.default_sessions_dir()))

    def _ensure_security_config() -> SecurityConfig:
        current = getattr(cfg, "security", None)
        if isinstance(current, SecurityConfig):
            return current
        payload = {}
        if current is not None and hasattr(current, "__dict__"):
            payload = dict(current.__dict__)
        normalized = SecurityConfig.model_validate(payload)
        setattr(cfg, "security", normalized)
        return normalized

    session_store = SessionStore(
        sessions_dir=_sessions_dir()
    )
    supported_search_engines = set(supported_web_search_provider_names(include_aliases=True))
    supported_fetch_extractors = {"trafilatura", "jina_reader", "htmlparser"}
    provider_specs = [
        spec
        for spec in iter_web_search_provider_specs()
        if spec.key_config_field is not None
    ]

    async def _build_runtime_service() -> object:
        _ensure_security_config()
        return await _create_tui_runtime_service(
            engine=engine,
            config=cfg,
            config_path=config_path,
        )

    async def _ensure_runtime_service() -> object:
        nonlocal runtime_service
        if runtime_service is None:
            runtime_service = await _build_runtime_service()
        return runtime_service

    async def _reset_engine() -> None:
        nonlocal engine, runtime_service
        if runtime_service is not None:
            await runtime_service.close()
        await engine.close()
        engine = AgentEngine(cfg)
        await engine.initialize()
        if runtime_service is not None:
            runtime_service = await _build_runtime_service()

    def _show_tool_settings() -> None:
        fallback = ", ".join(cfg.tools.web_search_fallback_engines) or "(none)"
        console.print(f"[dim]Web search engine: {cfg.tools.web_search_engine}[/dim]")
        console.print(f"[dim]Web search fallback: {fallback}[/dim]")
        console.print(f"[dim]Web fetch extractor: {cfg.tools.web_fetch_extractor}[/dim]")
        _show_tool_key_status()

    def _show_safety_settings() -> None:
        security = _ensure_security_config()
        console.print(f"[dim]Safety mode: {security.autonomy_mode}[/dim]")
        console.print(
            f"[dim]Exec approval: {'on' if security.require_approval_for_exec else 'off'}[/dim]"
        )
        console.print(
            "[dim]File write approval: "
            f"{'on' if security.require_approval_for_file_write else 'off'}[/dim]"
        )
        console.print(f"[dim]File scope: {security.file_ops_scope}[/dim]")

    def _print_approval_summary(item: dict[str, object]) -> None:
        approval_id = str(item.get("approval_id") or "?")
        status = str(item.get("status") or "unknown")
        kind = str(item.get("approval_kind") or "other")
        scope = str(item.get("approval_scope") or "workspace")
        command = item.get("command")
        tool_name = str(item.get("tool_name") or "")
        summary = f"{approval_id} [{status}] {kind}/{scope}"
        if isinstance(command, str) and command.strip():
            summary += f" cmd={command.strip()}"
        elif tool_name:
            summary += f" tool={tool_name}"
        console.print(summary, highlight=False, markup=False)
        reason = item.get("policy_reason") or item.get("reason")
        if isinstance(reason, str) and reason.strip():
            console.print(f"  {reason.strip()}", highlight=False)
        exec_session_id = item.get("exec_session_id")
        exec_status = item.get("exec_status")
        if isinstance(exec_session_id, str) and exec_session_id.strip():
            console.print(
                f"  exec={exec_session_id.strip()} status={exec_status or 'unknown'}",
                highlight=False,
            )
        file_changes = item.get("file_changes")
        if isinstance(file_changes, list) and file_changes:
            console.print(f"  file changes={len(file_changes)}", highlight=False)
            for change in file_changes[:5]:
                if not isinstance(change, dict):
                    continue
                label = change.get("relative_path") or change.get("path") or "unknown"
                added = int(change.get("added_lines") or 0)
                deleted = int(change.get("deleted_lines") or 0)
                console.print(
                    f"    {label} (+{added}/-{deleted})",
                    highlight=False,
                    markup=False,
                )

    def _print_exec_session_payload(payload: dict[str, object]) -> None:
        session = payload.get("session")
        if not isinstance(session, dict):
            console.print("[yellow]No exec session payload available.[/yellow]")
            return
        console.print(
            "[dim]Exec session: "
            f"{session.get('session_id')} status={session.get('status')} "
            f"shell={session.get('shell')}[/dim]"
        )
        exit_code = session.get("exit_code")
        if exit_code is not None:
            console.print(f"[dim]Exit code: {exit_code}[/dim]")
        stdout = session.get("stdout")
        stderr = session.get("stderr")
        if isinstance(stdout, str) and stdout.strip():
            console.print(stdout.rstrip(), highlight=False)
        if isinstance(stderr, str) and stderr.strip():
            console.print(f"[yellow]{stderr.rstrip()}[/yellow]", highlight=False)

    def _print_tool_approval_hint(event: ToolCallResultEvent) -> None:
        metadata = event.metadata
        if not isinstance(metadata, dict):
            return
        approval_id = metadata.get("approval_id")
        requires_approval = metadata.get("requires_approval") is True
        if not requires_approval and not isinstance(approval_id, str):
            return
        normalized_id = str(approval_id or "").strip()
        summary: list[str] = []
        if normalized_id:
            summary.append(f"id={normalized_id}")
        approval_kind = metadata.get("approval_kind")
        if isinstance(approval_kind, str) and approval_kind.strip():
            summary.append(f"kind={approval_kind.strip()}")
        approval_scope = metadata.get("approval_scope")
        if isinstance(approval_scope, str) and approval_scope.strip():
            summary.append(f"scope={approval_scope.strip()}")
        if summary:
            console.print(
                f"[yellow]Approval pending:[/yellow] {' '.join(summary)}",
                highlight=False,
            )
        if normalized_id:
            console.print(f"[dim]/approve {normalized_id}[/dim]")
            allowed_decisions = metadata.get("allowed_decisions")
            approval_kind_value = str(metadata.get("approval_kind") or "").strip().lower()
            if (
                (
                    not isinstance(allowed_decisions, list)
                    and approval_kind_value not in {"file_write", "file_edit", "apply_patch"}
                )
                or (isinstance(allowed_decisions, list) and "approve_and_save_rule" in allowed_decisions)
            ):
                console.print(f"[dim]/approve-save {normalized_id}[/dim]")
            console.print(f"[dim]/reject {normalized_id}[/dim]")

    def _show_tool_key_status() -> None:
        for spec in provider_specs:
            configured = getattr(cfg.tools, spec.key_config_field, None) is not None
            status = "configured" if configured else "not configured"
            console.print(f"[dim]{spec.canonical_name}: {status}[/dim]")

    async def _persist_tool_settings() -> None:
        path = save_config(cfg, config_path)
        await _reset_engine()
        console.print(f"[dim]Saved config: {path}[/dim]")

    try:
        await engine.initialize()
    except Exception as exc:
        console.print(f"[red]Initialization failed: {exc}[/red]")
        sys.exit(1)

    console.print(f"[bold cyan]Mochi TUI[/bold cyan] model=[cyan]{cfg.model}[/cyan]")
    console.print(
        f"[dim]session={current_session} | max_turns={max_turns} | type /help for commands[/dim]"
    )
    await _show_terminal_goal_summary(
        session_store=session_store,
        session_id=current_session,
    )
    console.print()

    async def _run_chat_turn(user_text: str) -> bool:
        console.print("[green]Mochi[/green] ", end="")
        full_reply = ""
        saw_error = False
        async for event in engine.chat(user_text, session_id=current_session):
            if isinstance(event, TextChunkEvent) or (
                isinstance(event, FinalAnswerEvent) and not full_reply
            ):
                console.print(event.content, end="", highlight=False)
                full_reply += event.content
            elif isinstance(event, ToolCallResultEvent) and event.error:
                console.print(f"\n[yellow]Tool {event.tool_name} failed: {event.error}[/yellow]")
                _print_tool_approval_hint(event)
            elif isinstance(event, ErrorEvent):
                saw_error = True
                console.print(f"\n[red]Error: {event.message}[/red]")
                break
        console.print()
        return not saw_error

    turns = 0
    try:
        while turns < max_turns:
            try:
                raw = console.input("[bold cyan]You[/bold cyan] > ")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Exiting TUI.[/dim]")
                break

            text = raw.strip()
            if not text:
                continue

            if text.startswith("/"):
                command, args = _parse_tui_slash_command(text)
                if command in {"exit", "quit"}:
                    console.print("[dim]Exiting TUI.[/dim]")
                    break
                if command == "clear":
                    await session_store.delete_session(current_session)
                    await _reset_engine()
                    console.print(f"[green]Cleared session:[/green] {current_session}")
                    continue
                if command == "help":
                    _print_tui_help()
                    continue
                if command == "approvals":
                    service = await _ensure_runtime_service()
                    approvals = await service.list_approvals()
                    if not approvals:
                        console.print("[dim]No approvals found.[/dim]")
                        continue
                    for item in approvals:
                        _print_approval_summary(item)
                    continue
                if command in {"approve", "approve-save", "reject"}:
                    if len(args) != 1:
                        usage = {
                            "approve": "/approve <approval_id>",
                            "approve-save": "/approve-save <approval_id>",
                            "reject": "/reject <approval_id>",
                        }[command]
                        console.print(f"[red]Usage: {usage}[/red]")
                        continue
                    decision = {
                        "approve": "approve_once",
                        "approve-save": "approve_and_save_rule",
                        "reject": "reject",
                    }[command]
                    service = await _ensure_runtime_service()
                    resolved = await service.resolve_approval(args[0], decision=decision)
                    if resolved is None:
                        console.print(f"[red]Approval not found: {args[0]}[/red]")
                        continue
                    console.print(
                        f"[green]Approval updated:[/green] "
                        f"{resolved.get('approval_id')} -> {resolved.get('status')}"
                    )
                    execution_result = resolved.get("execution_result")
                    if isinstance(execution_result, dict) and execution_result.get("session_id"):
                        console.print(
                            f"[dim]Exec session: {execution_result.get('session_id')}[/dim]"
                        )
                    continue
                if command == "exec-read":
                    if len(args) != 1:
                        console.print("[red]Usage: /exec-read <approval_id>[/red]")
                        continue
                    service = await _ensure_runtime_service()
                    payload = await service.get_approval_exec_session(args[0])
                    if isinstance(payload, tuple):
                        console.print(f"[yellow]{payload[0].replace('_', ' ')}[/yellow]")
                        continue
                    _print_exec_session_payload(payload)
                    continue
                if command == "exec-stop":
                    if len(args) != 1:
                        console.print("[red]Usage: /exec-stop <approval_id>[/red]")
                        continue
                    service = await _ensure_runtime_service()
                    payload = await service.stop_approval_exec_session(args[0])
                    if isinstance(payload, tuple):
                        console.print(f"[yellow]{payload[0].replace('_', ' ')}[/yellow]")
                        continue
                    console.print(
                        f"[green]Exec session stop requested:[/green] "
                        f"{payload.get('stop_status') or 'unknown'}"
                    )
                    _print_exec_session_payload(payload)
                    continue
                if command == "safety":
                    if not args:
                        _show_safety_settings()
                        continue
                    if len(args) != 1:
                        console.print("[red]Usage: /safety <mode>[/red]")
                        continue
                    next_mode = args[0].strip().lower()
                    supported_modes = {
                        "strict",
                        "trusted_workspace",
                        "auto_review",
                        "high_autonomy",
                    }
                    if next_mode not in supported_modes:
                        console.print(
                            "[red]Unsupported safety mode.[/red] "
                            f"Choose from: {', '.join(sorted(supported_modes))}"
                        )
                        continue
                    cfg.security = _ensure_security_config().model_copy(
                        update=autonomy_mode_defaults(next_mode)
                    )
                    await _persist_tool_settings()
                    console.print(f"[green]Safety mode updated:[/green] {cfg.security.autonomy_mode}")
                    continue
                if command == "model":
                    if not args:
                        console.print(f"[dim]Current model: {cfg.model}[/dim]")
                        continue
                    next_model = args[0]
                    try:
                        info = await engine.switch_model(next_model)
                    except Exception as exc:
                        console.print(f"[red]Model switch failed: {exc}[/red]")
                    else:
                        cfg.model = next_model
                        console.print(
                            f"[green]Model switched:[/green] {info.name} ({info.backend_type})"
                        )
                    continue
                if command == "session":
                    if not args:
                        console.print(f"[dim]Current session: {current_session}[/dim]")
                        await _show_terminal_goal_summary(
                            session_store=session_store,
                            session_id=current_session,
                        )
                        continue
                    next_session = args[0].strip()
                    if not next_session:
                        console.print("[red]Session id must not be empty.[/red]")
                        continue
                    current_session = next_session
                    console.print(f"[green]Session switched:[/green] {current_session}")
                    await _show_terminal_goal_summary(
                        session_store=session_store,
                        session_id=current_session,
                    )
                    continue
                if command == "tools":
                    if not args:
                        _show_tool_settings()
                        continue
                    subcommand = args[0].strip().lower()
                    subargs = [arg.strip().lower() for arg in args[1:] if arg.strip()]
                    if subcommand == "search-engine":
                        if len(subargs) != 1:
                            console.print("[red]Usage: /tools search-engine <engine>[/red]")
                            continue
                        next_engine = normalize_web_search_provider(subargs[0])
                        if next_engine not in supported_search_engines:
                            console.print(
                                "[red]Unsupported search engine.[/red] "
                                f"Choose from: {', '.join(sorted(supported_search_engines))}"
                            )
                            continue
                        cfg.tools.web_search_engine = next_engine
                        await _persist_tool_settings()
                        console.print(
                            f"[green]Web search engine updated:[/green] {cfg.tools.web_search_engine}"
                        )
                        continue
                    if subcommand == "fallback":
                        if not subargs:
                            console.print("[red]Usage: /tools fallback <engine...>[/red]")
                            continue
                        normalized_fallback = [normalize_web_search_provider(engine_name) for engine_name in subargs]
                        invalid = [engine_name for engine_name in normalized_fallback if engine_name not in supported_search_engines]
                        if invalid:
                            console.print(
                                "[red]Unsupported fallback engine(s):[/red] "
                                + ", ".join(invalid)
                            )
                            continue
                        deduped_fallback = list(dict.fromkeys(normalized_fallback))
                        cfg.tools.web_search_fallback_engines = deduped_fallback
                        await _persist_tool_settings()
                        console.print(
                            "[green]Web search fallback updated:[/green] "
                            + ", ".join(cfg.tools.web_search_fallback_engines)
                        )
                        continue
                    if subcommand == "fetch-extractor":
                        if len(subargs) != 1:
                            console.print("[red]Usage: /tools fetch-extractor <extractor>[/red]")
                            continue
                        next_extractor = subargs[0]
                        if next_extractor not in supported_fetch_extractors:
                            console.print(
                                "[red]Unsupported fetch extractor.[/red] "
                                f"Choose from: {', '.join(sorted(supported_fetch_extractors))}"
                            )
                            continue
                        cfg.tools.web_fetch_extractor = next_extractor
                        await _persist_tool_settings()
                        console.print(
                            f"[green]Web fetch extractor updated:[/green] {cfg.tools.web_fetch_extractor}"
                        )
                        continue
                    if subcommand == "key-status":
                        _show_tool_key_status()
                        continue
                    if subcommand == "key":
                        if len(subargs) != 1:
                            console.print("[red]Usage: /tools key <provider>[/red]")
                            continue
                        provider_name = normalize_web_search_provider(subargs[0])
                        field_name = provider_key_config_field(provider_name)
                        if field_name is None:
                            console.print(f"[red]Provider {provider_name} does not use a managed API key.[/red]")
                            continue
                        secret_value = console.input(
                            f"[bold cyan]{provider_name} key[/bold cyan] > ",
                            password=True,
                        ).strip()
                        if not secret_value:
                            console.print(f"[yellow]No key entered for {provider_name}; existing value kept.[/yellow]")
                            continue
                        setattr(cfg.tools, field_name, secret_value)
                        await _persist_tool_settings()
                        console.print(f"[green]Saved key for {provider_name}.[/green]")
                        continue
                    if subcommand == "key-clear":
                        if len(subargs) != 1:
                            console.print("[red]Usage: /tools key-clear <provider>[/red]")
                            continue
                        provider_name = normalize_web_search_provider(subargs[0])
                        field_name = provider_key_config_field(provider_name)
                        if field_name is None:
                            console.print(f"[red]Provider {provider_name} does not use a managed API key.[/red]")
                            continue
                        setattr(cfg.tools, field_name, None)
                        await _persist_tool_settings()
                        console.print(f"[green]Cleared key for {provider_name}.[/green]")
                        continue

                    console.print(f"[yellow]Unknown /tools command: {subcommand}[/yellow]")
                    continue

                routing_result = await _handle_terminal_goal_input(
                    text=text,
                    session_id=current_session,
                    current_model=cfg.model,
                    autonomy_mode=_ensure_security_config().autonomy_mode,
                    session_store=session_store,
                    ensure_runtime_service=_ensure_runtime_service,
                    intent_invoker=engine,
                )
                if routing_result["handled"]:
                    continue
                if routing_result["chat_text"]:
                    text = str(routing_result["chat_text"])
                else:
                    console.print(f"[yellow]Unknown command: /{command}[/yellow]")
                    continue

            else:
                routing_result = await _handle_terminal_goal_input(
                    text=text,
                    session_id=current_session,
                    current_model=cfg.model,
                    autonomy_mode=_ensure_security_config().autonomy_mode,
                    session_store=session_store,
                    ensure_runtime_service=_ensure_runtime_service,
                    intent_invoker=engine,
                )
                if routing_result["handled"]:
                    continue
                if routing_result["chat_text"]:
                    text = str(routing_result["chat_text"])

            turns += 1
            try:
                await _run_chat_turn(text)
            except Exception as exc:
                console.print(f"[red]Chat failed: {exc}[/red]")

        if turns >= max_turns:
            console.print(f"[dim]Reached max turns ({max_turns}). Exiting TUI.[/dim]")
    finally:
        await engine.close()


async def _skills_list_async(db_path: str | None) -> None:
    """列出技能庫內容。"""
    from mochi.learning.skill_library import SkillLibrary
    from mochi.learning.skill_loader import SkillLoader, default_system_skills_dir

    library = SkillLibrary(db_path=_resolve_skills_db_path(db_path))
    if db_path is None and _auto_sync_filesystem_skills_enabled():
        await SkillLoader.from_paths(
            _default_skills_dir(),
            system_skills_dir=default_system_skills_dir(),
        ).sync(library)
    skills = await library.list()

    table = Table(title="Learned Skills")
    table.add_column("name")
    table.add_column("id")
    table.add_column("version", justify="right")
    table.add_column("times_used", justify="right")
    table.add_column("success_rate", justify="right")

    for skill in skills:
        success_rate = _skill_value(skill, "success_rate", 0.0)
        try:
            success_rate_text = f"{float(success_rate):.2f}"
        except (TypeError, ValueError):
            success_rate_text = str(success_rate)
        table.add_row(
            str(_skill_value(skill, "name")),
            str(_skill_value(skill, "skill_id", _skill_value(skill, "id"))),
            str(_skill_value(skill, "version", 1)),
            str(_skill_value(skill, "times_used", 0)),
            success_rate_text,
        )

    console.print(table)


async def _skills_show_async(skill_id: str, db_path: str | None) -> None:
    """顯示單一技能詳情。"""
    from mochi.learning.skill_library import SkillLibrary
    from mochi.learning.skill_loader import SkillLoader, default_system_skills_dir

    library = SkillLibrary(db_path=_resolve_skills_db_path(db_path))
    if db_path is None and _auto_sync_filesystem_skills_enabled():
        await SkillLoader.from_paths(
            _default_skills_dir(),
            system_skills_dir=default_system_skills_dir(),
        ).sync(library)
    skill = await library.get(skill_id)
    if skill is None:
        console.print(f"[red]Skill not found: {skill_id}[/red]")
        sys.exit(1)

    console.print(f"[bold]{_skill_value(skill, 'name')}[/bold]")
    fields = (
        ("id", _skill_value(skill, "skill_id", _skill_value(skill, "id"))),
        ("version", _skill_value(skill, "version", 1)),
        ("description", _skill_value(skill, "description")),
        ("preconditions", _skill_value(skill, "preconditions")),
        ("steps", _skill_value(skill, "steps", [])),
        ("tools_used", _skill_value(skill, "tools_used", [])),
        ("times_used", _skill_value(skill, "times_used", 0)),
        ("success_rate", _skill_value(skill, "success_rate", 0.0)),
    )
    for label, value in fields:
        if isinstance(value, list):
            console.print(f"  {label}:")
            for item in value:
                console.print(f"    - {item}")
        else:
            console.print(f"  {label}: {value}")


async def _skills_delete_async(skill_id: str, db_path: str | None) -> None:
    """刪除單一技能。"""
    from mochi.learning.skill_library import SkillLibrary

    library = SkillLibrary(db_path=_resolve_skills_db_path(db_path))
    skill = await library.get(skill_id)
    if skill is None:
        console.print(f"[red]Skill not found: {skill_id}[/red]")
        sys.exit(1)

    await library.delete(skill_id)
    console.print(f"[green]Deleted skill: {skill_id}[/green]")


async def _skills_export_async(db_path: str | None, output_path: str | None) -> None:
    """匯出技能庫。"""
    from mochi.learning.skill_library import SkillLibrary
    from mochi.learning.skill_loader import SkillLoader, default_system_skills_dir

    library = SkillLibrary(db_path=_resolve_skills_db_path(db_path))
    if db_path is None and _auto_sync_filesystem_skills_enabled():
        await SkillLoader.from_paths(
            _default_skills_dir(),
            system_skills_dir=default_system_skills_dir(),
        ).sync(library)
    payload = await library.export()
    output = _format_json_payload(payload)

    if output_path:
        path = Path(os.path.expanduser(output_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
        console.print(f"[green]Exported skill library: {path}[/green]")
        return

    console.print(output)


async def _doctor_async() -> None:
    """非同步執行系統診斷。"""
    from mochi.backends.ollama import OllamaBackend
    from mochi.config.manager import load_config

    cfg = load_config()
    console.print("[bold]Mochi System Diagnostics[/bold]\n")

    # Python version
    py_ver = sys.version_info
    status = "[green]OK[/green]" if py_ver >= (3, 11) else "[red]requires Python 3.11+[/red]"
    console.print(f"  Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}  {status}")

    # Ollama connection
    backend = OllamaBackend(model="", base_url=cfg.ollama.base_url)
    try:
        try:
            ok = await backend.health_check()
            status = "[green]OK[/green]" if ok else "[red]unreachable[/red]"
        except Exception as exc:
            ok = False
            status = f"[red]diagnostics failed: {exc}[/red]"
        console.print(f"  Ollama @ {cfg.ollama.base_url}  {status}")
    finally:
        await backend.close()

    info, model_ready, model_error = await _inspect_configured_model(
        cfg.model,
        cfg.ollama.base_url,
    )
    if info is None:
        console.print(f"  Configured model {cfg.model}  [red]unresolved[/red]")
        console.print(f"    issue: {model_error}")
    else:
        issue = None if model_ready else _describe_backend_issue(
            info.backend_type,
            info.metadata,
            model_error,
        )
        console.print(
            f"  Configured model {cfg.model} ({info.backend_type})  "
            f"{_health_status_markup(model_ready)}"
        )
        if issue:
            console.print(f"    issue: {issue}")

    # Local voice/audio (bounded continuous path)
    voice_cfg = getattr(cfg, "voice", None)
    sample_rate = getattr(voice_cfg, "sample_rate", None)
    channels = getattr(voice_cfg, "channels", None)
    try:
        audio_io = create_default_audio_io()
    except Exception as exc:  # pragma: no cover - defensive branch
        console.print("  Voice/Audio (bounded continuous)  [red]unavailable[/red]")
        console.print(f"    issue: {exc}")
    else:
        backend_name = audio_io.__class__.__name__
        has_record_stream = callable(getattr(audio_io, "record_stream", None))
        unavailable_reason = None
        if backend_name == "UnavailableAudioIO":
            unavailable_reason = str(getattr(audio_io, "_reason", "audio backend init failed"))

        is_audio_ready = has_record_stream and unavailable_reason is None
        console.print(
            "  Voice/Audio (bounded continuous)  "
            f"{_health_status_markup(is_audio_ready)}"
        )
        console.print(f"    audio_io: {backend_name}")
        if isinstance(sample_rate, int) and isinstance(channels, int):
            console.print(f"    audio_config: {sample_rate} Hz / {channels} ch")
        console.print(f"    record_stream: {'available' if has_record_stream else 'missing'}")
        if unavailable_reason is not None:
            console.print(f"    issue: {unavailable_reason}")

        if unavailable_reason is not None:
            console.print("    diagnostics: skipped (audio backend unavailable)")
        else:
            helper_name, helper = _find_audio_runtime_diagnostics_helper(audio_io)
            if helper is None:
                console.print("    diagnostics: not exposed by audio_io backend")
            else:
                try:
                    details = helper()
                    if inspect.isawaitable(details):
                        details = await details
                except Exception as exc:
                    console.print(
                        f"    diagnostics ({helper_name}): [yellow]failed[/yellow] ({exc})"
                    )
                else:
                    if isinstance(details, dict):
                        if details:
                            for key, value in details.items():
                                console.print(f"    {key}: {value}")
                        else:
                            console.print(f"    diagnostics ({helper_name}): empty")
                    else:
                        console.print(f"    diagnostics ({helper_name}): {details}")

    # Config file
    from mochi.config.manager import user_config_path

    cfg_path = user_config_path()
    exists = cfg_path.exists()
    status = "[green]exists[/green]" if exists else "[yellow]not created (using defaults)[/yellow]"
    console.print(f"  Config file {cfg_path}  {status}")

    console.print()


async def _model_info_async(config_path: str | None) -> None:
    """非同步顯示當前模型資訊。"""
    from mochi.config.manager import load_config

    cfg = load_config(config_path)
    info, is_ready, error = await _inspect_configured_model(
        cfg.model,
        cfg.ollama.base_url,
    )

    console.print("[bold]Current Model Information[/bold]")
    console.print(f"  configured: [cyan]{cfg.model}[/cyan]")
    if info is None:
        console.print("  status: [red]unresolved[/red]")
        console.print(f"  issue: {error}")
        return

    issue = None if is_ready else _describe_backend_issue(
        info.backend_type,
        info.metadata,
        error,
    )
    console.print(f"  name: [cyan]{info.name}[/cyan]")
    console.print(f"  backend: {info.backend_type}")
    console.print(f"  context_length: {info.context_length}")
    console.print(f"  supports_tool_calling: {info.supports_tool_calling}")
    console.print(f"  status: {_health_status_markup(is_ready)}")
    if issue:
        console.print(f"  issue: {issue}")
    if "dependency_ready" in info.metadata:
        console.print(f"  dependency_ready: {info.metadata['dependency_ready']}")


async def _model_switch_async(model_spec: str, config_path: str | None) -> None:
    """非同步切換模型並輸出結果。"""
    from mochi.agents.engine import AgentEngine
    from mochi.config.manager import load_config

    cfg = load_config(config_path)
    engine = AgentEngine(cfg)
    try:
        await engine.initialize()
        info = await engine.switch_model(model_spec)
    except Exception as exc:
        console.print(f"[red]Model switch failed: {exc}[/red]")
        sys.exit(1)
    finally:
        await engine.close()

    console.print("[green]Model switched successfully[/green]")
    console.print(f"  name: [cyan]{info.name}[/cyan]")
    console.print(f"  backend: {info.backend_type}")


async def _voice_async(
    *,
    config_path: str | None,
    session_id: str | None,
    max_record_seconds: float,
    playback: bool,
    input_audio: str | None,
    output_audio: str | None,
    continuous: bool = False,
    chunk_seconds: float = 0.25,
    max_turns: int = 0,
    audio_io: BaseAudioIO | None = None,
) -> None:
    """非同步執行語音流程（單輪或連續回合）。"""
    from mochi.agents.engine import AgentEngine
    from mochi.config.manager import load_config
    from mochi.voice.events import (
        AgentFinalTextEvent,
        SynthesizedAudioChunkEvent,
        TranscriptionEvent,
        VoiceErrorEvent,
    )

    cfg = load_config(config_path)
    io = audio_io or create_default_audio_io()
    engine = AgentEngine(cfg)

    if max_turns < 0:
        console.print("[red]max_turns must be greater than or equal to 0.[/red]")
        sys.exit(1)
    if continuous and input_audio:
        console.print("[red]Continuous mode does not support --input-audio.[/red]")
        sys.exit(1)
    if continuous and chunk_seconds <= 0:
        console.print("[red]chunk_seconds must be greater than 0.[/red]")
        sys.exit(1)

    async def _consume_voice_events(event_iter) -> tuple[list[str], bytes]:  # noqa: ANN001
        error_codes: list[str] = []
        synthesized_chunks: list[bytes] = []
        async for event in event_iter:
            runtime_status = _describe_voice_runtime_event(event)
            if runtime_status is not None:
                label, value = runtime_status
                console.print(f"[dim]{label}[/dim] {value}")
            elif isinstance(event, TranscriptionEvent):
                console.print(f"[cyan]STT[/cyan] {event.text}")
            elif isinstance(event, AgentFinalTextEvent):
                console.print(f"[green]Agent[/green] {event.text}")
            elif isinstance(event, SynthesizedAudioChunkEvent):
                synthesized_chunks.append(event.chunk)
            elif isinstance(event, VoiceErrorEvent):
                console.print(f"[red]Voice error ({event.code}): {event.message}[/red]")
                error_codes.append(event.code)
        return error_codes, b"".join(synthesized_chunks)

    try:
        await engine.initialize()

        if continuous:
            console.print(
                "[dim]Continuous voice mode "
                f"(max duration {max_record_seconds:.1f}s, chunk {chunk_seconds:.2f}s)[/dim]"
            )
            voice_session = await engine.get_or_create_voice_session(session_id=session_id)
            append_chunk = getattr(voice_session, "append_audio_chunk_with_vad", None)
            clear_buffer = getattr(voice_session, "interrupt_buffered_input", None)

            if not callable(append_chunk):
                raise RuntimeError("Current voice session does not support continuous buffered mode.")

            queued_utterances: deque[bytes] = deque()
            queued_utterance_limit = 8
            total_synthesized_chunks: list[bytes] = []
            vad_prev_is_speech: bool | None = None
            turn_errors: list[str] = []
            turn_tasks: dict[int, asyncio.Task[None]] = {}
            next_turn_id = 0
            active_turn_id: int | None = None
            started_turns = 0
            turn_processing_done: set[int] = set()
            turn_pending_playback_chunks: dict[int, int] = {}
            utterance_buffer = bytearray()
            playback_queue: asyncio.Queue[tuple[int, bytes] | None] = asyncio.Queue()

            def _is_turn_active(turn_id: int) -> bool:
                return active_turn_id == turn_id

            @asynccontextmanager
            async def _fallback_playback_session():  # noqa: ANN202
                async def _play_chunk(chunk: bytes) -> None:
                    await io.play_once(
                        chunk,
                        sample_rate=cfg.voice.sample_rate,
                        channels=cfg.voice.channels,
                    )

                yield _play_chunk

            async def _playback_worker() -> None:
                session_factory = getattr(io, "playback_session", None)
                playback_session = (
                    session_factory(
                        sample_rate=cfg.voice.sample_rate,
                        channels=cfg.voice.channels,
                    )
                    if callable(session_factory)
                    else _fallback_playback_session()
                )
                async with playback_session as play_chunk:
                    while True:
                        item = await playback_queue.get()
                        if item is None:
                            return
                        turn_id, chunk = item
                        if not _is_turn_active(turn_id):
                            continue
                        try:
                            await play_chunk(chunk)
                        except Exception as exc:
                            console.print(
                                "[red]Voice error (CONTINUOUS_PLAYBACK_ERROR): "
                                f"{exc}[/red]"
                            )
                            turn_errors.append("CONTINUOUS_PLAYBACK_ERROR")
                        finally:
                            pending = turn_pending_playback_chunks.get(turn_id, 0)
                            if pending > 0:
                                turn_pending_playback_chunks[turn_id] = pending - 1
                            await _finalize_turn_if_ready(turn_id)

            async def _start_turn(audio: bytes) -> None:
                nonlocal active_turn_id, next_turn_id, started_turns
                next_turn_id += 1
                turn_id = next_turn_id
                active_turn_id = turn_id
                started_turns += 1
                turn_pending_playback_chunks[turn_id] = 0
                task = asyncio.create_task(_run_turn(turn_id, audio))
                turn_tasks[turn_id] = task
                task.add_done_callback(lambda _task, done_turn_id=turn_id: turn_tasks.pop(done_turn_id, None))

            def _can_enqueue_utterance() -> bool:
                if max_turns > 0:
                    remaining_turns = max_turns - started_turns - len(queued_utterances)
                    if remaining_turns <= 0:
                        return False
                return len(queued_utterances) < queued_utterance_limit

            async def _maybe_start_next_turn() -> None:
                if active_turn_id is not None:
                    return
                if not queued_utterances:
                    return
                if max_turns > 0 and started_turns >= max_turns:
                    queued_utterances.clear()
                    return
                next_audio = queued_utterances.popleft()
                await _start_turn(next_audio)

            async def _finalize_turn_if_ready(turn_id: int) -> None:
                nonlocal active_turn_id
                if active_turn_id != turn_id:
                    return
                if turn_id not in turn_processing_done:
                    return
                if turn_pending_playback_chunks.get(turn_id, 0) > 0:
                    return
                active_turn_id = None
                turn_processing_done.discard(turn_id)
                turn_pending_playback_chunks.pop(turn_id, None)
                await _maybe_start_next_turn()

            async def _run_turn(turn_id: int, audio: bytes) -> None:
                try:
                    async for event in engine.voice_chat(audio, session_id=session_id):
                        if not _is_turn_active(turn_id):
                            continue
                        runtime_status = _describe_voice_runtime_event(event)
                        if runtime_status is not None:
                            label, value = runtime_status
                            console.print(f"[dim]{label}[/dim] {value}")
                        elif isinstance(event, TranscriptionEvent):
                            console.print(f"[cyan]STT[/cyan] {event.text}")
                        elif isinstance(event, AgentFinalTextEvent):
                            console.print(f"[green]Agent[/green] {event.text}")
                        elif isinstance(event, SynthesizedAudioChunkEvent):
                            total_synthesized_chunks.append(event.chunk)
                            if playback and event.chunk:
                                turn_pending_playback_chunks[turn_id] = (
                                    turn_pending_playback_chunks.get(turn_id, 0) + 1
                                )
                                await playback_queue.put((turn_id, event.chunk))
                        elif isinstance(event, VoiceErrorEvent):
                            console.print(f"[red]Voice error ({event.code}): {event.message}[/red]")
                            turn_errors.append(event.code)
                except Exception as exc:
                    if _is_turn_active(turn_id):
                        console.print(f"[red]Voice error (CONTINUOUS_TURN_ERROR): {exc}[/red]")
                        turn_errors.append("CONTINUOUS_TURN_ERROR")
                finally:
                    turn_processing_done.add(turn_id)
                    await _finalize_turn_if_ready(turn_id)

            playback_task: asyncio.Task[None] | None = None
            if playback:
                playback_task = asyncio.create_task(_playback_worker())
            completed_normally = False

            try:
                async for chunk in io.record_stream(
                    sample_rate=cfg.voice.sample_rate,
                    channels=cfg.voice.channels,
                    chunk_seconds=chunk_seconds,
                    max_seconds=max_record_seconds,
                ):
                    utterance_buffer.extend(chunk)
                    observation = await append_chunk(
                        chunk,
                        session_id=session_id,
                        include_vad_state=True,
                    )
                    endpoint = False
                    is_speech: bool | None = None
                    if isinstance(observation, dict):
                        endpoint = bool(observation.get("endpoint", False))
                        speech_state = observation.get("is_speech")
                        if isinstance(speech_state, bool):
                            is_speech = speech_state
                    else:
                        endpoint = bool(observation)

                    speech_started = is_speech is True and vad_prev_is_speech is not True
                    speech_ended = endpoint or (vad_prev_is_speech is True and is_speech is False)
                    if speech_started:
                        console.print("[dim]VAD[/dim] speech_started")
                    if speech_ended:
                        console.print("[dim]VAD[/dim] speech_ended")
                    if is_speech is not None:
                        vad_prev_is_speech = is_speech
                    elif endpoint:
                        vad_prev_is_speech = False

                    if not endpoint:
                        continue

                    utterance_audio = bytes(utterance_buffer)
                    utterance_buffer.clear()
                    if callable(clear_buffer):
                        with suppress(Exception):
                            await clear_buffer()
                    if utterance_audio and _can_enqueue_utterance():
                        queued_utterances.append(utterance_audio)
                    await _maybe_start_next_turn()
                    if max_turns > 0 and started_turns >= max_turns and active_turn_id is not None:
                        break

                await _maybe_start_next_turn()

                while turn_tasks:
                    await asyncio.gather(*list(turn_tasks.values()), return_exceptions=True)
                completed_normally = True
            finally:
                for task in list(turn_tasks.values()):
                    if not task.done():
                        task.cancel()
                if turn_tasks:
                    await asyncio.gather(*list(turn_tasks.values()), return_exceptions=True)
                if playback_task is not None:
                    if not completed_normally:
                        playback_task.cancel()
                    await playback_queue.put(None)
                    with suppress(asyncio.CancelledError):
                        await playback_task

            if turn_errors:
                non_empty_errors = [code for code in turn_errors if code != "EMPTY_AUDIO_BUFFER"]
                if non_empty_errors:
                    sys.exit(1)

            synthesized_audio = b"".join(total_synthesized_chunks)
            if output_audio:
                write_audio_file_from_pcm16(
                    output_audio,
                    synthesized_audio,
                    sample_rate=cfg.voice.sample_rate,
                )
                console.print(f"[dim]Wrote audio: {output_audio}[/dim]")
            return

        if input_audio:
            audio = read_audio_file_as_pcm16(
                input_audio,
                sample_rate=cfg.voice.sample_rate,
            )
            console.print(f"[dim]Loaded audio: {input_audio}[/dim]")
        else:
            console.print(f"[dim]Recording (up to {max_record_seconds:.1f}s)...[/dim]")
            audio = await io.record_once(
                sample_rate=cfg.voice.sample_rate,
                channels=cfg.voice.channels,
                max_seconds=max_record_seconds,
            )

        if not audio:
            console.print("[red]No audio data captured.[/red]")
            sys.exit(1)

        error_codes, synthesized_audio = await _consume_voice_events(
            engine.voice_chat(audio, session_id=session_id)
        )
        if error_codes:
            sys.exit(1)
        if output_audio:
            write_audio_file_from_pcm16(
                output_audio,
                synthesized_audio,
                sample_rate=cfg.voice.sample_rate,
            )
            console.print(f"[dim]Wrote audio: {output_audio}[/dim]")

        if playback and synthesized_audio:
            await io.play_once(
                synthesized_audio,
                sample_rate=cfg.voice.sample_rate,
                channels=cfg.voice.channels,
            )
    except Exception as exc:
        console.print(f"[red]Voice flow failed: {exc}[/red]")
        sys.exit(1)
    finally:
        await engine.close()


async def _channels_run_async(config_path: str | None) -> None:
    """啟動頻道適配器並阻塞直到中斷。"""
    from mochi.agents.engine import AgentEngine
    from mochi.channels.manager import build_channel_manager
    from mochi.config.manager import load_config

    cfg = load_config(config_path)
    engine = AgentEngine(cfg)
    manager = None
    try:
        await engine.initialize()
        manager = build_channel_manager(
            cfg,
            engine,
            config_path=config_path,
            persist_config_updates=True,
        )
        channel_names = manager.list_channels()
        if not channel_names:
            console.print("[yellow]No channels are enabled. Enable discord or telegram in config.channels.[/yellow]")
            return

        await manager.start_all()
        console.print(f"[green]Started channels: {', '.join(channel_names)}[/green]")
        console.print("[dim]Press Ctrl+C to stop.[/dim]")
        stop_event = asyncio.Event()
        await stop_event.wait()
    except asyncio.CancelledError:
        console.print("\n[dim]Stopping channels...[/dim]")
    except KeyboardInterrupt:
        console.print("\n[dim]Stopping channels...[/dim]")
    except Exception as exc:
        console.print(f"[red]Channel startup failed: {exc}[/red]")
        sys.exit(1)
    finally:
        if manager is not None:
            await manager.stop_all()
        await engine.close()


async def _channels_voice_settings_async(
    *,
    config_path: str | None,
    tts_voice: str | None,
    session_mode: str | None,
    reply_model_mode: str | None,
    reply_model: str | None,
) -> None:
    from mochi.config.manager import load_config, save_config

    cfg = load_config(config_path)
    updates_requested = any(
        value is not None
        for value in (tts_voice, session_mode, reply_model_mode, reply_model)
    )
    if not updates_requested:
        console.print("[bold]Discord voice settings[/bold]")
        console.print(f"  tts_voice: {getattr(cfg.voice, 'tts_voice', '')}")
        console.print(
            "  session_mode: "
            f"{getattr(cfg.voice, 'session_mode', 'append_current')}"
        )
        console.print(
            "  reply_model_mode: "
            f"{getattr(cfg.voice, 'reply_model_mode', 'inherit_active')}"
        )
        console.print(
            "  reply_model_id: "
            f"{getattr(cfg.voice, 'reply_model_id', '')}"
        )
        return

    if tts_voice is not None:
        cfg.voice.tts_voice = tts_voice.strip()

    if session_mode is not None:
        normalized_session_mode = session_mode.strip()
        if normalized_session_mode not in {"append_current", "isolated_voice"}:
            console.print(
                "[red]Invalid session_mode. Use one of: "
                "append_current, isolated_voice.[/red]"
            )
            sys.exit(1)
        if hasattr(cfg.voice, "session_mode"):
            setattr(cfg.voice, "session_mode", normalized_session_mode)
        else:
            console.print(
                "[red]Config schema is missing voice.session_mode. "
                "Please add this field in shared config/schema first.[/red]"
            )
            sys.exit(1)

    if reply_model_mode is not None:
        normalized_mode = reply_model_mode.strip().replace("-", "_")
        if normalized_mode not in {"inherit_active", "configured_model"}:
            console.print(
                "[red]Invalid reply_model_mode. Use one of: "
                "inherit_active, configured_model.[/red]"
            )
            sys.exit(1)
        if hasattr(cfg.voice, "reply_model_mode"):
            setattr(cfg.voice, "reply_model_mode", normalized_mode)
        else:
            console.print(
                "[red]Config schema is missing voice.reply_model_mode. "
                "Please add this field in shared config/schema first.[/red]"
            )
            sys.exit(1)

    if reply_model is not None:
        if hasattr(cfg.voice, "reply_model_id"):
            setattr(cfg.voice, "reply_model_id", reply_model.strip())
        else:
            console.print(
                "[red]Config schema is missing voice.reply_model_id. "
                "Please add this field in shared config/schema first.[/red]"
            )
            sys.exit(1)

    path = save_config(cfg, config_path)
    console.print(f"[green]Saved Discord voice settings:[/green] {path}")


if __name__ == "__main__":
    app()
