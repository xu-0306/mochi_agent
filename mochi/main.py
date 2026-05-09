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
from pathlib import Path
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

    cfg = load_config()
    return Path(os.path.expanduser(cfg.skills_dir)) / "skills.db"


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
        "  tts_backend: \"edge-tts\"",
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
) -> None:
    """Send one message and print the agent response."""
    asyncio.run(_chat_async(text, model or None, config_path or None))


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
) -> None:
    """非同步執行對話。"""
    from mochi.agents.engine import AgentEngine
    from mochi.agents.events import ErrorEvent, FinalAnswerEvent, TextChunkEvent
    from mochi.config.manager import load_config

    cfg = load_config(config_path)
    if model:
        cfg.model = model

    engine = AgentEngine(cfg)
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
    from mochi.config.manager import load_config
    from mochi.sessions.store import SessionStore

    if max_turns <= 0:
        console.print("[red]max_turns must be greater than 0.[/red]")
        sys.exit(1)

    cfg = load_config(config_path)
    if model:
        cfg.model = model

    engine = AgentEngine(cfg)
    current_session = session_id.strip() or DEFAULT_TUI_SESSION_ID
    session_store = SessionStore(sessions_dir=getattr(cfg, "sessions_dir", "~/.mochi/sessions"))

    async def _reset_engine() -> None:
        nonlocal engine
        await engine.close()
        engine = AgentEngine(cfg)

    try:
        await engine.initialize()
    except Exception as exc:
        console.print(f"[red]Initialization failed: {exc}[/red]")
        sys.exit(1)

    console.print(f"[bold cyan]Mochi TUI[/bold cyan] model=[cyan]{cfg.model}[/cyan]")
    console.print(
        f"[dim]session={current_session} | max_turns={max_turns} | type /help for commands[/dim]"
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
                        continue
                    next_session = args[0].strip()
                    if not next_session:
                        console.print("[red]Session id must not be empty.[/red]")
                        continue
                    current_session = next_session
                    console.print(f"[green]Session switched:[/green] {current_session}")
                    continue

                console.print(f"[yellow]Unknown command: /{command}[/yellow]")
                continue

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
        manager = build_channel_manager(cfg, engine)
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


if __name__ == "__main__":
    app()
