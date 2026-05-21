"""頻道管理器。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from mochi.agents.events import ErrorEvent, FinalAnswerEvent, TextChunkEvent
from mochi.channels.base import BaseChannel
from mochi.channels.events import (
    Attachment,
    AttachmentEvent,
    ChannelEvent,
    CommandEvent,
    MessageEvent,
)
from mochi.channels.session_bridge import SessionBridge
from mochi.config.schema import MochiConfig


class ChannelConfigurationError(RuntimeError):
    """頻道設定不足或不可啟動。"""


class ChannelNotRegisteredError(RuntimeError):
    """指定頻道尚未註冊，無法執行管理操作。"""


class ChannelManager:
    """啟停多頻道，並將平台事件路由到 AgentEngine。"""

    def __init__(
        self,
        *,
        engine: Any | None = None,
        config: MochiConfig | None = None,
        session_bridge: SessionBridge | None = None,
        config_path: str | Path | None = None,
        persist_config_updates: bool = False,
    ) -> None:
        self._engine = engine
        self._config = config
        self._session_bridge = session_bridge or SessionBridge()
        self._config_path = Path(config_path).expanduser() if config_path is not None else None
        self._persist_config_updates = persist_config_updates
        self._channels: dict[str, BaseChannel] = {}
        self._running_channels: set[str] = set()
        self._running = False

    def register(self, channel: BaseChannel) -> None:
        """注冊一個頻道適配器。"""
        channel.set_event_handler(self.handle_event)
        self._channels[channel.name] = channel

    def get(self, name: str) -> BaseChannel | None:
        """取得已注冊的頻道。"""
        return self._channels.get(name)

    def list_channels(self) -> list[str]:
        """列出已注冊頻道名稱。"""
        return sorted(self._channels)

    async def start_all(self) -> None:
        """啟動所有已注冊的頻道。"""
        errors: list[str] = []
        for name in self._channels:
            try:
                await self.start_channel(name)
            except Exception as exc:  # pragma: no cover - 防禦性收斂
                errors.append(f"{name}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    async def stop_all(self) -> None:
        """停止所有頻道。"""
        errors: list[str] = []
        for name in reversed(list(self._channels)):
            try:
                await self.stop_channel(name)
            except Exception as exc:  # pragma: no cover - 防禦性收斂
                errors.append(f"{name}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    async def start_channel(self, name: str) -> None:
        """啟動單一已註冊頻道。"""
        channel = self._channels.get(name)
        if channel is None:
            raise ChannelNotRegisteredError(f"Channel '{name}' is not registered.")
        if name in self._running_channels:
            return

        await channel.start()
        self._running_channels.add(name)
        self._running = bool(self._running_channels)

    async def stop_channel(self, name: str) -> None:
        """停止單一已註冊頻道。"""
        channel = self._channels.get(name)
        if channel is None:
            raise ChannelNotRegisteredError(f"Channel '{name}' is not registered.")
        if name not in self._running_channels:
            return

        await channel.stop()
        self._running_channels.discard(name)
        self._running = bool(self._running_channels)

    def running_channels(self) -> list[str]:
        """列出目前已啟動中的頻道名稱。"""
        return sorted(self._running_channels)

    def is_running(self, name: str) -> bool:
        """判斷指定頻道是否處於啟動狀態。"""
        return name in self._running_channels

    async def handle_event(self, event: ChannelEvent) -> None:
        """處理平台事件並回覆到原頻道。"""
        channel = self._channels.get(event.channel)
        if channel is None:
            return

        if isinstance(event, CommandEvent):
            await self._handle_command(channel, event)
            return

        if isinstance(event, AttachmentEvent):
            user_text = self._format_attachment_input(event)
        elif isinstance(event, MessageEvent):
            user_text = event.text.strip()
        else:
            return

        if not user_text:
            return
        if self._engine is None:
            await channel.send_message(
                event.chat_id,
                "Mochi channel manager is not connected to an AgentEngine.",
                reply_to=event.message_id or None,
            )
            return

        session_id = self._session_bridge.get_or_create_session_id(
            event.channel,
            event.chat_id,
        )
        reply = await self._collect_agent_reply(
            self._engine.chat(user_text, session_id=session_id)
        )
        if reply:
            await channel.send_message(
                event.chat_id,
                reply,
                reply_to=event.message_id or None,
            )
            await self._maybe_send_discord_voice_reply(
                channel=channel,
                event=event,
                reply=reply,
            )

    def _format_attachment_input(self, event: AttachmentEvent) -> str:
        """將附件 metadata 收斂為 Agent 可讀文字，不下載附件內容。"""
        lines: list[str] = []
        caption = event.text.strip()
        if caption:
            lines.append(f"Caption: {caption}")

        for index, attachment in enumerate(event.attachments, start=1):
            parts = self._format_attachment_metadata(attachment)
            if parts:
                lines.append(f"Attachment {index}: " + ", ".join(parts))

        return "\n".join(lines).strip()

    def _format_attachment_metadata(self, attachment: Attachment) -> list[str]:
        """格式化單一附件的安全 metadata。"""
        parts: list[str] = []
        if attachment.filename:
            parts.append(f"filename={attachment.filename}")
        if attachment.content_type:
            parts.append(f"content_type={attachment.content_type}")
        if attachment.url:
            parts.append(f"url={attachment.url}")
        if attachment.id:
            parts.append(f"file_id={attachment.id}")
        if attachment.size is not None:
            parts.append(f"size={attachment.size}")
        return parts

    async def _handle_command(self, channel: BaseChannel, event: CommandEvent) -> None:
        """處理平台 command。"""
        command = event.command.lower().strip()
        if self._is_discord_mutation_command(command) and not self._is_discord_admin_command(
            channel=channel,
            event=event,
        ):
            await channel.send_message(
                event.chat_id,
                f"Discord command `/{command}` is admin-only.",
                reply_to=event.message_id or None,
            )
            return

        if command in {"start", "help"}:
            await channel.send_message(
                event.chat_id,
                "Mochi is ready. Send a message and I will reply in this chat.",
                reply_to=event.message_id or None,
            )
            return

        if command == "status":
            status_text = self._format_channel_status(channel)
            await channel.send_message(
                event.chat_id,
                status_text,
                reply_to=event.message_id or None,
            )
            return

        if command == "join":
            guild_id = self._metadata_int(event.metadata, "discord_guild_id")
            voice_channel_id = self._metadata_int(event.metadata, "discord_voice_channel_id")
            join_voice = getattr(channel, "join_voice_channel", None)
            if not callable(join_voice):
                await channel.send_message(
                    event.chat_id,
                    "Discord voice join is unavailable on this channel adapter.",
                    reply_to=event.message_id or None,
                )
                return
            if guild_id <= 0:
                await channel.send_message(
                    event.chat_id,
                    "Discord voice join requires a guild context.",
                    reply_to=event.message_id or None,
                )
                return
            if voice_channel_id <= 0:
                await channel.send_message(
                    event.chat_id,
                    "Join a Discord voice channel first, then run /join again.",
                    reply_to=event.message_id or None,
                )
                return
            try:
                payload = await join_voice(guild_id, voice_channel_id)
            except Exception as exc:
                await channel.send_message(
                    event.chat_id,
                    f"Discord voice join failed: {exc}",
                    reply_to=event.message_id or None,
                )
                return
            await channel.send_message(
                event.chat_id,
                "Discord voice joined. "
                f"guild_id={payload.get('guild_id')} channel_id={payload.get('channel_id')}",
                reply_to=event.message_id or None,
            )
            return

        if command == "leave":
            guild_id = self._metadata_int(event.metadata, "discord_guild_id")
            leave_voice = getattr(channel, "leave_voice_channel", None)
            if not callable(leave_voice):
                await channel.send_message(
                    event.chat_id,
                    "Discord voice leave is unavailable on this channel adapter.",
                    reply_to=event.message_id or None,
                )
                return
            if guild_id <= 0:
                await channel.send_message(
                    event.chat_id,
                    "Discord voice leave requires a guild context.",
                    reply_to=event.message_id or None,
                )
                return
            left = await leave_voice(guild_id)
            await channel.send_message(
                event.chat_id,
                (
                    f"Discord voice left guild {guild_id}."
                    if left
                    else f"Discord voice was not active in guild {guild_id}."
                ),
                reply_to=event.message_id or None,
            )
            return

        if command == "voice-status":
            status_text = self._format_discord_voice_status(channel)
            await channel.send_message(
                event.chat_id,
                status_text,
                reply_to=event.message_id or None,
            )
            return

        if command == "voice-config":
            get_settings = getattr(channel, "get_voice_conversation_settings", None)
            if not callable(get_settings):
                await channel.send_message(
                    event.chat_id,
                    "Discord voice settings are unavailable on this channel adapter.",
                    reply_to=event.message_id or None,
                )
                return
            settings = get_settings()
            if not isinstance(settings, dict):
                await channel.send_message(
                    event.chat_id,
                    "Discord voice settings returned an invalid payload.",
                    reply_to=event.message_id or None,
                )
                return
            lines = ["Discord voice settings"]
            for key in ("session_mode", "reply_model_mode", "reply_model_id", "tts_voice"):
                if key in settings:
                    lines.append(f"{key}: {settings.get(key)}")
            await channel.send_message(
                event.chat_id,
                "\n".join(lines),
                reply_to=event.message_id or None,
            )
            return

        if command == "voice-set":
            if len(event.args) < 2:
                await channel.send_message(
                    event.chat_id,
                    "Usage: /voice-set <key> <value>",
                    reply_to=event.message_id or None,
                )
                return
            key = event.args[0]
            value = " ".join(event.args[1:]).strip()
            update_setting = getattr(channel, "update_voice_conversation_setting", None)
            if not callable(update_setting):
                await channel.send_message(
                    event.chat_id,
                    "Discord voice settings are unavailable on this channel adapter.",
                    reply_to=event.message_id or None,
                )
                return
            try:
                updated_key, updated_value = update_setting(key, value)
            except Exception as exc:
                await channel.send_message(
                    event.chat_id,
                    f"Discord voice setting update failed: {exc}",
                    reply_to=event.message_id or None,
                )
                return
            await self._update_shared_voice_setting(updated_key, updated_value)
            await channel.send_message(
                event.chat_id,
                f"Updated Discord voice setting: {updated_key}={updated_value}",
                reply_to=event.message_id or None,
            )
            return

        text = " ".join([event.command, *event.args]).strip()
        if not text:
            return
        message_event = MessageEvent(
            channel=event.channel,
            chat_id=event.chat_id,
            user_id=event.user_id,
            text=text,
            message_id=event.message_id,
            metadata=event.metadata,
        )
        await self.handle_event(message_event)

    async def _collect_agent_reply(self, events: AsyncIterator[Any]) -> str:
        """從 Agent event stream 收斂成要回覆到平台的一段文字。"""
        streamed_chunks: list[str] = []
        final_answer = ""
        async for event in events:
            if isinstance(event, TextChunkEvent):
                streamed_chunks.append(event.content)
            elif isinstance(event, FinalAnswerEvent):
                final_answer = event.content
            elif isinstance(event, ErrorEvent):
                return f"Error: {event.message}"
        return final_answer or "".join(streamed_chunks).strip()

    def _format_channel_status(self, channel: BaseChannel) -> str:
        """格式化單一 channel 的最小狀態摘要。"""
        get_status = getattr(channel, "get_runtime_status", None)
        if not callable(get_status):
            return f"{channel.name} is registered, but detailed runtime status is unavailable."

        status = get_status()
        if not isinstance(status, dict):
            return f"{channel.name} returned an invalid runtime status payload."

        lines = [f"Channel: {channel.name}"]
        for key in (
            "text_enabled",
            "voice_enabled",
            "message_mode",
            "auto_join_policy",
            "application_commands_synced",
            "rate_limit_per_user",
        ):
            if key in status:
                lines.append(f"{key}: {status[key]}")

        voice_runtime = status.get("voice_runtime")
        if isinstance(voice_runtime, dict):
            lines.append(
                "voice_runtime: "
                f"enabled={voice_runtime.get('enabled')}, "
                f"phase={voice_runtime.get('phase')}, "
                f"active_voice_room_count={voice_runtime.get('active_voice_room_count')}"
            )

        return "\n".join(lines)

    async def _maybe_send_discord_voice_reply(
        self,
        *,
        channel: BaseChannel,
        event: AttachmentEvent | MessageEvent,
        reply: str,
    ) -> None:
        """若 Discord voice room 已啟用，則同步播報同一段 reply。"""
        if event.channel != "discord" or not reply.strip():
            return

        guild_id = self._metadata_int(event.metadata, "discord_guild_id")
        if guild_id <= 0:
            return

        voice_auto_reply = getattr(channel, "is_voice_auto_reply_enabled", None)
        if callable(voice_auto_reply) and not bool(voice_auto_reply()):
            return

        speak_voice_reply = getattr(channel, "speak_voice_reply", None)
        if not callable(speak_voice_reply):
            return

        synthesize_audio = getattr(self._engine, "synthesize_speech", None) if self._engine is not None else None
        if not callable(synthesize_audio):
            return

        try:
            await speak_voice_reply(
                guild_id,
                reply,
                synthesize_audio=synthesize_audio,
            )
        except Exception:
            return

    def _format_discord_voice_status(self, channel: BaseChannel) -> str:
        """格式化 Discord voice runtime 狀態。"""
        get_status = getattr(channel, "get_runtime_status", None)
        if not callable(get_status):
            return "Discord voice runtime status is unavailable."

        status = get_status()
        if not isinstance(status, dict):
            return "Discord voice runtime returned an invalid status payload."

        voice_runtime = status.get("voice_runtime")
        if not isinstance(voice_runtime, dict):
            return "Discord voice runtime is unavailable."

        lines = [
            "Discord voice runtime",
            f"enabled: {voice_runtime.get('enabled')}",
            f"phase: {voice_runtime.get('phase')}",
            f"active_voice_room_count: {voice_runtime.get('active_voice_room_count')}",
        ]
        last_error = voice_runtime.get("last_error")
        if last_error:
            lines.append(f"last_error: {last_error}")

        rooms = voice_runtime.get("active_voice_rooms")
        if isinstance(rooms, list):
            for index, room in enumerate(rooms, start=1):
                if not isinstance(room, dict):
                    continue
                lines.append(
                    "room "
                    f"{index}: guild_id={room.get('guild_id')}, "
                    f"channel_id={room.get('channel_id')}, "
                    f"playback_state={room.get('playback_state')}"
                )
        return "\n".join(lines)

    @staticmethod
    def _metadata_int(metadata: dict[str, Any], key: str) -> int:
        """安全將 metadata 某欄位轉成 int。"""
        raw = metadata.get(key)
        if isinstance(raw, bool):
            return 0
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str) and raw.strip().isdigit():
            return int(raw.strip())
        return 0

    def _is_discord_mutation_command(self, command: str) -> bool:
        if not command:
            return False
        return command in {"join", "leave", "voice-set"}

    def _is_discord_admin_command(self, *, channel: BaseChannel, event: CommandEvent) -> bool:
        if event.channel != "discord":
            return True
        checker = getattr(channel, "is_admin_user", None)
        if not callable(checker):
            return True
        try:
            return bool(checker(event.user_id))
        except Exception:
            return False

    async def _update_shared_voice_setting(self, key: str, value: str) -> None:
        if self._config is None:
            return
        if key == "session_mode":
            self._config.voice.session_mode = (
                "append_current"
                if value in {"append_current", "shared"}
                else "isolated_voice"
            )
        elif key == "reply_model_mode":
            self._config.voice.reply_model_mode = (
                "inherit_active"
                if value in {"inherit_active", "agent-default"}
                else "configured_model"
            )
        elif key == "reply_model_id":
            self._config.voice.reply_model_id = value
        elif key == "tts_voice":
            self._config.voice.tts_voice = value
        else:
            return

        if self._engine is not None:
            apply_config = getattr(self._engine, "apply_config", None)
            if callable(apply_config):
                await apply_config(self._config, reload_voice=True)

        self._persist_config_updates_if_needed()

    def _persist_config_updates_if_needed(self) -> None:
        if self._config is None or not self._persist_config_updates:
            return

        from mochi.config.manager import save_config

        save_config(self._config, self._config_path)


def build_channel_manager(
    config: MochiConfig,
    engine: Any,
    *,
    config_path: str | Path | None = None,
    persist_config_updates: bool = False,
) -> ChannelManager:
    """依設定建立並註冊啟用中的 Discord / Telegram adapter。"""
    import os

    from mochi.channels.discord_adapter import DiscordAdapter
    from mochi.channels.telegram_adapter import TelegramAdapter

    manager = ChannelManager(
        engine=engine,
        config=config,
        config_path=config_path,
        persist_config_updates=persist_config_updates,
    )

    discord_cfg = config.channels.discord
    if discord_cfg.enabled:
        token = (
            discord_cfg.bot_token.get_secret_value()
            if discord_cfg.bot_token is not None
            else os.getenv("DISCORD_BOT_TOKEN")
        )
        if not token:
            raise ChannelConfigurationError(
                "Discord channel is enabled but bot_token/DISCORD_BOT_TOKEN is missing."
            )
        discord_adapter = DiscordAdapter(
                bot_token=token,
                text_enabled=discord_cfg.text_enabled,
                voice_enabled=discord_cfg.voice_enabled,
                allowed_guild_ids=discord_cfg.allowed_guild_ids,
                allowed_channel_ids=discord_cfg.allowed_channel_ids,
                allowed_voice_channel_ids=discord_cfg.allowed_voice_channel_ids,
                allowed_user_ids=discord_cfg.allowed_user_ids,
                admin_user_ids=getattr(discord_cfg, "admin_user_ids", []),
                rate_limit_per_user=discord_cfg.rate_limit_per_user,
                message_mode=discord_cfg.message_mode,
                auto_join_policy=discord_cfg.auto_join_policy,
                voice_auto_reply=discord_cfg.voice_auto_reply,
                voice_stt_enabled=discord_cfg.voice_stt_enabled,
                voice_tts_enabled=discord_cfg.voice_tts_enabled,
                voice_session_mode=getattr(config.voice, "session_mode", "append_current"),
                voice_reply_model_mode=getattr(config.voice, "reply_model_mode", "inherit_active"),
                voice_reply_model_id=getattr(config.voice, "reply_model_id", "") or "",
                voice_reply_tts_voice=getattr(config.voice, "tts_voice", ""),
                voice_sample_rate=config.voice.sample_rate,
            )
        if callable(getattr(engine, "get_or_create_voice_session", None)) or callable(
            getattr(engine, "synthesize_speech", None)
        ):
            discord_adapter.configure_runtime_integrations(
                voice_session_factory=getattr(engine, "get_or_create_voice_session", None),
                reply_synthesizer=getattr(engine, "synthesize_speech", None),
            )
        manager.register(discord_adapter)

    telegram_cfg = config.channels.telegram
    if telegram_cfg.enabled:
        token = (
            telegram_cfg.bot_token.get_secret_value()
            if telegram_cfg.bot_token is not None
            else os.getenv("TELEGRAM_BOT_TOKEN")
        )
        if not token:
            raise ChannelConfigurationError(
                "Telegram channel is enabled but bot_token/TELEGRAM_BOT_TOKEN is missing."
            )
        manager.register(
            TelegramAdapter(
                bot_token=token,
                allowed_chat_ids=telegram_cfg.allowed_chat_ids,
                allowed_user_ids=telegram_cfg.allowed_user_ids,
                rate_limit_per_user=telegram_cfg.rate_limit_per_user,
            )
        )

    return manager
