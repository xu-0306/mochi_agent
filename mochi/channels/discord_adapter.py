# Inspired by hermes-agent/gateway/platforms/discord.py design pattern
"""Discord Bot 適配器。"""

from __future__ import annotations

import asyncio
import audioop
import time
from collections import defaultdict, deque
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Literal

from mochi.channels.base import BaseChannel, SendResult
from mochi.channels.discord_voice_ingress import DiscordVoiceIngress
from mochi.channels.discord_voice_runtime import DiscordVoiceRuntime
from mochi.channels.events import Attachment, AttachmentEvent, CommandEvent, MessageEvent

try:  # pragma: no cover - optional dependency is covered by adapter tests via injection.
    import discord
except ImportError:  # pragma: no cover
    discord = None  # type: ignore[assignment]


class DiscordAdapter(BaseChannel):
    """Discord Bot 頻道適配器（discord.py）。"""

    name = "discord"

    def __init__(
        self,
        bot_token: str,
        *,
        text_enabled: bool = True,
        voice_enabled: bool = False,
        allowed_guild_ids: Sequence[int] | None = None,
        allowed_channel_ids: Sequence[int] | None = None,
        allowed_voice_channel_ids: Sequence[int] | None = None,
        allowed_user_ids: Sequence[int] | None = None,
        rate_limit_per_user: int = 10,
        message_mode: Literal["all_messages", "mentions_only", "slash_only"] = "mentions_only",
        auto_join_policy: Literal["manual_only"] = "manual_only",
        voice_auto_reply: bool = True,
        voice_stt_enabled: bool = True,
        voice_tts_enabled: bool = True,
        voice_runtime: DiscordVoiceRuntime | None = None,
        voice_sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self._bot_token = bot_token
        self._text_enabled = text_enabled
        self._voice_enabled = voice_enabled
        self._allowed_guild_ids = set(allowed_guild_ids or [])
        self._allowed_channel_ids = set(allowed_channel_ids or [])
        self._allowed_voice_channel_ids = set(allowed_voice_channel_ids or [])
        self._allowed_user_ids = set(allowed_user_ids or [])
        self._rate_limit_per_user = max(1, rate_limit_per_user)
        self._message_mode = message_mode
        self._auto_join_policy = auto_join_policy
        self._voice_auto_reply = voice_auto_reply
        self._voice_stt_enabled = voice_stt_enabled
        self._voice_tts_enabled = voice_tts_enabled
        self._recent_user_events: dict[str, deque[float]] = defaultdict(deque)
        self._pending_slash_interactions: dict[str, Any] = {}
        self._client: Any | None = None
        self._task: asyncio.Task[None] | None = None
        self._application_commands_synced = False
        self._voice_sample_rate = voice_sample_rate
        self._voice_runtime = voice_runtime or DiscordVoiceRuntime(
            enabled=voice_enabled,
            stt_enabled=voice_stt_enabled,
            tts_enabled=voice_tts_enabled,
            connect_voice_channel=self._connect_voice_channel,
            play_audio=self._play_audio_to_voice_client,
        )
        self._voice_ingress = DiscordVoiceIngress(
            enabled=voice_enabled and voice_stt_enabled,
            sample_rate=voice_sample_rate,
            on_audio_chunk=self._ingest_voice_chunk_from_transport,
            on_end_turn=self.end_voice_turn,
            on_interrupt_input=self.interrupt_voice_input,
        )

    async def start(self) -> None:
        """啟動 discord.py client。"""
        if discord is None:
            raise RuntimeError(
                "discord.py is not installed. Install channels extra: "
                "`uv sync --extra channels`."
            )
        if self._task is not None and not self._task.done():
            return

        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.dm_messages = True
        intents.voice_states = True

        client = discord.Client(intents=intents)
        self._client = client

        @client.event
        async def on_message(message: Any) -> None:
            await self._handle_message(message)

        command_tree = self._register_application_commands(client)

        @client.event
        async def on_ready() -> None:
            if self._application_commands_synced:
                return
            await command_tree.sync()
            self._application_commands_synced = True

        self._task = asyncio.create_task(client.start(self._bot_token))

    async def stop(self) -> None:
        """停止 discord.py client。"""
        client = self._client
        await self._voice_ingress.detach_all(flush=False)
        if client is not None:
            await client.close()
        if self._task is not None:
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
    ) -> SendResult:
        """傳送 Discord 訊息。"""
        if reply_to:
            interaction = self._pending_slash_interactions.get(reply_to)
            if interaction is not None:
                result = await self._send_interaction_followup(
                    interaction=interaction,
                    text=text,
                    interaction_id=reply_to,
                )
                if result.success:
                    self._pending_slash_interactions.pop(reply_to, None)
                return result

        client = self._client
        if client is None:
            return SendResult(success=False, error="Discord client is not started.")

        try:
            target = await self._resolve_send_target(client, chat_id)
            message_id: str | None = None
            for chunk in _split_message(text, max_length=2000):
                sent = await target.send(chunk)
                message_id = str(getattr(sent, "id", "") or message_id or "")
            return SendResult(success=True, message_id=message_id)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def _send_interaction_followup(
        self,
        *,
        interaction: Any,
        text: str,
        interaction_id: str,
    ) -> SendResult:
        followup = getattr(interaction, "followup", None)
        if followup is None:
            return SendResult(success=False, error="Discord interaction followup is unavailable.")

        try:
            message_id: str | None = None
            for chunk in _split_message(text, max_length=2000):
                sent = await followup.send(chunk)
                message_id = str(getattr(sent, "id", "") or message_id or interaction_id)
            return SendResult(success=True, message_id=message_id)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def _handle_message(self, message: Any) -> None:
        """將 discord.py message 轉成 Mochi ChannelEvent。"""
        author = getattr(message, "author", None)
        if bool(getattr(author, "bot", False)):
            return
        if not self._text_enabled:
            return

        channel = getattr(message, "channel", None)
        channel_id = int(getattr(channel, "id", 0) or 0)
        user_id = int(getattr(author, "id", 0) or 0)
        guild = getattr(message, "guild", None)
        guild_id = int(getattr(guild, "id", 0) or 0)
        if not self._is_allowed(guild_id=guild_id, channel_id=channel_id, user_id=user_id):
            return
        if self._is_rate_limited(str(user_id)):
            return

        content = str(getattr(message, "content", "") or "").strip()
        if not self._should_accept_message(message=message, content=content, guild=guild):
            return
        chat_id = self._chat_id_for_message(message)
        message_id = str(getattr(message, "id", "") or "")
        attachments = _attachments_for_message(message)
        if attachments:
            await self.emit_event(
                AttachmentEvent(
                    channel=self.name,
                    chat_id=chat_id,
                    user_id=str(user_id),
                    text=content,
                    attachments=attachments,
                    message_id=message_id,
                    metadata={"discord_channel_id": channel_id},
                )
            )
            return

        if not content:
            return

        if content.startswith("/"):
            command, args = _parse_command_text(content)
            await self.emit_event(
                CommandEvent(
                    channel=self.name,
                    chat_id=chat_id,
                    user_id=str(user_id),
                    command=command,
                    args=args,
                    message_id=message_id,
                    metadata={
                        "discord_channel_id": channel_id,
                        "discord_guild_id": guild_id,
                    },
                )
            )
            return

        await self.emit_event(
            MessageEvent(
                channel=self.name,
                chat_id=chat_id,
                user_id=str(user_id),
                text=content,
                message_id=message_id,
                metadata={
                    "discord_channel_id": channel_id,
                    "discord_guild_id": guild_id,
                },
            )
        )

    def _is_allowed(self, *, guild_id: int, channel_id: int, user_id: int) -> bool:
        if self._allowed_guild_ids and guild_id and guild_id not in self._allowed_guild_ids:
            return False
        if self._allowed_channel_ids and channel_id not in self._allowed_channel_ids:
            return False
        return not (self._allowed_user_ids and user_id not in self._allowed_user_ids)

    def _should_accept_message(self, *, message: Any, content: str, guild: Any) -> bool:
        if guild is None:
            return True
        if self._message_mode == "all_messages":
            return True
        if self._message_mode == "slash_only":
            return content.startswith("/")
        if content.startswith("/"):
            return True
        bot_user = getattr(self._client, "user", None)
        bot_user_id = getattr(bot_user, "id", None)
        if not isinstance(bot_user_id, int):
            # Tests and early startup can hit this path before client.user is ready.
            return True
        if isinstance(bot_user_id, int):
            raw_mentions = getattr(message, "raw_mentions", None)
            if isinstance(raw_mentions, list) and bot_user_id in raw_mentions:
                return True
            mentions = getattr(message, "mentions", None)
            if isinstance(mentions, list):
                for user in mentions:
                    if int(getattr(user, "id", 0) or 0) == bot_user_id:
                        return True
        return bool(getattr(message, "mention_everyone", False))

    def _is_rate_limited(self, user_id: str) -> bool:
        now = time.monotonic()
        window = self._recent_user_events[user_id]
        while window and now - window[0] >= 60.0:
            window.popleft()
        if len(window) >= self._rate_limit_per_user:
            return True
        window.append(now)
        return False

    def _chat_id_for_message(self, message: Any) -> str:
        channel = getattr(message, "channel", None)
        author = getattr(message, "author", None)
        guild = getattr(message, "guild", None)
        if guild is None:
            return f"dm-{getattr(author, 'id', '')}"
        return f"channel-{getattr(channel, 'id', '')}"

    def _register_application_commands(self, client: Any) -> Any:
        """註冊 Discord native slash command；測試可直接覆蓋 callback factory。"""
        tree = discord.app_commands.CommandTree(client)

        @tree.command(name="ask", description="Ask Mochi a question.")
        async def ask(interaction: Any, question: str = "") -> None:
            await self._handle_slash_command(interaction, "ask", question.split())

        @tree.command(name="help", description="Show Mochi help.")
        async def help_command(interaction: Any) -> None:
            await self._handle_slash_command(interaction, "help", [])

        @tree.command(name="status", description="Show Mochi channel status.")
        async def status_command(interaction: Any) -> None:
            await self._handle_slash_command(interaction, "status", [])

        @tree.command(name="join", description="Ask Mochi to join your current voice channel.")
        async def join_command(interaction: Any) -> None:
            await self._handle_slash_command(interaction, "join", [])

        @tree.command(name="leave", description="Ask Mochi to leave the current voice channel.")
        async def leave_command(interaction: Any) -> None:
            await self._handle_slash_command(interaction, "leave", [])

        client.tree = tree
        return tree

    async def _handle_slash_command(
        self,
        interaction: Any,
        command: str,
        args: Sequence[str],
    ) -> None:
        event = self._command_event_from_interaction(interaction, command, args)
        if event is None:
            return
        await self._defer_interaction_response(interaction)
        interaction_id = str(event.message_id or "")
        if interaction_id:
            self._pending_slash_interactions[interaction_id] = interaction
        await self.emit_event(event)

    async def _defer_interaction_response(self, interaction: Any) -> None:
        response = getattr(interaction, "response", None)
        if response is None:
            return
        is_done = getattr(response, "is_done", None)
        if callable(is_done) and bool(is_done()):
            return
        defer = getattr(response, "defer", None)
        if callable(defer):
            await defer()

    def _command_event_from_interaction(
        self,
        interaction: Any,
        command: str,
        args: Sequence[str],
    ) -> CommandEvent | None:
        user = getattr(interaction, "user", None) or getattr(interaction, "author", None)
        channel = getattr(interaction, "channel", None)
        guild = getattr(interaction, "guild", None)
        voice_state = getattr(getattr(user, "voice", None), "channel", None)
        guild_id = int(getattr(guild, "id", 0) or 0)
        channel_id = int(getattr(channel, "id", 0) or getattr(interaction, "channel_id", 0) or 0)
        voice_channel_id = int(getattr(voice_state, "id", 0) or 0)
        user_id = int(getattr(user, "id", 0) or 0)
        if not self._is_allowed(guild_id=guild_id, channel_id=channel_id, user_id=user_id):
            return None
        if self._is_rate_limited(str(user_id)):
            return None

        return CommandEvent(
            channel=self.name,
            chat_id=self._chat_id_for_interaction(interaction),
            user_id=str(user_id),
            command=command,
            args=list(args),
            message_id=str(getattr(interaction, "id", "") or ""),
            metadata={
                "discord_guild_id": guild_id,
                "discord_channel_id": channel_id,
                "discord_voice_channel_id": voice_channel_id,
                "discord_interaction_id": str(getattr(interaction, "id", "") or ""),
                "native_slash_command": True,
            },
        )

    def _chat_id_for_interaction(self, interaction: Any) -> str:
        channel = getattr(interaction, "channel", None)
        user = getattr(interaction, "user", None) or getattr(interaction, "author", None)
        guild = getattr(interaction, "guild", None)
        if guild is None:
            return f"dm-{getattr(user, 'id', '')}"
        return f"channel-{getattr(channel, 'id', '') or getattr(interaction, 'channel_id', '')}"

    async def _resolve_send_target(self, client: Any, chat_id: str) -> Any:
        if chat_id.startswith("dm-"):
            user_id = int(chat_id.removeprefix("dm-"))
            user = client.get_user(user_id)
            if user is None:
                user = await client.fetch_user(user_id)
            return user

        raw_channel_id = chat_id.removeprefix("channel-")
        channel_id = int(raw_channel_id)
        channel = client.get_channel(channel_id)
        if channel is None:
            channel = await client.fetch_channel(channel_id)
        return channel

    def get_runtime_status(self) -> dict[str, Any]:
        """回傳非敏感 Discord adapter/runtime 狀態。"""
        return {
            "text_enabled": self._text_enabled,
            "voice_enabled": self._voice_enabled,
            "message_mode": self._message_mode,
            "auto_join_policy": self._auto_join_policy,
            "voice_auto_reply": self._voice_auto_reply,
            "voice_stt_enabled": self._voice_stt_enabled,
            "voice_tts_enabled": self._voice_tts_enabled,
            "allowed_guild_ids": sorted(self._allowed_guild_ids),
            "allowed_channel_ids": sorted(self._allowed_channel_ids),
            "allowed_voice_channel_ids": sorted(self._allowed_voice_channel_ids),
            "allowed_user_ids": sorted(self._allowed_user_ids),
            "rate_limit_per_user": self._rate_limit_per_user,
            "application_commands_synced": self._application_commands_synced,
            "pending_interaction_count": len(self._pending_slash_interactions),
            "bot_token_configured": bool(self._bot_token),
            "voice_ingress": self._voice_ingress.get_status(),
            "voice_runtime": self._voice_runtime.get_status(),
        }

    async def join_voice_channel(self, guild_id: int, channel_id: int) -> dict[str, Any]:
        """要求 bot 加入指定語音頻道。"""
        if not self._voice_enabled:
            raise RuntimeError("Discord voice is disabled in current config.")
        if guild_id <= 0:
            raise RuntimeError("Discord guild_id is required for voice join.")
        if channel_id <= 0:
            raise RuntimeError("Discord voice channel_id is required for voice join.")
        if self._allowed_guild_ids and guild_id not in self._allowed_guild_ids:
            raise RuntimeError(f"Guild {guild_id} is not allowed for Discord voice.")
        if self._allowed_voice_channel_ids and channel_id not in self._allowed_voice_channel_ids:
            raise RuntimeError(f"Voice channel {channel_id} is not allowed for Discord voice.")
        room = await self._voice_runtime.join_voice_channel(guild_id, channel_id)
        return room.to_dict()

    async def leave_voice_channel(self, guild_id: int) -> bool:
        """要求 bot 離開指定 guild 的 active room。"""
        if guild_id <= 0:
            raise RuntimeError("Discord guild_id is required for voice leave.")
        await self._voice_ingress.detach(guild_id, flush=False)
        return await self._voice_runtime.leave_voice_channel(guild_id)

    async def interrupt_voice_playback(self, guild_id: int | None = None) -> bool:
        """中斷 Discord 語音播放。"""
        return await self._voice_runtime.interrupt_playback(guild_id)

    async def ingest_voice_audio_chunk(
        self,
        guild_id: int,
        *,
        chunk: bytes,
        speaker_id: str | None = None,
        auto_end: bool = True,
    ) -> dict[str, Any]:
        """將 Discord 語音 chunk 送入 runtime。"""
        return await self._voice_runtime.ingest_audio_chunk(
            guild_id,
            chunk=chunk,
            speaker_id=speaker_id,
            auto_end=auto_end,
        )

    async def end_voice_turn(self, guild_id: int) -> bool:
        """要求 Discord runtime 結束目前 buffered turn。"""
        return await self._voice_runtime.end_listening_turn(guild_id)

    async def interrupt_voice_input(self, guild_id: int) -> int:
        """中斷 Discord runtime 的 buffered input。"""
        return await self._voice_runtime.interrupt_listening(guild_id)

    async def speak_voice_reply(
        self,
        guild_id: int,
        text: str,
        *,
        synthesize_audio: Any,
    ) -> bool:
        """將回覆文字以 TTS 播放到該 guild 的 active voice room。"""
        if not text.strip():
            return False
        return await self._voice_runtime.speak_text(
            guild_id,
            text,
            synthesize_audio=synthesize_audio,
        )

    def is_voice_auto_reply_enabled(self) -> bool:
        """回傳目前是否允許 Discord voice 自動回話。"""
        return self._voice_enabled and self._voice_auto_reply and self._voice_tts_enabled

    def configure_runtime_integrations(
        self,
        *,
        voice_session_factory: Any | None = None,
        reply_synthesizer: Any | None = None,
    ) -> None:
        """注入 Discord voice runtime 所需的 engine 整合點。"""
        self._voice_runtime.configure_integrations(
            voice_session_factory=voice_session_factory,
            reply_synthesizer=reply_synthesizer,
        )

    async def _ingest_voice_chunk_from_transport(
        self,
        guild_id: int,
        chunk: bytes,
        speaker_id: str | None,
    ) -> dict[str, Any]:
        if self._voice_tts_enabled:
            await self.interrupt_voice_playback(guild_id)
        return await self.ingest_voice_audio_chunk(
            guild_id,
            chunk=chunk,
            speaker_id=speaker_id,
            auto_end=False,
        )

    async def _connect_voice_channel(self, guild_id: int, channel_id: int) -> object:
        client = self._client
        if client is None:
            raise RuntimeError("Discord client is not started.")

        channel = client.get_channel(channel_id)
        if channel is None:
            fetch_channel = getattr(client, "fetch_channel", None)
            if callable(fetch_channel):
                channel = await fetch_channel(channel_id)
        if channel is None:
            raise RuntimeError(f"Discord voice channel {channel_id} is not available.")

        guild = getattr(channel, "guild", None)
        existing_voice_client = getattr(guild, "voice_client", None) if guild is not None else None
        if existing_voice_client is not None:
            current_channel = getattr(existing_voice_client, "channel", None)
            current_channel_id = int(getattr(current_channel, "id", 0) or 0)
            if current_channel_id == channel_id:
                await self._try_attach_voice_ingress(guild_id, existing_voice_client)
                return existing_voice_client
            move_to = getattr(existing_voice_client, "move_to", None)
            if callable(move_to):
                await move_to(channel)
                await self._try_attach_voice_ingress(guild_id, existing_voice_client)
                return existing_voice_client

        connect = getattr(channel, "connect", None)
        if not callable(connect):
            raise RuntimeError(
                f"Discord channel {channel_id} does not support voice connect()."
            )
        connect_kwargs: dict[str, Any] = {}
        voice_client_cls = self._voice_ingress.preferred_voice_client_cls()
        if voice_client_cls is not None:
            connect_kwargs["cls"] = voice_client_cls
        voice_client = await connect(**connect_kwargs)
        resolved_guild_id = int(getattr(guild, "id", 0) or 0)
        if resolved_guild_id and resolved_guild_id != guild_id:
            raise RuntimeError(
                f"Resolved voice channel guild mismatch: expected {guild_id}, got {resolved_guild_id}."
            )
        await self._try_attach_voice_ingress(guild_id, voice_client)
        return voice_client

    async def _try_attach_voice_ingress(self, guild_id: int, voice_client: object) -> None:
        if not self._voice_ingress.enabled:
            return
        try:
            await self._voice_ingress.attach(guild_id, voice_client)
        except Exception as exc:
            set_last_error = getattr(self._voice_runtime, "set_last_error", None)
            if callable(set_last_error):
                set_last_error(str(exc))

    async def _play_audio_to_voice_client(self, voice_client: object, audio: bytes) -> None:
        if discord is None:
            raise RuntimeError("discord.py is not installed.")
        if not audio:
            return

        source = _DiscordPCM16AudioSource(audio, sample_rate=self._voice_sample_rate)
        play = getattr(voice_client, "play", None)
        if not callable(play):
            raise RuntimeError("Discord voice client does not provide play().")

        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()

        def _after(error: Exception | None) -> None:
            if done.done():
                return
            if error is None:
                loop.call_soon_threadsafe(done.set_result, None)
            else:
                loop.call_soon_threadsafe(done.set_exception, error)

        play(source, after=_after)
        await done


def _parse_command_text(content: str) -> tuple[str, list[str]]:
    parts = content.removeprefix("/").split()
    if not parts:
        return "", []
    return parts[0], parts[1:]


def _attachments_for_message(message: Any) -> list[Attachment]:
    return [_attachment_from_discord(attachment) for attachment in getattr(message, "attachments", [])]


def _attachment_from_discord(attachment: Any) -> Attachment:
    return Attachment(
        id=str(getattr(attachment, "id", "") or ""),
        filename=str(getattr(attachment, "filename", "") or ""),
        url=str(getattr(attachment, "url", "") or ""),
        content_type=getattr(attachment, "content_type", None),
        size=getattr(attachment, "size", None),
        metadata={"discord_attachment": True},
    )


def _split_message(text: str, *, max_length: int) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + max_length] for index in range(0, len(text), max_length)]


class _DiscordPCM16AudioSource:
    """將 mono PCM16 轉成 discord.py 可播放的 48kHz stereo PCM source。"""

    _FRAME_SIZE_BYTES = 3_840

    def __init__(self, audio: bytes, *, sample_rate: int) -> None:
        normalized = _normalize_pcm16_for_discord(audio, sample_rate=sample_rate)
        self._audio = normalized
        self._offset = 0

    def read(self) -> bytes:
        if self._offset >= len(self._audio):
            return b""
        chunk = self._audio[self._offset : self._offset + self._FRAME_SIZE_BYTES]
        self._offset += self._FRAME_SIZE_BYTES
        if len(chunk) < self._FRAME_SIZE_BYTES:
            chunk += b"\x00" * (self._FRAME_SIZE_BYTES - len(chunk))
        return chunk

    def is_opus(self) -> bool:
        return False

    def cleanup(self) -> None:
        return None


def _normalize_pcm16_for_discord(audio: bytes, *, sample_rate: int) -> bytes:
    """將 mono PCM16 重採樣為 Discord 語音輸出的 48kHz stereo PCM。"""
    if not audio:
        return b""
    resampled = audio
    if sample_rate != 48_000:
        resampled, _ = audioop.ratecv(audio, 2, 1, sample_rate, 48_000, None)
    return audioop.tostereo(resampled, 2, 1, 1)
