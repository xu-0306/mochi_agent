"""Phase 4.5 頻道適配層測試。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mochi.agents.events import FinalAnswerEvent, TextChunkEvent
from mochi.channels.base import BaseChannel, SendResult
from mochi.channels.discord_adapter import DiscordAdapter, _normalize_pcm16_for_discord
from mochi.channels.discord_voice_ingress import DiscordVoiceIngress, _normalize_discord_receive_pcm
from mochi.channels.discord_voice_runtime import DiscordVoiceRuntime
from mochi.channels.events import (
    Attachment,
    AttachmentEvent,
    ChannelEvent,
    CommandEvent,
    MessageEvent,
)
from mochi.channels.manager import ChannelManager, build_channel_manager
from mochi.channels.telegram_adapter import TelegramAdapter
from mochi.config.manager import load_config
from mochi.config.schema import MochiConfig


class FakeEngine:
    """最小 AgentEngine stub。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.voice_session = object()
        self.apply_config_calls: list[tuple[Any, bool]] = []

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
    ) -> AsyncIterator[object]:
        self.calls.append((message, session_id))
        yield TextChunkEvent(content="partial")
        yield FinalAnswerEvent(content=f"reply:{message}")

    async def synthesize_speech(self, text: str) -> bytes:
        return f"tts:{text}".encode()

    async def get_or_create_voice_session(self, session_id: str | None = None) -> object:  # noqa: ARG002
        return self.voice_session

    async def apply_config(self, config: Any, *, reload_voice: bool = False) -> None:
        self.apply_config_calls.append((config, reload_voice))


class FakeChannel(BaseChannel):
    """可觀察 send/start/stop 的 channel stub。"""

    name = "fake"

    def __init__(self) -> None:
        super().__init__()
        self.started = False
        self.stopped = False
        self.sent: list[tuple[str, str, str | None]] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
    ) -> SendResult:
        self.sent.append((chat_id, text, reply_to))
        return SendResult(success=True, message_id="sent-1")


@pytest.mark.asyncio
async def test_channel_manager_routes_message_to_agent_session() -> None:
    engine = FakeEngine()
    channel = FakeChannel()
    manager = ChannelManager(engine=engine)
    manager.register(channel)

    await manager.start_all()
    await manager.handle_event(
        MessageEvent(
            channel="fake",
            chat_id="chat-1",
            user_id="user-1",
            text="hello",
            message_id="msg-1",
        )
    )
    await manager.stop_all()

    assert channel.started is True
    assert channel.stopped is True
    assert engine.calls == [("hello", "fake:chat-1")]
    assert channel.sent == [("chat-1", "reply:hello", "msg-1")]


@pytest.mark.asyncio
async def test_channel_manager_handles_help_command_without_engine_call() -> None:
    engine = FakeEngine()
    channel = FakeChannel()
    manager = ChannelManager(engine=engine)
    manager.register(channel)

    await manager.handle_event(
        CommandEvent(
            channel="fake",
            chat_id="chat-1",
            user_id="user-1",
            command="help",
            message_id="msg-1",
        )
    )

    assert engine.calls == []
    assert channel.sent == [
        (
            "chat-1",
            "Mochi is ready. Send a message and I will reply in this chat.",
            "msg-1",
        )
    ]


@pytest.mark.asyncio
async def test_channel_manager_routes_attachment_metadata_to_agent() -> None:
    engine = FakeEngine()
    channel = FakeChannel()
    manager = ChannelManager(engine=engine)
    manager.register(channel)

    await manager.handle_event(
        AttachmentEvent(
            channel="fake",
            chat_id="chat-1",
            user_id="user-1",
            text="please inspect",
            message_id="msg-1",
            attachments=[
                Attachment(
                    id="file-1",
                    filename="report.pdf",
                    url="https://cdn.example/report.pdf",
                    content_type="application/pdf",
                    size=12345,
                )
            ],
        )
    )

    expected_text = (
        "Caption: please inspect\n"
        "Attachment 1: filename=report.pdf, content_type=application/pdf, "
        "url=https://cdn.example/report.pdf, file_id=file-1, size=12345"
    )
    assert engine.calls == [(expected_text, "fake:chat-1")]
    assert channel.sent == [("chat-1", f"reply:{expected_text}", "msg-1")]


@pytest.mark.asyncio
async def test_channel_manager_routes_unknown_command_to_agent() -> None:
    engine = FakeEngine()
    channel = FakeChannel()
    manager = ChannelManager(engine=engine)
    manager.register(channel)

    await manager.handle_event(
        CommandEvent(
            channel="fake",
            chat_id="chat-1",
            user_id="user-1",
            command="ask",
            args=["hello", "world"],
            message_id="msg-1",
        )
    )

    assert engine.calls == [("ask hello world", "fake:chat-1")]
    assert channel.sent == [("chat-1", "reply:ask hello world", "msg-1")]


@pytest.mark.asyncio
async def test_channel_manager_discord_join_command_uses_voice_runtime() -> None:
    adapter = DiscordAdapter(bot_token="token", voice_enabled=True)
    adapter._voice_runtime = _FakeDiscordVoiceRuntime()
    manager = ChannelManager(engine=FakeEngine())
    manager.register(adapter)

    sent: list[tuple[str, str, str | None]] = []

    async def _send_message(chat_id: str, text: str, reply_to: str | None = None) -> SendResult:
        sent.append((chat_id, text, reply_to))
        return SendResult(success=True, message_id="sent-1")

    adapter.send_message = _send_message  # type: ignore[method-assign]

    await manager.handle_event(
        CommandEvent(
            channel="discord",
            chat_id="channel-100",
            user_id="200",
            command="join",
            message_id="300",
            metadata={
                "discord_guild_id": 400,
                "discord_voice_channel_id": 500,
            },
        )
    )

    assert adapter._voice_runtime.join_calls == [(400, 500)]
    assert sent == [
        (
            "channel-100",
            "Discord voice joined. guild_id=400 channel_id=500",
            "300",
        )
    ]


@pytest.mark.asyncio
async def test_channel_manager_discord_voice_status_formats_active_rooms() -> None:
    adapter = DiscordAdapter(bot_token="token", voice_enabled=True)
    adapter._voice_runtime = _FakeDiscordVoiceRuntime(
        active_rooms=[
            {
                "guild_id": "400",
                "channel_id": "500",
                "session_id": "discord:voice:400:500",
                "playback_state": "playing",
            }
        ]
    )
    manager = ChannelManager(engine=FakeEngine())
    manager.register(adapter)

    sent: list[tuple[str, str, str | None]] = []

    async def _send_message(chat_id: str, text: str, reply_to: str | None = None) -> SendResult:
        sent.append((chat_id, text, reply_to))
        return SendResult(success=True, message_id="sent-1")

    adapter.send_message = _send_message  # type: ignore[method-assign]

    await manager.handle_event(
        CommandEvent(
            channel="discord",
            chat_id="channel-100",
            user_id="200",
            command="voice-status",
            message_id="301",
        )
    )

    assert len(sent) == 1
    assert sent[0][0] == "channel-100"
    assert sent[0][2] == "301"
    assert "active_voice_room_count: 1" in sent[0][1]
    assert "room 1: guild_id=400, channel_id=500, playback_state=playing" in sent[0][1]


@pytest.mark.asyncio
async def test_channel_manager_blocks_discord_mutation_for_non_admin_user() -> None:
    adapter = DiscordAdapter(
        bot_token="token",
        voice_enabled=True,
        admin_user_ids=[999],
    )
    adapter._voice_runtime = _FakeDiscordVoiceRuntime()
    manager = ChannelManager(engine=FakeEngine())
    manager.register(adapter)

    sent: list[tuple[str, str, str | None]] = []

    async def _send_message(chat_id: str, text: str, reply_to: str | None = None) -> SendResult:
        sent.append((chat_id, text, reply_to))
        return SendResult(success=True, message_id="sent-1")

    adapter.send_message = _send_message  # type: ignore[method-assign]

    await manager.handle_event(
        CommandEvent(
            channel="discord",
            chat_id="channel-100",
            user_id="200",
            command="join",
            message_id="302",
            metadata={
                "discord_guild_id": 400,
                "discord_voice_channel_id": 500,
            },
        )
    )

    assert adapter._voice_runtime.join_calls == []
    assert sent == [
        (
            "channel-100",
            "Discord command `/join` is admin-only.",
            "302",
        )
    ]


@pytest.mark.asyncio
async def test_channel_manager_allows_discord_voice_set_for_admin_user() -> None:
    adapter = DiscordAdapter(
        bot_token="token",
        voice_enabled=True,
        admin_user_ids=[200],
    )
    manager = ChannelManager(engine=FakeEngine())
    manager.register(adapter)

    sent: list[tuple[str, str, str | None]] = []

    async def _send_message(chat_id: str, text: str, reply_to: str | None = None) -> SendResult:
        sent.append((chat_id, text, reply_to))
        return SendResult(success=True, message_id="sent-1")

    adapter.send_message = _send_message  # type: ignore[method-assign]

    await manager.handle_event(
        CommandEvent(
            channel="discord",
            chat_id="channel-100",
            user_id="200",
            command="voice-set",
            args=["session_mode", "shared"],
            message_id="303",
        )
    )

    assert adapter.get_voice_conversation_settings()["session_mode"] == "shared"
    assert len(sent) == 1
    assert sent[0][0] == "channel-100"
    assert sent[0][2] == "303"
    assert "Updated Discord voice setting: session_mode=shared" in sent[0][1]


@pytest.mark.asyncio
async def test_channel_manager_persists_shared_voice_updates_to_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "mochi.yaml"
    cfg = MochiConfig.model_validate(
        {
            "channels": {
                "discord": {
                    "enabled": True,
                    "voice_enabled": True,
                    "admin_user_ids": [200],
                }
            }
        }
    )
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord-token")
    engine = FakeEngine()
    manager = build_channel_manager(
        cfg,
        engine=engine,
        config_path=config_path,
        persist_config_updates=True,
    )
    adapter = manager.get("discord")
    assert adapter is not None

    sent: list[tuple[str, str, str | None]] = []

    async def _send_message(chat_id: str, text: str, reply_to: str | None = None) -> SendResult:
        sent.append((chat_id, text, reply_to))
        return SendResult(success=True, message_id="sent-1")

    adapter.send_message = _send_message  # type: ignore[method-assign]

    await manager.handle_event(
        CommandEvent(
            channel="discord",
            chat_id="channel-100",
            user_id="200",
            command="voice-set",
            args=["reply_model_mode", "configured_model"],
            message_id="304",
        )
    )
    await manager.handle_event(
        CommandEvent(
            channel="discord",
            chat_id="channel-100",
            user_id="200",
            command="voice-set",
            args=["reply_model_id", "ollama:qwen2.5"],
            message_id="305",
        )
    )

    saved = load_config(config_path)
    assert sent[-1][1] == "Updated Discord voice setting: reply_model_id=ollama:qwen2.5"
    assert saved.voice.reply_model_mode == "configured_model"
    assert saved.voice.reply_model_id == "ollama:qwen2.5"
    assert engine.apply_config_calls[-1][1] is True


@pytest.mark.asyncio
async def test_channel_manager_sends_discord_voice_reply_for_active_room() -> None:
    engine = FakeEngine()
    adapter = DiscordAdapter(
        bot_token="token",
        voice_enabled=True,
        voice_auto_reply=True,
        voice_tts_enabled=True,
    )
    adapter._voice_runtime = _FakeDiscordVoiceRuntime()
    manager = ChannelManager(engine=engine)
    manager.register(adapter)

    text_sent: list[tuple[str, str, str | None]] = []

    async def _send_message(chat_id: str, text: str, reply_to: str | None = None) -> SendResult:
        text_sent.append((chat_id, text, reply_to))
        return SendResult(success=True, message_id="sent-1")

    adapter.send_message = _send_message  # type: ignore[method-assign]

    await manager.handle_event(
        MessageEvent(
            channel="discord",
            chat_id="channel-100",
            user_id="200",
            text="hello",
            message_id="300",
            metadata={"discord_guild_id": 400},
        )
    )

    assert text_sent == [("channel-100", "reply:hello", "300")]
    assert adapter._voice_runtime.speak_calls == [
        (400, "reply:hello", b"tts:reply:hello")
    ]


@pytest.mark.asyncio
async def test_discord_adapter_emits_message_event() -> None:
    adapter = DiscordAdapter(
        bot_token="token",
        allowed_channel_ids=[100],
        allowed_user_ids=[200],
        rate_limit_per_user=2,
    )
    events: list[ChannelEvent] = []
    adapter.set_event_handler(lambda event: _append_event(events, event))

    await adapter._handle_message(
        _DiscordMessage(
            content="hi",
            message_id=300,
            channel_id=100,
            author_id=200,
            guild_id=400,
        )
    )

    assert events == [
        MessageEvent(
            channel="discord",
            chat_id="channel-100",
            user_id="200",
            text="hi",
            message_id="300",
            metadata={"discord_channel_id": 100, "discord_guild_id": 400},
        )
    ]


@pytest.mark.asyncio
async def test_discord_adapter_filters_disallowed_user() -> None:
    adapter = DiscordAdapter(bot_token="token", allowed_user_ids=[200])
    events: list[ChannelEvent] = []
    adapter.set_event_handler(lambda event: _append_event(events, event))

    await adapter._handle_message(
        _DiscordMessage(
            content="hi",
            message_id=300,
            channel_id=100,
            author_id=999,
            guild_id=400,
        )
    )

    assert events == []


@pytest.mark.asyncio
async def test_discord_adapter_emits_attachment_event() -> None:
    adapter = DiscordAdapter(
        bot_token="token",
        allowed_channel_ids=[100],
        allowed_user_ids=[200],
        rate_limit_per_user=2,
    )
    events: list[ChannelEvent] = []
    adapter.set_event_handler(lambda event: _append_event(events, event))

    await adapter._handle_message(
        _DiscordMessage(
            content="caption",
            message_id=300,
            channel_id=100,
            author_id=200,
            guild_id=400,
            attachments=[
                _DiscordAttachment(
                    id=500,
                    filename="report.txt",
                    url="https://cdn.example/report.txt",
                    content_type="text/plain",
                    size=42,
                )
            ],
        )
    )

    assert events == [
        AttachmentEvent(
            channel="discord",
            chat_id="channel-100",
            user_id="200",
            text="caption",
            message_id="300",
            attachments=[
                Attachment(
                    id="500",
                    filename="report.txt",
                    url="https://cdn.example/report.txt",
                    content_type="text/plain",
                    size=42,
                    metadata={"discord_attachment": True},
                )
            ],
            metadata={"discord_channel_id": 100},
        )
    ]


@pytest.mark.asyncio
async def test_discord_adapter_command_parsing_still_emits_command_event() -> None:
    adapter = DiscordAdapter(bot_token="token")
    events: list[ChannelEvent] = []
    adapter.set_event_handler(lambda event: _append_event(events, event))

    await adapter._handle_message(
        _DiscordMessage(
            content="/ask hello world",
            message_id=300,
            channel_id=100,
            author_id=200,
            guild_id=400,
        )
    )

    assert events == [
        CommandEvent(
            channel="discord",
            chat_id="channel-100",
            user_id="200",
            command="ask",
            args=["hello", "world"],
            message_id="300",
            metadata={"discord_channel_id": 100, "discord_guild_id": 400},
        )
    ]


def test_discord_native_slash_interaction_helper_builds_command_event() -> None:
    adapter = DiscordAdapter(
        bot_token="token",
        allowed_channel_ids=[100],
        allowed_user_ids=[200],
    )

    event = adapter._command_event_from_interaction(
        _DiscordInteraction(
            interaction_id=300,
            channel_id=100,
            user_id=200,
            guild_id=400,
        ),
        "ask",
        ["hello", "world"],
    )

    assert event == CommandEvent(
        channel="discord",
        chat_id="channel-100",
        user_id="200",
        command="ask",
        args=["hello", "world"],
        message_id="300",
        metadata={
            "discord_guild_id": 400,
            "discord_channel_id": 100,
            "discord_voice_channel_id": 0,
            "discord_interaction_id": "300",
            "native_slash_command": True,
        },
    )


@pytest.mark.asyncio
async def test_discord_native_slash_handler_emits_with_deferring() -> None:
    adapter = DiscordAdapter(bot_token="token")
    events: list[ChannelEvent] = []
    adapter.set_event_handler(lambda event: _append_event(events, event))
    interaction = _DiscordInteraction(
        interaction_id=300,
        channel_id=100,
        user_id=200,
        guild_id=400,
    )

    await adapter._handle_slash_command(interaction, "ask", ["hello"])

    assert interaction.response.deferred is True
    assert events == [
        CommandEvent(
            channel="discord",
            chat_id="channel-100",
            user_id="200",
            command="ask",
            args=["hello"],
            message_id="300",
            metadata={
                "discord_guild_id": 400,
                "discord_channel_id": 100,
                "discord_voice_channel_id": 0,
                "discord_interaction_id": "300",
                "native_slash_command": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_discord_adapter_send_message_uses_interaction_followup_when_reply_to_matches() -> None:
    adapter = DiscordAdapter(bot_token="token")
    interaction = _DiscordInteraction(
        interaction_id=300,
        channel_id=100,
        user_id=200,
        guild_id=400,
    )

    await adapter._handle_slash_command(interaction, "ask", ["hello"])
    result = await adapter.send_message("channel-100", "slash-reply", reply_to="300")

    assert result.success is True
    assert result.message_id == "7001"
    assert interaction.followup.sent == ["slash-reply"]
    assert "300" not in adapter._pending_slash_interactions


@pytest.mark.asyncio
async def test_discord_adapter_send_message_regular_target_path_unchanged() -> None:
    adapter = DiscordAdapter(bot_token="token")
    client = _DiscordClient(
        channels={
            100: _DiscordSendTarget(message_ids=[9001, 9002]),
        }
    )
    adapter._client = client

    result = await adapter.send_message("channel-100", "hello")

    assert result.success is True
    assert result.message_id == "9001"
    assert client.channels[100].sent == ["hello"]


@pytest.mark.asyncio
async def test_discord_adapter_join_voice_channel_uses_runtime_and_allowlist() -> None:
    adapter = DiscordAdapter(
        bot_token="token",
        voice_enabled=True,
        allowed_guild_ids=[400],
        allowed_voice_channel_ids=[500],
    )
    adapter._voice_runtime = _FakeDiscordVoiceRuntime()

    payload = await adapter.join_voice_channel(400, 500)

    assert payload["guild_id"] == "400"
    assert payload["channel_id"] == "500"
    assert adapter._voice_runtime.join_calls == [(400, 500)]


@pytest.mark.asyncio
async def test_discord_adapter_join_voice_channel_rejects_disallowed_voice_channel() -> None:
    adapter = DiscordAdapter(
        bot_token="token",
        voice_enabled=True,
        allowed_voice_channel_ids=[500],
    )
    adapter._voice_runtime = _FakeDiscordVoiceRuntime()

    with pytest.raises(RuntimeError, match="Voice channel 999 is not allowed"):
        await adapter.join_voice_channel(400, 999)


@pytest.mark.asyncio
async def test_discord_adapter_ingest_voice_audio_chunk_delegates_to_runtime() -> None:
    adapter = DiscordAdapter(
        bot_token="token",
        voice_enabled=True,
        voice_stt_enabled=True,
    )
    adapter._voice_runtime = _FakeDiscordVoiceRuntime()

    payload = await adapter.ingest_voice_audio_chunk(
        400,
        chunk=b"hello",
        speaker_id="user-1",
        auto_end=False,
    )

    assert payload == {
        "accepted": True,
        "endpoint": False,
        "buffered_audio_bytes": 5,
        "transcriptions": [],
    }
    assert adapter._voice_runtime.ingest_calls == [
        (400, b"hello", "user-1", False)
    ]


@pytest.mark.asyncio
async def test_discord_adapter_end_voice_turn_delegates_to_runtime() -> None:
    adapter = DiscordAdapter(
        bot_token="token",
        voice_enabled=True,
        voice_stt_enabled=True,
    )
    adapter._voice_runtime = _FakeDiscordVoiceRuntime()

    ended = await adapter.end_voice_turn(400)

    assert ended is True
    assert adapter._voice_runtime.end_turn_calls == [400]


@pytest.mark.asyncio
async def test_discord_adapter_interrupt_voice_input_delegates_to_runtime() -> None:
    adapter = DiscordAdapter(
        bot_token="token",
        voice_enabled=True,
        voice_stt_enabled=True,
    )
    adapter._voice_runtime = _FakeDiscordVoiceRuntime()

    cleared = await adapter.interrupt_voice_input(400)

    assert cleared == 5
    assert adapter._voice_runtime.interrupt_input_calls == [400]


@pytest.mark.asyncio
async def test_discord_adapter_connect_voice_channel_uses_voice_recv_client_and_attaches_ingress() -> None:
    recv_module = _FakeVoiceRecvModule()
    adapter = DiscordAdapter(
        bot_token="token",
        voice_enabled=True,
        voice_stt_enabled=True,
    )
    adapter._voice_ingress = DiscordVoiceIngress(
        enabled=True,
        sample_rate=16_000,
        on_audio_chunk=adapter._ingest_voice_chunk_from_transport,
        on_end_turn=adapter.end_voice_turn,
        on_interrupt_input=adapter.interrupt_voice_input,
        recv_module=recv_module,
    )
    voice_client = _FakeRecvVoiceClient()
    guild = _DiscordVoiceGuild(400, voice_client=voice_client)
    channel = _DiscordVoiceChannel(500, guild=guild, voice_client=voice_client)
    adapter._client = _DiscordVoiceClient(channels={500: channel})

    resolved = await adapter._connect_voice_channel(400, 500)

    assert resolved is voice_client
    assert channel.connect_calls == [{"cls": recv_module.VoiceRecvClient}]
    assert voice_client.listen_calls
    assert adapter.get_runtime_status()["voice_ingress"]["active_guild_ids"] == [400]


@pytest.mark.asyncio
async def test_discord_adapter_leave_voice_channel_detaches_ingress_before_runtime_leave() -> None:
    adapter = DiscordAdapter(
        bot_token="token",
        voice_enabled=True,
        voice_stt_enabled=True,
    )
    adapter._voice_runtime = _FakeDiscordVoiceRuntime()
    adapter._voice_ingress = _FakeDiscordVoiceIngress()

    left = await adapter.leave_voice_channel(400)

    assert left is True
    assert adapter._voice_ingress.detach_calls == [(400, False)]
    assert adapter._voice_runtime.leave_calls == [400]


@pytest.mark.asyncio
async def test_discord_adapter_stop_detaches_all_voice_ingress_sessions() -> None:
    adapter = DiscordAdapter(bot_token="token", voice_enabled=True, voice_stt_enabled=True)
    adapter._voice_ingress = _FakeDiscordVoiceIngress(active_guild_ids=[400, 401])
    adapter._client = _DiscordCloseClient()

    await adapter.stop()

    assert adapter._voice_ingress.detach_all_calls == [False]
    assert adapter._client.closed is True


def test_discord_pcm_normalizer_outputs_48khz_stereo_frame_bytes() -> None:
    mono_16k = b"\x01\x00\x02\x00\x03\x00\x04\x00"

    normalized = _normalize_pcm16_for_discord(mono_16k, sample_rate=16_000)

    assert len(normalized) > len(mono_16k)
    assert len(normalized) % 4 == 0


def test_discord_receive_pcm_normalizer_outputs_16khz_mono() -> None:
    stereo_48k = (
        b"\x01\x00\x03\x00"
        b"\x02\x00\x04\x00"
        b"\x05\x00\x07\x00"
        b"\x06\x00\x08\x00"
    )

    normalized = _normalize_discord_receive_pcm(stereo_48k, sample_rate=16_000)

    assert normalized
    assert len(normalized) % 2 == 0
    assert len(normalized) < len(stereo_48k)


@pytest.mark.asyncio
async def test_discord_voice_ingress_sink_forwards_chunks_and_flushes_turn_on_close() -> None:
    ingested: list[tuple[int, bytes, str | None]] = []
    ended: list[int] = []
    recv_module = _FakeVoiceRecvModule()
    ingress = DiscordVoiceIngress(
        enabled=True,
        sample_rate=16_000,
        inactivity_timeout_ms=10,
        on_audio_chunk=_record_ingest(ingested),
        on_end_turn=_record_end_turn(ended),
        recv_module=recv_module,
    )
    voice_client = _FakeRecvVoiceClient()

    attached = await ingress.attach(400, voice_client)
    assert attached is True

    sink = voice_client.sink
    assert sink is not None
    sink.write(_DiscordMember(200), _FakeVoiceData(pcm48=b"\x01\x00\x02\x00" * 480))
    sink.close(flush=True)
    await asyncio.sleep(0.05)

    assert ingested
    assert ingested[0][0] == 400
    assert ingested[0][2] == "200"
    assert ended == [400]


@pytest.mark.asyncio
async def test_telegram_adapter_emits_command_event() -> None:
    adapter = TelegramAdapter(
        bot_token="token",
        allowed_chat_ids=[100],
        allowed_user_ids=[200],
        rate_limit_per_user=2,
    )
    events: list[ChannelEvent] = []
    adapter.set_event_handler(lambda event: _append_event(events, event))

    await adapter._handle_update(
        _TelegramUpdate(
            text="/ask@mochi_bot hello world",
            message_id=300,
            chat_id=100,
            user_id=200,
        ),
        context=None,
    )

    assert events == [
        CommandEvent(
            channel="telegram",
            chat_id="100",
            user_id="200",
            command="ask",
            args=["hello", "world"],
            message_id="300",
            metadata={"telegram_chat_id": 100},
        )
    ]


@pytest.mark.asyncio
async def test_telegram_adapter_filters_disallowed_user() -> None:
    adapter = TelegramAdapter(bot_token="token", allowed_user_ids=[200])
    events: list[ChannelEvent] = []
    adapter.set_event_handler(lambda event: _append_event(events, event))

    await adapter._handle_update(
        _TelegramUpdate(
            text="hi",
            message_id=300,
            chat_id=100,
            user_id=999,
        ),
        context=None,
    )

    assert events == []


@pytest.mark.asyncio
async def test_telegram_adapter_emits_document_attachment_event() -> None:
    adapter = TelegramAdapter(bot_token="token", rate_limit_per_user=2)
    events: list[ChannelEvent] = []
    adapter.set_event_handler(lambda event: _append_event(events, event))

    await adapter._handle_update(
        _TelegramUpdate(
            text="",
            caption="please inspect",
            message_id=300,
            chat_id=100,
            user_id=200,
            document=_TelegramFile(
                file_id="doc-file-id",
                file_name="report.pdf",
                mime_type="application/pdf",
                file_size=12345,
            ),
        ),
        context=None,
    )

    assert events == [
        AttachmentEvent(
            channel="telegram",
            chat_id="100",
            user_id="200",
            message_id="300",
            text="please inspect",
            attachments=[
                Attachment(
                    id="doc-file-id",
                    filename="report.pdf",
                    content_type="application/pdf",
                    size=12345,
                    metadata={"telegram_attachment_type": "document"},
                )
            ],
            metadata={"telegram_chat_id": 100},
        )
    ]


@pytest.mark.asyncio
async def test_telegram_adapter_emits_photo_attachment_event() -> None:
    adapter = TelegramAdapter(bot_token="token", rate_limit_per_user=2)
    events: list[ChannelEvent] = []
    adapter.set_event_handler(lambda event: _append_event(events, event))

    await adapter._handle_update(
        _TelegramUpdate(
            text="",
            message_id=300,
            chat_id=100,
            user_id=200,
            photo=[
                _TelegramFile(file_id="small", width=90, height=90, file_size=100),
                _TelegramFile(file_id="large", width=900, height=900, file_size=1000),
            ],
        ),
        context=None,
    )

    assert events == [
        AttachmentEvent(
            channel="telegram",
            chat_id="100",
            user_id="200",
            message_id="300",
            attachments=[
                Attachment(
                    id="large",
                    size=1000,
                    metadata={
                        "telegram_attachment_type": "photo",
                        "width": 900,
                        "height": 900,
                    },
                )
            ],
            metadata={"telegram_chat_id": 100},
        )
    ]


@pytest.mark.asyncio
async def test_telegram_adapter_emits_voice_attachment_event() -> None:
    adapter = TelegramAdapter(bot_token="token", rate_limit_per_user=2)
    events: list[ChannelEvent] = []
    adapter.set_event_handler(lambda event: _append_event(events, event))

    await adapter._handle_update(
        _TelegramUpdate(
            text="",
            message_id=300,
            chat_id=100,
            user_id=200,
            voice=_TelegramFile(
                file_id="voice-file-id",
                mime_type="audio/ogg",
                file_size=2048,
                duration=7,
            ),
        ),
        context=None,
    )

    assert events == [
        AttachmentEvent(
            channel="telegram",
            chat_id="100",
            user_id="200",
            message_id="300",
            attachments=[
                Attachment(
                    id="voice-file-id",
                    content_type="audio/ogg",
                    size=2048,
                    metadata={"telegram_attachment_type": "voice", "duration": 7},
                )
            ],
            metadata={"telegram_chat_id": 100},
        )
    ]


def test_build_channel_manager_uses_environment_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MochiConfig.model_validate(
        {
            "channels": {
                "discord": {"enabled": True},
                "telegram": {"enabled": True},
            }
        }
    )
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-token")

    manager = build_channel_manager(cfg, engine=FakeEngine())

    assert manager.list_channels() == ["discord", "telegram"]


def test_build_channel_manager_injects_voice_runtime_integrations(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MochiConfig.model_validate(
        {
            "channels": {
                "discord": {
                    "enabled": True,
                    "voice_enabled": True,
                    "voice_stt_enabled": True,
                    "voice_tts_enabled": True,
                }
            }
        }
    )
    engine = FakeEngine()
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord-token")

    manager = build_channel_manager(cfg, engine=engine)
    adapter = manager.get("discord")

    assert adapter is not None
    runtime = getattr(adapter, "_voice_runtime")
    assert runtime._voice_session_factory is not None
    assert runtime._reply_synthesizer is not None


async def _append_event(events: list[ChannelEvent], event: ChannelEvent) -> None:
    events.append(event)


@dataclass
class _FakeAuthor:
    id: int
    bot: bool = False


@dataclass
class _FakeChannel:
    id: int


@dataclass
class _FakeGuild:
    id: int


class _DiscordMessage:
    def __init__(
        self,
        *,
        content: str,
        message_id: int,
        channel_id: int,
        author_id: int,
        guild_id: int | None,
        attachments: list[_DiscordAttachment] | None = None,
    ) -> None:
        self.content = content
        self.id = message_id
        self.channel = _FakeChannel(channel_id)
        self.author = _FakeAuthor(author_id)
        self.guild = _FakeGuild(guild_id) if guild_id is not None else None
        self.attachments = attachments or []


@dataclass
class _DiscordAttachment:
    id: int
    filename: str
    url: str
    content_type: str | None
    size: int


class _DiscordInteraction:
    def __init__(
        self,
        *,
        interaction_id: int,
        channel_id: int,
        user_id: int,
        guild_id: int | None,
    ) -> None:
        self.id = interaction_id
        self.channel = _FakeChannel(channel_id)
        self.channel_id = channel_id
        self.user = _FakeAuthor(user_id)
        self.guild = _FakeGuild(guild_id) if guild_id is not None else None
        self.response = _DiscordInteractionResponse()
        self.followup = _DiscordFollowup()


class _DiscordInteractionResponse:
    def __init__(self) -> None:
        self.deferred = False

    def is_done(self) -> bool:
        return self.deferred

    async def defer(self) -> None:
        self.deferred = True


class _DiscordFollowup:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._next_id = 7001

    async def send(self, text: str) -> Any:
        self.sent.append(text)
        message_id = self._next_id
        self._next_id += 1
        return _DiscordSentMessage(message_id=message_id)


@dataclass
class _DiscordSentMessage:
    message_id: int

    @property
    def id(self) -> int:
        return self.message_id


class _DiscordSendTarget:
    def __init__(self, *, message_ids: list[int]) -> None:
        self.sent: list[str] = []
        self._message_ids = list(message_ids)
        self._index = 0

    async def send(self, text: str) -> _DiscordSentMessage:
        self.sent.append(text)
        if self._index >= len(self._message_ids):
            message_id = self._message_ids[-1]
        else:
            message_id = self._message_ids[self._index]
        self._index += 1
        return _DiscordSentMessage(message_id=message_id)


class _DiscordClient:
    def __init__(self, *, channels: dict[int, _DiscordSendTarget]) -> None:
        self.channels = channels

    def get_channel(self, channel_id: int) -> _DiscordSendTarget | None:
        return self.channels.get(channel_id)

    async def fetch_channel(self, channel_id: int) -> _DiscordSendTarget | None:
        return self.channels.get(channel_id)

    def get_user(self, user_id: int) -> None:
        return None

    async def fetch_user(self, user_id: int) -> None:
        return None


class _DiscordCloseClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _DiscordVoiceClient:
    def __init__(self, *, channels: dict[int, Any]) -> None:
        self.channels = channels

    def get_channel(self, channel_id: int) -> Any | None:
        return self.channels.get(channel_id)

    async def fetch_channel(self, channel_id: int) -> Any | None:
        return self.channels.get(channel_id)


class _DiscordVoiceGuild:
    def __init__(self, guild_id: int, *, voice_client: object | None = None) -> None:
        self.id = guild_id
        self.voice_client = voice_client


class _DiscordVoiceChannel:
    def __init__(self, channel_id: int, *, guild: _DiscordVoiceGuild, voice_client: object) -> None:
        self.id = channel_id
        self.guild = guild
        self._voice_client = voice_client
        self.connect_calls: list[dict[str, Any]] = []

    async def connect(self, **kwargs: Any) -> object:
        self.connect_calls.append(dict(kwargs))
        return self._voice_client


class _FakeVoiceRecvClient:
    pass


class _FakeVoiceRecvModule:
    VoiceRecvClient = _FakeVoiceRecvClient

    class AudioSink:
        def __init__(self) -> None:
            self.voice_client: object | None = None

        def wants_opus(self) -> bool:
            return False

        def write(self, user: Any, data: Any) -> None:
            return None

        def cleanup(self) -> None:
            return None


class _FakeRecvVoiceClient:
    def __init__(self) -> None:
        self.listen_calls: list[tuple[object, Any]] = []
        self.stop_listening_calls = 0
        self._is_listening = False
        self.sink: Any | None = None

    def listen(self, sink: Any, *, after: Any = None) -> None:
        self.listen_calls.append((sink, after))
        self.sink = sink
        self._is_listening = True

    def stop_listening(self) -> None:
        self.stop_listening_calls += 1
        self._is_listening = False

    def is_listening(self) -> bool:
        return self._is_listening


class _FakeDiscordVoiceIngress:
    def __init__(self, *, active_guild_ids: list[int] | None = None) -> None:
        self.detach_calls: list[tuple[int, bool]] = []
        self.detach_all_calls: list[bool] = []
        self._active_guild_ids = active_guild_ids or []

    @property
    def enabled(self) -> bool:
        return True

    def preferred_voice_client_cls(self) -> type[object] | None:
        return _FakeVoiceRecvClient

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "extension_available": True,
            "active_guild_ids": list(self._active_guild_ids),
            "last_error": None,
            "sample_rate": 16_000,
            "inactivity_timeout_ms": 900,
        }

    async def detach(self, guild_id: int, *, flush: bool = False) -> bool:
        self.detach_calls.append((guild_id, flush))
        return True

    async def detach_all(self, *, flush: bool = False) -> int:
        self.detach_all_calls.append(flush)
        return len(self._active_guild_ids)


@dataclass
class _DiscordMember:
    id: int


@dataclass
class _FakeVoiceData:
    pcm48: bytes

    @property
    def pcm(self) -> bytes:
        return self.pcm48


def _record_ingest(store: list[tuple[int, bytes, str | None]]) -> Any:
    async def _handler(guild_id: int, chunk: bytes, speaker_id: str | None) -> dict[str, Any]:
        store.append((guild_id, chunk, speaker_id))
        return {"accepted": True, "endpoint": False, "buffered_audio_bytes": len(chunk)}

    return _handler


def _record_end_turn(store: list[int]) -> Any:
    async def _handler(guild_id: int) -> bool:
        store.append(guild_id)
        return True

    return _handler


class _FakeDiscordVoiceRuntime:
    def __init__(self, *, active_rooms: list[dict[str, Any]] | None = None) -> None:
        self.join_calls: list[tuple[int, int]] = []
        self.leave_calls: list[int] = []
        self.interrupt_calls: list[int | None] = []
        self.end_turn_calls: list[int] = []
        self.interrupt_input_calls: list[int] = []
        self.speak_calls: list[tuple[int, str, bytes]] = []
        self.ingest_calls: list[tuple[int, bytes, str | None, bool]] = []
        self.active_rooms = active_rooms or []

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "phase": "d2_voice_output",
            "active_voice_room_count": len(self.active_rooms),
            "active_voice_rooms": list(self.active_rooms),
            "last_error": None,
        }

    async def join_voice_channel(self, guild_id: int, channel_id: int) -> Any:
        self.join_calls.append((guild_id, channel_id))
        return _FakeRoomState(guild_id=guild_id, channel_id=channel_id)

    async def leave_voice_channel(self, guild_id: int) -> bool:
        self.leave_calls.append(guild_id)
        return True

    async def interrupt_playback(self, guild_id: int | None = None) -> bool:
        self.interrupt_calls.append(guild_id)
        return True

    async def speak_text(
        self,
        guild_id: int,
        text: str,
        *,
        synthesize_audio: Any,
    ) -> bool:
        audio = await synthesize_audio(text)
        self.speak_calls.append((guild_id, text, audio))
        return True

    async def ingest_audio_chunk(
        self,
        guild_id: int,
        *,
        chunk: bytes,
        speaker_id: str | None = None,
        auto_end: bool = True,
    ) -> dict[str, Any]:
        self.ingest_calls.append((guild_id, chunk, speaker_id, auto_end))
        return {
            "accepted": True,
            "endpoint": False,
            "buffered_audio_bytes": len(chunk),
            "transcriptions": [],
        }

    async def end_listening_turn(self, guild_id: int) -> bool:
        self.end_turn_calls.append(guild_id)
        return True

    async def interrupt_listening(self, guild_id: int) -> int:
        self.interrupt_input_calls.append(guild_id)
        return 5


@dataclass
class _FakeRoomState:
    guild_id: int
    channel_id: int

    def to_dict(self) -> dict[str, str]:
        return {
            "guild_id": str(self.guild_id),
            "channel_id": str(self.channel_id),
        }


@dataclass
class _TelegramFile:
    file_id: str
    file_name: str = ""
    mime_type: str | None = None
    file_size: int | None = None
    file_unique_id: str | None = None
    width: int | None = None
    height: int | None = None
    duration: int | None = None


class _TelegramMessage:
    def __init__(
        self,
        *,
        text: str,
        message_id: int,
        caption: str = "",
        document: _TelegramFile | None = None,
        photo: list[_TelegramFile] | None = None,
        voice: _TelegramFile | None = None,
        audio: _TelegramFile | None = None,
        video: _TelegramFile | None = None,
    ) -> None:
        self.text = text
        self.message_id = message_id
        self.caption = caption
        self.document = document
        self.photo = photo or []
        self.voice = voice
        self.audio = audio
        self.video = video


@dataclass
class _TelegramChat:
    id: int


@dataclass
class _TelegramUser:
    id: int


class _TelegramUpdate:
    def __init__(
        self,
        *,
        text: str,
        message_id: int,
        chat_id: int,
        user_id: int,
        caption: str = "",
        document: _TelegramFile | None = None,
        photo: list[_TelegramFile] | None = None,
        voice: _TelegramFile | None = None,
        audio: _TelegramFile | None = None,
        video: _TelegramFile | None = None,
    ) -> None:
        self.effective_message = _TelegramMessage(
            text=text,
            message_id=message_id,
            caption=caption,
            document=document,
            photo=photo,
            voice=voice,
            audio=audio,
            video=video,
        )
        self.effective_chat = _TelegramChat(id=chat_id)
        self.effective_user = _TelegramUser(id=user_id)
