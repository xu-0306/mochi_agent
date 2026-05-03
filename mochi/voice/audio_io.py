"""音訊 I/O 抽象（單輪 bounded 流程）。"""

from __future__ import annotations

import asyncio
import logging
import wave
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BaseAudioIO(ABC):
    """音訊輸入輸出抽象介面。"""

    @abstractmethod
    async def record_once(
        self,
        *,
        sample_rate: int,
        channels: int,
        max_seconds: float,
    ) -> bytes:
        """一次性錄音並回傳 PCM16 bytes。"""
        ...

    @abstractmethod
    async def play_once(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        channels: int,
    ) -> None:
        """一次性播放 PCM16 bytes。"""
        ...

    async def record_stream(
        self,
        *,
        sample_rate: int,
        channels: int,
        chunk_seconds: float,
        max_seconds: float,
    ) -> AsyncIterator[bytes]:
        """以分塊方式連續錄音（預設以多次 `record_once` 組成）。"""
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be > 0.")
        if max_seconds <= 0:
            raise ValueError("max_seconds must be > 0.")

        remaining = max_seconds
        while remaining > 0:
            duration = min(chunk_seconds, remaining)
            chunk = await self.record_once(
                sample_rate=sample_rate,
                channels=channels,
                max_seconds=duration,
            )
            if chunk:
                yield chunk
            remaining -= duration

    async def stop_playback(self) -> bool:
        """嘗試停止目前播放；不支援時回傳 False。"""
        return False

    @asynccontextmanager
    async def playback_session(
        self,
        *,
        sample_rate: int,
        channels: int,
    ) -> AsyncIterator[Callable[[bytes], Any]]:
        """建立可重複寫入的播放 session；預設退回逐次 `play_once()`。"""

        async def _play(audio: bytes) -> None:
            if not audio:
                return
            await self.play_once(audio, sample_rate=sample_rate, channels=channels)

        yield _play


class UnavailableAudioIO(BaseAudioIO):
    """音訊後端不可用時的 fallback。"""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    async def record_once(
        self,
        *,
        sample_rate: int,  # noqa: ARG002
        channels: int,  # noqa: ARG002
        max_seconds: float,  # noqa: ARG002
    ) -> bytes:
        raise RuntimeError(f"Audio I/O unavailable: {self._reason}")

    async def play_once(
        self,
        audio: bytes,  # noqa: ARG002
        *,
        sample_rate: int,  # noqa: ARG002
        channels: int,  # noqa: ARG002
    ) -> None:
        raise RuntimeError(f"Audio I/O unavailable: {self._reason}")


class SoundDeviceAudioIO(BaseAudioIO):
    """以 sounddevice 實作的一次錄音/播放。"""

    def __init__(
        self,
        *,
        recorder: Callable[..., Any] | None = None,
        player: Callable[..., Any] | None = None,
        input_stream_factory: Callable[..., Any] | None = None,
        output_stream_factory: Callable[..., Any] | None = None,
        stopper: Callable[[], Any] | None = None,
        waiter: Callable[[], Any] | None = None,
    ) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("sounddevice import failed; install `mochi[voice]`.") from exc

        self._recorder = recorder or sd.rec
        self._player = player or sd.play
        self._input_stream_factory = input_stream_factory or getattr(sd, "InputStream", None)
        self._output_stream_factory = output_stream_factory or getattr(sd, "OutputStream", None)
        self._stopper = stopper or getattr(sd, "stop", None)
        self._waiter = waiter or sd.wait
        self._check_input_settings = getattr(sd, "check_input_settings", None)
        self._check_output_settings = getattr(sd, "check_output_settings", None)
        self._query_devices = getattr(sd, "query_devices", None)
        self._portaudio_version_getter = getattr(sd, "get_portaudio_version", None)
        self._default = getattr(sd, "default", None)
        self._runtime_diagnostics = self._collect_device_diagnostics()
        self._runtime_diagnostics["last_input_settings_supported"] = None
        self._runtime_diagnostics["last_output_settings_supported"] = None
        self._runtime_diagnostics["last_input_settings_error"] = None
        self._runtime_diagnostics["last_output_settings_error"] = None
        self._runtime_diagnostics["input_overflow_events"] = 0
        self._runtime_diagnostics["output_underflow_events"] = 0
        self._runtime_diagnostics["last_runtime_error"] = None

    def get_runtime_diagnostics(self) -> dict[str, Any]:
        """回傳目前音訊後端裝置與執行時診斷資訊。"""
        return dict(self._runtime_diagnostics)

    async def record_once(
        self,
        *,
        sample_rate: int,
        channels: int,
        max_seconds: float,
    ) -> bytes:
        """一次錄音，回傳 mono PCM16 bytes。"""
        if max_seconds <= 0:
            raise ValueError("max_seconds must be > 0.")

        self._probe_stream_settings(
            is_input=True,
            sample_rate=sample_rate,
            channels=channels,
        )
        try:
            return await asyncio.to_thread(
                self._record_blocking,
                sample_rate,
                channels,
                max_seconds,
            )
        except Exception as exc:
            self._remember_runtime_error("record_once", exc)
            raise RuntimeError(
                f"Audio input record_once failed: {exc}. diagnostics={self.get_runtime_diagnostics()}"
            ) from exc

    async def play_once(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        channels: int,
    ) -> None:
        """一次播放 PCM16 bytes。"""
        if not audio:
            return

        self._probe_stream_settings(
            is_input=False,
            sample_rate=sample_rate,
            channels=channels,
        )
        try:
            underflowed = await asyncio.to_thread(
                self._play_blocking,
                audio,
                sample_rate,
                channels,
            )
        except Exception as exc:
            self._remember_runtime_error("play_once", exc)
            raise RuntimeError(
                f"Audio output play_once failed: {exc}. diagnostics={self.get_runtime_diagnostics()}"
            ) from exc
        if underflowed:
            self._runtime_diagnostics["output_underflow_events"] += 1
            logger.warning("sounddevice OutputStream.write() reported underflow")

    async def record_stream(
        self,
        *,
        sample_rate: int,
        channels: int,
        chunk_seconds: float,
        max_seconds: float,
    ) -> AsyncIterator[bytes]:
        """使用明確 InputStream 連續錄音，避免 sounddevice 全域 rec/play 互相干擾。"""
        if self._input_stream_factory is None:
            async for chunk in super().record_stream(
                sample_rate=sample_rate,
                channels=channels,
                chunk_seconds=chunk_seconds,
                max_seconds=max_seconds,
            ):
                yield chunk
            return
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be > 0.")
        if max_seconds <= 0:
            raise ValueError("max_seconds must be > 0.")

        self._probe_stream_settings(
            is_input=True,
            sample_rate=sample_rate,
            channels=channels,
        )
        frames_per_chunk = max(1, int(sample_rate * chunk_seconds))
        remaining_frames = max(1, int(sample_rate * max_seconds))
        stream = self._input_stream_factory(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
        )
        entered = False
        try:
            await asyncio.to_thread(stream.__enter__)
            entered = True
            while remaining_frames > 0:
                frames = min(frames_per_chunk, remaining_frames)
                chunk, overflowed = await asyncio.to_thread(self._read_stream_chunk, stream, frames)
                if overflowed:
                    self._runtime_diagnostics["input_overflow_events"] += 1
                    logger.warning("sounddevice InputStream.read() reported overflow")
                if chunk:
                    yield chunk
                remaining_frames -= frames
        except Exception as exc:
            self._remember_runtime_error("record_stream", exc)
            raise RuntimeError(
                f"Audio input record_stream failed: {exc}. diagnostics={self.get_runtime_diagnostics()}"
            ) from exc
        finally:
            await asyncio.to_thread(self._close_stream_safely, stream, entered)

    async def stop_playback(self) -> bool:
        """停止目前播放。"""
        if not callable(self._stopper):
            return False
        try:
            await asyncio.to_thread(self._stopper)
        except Exception as exc:
            self._remember_runtime_error("stop_playback", exc)
            return False
        return True

    @asynccontextmanager
    async def playback_session(
        self,
        *,
        sample_rate: int,
        channels: int,
    ) -> AsyncIterator[Callable[[bytes], Any]]:
        """建立長生命週期播放 stream，避免每個 chunk 反覆開關 OutputStream。"""
        self._probe_stream_settings(
            is_input=False,
            sample_rate=sample_rate,
            channels=channels,
        )
        if self._output_stream_factory is None:
            async with super().playback_session(
                sample_rate=sample_rate,
                channels=channels,
            ) as play:
                yield play
            return

        stream = self._output_stream_factory(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
        )
        entered = False
        try:
            await asyncio.to_thread(stream.__enter__)
            entered = True

            async def _play(audio: bytes) -> None:
                if not audio:
                    return
                try:
                    underflowed = await asyncio.to_thread(
                        self._write_stream_chunk,
                        stream,
                        audio,
                        channels,
                    )
                except Exception as exc:
                    self._remember_runtime_error("playback_session.write", exc)
                    raise RuntimeError(
                        "Audio output playback_session write failed: "
                        f"{exc}. diagnostics={self.get_runtime_diagnostics()}"
                    ) from exc
                if underflowed:
                    self._runtime_diagnostics["output_underflow_events"] += 1
                    logger.warning("sounddevice OutputStream.write() reported underflow")

            yield _play
        finally:
            await asyncio.to_thread(self._close_stream_safely, stream, entered)

    def _record_blocking(self, sample_rate: int, channels: int, max_seconds: float) -> bytes:
        import numpy as np

        frames = max(1, int(sample_rate * max_seconds))
        raw = self._recorder(
            frames,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
        )
        self._waiter()

        audio = np.asarray(raw, dtype=np.float32)
        audio = audio.mean(axis=1) if audio.ndim == 2 and audio.shape[1] > 1 else audio.reshape(-1)
        return self._float_audio_to_pcm16(audio)

    def _read_stream_chunk(self, stream: Any, frames: int) -> tuple[bytes, bool]:
        raw = stream.read(frames)
        overflowed = False
        if isinstance(raw, tuple):
            overflowed = bool(raw[1]) if len(raw) > 1 else False
            raw = raw[0]
        import numpy as np

        audio = np.asarray(raw, dtype=np.float32)
        audio = audio.mean(axis=1) if audio.ndim == 2 and audio.shape[1] > 1 else audio.reshape(-1)
        return self._float_audio_to_pcm16(audio), overflowed

    @staticmethod
    def _float_audio_to_pcm16(audio: Any) -> bytes:
        import numpy as np

        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    def _play_blocking(self, audio: bytes, sample_rate: int, channels: int) -> bool:
        import numpy as np

        wave = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32767.0
        if channels > 1:
            wave = np.repeat(wave[:, None], channels, axis=1)
        if self._output_stream_factory is not None:
            output_wave = wave.reshape(-1, channels) if channels == 1 else wave
            with self._output_stream_factory(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
            ) as stream:
                write_result = stream.write(output_wave)
            return bool(write_result) if isinstance(write_result, bool) else False
        self._player(wave, samplerate=sample_rate)
        self._waiter()
        return False

    def _write_stream_chunk(self, stream: Any, audio: bytes, channels: int) -> bool:
        import numpy as np

        wave = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32767.0
        output_wave = wave.reshape(-1, channels) if channels == 1 else np.repeat(
            wave[:, None],
            channels,
            axis=1,
        )
        write_result = stream.write(output_wave)
        return bool(write_result) if isinstance(write_result, bool) else False

    def _collect_device_diagnostics(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "backend": "sounddevice",
            "uses_input_stream": self._input_stream_factory is not None,
            "uses_output_stream": self._output_stream_factory is not None,
            "persistent_output_stream_supported": self._output_stream_factory is not None,
            "default_input_device": None,
            "default_output_device": None,
            "default_samplerate": None,
            "device_count": None,
            "portaudio_version": None,
        }
        default_device = getattr(self._default, "device", None)
        if isinstance(default_device, tuple | list):
            if len(default_device) >= 2:
                diagnostics["default_input_device"] = default_device[0]
                diagnostics["default_output_device"] = default_device[1]
        else:
            diagnostics["default_input_device"] = default_device
            diagnostics["default_output_device"] = default_device
        diagnostics["default_samplerate"] = getattr(self._default, "samplerate", None)

        if callable(self._query_devices):
            try:
                devices = self._query_devices()
                diagnostics["device_count"] = len(devices) if hasattr(devices, "__len__") else None
            except Exception as exc:
                diagnostics["device_count"] = f"error: {type(exc).__name__}: {exc}"

        if callable(self._portaudio_version_getter):
            try:
                diagnostics["portaudio_version"] = self._portaudio_version_getter()
            except Exception as exc:
                diagnostics["portaudio_version"] = f"error: {type(exc).__name__}: {exc}"
        return diagnostics

    def _probe_stream_settings(self, *, is_input: bool, sample_rate: int, channels: int) -> None:
        checker = self._check_input_settings if is_input else self._check_output_settings
        support_key = "last_input_settings_supported" if is_input else "last_output_settings_supported"
        error_key = "last_input_settings_error" if is_input else "last_output_settings_error"
        if not callable(checker):
            self._runtime_diagnostics[support_key] = None
            self._runtime_diagnostics[error_key] = None
            return
        try:
            checker(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
            )
            self._runtime_diagnostics[support_key] = True
            self._runtime_diagnostics[error_key] = None
        except Exception as exc:
            self._runtime_diagnostics[support_key] = False
            self._runtime_diagnostics[error_key] = f"{type(exc).__name__}: {exc}"

    def _remember_runtime_error(self, operation: str, exc: Exception) -> None:
        self._runtime_diagnostics["last_runtime_error"] = f"{operation}: {type(exc).__name__}: {exc}"

    def _close_stream_safely(self, stream: Any, entered: bool) -> None:
        if entered:
            with suppress(Exception):
                stream.__exit__(None, None, None)
                return
        stop_fn = getattr(stream, "stop", None)
        close_fn = getattr(stream, "close", None)
        abort_fn = getattr(stream, "abort", None)
        if callable(stop_fn):
            with suppress(Exception):
                stop_fn()
        elif callable(abort_fn):
            with suppress(Exception):
                abort_fn()
        if callable(close_fn):
            with suppress(Exception):
                close_fn()


def create_default_audio_io() -> BaseAudioIO:
    """建立預設 Audio I/O；若不可用則回傳 fallback。"""
    try:
        return SoundDeviceAudioIO()
    except Exception as exc:  # pragma: no cover
        return UnavailableAudioIO(str(exc))


def read_audio_file_as_pcm16(
    path: str | Path,
    *,
    sample_rate: int,
) -> bytes:
    """讀取音訊檔並回傳 PCM16 bytes（支援 .wav 與 raw PCM16）。"""
    file_path = Path(path)
    if file_path.suffix.lower() != ".wav":
        return file_path.read_bytes()

    with wave.open(str(file_path), "rb") as wav_file:
        wav_sample_rate = wav_file.getframerate()
        wav_channels = wav_file.getnchannels()
        wav_sample_width = wav_file.getsampwidth()
        if wav_sample_width != 2:
            raise ValueError("WAV sample width must be 16-bit PCM.")
        if wav_sample_rate != sample_rate:
            raise ValueError(
                f"WAV sample rate mismatch: expected {sample_rate}, got {wav_sample_rate}."
            )
        if wav_channels != 1:
            raise ValueError(f"WAV channels must be mono (1); got {wav_channels}.")
        return wav_file.readframes(wav_file.getnframes())


def write_audio_file_from_pcm16(
    path: str | Path,
    audio: bytes,
    *,
    sample_rate: int,
) -> None:
    """將 PCM16 bytes 寫入檔案（.wav 輸出 WAV，其他副檔名維持 raw PCM16）。"""
    file_path = Path(path)
    if file_path.suffix.lower() != ".wav":
        file_path.write_bytes(audio)
        return

    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)
