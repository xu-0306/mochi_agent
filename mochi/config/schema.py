"""Mochi 完整 Pydantic v2 設定 Schema。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from mochi.backends.inference_capabilities import ReasoningEffort
from mochi.config import defaults
from mochi.config.defaults import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_FALLBACK_CHAIN,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_MODEL_SETUP_MODE,
    DEFAULT_MODEL_SETUP_REQUIRED,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_REGION_PROFILE,
    DEFAULT_RESPONSE_LANGUAGE,
    DEFAULT_TIMEZONE,
    DEFAULT_TTS_VOICE,
    DEFAULT_UI_LOCALE,
    DEFAULT_UI_LOCALE_FALLBACK,
)
from mochi.security.policy import autonomy_mode_defaults, infer_autonomy_mode
from mochi.tools.web_search_providers import normalize_web_search_provider

# ---------------------------------------------------------------------------
# 子設定段
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _empty_list_typed(item_type: type[T]) -> list[T]:
    _ = item_type
    return []


class OllamaConfig(BaseModel):
    """Ollama 後端設定。"""

    base_url: str = DEFAULT_OLLAMA_BASE_URL
    """Ollama 服務地址。"""

    timeout: float = 120.0
    """HTTP 請求逾時秒數。"""


class GGUFConfig(BaseModel):
    """GGUF backend settings."""

    n_ctx: int = 4096
    """上下文視窗大小。"""

    n_gpu_layers: int = -1
    """GPU 層數，-1 表示全部卸載到 GPU。"""

    n_threads: int | None = None
    """CPU 執行緒數，None 表示自動。"""


    n_batch: int = 512
    """Prompt/eval batch size，較大通常更能吃滿 GPU。"""

    n_ubatch: int = 512
    """Micro-batch size，控制單次提交給後端的批次大小。"""

    n_threads_batch: int | None = None
    """Batch 評估階段使用的 CPU 執行緒數。"""

    flash_attn: bool = False
    """是否啟用 flash attention。"""

    offload_kqv: bool = True
    """是否將 K/Q/V 與 KV cache 相關計算盡量卸載到 GPU。"""

    use_mmap: bool = True
    """是否使用 mmap 載入模型。"""

    use_mlock: bool = False
    """是否鎖定模型頁面避免被換出。"""


class HuggingFaceConfig(BaseModel):
    """HuggingFace Safetensors 後端設定。"""

    device: str = "auto"
    """推理設備（auto / cpu / cuda）。"""

    torch_dtype: str = "auto"
    """Torch 資料類型（auto / float16 / bfloat16）。"""

    trust_remote_code: bool = False
    """是否信任遠端程式碼。"""


class LocalModelConfig(BaseModel):
    """本地模型掃描與 runtime 設定。"""

    roots: list[Path] = Field(default_factory=lambda: _empty_list_typed(Path))
    """允許 WebGUI 掃描的根目錄；空列表表示不限制。"""

    scan_max_depth: int = Field(default=3, ge=0, le=32)
    """單次掃描允許的最大目錄深度。"""

    scan_max_entries: int = Field(default=500, ge=1, le=100_000)
    """單次掃描最多檢查的 filesystem entry 數。"""

    runtime: Literal["inprocess", "worker"] = "inprocess"
    """本地模型 runtime 模式。"""

    idle_unload_enabled: bool = False
    """是否啟用本地模型閒置超時卸載。"""

    idle_unload_seconds: int | None = Field(default=300, ge=0, le=86_400)
    """本地模型閒置多久後自動卸載；`0` 或 `None` 表示停用。"""

    llama_cpp: "LlamaCppRuntimeConfig" = Field(default_factory=lambda: LlamaCppRuntimeConfig())
    """llama.cpp 轉換 runtime 的保守 managed/registered metadata。"""

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_idle_unload_defaults(cls, value: Any) -> Any:
        """相容舊設定：若缺少新布林欄位，依 idle_unload_seconds 推導。"""
        if not isinstance(value, dict):
            return value

        if "idle_unload_enabled" in value:
            return value

        raw_seconds = value.get("idle_unload_seconds")
        if isinstance(raw_seconds, int):
            value = dict(value)
            value["idle_unload_enabled"] = raw_seconds > 0
            return value
        if raw_seconds is None:
            value = dict(value)
            value["idle_unload_enabled"] = False
        return value


class LlamaCppRuntimeConfig(BaseModel):
    """llama.cpp 轉換工具鏈設定。"""

    source: Literal["managed", "existing_path"] | None = None
    """若有保存 runtime 來源，僅限 managed install 或既有路徑註冊。"""

    root_dir: Path | None = None
    """已保存的 llama.cpp 根目錄；可指向 managed install 或既有路徑。"""

    python_executable: str | None = None
    """可選覆蓋 llama.cpp convert script 使用的 Python。"""

    version: str | None = None
    """可選保存的 llama.cpp 版本/來源 tag。"""


class OpenAICompatConfig(BaseModel):
    """OpenAI-compatible 遠端 API 設定。"""

    base_url: str = "https://api.openai.com/v1"
    """OpenAI-compatible API base URL。"""

    model: str = "gpt-4o-mini"
    """預設遠端模型名稱。"""

    api_key: SecretStr | None = None
    """遠端 API key（敏感資料）。"""

    provider: Literal["openai_compat", "gemini", "anthropic", "vllm"] = "openai_compat"
    """UI/provider preset；底層目前皆走 OpenAI-compatible protocol。"""

    timeout: float = 120.0
    """HTTP 請求逾時秒數。"""


class OpenAICodexConfig(BaseModel):
    """OpenAI Codex OAuth-backed transport settings."""

    base_url: str = "https://chatgpt.com/backend-api"
    """OpenAI Codex backend base URL."""

    model: str = "gpt-5.4"
    """OpenAI Codex model identifier."""

    auth_profile_id: str | None = None
    """External auth profile id stored outside config.yaml."""

    timeout: float = 120.0
    """HTTP request timeout."""


class VLLMConfig(BaseModel):
    """Managed vLLM runtime 閮剖?"""

    enabled: bool = False
    """?臬? managed vLLM runtime。"""

    host: str = "127.0.0.1"
    """vLLM API host。"""

    port: int | None = None
    """vLLM API port；`None` 甇?? route/runtime 層分配。"""

    api_key: SecretStr | None = None
    """vLLM API key。"""

    tensor_parallel_size: int = 1
    """vLLM tensor parallel size。"""

    dtype: str = "auto"
    """vLLM dtype。"""

    gpu_memory_utilization: float = Field(default=0.9, ge=0.0, le=1.0)
    """GPU 憭扳? ratio嚗?.0~1.0嚗?"""

    max_model_len: int | None = None
    """vLLM max model length。"""

    trust_remote_code: bool = False
    """?臬 trust remote code。"""

    quantization: str | None = None
    """vLLM quantization preset。"""

    startup_timeout_seconds: int = 180
    """managed vLLM ?臭?逾時秒數。"""

    cuda_visible_devices: str | None = None
    """?阡? runtime ??`CUDA_VISIBLE_DEVICES`。"""


class LocaleDefaultsConfig(BaseModel):
    """跨地區部署與首次啟動的語言/時區預設。"""

    region_profile: str = DEFAULT_REGION_PROFILE
    """地區預設檔名稱；`global` 表示不偏特定地區。"""

    ui_locale: str = DEFAULT_UI_LOCALE
    """UI 偏好語言；`auto` 表示由瀏覽器或客戶端偵測。"""

    ui_locale_fallback: str = DEFAULT_UI_LOCALE_FALLBACK
    """無法偵測 UI locale 時使用的 fallback。"""

    response_language: str = DEFAULT_RESPONSE_LANGUAGE
    """回覆語言偏好；`same_as_user` 表示跟隨使用者輸入語言。"""

    default_tts_voice: str = DEFAULT_TTS_VOICE
    """首次啟動或未設定 voice 時建議使用的 TTS voice。"""

    timezone: str = DEFAULT_TIMEZONE
    """顯示層時區偏好；`auto` 表示由客戶端或系統偵測。"""


class ConfiguredModelConfig(BaseModel):
    """WebGUI 可選模型清單中的非敏感模型設定。"""

    id: str = Field(min_length=1)
    """模型清單項目的穩定識別碼；可由 `/v1/models/switch` 使用。"""

    provider: Literal["ollama", "openai_compat", "openai_codex", "gemini", "anthropic", "vllm", "local"]
    """模型供應商 preset。"""

    model: str = Field(min_length=1)
    """供應商內部模型名稱。"""

    model_spec: str = Field(min_length=1)
    """後端模型規格；Ollama 為 `ollama:<model>`，遠端相容 API 為 base URL。"""

    base_url: str | None = None
    """模型服務 base URL；不包含任何 API key。"""

    label: str | None = None
    """UI 顯示名稱。"""

    backend_type: str | None = None
    """摨惜 backend family??"""

    launch_mode: Literal["external", "managed"] | None = None
    """底層 backend family。"""


    auth_profile_id: str | None = None
    """External auth profile id used by OAuth-backed providers."""

    auth_mode: Literal["none", "api_key", "oauth"] | None = None
    """Auth mode metadata for the configured model entry."""


class ModelSetupConfig(BaseModel):
    """模型首次啟動與 provider-aware setup metadata。"""

    mode: Literal["configured_or_setup"] = DEFAULT_MODEL_SETUP_MODE
    """啟動模式；保留既有 model，但允許 UI 在不可用時進入 setup。"""

    default_provider: str = DEFAULT_MODEL_PROVIDER
    """開發預設 provider；不是唯一支援的部署選項。"""

    default_model: str = DEFAULT_MODEL_NAME
    """預設 provider 下的模型名稱。"""

    default_model_spec: str = DEFAULT_MODEL
    """完整預設模型規格，保留向後相容。"""

    setup_required: bool = DEFAULT_MODEL_SETUP_REQUIRED
    """首次啟動時應檢查模型是否可用，不可用則引導設定 provider。"""

    fallback_chain: list[str] = Field(default_factory=lambda: list(DEFAULT_MODEL_FALLBACK_CHAIN))
    """建議 setup 順序：使用者設定 → 本機探測 → 遠端 OpenAI-compatible provider。"""

    configured_models: list[ConfiguredModelConfig] = Field(
        default_factory=lambda: _empty_list_typed(ConfiguredModelConfig)
    )
    """WebGUI 已成功設定過、可在對話下拉選擇的非敏感模型清單。"""


class RegisteredTTSVoiceConfig(BaseModel):
    """已註冊的自訂 TTS 聲音包設定。"""

    id: str = Field(min_length=1)
    backend: Literal["coqui-tts", "kokoro-tts", "openai-tts", "piper"] | None = None
    path: Path
    label: str | None = None
    source: Literal["registered_path", "upload"] = "registered_path"


class VoiceConfig(BaseModel):
    """語音管線設定（STT + TTS + VAD + Audio）。"""

    enabled: bool = False
    """是否啟用語音功能。"""

    # STT 設定（Phase 4）
    stt_backend: Literal[
        "auto",
        "faster-whisper",
        "openai-api",
        "external-api",
        "openai-whisper",
        "qwen-asr",
        "vosk",
        "whisper-cpp",
        "whisperlivekit",
    ] = "faster-whisper"
    """STT 後端選擇。"""

    stt_model: str = "medium"
    """STT 模型名稱、repo id 或本地路徑別名。"""

    stt_language: str = "auto"
    """辨識語言（auto/en/zh/ja/...）。"""

    stt_device: str = "auto"
    """STT 推理設備（auto/cpu/cuda）。"""

    stt_model_cache_dir: Path = Path.home() / ".cache" / "mochi" / "models"
    """Whisper 模型快取目錄。"""

    stt_model_path: Path | None = None
    """可選本地 STT 模型路徑（優先於 stt_model 名稱）。"""

    stt_openai_base_url: str | None = None
    """`openai-api` STT 使用的 API base URL（可含 `/v1`）。"""

    stt_openai_api_key: SecretStr | None = None
    """`openai-api` STT 使用的 Bearer token。"""

    stt_openai_timeout: float = 60.0
    """`openai-api` STT HTTP 請求逾時秒數。"""

    # TTS 設定（Phase 4）
    tts_backend: Literal[
        "auto",
        "coqui-tts",
        "edge-tts",
        "external-api",
        "kokoro-tts",
        "openai-tts",
        "piper",
    ] = "kokoro-tts"
    """TTS 後端選擇。"""

    tts_model: str | None = None
    """TTS 模型名稱；僅特定後端使用。"""

    tts_voice: str = DEFAULT_TTS_VOICE
    """TTS 聲音名稱。"""

    tts_language: str | None = None
    """TTS 語言或 speaker language；僅特定後端使用。"""

    tts_speed: float = 1.0
    """語速倍率（0.5–2.0）。"""

    tts_use_gpu: bool = False
    """是否允許 TTS 後端使用 GPU。"""

    tts_kokoro_lang_code: str = "a"
    """Kokoro `KPipeline` 語言代碼。"""

    tts_split_pattern: str = r"\n+"
    """Kokoro/句段切分正則。"""

    tts_openai_base_url: str | None = None
    """`openai-tts` 使用的 API base URL（可含 `/v1`）。"""

    tts_openai_api_key: SecretStr | None = None
    """`openai-tts` 使用的 Bearer token。"""

    tts_openai_timeout: float = 60.0
    """`openai-tts` HTTP 請求逾時秒數。"""

    tts_openai_response_format: Literal["pcm", "wav"] = "pcm"
    """`openai-tts` 回應音訊格式。"""

    reply_model_mode: Literal["inherit_active", "configured_model"] = "inherit_active"
    """語音回覆使用中的回答模型模式。"""

    reply_model_id: str | None = None
    """當 `reply_model_mode=configured_model` 時使用的 configured model id。"""

    session_mode: Literal["append_current", "isolated_voice"] = "append_current"
    """語音對話是附加到目前 chat session，或使用隔離的 voice session。"""

    voice_pack_dir: Path = Path.home() / ".cache" / "mochi" / "voice-packs"
    """瀏覽器上傳的自訂語音包儲存目錄。"""

    registered_tts_voices: list[RegisteredTTSVoiceConfig] = Field(
        default_factory=lambda: _empty_list_typed(RegisteredTTSVoiceConfig)
    )
    """共用的自訂 TTS 聲音包註冊表。"""

    # VAD 設定
    vad_threshold: float = 0.5
    """語音偵測靈敏度（0.0–1.0）。"""

    vad_min_speech_ms: int = 250
    """最短語音片段（毫秒）。"""

    vad_max_silence_ms: int = 700
    """最長靜音後結束（毫秒）。"""

    # Audio I/O
    sample_rate: int = 16000
    """音訊取樣率（Hz）。"""

    channels: int = 1
    """`/v1/voice` 音訊輸入聲道數；目前僅支援 mono。"""

    @field_validator("channels")
    @classmethod
    def validate_channels(cls, value: int) -> int:
        """明確限制 `/v1/voice` 只接受 mono 聲道設定。"""
        if value != 1:
            raise ValueError(
                "voice.channels must be 1 because /v1/voice only accepts mono "
                "PCM16 input (base64-encoded s16le)."
            )
        return value

    @property
    def voice_input_channel_policy(self) -> dict[str, Any]:
        """回傳 `/v1/voice` 的聲道限制說明。"""
        return {
            "mode": "mono-only",
            "configured_channels": self.channels,
            "supported_channels": [1],
            "validation_message": (
                "voice.channels must be 1 because /v1/voice only accepts mono "
                "PCM16 input (base64-encoded s16le)."
            ),
        }

    @property
    def voice_input_contract(self) -> dict[str, Any]:
        """回傳 `/v1/voice` websocket 音訊輸入契約。"""
        return {
            "transport": "websocket",
            "message_type": "audio_chunk",
            "message_field": "data",
            "payload_encoding": "base64",
            "encoding": "pcm16",
            "sample_format": "s16le",
            "endianness": "little",
            "channels": self.channels,
            "channel_layout": "mono",
            "sample_rate_hz": self.sample_rate,
            "pcm_input": True,
        }


class DiscordPlatformConfig(BaseModel):
    """Discord Bot 設定。"""

    enabled: bool = False
    """是否啟用 Discord Bot。"""

    text_enabled: bool = True
    """是否啟用 Discord 文字/指令路徑。"""

    voice_enabled: bool = False
    """是否啟用 Discord 語音 runtime。"""

    bot_token: SecretStr | None = None
    """Discord Bot Token（敏感資料）。"""

    allowed_guild_ids: list[int] = Field(default_factory=lambda: _empty_list_typed(int))
    """允許回應的 guild ID 白名單，空列表表示全部允許。"""

    allowed_channel_ids: list[int] = Field(default_factory=lambda: _empty_list_typed(int))
    """允許回應的頻道 ID 白名單，空列表表示全部允許。"""

    allowed_voice_channel_ids: list[int] = Field(default_factory=lambda: _empty_list_typed(int))
    """允許加入的語音頻道 ID 白名單，空列表表示全部允許。"""

    allowed_user_ids: list[int] = Field(default_factory=lambda: _empty_list_typed(int))
    """允許使用的使用者 ID 白名單，空列表表示全部允許。"""

    admin_user_ids: list[int] = Field(default_factory=lambda: _empty_list_typed(int))
    """允許變更 Discord 共用設定的管理者 user id 清單。"""

    rate_limit_per_user: int = 10
    """每使用者 60 秒內最多幾則訊息。"""

    message_mode: Literal["all_messages", "mentions_only", "slash_only"] = "mentions_only"
    """Guild 文字頻道的收訊策略。"""

    auto_join_policy: Literal["manual_only"] = "manual_only"
    """語音頻道加入策略；v1 僅支援手動 join。"""

    voice_auto_reply: bool = True
    """語音頻道是否在完成 STT/推理後自動回覆。"""

    voice_stt_enabled: bool = True
    """是否允許 Discord voice runtime 啟用 STT ingress。"""

    voice_tts_enabled: bool = True
    """是否允許 Discord voice runtime 啟用 TTS playback。"""


class TelegramPlatformConfig(BaseModel):
    """Telegram Bot 設定。"""

    enabled: bool = False
    """是否啟用 Telegram Bot。"""

    bot_token: SecretStr | None = None
    """Telegram Bot Token（敏感資料）。"""

    allowed_chat_ids: list[int] = Field(default_factory=lambda: _empty_list_typed(int))
    """允許回應的聊天室 ID 白名單，空列表表示全部允許。"""

    allowed_user_ids: list[int] = Field(default_factory=lambda: _empty_list_typed(int))
    """允許使用的使用者 ID 白名單，空列表表示全部允許。"""

    rate_limit_per_user: int = 10
    """每使用者 60 秒內最多幾則訊息。"""


class ChannelsConfig(BaseModel):
    """頻道適配層設定。"""

    discord: DiscordPlatformConfig = Field(default_factory=DiscordPlatformConfig)
    """Discord Bot 設定。"""

    telegram: TelegramPlatformConfig = Field(default_factory=TelegramPlatformConfig)
    """Telegram Bot 設定。"""


class LearningConfig(BaseModel):
    """持續學習系統設定。"""

    enabled: bool = True
    """是否啟用學習系統。"""

    auto_extract_skills: bool = True
    """成功任務完成後是否自動萃取技能。"""

    auto_sync_filesystem_skills: bool = True
    """是否自動同步 skills_dir 底下的 SKILL.md 檔案。"""

    min_steps_for_extraction: int = 3
    """少於此步驟數的任務不進行技能萃取。"""

    min_tool_calls_for_extraction: int = 2
    """少於此工具呼叫次數的任務不進行技能萃取。"""

    trajectory_retention_days: int = 30
    """軌跡記錄保留天數。"""

    skill_improvement_threshold: float = 0.7
    """相似度超過此值時觸發技能改進（0.0–1.0）。"""

    max_skills: int = 500
    """技能庫上限。"""


class MemoryConfig(BaseModel):
    """記憶系統設定。"""

    db_path: Path = Field(default_factory=defaults.default_memory_db_path)
    """SQLite 資料庫路徑。"""

    max_short_term_messages: int = 50

    semantic_compaction_enabled: bool = True

    semantic_summary_mode: Literal["deterministic", "hybrid"] = "hybrid"

    max_short_term_tokens: int | None = Field(default=6000, ge=256, le=1_000_000)

    semantic_keep_recent_messages: int = Field(default=8, ge=2, le=200)
    """短期記憶最大訊息數量。"""

    fts_top_k: int = 5
    """FTS5 搜尋返回最大結果數。"""


class InferencePreset(BaseModel):
    """推理參數 preset。"""

    name: str = Field(min_length=1, max_length=64)
    """Preset 名稱。"""

    system_prompt: str = ""
    """Preset 的系統提示詞。"""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    """採樣溫度（0.0–2.0）。"""

    max_tokens: int = Field(default=4096, ge=1, le=131072)
    """最大輸出 token 數。"""

    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    """Top-p 取樣機率。"""

    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    """Min-p 取樣機率。"""

    top_k: int = Field(default=0, ge=0)
    """Top-k 取樣數量；0 代表停用。"""

    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    """Frequency penalty。"""

    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    """Presence penalty。"""

    repeat_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    reasoning_effort: ReasoningEffort | None = None
    """Repeat penalty。"""


class SecurityConfig(BaseModel):
    """安全設定。"""

    autonomy_mode: Literal["trusted_workspace", "strict", "high_autonomy", "auto_review"] = "trusted_workspace"
    """Autonomy preset mapped to runtime approval behavior."""

    require_approval_for_shell: bool = True
    """執行 Shell 命令前是否需要使用者確認。"""

    require_approval_for_file_write: bool = False
    """寫入檔案前是否需要使用者確認。"""

    require_approval_for_exec: bool = True
    """Exec runtime 命令是否預設需要顯式審批。"""

    agent_run_default_max_wall_clock_sec: int | None = Field(default=None, ge=1, le=86_400)
    """Default agent-run wall-clock guard. `None` disables the default deadline."""

    agent_run_default_heartbeat_timeout_sec: int | None = Field(default=None, ge=1, le=86_400)
    """Default subagent heartbeat timeout. `None` disables the stall watchdog."""

    agent_run_default_checkpoint_interval_steps: int = Field(default=1, ge=1, le=10_000)
    """Default checkpoint cadence for agent runs."""

    agent_run_default_max_subagent_failures_per_role: int = Field(default=2, ge=0, le=100)
    """Default retry budget before one role is considered degraded or stalled."""

    agent_run_default_on_budget_exhausted: Literal["pause", "finalize_partial"] = "pause"
    """Default action when the run-level wall-clock budget is exhausted."""

    agent_run_default_on_subagent_disconnect: Literal["retry_then_degrade", "pause", "fail"] = "retry_then_degrade"
    """Default action when one subagent disconnects or stalls."""

    exec_allowed_env_vars: list[str] = Field(default_factory=lambda: _empty_list_typed(str))
    """Exec runtime 可直接覆寫的環境變數白名單。"""

    exec_default_shell: Literal["auto", "bash", "sh", "pwsh", "powershell", "cmd"] = "auto"
    """Exec runtime 預設 shell provider。"""

    exec_session_output_limit: int = Field(default=8000, ge=256, le=1_000_000)
    """Exec session tail 緩衝上限（字元數）。"""

    exec_default_timeout_sec: int = Field(default=30, ge=1, le=86_400)
    """Exec runtime 預設 timeout（秒）。"""

    shell_command_allowlist: list[str] = Field(
        default_factory=lambda: ["ls", "cat", "pwd", "echo", "date", "which"]
    )
    """無需確認可直接執行的 Shell 命令白名單。"""

    max_file_write_size_mb: float = 10.0
    """允許寫入的最大檔案大小（MB）。"""

    file_ops_scope: Literal["workspace", "any"] = "workspace"
    """檔案操作範圍（workspace / any）。"""

    file_undo_max_size_mb: float = 2.0
    """允許保存 undo 的最大檔案大小（MB）。"""


    @model_validator(mode="before")
    @classmethod
    def _infer_legacy_autonomy_mode(cls, value: Any) -> Any:
        """??舐?蝚砍??怠?autonomy mode??"""
        if not isinstance(value, dict):
            return value
        if "autonomy_mode" in value:
            autonomy_mode = value.get("autonomy_mode")
            if isinstance(autonomy_mode, str):
                mode_defaults = autonomy_mode_defaults(autonomy_mode)
                return {
                    **{key: default for key, default in mode_defaults.items() if key != "autonomy_mode"},
                    **value,
                }
            return value

        relevant_keys = {
            "require_approval_for_shell",
            "require_approval_for_file_write",
            "file_ops_scope",
        }
        if not any(key in value for key in relevant_keys):
            return value

        require_shell = bool(
            value.get(
                "require_approval_for_shell",
                cls.model_fields["require_approval_for_shell"].default,
            )
        )
        require_file_write = bool(
            value.get(
                "require_approval_for_file_write",
                cls.model_fields["require_approval_for_file_write"].default,
            )
        )
        file_ops_scope = str(
            value.get(
                "file_ops_scope",
                cls.model_fields["file_ops_scope"].default,
            )
        )
        return {
            **value,
            "autonomy_mode": infer_autonomy_mode(
                require_approval_for_shell=require_shell,
                require_approval_for_file_write=require_file_write,
                file_ops_scope=file_ops_scope,
            ),
        }


class WebConfig(BaseModel):
    """WebAPI 伺服器設定。"""

    host: str = "0.0.0.0"
    """監聽地址。"""

    port: int = 8000
    """監聽埠號。"""

    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    """允許的 CORS 來源列表（開發預設值，可由設定檔或環境變數覆蓋）。"""

    reload: bool = False
    """開發模式熱重載（生產環境應設為 False）。"""


class ToolsConfig(BaseModel):
    """工具系統設定。"""

    extra_tools_dirs: list[str] = Field(default_factory=lambda: _empty_list_typed(str))
    """額外工具目錄，自動掃描 BaseTool 子類。"""

    # --- 搜尋引擎 ---

    web_search_engine: Literal[
        "tavily", "serper", "jina", "exa",
        "brave", "searxng", "duckduckgo", "duckduckgo_html",
    ] = "tavily"
    """網頁搜尋 provider 選擇（預設 Tavily，agent-native）。"""

    web_search_fallback_engines: list[str] = Field(
        default_factory=lambda: ["brave", "duckduckgo_html"],
    )
    """搜尋引擎 fallback chain，主引擎失敗時依序嘗試。"""

    web_search_tavily_api_key: SecretStr | None = None
    """Tavily Search API key (env: MOCHI_TAVILY_API_KEY)。"""

    web_search_serper_api_key: SecretStr | None = None
    """Serper.dev API key (env: MOCHI_SERPER_API_KEY)。"""

    web_search_jina_api_key: SecretStr | None = None
    """Jina AI API key (env: MOCHI_JINA_API_KEY)。"""

    web_search_exa_api_key: SecretStr | None = None
    """Exa (Metaphor) API key (env: MOCHI_EXA_API_KEY)。"""

    web_search_brave_api_key: SecretStr | None = None
    """Brave Search API key (env: MOCHI_BRAVE_API_KEY)。"""

    web_search_searxng_base_url: str | None = None
    """SearXNG 實例 base URL，例如 `https://search.example.com`。"""

    web_search_language: str | None = None
    """搜尋語言偏好 (e.g. 'zh-TW', 'en')。"""

    web_search_region: str | None = None
    """搜尋地區偏好 (e.g. 'tw', 'us')。"""

    # --- 網頁擷取 ---

    web_fetch_extractor: Literal["trafilatura", "jina_reader", "htmlparser"] = "trafilatura"
    """網頁內容擷取器選擇。"""

    web_fetch_jina_api_key: SecretStr | None = None
    """Jina Reader API key（可與 web_search_jina_api_key 共用）。"""

    # --- 文獻搜尋 ---

    semantic_scholar_api_key: SecretStr | None = None
    """Semantic Scholar API key (env: MOCHI_S2_API_KEY)。"""

    pubmed_api_key: SecretStr | None = None
    """PubMed/NCBI API key (env: MOCHI_PUBMED_API_KEY)。"""

    pubmed_email: str | None = None
    """PubMed E-utilities 用的聯繫 email。"""

    crossref_mailto: str | None = None
    """Crossref polite pool 用的 email。"""

    # --- HTTP 共用 ---

    http_timeout: float = 20.0
    """HTTP 工具共用逾時秒數。"""

    http_max_retries: int = 3
    """HTTP 工具共用最大重試次數。"""

    http_backoff_base: float = 1.0
    """HTTP 工具指數退避基數（秒）。"""


    @field_validator("web_search_engine", mode="before")
    @classmethod
    def _normalize_web_search_engine(cls, value: Any) -> Any:
        if isinstance(value, str):
            return normalize_web_search_provider(value)
        return value

    @field_validator("web_search_fallback_engines", mode="before")
    @classmethod
    def _normalize_web_search_fallback_engines(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            candidate = normalize_web_search_provider(item)
            if candidate and candidate not in normalized:
                normalized.append(candidate)
        return normalized


class AgentConfig(BaseModel):
    """Agent 核心設定。"""

    system_prompt: str = (
        "You are Mochi, an efficient and honest AI assistant. "
        "Reply in the same language as the user unless they request otherwise."
    )
    """系統提示詞。"""

    max_react_iterations: int = 10
    """ReAct 迴圈最大步驟數，防止無限循環。"""

    max_context_tokens: int = 3000
    """傳入 LLM 的最大 context token 數（不含輸出）。"""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    """採樣溫度（0.0–2.0）。"""

    max_tokens: int = Field(default=4096, ge=1, le=131072)
    """最大輸出 token 數。"""

    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    """Top-p 取樣機率。"""

    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    """Min-p 取樣機率。"""

    top_k: int = Field(default=0, ge=0)
    """Top-k 取樣數量；0 代表停用。"""

    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    """Frequency penalty。"""

    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    """Presence penalty。"""

    repeat_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    reasoning_effort: ReasoningEffort | None = None
    """Repeat penalty。"""

    show_token_stats: bool = False
    """是否顯示 token 統計資訊。"""

    presets: list[InferencePreset] = Field(default_factory=lambda: [InferencePreset(name="default")])
    """推理參數 preset 列表。"""

    active_preset: str = "default"
    """目前啟用的 preset 名稱。"""


# ---------------------------------------------------------------------------
# 頂層設定
# ---------------------------------------------------------------------------


class MochiConfig(BaseModel):
    """Mochi 主設定（頂層 Schema）。"""

    # 代理設定
    agent: AgentConfig = Field(default_factory=AgentConfig)

    # 模型後端
    model: str = DEFAULT_MODEL
    """預設模型規格（格式：ollama:<name> / /path/to/.gguf / http://host/v1）。"""

    # 語言/地區預設
    locale_defaults: LocaleDefaultsConfig = Field(default_factory=LocaleDefaultsConfig)

    # 模型 setup metadata
    model_setup: ModelSetupConfig = Field(default_factory=ModelSetupConfig)

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)
    openai_codex: OpenAICodexConfig = Field(default_factory=OpenAICodexConfig)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    gguf: GGUFConfig = Field(default_factory=GGUFConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)
    local_models: LocalModelConfig = Field(default_factory=LocalModelConfig)

    # 語音
    voice: VoiceConfig = Field(default_factory=VoiceConfig)

    # 工具
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    # 記憶
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # 學習
    learning: LearningConfig = Field(default_factory=LearningConfig)

    # 頻道
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)

    # 安全
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # Web API
    web: WebConfig = Field(default_factory=WebConfig)

    # 路徑
    workspace_dir: str = Field(default_factory=defaults.default_workspace_dir)
    sessions_dir: str = Field(default_factory=defaults.default_sessions_dir)
    skills_dir: str = Field(default_factory=defaults.default_skills_dir)
    plugins_dir: str = Field(default_factory=defaults.default_plugins_dir)

    # 日誌
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
