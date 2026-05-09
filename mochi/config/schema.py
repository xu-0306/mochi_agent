"""Mochi 完整 Pydantic v2 設定 Schema。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, SecretStr, field_validator

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
    """GGUF（llama-cpp-python）後端設定。"""

    n_ctx: int = 4096
    """上下文視窗大小。"""

    n_gpu_layers: int = -1
    """GPU 層數，-1 表示全部卸載到 GPU。"""

    n_threads: int | None = None
    """CPU 執行緒數，None 表示自動。"""


class HuggingFaceConfig(BaseModel):
    """HuggingFace Safetensors 後端設定。"""

    device: str = "auto"
    """推理設備（auto / cpu / cuda）。"""

    torch_dtype: str = "auto"
    """Torch 資料類型（auto / float16 / bfloat16）。"""

    trust_remote_code: bool = False
    """是否信任遠端程式碼。"""


class OpenAICompatConfig(BaseModel):
    """OpenAI-compatible 遠端 API 設定。"""

    base_url: str = "https://api.openai.com/v1"
    """OpenAI-compatible API base URL。"""

    model: str = "gpt-4o-mini"
    """預設遠端模型名稱。"""

    api_key: SecretStr | None = None
    """遠端 API key（敏感資料）。"""

    provider: Literal["openai_compat", "gemini", "anthropic"] = "openai_compat"
    """UI/provider preset；底層目前皆走 OpenAI-compatible protocol。"""

    timeout: float = 120.0
    """HTTP 請求逾時秒數。"""


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

    provider: Literal["ollama", "openai_compat", "gemini", "anthropic", "local"]
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
    """底層 backend family。"""


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


class VoiceConfig(BaseModel):
    """語音管線設定（STT + TTS + VAD + Audio）。"""

    enabled: bool = False
    """是否啟用語音功能。"""

    # STT 設定（Phase 4）
    stt_backend: Literal[
        "auto",
        "faster-whisper",
        "openai-api",
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
        "kokoro-tts",
        "openai-tts",
        "piper",
    ] = "edge-tts"
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
    """短期記憶最大訊息數量。"""

    fts_top_k: int = 5
    """FTS5 搜尋返回最大結果數。"""


class SecurityConfig(BaseModel):
    """安全設定。"""

    require_approval_for_shell: bool = True
    """執行 Shell 命令前是否需要使用者確認。"""

    require_approval_for_file_write: bool = True
    """寫入檔案前是否需要使用者確認。"""

    shell_command_allowlist: list[str] = Field(
        default_factory=lambda: ["ls", "cat", "pwd", "echo", "date", "which"]
    )
    """無需確認可直接執行的 Shell 命令白名單。"""

    max_file_write_size_mb: float = 10.0
    """允許寫入的最大檔案大小（MB）。"""


class WebConfig(BaseModel):
    """WebAPI 伺服器設定。"""

    host: str = "0.0.0.0"
    """監聽地址。"""

    port: int = 8000
    """監聽埠號。"""

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
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
    gguf: GGUFConfig = Field(default_factory=GGUFConfig)
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)

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
