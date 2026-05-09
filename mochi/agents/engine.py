"""AgentEngine — 頂層入口，協調所有子系統。"""

from __future__ import annotations

import copy
import inspect
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from loguru import logger
from pydantic import SecretStr

from mochi.agents.compaction import ConversationCompactor
from mochi.agents.context import ContextManager
from mochi.agents.events import (
    AgentEvent,
    ErrorEvent,
    FinalAnswerEvent,
    ThinkingEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.agents.prompt_builder import PromptBuilder
from mochi.agents.react_loop import AsyncReActLoop
from mochi.backends.router import BackendRouter
from mochi.backends.types import Message, ModelInfo
from mochi.config.schema import MochiConfig
from mochi.learning.evaluator import OutcomeEvaluator
from mochi.learning.extractor import SkillExtractor
from mochi.learning.improver import SkillImprover
from mochi.learning.skill_library import SkillLibrary
from mochi.learning.skill_loader import SkillLoader, default_system_skills_dir
from mochi.learning.trajectory import TrajectoryLogger
from mochi.learning.types import Trajectory, TrajectoryStep
from mochi.memory.conversation import ConversationMemory
from mochi.memory.store import MemoryStore
from mochi.sessions.store import SessionStore
from mochi.tools.execute_code import ExecuteCodeTool
from mochi.tools.file_ops import FileReadTool, FileWriteTool
from mochi.tools.literature_search import (
    ArxivSearchTool,
    CrossrefSearchTool,
    PubMedSearchTool,
    SemanticScholarSearchTool,
)
from mochi.tools.mcp_client import MCPCallTool
from mochi.tools.memory_save import MemorySaveTool
from mochi.tools.memory_search import MemorySearchTool
from mochi.tools.registry import ToolRegistry
from mochi.tools.shell import ShellTool
from mochi.tools.web_fetch import WebFetchTool
from mochi.tools.web_search import WebSearchTool
from mochi.voice.events import VoiceEvent
from mochi.voice.router import SUPPORTED_STT_BACKENDS, SUPPORTED_TTS_BACKENDS, VoiceRouter
from mochi.voice.session_manager import VoiceSessionManager
from mochi.voice.status import build_voice_runtime_status
from mochi.voice.voice_session import VoiceSession


class AgentEngine:
    """頂層 Agent 引擎，整合後端、工具、Prompt 組裝與 ReAct 迴圈。

    使用前請先呼叫 initialize() 完成非同步初始化。
    """

    def __init__(
        self,
        config: MochiConfig,
        *,
        voice_vad: object | None = None,
        voice_stt: object | None = None,
        voice_tts: object | None = None,
    ) -> None:
        """初始化 AgentEngine（同步部分）。

        Args:
            config: Mochi 完整設定。
        """
        self._config = config
        self._router = BackendRouter(
            ollama_base_url=config.ollama.base_url,
            openai_default_model=config.openai_compat.model,
            openai_api_key=(
                config.openai_compat.api_key.get_secret_value()
                if config.openai_compat.api_key is not None
                else ""
            ),
        )
        self._tool_registry = ToolRegistry(
            extra_dirs=config.tools.extra_tools_dirs or None,
            discover_builtin=False,
        )
        self._prompt_builder = PromptBuilder(config.agent.system_prompt)
        self._memory_store = MemoryStore(db_path=config.memory.db_path)
        self._session_store = SessionStore(sessions_dir=config.sessions_dir)
        self._contexts: dict[str, ContextManager] = {}
        self._skill_library = SkillLibrary(db_path=self._skills_db_path())
        self._skill_loader = self._make_skill_loader()
        self._trajectory_logger = TrajectoryLogger(storage_path=self._trajectories_jsonl_path())
        self._outcome_evaluator = OutcomeEvaluator()
        self._skill_extractor = SkillExtractor()
        self._skill_improver = SkillImprover()
        self._voice_vad_seed = voice_vad
        self._voice_vad_factory = self._make_injected_vad_factory(voice_vad)
        self._voice_stt = voice_stt
        self._voice_tts = voice_tts
        self._voice_router: VoiceRouter | None = None
        self._voice_last_load_error: str | None = None
        self._voice_session_manager = VoiceSessionManager()
        self._initialized = False
        self._register_builtin_tools()

    async def initialize(self) -> None:
        """非同步初始化：載入後端並完成準備。"""
        await self._router.load(self._config.model)
        logger.info(f"AgentEngine initialized with model: {self._config.model}")
        self._initialized = True

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """執行單輪對話，回傳事件串流。

        Args:
            message: 使用者輸入文字。
            session_id: 會話 ID（預留，Phase 2 實作持久化）。

        Yields:
            AgentEvent 事件流。
        """
        if not self._initialized:
            await self.initialize()

        session_key = session_id or "default"
        context = await self._get_context(session_key)
        prompt_context = await context.prepare_prompt_context(
            message,
            history_limit=self._config.memory.max_short_term_messages,
            memory_top_k=self._config.memory.fts_top_k,
        )
        skills_context = await self._build_skills_context(message)
        system_prompt = self._prompt_builder.build_system_prompt(
            skills_context=skills_context,
            memory_context=self._merge_memory_and_summary_context(
                memory_context=prompt_context.memory_context,
                summary=prompt_context.summary,
            ),
        )
        trajectory_id = self._start_trajectory(message)
        turn_id = str(uuid4())
        turn_event_seq = 0
        user_msg = Message(role="user", content=message)
        await self._persist_session_message(session_key, user_msg, turn_id=turn_id)

        react_loop = AsyncReActLoop(
            backend=self._router.active,
            tool_registry=self._tool_registry,
            max_iterations=self._config.agent.max_react_iterations,
        )

        final_text = ""
        async for event in react_loop.run(
            system_prompt=system_prompt,
            history=prompt_context.history,
            user_message=message,
        ):
            self._log_agent_event(trajectory_id, event)
            if isinstance(event, FinalAnswerEvent):
                final_text = event.content
                event.trajectory_id = trajectory_id
            event.turn_id = turn_id  # type: ignore[attr-defined]
            turn_event_seq += 1
            await self._persist_turn_event(
                session_key,
                event,
                turn_id=turn_id,
                seq=turn_event_seq,
            )
            yield event

        await self._finish_learning_cycle(trajectory_id)

        assistant_msg = Message(role="assistant", content=final_text)
        context.add_message(user_msg)
        context.add_message(assistant_msg)
        await self._persist_session_message(session_key, assistant_msg, turn_id=turn_id)

    async def switch_model(self, model_spec: str) -> ModelInfo:
        """切換活躍模型並回傳新模型資訊。"""
        backend = await self._router.switch(model_spec)
        self._config.model = model_spec
        self._initialized = True
        return backend.get_model_info()

    def get_model_info(self) -> ModelInfo:
        """回傳目前活躍模型資訊；尚未初始化時依 config 產生摘要。"""
        if self._initialized:
            return self._router.active.get_model_info()

        model_spec = self._config.model
        if model_spec.startswith("ollama:"):
            return ModelInfo(
                name=model_spec[len("ollama:"):],
                backend_type="ollama",
                supports_tool_calling=True,
            )
        if model_spec.startswith(("http://", "https://")):
            return ModelInfo(
                name=model_spec,
                backend_type="openai_compat",
                supports_tool_calling=True,
            )
        if model_spec.lower().endswith(".gguf"):
            return ModelInfo(name=model_spec, backend_type="gguf")
        return ModelInfo(name=model_spec, backend_type="safetensors")

    async def switch_ollama_backend(
        self,
        *,
        model: str,
        base_url: str | None = None,
    ) -> ModelInfo:
        """以指定 Ollama endpoint 與模型切換活躍後端。"""
        backend = await self._router.switch_ollama(model=model, base_url=base_url)
        self._config.model = f"ollama:{model.strip()}"
        if base_url:
            self._config.ollama.base_url = base_url.strip().rstrip("/")
        self._initialized = True
        return backend.get_model_info()

    async def switch_openai_compat_backend(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        provider: Literal["openai_compat", "gemini", "anthropic"] = "openai_compat",
    ) -> ModelInfo:
        """以 OpenAI-compatible API 設定切換活躍後端。"""
        backend = await self._router.switch_openai_compat(
            base_url=base_url,
            model=model,
            api_key=api_key,
        )
        normalized_base_url = base_url.strip().rstrip("/")
        self._config.model = normalized_base_url
        self._config.openai_compat.base_url = normalized_base_url
        self._config.openai_compat.model = model.strip()
        self._config.openai_compat.provider = cast(
            Literal["openai_compat", "gemini", "anthropic"],
            provider,
        )
        if api_key:
            from pydantic import SecretStr

            self._config.openai_compat.api_key = SecretStr(api_key)
        self._initialized = True
        return backend.get_model_info()

    async def list_skills(self) -> list:
        """列出目前技能庫中的技能。"""
        await self._sync_filesystem_skills()
        return await self._skill_library.list()

    async def search_skills(self, query: str, top_k: int = 3) -> list:
        """搜尋目前技能庫中的相關技能。"""
        await self._sync_filesystem_skills()
        return await self._skill_library.search(query, top_k=top_k)

    async def provide_feedback(self, trajectory_id: str, feedback: str) -> None:
        """補充指定 trajectory 的使用者回饋。"""
        trajectory = self._trajectory_logger.export(trajectory_id)
        self._trajectory_logger.finish(trajectory_id, trajectory.outcome, feedback=feedback)

    async def apply_config(self, config: MochiConfig, *, reload_voice: bool = False) -> None:
        """套用新的 runtime 設定，並重建與路徑相關的共享元件。"""
        previous_voice = self._config.voice
        self._config = config
        self._memory_store = MemoryStore(db_path=config.memory.db_path)
        self._session_store = SessionStore(sessions_dir=config.sessions_dir)
        self._skill_library = SkillLibrary(db_path=self._skills_db_path())
        self._skill_loader = self._make_skill_loader()
        self._trajectory_logger = TrajectoryLogger(storage_path=self._trajectories_jsonl_path())
        self._contexts.clear()
        self._register_builtin_tools()

        if reload_voice or config.voice != previous_voice:
            await self._voice_session_manager.release_all()
            if self._voice_router is not None:
                await self._voice_router.close()
                self._voice_router = None
            self._voice_stt = None
            self._voice_tts = None
            self._voice_vad_seed = None
            self._voice_vad_factory = None
            self._voice_last_load_error = None

    async def voice_chat(
        self,
        audio: bytes,
        session_id: str | None = None,
    ) -> AsyncIterator[VoiceEvent]:
        """執行單輪語音對話（VAD → STT → Agent → TTS）。"""
        voice_session = await self.get_or_create_voice_session(session_id=session_id)

        async for event in voice_session.handle_turn(audio, session_id=session_id):
            yield event

    async def synthesize_speech(self, text: str) -> bytes:
        """使用共享 voice runtime 的 TTS 將文字轉為 PCM16 bytes。"""
        await self._ensure_voice_runtime_loaded()
        if self._voice_tts is None:
            raise RuntimeError("Voice TTS is not initialized.")

        synthesize = getattr(self._voice_tts, "synthesize", None)
        if not callable(synthesize):
            raise AttributeError("Voice TTS must provide synthesize().")

        result = synthesize(text)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, bytearray):
            return bytes(result)
        if isinstance(result, bytes):
            return result
        raise TypeError("Voice TTS synthesize() must return bytes.")

    async def get_or_create_voice_session(
        self,
        session_id: str | None = None,
    ) -> VoiceSession:
        """取得或 lazy 建立可重用的語音會話物件（依 session_id 隔離）。"""
        return await self._voice_session_manager.get_or_create(
            session_id=session_id,
            factory=self._create_voice_session,
        )

    async def release_voice_session(self, session_id: str | None = None) -> bool:
        """釋放指定 session_id 的語音會話快取。"""
        return await self._voice_session_manager.release(session_id=session_id)

    async def get_voice_runtime_status(self) -> dict[str, Any]:
        """取得共享語音 runtime 狀態摘要（供 API 與監看使用）。"""
        active_runtime = None
        stt_runtime_spec: dict[str, Any] | None = None
        if self._voice_router is not None:
            with_active = True
            try:
                active_runtime = self._voice_router.active
            except RuntimeError:
                with_active = False
            if with_active and self._voice_router.last_stt_runtime_spec is not None:
                stt_runtime_spec = self._voice_router.last_stt_runtime_spec.to_dict()

        stt_component = getattr(active_runtime, "stt", None) if active_runtime is not None else self._voice_stt
        tts_component = getattr(active_runtime, "tts", None) if active_runtime is not None else self._voice_tts
        vad_component = getattr(active_runtime, "vad", None) if active_runtime is not None else self._voice_vad_seed
        last_load_error = self._voice_last_load_error
        if self._voice_router is not None and self._voice_router.last_load_error:
            last_load_error = self._voice_router.last_load_error
        session_diagnostics = await self._voice_session_manager.get_runtime_diagnostics()

        return await build_voice_runtime_status(
            config=self._config.voice,
            supported_stt_backends=sorted(SUPPORTED_STT_BACKENDS),
            supported_tts_backends=sorted(SUPPORTED_TTS_BACKENDS),
            stt_component=stt_component,
            tts_component=tts_component,
            vad_component=vad_component,
            has_vad_factory=self._voice_vad_factory is not None,
            stt_runtime_spec=stt_runtime_spec,
            last_load_error=last_load_error,
            session_diagnostics=session_diagnostics,
        )

    def reset_history(self) -> None:
        """清空對話歷史（開新會話時使用）。"""
        default_context = self._contexts.get("default")
        if default_context is not None:
            default_context.clear_history()

    async def close(self) -> None:
        """釋放所有資源。"""
        for tool in self._tool_registry.list_tools():
            close_method = getattr(tool, "close", None)
            if close_method is None:
                continue
            maybe_awaitable = close_method()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        if self._initialized:
            await self._router.active.close()
            logger.info("AgentEngine closed.")
        await self._voice_session_manager.release_all()
        if self._voice_router is not None:
            await self._voice_router.close()
            self._voice_router = None

    def _skills_db_path(self) -> Path:
        """取得本地技能庫 SQLite 路徑。"""
        return Path(self._config.skills_dir).expanduser() / "skills.db"

    def _trajectories_jsonl_path(self) -> Path:
        """取得本地 trajectory JSONL 路徑。"""
        return Path(self._config.workspace_dir).expanduser() / "trajectories.jsonl"

    async def _build_skills_context(self, message: str) -> str:
        """搜尋相關技能並格式化為 system prompt context。"""
        if not self._config.learning.enabled:
            return ""
        try:
            await self._sync_filesystem_skills()
            skills = await self._skill_library.search(message, top_k=3)
        except Exception as exc:  # pragma: no cover - 防禦性收斂
            logger.warning(f"Skill search failed: {exc}")
            return ""
        return self._prompt_builder.format_skills_context(skills)

    def _make_skill_loader(self) -> SkillLoader:
        return SkillLoader.from_paths(
            self._config.skills_dir,
            system_skills_dir=default_system_skills_dir(),
        )

    async def _sync_filesystem_skills(self) -> None:
        if not self._config.learning.auto_sync_filesystem_skills:
            return
        result = await self._skill_loader.sync(self._skill_library)
        if result.errors:
            logger.warning(f"Filesystem skill sync completed with errors: {result.errors}")

    def _start_trajectory(self, message: str) -> str | None:
        """依設定啟動本輪 trajectory 記錄。"""
        if not self._config.learning.enabled:
            return None
        return self._trajectory_logger.start(message)

    def _log_agent_event(self, trajectory_id: str | None, event: AgentEvent) -> None:
        """將 AgentEvent 轉成 trajectory step。"""
        if trajectory_id is None:
            return
        step = self._trajectory_step_from_event(event)
        if step is None:
            return
        self._trajectory_logger.log_step(trajectory_id, step)

    def _trajectory_step_from_event(self, event: AgentEvent) -> TrajectoryStep | None:
        """建立學習系統使用的 trajectory step。"""
        now = datetime.now(UTC).timestamp()
        if isinstance(event, ThinkingEvent):
            return TrajectoryStep(
                step_id=self._next_trajectory_step_id(),
                timestamp=now,
                step_type="llm_call",
                input_data={},
                output_data={"content": event.content},
                tokens_used=0,
                duration_ms=0,
            )
        if isinstance(event, ToolCallRequestEvent):
            return TrajectoryStep(
                step_id=self._next_trajectory_step_id(),
                timestamp=now,
                step_type="tool_call",
                input_data={"tool_name": event.tool_name, "arguments": event.arguments},
                output_data={},
                tokens_used=0,
                duration_ms=0,
                metadata={"call_id": event.call_id},
            )
        if isinstance(event, ToolCallResultEvent):
            return TrajectoryStep(
                step_id=self._next_trajectory_step_id(),
                timestamp=now,
                step_type="tool_result",
                input_data={"tool_name": event.tool_name},
                output_data={"result": event.result, "error": event.error},
                tokens_used=0,
                duration_ms=0,
                metadata={"call_id": event.call_id},
            )
        if isinstance(event, FinalAnswerEvent):
            return TrajectoryStep(
                step_id=self._next_trajectory_step_id(),
                timestamp=now,
                step_type="final_answer",
                input_data={},
                output_data={"content": event.content},
                tokens_used=0,
                duration_ms=0,
            )
        if isinstance(event, ErrorEvent):
            return TrajectoryStep(
                step_id=self._next_trajectory_step_id(),
                timestamp=now,
                step_type="final_answer",
                input_data={},
                output_data={},
                tokens_used=0,
                duration_ms=0,
                metadata={"error": event.message, "code": event.code},
            )
        return None

    def _next_trajectory_step_id(self) -> int:
        """產生本輪 process 內遞增的 trajectory step id。"""
        current = getattr(self, "_trajectory_step_counter", 0) + 1
        self._trajectory_step_counter = current
        return current

    async def _finish_learning_cycle(self, trajectory_id: str | None) -> None:
        """完成 trajectory 評估與可選 skill extraction。"""
        if trajectory_id is None:
            return
        trajectory = self._trajectory_logger.export(trajectory_id)
        outcome = await self._outcome_evaluator.evaluate(trajectory)
        self._trajectory_logger.finish(trajectory_id, outcome)
        trajectory = self._trajectory_logger.export(trajectory_id)
        await self._maybe_extract_skill(trajectory)

    async def _maybe_extract_skill(self, trajectory: Trajectory) -> None:
        """成功且足夠複雜時，自動萃取或合併技能。"""
        if not self._config.learning.enabled or not self._config.learning.auto_extract_skills:
            return
        if trajectory.outcome != "success":
            return
        if len(trajectory.steps) < self._config.learning.min_steps_for_extraction:
            return
        try:
            extracted = await self._skill_extractor.extract(trajectory, self._router.active)
            matches = await self._skill_library.search(
                " ".join([extracted.name, extracted.description, *extracted.trigger_keywords]),
                top_k=3,
            )
            learned_match = next(
                (match for match in matches if getattr(match, "source_type", "learned") == "learned"),
                None,
            )
            if (
                learned_match
                and learned_match.success_rate >= self._config.learning.skill_improvement_threshold
            ):
                improved = await self._skill_improver.improve(learned_match, trajectory, self._router.active)
                await self._skill_library.update(improved.skill_id, improved.to_dict())
            else:
                await self._skill_library.add(extracted)
        except Exception as exc:  # pragma: no cover - 學習失敗不應影響使用者回覆
            logger.warning(f"Skill extraction skipped: {exc}")

    def _register_builtin_tools(self) -> None:
        """以共享 runtime 物件覆蓋內建工具預設實例。"""
        from mochi.tools.calculator import CalculatorTool
        from mochi.tools.datetime_tool import DateTimeTool

        tc = self._config.tools  # shortcut

        self._tool_registry.register(
            ShellTool(
                allowlist=self._config.security.shell_command_allowlist,
                workspace_dir=self._config.workspace_dir,
                require_approval=self._config.security.require_approval_for_shell,
            )
        )
        self._tool_registry.register(
            FileReadTool(workspace_dir=self._config.workspace_dir)
        )
        self._tool_registry.register(
            FileWriteTool(
                workspace_dir=self._config.workspace_dir,
                require_approval=self._config.security.require_approval_for_file_write,
                max_write_size_mb=self._config.security.max_file_write_size_mb,
            )
        )

        # --- 搜尋工具 ---
        def _secret(s: SecretStr | None) -> str | None:
            return s.get_secret_value() if s is not None else None

        self._tool_registry.register(
            WebSearchTool(
                engine=tc.web_search_engine,
                timeout=tc.http_timeout,
                fallback_engines=tc.web_search_fallback_engines,
                searxng_base_url=tc.web_search_searxng_base_url,
                brave_api_key=_secret(tc.web_search_brave_api_key),
                tavily_api_key=_secret(tc.web_search_tavily_api_key),
                serper_api_key=_secret(tc.web_search_serper_api_key),
                jina_api_key=_secret(tc.web_search_jina_api_key),
                exa_api_key=_secret(tc.web_search_exa_api_key),
                language=tc.web_search_language,
                region=tc.web_search_region,
            )
        )

        # --- 網頁擷取 ---
        jina_key = _secret(tc.web_fetch_jina_api_key) or _secret(tc.web_search_jina_api_key)
        self._tool_registry.register(
            WebFetchTool(
                timeout=tc.http_timeout,
                jina_api_key=jina_key,
                extractor=tc.web_fetch_extractor,
            )
        )

        # --- 文獻工具 ---
        self._tool_registry.register(ArxivSearchTool(timeout=tc.http_timeout))
        self._tool_registry.register(
            SemanticScholarSearchTool(
                timeout=tc.http_timeout,
                api_key=_secret(tc.semantic_scholar_api_key),
            )
        )
        self._tool_registry.register(
            CrossrefSearchTool(
                timeout=tc.http_timeout,
                mailto=tc.crossref_mailto,
            )
        )
        self._tool_registry.register(
            PubMedSearchTool(
                timeout=tc.http_timeout,
                email=tc.pubmed_email,
                api_key=_secret(tc.pubmed_api_key),
            )
        )

        # --- 程式碼執行 ---
        self._tool_registry.register(
            ExecuteCodeTool(
                workspace_dir=self._config.workspace_dir,
                require_approval=self._config.security.require_approval_for_shell,
            )
        )

        # --- MCP ---
        self._tool_registry.register(MCPCallTool())

        # --- 記憶 ---
        self._tool_registry.register(
            MemorySearchTool(
                memory_store=self._memory_store,
                workspace_dir=self._config.workspace_dir,
                default_top_k=self._config.memory.fts_top_k,
            )
        )
        self._tool_registry.register(
            MemorySaveTool(
                memory_store=self._memory_store,
                workspace_dir=self._config.workspace_dir,
            )
        )

        # --- 實用工具 ---
        self._tool_registry.register(CalculatorTool())
        self._tool_registry.register(DateTimeTool())

    async def _get_context(self, session_id: str) -> ContextManager:
        """取得或建立指定 session 的上下文管理器。"""
        context = self._contexts.get(session_id)
        if context is not None:
            return context

        context = ContextManager(
            conversation_memory=ConversationMemory(
                max_messages=self._config.memory.max_short_term_messages
            ),
            memory_store=self._memory_store,
            compactor=ConversationCompactor.from_max_messages(
                self._config.memory.max_short_term_messages
            ),
            history_window=self._config.memory.max_short_term_messages,
            memory_top_k=self._config.memory.fts_top_k,
        )
        await self._restore_session_history(session_id, context)
        self._contexts[session_id] = context
        return context

    def _merge_memory_and_summary_context(
        self,
        *,
        memory_context: str | None,
        summary: str | None,
    ) -> str | None:
        """合併長期記憶與短期對話摘要為單一 memory context。"""
        memory_text = memory_context.strip() if isinstance(memory_context, str) else ""
        summary_text = summary.strip() if isinstance(summary, str) else ""

        if not memory_text and not summary_text:
            return None
        if memory_text and not summary_text:
            return memory_text
        if summary_text and not memory_text:
            return f"Conversation summary:\n{summary_text}"
        return f"{memory_text}\n\nConversation summary:\n{summary_text}"

    async def _restore_session_history(
        self,
        session_id: str,
        context: ContextManager,
    ) -> None:
        """從 JSONL 還原已持久化的會話歷史。"""
        events = await self._session_store.load_session(session_id)
        for event in events:
            if event.get("type") != "message":
                continue
            role = event.get("role")
            content = event.get("content")
            if role in {"system", "user", "assistant", "tool"} and isinstance(content, str):
                context.add_message(Message(role=role, content=content))

    async def _persist_session_messages(
        self,
        session_id: str,
        user_message: Message,
        assistant_message: Message,
    ) -> None:
        """將本輪核心訊息持久化到 session store。"""
        turn_id = str(uuid4())
        await self._persist_session_message(session_id, user_message, turn_id=turn_id)
        await self._persist_session_message(session_id, assistant_message, turn_id=turn_id)

    async def _persist_session_message(
        self,
        session_id: str,
        message: Message,
        *,
        turn_id: str,
    ) -> None:
        """將 canonical message 持久化到 session store。"""
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        await self._session_store.save_event(
            session_id,
            {
                "type": "message",
                "schema_version": 1,
                "turn_id": turn_id,
                "role": message.role,
                "content": message.content,
                "timestamp": timestamp,
            },
        )

    async def _persist_turn_event(
        self,
        session_id: str,
        event: AgentEvent,
        *,
        turn_id: str,
        seq: int,
    ) -> None:
        """將 UI replay event 持久化到 session store。"""
        phase, payload = self._turn_event_payload(event)
        if phase is None:
            return

        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        await self._session_store.save_event(
            session_id,
            {
                "type": "turn_event",
                "schema_version": 1,
                "turn_id": turn_id,
                "event_id": f"{turn_id}:{seq}",
                "seq": seq,
                "phase": phase,
                "timestamp": timestamp,
                "payload": payload,
            },
        )

    def _turn_event_payload(self, event: AgentEvent) -> tuple[str | None, dict[str, Any]]:
        """將 AgentEvent 轉成 session replay payload。"""
        if isinstance(event, ThinkingEvent):
            return "thinking", {"content": event.content}
        if isinstance(event, ToolCallRequestEvent):
            return "tool_call_request", {
                "call_id": event.call_id,
                "tool_name": event.tool_name,
                "arguments": copy.deepcopy(event.arguments),
            }
        if isinstance(event, ToolCallResultEvent):
            return "tool_call_result", {
                "call_id": event.call_id,
                "tool_name": event.tool_name,
                "result": copy.deepcopy(event.result),
                "error": event.error,
            }
        if isinstance(event, FinalAnswerEvent):
            return "final_answer", {
                "content": event.content,
                "trajectory_id": event.trajectory_id,
            }
        if isinstance(event, ErrorEvent):
            return "error", {"message": event.message, "code": event.code}
        return None, {}

    async def _create_voice_session(self) -> VoiceSession:
        """建立語音會話協調器（lazy）。"""
        await self._ensure_voice_runtime_loaded()
        if self._voice_stt is None or self._voice_tts is None:
            raise RuntimeError("Voice STT/TTS is not initialized.")

        return VoiceSession(
            vad=self._acquire_voice_vad(),
            stt=self._voice_stt,
            tts=self._voice_tts,
            agent_chat=self.chat,
            sample_rate=self._config.voice.sample_rate,
        )

    async def _ensure_voice_runtime_loaded(self) -> None:
        """確保共享 voice runtime 已載入。"""
        needs_runtime = (
            self._voice_stt is None
            or self._voice_tts is None
            or (self._voice_vad_seed is None and self._voice_vad_factory is None)
        )
        if not needs_runtime:
            return

        self._voice_router = self._voice_router or VoiceRouter()
        try:
            voice_runtime = await self._voice_router.load(self._config.voice)
        except Exception as exc:
            self._voice_last_load_error = str(exc)
            raise

        self._voice_stt = self._voice_stt or voice_runtime.stt
        self._voice_tts = self._voice_tts or voice_runtime.tts
        self._voice_vad_seed = self._voice_vad_seed or voice_runtime.vad
        if self._voice_vad_factory is None:
            self._voice_vad_factory = lambda: self._voice_router.create_vad(self._config.voice)
        self._voice_last_load_error = None

    def _acquire_voice_vad(self) -> object:
        """取得當前 session 專屬 VAD 實例。"""
        if self._voice_vad_factory is None:
            if self._voice_vad_seed is not None:
                vad = self._voice_vad_seed
                self._voice_vad_seed = None
                return vad
            raise RuntimeError("Voice VAD is not initialized.")
        return self._voice_vad_factory()

    @staticmethod
    def _make_injected_vad_factory(vad: object | None) -> Callable[[], object] | None:
        """將注入的 VAD 轉為可產生獨立實例的工廠。"""
        if vad is None:
            return None

        def _factory() -> object:
            try:
                return copy.deepcopy(vad)
            except Exception as exc:  # pragma: no cover - 防禦性分支
                raise RuntimeError(
                    "Injected voice_vad must support deepcopy for session isolation."
                ) from exc

        return _factory
