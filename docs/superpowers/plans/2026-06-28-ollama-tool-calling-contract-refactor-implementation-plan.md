# Ollama Tool-Calling Contract Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `@superpowers:subagent-driven-development` (recommended) or `@superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Mochi's tool-calling backend boundary so Ollama and other runtimes share a strict result contract, explicit tool-mode state machine, and recoverable probe flow without empty-final-response regressions.

**Architecture:** Keep one canonical internal tool protocol (`Message`, `ToolCall`, `GenerationResult`) and move provider-specific behavior behind backend-local protocol/state helpers. Add a shared validator for tool-eligible turns, a reusable simulated-tool protocol helper, and an explicit tool-calling state model that Ollama can use without leaking provider heuristics into the ReAct loop.

**Tech Stack:** Python 3.12, `httpx`, `pytest`, Mochi backend adapters (`ollama`, `openai_compat`), ReAct agent loop.

---

## Scope and Constraints

- Fix the bug at the backend adapter boundary, not in [mochi/agents/react_loop.py](/H:/_python/agent_mochi/mochi/agents/react_loop.py:611).
- Preserve the existing canonical internal protocol in [mochi/backends/types.py](/H:/_python/agent_mochi/mochi/backends/types.py:145) and [mochi/backends/types.py](/H:/_python/agent_mochi/mochi/backends/types.py:198).
- Do not introduce provider-specific branches into generic engine or ReAct code unless the branch is capability-based and reusable.
- Do not assume Ollama fallback is always valid. Prompt-simulated fallback must be treated as a distinct protocol/mode with its own validation.
- Keep `Gemini`, `vLLM`, `SGLang`, and `TensorRT-LLM` on the `openai_compat` route green while refactoring Ollama.
- Keep `ModelInfo.supports_tool_calling` current-state-based for runtime exposure: when `tool_call_mode == "unavailable"` or `tool_calling_blocked is True`, expose `supports_tool_calling=False`.

## File Structure

**Create**

- `mochi/backends/tool_call_state.py`
  - Canonical tool-calling state model shared by backends.
  - Owns mode/status transitions such as `native -> simulated_fallback -> native` and `native -> unavailable`.
- `mochi/backends/tool_call_contract.py`
  - Shared validator for tool-eligible turn results.
  - Decides whether a backend result is valid final content, valid tool call output, or invalid thinking-only / empty output.
- `mochi/backends/simulated_tool_protocol.py`
  - Shared wrapper around prompt-simulated tool calling.
  - Owns message flattening, tool instruction injection, simulated output parsing, and text extraction.

**Modify**

- `mochi/backends/base.py`
  - Add a default `probe_tool_calling()` contract returning `None`.
- `mochi/backends/ollama.py`
  - Replace the `_tool_calling_enabled` toggle with explicit state and probe/recovery logic.
  - Route simulated fallback through the shared protocol helper and shared contract validator.
- `mochi/backends/openai_compat.py`
  - Align existing fallback/probe behavior with the new shared state helper without changing provider semantics.
- `mochi/backends/tool_call_simulator.py`
  - Keep only low-level parsing/injection primitives or convert it into the low-level dependency used by `simulated_tool_protocol.py`.
- `mochi/agents/engine.py`
  - Generalize preflight probing so Ollama can participate when its status is unknown.

**Test**

- `tests/test_backends.py`
  - Ollama contract, fallback, recovery, and shared simulated-protocol regression tests.
- `tests/test_engine_phase2.py`
  - Backend-agnostic preflight probe behavior.
- `tests/test_api_chat_models.py`
  - Probe endpoint / diagnostics exposure remains stable.
- `tests/test_tool_call_simulator.py`
  - Low-level simulator parser/injection coverage stays intact after helper extraction.
- `tests/test_gguf_backend_runtime.py`
  - Flattened-text fallback behavior stays compatible with the shared simulated protocol primitives.
- `tests/test_safetensors_backend_runtime.py`
  - Safetensors fallback behavior stays compatible with the shared simulator/protocol plumbing.

## Recommended PR Boundaries

1. Local commit only: Task 1 red tests. Do not open a PR while the branch is intentionally red.
2. PR1: Task 2 + Task 3
3. PR2: Task 4
4. PR3: Task 5

Do not combine all three PRs unless the branch is private and short-lived.

## Tool State Semantics

Lock these semantics before writing code:

- `native` is the preferred active mode.
- `simulated_fallback` is recoverable, not terminal.
- `unavailable` is terminal for automatic tool exposure in the current backend instance.
- Entering `simulated_fallback` does **not** prove the simulated protocol is valid.
- A simulated retry becomes validated only after it yields structured simulated `tool_calls` or visible final `content`.
- A simulated retry that raises HTTP error, transport error, empty output, or `thinking_only` output moves the backend to `unavailable`.
- Automatic preflight probing may re-probe recoverable fallback states such as:
  - `active_mode == "simulated_fallback"`
  - `native_status in {"unknown", "native_tool_calls_missing"}`
- Automatic preflight probing must not re-probe terminal states such as:
  - `active_mode == "unavailable"`
  - `tool_calling_blocked is True`
- Manual `/v1/models/probe-tool-calling` remains allowed to test recovery explicitly even after exposure was disabled.

### Task 1: Lock the Contract With Failing Tests

**Files:**
- Modify: `tests/test_backends.py`
- Modify: `tests/test_engine_phase2.py`
- Modify: `tests/test_api_chat_models.py`

- [ ] **Step 1: Write the failing Ollama backend tests**

```python
@pytest.mark.asyncio
async def test_ollama_retry_that_returns_only_thinking_raises_backend_error() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    tools = [
        ToolSchema(
            name="web_search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
    ]
    native_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "", "thinking": "Need web context."},
            "done": True,
            "done_reason": "stop",
        }
    )
    retry_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "", "thinking": "Still deciding."},
            "done": True,
            "done_reason": "stop",
        }
    )

    with patch.object(backend._client, "post", new_callable=AsyncMock, side_effect=[native_response, retry_response]):
        with pytest.raises(RuntimeError, match="tool-eligible turn"):
            await backend.generate(messages=[Message(role="user", content="Search Mochi AI")], tools=tools, stream=False)
```

```python
@pytest.mark.asyncio
async def test_ollama_probe_reenables_native_mode_after_fallback() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    backend._tool_state.active_mode = "simulated_fallback"  # noqa: SLF001
    backend._tool_state.native_status = "native_tool_calls_missing"  # noqa: SLF001
    backend._tool_state.fallback_validation_status = "validated"  # noqa: SLF001
    probe_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "probe-call-1", "function": {"name": "mochi_tool_probe", "arguments": {"value": "ok"}}}
                ],
            },
            "done": True,
            "done_reason": "tool_calls",
        }
    )

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=probe_response):
        result = await backend.probe_tool_calling()

    assert result is not None
    assert result["status"] == "supported"
    metadata = backend.get_model_info().metadata
    assert metadata["tool_call_mode"] == "native"
    assert metadata["native_tool_calling_status"] == "supported"
```

```python
@pytest.mark.asyncio
async def test_ollama_simulated_retry_http_error_marks_backend_unavailable() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    tools = [
        ToolSchema(
            name="web_search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
    ]
    native_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "", "thinking": "Need web context."},
            "done": True,
            "done_reason": "stop",
        }
    )
    simulated_error = httpx.HTTPStatusError(
        "EOF",
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        response=httpx.Response(500, request=httpx.Request("POST", "http://localhost:11434/api/chat"), text='{\"error\":\"EOF\"}'),
    )

    with patch.object(backend._client, "post", new_callable=AsyncMock, side_effect=[native_response, simulated_error]):
        with pytest.raises(RuntimeError, match="invalid tool-eligible turn|API error 500"):
            await backend.generate(messages=[Message(role="user", content="Search Mochi AI")], tools=tools, stream=False)

    assert backend.get_model_info().metadata["tool_call_mode"] == "unavailable"
```

```python
@pytest.mark.asyncio
async def test_ollama_probe_failure_from_native_mode_marks_backend_unavailable() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    failure_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "", "thinking": "Need a tool."},
            "done": True,
            "done_reason": "stop",
        }
    )

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=failure_response):
        result = await backend.probe_tool_calling()

    assert result is not None
    assert result["status"] == "thinking_only"
    assert backend.get_model_info().metadata["tool_call_mode"] == "unavailable"
```

```python
@pytest.mark.asyncio
async def test_ollama_failed_reprobe_after_validated_fallback_stays_in_simulated_mode() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    backend._tool_state.active_mode = "simulated_fallback"  # noqa: SLF001
    backend._tool_state.native_status = "native_tool_calls_missing"  # noqa: SLF001
    backend._tool_state.fallback_validation_status = "validated"  # noqa: SLF001
    failure_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "", "thinking": "Need a tool."},
            "done": True,
            "done_reason": "stop",
        }
    )

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=failure_response):
        result = await backend.probe_tool_calling()

    assert result is not None
    assert result["status"] == "thinking_only"
    assert backend.get_model_info().metadata["tool_call_mode"] == "simulated_fallback"
```

```python
@pytest.mark.asyncio
async def test_ollama_manual_probe_can_recover_from_unavailable_state() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    backend._tool_state.active_mode = "unavailable"  # noqa: SLF001
    backend._tool_state.native_status = "simulated_protocol_rejected"  # noqa: SLF001
    probe_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "probe-call-1", "function": {"name": "mochi_tool_probe", "arguments": {"value": "ok"}}}
                ],
            },
            "done": True,
            "done_reason": "tool_calls",
        }
    )

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=probe_response):
        result = await backend.probe_tool_calling()

    assert result is not None
    assert result["status"] == "supported"
    assert backend.get_model_info().metadata["tool_call_mode"] == "native"
```

```python
def test_ollama_supports_tool_calling_false_when_mode_is_unavailable() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    backend._tool_state.active_mode = "unavailable"  # noqa: SLF001
    assert backend.supports_tool_calling() is False
```

- [ ] **Step 2: Write the failing engine/API tests**

```python
@pytest.mark.asyncio
async def test_engine_preflight_probe_can_probe_ollama_when_status_unknown(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {"model": "ollama:test", "workspace_dir": str(tmp_path), "sessions_dir": str(tmp_path / "sessions"), "memory": {"db_path": str(tmp_path / "memory.db")}}
    )
    engine = AgentEngine(config)
    backend = FakeBackend(
        backend_type="ollama",
        metadata={"native_tool_calling_status": "unknown"},
        probe_result={"status": "supported", "metadata": {"tool_call_mode": "native"}},
    )
    plan = ToolExposurePlan(tool_names=["web_search"], matched_groups=["web"], limit=10)
    filtered = await engine._probe_tool_calling_before_exposure(backend, plan)  # noqa: SLF001
    assert backend.probe_calls == 1
    assert filtered.tool_names == ["web_search"]
```

```python
@pytest.mark.asyncio
async def test_engine_preflight_probe_retries_recoverable_fallback_state(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {"model": "ollama:test", "workspace_dir": str(tmp_path), "sessions_dir": str(tmp_path / "sessions"), "memory": {"db_path": str(tmp_path / "memory.db")}}
    )
    engine = AgentEngine(config)
    backend = FakeBackend(
        backend_type="ollama",
        metadata={"tool_call_mode": "simulated_fallback", "native_tool_calling_status": "native_tool_calls_missing"},
        probe_result={"status": "supported", "metadata": {"tool_call_mode": "native", "native_tool_calling_status": "supported"}},
    )
    plan = ToolExposurePlan(tool_names=["web_search"], matched_groups=["web"], limit=10)
    filtered = await engine._probe_tool_calling_before_exposure(backend, plan)  # noqa: SLF001
    assert backend.probe_calls == 1
    assert filtered.tool_names == ["web_search"]
```

```python
def test_models_probe_tool_calling_returns_post_probe_active_model_metadata(tmp_path: Path, monkeypatch: Any) -> None:
    config = MochiConfig.model_validate(
        {"model": "ollama:qwen2.5", "workspace_dir": str(tmp_path), "sessions_dir": str(tmp_path / "sessions"), "memory": {"db_path": str(tmp_path / "memory.db")}}
    )
    app, _fake_engine = _build_app()
    real_engine = AgentEngine(config)
    backend = FakeBackend(
        backend_type="ollama",
        metadata={"tool_call_mode": "unavailable", "native_tool_calling_status": "simulated_protocol_rejected"},
        probe_result={"status": "supported", "metadata": {"tool_call_mode": "native", "native_tool_calling_status": "supported"}},
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        real_engine._router._active = backend  # noqa: SLF001
        return backend

    async def fake_probe_active_tool_calling() -> dict[str, Any] | None:
        await backend.probe_tool_calling()
        return {"status": "supported", "message": "native structured tool calling succeeded"}

    monkeypatch.setattr(real_engine._router, "load", fake_load)  # noqa: SLF001
    monkeypatch.setattr(real_engine, "probe_active_tool_calling", fake_probe_active_tool_calling)
    app.state.agent_engine = real_engine
    with TestClient(app) as client:
        response = client.post("/v1/models/probe-tool-calling")
    assert response.status_code == 200
    payload = response.json()
    assert payload["probe"]["status"] == "supported"
    assert payload["active_model"]["metadata"]["tool_call_mode"] == "native"
    assert payload["active_model"]["metadata"]["native_tool_calling_status"] == "supported"
    assert payload["active_model"]["supports_tool_calling"] is True
```

- [ ] **Step 3: Run the new targeted tests and confirm they fail for the right reason**

Run:

```bash
python -m pytest -q tests/test_backends.py::test_ollama_retry_that_returns_only_thinking_raises_backend_error tests/test_backends.py::test_ollama_probe_reenables_native_mode_after_fallback tests/test_backends.py::test_ollama_simulated_retry_http_error_marks_backend_unavailable tests/test_backends.py::test_ollama_probe_failure_from_native_mode_marks_backend_unavailable tests/test_backends.py::test_ollama_failed_reprobe_after_validated_fallback_stays_in_simulated_mode tests/test_backends.py::test_ollama_manual_probe_can_recover_from_unavailable_state tests/test_backends.py::test_ollama_supports_tool_calling_false_when_mode_is_unavailable
python -m pytest -q tests/test_engine_phase2.py::test_engine_preflight_probe_can_probe_ollama_when_status_unknown tests/test_engine_phase2.py::test_engine_preflight_probe_retries_recoverable_fallback_state
python -m pytest -q tests/test_api_chat_models.py::test_models_probe_tool_calling_returns_post_probe_active_model_metadata
```

Expected:

- `tests/test_backends.py` fails because `retry_result.thinking` is currently accepted as success.
- `tests/test_engine_phase2.py` fails because engine probing is currently hard-coded to `openai_compat`.
- `tests/test_api_chat_models.py` either passes unchanged or exposes missing metadata updates that need to be added deliberately.

- [ ] **Step 4: Commit the red test baseline**

```bash
git add tests/test_backends.py tests/test_engine_phase2.py tests/test_api_chat_models.py
git commit -m "test: lock ollama tool-calling contract regressions"
```

### Task 2: Introduce Shared Tool-Calling State and Result Contract

**Files:**
- Create: `mochi/backends/tool_call_state.py`
- Create: `mochi/backends/tool_call_contract.py`
- Modify: `mochi/backends/base.py`
- Test: `tests/test_backends.py`

- [ ] **Step 1: Write the state/contract unit tests first**

```python
def test_validate_tool_turn_accepts_structured_tool_calls() -> None:
    result = GenerationResult(content="", thinking="plan", tool_calls=[ToolCall(id="1", name="web_search", arguments={})])
    verdict = validate_tool_turn_result(result=result, tools_requested=True)
    assert verdict.is_valid is True
    assert verdict.reason == "tool_calls"
```

```python
def test_validate_tool_turn_rejects_thinking_only_output() -> None:
    result = GenerationResult(content="", thinking="planning only", tool_calls=[])
    verdict = validate_tool_turn_result(result=result, tools_requested=True)
    assert verdict.is_valid is False
    assert verdict.reason == "thinking_only"
```

- [ ] **Step 2: Run the new validator tests and verify they fail**

Run:

```bash
python -m pytest tests/test_backends.py -q -k "validate_tool_turn"
```

Expected: `ImportError` or `NameError` for missing validator/state helpers.

- [ ] **Step 3: Implement the shared state and validator**

```python
@dataclass
class ToolCallingState:
    active_mode: Literal["native", "simulated_fallback", "unavailable"] = "native"
    native_status: str = "unknown"
    fallback_validation_status: Literal["not_attempted", "validated", "rejected"] = "not_attempted"

    def enter_simulated(self, status: str) -> bool:
        changed = self.active_mode != "simulated_fallback" or self.native_status != status
        self.active_mode = "simulated_fallback"
        self.native_status = status
        self.fallback_validation_status = "not_attempted"
        return changed

    def validate_simulated(self) -> bool:
        changed = self.fallback_validation_status != "validated"
        self.fallback_validation_status = "validated"
        return changed

    def recover_native(self, status: str) -> bool:
        changed = self.active_mode != "native" or self.native_status != status
        self.active_mode = "native"
        self.native_status = status
        self.fallback_validation_status = "not_attempted"
        return changed

    def mark_unavailable(self, status: str) -> bool:
        changed = self.active_mode != "unavailable" or self.native_status != status
        self.active_mode = "unavailable"
        self.native_status = status
        self.fallback_validation_status = "rejected"
        return changed
```

```python
@dataclass(frozen=True)
class ToolTurnVerdict:
    is_valid: bool
    reason: Literal["tool_calls", "content", "thinking_only", "empty"]

def validate_tool_turn_result(*, result: GenerationResult, tools_requested: bool) -> ToolTurnVerdict:
    if result.tool_calls:
        return ToolTurnVerdict(is_valid=True, reason="tool_calls")
    if result.content.strip():
        return ToolTurnVerdict(is_valid=True, reason="content")
    if tools_requested and result.thinking.strip():
        return ToolTurnVerdict(is_valid=False, reason="thinking_only")
    return ToolTurnVerdict(is_valid=not tools_requested, reason="empty")
```

Also add a default no-op contract to `BaseLLMBackend`:

```python
async def probe_tool_calling(self) -> dict[str, Any] | None:
    return None
```

- [ ] **Step 4: Run the validator tests and backend smoke tests**

Run:

```bash
python -m pytest tests/test_backends.py -q -k "validate_tool_turn or ollama"
```

Expected: new validator tests pass; Ollama tests still fail until Task 3 refactors the adapter.

- [ ] **Step 5: Commit the shared contract/state layer**

```bash
git add mochi/backends/base.py mochi/backends/tool_call_state.py mochi/backends/tool_call_contract.py tests/test_backends.py
git commit -m "feat: add shared tool-calling state and result contract"
```

### Task 3: Extract a Shared Simulated Tool Protocol and Refactor Ollama Onto It

**Files:**
- Create: `mochi/backends/simulated_tool_protocol.py`
- Modify: `mochi/backends/tool_call_simulator.py`
- Modify: `mochi/backends/ollama.py`
- Test: `tests/test_backends.py`

- [ ] **Step 1: Write the failing protocol-helper tests**

```python
def test_simulated_tool_protocol_flattens_prior_tool_messages() -> None:
    protocol = SimulatedToolProtocol(ToolCallSimulator())
    tools = [
        ToolSchema(
            name="web_search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
    ]
    messages = [
        Message(role="assistant", content="", tool_calls=[ToolCall(id="1", name="web_search", arguments={"query": "Mochi"})]),
        Message(role="tool", content="Tool web_search result:\nfound", tool_call_id="1", name="web_search"),
    ]
    prepared = protocol.prepare_messages(messages=messages, tools=tools)
    assert any(message.role == "assistant" and "Tool request: web_search" in message.content for message in prepared)
    assert any(message.role == "user" and message.content.startswith("Tool web_search result:") for message in prepared)
```

- [ ] **Step 2: Run the protocol tests and confirm the helper is missing**

Run:

```bash
python -m pytest tests/test_backends.py -q -k "simulated_tool_protocol"
```

Expected: failure because `SimulatedToolProtocol` does not exist yet.

- [ ] **Step 3: Implement the helper and refactor Ollama**

Implementation requirements:

- Replace `_tool_calling_enabled` in `OllamaBackend` with `self._tool_state = ToolCallingState()`.
- Replace the current always-true [mochi/backends/ollama.py](/H:/_python/agent_mochi/mochi/backends/ollama.py:52) exposure with a state-based `supports_tool_calling()` implementation.
- Replace [mochi/backends/ollama.py](/H:/_python/agent_mochi/mochi/backends/ollama.py:362) with contract-based fallback conditions instead of raw `thinking` checks.
- Replace [mochi/backends/ollama.py](/H:/_python/agent_mochi/mochi/backends/ollama.py:392) with state transitions that:
  - record diagnostics on `native -> simulated_fallback`
  - keep `simulated_fallback` marked as unvalidated until a retry succeeds
  - reject invalid simulated retries
  - move to `unavailable` when the simulated protocol itself errors or returns invalid output
  - allow `simulated_fallback -> native` recovery after a successful probe
- Keep [mochi/backends/ollama.py](/H:/_python/agent_mochi/mochi/backends/ollama.py:218) responsible only for Ollama response parsing, not policy decisions.

Suggested code shape:

```python
verdict = validate_tool_turn_result(result=result, tools_requested=bool(tools))
if use_native_tools and verdict.reason == "thinking_only":
    self._tool_state.enter_simulated("native_tool_calls_missing")
retry_result = await self._blocking_generate(retry_payload, tools=tools, use_native_tools=False)
retry_verdict = validate_tool_turn_result(result=retry_result, tools_requested=bool(tools))
if retry_verdict.is_valid:
    self._tool_state.validate_simulated()
else:
    self._tool_state.mark_unavailable("simulated_protocol_rejected")
    raise BackendRequestError(
        "Ollama returned an invalid tool-eligible turn.",
        metadata={"backend_name": "ollama", "tool_turn_reason": retry_verdict.reason, "model": self.model},
    )
```

- [ ] **Step 4: Add and implement `OllamaBackend.probe_tool_calling()`**

Use a single safe probe tool and a direct native request path. Do not probe through the same fallback loop and do not run simulated retry logic inside `probe_tool_calling()`.

```python
async def probe_tool_calling(self) -> dict[str, Any] | None:
    probe_tool = ToolSchema(
        name="mochi_tool_probe",
        description="Diagnostic probe tool. Echo the requested value.",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
    )
    payload = self._build_request_payload(
        messages=[Message(role="user", content="Call mochi_tool_probe with value='ok'.")],
        tools=[probe_tool],
        options={"temperature": 0.0, "num_predict": 128, "top_p": 1.0, "top_k": 0, "repeat_penalty": 1.0},
        stream=False,
        think_value=None,
    )
    result = await self._blocking_generate(payload, tools=[probe_tool], use_native_tools=True)
    verdict = validate_tool_turn_result(result=result, tools_requested=True)
    if verdict.reason == "tool_calls":
        return self._record_probe_success(status="supported", message="native structured tool calling succeeded")
    return self._record_probe_failure(status=verdict.reason, message="probe did not return structured tool calls")
```

Success criteria:

- successful native probe sets `tool_call_mode` to `native`
- failed native probe while `active_mode == "native"` and no validated fallback exists sets `tool_call_mode` to `unavailable`
- failed native probe while `active_mode == "simulated_fallback"` and `fallback_validation_status == "validated"` keeps the backend in `simulated_fallback` and records recovery diagnostics
- manual probe from `unavailable` is allowed and can recover back to `native`
- invalid simulated retry during normal generation, not during probing, sets `tool_call_mode` to `unavailable`
- probe never treats `thinking`-only output as supported
- automatic preflight reprobes only recoverable states, while terminal `unavailable` waits for manual operator action or backend recreation

- [ ] **Step 5: Run the Ollama backend tests**

Run:

```bash
python -m pytest -q tests/test_backends.py::test_ollama_retry_that_returns_only_thinking_raises_backend_error tests/test_backends.py::test_ollama_probe_reenables_native_mode_after_fallback tests/test_backends.py::test_ollama_simulated_retry_http_error_marks_backend_unavailable tests/test_backends.py::test_ollama_probe_failure_from_native_mode_marks_backend_unavailable tests/test_backends.py::test_ollama_failed_reprobe_after_validated_fallback_stays_in_simulated_mode tests/test_backends.py::test_ollama_manual_probe_can_recover_from_unavailable_state tests/test_backends.py::test_ollama_supports_tool_calling_false_when_mode_is_unavailable
```

Expected: all Ollama tests pass, including the new thinking-only regression and recovery tests.

- [ ] **Step 6: Commit the Ollama refactor**

```bash
git add mochi/backends/tool_call_simulator.py mochi/backends/simulated_tool_protocol.py mochi/backends/ollama.py tests/test_backends.py
git commit -m "refactor: harden ollama tool-calling state and fallback flow"
```

### Task 4: Generalize Engine Probe Behavior and Stabilize API Diagnostics

**Files:**
- Modify: `mochi/agents/engine.py`
- Modify: `tests/test_engine_phase2.py`
- Modify: `tests/test_api_chat_models.py`

- [ ] **Step 1: Write the failing engine probe tests**

```python
@pytest.mark.asyncio
async def test_engine_preflight_probe_calls_backend_probe_when_status_unknown(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {"model": "ollama:test", "workspace_dir": str(tmp_path), "sessions_dir": str(tmp_path / "sessions"), "memory": {"db_path": str(tmp_path / "memory.db")}}
    )
    engine = AgentEngine(config)
    backend = FakeBackend(
        backend_type="ollama",
        metadata={"native_tool_calling_status": "unknown"},
        probe_result={"status": "supported", "metadata": {"tool_call_mode": "native"}},
    )
    plan = ToolExposurePlan(tool_names=["web_search"], matched_groups=["web"], limit=10)
    filtered = await engine._probe_tool_calling_before_exposure(backend, plan)  # noqa: SLF001
    assert backend.probe_calls == 1
    assert filtered.tool_names == ["web_search"]
```

```python
@pytest.mark.asyncio
async def test_engine_preflight_probe_disables_tools_when_backend_reports_unavailable(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {"model": "ollama:test", "workspace_dir": str(tmp_path), "sessions_dir": str(tmp_path / "sessions"), "memory": {"db_path": str(tmp_path / "memory.db")}}
    )
    engine = AgentEngine(config)
    backend = FakeBackend(
        backend_type="ollama",
        metadata={"native_tool_calling_status": "unknown"},
        probe_result={"status": "all_tool_protocols_rejected_by_provider", "metadata": {"tool_call_mode": "unavailable"}},
    )
    plan = ToolExposurePlan(tool_names=["web_search"], matched_groups=["web"], limit=10)
    filtered = await engine._probe_tool_calling_before_exposure(backend, plan)  # noqa: SLF001
    assert backend.probe_calls == 1
    assert filtered.tool_names == []
    assert filtered.limit == 0
```

- [ ] **Step 2: Run the engine/API tests and verify current behavior is too narrow**

Run:

```bash
python -m pytest tests/test_engine_phase2.py -q -k "preflight_probe"
python -m pytest tests/test_api_chat_models.py -q -k "probe_tool_calling"
```

Expected: engine tests fail because probing is currently hard-coded to [mochi/agents/engine.py](/H:/_python/agent_mochi/mochi/agents/engine.py:923) `openai_compat`.

- [ ] **Step 3: Implement backend-agnostic probe gating**

Refactor [mochi/agents/engine.py](/H:/_python/agent_mochi/mochi/agents/engine.py:904) so it:

- skips probing only when the exposure plan is empty
- skips probing when backend metadata already declares a terminal state
- re-probes recoverable fallback states such as `simulated_fallback + native_tool_calls_missing`
- probes any backend that exposes a callable `probe_tool_calling()`
- disables tools only when refreshed metadata reports `tool_call_mode == "unavailable"` or `tool_calling_blocked is True`

Keep the gate capability-based, not provider-name-based.

- [ ] **Step 4: Update probe endpoint expectations**

Ensure `/v1/models/probe-tool-calling` keeps returning:

- `probe.status`
- updated active-model metadata
- stable `tool_call_mode`
- stable `native_tool_calling_status`
- stable `supports_tool_calling`

Use a real uninitialized `AgentEngine` path in at least one route test so the serialized `active_model` reflects post-probe backend state, not only fake pre-seeded metadata.

- [ ] **Step 5: Run engine/API regression tests**

Run:

```bash
python -m pytest tests/test_engine_phase2.py -q -k "probe"
python -m pytest -q tests/test_api_chat_models.py::test_models_probe_tool_calling_returns_post_probe_active_model_metadata
```

Expected: pass.

- [ ] **Step 6: Commit the engine/API integration**

```bash
git add mochi/agents/engine.py tests/test_engine_phase2.py tests/test_api_chat_models.py
git commit -m "refactor: generalize backend tool-calling probes"
```

### Task 5: Align OpenAI-Compatible Backends to the Shared Helpers and Run the Matrix

**Files:**
- Modify: `mochi/backends/openai_compat.py`
- Modify: `tests/test_backends.py`
- Modify: `tests/test_openai_compat_backend.py`

- [ ] **Step 1: Write the alignment tests before changing behavior**

```python
@pytest.mark.asyncio
async def test_openai_compat_shared_contract_still_accepts_structured_tool_calls() -> None:
    backend = OpenAICompatBackend(base_url="http://localhost:8000/v1", model="google/gemma-4-26B-A4B-it", provider="vllm")
    response = _mock_response(
        {
            "model": "google/gemma-4-26B-A4B-it",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "call-1", "function": {"name": "web_search", "arguments": '{"query":"Mochi AI"}'}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )
    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=response):
        result = await backend.generate(
            messages=[Message(role="user", content="Search Mochi AI")],
            tools=[ToolSchema(name="web_search", description="Search the web", parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})],
            stream=False,
        )
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "web_search"
```

```python
@pytest.mark.asyncio
async def test_openai_compat_probe_recovery_keeps_native_mode_after_success() -> None:
    backend = OpenAICompatBackend(base_url="http://localhost:8000/v1", model="google/gemma-4-26B-A4B-it", provider="vllm")
    backend._tool_state.active_mode = "simulated_fallback"  # noqa: SLF001
    backend._tool_state.native_status = "rejected_missing_parser"  # noqa: SLF001
    success_response = _mock_response(
        {
            "model": "google/gemma-4-26B-A4B-it",
            "choices": [
                {
                    "message": {"role": "assistant", "tool_calls": [{"id": "probe-call-1", "function": {"name": "mochi_tool_probe", "arguments": '{"value":"ok"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )
    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=success_response):
        result = await backend.probe_tool_calling()
    assert result is not None
    assert result["status"] == "supported"
    assert backend.get_model_info().metadata["tool_call_mode"] == "native"
```

- [ ] **Step 2: Run the OpenAI-compatible backend tests**

Run:

```bash
python -m pytest -q tests/test_backends.py::test_openai_compat_shared_contract_still_accepts_structured_tool_calls tests/test_backends.py::test_openai_compat_probe_recovery_keeps_native_mode_after_success
python -m pytest -q tests/test_openai_compat_backend.py -k "probe or fallback"
```

Expected: any failures should be due to helper wiring, not intentional semantic changes.

- [ ] **Step 3: Replace duplicated helper logic with the shared modules**

Refactor `openai_compat.py` to use:

- `ToolCallingState` for native/simulated/unavailable transitions
- `SimulatedToolProtocol` for retry payload preparation where feasible
- `validate_tool_turn_result(...)` when interpreting simulated outputs

Do not regress:

- responses vs chat-completions transport fallback
- responses alias detection
- provider-specific diagnostic messages in [mochi/backends/openai_compat.py](/H:/_python/agent_mochi/mochi/backends/openai_compat.py:2017)

- [ ] **Step 4: Run the non-regression matrix**

Run:

```bash
python -m pytest -q tests/test_backends.py::test_ollama_retry_that_returns_only_thinking_raises_backend_error tests/test_backends.py::test_ollama_probe_reenables_native_mode_after_fallback tests/test_backends.py::test_ollama_simulated_retry_http_error_marks_backend_unavailable tests/test_backends.py::test_ollama_probe_failure_from_native_mode_marks_backend_unavailable tests/test_backends.py::test_ollama_failed_reprobe_after_validated_fallback_stays_in_simulated_mode tests/test_backends.py::test_ollama_manual_probe_can_recover_from_unavailable_state tests/test_backends.py::test_ollama_supports_tool_calling_false_when_mode_is_unavailable tests/test_backends.py::test_openai_compat_shared_contract_still_accepts_structured_tool_calls tests/test_backends.py::test_openai_compat_probe_recovery_keeps_native_mode_after_success
python -m pytest tests/test_tool_call_simulator.py -q
python -m pytest tests/test_gguf_backend_runtime.py -q -k "flattened_text"
python -m pytest tests/test_safetensors_backend_runtime.py -q
python -m pytest tests/test_openai_compat_backend.py -q
python -m pytest tests/test_engine_phase2.py -q -k "probe"
python -m pytest tests/test_api_chat_models.py -q -k "gemini or sglang or tensorrt_llm or probe_tool_calling"
```

Expected:

- Ollama passes
- OpenAI-compatible tests pass
- `gemini`, `sglang`, and `tensorrt_llm` API tests pass unchanged

- [ ] **Step 5: Commit the shared-backend alignment**

```bash
git add mochi/backends/openai_compat.py tests/test_backends.py tests/test_openai_compat_backend.py tests/test_engine_phase2.py tests/test_api_chat_models.py
git commit -m "refactor: unify tool-calling helpers across backends"
```

## Non-Goals

- Do not redesign `Message`, `ToolCall`, or `GenerationResult` wire types.
- Do not add provider-specific heuristics to the ReAct loop.
- Do not redesign tool exposure policy beyond making probe gating backend-agnostic.
- Do not broaden this into VLM-specific multimodal transport work in the same branch.

## Final Verification Checklist

- [ ] A tool-eligible backend turn is considered successful only if it yields structured `tool_calls` or visible final `content`.
- [ ] `thinking`-only output is diagnostic data, never a completed tool turn.
- [ ] Ollama can recover from `simulated_fallback` back to `native` without reconstructing the backend instance.
- [ ] Manual `/v1/models/probe-tool-calling` can recover an `unavailable` backend without allowing automatic preflight reprobes from that same terminal state.
- [ ] Engine preflight probing is capability-based rather than `backend_type == "openai_compat"`.
- [ ] `Gemini`, `SGLang`, and `TensorRT-LLM` coverage still passes through the `openai_compat` route.
- [ ] Probe/API diagnostics expose stable mode/status metadata after refactor.
