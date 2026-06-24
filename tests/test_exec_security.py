"""Exec command security and approval primitive tests."""

from __future__ import annotations

from mochi.config.schema import MochiConfig
from mochi.runtime.approvals import InMemoryApprovalStore, PersistentApprovalStore
from mochi.utils.command_security import CommandSecurityPolicy
from mochi.utils.security import explain_unsafe_shell_command, is_safe_command


def _allow_rule(*tokens: str, shells: list[str] | None = None) -> dict[str, object]:
    return {
        "tokens": list(tokens),
        "decision": "allow",
        "match": "exact",
        "shells": list(shells or []),
    }


def test_command_security_allows_persisted_command_rule() -> None:
    policy = CommandSecurityPolicy(command_rules=[_allow_rule("echo", "hello")])
    result = policy.classify("echo hello")
    assert result.action == "allow"
    assert result.rule_id == "persisted_command_rule"
    assert result.parsed_tokens == ("echo", "hello")


def test_command_security_unknown_command_now_requests_approval() -> None:
    policy = CommandSecurityPolicy(command_rules=[_allow_rule("echo")])
    result = policy.classify("git status")
    assert result.action == "ask"
    assert result.rule_id == "unknown_requires_approval"
    assert result.parsed_tokens == ("git", "status")


def test_command_security_denies_shell_chaining_redirection_and_subshell() -> None:
    policy = CommandSecurityPolicy(command_rules=[_allow_rule("echo")])
    assert policy.classify("echo hi && whoami").rule_id == "shell_chaining"
    assert policy.classify("echo hi > out.txt").rule_id == "shell_chaining"
    assert policy.classify("echo $(whoami)").rule_id == "subshell"


def test_command_security_denies_interpreter_inline_eval() -> None:
    policy = CommandSecurityPolicy()
    result = policy.classify('python -c "print(1)"')
    assert result.action == "deny"
    assert result.rule_id == "interpreter_inline_eval"


def test_command_security_denies_powershell_invoke_expression_and_encoded_command() -> None:
    policy = CommandSecurityPolicy(allow_dangerous_interpreters=True)
    invoke_expr = policy.classify("pwsh -Command Invoke-Expression whoami")
    encoded = policy.classify("powershell -EncodedCommand ZQBjAGgAbwAgAGgAaQA=")
    assert invoke_expr.action == "deny"
    assert invoke_expr.rule_id == "powershell_invoke_expression"
    assert encoded.action == "deny"
    assert encoded.rule_id == "powershell_encoded_command"


def test_command_security_allows_read_only_windows_browse_commands() -> None:
    policy = CommandSecurityPolicy(allow_dangerous_interpreters=True)

    cmd_result = policy.classify("cmd /c dir")
    powershell_result = policy.classify(
        "powershell -Command Get-ChildItem -Path src | Select-String -Pattern TODO"
    )
    provider_result = policy.classify(
        "Get-ChildItem -Path src | Select-String -Pattern TODO",
        shell="powershell",
    )

    assert cmd_result.action == "allow"
    assert cmd_result.rule_id == "cmd_read_only"
    assert powershell_result.action == "allow"
    assert powershell_result.rule_id == "powershell_read_only"
    assert provider_result.action == "allow"
    assert provider_result.rule_id == "powershell_read_only"


def test_command_security_windows_read_only_commands_still_enforce_workspace_paths() -> None:
    policy = CommandSecurityPolicy(
        workspace_dir="H:/_python/agent_mochi",
        allow_dangerous_interpreters=True,
    )

    cmd_result = policy.classify(r"cmd /c type ..\secret.txt")
    powershell_result = policy.classify(r"powershell -Command Get-Content ..\secret.txt")

    assert cmd_result.action == "deny"
    assert cmd_result.rule_id == "workspace_escape"
    assert powershell_result.action == "deny"
    assert powershell_result.rule_id == "workspace_escape"


def test_command_security_powershell_pipeline_allows_quoted_pipe_literals() -> None:
    policy = CommandSecurityPolicy(allow_dangerous_interpreters=True)

    result = policy.classify(
        'powershell -Command Get-ChildItem -Path src | Select-String -Pattern "a|b"'
    )

    assert result.action == "allow"
    assert result.rule_id == "powershell_read_only"


def test_command_security_denies_powershell_chaining_even_when_read_only_prefix_matches() -> None:
    policy = CommandSecurityPolicy(allow_dangerous_interpreters=True)

    and_result = policy.classify("powershell -Command Get-ChildItem && Remove-Item foo")
    semicolon_result = policy.classify("powershell -Command Get-ChildItem ; Remove-Item foo")

    assert and_result.action == "deny"
    assert and_result.rule_id == "powershell_chaining"
    assert semicolon_result.action == "deny"
    assert semicolon_result.rule_id == "powershell_chaining"


def test_command_security_denies_windows_spawns_and_powershell_writes() -> None:
    policy = CommandSecurityPolicy(allow_dangerous_interpreters=True)

    cmd_spawn = policy.classify("cmd /c powershell -Command Get-ChildItem")
    ps_spawn = policy.classify("powershell -Command Start-Process notepad.exe")
    ps_write = policy.classify("powershell -Command Set-Content notes.txt hi")

    assert cmd_spawn.action == "deny"
    assert cmd_spawn.rule_id == "cmd_blocked_payload"
    assert ps_spawn.action == "deny"
    assert ps_spawn.rule_id == "powershell_blocked_cmdlet"
    assert ps_write.action == "deny"
    assert ps_write.rule_id == "powershell_blocked_cmdlet"


def test_command_security_cmd_c_classification() -> None:
    policy = CommandSecurityPolicy(allow_dangerous_interpreters=True)
    ask_result = policy.classify("cmd /c dir")
    deny_result = policy.classify("cmd /c dir && whoami")
    assert ask_result.action == "allow"
    assert ask_result.rule_id == "cmd_read_only"
    assert deny_result.action == "deny"
    assert deny_result.rule_id == "cmd_high_risk_chaining"


def test_command_security_saved_rules_apply_inside_windows_shell_branches() -> None:
    powershell_policy = CommandSecurityPolicy(
        command_rules=[_allow_rule("Get-Process", shells=["powershell"])],
        allow_dangerous_interpreters=True,
    )
    cmd_policy = CommandSecurityPolicy(
        command_rules=[_allow_rule("cmd", "/c", "more", "notes.txt")],
        allow_dangerous_interpreters=True,
    )

    powershell_result = powershell_policy.classify("Get-Process", shell="powershell")
    cmd_result = cmd_policy.classify("cmd /c more notes.txt")

    assert powershell_result.action == "allow"
    assert powershell_result.rule_id == "persisted_command_rule"
    assert cmd_result.action == "allow"
    assert cmd_result.rule_id == "persisted_command_rule"


def test_command_security_denies_heredoc_sensitive_path_workspace_escape_and_path_override() -> None:
    policy = CommandSecurityPolicy(
        command_rules=[_allow_rule("cat")],
        workspace_dir="H:/_python/agent_mochi",
    )
    assert policy.classify("cat <<EOF").rule_id == "heredoc"
    assert policy.classify("cat ../secrets.txt").rule_id == "workspace_escape"
    assert policy.classify("cat ~/.ssh/id_rsa").rule_id == "sensitive_path"
    assert policy.classify("PATH=/tmp/bin cat notes.txt").rule_id == "env_path_override"


def test_command_security_denies_workspace_escape_when_executable_token_is_a_path() -> None:
    policy = CommandSecurityPolicy(
        workspace_dir="H:/_python/agent_mochi",
        allow_dangerous_interpreters=True,
    )

    direct_result = policy.classify("../outside-script.sh")
    powershell_result = policy.classify("powershell -Command ../outside-script.ps1")
    cmd_result = policy.classify("cmd /c ../outside-script.bat")

    assert direct_result.action == "deny"
    assert direct_result.rule_id == "workspace_escape"
    assert powershell_result.action == "deny"
    assert powershell_result.rule_id == "workspace_escape"
    assert cmd_result.action == "deny"
    assert cmd_result.rule_id == "workspace_escape"


def test_command_security_asks_for_non_allowlisted_env_override() -> None:
    policy = CommandSecurityPolicy(command_rules=[_allow_rule("echo", "hi")])
    result = policy.classify("FOO=bar echo hi")
    assert result.action == "ask"
    assert result.rule_id == "env_override"
    assert result.parsed_tokens == ("FOO=bar", "echo", "hi")


def test_legacy_shell_security_keeps_pwsh_denied_without_override() -> None:
    assert is_safe_command("pwsh", ["pwsh"]) is False
    assert explain_unsafe_shell_command("pwsh", ["pwsh"]) == "Command matched a protected shell policy."


def test_legacy_shell_security_supports_explicit_dangerous_override() -> None:
    allowlist = ["__allow_dangerous_shells__", "pwsh"]
    assert is_safe_command("pwsh", allowlist) is False
    assert explain_unsafe_shell_command("pwsh", allowlist) == "PowerShell command is blocked by policy: pwsh."


def test_in_memory_approval_store_roundtrip() -> None:
    store = InMemoryApprovalStore()
    created = store.create(
        approval_id="approval-1",
        command="cmd /c dir",
        shell="cmd",
        scope="dangerous_command",
        reason="requires review",
    )
    assert created.status == "pending"
    assert created.resolved_at is None
    assert store.get("approval-1") is not None
    assert [item.approval_id for item in store.list(status="pending")] == ["approval-1"]

    resolved = store.resolve("approval-1", decision="approve_once", reason="approved")
    assert resolved is not None
    assert resolved.status == "approved_once"
    assert resolved.reason == "approved"
    assert resolved.resolved_at is not None


def test_persistent_approval_store_roundtrip(tmp_path) -> None:
    db_path = tmp_path / "exec-approvals.db"
    store = PersistentApprovalStore(db_path)
    created = store.create(
        approval_id="approval-persistent-1",
        command="cmd /c dir",
        shell="cmd",
        scope="dangerous_command",
        reason="requires review",
        metadata={"policy_state": "ask"},
        command_payload={"command": "cmd /c dir", "shell": "cmd"},
    )
    assert created.status == "pending"
    assert created.resolved_at is None

    reopened = PersistentApprovalStore(db_path)
    pending = reopened.get("approval-persistent-1")
    assert pending is not None
    assert pending.status == "pending"
    assert pending.metadata["policy_state"] == "ask"
    assert pending.command_payload == {"command": "cmd /c dir", "shell": "cmd"}

    resolved = reopened.resolve(
        "approval-persistent-1",
        decision="approve_once",
        reason="approved",
        execution_result={"status": "completed", "stdout": "ok"},
    )
    assert resolved is not None
    assert resolved.status == "approved_once"
    assert resolved.execution_result == {"status": "completed", "stdout": "ok"}

    reloaded = PersistentApprovalStore(db_path)
    approved = reloaded.get("approval-persistent-1")
    assert approved is not None
    assert approved.status == "approved_once"
    assert approved.reason == "approved"
    assert approved.execution_result == {"status": "completed", "stdout": "ok"}


def test_exec_security_config_defaults_and_roundtrip() -> None:
    cfg = MochiConfig()
    assert cfg.security.require_approval_for_exec is True
    assert cfg.security.agent_run_default_max_wall_clock_sec is None
    assert cfg.security.agent_run_default_heartbeat_timeout_sec is None
    assert cfg.security.agent_run_default_checkpoint_interval_steps == 1
    assert cfg.security.agent_run_default_max_subagent_failures_per_role == 2
    assert cfg.security.agent_run_default_on_budget_exhausted == "pause"
    assert cfg.security.agent_run_default_on_subagent_disconnect == "retry_then_degrade"
    assert cfg.security.exec_allowed_env_vars == []
    assert cfg.security.exec_default_shell == "auto"
    assert cfg.security.exec_session_output_limit == 8000
    assert cfg.security.exec_default_timeout_sec == 30

    payload = cfg.model_dump(mode="python")
    payload["security"]["require_approval_for_exec"] = False
    payload["security"]["agent_run_default_max_wall_clock_sec"] = 1800
    payload["security"]["agent_run_default_heartbeat_timeout_sec"] = 45
    payload["security"]["agent_run_default_checkpoint_interval_steps"] = 3
    payload["security"]["agent_run_default_max_subagent_failures_per_role"] = 4
    payload["security"]["agent_run_default_on_budget_exhausted"] = "finalize_partial"
    payload["security"]["agent_run_default_on_subagent_disconnect"] = "pause"
    payload["security"]["exec_allowed_env_vars"] = ["LANG", "HOME"]
    payload["security"]["exec_default_shell"] = "pwsh"
    payload["security"]["exec_session_output_limit"] = 16384
    payload["security"]["exec_default_timeout_sec"] = 90
    parsed = MochiConfig.model_validate(payload)
    assert parsed.security.require_approval_for_exec is False
    assert parsed.security.agent_run_default_max_wall_clock_sec == 1800
    assert parsed.security.agent_run_default_heartbeat_timeout_sec == 45
    assert parsed.security.agent_run_default_checkpoint_interval_steps == 3
    assert parsed.security.agent_run_default_max_subagent_failures_per_role == 4
    assert parsed.security.agent_run_default_on_budget_exhausted == "finalize_partial"
    assert parsed.security.agent_run_default_on_subagent_disconnect == "pause"
    assert parsed.security.exec_allowed_env_vars == ["LANG", "HOME"]
    assert parsed.security.exec_default_shell == "pwsh"
    assert parsed.security.exec_session_output_limit == 16384
    assert parsed.security.exec_default_timeout_sec == 90
