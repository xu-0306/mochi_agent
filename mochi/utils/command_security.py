"""Structured command security policy for exec/shell classification."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from mochi.utils.command_path_policy import classify_path_risks
from mochi.utils.command_policy_rules import (
    DISALLOWED_INTERPRETERS,
    ENV_OVERRIDE_RE,
    HEREDOC_RE,
    INTERPRETER_INLINE_EVAL,
    PATH_ENV_KEYS,
    POWERSHELL_BLOCKED_CMDLETS,
    POWERSHELL_ENV_ASSIGN_RE,
    POWERSHELL_READ_ONLY_CMDLETS,
    SHELL_CHAINING_RE,
    SHELL_REDIRECTION_RE,
    SUBSHELL_RE,
    WINDOWS_BLOCKED_CMD_PAYLOADS,
    WINDOWS_READ_ONLY_COMMANDS,
    contains_unquoted_operator,
    command_rule_matches,
    extract_powershell_payload,
    normalize_command_name,
    split_powershell_pipeline,
    tokenize_command,
)

CommandSecurityAction = Literal["allow", "ask", "deny"]


@dataclass(frozen=True)
class CommandSecurityResult:
    """Command classification result."""

    action: CommandSecurityAction
    reason: str
    rule_id: str
    parsed_tokens: tuple[str, ...] = ()


def _with_parsed_tokens(
    result: CommandSecurityResult,
    *,
    tokens: list[str],
) -> CommandSecurityResult:
    if result.parsed_tokens:
        return result
    return replace(result, parsed_tokens=tuple(tokens))


class CommandSecurityPolicy:
    """Classify command strings into allow/ask/deny decisions."""

    def __init__(
        self,
        *,
        command_rules: list[dict[str, object]] | None = None,
        workspace_dir: str | Path | None = None,
        allowed_env_vars: list[str] | None = None,
        allow_dangerous_interpreters: bool = False,
    ) -> None:
        self._command_rules = [dict(item) for item in (command_rules or []) if isinstance(item, dict)]
        self._workspace = Path(workspace_dir).expanduser().resolve(strict=False) if workspace_dir else None
        self._allowed_env_vars = {item.upper() for item in (allowed_env_vars or []) if item}
        self._allow_dangerous_interpreters = allow_dangerous_interpreters

    def classify(self, command: str, *, shell: str | None = None, env: dict[str, str] | None = None) -> CommandSecurityResult:
        stripped = command.strip()
        if not stripped:
            return CommandSecurityResult(
                action="deny",
                reason="Command must not be empty.",
                rule_id="empty_command",
            )

        if HEREDOC_RE.search(stripped):
            return CommandSecurityResult(
                action="deny",
                reason="Heredoc-like command input is not allowed.",
                rule_id="heredoc",
            )

        try:
            tokens = tokenize_command(stripped)
        except ValueError:
            return CommandSecurityResult(
                action="deny",
                reason="Command could not be parsed safely.",
                rule_id="parse_error",
            )
        if not tokens:
            return CommandSecurityResult(
                action="deny",
                reason="Command must not be empty.",
                rule_id="empty_command",
            )

        program = normalize_command_name(tokens[0])

        cmd_result = self._classify_cmd(tokens=tokens, shell=shell, program=program)
        if cmd_result is not None:
            return _with_parsed_tokens(cmd_result, tokens=tokens)

        env_result = self._classify_env_overrides(tokens=tokens, env=env)
        if env_result is not None:
            return _with_parsed_tokens(env_result, tokens=tokens)

        interpreter_result = self._classify_interpreter_risks(tokens=tokens, program=program)
        if interpreter_result is not None:
            return _with_parsed_tokens(interpreter_result, tokens=tokens)

        ps_result = self._classify_powershell(
            command=stripped,
            tokens=tokens,
            shell=shell,
            program=program,
        )
        if ps_result is not None:
            return _with_parsed_tokens(ps_result, tokens=tokens)

        if SHELL_CHAINING_RE.search(stripped) or SHELL_REDIRECTION_RE.search(stripped):
            return CommandSecurityResult(
                action="deny",
                reason="Shell chaining or redirection syntax is not allowed.",
                rule_id="shell_chaining",
                parsed_tokens=tuple(tokens),
            )

        if SUBSHELL_RE.search(stripped):
            return CommandSecurityResult(
                action="deny",
                reason="Subshell or command-substitution syntax is not allowed.",
                rule_id="subshell",
                parsed_tokens=tuple(tokens),
            )

        path_result = self._classify_path_risks(tokens=tokens)
        if path_result is not None:
            return _with_parsed_tokens(path_result, tokens=tokens)

        if self._matches_persisted_rule(tokens=tokens, shell=shell):
            return CommandSecurityResult(
                action="allow",
                reason="Command is allowed by a persisted command rule.",
                rule_id="persisted_command_rule",
                parsed_tokens=tuple(tokens),
            )

        return CommandSecurityResult(
            action="ask",
            reason="Command requires approval because it is not covered by a read-only policy or persisted command rule.",
            rule_id="unknown_requires_approval",
            parsed_tokens=tuple(tokens),
        )

    def _classify_env_overrides(self, *, tokens: list[str], env: dict[str, str] | None) -> CommandSecurityResult | None:
        for token in tokens:
            if not ENV_OVERRIDE_RE.match(token):
                break
            key = token.split("=", 1)[0].upper()
            if key in PATH_ENV_KEYS:
                return CommandSecurityResult(
                    action="deny",
                    reason=f"{key} override is not allowed.",
                    rule_id="env_path_override",
                )
            if key not in self._allowed_env_vars:
                return CommandSecurityResult(
                    action="ask",
                    reason=f"Environment override requires approval: {key}.",
                    rule_id="env_override",
                )

        for token in tokens:
            env_match = POWERSHELL_ENV_ASSIGN_RE.match(token)
            if env_match:
                key = env_match.group(1).upper()
                if key in PATH_ENV_KEYS:
                    return CommandSecurityResult(
                        action="deny",
                        reason=f"{key} override is not allowed.",
                        rule_id="env_path_override",
                    )
                if key not in self._allowed_env_vars:
                    return CommandSecurityResult(
                        action="ask",
                        reason=f"Environment override requires approval: {key}.",
                        rule_id="env_override",
                    )

        if isinstance(env, dict):
            for key in env:
                upper_key = str(key).upper()
                if upper_key in PATH_ENV_KEYS:
                    return CommandSecurityResult(
                        action="deny",
                        reason=f"{upper_key} override is not allowed.",
                        rule_id="env_path_override",
                    )
                if upper_key not in self._allowed_env_vars:
                    return CommandSecurityResult(
                        action="ask",
                        reason=f"Environment override requires approval: {upper_key}.",
                        rule_id="env_override",
                    )

        return None

    def _classify_interpreter_risks(self, *, tokens: list[str], program: str) -> CommandSecurityResult | None:
        if program in INTERPRETER_INLINE_EVAL:
            for token in tokens[1:]:
                if token in INTERPRETER_INLINE_EVAL[program]:
                    return CommandSecurityResult(
                        action="deny",
                        reason="Inline interpreter eval is not allowed.",
                        rule_id="interpreter_inline_eval",
                    )

        if program in DISALLOWED_INTERPRETERS and not self._allow_dangerous_interpreters:
            return CommandSecurityResult(
                action="deny",
                reason="Command matched a protected shell policy.",
                rule_id="dangerous_interpreter",
            )
        return None

    def _classify_powershell(self, *, command: str, tokens: list[str], shell: str | None, program: str) -> CommandSecurityResult | None:
        shell_name = normalize_command_name(shell) if shell else ""
        is_powershell = program in {"pwsh", "powershell"} or shell_name in {"pwsh", "powershell"}
        if not is_powershell:
            return None

        lowered = [token.lower() for token in tokens]
        if "invoke-expression" in lowered or "iex" in lowered:
            return CommandSecurityResult(
                action="deny",
                reason="PowerShell Invoke-Expression is not allowed.",
                rule_id="powershell_invoke_expression",
            )

        for token in lowered[1:]:
            if token in {"-encodedcommand", "-enc"}:
                return CommandSecurityResult(
                    action="deny",
                    reason="PowerShell encoded command is not allowed.",
                    rule_id="powershell_encoded_command",
                )

        payload_tokens, payload = extract_powershell_payload(command, tokens)
        if not payload_tokens:
            return CommandSecurityResult(
                action="deny",
                reason="Interactive PowerShell sessions are not allowed.",
                rule_id="powershell_interactive_session",
            )
        if HEREDOC_RE.search(payload):
            return CommandSecurityResult(
                action="deny",
                reason="Heredoc-like command input is not allowed.",
                rule_id="heredoc",
            )
        if contains_unquoted_operator(payload, ("&&", "||", ";"), escape_char="`"):
            return CommandSecurityResult(
                action="deny",
                reason="PowerShell chaining syntax is not allowed.",
                rule_id="powershell_chaining",
            )
        if SHELL_REDIRECTION_RE.search(payload):
            return CommandSecurityResult(
                action="deny",
                reason="PowerShell redirection syntax is not allowed.",
                rule_id="powershell_redirection",
            )
        if SUBSHELL_RE.search(payload):
            return CommandSecurityResult(
                action="deny",
                reason="Subshell or command-substitution syntax is not allowed.",
                rule_id="subshell",
            )

        segments = split_powershell_pipeline(payload)
        if not segments:
            return CommandSecurityResult(
                action="deny",
                reason="Interactive PowerShell sessions are not allowed.",
                rule_id="powershell_interactive_session",
            )

        requires_approval_for_segment: str | None = None
        for segment in segments:
            try:
                segment_tokens = tokenize_command(segment)
            except ValueError:
                return CommandSecurityResult(
                    action="deny",
                    reason="PowerShell command could not be parsed safely.",
                    rule_id="parse_error",
                )
            if not segment_tokens:
                return CommandSecurityResult(
                    action="deny",
                    reason="PowerShell pipeline contains an empty segment.",
                    rule_id="powershell_empty_segment",
                )
            segment_program = normalize_command_name(segment_tokens[0])
            if segment_program in POWERSHELL_BLOCKED_CMDLETS:
                return CommandSecurityResult(
                    action="deny",
                    reason=f"PowerShell command is blocked by policy: {segment_tokens[0]}.",
                    rule_id="powershell_blocked_cmdlet",
                )
            if segment_program in DISALLOWED_INTERPRETERS:
                return CommandSecurityResult(
                    action="deny",
                    reason=f"PowerShell command is blocked by policy: {segment_tokens[0]}.",
                    rule_id="powershell_blocked_spawn",
                )
            path_result = self._classify_path_risks(tokens=segment_tokens)
            if path_result is not None:
                return path_result
            if segment_program not in POWERSHELL_READ_ONLY_CMDLETS:
                requires_approval_for_segment = requires_approval_for_segment or segment_tokens[0]

        if self._matches_persisted_rule(tokens=tokens, shell=shell):
            return CommandSecurityResult(
                action="allow",
                reason="Command is allowed by a persisted command rule.",
                rule_id="persisted_command_rule",
            )
        if requires_approval_for_segment is not None:
            return CommandSecurityResult(
                action="ask",
                reason=f"PowerShell command requires approval: {requires_approval_for_segment}.",
                rule_id="powershell_requires_approval",
            )

        return CommandSecurityResult(
            action="allow",
            reason="Read-only PowerShell command is allowed by security policy.",
            rule_id="powershell_read_only",
        )

    def _classify_cmd(self, *, tokens: list[str], shell: str | None, program: str) -> CommandSecurityResult | None:
        if program != "cmd":
            return None
        lowered = [token.lower() for token in tokens]
        if "/c" in lowered:
            tail = " ".join(tokens[lowered.index("/c") + 1 :]).strip()
            if not tail:
                return CommandSecurityResult(
                    action="deny",
                    reason="cmd /c payload must not be empty.",
                    rule_id="cmd_empty_payload",
                )
            if SHELL_CHAINING_RE.search(tail) or SHELL_REDIRECTION_RE.search(tail):
                return CommandSecurityResult(
                    action="deny",
                    reason="cmd /c with chained or redirected payload is not allowed.",
                    rule_id="cmd_high_risk_chaining",
                )
            try:
                payload_tokens = tokenize_command(tail)
            except ValueError:
                return CommandSecurityResult(
                    action="deny",
                    reason="cmd /c payload could not be parsed safely.",
                    rule_id="parse_error",
                )
            if not payload_tokens:
                return CommandSecurityResult(
                    action="deny",
                    reason="cmd /c payload must not be empty.",
                    rule_id="cmd_empty_payload",
                )
            payload_program = normalize_command_name(payload_tokens[0])
            if payload_program in WINDOWS_BLOCKED_CMD_PAYLOADS:
                return CommandSecurityResult(
                    action="deny",
                    reason=f"cmd /c payload is blocked by policy: {payload_tokens[0]}.",
                    rule_id="cmd_blocked_payload",
                )
            path_result = self._classify_path_risks(tokens=payload_tokens)
            if path_result is not None:
                return path_result
            if self._matches_persisted_rule(tokens=tokens, shell=shell):
                return CommandSecurityResult(
                    action="allow",
                    reason="Command is allowed by a persisted command rule.",
                    rule_id="persisted_command_rule",
                )
            if payload_program in WINDOWS_READ_ONLY_COMMANDS:
                return CommandSecurityResult(
                    action="allow",
                    reason="Read-only cmd /c command is allowed by security policy.",
                    rule_id="cmd_read_only",
                )
            return CommandSecurityResult(
                action="ask",
                reason=f"cmd /c command requires approval: {payload_tokens[0]}.",
                rule_id="cmd_c_requires_approval",
            )
        return None

    def _classify_path_risks(self, *, tokens: list[str]) -> CommandSecurityResult | None:
        path_result = classify_path_risks(tokens, workspace_dir=self._workspace)
        if path_result is None:
            return None
        return CommandSecurityResult(
            action="deny",
            reason=path_result.reason,
            rule_id=path_result.rule_id,
        )

    def _matches_persisted_rule(self, *, tokens: list[str], shell: str | None) -> bool:
        return any(command_rule_matches(rule, tokens, shell) for rule in self._command_rules)


def classify_command(
    command: str,
    *,
    command_rules: list[dict[str, object]] | None = None,
    shell: str | None = None,
    workspace_dir: str | Path | None = None,
    env: dict[str, str] | None = None,
    allowed_env_vars: list[str] | None = None,
    allow_dangerous_interpreters: bool = False,
) -> CommandSecurityResult:
    """Convenience helper for one-shot command classification."""
    policy = CommandSecurityPolicy(
        command_rules=command_rules,
        workspace_dir=workspace_dir,
        allowed_env_vars=allowed_env_vars,
        allow_dangerous_interpreters=allow_dangerous_interpreters,
    )
    return policy.classify(command, shell=shell, env=env)
