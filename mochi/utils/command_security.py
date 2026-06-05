"""Structured command security policy for exec/shell classification."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CommandSecurityAction = Literal["allow", "ask", "deny"]

_SHELL_CHAINING_RE = re.compile(r"(\|\||&&|[|;`])")
_SHELL_REDIRECTION_RE = re.compile(r"(?:^|[\s])(?:>|>>|<|<<|<<<|\d>&\d|\d>|\d<)")
_SUBSHELL_RE = re.compile(r"(\$\(|\$\{|\(\s*[^)]*\)\s*$)")
_HEREDOC_RE = re.compile(r"(<<[-~]?\s*['\"]?[A-Za-z0-9_]+['\"]?)")
_ENV_OVERRIDE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_POWERSHELL_ENV_ASSIGN_RE = re.compile(r"^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=", re.IGNORECASE)
_SENSITIVE_PATH_RE = re.compile(
    r"(?i)(?:\b|[/\\])(?:\.git|\.ssh|\.aws|\.gnupg|\.kube|system32|etc/passwd|id_rsa|id_ed25519)(?:\b|[/\\])"
)
_ABS_PATH_RE = re.compile(r"(?i)^[A-Z]:[/\\]|^/|^~[/\\]|^\\\\")
_INTERPRETER_INLINE_EVAL = {
    "python": {"-c"},
    "python3": {"-c"},
    "py": {"-c"},
    "node": {"-e", "--eval"},
    "deno": {"eval"},
    "ruby": {"-e"},
    "perl": {"-e"},
    "php": {"-r"},
    "bash": {"-c"},
    "sh": {"-c"},
    "zsh": {"-c"},
    "fish": {"-c"},
}
_DISALLOWED_INTERPRETERS = {
    "python",
    "python3",
    "py",
    "node",
    "deno",
    "bash",
    "sh",
    "pwsh",
    "powershell",
    "cmd",
    "npx",
    "npm",
    "yarn",
    "pnpm",
    "curl",
    "wget",
    "ssh",
    "perl",
    "ruby",
    "php",
}
_PATH_ENV_KEYS = {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATHEXT"}


def _normalize_cmd_name(raw: str) -> str:
    return Path(raw.strip()).name.lower()


def _tokenize(command: str) -> list[str]:
    return shlex.split(command, posix=True)


def _allowlist_entry_matches_tokens(entry: str, tokens: list[str]) -> bool:
    if not entry.strip():
        return False
    try:
        entry_tokens = _tokenize(entry.strip())
    except ValueError:
        entry_tokens = entry.strip().split()
    if not entry_tokens or len(tokens) < len(entry_tokens):
        return False
    normalized_entry = [_normalize_cmd_name(entry_tokens[0]), *[part.lower() for part in entry_tokens[1:]]]
    normalized_tokens = [_normalize_cmd_name(tokens[0]), *[part.lower() for part in tokens[1: len(entry_tokens)]]]
    return normalized_entry == normalized_tokens


def _looks_like_path(token: str) -> bool:
    normalized = token.strip().strip("\"'")
    if not normalized:
        return False
    if normalized in {".", ".."}:
        return False
    if _ABS_PATH_RE.match(normalized):
        return True
    if "/" in normalized or "\\" in normalized:
        return True
    return False


@dataclass(frozen=True)
class CommandSecurityResult:
    """Command classification result."""

    action: CommandSecurityAction
    reason: str
    rule_id: str
    parsed_tokens: tuple[str, ...] = ()


class CommandSecurityPolicy:
    """Classify command strings into allow/ask/deny decisions."""

    def __init__(
        self,
        *,
        allowlist: list[str] | None = None,
        workspace_dir: str | Path | None = None,
        allowed_env_vars: list[str] | None = None,
        allow_dangerous_interpreters: bool = False,
    ) -> None:
        self._allowlist = allowlist or []
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

        if _HEREDOC_RE.search(stripped):
            return CommandSecurityResult(
                action="deny",
                reason="Heredoc-like command input is not allowed.",
                rule_id="heredoc",
            )

        try:
            tokens = _tokenize(stripped)
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

        program = _normalize_cmd_name(tokens[0])
        cmd_result = self._classify_cmd(tokens=tokens, program=program)
        if cmd_result is not None:
            return cmd_result

        if _SHELL_CHAINING_RE.search(stripped) or _SHELL_REDIRECTION_RE.search(stripped):
            return CommandSecurityResult(
                action="deny",
                reason="Shell chaining or redirection syntax is not allowed.",
                rule_id="shell_chaining",
            )

        if _SUBSHELL_RE.search(stripped):
            return CommandSecurityResult(
                action="deny",
                reason="Subshell or command-substitution syntax is not allowed.",
                rule_id="subshell",
            )

        env_result = self._classify_env_overrides(tokens=tokens, env=env)
        if env_result is not None:
            return env_result

        interpreter_result = self._classify_interpreter_risks(tokens=tokens, program=program)
        if interpreter_result is not None:
            return interpreter_result

        ps_result = self._classify_powershell(tokens=tokens, shell=shell, program=program)
        if ps_result is not None:
            return ps_result

        path_result = self._classify_path_risks(tokens=tokens)
        if path_result is not None:
            return path_result

        if self._allowlist and not any(
            _allowlist_entry_matches_tokens(entry, tokens)
            for entry in self._allowlist
            if isinstance(entry, str)
        ):
            return CommandSecurityResult(
                action="deny",
                reason="Command denied by allowlist policy.",
                rule_id="allowlist",
                parsed_tokens=tuple(tokens),
            )

        return CommandSecurityResult(
            action="allow",
            reason="Command is allowed by security policy.",
            rule_id="allow",
            parsed_tokens=tuple(tokens),
        )

    def _classify_env_overrides(self, *, tokens: list[str], env: dict[str, str] | None) -> CommandSecurityResult | None:
        for token in tokens:
            if not _ENV_OVERRIDE_RE.match(token):
                break
            key = token.split("=", 1)[0].upper()
            if key in _PATH_ENV_KEYS:
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
            env_match = _POWERSHELL_ENV_ASSIGN_RE.match(token)
            if env_match:
                key = env_match.group(1).upper()
                if key in _PATH_ENV_KEYS:
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
                if upper_key in _PATH_ENV_KEYS:
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
        if program in _INTERPRETER_INLINE_EVAL:
            for token in tokens[1:]:
                if token in _INTERPRETER_INLINE_EVAL[program]:
                    return CommandSecurityResult(
                        action="deny",
                        reason="Inline interpreter eval is not allowed.",
                        rule_id="interpreter_inline_eval",
                    )

        if program in _DISALLOWED_INTERPRETERS and not self._allow_dangerous_interpreters:
            return CommandSecurityResult(
                action="deny",
                reason="Command matched a protected shell policy.",
                rule_id="dangerous_interpreter",
            )
        return None

    def _classify_powershell(self, *, tokens: list[str], shell: str | None, program: str) -> CommandSecurityResult | None:
        shell_name = _normalize_cmd_name(shell) if shell else ""
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
        return None

    def _classify_cmd(self, *, tokens: list[str], program: str) -> CommandSecurityResult | None:
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
            if _SHELL_CHAINING_RE.search(tail) or _SHELL_REDIRECTION_RE.search(tail):
                return CommandSecurityResult(
                    action="deny",
                    reason="cmd /c with chained or redirected payload is not allowed.",
                    rule_id="cmd_high_risk_chaining",
                )
            return CommandSecurityResult(
                action="ask",
                reason="cmd /c execution requires approval.",
                rule_id="cmd_c_requires_approval",
            )
        return None

    def _classify_path_risks(self, *, tokens: list[str]) -> CommandSecurityResult | None:
        if any(_SENSITIVE_PATH_RE.search(token) for token in tokens):
            return CommandSecurityResult(
                action="deny",
                reason="Command references a sensitive path.",
                rule_id="sensitive_path",
            )

        if self._workspace is None:
            return None

        for token in tokens[1:]:
            if not _looks_like_path(token):
                continue
            normalized = token.strip().strip("\"'")
            if normalized.startswith("..") or "/../" in normalized.replace("\\", "/"):
                return CommandSecurityResult(
                    action="deny",
                    reason="Command attempts to escape workspace boundaries.",
                    rule_id="workspace_escape",
                )
            candidate = Path(normalized).expanduser()
            if not candidate.is_absolute():
                candidate = self._workspace / candidate
            resolved = candidate.resolve(strict=False)
            try:
                resolved.relative_to(self._workspace)
            except ValueError:
                return CommandSecurityResult(
                    action="deny",
                    reason="Command path is outside the workspace.",
                    rule_id="workspace_escape",
                )
        return None


def classify_command(
    command: str,
    *,
    allowlist: list[str] | None = None,
    shell: str | None = None,
    workspace_dir: str | Path | None = None,
    env: dict[str, str] | None = None,
    allowed_env_vars: list[str] | None = None,
    allow_dangerous_interpreters: bool = False,
) -> CommandSecurityResult:
    """Convenience helper for one-shot command classification."""
    policy = CommandSecurityPolicy(
        allowlist=allowlist,
        workspace_dir=workspace_dir,
        allowed_env_vars=allowed_env_vars,
        allow_dangerous_interpreters=allow_dangerous_interpreters,
    )
    return policy.classify(command, shell=shell, env=env)
