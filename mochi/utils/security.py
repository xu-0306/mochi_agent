"""Security helpers for shell and file tool policy enforcement."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Literal

from mochi.utils.command_security import CommandSecurityResult, classify_command
from mochi.security.decision import SecurityDecision, deny_security_decision

FORBIDDEN_SHELL_PATTERNS: tuple[str, ...] = (
    "&&",
    "||",
    ";",
    "|",
    "`",
    "$(",
    "\n",
    "\r",
    ">",
    "<",
    "&",
)

DANGEROUS_SHELL_PREFIXES: tuple[str, ...] = (
    "python",
    "python3",
    "node",
    "deno",
    "bash",
    "sh",
    "pwsh",
    "powershell",
    "cmd",
    "ssh",
    "npx",
    "bun run",
    "npm run",
    "yarn run",
    "pnpm run",
    "curl",
    "wget",
)

PROTECTED_DIRECTORIES: tuple[str, ...] = (".git", ".mochi", ".vscode", ".idea")
PROTECTED_FILE_NAMES: tuple[str, ...] = (
    ".gitconfig",
    ".bashrc",
    ".bash_profile",
    ".profile",
    ".zshrc",
    ".zprofile",
    ".zshenv",
    "profile.ps1",
    "microsoft.powershell_profile.ps1",
)

WINDOWS_SHORT_NAME_RE = re.compile(r"^[A-Za-z0-9]{1,6}~\d(?:\..+)?$")
WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:$")
WINDOWS_LONG_PATH_PREFIXES: tuple[str, ...] = ("\\\\?\\", "\\\\.\\", "//?/", "//./")


def _normalize_cmd_name(raw: str) -> str:
    return Path(raw.strip()).name.lower()


def _tokenize_shell_command(command: str) -> list[str]:
    return shlex.split(command.strip(), posix=True)


def _allowlist_entry_matches_tokens(entry: str, tokens: list[str]) -> bool:
    if not entry.strip():
        return False
    try:
        entry_tokens = _tokenize_shell_command(entry)
    except ValueError:
        entry_tokens = entry.strip().split()
    if not entry_tokens or len(tokens) < len(entry_tokens):
        return False
    normalized_entry = [_normalize_cmd_name(entry_tokens[0]), *[part.lower() for part in entry_tokens[1:]]]
    normalized_tokens = [_normalize_cmd_name(tokens[0]), *[part.lower() for part in tokens[1: len(entry_tokens)]]]
    return normalized_entry == normalized_tokens


def _matches_dangerous_shell_prefix(tokens: list[str]) -> bool:
    lowered = [_normalize_cmd_name(tokens[0]), *[part.lower() for part in tokens[1:]]]
    joined = " ".join(lowered)
    return any(
        joined == prefix or joined.startswith(prefix + " ")
        for prefix in DANGEROUS_SHELL_PREFIXES
    )


def is_safe_command(command: str, allowlist: list[str]) -> bool:
    """Return True only for allowlisted commands that also pass hard safety rules."""
    return classify_legacy_shell_command(command, allowlist).action == "allow"


def classify_legacy_shell_command(
    command: str,
    allowlist: list[str],
) -> CommandSecurityResult:
    """Classify a legacy shell command using the shared command policy."""
    allow_dangerous_interpreters = any(
        str(item).strip().lower() == "__allow_dangerous_shells__"
        for item in allowlist
        if isinstance(item, str)
    )
    effective_allowlist = [
        item
        for item in allowlist
        if isinstance(item, str) and item.strip().lower() != "__allow_dangerous_shells__"
    ]
    command_rules = [
        {
            "tokens": _tokenize_shell_command(item),
            "decision": "allow",
            "match": "exact",
            "shells": [],
        }
        for item in effective_allowlist
        if item.strip()
    ]
    result = classify_command(
        command,
        command_rules=command_rules,
        allow_dangerous_interpreters=allow_dangerous_interpreters,
    )
    if result.rule_id == "unknown_requires_approval":
        return CommandSecurityResult(
            action="deny",
            reason="Command denied by allowlist policy.",
            rule_id="allowlist",
            parsed_tokens=result.parsed_tokens,
        )
    return result


def explain_unsafe_shell_command(command: str, allowlist: list[str]) -> str | None:
    """Return a human-readable reason when a shell command is denied by policy."""
    result = classify_legacy_shell_command(command, allowlist)
    if result.action == "allow":
        return None
    return result.reason


PolicyState = Literal["allow", "ask", "deny"]


def security_decision_policy_state(decision: SecurityDecision) -> PolicyState:
    """Map legacy SecurityDecision actions onto the UI-facing allow/ask/deny states."""
    if decision.action == "require_approval":
        return "ask"
    return decision.action


def build_policy_metadata(
    *,
    decision: SecurityDecision | None = None,
    policy_state: PolicyState | None = None,
    policy_reason: str | None = None,
    legacy_tool: bool | None = None,
    preferred_tool: str | None = None,
    preferred_tools: list[str] | None = None,
) -> dict[str, object]:
    """Attach stable allow/ask/deny metadata without dropping legacy decision fields."""
    metadata: dict[str, object] = {}
    if decision is not None:
        metadata.update(decision.to_metadata())
        policy_state = policy_state or security_decision_policy_state(decision)
        policy_reason = policy_reason or decision.reason
    if policy_state is not None:
        metadata["policy_state"] = policy_state
    if policy_reason:
        metadata["policy_reason"] = policy_reason
    if legacy_tool is not None:
        metadata["legacy_tool"] = legacy_tool
    if preferred_tool:
        metadata["preferred_tool"] = preferred_tool
    if preferred_tools:
        metadata["preferred_tools"] = list(preferred_tools)
    return metadata


def normalize_workspace_dir(workspace_dir: str | Path) -> Path:
    return Path(workspace_dir).expanduser().resolve(strict=False)


def is_path_within_workspace(path: str | Path, workspace_dir: str | Path) -> bool:
    workspace = normalize_workspace_dir(workspace_dir)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate

    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError:
        return False
    return True


def _is_suspicious_raw_path(path: str) -> bool:
    raw = path.strip()
    if not raw:
        return True
    if raw.startswith("\\\\") or raw.startswith("//"):
        return True
    if raw.startswith(WINDOWS_LONG_PATH_PREFIXES):
        return True
    if "..." in raw:
        return True
    normalized = raw.replace("\\", "/")
    for segment in normalized.split("/"):
        if not segment:
            continue
        if segment.rstrip(" .") != segment:
            return True
        if WINDOWS_SHORT_NAME_RE.match(segment):
            return True
        if ":" in segment and not WINDOWS_DRIVE_RE.match(segment):
            return True
    return False


def _is_protected_path(candidate: Path) -> bool:
    for segment in candidate.parts:
        lowered = segment.lower()
        if lowered in PROTECTED_DIRECTORIES:
            return True
        if lowered in PROTECTED_FILE_NAMES:
            return True
    return False


def resolve_path_in_workspace(path: str | Path, workspace_dir: str | Path) -> Path:
    workspace = normalize_workspace_dir(workspace_dir)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve(strict=False)

    if not is_path_within_workspace(resolved, workspace):
        raise ValueError(f"Path '{path}' is outside workspace '{workspace}'.")
    return resolved


def resolve_path_with_scope(
    path: str | Path,
    workspace_dir: str | Path,
    scope: str,
) -> Path:
    workspace = normalize_workspace_dir(workspace_dir)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve(strict=False)

    if scope == "workspace" and not is_path_within_workspace(resolved, workspace):
        raise ValueError(f"Path '{path}' is outside workspace '{workspace}'.")
    return resolved


def check_file_tool_path(
    path: str | Path,
    *,
    workspace_dir: str | Path,
    scope: str,
    access: str = "write",
) -> tuple[Path | None, SecurityDecision | None]:
    """Resolve a tool path and reject suspicious or protected locations."""
    raw_path = str(path)
    if _is_suspicious_raw_path(raw_path):
        return (
            None,
            deny_security_decision(
                reason="Suspicious path denied by security policy.",
                approval_scope="protected_path",
                policy_source="path_policy",
            ),
        )

    try:
        effective_scope = "any" if access == "read" else scope
        resolved = resolve_path_with_scope(raw_path, workspace_dir, effective_scope)
    except ValueError as exc:
        return (
            None,
            deny_security_decision(
                reason=str(exc),
                approval_scope="workspace",
                policy_source="workspace_scope",
            ),
        )

    if access != "read" and (
        _is_protected_path(Path(raw_path).expanduser()) or _is_protected_path(resolved)
    ):
        return (
            None,
            deny_security_decision(
                reason="Protected path denied by security policy.",
                approval_scope="protected_path",
                policy_source="path_policy",
            ),
        )

    return resolved, None


def content_size_bytes(content: str, encoding: str = "utf-8") -> int:
    return len(content.encode(encoding))


def size_limit_bytes(max_size_mb: float) -> int:
    return max(0, int(max_size_mb * 1024 * 1024))


def is_within_write_size_limit(
    content: str,
    max_size_mb: float,
    encoding: str = "utf-8",
) -> bool:
    return content_size_bytes(content, encoding=encoding) <= size_limit_bytes(max_size_mb)
