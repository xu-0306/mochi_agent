"""Workspace/path checks for command security."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SENSITIVE_PATH_RE = re.compile(
    r"(?i)(?:\b|[/\\])(?:\.git|\.ssh|\.aws|\.gnupg|\.kube|system32|etc/passwd|id_rsa|id_ed25519)(?:\b|[/\\])"
)
_ABS_PATH_RE = re.compile(r"(?i)^[A-Z]:[/\\]|^/|^~[/\\]|^\\\\")
_CMD_PROGRAMS = frozenset({"cmd", "cmd.exe"})
_CMD_SWITCHES = frozenset({"/a", "/c", "/d", "/k", "/q", "/s", "/u"})


@dataclass(frozen=True)
class CommandPathDecision:
    rule_id: str
    reason: str


def _looks_like_path(token: str) -> bool:
    normalized = token.strip().strip("\"'")
    if not normalized:
        return False
    if normalized in {".", ".."}:
        return False
    if normalized.startswith(".."):
        return True
    if _ABS_PATH_RE.match(normalized):
        return True
    return "/" in normalized or "\\" in normalized


def _cmd_switch_indexes(tokens: list[str]) -> frozenset[int]:
    """Return cmd.exe option positions that must not be parsed as POSIX paths."""

    ignored: set[int] = set()
    for index, token in enumerate(tokens):
        program = token.strip().strip("\"'").lower()
        if program not in _CMD_PROGRAMS:
            continue
        for option_index in range(index + 1, len(tokens)):
            option = tokens[option_index].strip().strip("\"'").lower()
            if option not in _CMD_SWITCHES:
                break
            ignored.add(option_index)
            if option in {"/c", "/k"}:
                break
    return frozenset(ignored)


def classify_path_risks(
    tokens: list[str],
    *,
    workspace_dir: str | Path | None,
) -> CommandPathDecision | None:
    if any(_SENSITIVE_PATH_RE.search(token) for token in tokens):
        return CommandPathDecision(
            rule_id="sensitive_path",
            reason="Command references a sensitive path.",
        )

    if workspace_dir is None:
        return None

    workspace = Path(workspace_dir).expanduser().resolve(strict=False)
    cmd_switch_indexes = _cmd_switch_indexes(tokens)
    for index, token in enumerate(tokens):
        if index in cmd_switch_indexes:
            continue
        if not _looks_like_path(token):
            continue
        normalized = token.strip().strip("\"'")
        if normalized.startswith("..") or "/../" in normalized.replace("\\", "/"):
            return CommandPathDecision(
                rule_id="workspace_escape",
                reason="Command attempts to escape workspace boundaries.",
            )
        candidate = Path(normalized).expanduser()
        if not candidate.is_absolute():
            candidate = workspace / candidate
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(workspace)
        except ValueError:
            return CommandPathDecision(
                rule_id="workspace_escape",
                reason="Command path is outside the workspace.",
            )
    return None
