"""Reusable rule tables and token helpers for command security."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

SHELL_CHAINING_RE = re.compile(r"(\|\||&&|[|;`])")
SHELL_REDIRECTION_RE = re.compile(r"(?:^|[\s])(?:>|>>|<|<<|<<<|\d>&\d|\d>|\d<)")
SUBSHELL_RE = re.compile(r"(\$\(|\$\{|\(\s*[^)]*\)\s*$)")
HEREDOC_RE = re.compile(r"(<<[-~]?\s*['\"]?[A-Za-z0-9_]+['\"]?)")
ENV_OVERRIDE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
POWERSHELL_ENV_ASSIGN_RE = re.compile(r"^\$env:([A-Za-z_][A-Za-z0-9_]*)\s*=", re.IGNORECASE)

INTERPRETER_INLINE_EVAL = {
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

DISALLOWED_INTERPRETERS = {
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

PATH_ENV_KEYS = {"PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATHEXT"}

WINDOWS_READ_ONLY_COMMANDS = {
    "dir",
    "type",
    "where",
    "cd",
    "chdir",
}

WINDOWS_BLOCKED_CMD_PAYLOADS = {
    "call",
    "cmd",
    "powershell",
    "pwsh",
    "python",
    "python3",
    "py",
    "node",
    "deno",
    "start",
    "wscript",
    "cscript",
}

POWERSHELL_READ_ONLY_CMDLETS = {
    "cat",
    "dir",
    "gc",
    "gci",
    "get-childitem",
    "get-content",
    "get-location",
    "gl",
    "ls",
    "pwd",
    "resolve-path",
    "select-string",
    "sls",
    "test-path",
    "type",
}

POWERSHELL_BLOCKED_CMDLETS = {
    ".",
    "&",
    "ac",
    "add-content",
    "add-type",
    "clear-content",
    "clc",
    "copy",
    "copy-item",
    "cp",
    "del",
    "erase",
    "icm",
    "iex",
    "ii",
    "invoke-command",
    "invoke-expression",
    "invoke-item",
    "md",
    "mi",
    "mkdir",
    "move",
    "move-item",
    "mv",
    "new-item",
    "ni",
    "out-file",
    "remove-item",
    "ren",
    "rename-item",
    "ri",
    "rm",
    "rmdir",
    "rni",
    "saps",
    "sc",
    "set-content",
    "set-item",
    "si",
    "start-job",
    "start-process",
}


def normalize_command_name(raw: str) -> str:
    return Path(raw.strip()).name.lower()


def tokenize_command(command: str) -> list[str]:
    return shlex.split(command, posix=True)


def command_rule_matches(
    rule: dict[str, object],
    tokens: list[str],
    shell: str | None,
) -> bool:
    rule_tokens = rule.get("tokens")
    if not isinstance(rule_tokens, list) or not rule_tokens:
        return False
    normalized_rule_tokens = [str(item).strip().lower() for item in rule_tokens if str(item).strip()]
    if not normalized_rule_tokens:
        return False
    normalized_tokens = [normalize_command_name(tokens[0]), *[part.lower() for part in tokens[1:]]]
    match_mode = str(rule.get("match") or "prefix").lower()
    if match_mode == "exact" and normalized_tokens != normalized_rule_tokens:
        return False
    if match_mode != "exact" and normalized_tokens[: len(normalized_rule_tokens)] != normalized_rule_tokens:
        return False
    rule_shells = rule.get("shells")
    if isinstance(rule_shells, list) and rule_shells:
        normalized_shell = normalize_command_name(shell) if shell else ""
        allowed_shells = {str(item).strip().lower() for item in rule_shells if str(item).strip()}
        if normalized_shell not in allowed_shells:
            return False
    return str(rule.get("decision") or "allow").lower() == "allow"


def contains_unquoted_operator(
    command: str,
    operators: tuple[str, ...],
    *,
    escape_char: str | None = None,
) -> bool:
    quote: str | None = None
    escape_next = False
    index = 0
    while index < len(command):
        char = command[index]
        if escape_next:
            escape_next = False
            index += 1
            continue
        if escape_char is not None and char == escape_char:
            escape_next = True
            index += 1
            continue
        if quote is not None:
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        for operator in operators:
            if command.startswith(operator, index):
                return True
        index += 1
    return False


def split_powershell_pipeline(command: str) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escape_next = False
    for char in command:
        if escape_next:
            current.append(char)
            escape_next = False
            continue
        if char == "`":
            current.append(char)
            escape_next = True
            continue
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char == "|":
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            continue
        current.append(char)
    trailing = "".join(current).strip()
    if trailing:
        segments.append(trailing)
    return segments


def extract_powershell_payload(command: str, tokens: list[str]) -> tuple[list[str], str]:
    lowered = [token.lower() for token in tokens]
    for flag in ("-command", "-c", "/c"):
        if flag in lowered:
            index = lowered.index(flag)
            payload_tokens = tokens[index + 1 :]
            marker_match = re.search(rf"(?i)(?:^|\s){re.escape(flag)}\s+", command)
            if marker_match is not None:
                return payload_tokens, command[marker_match.end() :].strip()
            return payload_tokens, " ".join(payload_tokens).strip()
    return tokens, command.strip()
