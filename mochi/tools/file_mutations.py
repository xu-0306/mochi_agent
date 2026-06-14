"""Shared file-mutation planning, metadata, and patch parsing helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import difflib
from pathlib import Path
from typing import Any, Literal

from mochi.security.decision import SecurityDecision
from mochi.utils.security import (
    check_file_tool_path,
    content_size_bytes,
    normalize_workspace_dir,
    size_limit_bytes,
)

ChangeType = Literal["add", "update", "delete"]


class PatchValidationError(ValueError):
    """Structured patch validation error."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class PatchSection:
    """One update section inside an apply_patch update file block."""

    header: str
    lines: tuple[tuple[str, str], ...]
    reaches_eof: bool = False


@dataclass(frozen=True)
class PatchOperation:
    """One parsed apply_patch file operation."""

    kind: ChangeType
    path: str
    added_lines: tuple[str, ...] = ()
    sections: tuple[PatchSection, ...] = ()


@dataclass(frozen=True)
class PreparedPatchOperation:
    """Resolved patch operation with current and next file state."""

    operation: PatchOperation
    target: Path
    existed_before: bool
    original_content: str | None
    new_content: str | None
    file_change: dict[str, Any]


def build_file_change_entry(
    *,
    target: Path,
    workspace_root: Path | None,
    tool_name: str,
    change_type: ChangeType,
    original_content: str | None,
    new_content: str | None,
    encoding: str,
    undo_max_size_mb: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the normalized file-change payload for one mutation."""

    undo_limit_bytes = size_limit_bytes(undo_max_size_mb)
    before_text = original_content or ""
    after_text = new_content or ""
    original_size = content_size_bytes(before_text, encoding=encoding)
    new_size = content_size_bytes(after_text, encoding=encoding)
    diff_lines = list(
        difflib.unified_diff(
            before_text.splitlines(),
            after_text.splitlines(),
            fromfile=f"a/{target.name}",
            tofile=f"b/{target.name}",
            lineterm="",
            n=3,
        )
    )
    diff_text = "\n".join(diff_lines) if diff_lines else None
    if diff_text is not None and undo_limit_bytes > 0:
        if content_size_bytes(diff_text, encoding=encoding) > undo_limit_bytes:
            diff_text = None

    undo_available = False
    undo_reason: str | None = None
    stored_original = None
    stored_new = None
    if undo_limit_bytes > 0:
        if max(original_size, new_size) <= undo_limit_bytes:
            undo_available = True
            stored_original = original_content
            stored_new = new_content
        else:
            undo_reason = "file_too_large"

    undo_action: str | None
    if not undo_available:
        undo_action = None
    elif change_type == "add":
        undo_action = "delete"
    else:
        undo_action = "restore"

    relative_path: str | None = None
    if workspace_root is not None:
        try:
            relative_path = target.resolve(strict=False).relative_to(workspace_root).as_posix() or "."
        except ValueError:
            relative_path = None

    added_lines = sum(
        1
        for line in diff_lines
        if line.startswith("+") and not line.startswith("+++")
    )
    deleted_lines = sum(
        1
        for line in diff_lines
        if line.startswith("-") and not line.startswith("---")
    )

    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "path": str(target),
        "file_path": str(target),
        "relative_path": relative_path,
        "status": change_type,
        "change_type": change_type,
        "encoding": encoding,
        "original_size_bytes": original_size,
        "new_size_bytes": new_size,
        "added_lines": added_lines,
        "deleted_lines": deleted_lines,
        "original_content": stored_original,
        "new_content": stored_new,
        "undo_available": undo_available,
        "undo_action": undo_action,
        "undo_reason": undo_reason,
        "diff": diff_text,
        "diff_available": diff_text is not None,
        "undo_size_limit_bytes": undo_limit_bytes,
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return payload


def build_editable_patch_text(
    *,
    file_changes: list[dict[str, Any]],
    fallback_patch_text: str | None = None,
) -> str | None:
    """Build an editable apply_patch payload for one or more normalized changes."""

    if fallback_patch_text is not None:
        normalized_fallback = fallback_patch_text.strip("\n")
        if normalized_fallback:
            return f"{normalized_fallback}\n"
        return None

    blocks: list[str] = ["*** Begin Patch"]
    added_any = False

    for change in file_changes:
        if not isinstance(change, dict):
            continue
        raw_path = change.get("relative_path") or change.get("path") or change.get("file_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = raw_path.strip()
        change_type = str(change.get("change_type") or "update").strip().lower()
        original_content = change.get("original_content")
        new_content = change.get("new_content")
        if original_content is not None and not isinstance(original_content, str):
            continue
        if new_content is not None and not isinstance(new_content, str):
            continue

        if change_type == "add":
            blocks.append(f"*** Add File: {path}")
            for line in _split_patch_content_lines(new_content or ""):
                blocks.append(f"+{line}")
            added_any = True
            continue

        if change_type == "delete":
            blocks.append(f"*** Delete File: {path}")
            added_any = True
            continue

        if original_content is None or new_content is None:
            continue

        blocks.append(f"*** Update File: {path}")
        blocks.append("@@")
        for line in _split_patch_content_lines(original_content):
            blocks.append(f"-{line}")
        for line in _split_patch_content_lines(new_content):
            blocks.append(f"+{line}")
        if not new_content.endswith("\n"):
            blocks.append("*** End of File")
        added_any = True

    if not added_any:
        return None

    blocks.append("*** End Patch")
    return "\n".join(blocks) + "\n"


def build_file_change_payload(
    file_changes: list[dict[str, Any]],
    *,
    editable_patch_text: str | None = None,
) -> dict[str, Any]:
    """Build shared response metadata for one or more file mutations."""

    payload: dict[str, Any] = {
        "file_changes": file_changes,
        "change_count": len(file_changes),
        "paths": [str(item.get("path")) for item in file_changes if isinstance(item.get("path"), str)],
        "diff_available": any(bool(item.get("diff_available")) for item in file_changes),
        "editable_patch_text": build_editable_patch_text(
            file_changes=file_changes,
            fallback_patch_text=editable_patch_text,
        ),
        "patch_validation_supported": True,
    }
    payload["editable_patch"] = payload["editable_patch_text"]
    if len(file_changes) == 1:
        first = file_changes[0]
        for key in (
            "path",
            "file_path",
            "relative_path",
            "status",
            "change_type",
            "encoding",
            "original_size_bytes",
            "new_size_bytes",
            "added_lines",
            "deleted_lines",
            "original_content",
            "new_content",
            "undo_available",
            "undo_action",
            "undo_reason",
            "diff",
            "diff_available",
            "undo_size_limit_bytes",
            "append",
            "edit_type",
        ):
            if key in first:
                payload[key] = first[key]
    return payload


def summarize_file_change_payload(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the shared file-change summary shape from metadata."""

    if not isinstance(metadata, dict):
        return {}
    file_changes = metadata.get("file_changes")
    if not isinstance(file_changes, list):
        return {}
    summary: dict[str, Any] = {
        "file_changes": file_changes,
        "change_count": int(metadata.get("change_count") or len(file_changes)),
        "paths": list(metadata.get("paths") or []),
        "diff_available": bool(metadata.get("diff_available", False)),
        "editable_patch_text": metadata.get("editable_patch_text"),
        "editable_patch": metadata.get("editable_patch"),
        "patch_validation_supported": bool(metadata.get("patch_validation_supported", False)),
    }
    for key in (
        "path",
        "file_path",
        "relative_path",
        "status",
        "change_type",
        "encoding",
        "original_size_bytes",
        "new_size_bytes",
        "added_lines",
        "deleted_lines",
        "original_content",
        "new_content",
        "undo_available",
        "undo_action",
        "undo_reason",
        "diff",
        "undo_size_limit_bytes",
        "append",
        "edit_type",
    ):
        if key in metadata:
            summary[key] = metadata[key]
    return summary


def parse_apply_patch(patch: str) -> list[PatchOperation]:
    """Parse the strict apply_patch grammar used by Exec-First editing."""

    if not patch.strip():
        raise PatchValidationError("`patch` must not be empty.")

    lines = patch.splitlines()
    if not lines or lines[0] != "*** Begin Patch":
        raise PatchValidationError("Patch must start with '*** Begin Patch'.")
    if len(lines) < 2 or lines[-1] != "*** End Patch":
        raise PatchValidationError("Patch must end with '*** End Patch'.")

    operations: list[PatchOperation] = []
    seen_paths: set[str] = set()
    index = 1
    last_index = len(lines) - 1

    while index < last_index:
        header = lines[index]
        if header.startswith("*** Add File: "):
            path = header[len("*** Add File: ") :].strip()
            _validate_patch_path(path, seen_paths)
            index += 1
            body: list[str] = []
            while index < last_index and not _is_patch_header(lines[index]):
                body.append(lines[index])
                index += 1
            if not body:
                raise PatchValidationError(f"Add File block for '{path}' must include content lines.")
            added_lines: list[str] = []
            for line in body:
                if not line.startswith("+"):
                    raise PatchValidationError(
                        f"Add File block for '{path}' only allows '+' lines."
                    )
                added_lines.append(line[1:])
            seen_paths.add(path)
            operations.append(
                PatchOperation(kind="add", path=path, added_lines=tuple(added_lines))
            )
            continue

        if header.startswith("*** Update File: "):
            path = header[len("*** Update File: ") :].strip()
            _validate_patch_path(path, seen_paths)
            index += 1
            body = []
            while index < last_index and not _is_patch_header(lines[index]):
                body.append(lines[index])
                index += 1
            sections = _parse_patch_sections(path, body)
            seen_paths.add(path)
            operations.append(PatchOperation(kind="update", path=path, sections=sections))
            continue

        if header.startswith("*** Delete File: "):
            path = header[len("*** Delete File: ") :].strip()
            _validate_patch_path(path, seen_paths)
            seen_paths.add(path)
            operations.append(PatchOperation(kind="delete", path=path))
            index += 1
            continue

        raise PatchValidationError(
            "Patch body may only contain '*** Add File', '*** Update File', or "
            "'*** Delete File' blocks."
        )

    if not operations:
        raise PatchValidationError("Patch must contain at least one file operation.")
    return operations


async def prepare_apply_patch(
    *,
    patch: str,
    workspace_dir: str | Path,
    path_scope: str,
    encoding: str,
    undo_max_size_mb: float,
    tool_name: str = "apply_patch",
) -> tuple[list[PreparedPatchOperation], dict[str, Any]]:
    """Resolve an apply_patch request into prepared file changes."""

    workspace_root = normalize_workspace_dir(workspace_dir)
    operations = parse_apply_patch(patch)
    prepared: list[PreparedPatchOperation] = []

    for operation in operations:
        target, security_decision = check_file_tool_path(
            operation.path,
            workspace_dir=workspace_root,
            scope=path_scope,
        )
        if security_decision is not None or target is None:
            raise _patch_validation_from_security(security_decision)

        existed_before = await asyncio.to_thread(target.exists)
        original_content: str | None = None
        new_content: str | None = None

        if operation.kind == "add":
            if existed_before:
                raise PatchValidationError(f"Cannot add file because it already exists: {target}")
            new_content = _render_added_lines(operation.added_lines)
        else:
            if not existed_before:
                raise PatchValidationError(f"Target file not found for patch: {target}")
            if not await asyncio.to_thread(target.is_file):
                raise PatchValidationError(f"Path is not a file: {target}")
            try:
                original_content = await asyncio.to_thread(target.read_text, encoding=encoding)
            except UnicodeDecodeError as exc:
                raise PatchValidationError(f"File is not valid {encoding} text: {target}") from exc
            if operation.kind == "delete":
                new_content = None
            else:
                new_content = _apply_update_sections(
                    target=target,
                    original_content=original_content,
                    sections=operation.sections,
                )

        file_change = build_file_change_entry(
            target=target,
            workspace_root=workspace_root,
            tool_name=tool_name,
            change_type=operation.kind,
            original_content=original_content,
            new_content=new_content,
            encoding=encoding,
            undo_max_size_mb=undo_max_size_mb,
        )
        prepared.append(
            PreparedPatchOperation(
                operation=operation,
                target=target,
                existed_before=existed_before,
                original_content=original_content,
                new_content=new_content,
                file_change=file_change,
            )
        )

    return prepared, build_file_change_payload(
        [item.file_change for item in prepared],
        editable_patch_text=patch,
    )


def _patch_validation_from_security(decision: SecurityDecision | None) -> PatchValidationError:
    if decision is None:
        return PatchValidationError("Path denied.")
    status_code = 403 if decision.approval_scope in {"workspace", "protected_path"} else 400
    return PatchValidationError(decision.reason or "Path denied.", status_code=status_code)


def _validate_patch_path(path: str, seen_paths: set[str]) -> None:
    if not path:
        raise PatchValidationError("Patch file paths must not be empty.")
    if path in seen_paths:
        raise PatchValidationError(f"Patch may only reference each path once: {path}")


def _is_patch_header(line: str) -> bool:
    return line.startswith("*** Add File: ") or line.startswith("*** Update File: ") or line.startswith(
        "*** Delete File: "
    ) or line == "*** End Patch"


def _parse_patch_sections(path: str, body: list[str]) -> tuple[PatchSection, ...]:
    if not body:
        raise PatchValidationError(f"Update File block for '{path}' must include at least one hunk.")

    sections: list[PatchSection] = []
    current_header: str | None = None
    current_lines: list[tuple[str, str]] = []
    reaches_eof = False
    saw_change = False

    def _flush_section() -> None:
        nonlocal current_header, current_lines, reaches_eof
        if current_header is None:
            return
        if not current_lines:
            raise PatchValidationError(
                f"Update File block for '{path}' contains an empty hunk."
            )
        sections.append(
            PatchSection(
                header=current_header,
                lines=tuple(current_lines),
                reaches_eof=reaches_eof,
            )
        )
        current_header = None
        current_lines = []
        reaches_eof = False

    for line in body:
        if line.startswith("@@"):
            _flush_section()
            current_header = line
            continue
        if line == "*** End of File":
            if current_header is None:
                raise PatchValidationError(
                    f"Update File block for '{path}' cannot use '*** End of File' before a hunk."
                )
            reaches_eof = True
            continue
        if not line or line[0] not in {" ", "+", "-"}:
            raise PatchValidationError(
                f"Update File block for '{path}' has an invalid line: {line!r}"
            )
        if current_header is None:
            raise PatchValidationError(
                f"Update File block for '{path}' must start each hunk with '@@'."
            )
        current_lines.append((line[0], line[1:]))
        if line[0] in {"+", "-"}:
            saw_change = True

    _flush_section()
    if not sections:
        raise PatchValidationError(f"Update File block for '{path}' must include at least one hunk.")
    if not saw_change:
        raise PatchValidationError(
            f"Update File block for '{path}' must include at least one '+' or '-' line."
        )
    return tuple(sections)


def _render_added_lines(lines: tuple[str, ...]) -> str:
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _split_patch_content_lines(content: str) -> list[str]:
    if not content:
        return []
    return content.splitlines()


def _apply_update_sections(
    *,
    target: Path,
    original_content: str,
    sections: tuple[PatchSection, ...],
) -> str:
    original_lines = original_content.splitlines()
    trailing_newline = original_content.endswith("\n")
    cursor = 0
    output: list[str] = []

    for section in sections:
        before_lines = [text for prefix, text in section.lines if prefix in {" ", "-"}]
        after_lines = [text for prefix, text in section.lines if prefix in {" ", "+"}]
        if not before_lines:
            output.extend(after_lines)
            continue

        match_index = _find_matching_block(
            haystack=original_lines,
            needle=before_lines,
            start_index=cursor,
        )
        if match_index is None:
            raise PatchValidationError(
                f"Failed to apply patch for '{target}': hunk context was not found."
            )
        output.extend(original_lines[cursor:match_index])
        output.extend(after_lines)
        cursor = match_index + len(before_lines)

    output.extend(original_lines[cursor:])
    if not output:
        return ""
    if trailing_newline:
        return "\n".join(output) + "\n"
    return "\n".join(output)


def _find_matching_block(
    *,
    haystack: list[str],
    needle: list[str],
    start_index: int,
) -> int | None:
    if not needle:
        return start_index

    for index in range(start_index, len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return index

    matches = [
        index
        for index in range(0, len(haystack) - len(needle) + 1)
        if haystack[index : index + len(needle)] == needle
    ]
    if len(matches) == 1:
        return matches[0]
    return None
