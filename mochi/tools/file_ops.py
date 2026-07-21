"""File tools with workspace boundaries, stale-write guards, and patch support."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import stat
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from mochi.config import defaults
from mochi.config.schema import SecurityConfig
from mochi.runtime.change_sets import ChangeSetStore
from mochi.runtime.store import RuntimeStore
from mochi.security import require_approval_decision, with_task_isolation_scope
from mochi.security.file_contract import (
    AuthorizationContext,
    AuthorizationEnvelope,
    ChangeEntry,
    ChangeManifest,
    FileChangeRequest,
    FileIdentity,
    authorization_request_digest,
    canonical_json,
)
from mochi.tools.base import BaseTool, FileReadState, ToolExecutionContext, ToolResult
from mochi.tools.file_mutations import (
    PatchValidationError,
    build_file_change_entry,
    build_file_change_payload,
    prepare_apply_patch,
)
from mochi.utils.security import (
    check_file_tool_path,
    content_size_bytes,
    is_within_write_size_limit,
    normalize_workspace_dir,
    resolve_path_with_scope,
    size_limit_bytes,
)

FileReader = Callable[[Path, str], Awaitable[str]]
FileWriter = Callable[[Path, str, bool, str], Awaitable[int]]
_TOOL_RESULT_PATH_PREFIX = "tool-result://"


def _build_path_denial_metadata(
    *,
    path: str | Path,
    workspace_root: Path,
    path_scope: str,
    security_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Expose the effective path boundary in structured write-denial diagnostics."""
    metadata = dict(security_metadata)
    metadata.update(
        {
            "workspace_dir": str(workspace_root),
            "requested_path": str(path),
            "path_scope": path_scope,
        }
    )
    with contextlib.suppress(OSError, RuntimeError, ValueError):
        metadata["resolved_path"] = str(resolve_path_with_scope(path, workspace_root, "any"))
    return metadata


class FileChangeContractConflict(RuntimeError):
    """Raised when an immutable patch preview no longer matches execution state."""


def file_change_policy_version(security: SecurityConfig) -> str:
    """Return a stable digest for file-authorization policy inputs."""

    projection = {
        "change_contract_mode": security.change_contract_mode,
        "autonomy_mode": security.autonomy_mode,
        "require_approval_for_file_write": security.require_approval_for_file_write,
        "file_write_scope": security.file_write_scope,
        "max_file_write_size_mb": str(security.max_file_write_size_mb),
        "file_undo_max_size_mb": str(security.file_undo_max_size_mb),
    }
    digest = hashlib.sha256(canonical_json(projection).encode("utf-8")).hexdigest()
    return f"file-policy-v1:{digest}"


def _capture_file_identity(path: Path) -> FileIdentity:
    info = path.stat()
    platform = "windows" if os.name == "nt" else "posix"
    attributes = int(getattr(info, "st_file_attributes", 0))
    return FileIdentity(
        platform=platform,
        volume_id=str(int(info.st_dev)),
        file_id=str(int(info.st_ino)),
        link_count=max(1, int(info.st_nlink)),
        is_reparse_point=bool(attributes & 0x400),
    )


def _content_digest(content: bytes | None) -> str | None:
    return None if content is None else hashlib.sha256(content).hexdigest()


def _context_digest(context: AuthorizationContext) -> str:
    payload = canonical_json(context.to_dict()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def prepare_patch_change_contract(
    *,
    runtime_store: RuntimeStore,
    patch: str,
    workspace_dir: str | Path,
    security: SecurityConfig,
    requester_id: str,
    session_id: str,
    task_id: str | None,
    encoding: str = "utf-8",
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Prepare and persist a canonical immutable patch manifest."""

    workspace_root = normalize_workspace_dir(workspace_dir)
    prepared, change_payload = await prepare_apply_patch(
        patch=patch,
        workspace_dir=workspace_root,
        path_scope=security.file_write_scope,
        encoding=encoding,
        undo_max_size_mb=security.file_undo_max_size_mb,
    )
    change_store = ChangeSetStore(runtime_store)
    workspace_identity = await asyncio.to_thread(_capture_file_identity, workspace_root)
    policy_version = file_change_policy_version(security)
    patch_sha256 = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    context = AuthorizationContext(
        requester_id=requester_id,
        session_id=session_id,
        task_id=task_id,
        workspace_root=str(workspace_root),
        workspace_identity=workspace_identity,
    )

    entries: list[ChangeEntry] = []
    for ordinal, item in enumerate(prepared):
        before = (
            None
            if not item.existed_before
            else await asyncio.to_thread(item.target.read_bytes)
        )
        after = None if item.new_content is None else item.new_content.encode(encoding)
        before_blob_id = None if before is None else await change_store.put_blob(before)
        after_blob_id = None if after is None else await change_store.put_blob(after)
        base_identity = (
            None
            if not item.existed_before
            else await asyncio.to_thread(_capture_file_identity, item.target)
        )
        mode_before = None
        if item.existed_before:
            info = await asyncio.to_thread(item.target.stat)
            mode_before = stat.S_IMODE(info.st_mode)
        relative_path = item.target.relative_to(workspace_root).as_posix()
        entry_seed = canonical_json(
            {
                "context": context.to_dict(),
                "policy_version": policy_version,
                "patch_sha256": patch_sha256,
                "ordinal": ordinal,
                "relative_path": relative_path,
                "operation": item.operation.kind,
                "base_sha256": _content_digest(before),
                "after_sha256": _content_digest(after),
            }
        )
        entries.append(
            ChangeEntry(
                entry_id=hashlib.sha256(entry_seed.encode("utf-8")).hexdigest(),
                relative_path=relative_path,
                operation=item.operation.kind,
                base_sha256=_content_digest(before),
                after_sha256=_content_digest(after),
                base_identity=base_identity,
                before_blob_id=before_blob_id,
                after_blob_id=after_blob_id,
                mode_before=mode_before,
                mode_after=mode_before if after is not None else None,
                base_metadata_sha256=None,
                after_metadata_sha256=None,
                rename_source=None,
                dependency_group=None,
            )
        )

    file_request = FileChangeRequest(entries=tuple(entries), patch_sha256=patch_sha256)
    envelope = AuthorizationEnvelope(
        schema_version=1,
        kind="file_change",
        context=context,
        policy_version=policy_version,
        file_request=file_request,
        exec_request=None,
    )
    request_digest = authorization_request_digest(envelope)
    now = datetime.now(UTC)
    manifest = ChangeManifest(
        version=1,
        change_set_id=str(uuid4()),
        workspace_root=str(workspace_root),
        workspace_identity=workspace_identity,
        tool_name="apply_patch",
        intent="mutate",
        entries=file_request.entries,
        patch_sha256=patch_sha256,
        policy_version=policy_version,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=15)).isoformat(),
        request_digest=request_digest,
        ui_metadata={
            "encoding": encoding,
            "change_count": change_payload.get("change_count", 0),
        },
    )
    persisted = await change_store.persist_manifest(manifest, envelope)
    stored_manifest = persisted["manifest"]
    contract = {
        "change_set_id": stored_manifest.change_set_id,
        "request_digest": stored_manifest.request_digest,
        "expires_at": stored_manifest.expires_at,
        "policy_version": stored_manifest.policy_version,
        "change_contract_mode": security.change_contract_mode,
        "context_digest": _context_digest(envelope.context),
    }
    return prepared, change_payload, contract


async def revalidate_patch_change_contract(
    *,
    runtime_store: RuntimeStore,
    approval: Mapping[str, Any],
    task: Mapping[str, Any],
    security: SecurityConfig,
) -> ChangeManifest:
    """Reload and compare a bound manifest immediately before mutation."""

    metadata = approval.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    change_set_id = metadata.get("change_set_id")
    if not isinstance(change_set_id, str) or not change_set_id:
        raise FileChangeContractConflict("change_set_missing")

    change_store = ChangeSetStore(runtime_store)
    persisted = await change_store.get_change_set(change_set_id)
    if persisted is None:
        raise FileChangeContractConflict("change_set_missing")
    manifest = persisted["manifest"]
    envelope = persisted["envelope"]
    try:
        if persisted["status"] != "prepared":
            raise FileChangeContractConflict("change_set_not_prepared")
        if datetime.fromisoformat(manifest.expires_at) <= datetime.now(UTC):
            raise FileChangeContractConflict("change_set_expired")
        if approval.get("request_digest") != manifest.request_digest:
            raise FileChangeContractConflict("approval_digest_mismatch")
        if metadata.get("request_digest") != manifest.request_digest:
            raise FileChangeContractConflict("approval_metadata_digest_mismatch")
        if manifest.policy_version != file_change_policy_version(security):
            raise FileChangeContractConflict("policy_changed")
        if str(task.get("id") or "") != str(envelope.context.task_id or ""):
            raise FileChangeContractConflict("task_context_changed")
        if str(task.get("session_id") or "draft-session") != envelope.context.session_id:
            raise FileChangeContractConflict("session_context_changed")

        workspace_value = (
            task.get("task_workspace_dir")
            or task.get("project_workspace_dir")
            or task.get("workspace_dir")
        )
        if not isinstance(workspace_value, str):
            raise FileChangeContractConflict("workspace_context_missing")
        workspace_root = normalize_workspace_dir(workspace_value)
        if str(workspace_root) != manifest.workspace_root:
            raise FileChangeContractConflict("workspace_context_changed")
        current_workspace_identity = await asyncio.to_thread(
            _capture_file_identity,
            workspace_root,
        )
        if current_workspace_identity != manifest.workspace_identity:
            raise FileChangeContractConflict("workspace_identity_changed")

        arguments = approval.get("arguments")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        patch = arguments.get("patch")
        if not isinstance(patch, str):
            raise FileChangeContractConflict("server_patch_missing")
        if hashlib.sha256(patch.encode("utf-8")).hexdigest() != manifest.patch_sha256:
            raise FileChangeContractConflict("server_patch_digest_mismatch")

        for entry in manifest.entries:
            target = workspace_root / Path(entry.relative_path)
            exists = await asyncio.to_thread(target.exists)
            if entry.operation == "add":
                if exists:
                    raise FileChangeContractConflict("base_path_now_exists")
                continue
            if not exists or not await asyncio.to_thread(target.is_file):
                raise FileChangeContractConflict("base_path_missing")
            current = await asyncio.to_thread(target.read_bytes)
            if _content_digest(current) != entry.base_sha256:
                raise FileChangeContractConflict("base_content_changed")
            identity = await asyncio.to_thread(_capture_file_identity, target)
            if identity != entry.base_identity:
                raise FileChangeContractConflict("base_identity_changed")
    except FileChangeContractConflict:
        await change_store.mark_change_set_status(change_set_id, status="conflicted")
        raise
    return manifest


class FileReadTool(BaseTool):
    """Read a text file from the local filesystem."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        path_scope: str = "workspace",
        default_encoding: str = "utf-8",
        max_read_bytes: int = 1024 * 1024,
        reader: FileReader | None = None,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._path_scope = path_scope
        self._default_encoding = default_encoding
        self._max_read_bytes = max_read_bytes
        self._reader = reader or self._default_reader

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a local text file. Returns full text content. "
            "Use when you need to inspect code, configuration, or notes. "
            "Cannot read binary files."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local file path to read."},
                "encoding": {"type": "string", "default": "utf-8"},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum bytes allowed for this read. Overrides the default limit.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Starting line number for partial reads. Uses 1-based indexing.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of lines to return from the starting line.",
                },
                "line_numbers": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to prefix returned lines with their line number.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    @property
    def allow_plain_text_result_for_model(self) -> bool:
        return True

    def _resolve_workspace_root(self, context: ToolExecutionContext | None) -> Path:
        if context is not None:
            for candidate in (
                context.task_sandbox_dir,
                context.project_workspace,
                context.workspace_dir,
            ):
                if candidate:
                    return normalize_workspace_dir(candidate)
        return self._workspace_dir

    async def execute(
        self,
        *,
        path: str,
        encoding: str | None = None,
        max_bytes: int | None = None,
        offset: int = 1,
        limit: int | None = None,
        line_numbers: bool = True,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not path.strip():
            return ToolResult(error="`path` must not be empty.")
        if offset <= 0:
            return ToolResult(error="`offset` must be greater than 0.")
        if limit is not None and limit <= 0:
            return ToolResult(error="`limit` must be greater than 0.")

        target: Path
        metadata_path = path
        reference_metadata: dict[str, Any] = {}
        if path.startswith(_TOOL_RESULT_PATH_PREFIX):
            target_result = self._resolve_tool_result_reference(path=path, context=context)
            if target_result.error is not None:
                return target_result
            target = Path(str(target_result.output))
            reference_metadata = dict(target_result.metadata)
        else:
            workspace_root = self._resolve_workspace_root(context)
            target, security_decision = check_file_tool_path(
                path,
                workspace_dir=workspace_root,
                scope=self._path_scope,
                access="read",
            )
            if security_decision is not None or target is None:
                return ToolResult(
                    error=security_decision.reason if security_decision is not None else "Path denied.",
                    metadata=security_decision.to_metadata() if security_decision is not None else {},
                )
            metadata_path = str(target)

        active_encoding = (
            str(reference_metadata.get("encoding"))
            if encoding is None and isinstance(reference_metadata.get("encoding"), str)
            else (encoding or self._default_encoding)
        )

        if not target.exists():
            return ToolResult(error=f"File not found: {target}")
        if not target.is_file():
            return ToolResult(error=f"Path is not a file: {target}")

        effective_max_bytes = max_bytes if max_bytes is not None else self._max_read_bytes
        if effective_max_bytes <= 0:
            return ToolResult(error="`max_bytes` must be greater than 0.")

        file_size = await asyncio.to_thread(lambda: target.stat().st_size)
        if file_size > effective_max_bytes:
            if limit is None:
                retry_call = (
                    f'file_read(path="{path}", offset=1, limit=200, line_numbers=True)'
                )
                message = (
                    f"File is larger than the current read limit ({file_size} bytes exceeds "
                    f"{effective_max_bytes} bytes). Retry with a bounded line chunk, for example: "
                    f"{retry_call}"
                )
                return ToolResult(
                    output=message,
                    metadata={
                        "path": metadata_path,
                        "size_bytes": file_size,
                        "partial": True,
                        "line_numbers": True,
                        "encoding": active_encoding,
                        **reference_metadata,
                    },
                    suggestion=retry_call,
                )

            chunk_result = await self._read_chunk_from_path(
                target=target,
                encoding=active_encoding,
                offset=offset,
                limit=limit,
                line_numbers=line_numbers,
            )
            chunk_result.metadata.update(
                {
                    "path": metadata_path,
                    "size_bytes": file_size,
                    "encoding": active_encoding,
                    **reference_metadata,
                }
            )
            if chunk_result.error is not None:
                return chunk_result
            return chunk_result

        text = await self._reader(target, active_encoding)
        lines = text.splitlines()
        if lines and offset > len(lines):
            return ToolResult(
                error=(
                    f"File exists but is shorter than the provided offset ({offset}). "
                    f"The file has {len(lines)} lines."
                ),
                metadata={
                    "path": metadata_path,
                    "total_lines": len(lines),
                    "partial": False,
                    "encoding": active_encoding,
                    **reference_metadata,
                },
            )

        rendered_text = text
        partial = False
        start_line = 1
        end_line = len(lines)
        if limit is not None or offset != 1:
            partial = True
            start_idx = offset - 1
            selected_lines = lines[start_idx:] if limit is None else lines[start_idx : start_idx + limit]
            start_line = offset
            end_line = offset + len(selected_lines) - 1 if selected_lines else offset - 1
            if line_numbers:
                rendered_text = "\n".join(
                    f"{line_no}: {line}"
                    for line_no, line in enumerate(selected_lines, start=offset)
                )
            else:
                rendered_text = "\n".join(selected_lines)

        if context is not None:
            stat = await asyncio.to_thread(target.stat)
            context.read_state_cache[str(target)] = FileReadState(
                path=str(target),
                content=text,
                encoding=active_encoding,
                mtime_ns=getattr(stat, "st_mtime_ns", None),
                size_bytes=file_size,
                partial=partial,
            )

        return ToolResult(
            output=rendered_text,
            metadata={
                "path": metadata_path,
                "size_bytes": file_size,
                "partial": partial,
                "start_line": start_line,
                "end_line": end_line,
                "total_lines": len(lines),
                "line_numbers": line_numbers,
                "encoding": active_encoding,
                **reference_metadata,
            },
        )

    @staticmethod
    async def _default_reader(path: Path, encoding: str) -> str:
        return await asyncio.to_thread(path.read_text, encoding=encoding)

    def _resolve_tool_result_reference(
        self,
        *,
        path: str,
        context: ToolExecutionContext | None,
    ) -> ToolResult:
        if context is None:
            return ToolResult(error="`tool-result://` reads require an execution context.")

        reference_id = path[len(_TOOL_RESULT_PATH_PREFIX) :].strip()
        if not reference_id:
            return ToolResult(error="`tool-result://` path must include a reference id.")

        return self._resolve_tool_result_reference_id(reference_id=reference_id, context=context)

    def _resolve_tool_result_reference_id(
        self,
        *,
        reference_id: str,
        context: ToolExecutionContext | None,
    ) -> ToolResult:
        if context is None:
            return ToolResult(error="Tool result reads require an execution context.")

        if not reference_id.strip():
            return ToolResult(error="`reference_id` must not be empty.")

        reference = context.tool_result_references.get(reference_id)
        if not isinstance(reference, dict):
            return ToolResult(error=f"Unknown tool result reference: {reference_id}")

        artifact_path = reference.get("artifact_path")
        if not isinstance(artifact_path, str) or not artifact_path.strip():
            return ToolResult(error=f"Tool result reference is missing artifact_path: {reference_id}")

        source_path = reference.get("source_path")
        artifact_encoding = reference.get("artifact_encoding", self._default_encoding)
        continuation_target = artifact_path
        continuation_encoding = artifact_encoding
        if isinstance(source_path, str) and source_path.strip():
            source_candidate = Path(source_path)
            if source_candidate.exists() and source_candidate.is_file():
                continuation_target = source_path
                continuation_encoding = reference.get("encoding", self._default_encoding)

        return ToolResult(
            output=continuation_target,
            metadata={
                "reference_id": reference_id,
                "artifact_path": artifact_path,
                "artifact_encoding": artifact_encoding,
                "source_path": source_path,
                "tool_name": reference.get("tool_name"),
                "encoding": continuation_encoding,
            },
        )

    async def _read_chunk_from_path(
        self,
        *,
        target: Path,
        encoding: str,
        offset: int,
        limit: int,
        line_numbers: bool,
    ) -> ToolResult:
        def _sync_read() -> tuple[str, int, int, int]:
            selected_lines: list[str] = []
            total_lines = 0
            with target.open("r", encoding=encoding) as file:
                for line_no, raw_line in enumerate(file, start=1):
                    total_lines = line_no
                    if line_no < offset:
                        continue
                    if len(selected_lines) >= limit:
                        continue
                    selected_lines.append(raw_line.rstrip("\r\n"))

            if total_lines > 0 and offset > total_lines:
                raise ValueError(
                    f"File exists but is shorter than the provided offset ({offset}). "
                    f"The file has {total_lines} lines."
                )

            start_line = offset
            end_line = offset + len(selected_lines) - 1 if selected_lines else offset - 1
            if line_numbers:
                rendered = "\n".join(
                    f"{line_no}: {line}"
                    for line_no, line in enumerate(selected_lines, start=offset)
                )
            else:
                rendered = "\n".join(selected_lines)
            return rendered, start_line, end_line, total_lines

        try:
            rendered_text, start_line, end_line, total_lines = await asyncio.to_thread(_sync_read)
        except UnicodeDecodeError:
            return ToolResult(error=f"File is not valid {encoding} text: {target}")
        except ValueError as exc:
            return ToolResult(error=str(exc))

        return ToolResult(
            output=rendered_text,
            metadata={
                "partial": True,
                "start_line": start_line,
                "end_line": end_line,
                "total_lines": total_lines,
                "line_numbers": line_numbers,
            },
        )


class FileWriteTool(BaseTool):
    """Write text to a workspace file with undo metadata."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        path_scope: str = "workspace",
        require_approval: bool = True,
        max_write_size_mb: float = 10.0,
        undo_max_size_mb: float = 2.0,
        default_encoding: str = "utf-8",
        writer: FileWriter | None = None,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._path_scope = path_scope
        self._require_approval = require_approval
        self._max_write_size_mb = max_write_size_mb
        self._undo_max_size_mb = undo_max_size_mb
        self._default_encoding = default_encoding
        self._writer = writer or self._default_writer

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return (
            "Write text to a file inside the workspace, either replacing or appending. "
            "Use for controlled text or code updates after deciding on the content. "
            "Cannot write outside the workspace and may require approval."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path. Must be inside the workspace."},
                "content": {"type": "string", "description": "Text content to write."},
                "append": {"type": "boolean", "default": False, "description": "Append instead of replacing the file."},
                "encoding": {"type": "string", "default": "utf-8"},
                "approved": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether user approval has been granted. Required when require_approval is true.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    @property
    def requires_approval(self) -> bool:
        return self._require_approval

    def _resolve_workspace_root(self, context: ToolExecutionContext | None) -> Path:
        if context is not None:
            for candidate in (
                context.task_sandbox_dir,
                context.project_workspace,
                context.workspace_dir,
            ):
                if candidate:
                    return normalize_workspace_dir(candidate)
        return self._workspace_dir

    async def execute(
        self,
        *,
        path: str,
        content: str,
        append: bool = False,
        encoding: str | None = None,
        approved: bool = False,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not path.strip():
            return ToolResult(error="`path` must not be empty.")

        workspace_root = self._resolve_workspace_root(context)
        active_encoding = encoding or self._default_encoding
        if not is_within_write_size_limit(
            content=content,
            max_size_mb=self._max_write_size_mb,
            encoding=active_encoding,
        ):
            content_size = content_size_bytes(content, encoding=active_encoding)
            return ToolResult(
                error=(
                    f"Write content too large: {content_size} bytes exceeds limit "
                    f"{size_limit_bytes(self._max_write_size_mb)} bytes."
                ),
                metadata={"size_bytes": content_size},
            )

        target, security_decision = check_file_tool_path(
            path,
            workspace_dir=workspace_root,
            scope=self._path_scope,
        )
        if security_decision is not None or target is None:
            metadata = _build_path_denial_metadata(
                path=path,
                workspace_root=workspace_root,
                path_scope=self._path_scope,
                security_metadata=(
                    security_decision.to_metadata() if security_decision is not None else {}
                ),
            )
            if security_decision is not None:
                metadata.update(
                    {
                        "runtime_category": "permission",
                        "error_type": "file_path_denied",
                        "recoverability": "requires_changed_path_or_policy",
                    }
                )
            return ToolResult(
                error=security_decision.reason if security_decision is not None else "Path denied.",
                metadata=metadata,
            )

        existing_result = await _load_existing_content(target, active_encoding)
        if existing_result.error is not None:
            return existing_result
        existing_content = str(existing_result.output)
        existed_before = target.exists()

        guard_error = await _check_stale_write_guard(
            target=target,
            current_content=existing_content,
            context=context,
        )
        if guard_error is not None:
            return guard_error

        merged_content = existing_content + content if append else content
        file_change = build_file_change_entry(
            target=target,
            workspace_root=workspace_root,
            tool_name=self.name,
            change_type="add" if not existed_before and not append else "update",
            original_content=existing_content,
            new_content=merged_content,
            encoding=active_encoding,
            undo_max_size_mb=self._undo_max_size_mb,
            extra={"append": append},
        )
        metadata = build_file_change_payload([file_change])
        metadata.update(
            {
                'workspace_dir': str(workspace_root),
                'resolved_path': str(target),
                'path_scope': self._path_scope,
            }
        )
        policy_requires_approval = (
            context is not None
            and bool(context.permission_policy.get("require_approval_for_file_write"))
        )
        if (self._require_approval or policy_requires_approval) and not approved:
            decision = require_approval_decision(
                reason="File writes require explicit approval in the current autonomy mode.",
                approval_kind="file_write",
                approval_scope="workspace",
                replay_safe=True,
                policy_source="runtime_policy",
            )
            decision = with_task_isolation_scope(
                decision,
                task_sandbox_dir=context.task_sandbox_dir if context is not None else None,
            )
            metadata.update(decision.to_metadata())
            return ToolResult(
                error="File write requires approval.",
                metadata=metadata,
            )

        bytes_written = await self._writer(target, content if append else merged_content, append, active_encoding)

        await _refresh_read_state_cache(
            context=context,
            target=target,
            content=merged_content,
            encoding=active_encoding,
        )
        metadata["bytes_written"] = bytes_written
        metadata["append"] = append
        return ToolResult(output=str(target), metadata=metadata)

    @staticmethod
    async def _default_writer(path: Path, content: str, append: bool, encoding: str) -> int:
        def _sync_write() -> int:
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with path.open(mode=mode, encoding=encoding, newline="") as file:
                file.write(content)
            return len(content.encode(encoding))

        return await asyncio.to_thread(_sync_write)


class FileEditTool(FileWriteTool):
    """Edit a file using old_string/new_string replacement semantics."""

    @property
    def name(self) -> str:
        return "file_edit"

    @property
    def description(self) -> str:
        return (
            "Edit a previously read text file by replacing old_string with new_string. "
            "Use this for incremental code or document edits instead of full-file rewrites."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path. Must be inside the workspace."},
                "old_string": {"type": "string", "description": "Text to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "default": False,
                    "description": "Replace every match instead of only the first one.",
                },
                "encoding": {"type": "string", "default": "utf-8"},
                "approved": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether user approval has been granted when required.",
                },
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        encoding: str | None = None,
        approved: bool = False,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not old_string:
            return ToolResult(error="`old_string` must not be empty.")

        workspace_root = self._resolve_workspace_root(context)
        active_encoding = encoding or self._default_encoding
        target, security_decision = check_file_tool_path(
            path,
            workspace_dir=workspace_root,
            scope=self._path_scope,
        )
        if security_decision is not None or target is None:
            metadata = _build_path_denial_metadata(
                path=path,
                workspace_root=workspace_root,
                path_scope=self._path_scope,
                security_metadata=(
                    security_decision.to_metadata() if security_decision is not None else {}
                ),
            )
            if security_decision is not None:
                metadata.update(
                    {
                        "runtime_category": "permission",
                        "error_type": "file_path_denied",
                        "recoverability": "requires_changed_path_or_policy",
                    }
                )
            return ToolResult(
                error=security_decision.reason if security_decision is not None else "Path denied.",
                metadata=metadata,
            )

        existing_result = await _load_existing_content(target, active_encoding)
        if existing_result.error is not None:
            return existing_result
        existing_content = str(existing_result.output)

        guard_error = await _check_stale_write_guard(
            target=target,
            current_content=existing_content,
            context=context,
            require_prior_read=True,
        )
        if guard_error is not None:
            return guard_error

        if old_string not in existing_content:
            return ToolResult(
                error="`old_string` was not found in the file. Re-read the file before editing.",
                suggestion="Read the latest file contents and retry with an exact match.",
            )

        if replace_all:
            new_content = existing_content.replace(old_string, new_string)
        else:
            new_content = existing_content.replace(old_string, new_string, 1)

        file_change = build_file_change_entry(
            target=target,
            workspace_root=workspace_root,
            tool_name=self.name,
            change_type="update",
            original_content=existing_content,
            new_content=new_content,
            encoding=active_encoding,
            undo_max_size_mb=self._undo_max_size_mb,
            extra={"append": False, "edit_type": "replace_all" if replace_all else "replace_first"},
        )
        metadata = build_file_change_payload([file_change])
        metadata.update(
            {
                'workspace_dir': str(workspace_root),
                'resolved_path': str(target),
                'path_scope': self._path_scope,
            }
        )
        policy_requires_approval = (
            context is not None
            and bool(context.permission_policy.get("require_approval_for_file_write"))
        )
        if (self._require_approval or policy_requires_approval) and not approved:
            decision = require_approval_decision(
                reason="File edits require explicit approval in the current autonomy mode.",
                approval_kind="file_edit",
                approval_scope="workspace",
                replay_safe=True,
                policy_source="runtime_policy",
            )
            decision = with_task_isolation_scope(
                decision,
                task_sandbox_dir=context.task_sandbox_dir if context is not None else None,
            )
            metadata.update(decision.to_metadata())
            return ToolResult(
                error="File edit requires approval.",
                metadata=metadata,
            )

        bytes_written = await self._writer(target, new_content, False, active_encoding)
        await _refresh_read_state_cache(
            context=context,
            target=target,
            content=new_content,
            encoding=active_encoding,
        )
        metadata["bytes_written"] = bytes_written
        return ToolResult(output=str(target), metadata=metadata)


class ApplyPatchTool(FileWriteTool):
    """Apply a strict multi-file patch inside the workspace."""

    @property
    def name(self) -> str:
        return "apply_patch"

    @property
    def description(self) -> str:
        return (
            "Apply a strict patch using *** Begin Patch / *** Add File / "
            "*** Update File / *** Delete File / *** End Patch blocks."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Strict apply_patch payload."},
                "encoding": {"type": "string", "default": "utf-8"},
                "approved": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether user approval has been granted when required.",
                },
            },
            "required": ["patch"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        patch: str,
        encoding: str | None = None,
        approved: bool = False,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        workspace_root = self._resolve_workspace_root(context)
        active_encoding = encoding or self._default_encoding
        try:
            prepared, metadata = await prepare_apply_patch(
                patch=patch,
                workspace_dir=workspace_root,
                path_scope=self._path_scope,
                encoding=active_encoding,
                undo_max_size_mb=self._undo_max_size_mb,
                tool_name=self.name,
            )
        except PatchValidationError as exc:
            return ToolResult(
                error=str(exc),
                metadata=dict(getattr(exc, "metadata", {}) or {}),
            )

        for item in prepared:
            if item.new_content is not None and not is_within_write_size_limit(
                content=item.new_content,
                max_size_mb=self._max_write_size_mb,
                encoding=active_encoding,
            ):
                size_bytes = content_size_bytes(item.new_content, encoding=active_encoding)
                return ToolResult(
                    error=(
                        f"Write content too large: {size_bytes} bytes exceeds limit "
                        f"{size_limit_bytes(self._max_write_size_mb)} bytes."
                    ),
                    metadata={"size_bytes": size_bytes, **metadata},
                )

            if item.original_content is not None:
                guard_error = await _check_stale_write_guard(
                    target=item.target,
                    current_content=item.original_content,
                    context=context,
                    require_prior_read=item.operation.kind != "add",
                )
                if guard_error is not None:
                    return guard_error

        policy_requires_approval = (
            context is not None
            and bool(context.permission_policy.get("require_approval_for_file_write"))
        )
        if (self._require_approval or policy_requires_approval) and not approved:
            decision = require_approval_decision(
                reason="Patch application requires explicit approval in the current autonomy mode.",
                approval_kind="apply_patch",
                approval_scope="workspace",
                replay_safe=True,
                policy_source="runtime_policy",
            )
            decision = with_task_isolation_scope(
                decision,
                task_sandbox_dir=context.task_sandbox_dir if context is not None else None,
            )
            metadata.update(decision.to_metadata())
            return ToolResult(
                error="Patch application requires approval.",
                metadata=metadata,
            )

        total_bytes_written = 0
        for item in prepared:
            if item.operation.kind == "delete":
                if await asyncio.to_thread(item.target.exists):
                    await asyncio.to_thread(item.target.unlink)
                if context is not None:
                    context.read_state_cache.pop(str(item.target), None)
                continue

            new_content = item.new_content or ""
            total_bytes_written += await self._writer(item.target, new_content, False, active_encoding)
            await _refresh_read_state_cache(
                context=context,
                target=item.target,
                content=new_content,
                encoding=active_encoding,
            )

        metadata["bytes_written"] = total_bytes_written
        return ToolResult(
            output={"paths": metadata.get("paths", []), "change_count": metadata.get("change_count", 0)},
            metadata=metadata,
        )


async def _load_existing_content(target: Path, encoding: str) -> ToolResult:
    if not target.exists():
        return ToolResult(output="")
    if not target.is_file():
        return ToolResult(error=f"Path is not a file: {target}")
    try:
        text = await asyncio.to_thread(target.read_text, encoding=encoding)
    except UnicodeDecodeError:
        return ToolResult(error=f"File is not valid {encoding} text: {target}")
    return ToolResult(output=text)


async def _check_stale_write_guard(
    *,
    target: Path,
    current_content: str,
    context: ToolExecutionContext | None,
    require_prior_read: bool | None = None,
) -> ToolResult | None:
    if context is None:
        return None
    if context.state.get("approval_replay") is True:
        return None

    if require_prior_read is None:
        require_prior_read = target.exists()

    snapshot = context.read_state_cache.get(str(target))
    if snapshot is None:
        if require_prior_read:
            return ToolResult(
                error="File must be read before write/edit.",
                suggestion="Use file_read on the target file before modifying it.",
            )
        return None

    if snapshot.partial:
        return ToolResult(
            error="Partial reads cannot be used for write/edit. Re-read the full file first.",
            suggestion="Read the full file contents before modifying it.",
        )

    if target.exists():
        stat = await asyncio.to_thread(target.stat)
        current_mtime = getattr(stat, "st_mtime_ns", None)
        if snapshot.mtime_ns is not None and current_mtime != snapshot.mtime_ns:
            return ToolResult(
                error="File changed after it was read. Re-read before writing.",
                retryable=True,
                suggestion="Run file_read again to refresh the cached snapshot.",
            )
    elif require_prior_read:
        return ToolResult(
            error="File was removed after it was read. Re-read before writing.",
            retryable=True,
        )

    if snapshot.content != current_content:
        return ToolResult(
            error="File contents are stale compared with the cached read. Re-read before writing.",
            retryable=True,
        )

    return None


async def _refresh_read_state_cache(
    *,
    context: ToolExecutionContext | None,
    target: Path,
    content: str,
    encoding: str,
) -> None:
    if context is None:
        return
    stat = await asyncio.to_thread(target.stat)
    context.read_state_cache[str(target)] = FileReadState(
        path=str(target),
        content=content,
        encoding=encoding,
        mtime_ns=getattr(stat, "st_mtime_ns", None),
        size_bytes=getattr(stat, "st_size", content_size_bytes(content, encoding=encoding)),
        partial=False,
    )
