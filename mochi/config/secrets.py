"""Secret-field discovery and encrypted persistence for MochiConfig."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, get_args, get_origin

from pydantic import BaseModel, SecretStr

from mochi.config.identity import configured_model_target_id
from mochi.security.secret_store import SecretStore, SecretStoreError, SecretStoreUnavailable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SecretSlot:
    """A mutable SecretStr field and its stable logical scope."""

    owner: BaseModel
    field_name: str
    scope: str
    value: SecretStr | None


def _contains_secret(annotation: Any) -> bool:
    if annotation is SecretStr:
        return True
    origin = get_origin(annotation)
    if origin in (UnionType,):
        return any(_contains_secret(item) for item in get_args(annotation))
    # typing.Union and Annotated/other wrappers expose their children through
    # get_args as well; inspecting all arguments is harmless for scalar types.
    return bool(get_args(annotation)) and any(
        _contains_secret(item) for item in get_args(annotation)
    )


def _collection_segment(item: Any, index: int) -> str:
    # Configured model ids must remain stable when the UI reorders entries.
    if hasattr(item, "provider") and hasattr(item, "model_spec") and hasattr(item, "model"):
        try:
            return f"target:{configured_model_target_id(item)}"
        except Exception:  # pragma: no cover - defensive for third-party models
            pass
    item_id = getattr(item, "id", None)
    if isinstance(item_id, str) and item_id.strip():
        return f"id:{item_id.strip()}"
    return f"index:{index}"


def iter_secret_slots(value: Any, *, _path: tuple[str, ...] = ()) -> Iterator[SecretSlot]:
    """Yield all Pydantic SecretStr fields, including currently empty fields."""

    if isinstance(value, BaseModel):
        for field_name, field in type(value).model_fields.items():
            child = getattr(value, field_name, None)
            child_path = (*_path, field_name)
            if _contains_secret(field.annotation):
                yield SecretSlot(
                    owner=value,
                    field_name=field_name,
                    scope="/".join(child_path),
                    value=child if isinstance(child, SecretStr) else None,
                )
            else:
                yield from iter_secret_slots(child, _path=child_path)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_secret_slots(child, _path=(*_path, str(key)))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from iter_secret_slots(
                child,
                _path=(*_path, _collection_segment(child, index)),
            )


def secret_reference(scope: str) -> str:
    """Return a non-sensitive reference key for the encrypted store."""

    digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()
    return f"config-v1:{digest}"


def _store_for_config(config_path: str | Path | None) -> SecretStore:
    return SecretStore.for_config_path(config_path)


def hydrate_config_secrets(
    config: BaseModel,
    config_path: str | Path | None,
) -> bool:
    """Load encrypted values and migrate any legacy plaintext SecretStr values.

    Loading may continue when no encrypted store exists and all fields are empty;
    a legacy plaintext value instead fails closed until a protected key source is
    configured, so it can never be written back accidentally.
    """

    slots = list(iter_secret_slots(config))
    raw_slots = [
        slot
        for slot in slots
        if slot.value is not None and slot.value.get_secret_value().strip()
    ]
    store_path = (
        Path(config_path).expanduser().parent / "secrets.enc"
        if config_path is not None
        else Path(".mochi") / "secrets.enc"
    )
    if not raw_slots and not store_path.exists():
        return False
    try:
        store = _store_for_config(config_path)
    except SecretStoreUnavailable as exc:
        if raw_slots or store_path.exists():
            logger.error("Mochi secret migration cannot proceed: %s", exc)
            raise
        return False
    migrated = False
    changes: dict[str, str] = {}
    pending: list[tuple[SecretSlot, SecretStr | None]] = []
    for slot in slots:
        reference = secret_reference(slot.scope)
        if slot.value is not None:
            raw = slot.value.get_secret_value().strip()
            if raw:
                # Defer the in-memory update until every encrypted write has
                # succeeded, so a failed migration leaves the config intact.
                changes[reference] = raw
                migrated = True
            else:
                pending.append((slot, None))
            continue
        stored = store.get(reference)
        if stored:
            pending.append((slot, SecretStr(stored)))

    if changes:
        store.mutate(changes)
    for slot, value in pending:
        setattr(slot.owner, slot.field_name, value)
    return migrated


def persist_config_secrets(
    config: BaseModel,
    config_path: str | Path | None,
    *,
    clear_scopes: Iterable[str] = (),
) -> None:
    """Persist SecretStr values encrypted and fail closed before YAML encoding."""

    slots = list(iter_secret_slots(config))
    if not slots:
        return
    store_path = (
        Path(config_path).expanduser().parent / "secrets.enc"
        if config_path is not None
        else Path(".mochi") / "secrets.enc"
    )
    has_values = any(
        slot.value is not None and slot.value.get_secret_value().strip()
        for slot in slots
    )
    if not has_values and not store_path.exists():
        return
    try:
        store = _store_for_config(config_path)
    except SecretStoreUnavailable:
        # No secret value means there is nothing to persist. If values exist,
        # propagate the error so the caller cannot accidentally write plaintext.
        if has_values:
            raise
        return
    changes: dict[str, str | None] = {}
    for slot in slots:
        reference = secret_reference(slot.scope)
        value = slot.value.get_secret_value().strip() if slot.value is not None else ""
        if value:
            changes[reference] = value
    for scope in clear_scopes:
        normalized = scope.strip()
        if normalized:
            changes[secret_reference(normalized)] = None
    store.mutate(changes)


def delete_config_secret_scopes(
    config_path: str | Path | None,
    scopes: Iterable[str],
) -> None:
    """Delete explicitly cleared credentials after a successful config save."""

    normalized = [scope.strip() for scope in scopes if scope and scope.strip()]
    if not normalized:
        return
    store_path = (
        Path(config_path).expanduser().parent / "secrets.enc"
        if config_path is not None
        else Path(".mochi") / "secrets.enc"
    )
    if not store_path.exists():
        return
    try:
        store = _store_for_config(config_path)
    except SecretStoreUnavailable:
        raise
    store.mutate({secret_reference(scope): None for scope in normalized})


__all__ = [
    "SecretSlot",
    "hydrate_config_secrets",
    "iter_secret_slots",
    "delete_config_secret_scopes",
    "persist_config_secrets",
    "secret_reference",
]
