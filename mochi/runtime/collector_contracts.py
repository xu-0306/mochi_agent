"""Collector shard/provenance normalization helpers for dataset packages."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
import json
from typing import Any, Mapping

COLLECTOR_SHARD_MANIFEST_ARTIFACT_TYPE = "collector_shard_manifest"
COLLECTOR_SHARD_MANIFEST_SCHEMA_VERSION = "1.0"
COLLECTOR_PROVENANCE_SCHEMA_VERSION = "1.0"
ACTIVE_COLLECTOR_SHARD_STATUSES = frozenset({"queued", "pending", "running", "retrying", "throttled"})
FAILED_COLLECTOR_SHARD_STATUSES = frozenset({"failed", "error"})


def collector_shard_manifests_from_result(
    artifacts: Mapping[str, Any],
    *,
    attempt_id: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize collector shard manifests emitted by a live run result."""

    raw_value = artifacts.get("collector_shard_manifests")
    if raw_value is None:
        raw_value = artifacts.get("collector_shards")
    return normalize_collector_shard_manifests(raw_value, attempt_id=attempt_id)


def normalize_collector_shard_manifests(
    value: Any,
    *,
    attempt_id: str | None = None,
    shard_namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize collector shard payloads into a stable additive contract."""

    manifests: list[dict[str, Any]] = []
    namespace = _string(shard_namespace)
    for index, manifest in enumerate(_collect_shard_items(value), start=1):
        payload = deepcopy(manifest)
        payload["schema_version"] = (
            _string(payload.get("schema_version")) or COLLECTOR_SHARD_MANIFEST_SCHEMA_VERSION
        )
        fallback_shard_id = f"shard-{index}"
        if namespace:
            fallback_shard_id = f"{namespace}::shard-{index}"
        payload["shard_id"] = _string(payload.get("shard_id")) or _string(payload.get("id")) or (
            fallback_shard_id
        )
        payload["adapter_name"] = _string(payload.get("adapter_name")) or _string(
            payload.get("adapter")
        )
        payload["status"] = _string(payload.get("status")) or "pending"
        payload["attempt_id"] = _string(payload.get("attempt_id")) or attempt_id
        source = _mapping(payload.get("source")) or {}
        if not source:
            source = _build_source_payload(payload)
        if source:
            payload["source"] = source
        tool = _mapping(payload.get("tool")) or {}
        if not tool:
            tool = _build_tool_payload(payload)
        if tool:
            payload["tool"] = tool
        policy = _mapping(payload.get("policy")) or {}
        if not policy:
            policy = _build_policy_payload(payload)
        if policy:
            payload["policy"] = policy
        manifests.append(payload)
    return manifests


def dedupe_collector_shard_manifests(
    manifests: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the freshest snapshot for each collector shard."""

    latest_by_shard_id: dict[str, dict[str, Any]] = {}
    latest_timestamp_by_shard_id: dict[str, datetime] = {}
    for manifest in manifests:
        shard_id = _string(manifest.get("shard_id")) or _string(manifest.get("artifact_id"))
        if shard_id is None:
            continue
        offset = build_collector_shard_offsets([manifest])
        last_activity_at = _string(offset[0].get("last_activity_at")) if offset else None
        last_activity_dt = _parse_iso_datetime(last_activity_at) or datetime.min.replace(tzinfo=UTC)
        existing_dt = latest_timestamp_by_shard_id.get(shard_id)
        if existing_dt is None or last_activity_dt >= existing_dt:
            latest_by_shard_id[shard_id] = dict(manifest)
            latest_timestamp_by_shard_id[shard_id] = last_activity_dt
    return list(latest_by_shard_id.values())


def collector_record_provenance_for_index(
    artifacts: Mapping[str, Any],
    *,
    index: int,
    shard_manifests: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Resolve collector provenance for one dataset record index."""

    raw_value = artifacts.get("collector_record_provenance")
    if raw_value is None:
        raw_value = artifacts.get("collector_provenance_records")
    candidate = _resolve_record_provenance_candidate(raw_value, index=index)
    normalized = normalize_collector_record_provenance(candidate)
    if normalized is not None:
        return normalized
    if shard_manifests:
        shard = _select_shard_for_record(shard_manifests, index=index)
        if shard is not None:
            return normalize_collector_record_provenance(_collector_provenance_from_shard(shard))
    return None


def normalize_collector_record_provenance(value: Any) -> dict[str, Any] | None:
    """Normalize per-record collector provenance into a stable additive payload."""

    payload = _mapping(value)
    if not payload:
        return None
    normalized = deepcopy(payload)
    normalized["schema_version"] = (
        _string(normalized.get("schema_version")) or COLLECTOR_PROVENANCE_SCHEMA_VERSION
    )
    source = _mapping(normalized.get("source")) or {}
    tool = _mapping(normalized.get("tool")) or {}
    policy = _mapping(normalized.get("policy")) or {}

    source_url = _string(normalized.get("source_url")) or _string(source.get("url"))
    source_id = _string(normalized.get("source_id")) or _string(source.get("id")) or _string(
        source.get("source_id")
    )
    collected_at = _string(normalized.get("collected_at")) or _string(
        normalized.get("captured_at")
    )
    adapter_name = _string(normalized.get("adapter_name")) or _string(
        normalized.get("adapter")
    )
    tool_name = _string(normalized.get("tool_name")) or _string(tool.get("name"))
    tool_arguments = _mapping(normalized.get("tool_arguments")) or _mapping(tool.get("arguments"))
    license_name = (
        _string(normalized.get("license"))
        or _string(normalized.get("license_name"))
        or _string(policy.get("license"))
    )
    policy_disposition = _string(normalized.get("policy_disposition")) or _string(
        policy.get("disposition")
    )
    dedupe_hash = _string(normalized.get("dedupe_hash"))
    shard_id = _string(normalized.get("shard_id"))

    if source_url is not None:
        normalized["source_url"] = source_url
    if source_id is not None:
        normalized["source_id"] = source_id
    if collected_at is not None:
        normalized["collected_at"] = collected_at
    if adapter_name is not None:
        normalized["adapter_name"] = adapter_name
    if tool_name is not None:
        normalized["tool_name"] = tool_name
    if tool_arguments:
        normalized["tool_arguments"] = tool_arguments
    if license_name is not None:
        normalized["license"] = license_name
    if policy_disposition is not None:
        normalized["policy_disposition"] = policy_disposition
    if dedupe_hash is not None:
        normalized["dedupe_hash"] = dedupe_hash
    if shard_id is not None:
        normalized["shard_id"] = shard_id
    return normalized


def collector_dataset_records_from_result(
    artifacts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Normalize collector-emitted dataset records from a run result artifact payload."""

    raw_value = artifacts.get("collector_dataset_records")
    if raw_value is None:
        raw_value = artifacts.get("collector_records")
    return normalize_collector_dataset_records(raw_value)


def normalize_collector_dataset_records(value: Any) -> list[dict[str, Any]]:
    """Normalize collector-emitted dataset records into dataset_record payloads."""

    records: list[dict[str, Any]] = []
    for item in _collect_dataset_record_items(value):
        record = _dataset_record_payload(item)
        if not record:
            continue
        payload = deepcopy(record)
        payload["contract"] = _string(payload.get("contract")) or "dataset_record"
        payload["schema_version"] = _string(payload.get("schema_version")) or "1.0"
        metadata = _mapping(payload.get("metadata")) or {}
        if metadata:
            payload["metadata"] = metadata
        supervision = _mapping(payload.get("supervision")) or {}
        if supervision:
            payload["supervision"] = supervision
        target = _mapping(payload.get("target")) or {}
        if target:
            payload["target"] = target
        records.append(payload)
    return records


def extract_persisted_collector_dataset_records(
    artifacts: list[Mapping[str, Any]],
    *,
    attempt_id: str | None = None,
) -> list[dict[str, Any]]:
    """Extract persisted collector dataset-record artifacts."""

    records: list[dict[str, Any]] = []
    for artifact in artifacts:
        if str(artifact.get("artifact_type") or "") != "dataset_record":
            continue
        metadata = _mapping(artifact.get("metadata")) or {}
        artifact_attempt_id = _string(metadata.get("attempt_id"))
        if attempt_id is not None and artifact_attempt_id != attempt_id:
            continue
        record = _dataset_record_payload(metadata)
        if not record:
            continue
        normalized_records = normalize_collector_dataset_records(record)
        if not normalized_records:
            continue
        normalized = normalized_records[0]
        if not _is_collector_dataset_record(normalized):
            continue
        payload = deepcopy(normalized)
        if artifact_attempt_id is not None:
            payload["attempt_id"] = artifact_attempt_id
        if (artifact_id := _string(artifact.get("artifact_id"))) is not None:
            payload["artifact_id"] = artifact_id
        if (created_at := _string(artifact.get("created_at"))) is not None:
            payload["artifact_created_at"] = created_at
        if (updated_at := _string(artifact.get("updated_at"))) is not None:
            payload["artifact_updated_at"] = updated_at
        records.append(payload)
    return records


def collector_dataset_record_identity(record: Mapping[str, Any]) -> str:
    """Build a stable identity for one collector dataset record."""

    metadata = _mapping(record.get("metadata")) or {}
    collector_provenance = normalize_collector_record_provenance(
        metadata.get("collector_provenance")
    )
    if collector_provenance is not None:
        dedupe_hash = _string(collector_provenance.get("dedupe_hash"))
        if dedupe_hash is not None:
            return f"dedupe:{dedupe_hash}"
        source_id = _string(collector_provenance.get("source_id"))
        if source_id is not None:
            return f"source:{source_id}"
        source_url = _string(collector_provenance.get("source_url"))
        if source_url is not None:
            return f"url:{source_url}"
    payload = _mapping(_dataset_record_payload(record)) or dict(record)
    return "payload:" + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def dedupe_collector_dataset_records(
    records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the latest collector dataset record for each attempt/identity pair."""

    latest_by_key: dict[tuple[str | None, str], dict[str, Any]] = {}
    latest_timestamp_by_key: dict[tuple[str | None, str], datetime] = {}
    ordered_keys: list[tuple[str | None, str]] = []
    floor = datetime.min.replace(tzinfo=UTC)

    for record in records:
        payload = _mapping(record) or {}
        if not payload:
            continue
        key = (_string(payload.get("attempt_id")), collector_dataset_record_identity(payload))
        if key not in latest_by_key:
            ordered_keys.append(key)
        last_activity_dt = (
            _parse_iso_datetime(_collector_dataset_record_last_activity_at(payload)) or floor
        )
        existing_dt = latest_timestamp_by_key.get(key)
        if existing_dt is None or last_activity_dt >= existing_dt:
            latest_by_key[key] = deepcopy(payload)
            latest_timestamp_by_key[key] = last_activity_dt

    return [latest_by_key[key] for key in ordered_keys if key in latest_by_key]


def collector_record_provenance_list_from_dataset_records(
    records: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract normalized collector provenance entries from dataset records."""

    provenance_records: list[dict[str, Any]] = []
    for record in records:
        metadata = _mapping(record.get("metadata")) or {}
        normalized = normalize_collector_record_provenance(metadata.get("collector_provenance"))
        if normalized is not None:
            provenance_records.append(normalized)
    return provenance_records


def extract_collector_shard_manifests(
    artifacts: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extract persisted collector shard manifest artifacts."""

    manifests: list[dict[str, Any]] = []
    for artifact_index, artifact in enumerate(artifacts, start=1):
        if str(artifact.get("artifact_type") or "") != COLLECTOR_SHARD_MANIFEST_ARTIFACT_TYPE:
            continue
        metadata = _mapping(artifact.get("metadata")) or {}
        content = _mapping(metadata.get("content"))
        if not content:
            continue
        normalized_items = normalize_collector_shard_manifests(
            content,
            attempt_id=_string(metadata.get("attempt_id")),
            shard_namespace=_string(artifact.get("artifact_id")) or f"artifact-{artifact_index}",
        )
        for normalized in normalized_items:
            normalized["artifact_id"] = artifact.get("artifact_id")
            normalized["title"] = artifact.get("title")
            normalized["uri"] = artifact.get("uri")
            if isinstance(artifact.get("created_at"), str) and str(artifact.get("created_at")).strip():
                normalized.setdefault("artifact_created_at", str(artifact.get("created_at")).strip())
            if isinstance(artifact.get("updated_at"), str) and str(artifact.get("updated_at")).strip():
                normalized.setdefault("artifact_updated_at", str(artifact.get("updated_at")).strip())
            manifests.append(normalized)
    return manifests


def dedupe_collector_shard_manifests(
    manifests: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the latest shard-progress snapshot for each attempt/shard pair."""

    latest_by_key: dict[tuple[str | None, str], dict[str, Any]] = {}
    latest_timestamp_by_key: dict[tuple[str | None, str], datetime] = {}
    ordered_keys: list[tuple[str | None, str]] = []
    floor = datetime.min.replace(tzinfo=UTC)

    for manifest in manifests:
        payload = _mapping(manifest) or {}
        shard_id = (
            _string(payload.get("shard_id"))
            or _string(payload.get("id"))
            or _string(payload.get("artifact_id"))
        )
        if shard_id is None:
            continue
        attempt_id = _string(payload.get("attempt_id"))
        key = (attempt_id, shard_id)
        if key not in latest_by_key:
            ordered_keys.append(key)
        offsets = build_collector_shard_offsets([payload])
        last_activity_at = _string(offsets[0].get("last_activity_at")) if offsets else None
        last_activity_dt = _parse_iso_datetime(last_activity_at) or floor
        existing_dt = latest_timestamp_by_key.get(key)
        if existing_dt is None or last_activity_dt >= existing_dt:
            latest_by_key[key] = deepcopy(payload)
            latest_timestamp_by_key[key] = last_activity_dt

    return [latest_by_key[key] for key in ordered_keys if key in latest_by_key]


def build_collector_shard_offsets(
    manifests: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build compact shard offset/progress payloads from normalized manifests."""

    offsets: list[dict[str, Any]] = []
    for manifest in manifests:
        payload = _mapping(manifest) or {}
        progress = _mapping(payload.get("progress")) or {}
        source = _mapping(payload.get("source")) or {}
        offset = {
            "shard_id": _string(payload.get("shard_id")) or _string(payload.get("id")),
            "attempt_id": _string(payload.get("attempt_id")),
            "status": _string(payload.get("status")) or "pending",
            "adapter_name": _string(payload.get("adapter_name")) or _string(payload.get("adapter")),
            "source_url": _string(source.get("url")) or _string(payload.get("source_url")),
            "source_id": _string(source.get("id"))
            or _string(source.get("source_id"))
            or _string(payload.get("source_id")),
            "cursor": _string(progress.get("cursor")) or _string(payload.get("cursor")),
            "items_collected": _int(progress.get("items_collected")),
            "items_emitted": _int(progress.get("items_emitted")),
            "last_activity_at": _collector_shard_last_activity_at(payload),
        }
        offsets.append(
            {
                key: value
                for key, value in offset.items()
                if value is not None
            }
        )
    return offsets


def build_collector_state_manifest(
    manifests: list[Mapping[str, Any]],
    *,
    stall_timeout_sec: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Summarize shard state for goal health, checkpoints, and UI."""

    offsets = build_collector_shard_offsets(manifests)
    if not offsets:
        return None
    reference_now = now or datetime.now(UTC)
    status_counts: Counter[str] = Counter()
    active_offsets: list[dict[str, Any]] = []
    stalled_offsets: list[dict[str, Any]] = []
    latest_activity_dt: datetime | None = None

    for offset in offsets:
        status = _string(offset.get("status")) or "pending"
        status_counts[status] += 1
        last_activity_at = _string(offset.get("last_activity_at"))
        last_activity_dt = _parse_iso_datetime(last_activity_at)
        if last_activity_dt is not None and (
            latest_activity_dt is None or last_activity_dt > latest_activity_dt
        ):
            latest_activity_dt = last_activity_dt
        if status in ACTIVE_COLLECTOR_SHARD_STATUSES:
            active_offsets.append(offset)
            if stall_timeout_sec is not None and stall_timeout_sec > 0 and last_activity_dt is not None:
                age_sec = max(0, int((reference_now - last_activity_dt).total_seconds()))
                if age_sec >= stall_timeout_sec:
                    stalled_offsets.append({**offset, "stalled_age_sec": age_sec})

    payload = {
        "shard_count": len(offsets),
        "active_shard_count": len(active_offsets),
        "completed_shard_count": int(status_counts.get("completed", 0)),
        "failed_shard_count": sum(
            count for status, count in status_counts.items() if status in FAILED_COLLECTOR_SHARD_STATUSES
        ),
        "status_counts": _counter_payload(status_counts),
        "latest_activity_at": latest_activity_dt.isoformat() if latest_activity_dt is not None else None,
        "stall_timeout_sec": stall_timeout_sec,
        "stalled_shard_count": len(stalled_offsets),
        "stalled_shards": stalled_offsets,
        "shards": offsets,
    }
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, [], {})
    }


def build_collector_provenance_manifest(
    records: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Build a package-facing collector provenance manifest from dataset records."""

    manifest_records: list[dict[str, Any]] = []
    adapter_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    license_counts: Counter[str] = Counter()

    for index, item in enumerate(records, start=1):
        record = _dataset_record_payload(item)
        if not record:
            continue
        metadata = _mapping(record.get("metadata")) or {}
        provenance = normalize_collector_record_provenance(metadata.get("collector_provenance"))
        if provenance is None:
            continue
        entry = deepcopy(provenance)
        entry["record_index"] = index
        artifact_id = _string(item.get("artifact_id"))
        attempt_id = _string(item.get("attempt_id"))
        if artifact_id is not None:
            entry["artifact_id"] = artifact_id
        if attempt_id is not None:
            entry["attempt_id"] = attempt_id
        manifest_records.append(entry)
        adapter_name = _string(entry.get("adapter_name"))
        if adapter_name is not None:
            adapter_counts[adapter_name] += 1
        policy_disposition = _string(entry.get("policy_disposition"))
        if policy_disposition is not None:
            policy_counts[policy_disposition] += 1
        license_name = _string(entry.get("license"))
        if license_name is not None:
            license_counts[license_name] += 1

    if not manifest_records:
        return None

    return {
        "schema_version": COLLECTOR_PROVENANCE_SCHEMA_VERSION,
        "record_count": len(manifest_records),
        "records": manifest_records,
        "adapter_counts": _counter_payload(adapter_counts),
        "policy_disposition_counts": _counter_payload(policy_counts),
        "license_counts": _counter_payload(license_counts),
    }


def collector_record_provenance_list_from_result(
    artifacts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Normalize collector provenance payloads emitted by a live run result."""

    raw_value = artifacts.get("collector_record_provenance")
    if raw_value is None:
        raw_value = artifacts.get("collector_provenance_records")
    return normalize_collector_record_provenance_list(raw_value)


def normalize_collector_record_provenance_list(value: Any) -> list[dict[str, Any]]:
    """Normalize per-record collector provenance payloads into a stable list."""

    payload = _mapping(value)
    if payload:
        default = _mapping(payload.get("default")) or {}
        records = payload.get("records")
        if isinstance(records, list):
            normalized_records: list[dict[str, Any]] = []
            for item in records:
                candidate = {**default, **dict(item)} if isinstance(item, Mapping) else default
                normalized = normalize_collector_record_provenance(candidate)
                if normalized is not None:
                    normalized_records.append(normalized)
            if normalized_records:
                return normalized_records
            normalized_default = normalize_collector_record_provenance(default)
            return [normalized_default] if normalized_default is not None else []
        normalized_payload = normalize_collector_record_provenance(payload)
        return [normalized_payload] if normalized_payload is not None else []
    if not isinstance(value, list):
        return []
    normalized_records: list[dict[str, Any]] = []
    for item in value:
        normalized = normalize_collector_record_provenance(item)
        if normalized is not None:
            normalized_records.append(normalized)
    return normalized_records


def _collect_shard_items(value: Any) -> list[dict[str, Any]]:
    payload = _mapping(value)
    if payload and isinstance(payload.get("shards"), list):
        return [dict(item) for item in payload["shards"] if isinstance(item, Mapping)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if payload:
        return [payload]
    return []


def _collect_dataset_record_items(value: Any) -> list[dict[str, Any]]:
    payload = _mapping(value)
    if payload and isinstance(payload.get("records"), list):
        return [dict(item) for item in payload["records"] if isinstance(item, Mapping)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if payload:
        return [payload]
    return []


def _is_collector_dataset_record(record: Mapping[str, Any]) -> bool:
    metadata = _mapping(record.get("metadata")) or {}
    if normalize_collector_record_provenance(metadata.get("collector_provenance")) is not None:
        return True
    if _string(metadata.get("capability_family")) == "dataset_collection":
        return True
    return False


def _resolve_record_provenance_candidate(value: Any, *, index: int) -> dict[str, Any] | None:
    payload = _mapping(value)
    if payload:
        default = _mapping(payload.get("default")) or {}
        records = payload.get("records")
        if isinstance(records, list):
            item = _mapping(records[index]) if 0 <= index < len(records) else None
            if item:
                return {**default, **item}
            return default or None
        return payload
    if isinstance(value, list):
        if 0 <= index < len(value):
            return _mapping(value[index])
        return None
    return None


def _select_shard_for_record(
    shard_manifests: list[Mapping[str, Any]],
    *,
    index: int,
) -> Mapping[str, Any] | None:
    if len(shard_manifests) == 1:
        return shard_manifests[0]
    if 0 <= index < len(shard_manifests):
        return shard_manifests[index]
    return None


def _collector_provenance_from_shard(shard: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(shard.get("source")) or {}
    tool = _mapping(shard.get("tool")) or {}
    policy = _mapping(shard.get("policy")) or {}
    payload: dict[str, Any] = {}
    if (source_url := _string(source.get("url")) or _string(shard.get("source_url"))) is not None:
        payload["source_url"] = source_url
    if (source_id := _string(source.get("id")) or _string(source.get("source_id")) or _string(shard.get("source_id"))) is not None:
        payload["source_id"] = source_id
    if (collected_at := _string(shard.get("collected_at")) or _string(shard.get("completed_at")) or _string(shard.get("updated_at"))) is not None:
        payload["collected_at"] = collected_at
    if (adapter_name := _string(shard.get("adapter_name")) or _string(shard.get("adapter"))) is not None:
        payload["adapter_name"] = adapter_name
    if (tool_name := _string(tool.get("name")) or _string(shard.get("tool_name"))) is not None:
        payload["tool_name"] = tool_name
    tool_arguments = _mapping(tool.get("arguments")) or _mapping(shard.get("tool_arguments"))
    if tool_arguments:
        payload["tool_arguments"] = tool_arguments
    if (license_name := _string(policy.get("license")) or _string(shard.get("license"))) is not None:
        payload["license"] = license_name
    if (policy_disposition := _string(policy.get("disposition")) or _string(shard.get("policy_disposition"))) is not None:
        payload["policy_disposition"] = policy_disposition
    if (dedupe_hash := _string(shard.get("dedupe_hash"))) is not None:
        payload["dedupe_hash"] = dedupe_hash
    if (shard_id := _string(shard.get("shard_id")) or _string(shard.get("id"))) is not None:
        payload["shard_id"] = shard_id
    return payload


def _build_source_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    source: dict[str, Any] = {}
    if (source_url := _string(payload.get("source_url"))) is not None:
        source["url"] = source_url
    if (source_id := _string(payload.get("source_id"))) is not None:
        source["id"] = source_id
    if (source_type := _string(payload.get("source_type"))) is not None:
        source["source_type"] = source_type
    return source


def _build_tool_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    tool: dict[str, Any] = {}
    if (tool_name := _string(payload.get("tool_name"))) is not None:
        tool["name"] = tool_name
    tool_arguments = _mapping(payload.get("tool_arguments"))
    if tool_arguments:
        tool["arguments"] = tool_arguments
    return tool


def _build_policy_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    policy: dict[str, Any] = {}
    if (license_name := _string(payload.get("license"))) is not None:
        policy["license"] = license_name
    if (disposition := _string(payload.get("policy_disposition"))) is not None:
        policy["disposition"] = disposition
    return policy


def _dataset_record_payload(item: Mapping[str, Any]) -> Mapping[str, Any] | None:
    record = _mapping(item.get("record"))
    if record:
        return record
    return _mapping(item)


def _counter_payload(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items())
    ]


def _collector_shard_last_activity_at(manifest: Mapping[str, Any]) -> str | None:
    progress = _mapping(manifest.get("progress")) or {}
    primary_candidates = (
        manifest.get("updated_at"),
        progress.get("updated_at"),
        manifest.get("completed_at"),
        manifest.get("collected_at"),
        manifest.get("captured_at"),
        manifest.get("artifact_updated_at"),
    )
    fallback_candidates = (
        manifest.get("created_at"),
        manifest.get("artifact_created_at"),
    )

    def _select_freshest(candidates: tuple[Any, ...]) -> tuple[datetime | None, str | None]:
        best_dt: datetime | None = None
        best_value: str | None = None
        for candidate in candidates:
            value = _string(candidate)
            if value is None:
                continue
            parsed = _parse_iso_datetime(value)
            if parsed is None:
                if best_value is None:
                    best_value = value
                continue
            if best_dt is None or parsed > best_dt:
                best_dt = parsed
                best_value = parsed.isoformat()
        return best_dt, best_value

    best_dt, best_value = _select_freshest(primary_candidates)
    if best_value is not None:
        return best_value
    _, fallback_value = _select_freshest(fallback_candidates)
    return fallback_value


def _collector_dataset_record_last_activity_at(record: Mapping[str, Any]) -> str | None:
    metadata = _mapping(record.get("metadata")) or {}
    provenance = normalize_collector_record_provenance(metadata.get("collector_provenance")) or {}
    primary_candidates = (
        provenance.get("collected_at"),
        record.get("updated_at"),
        record.get("artifact_updated_at"),
    )
    fallback_candidates = (
        record.get("created_at"),
        record.get("artifact_created_at"),
    )

    def _select_freshest(candidates: tuple[Any, ...]) -> tuple[datetime | None, str | None]:
        best_dt: datetime | None = None
        best_value: str | None = None
        for candidate in candidates:
            value = _string(candidate)
            if value is None:
                continue
            parsed = _parse_iso_datetime(value)
            if parsed is None:
                if best_value is None:
                    best_value = value
                continue
            if best_dt is None or parsed > best_dt:
                best_dt = parsed
                best_value = parsed.isoformat()
        return best_dt, best_value

    best_dt, best_value = _select_freshest(primary_candidates)
    if best_value is not None:
        return best_value
    _, fallback_value = _select_freshest(fallback_candidates)
    return fallback_value


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None
