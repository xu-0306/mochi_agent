"""Shared package builders for agent-run exports and snapshots."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any, Mapping

ATTEMPT_BUNDLE_MANIFEST_VERSION = "mochi.agent_run.attempt_bundle.v1"
DATASET_PACKAGE_MANIFEST_VERSION = "mochi.agent_run.dataset_package.v1"


def build_attempt_bundle(
    run: Mapping[str, Any],
    *,
    attempt_id: str | None,
    selected_scope: str,
) -> dict[str, Any]:
    """Build an attempt-scoped replay/evaluator package."""

    artifacts = _scoped_artifacts(run, attempt_id)
    events = _scoped_events(run, attempt_id)
    role_outputs = _extract_role_outputs(events)
    evaluation_events = [
        event for event in events if str(event.get("type") or "") == "evaluation"
    ]
    dataset_records = [
        dict(artifact.get("metadata", {}).get("record") or {})
        for artifact in artifacts
        if str(artifact.get("artifact_type") or "") == "dataset_record"
        and isinstance(artifact.get("metadata", {}).get("record"), dict)
    ]
    selected_candidate_id = _selected_candidate_id(run, dataset_records=dataset_records)
    final_answer = _final_answer(run, dataset_records=dataset_records)

    payload = {
        "manifest_version": ATTEMPT_BUNDLE_MANIFEST_VERSION,
        "package_type": "attempt_bundle",
        "exported_at": _now_iso(),
        "run_id": str(run.get("run_id") or run.get("id") or ""),
        "protocol_id": str(run.get("protocol_id") or ""),
        "attempt_id": attempt_id,
        "selected_scope": selected_scope,
        "schedule_attempt": _schedule_attempt(run, attempt_id),
        "artifact_count": len(artifacts),
        "event_count": len(events),
        "role_output_count": len(role_outputs),
        "artifacts": [_serialize_artifact(artifact) for artifact in artifacts],
        "events": [dict(event) for event in events],
        "role_outputs": role_outputs,
        "evaluation_events": [dict(event) for event in evaluation_events],
        "dataset_records": dataset_records,
        "run_summary": _run_summary_payload(run, artifacts, attempt_id),
        "evidence_summary": _artifact_content(artifacts, "evidence_summary"),
        "verification_summary": _artifact_content(artifacts, "verification_summary"),
        "final_selected_candidate": {
            "candidate_id": selected_candidate_id,
            "final_answer": final_answer,
        }
        if selected_candidate_id is not None or final_answer is not None
        else None,
    }
    payload["replay_ready"] = bool(events) and payload["final_selected_candidate"] is not None
    return payload


def build_dataset_package(run: Mapping[str, Any]) -> dict[str, Any]:
    """Build a training-facing dataset package for an entire run."""

    run_id = str(run.get("run_id") or run.get("id") or "")
    protocol_id = str(run.get("protocol_id") or "")
    recent_attempts = _record_array(_schedule_payload(run).get("recent_attempts"))
    attempt_lookup = {
        attempt_id: attempt
        for attempt in recent_attempts
        if (attempt_id := _string(attempt.get("attempt_id"))) is not None
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    grouped_attempt_records: dict[str, list[dict[str, Any]]] = {}

    for artifact in _artifacts(run):
        if str(artifact.get("artifact_type") or "") != "dataset_record":
            continue
        record = artifact.get("metadata", {}).get("record")
        if not isinstance(record, dict):
            continue
        attempt_id = _artifact_attempt_id(artifact) or "unscoped"
        governed_record = _govern_dataset_record(
            run,
            attempt_id=None if attempt_id == "unscoped" else attempt_id,
            artifact=artifact,
            record=record,
        )
        grouped.setdefault(attempt_id, []).append(governed_record)
        grouped_attempt_records.setdefault(attempt_id, []).append(governed_record)

    attempts: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    training_ready_records: list[dict[str, Any]] = []
    excluded_reason_counts: Counter[str] = Counter()

    ordered_attempt_ids = [
        attempt_id
        for attempt_id in attempt_lookup
        if attempt_id in grouped_attempt_records
    ]
    for attempt_id in grouped_attempt_records:
        if attempt_id != "unscoped" and attempt_id not in ordered_attempt_ids:
            ordered_attempt_ids.append(attempt_id)
    if "unscoped" in grouped_attempt_records:
        ordered_attempt_ids.append("unscoped")

    for grouped_attempt_id in ordered_attempt_ids:
        records = grouped_attempt_records.get(grouped_attempt_id, [])
        if not records:
            continue
        training_ready = [record for record in records if bool(record.get("training_ready"))]
        excluded = [record for record in records if not bool(record.get("training_ready"))]
        for record in excluded:
            for reason in _string_array(record.get("exclusion_reasons")):
                excluded_reason_counts[reason] += 1
        attempts.append(
            {
                "attempt_id": None if grouped_attempt_id == "unscoped" else grouped_attempt_id,
                "schedule_attempt": None
                if grouped_attempt_id == "unscoped"
                else attempt_lookup.get(grouped_attempt_id),
                "dataset_record_count": len(records),
                "training_ready_count": len(training_ready),
                "excluded_record_count": len(excluded),
                "dataset_records": records,
            }
        )
        all_records.extend(records)
        training_ready_records.extend(training_ready)

    excluded_records_summary = {
        "excluded_count": len([record for record in all_records if not bool(record.get("training_ready"))]),
        "reasons": [
            {"reason": reason, "count": count}
            for reason, count in sorted(excluded_reason_counts.items())
        ],
    }

    return {
        "manifest_version": DATASET_PACKAGE_MANIFEST_VERSION,
        "package_type": "dataset_package",
        "exported_at": _now_iso(),
        "run_id": run_id,
        "protocol_id": protocol_id,
        "attempt_count": len(attempts),
        "dataset_record_count": len(all_records),
        "training_ready_count": len(training_ready_records),
        "excluded_record_count": excluded_records_summary["excluded_count"],
        "attempts": attempts,
        "all_records": all_records,
        "training_ready_records": training_ready_records,
        "excluded_records_summary": excluded_records_summary,
    }


def attempt_id_exists(run: Mapping[str, Any], attempt_id: str) -> bool:
    """Check whether an attempt id is present in the run payload."""

    if _schedule_attempt(run, attempt_id) is not None:
        return True
    if any(_artifact_attempt_id(artifact) == attempt_id for artifact in _artifacts(run)):
        return True
    return any(_event_attempt_id(event) == attempt_id for event in _events(run))


def summarize_package_materialization(
    attempt_bundle: Mapping[str, Any],
    dataset_package: Mapping[str, Any],
    *,
    attempt_id: str | None,
    package_error: str | None = None,
) -> dict[str, Any]:
    """Summarize package health for schedule metadata."""

    package_attempt_id = attempt_id or "unscoped"
    attempt_entry = next(
        (
            entry
            for entry in _record_array(dataset_package.get("attempts"))
            if _string(entry.get("attempt_id")) == attempt_id
            or (attempt_id is None and entry.get("attempt_id") is None)
        ),
        None,
    )
    package_ready = package_error is None
    return {
        "attempt_id": package_attempt_id,
        "artifact_count": int(attempt_bundle.get("artifact_count") or 0) + 2,
        "dataset_record_count": int(
            (attempt_entry or {}).get("dataset_record_count") or len(_record_array(attempt_bundle.get("dataset_records")))
        ),
        "training_ready_count": int((attempt_entry or {}).get("training_ready_count") or 0),
        "package_ready": package_ready,
        "last_package_error": package_error,
        "last_package_completed_at": _now_iso() if package_ready else None,
    }


def _govern_dataset_record(
    run: Mapping[str, Any],
    *,
    attempt_id: str | None,
    artifact: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    selected_models = _selected_models_by_role(run)
    selected_candidate_id = _selected_candidate_id(run, dataset_records=[record])
    final_answer = _final_answer(run, dataset_records=[record])
    verification_summary = _artifact_content(
        _scoped_artifacts(run, attempt_id),
        "verification_summary",
    )
    verification_status = _verification_status(verification_summary, selected_candidate_id)
    evidence_gate_status = _evidence_gate_status(record, selected_candidate_id)

    exclusion_reasons: list[str] = []
    if not final_answer:
        exclusion_reasons.append("missing_final_answer")
    if verification_status == "failed":
        exclusion_reasons.append("verification_failed")
    if evidence_gate_status == "failed":
        exclusion_reasons.append("evidence_gate_failed")

    return {
        "artifact_id": artifact.get("artifact_id"),
        "title": artifact.get("title"),
        "uri": artifact.get("uri"),
        "run_id": str(run.get("run_id") or run.get("id") or ""),
        "attempt_id": attempt_id,
        "protocol_id": str(run.get("protocol_id") or ""),
        "selected_candidate_id": selected_candidate_id,
        "verification_status": verification_status,
        "evidence_gate_status": evidence_gate_status,
        "teacher_model_id": selected_models.get("teacher"),
        "student_model_id": selected_models.get("student"),
        "judge_model_id": selected_models.get("judge"),
        "verifier_model_id": selected_models.get("verifier"),
        "training_ready": len(exclusion_reasons) == 0,
        "exclusion_reasons": exclusion_reasons,
        "record": dict(record),
    }


def _selected_models_by_role(run: Mapping[str, Any]) -> dict[str, str | None]:
    payload = run.get("selected_models_roles")
    if not isinstance(payload, dict):
        return {}
    by_role = payload.get("by_role")
    if not isinstance(by_role, dict):
        return {}
    return {
        role: _string(model_id)
        for role, model_id in by_role.items()
        if isinstance(role, str)
    }


def _verification_status(
    verification_summary: Mapping[str, Any] | None,
    selected_candidate_id: str | None,
) -> str | None:
    if selected_candidate_id is None or not isinstance(verification_summary, Mapping):
        return None
    for item in _record_array(verification_summary.get("verifications")):
        if _string(item.get("candidate_id")) == selected_candidate_id:
            return _string(item.get("status"))
    if selected_candidate_id in _string_array(verification_summary.get("failed_candidate_ids")):
        return "failed"
    if selected_candidate_id in _string_array(verification_summary.get("verified_candidate_ids")):
        return "verified"
    if selected_candidate_id in _string_array(verification_summary.get("skipped_candidate_ids")):
        return "skipped"
    return None


def _evidence_gate_status(
    record: Mapping[str, Any],
    selected_candidate_id: str | None,
) -> str | None:
    evidence = record.get("evidence")
    if not isinstance(evidence, dict):
        return None
    evaluation = evidence.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    for score in _record_array(evaluation.get("scores")):
        if selected_candidate_id is not None and _string(score.get("candidate_id")) != selected_candidate_id:
            continue
        gate = score.get("evidence_gate")
        if isinstance(gate, dict):
            return _string(gate.get("status"))
    return None


def _selected_candidate_id(
    run: Mapping[str, Any],
    *,
    dataset_records: list[Mapping[str, Any]],
) -> str | None:
    for record in dataset_records:
        target = record.get("target")
        if isinstance(target, dict):
            selected = _string(target.get("candidate_id"))
            if selected is not None:
                return selected
    summary = run.get("summary")
    if isinstance(summary, dict):
        selected = _string(summary.get("selected_candidate_id"))
        if selected is not None:
            return selected
    return None


def _final_answer(
    run: Mapping[str, Any],
    *,
    dataset_records: list[Mapping[str, Any]],
) -> str | None:
    for record in dataset_records:
        target = record.get("target")
        if isinstance(target, dict):
            answer = _string(target.get("answer"))
            if answer is not None:
                return answer
    summary = run.get("summary")
    if isinstance(summary, dict):
        answer = _string(summary.get("final_answer"))
        if answer is not None:
            return answer
    return None


def _run_summary_payload(
    run: Mapping[str, Any],
    artifacts: list[dict[str, Any]],
    attempt_id: str | None,
) -> dict[str, Any]:
    summary = run.get("summary")
    payload = dict(summary) if isinstance(summary, dict) else {}
    artifact_payload = _artifact_content(artifacts, "run_summary")
    if isinstance(artifact_payload, dict):
        payload["artifact_content"] = artifact_payload
    payload["attempt_id"] = attempt_id
    return payload


def _extract_role_outputs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for event in events:
        if str(event.get("type") or "") != "role_output":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        outputs.append(
            {
                "role_id": _string(payload.get("role_id")) or "unknown",
                "candidate_id": _string(payload.get("candidate_id")),
                "model_id": _string(payload.get("model_id")),
                "round_index": payload.get("round_index")
                if isinstance(payload.get("round_index"), int)
                else None,
                "content": _string(payload.get("content")) or "",
                "timestamp": _string(event.get("timestamp")),
            }
        )
    return outputs


def _serialize_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_id": artifact.get("artifact_id"),
        "artifact_type": artifact.get("artifact_type"),
        "title": artifact.get("title"),
        "uri": artifact.get("uri"),
        "mime_type": artifact.get("mime_type"),
        "size_bytes": artifact.get("size_bytes"),
        "metadata": dict(artifact.get("metadata") or {}),
    }


def _artifact_content(
    artifacts: list[dict[str, Any]],
    artifact_type: str,
) -> dict[str, Any] | None:
    for artifact in artifacts:
        if str(artifact.get("artifact_type") or "") != artifact_type:
            continue
        metadata = artifact.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("content"), dict):
            return dict(metadata["content"])
    return None


def _scoped_artifacts(run: Mapping[str, Any], attempt_id: str | None) -> list[dict[str, Any]]:
    artifacts = _artifacts(run)
    if attempt_id is None:
        return [artifact for artifact in artifacts if _artifact_attempt_id(artifact) is None]
    return [artifact for artifact in artifacts if _artifact_attempt_id(artifact) == attempt_id]


def _scoped_events(run: Mapping[str, Any], attempt_id: str | None) -> list[dict[str, Any]]:
    events = _events(run)
    if attempt_id is None:
        return [event for event in events if _event_attempt_id(event) is None]
    return [event for event in events if _event_attempt_id(event) == attempt_id]


def _schedule_attempt(run: Mapping[str, Any], attempt_id: str | None) -> dict[str, Any] | None:
    if attempt_id is None:
        return None
    for attempt in _record_array(_schedule_payload(run).get("recent_attempts")):
        if _string(attempt.get("attempt_id")) == attempt_id:
            return attempt
    return None


def _events(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(event) for event in _record_array(run.get("events"))]


def _artifacts(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(artifact) for artifact in _record_array(run.get("artifacts"))]


def _schedule_payload(run: Mapping[str, Any]) -> dict[str, Any]:
    schedule = run.get("schedule")
    return dict(schedule) if isinstance(schedule, dict) else {}


def _artifact_attempt_id(artifact: Mapping[str, Any]) -> str | None:
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return _string(metadata.get("attempt_id"))


def _event_attempt_id(event: Mapping[str, Any]) -> str | None:
    return _string(event.get("attempt_id"))


def _record_array(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_array(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _string(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
