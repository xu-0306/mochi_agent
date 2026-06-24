from __future__ import annotations

from collections.abc import AsyncIterator
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.backends.types import GenerationResult, Message
from mochi.config.schema import MochiConfig
from mochi.runtime.models import (
    AgentRunAttemptPackageResponse,
    AgentRunDatasetPackageResponse,
)
from mochi.runtime.agent_run_packages import (
    ATTEMPT_BUNDLE_MANIFEST_VERSION,
    DATASET_PACKAGE_MANIFEST_VERSION,
    build_attempt_bundle,
    build_dataset_package,
)
from mochi.runtime.service import RuntimeService


def test_build_attempt_bundle_filters_to_single_attempt() -> None:
    run = {
        "run_id": "run-1",
        "protocol_id": "teacher_student_distill",
        "summary": {
            "selected_candidate_id": "student",
            "final_answer": "Student final answer",
        },
        "schedule": {
            "recent_attempts": [
                {"attempt_id": "attempt-2", "attempt_number": 2},
                {"attempt_id": "attempt-1", "attempt_number": 1},
            ]
        },
        "artifacts": [
            {
                "artifact_id": "a-1",
                "artifact_type": "dataset_record",
                "metadata": {
                    "attempt_id": "attempt-1",
                    "record": {"target": {"candidate_id": "teacher", "answer": "old answer"}},
                },
            },
            {
                "artifact_id": "a-2",
                "artifact_type": "dataset_record",
                "metadata": {
                    "attempt_id": "attempt-2",
                    "record": {"target": {"candidate_id": "student", "answer": "new answer"}},
                },
            },
            {
                "artifact_id": "verify-2",
                "artifact_type": "verification_summary",
                "metadata": {
                    "attempt_id": "attempt-2",
                    "content": {"verified_candidate_ids": ["student"], "verifications": []},
                },
            },
        ],
        "events": [
            {"type": "role_output", "attempt_id": "attempt-1", "timestamp": "2026-01-01T00:00:00+00:00", "payload": {"role_id": "teacher", "candidate_id": "teacher", "round_index": 0, "content": "old"}},
            {"type": "role_output", "attempt_id": "attempt-2", "timestamp": "2026-01-02T00:00:00+00:00", "payload": {"role_id": "student", "candidate_id": "student", "round_index": 0, "content": "new"}},
            {"type": "evaluation", "attempt_id": "attempt-2", "timestamp": "2026-01-02T00:00:01+00:00", "payload": {"selected_candidate_id": "student"}},
        ],
    }

    payload = build_attempt_bundle(run, attempt_id="attempt-2", selected_scope="attempt-2")

    assert payload["manifest_version"] == ATTEMPT_BUNDLE_MANIFEST_VERSION
    assert payload["attempt_id"] == "attempt-2"
    assert payload["artifact_count"] == 2
    assert payload["event_count"] == 2
    assert payload["role_output_count"] == 1
    assert payload["replay_ready"] is True
    assert payload["final_selected_candidate"]["candidate_id"] == "student"
    assert payload["dataset_records"] == [
        {"target": {"candidate_id": "student", "answer": "new answer"}}
    ]


def test_build_dataset_package_marks_training_ready_and_exclusions() -> None:
    run = {
        "run_id": "run-2",
        "protocol_id": "teacher_student_distill",
        "selected_models_roles": {
            "by_role": {
                "teacher": "teacher-model",
                "student": "student-model",
                "judge": "judge-model",
                "verifier": "verifier-model",
            }
        },
        "artifacts": [
            {
                "artifact_id": "record-1",
                "artifact_type": "dataset_record",
                "title": "Record 1",
                "uri": "agent-run://run-2/record-1",
                "metadata": {
                    "attempt_id": "attempt-1",
                    "record": {
                        "target": {"candidate_id": "student", "answer": "safe answer"},
                        "evidence": {
                            "evaluation": {
                                "scores": [
                                    {"candidate_id": "student", "evidence_gate": {"status": "verified"}}
                                ]
                            }
                        },
                    },
                },
            },
            {
                "artifact_id": "record-2",
                "artifact_type": "dataset_record",
                "title": "Record 2",
                "uri": "agent-run://run-2/record-2",
                "metadata": {
                    "attempt_id": "attempt-2",
                    "record": {
                        "target": {"candidate_id": "teacher", "answer": "unsupported answer"},
                        "evidence": {
                            "evaluation": {
                                "scores": [
                                    {"candidate_id": "teacher", "evidence_gate": {"status": "failed"}}
                                ]
                            }
                        },
                    },
                },
            },
            {
                "artifact_id": "verify-1",
                "artifact_type": "verification_summary",
                "metadata": {
                    "attempt_id": "attempt-1",
                    "content": {
                        "verified_candidate_ids": ["student"],
                        "failed_candidate_ids": [],
                        "verifications": [{"candidate_id": "student", "status": "verified"}],
                    },
                },
            },
            {
                "artifact_id": "verify-2",
                "artifact_type": "verification_summary",
                "metadata": {
                    "attempt_id": "attempt-2",
                    "content": {
                        "verified_candidate_ids": [],
                        "failed_candidate_ids": ["teacher"],
                        "verifications": [{"candidate_id": "teacher", "status": "failed"}],
                    },
                },
            },
        ],
        "schedule": {
            "recent_attempts": [
                {"attempt_id": "attempt-2", "attempt_number": 2},
                {"attempt_id": "attempt-1", "attempt_number": 1},
            ]
        },
    }

    payload = build_dataset_package(run)

    assert payload["manifest_version"] == DATASET_PACKAGE_MANIFEST_VERSION
    assert payload["dataset_record_count"] == 2
    assert payload["training_ready_count"] == 1
    assert len(payload["all_records"]) == 2
    assert len(payload["training_ready_records"]) == 1
    excluded = [record for record in payload["all_records"] if not record["training_ready"]]
    assert excluded[0]["verification_status"] == "failed"
    assert excluded[0]["evidence_gate_status"] == "failed"
    assert "verification_failed" in excluded[0]["exclusion_reasons"]
    assert "evidence_gate_failed" in excluded[0]["exclusion_reasons"]
    assert payload["excluded_records_summary"]["excluded_count"] == 1


def test_build_packages_include_collector_shards_and_provenance_manifests() -> None:
    run = {
        "run_id": "run-collector-package",
        "protocol_id": "teacher_student_distill",
        "summary": {
            "selected_candidate_id": "student",
            "final_answer": "Curated training record",
        },
        "artifacts": [
            {
                "artifact_id": "record-1",
                "artifact_type": "dataset_record",
                "title": "Record 1",
                "uri": "agent-run://run-collector-package/record-1",
                "metadata": {
                    "attempt_id": "attempt-1",
                    "record": {
                        "target": {"candidate_id": "student", "answer": "Curated training record"},
                        "metadata": {
                            "collector_provenance": {
                                "source_url": "https://forum.example/thread-1",
                                "source_id": "thread-1",
                                "collected_at": "2026-06-24T00:00:00+00:00",
                                "adapter_name": "forum_thread_adapter",
                                "tool_name": "web_fetch",
                                "tool_arguments": {"url": "https://forum.example/thread-1"},
                                "license": "cc-by-sa-4.0",
                                "policy_disposition": "allow",
                                "dedupe_hash": "sha256:abc123",
                                "shard_id": "forum-thread-1",
                            }
                        },
                    },
                },
            },
            {
                "artifact_id": "collector-shard-1",
                "artifact_type": "collector_shard_manifest",
                "title": "Collector Shard forum-thread-1",
                "uri": "agent-run://run-collector-package/collector-shard-1",
                "metadata": {
                    "attempt_id": "attempt-1",
                    "content": {
                        "shards": [
                            {
                                "shard_id": "forum-thread-1",
                                "adapter_name": "forum_thread_adapter",
                                "status": "completed",
                                "source": {
                                    "url": "https://forum.example/thread-1",
                                    "id": "thread-1",
                                },
                                "progress": {
                                    "items_collected": 24,
                                    "items_emitted": 24,
                                    "cursor": "post-24",
                                },
                            },
                            {
                                "shard_id": "forum-thread-2",
                                "adapter_name": "forum_thread_adapter",
                                "status": "running",
                                "source": {
                                    "url": "https://forum.example/thread-2",
                                    "id": "thread-2",
                                },
                                "progress": {
                                    "items_collected": 8,
                                    "items_emitted": 6,
                                    "cursor": "post-8",
                                },
                            },
                        ],
                    },
                },
            },
            {
                "artifact_id": "verify-1",
                "artifact_type": "verification_summary",
                "metadata": {
                    "attempt_id": "attempt-1",
                    "content": {
                        "verified_candidate_ids": ["student"],
                        "failed_candidate_ids": [],
                        "verifications": [{"candidate_id": "student", "status": "verified"}],
                    },
                },
            },
        ],
        "events": [
            {
                "type": "role_output",
                "attempt_id": "attempt-1",
                "timestamp": "2026-06-24T00:00:00+00:00",
                "payload": {
                    "role_id": "student",
                    "candidate_id": "student",
                    "round_index": 0,
                    "content": "Curated training record",
                },
            }
        ],
        "schedule": {
            "recent_attempts": [
                {"attempt_id": "attempt-1", "attempt_number": 1},
            ]
        },
    }

    attempt_payload = build_attempt_bundle(run, attempt_id="attempt-1", selected_scope="attempt-1")
    dataset_payload = build_dataset_package(run)

    assert len(attempt_payload["collector_shard_manifests"]) == 2
    assert attempt_payload["collector_shard_manifests"][1]["shard_id"] == "forum-thread-2"
    assert attempt_payload["collector_provenance_manifest"]["record_count"] == 1
    assert attempt_payload["collector_provenance_manifest"]["records"][0]["source_url"] == (
        "https://forum.example/thread-1"
    )

    assert len(dataset_payload["collector_shard_manifests"]) == 2
    assert dataset_payload["collector_shard_manifests"][1]["status"] == "running"
    assert dataset_payload["collector_provenance_manifest"]["adapter_counts"] == [
        {"value": "forum_thread_adapter", "count": 1}
    ]
    attempt_entry = dataset_payload["attempts"][0]
    assert len(attempt_entry["collector_shard_manifests"]) == 2
    assert attempt_entry["collector_shard_manifests"][0]["progress"]["items_emitted"] == 24
    assert attempt_entry["collector_provenance_manifest"]["records"][0]["shard_id"] == (
        "forum-thread-1"
    )

    attempt_model = AgentRunAttemptPackageResponse.model_validate(attempt_payload).model_dump()
    dataset_model = AgentRunDatasetPackageResponse.model_validate(dataset_payload).model_dump()
    assert attempt_model["collector_shard_manifests"][0]["artifact_id"] == "collector-shard-1"
    assert dataset_model["collector_provenance_manifest"]["record_count"] == 1


def test_build_packages_dedupe_live_collector_snapshots_to_latest_state() -> None:
    run = {
        "run_id": "run-collector-live-dedupe",
        "protocol_id": "teacher_student_distill",
        "summary": {},
        "artifacts": [
            {
                "artifact_id": "collector-live-snapshot-1",
                "artifact_type": "collector_shard_manifest",
                "title": "Collector live snapshot",
                "uri": "agent-run://run-collector-live-dedupe/collector-live-snapshot-1",
                "metadata": {
                    "attempt_id": "attempt-1",
                    "content": {
                        "shards": [
                            {
                                "shard_id": "forum-thread-1",
                                "adapter_name": "forum_thread_adapter",
                                "status": "running",
                                "source": {
                                    "url": "https://forum.example/thread-1",
                                    "id": "thread-1",
                                },
                                "progress": {
                                    "items_collected": 8,
                                    "items_emitted": 6,
                                    "cursor": "post-8",
                                },
                            }
                        ],
                    },
                },
            },
            {
                "artifact_id": "collector-final-shard",
                "artifact_type": "collector_shard_manifest",
                "title": "Collector final shard",
                "uri": "agent-run://run-collector-live-dedupe/collector-final-shard",
                "metadata": {
                    "attempt_id": "attempt-1",
                    "content": {
                        "shards": [
                            {
                                "shard_id": "forum-thread-1",
                                "adapter_name": "forum_thread_adapter",
                                "status": "completed",
                                "source": {
                                    "url": "https://forum.example/thread-1",
                                    "id": "thread-1",
                                },
                                "progress": {
                                    "items_collected": 24,
                                    "items_emitted": 24,
                                    "cursor": "post-24",
                                },
                            }
                        ],
                    },
                },
            },
        ],
        "events": [],
        "schedule": {
            "recent_attempts": [
                {"attempt_id": "attempt-1", "attempt_number": 1},
            ]
        },
    }

    attempt_payload = build_attempt_bundle(run, attempt_id="attempt-1", selected_scope="attempt-1")
    dataset_payload = build_dataset_package(run)

    assert len(attempt_payload["collector_shard_manifests"]) == 1
    assert attempt_payload["collector_shard_manifests"][0]["status"] == "completed"
    assert attempt_payload["collector_shard_manifests"][0]["progress"]["cursor"] == "post-24"

    assert len(dataset_payload["collector_shard_manifests"]) == 1
    assert dataset_payload["collector_shard_manifests"][0]["status"] == "completed"


class _PackageModelBackedEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate_with_configured_model(
        self,
        *,
        model_id: str,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ) -> GenerationResult:
        self.calls.append(
            {
                "model_id": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
            }
        )
        if model_id == "teacher-model":
            return GenerationResult(content="Teacher evidence-backed draft.", model=model_id)
        if model_id == "student-model":
            return GenerationResult(content="Student concise final answer.", model=model_id)
        if model_id == "verifier-model":
            return GenerationResult(
                content=(
                    '{"candidate_verifications":['
                    '{"candidate_id":"teacher","status":"failed","rationale":"unsupported","citations":[],"issues":["unsupported"]},'
                    '{"candidate_id":"student","status":"verified","rationale":"supported","citations":[],"issues":[]}'
                    ']}'
                ),
                model=model_id,
            )
        if model_id == "judge-model":
            return GenerationResult(
                content='{"selected_candidate_id":"teacher","scores":[{"candidate_id":"teacher","score":0.96,"rationale":"looks comprehensive","evidence_gate":{"status":"skipped"}},{"candidate_id":"student","score":0.82,"rationale":"clear but shorter","evidence_gate":{"status":"skipped"}}]}',
                model=model_id,
            )
        raise AssertionError(f"Unexpected model_id: {model_id}")

    async def collect_agent_run_evidence(
        self,
        *,
        queries: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        _ = metadata
        return (
            [
                {
                    "evidence_id": "src-1",
                    "title": "Deployment note",
                    "content": "Student matches source.",
                    "url": "https://example.com/deployment-note",
                    "source_type": "web_fetch",
                    "query": queries[0] if queries else "",
                }
            ],
            {
                "query_count": len(queries),
                "collected_packet_count": 1,
                "provider_counts": {"stub-search": 1},
                "queries": [{"query": queries[0] if queries else "", "packet_count": 1}],
            },
        )


def _wait_agent_run_until(
    client: TestClient,
    run_id: str,
    statuses: set[str],
    *,
    timeout_seconds: float = 4.0,
) -> dict[str, Any]:
    steps = max(1, int(timeout_seconds / 0.05))
    payload: dict[str, Any] = {}
    for _ in range(steps):
        response = client.get(f"/v1/agent-runs/{run_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in statuses:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Agent run did not reach statuses {statuses}: {payload}")


def test_package_endpoints_and_materialization_for_scheduled_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RuntimeService, "_DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS", 0.05)
    app = create_app()
    app.state.engine_factory = lambda: _PackageModelBackedEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Scheduled package run",
                "topic": "summarize deployment note",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                },
                "schedule": {
                    "enabled": True,
                    "interval_seconds": 60,
                    "start_immediately": True,
                    "max_runs": 1,
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        payload = _wait_agent_run_until(client, run_id, {"succeeded"})
        latest_attempt = payload["schedule"]["recent_attempts"][0]
        attempt_id = latest_attempt["attempt_id"]
        artifact_types = [artifact["artifact_type"] for artifact in payload["artifacts"]]
        assert "attempt_bundle" in artifact_types
        assert "dataset_package_snapshot" in artifact_types
        assert latest_attempt["package_ready"] is True
        assert latest_attempt["artifact_count"] >= 6
        assert latest_attempt["dataset_record_count"] == 1
        assert latest_attempt["training_ready_count"] == 1
        assert payload["schedule"]["last_package_completed_at"] is not None
        assert payload["schedule"]["last_package_error"] is None

        attempt_package_response = client.get(
            f"/v1/agent-runs/{run_id}/packages/attempts/{attempt_id}"
        )
        assert attempt_package_response.status_code == 200
        attempt_package = attempt_package_response.json()
        assert attempt_package["manifest_version"] == ATTEMPT_BUNDLE_MANIFEST_VERSION
        assert attempt_package["attempt_id"] == attempt_id
        assert attempt_package["package_type"] == "attempt_bundle"
        assert attempt_package["replay_ready"] is True

        dataset_package_response = client.get(f"/v1/agent-runs/{run_id}/packages/dataset")
        assert dataset_package_response.status_code == 200
        dataset_package = dataset_package_response.json()
        assert dataset_package["manifest_version"] == DATASET_PACKAGE_MANIFEST_VERSION
        assert dataset_package["dataset_record_count"] == 1
        assert dataset_package["training_ready_count"] == 1
        assert len(dataset_package["training_ready_records"]) == 1

        missing_attempt_response = client.get(
            f"/v1/agent-runs/{run_id}/packages/attempts/missing-attempt"
        )
        assert missing_attempt_response.status_code == 404
