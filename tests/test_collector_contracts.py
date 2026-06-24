from datetime import UTC, datetime

from mochi.runtime.collector_contracts import (
    build_collector_state_manifest,
    collector_dataset_record_identity,
    collector_record_provenance_list_from_dataset_records,
    dedupe_collector_shard_manifests,
    dedupe_collector_dataset_records,
    extract_collector_shard_manifests,
    extract_persisted_collector_dataset_records,
)


def test_extract_collector_shard_manifests_uses_artifact_scoped_fallback_shard_ids() -> None:
    artifacts = [
        {
            "artifact_id": "collector-shard-artifact-a",
            "artifact_type": "collector_shard_manifest",
            "metadata": {
                "attempt_id": "attempt-1",
                "content": {
                    "status": "running",
                    "progress": {
                        "cursor": "post-3",
                    },
                },
            },
        },
        {
            "artifact_id": "collector-shard-artifact-b",
            "artifact_type": "collector_shard_manifest",
            "metadata": {
                "attempt_id": "attempt-1",
                "content": {
                    "status": "running",
                    "progress": {
                        "cursor": "post-7",
                    },
                },
            },
        },
    ]

    manifests = extract_collector_shard_manifests(artifacts)
    shard_ids = [item["shard_id"] for item in manifests]

    assert shard_ids == [
        "collector-shard-artifact-a::shard-1",
        "collector-shard-artifact-b::shard-1",
    ]

    collector_state = build_collector_state_manifest(manifests)

    assert collector_state is not None
    assert collector_state["shard_count"] == 2


def test_build_collector_state_manifest_uses_freshest_timestamp_for_last_activity() -> None:
    manifests = [
        {
            "shard_id": "forum-thread-1",
            "status": "running",
            "updated_at": "2026-06-24T00:00:00+00:00",
            "completed_at": "2026-06-24T00:10:00+00:00",
            "artifact_updated_at": "2026-06-24T00:20:00+00:00",
            "progress": {
                "cursor": "post-24",
            },
        }
    ]

    collector_state = build_collector_state_manifest(
        manifests,
        stall_timeout_sec=300,
        now=datetime(2026, 6, 24, 0, 21, tzinfo=UTC),
    )

    assert collector_state is not None
    assert collector_state["latest_activity_at"] == "2026-06-24T00:20:00+00:00"
    assert collector_state["shards"][0]["last_activity_at"] == "2026-06-24T00:20:00+00:00"
    assert collector_state["stalled_shard_count"] == 0


def test_dedupe_collector_shard_manifests_keeps_latest_snapshot_per_shard() -> None:
    manifests = [
        {
            "shard_id": "forum-thread-1",
            "status": "running",
            "artifact_id": "collector-live-1",
            "artifact_updated_at": "2026-06-24T00:05:00+00:00",
            "progress": {
                "cursor": "post-8",
                "items_collected": 8,
                "items_emitted": 6,
            },
        },
        {
            "shard_id": "forum-thread-1",
            "status": "completed",
            "artifact_id": "collector-live-2",
            "artifact_updated_at": "2026-06-24T00:10:00+00:00",
            "completed_at": "2026-06-24T00:10:00+00:00",
            "progress": {
                "cursor": "post-24",
                "items_collected": 24,
                "items_emitted": 24,
            },
        },
    ]

    deduped = dedupe_collector_shard_manifests(manifests)

    assert len(deduped) == 1
    assert deduped[0]["artifact_id"] == "collector-live-2"
    assert deduped[0]["status"] == "completed"
    assert deduped[0]["progress"]["cursor"] == "post-24"


def test_dedupe_collector_shard_manifests_preserves_distinct_attempts() -> None:
    manifests = [
        {
            "attempt_id": "attempt-1",
            "shard_id": "forum-thread-1",
            "status": "completed",
            "artifact_updated_at": "2026-06-24T00:10:00+00:00",
            "progress": {"cursor": "post-24"},
        },
        {
            "attempt_id": "attempt-2",
            "shard_id": "forum-thread-1",
            "status": "running",
            "artifact_updated_at": "2026-06-24T00:20:00+00:00",
            "progress": {"cursor": "post-3"},
        },
    ]

    deduped = dedupe_collector_shard_manifests(manifests)

    assert len(deduped) == 2
    assert deduped[0]["attempt_id"] == "attempt-1"
    assert deduped[1]["attempt_id"] == "attempt-2"


def test_extract_persisted_collector_dataset_records_filters_and_keeps_provenance() -> None:
    artifacts = [
        {
            "artifact_id": "collector-record-live-1",
            "artifact_type": "dataset_record",
            "updated_at": "2026-06-24T00:05:00+00:00",
            "metadata": {
                "attempt_id": "attempt-1",
                "record": {
                    "input": "Topic: API examples",
                    "target": {"answer": "First collected post"},
                    "metadata": {
                        "capability_family": "dataset_collection",
                        "collector_provenance": {
                            "source_id": "topic:274354:post:1",
                            "source_url": "https://forum.example/t/api-examples/274354/1",
                            "collected_at": "2026-06-24T00:04:00+00:00",
                        },
                    },
                },
            },
        },
        {
            "artifact_id": "non-collector-record",
            "artifact_type": "dataset_record",
            "metadata": {
                "attempt_id": "attempt-1",
                "record": {
                    "input": "Summarize deployment note",
                    "target": {"answer": "Done"},
                },
            },
        },
    ]

    records = extract_persisted_collector_dataset_records(artifacts, attempt_id="attempt-1")

    assert len(records) == 1
    assert records[0]["artifact_id"] == "collector-record-live-1"
    assert records[0]["metadata"]["collector_provenance"]["source_id"] == "topic:274354:post:1"


def test_dedupe_collector_dataset_records_keeps_latest_by_source_identity() -> None:
    records = [
        {
            "attempt_id": "attempt-1",
            "artifact_updated_at": "2026-06-24T00:05:00+00:00",
            "metadata": {
                "collector_provenance": {
                    "source_id": "topic:274354:post:1",
                    "collected_at": "2026-06-24T00:04:00+00:00",
                }
            },
            "target": {"answer": "Older version"},
        },
        {
            "attempt_id": "attempt-1",
            "artifact_updated_at": "2026-06-24T00:07:00+00:00",
            "metadata": {
                "collector_provenance": {
                    "source_id": "topic:274354:post:1",
                    "collected_at": "2026-06-24T00:06:00+00:00",
                }
            },
            "target": {"answer": "Latest version"},
        },
    ]

    deduped = dedupe_collector_dataset_records(records)

    assert len(deduped) == 1
    assert deduped[0]["target"]["answer"] == "Latest version"
    assert collector_dataset_record_identity(deduped[0]) == "source:topic:274354:post:1"


def test_collector_record_provenance_list_from_dataset_records_preserves_order() -> None:
    records = [
        {
            "metadata": {
                "collector_provenance": {
                    "source_id": "topic:274354:post:1",
                    "shard_id": "discourse-topic-274354",
                }
            }
        },
        {
            "metadata": {
                "collector_provenance": {
                    "source_id": "topic:274354:post:2",
                    "shard_id": "discourse-topic-274354",
                }
            }
        },
    ]

    provenance = collector_record_provenance_list_from_dataset_records(records)

    assert [item["source_id"] for item in provenance] == [
        "topic:274354:post:1",
        "topic:274354:post:2",
    ]
