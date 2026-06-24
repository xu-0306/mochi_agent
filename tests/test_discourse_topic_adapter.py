from __future__ import annotations

import httpx
import pytest

from mochi.tools.collector_adapter import CollectorRequestPolicy
from mochi.tools.discourse_topic_adapter import (
    DiscourseTopicCollectTool,
    DiscourseTopicCollectorAdapter,
)


def _topic_post(
    *,
    post_id: int,
    post_number: int,
    body: str,
    hidden: bool = False,
    user_deleted: bool = False,
    deleted_at: str | None = None,
) -> dict[str, object]:
    return {
        "id": post_id,
        "post_number": post_number,
        "post_url": f"/t/api-examples/274354/{post_number}",
        "cooked": f"<p>{body}</p>",
        "username": "api-bot",
        "display_username": "API Bot",
        "created_at": f"2026-06-24T00:0{post_number}:00+00:00",
        "updated_at": f"2026-06-24T00:1{post_number}:00+00:00",
        "hidden": hidden,
        "user_deleted": user_deleted,
        "deleted_at": deleted_at,
    }


@pytest.mark.asyncio
async def test_discourse_topic_collector_collects_bounded_batch_and_fetches_missing_posts() -> None:
    requests: list[tuple[str, list[str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        post_ids = request.url.params.get_list("post_ids[]")
        requests.append((request.url.path, post_ids))
        if request.url.path == "/t/274354.json":
            return httpx.Response(
                200,
                json={
                    "id": 274354,
                    "slug": "api-examples",
                    "title": "API examples",
                    "category_id": 7,
                    "tags": ["api", "docs"],
                    "posts_count": 3,
                    "highest_post_number": 3,
                    "post_stream": {
                        "stream": [100, 101, 102],
                        "posts": [
                            _topic_post(post_id=100, post_number=1, body="First collected post"),
                        ],
                    },
                },
                request=request,
            )
        if request.url.path == "/t/274354/posts.json":
            assert post_ids == ["101"]
            return httpx.Response(
                200,
                json={
                    "post_stream": {
                        "posts": [
                            _topic_post(post_id=101, post_number=2, body="Second collected post"),
                        ]
                    }
                },
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = DiscourseTopicCollectorAdapter(
        client=client,
        request_policy=CollectorRequestPolicy(
            max_retries=0,
            backoff_base=0.0,
            backoff_max=0.0,
        ),
    )

    try:
        result = await adapter.collect_shard(
            {
                "source": {"url": "https://forum.example/t/api-examples/274354"},
                "max_posts": 2,
                "policy": {"license": "cc-by-sa-4.0", "disposition": "allow"},
            }
        )
    finally:
        await client.aclose()

    records = result["collector_dataset_records"]
    manifest = result["collector_shard_manifests"][0]

    assert [item[0] for item in requests] == ["/t/274354.json", "/t/274354/posts.json"]
    assert len(records) == 2
    assert records[0]["target"]["answer"] == "First collected post"
    assert records[1]["target"]["answer"] == "Second collected post"
    assert records[0]["metadata"]["collector_provenance"]["adapter_name"] == "discourse_topic_adapter"
    assert records[0]["metadata"]["collector_provenance"]["tool_name"] == "discourse_topic_collect"
    assert manifest["status"] == "running"
    assert manifest["tool"]["name"] == "discourse_topic_collect"
    assert manifest["progress"]["cursor"] == "101"
    assert manifest["progress"]["items_collected"] == 2
    assert manifest["progress"]["items_emitted"] == 2
    assert manifest["progress"]["remaining_item_count"] == 1


@pytest.mark.asyncio
async def test_discourse_topic_collector_resumes_from_cursor_and_tracks_filtered_posts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/t/274354.json"
        return httpx.Response(
            200,
            json={
                "id": 274354,
                "slug": "api-examples",
                "title": "API examples",
                "posts_count": 3,
                "highest_post_number": 3,
                "post_stream": {
                    "stream": [100, 101, 102],
                    "posts": [
                        _topic_post(post_id=101, post_number=2, body="Visible resumed post"),
                        _topic_post(
                            post_id=102,
                            post_number=3,
                            body="Deleted resumed post",
                            user_deleted=True,
                            deleted_at="2026-06-24T00:30:00+00:00",
                        ),
                    ],
                },
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = DiscourseTopicCollectorAdapter(client=client)

    try:
        result = await adapter.collect_shard(
            {
                "base_url": "https://forum.example",
                "topic_id": "274354",
                "progress": {
                    "cursor": "100",
                    "items_collected": 4,
                    "items_emitted": 3,
                },
                "policy": {"disposition": "allow"},
            }
        )
    finally:
        await client.aclose()

    records = result["collector_dataset_records"]
    manifest = result["collector_shard_manifests"][0]

    assert len(records) == 1
    assert records[0]["target"]["answer"] == "Visible resumed post"
    assert manifest["status"] == "completed"
    assert manifest["progress"]["cursor"] == "102"
    assert manifest["progress"]["items_collected"] == 6
    assert manifest["progress"]["items_emitted"] == 4
    assert manifest["progress"]["remaining_item_count"] == 0
    assert manifest["progress"]["last_emitted_post_id"] == 102
    assert manifest["progress"]["last_emitted_post_number"] == 3


@pytest.mark.asyncio
async def test_discourse_topic_collect_tool_schema_allows_base_url_topic_id_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/t/274354.json"
        return httpx.Response(
            200,
            json={
                "id": 274354,
                "slug": "api-examples",
                "title": "API examples",
                "post_stream": {
                    "stream": [100],
                    "posts": [
                        _topic_post(post_id=100, post_number=1, body="Only collected post"),
                    ],
                },
            },
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = DiscourseTopicCollectTool()
    tool._adapter = DiscourseTopicCollectorAdapter(client=client)

    try:
        result = await tool.execute(base_url="https://forum.example", topic_id="274354")
    finally:
        await client.aclose()

    assert result.error is None
    assert result.output["collector_shard_manifests"][0]["tool"]["name"] == "discourse_topic_collect"
    assert tool.tool_capabilities["domains"] == ["web"]
    assert "dataset_collection" in tool.tool_capabilities["preference_tags"]
    assert any(
        set(option.get("required", [])) == {"base_url", "topic_id"}
        for option in tool.parameters_schema["anyOf"]
    )
