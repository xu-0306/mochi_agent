"""Source-specific collector adapter for Discourse topics."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from typing import Any, Mapping
from urllib.parse import urljoin, urlparse

from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.collector_adapter import BaseCollectorAdapter, CollectorRequestPolicy

_DISCOURSE_TOPIC_URL_RE = re.compile(r"/t/(?:[^/]+/)?(?P<topic_id>\d+)(?:/.*)?$")
_DEFAULT_SPECIFIC_POSTS_BATCH_SIZE = 20
_DISCOURSE_TOPIC_ADAPTER_NAME = "discourse_topic_adapter"
_DISCOURSE_TOPIC_TOOL_NAME = "discourse_topic_collect"


@dataclass(frozen=True, slots=True)
class _DiscourseTopicShardRequest:
    base_url: str
    topic_id: str
    topic_slug: str | None
    topic_url: str
    shard_id: str
    cursor: str | None
    max_posts: int | None
    include_deleted: bool
    api_key: str | None
    api_username: str | None
    policy_license: str | None
    policy_disposition: str | None
    previous_items_collected: int
    previous_items_emitted: int


class _CookedHtmlTextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "ol",
        "p",
        "pre",
        "section",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        text = unescape("".join(self._parts))
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in text.split("\n")]
        compact_lines: list[str] = []
        previous_blank = False
        for line in lines:
            if not line:
                if previous_blank:
                    continue
                compact_lines.append("")
                previous_blank = True
                continue
            compact_lines.append(line)
            previous_blank = False
        return "\n".join(compact_lines).strip()


class DiscourseTopicCollectorAdapter(BaseCollectorAdapter):
    """Collect topic posts from a Discourse forum with shard-level resume semantics."""

    def __init__(
        self,
        *,
        request_policy: CollectorRequestPolicy | None = None,
        specific_posts_batch_size: int = _DEFAULT_SPECIFIC_POSTS_BATCH_SIZE,
        client: Any = None,
    ) -> None:
        super().__init__(request_policy=request_policy, client=client)
        self._specific_posts_batch_size = max(1, int(specific_posts_batch_size))

    @property
    def name(self) -> str:
        return _DISCOURSE_TOPIC_ADAPTER_NAME

    async def collect_shard(self, shard: Mapping[str, Any]) -> dict[str, Any]:
        request = _normalize_shard_request(shard)
        topic_payload = await self.request_json(
            "GET",
            f"{request.base_url}/t/{request.topic_id}.json",
            headers=_request_headers(request),
        )
        topic = dict(topic_payload or {})
        topic_post_stream = (
            dict(topic.get("post_stream"))
            if isinstance(topic.get("post_stream"), dict)
            else {}
        )
        stream_ids = _stream_ids(topic_post_stream.get("stream"))
        bootstrap_posts = _post_payloads(topic_post_stream.get("posts"))
        post_lookup = {int(post["id"]): post for post in bootstrap_posts if isinstance(post.get("id"), int)}

        next_post_ids = _select_next_post_ids(
            stream_ids,
            cursor=request.cursor,
            max_posts=request.max_posts,
        )
        missing_post_ids = [post_id for post_id in next_post_ids if post_id not in post_lookup]
        for chunk in _chunked(missing_post_ids, self._specific_posts_batch_size):
            specific_posts_payload = await self.request_json(
                "GET",
                f"{request.base_url}/t/{request.topic_id}/posts.json",
                headers=_request_headers(request),
                params=[("post_ids[]", str(post_id)) for post_id in chunk],
            )
            fetched_stream = (
                dict(specific_posts_payload.get("post_stream"))
                if isinstance(specific_posts_payload, dict)
                else {}
            )
            for post in _post_payloads(fetched_stream.get("posts")):
                if isinstance(post.get("id"), int):
                    post_lookup[int(post["id"])] = post

        selected_posts = [
            post_lookup[post_id]
            for post_id in next_post_ids
            if post_id in post_lookup
        ]
        emitted_posts = [
            post
            for post in selected_posts
            if request.include_deleted or _post_is_collectable(post)
        ]
        collector_records = [
            _build_dataset_record(
                request=request,
                topic=topic,
                post=post,
                tool_name=_DISCOURSE_TOPIC_TOOL_NAME,
            )
            for post in emitted_posts
        ]
        collector_provenance = [
            dict(record.get("metadata", {}).get("collector_provenance") or {})
            for record in collector_records
            if isinstance(record.get("metadata"), dict)
            and isinstance(record.get("metadata", {}).get("collector_provenance"), dict)
        ]

        previous_cursor = request.cursor
        next_cursor = previous_cursor
        if next_post_ids:
            next_cursor = str(next_post_ids[-1])
        last_activity_at = _last_post_activity_at(selected_posts) or _utcnow().isoformat()
        completed = not bool(_remaining_post_ids(stream_ids, cursor=next_cursor))
        processed_count = len(selected_posts)
        emitted_count = len(collector_records)
        shard_manifest = {
            "shard_id": request.shard_id,
            "adapter_name": self.name,
            "attempt_id": _string(shard.get("attempt_id")),
            "status": "completed" if completed else "running",
            "updated_at": last_activity_at,
            "completed_at": last_activity_at if completed else None,
            "source": {
                "url": request.topic_url,
                "id": f"topic:{request.topic_id}",
                "source_type": "discourse_topic",
            },
            "tool": {
                "name": _DISCOURSE_TOPIC_TOOL_NAME,
                "arguments": {
                    "base_url": request.base_url,
                    "topic_id": request.topic_id,
                    "topic_slug": request.topic_slug,
                    "max_posts": request.max_posts,
                    "include_deleted": request.include_deleted,
                },
            },
            "policy": {
                "license": request.policy_license,
                "disposition": request.policy_disposition,
            },
            "topic": {
                "id": int(request.topic_id),
                "slug": _string(topic.get("slug")) or request.topic_slug,
                "title": _string(topic.get("title")),
                "category_id": topic.get("category_id"),
                "tags": [str(item) for item in topic.get("tags", []) if str(item).strip()],
                "posts_count": topic.get("posts_count"),
                "highest_post_number": topic.get("highest_post_number"),
            },
            "progress": {
                "cursor": next_cursor,
                "items_collected": request.previous_items_collected + processed_count,
                "items_emitted": request.previous_items_emitted + emitted_count,
                "remaining_item_count": len(_remaining_post_ids(stream_ids, cursor=next_cursor)),
                "last_emitted_post_id": int(next_cursor) if next_cursor and next_cursor.isdigit() else None,
                "last_emitted_post_number": _last_post_number(selected_posts),
            },
        }
        return {
            "collector_dataset_records": collector_records,
            "collector_record_provenance": collector_provenance,
            "collector_shard_manifests": [_strip_none(shard_manifest)],
        }


class DiscourseTopicCollectTool(BaseTool):
    """Collect a Discourse topic incrementally into dataset and shard payloads."""

    def __init__(
        self,
        *,
        request_policy: CollectorRequestPolicy | None = None,
        specific_posts_batch_size: int = _DEFAULT_SPECIFIC_POSTS_BATCH_SIZE,
    ) -> None:
        self._adapter = DiscourseTopicCollectorAdapter(
            request_policy=request_policy,
            specific_posts_batch_size=specific_posts_batch_size,
        )

    @property
    def name(self) -> str:
        return _DISCOURSE_TOPIC_TOOL_NAME

    @property
    def description(self) -> str:
        return "Collect posts from a Discourse topic into dataset records, provenance records, and shard manifests."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "id": {"type": "string"},
                        "base_url": {"type": "string"},
                        "topic_id": {"type": "string"},
                        "topic_slug": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
                "auth": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string"},
                        "api_username": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
                "policy": {
                    "type": "object",
                    "properties": {
                        "license": {"type": "string"},
                        "disposition": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
                "progress": {
                    "type": "object",
                    "properties": {
                        "cursor": {"type": "string"},
                        "items_collected": {"type": "integer"},
                        "items_emitted": {"type": "integer"},
                    },
                    "additionalProperties": True,
                },
                "base_url": {"type": "string"},
                "topic_id": {"type": "string"},
                "topic_url": {"type": "string"},
                "topic_slug": {"type": "string"},
                "shard_id": {"type": "string"},
                "max_posts": {"type": "integer", "minimum": 1, "maximum": 100},
                "include_deleted": {"type": "boolean", "default": False},
            },
            "anyOf": [
                {
                    "required": ["source"],
                    "properties": {
                        "source": {
                            "type": "object",
                            "required": ["url"],
                        }
                    },
                },
                {
                    "required": ["base_url", "topic_id"],
                },
                {
                    "required": ["source"],
                    "properties": {
                        "source": {
                            "type": "object",
                            "required": ["base_url", "topic_id"],
                        }
                    },
                },
            ],
            "additionalProperties": True,
        }

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_open_world(self) -> bool:
        return True

    @property
    def tool_capabilities(self) -> dict[str, Any]:
        return {
            "domains": ["web"],
            "retrieval_modes": ["fetch"],
            "preference_tags": [
                "open_web",
                "structured_source_api",
                "forum_thread",
                "dataset_collection",
                "source_capture",
            ],
            "read_only": self.is_read_only,
            "destructive": self.is_destructive,
            "open_world": self.is_open_world,
        }

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    @property
    def search_hint(self) -> str | None:
        return "Use this to collect a Discourse topic in bounded batches and preserve a shard cursor."

    async def execute(
        self,
        *,
        context: ToolExecutionContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        del context
        try:
            output = await self._adapter.collect_shard(kwargs)
        except ValueError as exc:
            return ToolResult(error=str(exc))
        return ToolResult(output=output, metadata=output)


def _normalize_shard_request(
    shard: Mapping[str, Any],
) -> _DiscourseTopicShardRequest:
    payload = dict(shard)
    source = dict(payload.get("source") or {}) if isinstance(payload.get("source"), Mapping) else {}
    auth = dict(payload.get("auth") or {}) if isinstance(payload.get("auth"), Mapping) else {}
    progress = dict(payload.get("progress") or {}) if isinstance(payload.get("progress"), Mapping) else {}
    policy = dict(payload.get("policy") or {}) if isinstance(payload.get("policy"), Mapping) else {}

    source_url = (
        _string(source.get("url"))
        or _string(payload.get("topic_url"))
        or _string(payload.get("source_url"))
    )
    base_url = (
        _string(source.get("base_url"))
        or _string(payload.get("base_url"))
        or _base_url_from_source_url(source_url)
    )
    topic_id = (
        _string(source.get("topic_id"))
        or _string(payload.get("topic_id"))
        or _topic_id_from_source_url(source_url)
    )
    source_id = _string(source.get("id"))
    if topic_id is None and source_id and source_id.isdigit():
        topic_id = source_id
    topic_slug = _string(source.get("topic_slug")) or _string(payload.get("topic_slug")) or _topic_slug_from_source_url(source_url)
    if base_url is None or topic_id is None:
        raise ValueError("Discourse topic collector shard requires `base_url` and `topic_id`, or a parseable `source.url`/`topic_url`.")
    topic_url = source_url or _build_topic_url(base_url, topic_id, topic_slug)
    shard_id = (
        _string(payload.get("shard_id"))
        or _string(source.get("id"))
        or f"discourse-topic-{topic_id}"
    )
    cursor = _string(progress.get("cursor")) or _string(payload.get("cursor"))
    max_posts = _int(payload.get("max_posts"))
    if max_posts is None or max_posts <= 0:
        max_posts = _DEFAULT_SPECIFIC_POSTS_BATCH_SIZE
    include_deleted = bool(payload.get("include_deleted") or source.get("include_deleted"))
    api_key = _string(auth.get("api_key")) or _string(payload.get("api_key"))
    api_username = _string(auth.get("api_username")) or _string(payload.get("api_username"))
    policy_license = _string(policy.get("license")) or _string(payload.get("license"))
    policy_disposition = (
        _string(policy.get("disposition"))
        or _string(payload.get("policy_disposition"))
        or "review_required"
    )
    previous_items_collected = _int(progress.get("items_collected")) or 0
    previous_items_emitted = _int(progress.get("items_emitted")) or 0
    return _DiscourseTopicShardRequest(
        base_url=base_url.rstrip("/"),
        topic_id=topic_id,
        topic_slug=topic_slug,
        topic_url=topic_url,
        shard_id=shard_id,
        cursor=cursor,
        max_posts=max_posts,
        include_deleted=include_deleted,
        api_key=api_key,
        api_username=api_username,
        policy_license=policy_license,
        policy_disposition=policy_disposition,
        previous_items_collected=previous_items_collected,
        previous_items_emitted=previous_items_emitted,
    )


def _request_headers(request: _DiscourseTopicShardRequest) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
    }
    if request.api_key:
        headers["Api-Key"] = request.api_key
    if request.api_username:
        headers["Api-Username"] = request.api_username
    return headers


def _stream_ids(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    ids: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            ids.append(item)
            continue
        if isinstance(item, str) and item.strip().isdigit():
            ids.append(int(item.strip()))
    return ids


def _post_payloads(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _select_next_post_ids(
    stream_ids: list[int],
    *,
    cursor: str | None,
    max_posts: int | None,
) -> list[int]:
    remaining = _remaining_post_ids(stream_ids, cursor=cursor)
    if max_posts is not None:
        return remaining[:max_posts]
    return remaining


def _remaining_post_ids(stream_ids: list[int], *, cursor: str | None) -> list[int]:
    if not stream_ids:
        return []
    if cursor is None:
        return list(stream_ids)
    if cursor.isdigit():
        cursor_id = int(cursor)
        if cursor_id in stream_ids:
            cursor_index = stream_ids.index(cursor_id)
            return stream_ids[cursor_index + 1 :]
        return [post_id for post_id in stream_ids if post_id > cursor_id]
    return list(stream_ids)


def _post_is_collectable(post: Mapping[str, Any]) -> bool:
    if bool(post.get("hidden")):
        return False
    if bool(post.get("user_deleted")):
        return False
    if _string(post.get("deleted_at")) is not None:
        return False
    return True


def _build_dataset_record(
    *,
    request: _DiscourseTopicShardRequest,
    topic: Mapping[str, Any],
    post: Mapping[str, Any],
    tool_name: str,
) -> dict[str, Any]:
    post_id = int(post.get("id"))
    post_number = int(post.get("post_number"))
    topic_slug = _string(topic.get("slug")) or request.topic_slug
    post_url = _post_source_url(request.base_url, post, topic_id=request.topic_id, topic_slug=topic_slug)
    title = _string(topic.get("title")) or f"Discourse topic {request.topic_id}"
    answer = _post_text(post)
    collector_provenance = {
        "source_url": post_url,
        "source_id": f"topic:{request.topic_id}:post:{post_id}",
        "collected_at": _string(post.get("updated_at")) or _string(post.get("created_at")) or _utcnow().isoformat(),
        "adapter_name": _DISCOURSE_TOPIC_ADAPTER_NAME,
        "tool_name": tool_name,
        "tool_arguments": {
            "base_url": request.base_url,
            "topic_id": request.topic_id,
            "topic_slug": topic_slug,
            "post_id": post_id,
            "post_number": post_number,
        },
        "license": request.policy_license,
        "policy_disposition": request.policy_disposition,
        "dedupe_hash": _dedupe_hash(
            request.base_url,
            request.topic_id,
            post_id,
            answer,
        ),
        "shard_id": request.shard_id,
    }
    metadata = {
        "dataset_mode": "source_capture",
        "capability_family": "dataset_collection",
        "supervision_shape": "source_capture",
        "source_provenance": {
            "source_type": "discourse_topic_post",
            "site_url": request.base_url,
            "topic_id": int(request.topic_id),
            "topic_slug": topic_slug,
            "topic_title": title,
            "post_id": post_id,
            "post_number": post_number,
        },
        "collector_provenance": _strip_none(collector_provenance),
        "discourse": _strip_none(
            {
                "site_url": request.base_url,
                "topic_id": int(request.topic_id),
                "topic_slug": topic_slug,
                "topic_title": title,
                "category_id": topic.get("category_id"),
                "tags": [str(item) for item in topic.get("tags", []) if str(item).strip()],
                "post_id": post_id,
                "post_number": post_number,
                "reply_to_post_number": post.get("reply_to_post_number"),
                "username": _string(post.get("username")),
                "display_username": _string(post.get("display_username")),
                "created_at": _string(post.get("created_at")),
                "updated_at": _string(post.get("updated_at")),
            }
        ),
    }
    return {
        "contract": "dataset_record",
        "schema_version": "1.0",
        "input": f"Topic: {title}",
        "target": {
            "answer": answer,
        },
        "metadata": metadata,
        "supervision": {
            "type": "source_capture",
            "loss_mask": "target_only",
        },
    }


def _post_text(post: Mapping[str, Any]) -> str:
    cooked = _string(post.get("cooked")) or ""
    if not cooked:
        return ""
    parser = _CookedHtmlTextExtractor()
    parser.feed(cooked)
    parser.close()
    return parser.get_text()


def _post_source_url(
    base_url: str,
    post: Mapping[str, Any],
    *,
    topic_id: str,
    topic_slug: str | None,
) -> str:
    relative_url = _string(post.get("post_url"))
    if relative_url:
        return urljoin(f"{base_url}/", relative_url)
    post_number = int(post.get("post_number"))
    return _build_topic_url(base_url, topic_id, topic_slug, post_number=post_number)


def _build_topic_url(
    base_url: str,
    topic_id: str,
    topic_slug: str | None,
    *,
    post_number: int | None = None,
) -> str:
    slug = topic_slug or "-"
    suffix = f"/{post_number}" if post_number is not None else ""
    return f"{base_url}/t/{slug}/{topic_id}{suffix}"


def _base_url_from_source_url(source_url: str | None) -> str | None:
    if not source_url:
        return None
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _topic_id_from_source_url(source_url: str | None) -> str | None:
    if not source_url:
        return None
    parsed = urlparse(source_url)
    match = _DISCOURSE_TOPIC_URL_RE.search(parsed.path)
    if match is None:
        return None
    return match.group("topic_id")


def _topic_slug_from_source_url(source_url: str | None) -> str | None:
    if not source_url:
        return None
    parsed = urlparse(source_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 3 and parts[0] == "t" and not parts[1].isdigit():
        return parts[1]
    return None


def _last_post_activity_at(posts: list[Mapping[str, Any]]) -> str | None:
    best_dt: datetime | None = None
    best_value: str | None = None
    for post in posts:
        for candidate in (
            _string(post.get("updated_at")),
            _string(post.get("created_at")),
        ):
            if candidate is None:
                continue
            parsed = _parse_iso_datetime(candidate)
            if parsed is None:
                if best_value is None:
                    best_value = candidate
                continue
            if best_dt is None or parsed > best_dt:
                best_dt = parsed
                best_value = parsed.isoformat()
    return best_value


def _last_post_number(posts: list[Mapping[str, Any]]) -> int | None:
    post_numbers = [
        int(post["post_number"])
        for post in posts
        if isinstance(post.get("post_number"), int)
    ]
    if not post_numbers:
        return None
    return max(post_numbers)


def _dedupe_hash(base_url: str, topic_id: str, post_id: int, answer: str) -> str:
    payload = json.dumps(
        {
            "base_url": base_url,
            "topic_id": topic_id,
            "post_id": post_id,
            "answer": answer,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _chunked(items: list[int], size: int) -> list[list[int]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _strip_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, [], {})
    }


def _string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _utcnow() -> datetime:
    return datetime.now(UTC)
