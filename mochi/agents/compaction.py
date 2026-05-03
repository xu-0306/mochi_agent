"""長對話壓縮：將過舊訊息摘要為滾動摘要。"""

from __future__ import annotations

from dataclasses import dataclass

from mochi.backends.types import Message


@dataclass(frozen=True)
class CompactionPolicy:
    """對話壓縮策略。"""

    trigger_messages: int
    """訊息數超過此門檻時觸發壓縮。"""

    retain_recent_messages: int
    """壓縮後保留的最近訊息數。"""

    max_summary_chars: int = 1200
    """摘要最大字元數。"""

    max_points: int = 8
    """每次壓縮最多保留幾個重點。"""

    max_chars_per_point: int = 120
    """每個重點的最大字元數。"""


@dataclass(frozen=True)
class CompactionResult:
    """單次壓縮結果。"""

    summary: str
    retained_history: list[Message]
    compacted_count: int


class ConversationCompactor:
    """根據策略對對話歷史進行滾動壓縮。"""

    def __init__(self, policy: CompactionPolicy) -> None:
        self._policy = policy

    @classmethod
    def from_max_messages(cls, max_messages: int) -> ConversationCompactor:
        """由短期記憶上限推導保守壓縮策略。"""
        safe_max = max(8, max_messages)
        trigger = max(6, int(safe_max * 0.8))
        retain = max(4, int(safe_max * 0.4))
        if retain >= trigger:
            retain = max(4, trigger // 2)
        return cls(
            CompactionPolicy(
                trigger_messages=trigger,
                retain_recent_messages=retain,
            )
        )

    def compact(
        self,
        history: list[Message],
        *,
        previous_summary: str | None = None,
    ) -> CompactionResult | None:
        """若歷史過長則壓縮舊訊息並回傳結果。"""
        if len(history) <= self._policy.trigger_messages:
            return None

        retain = min(self._policy.retain_recent_messages, len(history))
        if retain <= 0 or len(history) <= retain:
            return None

        compacted = history[:-retain]
        if not compacted:
            return None

        summary = self._merge_summary(previous_summary=previous_summary, compacted=compacted)
        return CompactionResult(
            summary=summary,
            retained_history=list(history[-retain:]),
            compacted_count=len(compacted),
        )

    def _merge_summary(
        self,
        *,
        previous_summary: str | None,
        compacted: list[Message],
    ) -> str:
        points: list[str] = []
        if previous_summary and previous_summary.strip():
            points.append(f"Earlier summary: {self._truncate(previous_summary.strip(), 400)}")

        for message in compacted[-self._policy.max_points:]:
            text = self._normalize_text(message.content)
            if not text:
                continue
            role = message.role.capitalize()
            points.append(f"{role}: {self._truncate(text, self._policy.max_chars_per_point)}")

        if not points:
            points.append("Conversation continued with low-signal messages.")

        summary = "\n".join(f"- {point}" for point in points)
        return self._truncate(summary, self._policy.max_summary_chars)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().split())

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 1]}…"
