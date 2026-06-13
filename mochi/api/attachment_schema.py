"""Shared structured attachment request schema."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from mochi.backends.types import AttachmentRef


class AttachmentPayload(BaseModel):
    """Canonical structured attachment payload used by chat and workflow APIs."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    size: int | None = Field(default=None, ge=0)
    content_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("content_type", "contentType"),
    )
    source: str | None = None
    line_start: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("line_start", "lineStart"),
    )
    line_end: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("line_end", "lineEnd"),
    )
    quote: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _validate_line_range(self) -> AttachmentPayload:
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            msg = "`line_end` must be greater than or equal to `line_start`."
            raise ValueError(msg)
        return self

    def to_attachment_ref(self) -> AttachmentRef:
        return AttachmentRef(
            name=self.name,
            path=self.path,
            size=self.size,
            content_type=self.content_type,
            source=self.source,
            line_start=self.line_start,
            line_end=self.line_end,
            quote=self.quote,
            note=self.note,
        )

    def to_attachment_dict(self) -> dict[str, Any]:
        return self.to_attachment_ref().to_dict()
