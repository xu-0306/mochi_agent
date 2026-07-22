"""Strict JSON-lines protocol for native sandbox brokers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from mochi.runtime.sandbox.base import SandboxCapabilities, SandboxPlan

BROKER_PROTOCOL_VERSION = 1
BrokerMessageType = Literal["hello", "run", "cancel", "result", "error"]
_MESSAGE_TYPES = {"hello", "run", "cancel", "result", "error"}
_PAYLOAD_FIELDS = {
    "hello": {"backend", "version", "capabilities"},
    "run": {"plan"},
    "cancel": {"run_id"},
    "result": {"run_id", "exit_code", "stdout", "stderr", "timed_out"},
    "error": {"code", "message", "retryable"},
}


class BrokerProtocolError(ValueError):
    """Raised when a broker frame is malformed or not bound to the request."""


@dataclass(frozen=True, slots=True)
class BrokerFrame:
    message_type: BrokerMessageType
    nonce: str
    payload: dict[str, Any]
    protocol_version: int = BROKER_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != BROKER_PROTOCOL_VERSION:
            raise BrokerProtocolError("Unsupported sandbox broker protocol version.")
        if self.message_type not in _MESSAGE_TYPES:
            raise BrokerProtocolError("Unsupported sandbox broker message type.")
        if len(self.nonce) < 16 or len(self.nonce) > 128:
            raise BrokerProtocolError("Sandbox broker nonce is invalid.")
        if len(self.payload) > 128:
            raise BrokerProtocolError("Sandbox broker payload is too large.")
        _validate_payload(self.message_type, self.payload)

    def encode(self) -> bytes:
        value = {
            "protocol_version": self.protocol_version,
            "type": self.message_type,
            "nonce": self.nonce,
            "payload": self.payload,
        }
        encoded = (
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        if len(encoded) > 1_048_576:
            raise BrokerProtocolError("Sandbox broker frame exceeds the size limit.")
        return encoded


def decode_frame(raw: bytes | str, *, expected_nonce: str | None = None) -> BrokerFrame:
    if isinstance(raw, bytes):
        if len(raw) > 1_048_576:
            raise BrokerProtocolError("Sandbox broker frame exceeds the size limit.")
        text = raw.decode("utf-8", errors="strict")
    else:
        text = raw
        if len(text.encode("utf-8")) > 1_048_576:
            raise BrokerProtocolError("Sandbox broker frame exceeds the size limit.")
    if text.count("\n") > 1 or ("\n" in text and not text.endswith("\n")):
        raise BrokerProtocolError("Sandbox broker accepts exactly one JSON line.")
    try:
        decoded: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BrokerProtocolError("Sandbox broker frame is not valid JSON.") from exc
    if not isinstance(decoded, Mapping):
        raise BrokerProtocolError("Sandbox broker frame must be an object.")
    value = cast(Mapping[str, Any], decoded)
    allowed = {"protocol_version", "type", "nonce", "payload"}
    if set(value) != allowed:
        raise BrokerProtocolError("Sandbox broker frame fields do not match the protocol.")
    payload = value.get("payload")
    if not isinstance(payload, dict):
        raise BrokerProtocolError("Sandbox broker payload must be an object.")
    typed_payload = cast(dict[str, Any], payload)
    message_type = cast(BrokerMessageType, str(value.get("type") or ""))
    protocol_version = value.get("protocol_version")
    if not isinstance(protocol_version, int) or isinstance(protocol_version, bool):
        raise BrokerProtocolError("Sandbox broker protocol version must be an integer.")
    frame = BrokerFrame(
        protocol_version=protocol_version,
        message_type=message_type,
        nonce=str(value.get("nonce") or ""),
        payload=typed_payload,
    )
    if expected_nonce is not None and frame.nonce != expected_nonce:
        raise BrokerProtocolError("Sandbox broker nonce mismatch.")
    return frame


def _validate_payload(
    message_type: BrokerMessageType,
    payload: Mapping[str, Any],
) -> None:
    expected_payload_fields = _PAYLOAD_FIELDS.get(message_type)
    if expected_payload_fields is None or set(payload) != expected_payload_fields:
        raise BrokerProtocolError("Sandbox broker payload fields do not match the message type.")
    if message_type == "run":
        raw_plan = payload.get("plan")
        if not isinstance(raw_plan, Mapping):
            raise BrokerProtocolError("Sandbox broker run plan must be an object.")
        SandboxPlan.from_dict(cast(Mapping[str, Any], raw_plan))
        return
    if message_type == "hello":
        raw_capabilities = payload.get("capabilities")
        if not isinstance(raw_capabilities, Mapping):
            raise BrokerProtocolError("Sandbox broker hello capabilities must be an object.")
        capabilities = SandboxCapabilities.from_dict(
            cast(Mapping[str, Any], raw_capabilities)
        )
        if payload.get("backend") != capabilities.backend or payload.get("version") != capabilities.version:
            raise BrokerProtocolError("Sandbox broker hello capability identity mismatch.")
        return
    if message_type == "cancel":
        if not _bounded_protocol_string(payload.get("run_id"), maximum=128):
            raise BrokerProtocolError("Sandbox broker cancel run_id is invalid.")
        return
    if message_type == "result":
        exit_code = payload.get("exit_code")
        if (
            not _bounded_protocol_string(payload.get("run_id"), maximum=128)
            or not isinstance(exit_code, int)
            or isinstance(exit_code, bool)
            or not isinstance(payload.get("stdout"), str)
            or not isinstance(payload.get("stderr"), str)
            or not isinstance(payload.get("timed_out"), bool)
        ):
            raise BrokerProtocolError("Sandbox broker result payload is invalid.")
        return
    if (
        not _bounded_protocol_string(payload.get("code"), maximum=128)
        or not _bounded_protocol_string(payload.get("message"), maximum=4096)
        or not isinstance(payload.get("retryable"), bool)
    ):
        raise BrokerProtocolError("Sandbox broker error payload is invalid.")


def _bounded_protocol_string(value: Any, *, maximum: int) -> bool:
    return isinstance(value, str) and bool(value) and len(value) <= maximum


def hello_frame(*, nonce: str, backend: str, version: str, capabilities: Mapping[str, Any]) -> BrokerFrame:
    return BrokerFrame(
        message_type="hello",
        nonce=nonce,
        payload={
            "backend": backend,
            "version": version,
            "capabilities": dict(capabilities),
        },
    )


def run_frame(plan: SandboxPlan) -> BrokerFrame:
    return BrokerFrame(
        message_type="run",
        nonce=plan.request_nonce,
        payload={"plan": plan.to_dict()},
    )


def cancel_frame(*, nonce: str, run_id: str) -> BrokerFrame:
    return BrokerFrame(message_type="cancel", nonce=nonce, payload={"run_id": run_id})


__all__ = [
    "BROKER_PROTOCOL_VERSION",
    "BrokerFrame",
    "BrokerMessageType",
    "BrokerProtocolError",
    "cancel_frame",
    "decode_frame",
    "hello_frame",
    "run_frame",
]
