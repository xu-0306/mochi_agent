"""Stable identity helpers for configured model targets.

Display labels are intentionally not used as identities.  A label is a UI
concern and two providers may quite legitimately expose the same model name.
The target id is derived from the provider/runtime target and therefore remains
stable when a label is changed or translated.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

_SENSITIVE_URL_QUERY_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "key",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}


def normalize_model_endpoint(value: str | None) -> str:
    """Normalize an endpoint for identity comparisons without touching secrets."""

    raw = (value or "").strip()
    if not raw:
        return ""
    # URLs are case-insensitive only for scheme/host. Preserve non-sensitive
    # path/query routing information while dropping credential-shaped values.
    try:
        parsed = urlsplit(raw)
    except ValueError:
        parsed = None
    if parsed is not None and parsed.scheme and parsed.netloc:
        scheme = parsed.scheme.casefold()
        # Never carry URL userinfo (which may contain a token) into an
        # identity, DOM key, or encrypted-secret reference.
        host = (parsed.hostname or "").casefold()
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        netloc = f"{host}:{port}" if port is not None else host
        path = parsed.path.rstrip("/") or ""
        query = urlencode(
            [
                (name, item)
                for name, item in parse_qsl(parsed.query, keep_blank_values=True)
                if name.casefold().replace("-", "_") not in _SENSITIVE_URL_QUERY_NAMES
            ],
            doseq=True,
        )
        # Fragments are client-only and can accidentally contain credentials;
        # they must not participate in a target identity.
        return urlunsplit((scheme, netloc, path, query, "")).rstrip("/")
    return raw.rstrip("/")


def model_target_id(
    *,
    provider: str | None,
    model: str | None,
    model_spec: str | None,
    base_url: str | None = None,
    backend_type: str | None = None,
    auth_profile_id: str | None = None,
) -> str:
    """Return an opaque, deterministic id for one runtime model target.

    The digest deliberately excludes API key material.  Rotating a key keeps
    the same target (and therefore updates the same saved entry), while two
    endpoints or providers exposing the same model name remain distinct.
    """

    normalized_provider = (provider or "unknown").strip().casefold() or "unknown"
    normalized_model = (model or "").strip()
    normalized_spec = normalize_model_endpoint(model_spec)
    normalized_base = normalize_model_endpoint(base_url)
    identity = {
        "provider": normalized_provider,
        "model": normalized_model,
        "model_spec": normalized_spec,
        "base_url": normalized_base,
        "backend_type": (backend_type or "").strip().casefold(),
        # OAuth profiles are identities; API keys are not.  This permits key
        # rotation without creating duplicate rows while keeping OAuth
        # accounts separate when a provider supports multiple profiles.
        "auth_profile_id": (auth_profile_id or "").strip(),
    }
    canonical = json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
    return f"model:{normalized_provider}:{digest}"


def configured_model_target_id(model: Any) -> str:
    """Resolve a target id from a ConfiguredModelConfig-like object."""

    explicit = getattr(model, "target_id", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return model_target_id(
        provider=getattr(model, "provider", None),
        model=getattr(model, "model", None),
        model_spec=getattr(model, "model_spec", None),
        base_url=getattr(model, "base_url", None),
        backend_type=getattr(model, "backend_type", None),
        auth_profile_id=getattr(model, "auth_profile_id", None),
    )


def model_target_alias(
    *,
    provider: str | None,
    model: str | None,
    model_spec: str | None,
    base_url: str | None = None,
    backend_type: str | None = None,
    auth_profile_id: str | None = None,
) -> str:
    """Return a readable fallback alias for clients without target_id.

    The first four segments preserve the legacy alias shape. Optional spec,
    backend and auth segments are appended only when those dimensions are
    needed to distinguish otherwise-identical runtime targets. This keeps old
    clients working while preventing collisions between OAuth profiles or
    managed/external targets sharing an endpoint and model name.
    """

    normalized_provider = (provider or "unknown").strip() or "unknown"
    normalized_model = (model or "").strip()
    normalized_spec = normalize_model_endpoint(model_spec)
    endpoint = normalize_model_endpoint(base_url) or normalized_spec
    parts = ["target", quote(normalized_provider, safe="-_.!~*'()")]
    if endpoint:
        parts.append(quote(endpoint, safe="-_.!~*'()"))
    if normalized_model:
        parts.append(quote(normalized_model, safe="-_.!~*'()"))
    if normalized_spec and endpoint and normalized_spec != endpoint:
        parts.extend(["spec", quote(normalized_spec, safe="-_.!~*'()")])
    normalized_backend = (backend_type or "").strip()
    if normalized_backend:
        parts.extend(["backend", quote(normalized_backend, safe="-_.!~*'()")])
    normalized_auth = (auth_profile_id or "").strip()
    if normalized_auth:
        parts.extend(["auth", quote(normalized_auth, safe="-_.!~*'()")])
    return ":".join(parts)


__all__ = [
    "configured_model_target_id",
    "model_target_id",
    "model_target_alias",
    "normalize_model_endpoint",
]
