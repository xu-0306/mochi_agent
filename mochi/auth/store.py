"""Persistent auth store used for OAuth-backed providers."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from pydantic import SecretStr

from .models import AuthStoreData, OpenAICodexAuthProfile, OpenAICodexPendingOAuthFlow

AUTH_STORE_LOCK_TIMEOUT_SECONDS = 30.0
AUTH_STORE_LOCK_STALE_SECONDS = 120.0
AUTH_STORE_LOCK_POLL_SECONDS = 0.1

T = TypeVar("T")


def _serialize_secret_value(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, dict):
        return {key: _serialize_secret_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_secret_value(item) for item in value]
    return value


class AuthStore:
    """JSON-backed auth profile store."""

    def __init__(self, path: Path, *, fallback_paths: Iterable[Path] = ()) -> None:
        self._path = path
        self._fallback_paths = tuple(fallback_paths)

    @property
    def path(self) -> Path:
        return self._path

    def _lock_path(self) -> Path:
        return self._path.parent / "locks" / "auth-store.lock"

    @contextmanager
    def _file_lock(self):
        lock_path = self._lock_path()
        deadline = time.monotonic() + AUTH_STORE_LOCK_TIMEOUT_SECONDS
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "pid": os.getpid(),
                                "created_at": int(time.time()),
                            },
                            ensure_ascii=False,
                        )
                    )
                break
            except FileExistsError:
                try:
                    age_seconds = time.time() - lock_path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age_seconds >= AUTH_STORE_LOCK_STALE_SECONDS:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Timed out waiting for auth store lock at {lock_path.name}."
                    )
                time.sleep(AUTH_STORE_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    def _load_path(self) -> Path | None:
        if self._path.is_file():
            return self._path
        for fallback_path in self._fallback_paths:
            if fallback_path.is_file():
                return fallback_path
        return None

    def _load_unlocked(self) -> AuthStoreData:
        load_path = self._load_path()
        if load_path is None:
            return AuthStoreData()
        payload = json.loads(load_path.read_text(encoding="utf-8"))
        return AuthStoreData.model_validate(payload if isinstance(payload, dict) else {})

    def load(self) -> AuthStoreData:
        return self._load_unlocked()

    def _save_unlocked(self, data: AuthStoreData) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = _serialize_secret_value(data.model_dump(mode="python", exclude_none=True))
        payload = json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True)
        temp_path = self._path.with_name(
            f"{self._path.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )
        temp_path.write_text(payload, encoding="utf-8")
        os.replace(temp_path, self._path)
        return self._path

    def save(self, data: AuthStoreData) -> Path:
        with self._file_lock():
            return self._save_unlocked(data)

    def mutate(self, mutator: Callable[[AuthStoreData], T]) -> T:
        with self._file_lock():
            data = self._load_unlocked()
            result = mutator(data)
            self._save_unlocked(data)
            return result

    def list_openai_codex_profiles(self) -> list[OpenAICodexAuthProfile]:
        return list(self.load().openai_codex_profiles)

    def get_openai_codex_profile(self, profile_id: str) -> OpenAICodexAuthProfile | None:
        for profile in self.list_openai_codex_profiles():
            if profile.profile_id == profile_id:
                return profile
        return None

    def upsert_openai_codex_profile(self, profile: OpenAICodexAuthProfile) -> Path:
        def _mutate(data: AuthStoreData) -> Path:
            profiles = [
                item
                for item in data.openai_codex_profiles
                if item.profile_id != profile.profile_id
            ]
            data.openai_codex_profiles = [profile, *profiles]
            return self._path

        return self.mutate(_mutate)

    def delete_openai_codex_profile(self, profile_id: str) -> bool:
        def _mutate(data: AuthStoreData) -> bool:
            remaining = [
                item
                for item in data.openai_codex_profiles
                if item.profile_id != profile_id
            ]
            if len(remaining) == len(data.openai_codex_profiles):
                return False
            data.openai_codex_profiles = remaining
            return True

        return self.mutate(_mutate)

    def list_openai_codex_pending_flows(self) -> list[OpenAICodexPendingOAuthFlow]:
        return list(self.load().openai_codex_pending_flows)

    def get_openai_codex_pending_flow_by_state(
        self, state: str
    ) -> OpenAICodexPendingOAuthFlow | None:
        for flow in self.list_openai_codex_pending_flows():
            if flow.state == state:
                return flow
        return None

    def upsert_openai_codex_pending_flow(
        self, flow: OpenAICodexPendingOAuthFlow
    ) -> Path:
        def _mutate(data: AuthStoreData) -> Path:
            pending = [
                item
                for item in data.openai_codex_pending_flows
                if item.flow_id != flow.flow_id
            ]
            data.openai_codex_pending_flows = [flow, *pending]
            return self._path

        return self.mutate(_mutate)
