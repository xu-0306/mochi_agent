"""Encrypted local secret storage for configuration credentials.

Secrets are deliberately kept out of YAML, SQLite, and API payloads. On
Windows the encryption key is protected by the current user's DPAPI profile;
on other platforms an operator-provided MOCHI_SECRET_STORE_KEY is required so
we never silently fall back to plaintext storage.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path

try:  # cryptography is a runtime dependency, but keep import errors actionable.
    from cryptography.fernet import Fernet, InvalidToken
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal installs
    Fernet = None  # type: ignore[assignment,misc]
    InvalidToken = ValueError  # type: ignore[assignment,misc]


SECRET_STORE_SERVICE = "mochi-config-secrets-v1"
SECRET_STORE_LOCK_TIMEOUT_SECONDS = 30.0
SECRET_STORE_LOCK_STALE_SECONDS = 120.0
SECRET_STORE_LOCK_POLL_SECONDS = 0.05


class SecretStoreError(RuntimeError):
    """Base error raised for secret-store failures."""


class SecretStoreUnavailable(SecretStoreError):
    """Raised when no protected key source is available."""


if os.name == "nt":

    class _DataBlob(ctypes.Structure):
        _fields_ = [
            ("cbData", ctypes.c_uint32),
            ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
        ]


def _dpapi_transform(data: bytes, *, unprotect: bool) -> bytes:
    if os.name != "nt":  # pragma: no cover - guarded by callers
        raise SecretStoreUnavailable("Windows DPAPI is unavailable on this platform.")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    operation = crypt32.CryptUnprotectData if unprotect else crypt32.CryptProtectData
    operation.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(_DataBlob),
    ]
    operation.restype = ctypes.c_int

    source = ctypes.create_string_buffer(data)
    source_blob = _DataBlob(
        len(data),
        ctypes.cast(source, ctypes.POINTER(ctypes.c_ubyte)),
    )
    result_blob = _DataBlob()
    if not operation(
        ctypes.byref(source_blob),
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(result_blob),
    ):
        error = ctypes.get_last_error()
        raise SecretStoreError(
            f"Windows DPAPI {'unprotect' if unprotect else 'protect'} failed ({error})."
        )
    try:
        return ctypes.string_at(result_blob.pbData, result_blob.cbData)
    finally:
        if result_blob.pbData:
            kernel32.LocalFree(ctypes.cast(result_blob.pbData, ctypes.c_void_p))


def _protect_key(key: bytes) -> bytes:
    return _dpapi_transform(key, unprotect=False)


def _unprotect_key(value: bytes) -> bytes:
    return _dpapi_transform(value, unprotect=True)


def _derive_env_key(value: str) -> bytes:
    candidate = value.strip().encode("utf-8")
    if not candidate:
        raise SecretStoreUnavailable("MOCHI_SECRET_STORE_KEY must not be empty.")
    # Accept a native Fernet key, while also allowing a passphrase in local
    # deployments without weakening the no-plaintext-at-rest guarantee.
    if len(candidate) == 44:
        try:
            Fernet(candidate)  # type: ignore[misc]
            return candidate
        except (TypeError, ValueError):
            pass
    return base64.urlsafe_b64encode(hashlib.sha256(candidate).digest())


def _secure_file_mode(path: Path) -> None:
    with suppress(OSError):
        path.chmod(0o600)


class SecretStore:
    """Small encrypted key/value store scoped to one config directory."""

    def __init__(self, path: Path) -> None:
        if Fernet is None:
            raise SecretStoreUnavailable(
                "The 'cryptography' package is required for encrypted secret storage."
            )
        self.path = Path(path).expanduser()
        self.key_path = self.path.with_name(f"{self.path.name}.key")
        self._fernet = Fernet(self._load_or_create_key())

    @classmethod
    def for_config_path(cls, config_path: str | Path | None) -> SecretStore:
        root = Path(".mochi") if config_path is None else Path(config_path).expanduser().parent
        return cls(root / "secrets.enc")

    def _load_or_create_key(self) -> bytes:
        if os.name == "nt":
            if self.key_path.is_file():
                try:
                    return _unprotect_key(self.key_path.read_bytes())
                except (OSError, SecretStoreError) as exc:
                    raise SecretStoreError(
                        f"Unable to unlock protected Mochi secret key: {self.key_path}"
                    ) from exc
            if self.path.exists():
                raise SecretStoreError(
                    f"Encrypted secrets exist but their protected key is missing: {self.key_path}"
                )
            # Key creation must be serialized with secret mutations.  Recheck
            # after acquiring the lock because another process may have won.
            with self._lock():
                if self.key_path.is_file():
                    try:
                        return _unprotect_key(self.key_path.read_bytes())
                    except (OSError, SecretStoreError) as exc:
                        raise SecretStoreError(
                            f"Unable to unlock protected Mochi secret key: {self.key_path}"
                        ) from exc
                if self.path.exists():
                    raise SecretStoreError(
                        f"Encrypted secrets exist but their protected key is missing: {self.key_path}"
                    )
                key = Fernet.generate_key()  # type: ignore[union-attr]
                self.key_path.parent.mkdir(parents=True, exist_ok=True)
                protected = _protect_key(key)
                temp = self.key_path.with_name(
                    f"{self.key_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
                )
                temp.write_bytes(protected)
                _secure_file_mode(temp)
                os.replace(temp, self.key_path)
                _secure_file_mode(self.key_path)
                return key

        env_key = os.getenv("MOCHI_SECRET_STORE_KEY")
        if not env_key:
            raise SecretStoreUnavailable(
                "No protected secret-store key is available. Set MOCHI_SECRET_STORE_KEY "
                "on non-Windows hosts (or configure an OS keyring)."
            )
        return _derive_env_key(env_key)

    @staticmethod
    def _reference_key(reference: str) -> str:
        # References are logical paths, never credential material. Hash them so
        # endpoint names and config structure are not disclosed in metadata.
        return hashlib.sha256(reference.encode("utf-8")).hexdigest()

    @contextmanager
    def _lock(self) -> Iterator[None]:
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + SECRET_STORE_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                try:
                    os.write(fd, str(os.getpid()).encode("ascii"))
                finally:
                    os.close(fd)
                break
            except FileExistsError:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                except FileNotFoundError:
                    continue
                owner_pid: int | None = None
                with suppress(OSError, ValueError):
                    owner_pid = int(lock_path.read_text(encoding="ascii").strip())
                # Only recover locks from a process that is demonstrably dead.
                # A long-running live writer must never be evicted by age.
                if age >= SECRET_STORE_LOCK_STALE_SECONDS and (
                    owner_pid is None or not _pid_alive(owner_pid)
                ):
                    with suppress(FileNotFoundError):
                        lock_path.unlink()
                    continue
                if time.monotonic() >= deadline:
                    raise SecretStoreError(
                        f"Timed out waiting for secret store lock at {lock_path}."
                    ) from None
                time.sleep(SECRET_STORE_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            with suppress(FileNotFoundError):
                lock_path.unlink()

    def _read(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        try:
            token = self.path.read_bytes()
            payload = json.loads(self._fernet.decrypt(token).decode("utf-8"))
        except (OSError, InvalidToken, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SecretStoreError(f"Unable to read encrypted secrets: {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise SecretStoreError(f"Unsupported encrypted secret format: {self.path}")
        values = payload.get("values")
        if not isinstance(values, dict):
            raise SecretStoreError(f"Malformed encrypted secret payload: {self.path}")
        return {
            str(key): str(value)
            for key, value in values.items()
            if isinstance(key, str) and isinstance(value, str)
        }

    def _write(self, values: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": 1, "service": SECRET_STORE_SERVICE, "values": values},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        encrypted = self._fernet.encrypt(payload)
        temp = self.path.with_name(
            f"{self.path.name}.tmp-{os.getpid()}-{time.time_ns()}"
        )
        temp.write_bytes(encrypted)
        _secure_file_mode(temp)
        os.replace(temp, self.path)
        _secure_file_mode(self.path)

    def get(self, reference: str) -> str | None:
        if not reference:
            return None
        values = self._read()
        return values.get(self._reference_key(reference))

    def mutate(self, changes: Mapping[str, str | None]) -> None:
        """Apply multiple set/delete operations under one lock and one write."""
        normalized: dict[str, str | None] = {}
        for reference, value in changes.items():
            key = reference.strip()
            if not key:
                raise ValueError("Secret reference must not be empty.")
            normalized[key] = value.strip() if isinstance(value, str) else None
        if not normalized:
            return
        with self._lock():
            values = self._read()
            for reference, value in normalized.items():
                key = self._reference_key(reference)
                if value:
                    values[key] = value
                else:
                    values.pop(key, None)
            if values:
                self._write(values)
            elif self.path.exists():
                with suppress(FileNotFoundError):
                    self.path.unlink()

    def set(self, reference: str, value: str) -> None:
        normalized_reference = reference.strip()
        normalized_value = value.strip()
        if not normalized_reference:
            raise ValueError("Secret reference must not be empty.")
        if not normalized_value:
            self.delete(normalized_reference)
            return
        self.mutate({normalized_reference: normalized_value})

    def delete(self, reference: str) -> None:
        if not reference:
            return
        self.mutate({reference: None})


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


__all__ = ["SecretStore", "SecretStoreError", "SecretStoreUnavailable"]
