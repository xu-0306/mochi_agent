"""Durable process identity and fail-closed recovered-process handles."""

from __future__ import annotations

import contextlib
import ctypes
import os
import select
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Literal

IdentityPlatform = Literal["windows", "linux"]
VerificationState = Literal["verified", "exited", "unknown", "mismatch"]


@dataclass(frozen=True)
class DurableProcessIdentity:
    """A process identity that remains meaningful after a runtime restart."""

    platform: IdentityPlatform
    pid: int
    windows_process_creation_filetime: int | None = None
    linux_boot_id: str | None = None
    linux_proc_start_time: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "pid": self.pid,
            "windows_process_creation_filetime": self.windows_process_creation_filetime,
            "linux_boot_id": self.linux_boot_id,
            "linux_proc_start_time": self.linux_proc_start_time,
        }

    @classmethod
    def from_dict(cls, payload: object) -> DurableProcessIdentity | None:
        if not isinstance(payload, dict):
            return None
        platform = payload.get("platform")
        pid = payload.get("pid")
        if platform not in {"windows", "linux"} or isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
            return None
        if platform == "windows":
            creation_time = payload.get("windows_process_creation_filetime")
            if isinstance(creation_time, bool) or not isinstance(creation_time, int) or creation_time < 0:
                return None
            return cls(
                platform="windows",
                pid=pid,
                windows_process_creation_filetime=creation_time,
            )
        boot_id = payload.get("linux_boot_id")
        start_time = payload.get("linux_proc_start_time")
        if not isinstance(boot_id, str) or not boot_id.strip():
            return None
        if isinstance(start_time, bool) or not isinstance(start_time, int) or start_time < 0:
            return None
        return cls(
            platform="linux",
            pid=pid,
            linux_boot_id=boot_id,
            linux_proc_start_time=start_time,
        )


@dataclass
class VerifiedProcessHandle:
    """One stable operating-system handle used for liveness and termination."""

    identity: DurableProcessIdentity
    windows_handle: int | None = None
    linux_pidfd: int | None = None

    def is_running(self) -> bool:
        if self.windows_handle is not None:
            return _windows_handle_is_running(self.windows_handle)
        if self.linux_pidfd is not None:
            try:
                readable, _, _ = select.select([self.linux_pidfd], [], [], 0)
            except OSError:
                return False
            return not readable
        return False

    def request_termination(self) -> bool:
        if self.windows_handle is not None:
            return _windows_terminate_handle(self.windows_handle)
        if self.linux_pidfd is not None:
            sender = getattr(signal, "pidfd_send_signal", None)
            if not callable(sender):
                return False
            try:
                sender(self.linux_pidfd, signal.SIGTERM, None, 0)
            except OSError:
                return False
            return True
        return False

    def force_termination(self) -> bool:
        if self.windows_handle is not None:
            return _windows_terminate_handle(self.windows_handle)
        if self.linux_pidfd is not None:
            sender = getattr(signal, "pidfd_send_signal", None)
            if not callable(sender):
                return False
            try:
                sender(self.linux_pidfd, signal.SIGKILL, None, 0)
            except OSError:
                return False
            return True
        return False

    def close(self) -> None:
        if self.windows_handle is not None:
            _windows_close_handle(self.windows_handle)
            self.windows_handle = None
        if self.linux_pidfd is not None:
            with contextlib.suppress(OSError):
                os.close(self.linux_pidfd)
            self.linux_pidfd = None


@dataclass(frozen=True)
class ProcessVerification:
    state: VerificationState
    handle: VerifiedProcessHandle | None
    reason: str


def capture_process_identity(pid: int) -> DurableProcessIdentity | None:
    """Capture the identity of a newly launched process without trusting PID alone."""

    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return None
    if sys.platform == "win32":
        handle, _ = _windows_open_process(pid, _PROCESS_QUERY_LIMITED_INFORMATION)
        if handle is None:
            return None
        try:
            if not _windows_handle_is_running(handle):
                return None
            creation_time = _windows_creation_filetime(handle)
            if creation_time is None:
                return None
            return DurableProcessIdentity(
                platform="windows",
                pid=pid,
                windows_process_creation_filetime=creation_time,
            )
        finally:
            _windows_close_handle(handle)
    if not sys.platform.startswith("linux"):
        return None
    return _read_linux_identity(pid)


def open_verified_process(identity: DurableProcessIdentity) -> ProcessVerification:
    """Return a stable handle only when the persisted process identity still matches."""

    if identity.platform == "windows":
        if sys.platform != "win32":
            return ProcessVerification("unknown", None, "platform_unavailable")
        handle, error_code = _windows_open_process(
            identity.pid,
            _PROCESS_QUERY_LIMITED_INFORMATION | _PROCESS_TERMINATE,
        )
        if handle is None:
            if error_code == _ERROR_INVALID_PARAMETER:
                return ProcessVerification("exited", None, "process_exited")
            return ProcessVerification("unknown", None, "process_query_failed")
        if not _windows_handle_is_running(handle):
            _windows_close_handle(handle)
            return ProcessVerification("exited", None, "process_exited")
        creation_time = _windows_creation_filetime(handle)
        if creation_time is None:
            _windows_close_handle(handle)
            return ProcessVerification("unknown", None, "creation_time_lookup_failed")
        if creation_time != identity.windows_process_creation_filetime:
            _windows_close_handle(handle)
            return ProcessVerification("mismatch", None, "creation_time_mismatch")
        return ProcessVerification(
            "verified",
            VerifiedProcessHandle(identity=identity, windows_handle=handle),
            "verified",
        )

    if identity.platform != "linux" or not sys.platform.startswith("linux"):
        return ProcessVerification("unknown", None, "platform_unavailable")
    current, lookup_state = _read_linux_identity_with_state(identity.pid)
    if current is None:
        return ProcessVerification(lookup_state, None, f"process_{lookup_state}")
    if (
        current.linux_boot_id != identity.linux_boot_id
        or current.linux_proc_start_time != identity.linux_proc_start_time
    ):
        return ProcessVerification("mismatch", None, "linux_identity_mismatch")
    pidfd_open = getattr(os, "pidfd_open", None)
    sender = getattr(signal, "pidfd_send_signal", None)
    if not callable(pidfd_open) or not callable(sender):
        return ProcessVerification("unknown", None, "pidfd_unavailable")
    try:
        pidfd = pidfd_open(identity.pid, 0)
    except OSError:
        return ProcessVerification("unknown", None, "pidfd_open_failed")
    current_after_pidfd, lookup_state = _read_linux_identity_with_state(identity.pid)
    if current_after_pidfd is None:
        os.close(pidfd)
        return ProcessVerification(lookup_state, None, f"process_{lookup_state}_after_pidfd")
    if (
        current_after_pidfd.linux_boot_id != identity.linux_boot_id
        or current_after_pidfd.linux_proc_start_time != identity.linux_proc_start_time
    ):
        os.close(pidfd)
        return ProcessVerification("mismatch", None, "linux_identity_mismatch_after_pidfd")
    return ProcessVerification(
        "verified",
        VerifiedProcessHandle(identity=identity, linux_pidfd=pidfd),
        "verified",
    )


def wait_for_process_exit(handle: VerifiedProcessHandle, *, timeout_seconds: float) -> bool:
    """Wait only through a verified handle; never reopen a PID after verification."""

    deadline = monotonic() + max(0.0, timeout_seconds)
    while handle.is_running() and monotonic() < deadline:
        if handle.linux_pidfd is not None:
            try:
                select.select([handle.linux_pidfd], [], [], min(0.1, deadline - monotonic()))
            except OSError:
                return False
        else:
            # Windows has no portable wait primitive in the standard library; the
            # caller only holds this verified process handle during the short poll.
            import time

            time.sleep(0.05)
    return not handle.is_running()


_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_INVALID_PARAMETER = 87


def _read_linux_identity(pid: int) -> DurableProcessIdentity | None:
    identity, _ = _read_linux_identity_with_state(pid)
    return identity


def _read_linux_identity_with_state(
    pid: int,
) -> tuple[DurableProcessIdentity | None, VerificationState]:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, "exited"
    except OSError:
        return None, "unknown"
    closing_parenthesis = stat_text.rfind(")")
    if closing_parenthesis < 0:
        return None, "unknown"
    fields_after_comm = stat_text[closing_parenthesis + 2 :].split()
    if len(fields_after_comm) <= 19 or fields_after_comm[0] == "Z" or not boot_id:
        return None, "exited" if fields_after_comm and fields_after_comm[0] == "Z" else "unknown"
    try:
        start_time = int(fields_after_comm[19])
    except ValueError:
        return None, "unknown"
    return (
        DurableProcessIdentity(
            platform="linux",
            pid=pid,
            linux_boot_id=boot_id,
            linux_proc_start_time=start_time,
        ),
        "verified",
    )


def _windows_open_process(pid: int, access: int) -> tuple[int | None, int]:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    ctypes.set_last_error(0)
    handle = kernel32.OpenProcess(access, False, pid)
    return (int(handle), 0) if handle else (None, ctypes.get_last_error())


def _windows_close_handle(handle: int) -> None:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(handle)


def _windows_handle_is_running(handle: int) -> bool:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    exit_code = wintypes.DWORD()
    if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
        return False
    return exit_code.value == _STILL_ACTIVE


def _windows_creation_filetime(handle: int) -> int | None:
    from ctypes import wintypes

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    created = FILETIME()
    exited = FILETIME()
    kernel_time = FILETIME()
    user_time = FILETIME()
    if not kernel32.GetProcessTimes(
        handle,
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(kernel_time),
        ctypes.byref(user_time),
    ):
        return None
    return (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)


def _windows_terminate_handle(handle: int) -> bool:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateProcess.restype = wintypes.BOOL
    return bool(kernel32.TerminateProcess(handle, 1))
