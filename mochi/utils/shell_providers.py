"""Shell provider abstraction for exec runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SubprocessSpec:
    """描述 subprocess 啟動參數。"""

    executable: str
    args: tuple[str, ...]

    @property
    def argv(self) -> tuple[str, ...]:
        """回傳完整 argv（含 executable）。"""
        return (self.executable, *self.args)


class BaseShellProvider(ABC):
    """Shell 命令包裝抽象層。"""

    @property
    @abstractmethod
    def canonical_name(self) -> str:
        """Provider 的標準 shell 名稱。"""
        ...

    @property
    @abstractmethod
    def aliases(self) -> tuple[str, ...]:
        """可接受的 shell 別名。"""
        ...

    @abstractmethod
    def build_subprocess_spec(self, command: str, *, tty: bool = False) -> SubprocessSpec:
        """將命令包裝為 subprocess 啟動參數。"""
        ...

    def supports_shell(self, shell: str) -> bool:
        """判斷是否支援指定 shell 字串。"""
        normalized = shell.strip().lower()
        return normalized in self.aliases


class PowerShellProvider(BaseShellProvider):
    """PowerShell / pwsh 命令包裝。"""

    def __init__(self, *, executable: str = "pwsh") -> None:
        self._executable = executable

    @property
    def canonical_name(self) -> str:
        return "powershell"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("powershell", "pwsh")

    def build_subprocess_spec(self, command: str, *, tty: bool = False) -> SubprocessSpec:
        args = ["-NoLogo", "-NoProfile"]
        if not tty:
            args.append("-NonInteractive")
        args.extend(["-Command", command])
        return SubprocessSpec(executable=self._executable, args=tuple(args))


class BashProvider(BaseShellProvider):
    """Bash / sh 命令包裝。"""

    def __init__(self, *, executable: str = "bash") -> None:
        self._executable = executable

    @property
    def canonical_name(self) -> str:
        return "bash"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("bash", "sh")

    def build_subprocess_spec(self, command: str, *, tty: bool = False) -> SubprocessSpec:
        flag = "-ic" if tty else "-lc"
        return SubprocessSpec(executable=self._executable, args=(flag, command))


class CmdProvider(BaseShellProvider):
    """Windows cmd.exe 命令包裝。"""

    def __init__(self, *, executable: str = "cmd.exe") -> None:
        self._executable = executable

    @property
    def canonical_name(self) -> str:
        return "cmd"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("cmd", "cmd.exe")

    def build_subprocess_spec(self, command: str, *, tty: bool = False) -> SubprocessSpec:
        del tty
        return SubprocessSpec(
            executable=self._executable,
            args=("/d", "/s", "/c", command),
        )


def default_shell_providers() -> dict[str, BaseShellProvider]:
    """建立預設 shell provider 集合。"""

    providers: list[BaseShellProvider] = [
        PowerShellProvider(),
        BashProvider(),
        CmdProvider(),
    ]
    mapping: dict[str, BaseShellProvider] = {}
    for provider in providers:
        for alias in provider.aliases:
            mapping[alias] = provider
    return mapping
