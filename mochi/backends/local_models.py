"""本地模型候選掃描工具。"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Literal
import zipfile

import httpx
try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LocalModelCandidate:
    """本地模型候選項。"""

    model_spec: str
    model: str
    backend_type: str
    metadata: dict[str, int | str] = field(default_factory=dict)


@dataclass(slots=True)
class LocalModelDiscoveryResult:
    """本地模型掃描結果。"""

    root: str
    models: list[LocalModelCandidate]
    warnings: list[str]


@dataclass(slots=True)
class QuantizationOption:
    """量化選項描述。"""

    id: str
    name: str
    bits: str
    description: str


@dataclass(slots=True)
class QuantizationFormatCapability:
    """單一格式的量化能力描述。"""

    format_id: str
    format_name: str
    supported: bool
    priority: Literal["primary", "secondary"]
    reason: str
    warnings: list[str] = field(default_factory=list)
    quantization_options: list[QuantizationOption] = field(default_factory=list)
    suggested_default_quantization: str | None = None


@dataclass(slots=True)
class HardwareSummary:
    """本地硬體摘要（可選）。"""

    provider: str
    cuda_available: bool
    gpu_count: int
    gpu_vendor: str | None = None
    primary_gpu_name: str | None = None
    total_vram_gb: float | None = None
    recommended_runtime_backend: str | None = None
    recommended_runtime_label: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LocalModelQuantizationCapabilitiesResult:
    """本地 HF 模型量化能力探測結果。"""

    model_dir: str
    model_family: str | None
    formats: list[QuantizationFormatCapability]
    warnings: list[str] = field(default_factory=list)
    hardware: HardwareSummary | None = None


@dataclass(slots=True)
class LocalModelConvertRequest:
    """本地模型轉換請求（bounded phase 1）。"""

    source_model_dir: str
    target_format: str
    quantization: str
    persist: bool = True


@dataclass(slots=True)
class LocalModelConvertExecutionResult:
    """本地模型轉換執行結果（不含 config 持久化狀態）。"""

    target_format: str
    quantization: str
    source_model_dir: str
    output_model_path: str
    converted: bool
    message: str


@dataclass(slots=True)
class LlamaCppToolchain:
    """llama.cpp 轉換工具鏈定位結果。"""

    python_executable: str
    convert_script: Path | None
    quantize_binary: Path | None
    root_dir: Path | None = None
    version: str | None = None
    source: Literal["env", "managed", "existing_path", "auto"] = "auto"
    search_roots: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ManagedLlamaCppRuntimeStatus:
    """managed llama.cpp runtime 狀態摘要。"""

    state: Literal["ready", "degraded", "missing"]
    installed: bool
    source: Literal["managed", "existing_path", "env", "auto", "unknown"]
    root_dir: str | None
    install_dir: str | None
    python_executable: str
    version: str | None
    convert_script: str | None
    quantize_binary: str | None
    platform: str | None = None
    binary_asset: str | None = None
    missing_components: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ManagedLlamaCppPlatformTarget:
    """managed llama.cpp 安裝目標平台。"""

    platform_id: str
    system: str
    machine: str
    asset_include_tokens: tuple[str, ...]
    asset_exclude_tokens: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class ManagedLlamaCppReleaseAsset:
    """單一 llama.cpp release asset 摘要。"""

    name: str
    download_url: str
    size_bytes: int | None = None
    content_type: str | None = None


@dataclass(slots=True)
class ManagedLlamaCppReleaseMetadata:
    """managed installer 需要的 release metadata。"""

    version: str
    platform: ManagedLlamaCppPlatformTarget
    source_archive_url: str
    source_archive_name: str
    binary_asset: ManagedLlamaCppReleaseAsset


@dataclass(slots=True)
class ManagedLlamaCppInstallPlan:
    """保守 managed installer contract。"""

    install_dir: str
    state: Literal["manual_download_required", "network_unavailable", "ready"]
    source: Literal["managed", "existing_path"]
    action: Literal["prepare", "register_existing"]
    version: str | None
    message: str
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ManagedLlamaCppInstallResult:
    """managed llama.cpp runtime 安裝結果。"""

    install_dir: str
    state: Literal["installed", "already_installed"]
    source: Literal["managed"]
    action: Literal["install"]
    version: str
    root_dir: str
    python_executable: str
    warnings: list[str] = field(default_factory=list)
    message: str = ""


class LocalModelConversionError(RuntimeError):
    """本地模型轉換統一錯誤基類。"""

    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)


class LocalModelConversionValidationError(LocalModelConversionError):
    """轉換請求參數/輸入資料驗證錯誤。"""

    status_code = 400


class LocalModelConversionNotImplementedError(LocalModelConversionError):
    """尚未實作的轉換格式錯誤。"""

    status_code = 501


class LocalModelConversionRuntimeUnavailableError(LocalModelConversionError):
    """轉換 runtime 尚不可用（phase 1 placeholder）。"""

    status_code = 503


class ManagedLlamaCppInstallError(RuntimeError):
    """managed llama.cpp 安裝失敗。"""

    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ManagedLlamaCppInstallValidationError(ManagedLlamaCppInstallError):
    """managed installer 參數/平台不支援。"""

    status_code = 400


class ManagedLlamaCppInstallNetworkError(ManagedLlamaCppInstallError):
    """managed installer 網路或遠端服務不可用。"""

    status_code = 503


DEFAULT_MANAGED_LLAMA_CPP_VERSION = "b9058"
LLAMA_CPP_GITHUB_RELEASE_API_TEMPLATE = (
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{version}"
)
LLAMA_CPP_SOURCE_ARCHIVE_TEMPLATE = (
    "https://github.com/ggml-org/llama.cpp/archive/refs/tags/{version}.tar.gz"
)
LLAMA_CPP_GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "Mochi/managed-llama-cpp-installer",
}
LLAMA_CPP_GPU_VARIANT_TOKENS = (
    "cuda",
    "vulkan",
    "rocm",
    "openvino",
    "sycl",
    "hip",
    "metal",
)


class BaseLocalModelConverter(ABC):
    """本地模型轉換器抽象介面。"""

    @abstractmethod
    async def convert(self, request: LocalModelConvertRequest) -> LocalModelConvertExecutionResult:
        """執行模型轉換。"""


class PlaceholderLocalModelConverter(BaseLocalModelConverter):
    """bounded 本地模型轉換骨架（預設 runtime unavailable）。"""

    def __init__(self, *, runtime_available: bool = False) -> None:
        self._runtime_available = runtime_available

    async def convert(self, request: LocalModelConvertRequest) -> LocalModelConvertExecutionResult:
        source_model_dir = request.source_model_dir.strip()
        target_format = request.target_format.strip().lower()
        raw_quantization = request.quantization.strip()

        if target_format != "gguf":
            raise LocalModelConversionValidationError(
                f"Unsupported target_format '{request.target_format}'. Only 'gguf' is supported."
            )

        quantization = raw_quantization.upper()
        source_path = _validate_hf_source_model_dir(source_model_dir)
        _validate_gguf_quantization_option(quantization)
        output_path = build_gguf_output_model_path(source_path, quantization)

        if not self._runtime_available:
            raise LocalModelConversionRuntimeUnavailableError(
                "GGUF llama.cpp tools/runtime is unavailable in phase 1 placeholder. "
                "Inject a converter via app.state.local_model_converter(_factory) to enable execution."
            )

        return LocalModelConvertExecutionResult(
            target_format="gguf",
            quantization=quantization,
            source_model_dir=str(source_path),
            output_model_path=str(output_path),
            converted=True,
            message=f"Converted HF model directory to GGUF: {output_path}",
        )


class LlamaCppLocalModelConverter(BaseLocalModelConverter):
    """以 llama.cpp 工具鏈執行 HF -> GGUF 轉換。"""

    def __init__(
        self,
        *,
        toolchain: LlamaCppToolchain | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self._toolchain = toolchain
        self._env = dict(env or os.environ)
        self._cwd = Path(cwd).expanduser().resolve(strict=False) if cwd is not None else Path.cwd()

    async def convert(self, request: LocalModelConvertRequest) -> LocalModelConvertExecutionResult:
        source_model_dir = request.source_model_dir.strip()
        target_format = request.target_format.strip().lower()
        raw_quantization = request.quantization.strip()
        logger.info(
            "Starting local model conversion: source={} target={} quantization={}",
            source_model_dir,
            target_format,
            raw_quantization,
        )

        if target_format != "gguf":
            raise LocalModelConversionValidationError(
                f"Unsupported target_format '{request.target_format}'. Only 'gguf' is supported."
            )

        quantization = raw_quantization.upper()
        source_path = _validate_hf_source_model_dir(source_model_dir)
        _validate_gguf_quantization_option(quantization)

        toolchain = self._toolchain or discover_llama_cpp_toolchain(
            env=self._env,
            cwd=self._cwd,
        )
        _ensure_llama_cpp_runtime_available(toolchain, quantization)

        output_path = build_gguf_output_model_path(source_path, quantization)

        if _is_direct_gguf_outtype(quantization):
            await self._run_convert_command(
                toolchain=toolchain,
                source_model_dir=source_path,
                output_model_path=output_path,
                outtype=_convert_outtype_for_quantization(quantization),
            )
        else:
            intermediate_path = build_gguf_output_model_path(source_path, "F16")
            await self._run_convert_command(
                toolchain=toolchain,
                source_model_dir=source_path,
                output_model_path=intermediate_path,
                outtype="f16",
            )
            await self._run_quantize_command(
                toolchain=toolchain,
                input_model_path=intermediate_path,
                output_model_path=output_path,
                quantization=quantization,
            )

        if not output_path.is_file():
            raise LocalModelConversionError(
                f"GGUF conversion finished but output file is missing: {output_path}"
            )

        logger.info(
            "Completed local model conversion: source={} target=gguf quantization={} output={}",
            source_path,
            quantization,
            output_path,
        )
        return LocalModelConvertExecutionResult(
            target_format="gguf",
            quantization=quantization,
            source_model_dir=str(source_path),
            output_model_path=str(output_path),
            converted=True,
            message=f"Converted HF model directory to GGUF: {output_path}",
        )

    async def _run_convert_command(
        self,
        *,
        toolchain: LlamaCppToolchain,
        source_model_dir: Path,
        output_model_path: Path,
        outtype: str,
    ) -> None:
        command = build_llama_cpp_convert_command(
            toolchain=toolchain,
            source_model_dir=source_model_dir,
            output_model_path=output_model_path,
            outtype=outtype,
        )
        await self._run_command(command)

    async def _run_quantize_command(
        self,
        *,
        toolchain: LlamaCppToolchain,
        input_model_path: Path,
        output_model_path: Path,
        quantization: str,
    ) -> None:
        command = build_llama_cpp_quantize_command(
            toolchain=toolchain,
            input_model_path=input_model_path,
            output_model_path=output_model_path,
            quantization=quantization,
        )
        await self._run_command(command)

    async def _run_command(self, command: Sequence[str]) -> None:
        logger.info("Running local model conversion command: {}", shlex.join(command))
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(self._cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._env,
        )

        stdout_task = asyncio.create_task(
            _drain_process_stream(
                process.stdout,
                prefix="local-model-convert stdout",
            )
        )
        stderr_task = asyncio.create_task(
            _drain_process_stream(
                process.stderr,
                prefix="local-model-convert stderr",
            )
        )
        await process.wait()
        stdout_text, stderr_text = await asyncio.gather(stdout_task, stderr_task)
        if process.returncode == 0:
            logger.info("Local model conversion command completed: {}", shlex.join(command))
            return

        detail_parts = [f"command exited with code {process.returncode}"]
        if stderr_text:
            detail_parts.append(f"stderr: {stderr_text[-1200:]}")
        if stdout_text:
            detail_parts.append(f"stdout: {stdout_text[-1200:]}")
        raise LocalModelConversionError("GGUF conversion command failed: " + " | ".join(detail_parts))


async def _drain_process_stream(
    stream: asyncio.StreamReader | None,
    *,
    prefix: str,
    tail_lines: int = 120,
) -> str:
    """即時轉發子程序輸出到 backend log，並保留尾端供失敗訊息使用。"""
    if stream is None:
        return ""

    tail: list[str] = []
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        if not text:
            continue
        logger.debug("{}: {}", prefix, text)
        tail.append(text)
        if len(tail) > tail_lines:
            del tail[0]
    return "\n".join(tail)


def discover_local_models(
    root: str | Path,
    *,
    max_depth: int = 3,
    max_entries: int = 500,
) -> LocalModelDiscoveryResult:
    """掃描 root 底下可用本地模型候選（bounded、不可跟隨 symlink）。"""
    raw_root_path = Path(root).expanduser()
    if raw_root_path.exists() and raw_root_path.is_symlink():
        raise ValueError(f"Symlink root paths are not supported: {raw_root_path}")
    root_path = raw_root_path.resolve(strict=False)
    if not root_path.exists():
        raise FileNotFoundError(f"Root path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_path}")
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0.")
    if max_entries <= 0:
        raise ValueError("max_entries must be > 0.")

    entries_checked = 0
    scan_truncated = False
    warnings: list[str] = []
    models: list[LocalModelCandidate] = []
    queue: list[tuple[Path, int]] = [(root_path, 0)]
    visited_dirs: set[Path] = set()

    while queue and entries_checked < max_entries:
        directory, depth = queue.pop(0)
        if directory in visited_dirs:
            continue
        visited_dirs.add(directory)

        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name.lower())
        except PermissionError:
            warnings.append(f"Permission denied while scanning: {directory}")
            continue

        for child in children:
            if entries_checked >= max_entries:
                scan_truncated = True
                break
            entries_checked += 1

            # 不跟隨 symlink，避免跳離 root 或形成循環。
            if child.is_symlink():
                continue

            if child.is_file() and child.suffix.lower() == ".gguf":
                models.append(_build_gguf_candidate(child))
                continue

            if child.is_dir():
                hf_candidate = _build_hf_candidate(child)
                if hf_candidate is not None:
                    models.append(hf_candidate)
                    continue
                if depth < max_depth:
                    queue.append((child, depth + 1))

    if scan_truncated or (entries_checked >= max_entries and queue):
        warnings.append(
            f"Scan stopped after reaching max_entries={max_entries}. "
            "Increase limit to discover more models."
        )

    return LocalModelDiscoveryResult(
        root=str(root_path),
        models=models,
        warnings=warnings,
    )


def discover_hf_quantization_capabilities(
    model_dir: str | Path,
    *,
    include_hardware: bool = True,
) -> LocalModelQuantizationCapabilitiesResult:
    """探測本地 HF 目錄可用的量化能力（只回傳 capability，不做轉換）。"""
    raw_path = Path(model_dir).expanduser()
    if raw_path.exists() and raw_path.is_symlink():
        raise ValueError(f"Symlink model paths are not supported: {raw_path}")
    path = raw_path.resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(f"Local model path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Local model path is not a directory: {path}")

    missing = _hf_layout_missing_requirements(path)
    if missing:
        detail = ", ".join(missing)
        raise ValueError(
            "Local model directory is not a valid HuggingFace safetensors directory. "
            f"Missing: {detail}"
        )

    family = _read_hf_model_family(path)
    hardware = _detect_hardware_summary() if include_hardware else None
    warnings = [
        "Capability probe reflects local runtime/tool availability at request time.",
    ]
    if family is None:
        warnings.append("config.json has no model_type; family allowlist checks are deferred.")

    return LocalModelQuantizationCapabilitiesResult(
        model_dir=str(path),
        model_family=family,
        formats=[_build_gguf_format_capability(hardware)],
        warnings=warnings,
        hardware=hardware,
    )


def _build_gguf_candidate(path: Path) -> LocalModelCandidate:
    """建立 GGUF 候選項。"""
    size_bytes = 0
    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0
    return LocalModelCandidate(
        model_spec=str(path),
        model=path.name,
        backend_type="gguf",
        metadata={
            "path": str(path),
            "size_bytes": size_bytes,
        },
    )


def _build_hf_candidate(path: Path) -> LocalModelCandidate | None:
    """判斷並建立 HuggingFace safetensors 目錄候選項。"""
    if _hf_layout_missing_requirements(path):
        return None

    file_count = 0
    size_bytes = 0
    try:
        for file_path in path.iterdir():
            if file_path.is_symlink() or not file_path.is_file():
                continue
            file_count += 1
            try:
                size_bytes += file_path.stat().st_size
            except OSError:
                pass
    except PermissionError:
        return None

    return LocalModelCandidate(
        model_spec=str(path),
        model=path.name,
        backend_type="safetensors",
        metadata={
            "path": str(path),
            "size_bytes": size_bytes,
            "file_count": file_count,
        },
    )


def build_gguf_output_model_path(
    source_model_dir: str | Path,
    quantization: str,
) -> Path:
    """建立 GGUF 輸出路徑（deterministic naming）。"""
    source_path = Path(source_model_dir).expanduser().resolve(strict=False)
    quant = quantization.strip().upper()
    return source_path.parent / f"{source_path.name}-{quant}.gguf"


def discover_llama_cpp_toolchain(
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    managed_root: str | Path | None = None,
    preferred_root: str | Path | None = None,
    preferred_python: str | None = None,
    preferred_version: str | None = None,
    preferred_source: Literal["managed", "existing_path"] | None = None,
) -> LlamaCppToolchain:
    """保守地定位 llama.cpp 轉換工具鏈。"""
    effective_env = env or os.environ
    search_roots = _candidate_llama_cpp_search_roots(
        env=effective_env,
        cwd=cwd,
        managed_root=managed_root,
        preferred_root=preferred_root,
        preferred_version=preferred_version,
        preferred_source=preferred_source,
    )
    python_executable = (
        preferred_python
        or effective_env.get("MOCHI_LLAMA_CPP_PYTHON")
        or sys.executable
        or shutil.which("python3")
        or shutil.which("python")
        or "python3"
    )
    root_dir = _infer_llama_cpp_root_dir(
        search_roots=search_roots,
        env=effective_env,
        preferred_root=preferred_root,
    )
    source: Literal["env", "managed", "existing_path", "auto"] = "auto"
    if preferred_source is not None:
        source = preferred_source
    elif _env_declares_llama_cpp(effective_env):
        source = "env"
    return LlamaCppToolchain(
        python_executable=python_executable,
        convert_script=_resolve_llama_cpp_convert_script(search_roots=search_roots, env=effective_env),
        quantize_binary=_resolve_llama_cpp_quantize_binary(search_roots=search_roots, env=effective_env),
        root_dir=root_dir,
        version=preferred_version,
        source=source,
        search_roots=[str(path) for path in search_roots],
    )


def build_llama_cpp_convert_command(
    *,
    toolchain: LlamaCppToolchain,
    source_model_dir: str | Path,
    output_model_path: str | Path,
    outtype: str,
) -> list[str]:
    """建立 llama.cpp `convert_hf_to_gguf.py` 命令列。"""
    if toolchain.convert_script is None:
        raise LocalModelConversionRuntimeUnavailableError(
            "GGUF llama.cpp tools/runtime is unavailable: missing convert_hf_to_gguf.py."
        )
    source_path = Path(source_model_dir).expanduser().resolve(strict=False)
    output_path = Path(output_model_path).expanduser().resolve(strict=False)
    return [
        toolchain.python_executable,
        str(toolchain.convert_script),
        str(source_path),
        "--outfile",
        str(output_path),
        "--outtype",
        outtype.strip().lower(),
    ]


def build_llama_cpp_quantize_command(
    *,
    toolchain: LlamaCppToolchain,
    input_model_path: str | Path,
    output_model_path: str | Path,
    quantization: str,
    threads: int | None = None,
) -> list[str]:
    """建立 llama.cpp `llama-quantize` 命令列。"""
    if toolchain.quantize_binary is None:
        raise LocalModelConversionRuntimeUnavailableError(
            "GGUF llama.cpp tools/runtime is unavailable: missing llama-quantize."
        )
    input_path = Path(input_model_path).expanduser().resolve(strict=False)
    output_path = Path(output_model_path).expanduser().resolve(strict=False)
    command = [
        str(toolchain.quantize_binary),
        str(input_path),
        str(output_path),
        quantization.strip().upper(),
    ]
    if threads is not None and threads > 0:
        command.append(str(threads))
    return command


def _validate_hf_source_model_dir(source_model_dir: str | Path) -> Path:
    """驗證 source 是否為可轉換的 HF safetensors 目錄。"""
    raw_path = Path(source_model_dir).expanduser()
    if raw_path.exists() and raw_path.is_symlink():
        raise LocalModelConversionValidationError("Symlink model paths are not supported.")
    path = raw_path.resolve(strict=False)
    if not path.exists():
        raise LocalModelConversionValidationError(f"Local model path does not exist: {path}")
    if not path.is_dir():
        raise LocalModelConversionValidationError(
            "GGUF conversion currently supports local HuggingFace model directories only."
        )
    missing = _hf_layout_missing_requirements(path)
    if missing:
        detail = ", ".join(missing)
        raise LocalModelConversionValidationError(
            "Local model directory is not a valid HuggingFace safetensors directory. "
            f"Missing: {detail}"
        )
    return path


def _candidate_llama_cpp_search_roots(
    *,
    env: Mapping[str, str],
    cwd: str | Path | None,
    managed_root: str | Path | None = None,
    preferred_root: str | Path | None = None,
    preferred_version: str | None = None,
    preferred_source: Literal["managed", "existing_path"] | None = None,
) -> list[Path]:
    """建立 llama.cpp 常見搜尋根目錄。"""
    candidates: list[Path] = []
    raw_cwd = Path(cwd).expanduser().resolve(strict=False) if cwd is not None else Path.cwd()
    if preferred_root is not None:
        candidates.append(Path(preferred_root).expanduser().resolve(strict=False))
    if managed_root is not None and preferred_source == "managed":
        candidates.append(
            get_managed_llama_cpp_install_dir(
                managed_root,
                version=preferred_version,
            ).resolve(strict=False)
        )
    for key in ("MOCHI_LLAMA_CPP_DIR", "LLAMA_CPP_DIR"):
        raw_value = env.get(key, "").strip()
        if raw_value:
            candidates.append(Path(raw_value).expanduser().resolve(strict=False))
    candidates.extend(
        [
            raw_cwd,
            raw_cwd / "llama.cpp",
            raw_cwd / "vendor" / "llama.cpp",
            raw_cwd / "third_party" / "llama.cpp",
            raw_cwd / "extern" / "llama.cpp",
            raw_cwd.parent / "llama.cpp",
        ]
    )
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _env_declares_llama_cpp(env: Mapping[str, str]) -> bool:
    """判斷 env 是否明確宣告 llama.cpp 位置。"""
    for key in (
        "MOCHI_LLAMA_CPP_DIR",
        "LLAMA_CPP_DIR",
        "MOCHI_LLAMA_CPP_CONVERT_SCRIPT",
        "LLAMA_CPP_CONVERT_SCRIPT",
        "MOCHI_LLAMA_CPP_QUANTIZE_BIN",
        "LLAMA_CPP_QUANTIZE_BIN",
    ):
        if env.get(key, "").strip():
            return True
    return False


def _infer_llama_cpp_root_dir(
    *,
    search_roots: Sequence[Path],
    env: Mapping[str, str],
    preferred_root: str | Path | None,
) -> Path | None:
    """推測目前最合理的 llama.cpp 根目錄。"""
    if preferred_root is not None:
        return Path(preferred_root).expanduser().resolve(strict=False)
    for key in ("MOCHI_LLAMA_CPP_DIR", "LLAMA_CPP_DIR"):
        raw_value = env.get(key, "").strip()
        if raw_value:
            return Path(raw_value).expanduser().resolve(strict=False)
    for root in search_roots:
        if _looks_like_llama_cpp_root(root):
            return root.resolve(strict=False)
    return None


def _looks_like_llama_cpp_root(root: Path) -> bool:
    """Detect whether a directory looks like a usable llama.cpp runtime root."""
    candidates = (
        root / "convert_hf_to_gguf.py",
        root / "examples" / "convert_hf_to_gguf.py",
        root / "llama-server",
        root / "llama-server.exe",
        root / "llama-quantize",
        root / "llama-quantize.exe",
        root / "build" / "bin" / "llama-server",
        root / "build" / "bin" / "llama-server.exe",
        root / "build" / "bin" / "llama-quantize",
        root / "build" / "bin" / "llama-quantize.exe",
    )
    return any(candidate.is_file() for candidate in candidates)


def _resolve_llama_cpp_convert_script(
    *,
    search_roots: Sequence[Path],
    env: Mapping[str, str],
) -> Path | None:
    """定位 `convert_hf_to_gguf.py`。"""
    for key in ("MOCHI_LLAMA_CPP_CONVERT_SCRIPT", "LLAMA_CPP_CONVERT_SCRIPT"):
        raw_value = env.get(key, "").strip()
        if not raw_value:
            continue
        candidate = Path(raw_value).expanduser().resolve(strict=False)
        if candidate.is_file():
            return candidate
    for root in search_roots:
        for candidate in (
            root / "convert_hf_to_gguf.py",
            root / "examples" / "convert_hf_to_gguf.py",
        ):
            if candidate.is_file():
                return candidate
    return None


def _resolve_llama_cpp_quantize_binary(
    *,
    search_roots: Sequence[Path],
    env: Mapping[str, str],
) -> Path | None:
    """定位 `llama-quantize`。"""
    for key in ("MOCHI_LLAMA_CPP_QUANTIZE_BIN", "LLAMA_CPP_QUANTIZE_BIN"):
        raw_value = env.get(key, "").strip()
        if not raw_value:
            continue
        candidate = Path(raw_value).expanduser().resolve(strict=False)
        if candidate.is_file():
            return candidate

    for program in ("llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"):
        resolved = shutil.which(program)
        if resolved:
            return Path(resolved).expanduser().resolve(strict=False)

    binary_names = ("llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe")
    relative_roots = (
        (),
        ("build", "bin"),
        ("build", "bin", "Release"),
        ("build", "bin", "Debug"),
        ("bin",),
    )
    for root in search_roots:
        for relative_root in relative_roots:
            base = root.joinpath(*relative_root) if relative_root else root
            for binary_name in binary_names:
                candidate = base / binary_name
                if candidate.is_file():
                    return candidate
    return None


def _ensure_llama_cpp_runtime_available(
    toolchain: LlamaCppToolchain,
    quantization: str,
) -> None:
    """確認目前請求所需的 llama.cpp 工具鏈可用。"""
    missing: list[str] = []
    if toolchain.convert_script is None:
        missing.append("convert_hf_to_gguf.py")
    if not _is_direct_gguf_outtype(quantization) and toolchain.quantize_binary is None:
        missing.append("llama-quantize")
    if not missing:
        return

    hints = [
        "Build or clone llama.cpp and expose it with MOCHI_LLAMA_CPP_DIR, "
        "or set MOCHI_LLAMA_CPP_CONVERT_SCRIPT / MOCHI_LLAMA_CPP_QUANTIZE_BIN directly."
    ]
    if toolchain.search_roots:
        hints.append("Searched: " + ", ".join(toolchain.search_roots))
    raise LocalModelConversionRuntimeUnavailableError(
        "GGUF llama.cpp tools/runtime is unavailable: missing "
        + ", ".join(missing)
        + ". "
        + " ".join(hints)
    )


def get_managed_llama_cpp_install_dir(
    base_dir: str | Path,
    *,
    version: str | None = None,
) -> Path:
    """回傳 managed llama.cpp install 預設目錄。"""
    root = Path(base_dir).expanduser().resolve(strict=False)
    tag = (version or "current").strip() or "current"
    return root / "runtimes" / "llama.cpp" / tag


def get_managed_llama_cpp_runtime_status(
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    managed_root: str | Path | None = None,
    preferred_root: str | Path | None = None,
    preferred_python: str | None = None,
    preferred_version: str | None = None,
    preferred_source: Literal["managed", "existing_path"] | None = None,
) -> ManagedLlamaCppRuntimeStatus:
    """彙整 llama.cpp runtime readiness。"""
    effective_env = env or os.environ
    effective_version = (preferred_version or "").strip() or DEFAULT_MANAGED_LLAMA_CPP_VERSION
    install_dir = (
        str(get_managed_llama_cpp_install_dir(managed_root, version=effective_version))
        if managed_root is not None
        else None
    )
    toolchain = discover_llama_cpp_toolchain(
        env=effective_env,
        cwd=cwd,
        managed_root=managed_root,
        preferred_root=preferred_root,
        preferred_python=preferred_python,
        preferred_version=effective_version,
        preferred_source=preferred_source,
    )

    missing_components: list[str] = []
    if toolchain.convert_script is None:
        missing_components.append("convert_hf_to_gguf.py")
    if toolchain.quantize_binary is None:
        missing_components.append("llama-quantize")

    warnings: list[str] = []
    actions: list[str] = []
    platform_name: str | None = None
    binary_asset: str | None = None
    metadata_root = toolchain.root_dir
    if metadata_root is not None:
        metadata_path = metadata_root / ".mochi-managed-runtime.json"
        if metadata_path.is_file():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                payload = {}
            if isinstance(payload, dict):
                raw_platform = payload.get("platform")
                raw_binary_asset = payload.get("binary_asset")
                platform_name = raw_platform.strip() if isinstance(raw_platform, str) and raw_platform.strip() else None
                binary_asset = (
                    raw_binary_asset.strip()
                    if isinstance(raw_binary_asset, str) and raw_binary_asset.strip()
                    else None
                )
    if missing_components:
        actions.append("register_existing_path")
        actions.append("prepare_managed_runtime")
        if toolchain.source in {"managed", "existing_path"} and toolchain.root_dir is not None:
            warnings.append(
                "Registered llama.cpp runtime is incomplete; required converter components are missing."
            )
        else:
            warnings.append("No complete llama.cpp tools/runtime installation has been discovered yet.")
    else:
        actions.append("ready_for_conversion")

    state: Literal["ready", "degraded", "missing"]
    if not missing_components:
        state = "ready"
    elif toolchain.root_dir is not None or toolchain.source in {"managed", "existing_path", "env"}:
        state = "degraded"
    else:
        state = "missing"

    source: Literal["managed", "existing_path", "env", "auto", "unknown"]
    if toolchain.source in {"managed", "existing_path", "env", "auto"}:
        source = toolchain.source
    else:
        source = "unknown"

    return ManagedLlamaCppRuntimeStatus(
        state=state,
        installed=state != "missing",
        source=source,
        root_dir=str(toolchain.root_dir) if toolchain.root_dir is not None else None,
        install_dir=install_dir,
        python_executable=toolchain.python_executable,
        version=toolchain.version,
        platform=platform_name,
        binary_asset=binary_asset,
        convert_script=str(toolchain.convert_script) if toolchain.convert_script is not None else None,
        quantize_binary=str(toolchain.quantize_binary) if toolchain.quantize_binary is not None else None,
        missing_components=missing_components,
        warnings=warnings,
        actions=actions,
    )


def prepare_managed_llama_cpp_install_plan(
    *,
    managed_root: str | Path,
    version: str | None = None,
    network_available: bool = False,
) -> ManagedLlamaCppInstallPlan:
    """建立 bounded managed install plan，不直接下載。"""
    effective_version = (version or "").strip() or DEFAULT_MANAGED_LLAMA_CPP_VERSION
    install_dir = get_managed_llama_cpp_install_dir(managed_root, version=effective_version)
    if network_available:
        state: Literal["manual_download_required", "network_unavailable", "ready"] = "ready"
        message = "Managed llama.cpp runtime install can proceed automatically."
        warnings = []
    else:
        state = "network_unavailable"
        message = (
            "Managed llama.cpp runtime directory prepared, but network-assisted installation is unavailable. "
            "Manual download/build is required."
        )
        warnings = [
            "No network dependency is used here; register an existing llama.cpp path or populate the managed directory manually.",
        ]

    install_dir.mkdir(parents=True, exist_ok=True)
    return ManagedLlamaCppInstallPlan(
        install_dir=str(install_dir),
        state=state,
        source="managed",
        action="prepare",
        version=effective_version,
        message=message,
        warnings=warnings,
    )


def detect_managed_llama_cpp_platform_target(
    *,
    system: str | None = None,
    machine: str | None = None,
    hardware: HardwareSummary | None = None,
) -> ManagedLlamaCppPlatformTarget:
    """將目前平台映射到預設可下載的 llama.cpp CPU release asset。"""
    resolved_system = (system or platform.system()).strip().lower()
    resolved_machine = (machine or platform.machine()).strip().lower()
    resolved_gpu_vendor = (
        (hardware.gpu_vendor or "").strip().lower()
        if hardware is not None and hardware.gpu_vendor
        else _infer_gpu_vendor(hardware.primary_gpu_name if hardware is not None else None) or ""
    )

    if resolved_system == "windows":
        if resolved_machine in {"amd64", "x86_64", "x64"}:
            if hardware is not None and hardware.cuda_available:
                return ManagedLlamaCppPlatformTarget(
                    platform_id="windows-x64-cuda",
                    system="windows",
                    machine="x86_64",
                    asset_include_tokens=("win", "x64", "cuda"),
                    asset_exclude_tokens=("arm64",),
                )
            if resolved_gpu_vendor == "amd":
                return ManagedLlamaCppPlatformTarget(
                    platform_id="windows-x64-hip",
                    system="windows",
                    machine="x86_64",
                    asset_include_tokens=("win", "x64", "hip"),
                    asset_exclude_tokens=("arm64",),
                )
            if resolved_gpu_vendor == "intel":
                return ManagedLlamaCppPlatformTarget(
                    platform_id="windows-x64-vulkan",
                    system="windows",
                    machine="x86_64",
                    asset_include_tokens=("win", "x64", "vulkan"),
                    asset_exclude_tokens=("arm64",),
                )
            return ManagedLlamaCppPlatformTarget(
                platform_id="windows-x64-cpu",
                system="windows",
                machine="x86_64",
                asset_include_tokens=("win", "x64"),
                asset_exclude_tokens=("arm64", "cuda", "vulkan", "sycl", "hip"),
            )
        if resolved_machine in {"arm64", "aarch64"}:
            return ManagedLlamaCppPlatformTarget(
                platform_id="windows-arm64-cpu",
                system="windows",
                machine="arm64",
                asset_include_tokens=("win", "arm64"),
            )
    elif resolved_system == "linux":
        if resolved_machine in {"x86_64", "amd64"}:
            if resolved_gpu_vendor == "amd":
                return ManagedLlamaCppPlatformTarget(
                    platform_id="linux-x64-rocm",
                    system="linux",
                    machine="x86_64",
                    asset_include_tokens=("ubuntu", "x64", "rocm"),
                    asset_exclude_tokens=("arm64", "s390x"),
                )
            if resolved_gpu_vendor == "intel":
                return ManagedLlamaCppPlatformTarget(
                    platform_id="linux-x64-sycl",
                    system="linux",
                    machine="x86_64",
                    asset_include_tokens=("ubuntu", "x64", "sycl"),
                    asset_exclude_tokens=("arm64", "s390x"),
                )
            return ManagedLlamaCppPlatformTarget(
                platform_id="linux-x64-cpu",
                system="linux",
                machine="x86_64",
                asset_include_tokens=("ubuntu", "x64"),
                asset_exclude_tokens=("arm64", "s390x", "cuda", "vulkan", "sycl", "hip"),
            )
        if resolved_machine in {"arm64", "aarch64"}:
            return ManagedLlamaCppPlatformTarget(
                platform_id="linux-arm64-cpu",
                system="linux",
                machine="arm64",
                asset_include_tokens=("ubuntu", "arm64"),
                asset_exclude_tokens=("x64", "s390x", "cuda", "vulkan", "sycl", "hip"),
            )
    elif resolved_system == "darwin":
        if resolved_machine in {"arm64", "aarch64"}:
            return ManagedLlamaCppPlatformTarget(
                platform_id="macos-arm64-cpu",
                system="darwin",
                machine="arm64",
                asset_include_tokens=("macos", "arm64"),
                asset_exclude_tokens=("x64", "vulkan", "cuda", "sycl", "hip"),
            )
        if resolved_machine in {"x86_64", "amd64"}:
            return ManagedLlamaCppPlatformTarget(
                platform_id="macos-x64-cpu",
                system="darwin",
                machine="x86_64",
                asset_include_tokens=("macos", "x64"),
                asset_exclude_tokens=("arm64", "vulkan", "cuda", "sycl", "hip"),
            )

    raise ManagedLlamaCppInstallValidationError(
        "Managed llama.cpp install is not supported on this platform: "
        f"system={resolved_system}, machine={resolved_machine}"
    )


def select_managed_llama_cpp_release_asset(
    assets: Sequence[ManagedLlamaCppReleaseAsset],
    *,
    target: ManagedLlamaCppPlatformTarget,
) -> ManagedLlamaCppReleaseAsset:
    """從 release assets 中選出最符合目前平台的 CPU binary。"""
    matches: list[ManagedLlamaCppReleaseAsset] = []
    for asset in assets:
        name = asset.name.strip().lower()
        if not name or not _is_supported_archive_name(name):
            continue
        if any(token not in name for token in target.asset_include_tokens):
            continue
        if any(token in name for token in target.asset_exclude_tokens):
            continue
        matches.append(asset)

    if not matches:
        raise ManagedLlamaCppInstallValidationError(
            "No compatible llama.cpp binary release asset was found for "
            f"{target.platform_id}."
        )

    def _sort_key(asset: ManagedLlamaCppReleaseAsset) -> tuple[int, int]:
        name = asset.name.lower()
        return (0 if ".zip" in name else 1, len(name))

    return sorted(matches, key=_sort_key)[0]


async def fetch_managed_llama_cpp_release_metadata(
    *,
    version: str | None = None,
    client: httpx.AsyncClient | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> ManagedLlamaCppReleaseMetadata:
    """讀取官方 GitHub release metadata，選出對應平台 asset。"""
    effective_version = (version or "").strip() or DEFAULT_MANAGED_LLAMA_CPP_VERSION
    target = detect_managed_llama_cpp_platform_target(
        system=system,
        machine=machine,
        hardware=_detect_hardware_summary(),
    )
    own_client = client is None
    http_client = client or httpx.AsyncClient(
        headers=dict(LLAMA_CPP_GITHUB_HEADERS),
        follow_redirects=True,
        timeout=120.0,
    )

    try:
        response = await http_client.get(
            LLAMA_CPP_GITHUB_RELEASE_API_TEMPLATE.format(version=effective_version)
        )
    except httpx.HTTPError as exc:
        raise ManagedLlamaCppInstallNetworkError(
            f"Failed to fetch llama.cpp release metadata for {effective_version}: {exc}"
        ) from exc
    finally:
        if own_client:
            await http_client.aclose()

    if response.status_code == 404:
        raise ManagedLlamaCppInstallValidationError(
            f"llama.cpp release tag does not exist: {effective_version}"
        )
    if response.status_code >= 400:
        raise ManagedLlamaCppInstallNetworkError(
            "Failed to fetch llama.cpp release metadata: "
            f"HTTP {response.status_code}"
        )

    payload = response.json()
    raw_assets = payload.get("assets") or []
    assets = [
        ManagedLlamaCppReleaseAsset(
            name=str(item.get("name", "")),
            download_url=str(item.get("browser_download_url", "")),
            size_bytes=int(item["size"]) if item.get("size") is not None else None,
            content_type=str(item.get("content_type")) if item.get("content_type") else None,
        )
        for item in raw_assets
        if item.get("name") and item.get("browser_download_url")
    ]
    binary_asset = select_managed_llama_cpp_release_asset(assets, target=target)

    return ManagedLlamaCppReleaseMetadata(
        version=effective_version,
        platform=target,
        source_archive_url=LLAMA_CPP_SOURCE_ARCHIVE_TEMPLATE.format(version=effective_version),
        source_archive_name=f"llama.cpp-{effective_version}.tar.gz",
        binary_asset=binary_asset,
    )


async def install_managed_llama_cpp_runtime(
    *,
    managed_root: str | Path,
    version: str | None = None,
    python_executable: str | None = None,
    client: httpx.AsyncClient | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> ManagedLlamaCppInstallResult:
    """下載、解壓並驗證 managed llama.cpp runtime。"""
    metadata = await fetch_managed_llama_cpp_release_metadata(
        version=version,
        client=client,
        system=system,
        machine=machine,
    )
    install_dir = get_managed_llama_cpp_install_dir(managed_root, version=metadata.version)
    install_dir.mkdir(parents=True, exist_ok=True)

    existing_status = get_managed_llama_cpp_runtime_status(
        managed_root=managed_root,
        preferred_root=install_dir,
        preferred_python=python_executable,
        preferred_version=metadata.version,
        preferred_source="managed",
    )
    if existing_status.state == "ready":
        return ManagedLlamaCppInstallResult(
            install_dir=str(install_dir),
            state="already_installed",
            source="managed",
            action="install",
            version=metadata.version,
            root_dir=str(install_dir),
            python_executable=existing_status.python_executable,
            warnings=[],
            message=f"Managed llama.cpp runtime already installed at {install_dir}.",
        )

    own_client = client is None
    http_client = client or httpx.AsyncClient(follow_redirects=True, timeout=300.0)

    temp_dir = _create_managed_install_temp_dir(managed_root)
    staging_dir = install_dir / ".i"
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_archive = temp_dir / "src.tar.gz"
        binary_archive = temp_dir / (
            "bin.zip"
            if metadata.binary_asset.name.lower().endswith(".zip")
            else "bin.tar.gz"
        )
        source_extract_dir = temp_dir / "s"
        binary_extract_dir = temp_dir / "b"
        await _download_to_file(http_client, metadata.source_archive_url, source_archive)
        await _download_to_file(http_client, metadata.binary_asset.download_url, binary_archive)

        _extract_archive_to_directory(source_archive, source_extract_dir)
        _flatten_single_directory(source_extract_dir)
        _merge_directory_tree(source_extract_dir, staging_dir)

        _extract_archive_to_directory(binary_archive, binary_extract_dir)
        _flatten_single_directory(binary_extract_dir)
        _merge_directory_tree(binary_extract_dir, staging_dir)

        toolchain = discover_llama_cpp_toolchain(
            preferred_root=staging_dir,
            preferred_python=python_executable,
            preferred_version=metadata.version,
            preferred_source="managed",
        )
        _ensure_llama_cpp_runtime_available(toolchain, "Q4_K_M")

        _merge_directory_tree(staging_dir, install_dir)
    except ManagedLlamaCppInstallError:
        raise
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ManagedLlamaCppInstallError(f"Failed to install managed llama.cpp runtime: {exc}") from exc
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        if own_client:
            await http_client.aclose()

    final_toolchain = discover_llama_cpp_toolchain(
        preferred_root=install_dir,
        preferred_python=python_executable,
        preferred_version=metadata.version,
        preferred_source="managed",
    )
    _ensure_llama_cpp_runtime_available(final_toolchain, "Q4_K_M")

    metadata_path = install_dir / ".mochi-managed-runtime.json"
    metadata_path.write_text(
        json.dumps(
            {
                "runtime": "llama.cpp",
                "version": metadata.version,
                "platform": metadata.platform.platform_id,
                "installed_at": datetime.now(tz=UTC).isoformat(),
                "binary_asset": metadata.binary_asset.name,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    return ManagedLlamaCppInstallResult(
        install_dir=str(install_dir),
        state="installed",
        source="managed",
        action="install",
        version=metadata.version,
        root_dir=str(install_dir),
        python_executable=final_toolchain.python_executable,
        warnings=[],
        message=f"Installed managed llama.cpp runtime {metadata.version} to {install_dir}.",
    )


def _is_direct_gguf_outtype(quantization: str) -> bool:
    """判斷是否可直接由 convert script 產出。"""
    return quantization in {"F16", "BF16"}


def _convert_outtype_for_quantization(quantization: str) -> str:
    """將 GGUF 量化值映射為 convert script outtype。"""
    return quantization.strip().lower()


async def _download_to_file(
    client: httpx.AsyncClient,
    url: str,
    destination: Path,
) -> None:
    """下載檔案到指定路徑。"""
    try:
        async with client.stream("GET", url, headers=LLAMA_CPP_GITHUB_HEADERS) as response:
            if response.status_code >= 400:
                raise ManagedLlamaCppInstallNetworkError(
                    f"Failed to download {url}: HTTP {response.status_code}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        handle.write(chunk)
    except httpx.HTTPError as exc:
        raise ManagedLlamaCppInstallNetworkError(f"Failed to download {url}: {exc}") from exc


def _create_managed_install_temp_dir(managed_root: str | Path) -> Path:
    """Create a short writable temp directory to avoid Windows path-length failures."""
    managed_base = Path(managed_root).expanduser().resolve(strict=False)
    candidates = [
        managed_base / ".tmp",
        managed_base,
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return Path(tempfile.mkdtemp(prefix="mli-", dir=candidate))
        except OSError:
            continue
    return Path(tempfile.mkdtemp(prefix="mli-"))


def _is_supported_archive_name(name: str) -> bool:
    """判斷是否為支援的壓縮檔名。"""
    lowered = name.strip().lower()
    return lowered.endswith(".zip") or lowered.endswith(".tar.gz") or lowered.endswith(".tgz")


def _extract_archive_to_directory(archive_path: Path, destination: Path) -> None:
    """解壓縮 archive 到目標目錄。"""
    name = archive_path.name.lower()
    destination.mkdir(parents=True, exist_ok=True)
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(destination)
        return
    raise ManagedLlamaCppInstallValidationError(
        f"Unsupported archive format for managed llama.cpp install: {archive_path.name}"
    )


def _flatten_single_directory(root: Path) -> None:
    """若解壓後只有單一外層資料夾，將內容提升一層。"""
    try:
        children = [child for child in root.iterdir() if child.name not in {".", ".."}]
    except FileNotFoundError:
        return
    if len(children) != 1 or not children[0].is_dir():
        return
    nested_root = children[0]
    temp_root = root.parent / f"{root.name}.flatten"
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    nested_root.rename(temp_root)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    for child in temp_root.iterdir():
        shutil.move(str(child), root / child.name)
    shutil.rmtree(temp_root, ignore_errors=True)


def _merge_directory_tree(source: Path, destination: Path) -> None:
    """將 staging 目錄內容合併進最終安裝路徑。"""
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            if target.exists() and target.is_dir():
                _merge_directory_tree(child, target)
            else:
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target, ignore_errors=True)
                    else:
                        target.unlink()
                shutil.move(str(child), target)
            continue
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink()
        shutil.move(str(child), target)


def _supported_gguf_quantization_ids() -> set[str]:
    """回傳 phase 1 支援的 GGUF 量化選項 id。"""
    return {item.id for item in _build_gguf_format_capability(None).quantization_options}


def _validate_gguf_quantization_option(quantization: str) -> None:
    """驗證 GGUF 量化選項是否合法。"""
    supported = _supported_gguf_quantization_ids()
    if quantization in supported:
        return
    options = ", ".join(sorted(supported))
    raise LocalModelConversionValidationError(
        f"Unsupported GGUF quantization '{quantization}'. Supported options: {options}"
    )


def _has_safetensors_weights(path: Path) -> bool:
    """判斷目錄是否包含 safetensors 權重。"""
    if (path / "model.safetensors.index.json").is_file():
        return True
    try:
        for child in path.iterdir():
            if child.is_symlink():
                continue
            if child.is_file() and child.suffix.lower() == ".safetensors":
                return True
    except PermissionError:
        return False
    return False


def _hf_layout_missing_requirements(path: Path) -> list[str]:
    """回傳 HF safetensors 目錄缺少的關鍵檔案。"""
    missing: list[str] = []
    if not (path / "config.json").is_file():
        missing.append("config.json")

    has_tokenizer = any(
        (path / filename).is_file()
        for filename in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )
    if not has_tokenizer:
        missing.append("tokenizer file (tokenizer.json/tokenizer.model/tokenizer_config.json)")

    if not _has_safetensors_weights(path):
        missing.append("safetensors weights (*.safetensors or model.safetensors.index.json)")

    return missing


def _read_hf_model_family(path: Path) -> str | None:
    """嘗試讀取 HF config.json 的 model_type。"""
    config_path = path / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    model_type = payload.get("model_type")
    if isinstance(model_type, str) and model_type.strip():
        return model_type.strip()
    return None


def _build_gguf_format_capability(
    hardware: HardwareSummary | None,
) -> QuantizationFormatCapability:
    """建立 GGUF 量化能力描述（第一優先）。"""
    options = [
        QuantizationOption(
            id="Q2_K",
            name="Q2_K",
            bits="2-3 bit",
            description="最小體積，品質折損明顯，適合極限記憶體情境。",
        ),
        QuantizationOption(
            id="Q3_K_M",
            name="Q3_K_M",
            bits="3-4 bit",
            description="低記憶體優先，品質較 Q4_K_M 再下降。",
        ),
        QuantizationOption(
            id="Q4_K_M",
            name="Q4_K_M",
            bits="4-5 bit",
            description="平衡大小與品質，預設建議優先。",
        ),
        QuantizationOption(
            id="Q5_K_M",
            name="Q5_K_M",
            bits="5-6 bit",
            description="較高品質，記憶體需求高於 Q4_K_M。",
        ),
        QuantizationOption(
            id="Q6_K",
            name="Q6_K",
            bits="6-7 bit",
            description="接近高精度品質，體積與延遲較高。",
        ),
        QuantizationOption(
            id="Q8_0",
            name="Q8_0",
            bits="8 bit",
            description="高品質低折損，記憶體需求明顯提高。",
        ),
        QuantizationOption(
            id="F16",
            name="F16",
            bits="16 bit",
            description="半精度浮點，品質高但體積較大。",
        ),
        QuantizationOption(
            id="BF16",
            name="BF16",
            bits="16 bit",
            description="bfloat16，品質高且常見於新一代硬體。",
        ),
    ]

    suggested = _suggest_default_gguf_quantization(hardware)
    warnings = [
        "Model-family-specific conversion quirks are not pre-validated; verify the converted GGUF with a real load/run after conversion.",
    ]
    reason = (
        "HF safetensors directory is valid and GGUF conversion is available through the llama.cpp tools/runtime."
    )
    return QuantizationFormatCapability(
        format_id="gguf",
        format_name="GGUF",
        supported=True,
        priority="primary",
        reason=reason,
        warnings=warnings,
        quantization_options=options,
        suggested_default_quantization=suggested,
    )


def _suggest_default_gguf_quantization(hardware: HardwareSummary | None) -> str:
    """依可用 VRAM 給保守 GGUF 預設。"""
    if hardware is None or hardware.total_vram_gb is None:
        return "Q4_K_M"
    vram = hardware.total_vram_gb
    if vram < 6.0:
        return "Q3_K_M"
    if vram < 12.0:
        return "Q4_K_M"
    if vram < 18.0:
        return "Q5_K_M"
    if vram < 24.0:
        return "Q6_K"
    return "Q8_0"


def _infer_gpu_vendor(primary_gpu_name: str | None) -> str | None:
    """Infer GPU vendor from the primary adapter name when possible."""
    if not primary_gpu_name:
        return None
    normalized = primary_gpu_name.strip().lower()
    if not normalized:
        return None
    if "nvidia" in normalized or "geforce" in normalized or "rtx" in normalized or "gtx" in normalized:
        return "nvidia"
    if "amd" in normalized or "radeon" in normalized or "rx " in normalized or normalized.startswith("rx"):
        return "amd"
    if "intel" in normalized or "arc" in normalized:
        return "intel"
    if "apple" in normalized or normalized.startswith("m1") or normalized.startswith("m2") or normalized.startswith("m3") or normalized.startswith("m4"):
        return "apple"
    return None


def _recommended_llama_cpp_backend(*, system_name: str, gpu_vendor: str | None, cuda_available: bool) -> tuple[str | None, str | None]:
    """Recommend the best llama.cpp runtime backend for the detected environment."""
    system_key = system_name.strip().lower()
    vendor = (gpu_vendor or "").strip().lower() or None

    if system_key == "darwin":
        return "metal", "Metal"
    if vendor == "nvidia":
        return ("cuda", "CUDA") if cuda_available else ("vulkan", "Vulkan")
    if vendor == "amd":
        if system_key == "windows":
            return "hip", "HIP"
        if system_key == "linux":
            return "hip", "ROCm/HIP"
        return "vulkan", "Vulkan"
    if vendor == "intel":
        if system_key == "linux":
            return "sycl", "SYCL"
        return "vulkan", "Vulkan"
    if cuda_available:
        return "cuda", "CUDA"
    return "cpu", "CPU"


def _probe_nvidia_smi_hardware_summary() -> HardwareSummary | None:
    """Probe NVIDIA GPU details on Windows without depending on PyTorch."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None

    try:
        completed = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:  # noqa: BLE001
        return None

    if completed.returncode != 0 or not completed.stdout.strip():
        return None

    entries = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not entries:
        return None

    primary_gpu_name: str | None = None
    total_vram_gb: float | None = None
    for index, entry in enumerate(entries):
        name = entry
        memory_mb: float | None = None
        if "," in entry:
            raw_name, raw_memory = entry.rsplit(",", 1)
            name = raw_name.strip()
            try:
                memory_mb = float(raw_memory.strip())
            except ValueError:
                memory_mb = None
        if index == 0:
            primary_gpu_name = name or None
            if memory_mb is not None and memory_mb > 0:
                total_vram_gb = round(memory_mb / 1024, 2)

    backend_key, backend_label = _recommended_llama_cpp_backend(
        system_name="Windows",
        gpu_vendor="nvidia",
        cuda_available=True,
    )
    return HardwareSummary(
        provider="nvidia-smi",
        cuda_available=True,
        gpu_count=len(entries),
        gpu_vendor="nvidia",
        primary_gpu_name=primary_gpu_name,
        total_vram_gb=total_vram_gb,
        recommended_runtime_backend=backend_key,
        recommended_runtime_label=backend_label,
        warnings=[],
    )


def _probe_windows_video_controller_hardware_summary() -> HardwareSummary | None:
    """Fallback Windows GPU probe via Win32_VideoController."""
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return None

    try:
        completed = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:  # noqa: BLE001
        return None

    if completed.returncode != 0 or not completed.stdout.strip():
        return None

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None

    entries = payload if isinstance(payload, list) else [payload]
    gpu_names: list[str] = []
    total_vram_gb: float | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_name = entry.get("Name")
        if isinstance(raw_name, str) and raw_name.strip():
            gpu_names.append(raw_name.strip())
        raw_adapter_ram = entry.get("AdapterRAM")
        if total_vram_gb is None and isinstance(raw_adapter_ram, int) and raw_adapter_ram > 0:
            total_vram_gb = round(raw_adapter_ram / (1024**3), 2)

    if not gpu_names:
        return None

    primary_gpu_name = gpu_names[0]
    gpu_vendor = _infer_gpu_vendor(primary_gpu_name)
    cuda_available = gpu_vendor == "nvidia"
    backend_key, backend_label = _recommended_llama_cpp_backend(
        system_name="Windows",
        gpu_vendor=gpu_vendor,
        cuda_available=cuda_available,
    )
    return HardwareSummary(
        provider="windows-cim",
        cuda_available=cuda_available,
        gpu_count=len(gpu_names),
        gpu_vendor=gpu_vendor,
        primary_gpu_name=primary_gpu_name,
        total_vram_gb=total_vram_gb,
        recommended_runtime_backend=backend_key,
        recommended_runtime_label=backend_label,
        warnings=[],
    )


def _detect_windows_hardware_summary() -> HardwareSummary | None:
    """Use Windows-native probes when PyTorch cannot provide a useful CUDA signal."""
    if platform.system().strip().lower() != "windows":
        return None
    return _probe_nvidia_smi_hardware_summary() or _probe_windows_video_controller_hardware_summary()


def _detect_windows_hardware_fallback(*, warning: str) -> HardwareSummary | None:
    """Attach the original probe warning to a successful Windows-native fallback."""
    summary = _detect_windows_hardware_summary()
    if summary is None:
        return None
    summary.warnings = [warning, *summary.warnings]
    return summary


def _detect_hardware_summary() -> HardwareSummary:
    """bounded CUDA/VRAM 偵測；無 CUDA 或缺依賴時回退不報錯。"""
    if importlib.util.find_spec("torch") is None:
        fallback = _detect_windows_hardware_fallback(
            warning="PyTorch is not installed; using Windows hardware fallback.",
        )
        if fallback is not None:
            return fallback
        return HardwareSummary(
            provider="none",
            cuda_available=False,
            gpu_count=0,
            gpu_vendor=None,
            primary_gpu_name=None,
            total_vram_gb=None,
            recommended_runtime_backend="cpu",
            recommended_runtime_label="CPU",
            warnings=["PyTorch is not installed; CUDA/VRAM probe is skipped."],
        )

    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        fallback = _detect_windows_hardware_fallback(
            warning=f"Failed to import torch for hardware probe: {exc}; using Windows hardware fallback.",
        )
        if fallback is not None:
            return fallback
        return HardwareSummary(
            provider="torch",
            cuda_available=False,
            gpu_count=0,
            gpu_vendor=None,
            primary_gpu_name=None,
            total_vram_gb=None,
            recommended_runtime_backend="cpu",
            recommended_runtime_label="CPU",
            warnings=[f"Failed to import torch for hardware probe: {exc}"],
        )

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        fallback = _detect_windows_hardware_fallback(
            warning=f"torch.cuda.is_available() failed: {exc}; using Windows hardware fallback.",
        )
        if fallback is not None:
            return fallback
        return HardwareSummary(
            provider="torch",
            cuda_available=False,
            gpu_count=0,
            gpu_vendor=None,
            primary_gpu_name=None,
            total_vram_gb=None,
            recommended_runtime_backend="cpu",
            recommended_runtime_label="CPU",
            warnings=[f"torch.cuda.is_available() failed: {exc}"],
        )

    if not cuda_available:
        fallback = _detect_windows_hardware_fallback(
            warning="torch.cuda.is_available() returned False; using Windows hardware fallback.",
        )
        if fallback is not None:
            return fallback
        return HardwareSummary(
            provider="torch",
            cuda_available=False,
            gpu_count=0,
            gpu_vendor=None,
            primary_gpu_name=None,
            total_vram_gb=None,
            recommended_runtime_backend="cpu",
            recommended_runtime_label="CPU",
            warnings=[],
        )

    warnings: list[str] = []
    try:
        gpu_count = int(torch.cuda.device_count())
    except Exception as exc:  # noqa: BLE001
        return HardwareSummary(
            provider="torch",
            cuda_available=True,
            gpu_count=0,
            gpu_vendor=None,
            primary_gpu_name=None,
            total_vram_gb=None,
            recommended_runtime_backend="cuda",
            recommended_runtime_label="CUDA",
            warnings=[f"torch.cuda.device_count() failed: {exc}"],
        )

    primary_gpu_name: str | None = None
    total_vram_gb: float | None = None
    if gpu_count > 0:
        try:
            props = torch.cuda.get_device_properties(0)
            primary_gpu_name = getattr(props, "name", None)
            total_memory = getattr(props, "total_memory", None)
            if isinstance(total_memory, int) and total_memory > 0:
                total_vram_gb = round(total_memory / (1024**3), 2)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"torch.cuda.get_device_properties(0) failed: {exc}")

    gpu_vendor = _infer_gpu_vendor(primary_gpu_name)
    backend_key, backend_label = _recommended_llama_cpp_backend(
        system_name=platform.system(),
        gpu_vendor=gpu_vendor,
        cuda_available=cuda_available,
    )
    return HardwareSummary(
        provider="torch",
        cuda_available=True,
        gpu_count=gpu_count,
        gpu_vendor=gpu_vendor,
        primary_gpu_name=primary_gpu_name,
        total_vram_gb=total_vram_gb,
        recommended_runtime_backend=backend_key,
        recommended_runtime_label=backend_label,
        warnings=warnings,
    )
