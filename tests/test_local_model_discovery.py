"""本地模型掃描邏輯測試。"""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import sys
import tarfile
import zipfile

import pytest

from mochi.backends.local_models import (
    DEFAULT_MANAGED_LLAMA_CPP_VERSION,
    HardwareSummary,
    LlamaCppLocalModelConverter,
    LlamaCppToolchain,
    ManagedLlamaCppReleaseAsset,
    ManagedLlamaCppRuntimeStatus,
    LocalModelConversionRuntimeUnavailableError,
    LocalModelConversionValidationError,
    LocalModelConvertRequest,
    PlaceholderLocalModelConverter,
    build_llama_cpp_convert_command,
    build_llama_cpp_quantize_command,
    build_gguf_output_model_path,
    detect_managed_llama_cpp_platform_target,
    discover_hf_quantization_capabilities,
    discover_llama_cpp_toolchain,
    discover_local_models,
    fetch_managed_llama_cpp_release_metadata,
    get_managed_llama_cpp_runtime_status,
    install_managed_llama_cpp_runtime,
    select_managed_llama_cpp_release_asset,
    _create_managed_install_temp_dir,
    _detect_hardware_summary,
)


def _write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_discover_local_models_finds_gguf_and_safetensors_dir(tmp_path: Path) -> None:
    """應能同時找到 GGUF 檔案與 HuggingFace safetensors 目錄。"""
    gguf = tmp_path / "qwen2.5-q4.gguf"
    _write_file(gguf, "gguf")

    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    _write_file(hf_dir / "config.json", "{}")
    _write_file(hf_dir / "tokenizer.json", "{}")
    _write_file(hf_dir / "model-00001-of-00002.safetensors", "x")

    result = discover_local_models(tmp_path, max_depth=3, max_entries=100)

    assert result.root == str(tmp_path.resolve())
    specs = {candidate.model_spec for candidate in result.models}
    assert str(gguf.resolve()) in specs
    assert str(hf_dir.resolve()) in specs
    by_spec = {candidate.model_spec: candidate for candidate in result.models}
    assert by_spec[str(gguf.resolve())].backend_type == "gguf"
    assert by_spec[str(hf_dir.resolve())].backend_type == "safetensors"


def test_discover_local_models_respects_max_depth(tmp_path: Path) -> None:
    """超過 max_depth 的模型不應被掃描到。"""
    shallow = tmp_path / "shallow.gguf"
    _write_file(shallow, "gguf")
    deep = tmp_path / "a" / "b" / "c" / "d" / "too-deep.gguf"
    _write_file(deep, "gguf")

    result = discover_local_models(tmp_path, max_depth=2, max_entries=200)
    specs = {candidate.model_spec for candidate in result.models}

    assert str(shallow.resolve()) in specs
    assert str(deep.resolve()) not in specs


def test_discover_local_models_does_not_follow_symlink(tmp_path: Path) -> None:
    """掃描時不應跟隨 symlink。"""
    outside = tmp_path / "outside"
    outside.mkdir()
    hidden = outside / "hidden.gguf"
    _write_file(hidden, "gguf")

    root = tmp_path / "root"
    root.mkdir()
    (root / "outside_link").symlink_to(outside, target_is_directory=True)

    result = discover_local_models(root, max_depth=5, max_entries=200)
    specs = {candidate.model_spec for candidate in result.models}

    assert str(hidden.resolve()) not in specs


def test_discover_local_models_respects_max_entries(tmp_path: Path) -> None:
    """超過 max_entries 時應中止掃描並回傳 warning。"""
    for idx in range(30):
        _write_file(tmp_path / f"file-{idx}.txt", "x")
    _write_file(tmp_path / "model.gguf", "gguf")

    result = discover_local_models(tmp_path, max_depth=1, max_entries=10)

    assert result.models == []
    assert any("max_entries=10" in warning for warning in result.warnings)


def test_discover_hf_quantization_capabilities_returns_gguf(tmp_path: Path, monkeypatch) -> None:
    """HF 目錄量化能力探測應至少包含 GGUF 格式。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    _write_file(hf_dir / "config.json", '{"model_type":"qwen2"}')
    _write_file(hf_dir / "tokenizer.json", "{}")
    _write_file(hf_dir / "model.safetensors", "x")

    monkeypatch.setattr(
        "mochi.backends.local_models._detect_hardware_summary",
        lambda: HardwareSummary(
            provider="torch",
            cuda_available=True,
            gpu_count=1,
            primary_gpu_name="RTX 4090",
            total_vram_gb=24.0,
            warnings=[],
        ),
    )
    result = discover_hf_quantization_capabilities(hf_dir)

    assert result.model_dir == str(hf_dir.resolve())
    assert result.model_family == "qwen2"
    assert result.hardware is not None
    assert result.hardware.cuda_available is True
    by_format = {item.format_id: item for item in result.formats}
    assert set(by_format.keys()) == {"gguf"}
    assert by_format["gguf"].supported is True
    assert by_format["gguf"].priority == "primary"
    assert "llama.cpp tools/runtime" in (by_format["gguf"].reason or "")
    assert any("verify the converted GGUF" in warning for warning in by_format["gguf"].warnings)
    option_ids = {item.id for item in by_format["gguf"].quantization_options}
    assert {"Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16"} <= option_ids
    assert by_format["gguf"].suggested_default_quantization == "Q8_0"


def test_discover_hf_quantization_capabilities_fallback_default_without_vram(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """偵測不到 VRAM 時，GGUF 建議預設應回退為保守值。"""
    hf_dir = tmp_path / "Llama"
    _write_file(hf_dir / "config.json", "{}")
    _write_file(hf_dir / "tokenizer.json", "{}")
    _write_file(hf_dir / "model-00001-of-00002.safetensors", "x")

    monkeypatch.setattr(
        "mochi.backends.local_models._detect_hardware_summary",
        lambda: HardwareSummary(
            provider="torch",
            cuda_available=False,
            gpu_count=0,
            primary_gpu_name=None,
            total_vram_gb=None,
            warnings=[],
        ),
    )
    result = discover_hf_quantization_capabilities(hf_dir)
    gguf = next(item for item in result.formats if item.format_id == "gguf")
    assert gguf.suggested_default_quantization == "Q4_K_M"


def test_discover_hf_quantization_capabilities_reports_missing_requirements(tmp_path: Path) -> None:
    """缺少 HF 必要檔案時應回傳可讀錯誤。"""
    broken = tmp_path / "broken"
    broken.mkdir()
    _write_file(broken / "config.json", "{}")

    with pytest.raises(ValueError, match="Missing: tokenizer file"):
        discover_hf_quantization_capabilities(broken)


def test_build_gguf_output_model_path_uses_deterministic_naming(tmp_path: Path) -> None:
    """GGUF 轉換輸出檔名應使用固定規則 `<dir>-<quant>.gguf`。"""
    source_dir = tmp_path / "Qwen2.5-7B-Instruct"
    source_dir.mkdir()

    output = build_gguf_output_model_path(source_dir, "Q4_K_M")

    assert output == source_dir.parent / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"


def test_placeholder_converter_rejects_unsupported_quantization(tmp_path: Path) -> None:
    """不支援的 GGUF 量化值應回傳 validation 錯誤。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    _write_file(hf_dir / "config.json", "{}")
    _write_file(hf_dir / "tokenizer.json", "{}")
    _write_file(hf_dir / "model.safetensors", "x")
    converter = PlaceholderLocalModelConverter()

    with pytest.raises(LocalModelConversionValidationError, match="Unsupported GGUF quantization"):
        asyncio.run(
            converter.convert(
                LocalModelConvertRequest(
                    source_model_dir=str(hf_dir),
                    target_format="gguf",
                    quantization="Q9_FAKE",
                )
            )
        )


def test_placeholder_converter_returns_runtime_unavailable_for_phase1(tmp_path: Path) -> None:
    """第一期 placeholder converter 預設應回傳 runtime unavailable。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    _write_file(hf_dir / "config.json", "{}")
    _write_file(hf_dir / "tokenizer.json", "{}")
    _write_file(hf_dir / "model.safetensors", "x")
    converter = PlaceholderLocalModelConverter()

    with pytest.raises(LocalModelConversionRuntimeUnavailableError, match="runtime is unavailable"):
        asyncio.run(
            converter.convert(
                LocalModelConvertRequest(
                    source_model_dir=str(hf_dir),
                    target_format="gguf",
                    quantization="Q4_K_M",
                )
            )
        )


def test_discover_llama_cpp_toolchain_prefers_env_paths(tmp_path: Path) -> None:
    """toolchain discovery 應優先採用明確 env path。"""
    llama_dir = tmp_path / "llama.cpp"
    convert_script = llama_dir / "convert_hf_to_gguf.py"
    quantize_binary = llama_dir / "build" / "bin" / "llama-quantize"
    _write_file(convert_script, "#!/usr/bin/env python3")
    _write_file(quantize_binary, "bin")

    toolchain = discover_llama_cpp_toolchain(
        env={
            "MOCHI_LLAMA_CPP_DIR": str(llama_dir),
            "MOCHI_LLAMA_CPP_PYTHON": "/usr/bin/python3",
        },
        cwd=tmp_path,
    )

    assert toolchain.python_executable == "/usr/bin/python3"
    assert toolchain.convert_script == convert_script.resolve()
    assert toolchain.quantize_binary == quantize_binary.resolve()
    assert str(llama_dir.resolve()) in toolchain.search_roots


def test_build_llama_cpp_convert_command_uses_outfile_and_outtype(tmp_path: Path) -> None:
    """convert command builder 應輸出穩定命令列。"""
    source_dir = tmp_path / "Qwen2.5-7B-Instruct"
    source_dir.mkdir()
    output_path = tmp_path / "Qwen2.5-7B-Instruct-F16.gguf"
    convert_script = tmp_path / "llama.cpp" / "convert_hf_to_gguf.py"
    _write_file(convert_script, "#!/usr/bin/env python3")
    toolchain = LlamaCppToolchain(
        python_executable="/usr/bin/python3",
        convert_script=convert_script.resolve(),
        quantize_binary=None,
    )

    command = build_llama_cpp_convert_command(
        toolchain=toolchain,
        source_model_dir=source_dir,
        output_model_path=output_path,
        outtype="f16",
    )

    assert command == [
        "/usr/bin/python3",
        str(convert_script.resolve()),
        str(source_dir.resolve()),
        "--outfile",
        str(output_path.resolve()),
        "--outtype",
        "f16",
    ]


def test_build_llama_cpp_quantize_command_uses_quantization_name(tmp_path: Path) -> None:
    """quantize command builder 應輸出穩定命令列。"""
    input_path = tmp_path / "Qwen2.5-7B-Instruct-F16.gguf"
    output_path = tmp_path / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    quantize_binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-quantize"
    _write_file(quantize_binary, "bin")
    toolchain = LlamaCppToolchain(
        python_executable="/usr/bin/python3",
        convert_script=None,
        quantize_binary=quantize_binary.resolve(),
    )

    command = build_llama_cpp_quantize_command(
        toolchain=toolchain,
        input_model_path=input_path,
        output_model_path=output_path,
        quantization="Q4_K_M",
    )

    assert command == [
        str(quantize_binary.resolve()),
        str(input_path.resolve()),
        str(output_path.resolve()),
        "Q4_K_M",
    ]


@pytest.mark.parametrize(
    ("requested_quantization", "normalized_quantization"),
    [
        ("f16", "F16"),
        ("bf16", "BF16"),
        ("q4_k_m", "Q4_K_M"),
    ],
)
def test_placeholder_converter_normalizes_supported_quantization_and_output_path(
    tmp_path: Path,
    requested_quantization: str,
    normalized_quantization: str,
) -> None:
    """converter contract: 支援值應正規化並回傳 deterministic GGUF output path。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    _write_file(hf_dir / "config.json", "{}")
    _write_file(hf_dir / "tokenizer.json", "{}")
    _write_file(hf_dir / "model.safetensors", "x")
    converter = PlaceholderLocalModelConverter(runtime_available=True)

    result = asyncio.run(
        converter.convert(
            LocalModelConvertRequest(
                source_model_dir=str(hf_dir),
                target_format="GGUF",
                quantization=requested_quantization,
            )
        )
    )

    expected_output = hf_dir.resolve().parent / f"{hf_dir.name}-{normalized_quantization}.gguf"
    assert result.target_format == "gguf"
    assert result.quantization == normalized_quantization
    assert result.source_model_dir == str(hf_dir.resolve())
    assert result.output_model_path == str(expected_output)
    assert result.converted is True


def test_llama_cpp_converter_runtime_unavailable_error_mentions_missing_tools(tmp_path: Path) -> None:
    """real converter 在缺工具時應提供可操作的訊息。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    _write_file(hf_dir / "config.json", "{}")
    _write_file(hf_dir / "tokenizer.json", "{}")
    _write_file(hf_dir / "model.safetensors", "x")
    converter = LlamaCppLocalModelConverter(env={}, cwd=tmp_path)

    with pytest.raises(LocalModelConversionRuntimeUnavailableError, match="convert_hf_to_gguf.py"):
        asyncio.run(
            converter.convert(
                LocalModelConvertRequest(
                    source_model_dir=str(hf_dir),
                    target_format="gguf",
                    quantization="Q4_K_M",
                )
            )
        )


def test_llama_cpp_converter_uses_direct_path_for_f16_and_bf16(tmp_path: Path, monkeypatch) -> None:
    """F16/BF16 應走 convert 直出，不呼叫 quantize。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    _write_file(hf_dir / "config.json", "{}")
    _write_file(hf_dir / "tokenizer.json", "{}")
    _write_file(hf_dir / "model.safetensors", "x")

    convert_script = tmp_path / "llama.cpp" / "convert_hf_to_gguf.py"
    quantize_binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-quantize"
    _write_file(convert_script, "#!/usr/bin/env python3")
    _write_file(quantize_binary, "bin")

    toolchain = LlamaCppToolchain(
        python_executable="/usr/bin/python3",
        convert_script=convert_script.resolve(),
        quantize_binary=quantize_binary.resolve(),
    )
    converter = LlamaCppLocalModelConverter(toolchain=toolchain, env={}, cwd=tmp_path)

    calls: dict[str, int] = {"convert": 0, "quantize": 0}

    async def _fake_convert(*, toolchain, source_model_dir, output_model_path, outtype) -> None:
        _ = toolchain, source_model_dir, outtype
        calls["convert"] += 1
        output_model_path.write_text("gguf", encoding="utf-8")

    async def _fake_quantize(*, toolchain, input_model_path, output_model_path, quantization) -> None:
        _ = toolchain, input_model_path, output_model_path, quantization
        calls["quantize"] += 1

    monkeypatch.setattr(converter, "_run_convert_command", _fake_convert)
    monkeypatch.setattr(converter, "_run_quantize_command", _fake_quantize)

    f16 = asyncio.run(
        converter.convert(
            LocalModelConvertRequest(
                source_model_dir=str(hf_dir),
                target_format="gguf",
                quantization="F16",
            )
        )
    )
    bf16 = asyncio.run(
        converter.convert(
            LocalModelConvertRequest(
                source_model_dir=str(hf_dir),
                target_format="gguf",
                quantization="BF16",
            )
        )
    )
    assert calls["convert"] == 2
    assert calls["quantize"] == 0
    assert f16.quantization == "F16"
    assert bf16.quantization == "BF16"


def test_llama_cpp_converter_quantized_path_uses_quantize_step(tmp_path: Path, monkeypatch) -> None:
    """Q4_K_M 等非直出格式應先 F16 再跑 quantize。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    _write_file(hf_dir / "config.json", "{}")
    _write_file(hf_dir / "tokenizer.json", "{}")
    _write_file(hf_dir / "model.safetensors", "x")

    convert_script = tmp_path / "llama.cpp" / "convert_hf_to_gguf.py"
    quantize_binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-quantize"
    _write_file(convert_script, "#!/usr/bin/env python3")
    _write_file(quantize_binary, "bin")
    toolchain = LlamaCppToolchain(
        python_executable="/usr/bin/python3",
        convert_script=convert_script.resolve(),
        quantize_binary=quantize_binary.resolve(),
    )
    converter = LlamaCppLocalModelConverter(toolchain=toolchain, env={}, cwd=tmp_path)
    seen: list[str] = []

    async def _fake_convert(*, toolchain, source_model_dir, output_model_path, outtype) -> None:
        _ = toolchain, source_model_dir
        seen.append(f"convert:{outtype}")
        output_model_path.write_text("gguf-f16", encoding="utf-8")

    async def _fake_quantize(*, toolchain, input_model_path, output_model_path, quantization) -> None:
        _ = toolchain, input_model_path
        seen.append(f"quantize:{quantization}")
        output_model_path.write_text("gguf-q4", encoding="utf-8")

    monkeypatch.setattr(converter, "_run_convert_command", _fake_convert)
    monkeypatch.setattr(converter, "_run_quantize_command", _fake_quantize)

    result = asyncio.run(
        converter.convert(
            LocalModelConvertRequest(
                source_model_dir=str(hf_dir),
                target_format="gguf",
                quantization="Q4_K_M",
            )
        )
    )
    assert seen == ["convert:f16", "quantize:Q4_K_M"]
    assert result.quantization == "Q4_K_M"


def test_discover_llama_cpp_toolchain_prefers_explicit_source_over_env(tmp_path: Path) -> None:
    """source selection helper：有 preferred_source 時應覆蓋 env 推導來源。"""
    llama_dir = tmp_path / "llama.cpp"
    _write_file(llama_dir / "convert_hf_to_gguf.py", "#!/usr/bin/env python3")
    _write_file(llama_dir / "build" / "bin" / "llama-quantize", "bin")

    toolchain = discover_llama_cpp_toolchain(
        env={"MOCHI_LLAMA_CPP_DIR": str(llama_dir)},
        cwd=tmp_path,
        preferred_source="managed",
    )

    assert toolchain.source == "managed"


def test_get_managed_runtime_status_reports_ready_with_fake_toolchain(monkeypatch, tmp_path: Path) -> None:
    """install orchestration success path（fake）：應映射為 ready runtime status。"""
    managed_root = tmp_path / "workspace"
    fake_root = managed_root / "runtimes" / "llama.cpp" / "v0.0.0-test"

    def _fake_discover(**_: object) -> LlamaCppToolchain:
        return LlamaCppToolchain(
            python_executable="/usr/bin/python3",
            convert_script=fake_root / "convert_hf_to_gguf.py",
            quantize_binary=fake_root / "build" / "bin" / "llama-quantize",
            root_dir=fake_root,
            version="v0.0.0-test",
            source="managed",
        )

    monkeypatch.setattr("mochi.backends.local_models.discover_llama_cpp_toolchain", _fake_discover)

    status = get_managed_llama_cpp_runtime_status(
        managed_root=managed_root,
        preferred_source="managed",
        preferred_version="v0.0.0-test",
    )

    assert isinstance(status, ManagedLlamaCppRuntimeStatus)
    assert status.state == "ready"
    assert status.installed is True
    assert status.source == "managed"
    assert status.root_dir == str(fake_root)
    assert status.install_dir == str(fake_root)
    assert status.convert_script == str(fake_root / "convert_hf_to_gguf.py")
    assert status.quantize_binary == str(fake_root / "build" / "bin" / "llama-quantize")
    assert status.actions == ["ready_for_conversion"]
    assert status.missing_components == []


def test_detect_managed_llama_cpp_platform_target_linux_x64() -> None:
    """platform target 應將 Linux x64 映射到 CPU asset 選擇規則。"""
    target = detect_managed_llama_cpp_platform_target(system="Linux", machine="x86_64")
    assert target.platform_id == "linux-x64-cpu"
    assert target.asset_include_tokens == ("ubuntu", "x64")
    assert "cuda" in target.asset_exclude_tokens


def test_select_managed_llama_cpp_release_asset_prefers_cpu_archive() -> None:
    """asset selection 應排除 GPU variant，挑出 CPU binary archive。"""
    target = detect_managed_llama_cpp_platform_target(system="Linux", machine="x86_64")
    assets = [
        ManagedLlamaCppReleaseAsset(
            name="llama-b9058-bin-ubuntu-x64-cuda-12.4.zip",
            download_url="https://example.invalid/cuda.zip",
        ),
        ManagedLlamaCppReleaseAsset(
            name="llama-b9058-bin-ubuntu-x64.zip",
            download_url="https://example.invalid/cpu.zip",
        ),
    ]

    selected = select_managed_llama_cpp_release_asset(assets, target=target)

    assert selected.name == "llama-b9058-bin-ubuntu-x64.zip"


def test_detect_managed_llama_cpp_platform_target_windows_x64_prefers_cuda_when_available() -> None:
    """Windows x64 + CUDA hardware should prefer CUDA runtime assets."""
    target = detect_managed_llama_cpp_platform_target(
        system="Windows",
        machine="AMD64",
        hardware=HardwareSummary(
            provider="torch",
            cuda_available=True,
            gpu_count=1,
            primary_gpu_name="RTX 3090",
            total_vram_gb=24.0,
        ),
    )

    assert target.platform_id == "windows-x64-cuda"
    assert target.asset_include_tokens == ("win", "x64", "cuda")


def test_select_managed_llama_cpp_release_asset_prefers_cuda_archive_for_windows_gpu() -> None:
    """Windows GPU target should choose CUDA archive over CPU archive."""
    target = detect_managed_llama_cpp_platform_target(
        system="Windows",
        machine="AMD64",
        hardware=HardwareSummary(
            provider="torch",
            cuda_available=True,
            gpu_count=1,
            primary_gpu_name="RTX 3090",
            total_vram_gb=24.0,
        ),
    )
    assets = [
        ManagedLlamaCppReleaseAsset(
            name="llama-b9058-bin-win-cuda-12.4-x64.zip",
            download_url="https://example.invalid/cuda.zip",
        ),
        ManagedLlamaCppReleaseAsset(
            name="llama-b9058-bin-win-x64.zip",
            download_url="https://example.invalid/cpu.zip",
        ),
    ]

    selected = select_managed_llama_cpp_release_asset(assets, target=target)

    assert selected.name == "llama-b9058-bin-win-cuda-12.4-x64.zip"


def test_detect_managed_llama_cpp_platform_target_windows_amd_prefers_hip() -> None:
    """Windows AMD GPUs should prefer HIP runtime assets."""
    target = detect_managed_llama_cpp_platform_target(
        system="Windows",
        machine="AMD64",
        hardware=HardwareSummary(
            provider="torch",
            cuda_available=False,
            gpu_count=1,
            gpu_vendor="amd",
            primary_gpu_name="AMD Radeon RX 7900 XTX",
            total_vram_gb=24.0,
            recommended_runtime_backend="hip",
            recommended_runtime_label="HIP",
        ),
    )

    assert target.platform_id == "windows-x64-hip"
    assert target.asset_include_tokens == ("win", "x64", "hip")


def test_detect_managed_llama_cpp_platform_target_linux_amd_prefers_rocm() -> None:
    """Linux AMD GPUs should prefer ROCm/HIP runtime assets."""
    target = detect_managed_llama_cpp_platform_target(
        system="Linux",
        machine="x86_64",
        hardware=HardwareSummary(
            provider="torch",
            cuda_available=False,
            gpu_count=1,
            gpu_vendor="amd",
            primary_gpu_name="AMD Radeon RX 7900 XTX",
            total_vram_gb=24.0,
            recommended_runtime_backend="hip",
            recommended_runtime_label="ROCm/HIP",
        ),
    )

    assert target.platform_id == "linux-x64-rocm"
    assert target.asset_include_tokens == ("ubuntu", "x64", "rocm")


def test_detect_hardware_summary_uses_windows_nvidia_fallback_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No torch on Windows/NVIDIA should still recommend CUDA runtime assets."""

    monkeypatch.setattr("mochi.backends.local_models.importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr("mochi.backends.local_models.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "mochi.backends.local_models._detect_windows_hardware_summary",
        lambda: HardwareSummary(
            provider="windows-system",
            cuda_available=True,
            gpu_count=1,
            gpu_vendor="nvidia",
            primary_gpu_name="NVIDIA GeForce RTX 4090",
            total_vram_gb=24.0,
            recommended_runtime_backend="cuda",
            recommended_runtime_label="CUDA",
            warnings=[],
        ),
    )

    summary = _detect_hardware_summary()

    assert summary.provider == "windows-system"
    assert summary.cuda_available is True
    assert summary.gpu_vendor == "nvidia"
    assert summary.recommended_runtime_backend == "cuda"
    assert summary.recommended_runtime_label == "CUDA"


def test_detect_hardware_summary_uses_windows_nvidia_fallback_when_torch_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-only torch should not force CPU recommendation on Windows/NVIDIA machines."""

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _FakeTorch:
        cuda = _FakeCuda()

    monkeypatch.setattr("mochi.backends.local_models.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("mochi.backends.local_models.platform.system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setattr(
        "mochi.backends.local_models._detect_windows_hardware_summary",
        lambda: HardwareSummary(
            provider="windows-system",
            cuda_available=True,
            gpu_count=1,
            gpu_vendor="nvidia",
            primary_gpu_name="NVIDIA GeForce RTX 4090",
            total_vram_gb=24.0,
            recommended_runtime_backend="cuda",
            recommended_runtime_label="CUDA",
            warnings=[],
        ),
    )

    summary = _detect_hardware_summary()

    assert summary.provider == "windows-system"
    assert summary.cuda_available is True
    assert summary.recommended_runtime_backend == "cuda"
    assert summary.recommended_runtime_label == "CUDA"


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: dict[str, object] | None = None,
        body: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self._body = body

    def json(self) -> dict[str, object]:
        return self._json_data

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def aiter_bytes(self):
        yield self._body


class _FakeHttpClient:
    def __init__(self, *, release_payload: dict[str, object], downloads: dict[str, bytes]) -> None:
        self.release_payload = release_payload
        self.downloads = downloads
        self.calls: list[str] = []

    async def get(self, url: str) -> _FakeResponse:
        self.calls.append(url)
        return _FakeResponse(status_code=200, json_data=self.release_payload)

    def stream(self, method: str, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:
        _ = method
        _ = headers
        self.calls.append(url)
        return _FakeResponse(status_code=200, body=self.downloads[url])

    async def aclose(self) -> None:
        return None


def _build_source_archive(path: Path) -> bytes:
    source_root = path / "llama.cpp-b9058"
    _write_file(source_root / "convert_hf_to_gguf.py", "#!/usr/bin/env python3\nprint('ok')\n")
    archive_path = path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_root, arcname=source_root.name)
    return archive_path.read_bytes()


def _build_binary_archive(path: Path) -> bytes:
    binary_root = path / "llama-b9058-bin-ubuntu-x64"
    _write_file(binary_root / "build" / "bin" / "llama-quantize", "bin")
    archive_path = path / "binary.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for file_path in binary_root.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(path))
    return archive_path.read_bytes()


def test_fetch_managed_llama_cpp_release_metadata_uses_default_version(tmp_path: Path) -> None:
    """release metadata helper 未指定版本時應使用 pinned default version。"""
    payload = {
        "assets": [
            {
                "name": f"llama-{DEFAULT_MANAGED_LLAMA_CPP_VERSION}-bin-ubuntu-x64.zip",
                "browser_download_url": "https://example.invalid/cpu.zip",
                "size": 123,
            }
        ]
    }
    client = _FakeHttpClient(release_payload=payload, downloads={})

    result = asyncio.run(
        fetch_managed_llama_cpp_release_metadata(
            client=client,
            system="Linux",
            machine="x86_64",
        )
    )

    assert result.version == DEFAULT_MANAGED_LLAMA_CPP_VERSION
    assert result.platform.platform_id == "linux-x64-cpu"
    assert result.binary_asset.name == f"llama-{DEFAULT_MANAGED_LLAMA_CPP_VERSION}-bin-ubuntu-x64.zip"


def test_install_managed_llama_cpp_runtime_downloads_and_installs(tmp_path: Path) -> None:
    """real managed installer 應在 mock download/extract 後產出 ready runtime。"""
    source_url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b9058.tar.gz"
    binary_url = "https://example.invalid/llama-b9058-bin-ubuntu-x64.zip"
    payload = {
        "assets": [
            {
                "name": "llama-b9058-bin-ubuntu-x64.zip",
                "browser_download_url": binary_url,
                "size": 456,
            }
        ]
    }
    downloads = {
        source_url: _build_source_archive(tmp_path / "archives-source"),
        binary_url: _build_binary_archive(tmp_path / "archives-binary"),
    }
    client = _FakeHttpClient(release_payload=payload, downloads=downloads)

    result = asyncio.run(
        install_managed_llama_cpp_runtime(
            managed_root=tmp_path / "workspace",
            version="b9058",
            python_executable="/usr/bin/python3",
            client=client,
            system="Linux",
            machine="x86_64",
        )
    )

    install_dir = tmp_path / "workspace" / "runtimes" / "llama.cpp" / "b9058"
    assert result.state == "installed"
    assert result.version == "b9058"
    assert result.root_dir == str(install_dir.resolve())
    assert (install_dir / "convert_hf_to_gguf.py").is_file()
    assert (install_dir / "build" / "bin" / "llama-quantize").is_file()
    assert (install_dir / ".mochi-managed-runtime.json").is_file()

    status = get_managed_llama_cpp_runtime_status(
        managed_root=tmp_path / "workspace",
        preferred_root=install_dir,
        preferred_python="/usr/bin/python3",
        preferred_version="b9058",
        preferred_source="managed",
    )
    assert status.state == "ready"
    assert status.root_dir == str(install_dir.resolve())


def test_create_managed_install_temp_dir_prefers_workspace_tmp(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    temp_dir = _create_managed_install_temp_dir(workspace)
    try:
        assert temp_dir.parent == (workspace / ".tmp").resolve()
        assert temp_dir.name.startswith("mli-")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
