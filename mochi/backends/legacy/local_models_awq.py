"""Legacy AWQ local-model conversion helpers kept for reference only."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
from pathlib import Path


class LegacyAwqConversionError(RuntimeError):
    """Legacy AWQ conversion error."""


class LegacyAwqRuntimeUnavailableError(LegacyAwqConversionError):
    """Legacy AWQ runtime unavailable error."""


def build_awq_output_model_path(
    source_model_dir: str | Path,
    quantization: str,
) -> Path:
    """建立 AWQ 輸出目錄路徑（legacy deterministic naming）。"""
    source_path = Path(source_model_dir).expanduser().resolve(strict=False)
    quant = normalize_awq_quantization(quantization)
    return source_path.parent / f"{source_path.name}-AWQ-{quant}"


def normalize_awq_quantization(quantization: str) -> str:
    """將 AWQ 量化值正規化為內部表示。"""
    normalized = quantization.strip().upper().replace("-", "_")
    alias = {
        "W4A16": "W4A16",
        "W4A16_G128": "W4A16_G128",
    }
    return alias.get(normalized, normalized)


def validate_awq_quantization_option(quantization: str) -> None:
    """驗證 AWQ 量化選項是否合法。"""
    supported = {"W4A16", "W4A16_G128"}
    if quantization in supported:
        return
    options = ", ".join(sorted(supported))
    raise LegacyAwqConversionError(
        f"Unsupported AWQ quantization '{quantization}'. Supported options: {options}"
    )


def awq_quant_config_for_quantization(quantization: str) -> dict[str, str | int | bool]:
    """將 AWQ 量化 id 映射為 AutoAWQ quant_config。"""
    normalized = normalize_awq_quantization(quantization)
    if normalized in {"W4A16", "W4A16_G128"}:
        return {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM",
        }
    raise LegacyAwqConversionError(f"Unsupported AWQ quantization '{quantization}'.")


def awq_runtime_missing_components(*, require_cuda: bool) -> list[str]:
    """檢查 AWQ 轉換 runtime 缺少的元件。"""
    missing: list[str] = []
    if importlib.util.find_spec("awq") is None:
        missing.append("autoawq (`awq` module)")
    if importlib.util.find_spec("transformers") is None:
        missing.append("transformers")
    if importlib.util.find_spec("torch") is None:
        missing.append("torch")
    elif require_cuda:
        try:
            torch_module = importlib.import_module("torch")
            cuda_available = bool(getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)())
        except Exception:
            cuda_available = False
        if not cuda_available:
            missing.append("CUDA-enabled torch runtime")
    return missing


def ensure_awq_runtime_available(*, require_cuda: bool) -> None:
    """確保 AWQ runtime 可用。"""
    missing = awq_runtime_missing_components(require_cuda=require_cuda)
    if not missing:
        return
    hints = ["Install dependencies: pip install autoawq transformers torch."]
    if require_cuda:
        hints.append("AWQ conversion path currently requires CUDA-enabled torch.")
    raise LegacyAwqRuntimeUnavailableError(
        "AWQ converter runtime is unavailable: missing "
        + ", ".join(missing)
        + ". "
        + " ".join(hints)
    )


def validate_awq_output_model_dir(output_dir: Path) -> None:
    """驗證 AWQ 輸出目錄包含最低限度可載入的檔案。"""
    if not output_dir.is_dir():
        raise LegacyAwqConversionError(
            f"AWQ conversion finished but output directory is missing: {output_dir}"
        )
    missing = []
    if not (output_dir / "config.json").is_file():
        missing.append("config.json")
    has_tokenizer = any(
        (output_dir / filename).is_file()
        for filename in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )
    if not has_tokenizer:
        missing.append("tokenizer file (tokenizer.json/tokenizer.model/tokenizer_config.json)")
    has_weights = (output_dir / "model.safetensors.index.json").is_file() or any(
        child.is_file() and child.suffix.lower() == ".safetensors"
        for child in output_dir.iterdir()
        if not child.is_symlink()
    )
    if not has_weights:
        missing.append("safetensors weights (*.safetensors or model.safetensors.index.json)")
    if missing:
        detail = ", ".join(missing)
        raise LegacyAwqConversionError(
            "AWQ conversion finished but output directory is incomplete. "
            f"Missing: {detail}"
        )
    config_path = output_dir / "config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LegacyAwqConversionError(
            f"AWQ conversion finished but output config.json is invalid: {exc}"
        ) from exc
    quant_cfg = payload.get("quantization_config")
    quant_method = quant_cfg.get("quant_method") if isinstance(quant_cfg, dict) else None
    if str(quant_method or "").strip().lower() != "awq":
        raise LegacyAwqConversionError(
            "AWQ conversion finished but output config.json has no "
            "quantization_config.quant_method=awq."
        )


async def run_awq_convert(
    *,
    source_model_dir: Path,
    output_model_dir: Path,
    quantization: str,
) -> None:
    """執行 legacy AWQ convert 路徑。"""
    ensure_awq_runtime_available(require_cuda=True)

    def _run_in_thread() -> None:
        awq_module = importlib.import_module("awq")
        transformers_module = importlib.import_module("transformers")
        auto_awq_class = getattr(awq_module, "AutoAWQForCausalLM", None)
        auto_tokenizer = getattr(transformers_module, "AutoTokenizer", None)
        if auto_awq_class is None or auto_tokenizer is None:
            raise LegacyAwqRuntimeUnavailableError(
                "AWQ converter runtime is unavailable: AutoAWQForCausalLM or AutoTokenizer is missing."
            )

        output_model_dir.mkdir(parents=True, exist_ok=True)
        quant_config = awq_quant_config_for_quantization(quantization)
        try:
            model = auto_awq_class.from_pretrained(
                str(source_model_dir),
                low_cpu_mem_usage=True,
                use_cache=False,
            )
            tokenizer = auto_tokenizer.from_pretrained(
                str(source_model_dir),
                trust_remote_code=True,
            )
            model.quantize(tokenizer, quant_config=quant_config)
            model.save_quantized(str(output_model_dir), safetensors=True)
            tokenizer.save_pretrained(str(output_model_dir))
        except LegacyAwqConversionError:
            raise
        except Exception as exc:
            raise LegacyAwqConversionError(f"AWQ conversion failed: {exc}") from exc

    await asyncio.to_thread(_run_in_thread)

