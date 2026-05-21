"""Legacy AWQ backend kept out of the active Mochi runtime path."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from mochi.backends.safetensors import PipelineFactory
from mochi.backends.safetensors import SafetensorsBackend
from mochi.backends.tool_call_simulator import ToolCallSimulator
from mochi.backends.types import ModelInfo


def is_awq_model_dir(model_dir: str | Path) -> bool:
    """判斷目錄是否為 AWQ 量化模型輸出。"""
    model_path = Path(model_dir).expanduser().resolve(strict=False)
    if not model_path.is_dir():
        return False
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return False
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    quant_cfg = payload.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        return False
    quant_method = quant_cfg.get("quant_method")
    return isinstance(quant_method, str) and quant_method.strip().lower() == "awq"


class AWQBackend(SafetensorsBackend):
    """AWQ 本地模型後端（legacy，重用 transformers pipeline 路徑）。"""

    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        *,
        pipeline_factory: PipelineFactory | None = None,
        tool_call_simulator: ToolCallSimulator | None = None,
    ) -> None:
        super().__init__(
            model_dir=model_dir,
            device=device,
            torch_dtype=torch_dtype,
            pipeline_factory=pipeline_factory,
            tool_call_simulator=tool_call_simulator,
        )
        if self._dependency_error is None:
            self._dependency_error = self._probe_awq_dependency_error()

    def get_model_info(self) -> ModelInfo:
        """回傳 AWQ 後端模型資訊。"""
        info = super().get_model_info()
        metadata = dict(info.metadata)
        metadata["quantization"] = "awq"
        metadata["awq_detected"] = is_awq_model_dir(self.model_dir)
        return ModelInfo(
            name=info.name,
            backend_type="awq",
            context_length=info.context_length,
            supports_tool_calling=info.supports_tool_calling,
            metadata=metadata,
        )

    async def health_check(self) -> bool:
        """檢查依賴、目錄與 AWQ 標記是否可用。"""
        return (
            self._dependency_error is None
            and Path(self.model_dir).is_dir()
            and is_awq_model_dir(self.model_dir)
        )

    def _probe_awq_dependency_error(self) -> str | None:
        """檢查 AWQ runtime 是否可用。"""
        if not is_awq_model_dir(self.model_dir):
            return (
                "Local model directory is not an AWQ model. "
                "Expected config.json with quantization_config.quant_method=awq."
            )

        if importlib.util.find_spec("awq") is None:
            return (
                "AWQ runtime dependency is missing: autoawq (`awq` module). "
                "Install with `uv sync --extra awq`."
            )
        return None

