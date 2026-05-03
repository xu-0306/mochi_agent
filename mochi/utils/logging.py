"""日誌初始化工具。"""

from __future__ import annotations

import sys

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """設定 loguru 日誌輸出格式與等級。

    Args:
        level: 日誌等級（DEBUG/INFO/WARNING/ERROR）。
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
