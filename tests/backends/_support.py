"""Shared HTTP response helpers for backend tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx


def _mock_response(data: dict) -> MagicMock:
    """建立 httpx Response mock。"""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _httpx_json_response(url: str, status_code: int, data: dict) -> httpx.Response:
    request = httpx.Request("POST", url)
    return httpx.Response(status_code, request=request, json=data)