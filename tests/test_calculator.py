"""calculator \u5de5\u5177\u6e2c\u8a66\u3002"""

from __future__ import annotations

import math

import pytest

from mochi.tools.calculator import CalculatorTool, safe_evaluate


@pytest.mark.asyncio
async def test_calculator_basic_arithmetic() -> None:
    """\u57fa\u672c\u7b97\u8853\u904b\u7b97\u3002"""
    tool = CalculatorTool()
    result = await tool.execute(expression="2 + 3 * 4")
    assert result.error is None
    assert result.output == 14


@pytest.mark.asyncio
async def test_calculator_math_functions() -> None:
    """math \u51fd\u6578\u3002"""
    tool = CalculatorTool()
    result = await tool.execute(expression="sqrt(144)")
    assert result.error is None
    assert result.output == 12.0


@pytest.mark.asyncio
async def test_calculator_constants() -> None:
    """\u6578\u5b78\u5e38\u6578\u3002"""
    tool = CalculatorTool()
    result = await tool.execute(expression="round(pi, 4)")
    assert result.error is None
    assert result.output == 3.1416


@pytest.mark.asyncio
async def test_calculator_trig() -> None:
    """\u4e09\u89d2\u51fd\u6578\u3002"""
    tool = CalculatorTool()
    result = await tool.execute(expression="round(sin(pi/2), 5)")
    assert result.error is None
    assert result.output == 1.0


@pytest.mark.asyncio
async def test_calculator_division_by_zero() -> None:
    """\u9664\u4ee5\u96f6\u61c9\u56de\u50b3\u932f\u8aa4\u3002"""
    tool = CalculatorTool()
    result = await tool.execute(expression="1/0")
    assert result.error is not None
    assert "Calculation error" in result.error


@pytest.mark.asyncio
async def test_calculator_rejects_dangerous_expressions() -> None:
    """\u5371\u96aa\u8868\u9054\u5f0f\uff08import/exec\uff09\u61c9\u88ab\u62d2\u7d55\u3002"""
    tool = CalculatorTool()
    result = await tool.execute(expression="__import__('os').system('ls')")
    assert result.error is not None


@pytest.mark.asyncio
async def test_calculator_rejects_huge_exponent() -> None:
    """\u904e\u5927 exponent \u61c9\u88ab\u62d2\u7d55\u3002"""
    tool = CalculatorTool()
    result = await tool.execute(expression="2**100000")
    assert result.error is not None
    assert "too large" in result.error.lower()


@pytest.mark.asyncio
async def test_calculator_empty_expression() -> None:
    """\u7a7a\u8868\u9054\u5f0f\u3002"""
    tool = CalculatorTool()
    result = await tool.execute(expression="")
    assert result.error == "`expression` must not be empty."


def test_safe_evaluate_floor_div() -> None:
    """Floor division."""
    assert safe_evaluate("17 // 3") == 5


def test_safe_evaluate_modulo() -> None:
    """Modulo."""
    assert safe_evaluate("17 % 5") == 2


def test_safe_evaluate_nested_functions() -> None:
    """\u5de2\u72c0\u51fd\u6578\u3002"""
    result = safe_evaluate("round(log(100, 10), 2)")
    assert result == 2.0
