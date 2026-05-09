"""\u5b89\u5168\u6578\u5b78\u904b\u7b97\u5de5\u5177\u3002"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from mochi.tools.base import BaseTool, ToolResult

# ---------------------------------------------------------------------------
# \u5b89\u5168\u904b\u7b97\u5668\uff08\u57fa\u65bc AST \u8a55\u4f30\uff0c\u4e0d\u4f7f\u7528 eval\uff09
# ---------------------------------------------------------------------------

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "int": int,
    "float": float,
    # math \u5e38\u7528\u51fd\u6578
    "sqrt": math.sqrt,
    "pow": math.pow,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "factorial": math.factorial,
    "gcd": math.gcd,
}

_SAFE_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

_MAX_POWER = 10000


def _safe_eval_expr(node: ast.expr) -> Any:
    """\u905e\u8ff4\u8a55\u4f30 AST \u7bc0\u9ede\uff08\u53ea\u5141\u8a31\u5b89\u5168\u904b\u7b97\uff09\u3002"""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.Name):
        name = node.id
        if name in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[name]
        raise ValueError(f"Unknown variable: {name}")

    if isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval_expr(node.operand))

    if isinstance(node, ast.BinOp):
        op_func = _SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _safe_eval_expr(node.left)
        right = _safe_eval_expr(node.right)
        # \u9632\u6b62\u904e\u5927\u7684 pow \u904b\u7b97
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > _MAX_POWER:
            raise ValueError(f"Exponent too large: {right} (max {_MAX_POWER})")
        return op_func(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported.")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Unsupported function: {func_name}")
        args = [_safe_eval_expr(arg) for arg in node.args]
        return _SAFE_FUNCTIONS[func_name](*args)

    raise ValueError(f"Unsupported expression: {type(node).__name__}")


def safe_evaluate(expression: str) -> Any:
    """\u5b89\u5168\u8a55\u4f30\u6578\u5b78\u904b\u7b97\u5f0f\u3002"""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    return _safe_eval_expr(tree.body)


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------


class CalculatorTool(BaseTool):
    """\u5b89\u5168\u7684\u6578\u5b78\u904b\u7b97\u5de5\u5177\u3002"""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression and return the result. "
            "Supports arithmetic (+, -, *, /, //, %, **), math functions "
            "(sqrt, log, sin, cos, tan, abs, round, ceil, floor, factorial, gcd), "
            "and constants (pi, e, tau). Use this instead of guessing calculations. "
            "Example: 'sqrt(144) + 2**10' returns 1036.0"
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Mathematical expression to evaluate. "
                        "Examples: '2+3*4', 'sqrt(144)', 'log(100, 10)', 'sin(pi/2)'"
                    ),
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """\u8a08\u7b97\u6578\u5b78\u904b\u7b97\u5f0f\u3002"""
        expression = str(kwargs.get("expression", "")).strip()
        if not expression:
            return ToolResult(error="`expression` must not be empty.")

        try:
            result = safe_evaluate(expression)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            return ToolResult(
                error=f"Calculation error: {exc}",
                suggestion="Check the expression syntax and ensure values are within valid ranges.",
            )

        return ToolResult(
            output=result,
            metadata={"expression": expression, "result_type": type(result).__name__},
        )
