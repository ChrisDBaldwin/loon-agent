"""A couple of trivial built-in tools to exercise the tool-calling loop.

These exist mainly so the hand-rolled ReAct loop has something concrete to call and so
tool-calling can be observed against each backend. Real tools come later.
"""

from __future__ import annotations

import ast
import datetime as _dt
import operator as _op

from langchain_core.tools import tool

# Whitelisted operators for the calculator — never use eval() on model output.
_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}
_UNARY_OPS = {ast.UAdd: _op.pos, ast.USub: _op.neg}


@tool
def get_current_time() -> str:
    """Return the current local date and time in ISO 8601 format."""
    return _dt.datetime.now().isoformat(timespec="seconds")


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (e.g. "3 * (4 + 5)").

    Supports + - * / // % ** and parentheses on numbers only.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        return f"error: {exc}"


DEFAULT_TOOLS = [get_current_time, calculator]
