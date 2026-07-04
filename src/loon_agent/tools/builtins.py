"""Built-in tools bound into the chat loop (``DEFAULT_TOOLS``).

These are the tools reachable by the model on *every* chat turn. That is exactly why
**exec/file tools must never be added here** — the chat loop also holds untrusted content
returned by ``search_web``/``read_web_page`` below, so an exec tool sharing this loop would
turn a prompt-injecting web page into remote code execution. Sandboxed exec lives only in
the skill registry, behind the deliberately-invoked ``/code`` skill (see ``app.py``).

Web search/fetch are read-only, so they are safe to expose conversationally — they wrap the
same functions the research skill uses (``tools/web.py``).

The chat loop also carries the site-management tools from ``tools/site.py`` (appended in
``app.build_runtime``, not here, because they bind to the deployment's web root). They can
share the loop because their writes are markdown-rendered pages confined to the web root —
see that module's docstring for the full rationale.
"""

from __future__ import annotations

import ast
import datetime as _dt
import operator as _op

from langchain_core.tools import tool

from .web import fetch_page, web_search

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


@tool
def search_web(query: str) -> str:
    """Search the web for current information and return the top results.

    Use this to look things up, research a topic, or answer questions that need
    up-to-date information. Returns a list of results (title, URL, snippet).
    """
    results = web_search(query)
    if not results:
        return "no results (search unavailable or nothing found)"
    return "\n\n".join(str(r) for r in results)


@tool
def read_web_page(url: str) -> str:
    """Fetch a single web page by URL and return its readable text.

    Use this after search_web to read a specific page in full. Returns the extracted
    article text, or an error line if the page could not be fetched.
    """
    page = fetch_page(url)
    return str(page)


DEFAULT_TOOLS = [get_current_time, calculator, search_web, read_web_page]
