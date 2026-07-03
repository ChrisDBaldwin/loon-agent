"""Entry point: ``python -m loon_agent [cli|telegram|web]`` (default: cli)."""

from __future__ import annotations

import sys

_ADAPTERS = ("cli", "telegram", "web")


def main() -> None:
    adapter = sys.argv[1] if len(sys.argv) > 1 else "cli"
    if adapter == "cli":
        from .adapters.cli import run_cli

        run_cli()
    elif adapter == "telegram":
        from .adapters.telegram import run_telegram

        run_telegram()
    elif adapter == "web":
        from .adapters.web import run_web

        run_web()
    else:
        raise SystemExit(f"unknown adapter {adapter!r} (expected one of: {', '.join(_ADAPTERS)})")


if __name__ == "__main__":
    main()
