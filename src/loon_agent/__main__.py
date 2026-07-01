"""Entry point: ``python -m loon_agent [cli|telegram]`` (default: cli)."""

from __future__ import annotations

import sys

_ADAPTERS = ("cli", "telegram")


def main() -> None:
    adapter = sys.argv[1] if len(sys.argv) > 1 else "cli"
    if adapter == "cli":
        from .adapters.cli import run_cli

        run_cli()
    elif adapter == "telegram":
        from .adapters.telegram import run_telegram

        run_telegram()
    else:
        raise SystemExit(f"unknown adapter {adapter!r} (expected one of: {', '.join(_ADAPTERS)})")


if __name__ == "__main__":
    main()
