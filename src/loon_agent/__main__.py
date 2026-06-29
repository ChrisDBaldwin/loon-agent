"""Entry point: ``python -m loon_agent`` launches the CLI REPL."""

from __future__ import annotations

from .adapters.cli import run_cli


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
