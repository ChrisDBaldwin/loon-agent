"""Masque identities: YAML lenses donned per skill step or by the chat agent.

Schema-compatible with https://github.com/ChrisDBaldwin/masques (the ``name`` /
``lens`` / ``context`` subset — attributes, knowledge, access etc. are ignored). Skills
say *what to do*; a masque says *who is doing it*: its lens (+ context) becomes the
system prompt of the LLM call that dons it. Point ``LOON_MASQUES_DIR`` at an external
personas catalog to reuse existing identities; loon-local files in ``masques/`` win on
name collisions.

Lookup is lenient by design: a missing masque logs a warning and the step runs bare
rather than aborting a long research run over a costume.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# <name>.yaml (loon-local convention) and <name>.masque.yaml (masques repo convention).
_FILE_PATTERNS = ("{name}.yaml", "{name}.yml", "{name}.masque.yaml")


@dataclass(frozen=True)
class Masque:
    name: str
    lens: str
    context: str = ""

    def system_block(self) -> str:
        parts = [self.lens.strip()]
        if self.context.strip():
            parts.append(self.context.strip())
        return "\n\n".join(parts)


class MasqueLoader:
    """Resolves masque names to system-prompt blocks across a list of directories."""

    def __init__(self, dirs: Sequence[Path | str]) -> None:
        self.dirs = [Path(d) for d in dirs]
        self._cache: dict[str, Masque | None] = {}

    def load(self, name: str) -> Masque | None:
        if name not in self._cache:
            self._cache[name] = self._read(name)
        return self._cache[name]

    def block(self, name: str) -> str | None:
        """The ``SkillRunner`` masque_loader hook: lens+context text, or None if absent."""
        masque = self.load(name)
        return masque.system_block() if masque else None

    def _read(self, name: str) -> Masque | None:
        for path in self._candidates(name):
            if not path.is_file():
                continue
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                logger.warning("masque %s is not valid YAML (%s); skipping", path, exc)
                continue
            if not isinstance(data, dict) or not str(data.get("lens", "")).strip():
                logger.warning("masque %s has no 'lens'; skipping", path)
                continue
            return Masque(
                name=str(data.get("name", name)),
                lens=str(data["lens"]),
                context=str(data.get("context", "") or ""),
            )
        logger.warning("masque %r not found in %s; running bare", name, self.dirs)
        return None

    def _candidates(self, name: str) -> Iterable[Path]:
        for directory in self.dirs:
            for pattern in _FILE_PATTERNS:
                yield directory / pattern.format(name=name)
