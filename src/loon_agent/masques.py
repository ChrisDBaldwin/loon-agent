"""Masque identities: a thin loon shim over ``masques_core``.

``masques_core.resolve()`` is the one authoritative resolver (pinned Identity +
optional ``<name>.persona.yaml`` sidecar), but it only knows its own catalog
(``~/.masques`` + the masques repo's bundled personas) and only reads
``*.masque.yaml``. loon keeps two local conveniences on top:

* plain ``<name>.yaml`` files in ``masques/`` (the loon-local convention) still
  load, with ``version`` optional — these parse into the same Persona shape,
  and a sidecar sitting next to the identity file is honored;
* lookup stays lenient: a missing or malformed masque logs a warning and
  returns ``None``, so a skill step runs bare rather than aborting a long
  research run over a costume.

Local dirs win on name collisions; anything else falls through to
``masques_core``'s own search paths. Point ``LOON_MASQUES_DIR`` at an external
personas catalog to extend the local list.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from masques_core import (
    Identity,
    MasqueError,
    Persona,
    PersonaRef,
    Version,
    parse_persona_config,
)
from masques_core import (
    resolve as core_resolve,
)

logger = logging.getLogger(__name__)

# <name>.yaml (loon-local convention) and <name>.masque.yaml (masques repo convention).
_FILE_PATTERNS = ("{name}.yaml", "{name}.yml", "{name}.masque.yaml")


@dataclass(frozen=True)
class Masque:
    """The lens-only projection skill steps don (``SkillRunner`` needs no bindings)."""

    name: str
    lens: str
    context: str = ""

    def system_block(self) -> str:
        parts = [self.lens.strip()]
        if self.context.strip():
            parts.append(self.context.strip())
        return "\n\n".join(parts)


class MasqueLoader:
    """Resolves masque names to Personas (and system-prompt blocks) across dirs."""

    def __init__(self, dirs: Sequence[Path | str]) -> None:
        self.dirs = [Path(d) for d in dirs]
        self._cache: dict[str, Persona | None] = {}

    def resolve(self, name: str) -> Persona | None:
        """The full Persona (identity + sidecar bindings), or None — never raises."""
        if name not in self._cache:
            self._cache[name] = self._resolve(name)
        return self._cache[name]

    def load(self, name: str) -> Masque | None:
        persona = self.resolve(name)
        if persona is None:
            return None
        return Masque(name=persona.name, lens=persona.lens, context=persona.context or "")

    def block(self, name: str) -> str | None:
        """The ``SkillRunner`` masque_loader hook: lens+context text, or None if absent."""
        masque = self.load(name)
        return masque.system_block() if masque else None

    def _resolve(self, name: str) -> Persona | None:
        for path in self._candidates(name):
            if not path.is_file():
                continue
            persona = self._local_persona(path, name)
            if persona is not None:
                return persona
        try:
            return core_resolve(name)
        except MasqueError:
            logger.warning(
                "masque %r not found in %s or the masques_core catalog; running bare",
                name,
                self.dirs,
            )
            return None

    def _local_persona(self, path: Path, name: str) -> Persona | None:
        """Parse a loon-local file into a masques_core Persona.

        Lenient where ``core_resolve`` is strict: only ``lens`` is required
        (``version`` defaults to 0.0.0). The sidecar is looked up next to the
        identity file; a malformed sidecar degrades to config=None, never
        denying the don.
        """
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            logger.warning("masque %s is not valid YAML (%s); skipping", path, exc)
            return None
        if not isinstance(data, dict) or not str(data.get("lens", "")).strip():
            logger.warning("masque %s has no 'lens'; skipping", path)
            return None

        identity = Identity(
            name=str(data.get("name", name)),
            version=Version.parse(data.get("version", "0.0.0")),
            lens=str(data["lens"]),
            context=str(data["context"]) if data.get("context") is not None else None,
            attributes=data.get("attributes") or {},
            rubric=data.get("rubric"),
            spinner_verbs=data.get("spinnerVerbs"),
            raw=data,
        )

        config = config_source = config_error = None
        sidecar = path.parent / f"{name}.persona.yaml"
        if sidecar.is_file():
            config, config_error = parse_persona_config(sidecar, identity.name)
            if config is not None:
                config_source = "local"
            if config_error is not None:
                logger.warning(
                    "sidecar %s is invalid (%s); donning without bindings",
                    config_error.path,
                    config_error.reason,
                )

        return Persona(
            identity=identity,
            ref=PersonaRef(identity.name, identity.version),
            identity_source="local",
            path=path,
            config=config,
            config_source=config_source,
            config_error=config_error,
        )

    def _candidates(self, name: str) -> Iterable[Path]:
        for directory in self.dirs:
            for pattern in _FILE_PATTERNS:
                yield directory / pattern.format(name=name)
