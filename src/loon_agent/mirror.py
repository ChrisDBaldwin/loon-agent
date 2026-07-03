"""The session mirror: ``data/masque.session.yaml`` (Phase 2 §4 evolved schema).

Written on every don/doff so external tooling (an `/id`-style introspector, the
masques audience) reads one format across hosts. loon writes it at the ``bind``
tier — the first mirror ever produced where ``capability_plan.host_apply`` is an
enforced tier and bindings carry ``enforced: true`` + ``effective_*`` names.

Only mirror-safe projections travel: credentials appear as alias+scheme+status
(``BoundRef.to_dict``), never material — the same allowlist discipline as
masques' own ``_mirrorable_view``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from masques_core import BoundRef, CapabilityPlan, Persona

MIRROR_FILENAME = "masque.session.yaml"


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def active_record(
    persona: Persona,
    intent: str | None,
    plan: CapabilityPlan,
    bound_refs: list[BoundRef],
) -> dict[str, Any]:
    """The mirror's ``active`` mapping — an explicit allowlist, never a __dict__ dump."""
    config_error = None
    if persona.config_error is not None:
        config_error = {
            "path": persona.config_error.path,
            "reason": persona.config_error.reason,
            "remediation": persona.config_error.remediation,
        }
    return {
        "name": persona.name,
        "source": persona.identity_source,
        "version": persona.version.raw,
        "config_source": persona.config_source,
        "config_error": config_error,
        "intent": intent,
        "donned_at": utc_now(),
        "bound_refs": [ref.to_dict() for ref in bound_refs],
        "capability_plan": plan.to_dict(),
    }


def write_mirror(
    data_dir: Path,
    active: dict[str, Any] | None,
    previous: dict[str, Any] | None,
) -> Path:
    """Write the mirror atomically-enough for a single-process host."""
    path = Path(data_dir) / MIRROR_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"active": active, "previous": previous}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
