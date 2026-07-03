"""Lazy persona credential resolution (Phase 2 §4 semantics, loon host).

Nothing resolves at don — the resolver only learns which persona's bindings are
resolvable. Material is fetched on first use by an allowed consumer, cached
in-process keyed by (persona ref, alias), and zeroed at doff. Audience is
fail-closed: a binding whose ``scope.audience`` does not name the consumer is
unresolvable, including the empty (unauthored) audience. env:// is static, so
doff only forgets the in-process copy — the environment variable itself is
untouched.

Material must never appear in logs, mirrors, or returned dicts; this module
returns the bare string to the calling tool and nothing else.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from masques_core import EnvAdapter

if TYPE_CHECKING:
    from masques_core import Persona, PersonaRef, ResolvedSecret, SecretPort

logger = logging.getLogger(__name__)

# The audience token for loon's native tool registry (mirrors HOST_NATIVE_SERVER).
HOST_AUDIENCE = "host"


class CredentialResolver:
    """The one path from a donned persona's CredentialBindings to secret material."""

    def __init__(self, adapter: SecretPort | None = None) -> None:
        self._adapter = adapter or EnvAdapter()
        self._persona: Persona | None = None
        self._cache: dict[tuple[PersonaRef, str], ResolvedSecret] = {}

    def activate(self, persona: Persona) -> None:
        """Swap resolvable bindings to the newly-donned persona. Resolves nothing."""
        self.deactivate()
        self._persona = persona

    def deactivate(self) -> None:
        """Doff: zero every in-process copy and forget the bindings."""
        for secret in self._cache.values():
            secret.material = None
        self._cache.clear()
        self._persona = None

    def material(self, alias: str, audience: str = HOST_AUDIENCE) -> str | None:
        """Secret material for ``alias``, or None with a warning — never raises.

        Degrading rather than raising keeps a long run alive over a missing
        credential; the tool sees None and reports its own failure.
        """
        persona = self._persona
        if persona is None or persona.config is None:
            logger.warning("credential %r requested but no persona bindings are donned", alias)
            return None
        binding = persona.config.credential(alias)
        if binding is None:
            logger.warning(
                "credential %r is not bound by persona %r", alias, persona.name
            )
            return None
        if audience not in binding.scope.audience:
            logger.warning(
                "credential %r is not scoped to audience %r (fail-closed)", alias, audience
            )
            return None

        key = (persona.ref, alias)
        secret = self._cache.get(key)
        if secret is None:
            secret = self._adapter.resolve(binding.ref, persona.ref, scope=binding.scope)
            self._cache[key] = secret
        if secret.status != "resolved":
            logger.warning("credential %r did not resolve (%s)", alias, secret.status)
            return None
        return secret.material
