"""CredentialResolver: lazy env:// resolution, audience fail-closed, zeroed at doff."""

from __future__ import annotations

from pathlib import Path

from masques_core import (
    CredentialBinding,
    CredentialScope,
    Identity,
    Persona,
    PersonaConfig,
    PersonaRef,
    SecretRef,
    Version,
)

from loon_agent.credentials import CredentialResolver


def _persona(*credentials: CredentialBinding) -> Persona:
    identity = Identity(name="Analyst", version=Version.parse("1.0.0"), lens="terse")
    return Persona(
        identity=identity,
        ref=PersonaRef(identity.name, identity.version),
        identity_source="local",
        path=Path("analyst.yaml"),
        config=PersonaConfig(persona="Analyst", credentials=list(credentials)),
    )


def _binding(audience: tuple[str, ...] = ("host",)) -> CredentialBinding:
    return CredentialBinding(
        alias="search_api",
        ref=SecretRef.parse("env://LOON_TEST_SECRET"),
        scope=CredentialScope(audience=audience),
    )


def test_resolves_lazily_and_caches(monkeypatch) -> None:
    monkeypatch.setenv("LOON_TEST_SECRET", "s3kr1t-material")
    resolver = CredentialResolver()
    resolver.activate(_persona(_binding()))

    assert resolver.material("search_api") == "s3kr1t-material"
    # Cached: the env var can vanish and the in-process copy still serves.
    monkeypatch.delenv("LOON_TEST_SECRET")
    assert resolver.material("search_api") == "s3kr1t-material"


def test_audience_is_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("LOON_TEST_SECRET", "s3kr1t-material")
    resolver = CredentialResolver()

    resolver.activate(_persona(_binding(audience=("some-mcp-server",))))
    assert resolver.material("search_api") is None  # host not in audience

    resolver.activate(_persona(_binding(audience=())))
    assert resolver.material("search_api") is None  # unauthored audience = deny-all


def test_unbound_alias_and_bare_baseline_return_none(monkeypatch) -> None:
    monkeypatch.setenv("LOON_TEST_SECRET", "s3kr1t-material")
    resolver = CredentialResolver()

    assert resolver.material("search_api") is None  # nothing donned

    resolver.activate(_persona(_binding()))
    assert resolver.material("other_alias") is None  # alias not bound


def test_deactivate_zeroes_in_process_material(monkeypatch) -> None:
    monkeypatch.setenv("LOON_TEST_SECRET", "s3kr1t-material")
    resolver = CredentialResolver()
    persona = _persona(_binding())
    resolver.activate(persona)
    assert resolver.material("search_api") == "s3kr1t-material"
    (secret,) = resolver._cache.values()

    resolver.deactivate()

    assert secret.material is None  # the cached copy itself is zeroed, not just dropped
    assert resolver.material("search_api") is None  # and nothing is resolvable anymore


def test_missing_env_var_degrades_with_none(monkeypatch) -> None:
    monkeypatch.delenv("LOON_TEST_SECRET", raising=False)
    resolver = CredentialResolver()
    resolver.activate(_persona(_binding()))

    assert resolver.material("search_api") is None
