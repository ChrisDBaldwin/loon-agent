"""Persona OTEL attribution: spans carry persona.* while donned, nothing at baseline.

Builds a private TracerProvider (never touching the global one) so this stays
clear of the setup_telemetry one-shot global state.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from loon_agent.telemetry import PersonaSpanProcessor, set_persona_attributes


@pytest.fixture(autouse=True)
def _baseline_after():
    yield
    set_persona_attributes(None)


def test_spans_carry_persona_trio_while_donned_and_none_at_baseline() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(PersonaSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    set_persona_attributes(
        {"persona.id": "Analyst", "persona.version": "0.1.0", "persona.identity_hash": "abc123"}
    )
    with tracer.start_as_current_span("donned-turn"):
        pass
    set_persona_attributes(None)
    with tracer.start_as_current_span("baseline-turn"):
        pass

    donned, baseline = exporter.get_finished_spans()
    assert donned.attributes["persona.id"] == "Analyst"
    assert donned.attributes["persona.version"] == "0.1.0"
    assert donned.attributes["persona.identity_hash"] == "abc123"
    assert all(not key.startswith("persona.") for key in baseline.attributes)
