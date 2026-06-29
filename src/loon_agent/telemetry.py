"""OpenTelemetry wiring for gen_ai observability.

Auto-instruments LangChain/LangGraph via OpenInference so the callback tree becomes a span
tree: an ``LLM`` span per model call (capturing the request params, input/output messages
and ``llm.token_count.*`` when the backend reports usage), nested ``TOOL`` and ``AGENT``
spans. OpenInference uses its own OTel-compatible taxonomy (``openinference.span.kind``,
``llm.*``/``tool.*`` attributes) which maps onto the OTel gen_ai semantic conventions and
exports to any OTLP collector or Arize Phoenix.

Gated by ``LOON_OTEL``:
* ``off`` (default) — no-op, zero overhead.
* ``console`` — pretty-print spans to stdout (handy for the learning loop).
* ``otlp`` — export over OTLP to a collector (``OTEL_EXPORTER_OTLP_ENDPOINT``).

Set ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`` for the newest attributes.
"""

from __future__ import annotations

from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from .config import Settings

_configured = False


def setup_telemetry(settings: Settings) -> None:
    """Configure tracing + LangChain instrumentation once, per ``settings.otel``."""
    global _configured
    mode = (settings.otel or "off").lower()
    if _configured or mode == "off":
        return

    resource = Resource.create({SERVICE_NAME: "loon-agent"})
    provider = TracerProvider(resource=resource)

    if mode == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif mode == "otlp":
        # grpc exporter; reads OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4317).
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        raise ValueError(f"Unknown LOON_OTEL mode {mode!r}; use off | console | otlp")

    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=provider)
    _configured = True
