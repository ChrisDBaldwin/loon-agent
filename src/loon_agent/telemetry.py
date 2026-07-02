"""OpenTelemetry wiring for gen_ai observability.

Two complementary layers, both exporting to the same provider:

* **OTel gen_ai semantic conventions** (the official ones) — the ``openai`` SDK that
  ``ChatOpenAI`` wraps is instrumented with ``opentelemetry-instrumentation-openai-v2``,
  so every model call (chat loop and skill steps alike) emits a ``chat {model}`` span
  with ``gen_ai.operation.name`` / ``gen_ai.request.model`` / ``gen_ai.usage.*`` /
  ``gen_ai.response.finish_reasons``, plus the ``gen_ai.client.token.usage`` and
  ``gen_ai.client.operation.duration`` metrics. The skill engine adds its own
  ``invoke_agent {skill}`` / ``execute_tool {tool}`` spans per the gen_ai agent
  conventions (see ``skills/engine.py``).
* **OpenInference LangChain instrumentation** — turns the LangChain/LangGraph callback
  tree into nested ``AGENT``/``CHAIN``/``LLM``/``TOOL`` spans (its own taxonomy;
  maps onto gen_ai and renders nicely in Arize Phoenix). The gen_ai spans nest under it.

``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`` is defaulted here so the
newest gen_ai attribute set is used. Prompt/completion content is NOT captured unless
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`` (privacy by default).

Gated by ``LOON_OTEL``:
* ``off`` (default) — no-op, zero overhead (engine spans become no-op proxies).
* ``console`` — pretty-print spans (and metrics each 60s) to stdout.
* ``otlp`` — export over OTLP to a collector (``OTEL_EXPORTER_OTLP_ENDPOINT``).
"""

from __future__ import annotations

import os

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
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
    """Configure tracing + metrics + instrumentation once, per ``settings.otel``."""
    global _configured
    mode = (settings.otel or "off").lower()
    if _configured or mode == "off":
        return
    if mode not in ("console", "otlp"):
        raise ValueError(f"Unknown LOON_OTEL mode {mode!r}; use off | console | otlp")

    # Instrumentors read this at setup: opt into the newest gen_ai attribute set.
    os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")

    resource = Resource.create({SERVICE_NAME: "loon-agent"})
    tracer_provider = TracerProvider(resource=resource)

    metric_reader: MetricReader
    if mode == "console":
        tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        metric_reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(), export_interval_millis=60_000
        )
    else:  # otlp
        # grpc exporters; read OTEL_EXPORTER_OTLP_ENDPOINT (default http://localhost:4317).
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())

    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])

    # Globals: the skill engine's tracer proxies resolve against these.
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)

    # Official gen_ai semconv spans + token/duration metrics on the openai SDK.
    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)

    # LangChain/LangGraph callback tree (OpenInference taxonomy) for loop structure.
    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _configured = True
