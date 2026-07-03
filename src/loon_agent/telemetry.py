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
* ``otlp`` — export traces, metrics, **and logs** over OTLP to a collector
  (``OTEL_EXPORTER_OTLP_ENDPOINT``). Python ``logging`` records (loon's ``INFO`` turn/skill
  logs, library ``WARNING``s) are bridged to OTLP log records via a root ``LoggingHandler``,
  so the same collector that stores traces also stores loon's logs.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.context import Context
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from .config import Settings

_configured = False

# The donned persona's attribution trio (persona.id / persona.version /
# persona.identity_hash — Phase 3 §6). Empty at baseline: no masque, no attributes.
_persona_attributes: dict[str, str] = {}


def set_persona_attributes(attributes: Mapping[str, str] | None) -> None:
    """Stamp (or, with None, clear) the active persona's OTEL attribution.

    Called by the runtime at don/doff. Safe to call with telemetry off — the
    dict just never reaches a span processor.
    """
    _persona_attributes.clear()
    if attributes:
        _persona_attributes.update(attributes)


class PersonaSpanProcessor(SpanProcessor):
    """Stamps persona.* onto every span started while a masque is donned.

    Attribution is event-carried on the host's existing pipeline (Phase 3 §6):
    masques provides the three attributes, loon provides the spans.
    """

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        for key, value in _persona_attributes.items():
            span.set_attribute(key, value)


def build_log_handler(
    resource: Resource, exporter: LogExporter
) -> tuple[LoggerProvider, LoggingHandler]:
    """Build a (provider, handler) pair that ships log records to ``exporter``.

    The handler is bound to its own provider (explicit, not the process global) so this is
    unit-testable without touching global state — the same pattern the exec-audit tests use
    to dodge the once-per-process provider landmine.
    """
    provider = LoggerProvider(resource=resource)
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    handler = LoggingHandler(level=logging.INFO, logger_provider=provider)
    return provider, handler


def _install_log_export(resource: Resource, exporter: LogExporter) -> None:
    """Wire the OTLP log handler onto the root logger + set the global logger provider."""
    provider, handler = build_log_handler(resource, exporter)
    set_logger_provider(provider)  # global, for any direct OTLP log emitters
    logging.getLogger().addHandler(handler)


def setup_telemetry(settings: Settings) -> None:
    """Configure tracing + metrics + logs + instrumentation once, per ``settings.otel``."""
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
    tracer_provider.add_span_processor(PersonaSpanProcessor())

    metric_reader: MetricReader
    if mode == "console":
        tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        metric_reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(), export_interval_millis=60_000
        )
    else:  # otlp
        # Exporters read OTEL_EXPORTER_OTLP_ENDPOINT. Protocol comes from
        # OTEL_EXPORTER_OTLP_PROTOCOL (grpc | http/protobuf); when unset, infer it from
        # the endpoint's conventional port — :4318 is OTLP/HTTP, :4317 (or none) is gRPC.
        protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "").strip().lower()
        if not protocol:
            endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
            protocol = "http/protobuf" if ":4318" in endpoint else "grpc"

        if protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        else:
            from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
        # Bridge Python logging -> OTLP logs so the collector's logs pipeline gets loon's
        # logs alongside its traces/metrics (same OTEL_EXPORTER_OTLP_ENDPOINT).
        _install_log_export(resource, OTLPLogExporter())

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
