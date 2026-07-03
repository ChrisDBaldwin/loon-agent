"""OTLP log export: Python logging records are bridged to OTLP log records.

Uses a local LoggerProvider via build_log_handler (no set_logger_provider / global state),
so this file is order-independent w.r.t. the other telemetry tests — the once-per-process
provider landmine that has bitten the trace/metric tests.
"""

from __future__ import annotations

import logging

from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

from loon_agent.telemetry import build_log_handler

_RESOURCE = Resource.create({SERVICE_NAME: "loon-agent"})


def _emit(handler, provider, level, message):
    log = logging.getLogger("loon_agent.test_logs")
    log.setLevel(logging.INFO)
    log.propagate = False  # isolate from root handlers other tests may have attached
    log.handlers = [handler]
    log.log(level, message)
    provider.force_flush()


def test_info_log_is_exported_as_otlp_record() -> None:
    exporter = InMemoryLogRecordExporter()
    provider, handler = build_log_handler(_RESOURCE, exporter)
    _emit(handler, provider, logging.INFO, "turn done (session=abc, reply=42 chars)")

    records = exporter.get_finished_logs()
    assert len(records) == 1
    assert "turn done" in str(records[0].log_record.body)


def test_service_name_resource_is_attached() -> None:
    exporter = InMemoryLogRecordExporter()
    provider, handler = build_log_handler(_RESOURCE, exporter)
    _emit(handler, provider, logging.WARNING, "something noisy")

    record = exporter.get_finished_logs()[0]
    assert record.resource.attributes.get("service.name") == "loon-agent"


def test_debug_below_handler_level_is_dropped() -> None:
    exporter = InMemoryLogRecordExporter()
    provider, handler = build_log_handler(_RESOURCE, exporter)  # handler is INFO
    _emit(handler, provider, logging.DEBUG, "too chatty")

    assert exporter.get_finished_logs() == ()
