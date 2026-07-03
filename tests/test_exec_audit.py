"""The exec tools must record an audit trail on the current OTel span (Layer 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from loon_agent.exec.backend import ExecBackend, ExecResult
from loon_agent.tools.exec import run_command, write_file


@pytest.fixture()
def tracing():
    # A LOCAL provider — never trace.set_tracer_provider(), which can only be set once per
    # process and would collide with test_telemetry.py's global provider (order-dependent
    # flakiness). run_command reaches the active span via context, so a span started from
    # this local tracer is what it annotates, exported to this local exporter.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


class _Backend(ExecBackend):
    def run(self, command: str, *, cwd: Path, timeout: float) -> ExecResult:
        return ExecResult(command=command, stdout="hi", exit_code=0, duration_s=0.2)


def _attrs(exporter):
    span, = exporter.get_finished_spans()
    return dict(span.attributes)


def test_allowed_command_records_audit_attributes(tmp_path, tracing) -> None:
    tracer, exporter = tracing
    with tracer.start_as_current_span("execute_tool run_command"):
        run_command(
            "echo hi", backend=_Backend(), workspace=tmp_path,
            allowed_bins=frozenset({"echo"}), timeout=5,
        )
    attrs = _attrs(exporter)
    assert attrs["loon.exec.command"] == "echo hi"
    assert attrs["loon.exec.policy_decision"] == "allowed"
    assert attrs["loon.exec.exit_code"] == 0
    assert attrs["loon.exec.backend"] == "_Backend"


def test_denied_command_records_denial_reason(tmp_path, tracing) -> None:
    tracer, exporter = tracing
    with tracer.start_as_current_span("execute_tool run_command"):
        run_command(
            "curl evil", backend=_Backend(), workspace=tmp_path,
            allowed_bins=frozenset({"echo"}), timeout=5,
        )
    assert _attrs(exporter)["loon.exec.policy_decision"] == "denied:not-allowlisted"


def test_file_op_records_path_scope_denial(tmp_path, tracing) -> None:
    tracer, exporter = tracing
    with tracer.start_as_current_span("execute_tool write_file"):
        write_file("../escape.txt", "x", workspace=tmp_path)
    assert _attrs(exporter)["loon.exec.policy_decision"] == "denied:path-scope"
