"""The deterministic skill engine.

Python orchestrates; the model only ever does one focused job per call. Every LLM step
gets a fresh prompt assembled from the accumulated context, with each substituted
variable truncated to its share of the input budget — context discipline is enforced
here, never trusted to prompts. Small-local-model realities are handled centrally:
think-block stripping (reasoning models), one retry per call, tolerant line parsing,
and per-item skip inside ``foreach`` fan-outs.
"""

from __future__ import annotations

import logging
import re
import string
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..textbudget import CHARS_PER_TOKEN, truncate_to_tokens
from .model import Skill, Step

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_LINE_MARKER_RE = re.compile(r"^(?:[-*•]\s+|\d+[.)]\s+)")

DEFAULT_INPUT_BUDGET = 4_000  # approx tokens per assembled prompt
DEFAULT_MAX_OUTPUT_TOKENS = 1_200  # reasoning models need headroom past the think block
DEFAULT_MAX_LINES = 8  # hard cap on parse:lines output, whatever the model says
_LLM_ATTEMPTS = 2


class SkillRunError(RuntimeError):
    """A skill run failed in a way the pipeline cannot survive."""


@dataclass
class RunResult:
    """Final context (every step's output by name) plus per-item skips."""

    outputs: dict[str, object]
    failures: list[str] = field(default_factory=list)


class SkillRunner:
    """Executes a :class:`Skill` against an LLM and a tool registry."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Mapping[str, Callable[[object], object]],
        *,
        masque_loader: Callable[[str], str | None] | None = None,
        input_budget: int = DEFAULT_INPUT_BUDGET,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        max_lines: int = DEFAULT_MAX_LINES,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.llm = llm.bind(max_tokens=max_output_tokens)
        self.tools = dict(tools)
        self.masque_loader = masque_loader
        self.input_budget = input_budget
        self.max_lines = max_lines
        self.progress = progress or (lambda message: None)

    def run(self, skill: Skill, args: Mapping[str, object]) -> RunResult:
        missing = [a for a in skill.args if a not in args]
        if missing:
            raise SkillRunError(f"skill {skill.name!r} missing args: {missing}")

        context: dict[str, object] = dict(args)
        failures: list[str] = []

        for step in skill.steps:
            self.progress(f"{skill.name}: {step.name}…")
            if step.kind == "tool" and step.tool not in self.tools:
                raise SkillRunError(
                    f"step {step.name!r}: unknown tool {step.tool!r} "
                    f"(registered: {sorted(self.tools)})"
                )
            if step.foreach is not None:
                items = context.get(step.foreach)
                if not isinstance(items, list):
                    raise SkillRunError(
                        f"step {step.name!r}: foreach {step.foreach!r} is not a list "
                        f"(got {type(items).__name__})"
                    )
                context[step.output] = self._run_foreach(skill, step, context, items, failures)
            else:
                context[step.output] = self._run_single(skill, step, context, failures=failures)
            logger.info("skill %s: step %s done", skill.name, step.name)

        return RunResult(outputs=context, failures=failures)

    # --- step execution ---------------------------------------------------------

    def _run_foreach(
        self,
        skill: Skill,
        step: Step,
        context: Mapping[str, object],
        items: list[object],
        failures: list[str],
    ) -> list[object]:
        results: list[object] = []
        for item in items:
            try:
                result = self._run_single(skill, step, context, item=item)
            except Exception as exc:  # noqa: BLE001 - per-item skip is the contract
                note = f"step {step.name!r} skipped item {_shorten(item)}: {exc}"
                logger.warning("skill %s: %s", skill.name, note)
                failures.append(note)
                continue
            # A per-item tool returning a list (e.g. one search's results) flattens
            # into the step output rather than nesting.
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)
        if not results:
            raise SkillRunError(f"step {step.name!r}: every item failed ({len(items)} tried)")
        return results

    def _run_single(
        self,
        skill: Skill,
        step: Step,
        context: Mapping[str, object],
        item: object | None = None,
        failures: list[str] | None = None,
    ) -> object:
        if step.kind == "tool":
            # Existence was checked in run(); contract: foreach tools receive the item,
            # whole-pipeline tools receive the full context mapping (plus the skips
            # accumulated so far, so e.g. a publish step can report them).
            fn = self.tools[step.tool]
            if step.foreach is not None:
                return fn(item)
            return fn({**context, "failures": list(failures or [])})

        template = skill.templates[step.name]
        variables = dict(context)
        if item is not None:
            variables["item"] = item
        system = self.masque_loader(step.masque) if (self.masque_loader and step.masque) else None
        prompt = self._render(step, template, variables, system_len=len(system or ""))
        text = self._call_llm(step, prompt, system)
        return _parse_lines(text, self.max_lines) if step.parse == "lines" else text

    # --- prompt assembly ---------------------------------------------------------

    def _render(
        self, step: Step, template: str, variables: Mapping[str, object], *, system_len: int
    ) -> str:
        parsed = list(string.Formatter().parse(template))
        fields = sorted({name for _, name, _, _ in parsed if name})
        unknown = [f for f in fields if f not in variables]
        if unknown:
            raise SkillRunError(f"step {step.name!r}: template references unknown {unknown}")

        literal_len = sum(len(literal) for literal, _, _, _ in parsed)
        budget_chars = self.input_budget * CHARS_PER_TOKEN - literal_len - system_len
        if budget_chars <= 0:
            raise SkillRunError(
                f"step {step.name!r}: input budget ({self.input_budget} tokens) is smaller "
                "than the template itself"
            )
        share = (budget_chars // max(len(fields), 1)) // CHARS_PER_TOKEN

        values = {
            name: truncate_to_tokens(_stringify(variables[name]), share) for name in fields
        }
        return template.format(**values)

    def _call_llm(self, step: Step, prompt: str, system: str | None) -> str:
        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(system))
        messages.append(HumanMessage(prompt))

        last_error: Exception | None = None
        for attempt in range(_LLM_ATTEMPTS):
            try:
                response = self.llm.invoke(messages)
                text = _strip_think(_message_text(response)).strip()
                if text:
                    return text
                last_error = SkillRunError("model returned empty text")
            except Exception as exc:  # noqa: BLE001 - retry once on any backend hiccup
                last_error = exc
            logger.warning("step %r llm attempt %d failed: %s", step.name, attempt + 1, last_error)
        raise SkillRunError(
            f"step {step.name!r}: llm failed after {_LLM_ATTEMPTS} attempts: {last_error}"
        )


# --- helpers -----------------------------------------------------------------------


def _message_text(message: BaseMessage) -> str:
    """Plain text across langchain content-block variants (property vs method)."""
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):
        return text()
    return str(message.content)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text)


def _parse_lines(text: str, max_lines: int) -> list[str]:
    """Tolerant line parser: strips bullets/numbering, drops blanks and preamble."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = _LINE_MARKER_RE.sub("", raw.strip()).strip().strip('"')
        if not line or line.endswith(":"):  # "Here are the queries:" preamble
            continue
        lines.append(line)
    return lines[:max_lines]


def _stringify(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value)


def _shorten(item: object, limit: int = 80) -> str:
    text = str(item).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"
