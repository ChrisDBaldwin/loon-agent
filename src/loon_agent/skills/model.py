"""Skill definitions: markdown files parsed into typed pipelines.

A skill file is YAML frontmatter (name, description, args, steps) followed by one
``## step: <name>`` prompt-template section per LLM step. The grammar is deliberately
frozen small (see docs/spec-research-skills.md): two step kinds (``llm`` / ``tool``),
optional ``foreach`` fan-out, and two parse modes (``text`` / ``lines``). Anything
fancier belongs in a Python tool, not in DSL syntax.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_STEP_HEADING_RE = re.compile(r"^##\s*step:\s*(\S+)\s*$", re.MULTILINE)

KINDS = ("llm", "tool")
PARSE_MODES = ("text", "lines")


class SkillParseError(ValueError):
    """A skill file is malformed; the message says which file and what's wrong."""

    def __init__(self, path: Path | str, problem: str) -> None:
        super().__init__(f"{path}: {problem}")


@dataclass(frozen=True)
class Step:
    name: str
    kind: str  # "llm" | "tool"
    output: str
    tool: str | None = None  # tool steps: registry key
    masque: str | None = None  # llm steps: lens to don
    foreach: str | None = None  # fan out over a prior list output
    parse: str = "text"  # llm steps: "text" | "lines"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    args: tuple[str, ...]
    steps: tuple[Step, ...]
    templates: dict[str, str] = field(default_factory=dict)  # llm step name -> prompt
    path: Path | None = None


def parse_skill(text: str, path: Path | str = "<string>") -> Skill:
    """Parse a skill markdown document; raise :class:`SkillParseError` with specifics."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise SkillParseError(path, "missing YAML frontmatter (--- ... ---)")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise SkillParseError(path, f"frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise SkillParseError(path, "frontmatter must be a YAML mapping")

    name = meta.get("name")
    if not name or not isinstance(name, str):
        raise SkillParseError(path, "frontmatter needs a string 'name'")
    raw_steps = meta.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise SkillParseError(path, "frontmatter needs a non-empty 'steps' list")

    templates = _parse_templates(text[match.end() :])
    steps = tuple(_parse_step(raw, index, path) for index, raw in enumerate(raw_steps))
    _validate(steps, templates, path)

    args = meta.get("args") or []
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise SkillParseError(path, "'args' must be a list of strings")

    return Skill(
        name=name,
        description=str(meta.get("description", "")),
        args=tuple(args),
        steps=steps,
        templates=templates,
        path=Path(path) if path != "<string>" else None,
    )


def load_skill(path: Path | str) -> Skill:
    path = Path(path)
    return parse_skill(path.read_text(encoding="utf-8"), path)


def discover_skills(directory: Path | str) -> dict[str, Skill]:
    """Load every ``*.md`` in ``directory`` (missing directory -> empty registry)."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    skills: dict[str, Skill] = {}
    for file in sorted(directory.glob("*.md")):
        skill = load_skill(file)
        if skill.name in skills:
            raise SkillParseError(file, f"duplicate skill name {skill.name!r}")
        skills[skill.name] = skill
    return skills


# --- internals -------------------------------------------------------------------


def _parse_step(raw: object, index: int, path: Path | str) -> Step:
    if not isinstance(raw, dict):
        raise SkillParseError(path, f"steps[{index}] must be a mapping")
    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise SkillParseError(path, f"steps[{index}] needs a string 'name'")
    kind = raw.get("kind")
    if kind not in KINDS:
        raise SkillParseError(path, f"step {name!r}: kind must be one of {KINDS}, got {kind!r}")
    output = raw.get("output")
    if not output or not isinstance(output, str):
        raise SkillParseError(path, f"step {name!r} needs a string 'output'")
    parse = raw.get("parse", "text")
    if parse not in PARSE_MODES:
        raise SkillParseError(
            path, f"step {name!r}: parse must be one of {PARSE_MODES}, got {parse!r}"
        )
    tool = raw.get("tool")
    if kind == "tool" and (not tool or not isinstance(tool, str)):
        raise SkillParseError(path, f"tool step {name!r} needs a string 'tool'")
    if kind == "llm" and tool:
        raise SkillParseError(path, f"llm step {name!r} must not set 'tool'")

    return Step(
        name=name,
        kind=kind,
        output=output,
        tool=tool,
        masque=raw.get("masque"),
        foreach=raw.get("foreach"),
        parse=parse,
    )


def _parse_templates(body: str) -> dict[str, str]:
    templates: dict[str, str] = {}
    matches = list(_STEP_HEADING_RE.finditer(body))
    for i, heading in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        templates[heading.group(1)] = body[heading.end() : end].strip()
    return templates


def _validate(steps: tuple[Step, ...], templates: dict[str, str], path: Path | str) -> None:
    names = [step.name for step in steps]
    if len(names) != len(set(names)):
        raise SkillParseError(path, "step names must be unique")
    for step in steps:
        if step.kind == "llm":
            if not templates.get(step.name):
                raise SkillParseError(
                    path, f"llm step {step.name!r} has no '## step: {step.name}' template"
                )
    orphans = set(templates) - set(names)
    if orphans:
        raise SkillParseError(path, f"templates without a matching step: {sorted(orphans)}")
