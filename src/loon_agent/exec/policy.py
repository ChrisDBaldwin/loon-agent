"""Allow/deny policy for sandboxed execution — the second defensive layer.

Pure and I/O-free so it is trivially unit-testable and cannot itself be the thing that
breaks. It is **not** the security boundary — the container backend is (see
``docker_backend.py``); this layer catches cooperative-mode mistakes early and gives a
clean, auditable denial reason before anything is spawned or written.

Two checks, two philosophies:

* :func:`check_command` — a tiny *unconditional* hardline denylist (things that are never
  OK regardless of allowlist: wiping the disk, fork bombs, escaping the sandbox) followed
  by a **default-deny allowlist** on the resolved program name. The allowlist is the
  primary control: denylists on command text are bypassable (symlinks, full paths,
  aliases), so anything not explicitly permitted is refused.
* :func:`check_path` — resolves a target path (symlinks included) and refuses anything that
  escapes the workspace root, so file create/edit/delete can never touch the host outside
  the sandbox dir even when the backend is a plain in-process write.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

# Unconditional refusals — never allowed no matter what the allowlist says, because the
# blast radius is catastrophic or the intent is to break out of the sandbox. Matched
# against the raw command string (case-insensitive). Kept deliberately small: the
# container bounds everything else, so this only needs to cover host-lethal patterns and
# obvious escape attempts.
_HARDLINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # rm -rf targeting a filesystem root or system dir (any flag order, incl. --no-preserve-root).
    # The path alternatives end in a lookahead (slash, whitespace, or end-of-string) rather than
    # \b, since a bare trailing "/" is a non-word char and \b would never fire after it.
    re.compile(
        r"\brm\b.*\s-\w*[rf]\w*.*\s+(?:~|/(?:etc|usr|bin|var|boot|dev|sys|lib)?)(?=/|\s|$)"
    ),
    re.compile(r"\brm\b.*--no-preserve-root"),
    # Classic fork bomb :(){ :|:& };:  and shell-function variants
    re.compile(r":\s*\(\s*\)\s*\{.*\|.*&.*\}"),
    re.compile(r"\.\s*\(\s*\)\s*\{.*\|.*&.*\}"),
    # Filesystem creation / raw block-device writes
    re.compile(r"\bmkfs(\.\w+)?\b"),
    re.compile(r"\bdd\b.*\bof=/dev/"),
    re.compile(r">\s*/dev/(sd|nvme|hd|disk|mmcblk)"),
    # Escaping the sandbox: talking to Docker or elevating privilege from inside a command
    re.compile(r"\bdocker\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bsu\b\s"),
)


@dataclass(frozen=True)
class PolicyDecision:
    """The verdict for one command or path check."""

    allowed: bool
    # Machine-readable reason on denial (e.g. "denied:hardline", "denied:not-allowlisted",
    # "denied:path-scope") — also used as the OTel loon.exec.policy_decision attribute.
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.allowed


_ALLOW = PolicyDecision(allowed=True, reason="allowed")


def check_command(command: str, allowed_bins: frozenset[str]) -> PolicyDecision:
    """Decide whether ``command`` may run: hardline denylist, then default-deny allowlist.

    The allowlist is checked against the *resolved program name* — the basename of the
    first shell token — so ``/usr/bin/git`` and ``git`` are both judged as ``git``. An
    empty allowlist denies everything (the safe default when exec is misconfigured).
    """
    text = command.strip()
    if not text:
        return PolicyDecision(allowed=False, reason="denied:empty")

    lowered = text.lower()
    for pattern in _HARDLINE_PATTERNS:
        if pattern.search(lowered):
            return PolicyDecision(allowed=False, reason="denied:hardline")

    try:
        tokens = shlex.split(text)
    except ValueError:
        # Unbalanced quotes etc. — refuse rather than guess at the program name.
        return PolicyDecision(allowed=False, reason="denied:unparseable")
    if not tokens:
        return PolicyDecision(allowed=False, reason="denied:empty")

    program = Path(tokens[0]).name
    if program not in allowed_bins:
        return PolicyDecision(allowed=False, reason="denied:not-allowlisted")

    return _ALLOW


def check_path(path: Path | str, workspace_root: Path | str) -> PolicyDecision:
    """Decide whether a file op on ``path`` stays inside ``workspace_root``.

    Both sides are fully resolved (``..`` collapsed, symlinks followed) before the
    containment test, so neither ``../../etc/passwd`` nor a symlink pointing out of the
    workspace can escape.
    """
    root = Path(workspace_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    resolved = target.resolve()

    if resolved == root or root in resolved.parents:
        return _ALLOW
    return PolicyDecision(allowed=False, reason="denied:path-scope")
