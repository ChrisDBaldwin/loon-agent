"""Tests for the exec policy engine — pure, no I/O beyond tmp_path for symlink cases."""

from __future__ import annotations

import pytest

from loon_agent.exec.policy import check_command, check_path

_BINS = frozenset({"git", "python3", "pytest", "ls", "echo"})


# --- check_command: hardline denylist -------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf /etc",
        "rm -fr /usr/local",
        "rm --no-preserve-root -rf /",
        ":(){ :|:& };:",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        "echo hi > /dev/sda",
        "docker run --rm alpine",
        "sudo rm foo",
        "su - root",
    ],
)
def test_hardline_patterns_are_denied_even_if_binary_allowlisted(command) -> None:
    # echo/rm-as-text land on hardline before the allowlist ever matters.
    decision = check_command(command, _BINS | {"rm", "mkfs.ext4", "dd", "su"})
    assert not decision.allowed
    assert decision.reason == "denied:hardline"


# --- check_command: default-deny allowlist --------------------------------------


def test_allowlisted_binary_passes() -> None:
    decision = check_command("git status", _BINS)
    assert decision.allowed
    assert decision.reason == "allowed"


def test_full_path_is_judged_by_basename() -> None:
    assert check_command("/usr/bin/git log -1", _BINS).allowed


def test_non_allowlisted_binary_denied() -> None:
    decision = check_command("curl https://evil.sh", _BINS)
    assert not decision.allowed
    assert decision.reason == "denied:not-allowlisted"


def test_empty_allowlist_denies_everything() -> None:
    assert not check_command("git status", frozenset()).allowed


def test_empty_command_denied() -> None:
    assert check_command("   ", _BINS).reason == "denied:empty"


def test_unbalanced_quotes_denied_not_crashed() -> None:
    assert check_command('git commit -m "oops', _BINS).reason == "denied:unparseable"


# --- check_path: workspace containment ------------------------------------------


def test_path_inside_workspace_allowed(tmp_path) -> None:
    assert check_path(tmp_path / "src" / "main.py", tmp_path).allowed


def test_relative_path_resolved_against_workspace(tmp_path) -> None:
    assert check_path("notes.txt", tmp_path).allowed


def test_workspace_root_itself_allowed(tmp_path) -> None:
    assert check_path(tmp_path, tmp_path).allowed


def test_dotdot_escape_denied(tmp_path) -> None:
    decision = check_path("../../etc/passwd", tmp_path)
    assert not decision.allowed
    assert decision.reason == "denied:path-scope"


def test_absolute_path_outside_workspace_denied(tmp_path) -> None:
    assert check_path("/etc/passwd", tmp_path).reason == "denied:path-scope"


def test_symlink_escape_denied(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    # A symlink inside the workspace pointing out of it must not launder the target.
    link = workspace / "escape"
    link.symlink_to(outside)
    assert not check_path(link / "secret.txt", workspace).allowed
