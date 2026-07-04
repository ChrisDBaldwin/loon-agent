"""Tests for the Docker exec backend.

argv construction is asserted without Docker (pure). The end-to-end container run is gated
behind a skipif so machines without a reachable Docker daemon (CI, laptops) don't fail —
mirroring how tools/web.py degrades when its esper-search CLI is absent.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from loon_agent.exec.docker_backend import DockerExecBackend, DockerLimits


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, timeout=10, check=False,
            ).returncode
            == 0
        )
    except Exception:  # noqa: BLE001 - any failure means "not usable for the test"
        return False


_DOCKER = pytest.mark.skipif(not _docker_ready(), reason="docker daemon not reachable")


# --- argv construction (no Docker needed) ---------------------------------------


def test_argv_hardening_flags(tmp_path) -> None:
    backend = DockerExecBackend(
        image="loon-toolbox@sha256:abc",
        limits=DockerLimits(network="none", memory="256m", cpus=2.0, pids=64, user="1000:1000"),
    )
    argv = backend._docker_run_argv("pytest -q", tmp_path)

    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert "--read-only" in argv
    assert argv[argv.index("--memory") + 1] == "256m"
    assert argv[argv.index("--cpus") + 1] == "2.0"
    assert argv[argv.index("--pids-limit") + 1] == "64"
    assert argv[argv.index("--user") + 1] == "1000:1000"
    # Exactly one mount, the workspace, and it is not the docker socket or a host secret.
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--volume"]
    assert mounts == [f"{tmp_path.resolve()}:/workspace:rw"]
    assert not any("docker.sock" in m or ".ssh" in m for m in mounts)
    # Command runs through a shell in the workdir, image before it.
    assert argv[-3:] == ["sh", "-c", "pytest -q"]
    assert argv[argv.index("--workdir") + 1] == "/workspace"


def test_empty_image_rejected() -> None:
    with pytest.raises(ValueError, match="LOON_EXEC_IMAGE"):
        DockerExecBackend(image="")


def test_preflight_reports_missing_binary(monkeypatch) -> None:
    monkeypatch.setattr("loon_agent.exec.docker_backend.shutil.which", lambda _: None)
    backend = DockerExecBackend(image="x")
    assert backend.preflight() == "docker binary not found on PATH"


def test_run_degrades_to_error_when_unusable(monkeypatch, tmp_path) -> None:
    backend = DockerExecBackend(image="x")
    monkeypatch.setattr(backend, "preflight", lambda: "docker daemon unreachable: nope")
    result = backend.run("ls", cwd=tmp_path, timeout=5)
    assert not result.ok
    assert "unreachable" in (result.error or "")


# --- real container run (gated) -------------------------------------------------


@_DOCKER
def test_echo_runs_in_real_container(tmp_path) -> None:
    backend = DockerExecBackend(image="alpine:3", limits=DockerLimits(network="none"))
    result = backend.run("echo hello-from-sandbox", cwd=tmp_path, timeout=60)
    assert result.ok, result.error
    assert result.exit_code == 0
    assert "hello-from-sandbox" in result.stdout


@_DOCKER
def test_workspace_write_persists_on_host(tmp_path) -> None:
    backend = DockerExecBackend(image="alpine:3", limits=DockerLimits(network="none"))
    result = backend.run("echo persisted > /workspace/out.txt", cwd=tmp_path, timeout=60)
    assert result.ok, result.error
    assert (tmp_path / "out.txt").read_text().strip() == "persisted"


@_DOCKER
def test_network_is_off_by_default(tmp_path) -> None:
    backend = DockerExecBackend(image="alpine:3", limits=DockerLimits(network="none"))
    # With --network=none there is no route out; wget/ping should fail (non-zero exit),
    # but the command itself still *runs* (result.ok), proving isolation not breakage.
    result = backend.run(
        "wget -T 3 -q -O- http://example.com || echo NETFAIL", cwd=tmp_path, timeout=60
    )
    assert result.ok, result.error
    assert "NETFAIL" in result.stdout


# --- read-only host mounts (LOON_EXEC_RO_MOUNTS) ---------------------------------


def test_ro_mounts_appear_in_argv_as_readonly_volumes(tmp_path) -> None:
    host = tmp_path / "repo-src"
    host.mkdir()
    backend = DockerExecBackend(
        image="x", limits=DockerLimits(ro_mounts=((str(host), "/repo/src"),))
    )
    argv = backend._docker_run_argv("ls /repo/src", tmp_path / "ws")
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--volume"]
    assert f"{(tmp_path / 'ws')}:/workspace:rw" in mounts[0]
    assert f"{host.resolve()}:/repo/src:ro" in mounts
    # The workspace stays the only writable mount.
    assert [m for m in mounts if m.endswith(":rw")] == [f"{(tmp_path / 'ws')}:/workspace:rw"]


def test_ro_mount_may_not_shadow_workspace(tmp_path) -> None:
    with pytest.raises(ValueError, match="shadows the workspace"):
        DockerExecBackend(
            image="x", limits=DockerLimits(ro_mounts=((str(tmp_path), "/workspace"),))
        )


def test_ro_mount_host_path_must_exist(tmp_path) -> None:
    with pytest.raises(ValueError, match="not an existing directory"):
        DockerExecBackend(
            image="x", limits=DockerLimits(ro_mounts=((str(tmp_path / "gone"), "/repo"),))
        )


@_DOCKER
def test_ro_mount_is_readable_but_not_writable_in_container(tmp_path) -> None:
    host = tmp_path / "shared"
    host.mkdir()
    (host / "note.txt").write_text("from-host", encoding="utf-8")
    backend = DockerExecBackend(
        image="alpine:3", limits=DockerLimits(ro_mounts=((str(host), "/repo/src"),))
    )
    result = backend.run(
        "cat /repo/src/note.txt && (touch /repo/src/x 2>/dev/null || echo READONLY)",
        cwd=tmp_path / "ws", timeout=60,
    )
    assert result.ok, result.error
    assert "from-host" in result.stdout
    assert "READONLY" in result.stdout
