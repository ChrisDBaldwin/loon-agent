# The sandbox toolbox image for loon's /code skill. Runs as an unprivileged user; loon
# mounts a workspace at /workspace and runs allowlisted commands inside this container.
# Build + pin by digest (see docs/exec-sandbox.md), never reference it as :latest at runtime.
FROM python:3.12-slim

# A small, coding-focused toolset. Keep this lean — every binary here is something loon can
# run once you also add its name to LOON_EXEC_ALLOWED_BINS (the image having it is necessary
# but not sufficient; the allowlist is the actual gate).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ca-certificates ripgrep make \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pytest ruff

# Unprivileged user; loon also passes --user, but bake a real home so tools have somewhere
# to write dotfiles under /tmp-style usage.
RUN useradd --create-home --uid 1000 loon
USER loon

WORKDIR /workspace
