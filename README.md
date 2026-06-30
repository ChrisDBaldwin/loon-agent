# loon-agent

A from-scratch LangGraph agent for the homelab — a learning-first project to get
comfortable building an LLM agent end to end (agent loop, tools, memory, telemetry,
chat interface) and run it on my own hardware.

## What it is

- **Hand-rolled ReAct loop** built directly on `langgraph.StateGraph` (no `create_agent`
  prebuilt) so the agent loop is fully visible and hackable.
- **Backend-agnostic.** One `ChatOpenAI(base_url=...)` interface talks to any
  OpenAI-compatible endpoint. Three homelab backends are pre-configured:
  | name       | host                       | example model                            |
  |------------|----------------------------|------------------------------------------|
  | `pontoon`  | MacBook M5 (LM Studio/MLX) | `mlx-community/Qwen2.5-7B-Instruct-4bit`  |
  | `ironwood` | 3080 Ti (vLLM)             | `Qwen/Qwen2.5-14B-Instruct-AWQ`          |
  | `wsl`      | this box, WSL2 (vLLM)      | `Qwen/Qwen2.5-7B-Instruct-AWQ`           |
- **Swappable memory** behind a `MemoryProvider` interface (patterned on
  Nous Research's hermes-agent: `system_prompt_block` / `prefetch` / `sync_turn`).
  Default impl is SQLite FTS5 + a markdown notes file — OpenViking can be added later
  as a drop-in provider.
- **OpenTelemetry gen_ai** spans/metrics via OpenInference instrumentation (optional).

## How it works

Each turn runs through a compiled `StateGraph`:

```
START → agent → (tool calls?) ──yes──→ tools → agent → … 
                      └──────no──────→ END
```

- **agent** assembles the prompt (system + memory block + recall + history) and calls the
  tool-bound model.
- **tools_condition** routes to the `ToolNode` when the model emitted tool calls, otherwise
  to `END`.
- Tool results feed back into **agent** for another reasoning step, until the model answers
  with no tool calls.
- A SQLite **checkpointer** makes each `thread_id` a durable conversation; the
  `MemoryProvider` injects recall *before* the turn and writes the turn back *after*.

Adapters (CLI now, Telegram later) only normalize input into a platform-neutral
`MessageEvent` and derive a stable `thread_id` via `build_session_key` — the agent core
never needs to know which platform a message came from.

## Quickstart

```bash
uv sync                       # install deps
cp .env.example .env          # set LOON_BACKEND to whichever box is up
uv run python -m loon_agent   # launch the CLI REPL
```

Smoke-test a backend directly:

```bash
uv run python -c "from loon_agent.llm import make_llm; print(make_llm('wsl').invoke('say hi').content)"
```

## Configuration

Settings come from the environment / `.env` (prefix `LOON_`); defaults live in
`src/loon_agent/config.py`. Each backend can be overridden per-field:

| variable                 | purpose                                             |
|--------------------------|-----------------------------------------------------|
| `LOON_BACKEND`           | which backend to use (`pontoon` / `ironwood` / `wsl`) |
| `LOON_<NAME>_BASE_URL`   | override a backend's OpenAI-compatible base URL      |
| `LOON_<NAME>_MODEL`      | override a backend's model id                        |
| `LOON_<NAME>_API_KEY`    | bearer token for that backend (see auth note below)  |
| `LOON_TEMPERATURE`       | sampling temperature                                 |
| `LOON_DATA_DIR`          | where the checkpointer + long-term memory live       |
| `LOON_OTEL`              | telemetry mode: `off` (default) / `console` / `otlp` |

Most local servers ignore auth, so the API key defaults to a placeholder. Set
`LOON_<NAME>_API_KEY` only when a backend actually requires a token — e.g. LM Studio with
"API token authentication" enabled.

## Backend serving notes

- **vLLM (ironwood / WSL2):** serve with tool-calling enabled, e.g.
  `vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes` — required for
  `bind_tools` to work.
- **Mac (pontoon):** LM Studio's OpenAI-compatible server on `:1234`, or `mlx_lm.server`.
  Use a tool-capable model. If LM Studio's token auth is on, set `LOON_PONTOON_API_KEY`.
  On memory-tight machines, keep the loaded model resident (free RAM, modest context) — a
  15 GB model that gets paged out between requests will cold-start slowly.

## Extending

Each axis of growth has its own seam — you rarely touch the core loop:

| To…              | Do this                                                                                   |
|------------------|-------------------------------------------------------------------------------------------|
| Add a tool       | Write an `@tool` in `tools/builtins.py`, add it to `DEFAULT_TOOLS`.                        |
| Add a backend    | Add to `DEFAULT_BACKENDS` in `config.py`, or set `LOON_<NAME>_*` in `.env`.               |
| Add an interface | New adapter in `adapters/` emitting `MessageEvent`s + a `SessionSource`; reuse `build_agent()` / `agent.stream()`. |
| Swap memory      | Implement `MemoryProvider` (`system_prompt_block` / `prefetch` / `sync_turn`), pass it to `LoonAgent`. |
| Extend state     | Add channels to `AgentState` (subclasses `MessagesState`).                                |

## Layout

```
src/loon_agent/
  config.py     backend registry + settings
  llm.py        make_llm(name) -> ChatOpenAI
  state.py      AgentState
  graph.py      hand-rolled StateGraph ReAct loop (LoonAgent: invoke/stream)
  app.py        build_agent() — wires llm + tools + checkpointer + memory + telemetry
  tools/        @tool definitions (DEFAULT_TOOLS)
  memory/       MemoryProvider interface + SQLite/FTS5 impl
  session.py    SessionSource / MessageEvent / build_session_key
  adapters/     cli.py (REPL); telegram later
  telemetry.py  OpenTelemetry gen_ai wiring
```

## Limitations & roadmap

An early scaffold — the architecture is in place, but capability and hardening are thin:

- **Tools are trivial** — only `calculator` and `get_current_time`; real homelab tools
  (shell, HTTP, files) are the obvious next step.
- **Memory recall is keyword-only** — `prefetch` is an OR of tokens over FTS5, and it is
  not yet scoped to the session (it can surface exchanges from other conversations). No
  embeddings / semantic recall yet; an OpenViking provider is the planned upgrade.
- **One adapter** — CLI only; the Telegram adapter is designed-for but not built.
- **No token streaming** — `stream()` yields per-node message updates, not tokens.
- **Few guards** — no `max_tokens` / iteration cap, no retries around LLM calls, and the
  checkpointed history never compacts.
- **Thin tests** — the graph loop and session keying are covered; memory, llm, config, and
  telemetry are not, and there's no CI yet.
