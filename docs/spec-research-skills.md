# Spec: Composable Skills, Deep Research, and Masque Identities

Status: draft for review · 2026-07-01

## 1. Problem statement

loon-agent can chat and call two toy tools. It cannot do real work. The first real
capability: **given a topic, loon goes out, reads the web, and comes back with "hey
Chris, here's what you need to know"** — a short cited briefing in the terminal, a
polished self-contained HTML report on disk, and a memory entry so later chats can
recall what was learned.

The hard constraint is the runtime: a local model behind LM Studio/vLLM with a small
practical context window (KV-cache pressure on a 24 GB Mac; see memory notes on
`pontoon`). A 7B–26B local model cannot hold a 30-source research session in its head
the way a frontier model can. Therefore **orchestration lives in Python, cognition
lives in small, focused LLM calls.**

### Success criteria (binary)

1. `you> /skill research <topic>` produces, on pontoon:
   - a terminal briefing (≤ ~300 words) with numbered source URLs,
   - a self-contained HTML report at `.loon/reports/<slug>-<date>.html`,
   - a memory row recallable by FTS in a later session.
2. No single LLM call's assembled prompt exceeds the configured input budget
   (`LOON_STEP_INPUT_BUDGET`, default 4 000 tokens ≈ 16 000 chars).
3. Adding a new skill = dropping a markdown file in `skills/` — zero core-code changes.
4. A skill step can declare `masque: analyst` and run under that lens; masque YAML is
   schema-compatible with `~/git/masques` (`name` / `lens` / `context` subset).
5. All new modules unit-tested without a live backend (fake LLM, mocked HTTP).

### Out of scope (v1)

- Masques' audience/scoring layer (OTel hook exists; defer).
- Embeddings / semantic memory (OpenViking remains the planned upgrade).
- JS-rendered pages (plain HTTP fetch only), Telegram adapter, token streaming.
- Conditionals/loops/nesting in the skill DSL — two step kinds, `foreach`, done.

## 2. Design

Two composable primitives, one deterministic engine:

- **Skill (markdown)** — *what to do.* Frontmatter declares the pipeline; the body
  holds one prompt template per LLM step. Authored/edited like a document.
- **Masque (YAML)** — *who is doing it.* A lens + context block injected as the
  system prompt of the steps that don it. Reuses the masques schema; loon reads the
  same files `~/git/masques/personas` ships.
- **Engine (Python)** — runs the declared steps in order, each LLM call with a fresh,
  budget-truncated context. The model never orchestrates; it only ever does one
  focused job per call.

```
/skill research <topic>
        │
        ▼
 plan (llm, analyst)      topic → 4 search queries
 search (tool, foreach)   ddgs → titles/snippets/urls
 select (llm, analyst)    pick N urls worth reading
 fetch (tool, foreach)    httpx + trafilatura → clean text (truncated per source)
 summarize (llm, analyst, foreach)   one call per source → focused notes
 synthesize (llm, briefer)           notes → cited briefing (markdown)
 publish (tool)           briefing → HTML report + memory write-back
```

### 2.1 New layout

```
skills/                      # repo root — skill definitions (the authoring surface)
  research.md
masques/                     # loon-local masques (masques-schema-compatible YAML)
  analyst.yaml
  briefer.yaml
src/loon_agent/
  skills/
    model.py                 # Skill/Step dataclasses + frontmatter/markdown parser
    engine.py                # SkillRunner: step execution, foreach, budgets, parsing
    __init__.py              # discover_skills(dir) -> dict[str, Skill]
  masques.py                 # load_masque(name) -> lens/context system block
  tools/web.py               # web_search (ddgs), fetch_page (httpx+trafilatura)
  report.py                  # markdown briefing -> styled self-contained HTML
  textbudget.py              # approx_tokens, truncate helpers (chars/4 heuristic)
```

### 2.2 Skill file format

```markdown
---
name: research
description: Deep-dive a topic on the web; produce a cited briefing + HTML report.
args: [topic]
steps:
  - {name: plan,       kind: llm,  masque: analyst, output: queries, parse: lines}
  - {name: search,     kind: tool, tool: web_search, foreach: queries, output: results}
  - {name: select,     kind: llm,  masque: analyst, output: urls, parse: lines}
  - {name: fetch,      kind: tool, tool: fetch_page, foreach: urls, output: pages}
  - {name: summarize,  kind: llm,  masque: analyst, foreach: pages, output: notes}
  - {name: synthesize, kind: llm,  masque: briefer, output: briefing}
  - {name: publish,    kind: tool, tool: publish_report, output: report_path}
---

## step: plan
Plan web research on: {topic}
Write exactly 4 short, diverse search queries, one per line. No numbering, no prose.

## step: select
...templates for each llm step, referencing {topic}, {results}, {item}, etc...
```

Engine semantics (deliberately dumb):

- A context dict accumulates step outputs under their `output` names; `{item}` is the
  current element inside a `foreach`.
- `kind: llm` renders the matching `## step:` template with the context (every
  substituted variable truncated to its share of the input budget), calls the model
  with the step's masque as system prompt, strips `<think>`/reasoning content, applies
  the `parse` mode (`text` default, or `lines` — tolerant: drops bullets/numbering,
  caps list length).
- `kind: tool` calls a Python function from a small registry
  (`web_search`, `fetch_page`, `publish_report`).
- Any step failure inside `foreach` skips that item and records it; a top-level step
  failure aborts the skill with a readable error.

**JSON is deliberately not a parse mode.** Line-oriented output is the reliable
contract for small local models.

### 2.3 Masques

`masques.py` reads YAML with the masques schema subset `{name, lens, context}`.
Search path: `./masques/` then `LOON_MASQUES_DIR` (point it at
`~/git/masques/personas` to reuse the existing catalog). A step's masque becomes the
system prompt for that call: lens + (optional) context. `LOON_MASQUE=<name>` optionally
dons a masque on the chat agent itself (appended to `SYSTEM_PROMPT`). Don/doff at the
REPL and audience scoring are future work.

### 2.4 Report rendering

The model writes **markdown**; Python renders HTML. `report.py` converts the briefing
via `markdown-it-py` (default mode — raw HTML from untrusted summaries stays escaped)
into a single-file template (embedded CSS, no external assets): title, date, TL;DR,
body with numbered citations, source list with fetch status, footer noting model +
backend. Written to `.loon/reports/`, path printed to the terminal.

### 2.5 Memory write-back

After `publish`, the existing provider records the turn:
`sync_turn(user="research: <topic>", assistant="<TL;DR + report path>", session_id="skill:research")`
— FTS-recallable from any future chat, no schema change. (MEMORY.md stays
human/agent-curated; no auto-append.)

### 2.6 Config additions (`LOON_` prefix)

| setting | default | purpose |
|---|---|---|
| `step_input_budget` | 4000 | max approx tokens of any assembled step prompt |
| `step_max_tokens` | 3000 | output cap per step call (reasoning models need headroom) |
| `research_sources` | 5 | pages fetched/summarized per run |
| `masques_dir` | `masques/` | extra masque search path |
| `masque` | — | optional masque donned by the chat agent |

### 2.7 Telegram adapter — the vertical slice (built first)

Chris's call (2026-07-01): the first shippable slice is **loon on Telegram** — take a
conversation from a Telegram bot, run the agent turn, reply back. The CLI stays as the
dev/base adapter; research skills land on top afterwards.

Reviewed hermes-agent's implementation (`plugins/platforms/telegram/adapter.py`): it is
correct in approach but coupled to hermes's gateway framework (media caching, fallback
transports, lazy installs) — we copy the *pattern*, not the code:

- **`python-telegram-bot` v22+, long-polling** (no webhook/public ingress needed in the
  homelab). Sequential update processing — one local model, one turn at a time.
- **Allowlist auth, deny-by-default.** `LOON_TELEGRAM_ALLOWED_USERS` is a comma-separated
  list of numeric Telegram user ids. Unknown users get a polite refusal that includes
  their id — which doubles as the way to discover your own id on first contact.
- **Session identity** falls out of the existing machinery: chat/user/topic →
  `SessionSource(platform="telegram", ...)` → `build_session_key` → durable checkpointed
  thread. DMs, groups, and forum topics each get isolated conversations for free.
- **Slow-model UX:** a background task refreshes the `typing…` chat action while the
  turn runs (pontoon cold starts can take a minute); replies are chunked at Telegram's
  4096-char limit on newline boundaries; errors come back as a short apologetic message,
  never a crash of the polling loop.
- The agent turn runs via `asyncio.to_thread` so the sync `LoonAgent.invoke` never
  blocks PTB's event loop.
- Entry point: `python -m loon_agent telegram` (default remains `cli`).
- Config: `LOON_TELEGRAM_TOKEN` (BotFather), `LOON_TELEGRAM_ALLOWED_USERS`.

### 2.8 Backend realities (pontoon)

- Reasoning model (`gemma-4-26b`): answers can land in `reasoning_content` with empty
  `.content` mid-think — the engine must read final content and strip think blocks.
- Warm ≈ 34 tok/s, cold start pays a ~15 GB fault-in — generous client timeouts
  (120 s) + one retry per step call.
- Small per-call contexts are also a KV-cache/RAM courtesy on the 24 GB box.

New deps: `ddgs`, `httpx`, `trafilatura`, `markdown-it-py`, `pyyaml`.

## 3. Critical review (spec survived these)

- **ddgs flakiness / rate limits** → retry with backoff; if search yields nothing, the
  skill aborts with a clear message rather than hallucinating a report.
- **Fetch failures / paywalls / binary content** → per-item skip, recorded and shown in
  the report's source table; run proceeds if ≥1 source survives.
- **Small-model discipline** (ignores "one per line", adds preamble) → tolerant line
  parser, hard caps (max 6 queries, max `research_sources` urls) regardless of output.
- **Prompt injection from fetched pages** → fetched text only ever reaches summarize
  prompts; report renderer escapes raw HTML; injected instructions can at worst skew a
  summary — acceptable for a homelab tool, noted in README.
- **Context overflow** → truncation is enforced by the engine on every substitution,
  not trusted to prompt discipline.
- **DSL creep** (the real overengineering risk) → v1 grammar frozen: `llm|tool`,
  `foreach`, `parse: text|lines`. Anything fancier is a Python tool, not DSL syntax.
- **Underengineering check** → seams already exist for the known futures: SearchProvider
  swap (SearXNG), MemoryProvider swap (OpenViking), masque audience via existing OTel.

## 4. Beads (build order)

| # | bead | size | tests |
|---|------|------|-------|
| 0 | **Telegram adapter (vertical slice, first)**: `adapters/telegram.py`, config + entry point, allowlist, chunking, typing indicator | M | unit (pure helpers) + manual bot smoke |
| 1 | deps + `textbudget.py` (approx_tokens, truncate) | S | unit |
| 2 | `tools/web.py`: web_search (ddgs, retries) + fetch_page (httpx+trafilatura, truncation) | M | mocked HTTP |
| 3 | `skills/model.py`: parse skill markdown → Skill/Step; validation errors readable | M | unit |
| 4 | `skills/engine.py`: SkillRunner — foreach, budgets, think-strip, tolerant parsers, per-item failure policy | M | fake LLM |
| 5 | `masques.py`: loader + engine/system-prompt integration + `LOON_MASQUE` | S | unit |
| 6 | `report.py`: briefing → self-contained HTML; write + return path | M | unit |
| 7 | `skills/research.md` + `masques/*.yaml` prompts; publish tool wiring memory write-back; CLI `/skill <name> <args>` (+ `/research` alias) | M | unit + fake run |
| 8 | E2E smoke on pontoon; README + this doc updated to reality | S | manual |
| 9 | (stretch) expose each skill as a chat tool so loon can trigger research mid-conversation | S | unit |

Order = dependency chain; bead 4 is the risk concentrate (do early, fail fast).
