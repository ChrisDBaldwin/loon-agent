# Self-directed processing loops

Loon has no background existence: it runs only while a turn is being processed, and
every `run_command` container vanishes when the command finishes. So "loon prompting
itself" is implemented from the outside in — the **host process** (the Telegram
adapter, a persistent LaunchAgent with an always-on event loop) wakes the agent up on
a cadence with a stored prompt. Loon isn't running in the background; it's leaving
itself work that the host replays on schedule.

## The pieces

**Loop definitions (`loops/*.md`)** — YAML frontmatter + an iteration prompt, in the
same file style as skills:

```markdown
---
name: self-audit
description: one line shown by /loop
interval: 900        # seconds between iterations (floor: 60)
max_iterations: 12   # hard cap (1..100), default 10
---
The prompt sent every iteration. {iteration} and {max_iterations} are substituted.
```

**One iteration = one ordinary agent turn** (`loops.py:run_iteration`), in its own
fresh thread `loop:<name>:i<n>` — a small local model never accumulates unbounded
context across a long run. Continuity lives outside the thread instead:

- **Follow-ups** (`tools/followups.py`): `add_followup` / `list_followups` /
  `resolve_followup`, a sqlite-backed notes-to-self list shared by chat turns and
  loops. A loop iteration records findings there; you see them in any later chat.
- **The internal website**: the loop reads and rewrites its own board page with the
  ordinary site tools.
- **Long-term memory**: each iteration's turn is written back like any chat turn.

**The protocol**: every iteration prompt gets a footer explaining that no human is
present and asking the model to end with `LOOP_DONE` (goal complete) or
`LOOP_CONTINUE`. The marker is only honored on the last few lines of the reply
(`loops.py:is_done`); absent both, the loop continues — `max_iterations` is the real
backstop, so a loop that never says done still terminates.

**The driver** (`adapters/telegram.py:LoopManager`) — one asyncio task per running
loop: run an iteration, deliver the reply to the chat that started the loop, sleep
the interval, repeat. It stops on `LOOP_DONE`, at the iteration cap, on `/loop stop`,
or after 3 consecutive failed iterations (a broken backend should not be retried
unattended forever).

**Persistence** (`loops.py:LoopStore`, `.loon/loops.sqlite`) — one row per loop:
chat id, iteration count, status (`running | done | stopped | failed`). At startup
the adapter resumes any loop the previous process left `running`, from its stored
iteration count — so `launchctl kickstart` restarts don't kill a run.

## Commands

```
/loop                 list defined loops and their state
/loop start <name>    run: one iteration immediately, then every interval
/loop stop <name>     cancel (an iteration already in flight finishes, reply dropped)
```

## Safety properties

- **One model, one turn at a time**: loop iterations and user turns share an
  `asyncio.Lock`, so they never hit the single local backend concurrently. A user
  message that arrives mid-iteration waits (typing indicator stays alive); the loop
  likewise waits for a user turn to finish.
- **Loop turns get exactly the chat loop's capabilities** — same tools, same
  sandboxing (`run_command` network-off, read-only mounts). Nothing is escalated for
  autonomy.
- **Bounded by construction**: interval floor of 60s, iteration cap of at most 100,
  loops can't define or start other loops. Definitions live in the repo, not in
  model-writable state.
- **Everything is observable**: iterations log through the normal OTLP pipeline, and
  every reply is delivered to the originating (allowlisted) chat.

## The first loop: self-audit

`loops/self-audit.md` — an introspective review of loon's own setup, one area per
iteration (secrets handling, telegram access control, the exec sandbox, memory,
telemetry, dependencies, …). Each iteration investigates one area — reading its own
source via the sandbox's read-only `/repo` mount when exec is enabled — records
weaknesses as follow-ups for the human, and maintains a running "Self-Audit" report
page on the internal board. The prompt forbids writing secret *values* anywhere;
locations only.
