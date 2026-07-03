---
name: code
description: Run a scoped coding task (tests, lint, git, build) inside the sandboxed workspace.
args: [task]
steps:
  - {name: plan,    kind: llm,  masque: analyst, output: commands, parse: lines}
  - {name: execute, kind: tool, tool: run_command, foreach: commands, output: results}
  - {name: report,  kind: llm,  masque: briefer,  output: summary}
---

## step: plan
Coding task to accomplish: {task}

The commands you write run one at a time inside an isolated sandbox container. The
working directory is the shared workspace; there is no network by default, and only an
allowlisted set of programs will run — anything else is refused and reported, not executed.

Write the shell commands needed to accomplish the task, in order, **one command per line**.
- Plain shell commands only. No numbering, no prose, no backticks, no explanation.
- Each line is run on its own — do not rely on `cd` persisting between lines; use paths.
- Prefer read/inspect commands before mutating ones so failures surface early.
- If the task cannot be done with shell commands, output a single line: NO COMMANDS

## step: report
Coding task: {task}

Command results, in order (each shows the command, its exit code, and output):

{results}

Write a short markdown summary of what happened:
- Start with one line: DONE, PARTIAL, or FAILED — did the task's goal get met?
- Then 2-5 bullets: what each command did, and call out any non-zero exit codes, refused
  (policy-denied) commands, or errors plainly. Do not claim success for a command that
  exited non-zero or was refused.
- If nothing ran (all refused, or NO COMMANDS), say so and why in one line.
