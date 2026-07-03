# Internal website

`python -m loon_agent web` (`adapters/web.py`) serves the HTML loon publishes over the LAN,
so Chris and Kayla can browse what loon creates — the way Grafana is reachable at
`ironwood:3000`. loon runs on Pontoon, so the site lives at `http://pontoon.local:8800`.

## What it serves

A curated **site directory** (`LOON_WEB_ROOT`, default `.loon/site`) — deliberately not all
of `.loon/`, which holds the sqlite DBs, checkpoints, and memory. Two things land there:

- **Research reports** — the `/research` skill now writes its HTML report straight into the
  web root, so a finished briefing is immediately browsable.
- **`/publish <topic>` pages** — the `publish` skill has loon write a markdown page and
  publish it via `publish_page`, which renders it through the same XSS-safe pipeline as
  reports (`report.render_page`: the model writes markdown, raw HTML stays escaped).

The root URL renders a generated **gallery** (newest first, with title/date/size); it reads
the directory live per request, so newly-published artifacts appear without a restart.

## Design

- **Static and read-only.** GET/HEAD only (other methods return 405); no model, no skill
  engine — serving is decoupled from the agent runtime, so the site is safe to run as its own
  always-on service next to the bot.
- **stdlib only.** `http.server.ThreadingHTTPServer` + a `SimpleHTTPRequestHandler` subclass —
  no web framework. A framework is the right call only if v2 adds browser→loon interactivity.
- **Contained.** The handler serves only files resolving inside the web root (a symlink-escape
  guard on top of the base handler's `..` sanitization).

## Configuration (`.env`)

```
LOON_WEB_HOST=0.0.0.0     # 0.0.0.0 = LAN-reachable; 127.0.0.1 = this machine only
LOON_WEB_PORT=8800
LOON_WEB_ROOT=.loon/site
```

## Running it

Locally: `uv run python -m loon_agent web`, then open `http://pontoon.local:8800`.

As a service: install `deploy/com.loon-agent.web.plist` (edit `WorkingDirectory` first) to
`~/Library/LaunchAgents/` and `launchctl bootstrap gui/$(id -u) …`. It's a separate
LaunchAgent from the Telegram bot — independent process and log
(`~/Library/Logs/loon-agent-web.log`). Restart after code changes with
`launchctl kickstart -k gui/$(id -u)/com.loon-agent.web`.

**macOS firewall:** on first launch macOS may prompt to allow the Python process to accept
incoming connections (it binds `0.0.0.0`). Allow it, or the site is only reachable from
localhost, not other nodes.

## Not yet (v2)

Browser→loon interactivity (buttons/forms/web chat), auth, and `loon.esper.internal` +
Caddy TLS branding are deliberate follow-ons — v1 is publish-and-browse, LAN-open,
`pontoon.local:PORT`.
