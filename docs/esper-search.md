# esper-search (SearXNG web search)

`web_search` (`src/loon_agent/tools/web.py`) shells out to the `esper-search` CLI
instead of calling a search engine directly. `esper-search` is a stdlib-only Python
client for the private SearXNG instance on Ironwood (`~/git/esper-searxng` there) —
it handles query caching and rate limiting via a shared Redis (Valkey) so every agent
on the network hits one cache instead of each reimplementing it. Full design:
`~/git/esper-searxng/README.md` on Ironwood.

## Installing the client on a new machine

1. Resolve `searxng.esper.internal` → `192.168.0.114` (Ironwood). Either point this
   machine's DNS at Ironwood, or add one line to `/etc/hosts`:
   ```
   192.168.0.114 searxng.esper.internal
   ```
2. Copy the CLI and Caddy's internal root CA from Ironwood:
   ```bash
   scp ironwood:~/git/esper-searxng/bin/esper-search ~/.local/bin/esper-search
   chmod +x ~/.local/bin/esper-search
   mkdir -p ~/.local/share/esper-search
   scp ironwood:~/git/esper-searxng/caddy/root.crt ~/.local/share/esper-search/root.crt
   ```
   `~/.local/bin` must be on `PATH` (it already is on Pontoon).
3. Set the client's env (loon-agent's `.env` — see `.env.example`):
   ```
   SEARXNG_URL=https://searxng.esper.internal
   SEARXNG_REDIS_URL=redis://:<VALKEY_PASSWORD>@192.168.0.114:6379/0
   SEARXNG_CA_BUNDLE=/path/to/esper-search/root.crt
   ```
   `VALKEY_PASSWORD` is in `~/git/esper-searxng/.env` on Ironwood (gitignored there —
   don't commit it here either; it only lives in loon-agent's own gitignored `.env`).

These are read directly by the `esper-search` binary from the process environment
(`load_dotenv()` in `config.py` populates it), not by loon-agent's `Settings`.

## Behavior

- `web_search()` returns `[]` if the binary is missing from `PATH`, times out, hits a
  bad exit code, or the network round-trip fails — the research skill already treats
  an empty result set as "this query found nothing" and moves on.
- Exit 2 (rate limited past the CLI's own wait cap) is not retried locally — the CLI
  already waited out the shared window before giving up.
- Only `web_search` goes through esper-search. `fetch_page` (reading a *known* URL in
  full) still fetches directly via `httpx` + `trafilatura` — that's not what SearXNG
  is for.
