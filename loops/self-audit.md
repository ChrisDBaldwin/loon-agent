---
name: self-audit
description: Review my own setup one area per iteration; findings go to follow-ups and a board page.
interval: 900
max_iterations: 12
---
You are running your **self-audit loop**: an autonomous review of your own setup and
creation, looking for areas that are lacking, weak, or insecure. You cover ONE area
per iteration and keep a running report on your website.

Areas to work through across iterations (plus anything you notice on the way):
secrets & configuration handling, telegram access control, the exec sandbox and its
mounts, the internal website, long-term memory, session/checkpoint storage, skills &
masques, telemetry, and dependency hygiene.

This iteration:

1. See where the audit stands: call `list_followups`, then `list_site_pages` and read
   your audit page (its filename starts with `self-audit-`) with `read_site_page` if
   it exists. Work out which areas are already covered.
2. Pick ONE area not yet covered and investigate it. If `run_command` is available,
   your own source code is mounted read-only at /repo — inspect the real code and
   configuration (`ls /repo`, `cat`, `grep -r`). Otherwise reason from what you know
   about your setup.
3. For each concrete weakness, gap, or risk you find, record it with
   `add_followup(topic, note)` — topic is a short label, the note says what is weak
   and what a fix might look like. These are for your human to follow up on.
4. Update the running report: the first iteration publishes a new page titled
   "Self-Audit" (`publish_site_page`); later iterations rewrite the existing page
   (`update_site_page`) adding a section for the area just audited, so the page reads
   as one complete report. Keep earlier sections intact.
5. Finish your reply with a 2-3 sentence summary of what you audited and found this
   iteration. If every area is now covered, declare the loop done; otherwise continue.

Never write secret VALUES (tokens, API keys, passwords) into follow-ups or the page —
name the file or setting where the secret lives instead.
