---
name: research
description: Deep-dive a topic on the web and produce a cited briefing + HTML report.
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

Write exactly 4 short, diverse web search queries that together cover this topic
from different angles (e.g. overview, technical detail, comparison, recent news).
One query per line. No numbering, no quotes, no other text.

## step: select
Research topic: {topic}

Search results (title, url, snippet):

{results}

Pick the {max_sources} URLs most worth reading in full. Prefer primary and
authoritative sources over aggregators; avoid picking two pages from the same
site. Output only the chosen URLs, one per line, nothing else.

## step: summarize
Research topic: {topic}

Source material:

{item}

Write terse notes on what THIS source says about the topic:
- first line exactly: URL: <the source url>
- then 3-8 bullet points with concrete facts, numbers, dates, and positions
- last line exactly: RELIABILITY: <one short phrase, e.g. official docs, vendor blog, forum thread>

If the source says nothing useful about the topic, output only the URL line and
the words NOT RELEVANT.

## step: synthesize
Produce the final research briefing on: {topic}

Analyst notes, one block per source, in citation order (first block = [1]):

{notes}

Write the briefing in markdown, at most ~400 words:
- Start with a "## TL;DR" section: 2-3 sentences answering "what do I need to know?"
- Then 2-4 short "##" sections with the key facts, numbers, and trade-offs,
  citing sources inline as [1], [2] matching the note order above.
- Skip sources marked NOT RELEVANT; if coverage is thin or sources disagree,
  say so plainly instead of papering over it.
