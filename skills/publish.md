---
name: publish
description: Write a page on a topic and publish it to loon's internal website.
args: [topic]
steps:
  - {name: write,     kind: llm,  masque: briefer, output: page}
  - {name: publish,   kind: tool, tool: publish_page, output: result}
---

## step: write
Write a page about: {topic}

Produce a clear, well-structured page in **markdown** for a reader who wants to understand
this topic. Use a top "# " title line, then "## " sections with concise prose, lists, and
tables where they help. Do not include a preamble like "here is your page" — output only the
page content itself, starting with the title.
