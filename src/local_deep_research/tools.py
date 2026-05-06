from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from local_deep_research.config import Settings
from local_deep_research.firecrawl_client import FirecrawlClient, FirecrawlError
from local_deep_research.llm import LLM
from local_deep_research.prompts import SUMMARIZE_PAGE_PROMPT


@dataclass
class Note:
    """A single piece of evidence the agent has gathered from a URL."""

    source_id: int
    url: str
    title: str
    goal: str
    relevant: bool
    rationale: str
    evidence: list[str]
    summary: str


@dataclass
class ToolEvent:
    """One observable step in the research run — useful for streaming UIs."""

    tool: str
    arguments: dict[str, Any]
    observation: str


@dataclass
class NotesStore:
    """Append-only collection of notes with stable, monotonic source ids.

    The same URL is never recorded twice; subsequent visits return the existing id.
    """

    notes: list[Note] = field(default_factory=list)
    _by_url: dict[str, int] = field(default_factory=dict)

    def upsert(self, note_kwargs: dict[str, Any]) -> Note:
        url = note_kwargs["url"]
        if url in self._by_url:
            existing = self.notes[self._by_url[url] - 1]
            return existing
        source_id = len(self.notes) + 1
        note = Note(source_id=source_id, **note_kwargs)
        self.notes.append(note)
        self._by_url[url] = source_id
        return note


class Toolbox:
    """Dispatches tool calls produced by the agent."""

    def __init__(self, settings: Settings, llm: LLM, firecrawl: FirecrawlClient):
        self.settings = settings
        self.llm = llm
        self.firecrawl = firecrawl
        self.notes = NotesStore()
        self._summarizer_model = self._pick_summarizer_model()

    def _pick_summarizer_model(self) -> str | None:
        # OllamaLLM and GeminiLLM both expose a `summarizer_model` attribute.
        return getattr(self.llm, "summarizer_model", None)

    # --- public dispatch -------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "search":
            return self._tool_search(arguments)
        if name == "visit":
            return self._tool_visit(arguments)
        if name == "think":
            return self._tool_think(arguments)
        if name == "answer":
            return self._tool_answer(arguments)
        return f"ERROR: unknown tool '{name}'. Valid tools: search, visit, think, answer."

    # --- individual tools ------------------------------------------------

    def _tool_search(self, args: dict[str, Any]) -> str:
        queries = args.get("queries") or []
        if isinstance(queries, str):
            queries = [queries]
        if not queries:
            return "ERROR: search requires `queries` (a non-empty list of strings)."

        limit = self.settings.max_search_results
        blocks: list[str] = []
        for q in queries[:5]:
            try:
                results = self.firecrawl.search(q, limit=limit, scrape=False)
            except FirecrawlError as e:
                blocks.append(f"### Query: {q}\n[search failed: {e}]")
                continue
            if not results:
                blocks.append(f"### Query: {q}\n(no results)")
                continue
            lines = [f"### Query: {q}"]
            for i, r in enumerate(results, start=1):
                desc = r.description.strip().replace("\n", " ")
                if len(desc) > 240:
                    desc = desc[:240] + "…"
                lines.append(f"{i}. {r.title}\n   {r.url}\n   {desc}")
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks) if blocks else "(no search results)"

    def _tool_visit(self, args: dict[str, Any]) -> str:
        url = (args.get("url") or "").strip()
        goal = (args.get("goal") or "").strip()
        if not url:
            return "ERROR: visit requires a `url` argument."
        if not goal:
            return "ERROR: visit requires a `goal` argument describing what you are looking for."

        try:
            scrape = self.firecrawl.scrape(url)
        except FirecrawlError as e:
            return f"[fetch failed for {url}: {e}]"

        markdown = scrape.markdown or ""
        if not markdown.strip():
            return f"[fetched {url} but no content was extractable]"

        truncated = markdown[: self.settings.page_char_limit]
        prompt = SUMMARIZE_PAGE_PROMPT.format(
            goal=goal,
            url=scrape.url,
            title=scrape.title or "(no title)",
            markdown=truncated,
        )
        raw = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self._summarizer_model,
            temperature=0.2,
            max_tokens=1500,
        )

        parsed = _safe_json_parse(raw)
        if parsed is None:
            return (
                f"[summarizer returned non-JSON for {url}; "
                f"first 500 chars of raw output: {raw[:500]}]"
            )

        note = self.notes.upsert(
            {
                "url": scrape.url,
                "title": scrape.title or scrape.url,
                "goal": goal,
                "relevant": bool(parsed.get("relevant", True)),
                "rationale": str(parsed.get("rationale", "")),
                "evidence": [str(x) for x in (parsed.get("evidence") or [])][:8],
                "summary": str(parsed.get("summary", "")),
            }
        )

        ev = "\n".join(f"  - {q}" for q in note.evidence) or "  (none extracted)"
        return (
            f"[source {note.source_id}] {note.title}\n"
            f"URL: {note.url}\n"
            f"Relevant: {note.relevant}\n"
            f"Rationale: {note.rationale}\n"
            f"Summary: {note.summary}\n"
            f"Evidence quotes:\n{ev}"
        )

    def _tool_think(self, args: dict[str, Any]) -> str:
        thought = (args.get("thought") or "").strip()
        if not thought:
            return "ERROR: think requires a non-empty `thought` argument."
        return f"[thought recorded: {thought[:200]}{'…' if len(thought) > 200 else ''}]"

    def _tool_answer(self, args: dict[str, Any]) -> str:
        summary = (args.get("summary") or "").strip()
        if not summary:
            return "ERROR: answer requires a `summary` argument."
        return f"[answer accepted: {summary[:200]}{'…' if len(summary) > 200 else ''}]"


def _safe_json_parse(raw: str) -> dict[str, Any] | None:
    """Try hard to extract a JSON object from a model response.

    Local models often add fences or stray prose. We strip code fences and
    trim to the outermost {...} before parsing.
    """
    text = raw.strip()
    if text.startswith("```"):
        # ```json\n...\n```  or  ```\n...\n```
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None
