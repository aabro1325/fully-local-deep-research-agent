from __future__ import annotations

from local_deep_research.llm import LLM, LLMError
from local_deep_research.prompts import FINAL_REPORT_PROMPT
from local_deep_research.tools import Note


def compose_report(
    *,
    question: str,
    agent_summary: str,
    notes: list[Note],
    llm: LLM,
    char_budget: int = 80_000,
) -> str:
    """Synthesize a long-form markdown report from collected notes.

    On first failure (typically context overflow), retry with progressively
    truncated notes — borrowed from langchain-ai/open_deep_research.
    """
    relevant_notes = [n for n in notes if n.relevant] or notes
    if not relevant_notes:
        return _fallback_report(question, agent_summary, notes)

    sources_block = "\n".join(f"[{n.source_id}] {n.title} — {n.url}" for n in relevant_notes)

    for truncate_factor in (1.0, 0.7, 0.5, 0.3):
        notes_block = _format_notes(relevant_notes, char_budget=int(char_budget * truncate_factor))
        prompt = FINAL_REPORT_PROMPT.format(
            question=question,
            agent_summary=agent_summary,
            notes=notes_block,
            sources=sources_block,
        )
        try:
            return llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8192,
            ).strip()
        except LLMError:
            if truncate_factor == 0.3:
                raise
            continue

    return _fallback_report(question, agent_summary, relevant_notes)


def _format_notes(notes: list[Note], *, char_budget: int) -> str:
    parts: list[str] = []
    used = 0
    for n in notes:
        ev = "\n".join(f"  - \"{q}\"" for q in n.evidence) or "  (no quotes extracted)"
        block = (
            f"### Source [{n.source_id}] {n.title}\n"
            f"URL: {n.url}\n"
            f"Goal probed: {n.goal}\n"
            f"Summary: {n.summary}\n"
            f"Evidence:\n{ev}\n"
        )
        if used + len(block) > char_budget:
            parts.append(f"### Source [{n.source_id}] {n.title}\nURL: {n.url}\n[truncated]\n")
            continue
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def _fallback_report(question: str, agent_summary: str, notes: list[Note]) -> str:
    """Used when the LLM final-synthesis step fails outright. Better than nothing."""
    lines = [
        f"# Research report\n",
        f"**Question:** {question}\n",
        f"## Agent's summary\n\n{agent_summary or '(none)'}\n",
        "## Collected sources\n",
    ]
    for n in notes:
        lines.append(f"### [{n.source_id}] {n.title}")
        lines.append(f"- URL: {n.url}")
        lines.append(f"- Goal: {n.goal}")
        lines.append(f"- Summary: {n.summary}")
        if n.evidence:
            lines.append("- Evidence:")
            for q in n.evidence:
                lines.append(f"  - {q}")
        lines.append("")
    return "\n".join(lines)
