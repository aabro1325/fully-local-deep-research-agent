from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from local_deep_research.config import Settings
from local_deep_research.firecrawl_client import FirecrawlClient
from local_deep_research.llm import LLM, build_llm
from local_deep_research.prompts import FORCE_ANSWER_PROMPT, system_prompt
from local_deep_research.report import compose_report
from local_deep_research.tools import Note, Toolbox, ToolEvent

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


@dataclass
class ResearchResult:
    question: str
    final_report: str
    answer_summary: str
    notes: list[Note]
    events: list[ToolEvent]
    iterations: int
    terminated_reason: str
    elapsed_seconds: float
    messages: list[dict] = field(default_factory=list)


EventCallback = Callable[[ToolEvent], None]


class ResearchAgent:
    """Single-agent ReAct deep research loop.

    Inspired by Tongyi DeepResearch (Alibaba) for the loop shape and stop-token
    discipline, langchain-ai/open_deep_research for the per-page summarization,
    and nickscamara/open-deep-research for the Firecrawl integration shape.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: LLM | None = None,
        firecrawl: FirecrawlClient | None = None,
    ):
        self.settings = settings or Settings()
        self.llm = llm or build_llm(self.settings)
        self.firecrawl = firecrawl or FirecrawlClient(self.settings)
        self.toolbox = Toolbox(self.settings, self.llm, self.firecrawl)

    def run(
        self,
        question: str,
        *,
        on_event: EventCallback | None = None,
    ) -> ResearchResult:
        start = time.monotonic()
        deadline = start + self.settings.time_limit_minutes * 60.0

        messages: list[dict] = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": question},
        ]
        events: list[ToolEvent] = []
        terminated = "iterations_exhausted"
        answer_summary = ""
        forced = False

        for iteration in range(1, self.settings.max_iterations + 1):
            if time.monotonic() > deadline and not forced:
                messages.append({"role": "user", "content": FORCE_ANSWER_PROMPT})
                forced = True
                terminated = "time_limit"

            raw = self.llm.chat(
                messages=messages,
                stop=["</tool_call>", "<tool_response>"],
                temperature=0.5,
                max_tokens=2048,
            )

            assistant_text = _normalize_assistant_output(raw)
            messages.append({"role": "assistant", "content": assistant_text})

            tool_call = _extract_tool_call(assistant_text)
            if tool_call is None:
                # The model did not produce a valid tool call. Nudge it.
                nudge = (
                    "Your last turn did not contain a valid <tool_call>...</tool_call> "
                    "block. Re-emit a single tool call in the required format."
                )
                messages.append({"role": "user", "content": _wrap_observation(nudge)})
                events.append(
                    ToolEvent(tool="<malformed>", arguments={}, observation=nudge)
                )
                if on_event:
                    on_event(events[-1])
                continue

            name, arguments = tool_call

            if name == "answer":
                answer_summary = (arguments.get("summary") or "").strip()
                events.append(
                    ToolEvent(
                        tool="answer",
                        arguments=arguments,
                        observation=f"final answer accepted ({len(answer_summary)} chars)",
                    )
                )
                if on_event:
                    on_event(events[-1])
                terminated = "answered" if not forced else terminated
                break

            observation = self.toolbox.call(name, arguments)
            events.append(
                ToolEvent(tool=name, arguments=arguments, observation=observation)
            )
            if on_event:
                on_event(events[-1])

            messages.append({"role": "user", "content": _wrap_observation(observation)})

            if forced:
                # We told the model to terminate but it called another tool. Force again.
                messages.append({"role": "user", "content": FORCE_ANSWER_PROMPT})
        else:
            # Loop exhausted without break → force-answer one last time.
            terminated = "iterations_exhausted"
            messages.append({"role": "user", "content": FORCE_ANSWER_PROMPT})
            raw = self.llm.chat(
                messages=messages,
                stop=["</tool_call>", "<tool_response>"],
                temperature=0.3,
                max_tokens=1500,
            )
            assistant_text = _normalize_assistant_output(raw)
            messages.append({"role": "assistant", "content": assistant_text})
            tool_call = _extract_tool_call(assistant_text)
            if tool_call and tool_call[0] == "answer":
                answer_summary = (tool_call[1].get("summary") or "").strip()

        elapsed = time.monotonic() - start

        report = compose_report(
            question=question,
            agent_summary=answer_summary or "(agent did not produce a final summary)",
            notes=self.toolbox.notes.notes,
            llm=self.llm,
            char_budget=self.settings.page_char_limit * 4,
        )

        return ResearchResult(
            question=question,
            final_report=report,
            answer_summary=answer_summary,
            notes=list(self.toolbox.notes.notes),
            events=events,
            iterations=len(events),
            terminated_reason=terminated,
            elapsed_seconds=elapsed,
            messages=messages,
        )

    def close(self) -> None:
        if hasattr(self.llm, "close"):
            self.llm.close()
        self.firecrawl.close()


def _wrap_observation(text: str) -> str:
    return f"<tool_response>\n{text}\n</tool_response>"


def _normalize_assistant_output(raw: str) -> str:
    """The LLM is stopped at `</tool_call>` so the closing tag is missing. Restore it."""
    text = raw.rstrip()
    if "<tool_call>" in text and "</tool_call>" not in text:
        text = text + "\n</tool_call>"
    return text


def _extract_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    m = TOOL_CALL_RE.search(text)
    if m is None:
        return None
    body = m.group("body")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    name = parsed.get("name")
    args = parsed.get("arguments") or {}
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return name, args
