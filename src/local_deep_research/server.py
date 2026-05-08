from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from local_deep_research.agent import ResearchAgent
from local_deep_research.config import Settings


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    max_iterations: int | None = None
    time_limit_minutes: float | None = None
    provider: Literal["ollama", "gemini"] | None = None
    model: str | None = None
    include_messages: bool = False


class NoteOut(BaseModel):
    source_id: int
    url: str
    title: str
    goal: str
    relevant: bool
    rationale: str
    evidence: list[str]
    summary: str


class ToolEventOut(BaseModel):
    tool: str
    arguments: dict[str, Any]
    observation: str


class ResearchResponse(BaseModel):
    question: str
    final_report: str
    answer_summary: str
    notes: list[NoteOut]
    events: list[ToolEventOut]
    iterations: int
    terminated_reason: str
    elapsed_seconds: float
    messages: list[dict[str, Any]] | None = None


# Local LLMs are GPU-bound — serialize concurrent runs instead of letting them thrash.
_run_lock = threading.Lock()


def _settings_for_request(req: ResearchRequest) -> Settings:
    settings = Settings()
    if req.provider:
        settings.llm_provider = req.provider
    if req.model:
        if settings.llm_provider == "ollama":
            settings.ollama_model = req.model
        else:
            settings.gemini_model = req.model
    if req.max_iterations is not None:
        settings.max_iterations = req.max_iterations
    if req.time_limit_minutes is not None:
        settings.time_limit_minutes = req.time_limit_minutes
    return settings


def create_app() -> FastAPI:
    app = FastAPI(title="local-deep-research", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/research", response_model=ResearchResponse)
    def research(req: ResearchRequest) -> ResearchResponse:
        if not _run_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409,
                detail="A research run is already in progress. Local LLMs cannot run in parallel.",
            )
        try:
            settings = _settings_for_request(req)
            agent = ResearchAgent(settings)
            try:
                result = agent.run(req.question)
            finally:
                agent.close()
        finally:
            _run_lock.release()

        return ResearchResponse(
            question=result.question,
            final_report=result.final_report,
            answer_summary=result.answer_summary,
            notes=[NoteOut(**asdict(n)) for n in result.notes],
            events=[ToolEventOut(**asdict(e)) for e in result.events],
            iterations=result.iterations,
            terminated_reason=result.terminated_reason,
            elapsed_seconds=result.elapsed_seconds,
            messages=result.messages if req.include_messages else None,
        )

    return app


app = create_app()


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
