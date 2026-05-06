from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from local_deep_research.agent import ResearchAgent
from local_deep_research.config import Settings
from local_deep_research.firecrawl_client import FirecrawlClient
from local_deep_research.tools import ToolEvent

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A fully local deep research agent (Ollama + self-hosted Firecrawl).",
)
console = Console()


def _build_settings_with_overrides(
    provider: str | None,
    model: str | None,
    max_iterations: int | None,
    time_limit: float | None,
) -> Settings:
    settings = Settings()
    if provider:
        settings.llm_provider = provider  # type: ignore[assignment]
    if model:
        if settings.llm_provider == "ollama":
            settings.ollama_model = model
        else:
            settings.gemini_model = model
    if max_iterations is not None:
        settings.max_iterations = max_iterations
    if time_limit is not None:
        settings.time_limit_minutes = time_limit
    return settings


@app.command()
def research(
    question: Annotated[str, typer.Argument(help="The research question to investigate.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the final report to this markdown file."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="LLM provider override: ollama | gemini."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Override the main model name."),
    ] = None,
    max_iterations: Annotated[
        int | None,
        typer.Option("--max-iterations", help="Override the max ReAct iterations."),
    ] = None,
    time_limit: Annotated[
        float | None,
        typer.Option("--time-limit", help="Override the wall-clock budget in minutes."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress per-step streaming output."),
    ] = False,
) -> None:
    """Run a deep research session and print or save a long-form report."""
    settings = _build_settings_with_overrides(provider, model, max_iterations, time_limit)
    _print_run_header(settings, question)

    agent = ResearchAgent(settings)
    try:
        on_event = None if quiet else _make_event_printer()
        result = agent.run(question, on_event=on_event)
    finally:
        agent.close()

    console.print()
    console.print(
        Panel(
            f"[bold]Iterations:[/bold] {result.iterations}    "
            f"[bold]Sources collected:[/bold] {len(result.notes)}    "
            f"[bold]Elapsed:[/bold] {result.elapsed_seconds:.1f}s    "
            f"[bold]Termination:[/bold] {result.terminated_reason}",
            title="Research complete",
            border_style="green",
        )
    )

    if output:
        output.write_text(result.final_report, encoding="utf-8")
        console.print(f"[green]Wrote report to {output}[/green]")
    else:
        console.print()
        console.print(Markdown(result.final_report))


@app.command()
def doctor() -> None:
    """Check that Ollama (or Gemini) and Firecrawl are reachable."""
    settings = Settings()
    table = Table(title="Local Deep Research — connectivity check")
    table.add_column("Component")
    table.add_column("Endpoint")
    table.add_column("Status")
    table.add_column("Detail")

    if settings.llm_provider == "ollama":
        ok, detail = _check_ollama(settings)
        table.add_row("LLM (Ollama)", settings.ollama_base_url, _badge(ok), detail)
    else:
        ok = settings.google_api_key is not None
        detail = (
            "GOOGLE_API_KEY set"
            if ok
            else "GOOGLE_API_KEY missing — required when LDR_LLM_PROVIDER=gemini"
        )
        table.add_row("LLM (Gemini)", "google-generativeai", _badge(ok), detail)

    fc_ok, fc_detail = _check_firecrawl(settings)
    table.add_row(
        "Firecrawl",
        settings.firecrawl_base_url,
        _badge(fc_ok),
        fc_detail,
    )

    console.print(table)
    if settings.llm_provider == "ollama":
        ok_overall = _check_ollama(settings)[0] and fc_ok
    else:
        ok_overall = (settings.google_api_key is not None) and fc_ok
    raise typer.Exit(code=0 if ok_overall else 1)


def _check_ollama(settings: Settings) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            r.raise_for_status()
        models = [m.get("name", "") for m in r.json().get("models", [])]
        if not models:
            return True, "reachable but no models pulled — `ollama pull <model>`"
        target = settings.ollama_model
        if target in models:
            return True, f"model '{target}' available ({len(models)} total)"
        return False, f"model '{target}' not found. Pulled: {', '.join(models[:5])}"
    except Exception as e:  # noqa: BLE001 — surface anything as a friendly diagnostic
        return False, f"unreachable: {e}"


def _check_firecrawl(settings: Settings) -> tuple[bool, str]:
    fc = FirecrawlClient(settings)
    try:
        if fc.health():
            return True, "reachable"
        return False, "unreachable (no /v1/health response)"
    except Exception as e:  # noqa: BLE001
        return False, f"error: {e}"
    finally:
        fc.close()


def _badge(ok: bool) -> str:
    return "[green]OK[/green]" if ok else "[red]FAIL[/red]"


def _make_event_printer():
    counter = {"n": 0}

    def _print(event: ToolEvent) -> None:
        counter["n"] += 1
        idx = counter["n"]
        if event.tool == "search":
            queries = event.arguments.get("queries") or []
            console.print(
                f"[cyan]{idx:>3}.[/cyan] [bold]search[/bold] "
                f"[dim]({len(queries)} queries)[/dim]: "
                + ", ".join(f'"{q}"' for q in queries[:3])
                + (" …" if len(queries) > 3 else "")
            )
        elif event.tool == "visit":
            url = event.arguments.get("url", "")
            goal = event.arguments.get("goal", "")
            console.print(
                f"[cyan]{idx:>3}.[/cyan] [bold]visit[/bold] {url}\n"
                f"      [dim]goal:[/dim] {goal[:120]}"
            )
        elif event.tool == "think":
            thought = event.arguments.get("thought", "")
            console.print(f"[cyan]{idx:>3}.[/cyan] [bold]think[/bold] [dim]{thought[:200]}[/dim]")
        elif event.tool == "answer":
            console.print(f"[cyan]{idx:>3}.[/cyan] [bold green]answer[/bold green]")
        else:
            console.print(f"[cyan]{idx:>3}.[/cyan] [yellow]{event.tool}[/yellow]")

    return _print


def _print_run_header(settings: Settings, question: str) -> None:
    if settings.llm_provider == "ollama":
        llm_line = f"Ollama @ {settings.ollama_base_url} — {settings.ollama_model}"
    else:
        llm_line = f"Gemini — {settings.gemini_model}"
    console.print(
        Panel.fit(
            f"[bold]Question:[/bold] {question}\n"
            f"[dim]LLM:[/dim] {llm_line}\n"
            f"[dim]Firecrawl:[/dim] {settings.firecrawl_base_url}\n"
            f"[dim]Budget:[/dim] {settings.max_iterations} iterations / "
            f"{settings.time_limit_minutes:.0f} min",
            title="local-deep-research",
            border_style="blue",
        )
    )


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
