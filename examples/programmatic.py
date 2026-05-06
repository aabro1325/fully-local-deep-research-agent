"""Run a research session from Python.

Prereqs: Ollama and Firecrawl running locally (see README).

    python examples/programmatic.py
"""

from __future__ import annotations

from local_deep_research import ResearchAgent, Settings


def main() -> None:
    settings = Settings(max_iterations=12, time_limit_minutes=8)
    agent = ResearchAgent(settings)
    try:
        result = agent.run(
            "What hardware/software stack do most autonomous warehouse robots use in 2026?",
            on_event=lambda e: print(f"[{e.tool}] {list(e.arguments.keys())}"),
        )
    finally:
        agent.close()

    print("\n" + "=" * 80)
    print(result.final_report)
    print("=" * 80)
    print(
        f"\nIterations: {result.iterations}  "
        f"Sources: {len(result.notes)}  "
        f"Elapsed: {result.elapsed_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
