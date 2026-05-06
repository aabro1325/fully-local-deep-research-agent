# local-deep-research

A fully local deep research agent. Plug it into a local LLM (Ollama) and a self-hosted Firecrawl instance, ask it a research question, and it will plan, search, scrape, and synthesize a long-form, cited markdown report — without sending your query to any third-party API.

The only **optional** API is a Google AI key for Gemini, in case you want to swap in a frontier model for the planning + writing steps and keep everything else local.

## Why this exists

There are several great open implementations of "deep research" agents:

- [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) — a tight ReAct loop with carefully designed XML-tagged tool calls and stop tokens.
- [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) — supervisor + parallel researchers in LangGraph, with mid-pipeline page summarization.
- [nickscamara/open-deep-research](https://github.com/nickscamara/open-deep-research) — a Next.js app that wires Firecrawl `/search` + `/extract` into a tight inner research loop.

All three depend on a hosted LLM provider, a hosted search/scrape API, or both. **`local-deep-research` borrows the best ideas from each and runs them against tools you already host yourself.**

## Architecture

A single ReAct agent drives the whole run, with these design choices borrowed verbatim:

| Idea | Source | Why |
| --- | --- | --- |
| `<tool_call>` / `<tool_response>` XML protocol | Alibaba | Robust against JSON-quoting failures from local models |
| Stop tokens on `</tool_call>` and `<tool_response>` | Alibaba | Prevents the model from forging tool outputs |
| Batched search queries (`queries: string[]`) | Alibaba | One turn fans out to many queries |
| Two-stage browse (fetch → summarizer LLM) | Alibaba + LangChain | Keeps raw HTML out of the agent's context |
| `think` tool as a no-op reflection slot | LangChain | Cheap planning lift without a separate node |
| Per-page summarization with a smaller model | LangChain | Biggest token-saving lever |
| Progressive truncation on context-overflow retries | LangChain | Graceful degradation |
| Force-answer prompt on budget exhaustion | Alibaba | No mid-stream truncation |
| Firecrawl `/search` + `/scrape` for web access | nickscamara | Self-hostable end-to-end |
| Time budget exposed inline in the prompt | nickscamara | Lets the model self-pace |

**Tools available to the agent:**

- `search(queries)` — parallel web searches via Firecrawl `/v1/search`
- `visit(url, goal)` — fetch a page via Firecrawl `/v1/scrape`, then summarize with the local LLM into `{relevant, rationale, evidence[], summary}`
- `think(thought)` — reflection sink; appends a planning note
- `answer(summary)` — terminator; triggers final report synthesis

**Two LLM calls per visit:** the visit tool runs a *summarizer* call (default: a smaller, faster model) before showing anything to the agent. This is the single largest cost saver.

## Prerequisites

You need three things running on the local box:

1. **Ollama** ([install](https://ollama.com/)) with at least one decent tool-following model pulled. Tested defaults:
   ```bash
   ollama pull qwen2.5:14b      # main agent model
   ollama pull qwen2.5:7b       # cheaper page summarizer
   ```
   Llama 3.1 / 3.3, Mistral, and DeepSeek-R1 work too — see the *Choosing a model* section.

2. **Firecrawl** running locally. The fastest path is the Docker setup from the [Firecrawl repo](https://github.com/mendableai/firecrawl):
   ```bash
   git clone https://github.com/mendableai/firecrawl.git
   cd firecrawl
   docker compose up -d
   # Defaults to http://localhost:3002
   ```
   For self-hosted Firecrawl with no API key you can leave `LDR_FIRECRAWL_API_KEY` blank.

3. **Python 3.10+**.

## Install

```bash
git clone <this repo> local_deep_research
cd local_deep_research
python -m venv .venv
source .venv/bin/activate
pip install -e .

# optional — only if you plan to use Gemini:
pip install -e '.[gemini]'
```

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
$EDITOR .env
```

## Verify your setup

```bash
ldr doctor
```

You should see `OK` next to both the LLM provider and Firecrawl. If not, the table tells you exactly which knob to turn.

## Run a research session

```bash
ldr research "What are the most important open problems in mechanistic interpretability as of 2026?"
```

Useful flags:

```bash
ldr research "..." \
  --output report.md \
  --max-iterations 30 \
  --time-limit 30 \
  --provider ollama \
  --model llama3.1:70b
```

While it runs you'll see streaming `search`/`visit`/`think` events; on completion it prints the markdown report (or writes it to `--output`).

## Use Gemini instead of Ollama

```bash
export GOOGLE_API_KEY=...
ldr research "..." --provider gemini --model gemini-2.5-pro
```

Everything else stays local — Firecrawl still runs on your machine, and only the model calls leave the box.

## Choosing a model

The agent must follow a strict XML tool-call format and emit valid JSON inside it. Recommended local models, roughly best-to-worst at this:

- `qwen2.5:14b` / `qwen2.5:32b` — best ReAct discipline at modest sizes
- `llama3.1:70b` / `llama3.3:70b` — strong if you have the VRAM
- `mistral-small:24b` — solid all-rounder
- `qwen2.5:7b` is good for the *summarizer* slot but flaky as the main agent

The summarizer model is set separately via `LDR_OLLAMA_SUMMARIZER_MODEL` so you can run a small fast model for per-page summarization and reserve the big one for planning.

## Programmatic use

```python
from local_deep_research import ResearchAgent, Settings

settings = Settings(max_iterations=15)
agent = ResearchAgent(settings)
result = agent.run("How does Apple's M4 Neural Engine compare to NVIDIA's Jetson Orin?")

print(result.final_report)
for note in result.notes:
    print(note.source_id, note.url)
```

## Files

```
src/local_deep_research/
├── agent.py             # ReAct loop + tool-call parsing
├── cli.py               # `ldr research` and `ldr doctor`
├── config.py            # pydantic-settings env config
├── firecrawl_client.py  # /v1/search + /v1/scrape
├── llm.py               # Ollama (OpenAI-compat) + Gemini providers
├── prompts.py           # System prompt, summarizer prompt, report prompt
├── report.py            # Final report composition with truncation retries
└── tools.py             # search / visit / think / answer dispatcher
```

## Limits and what's next

This is intentionally a v1: single-agent ReAct, no parallel researchers. With a local LLM the GPU is the bottleneck anyway, so async fan-out doesn't actually parallelize. Likely future work:

- A supervisor + parallel-researchers mode for the Gemini path (where API parallelism *does* help)
- Optional vector-store memory for very long sessions
- A web UI that consumes the streamed `ToolEvent`s

## License

MIT.
