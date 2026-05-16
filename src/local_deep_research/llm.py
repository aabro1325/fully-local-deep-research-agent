from __future__ import annotations

import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from local_deep_research.config import Settings

Message = dict[str, str]

http_logger = logging.getLogger("local_deep_research.llm.http")


class LLMError(RuntimeError):
    """Raised when the LLM provider fails irrecoverably."""


class LLM(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        stop: list[str] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
    ) -> str: ...

    def estimate_tokens(self, text: str) -> int:
        # Cheap heuristic — good enough for budget tracking. ~4 chars/token.
        return max(1, len(text) // 4)


class OllamaLLM(LLM):
    """Talks to Ollama via its OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.default_model = settings.ollama_model
        self.summarizer_model = settings.ollama_summarizer_model
        self.num_ctx = settings.ollama_num_ctx
        self.timeout = settings.request_timeout_seconds

        event_hooks: dict[str, list] = {}
        debug_stderr = bool(os.environ.get("LDR_DEBUG_HTTP"))
        debug_file_path = os.environ.get("LDR_DEBUG_HTTP_FILE")

        if debug_stderr or debug_file_path or http_logger.isEnabledFor(logging.DEBUG):
            def _emit(msg: str) -> None:
                http_logger.debug(msg)
                if debug_stderr:
                    print(msg, file=sys.stderr, flush=True)
                if debug_file_path:
                    with open(debug_file_path, "a", encoding="utf-8") as f:
                        f.write(msg + "\n")

            def _log_request(req: httpx.Request) -> None:
                _emit(f"\n[ollama→] {req.method} {req.url}")
                if req.content:
                    _emit(req.content.decode("utf-8", errors="replace"))

            def _log_response(resp: httpx.Response) -> None:
                resp.read()
                _emit(
                    f"[ollama←] {resp.status_code} {len(resp.content)}B "
                    f"in {resp.elapsed.total_seconds():.2f}s"
                )
                _emit(resp.text)

            event_hooks = {"request": [_log_request], "response": [_log_response]}

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            event_hooks=event_hooks,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        stop: list[str] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            # Ollama-specific: pass num_ctx through OpenAI-compat options bag
            "options": {"num_ctx": self.num_ctx},
        }
        if stop:
            payload["stop"] = stop

        try:
            r = self._client.post("/v1/chat/completions", json=payload)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Ollama returned {e.response.status_code}: {e.response.text}") from e

        data = r.json()
        if not data.get("choices"):
            raise LLMError(f"Ollama returned no choices: {data}")
        return data["choices"][0]["message"]["content"] or ""

    def close(self) -> None:
        self._client.close()


class GeminiLLM(LLM):
    """Talks to Google Gemini via the google-generativeai SDK."""

    def __init__(self, settings: Settings):
        if settings.google_api_key is None:
            raise LLMError(
                "GOOGLE_API_KEY is required when LDR_LLM_PROVIDER=gemini. "
                "Set it in .env or export it as an environment variable."
            )
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise LLMError(
                "google-generativeai is not installed. "
                "Install with: pip install 'local-deep-research[gemini]'"
            ) from e

        genai.configure(api_key=settings.google_api_key.get_secret_value())
        self._genai = genai
        self.default_model = settings.gemini_model
        self.summarizer_model = settings.gemini_summarizer_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        stop: list[str] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
    ) -> str:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        history = [m for m in messages if m["role"] != "system"]

        gen_model = self._genai.GenerativeModel(
            model_name=model or self.default_model,
            system_instruction="\n\n".join(system_parts) if system_parts else None,
        )
        contents = [
            {
                "role": "user" if m["role"] == "user" else "model",
                "parts": [m["content"]],
            }
            for m in history
        ]
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if stop:
            generation_config["stop_sequences"] = stop

        try:
            response = gen_model.generate_content(
                contents,
                generation_config=generation_config,
            )
        except Exception as e:
            raise LLMError(f"Gemini call failed: {e}") from e

        return (response.text or "").strip()


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "ollama":
        return OllamaLLM(settings)
    if settings.llm_provider == "gemini":
        return GeminiLLM(settings)
    raise LLMError(f"Unknown LLM provider: {settings.llm_provider}")
