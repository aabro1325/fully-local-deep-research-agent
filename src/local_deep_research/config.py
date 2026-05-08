from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["ollama", "gemini"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LDR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: Provider = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    ollama_summarizer_model: str = "qwen2.5:7b"
    ollama_num_ctx: int = 32768

    google_api_key: SecretStr | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    gemini_model: str = "gemini-2.5-pro"
    gemini_summarizer_model: str = "gemini-2.5-flash"

    firecrawl_base_url: str = "http://localhost:3002"
    firecrawl_api_key: SecretStr | None = None

    max_iterations: int = 20
    max_search_results: int = 8
    time_limit_minutes: float = 15.0
    page_char_limit: int = 20000
    request_timeout_seconds: float = 120.0
    min_notes_before_answer: int = 1
