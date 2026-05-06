from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from local_deep_research.config import Settings


class FirecrawlError(RuntimeError):
    """Raised when the Firecrawl instance returns a non-success response."""


@dataclass
class SearchResult:
    url: str
    title: str
    description: str
    markdown: str | None = None


@dataclass
class ScrapeResult:
    url: str
    title: str
    markdown: str
    metadata: dict[str, Any]


class FirecrawlClient:
    """Thin client for a self-hosted Firecrawl instance.

    Defaults assume the OSS Firecrawl Docker setup at http://localhost:3002,
    but works against Firecrawl Cloud too if you set the base URL + API key.
    """

    def __init__(self, settings: Settings):
        self.base_url = settings.firecrawl_base_url.rstrip("/")
        self.timeout = settings.request_timeout_seconds
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = (
            settings.firecrawl_api_key.get_secret_value()
            if settings.firecrawl_api_key is not None
            else ""
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
        )

    def close(self) -> None:
        self._client.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        scrape: bool = False,
    ) -> list[SearchResult]:
        body: dict[str, Any] = {"query": query, "limit": limit}
        if scrape:
            body["scrapeOptions"] = {"formats": ["markdown"], "onlyMainContent": True}

        try:
            r = self._client.post("/v1/search", json=body)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise FirecrawlError(
                f"Firecrawl /v1/search failed ({e.response.status_code}): {e.response.text[:300]}"
            ) from e

        payload = r.json()
        if not payload.get("success", False):
            raise FirecrawlError(f"Firecrawl /v1/search not successful: {payload}")

        results: list[SearchResult] = []
        for item in payload.get("data", []) or []:
            results.append(
                SearchResult(
                    url=item.get("url") or item.get("link") or "",
                    title=item.get("title") or "",
                    description=item.get("description") or item.get("snippet") or "",
                    markdown=item.get("markdown"),
                )
            )
        return [r for r in results if r.url]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    def scrape(self, url: str, *, only_main_content: bool = True) -> ScrapeResult:
        body = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": only_main_content,
        }
        try:
            r = self._client.post("/v1/scrape", json=body)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise FirecrawlError(
                f"Firecrawl /v1/scrape failed ({e.response.status_code}): {e.response.text[:300]}"
            ) from e

        payload = r.json()
        if not payload.get("success", False):
            raise FirecrawlError(f"Firecrawl /v1/scrape not successful: {payload}")

        data = payload.get("data") or {}
        metadata = data.get("metadata") or {}
        return ScrapeResult(
            url=metadata.get("sourceURL") or url,
            title=metadata.get("title") or "",
            markdown=data.get("markdown") or "",
            metadata=metadata,
        )

    def health(self) -> bool:
        """Liveness probe.

        The OSS Firecrawl image does not consistently expose a health endpoint
        across versions, so we treat the server as reachable if any of these
        respond with anything other than a transport error: GET /v1/health,
        GET /, or POST /v1/scrape (a 4xx from the API itself proves the
        server is up — only network errors mean it's not).
        """
        probes = [
            ("GET", "/v1/health", None),
            ("GET", "/", None),
            ("POST", "/v1/scrape", {}),
        ]
        for method, path, body in probes:
            try:
                if method == "GET":
                    r = self._client.get(path, timeout=10.0)
                else:
                    r = self._client.post(path, json=body, timeout=10.0)
            except httpx.HTTPError:
                continue
            if 200 <= r.status_code < 500:
                return True
        return False
