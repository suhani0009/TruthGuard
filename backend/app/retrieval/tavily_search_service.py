from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings


@dataclass(frozen=True)
class TavilySearchResult:
    """A normalized result returned by Tavily."""

    title: str
    url: str
    content: str
    score: float | None = None


class TavilySearchService:
    """Search the web through Tavily's HTTP API and normalize the results."""

    TAVILY_API_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self._client = client

    @property
    def client(self) -> Any:
        """Return the Tavily API key."""

        if self._client is None:
            if not self.settings.tavily_available:
                raise RuntimeError(
                    "Tavily is not configured. "
                    "Set TAVILY_API_KEY in the environment."
                )

            self._client = self.settings.tavily_api_key

        return self._client

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[TavilySearchResult]:
        """Search Tavily through its REST API."""

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if max_results < 1:
            raise ValueError("max_results must be at least 1.")

        payload = {
            "api_key": self.client,
            "query": query.strip(),
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            response = httpx.post(
                self.TAVILY_API_URL,
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Tavily search failed with HTTP "
                f"{exc.response.status_code}: {exc.response.text}"
            ) from exc

        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Could not connect to Tavily: {exc}"
            ) from exc

        data = response.json()

        results: list[TavilySearchResult] = []

        for item in data.get("results", []):
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            content = str(item.get("content", "")).strip()

            if not url or not content:
                continue

            results.append(
                TavilySearchResult(
                    title=title,
                    url=url,
                    content=content,
                    score=item.get("score"),
                )
            )

        return results