from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings


@dataclass(frozen=True)
class TavilySearchResult:
    """A normalized result returned by Tavily."""

    title: str
    url: str
    content: str
    score: float | None = None


class TavilySearchService:
    """Search the web through Tavily and normalize the results."""

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self._client = client

    @property
    def client(self) -> Any:
        """Create the Tavily client lazily."""

        if self._client is None:
            if not self.settings.tavily_available:
                raise RuntimeError(
                    "Tavily is not configured. "
                    "Set TAVILY_API_KEY in the environment."
                )

            from tavily import TavilyClient

            self._client = TavilyClient(
                api_key=self.settings.tavily_api_key
            )

        return self._client

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[TavilySearchResult]:
        """Search Tavily and return normalized web results."""

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if max_results < 1:
            raise ValueError("max_results must be at least 1.")

        response = self.client.search(
            query=query.strip(),
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )

        results: list[TavilySearchResult] = []

        for item in response.get("results", []):
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