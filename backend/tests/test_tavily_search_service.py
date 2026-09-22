from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.retrieval.tavily_search_service import (
    TavilySearchService,
)


class FakeTavilyClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_search_normalizes_tavily_results() -> None:
    response = {
        "results": [
            {
                "title": "Google History",
                "url": "https://example.org/google",
                "content": "Google was founded in 1998.",
                "score": 0.92,
            },
            {
                "title": "Python History",
                "url": "https://example.org/python",
                "content": "Python was first released in 1991.",
                "score": 0.87,
            },
        ]
    }

    client = FakeTavilyClient(response)

    settings = Settings(
        llm_api_key="",
        tavily_api_key="test-tavily-key",
    )

    service = TavilySearchService(
        settings=settings,
        client=client,
    )

    results = service.search(
        "Google founding date",
        max_results=2,
    )

    assert len(results) == 2

    assert results[0].title == "Google History"
    assert results[0].url == "https://example.org/google"
    assert results[0].content == "Google was founded in 1998."
    assert results[0].score == 0.92

    assert client.calls == [
        {
            "query": "Google founding date",
            "search_depth": "advanced",
            "max_results": 2,
            "include_answer": False,
            "include_raw_content": False,
        }
    ]


def test_search_skips_results_without_url_or_content() -> None:
    response = {
        "results": [
            {
                "title": "Valid",
                "url": "https://example.org/valid",
                "content": "Valid evidence.",
                "score": 0.9,
            },
            {
                "title": "Missing URL",
                "url": "",
                "content": "Some content.",
                "score": 0.8,
            },
            {
                "title": "Missing content",
                "url": "https://example.org/missing",
                "content": "",
                "score": 0.7,
            },
        ]
    }

    client = FakeTavilyClient(response)

    settings = Settings(
        llm_api_key="",
        tavily_api_key="test-tavily-key",
    )

    service = TavilySearchService(
        settings=settings,
        client=client,
    )

    results = service.search("test")

    assert len(results) == 1
    assert results[0].title == "Valid"


def test_empty_query_is_rejected() -> None:
    settings = Settings(
        llm_api_key="",
        tavily_api_key="test-tavily-key",
    )

    service = TavilySearchService(
        settings=settings,
        client=FakeTavilyClient({"results": []}),
    )

    with pytest.raises(ValueError):
        service.search("   ")


def test_invalid_max_results_is_rejected() -> None:
    settings = Settings(
        llm_api_key="",
        tavily_api_key="test-tavily-key",
    )

    service = TavilySearchService(
        settings=settings,
        client=FakeTavilyClient({"results": []}),
    )

    with pytest.raises(ValueError):
        service.search("Google", max_results=0)


def test_missing_tavily_key_raises_clear_error() -> None:
    settings = Settings(llm_api_key="")

    service = TavilySearchService(settings=settings)

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        service.search("Google")