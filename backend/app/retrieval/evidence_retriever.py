"""Stage 2 orchestration from claims to retrieved evidence."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse

from app.models.schemas import Claim, Evidence
from app.retrieval.tavily_search_service import TavilySearchService
from app.retrieval.vector_store import KnowledgeDocument, QdrantVectorStore


class EvidenceRetriever:
    """Retrieves evidence from the vector store and optionally from Tavily."""

    LOW_QUALITY_DOMAINS = {
        "facebook.com",
        "www.facebook.com",
        "x.com",
        "twitter.com",
        "www.twitter.com",
        "reddit.com",
        "www.reddit.com",
        "instagram.com",
        "www.instagram.com",
        "tiktok.com",
        "www.tiktok.com",
    }

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        tavily_service: TavilySearchService | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._tavily_service = tavily_service

    def ingest_documents(
        self,
        documents: Iterable[KnowledgeDocument],
    ) -> int:
        """Add evidence documents to the vector store."""
        return self._vector_store.ingest(documents)

    @classmethod
    def _is_low_quality_source(cls, url: str) -> bool:
        """Identify obvious social-media/user-generated sources."""

        if not url:
            return True

        hostname = urlparse(url).netloc.lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname in {
            domain.removeprefix("www.")
            for domain in cls.LOW_QUALITY_DOMAINS
        }

    def search_web(
        self,
        claim: Claim,
        max_results: int = 5,
    ) -> list[KnowledgeDocument]:
        """Search Tavily and convert reliable web results into KnowledgeDocuments."""

        if not claim.text or not claim.text.strip():
            raise ValueError("Claim cannot be empty.")

        if self._tavily_service is None:
            raise RuntimeError("Tavily search service is not configured.")

        # Request extra results so filtering does not leave us with too
        # little useful evidence.
        search_results = self._tavily_service.search(
            query=claim.text,
            max_results=max(max_results * 2, 10),
        )

        documents: list[KnowledgeDocument] = []

        for result in search_results:
            if self._is_low_quality_source(result.url):
                continue

            documents.append(
                KnowledgeDocument(
                    text=result.content,
                    source=result.title or "Web Search",
                    url=result.url,
                    metadata={
                        "tavily_score": result.score,
                    },
                    source_type="web",
                )
            )

            if len(documents) >= max_results:
                break

        return documents

    def retrieve_for_claim(
        self,
        claim: Claim,
        top_k: int | None = None,
    ) -> list[Evidence]:
        """Retrieve top-k evidence passages for a Stage 1 claim."""

        return self._vector_store.search(
            claim.text,
            limit=top_k,
        )

    def retrieve_for_claims(
        self,
        claims: Iterable[Claim],
        top_k: int | None = None,
    ) -> dict[str, list[Evidence]]:
        """Preserve Stage 1 claim IDs while returning evidence."""

        return {
            claim.id: self.retrieve_for_claim(
                claim,
                top_k=top_k,
            )
            for claim in claims
        }