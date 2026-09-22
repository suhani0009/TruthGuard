"""Stage 2 orchestration from claims to retrieved evidence."""

from __future__ import annotations

from collections.abc import Iterable

from app.models.schemas import Claim, Evidence
from app.retrieval.tavily_search_service import TavilySearchService
from app.retrieval.vector_store import KnowledgeDocument, QdrantVectorStore


class EvidenceRetriever:
    """Retrieves evidence from the vector store and optionally from Tavily."""

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

    def search_web(
        self,
        claim: Claim,
        max_results: int = 5,
    ) -> list[KnowledgeDocument]:
        """Search Tavily and convert web results into KnowledgeDocuments."""

        if not claim.text or not claim.text.strip():
            raise ValueError("Claim cannot be empty.")

        if self._tavily_service is None:
            raise RuntimeError("Tavily search service is not configured.")

        results = self._tavily_service.search(
            query=claim.text,
            max_results=max_results,
        )

        return [
            KnowledgeDocument(
                text=result.content,
                source=result.title or "Web Search",
                url=result.url,
                metadata={
                    "tavily_score": result.score,
                },
                source_type="web",
            )
            for result in results
        ]

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