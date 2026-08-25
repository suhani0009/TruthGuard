"""Stage 2 orchestration from extracted claims to retrieved evidence."""

from __future__ import annotations

from collections.abc import Iterable

from app.models.schemas import Claim, Evidence
from app.retrieval.vector_store import KnowledgeDocument, QdrantVectorStore


class EvidenceRetriever:
    """Reuses one vector store to ingest evidence and retrieve it per claim."""

    def __init__(self, vector_store: QdrantVectorStore) -> None:
        self._vector_store = vector_store

    def ingest_documents(self, documents: Iterable[KnowledgeDocument]) -> int:
        """Add local knowledge-base passages to the evidence store."""
        return self._vector_store.ingest(documents)

    def retrieve_for_claim(self, claim: Claim, top_k: int | None = None) -> list[Evidence]:
        """Retrieve top-k evidence passages for a Stage 1 claim."""
        return self._vector_store.search(claim.text, limit=top_k)

    def retrieve_for_claims(
        self, claims: Iterable[Claim], top_k: int | None = None
    ) -> dict[str, list[Evidence]]:
        """Preserve Stage 1 claim IDs while returning evidence for every claim."""
        return {
            claim.id: self.retrieve_for_claim(claim, top_k=top_k)
            for claim in claims
        }
