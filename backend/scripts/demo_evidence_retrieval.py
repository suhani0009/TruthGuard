"""Run the Stage 1 to Stage 2 flow against the included local knowledge base."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings
from app.models.claim_extractor import ClaimExtractor
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.retrieval.sample_knowledge_base import SAMPLE_KNOWLEDGE_BASE
from app.retrieval.vector_store import QdrantVectorStore


async def main() -> None:
    settings = Settings(llm_api_key="", top_k_results=3, similarity_threshold=0.0)
    extractor = ClaimExtractor(settings=settings)
    embeddings = EmbeddingService(settings=settings)
    store = QdrantVectorStore(embedding_service=embeddings, settings=settings)
    retriever = EvidenceRetriever(store)

    ingested = retriever.ingest_documents(SAMPLE_KNOWLEDGE_BASE)
    extraction = await extractor.extract(
        "Google was founded in 1999. The Earth revolves around the Sun."
    )
    evidence_by_claim = retriever.retrieve_for_claims(extraction.claims)

    print(f"Ingested {ingested} sample documents.")
    print(
        json.dumps(
            {
                "claims": [claim.model_dump() for claim in extraction.claims],
                "evidence": {
                    claim_id: [item.model_dump() for item in evidence]
                    for claim_id, evidence in evidence_by_claim.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
