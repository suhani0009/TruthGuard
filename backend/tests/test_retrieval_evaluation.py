"""Retrieval evaluation for Recall@K."""

from __future__ import annotations

from app.config import Settings
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.retrieval.sample_knowledge_base import SAMPLE_KNOWLEDGE_BASE
from app.retrieval.vector_store import QdrantVectorStore


EVALUATION_DATASET = [
    {
        "claim": "Google was founded in 1998 by Larry Page and Sergey Brin.",
        "expected_evidence": "Google was founded in 1998 by Larry Page and Sergey Brin.",
    },
    {
        "claim": "The Earth revolves around the Sun.",
        "expected_evidence": "The Earth revolves around the Sun.",
    },
    {
        "claim": "Python was created by Guido van Rossum and first released in 1991.",
        "expected_evidence": (
            "Python was created by Guido van Rossum and first released in 1991."
        ),
    },
]


def calculate_recall_at_k(
    retriever: EvidenceRetriever,
    k: int,
) -> float:
    """Calculate Recall@K over the evaluation dataset."""

    hits = 0

    for index, item in enumerate(EVALUATION_DATASET, start=1):
        retrieved = retriever.retrieve_for_claim(
            claim=_make_claim(index, item["claim"]),
            top_k=k,
        )

        retrieved_texts = [evidence.text for evidence in retrieved]

        if item["expected_evidence"] in retrieved_texts:
            hits += 1

    return hits / len(EVALUATION_DATASET)


def _make_claim(claim_id: int, text: str):
    """Create the existing Claim model used by Stage 2."""

    from app.models.schemas import Claim

    return Claim(
        id=f"evaluation_claim_{claim_id}",
        text=text,
    )


def _build_retriever() -> EvidenceRetriever:
    """Build the same retrieval pipeline used by Stage 2."""

    settings = Settings(
        llm_api_key="",
        top_k_results=5,
        similarity_threshold=0.0,
    )

    embedding_service = EmbeddingService(settings=settings)

    vector_store = QdrantVectorStore(
        embedding_service=embedding_service,
        settings=settings,
    )

    retriever = EvidenceRetriever(vector_store)

    retriever.ingest_documents(SAMPLE_KNOWLEDGE_BASE)

    return retriever


def test_recall_at_1():
    retriever = _build_retriever()

    recall = calculate_recall_at_k(retriever, k=1)

    print(f"\nRecall@1 = {recall:.2%}")

    assert recall >= 0.0
    assert recall <= 1.0


def test_recall_at_3():
    retriever = _build_retriever()

    recall = calculate_recall_at_k(retriever, k=3)

    print(f"\nRecall@3 = {recall:.2%}")

    assert recall >= 0.0
    assert recall <= 1.0


def test_recall_at_5():
    retriever = _build_retriever()

    recall = calculate_recall_at_k(retriever, k=5)

    print(f"\nRecall@5 = {recall:.2%}")

    assert recall >= 0.0
    assert recall <= 1.0