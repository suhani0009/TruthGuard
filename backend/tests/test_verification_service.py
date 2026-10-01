"""Integration tests for Retrieval + NLI verification."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.models.claim_extractor import ClaimExtractor
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.retrieval.sample_knowledge_base import SAMPLE_KNOWLEDGE_BASE
from app.retrieval.vector_store import QdrantVectorStore
from app.verification.nli_service import NLIService
from app.verification.verification_service import VerificationService


@pytest.fixture(scope="module")
def verification_service() -> VerificationService:
    settings = Settings(
        llm_api_key="",
        top_k_results=3,
        similarity_threshold=0.0,
    )

    embedding_service = EmbeddingService(settings=settings)

    vector_store = QdrantVectorStore(
        embedding_service=embedding_service,
        settings=settings,
    )

    retriever = EvidenceRetriever(vector_store)
    retriever.ingest_documents(SAMPLE_KNOWLEDGE_BASE)

    nli_service = NLIService()

    return VerificationService(
        evidence_retriever=retriever,
        nli_service=nli_service,
    )


@pytest.mark.parametrize(
    ("claim_text", "expected_label"),
    [
        (
            "Google was founded in 1998.",
            "entailment",
        ),
        (
            "Google was founded in 1999.",
            "contradiction",
        ),
        (
            "Google has offices in Paris.",
            "neutral",
        ),
    ],
)
def test_retrieval_plus_nli(
    verification_service: VerificationService,
    claim_text: str,
    expected_label: str,
) -> None:
    from app.models.schemas import Claim

    claim = Claim(
        id="test_claim",
        text=claim_text,
    )

    result = verification_service.verify_claim(
        claim,
        top_k=1,
    )

    assert len(result.results) == 1
    assert result.results[0].label == expected_label
    assert 0.0 <= result.results[0].confidence <= 1.0

    print(
        f"\nClaim: {claim_text}"
        f"\nEvidence: {result.results[0].evidence.text}"
        f"\nNLI: {result.results[0].label}"
        f"\nConfidence: {result.results[0].confidence:.4f}"
    )


@pytest.mark.parametrize(
    ("claim_text", "expected_status"),
    [
        (
            "Google was founded in 1998.",
            "SUPPORTED",
        ),
        (
            "Google was founded in 1999.",
            "CONTRADICTED",
        ),
        (
            "Google has offices in Paris.",
            "UNCERTAIN",
        ),
    ],
)
def test_claim_level_decision(
    verification_service: VerificationService,
    claim_text: str,
    expected_status: str,
) -> None:
    from app.models.schemas import Claim

    claim = Claim(
        id="test-claim",
        text=claim_text,
    )

    verification = verification_service.verify_claim(
        claim,
        top_k=1,
    )

    decision = verification_service.decide_claim(verification)

    print(f"\nClaim: {claim.text}")
    print(f"Decision: {decision.status}")
    print(f"NLI: {decision.nli_label}")
    print(f"Confidence: {decision.confidence:.4f}")

    assert decision.status == expected_status
    assert decision.nli_label in {
        "entailment",
        "contradiction",
        "neutral",
    }
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.evidence is not None
