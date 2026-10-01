"""Stage 1 → Stage 3 → Stage 4 integration with deterministic retrieved evidence."""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.models.schemas import Claim, Evidence
from app.pipeline.analysis_service import AnalysisService


class StaticEvidenceRetriever:
    """Supplies representative Stage 2 results without changing retrieval code."""

    def search_web(self, claim: Claim, max_results: int) -> list[object]:
        return []

    def ingest_documents(self, documents: list[object]) -> int:
        return 0

    def retrieve_for_claim(self, claim: Claim, top_k: int | None = None) -> list[Evidence]:
        if "Google" in claim.text:
            return [
                Evidence(
                    text="Google was founded in 1998 by Larry Page and Sergey Brin.",
                    source="Sample Knowledge Base",
                    url="https://example.com/google",
                    similarity_score=0.84,
                )
            ]
        return [
            Evidence(
                text="The Earth revolves around the Sun.",
                source="Sample Knowledge Base",
                url="https://example.com/earth-orbit",
                similarity_score=0.85,
            )
        ]


def test_google_and_earth_example_is_verified_and_scored(monkeypatch) -> None:
    """Use real DeBERTa against representative evidence returned by Stage 2."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    service = AnalysisService(
        settings=Settings(llm_api_key="", top_k_results=1)
    )
    service._evidence_retriever = StaticEvidenceRetriever()  # type: ignore[assignment]

    result = asyncio.run(
        service.analyze(
            "Google was founded in 1999. The Earth revolves around the Sun."
        )
    )

    assert [claim.status.value for claim in result.claims] == [
        "CONTRADICTED",
        "SUPPORTED",
    ]
    assert all(claim.confidence > 0.0 for claim in result.claims)
    assert result.overall_trust_score == 50.0
