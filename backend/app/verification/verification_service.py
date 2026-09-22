"""Connect retrieved evidence with DeBERTa NLI verification."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import Claim, Evidence
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.verification.nli_service import NLIService


@dataclass(frozen=True)
class EvidenceVerification:
    """NLI result for one retrieved evidence passage."""

    evidence: Evidence
    label: str
    confidence: float
    scores: dict[str, float]


@dataclass(frozen=True)
class ClaimVerification:
    """Verification results for one claim."""

    claim: Claim
    results: list[EvidenceVerification]


# ADD THIS HERE
@dataclass(frozen=True)
class ClaimDecision:
    """Final verification decision for one claim."""

    claim: Claim
    status: str
    confidence: float
    nli_label: str | None
    evidence: Evidence | None


class VerificationService:
    """Runs NLI over evidence retrieved for claims."""

    def __init__(
        self,
        evidence_retriever: EvidenceRetriever,
        nli_service: NLIService,
    ) -> None:
        self._evidence_retriever = evidence_retriever
        self._nli_service = nli_service

    def verify_claim(
        self,
        claim: Claim,
        top_k: int | None = None,
    ) -> ClaimVerification:
        """Retrieve evidence and classify each passage against the claim."""

        evidence_list = self._evidence_retriever.retrieve_for_claim(
            claim,
            top_k=top_k,
        )

        results = []

        for evidence in evidence_list:
            nli_result = self._nli_service.verify(
                claim=claim.text,
                evidence=evidence.text,
            )

            results.append(
                EvidenceVerification(
                    evidence=evidence,
                    label=nli_result.label,
                    confidence=nli_result.confidence,
                    scores=nli_result.scores,
                )
            )

        return ClaimVerification(
            claim=claim,
            results=results,
        )

    # ADD THIS FUNCTION HERE
    def decide_claim(
        self,
        verification: ClaimVerification,
    ) -> ClaimDecision:
        """Convert NLI results into one final claim-level decision."""

        if not verification.results:
            return ClaimDecision(
                claim=verification.claim,
                status="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                nli_label=None,
                evidence=None,
            )

        # Select the strongest NLI result.
        best_result = max(
            verification.results,
            key=lambda result: result.evidence.similarity_score,
        )

        status_map = {
            "entailment": "SUPPORTED",
            "contradiction": "CONTRADICTED",
            "neutral": "INSUFFICIENT_EVIDENCE",
        }

        status = status_map[best_result.label]

        return ClaimDecision(
            claim=verification.claim,
            status=status,
            confidence=best_result.confidence,
            nli_label=best_result.label,
            evidence=best_result.evidence,
        )