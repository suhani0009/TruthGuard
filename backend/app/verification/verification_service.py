"""Connect retrieved evidence with DeBERTa NLI verification."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import Claim, Evidence
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.verification.evidence_focus_service import EvidenceFocusService
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
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self._evidence_retriever = evidence_retriever
        self._nli_service = nli_service
        if embedding_service is None:
           embedding_service = EmbeddingService()

        self._evidence_focus_service = EvidenceFocusService(
           embedding_service
    )

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
            focused_evidence = self._evidence_focus_service.focus(
                claim=claim.text,
                evidence=evidence,
                max_sentences=2,
            )

            nli_result = self._nli_service.verify(
                claim=claim.text,
                evidence=focused_evidence.text,
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

    def decide_claim(
        self,
        verification: ClaimVerification,
    ) -> ClaimDecision:
        """Convert NLI results into one final claim-level decision."""

        if not verification.results:
            return ClaimDecision(
                claim=verification.claim,
                status="UNCERTAIN",
                confidence=0.0,
                nli_label=None,
                evidence=None,
            )

        print("\n" + "=" * 70)
        print(f"CLAIM: {verification.claim.text}")

        for index, result in enumerate(verification.results, start=1):
            print(f"\nEvidence {index}:")
            print(
                f"Similarity: "
                f"{result.evidence.similarity_score:.4f}"
            )
            print(f"NLI label: {result.label}")
            print(
                f"NLI confidence: "
                f"{result.confidence:.4f}"
            )
            print(f"NLI scores: {result.scores}")
            print(f"Source: {result.evidence.source}")

        # Prefer strong NLI decisions over embedding similarity.
        contradictions = [
            result
            for result in verification.results
            if result.label == "contradiction"
        ]

        entailments = [
            result
            for result in verification.results
            if result.label == "entailment"
        ]

        if contradictions:
            best_result = max(
                contradictions,
                key=lambda result: result.confidence,
            )
            status = "CONTRADICTED"

        elif entailments:
            best_result = max(
                entailments,
                key=lambda result: result.confidence,
            )
            status = "SUPPORTED"

        else:
            best_result = max(
                verification.results,
                key=lambda result: result.confidence,
            )
            status = "UNCERTAIN"

        return ClaimDecision(
            claim=verification.claim,
            status=status,
            confidence=best_result.confidence,
            nli_label=best_result.label,
            evidence=best_result.evidence,
        )