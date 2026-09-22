"""Seminar-demo orchestration for claim extraction, retrieval, and verification."""

from __future__ import annotations

from threading import Lock

from app.config import Settings
from app.models.claim_extractor import ClaimExtractor
from app.models.schemas import (
    AnalyzeResponse,
    AnalyzedClaim,
    ClaimStatus,
)
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.retrieval.sample_knowledge_base import SAMPLE_KNOWLEDGE_BASE
from app.retrieval.vector_store import QdrantVectorStore
from app.verification.nli_service import NLIService
from app.verification.verification_service import VerificationService


class AnalysisService:
    """Connect claim extraction, evidence retrieval, and NLI verification."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings
        self._claim_extractor: ClaimExtractor | None = None
        self._evidence_retriever: EvidenceRetriever | None = None
        self._verification_service: VerificationService | None = None
        self._knowledge_base_ready = False
        self._initialization_lock = Lock()

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            from app.config import settings

            self._settings = settings
        return self._settings

    @property
    def claim_extractor(self) -> ClaimExtractor:
        if self._claim_extractor is None:
            self._claim_extractor = ClaimExtractor(settings=self.settings)
        return self._claim_extractor

    @property
    def evidence_retriever(self) -> EvidenceRetriever:
        if self._evidence_retriever is None:
            embeddings = EmbeddingService(settings=self.settings)
            vector_store = QdrantVectorStore(
                embedding_service=embeddings,
                settings=self.settings,
            )
            self._evidence_retriever = EvidenceRetriever(vector_store)
        return self._evidence_retriever

    @property
    def verification_service(self) -> VerificationService:
        if self._verification_service is None:
            self._verification_service = VerificationService(
                evidence_retriever=self.evidence_retriever,
                nli_service=NLIService(),
            )
        return self._verification_service

    def _ensure_sample_knowledge_base(self) -> None:
        """Populate the in-memory demo collection exactly once per process."""
        if self._knowledge_base_ready:
            return

        with self._initialization_lock:
            if not self._knowledge_base_ready:
                self.evidence_retriever.ingest_documents(
                    SAMPLE_KNOWLEDGE_BASE
                )
                self._knowledge_base_ready = True

    async def analyze(self, response_text: str) -> AnalyzeResponse:
        """Run extraction → retrieval → NLI verification."""

        extraction = await self.claim_extractor.extract(response_text)

        self._ensure_sample_knowledge_base()

        analyzed_claims: list[AnalyzedClaim] = []

        for claim in extraction.claims:
            verification = self.verification_service.verify_claim(
                claim,
                top_k=self.settings.top_k_results,
            )

            decision = self.verification_service.decide_claim(
                verification
            )

            if decision.status == "SUPPORTED":
                status = ClaimStatus.SUPPORTED
            elif decision.status == "CONTRADICTED":
                status = ClaimStatus.CONTRADICTED
            else:
                status = ClaimStatus.UNCERTAIN

            analyzed_claims.append(
                AnalyzedClaim(
                    id=claim.id,
                    text=claim.text,
                    evidence=[
                        result.evidence
                        for result in verification.results
                    ],
                    status=status,
                    confidence=decision.confidence,
                    nli_label=decision.nli_label,
                )
            )

        return AnalyzeResponse(
            original_response=response_text,
            claims=analyzed_claims,
        )
