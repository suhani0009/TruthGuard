
"""Orchestration for claim extraction, web evidence retrieval, and verification."""

from __future__ import annotations

from app.config import Settings
from app.models.claim_extractor import ClaimExtractor
from app.models.schemas import (
    AnalyzeResponse,
    AnalyzedClaim,
    ClaimStatus,
)
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.retrieval.tavily_search_service import TavilySearchService
from app.retrieval.vector_store import QdrantVectorStore
from app.verification.nli_service import NLIService
from app.verification.verification_service import VerificationService


class AnalysisService:
    """Connect claim extraction, web evidence retrieval, and NLI verification."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings
        self._claim_extractor: ClaimExtractor | None = None
        self._evidence_retriever: EvidenceRetriever | None = None
        self._verification_service: VerificationService | None = None
        self._tavily_service: TavilySearchService | None = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            from app.config import settings

            self._settings = settings

        return self._settings

    @property
    def claim_extractor(self) -> ClaimExtractor:
        if self._claim_extractor is None:
            self._claim_extractor = ClaimExtractor(
                settings=self.settings
            )

        return self._claim_extractor

    @property
    def tavily_service(self) -> TavilySearchService:
        if self._tavily_service is None:
            self._tavily_service = TavilySearchService(
                settings=self.settings
            )

        return self._tavily_service

    @property
    def evidence_retriever(self) -> EvidenceRetriever:
        if self._evidence_retriever is None:
            embeddings = EmbeddingService(
                settings=self.settings
            )

            vector_store = QdrantVectorStore(
                embedding_service=embeddings,
                settings=self.settings,
            )

            self._evidence_retriever = EvidenceRetriever(
                vector_store=vector_store,
                tavily_service=self.tavily_service,
            )

        return self._evidence_retriever

    @property
    def verification_service(self) -> VerificationService:
        if self._verification_service is None:
            self._verification_service = VerificationService(
                evidence_retriever=self.evidence_retriever,
                nli_service=NLIService(),
            )

        return self._verification_service

    async def analyze(self, response_text: str) -> AnalyzeResponse:
        """Run extraction → real web retrieval → NLI verification."""

        extraction = await self.claim_extractor.extract(
            response_text
        )

        analyzed_claims: list[AnalyzedClaim] = []

        for claim in extraction.claims:
            # Step 2: Search the real web through Tavily.
            web_documents = self.evidence_retriever.search_web(
                claim,
                max_results=self.settings.top_k_results,
            )

            # Store the retrieved web evidence in Qdrant.
            self.evidence_retriever.ingest_documents(
                web_documents
            )

            # Step 3: Retrieve relevant evidence and verify with NLI.
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
