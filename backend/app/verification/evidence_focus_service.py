"""Select focused evidence sentences for NLI verification."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import Evidence
from app.retrieval.embedding_service import EmbeddingService


@dataclass(frozen=True)
class FocusedEvidence:
    """Original evidence plus the focused text used for NLI."""

    evidence: Evidence
    text: str
    similarity_score: float


class EvidenceFocusService:
    """Select the most claim-relevant sentences from evidence."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
    ) -> None:
        self._embedding_service = embedding_service

    def focus(
        self,
        claim: str,
        evidence: Evidence,
        max_sentences: int = 2,
    ) -> FocusedEvidence:
        """Select the most relevant sentences from one evidence passage."""

        sentences = self._split_sentences(evidence.text)

        if not sentences:
            return FocusedEvidence(
                evidence=evidence,
                text=evidence.text,
                similarity_score=evidence.similarity_score,
            )

        if len(sentences) <= max_sentences:
            focused_text = " ".join(sentences)
            return FocusedEvidence(
                evidence=evidence,
                text=focused_text,
                similarity_score=evidence.similarity_score,
            )

        claim_vector = self._embedding_service.embed_query(claim)
        sentence_vectors = self._embedding_service.embed_documents(
            sentences
        )

        scored_sentences = []

        for sentence, vector in zip(
            sentences,
            sentence_vectors,
            strict=True,
        ):
            score = self._cosine_similarity(
                claim_vector,
                vector,
            )
            scored_sentences.append((sentence, score))

        scored_sentences.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        selected = scored_sentences[:max_sentences]

        # Restore the original sentence order for readability.
        selected_sentences = {
            sentence
            for sentence, _ in selected
        }

        focused_text = " ".join(
            sentence
            for sentence in sentences
            if sentence in selected_sentences
        )

        return FocusedEvidence(
            evidence=evidence,
            text=focused_text,
            similarity_score=evidence.similarity_score,
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split a web passage into usable sentences."""

        cleaned = re.sub(r"\s+", " ", text).strip()

        if not cleaned:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+",
            cleaned,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    @staticmethod
    def _cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        """Calculate cosine similarity for normalized vectors."""

        if not vector_a or not vector_b:
            return 0.0

        numerator = sum(
            a * b
            for a, b in zip(vector_a, vector_b, strict=True)
        )

        return float(numerator)