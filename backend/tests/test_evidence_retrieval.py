"""Unit and integration-style tests for the Stage 2 retrieval services."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.models.claim_extractor import ClaimExtractor
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.retrieval.sample_knowledge_base import SAMPLE_KNOWLEDGE_BASE
from app.retrieval.vector_store import KnowledgeDocument, QdrantVectorStore


class FakeEmbeddingModel:
    """Small deterministic stand-in that keeps tests offline and fast."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append((texts, kwargs))
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "google" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "earth" in lowered or "sun" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            elif "python" in lowered:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.2, 0.2, 0.2])
        return vectors


@dataclass
class FakeVectorParams:
    size: int
    distance: str


@dataclass
class FakePointStruct:
    id: str
    vector: list[float]
    payload: dict[str, object]


class FakeModels:
    class Distance:
        COSINE = "cosine"

    VectorParams = FakeVectorParams
    PointStruct = FakePointStruct


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, list[FakePointStruct]] = {}
        self.created_configs: list[FakeVectorParams] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, vectors_config: FakeVectorParams) -> None:
        self.collections[collection_name] = []
        self.created_configs.append(vectors_config)

    def upsert(self, collection_name: str, points: list[FakePointStruct], wait: bool) -> None:
        existing = {point.id: point for point in self.collections[collection_name]}
        existing.update({point.id: point for point in points})
        self.collections[collection_name] = list(existing.values())

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        score_threshold: float,
        with_payload: bool,
    ) -> list[SimpleNamespace]:
        hits = [
            SimpleNamespace(
                payload=point.payload,
                score=sum(a * b for a, b in zip(query_vector, point.vector)),
            )
            for point in self.collections[collection_name]
        ]
        return sorted(
            (hit for hit in hits if hit.score >= score_threshold),
            key=lambda hit: hit.score,
            reverse=True,
        )[:limit]


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_api_key="", top_k_results=2, similarity_threshold=0.0)


@pytest.fixture
def embedding_service(settings: Settings) -> EmbeddingService:
    return EmbeddingService(settings=settings, model=FakeEmbeddingModel())


@pytest.fixture
def vector_store(
    settings: Settings, embedding_service: EmbeddingService
) -> QdrantVectorStore:
    return QdrantVectorStore(
        embedding_service=embedding_service,
        settings=settings,
        client=FakeQdrantClient(),
        models_module=FakeModels,
    )


class TestEmbeddingService:
    def test_generates_normalized_document_embeddings(
        self, embedding_service: EmbeddingService
    ) -> None:
        vectors = embedding_service.embed_documents(["Google was founded in 1998."])

        assert vectors == [[1.0, 0.0, 0.0]]
        _, kwargs = embedding_service.model.calls[0]
        assert kwargs["normalize_embeddings"] is True
        assert embedding_service.vector_size == 3

    def test_prefixes_queries_with_bge_instruction(
        self, embedding_service: EmbeddingService
    ) -> None:
        embedding_service.embed_query("Google was founded in 1999.")

        encoded_text, _ = embedding_service.model.calls[0]
        assert encoded_text == [
            "Represent this sentence for searching relevant passages: "
            "Google was founded in 1999."
        ]


class TestQdrantVectorStore:
    def test_ingestion_creates_collection_and_preserves_payload(
        self, vector_store: QdrantVectorStore
    ) -> None:
        document = KnowledgeDocument(
            text="Google was founded in 1998 by Larry Page and Sergey Brin.",
            source="Sample Knowledge Base",
            url="https://example.com/google",
            metadata={"topic": "Google"},
        )

        assert vector_store.ingest([document]) == 1
        client = vector_store.client
        stored = client.collections[vector_store.collection_name][0]
        assert client.created_configs[0].size == 3
        assert stored.payload == {
            "text": document.text,
            "source": "Sample Knowledge Base",
            "url": "https://example.com/google",
            "metadata": {"topic": "Google"},
            "source_type": "local",
        }

    def test_similarity_search_returns_top_evidence(
        self, vector_store: QdrantVectorStore
    ) -> None:
        vector_store.ingest(SAMPLE_KNOWLEDGE_BASE)

        evidence = vector_store.search("Google was founded in 1999.", limit=1)

        assert len(evidence) == 1
        assert evidence[0].text.startswith("Google was founded in 1998")
        assert evidence[0].source == "Sample Knowledge Base"
        assert evidence[0].url == "https://example.com/google"
        assert evidence[0].similarity_score == 1.0

    def test_reingestion_updates_instead_of_duplicating(
        self, vector_store: QdrantVectorStore
    ) -> None:
        document = KnowledgeDocument(text="The Earth revolves around the Sun.", source="KB")

        vector_store.ingest([document])
        vector_store.ingest([document])

        assert len(vector_store.client.collections[vector_store.collection_name]) == 1

    @pytest.mark.skipif(
        importlib.util.find_spec("qdrant_client") is None,
        reason="Requires the optional qdrant-client runtime dependency.",
    )
    def test_real_in_memory_qdrant_integration(
        self, settings: Settings, embedding_service: EmbeddingService
    ) -> None:
        """Exercise the actual installed Qdrant API without a Qdrant server."""
        from qdrant_client import QdrantClient

        store = QdrantVectorStore(
            embedding_service=embedding_service,
            settings=settings,
            client=QdrantClient(location=":memory:"),
        )
        store.ingest(SAMPLE_KNOWLEDGE_BASE)

        evidence = store.search("Google was founded in 1999.", limit=1)

        assert evidence[0].text.startswith("Google was founded in 1998")


class TestStageOneToTwoFlow:
    @pytest.mark.asyncio
    async def test_extracted_claims_receive_evidence(
        self, settings: Settings, vector_store: QdrantVectorStore
    ) -> None:
        extractor = ClaimExtractor(settings=settings)
        retriever = EvidenceRetriever(vector_store)
        retriever.ingest_documents(SAMPLE_KNOWLEDGE_BASE)

        extracted = await extractor.extract(
            "Google was founded in 1999. The Earth revolves around the Sun."
        )
        evidence_by_claim = retriever.retrieve_for_claims(extracted.claims, top_k=1)

        assert extracted.method == "fallback"
        assert list(evidence_by_claim) == ["claim_1", "claim_2"]
        assert evidence_by_claim["claim_1"][0].text.startswith("Google was founded")
        assert evidence_by_claim["claim_2"][0].text == "The Earth revolves around the Sun."
