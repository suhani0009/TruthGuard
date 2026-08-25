"""Qdrant-backed storage for TruthLens evidence passages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
from uuid import NAMESPACE_URL, uuid5

from app.config import Settings
from app.models.schemas import Evidence
from app.retrieval.embedding_service import EmbeddingService


@dataclass(frozen=True)
class KnowledgeDocument:
    """An evidence passage that can be embedded and stored in Qdrant."""

    text: str
    source: str
    url: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_type: str = "local"


class QdrantVectorStore:
    """Creates, populates, and queries the configured Qdrant collection."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        settings: Settings | None = None,
        client: Any | None = None,
        models_module: Any | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._settings = settings
        self._client = client
        self._models = models_module
        self._collection_ready = False

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            from app.config import settings

            self._settings = settings
        return self._settings

    @property
    def collection_name(self) -> str:
        return self.settings.qdrant_collection_name

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as error:
                raise RuntimeError(
                    "Stage 2 requires qdrant-client. "
                    "Install backend/requirements.txt before using retrieval."
                ) from error

            client_options: dict[str, Any] = {}
            if self.settings.qdrant_is_remote:
                client_options["url"] = self.settings.qdrant_url
                if self.settings.qdrant_api_key:
                    client_options["api_key"] = self.settings.qdrant_api_key
            else:
                client_options["location"] = ":memory:"
            self._client = QdrantClient(**client_options)
        return self._client

    @property
    def models(self) -> Any:
        if self._models is None:
            try:
                from qdrant_client import models
            except ImportError as error:
                raise RuntimeError(
                    "Stage 2 requires qdrant-client. "
                    "Install backend/requirements.txt before using retrieval."
                ) from error
            self._models = models
        return self._models

    def ensure_collection(self) -> None:
        """Create the collection once, leaving any existing data untouched."""
        if self._collection_ready:
            return
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.models.VectorParams(
                    size=self._embedding_service.vector_size,
                    distance=self.models.Distance.COSINE,
                ),
            )
        self._collection_ready = True

    def ingest(self, documents: Iterable[KnowledgeDocument]) -> int:
        """Embed and upsert documents, returning the number of passages stored."""
        document_list = list(documents)
        if not document_list:
            return 0

        self.ensure_collection()
        vectors = self._embedding_service.embed_documents(
            [document.text for document in document_list]
        )
        points = [
            self.models.PointStruct(
                id=self._document_id(document),
                vector=vector,
                payload={
                    "text": document.text,
                    "source": document.source,
                    "url": document.url,
                    "metadata": dict(document.metadata),
                    "source_type": document.source_type,
                },
            )
            for document, vector in zip(document_list, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(points)

    def search(self, claim: str, limit: int | None = None) -> list[Evidence]:
        """Return the most semantically relevant evidence passages for a claim."""
        self.ensure_collection()
        top_k = limit if limit is not None else self.settings.top_k_results
        if top_k <= 0:
            return []

        query_vector = self._embedding_service.embed_query(claim)
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=self.settings.similarity_threshold,
                with_payload=True,
            )
        else:
            # qdrant-client 1.19+ replaced ``search`` with ``query_points``.
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=self.settings.similarity_threshold,
                with_payload=True,
            ).points
        return [
            Evidence(
                text=str(hit.payload["text"]),
                source=str(hit.payload.get("source") or "Unknown"),
                url=hit.payload.get("url"),
                similarity_score=float(hit.score),
                source_type=str(hit.payload.get("source_type") or "local"),
            )
            for hit in hits
        ]

    @staticmethod
    def _document_id(document: KnowledgeDocument) -> str:
        """Create a stable ID so re-ingesting a document updates its vector."""
        identity = json.dumps(
            {
                "text": document.text,
                "source": document.source,
                "url": document.url,
                "metadata": dict(document.metadata),
            },
            sort_keys=True,
            default=str,
        )
        return str(uuid5(NAMESPACE_URL, identity))
