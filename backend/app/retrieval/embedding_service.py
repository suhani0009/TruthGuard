"""Embedding service for the Stage 2 evidence-retrieval pipeline."""

from __future__ import annotations

from threading import Lock
from typing import Any, Sequence

from app.config import Settings


class EmbeddingService:
    """Creates normalized BGE embeddings while loading each model only once."""

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
    _models: dict[str, Any] = {}
    _models_lock = Lock()

    def __init__(self, settings: Settings | None = None, model: Any | None = None) -> None:
        self._settings = settings
        self._model = model

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            from app.config import settings

            self._settings = settings
        return self._settings

    @property
    def model_name(self) -> str:
        return self.settings.embedding_model

    @property
    def model(self) -> Any:
        """Return the injected model or the shared sentence-transformer instance."""
        if self._model is not None:
            return self._model

        with self._models_lock:
            cached_model = self._models.get(self.model_name)
            if cached_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as error:
                    raise RuntimeError(
                        "Stage 2 requires sentence-transformers. "
                        "Install backend/requirements.txt before using retrieval."
                    ) from error
                cached_model = SentenceTransformer(self.model_name)
                self._models[self.model_name] = cached_model
            self._model = cached_model
        return self._model

    @property
    def vector_size(self) -> int:
        """Return the configured model's embedding dimension."""
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed evidence documents using the BGE document representation."""
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed one claim using BGE's retrieval-query instruction."""
        embeddings = self._encode([f"{self.QUERY_INSTRUCTION}{query}"])
        return embeddings[0]

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(value) for value in vector] for vector in vectors]
