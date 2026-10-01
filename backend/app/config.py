"""
TruthLens Configuration.

Loads settings from environment variables / .env file using Pydantic Settings.
All external service credentials are configured here.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- LLM Configuration ---
    llm_api_key: str = Field(
        default="",
        description="API key for the OpenAI-compatible LLM provider.",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the LLM API (OpenAI, Groq, Ollama, etc.).",
    )
    llm_model: str = Field(
        default="gpt-3.5-turbo",
        description="Model name to use for claim extraction and correction.",
    )

    # --- Embedding Model ---
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Sentence-transformer model for embeddings.",
    )

    # --- NLI Verification Model ---
    nli_model: str = Field(
        default="cross-encoder/nli-deberta-v3-base",
        description="Cross-encoder model for NLI-based claim verification.",
    )

    # --- Tavily Web Search (Optional) ---
    tavily_api_key: Optional[str] = Field(
        default=None,
        description="Tavily API key. If not set, only local evidence is used.",
    )

    # --- Qdrant Vector Database ---
    qdrant_url: Optional[str] = Field(
        default=None,
        description="Qdrant server URL. If not set, in-memory mode is used.",
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key for cloud/authenticated instances.",
    )
    qdrant_collection_name: str = Field(
        default="truthlens_knowledge",
        description="Name of the Qdrant collection for the knowledge base.",
    )

    # --- MongoDB (Optional) ---
    mongodb_uri: Optional[str] = Field(
        default=None,
        description="MongoDB connection URI. If not set, history is not persisted.",
    )
    mongodb_db_name: str = Field(
        default="truthlens",
        description="MongoDB database name.",
    )

    # --- Retrieval Settings ---
    top_k_results: int = Field(
        default=5,
        description="Number of top evidence passages to retrieve per claim.",
    )
    similarity_threshold: float = Field(
        default=0.3,
        description="Minimum similarity score to consider evidence relevant.",
    )

    # --- API Settings ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins for the frontend.",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @property
    def llm_available(self) -> bool:
        """Check if LLM API is configured with a real key."""
        return bool(self.llm_api_key and self.llm_api_key != "your-api-key-here")

    @property
    def tavily_available(self) -> bool:
        """Check if Tavily is configured."""
        return bool(self.tavily_api_key)

    @property
    def mongodb_available(self) -> bool:
        """Check if MongoDB is configured."""
        return bool(self.mongodb_uri)

    @property
    def qdrant_is_remote(self) -> bool:
        """Check if Qdrant is configured to use a remote server."""
        return bool(self.qdrant_url)


# Singleton settings instance
settings = Settings()
