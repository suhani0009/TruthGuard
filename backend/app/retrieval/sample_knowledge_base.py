"""Small local evidence set for running TruthLens without web search."""

from app.retrieval.vector_store import KnowledgeDocument


SAMPLE_KNOWLEDGE_BASE: tuple[KnowledgeDocument, ...] = (
    KnowledgeDocument(
        text="Google was founded in 1998 by Larry Page and Sergey Brin.",
        source="Sample Knowledge Base",
        url="https://example.com/google",
        metadata={"topic": "Google", "fact_type": "founding"},
    ),
    KnowledgeDocument(
        text="The Earth revolves around the Sun.",
        source="Sample Knowledge Base",
        url="https://example.com/earth-orbit",
        metadata={"topic": "astronomy"},
    ),
    KnowledgeDocument(
        text="Python was created by Guido van Rossum and first released in 1991.",
        source="Sample Knowledge Base",
        url="https://example.com/python",
        metadata={"topic": "Python", "fact_type": "history"},
    ),
)
