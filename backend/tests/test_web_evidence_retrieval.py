from unittest.mock import Mock

from app.models.schemas import Claim
from app.retrieval.evidence_retriever import EvidenceRetriever
from app.retrieval.tavily_search_service import TavilySearchResult
from app.retrieval.vector_store import KnowledgeDocument


def test_tavily_results_are_converted_to_web_documents():
    tavily = Mock()

    tavily.search.return_value = [
        TavilySearchResult(
            title="Google History",
            url="https://example.org/google",
            content="Google was founded in 1998.",
            score=0.91,
        ),
        TavilySearchResult(
            title="Company Information",
            url="https://example.org/company",
            content="Google was founded by Larry Page and Sergey Brin.",
            score=0.84,
        ),
    ]

    vector_store = Mock()

    retriever = EvidenceRetriever(
        vector_store=vector_store,
        tavily_service=tavily,
    )

    claim = Claim(
        id="claim_1",
        text="Google was founded in 1998.",
    )

    documents = retriever.search_web(claim)

    assert len(documents) == 2

    assert isinstance(documents[0], KnowledgeDocument)

    assert documents[0].text == "Google was founded in 1998."
    assert documents[0].source == "Google History"
    assert documents[0].url == "https://example.org/google"
    assert documents[0].source_type == "web"

    assert documents[0].metadata["tavily_score"] == 0.91

    tavily.search.assert_called_once_with(
        query="Google was founded in 1998.",
        max_results=5,
    )


def test_web_documents_can_be_ingested_into_vector_store():
    tavily = Mock()

    tavily.search.return_value = [
        TavilySearchResult(
            title="Google History",
            url="https://example.org/google",
            content="Google was founded in 1998.",
            score=0.91,
        )
    ]

    vector_store = Mock()
    vector_store.ingest.return_value = 1

    retriever = EvidenceRetriever(
        vector_store=vector_store,
        tavily_service=tavily,
    )

    claim = Claim(
        id="claim_1",
        text="Google was founded in 1998.",
    )

    documents = retriever.search_web(claim)

    count = retriever.ingest_documents(documents)

    assert count == 1

    vector_store.ingest.assert_called_once()

    ingested_documents = vector_store.ingest.call_args.args[0]

    assert len(ingested_documents) == 1
    assert ingested_documents[0].source_type == "web"
    assert ingested_documents[0].url == "https://example.org/google"


def test_search_web_requires_tavily_service():
    vector_store = Mock()

    retriever = EvidenceRetriever(
        vector_store=vector_store,
    )

    claim = Claim(
        id="claim_1",
        text="Google was founded in 1998.",
    )

    try:
        retriever.search_web(claim)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Tavily search service is not configured" in str(exc)