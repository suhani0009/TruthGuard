from app.models.schemas import Evidence
from app.retrieval.embedding_service import EmbeddingService
from app.verification.evidence_focus_service import EvidenceFocusService


def test_focus_selects_relevant_sentence():
    embedding_service = EmbeddingService()
    service = EvidenceFocusService(embedding_service)

    evidence = Evidence(
        text=(
            "Earth rotates on its axis every 24 hours. "
            "Earth revolves about the Sun once every year. "
            "The Moon orbits Earth. "
            "Earth has an elliptical orbit around the Sun."
        ),
        source="Test Source",
        url="https://example.com",
        similarity_score=0.8,
        source_type="web",
    )

    result = service.focus(
        claim="The Earth revolves around the Sun.",
        evidence=evidence,
        max_sentences=2,
    )

    print("\nOriginal:")
    print(result.evidence.text)

    print("\nFocused:")
    print(result.text)

    assert result.text
    assert "revolves about the Sun" in result.text