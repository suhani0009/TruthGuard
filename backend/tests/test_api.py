"""HTTP tests for the Stage 1 + Stage 2 seminar-demo API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.config import Settings
from app.models.schemas import AnalyzeResponse, AnalyzedClaim, Evidence


class StubAnalysisService:
    """Avoid model loading while testing HTTP contracts in isolation."""

    def __init__(self) -> None:
        self.received_responses: list[str] = []

    async def analyze(self, response_text: str) -> AnalyzeResponse:
        self.received_responses.append(response_text)
        return AnalyzeResponse(
            original_response=response_text,
            claims=[
                AnalyzedClaim(
                    id="claim_1",
                    text="Google was founded in 1999.",
                    evidence=[
                        Evidence(
                            text="Google was founded in 1998 by Larry Page and Sergey Brin.",
                            source="Sample Knowledge Base",
                            url="https://example.com/google",
                            similarity_score=0.84,
                        )
                    ],
                )
            ],
        )


def make_client() -> tuple[TestClient, StubAnalysisService]:
    service = StubAnalysisService()
    app = create_app(
        app_settings=Settings(llm_api_key=""),
        analysis_service=service,
    )
    return TestClient(app), service


def test_health_endpoint() -> None:
    client, _ = make_client()

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_endpoint_returns_claims_and_evidence() -> None:
    client, service = make_client()
    input_text = "Google was founded in 1999. The Earth revolves around the Sun."

    response = client.post("/api/analyze", json={"response": input_text})

    assert response.status_code == 200
    assert service.received_responses == [input_text]
    payload = response.json()
    assert payload["original_response"] == input_text
    assert payload["claims"][0]["id"] == "claim_1"
    assert payload["claims"][0]["evidence"][0] == {
        "text": "Google was founded in 1998 by Larry Page and Sergey Brin.",
        "source": "Sample Knowledge Base",
        "url": "https://example.com/google",
        "similarity_score": 0.84,
        "source_type": "local",
    }


def test_analyze_rejects_empty_response() -> None:
    client, service = make_client()

    response = client.post("/api/analyze", json={"response": "   "})

    assert response.status_code == 422
    assert service.received_responses == []


def test_analyze_requires_response_field() -> None:
    client, _ = make_client()

    response = client.post("/api/analyze", json={})

    assert response.status_code == 422
