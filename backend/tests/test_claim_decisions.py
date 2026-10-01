"""Focused Stage 3 tests for claim-level NLI status mapping."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.models.schemas import Claim, Evidence
from app.verification.nli_service import NLIResult
from app.verification.verification_service import VerificationService


@dataclass
class StubRetriever:
    evidence: list[Evidence]

    def retrieve_for_claim(self, claim: Claim, top_k: int | None = None) -> list[Evidence]:
        return self.evidence[:top_k]


class StubNLIService:
    def __init__(self, result: NLIResult) -> None:
        self.result = result

    def verify(self, claim: str, evidence: str) -> NLIResult:
        return self.result


@pytest.mark.parametrize(
    ("nli_label", "expected_status"),
    [
        ("entailment", "SUPPORTED"),
        ("contradiction", "CONTRADICTED"),
        ("neutral", "UNCERTAIN"),
    ],
)
def test_nli_labels_map_to_requested_claim_statuses(
    nli_label: str, expected_status: str
) -> None:
    evidence = Evidence(
        text="Google was founded in 1998.",
        source="Example",
        similarity_score=0.91,
    )
    nli_result = NLIResult(
        label=nli_label,
        confidence=0.87,
        scores={nli_label: 0.87},
    )
    service = VerificationService(
        evidence_retriever=StubRetriever([evidence]),
        nli_service=StubNLIService(nli_result),
    )

    verification = service.verify_claim(
        Claim(id="claim_1", text="Google was founded in 1999."),
        top_k=1,
    )
    decision = service.decide_claim(verification)

    assert decision.status == expected_status
    assert decision.confidence == 0.87
    assert decision.evidence == evidence


def test_missing_evidence_is_uncertain_without_nli_confidence() -> None:
    service = VerificationService(
        evidence_retriever=StubRetriever([]),
        nli_service=StubNLIService(
            NLIResult(label="entailment", confidence=1.0, scores={"entailment": 1.0})
        ),
    )

    decision = service.decide_claim(
        service.verify_claim(Claim(id="claim_1", text="An unverified fact."))
    )

    assert decision.status == "UNCERTAIN"
    assert decision.confidence == 0.0
    assert decision.nli_label is None
    assert decision.evidence is None
