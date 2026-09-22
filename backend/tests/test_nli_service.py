"""Tests for the DeBERTa NLI verification service."""

from __future__ import annotations

import pytest

from app.verification.nli_service import NLIService


@pytest.fixture(scope="module")
def nli_service() -> NLIService:
    return NLIService()


def test_entailment(nli_service: NLIService) -> None:
    evidence = (
        "Google was founded in 1998 by Larry Page and Sergey Brin."
    )
    claim = "Google was founded in 1998."

    result = nli_service.verify(claim, evidence)

    print(result)

    assert result.label == "entailment"
    assert 0.0 <= result.confidence <= 1.0


def test_contradiction(nli_service: NLIService) -> None:
    evidence = (
        "Google was founded in 1998 by Larry Page and Sergey Brin."
    )
    claim = "Google was founded in 1999."

    result = nli_service.verify(claim, evidence)

    print(result)

    assert result.label == "contradiction"
    assert 0.0 <= result.confidence <= 1.0


def test_neutral(nli_service: NLIService) -> None:
    evidence = (
        "Google was founded in 1998 by Larry Page and Sergey Brin."
    )
    claim = "Google has offices in Paris."

    result = nli_service.verify(claim, evidence)

    print(result)

    assert result.label == "neutral"
    assert 0.0 <= result.confidence <= 1.0


def test_empty_claim_is_rejected(nli_service: NLIService) -> None:
    with pytest.raises(ValueError):
        nli_service.verify(
            "",
            "Google was founded in 1998.",
        )


def test_empty_evidence_is_rejected(nli_service: NLIService) -> None:
    with pytest.raises(ValueError):
        nli_service.verify(
            "Google was founded in 1998.",
            "",
        )