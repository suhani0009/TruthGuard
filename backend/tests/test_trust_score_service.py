"""Tests for the Stage 4 overall trust-score calculation."""

from app.models.schemas import ClaimStatus
from app.scoring.trust_score_service import TrustScoreService


def test_supported_claims_produce_full_trust() -> None:
    assert TrustScoreService().calculate([ClaimStatus.SUPPORTED]) == 100.0


def test_contradicted_claims_produce_zero_trust() -> None:
    assert TrustScoreService().calculate([ClaimStatus.CONTRADICTED]) == 0.0


def test_uncertain_claims_contribute_half_weight() -> None:
    assert TrustScoreService().calculate([ClaimStatus.UNCERTAIN]) == 50.0


def test_supported_and_contradicted_claims_produce_fifty() -> None:
    score = TrustScoreService().calculate(
        [ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED]
    )

    assert score == 50.0


def test_no_claims_produce_zero_trust() -> None:
    assert TrustScoreService().calculate([]) == 0.0
