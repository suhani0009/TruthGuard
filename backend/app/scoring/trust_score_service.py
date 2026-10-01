"""Stage 4 claim-verification trust-score calculation."""

from __future__ import annotations

from collections.abc import Iterable

from app.models.schemas import ClaimStatus


class TrustScoreService:
    """Calculate an overall score from independently verified claims."""

    CLAIM_SCORES: dict[ClaimStatus, float] = {
        ClaimStatus.SUPPORTED: 1.0,
        ClaimStatus.UNCERTAIN: 0.5,
        ClaimStatus.CONTRADICTED: 0.0,
    }

    def calculate(self, statuses: Iterable[ClaimStatus]) -> float:
        """Return the mean claim score as a percentage, or zero for no claims."""
        status_list = list(statuses)
        if not status_list:
            return 0.0

        total = sum(self.CLAIM_SCORES[status] for status in status_list)
        return (total / len(status_list)) * 100
