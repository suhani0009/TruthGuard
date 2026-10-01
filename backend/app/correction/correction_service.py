"""
Evidence-based correction service for TruthGuard.

Generates corrections only for contradicted claims using
the evidence already retrieved by the verification pipeline.
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import AsyncOpenAI

from app.config import Settings
from app.models.schemas import ClaimStatus, Evidence

logger = logging.getLogger(__name__)


UNCERTAIN_MESSAGE = (
    "Insufficient reliable evidence to provide a correction."
)

LLM_FAILURE_MESSAGE = (
    "A correction could not be generated from the available evidence."
)


CORRECTION_SYSTEM_PROMPT = """You are an evidence-based factual correction assistant.

Your task is to correct a factual claim that has been classified as CONTRADICTED.

STRICT RULES:
1. Use ONLY the evidence provided to you.
2. Do NOT use outside knowledge.
3. Do NOT invent facts.
4. Do NOT add information that is not supported by the evidence.
5. Preserve the subject and meaning of the original claim whenever possible.
6. Return ONLY the corrected factual statement.
7. Do not provide explanations, introductions, bullet points, or commentary.
8. If the evidence does not provide enough information for a reliable correction,
   return exactly:
   "Insufficient reliable evidence to provide a correction."
"""


class CorrectionService:
    """
    Generates evidence-grounded corrections for contradicted claims.

    Supported claims are returned unchanged.
    Uncertain claims receive no fabricated correction.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings
        self._client: Optional[AsyncOpenAI] = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            from app.config import settings

            self._settings = settings

        return self._settings

    def _get_client(self) -> Optional[AsyncOpenAI]:
        """Lazily create the OpenAI-compatible client."""
        if self._client is None and self.settings.llm_available:
            self._client = AsyncOpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
            )

        return self._client

    async def correct(
        self,
        claim_text: str,
        status: ClaimStatus,
        evidence: list[Evidence],
    ) -> Optional[str]:
        """
        Generate a correction based on verification status and evidence.

        Args:
            claim_text: Original factual claim.
            status: Verification status of the claim.
            evidence: Evidence already retrieved for this claim.

        Returns:
            Corrected claim text, original claim, or None depending on status.
        """

        # Supported claims do not need correction.
        if status == ClaimStatus.SUPPORTED:
            return claim_text

        # Never invent corrections for uncertain claims.
        if status == ClaimStatus.UNCERTAIN:
            return UNCERTAIN_MESSAGE

        # A contradicted claim requires evidence.
        if status != ClaimStatus.CONTRADICTED:
            return None

        if not evidence:
            logger.warning(
                "Cannot correct contradicted claim without evidence: %s",
                claim_text,
            )
            return LLM_FAILURE_MESSAGE

        client = self._get_client()

        if client is None:
            logger.warning(
                "LLM is not configured; correction unavailable."
            )
            return LLM_FAILURE_MESSAGE

        evidence_text = self._format_evidence(evidence)

        user_prompt = f"""Original claim:
{claim_text}

Verification status:
CONTRADICTED

Retrieved evidence:
{evidence_text}

Using ONLY the evidence above, provide the corrected factual claim.
Return only the corrected claim."""

        try:
            completion = await client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": CORRECTION_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=300,
            )

            corrected = (
                completion.choices[0].message.content or ""
            ).strip()

            if not corrected:
                logger.warning(
                    "Correction LLM returned an empty response."
                )
                return LLM_FAILURE_MESSAGE

            return corrected

        except Exception as exc:
            logger.warning(
                "Correction LLM failed: %s",
                exc,
            )
            return LLM_FAILURE_MESSAGE

    @staticmethod
    def _format_evidence(
        evidence: list[Evidence],
    ) -> str:
        """Format retrieved evidence for the correction prompt."""

        formatted = []

        for index, item in enumerate(evidence, start=1):
            source = item.source or "Unknown source"
            url = item.url or "No URL available"

            formatted.append(
                f"Evidence {index}:\n"
                f"Source: {source}\n"
                f"URL: {url}\n"
                f"Text: {item.text}"
            )

        return "\n\n".join(formatted)