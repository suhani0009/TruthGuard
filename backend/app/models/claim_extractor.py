"""
TruthLens Claim Extractor.

Extracts atomic factual claims from an LLM-generated response.

Two extraction strategies:
1. LLM-based: Uses an OpenAI-compatible API to extract structured claims.
2. Fallback: Simple sentence-based heuristic when LLM is unavailable.
"""

import re
import json
import logging
from typing import Optional

from openai import AsyncOpenAI

from app.config import Settings
from app.models.schemas import Claim, ClaimExtractionResult

logger = logging.getLogger(__name__)

# System prompt that instructs the LLM to extract atomic factual claims
CLAIM_EXTRACTION_PROMPT = """You are a factual claim extractor. Your job is to extract individual, atomic factual claims from the given text.

Rules:
1. Each claim must be a single, self-contained factual statement.
2. Split compound sentences into individual claims when they contain multiple facts.
3. Skip greetings, opinions, filler phrases, and conversational statements that cannot be fact-checked.
4. Preserve the original meaning and wording as closely as possible.
5. Each claim should be verifiable against evidence.

Return ONLY a valid JSON object with this exact format:
{
    "claims": [
        {"id": "claim_1", "text": "..."},
        {"id": "claim_2", "text": "..."}
    ]
}

Do NOT include any text outside the JSON object."""


class ClaimExtractor:
    """
    Extracts atomic factual claims from LLM-generated responses.

    Uses an LLM when available, falls back to sentence-based heuristic.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings
        self._client: Optional[AsyncOpenAI] = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            from app.config import settings
            self._settings = settings
        return self._settings

    def _get_client(self) -> Optional[AsyncOpenAI]:
        """Lazily create OpenAI client if LLM is configured."""
        if self._client is None and self.settings.llm_available:
            self._client = AsyncOpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
            )
        return self._client

    async def extract(self, response_text: str) -> ClaimExtractionResult:
        """
        Extract claims from a response. Tries LLM first, falls back to heuristic.

        Args:
            response_text: The LLM-generated response to analyze.

        Returns:
            ClaimExtractionResult with extracted claims and method used.
        """
        if not response_text or not response_text.strip():
            return ClaimExtractionResult(claims=[], method="empty_input")

        # Try LLM-based extraction first
        client = self._get_client()
        if client is not None:
            try:
                result = await self._extract_with_llm(client, response_text)
                if result.claims:
                    return result
                logger.warning("LLM returned no claims, falling back to heuristic.")
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}. Falling back to heuristic.")

        # Fallback to sentence-based extraction
        return self._extract_with_fallback(response_text)

    async def _extract_with_llm(
        self, client: AsyncOpenAI, response_text: str
    ) -> ClaimExtractionResult:
        """Extract claims using an OpenAI-compatible LLM."""
        completion = await client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": CLAIM_EXTRACTION_PROMPT},
                {"role": "user", "content": response_text},
            ],
            temperature=0.0,
            max_tokens=2000,
        )

        raw_content = completion.choices[0].message.content or ""
        claims = self._parse_llm_response(raw_content)
        return ClaimExtractionResult(claims=claims, method="llm")

    def _parse_llm_response(self, raw: str) -> list[Claim]:
        """Parse LLM JSON response into Claim objects, with error recovery."""
        # Try to extract JSON from the response (LLMs sometimes wrap in markdown)
        json_str = self._extract_json(raw)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM JSON response: {raw[:200]}")
            return []

        # Handle both {"claims": [...]} and plain [...]
        if isinstance(data, dict) and "claims" in data:
            raw_claims = data["claims"]
        elif isinstance(data, list):
            raw_claims = data
        else:
            logger.warning(f"Unexpected LLM response structure: {type(data)}")
            return []

        claims = []
        for i, item in enumerate(raw_claims):
            if isinstance(item, dict) and "text" in item:
                claim_id = item.get("id", f"claim_{i + 1}")
                claims.append(Claim(id=claim_id, text=item["text"].strip()))
            elif isinstance(item, str):
                claims.append(Claim(id=f"claim_{i + 1}", text=item.strip()))

        return claims

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from text that may be wrapped in markdown code blocks."""
        # Try to find JSON in markdown code blocks
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try to find a JSON object or array directly
        json_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

        return text.strip()

    def _extract_with_fallback(self, response_text: str) -> ClaimExtractionResult:
        """
        Fallback claim extraction using sentence splitting heuristic.

        Splits text into sentences and filters out non-factual ones.
        """
        sentences = self._split_sentences(response_text)
        claims = []

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if self._is_factual_candidate(sentence):
                claims.append(Claim(id=f"claim_{i + 1}", text=sentence))

        # Re-number claims sequentially
        for idx, claim in enumerate(claims):
            claim.id = f"claim_{idx + 1}"

        return ClaimExtractionResult(claims=claims, method="fallback")

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split text into sentences.

        Uses a forward-scan approach to find sentence boundaries while
        respecting common abbreviations (Dr., Mr., etc.) and numbers.
        Compatible with Python 3.10+.
        """
        # Known abbreviations that should NOT trigger a sentence split
        abbreviations = {
            "mr", "mrs", "ms", "dr", "prof", "jr", "sr", "vs",
            "etc", "e.g", "i.e", "st", "ave", "dept", "approx",
        }

        # Find all positions where a period/!/?  is followed by whitespace
        result = []
        current_start = 0

        i = 0
        while i < len(text):
            char = text[i]

            if char in '.!?':
                # Check if this is a real sentence boundary
                is_boundary = False

                # Must be followed by whitespace + uppercase, or end of text
                rest = text[i + 1:]
                if not rest.strip():
                    # End of text — this is a boundary
                    is_boundary = True
                elif rest and rest[0] in ' \t\n\r':
                    # Check what follows the whitespace
                    after_space = rest.lstrip()
                    if after_space and after_space[0].isupper():
                        is_boundary = True

                # For periods, check if preceded by an abbreviation
                if is_boundary and char == '.':
                    # Look back for the preceding word
                    before = text[current_start:i + 1].rstrip()
                    # Extract the last word before the period
                    words = before.split()
                    if words:
                        last_word = words[-1].rstrip('.').lower()
                        if last_word in abbreviations:
                            is_boundary = False

                    # Check for decimal numbers (e.g., "3.14")
                    if i > 0 and text[i - 1].isdigit():
                        after = text[i + 1:i + 2]
                        if after and after[0].isdigit():
                            is_boundary = False

                if is_boundary:
                    sentence = text[current_start:i + 1].strip()
                    if sentence:
                        result.append(sentence)
                    # Move past whitespace to next sentence
                    current_start = i + 1
                    while current_start < len(text) and text[current_start] in ' \t\n\r':
                        current_start += 1
                    i = current_start
                    continue

            i += 1

        # Capture any remaining text
        remaining = text[current_start:].strip()
        if remaining:
            if remaining[-1] not in '.!?':
                remaining += '.'
            result.append(remaining)

        return result

    @staticmethod
    def _is_factual_candidate(sentence: str) -> bool:
        """
        Determine if a sentence is likely a factual claim worth verifying.

        Filters out greetings, questions, very short fragments, and
        purely subjective/conversational statements.
        """
        if not sentence or len(sentence) < 10:
            return False

        # Count words (very short sentences are rarely factual claims)
        words = sentence.split()
        if len(words) < 3:
            return False

        lower = sentence.lower().strip()

        # Skip questions
        if sentence.strip().endswith('?'):
            return False

        # Skip common greetings and conversational phrases
        skip_prefixes = [
            "hello", "hi ", "hey ", "thanks", "thank you",
            "sure", "of course", "you're welcome", "no problem",
            "i hope", "i think", "in my opinion", "personally",
            "let me know", "feel free", "don't hesitate",
            "i'd be happy", "glad to help",
        ]
        for prefix in skip_prefixes:
            if lower.startswith(prefix):
                return False

        # Skip sentences that are purely imperative / instructional
        # (starting with a verb in base form is a heuristic)
        skip_starters = [
            "please ", "note that", "keep in mind",
            "remember ", "make sure", "try to",
        ]
        for starter in skip_starters:
            if lower.startswith(starter):
                return False

        return True
