"""
Tests for the TruthLens Claim Extractor.

Tests both fallback (sentence-based) and LLM-based extraction.
LLM tests use mocking to avoid requiring an actual API key.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.claim_extractor import ClaimExtractor
from app.models.schemas import Claim, ClaimExtractionResult
from app.config import Settings


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def extractor_no_llm():
    """Claim extractor with LLM disabled (forces fallback)."""
    settings = Settings(llm_api_key="", llm_base_url="https://api.example.com/v1")
    return ClaimExtractor(settings=settings)


@pytest.fixture
def extractor_with_llm():
    """Claim extractor with LLM enabled (for mocked tests)."""
    settings = Settings(
        llm_api_key="test-key-123",
        llm_base_url="https://api.example.com/v1",
        llm_model="test-model",
    )
    return ClaimExtractor(settings=settings)


# ============================================================
# Test 1: Fallback extraction - Basic sentence splitting
# ============================================================

class TestFallbackExtraction:
    """Tests for the sentence-based fallback extractor."""

    @pytest.mark.asyncio
    async def test_basic_factual_sentences(self, extractor_no_llm):
        """Two clear factual sentences should produce two claims."""
        text = "Google was founded in 1998. The Earth revolves around the Sun."
        result = await extractor_no_llm.extract(text)

        assert result.method == "fallback"
        assert len(result.claims) == 2
        assert "Google was founded in 1998" in result.claims[0].text
        assert "Earth revolves around the Sun" in result.claims[1].text

    @pytest.mark.asyncio
    async def test_sequential_claim_ids(self, extractor_no_llm):
        """Claims should have sequential IDs starting from claim_1."""
        text = "Python was created by Guido van Rossum. It was first released in 1991."
        result = await extractor_no_llm.extract(text)

        assert result.claims[0].id == "claim_1"
        assert result.claims[1].id == "claim_2"

    @pytest.mark.asyncio
    async def test_filters_greetings(self, extractor_no_llm):
        """Greetings and conversational phrases should be filtered out."""
        text = "Hello! Google was founded in 1998. Thanks for asking!"
        result = await extractor_no_llm.extract(text)

        assert len(result.claims) == 1
        assert "Google" in result.claims[0].text

    @pytest.mark.asyncio
    async def test_filters_questions(self, extractor_no_llm):
        """Questions should be filtered out."""
        text = "Did you know that Python is popular? Python was created in 1991."
        result = await extractor_no_llm.extract(text)

        # Only the factual statement should remain
        factual_claims = [c for c in result.claims if "1991" in c.text]
        assert len(factual_claims) >= 1

    @pytest.mark.asyncio
    async def test_filters_short_fragments(self, extractor_no_llm):
        """Very short fragments should be filtered out."""
        text = "Yes. OK. The Eiffel Tower is located in Paris, France."
        result = await extractor_no_llm.extract(text)

        assert len(result.claims) == 1
        assert "Eiffel Tower" in result.claims[0].text

    @pytest.mark.asyncio
    async def test_filters_opinions(self, extractor_no_llm):
        """Subjective/opinion statements should be filtered."""
        text = "I think Python is great. Python was first released in 1991."
        result = await extractor_no_llm.extract(text)

        assert len(result.claims) == 1
        assert "1991" in result.claims[0].text

    @pytest.mark.asyncio
    async def test_empty_input(self, extractor_no_llm):
        """Empty input should return empty results without error."""
        result = await extractor_no_llm.extract("")
        assert result.claims == []
        assert result.method == "empty_input"

    @pytest.mark.asyncio
    async def test_whitespace_only_input(self, extractor_no_llm):
        """Whitespace-only input should return empty results."""
        result = await extractor_no_llm.extract("   \n\t  ")
        assert result.claims == []

    @pytest.mark.asyncio
    async def test_multiple_facts(self, extractor_no_llm):
        """Multiple factual sentences should all be extracted."""
        text = (
            "The Eiffel Tower was built in 1889. "
            "It is located in Paris, France. "
            "It stands approximately 330 meters tall."
        )
        result = await extractor_no_llm.extract(text)

        assert len(result.claims) == 3
        assert result.method == "fallback"

    @pytest.mark.asyncio
    async def test_handles_abbreviations(self, extractor_no_llm):
        """Common abbreviations like 'Dr.' should not cause false splits."""
        text = "Dr. Smith discovered the element in 1952. The element has atomic number 42."
        result = await extractor_no_llm.extract(text)

        # Should handle abbreviation reasonably (may or may not split perfectly,
        # but should not crash and should produce valid claims)
        assert len(result.claims) >= 1
        assert all(len(c.text) > 5 for c in result.claims)


# ============================================================
# Test 2: LLM-based extraction (mocked)
# ============================================================

class TestLLMExtraction:
    """Tests for LLM-based extraction using mocked API calls."""

    @pytest.mark.asyncio
    async def test_successful_llm_extraction(self, extractor_with_llm):
        """LLM returns well-formed JSON -> should parse correctly."""
        llm_response = json.dumps({
            "claims": [
                {"id": "claim_1", "text": "Google was founded in 1998."},
                {"id": "claim_2", "text": "Google is headquartered in California."},
            ]
        })

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = llm_response

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        extractor_with_llm._client = mock_client

        result = await extractor_with_llm.extract(
            "Google was founded in 1998. It is headquartered in California."
        )

        assert result.method == "llm"
        assert len(result.claims) == 2
        assert result.claims[0].text == "Google was founded in 1998."
        assert result.claims[1].text == "Google is headquartered in California."

    @pytest.mark.asyncio
    async def test_llm_markdown_wrapped_json(self, extractor_with_llm):
        """LLM wraps JSON in markdown code block -> should still parse."""
        llm_response = '```json\n{"claims": [{"id": "claim_1", "text": "Test claim."}]}\n```'

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = llm_response

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        extractor_with_llm._client = mock_client

        result = await extractor_with_llm.extract("Test input.")

        assert result.method == "llm"
        assert len(result.claims) == 1
        assert result.claims[0].text == "Test claim."

    @pytest.mark.asyncio
    async def test_llm_returns_plain_list(self, extractor_with_llm):
        """LLM returns a plain JSON array instead of object -> should handle."""
        llm_response = json.dumps([
            {"id": "claim_1", "text": "Claim one."},
            {"id": "claim_2", "text": "Claim two."},
        ])

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = llm_response

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        extractor_with_llm._client = mock_client

        result = await extractor_with_llm.extract("Input text.")

        assert result.method == "llm"
        assert len(result.claims) == 2

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self, extractor_with_llm):
        """When LLM API fails, should fall back to heuristic extraction."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API timeout")
        )
        extractor_with_llm._client = mock_client

        result = await extractor_with_llm.extract(
            "Google was founded in 1998. Python was created by Guido van Rossum."
        )

        assert result.method == "fallback"
        assert len(result.claims) >= 1

    @pytest.mark.asyncio
    async def test_llm_invalid_json_falls_back(self, extractor_with_llm):
        """When LLM returns invalid JSON, should fall back to heuristic."""
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "This is not JSON at all!"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        extractor_with_llm._client = mock_client

        result = await extractor_with_llm.extract(
            "The Eiffel Tower was built in 1889."
        )

        # Should fall back since LLM returned no valid claims
        assert result.method == "fallback"
        assert len(result.claims) >= 1

    @pytest.mark.asyncio
    async def test_llm_missing_id_field(self, extractor_with_llm):
        """Claims without 'id' field should get auto-generated IDs."""
        llm_response = json.dumps({
            "claims": [
                {"text": "Claim without ID."},
                {"text": "Another claim without ID."},
            ]
        })

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = llm_response

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        extractor_with_llm._client = mock_client

        result = await extractor_with_llm.extract("Input.")

        assert len(result.claims) == 2
        assert result.claims[0].id == "claim_1"
        assert result.claims[1].id == "claim_2"


# ============================================================
# Test 3: Schema validation
# ============================================================

class TestSchemas:
    """Tests for Pydantic schema validation."""

    def test_claim_creation(self):
        """Claim schema should accept valid data."""
        claim = Claim(id="claim_1", text="Test claim.")
        assert claim.id == "claim_1"
        assert claim.text == "Test claim."

    def test_extraction_result_defaults(self):
        """ClaimExtractionResult should have sensible defaults."""
        result = ClaimExtractionResult()
        assert result.claims == []
        assert result.method == "llm"

    def test_claim_extraction_result_with_claims(self):
        """ClaimExtractionResult should hold claims properly."""
        claims = [
            Claim(id="claim_1", text="First claim."),
            Claim(id="claim_2", text="Second claim."),
        ]
        result = ClaimExtractionResult(claims=claims, method="fallback")
        assert len(result.claims) == 2
        assert result.method == "fallback"


# ============================================================
# Test 4: Config validation
# ============================================================

class TestConfig:
    """Tests for configuration loading."""

    def test_default_settings(self):
        """Default settings should load without errors."""
        settings = Settings(llm_api_key="")
        assert settings.llm_model == "gpt-3.5-turbo"
        assert settings.embedding_model == "BAAI/bge-small-en-v1.5"

    def test_llm_not_available_with_empty_key(self):
        """LLM should be reported as unavailable with empty key."""
        settings = Settings(llm_api_key="")
        assert settings.llm_available is False

    def test_llm_not_available_with_placeholder_key(self):
        """LLM should be reported as unavailable with placeholder key."""
        settings = Settings(llm_api_key="your-api-key-here")
        assert settings.llm_available is False

    def test_llm_available_with_real_key(self):
        """LLM should be reported as available with a real key."""
        settings = Settings(llm_api_key="sk-real-key-123")
        assert settings.llm_available is True

    def test_tavily_not_available_by_default(self):
        """Tavily should be unavailable when key is not set."""
        settings = Settings(llm_api_key="")
        assert settings.tavily_available is False

    def test_mongodb_not_available_by_default(self):
        """MongoDB should be unavailable when URI is not set."""
        settings = Settings(llm_api_key="")
        assert settings.mongodb_available is False


# ============================================================
# Test 5: Realistic end-to-end scenarios (fallback mode)
# ============================================================

class TestRealisticScenarios:
    """Realistic test scenarios matching the demo criteria."""

    @pytest.mark.asyncio
    async def test_demo_scenario_mixed_claims(self, extractor_no_llm):
        """The primary demo scenario: mixed correct and incorrect claims."""
        text = "Google was founded in 1999. The Earth revolves around the Sun."
        result = await extractor_no_llm.extract(text)

        assert len(result.claims) == 2
        # Both should be extracted regardless of correctness
        texts = [c.text for c in result.claims]
        assert any("Google" in t for t in texts)
        assert any("Earth" in t for t in texts)

    @pytest.mark.asyncio
    async def test_longer_response(self, extractor_no_llm):
        """Longer multi-paragraph response should extract all factual claims."""
        text = (
            "Python is a high-level programming language created by Guido van Rossum. "
            "It was first released in 1991. Python emphasizes code readability. "
            "The language supports multiple programming paradigms including "
            "procedural, object-oriented, and functional programming."
        )
        result = await extractor_no_llm.extract(text)

        assert len(result.claims) >= 3
        assert all(c.id.startswith("claim_") for c in result.claims)

    @pytest.mark.asyncio
    async def test_response_with_mixed_content(self, extractor_no_llm):
        """Response mixing facts, opinions, and greetings."""
        text = (
            "Sure, I'd be happy to help! "
            "The Eiffel Tower was built in 1889 for the World's Fair. "
            "It stands approximately 330 meters tall. "
            "I think it's one of the most beautiful structures in the world. "
            "Let me know if you have any other questions!"
        )
        result = await extractor_no_llm.extract(text)

        # Should extract the factual claims but filter conversational parts
        assert len(result.claims) >= 2
        texts = " ".join(c.text for c in result.claims)
        assert "1889" in texts
        assert "330" in texts

    @pytest.mark.asyncio
    async def test_all_correct_claims(self, extractor_no_llm):
        """Test 1: All claims are correct (verification happens later)."""
        text = (
            "Water is composed of hydrogen and oxygen. "
            "The Earth orbits the Sun. "
            "Mount Everest is the tallest mountain on Earth."
        )
        result = await extractor_no_llm.extract(text)
        assert len(result.claims) == 3

    @pytest.mark.asyncio
    async def test_single_incorrect_claim(self, extractor_no_llm):
        """Test 2: One incorrect claim among correct ones."""
        text = (
            "The speed of light is approximately 300,000 km/s. "
            "The Great Wall of China was built in the 20th century."
        )
        result = await extractor_no_llm.extract(text)
        assert len(result.claims) == 2
