from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import Settings
from app.correction.correction_service import (
    CorrectionService,
    LLM_FAILURE_MESSAGE,
    UNCERTAIN_MESSAGE,
)
from app.models.schemas import ClaimStatus, Evidence


def make_evidence(
    text: str = "Google was founded in 1998.",
) -> Evidence:
    return Evidence(
        text=text,
        source="Example Source",
        url="https://example.org/google",
        similarity_score=0.95,
        source_type="web",
    )


@pytest.mark.asyncio
async def test_supported_claim_returns_original_claim():
    service = CorrectionService(
        settings=Settings(
            _env_file=None,
            llm_api_key="",
        )
    )

    result = await service.correct(
        claim_text="The Earth revolves around the Sun.",
        status=ClaimStatus.SUPPORTED,
        evidence=[],
    )

    assert result == "The Earth revolves around the Sun."


@pytest.mark.asyncio
async def test_uncertain_claim_returns_insufficient_evidence_message():
    service = CorrectionService(
        settings=Settings(
            _env_file=None,
            llm_api_key="",
        )
    )

    result = await service.correct(
        claim_text="Some uncertain claim.",
        status=ClaimStatus.UNCERTAIN,
        evidence=[],
    )

    assert result == UNCERTAIN_MESSAGE


@pytest.mark.asyncio
async def test_contradicted_claim_without_evidence_returns_failure_message():
    service = CorrectionService(
        settings=Settings(
            _env_file=None,
            llm_api_key="test-key",
        )
    )

    result = await service.correct(
        claim_text="Google was founded in 1999.",
        status=ClaimStatus.CONTRADICTED,
        evidence=[],
    )

    assert result == LLM_FAILURE_MESSAGE


@pytest.mark.asyncio
async def test_contradicted_claim_without_llm_returns_failure_message():
    service = CorrectionService(
        settings=Settings(
            _env_file=None,
            llm_api_key="",
        )
    )

    result = await service.correct(
        claim_text="Google was founded in 1999.",
        status=ClaimStatus.CONTRADICTED,
        evidence=[make_evidence()],
    )

    assert result == LLM_FAILURE_MESSAGE


@pytest.mark.asyncio
async def test_contradicted_claim_generates_correction():
    settings = Settings(
        _env_file=None,
        llm_api_key="test-key",
        llm_base_url="https://api.example.com/v1",
        llm_model="test-model",
    )

    service = CorrectionService(settings=settings)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="Google was founded in 1998."
                    )
                )
            ]
        )
    )

    service._client = mock_client

    result = await service.correct(
        claim_text="Google was founded in 1999.",
        status=ClaimStatus.CONTRADICTED,
        evidence=[
            make_evidence(
                "Google was founded in 1998."
            )
        ],
    )

    assert result == "Google was founded in 1998."

    mock_client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_correction_prompt_contains_claim_and_evidence():
    settings = Settings(
        _env_file=None,
        llm_api_key="test-key",
        llm_base_url="https://api.example.com/v1",
        llm_model="test-model",
    )

    service = CorrectionService(settings=settings)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="Google was founded in 1998."
                    )
                )
            ]
        )
    )

    service._client = mock_client

    evidence = make_evidence(
        "Google was founded in 1998 by Larry Page and Sergey Brin."
    )

    await service.correct(
        claim_text="Google was founded in 1999.",
        status=ClaimStatus.CONTRADICTED,
        evidence=[evidence],
    )

    call = mock_client.chat.completions.create.await_args
    messages = call.kwargs["messages"]

    user_message = messages[1]["content"]

    assert "Google was founded in 1999." in user_message
    assert "Google was founded in 1998 by Larry Page and Sergey Brin." in user_message
    assert "CONTRADICTED" in user_message


@pytest.mark.asyncio
async def test_llm_failure_returns_safe_message():
    settings = Settings(
        _env_file=None,
        llm_api_key="test-key",
        llm_base_url="https://api.example.com/v1",
        llm_model="test-model",
    )

    service = CorrectionService(settings=settings)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception("API timeout")
    )

    service._client = mock_client

    result = await service.correct(
        claim_text="Google was founded in 1999.",
        status=ClaimStatus.CONTRADICTED,
        evidence=[make_evidence()],
    )

    assert result == LLM_FAILURE_MESSAGE