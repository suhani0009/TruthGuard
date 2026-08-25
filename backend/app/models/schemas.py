"""
TruthLens Pydantic Schemas.

Defines all request/response models used throughout the pipeline.
These schemas enforce data contracts between pipeline stages.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime


# ============================================================
# Enums
# ============================================================

class ClaimStatus(str, Enum):
    """Verification status of a claim."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNCERTAIN = "UNCERTAIN"


# ============================================================
# Claim Extraction
# ============================================================

class Claim(BaseModel):
    """A single atomic factual claim extracted from a response."""
    id: str = Field(description="Unique claim identifier, e.g. 'claim_1'.")
    text: str = Field(description="The text of the extracted claim.")


class ClaimExtractionResult(BaseModel):
    """Result of extracting claims from a response."""
    claims: list[Claim] = Field(default_factory=list)
    method: str = Field(
        default="llm",
        description="Extraction method used: 'llm' or 'fallback'.",
    )


# ============================================================
# Evidence
# ============================================================

class Evidence(BaseModel):
    """A single piece of evidence retrieved for a claim."""
    text: str = Field(description="The evidence passage text.")
    source: str = Field(default="Unknown", description="Name of the source.")
    url: Optional[str] = Field(default=None, description="URL of the source.")
    similarity_score: float = Field(
        default=0.0,
        description="Semantic similarity score between claim and evidence.",
    )
    source_type: str = Field(
        default="local",
        description="Where this evidence came from: 'local' or 'web'.",
    )


# ============================================================
# Verification
# ============================================================

class ClaimVerification(BaseModel):
    """Verification result for a single claim."""
    claim_id: str
    claim_text: str
    status: ClaimStatus = ClaimStatus.UNCERTAIN
    trust_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Trust score for this claim (0-100).",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="NLI model confidence in the classification.",
    )
    evidence: list[Evidence] = Field(default_factory=list)
    corrected_text: Optional[str] = Field(
        default=None,
        description="Corrected claim text if status is CONTRADICTED.",
    )
    correction_explanation: Optional[str] = Field(
        default=None,
        description="Explanation of what was changed and why.",
    )


# ============================================================
# Change Tracking
# ============================================================

class Change(BaseModel):
    """A single correction made to the response."""
    original: str = Field(description="Original claim text.")
    corrected: str = Field(description="Corrected claim text.")
    reason: str = Field(description="Why the correction was made.")
    sources: list[str] = Field(
        default_factory=list,
        description="Source names/URLs used for the correction.",
    )


# ============================================================
# API Request / Response
# ============================================================

class VerifyRequest(BaseModel):
    """Request body for POST /api/verify."""
    response: str = Field(
        min_length=1,
        description="The LLM-generated response to verify.",
    )


class VerifyResponse(BaseModel):
    """Response body for POST /api/verify."""
    original_response: str
    corrected_response: str
    overall_trust_score: float = Field(ge=0.0, le=100.0)
    claims: list[ClaimVerification] = Field(default_factory=list)
    changes: list[Change] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Response body for GET /api/health."""
    status: str = "ok"
    llm_available: bool = False
    tavily_available: bool = False
    qdrant_available: bool = False
    mongodb_available: bool = False
