"""FastAPI application for the TruthLens Stage 1 + 2 seminar prototype."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

# Existing modules use ``app.*`` imports.  Add backend/ when this module is
# launched with the documented ``uvicorn backend.app.main:app --reload`` command.
BACKEND_DIRECTORY = str(Path(__file__).resolve().parents[1])
if BACKEND_DIRECTORY not in sys.path:
    sys.path.insert(0, BACKEND_DIRECTORY)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, settings as default_settings
from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.pipeline.analysis_service import AnalysisService


class Analyzer(Protocol):
    """Minimal interface that makes the HTTP layer independently testable."""

    async def analyze(self, response_text: str) -> AnalyzeResponse: ...


def create_app(
    app_settings: Settings | None = None,
    analysis_service: Analyzer | None = None,
) -> FastAPI:
    """Build the application with injectable services for isolated endpoint tests."""
    configured_settings = app_settings or default_settings
    service = analysis_service or AnalysisService(settings=configured_settings)

    application = FastAPI(
        title="TruthLens",
        description="Claim extraction and evidence retrieval for LLM-generated text.",
        version="0.2.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=configured_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/health")
    async def health() -> dict[str, str]:
        """Report that the lightweight API process is reachable."""
        return {"status": "ok"}

    @application.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
        """Run the existing Stage 1 → Stage 2 pipeline."""
        return await service.analyze(request.response)

    return application


app = create_app()
