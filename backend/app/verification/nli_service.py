"""DeBERTa-based Natural Language Inference for claim verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class NLIResult:
    """Result of comparing a claim against an evidence passage."""

    label: str
    confidence: float
    scores: dict[str, float]


class NLIService:
    """Classifies the relationship between evidence and a claim."""

    MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

    LABEL_MAP = {
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    }

    def __init__(
        self,
        model_name: str | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name or self.MODEL_NAME

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = model or AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        self.model.eval()

    def verify(self, claim: str, evidence: str) -> NLIResult:
        """Determine whether evidence entails, contradicts, or is neutral to a claim."""

        if not claim.strip():
            raise ValueError("Claim cannot be empty.")

        if not evidence.strip():
            raise ValueError("Evidence cannot be empty.")

        inputs = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        scores = {
            self.LABEL_MAP[index]: float(probabilities[index])
            for index in range(len(probabilities))
        }

        predicted_index = int(torch.argmax(probabilities).item())
        label = self.LABEL_MAP[predicted_index]
        confidence = float(probabilities[predicted_index])

        return NLIResult(
            label=label,
            confidence=confidence,
            scores=scores,
        )