from app.verification.nli_service import NLIService


def test_real_nli_examples():
    service = NLIService()

    cases = [
        (
            "Google was founded in 1998.",
            "Google was founded in 1999.",
        ),
        (
            "Earth revolves around the Sun.",
            "The Earth revolves around the Sun.",
        ),
    ]

    for evidence, claim in cases:
        result = service.verify(
            claim=claim,
            evidence=evidence,
        )

        print("\n" + "=" * 60)
        print("Evidence:", evidence)
        print("Claim:", claim)
        print("Label:", result.label)
        print("Confidence:", result.confidence)
        print("Scores:", result.scores)