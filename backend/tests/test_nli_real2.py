from app.verification.nli_service import NLIService


def test_real_nli_focused_evidence():
    service = NLIService()

    cases = [
        (
            "Earth revolves about the Sun.",
            "The Earth revolves around the Sun.",
        ),
        (
            "Earth revolves in orbit around the Sun.",
            "The Earth revolves around the Sun.",
        ),
        (
            "Earth completes one full trip around our home star.",
            "The Earth revolves around the Sun.",
        ),
        (
            "Earth doesn't just rotate, though—it also revolves around the Sun.",
            "The Earth revolves around the Sun.",
        ),
    ]

    for evidence, claim in cases:
        result = service.verify(
            claim=claim,
            evidence=evidence,
        )

        print("\n" + "=" * 70)
        print("Evidence:", evidence)
        print("Claim:", claim)
        print("Label:", result.label)
        print("Confidence:", result.confidence)
        print("Scores:", result.scores)