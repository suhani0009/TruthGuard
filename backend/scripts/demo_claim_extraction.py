"""
Quick demo of the TruthLens Claim Extractor (fallback mode).
Demonstrates extraction with the primary demo input.
"""

import asyncio
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.claim_extractor import ClaimExtractor
from app.config import Settings


async def main():
    # Use fallback mode (no LLM key)
    settings = Settings(llm_api_key="")
    extractor = ClaimExtractor(settings=settings)

    # ---- Demo 1: Primary test scenario ----
    print("=" * 60)
    print("DEMO 1: Primary test scenario")
    print("=" * 60)
    text1 = "Google was founded in 1999. The Earth revolves around the Sun."
    result1 = await extractor.extract(text1)

    print(f"\nInput:  \"{text1}\"")
    print(f"Method: {result1.method}")
    print(f"Claims found: {len(result1.claims)}")
    print("\nExtracted Claims:")
    print(json.dumps(
        {"claims": [{"id": c.id, "text": c.text} for c in result1.claims]},
        indent=2
    ))

    # ---- Demo 2: Mixed content ----
    print("\n" + "=" * 60)
    print("DEMO 2: Mixed content (facts + greetings + opinions)")
    print("=" * 60)
    text2 = (
        "Hello! Sure, I'd be happy to help. "
        "Python was created by Guido van Rossum. "
        "It was first released in 1991. "
        "I think it's a wonderful language. "
        "The Eiffel Tower was built in 1889 for the World's Fair. "
        "Let me know if you have any other questions!"
    )
    result2 = await extractor.extract(text2)

    print(f"\nInput:  \"{text2}\"")
    print(f"Method: {result2.method}")
    print(f"Claims found: {len(result2.claims)}")
    print("\nExtracted Claims:")
    print(json.dumps(
        {"claims": [{"id": c.id, "text": c.text} for c in result2.claims]},
        indent=2
    ))

    # ---- Demo 3: Multiple incorrect claims ----
    print("\n" + "=" * 60)
    print("DEMO 3: Multiple factual claims (some incorrect)")
    print("=" * 60)
    text3 = (
        "The Great Wall of China is visible from space. "
        "Water boils at 100 degrees Celsius at sea level. "
        "Albert Einstein invented the light bulb. "
        "The Amazon River is the longest river in the world."
    )
    result3 = await extractor.extract(text3)

    print(f"\nInput:  \"{text3}\"")
    print(f"Method: {result3.method}")
    print(f"Claims found: {len(result3.claims)}")
    print("\nExtracted Claims:")
    print(json.dumps(
        {"claims": [{"id": c.id, "text": c.text} for c in result3.claims]},
        indent=2
    ))

    print("\n" + "=" * 60)
    print("Stage 1 COMPLETE: Claim extraction is working!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
