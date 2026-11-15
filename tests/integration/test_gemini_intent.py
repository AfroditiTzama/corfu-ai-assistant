import os

import pytest

from corfu_assistant.config import get_settings
from corfu_assistant.intent_extractor import GeminiIntentExtractor


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("CORFU_RUN_GEMINI_INTEGRATION") != "1",
    reason="Set CORFU_RUN_GEMINI_INTEGRATION=1 to make a real Gemini request",
)
def test_real_gemini_returns_structured_intent() -> None:
    settings = get_settings()
    if settings.google_api_key is None:
        pytest.skip("Set GOOGLE_API_KEY in the environment or local .env")

    extractor = GeminiIntentExtractor(
        api_key=settings.google_api_key,
        model_name=settings.gemini_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    intent = extractor.extract(
        "Plan a walking visit focused on history tomorrow from 10:00 to 12:00."
    )

    assert intent.intent == "itinerary_creation"
    assert intent.time_window is not None
    assert intent.travel_mode == "walking"
