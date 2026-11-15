from __future__ import annotations

import logging
from dataclasses import dataclass

from corfu_assistant.answer_generator import (
    AnswerGeneratorConfigurationError,
    GeminiAnswerGenerator,
)
from corfu_assistant.config import Settings
from corfu_assistant.intent_extractor import (
    GeminiIntentExtractor,
    IntentConfigurationError,
    IntentExtractor,
)
from corfu_assistant.models import ConversationState, UserIntent
from corfu_assistant.orchestrator import AnswerGenerator, Orchestrator
from corfu_assistant.planner import DeterministicItineraryPlanner
from corfu_assistant.repository import POIRepository
from corfu_assistant.state_manager import ConversationStateManager
from corfu_assistant.weather import (
    FixtureWeatherProvider,
    OpenMeteoWeatherProvider,
    WeatherProvider,
)


_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ServiceOverrides:
    """Optional app-scoped collaborators for tests and explicit demos."""

    poi_repository: POIRepository | None = None
    state_manager: ConversationStateManager | None = None
    intent_extractor: IntentExtractor | None = None
    planner: DeterministicItineraryPlanner | None = None
    weather_provider: WeatherProvider | None = None
    answer_generator: AnswerGenerator | None = None


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    poi_repository: POIRepository
    state_manager: ConversationStateManager
    intent_extractor: IntentExtractor
    planner: DeterministicItineraryPlanner
    weather_provider: WeatherProvider
    answer_generator: AnswerGenerator | None
    orchestrator: Orchestrator


class _UnavailableIntentExtractor:
    """Keeps liveness available while making missing LLM configuration explicit."""

    def extract(
        self,
        _message: str,
        _conversation_state: ConversationState | None = None,
    ) -> UserIntent:
        raise IntentConfigurationError(
            "Gemini intent extraction is unavailable; configure the server-side GOOGLE_API_KEY "
            "and install the llm dependency extra"
        )


def build_application_services(
    settings: Settings,
    loaded_repository: POIRepository,
    overrides: ServiceOverrides | None = None,
) -> ApplicationServices:
    injected = overrides or ServiceOverrides()
    if (
        settings.environment != "development"
        and (
            settings.weather_fixture_enabled
            or isinstance(injected.weather_provider, FixtureWeatherProvider)
        )
    ):
        raise ValueError(
            "Synthetic weather fixtures are allowed only when "
            "CORFU_ENVIRONMENT=development"
        )
    repository = injected.poi_repository or loaded_repository
    state_manager = injected.state_manager or ConversationStateManager()
    intent_extractor = injected.intent_extractor
    if intent_extractor is None:
        try:
            intent_extractor = GeminiIntentExtractor(
                api_key=settings.google_api_key,
                model_name=settings.gemini_model,
                timeout_seconds=settings.llm_timeout_seconds,
                metadata_categories=repository.metadata_categories,
                metadata_tags=repository.metadata_tags,
                metadata_poi_names={
                    poi.id: poi.name for poi in repository.get_all()
                },
            )
        except IntentConfigurationError:
            intent_extractor = _UnavailableIntentExtractor()

    planner = injected.planner or DeterministicItineraryPlanner(repository)
    if injected.weather_provider is not None:
        weather_provider = injected.weather_provider
    elif settings.weather_fixture_enabled:
        weather_provider = FixtureWeatherProvider(settings.weather_fixture_path)
    else:
        weather_provider = OpenMeteoWeatherProvider(
            latitude=settings.weather_latitude,
            longitude=settings.weather_longitude,
            timezone_name=settings.weather_timezone,
            timeout_seconds=settings.weather_timeout_seconds,
            retry_budget=settings.weather_retry_budget,
        )
    answer_generator = injected.answer_generator
    if answer_generator is None and settings.google_api_key is not None:
        try:
            answer_generator = GeminiAnswerGenerator(
                api_key=settings.google_api_key,
                model_name=settings.gemini_model,
                timeout_seconds=settings.llm_timeout_seconds,
            )
        except AnswerGeneratorConfigurationError as error:
            _LOGGER.warning(
                "Gemini answer generation is unavailable; using deterministic responses",
                extra={"error_type": type(error).__name__},
            )

    orchestrator = Orchestrator(
        intent_extractor=intent_extractor,
        state_manager=state_manager,
        poi_repository=repository,
        planner=planner,
        weather_provider=weather_provider,
        answer_generator=answer_generator,
    )
    return ApplicationServices(
        poi_repository=repository,
        state_manager=state_manager,
        intent_extractor=intent_extractor,
        planner=planner,
        weather_provider=weather_provider,
        answer_generator=answer_generator,
        orchestrator=orchestrator,
    )
