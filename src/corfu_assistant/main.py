from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from corfu_assistant.api import router
from corfu_assistant.config import Settings, get_settings
from corfu_assistant.data import FixtureLoadError, load_dataset
from corfu_assistant.intent_extractor import (
    IntentConfigurationError,
    IntentExtractionError,
    IntentProviderError,
    IntentRateLimitError,
    IntentTimeoutError,
    IntentTransientProviderError,
    InvalidStructuredOutputError,
)
from corfu_assistant.orchestrator import (
    AnswerGenerationError,
    InvalidExtractedConstraintError,
    InvalidOrchestrationRequestError,
    OrchestrationError,
    UnknownConversationError,
)
from corfu_assistant.repository import InvalidMetadataFilter, POIRepository
from corfu_assistant.services import ServiceOverrides, build_application_services
from corfu_assistant.state_manager import ConversationNotFoundError


_LOGGER = logging.getLogger(__name__)


def _log_provider_failure(error: Exception, *, stage: str) -> None:
    """Render safe provider diagnostics in the message used by default loggers."""
    frames = [
        f"{Path(frame.f_code.co_filename).name}:{line_number} in {frame.f_code.co_name}"
        for frame, line_number in traceback.walk_tb(error.__traceback__)
    ]
    chain: list[BaseException] = []
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited and len(chain) < 8:
        visited.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__

    provider_status_code = _safe_provider_status_code(chain)
    provider_status_label, provider_rejection = _safe_provider_rejection(chain)
    exception_chain = ">".join(type(item).__name__ for item in chain)
    rendered_frames = " > ".join(frames) if frames else "unavailable"
    rendered_status = (
        str(provider_status_code) if provider_status_code is not None else "unknown"
    )
    validation_issues_value = getattr(error, "validation_issues", ())
    validation_issues = (
        ",".join(issue for issue in validation_issues_value if isinstance(issue, str))
        if isinstance(validation_issues_value, (tuple, list))
        else ""
    ) or "unavailable"
    attempts_used = getattr(error, "attempts_used", None)
    retry_budget = getattr(error, "retry_budget", None)
    structured_attempts = (
        str(attempts_used)
        if isinstance(attempts_used, int)
        else "unavailable"
    )
    structured_retries = (
        f"{max(0, attempts_used - 1)}/{retry_budget}"
        if isinstance(attempts_used, int) and isinstance(retry_budget, int)
        else "unavailable"
    )
    _LOGGER.error(
        "Mapped Gemini failure returned by API | "
        "failure_stage=%s | exception_class=%s | provider_status_code=%s | "
        "provider_status_label=%s | provider_rejection=%s | "
        "exception_chain=%s | validation_issues=%s | structured_attempts=%s | "
        "structured_retries=%s | "
        "traceback_frames=[%s]",
        stage,
        type(error).__name__,
        rendered_status,
        provider_status_label,
        provider_rejection,
        exception_chain,
        validation_issues,
        structured_attempts,
        structured_retries,
        rendered_frames,
        extra={
            "failure_stage": stage,
            "exception_class": type(error).__name__,
            "provider_status_code": provider_status_code,
            "provider_status_label": provider_status_label,
            "provider_rejection": provider_rejection,
            "exception_chain": exception_chain,
            "validation_issues": validation_issues,
            "structured_attempts": structured_attempts,
            "structured_retries": structured_retries,
            "traceback_frames": rendered_frames,
        },
    )


def _safe_provider_status_code(chain: list[BaseException]) -> int | None:
    """Read only numeric status metadata; never stringify provider exceptions."""
    for error in chain:
        for attribute in ("status_code", "code"):
            try:
                value = getattr(error, attribute, None)
            except Exception:
                continue
            if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
                return value
        try:
            response = getattr(error, "response", None)
            value = getattr(response, "status_code", None)
        except Exception:
            continue
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def _safe_provider_rejection(chain: list[BaseException]) -> tuple[str, str]:
    """Allowlist provider metadata; never render arbitrary response text."""
    for error in chain:
        try:
            status = getattr(error, "status", None)
            message = getattr(error, "message", None)
        except Exception:
            continue
        if status == "INVALID_ARGUMENT":
            reason = (
                "Request contains an invalid argument."
                if message == "Request contains an invalid argument."
                else "unavailable"
            )
            return "INVALID_ARGUMENT", reason
    return "unavailable", "unavailable"


def create_app(
    settings: Settings | None = None,
    *,
    service_overrides: ServiceOverrides | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        try:
            dataset = load_dataset(app_settings.dataset_path)
            repository = POIRepository(
                dataset,
                embedding_model_name=app_settings.embedding_model_name,
                embedding_cache_dir=app_settings.embedding_cache_dir,
            )
        except FixtureLoadError as error:
            raise RuntimeError(
                "Application startup failed while loading the required POI dataset "
                f"at {app_settings.dataset_path}: {error}"
            ) from error

        services = build_application_services(
            app_settings,
            repository,
            service_overrides,
        )
        application.state.services = services
        application.state.poi_repository = services.poi_repository
        application.state.state_manager = services.state_manager
        application.state.orchestrator = services.orchestrator
        yield

    application = FastAPI(title=app_settings.app_name, lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    application.include_router(router)

    @application.exception_handler(UnknownConversationError)
    @application.exception_handler(ConversationNotFoundError)
    async def unknown_conversation(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "Conversation not found"})

    @application.exception_handler(InvalidOrchestrationRequestError)
    async def invalid_orchestration_request(
        request: Request,
        error: InvalidOrchestrationRequestError,
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @application.exception_handler(InvalidExtractedConstraintError)
    @application.exception_handler(InvalidMetadataFilter)
    async def invalid_metadata_constraint(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": "A requested POI category or tag is not supported."},
        )

    @application.exception_handler(IntentConfigurationError)
    async def intent_configuration_error(
        request: Request,
        error: IntentConfigurationError,
    ) -> JSONResponse:
        _log_provider_failure(error, stage="intent_extraction")
        return JSONResponse(
            status_code=503,
            content={"detail": "Gemini intent extraction is not configured on this server."},
        )

    @application.exception_handler(IntentRateLimitError)
    async def intent_rate_limited(
        request: Request,
        error: IntentRateLimitError,
    ) -> JSONResponse:
        _log_provider_failure(error, stage="intent_extraction")
        return JSONResponse(
            status_code=429,
            content={"detail": "Gemini rate limit reached. Retry according to provider limits."},
        )

    @application.exception_handler(IntentTimeoutError)
    async def intent_timeout(
        request: Request,
        error: IntentTimeoutError,
    ) -> JSONResponse:
        _log_provider_failure(error, stage="intent_extraction")
        return JSONResponse(
            status_code=504,
            content={"detail": "Gemini intent extraction timed out."},
        )

    @application.exception_handler(IntentTransientProviderError)
    async def transient_intent_failure(
        request: Request,
        error: IntentTransientProviderError,
    ) -> JSONResponse:
        _log_provider_failure(error, stage="intent_extraction")
        return JSONResponse(
            status_code=503,
            content={"detail": "Gemini is temporarily unavailable."},
        )

    @application.exception_handler(InvalidStructuredOutputError)
    async def invalid_structured_output(
        request: Request,
        error: InvalidStructuredOutputError,
    ) -> JSONResponse:
        _log_provider_failure(error, stage="intent_extraction")
        return JSONResponse(
            status_code=502,
            content={"detail": "Gemini returned an invalid structured response."},
        )

    @application.exception_handler(IntentProviderError)
    async def intent_provider_failure(
        request: Request,
        error: IntentProviderError,
    ) -> JSONResponse:
        _log_provider_failure(error, stage="intent_extraction")
        return JSONResponse(
            status_code=502,
            content={"detail": "Gemini intent extraction failed."},
        )

    @application.exception_handler(IntentExtractionError)
    async def intent_extraction_failure(
        request: Request,
        error: IntentExtractionError,
    ) -> JSONResponse:
        _log_provider_failure(error, stage="intent_extraction")
        return JSONResponse(
            status_code=502,
            content={"detail": "Gemini intent extraction failed."},
        )

    @application.exception_handler(AnswerGenerationError)
    async def answer_generation_failure(
        request: Request,
        error: AnswerGenerationError,
    ) -> JSONResponse:
        _log_provider_failure(error, stage="answer_generation")
        return JSONResponse(
            status_code=502,
            content={"detail": "Grounded answer generation failed."},
        )

    @application.exception_handler(OrchestrationError)
    async def orchestration_failure(
        request: Request,
        error: OrchestrationError,
    ) -> JSONResponse:
        _LOGGER.warning("Orchestration request failed: %s", type(error).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "The assistant could not complete this request."},
        )

    @application.exception_handler(Exception)
    async def unexpected_failure(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        _LOGGER.error(
            "Unhandled API error",
            extra={
                "error_type": type(error).__name__,
                "path": request.url.path,
            },
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected server error occurred."},
        )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/ready")
    def readiness(request: Request) -> dict[str, str | int]:
        repository = getattr(request.app.state, "poi_repository", None)
        if not isinstance(repository, POIRepository):
            raise HTTPException(status_code=503, detail="POI repository is not initialized")
        return {"status": "ready", "poi_count": repository.count}

    return application


app = create_app()
