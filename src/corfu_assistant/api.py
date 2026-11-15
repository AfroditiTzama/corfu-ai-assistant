from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from corfu_assistant.models import ConversationState
from corfu_assistant.orchestrator import (
    OrchestrationRequest,
    OrchestrationResult,
    Orchestrator,
)
from corfu_assistant.state_manager import (
    ConversationNotFoundError,
    ConversationStateManager,
)
from corfu_assistant.services import ApplicationServices


router = APIRouter()


def get_application_services(request: Request) -> ApplicationServices:
    services = getattr(request.app.state, "services", None)
    if not isinstance(services, ApplicationServices):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application services are not initialized",
        )
    return services


def get_orchestrator(
    services: ApplicationServices = Depends(get_application_services),
) -> Orchestrator:
    return services.orchestrator


def get_state_manager(
    services: ApplicationServices = Depends(get_application_services),
) -> ConversationStateManager:
    return services.state_manager


@router.post(
    "/chat",
    response_model=OrchestrationResult,
    response_model_exclude_none=False,
)
def chat(
    payload: OrchestrationRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> OrchestrationResult:
    return orchestrator.process(payload)


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationState,
)
def get_conversation(
    conversation_id: UUID,
    state_manager: ConversationStateManager = Depends(get_state_manager),
) -> ConversationState:
    try:
        return state_manager.get(conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        ) from error


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_conversation(
    conversation_id: UUID,
    state_manager: ConversationStateManager = Depends(get_state_manager),
) -> Response:
    if not state_manager.delete(conversation_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
