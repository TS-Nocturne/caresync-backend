import logging
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from patient_care_backend.brain.graph import get_brain_graph
from patient_care_backend.config import get_settings
from patient_care_backend.schemas import (
    AssessmentRequest,
    AssessmentResult,
    AvailabilityRequest,
    DecisionRequest,
)
from patient_care_backend.services.context_indexer import build_context_indexer
from patient_care_backend.services.knowledge_base import build_knowledge_base_indexer

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title="Patient Care Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=not any(origin == "*" for origin in settings.allowed_origins),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Internal-API-Key"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if int(request.headers.get("content-length") or 0) > settings.max_request_bytes:
        return JSONResponse(status_code=413, content={"error": "Request body too large"})

    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled backend exception for %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error"},
    )


def require_internal_api_key(request: Request) -> None:
    expected = settings.internal_api_key
    if expected:
        if request.headers.get("x-internal-api-key") != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return
    # No key configured — only allow requests from localhost (dev mode).
    client_host = getattr(request.client, "host", None) if request.client else None
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _current_state(thread_id: str) -> dict:
    snapshot = get_brain_graph().get_state(_config(thread_id))
    values = snapshot.values or {}
    return dict(values)


def _status(state: dict) -> str:
    issues = state.get("validation_issues") or []
    if issues and not state.get("validation_confirmed"):
        severities = {i.get("severity") for i in issues if isinstance(i, dict)}
        if "error" in severities or "warning" in severities:
            return "needs_confirmation"

    risk_level = state.get("risk_level")
    family_decision = state.get("family_decision")
    executed_actions = state.get("executed_actions")
    if risk_level in {"warning", "critical"} and not family_decision and not executed_actions:
        return "waiting_for_human"
    return "completed"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/brain/assess", response_model=AssessmentResult, dependencies=[Depends(require_internal_api_key)])
def assess_patient(request: AssessmentRequest) -> AssessmentResult:
    thread_id = request.thread_id or str(uuid4())
    initial_state = {
        "patient_id": request.patient_id,
        "vitals": request.vitals,
        "symptoms": request.symptoms,
        "current_medications": request.current_medications,
        "family_decision": None,
        "family_availability": [],
        "validation_confirmed": request.validation_confirmed,
    }
    get_brain_graph().invoke(initial_state, _config(thread_id))
    state = _current_state(thread_id)
    return AssessmentResult(thread_id=thread_id, status=_status(state), state=state)


@app.post(
    "/brain/{thread_id}/decision",
    response_model=AssessmentResult,
    dependencies=[Depends(require_internal_api_key)],
)
def submit_family_decision(thread_id: str, request: DecisionRequest) -> AssessmentResult:
    state = _current_state(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail="Unknown thread_id")

    get_brain_graph().update_state(
        _config(thread_id),
        {"family_decision": request.family_decision},
        as_node="evaluate",
    )
    get_brain_graph().invoke(None, _config(thread_id))
    state = _current_state(thread_id)
    return AssessmentResult(thread_id=thread_id, status=_status(state), state=state)


@app.post(
    "/brain/{thread_id}/availability",
    response_model=AssessmentResult,
    dependencies=[Depends(require_internal_api_key)],
)
def update_family_availability(thread_id: str, request: AvailabilityRequest) -> AssessmentResult:
    state = _current_state(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail="Unknown thread_id")

    existing = list(state.get("family_availability") or [])
    existing.append(request.model_dump())
    get_brain_graph().update_state(
        _config(thread_id),
        {"family_availability": existing},
        as_node="evaluate",
    )
    state = _current_state(thread_id)
    return AssessmentResult(thread_id=thread_id, status=_status(state), state=state)


@app.get(
    "/brain/{thread_id}/state",
    response_model=AssessmentResult,
    dependencies=[Depends(require_internal_api_key)],
)
def get_brain_state(thread_id: str) -> AssessmentResult:
    state = _current_state(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    return AssessmentResult(thread_id=thread_id, status=_status(state), state=state)


@app.post("/brain/training-export", dependencies=[Depends(require_internal_api_key)])
def export_anonymized_training(records: list[dict]) -> dict:
    """Export anonymized records safe for future AI training."""
    from patient_care_backend.services.anonymizer import anonymize_training_record

    return {"records": [anonymize_training_record(r) for r in records]}


@app.post("/brain/index-patient", dependencies=[Depends(require_internal_api_key)])
def index_patient_context(profile: dict) -> dict:
    """Upsert patient medical context chunks into Pinecone for RAG."""
    indexer = build_context_indexer(get_settings())
    return indexer.index_patient(profile)


@app.post("/brain/index-knowledge-base", dependencies=[Depends(require_internal_api_key)])
def index_knowledge_base() -> dict:
    """Upsert curated PDF knowledge from the pland directory into Pinecone."""
    source_dir = get_settings().knowledge_base_dir
    indexer = build_knowledge_base_indexer(get_settings())
    return indexer.index_directory(source_dir)
