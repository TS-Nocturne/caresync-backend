"""Strip patient identifiers before AI training or external LLM logging."""

import hashlib
import re
from typing import Any


def _stable_patient_token(patient_id: str) -> str:
    digest = hashlib.sha256(patient_id.encode()).hexdigest()[:12]
    return f"anon-{digest}"


def anonymize_patient_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of PatientState with PII removed or tokenized."""
    result = dict(state)
    patient_id = result.get("patient_id")
    if patient_id:
        result["patient_id"] = _stable_patient_token(str(patient_id))

    vitals = dict(result.get("vitals") or {})
    result["vitals"] = vitals

    context = result.get("retrieved_medical_context") or ""
    if context:
        result["retrieved_medical_context"] = _redact_text(context)

    analysis = result.get("ai_analysis") or ""
    if analysis:
        result["ai_analysis"] = _redact_text(analysis)

    return result


def _redact_text(text: str) -> str:
    redacted = re.sub(
        r"\b(?:นาย|นาง|น\.ส\.|Mr\.|Mrs\.|Ms\.)\s*[\w\u0E00-\u0E7F]+",
        "[NAME]",
        text,
    )
    redacted = re.sub(r"\b\d{13}\b", "[NATIONAL_ID]", redacted)
    redacted = re.sub(
        r"\b(?:\+66|0)\d{8,9}\b",
        "[PHONE]",
        redacted,
    )
    redacted = re.sub(
        r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
        "[EMAIL]",
        redacted,
    )
    return redacted


def anonymize_training_record(record: dict[str, Any]) -> dict[str, Any]:
    """Prepare a full record for ML export — vitals/symptoms only, no identity."""
    return {
        "patient_token": _stable_patient_token(str(record.get("patient_id", "unknown"))),
        "vitals": record.get("vitals") or {},
        "symptoms": record.get("symptoms") or [],
        "risk_level": record.get("risk_level"),
        "current_medications": [
            _redact_text(str(m)) for m in (record.get("current_medications") or [])
        ],
    }
