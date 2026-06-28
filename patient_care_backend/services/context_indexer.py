"""Build and upsert patient care-context chunks into Pinecone for RAG."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from patient_care_backend.config import Settings

MOBILITY_LABELS = {
    "INDEPENDENT": "walks independently",
    "ASSISTED": "walks with assistance such as cane or walker",
    "WHEELCHAIR": "uses wheelchair",
    "BEDBOUND": "bedbound",
}

INSURANCE_LABELS = {
    "REIMBURSEMENT": "reimbursement",
    "SOCIAL_SECURITY": "social security",
    "SELF_PAY": "self pay",
}


def build_patient_context_chunks(profile: dict[str, Any]) -> list[dict[str, str]]:
    """Turn a patient profile dict into embeddable text chunks."""
    chunks: list[dict[str, str]] = []
    patient_id = str(profile.get("patient_id", ""))

    name = profile.get("nickname") or profile.get("first_name", "")
    if name:
        chunks.append({"category": "identity", "text": f"Patient nickname/preferred name: {name}"})

    diseases = profile.get("underlying_diseases") or []
    if diseases:
        chunks.append(
            {
                "category": "conditions",
                "text": f"Underlying diseases / comorbidities: {', '.join(diseases)}",
            }
        )

    allergies = profile.get("allergies") or []
    if allergies:
        chunks.append(
            {
                "category": "allergies",
                "text": f"Drug and food allergies - CRITICAL: {', '.join(allergies)}",
            }
        )

    mobility = profile.get("mobility_status")
    if mobility:
        label = MOBILITY_LABELS.get(mobility, mobility)
        chunks.append({"category": "mobility", "text": f"Physical limitations / mobility: {label}"})

    baseline_parts = []
    if profile.get("baseline_systolic") and profile.get("baseline_diastolic"):
        baseline_parts.append(
            f"BP {profile['baseline_systolic']}/{profile['baseline_diastolic']} mmHg"
        )
    for key, label in [
        ("baseline_temperature", "temp"),
        ("baseline_heart_rate", "HR"),
        ("baseline_oxygen_sat", "SpO2"),
    ]:
        if profile.get(key) is not None:
            baseline_parts.append(f"{label} {profile[key]}")
    if baseline_parts:
        chunks.append(
            {
                "category": "baseline_vitals",
                "text": (
                    "Personal baseline vitals (not general population norms): "
                    f"{', '.join(baseline_parts)}"
                ),
            }
        )

    dynamic_parts = []
    for lower_key, upper_key, label in [
        ("baseline_systolic_lower", "baseline_systolic_upper", "systolic"),
        ("baseline_diastolic_lower", "baseline_diastolic_upper", "diastolic"),
        ("baseline_temperature_lower", "baseline_temperature_upper", "temperature"),
        ("baseline_heart_rate_lower", "baseline_heart_rate_upper", "heart rate"),
        ("baseline_oxygen_sat_min", "baseline_oxygen_sat_max", "SpO2"),
    ]:
        if profile.get(lower_key) is not None and profile.get(upper_key) is not None:
            dynamic_parts.append(f"{label} {profile[lower_key]}-{profile[upper_key]}")
    if dynamic_parts or profile.get("baseline_insight_text"):
        chunks.append(
            {
                "category": "dynamic_baseline",
                "text": (
                    "Dynamic patient-specific thresholds from recent vitals: "
                    f"{', '.join(dynamic_parts)}. "
                    f"Latest insight: {profile.get('baseline_insight_text') or '-'}"
                ),
            }
        )

    if profile.get("weight_kg") and profile.get("height_cm"):
        h_m = profile["height_cm"] / 100
        bmi = round(profile["weight_kg"] / (h_m * h_m), 1)
        chunks.append(
            {
                "category": "anthropometry",
                "text": f"Weight {profile['weight_kg']} kg, height {profile['height_cm']} cm, BMI {bmi}",
            }
        )

    hospital = profile.get("preferred_hospital")
    hn = profile.get("hospital_number")
    if hospital:
        chunks.append(
            {
                "category": "hospital",
                "text": f"Preferred hospital: {hospital}" + (f", HN {hn}" if hn else ""),
            }
        )

    insurance = profile.get("insurance_type")
    if insurance:
        chunks.append(
            {
                "category": "insurance",
                "text": f"Insurance / payment: {INSURANCE_LABELS.get(insurance, insurance)}",
            }
        )

    for contact in profile.get("emergency_contacts") or []:
        relation = contact.get("relation") or "-"
        chunks.append(
            {
                "category": "emergency_contact",
                "text": (
                    f"Emergency contact: {contact.get('name')} ({relation}) "
                    f"tel {contact.get('phone')}"
                ),
            }
        )

    for med in profile.get("medications") or []:
        times = ", ".join(med.get("time_of_day") or []) or med.get("schedule_time", "")
        instruction = med.get("instruction") or ""
        dose = " ".join(
            str(part)
            for part in [med.get("dose_amount"), med.get("dose_unit")]
            if part not in (None, "")
        ) or med.get("dosage", "")
        prn = " PRN/as-needed." if med.get("is_prn") else ""
        frequency = med.get("frequency") or "DAILY"
        indication = f" Indication: {med.get('indication')}." if med.get("indication") else ""
        appearance = f" Appearance: {med.get('appearance')}." if med.get("appearance") else ""
        chunks.append(
            {
                "category": "medication",
                "text": (
                    f"Medication: {med.get('name')} {med.get('strength') or ''} {dose} "
                    f"at {times}. Frequency: {frequency}.{prn}{indication}{appearance} "
                    f"{instruction}".strip()
                ),
            }
        )

    for chunk in chunks:
        chunk["patient_id"] = patient_id

    return chunks


class PatientContextIndexer:
    def __init__(self, settings) -> None:
        self._settings = settings
        self._index = None
        self._embeddings = None

    def _connect(self) -> None:
        if self._index is not None and self._embeddings is not None:
            return

        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from pinecone import Pinecone

        pc = Pinecone(api_key=self._settings.pinecone_api_key)
        self._index = pc.Index(self._settings.pinecone_index_name)
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self._settings.gemini_embedding_model,
            google_api_key=self._settings.google_api_key,
        )

    def index_patient(self, profile: dict[str, Any]) -> dict[str, Any]:
        chunks = build_patient_context_chunks(profile)
        if not (
            self._settings.enable_external_ai
            and self._settings.pinecone_api_key
            and self._settings.google_api_key
        ):
            return {
                "indexed": False,
                "reason": "external_ai_not_configured",
                "chunk_count": len(chunks),
                "preview": [c["text"] for c in chunks[:3]],
            }

        self._connect()
        assert self._index is not None
        assert self._embeddings is not None

        patient_id = str(profile.get("patient_id", ""))
        vectors = []

        for chunk in chunks:
            text = chunk["text"]
            vector = self._embeddings.embed_query(text)
            vectors.append(
                {
                    "id": f"{patient_id}-{chunk['category']}-{uuid4().hex[:8]}",
                    "values": vector,
                    "metadata": {
                        "scope": "patient_context",
                        "patient_id": patient_id,
                        "category": chunk["category"],
                        "text": text,
                    },
                }
            )

        if vectors:
            self._index.upsert(vectors=vectors)

        return {"indexed": True, "chunk_count": len(vectors)}


def build_context_indexer(settings: Settings) -> PatientContextIndexer:
    return PatientContextIndexer(settings)
