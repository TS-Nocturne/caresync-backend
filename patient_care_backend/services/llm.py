from typing import Any, Protocol

from patient_care_backend.config import Settings
from patient_care_backend.schemas import PatientState, RiskAssessment, RiskLevel
from patient_care_backend.services.anonymizer import anonymize_patient_state


class SymptomEvaluator(Protocol):
    def evaluate(self, state: PatientState) -> RiskAssessment:
        """Summarize symptoms, vitals, and retrieved context for human review."""


class FallbackSymptomEvaluator:
    critical_terms = {
        "chest pain",
        "เจ็บหน้าอก",
        "severe bleeding",
        "เลือดออกมาก",
        "unconscious",
        "หมดสติ",
        "shortness of breath",
        "หายใจลำบาก",
    }
    warning_terms = {
        "fever",
        "มีไข้",
        "dizziness",
        "เวียนหัว",
        "vomiting",
        "อาเจียน",
        "confusion",
        "สับสน",
        "fall",
        "หกล้ม",
    }

    def evaluate(self, state: PatientState) -> RiskAssessment:
        symptoms = {symptom.strip().lower() for symptom in state.get("symptoms", [])}
        vitals = state.get("vitals", {})
        baseline = state.get("patient_baseline", {})

        risk = RiskLevel.NORMAL
        if symptoms & self.critical_terms or self._spo2(vitals) < self._critical_spo2_threshold(baseline):
            risk = RiskLevel.CRITICAL
        elif symptoms & self.warning_terms or self._temperature(vitals) >= 38:
            risk = RiskLevel.WARNING

        if risk == RiskLevel.CRITICAL:
            actions = [
                "notify_care_team_for_review",
                "show_emergency_contact_options",
                "record_information_flag",
            ]
            analysis = "พบข้อมูลที่ควรให้ผู้ดูแลตรวจสอบด่วน ระบบไม่ได้วินิจฉัยหรือสั่งการรักษา"
        elif risk == RiskLevel.WARNING:
            actions = ["notify_family_for_review", "request_caregiver_review", "record_information_flag"]
            analysis = "มีข้อมูลที่ควรติดตามและส่งให้ผู้ดูแลพิจารณา ไม่ใช่คำวินิจฉัยหรือคำสั่งรักษา"
        else:
            actions = ["record_observation", "continue_routine_logging"]
            analysis = "ข้อมูลล่าสุดอยู่ในช่วงที่ระบบติดตามไว้ ควรใช้ประกอบการดูแลตามปกติโดยมนุษย์"

        return RiskAssessment(
            risk_level=risk,
            ai_analysis=analysis,
            recommended_actions=actions,
        )

    @staticmethod
    def _spo2(vitals: dict[str, Any]) -> float:
        try:
            return float(vitals.get("spo2", 100))
        except (TypeError, ValueError):
            return 100

    @staticmethod
    def _critical_spo2_threshold(baseline: dict[str, Any]) -> float:
        try:
            baseline_spo2 = float(baseline.get("baseline_oxygen_sat"))
        except (TypeError, ValueError):
            return 92
        return max(50, min(92, baseline_spo2 - 4))

    @staticmethod
    def _temperature(vitals: dict[str, Any]) -> float:
        try:
            return float(vitals.get("temperature_c", vitals.get("temperature", 36.5)))
        except (TypeError, ValueError):
            return 36.5


class GeminiSymptomEvaluator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chain = None

    def _build_chain(self):
        if self._chain is not None:
            return self._chain

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an elder-care care-coordination summarization assistant. "
                    "Do not diagnose disease, determine treatment, triage, prescribe, adjust medication, "
                    "or direct transfer decisions. "
                    "Use patient-specific baseline vitals before general population thresholds. "
                    "Return concise, family-readable observations and suggest human review only.",
                ),
                (
                    "human",
                    "Summarize this patient report for caregiver review.\n"
                    "- Symptoms: {symptoms}\n"
                    "- Vitals: {vitals}\n"
                    "- Patient-specific baseline vitals: {patient_baseline}\n"
                    "- Current medications: {medications}\n"
                    "- Retrieved care context: {context}\n",
                ),
            ]
        )
        llm = ChatGoogleGenerativeAI(
            model=self._settings.gemini_model,
            temperature=0.2,
            google_api_key=self._settings.google_api_key,
        )
        self._chain = prompt | llm.with_structured_output(RiskAssessment)
        return self._chain

    def evaluate(self, state: PatientState) -> RiskAssessment:
        chain = self._build_chain()
        safe = anonymize_patient_state(dict(state))
        return chain.invoke(
            {
                "symptoms": safe.get("symptoms", []),
                "vitals": safe.get("vitals", {}),
                "patient_baseline": safe.get("patient_baseline", {}),
                "medications": safe.get("current_medications", []),
                "context": safe.get("retrieved_medical_context", ""),
            }
        )


def build_symptom_evaluator(settings: Settings) -> SymptomEvaluator:
    if settings.enable_external_ai and settings.google_api_key:
        return GeminiSymptomEvaluator(settings)
    return FallbackSymptomEvaluator()
