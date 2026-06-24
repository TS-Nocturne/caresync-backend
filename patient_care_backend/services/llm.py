from typing import Any, Protocol

from patient_care_backend.config import Settings
from patient_care_backend.schemas import PatientState, RiskAssessment, RiskLevel
from patient_care_backend.services.anonymizer import anonymize_patient_state


class SymptomEvaluator(Protocol):
    def evaluate(self, state: PatientState) -> RiskAssessment:
        """Evaluate symptoms, vitals, and retrieved context."""


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

        risk = RiskLevel.NORMAL
        if symptoms & self.critical_terms or self._spo2(vitals) < 92:
            risk = RiskLevel.CRITICAL
        elif symptoms & self.warning_terms or self._temperature(vitals) >= 38:
            risk = RiskLevel.WARNING

        if risk == RiskLevel.CRITICAL:
            actions = [
                "contact_emergency_service",
                "prepare_hospital_transfer",
                "skip_evening_medication",
                "telemed",
            ]
            analysis = "พบสัญญาณที่อาจฉุกเฉิน ควรแจ้งแพทย์หรือพยาบาลทันที"
        elif risk == RiskLevel.WARNING:
            actions = ["notify_family", "call_nurse", "monitor_vitals"]
            analysis = "อาการต้องติดตามใกล้ชิด แนะนำให้ครอบครัวพิจารณาก่อนดำเนินการขั้นถัดไป"
        else:
            actions = ["continue_monitoring", "record_symptoms"]
            analysis = "ยังไม่พบสัญญาณเสี่ยงสูงจากระบบประเมินอัตโนมัติ"

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
                    "You are an elder-care risk assessment assistant. "
                    "Assess urgency and suggest options. Do not diagnose disease. "
                    "Return concise, family-readable guidance.",
                ),
                (
                    "human",
                    "Assess this patient report.\n"
                    "- Symptoms: {symptoms}\n"
                    "- Vitals: {vitals}\n"
                    "- Current medications: {medications}\n"
                    "- Retrieved medical context: {context}\n",
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
                "medications": safe.get("current_medications", []),
                "context": safe.get("retrieved_medical_context", ""),
            }
        )


def build_symptom_evaluator(settings: Settings) -> SymptomEvaluator:
    if settings.enable_external_ai and settings.google_api_key:
        return GeminiSymptomEvaluator(settings)
    return FallbackSymptomEvaluator()
