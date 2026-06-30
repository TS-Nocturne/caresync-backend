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
        elif symptoms or symptoms & self.warning_terms or self._temperature(vitals) >= 38:
            risk = RiskLevel.WARNING

        if risk == RiskLevel.CRITICAL:
            actions = [
                "notify_care_team_for_review",
                "show_emergency_contact_options",
                "record_information_flag",
            ]
        elif risk == RiskLevel.WARNING:
            actions = ["notify_family_for_review", "request_caregiver_review", "record_information_flag"]
        else:
            actions = ["record_observation", "continue_routine_logging"]

        return RiskAssessment(
            risk_level=risk,
            ai_analysis=self._build_analysis(risk, state),
            recommended_actions=actions,
        )

    def _build_analysis(self, risk: RiskLevel, state: PatientState) -> str:
        symptoms = [symptom.strip() for symptom in state.get("symptoms", []) if symptom and symptom.strip()]
        vitals = state.get("vitals", {})
        baseline = state.get("patient_baseline", {})
        recent_context = (state.get("recent_care_context") or "").strip()
        vital_findings = self._vital_findings(vitals, baseline)
        context_findings = self._context_findings(recent_context)

        findings = []
        if vital_findings:
            findings.append(f"ค่าสถิติร่างกายที่ควรทบทวน: {', '.join(vital_findings)}")
        elif vitals:
            findings.append("ค่าสถิติร่างกายล่าสุดยังไม่พบค่าที่หลุดจากเกณฑ์หลัก")

        if symptoms:
            findings.append(f"มีอาการ/บันทึกล่าสุด: {', '.join(symptoms[:6])}")
        if context_findings:
            findings.append(f"บริบทการดูแลล่าสุด: {context_findings}")

        if not findings:
            return "ยังไม่พบอาการผิดปกติหรือค่าสถิติร่างกายที่ต้องส่งต่อเป็นพิเศษ ผู้ดูแลกะต่อไปควรติดตามตามรอบปกติ"

        watch_items = []
        lowered = " ".join([*symptoms, recent_context]).lower()
        if any(term in lowered for term in ["confusion", "สับสน", "wandering", "ออกจากบ้าน", "พลัดหลง"]):
            watch_items.append("เฝ้าระวังความสับสน การเดินออกนอกพื้นที่ หรือการพลัดหลง")
        if any(term in lowered for term in ["fall", "หกล้ม", "ล้ม", "dizziness", "เวียนหัว"]):
            watch_items.append("เฝ้าระวังการหกล้มและช่วยพยุงเมื่อลุกเดิน")
        if any(term in lowered for term in ["fever", "มีไข้", "vomiting", "อาเจียน"]):
            watch_items.append("ติดตามไข้ อาการอ่อนเพลีย การดื่มน้ำ และอาการแย่ลง")
        if self._spo2(vitals) < self._critical_spo2_threshold(baseline):
            watch_items.append("ตรวจซ้ำค่าออกซิเจนปลายนิ้วและอาการหายใจลำบาก")
        if self._temperature(vitals) >= 38:
            watch_items.append("ติดตามอุณหภูมิซ้ำและอาการร่วม")

        if not watch_items:
            watch_items.append("ส่งต่อให้ผู้ดูแลทบทวนรายละเอียดและติดตามการเปลี่ยนแปลงในกะถัดไป")

        prefix = "พบข้อมูลระดับวิกฤตที่ต้องให้ผู้ดูแลตรวจสอบทันที" if risk == RiskLevel.CRITICAL else "พบข้อมูลที่ควรส่งต่อให้ผู้ดูแลกะถัดไป"
        return f"{prefix}: {'; '.join(findings[:3])}. ควร{'; '.join(watch_items[:2])}"

    def _vital_findings(self, vitals: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
        findings: list[str] = []
        temp = self._temperature(vitals)
        spo2 = self._spo2(vitals)

        if temp >= 38:
            findings.append(f"อุณหภูมิ {temp:g}°C")
        if spo2 < self._critical_spo2_threshold(baseline):
            findings.append(f"SpO2 {spo2:g}%")

        systolic = self._float_value(vitals, "systolic")
        diastolic = self._float_value(vitals, "diastolic")
        if systolic is not None and diastolic is not None:
            low_sys = self._float_value(baseline, "baseline_systolic_lower")
            high_sys = self._float_value(baseline, "baseline_systolic_upper")
            low_dia = self._float_value(baseline, "baseline_diastolic_lower")
            high_dia = self._float_value(baseline, "baseline_diastolic_upper")
            if (low_sys is not None and systolic < low_sys) or (high_sys is not None and systolic > high_sys):
                findings.append(f"ความดันตัวบน {systolic:g}")
            if (low_dia is not None and diastolic < low_dia) or (high_dia is not None and diastolic > high_dia):
                findings.append(f"ความดันตัวล่าง {diastolic:g}")

        heart_rate = self._float_value(vitals, "heart_rate")
        if heart_rate is None:
            heart_rate = self._float_value(vitals, "heartRate")
        low_hr = self._float_value(baseline, "baseline_heart_rate_lower")
        high_hr = self._float_value(baseline, "baseline_heart_rate_upper")
        if heart_rate is not None and (
            (low_hr is not None and heart_rate < low_hr) or (high_hr is not None and heart_rate > high_hr)
        ):
            findings.append(f"ชีพจร {heart_rate:g}/นาที")

        return findings

    @staticmethod
    def _context_findings(recent_context: str) -> str:
        if not recent_context or recent_context.startswith("No abnormal symptoms"):
            return ""
        lines = [line.strip() for line in recent_context.splitlines() if line.strip()]
        cleaned = []
        for line in lines[:3]:
            cleaned.append(
                line.replace("Current submitted abnormal symptoms:", "อาการที่บันทึก").replace(
                    "Recent symptom", "อาการก่อนหน้า"
                )
            )
        return "; ".join(cleaned)

    @staticmethod
    def _float_value(values: dict[str, Any], key: str) -> float | None:
        try:
            value = values.get(key)
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

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
        self._fallback = FallbackSymptomEvaluator()

    def _build_chain(self):
        if self._chain is not None:
            return self._chain

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an elder-care care-coordination handoff assistant. "
                    "Write a concrete handoff note for the next caregiver shift in Thai, 1-2 concise sentences. "
                    "You must summarize what actually happened from symptoms, vitals, recent notes, pain logs, "
                    "PRN/as-needed medication events, and retrieved patient context. "
                    "Mention stable vitals only when useful, but do not call the patient normal if abnormal symptoms, "
                    "pain, PRN/as-needed medication use, or relevant context exists. "
                    "State what the next caregiver should watch, check, or review. "
                    "Do not output a generic disclaimer, legal warning, diagnosis, treatment instruction, triage order, "
                    "medication adjustment, or transfer decision. "
                    "The ai_analysis field must contain patient-specific facts from the input; it must not be only "
                    "a generic phrase such as 'send to caregiver review' or 'not a diagnosis'.",
                ),
                (
                    "human",
                    "Create the care coordination note from this report.\n"
                    "- Symptoms submitted now: {symptoms}\n"
                    "- Vitals submitted now: {vitals}\n"
                    "- Patient-specific baseline vitals: {patient_baseline}\n"
                    "- Current medications: {medications}\n"
                    "- Recent care context from the last 24 hours: {recent_care_context}\n"
                    "- Retrieved care context/RAG: {context}\n"
                    "Return structured output with risk_level, ai_analysis, and recommended_actions.",
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
        try:
            return chain.invoke(
                {
                    "symptoms": safe.get("symptoms", []),
                    "vitals": safe.get("vitals", {}),
                    "patient_baseline": safe.get("patient_baseline", {}),
                    "medications": safe.get("current_medications", []),
                    "recent_care_context": safe.get("recent_care_context", ""),
                    "context": safe.get("retrieved_medical_context", ""),
                }
            )
        except Exception:
            return self._fallback.evaluate(state)


def build_symptom_evaluator(settings: Settings) -> SymptomEvaluator:
    if settings.enable_external_ai and settings.google_api_key:
        return GeminiSymptomEvaluator(settings)
    return FallbackSymptomEvaluator()
