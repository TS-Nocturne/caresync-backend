from patient_care_backend.schemas import PatientState
from patient_care_backend.services.actions import DecisionExecutor
from patient_care_backend.services.llm import SymptomEvaluator
from patient_care_backend.services.vector_store import PatientContextRetriever
from patient_care_backend.services.vital_validation import (
    issues_to_dicts,
    validate_vitals,
)


class BrainNodes:
    def __init__(
        self,
        context_retriever: PatientContextRetriever,
        symptom_evaluator: SymptomEvaluator,
        decision_executor: DecisionExecutor,
    ) -> None:
        self._context_retriever = context_retriever
        self._symptom_evaluator = symptom_evaluator
        self._decision_executor = decision_executor

    def validate_data(self, state: PatientState) -> PatientState:
        """Data_Validation_Node — sanity check vitals before AI evaluation."""
        if state.get("validation_confirmed"):
            return {"validation_issues": []}

        vitals = state.get("vitals") or {}
        issues = validate_vitals(vitals)
        return {"validation_issues": issues_to_dicts(issues)}

    def retrieve_patient_context(self, state: PatientState) -> PatientState:
        try:
            return {"retrieved_medical_context": self._context_retriever.retrieve(state)}
        except Exception as exc:
            return {
                "retrieved_medical_context": "",
                "errors": [f"retrieve_patient_context_failed:{exc}"],
            }

    def evaluate_symptoms(self, state: PatientState) -> PatientState:
        try:
            result = self._symptom_evaluator.evaluate(state)
            return {
                "risk_level": result.risk_level.value,
                "ai_analysis": result.ai_analysis,
                "recommended_actions": result.recommended_actions,
            }
        except Exception as exc:
            return {
                "risk_level": "warning",
                "ai_analysis": "Risk evaluation failed; route to human review.",
                "recommended_actions": ["notify_family", "call_nurse"],
                "errors": [f"evaluate_symptoms_failed:{exc}"],
            }

    def execute_decision(self, state: PatientState) -> PatientState:
        try:
            return {"executed_actions": self._decision_executor.execute(state)}
        except Exception as exc:
            return {"errors": [f"execute_decision_failed:{exc}"]}
