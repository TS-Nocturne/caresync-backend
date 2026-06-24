from typing import Protocol

from patient_care_backend.schemas import PatientState


class DecisionExecutor(Protocol):
    def execute(self, state: PatientState) -> list[str]:
        """Execute side effects selected by the family decision."""


class LoggingDecisionExecutor:
    def execute(self, state: PatientState) -> list[str]:
        decision = state.get("family_decision")
        if not decision:
            return ["no_family_decision_provided"]

        if decision in {"prepare_hospital_transfer", "contact_emergency_service"}:
            return [
                "hold_non_critical_schedule",
                "notify_care_team",
                "record_family_hospital_transfer_decision",
            ]

        if decision == "call_nurse":
            return ["notify_assigned_nurse", "record_family_nurse_call_decision"]

        return [f"record_family_decision:{decision}"]
