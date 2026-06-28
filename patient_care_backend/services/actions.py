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

        if decision in {"request_professional_review", "contact_emergency_services_user_initiated"}:
            return [
                "notify_care_team",
                "record_family_requested_external_review",
            ]

        if decision in {"call_nurse", "notify_nurse", "request_caregiver_review"}:
            return ["notify_assigned_caregiver", "record_family_caregiver_contact_decision"]

        return [f"record_family_decision:{decision}"]
