from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


class RiskLevel(StrEnum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class ValidationIssueDict(TypedDict, total=False):
    field: str
    value: float | int | None
    severity: Literal["error", "warning"]
    message: str
    suggested_value: float | None


class PatientState(TypedDict, total=False):
    patient_id: str
    vitals: dict[str, Any]
    symptoms: list[str]
    current_medications: list[str]
    retrieved_medical_context: str
    validation_issues: list[ValidationIssueDict]
    validation_confirmed: bool
    risk_level: Literal["normal", "warning", "critical"]
    ai_analysis: str
    recommended_actions: list[str]
    family_decision: str | None
    family_availability: list[dict[str, Any]]
    executed_actions: list[str]
    errors: list[str]


class AssessmentRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=128)
    vitals: dict[str, Any] = Field(default_factory=dict)
    symptoms: list[str] = Field(default_factory=list, max_length=25)
    current_medications: list[str] = Field(default_factory=list, max_length=100)
    thread_id: str | None = Field(default=None, max_length=128)
    validation_confirmed: bool = False

    @field_validator("symptoms", "current_medications")
    @classmethod
    def limit_text_items(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value and value.strip()]
        if any(len(value) > 500 for value in cleaned):
            raise ValueError("Text items must be 500 characters or fewer")
        return cleaned

    @field_validator("vitals")
    @classmethod
    def limit_vitals(cls, values: dict[str, Any]) -> dict[str, Any]:
        if len(values) > 30:
            raise ValueError("Too many vital fields")
        return values


class DecisionRequest(BaseModel):
    family_decision: str = Field(min_length=1, max_length=500)


class AvailabilityRequest(BaseModel):
    member_name: str = Field(min_length=1, max_length=120)
    day: int = Field(ge=0, le=6)
    start_hour: int = Field(ge=0, le=23)
    end_hour: int = Field(ge=1, le=24)
    note: str = Field(default="available_for_doctor_visit", max_length=500)


class AssessmentResult(BaseModel):
    thread_id: str
    status: Literal["completed", "waiting_for_human", "needs_confirmation"]
    state: PatientState


class RiskAssessment(BaseModel):
    risk_level: RiskLevel
    ai_analysis: str
    recommended_actions: list[str]
