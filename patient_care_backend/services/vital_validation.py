"""Sanity checks for nurse-entered vitals — catches typos before AI evaluation."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    value: float | int | None
    severity: str  # "error" | "warning"
    message: str
    suggested_value: float | None = None


# Physiologically impossible → block
HARD_LIMITS: dict[str, tuple[float, float]] = {
    "systolic": (40, 300),
    "diastolic": (20, 200),
    "temperature_c": (30, 45),
    "temperature": (30, 45),
    "heart_rate": (20, 250),
    "spo2": (50, 100),
}

# Plausible but unusual → require nurse confirmation
SOFT_LIMITS: dict[str, tuple[float, float]] = {
    "systolic": (70, 220),
    "diastolic": (40, 130),
    "temperature_c": (34, 42),
    "temperature": (34, 42),
    "heart_rate": (40, 180),
    "spo2": (85, 100),
}

# Typo messages (Thai)
_TYPO_MSG_SYSTOLIC = "อาจพิมพ์ตก 0 — ตั้งใจ 120 แต่ได้ 12?"
_TYPO_MSG_DIASTOLIC = "อาจพิมพ์ตก 0 — ตั้งใจ 80 แต่ได้ 8?"
_TYPO_MSG_TEMPERATURE = "อาจพิมพ์ผิด — ตั้งใจ 37.5 แต่ได้ 375?"


def _get_float(vitals: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = vitals.get(key)
        if raw is not None and raw != "":
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None
    return None


def _suggest_temperature(value: float) -> float | None:
    """Suggest corrected temperature for likely decimal-shifted entries (e.g. 375 → 37.5)."""
    if value >= 100:
        return round(value / 10, 1)
    return None


def _typo_issue(field: str, value: float) -> ValidationIssue | None:
    """Detect common typo patterns that look like a missing digit or decimal shift."""
    if field == "systolic" and 10 <= value <= 19:
        return ValidationIssue(
            field=field,
            value=value,
            severity="warning",
            message=_TYPO_MSG_SYSTOLIC,
            suggested_value=value * 10,
        )
    if field == "diastolic" and 5 <= value <= 9:
        return ValidationIssue(
            field=field,
            value=value,
            severity="warning",
            message=_TYPO_MSG_DIASTOLIC,
            suggested_value=value * 10,
        )
    if field == "temperature_c" and value >= 100:
        return ValidationIssue(
            field=field,
            value=value,
            severity="warning",
            message=_TYPO_MSG_TEMPERATURE,
            suggested_value=_suggest_temperature(value),
        )
    return None


def validate_vitals(vitals: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    systolic = _get_float(vitals, "systolic")
    diastolic = _get_float(vitals, "diastolic")
    temperature = _get_float(vitals, "temperature_c", "temperature")
    heart_rate = _get_float(vitals, "heart_rate", "heartRate")
    spo2 = _get_float(vitals, "spo2", "oxygenSat")

    field_values = {
        "systolic": systolic,
        "diastolic": diastolic,
        "temperature_c": temperature,
        "heart_rate": heart_rate,
        "spo2": spo2,
    }

    for field, value in field_values.items():
        if value is None:
            continue

        # 1) Check for common typo patterns first (these override other checks).
        typo = _typo_issue(field, value)
        if typo:
            issues.append(typo)
            continue

        # 2) Hard limits — physiologically impossible values.
        hard = HARD_LIMITS.get(field)
        if hard and not (hard[0] <= value <= hard[1]):
            issues.append(
                ValidationIssue(
                    field=field,
                    value=value,
                    severity="error",
                    message=f"{field}={value} อยู่นอกช่วงที่เป็นไปได้ของร่างกายมนุษย์ ({hard[0]}–{hard[1]})",
                )
            )
            continue

        # 3) Soft limits — unusual but possible, ask nurse to double-check.
        soft = SOFT_LIMITS.get(field)
        if soft and not (soft[0] <= value <= soft[1]):
            issues.append(
                ValidationIssue(
                    field=field,
                    value=value,
                    severity="warning",
                    message=f"{field}={value} ดูผิดปกติ — กรุณายืนยันตัวเลขอีกครั้ง",
                )
            )

    # Cross-field: systolic must be greater than diastolic.
    if systolic is not None and diastolic is not None and systolic <= diastolic:
        issues.append(
            ValidationIssue(
                field="blood_pressure",
                value=None,
                severity="warning",
                message=f"ความดันบน ({systolic}) ต้องมากกว่าความดันล่าง ({diastolic}) — กรุณาตรวจสอบ",
            )
        )

    return issues


def validation_needs_confirmation(issues: list[ValidationIssue | dict[str, Any]]) -> bool:
    for issue in issues:
        if isinstance(issue, dict):
            if issue.get("severity") in {"error", "warning"}:
                return True
        elif issue.severity in {"error", "warning"}:
            return True
    return False


def issues_to_dicts(issues: list[ValidationIssue]) -> list[dict[str, Any]]:
    return [
        {
            "field": i.field,
            "value": i.value,
            "severity": i.severity,
            "message": i.message,
            "suggested_value": i.suggested_value,
        }
        for i in issues
    ]
