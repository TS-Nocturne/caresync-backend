from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


MetricSpec = tuple[str, str, str]

METRICS: tuple[MetricSpec, ...] = (
    ("systolic", "baseline_systolic_lower", "baseline_systolic_upper"),
    ("diastolic", "baseline_diastolic_lower", "baseline_diastolic_upper"),
    ("temperature", "baseline_temperature_lower", "baseline_temperature_upper"),
    ("heart_rate", "baseline_heart_rate_lower", "baseline_heart_rate_upper"),
    ("oxygen_sat", "baseline_oxygen_sat_min", "baseline_oxygen_sat_max"),
)


def _round_metric(metric: str, value: float | None) -> float | None:
    if value is None:
        return None
    digits = 1 if metric == "temperature" else 0
    return round(value, digits)


def _series(history: list[dict[str, Any]], metric: str) -> list[float]:
    frame = _history_frame(history)
    if metric not in frame:
        return []
    return frame[metric].dropna().astype(float).tolist()


def _history_frame(history: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(history)
    for metric, _, _ in METRICS:
        if metric not in frame:
            frame[metric] = np.nan
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    return frame


def calculate_dynamic_thresholds(
    history: list[dict[str, Any]],
    k: float = 1.5,
) -> dict[str, float | None]:
    thresholds: dict[str, float | None] = {}
    frame = _history_frame(history)

    for metric, lower_key, upper_key in METRICS:
        values = frame[metric].dropna()
        if values.empty:
            thresholds[lower_key] = None
            thresholds[upper_key] = None
            continue

        mean = float(values.mean())
        sigma = float(values.std(ddof=1)) if len(values) >= 2 else 0.0
        lower = mean - (k * sigma)
        upper = mean + (k * sigma)

        if metric == "oxygen_sat":
            lower = float(np.clip(lower, 50.0, 100.0))
            upper = float(np.clip(upper, 50.0, 100.0))

        thresholds[lower_key] = _round_metric(metric, lower)
        thresholds[upper_key] = _round_metric(metric, upper)

    return thresholds


def generate_baseline_insight(
    history: list[dict[str, Any]],
    thresholds: dict[str, float | None],
) -> str:
    if not history:
        return "ยังไม่มีข้อมูลสัญญาณชีพเพียงพอสำหรับสร้าง baseline เฉพาะบุคคล"

    frame = _history_frame(history)
    systolic_values = frame["systolic"].dropna()
    oxygen_values = frame["oxygen_sat"].dropna()

    messages: list[str] = []
    if not systolic_values.empty:
        midpoint = max(1, len(systolic_values) // 2)
        earlier = float(systolic_values.iloc[:midpoint].mean())
        recent = float(systolic_values.iloc[midpoint:].mean())
        if recent - earlier >= 5:
            messages.append("ความดันมีแนวโน้มสูงขึ้นเล็กน้อย")
        elif earlier - recent >= 5:
            messages.append("ความดันมีแนวโน้มลดลงเล็กน้อย")

    oxygen_min = thresholds.get("baseline_oxygen_sat_min")
    if not oxygen_values.empty and oxygen_min is not None and float(oxygen_values.min()) <= oxygen_min:
        messages.append("ค่าออกซิเจนมีบางช่วงแตะขอบล่างของ baseline")

    if not messages:
        return "สัญญาณชีพโดยรวมคงที่เมื่อเทียบกับ baseline เฉพาะบุคคลในช่วง 7 วันที่ผ่านมา"

    return "สัญญาณชีพโดยรวมยังติดตามได้ แต่" + " และ".join(messages)
