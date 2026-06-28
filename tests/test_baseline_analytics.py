from patient_care_backend.brain.baseline_graph import build_baseline_graph
from patient_care_backend.services.baseline_analytics import calculate_dynamic_thresholds


def test_calculate_dynamic_thresholds_uses_mu_plus_minus_k_sigma():
    history = [
        {"systolic": 120, "diastolic": 75, "temperature": 36.5, "heart_rate": 72, "oxygen_sat": 95},
        {"systolic": 122, "diastolic": 76, "temperature": 36.6, "heart_rate": 74, "oxygen_sat": 96},
        {"systolic": 124, "diastolic": 77, "temperature": 36.7, "heart_rate": 76, "oxygen_sat": 95},
    ]

    thresholds = calculate_dynamic_thresholds(history, k=1.5)

    assert thresholds["baseline_systolic_lower"] == 119
    assert thresholds["baseline_systolic_upper"] == 125
    assert thresholds["baseline_oxygen_sat_min"] == 94
    assert thresholds["baseline_oxygen_sat_max"] == 96


def test_baseline_graph_generates_thresholds_and_insight():
    graph = build_baseline_graph()
    config = {"configurable": {"thread_id": "baseline-test"}}

    graph.invoke(
        {
            "patient_id": "p-001",
            "k": 1.5,
            "vitals_history": [
                {"systolic": 120, "oxygen_sat": 95},
                {"systolic": 122, "oxygen_sat": 96},
                {"systolic": 124, "oxygen_sat": 95},
            ],
        },
        config,
    )

    state = graph.get_state(config).values
    assert state["calculated_thresholds"]["baseline_systolic_upper"] == 125
    assert state["ai_insight_text"]
