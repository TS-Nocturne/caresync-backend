from patient_care_backend.brain.graph import build_brain_graph
from patient_care_backend.services.vital_validation import validate_vitals, validation_needs_confirmation


def test_warning_assessment_waits_for_human_and_resumes():
    graph = build_brain_graph()
    config = {"configurable": {"thread_id": "test-warning"}}

    graph.invoke(
        {
            "patient_id": "p-001",
            "vitals": {"temperature_c": 38.5},
            "symptoms": ["fever"],
            "current_medications": ["metformin"],
            "family_decision": None,
            "validation_confirmed": True,
        },
        config,
    )

    state = graph.get_state(config).values
    assert state["risk_level"] == "warning"
    assert "executed_actions" not in state

    graph.update_state(config, {"family_decision": "call_nurse"}, as_node="evaluate")
    graph.invoke(None, config)

    resumed = graph.get_state(config).values
    assert resumed["family_decision"] == "call_nurse"
    assert "notify_assigned_caregiver" in resumed["executed_actions"]


def test_symptom_assessment_waits_for_human_even_with_stable_vitals():
    graph = build_brain_graph()
    config = {"configurable": {"thread_id": "test-symptom-warning"}}

    graph.invoke(
        {
            "patient_id": "p-002",
            "vitals": {"temperature_c": 36.7, "spo2": 98},
            "symptoms": ["mild tiredness"],
            "current_medications": [],
            "family_decision": None,
            "validation_confirmed": True,
        },
        config,
    )

    state = graph.get_state(config).values
    assert state["risk_level"] == "warning"
    assert "executed_actions" not in state


def test_normal_assessment_completes_without_human_pause():
    graph = build_brain_graph()
    config = {"configurable": {"thread_id": "test-normal"}}

    graph.invoke(
        {
            "patient_id": "p-002",
            "vitals": {"temperature_c": 36.7, "spo2": 98},
            "symptoms": [],
            "current_medications": [],
            "family_decision": None,
            "validation_confirmed": True,
        },
        config,
    )

    state = graph.get_state(config).values
    assert state["risk_level"] == "normal"
    assert "executed_actions" not in state


def test_validation_blocks_typo_before_evaluation():
    graph = build_brain_graph()
    config = {"configurable": {"thread_id": "test-typo"}}

    graph.invoke(
        {
            "patient_id": "p-003",
            "vitals": {"systolic": 12, "diastolic": 80, "temperature_c": 36.5},
            "symptoms": [],
            "current_medications": [],
            "validation_confirmed": False,
        },
        config,
    )

    state = graph.get_state(config).values
    assert state.get("validation_issues")
    assert "risk_level" not in state


def test_validation_confirmed_skips_recheck():
    graph = build_brain_graph()
    config = {"configurable": {"thread_id": "test-confirmed-typo"}}

    graph.invoke(
        {
            "patient_id": "p-004",
            "vitals": {"systolic": 12, "diastolic": 80},
            "symptoms": [],
            "current_medications": [],
            "validation_confirmed": True,
        },
        config,
    )

    state = graph.get_state(config).values
    assert state.get("validation_issues") == []


def test_vital_validation_detects_temperature_typo():
    issues = validate_vitals({"temperature_c": 375})
    assert validation_needs_confirmation(issues)
    assert any(i.field == "temperature_c" for i in issues)
