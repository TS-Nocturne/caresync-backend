import functools

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from patient_care_backend.brain.nodes import BrainNodes
from patient_care_backend.config import Settings, get_settings
from patient_care_backend.schemas import PatientState
from patient_care_backend.services.actions import LoggingDecisionExecutor
from patient_care_backend.services.llm import build_symptom_evaluator
from patient_care_backend.services.vector_store import build_context_retriever
from patient_care_backend.services.vital_validation import validation_needs_confirmation


def route_after_validation(state: PatientState) -> str:
    issues = state.get("validation_issues") or []
    if validation_needs_confirmation(issues):
        return END
    return "retrieve"


def route_risk(state: PatientState) -> str:
    if state.get("risk_level") in {"warning", "critical"}:
        return "execute_decision"
    return END


def build_brain_graph(settings: Settings | None = None):
    settings = settings or get_settings()
    nodes = BrainNodes(
        context_retriever=build_context_retriever(settings),
        symptom_evaluator=build_symptom_evaluator(settings),
        decision_executor=LoggingDecisionExecutor(),
    )

    workflow = StateGraph(PatientState)
    workflow.add_node("validate_data", nodes.validate_data)
    workflow.add_node("retrieve", nodes.retrieve_patient_context)
    workflow.add_node("evaluate", nodes.evaluate_symptoms)
    workflow.add_node("execute_decision", nodes.execute_decision)

    workflow.set_entry_point("validate_data")
    workflow.add_conditional_edges(
        "validate_data",
        route_after_validation,
        {
            "retrieve": "retrieve",
            END: END,
        },
    )
    workflow.add_edge("retrieve", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        route_risk,
        {
            "execute_decision": "execute_decision",
            END: END,
        },
    )
    workflow.add_edge("execute_decision", END)

    return workflow.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["execute_decision"],
    )

@functools.lru_cache(maxsize=1)
def get_brain_graph():
    return build_brain_graph()
