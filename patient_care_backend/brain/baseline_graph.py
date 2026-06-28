import functools

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from patient_care_backend.schemas import BaselineState
from patient_care_backend.services.baseline_analytics import (
    calculate_dynamic_thresholds,
    generate_baseline_insight,
)


class BaselineNodes:
    def fetch_history(self, state: BaselineState) -> BaselineState:
        history = state.get("vitals_history") or []
        if not history:
            return {
                "vitals_history": [],
                "errors": ["no_vitals_history"],
            }
        return {"vitals_history": history}

    def calculate_dynamic_thresholds(self, state: BaselineState) -> BaselineState:
        history = state.get("vitals_history") or []
        k = float(state.get("k") or 1.5)
        return {
            "calculated_thresholds": calculate_dynamic_thresholds(history, k=k),
        }

    def generate_ai_insight(self, state: BaselineState) -> BaselineState:
        history = state.get("vitals_history") or []
        thresholds = state.get("calculated_thresholds") or {}
        return {
            "ai_insight_text": generate_baseline_insight(history, thresholds),
        }

    def update_database(self, state: BaselineState) -> BaselineState:
        # FastAPI returns the persistence payload to Next.js, where Prisma owns DB writes.
        return {
            "calculated_thresholds": state.get("calculated_thresholds") or {},
            "ai_insight_text": state.get("ai_insight_text") or "",
        }


def build_baseline_graph():
    nodes = BaselineNodes()
    workflow = StateGraph(BaselineState)
    workflow.add_node("fetch_history", nodes.fetch_history)
    workflow.add_node("calculate_dynamic_thresholds", nodes.calculate_dynamic_thresholds)
    workflow.add_node("generate_ai_insight", nodes.generate_ai_insight)
    workflow.add_node("update_database", nodes.update_database)

    workflow.set_entry_point("fetch_history")
    workflow.add_edge("fetch_history", "calculate_dynamic_thresholds")
    workflow.add_edge("calculate_dynamic_thresholds", "generate_ai_insight")
    workflow.add_edge("generate_ai_insight", "update_database")
    workflow.add_edge("update_database", END)

    return workflow.compile(checkpointer=MemorySaver())


@functools.lru_cache(maxsize=1)
def get_baseline_graph():
    return build_baseline_graph()
