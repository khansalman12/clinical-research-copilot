import time
from langgraph.graph import StateGraph, START, END
from config import CRAG_HIGH, CRAG_LOW

from agent.state import CopilotState
from agent.nodes import (
    classify_node,
    retrieve_node,
    grade_node,
    rewrite_node,
    generate_node,
    refuse_node,
)


def decide_route(state: CopilotState) -> str:
    elapsed = time.time() - state["started_at"]
    if elapsed >= 30.0:
        print(f"[router] timeout at {elapsed:.2f}s -> refuse")
        return "refuse"

    if state["step_count"] >= 6:
        print(f"[router] step budget exhausted ({state['step_count']}) -> refuse")
        return "refuse"

    score = state["crag_score"]

    if score >= CRAG_HIGH:
        print(f"[router] score {score:.2f} >= {CRAG_HIGH} -> generate")
        return "generate"

    if score < CRAG_LOW:
        print(f"[router] score {score:.2f} < {CRAG_LOW} -> refuse")
        return "refuse"

    attempts = len(state.get("query_attempts") or [])
    if attempts >= 2:
        print(f"[router] score {score:.2f} partial, rewrite budget exhausted -> generate")
        return "generate"

    print(f"[router] score {score:.2f} partial, attempt {attempts + 1} -> rewrite")
    return "rewrite"


workflow = StateGraph(CopilotState)

workflow.add_node("classify", classify_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)
workflow.add_node("refuse", refuse_node)

workflow.add_edge(START, "classify")
workflow.add_edge("classify", "retrieve")
workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    decide_route,
    {
        "generate": "generate",
        "refuse": "refuse",
        "rewrite": "rewrite",
    },
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)
workflow.add_edge("refuse", END)

app = workflow.compile()
