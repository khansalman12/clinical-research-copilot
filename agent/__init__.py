from agent.graph import app as agent_app


def run_agent(query: str) -> dict:
    initial_state = {
        "query": query,
        "query_attempts": [],
        "retrieved_chunks": [],
        "crag_score": 0.0,
        "step_count": 0,
        "started_at": 0.0,
        "final_answer": None,
        "refusal_reason": None,
        "sources": [],
    }

    final_state = agent_app.invoke(initial_state)

    attempts = len(final_state.get("query_attempts") or [])
    attempts_count = max(1, attempts + 1)

    chunks = final_state.get("retrieved_chunks") or []
    context_text = "\n\n---\n\n".join(
        f"[PMID {c['pmid']}] {c['title']}\n{c['text']}" for c in chunks
    )

    return {
        "answer": final_state.get("final_answer"),
        "sources": final_state.get("sources") or [],
        "query_type": final_state.get("query_type", "factoid"),
        "crag_score": round(final_state.get("crag_score", 0.0), 2),
        "attempts": attempts_count,
        "context_text": context_text,
    }
