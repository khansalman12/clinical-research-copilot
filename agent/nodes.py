import time
import numpy as np
from groq import Groq

from corpus import load_abstracts
from retriever import build_index, retrieve, model as embed_model
from config import GROQ_API_KEY, LLM_MODEL, LLM_MODEL_FAST, FINAL_K, CRAG_HIGH, CRAG_LOW
from agent.state import CopilotState
from resilience.circuit_breaker import breaker

client = Groq(api_key=GROQ_API_KEY)
abstracts = load_abstracts()
dense, sparse = build_index(abstracts)
print("[Agent] Ready.")


def classify_node(state: CopilotState) -> dict:
    q = state["query"].lower()

    if any(w in q for w in ["compare", "vs", "versus", "difference"]):
        query_type = "multi-hop"
    elif len(state["query"].split()) <= 3:
        query_type = "vague"
    else:
        query_type = "factoid"

    print(f"[classify] type={query_type}")

    return {
        "query_type": query_type,
        "step_count": state.get("step_count", 0) + 1,
        "started_at": state.get("started_at") or time.time(),
        "query_attempts": state.get("query_attempts") or [],
    }


def retrieve_node(state: CopilotState) -> dict:
    current_query = (
        state["query_attempts"][-1]
        if state["query_attempts"]
        else state["query"]
    )
    print(f"[retrieve] query='{current_query}'")

    chunks = retrieve(current_query, abstracts, dense, sparse, k=FINAL_K)

    return {
        "retrieved_chunks": chunks,
        "step_count": state["step_count"] + 1,
    }


def grade_node(state: CopilotState) -> dict:
    chunks = state["retrieved_chunks"]
    if not chunks:
        print("[grade] no chunks → score=0.0")
        return {"crag_score": 0.0, "step_count": state["step_count"] + 1}

    q_vec = embed_model.encode([state["query"]], normalize_embeddings=True)
    texts = [c["title"] + " " + c["text"] for c in chunks]
    vecs = embed_model.encode(texts, normalize_embeddings=True)
    scores = (vecs @ q_vec.T).squeeze()

    for chunk, s in zip(chunks, scores):
        chunk["score"] = float(s)

    crag_score = float(np.mean(scores))
    print(f"[grade] crag_score={crag_score:.2f}")

    return {
        "crag_score": crag_score,
        "retrieved_chunks": chunks,
        "step_count": state["step_count"] + 1,
    }


def rewrite_node(state: CopilotState) -> dict:
    prompt = (
        f"Rewrite this clinical query to be more specific and search-friendly.\n"
        f"Return ONLY the rewritten query.\n\n"
        f"Original: {state['query']}"
    )

    rewritten = breaker.call(
        client, LLM_MODEL_FAST,
        [{"role": "user", "content": prompt}],
        temperature=0.1,
    ).strip()

    attempts = list(state["query_attempts"])
    attempts.append(rewritten)

    print(f"[rewrite] → '{rewritten}'")

    return {
        "query_attempts": attempts,
        "step_count": state["step_count"] + 1,
    }


def generate_node(state: CopilotState) -> dict:
    chunks = state["retrieved_chunks"]

    # Lost-in-middle mitigation: best chunk first and second-best last, since
    # LLMs attend less to context in the middle of a long prompt.
    sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
    left, right = [], []
    for i, c in enumerate(sorted_chunks):
        if i % 2 == 0:
            left.append(c)
        else:
            right.insert(0, c)
    reordered = left + right

    context = "\n\n---\n\n".join(
        f"[PMID {c['pmid']}] {c['title']}\n{c['text']}" for c in reordered
    )

    caveat = (
        "\nNote: Confidence is partial. State limitations in your answer."
        if state["crag_score"] < CRAG_HIGH
        else ""
    )

    prompt = (
        f"You are a clinical research assistant.\n"
        f"Answer using ONLY the context below. Cite PMID for every fact.{caveat}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\nAnswer:"
    )

    answer = breaker.call(
        client, LLM_MODEL,
        [{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    sources = [str(c["pmid"]) for c in chunks]

    print(f"[generate] answer ready. sources={sources}")

    return {
        "final_answer": answer,
        "sources": sources,
        "step_count": state["step_count"] + 1,
    }


def refuse_node(state: CopilotState) -> dict:
    reason = state.get("refusal_reason") or "low_confidence"

    messages = {
        "low_confidence": "I don't have enough reliable evidence to answer this safely.",
        "timeout": "The request timed out before I could verify an answer.",
        "budget_exhausted": "I couldn't find high-confidence sources within the step limit.",
    }

    print(f"[refuse] reason={reason}")

    return {
        "final_answer": messages.get(reason, messages["low_confidence"]),
        "sources": [],
        "refusal_reason": reason,
        "step_count": state["step_count"] + 1,
    }
