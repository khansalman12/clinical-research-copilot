import pytest
from agent import run_agent

pytestmark = pytest.mark.slow  # hits the real Groq API + loads the embedding model


def test_high_confidence_query_returns_grounded_answer():
    result = run_agent("What is the primary mechanism of resmetirom?")

    assert result["crag_score"] >= 0.70
    assert len(result["sources"]) > 0
    assert "PMID" in result["context_text"]
    assert result["answer"] is not None
    assert len(result["answer"]) > 0


def test_out_of_domain_query_refuses_instead_of_hallucinating():
    result = run_agent("What is the capital of France?")

    assert result["crag_score"] < 0.70
    low_confidence_markers = ["don't have", "couldn't find", "timed out"]
    assert any(m in result["answer"].lower() for m in low_confidence_markers)


def test_result_always_has_expected_shape():
    result = run_agent("What lifestyle changes help MASLD?")

    for key in ("answer", "sources", "query_type", "crag_score", "attempts", "context_text"):
        assert key in result
