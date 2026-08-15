import pytest
from corpus import load_abstracts
from retriever import build_index, retrieve

pytestmark = pytest.mark.slow  # loads a real sentence-transformers model


@pytest.fixture(scope="module")
def index():
    abstracts = load_abstracts()
    dense, sparse = build_index(abstracts)
    return abstracts, dense, sparse


def test_retrieve_returns_k_results(index):
    abstracts, dense, sparse = index
    results = retrieve("resmetirom mechanism of action", abstracts, dense, sparse, k=5)
    assert len(results) == 5


def test_retrieve_finds_relevant_drug_by_name(index):
    abstracts, dense, sparse = index
    results = retrieve("What is resmetirom?", abstracts, dense, sparse, k=5)
    titles = " ".join(r["title"].lower() for r in results)
    assert "resmetirom" in titles


def test_retrieve_ranks_semaglutide_query_correctly(index):
    abstracts, dense, sparse = index
    results = retrieve("semaglutide effects on liver steatosis", abstracts, dense, sparse, k=5)
    titles = " ".join(r["title"].lower() for r in results)
    assert "semaglutide" in titles
