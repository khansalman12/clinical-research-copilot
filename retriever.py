from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from config import TOP_K

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RRF_K = 60

model = SentenceTransformer(MODEL_NAME)


def build_index(abstracts):
    texts = [a["title"] + " " + a["text"] for a in abstracts]
    dense = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    sparse = BM25Okapi([t.lower().split() for t in texts])
    return dense, sparse


def retrieve(query, abstracts, dense, sparse, k=TOP_K):
    q_vec = model.encode([query], normalize_embeddings=True)
    dense_scores = (dense @ q_vec.T).squeeze()
    dense_top = np.argsort(dense_scores)[::-1][:k * 3]

    bm25_scores = sparse.get_scores(query.lower().split())
    sparse_top = np.argsort(bm25_scores)[::-1][:k * 3]

    # RRF merges by rank, not raw score — BM25 (0-40) and cosine (0-1) aren't
    # on comparable scales, so a rank-based fusion avoids normalizing either.
    rrf = {}
    for rank, idx in enumerate(dense_top):
        pmid = abstracts[idx]["pmid"]
        rrf[pmid] = rrf.get(pmid, 0) + 1 / (RRF_K + rank + 1)

    for rank, idx in enumerate(sparse_top):
        pmid = abstracts[idx]["pmid"]
        rrf[pmid] = rrf.get(pmid, 0) + 1 / (RRF_K + rank + 1)

    top_pmids = sorted(rrf, key=lambda p: rrf[p], reverse=True)[:k]
    pmid_map = {a["pmid"]: a for a in abstracts}
    return [pmid_map[pmid] for pmid in top_pmids]


if __name__ == "__main__":
    from corpus import load_abstracts

    abstracts = load_abstracts()
    print("Building index...")
    dense, sparse = build_index(abstracts)

    query = "What drugs reduce liver fibrosis in MASH?"
    results = retrieve(query, abstracts, dense, sparse)

    print(f"\nTop {TOP_K} results for: '{query}'\n")
    for i, a in enumerate(results, 1):
        print(f"  {i}. PMID {a['pmid']}: {a['title'][:80]}")
