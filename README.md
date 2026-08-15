# Clinical Research Copilot

Agentic RAG system over a corpus of 100 PubMed abstracts on MASLD/MASH (metabolic
dysfunction-associated steatotic liver disease), built as a LangGraph state machine
with hybrid retrieval, corrective self-grading, and a custom LLM-as-judge eval harness.

## Why this exists

Most RAG demos retrieve once and answer. This project treats retrieval quality as
something the system checks and reacts to, not something it assumes — the agent
grades its own retrieval, rewrites the query if the grade is weak, and refuses to
answer rather than hallucinate if it can't find reliable evidence.

## Architecture

```
Query
  │
  ▼
[classify] ── heuristic query-type tag (factoid / multi-hop / vague), no LLM cost
  │
  ▼
[retrieve] ── hybrid search: BM25 (sparse) + BGE-small (dense) fused with
  │           Reciprocal Rank Fusion — rank-based merge, not score-based
  ▼
[grade] ──── CRAG self-check: cosine similarity between query and each
  │          retrieved chunk, no LLM call
  ▼
 router (grade.py: decide_route)
  │
  ├── score ≥ 0.70 ─────────────────────────────► [generate] ─► answer + citations
  │
  ├── 0.35 ≤ score < 0.70, retries < 2 ──► [rewrite] ─► loop back to [retrieve]
  │        (cheap model rephrases the query, then re-runs hybrid search)
  │
  ├── 0.35 ≤ score < 0.70, retries exhausted ──► [generate] (with a stated caveat)
  │
  └── score < 0.35 ──────────────────────────────► [refuse] ─► honest "I don't know"

Safety gates on every router pass: wall-clock timeout (30s) and step-count budget (6)
both force a refusal, regardless of CRAG score, so a bad loop can't run forever.
```

Every LLM call (rewrite, generate) goes through a circuit breaker
([resilience/circuit_breaker.py](resilience/circuit_breaker.py)): 3 consecutive
failures trips it open and routes to a smaller fallback model (`llama-3.1-8b-instant`)
until a 60s recovery window passes.

### Why these choices

- **RRF over score-blending** — BM25 scores (0–40) and cosine similarity (0–1) aren't
  comparable on the same scale. RRF merges by *rank*, not raw score, so no
  normalization hack is needed.
- **CRAG grading is a cosine check, not an LLM call** — it runs on every single
  retrieval, including every rewrite loop iteration. An LLM-graded version would
  multiply cost and latency for a signal a cheap embedding comparison already gives.
- **Async job pattern in the API** — `POST /query` returns a `job_id` in under 100ms
  and runs the agent in a background task; the client polls `GET /result/{job_id}`.
  A single agent run can take several seconds (embedding + up to 2 LLM calls), and an
  HTTP framework should never hold a connection open for that long under real load.

## Project layout

```
agent/            LangGraph state machine (state.py, nodes.py, graph.py)
api/               FastAPI HTTP layer — async job submission + polling + feedback
db/                SQLAlchemy models + repository functions (SQLite by default)
resilience/        Circuit breaker for LLM calls
tests/             pytest suite (fast: corpus/circuit-breaker, slow: retriever/agent e2e)
retriever.py       Hybrid BM25 + dense retrieval with RRF fusion
corpus.py          Loads data.json into the shape the rest of the app expects
config.py          Single source of truth for models, thresholds, paths
evals.py           Eval harness: keyword/citation metrics + LLM-as-judge faithfulness & relevancy
eval_dataset.json  25 hand-written queries (factoid, multi-hop, vague, and deliberate refusal cases)
run_evals.py       Runs the full agent against eval_dataset.json, writes eval_report.json
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
```

## Running it

**Interactive smoke test:**
```bash
python -c "from agent import run_agent; print(run_agent('What is the primary mechanism of resmetirom?'))"
```

**As an HTTP API:**
```bash
uvicorn api.main:app --reload
# POST /query        {"query": "..."}  -> {"job_id": "..."}
# GET  /result/{id}                    -> status + answer + sources + crag_score
# POST /feedback/{run_id}  {"rating": "up"|"down"}
# GET  /health
```

**Run the eval suite (writes `eval_report.json`):**
```bash
python run_evals.py
```

**Run the test suite:**
```bash
pytest -m "not slow"   # fast: no model loading, no API calls
pytest                 # everything, including retriever + real agent e2e (costs Groq calls)
```

## Evaluation methodology

25 queries against the 100-abstract corpus, split across:
- **factoid** — direct fact lookup answerable from a specific abstract
- **multi-hop** — requires comparing across two or more abstracts
- **vague** — 1-2 word underspecified queries (tests the classifier + graceful handling)
- **refusal** — deliberately out-of-corpus questions (e.g. "capital of France", a drug
  not in the corpus) where the *correct* behavior is refusing to answer

Metrics, computed per query and averaged:
- **keyword accuracy** — does the answer contain the expected clinical terms?
- **citation recall** — for queries with a known expected PMID, did retrieval find it?
- **faithfulness** — custom LLM-as-judge (Groq `llama-3.3-70b-versatile`) scores
  whether every claim in the answer is actually supported by the retrieved context.
  This is the same concept the `ragas` library calls "faithfulness," implemented
  directly against Groq instead of pulling in `ragas`'s OpenAI-centric dependency
  chain. The judge deliberately uses the strong model, not the fast one used for
  query rewriting — an earlier version judged with `llama-3.1-8b-instant` and
  produced false-negative scores on long (~7.5k character) context, missing claims
  that were genuinely supported elsewhere in the retrieved abstracts. Judging is a
  one-time eval cost, so the larger model is worth it even though generation uses
  a cheaper one for some steps.
- **answer relevancy** — LLM-as-judge scores whether the answer addresses the
  question asked, independent of factual grounding.
- **refusal accuracy** — for the deliberate refusal test cases, did the agent
  correctly decline instead of hallucinating?

### Results

_Generated by `python run_evals.py` on 2026-08-15 — see `eval_report.json` in the repo
for the full per-query breakdown. Numbers below are pulled directly from that file._

| Metric | Value |
|---|---|
| Queries evaluated | 25 |
| Answer rate | 72% (18/25) |
| Avg keyword accuracy | 72.9% |
| Avg citation recall | 78.7% |
| Avg faithfulness (answered queries only) | 1.00 |
| Avg relevancy (answered queries only) | 0.97 |
| Refusal accuracy (3 deliberate out-of-corpus tests) | 66.7% (2/3) |
| Avg latency | 6.9s |
| p95 latency | 12.4s |

**What this actually shows:**
- On the 18 queries the agent chose to answer, faithfulness and relevancy are both
  near-perfect — when the CRAG gate lets a query through, the generated answer is
  reliably grounded in the retrieved abstracts and on-topic.
- The 28% refusal rate is mostly the CRAG gate being conservative, not the agent
  failing: several "medium" difficulty factoid queries (e.g. "What lifestyle changes
  are recommended for MASLD?") scored just under the 0.70 confidence threshold and
  were refused rather than answered with a lower-confidence caveat, even though the
  corpus does contain relevant material. This is a tunable threshold trade-off
  (fewer refusals vs. more caveats), not a retrieval bug.
- **Refusal accuracy is the most important number to be honest about: it's 2/3, not
  3/3.** One deliberately out-of-corpus query ("What is the recommended dosage of
  metformin for treating MASH-related insulin resistance?") scored CRAG 0.76 — high
  confidence — because metformin co-occurs with real MASH treatments in several
  retrieved abstracts, so the embedding similarity looks high even though none of
  those abstracts state a metformin dosage. The agent generated a low-faithfulness-risk
  but ultimately unhelpful answer ("there is no information about dosage in this
  context") rather than a clean refusal. This is a genuine limitation of using
  embedding similarity alone as the confidence signal — topically-adjacent content
  can pass the CRAG gate without actually answering the question.
- Latency is higher during evaluation (avg 6.9s, p95 12.4s) than in standalone
  interactive use (~1-2s) because every non-refused query in this run pays for an
  extra full-context LLM-judge call on top of generation — that's an eval-harness
  cost, not something a production caller of `run_agent()` incurs.
- Groq's free tier caps `llama-3.3-70b-versatile` at 100,000 tokens/day. Running
  this 25-query eval with a 70B judge on every answered query is close to that
  ceiling — `run_eval()` checkpoints progress to `eval_checkpoint.json` after every
  query specifically so a mid-run rate limit doesn't lose completed work.

## Known limitations

- Corpus is 100 abstracts on a single disease area (MASLD/MASH) — retrieval quality
  claims don't generalize beyond this domain without re-indexing.
- CRAG grading uses cosine similarity as a proxy for relevance; it doesn't catch
  answers that are on-topic but factually contradicted by the source. Confirmed in
  eval: a query about metformin dosage (not in the corpus) scored high CRAG
  confidence because metformin co-occurs with real MASH treatments in several
  retrieved abstracts — topical adjacency passed the gate even though no retrieved
  abstract actually answered the question.
- The FastAPI job store is an in-memory dict — fine for a single-process demo,
  not for multi-worker deployment (would need Redis, per the code comments in
  `api/main.py`).
- `db/database.py` defaults to SQLite; switching to Postgres is a one-line change
  to `DATABASE_URL` but hasn't been load-tested.
