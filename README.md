# 🧬 Clinical Research Copilot

**An agentic RAG system that knows when it doesn't know.**

Ask it about MASLD/MASH liver disease research and it'll dig through 100 PubMed
abstracts, grade its own retrieval quality, retry with a better query if the
first pass was weak, and — this is the part most RAG demos skip — refuse to
answer if it can't back up a claim with a real citation.

Built with LangGraph, hybrid search (BM25 + dense embeddings), and a
self-correcting retrieval loop (CRAG). No OpenAI. Runs entirely on Groq's
free tier.

```
"What is the primary mechanism of resmetirom?"
  → The primary mechanism of resmetirom is as a selective thyroid hormone
    receptor-β (THR-β) agonist [PMID 38771485]. It inhibits intestinal lipid
    absorption via remodeling bile acid profiles [PMID 38789494]...

"What's the recommended dosage of metformin for MASH?"
  → I don't have enough reliable evidence to answer this safely.
```

---

## The idea

Most RAG tutorials retrieve once, stuff it in a prompt, and hope. That's fine
for a demo, terrible for anything clinical — a confident wrong answer is
worse than no answer.

This system treats retrieval as something to *verify*, not assume:

1. **Retrieve** with hybrid search — BM25 catches exact terms, dense
   embeddings catch semantic matches, Reciprocal Rank Fusion merges both.
2. **Grade** the retrieval with a cosine-similarity confidence score.
3. **Route** based on that score:
   - High confidence → answer, with citations.
   - Medium confidence → rewrite the query and try again (up to 2 times).
   - Low confidence → **refuse**, honestly, instead of guessing.

It's a state machine, not a chain — built as a LangGraph graph with real
loops, real safety gates (step budget, wall-clock timeout), and a circuit
breaker in front of every LLM call so a flaky API doesn't take the whole
thing down.

## How it flows

```
                         ┌─────────────┐
                         │   classify  │  heuristic tag, zero LLM cost
                         └──────┬──────┘
                                ▼
                         ┌─────────────┐
              ┌─────────▶│   retrieve  │  BM25 + BGE-small, fused with RRF
              │          └──────┬──────┘
              │                 ▼
              │          ┌─────────────┐
              │          │    grade    │  cosine similarity, no LLM call
              │          └──────┬──────┘
              │                 ▼
              │            score ≥ 0.70 ──────────────► generate ──► ✅ answer + citations
              │                 │
              │            0.35–0.70, retries left
              └──── rewrite ◀───┤
                                 │
                            0.35–0.70, retries exhausted ──► generate (with caveat)
                                 │
                            score < 0.35 ──────────────────► refuse ──► 🚫 "I don't know"
```

Every path through the graph is bounded: a 30-second wall-clock timeout and
a 6-step budget both force a refusal if something loops longer than it
should. No infinite retries, no runaway cost.

## Decisions worth explaining

**Rank fusion, not score fusion.** BM25 scores live on a 0–40 scale, cosine
similarity lives on 0–1. Averaging them directly is meaningless. RRF sidesteps
that by merging on *rank position* instead of raw score — no normalization
hacks needed.

**The grader is math, not another LLM call.** Every retrieval — including
every retry inside the rewrite loop — gets graded. Making that an LLM call
would multiply cost and latency for a signal a cheap embedding comparison
already gives for free.

**The API returns before the work is done.** `POST /query` hands back a
`job_id` in under 100ms and runs the agent as a background task; the client
polls `GET /result/{job_id}`. A single run can take several seconds — an HTTP
server should never hold a connection open that long under real traffic.

**The eval judge deliberately uses the expensive model.** More on this below
— it's the one part of this project that broke in an interesting way.

## What's in here

```
agent/              LangGraph state machine — state.py, nodes.py, graph.py
api/                FastAPI layer: async job submission, polling, feedback
db/                 SQLAlchemy models + repository functions (SQLite)
resilience/         Circuit breaker wrapping every LLM call
tests/              pytest — fast unit tests + slow real-API/real-model tests
retriever.py        Hybrid BM25 + dense retrieval, RRF fusion
corpus.py           Loads the PubMed abstracts into a clean shape
config.py           One file, every tunable constant
evals.py            Eval harness — keyword/citation metrics + LLM-as-judge
eval_dataset.json   25 hand-written queries: factoid, multi-hop, vague, refusal
run_evals.py        Runs the eval suite end-to-end, writes eval_report.json
```

## Get it running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # drop in your GROQ_API_KEY — it's free
```

**Ask it something:**
```bash
python -c "from agent import run_agent; print(run_agent('What is the primary mechanism of resmetirom?'))"
```

**Run it as an API:**
```bash
uvicorn api.main:app --reload
```
```
POST /query              {"query": "..."}       → {"job_id": "..."}
GET  /result/{job_id}                            → status + answer + sources + crag_score
POST /feedback/{run_id}  {"rating": "up"|"down"} → thumbs up/down on a past run
GET  /health
```

**Run the tests:**
```bash
pytest -m "not slow"   # fast — no models, no API calls
pytest                 # everything, including real Groq calls
```

**Run the evals:**
```bash
python run_evals.py    # writes eval_report.json
```

## Does it actually work? Here's the receipt.

25 queries, run against the real agent, real Groq calls, no cherry-picking.
Split across four categories: **factoid** (direct lookups), **multi-hop**
(compare across abstracts), **vague** (1–2 word queries), and **refusal**
(deliberately unanswerable — "what's the capital of France" run through a
liver-disease RAG system, on purpose).

Scored on keyword accuracy, citation recall, and two LLM-as-judge metrics —
**faithfulness** (is every claim actually backed by the retrieved text?) and
**relevancy** (does the answer address what was asked?). Same ideas the
`ragas` library measures, implemented directly against Groq so the project
doesn't need to drag in an OpenAI-shaped dependency tree just to grade itself.

|  |  |
|---|---|
| Queries evaluated | 25 |
| Answer rate | **72%** (18/25) |
| Keyword accuracy | 72.9% |
| Citation recall | 78.7% |
| Faithfulness *(on answered queries)* | **1.00** |
| Relevancy *(on answered queries)* | 0.97 |
| Refusal accuracy *(3 deliberate traps)* | 2/3 |
| Latency (avg / p95) | 6.9s / 12.4s |

Full per-query breakdown lives in [`eval_report.json`](eval_report.json).

**The honest read of these numbers:**

✅ **When it answers, it's not making things up.** Faithfulness and relevancy
are both essentially perfect on the 18 queries the system chose to answer.
The self-grading loop is doing its job.

🤔 **28% refusal rate is mostly caution, not failure.** A handful of
medium-difficulty queries scored just under the 0.70 confidence bar and got
refused instead of answered-with-a-caveat, even though the corpus had
relevant material. That's a threshold tuning knob, not a broken retriever.

🐛 **The one real bug I found:** one of the three deliberate "this should be
refused" traps — a question about metformin dosage, a drug that's genuinely
not in this corpus — slipped past the confidence gate at 0.76. Why? Metformin
shows up *alongside* real MASH drugs in several retrieved abstracts, so the
embedding similarity looks high even though none of those abstracts state a
dosage. The agent didn't hallucinate a number (faithfulness held), but it
also didn't cleanly refuse — it hedged. That's a real limitation of using
cosine similarity alone as a confidence signal: topical adjacency can sneak
past a gate built to catch topical *irrelevance*.

⏱️ **Latency during eval (6.9s avg) is higher than standalone use (~1–2s)**
because every judged query here pays for an extra full-context LLM call on
top of generation. That's an eval-harness cost, not a production one.

🔧 **One bug I had to catch before trusting any of this:** the first version
of the eval judge used the small, fast model (`llama-3.1-8b-instant`) to
score faithfulness. It returned near-zero scores across the board — on
answers that were clearly, verifiably correct. Turned out the 8B model
couldn't reliably track claims across ~7.5k characters of context; it missed
support that was sitting right there in a different abstract. Swapping the
judge to the 70B model fixed it instantly. Lesson: a judge model needs to be
at least as capable as the thing it's judging, especially at longer context
lengths — and you should never trust an eval number without sanity-checking
the judge itself.

📉 **Also ran into Groq's free-tier ceiling** (100K tokens/day on the 70B
model) mid-eval, more than once — running a 70B judge on every answered query
adds up fast. Fixed by making the eval harness resumable: it checkpoints
after every single query to `eval_checkpoint.json`, so a rate limit mid-run
costs a few minutes, not the whole run.

## Where this would break in production

- **The corpus is 100 abstracts on one disease.** Everything above is true
  for this corpus; none of it is a claim about generalizing to a bigger,
  messier one without re-indexing and re-evaluating.
- **Cosine similarity is a relevance proxy, not a correctness check** — see
  the metformin example above. It catches "this is off-topic," not "this is
  topically-adjacent but doesn't actually answer the question."
- **The job store is an in-memory dict.** Fine for one process, wrong for
  anything with more than one worker — that's a Redis swap away, not built.
- **SQLite by default.** Switching to Postgres is a one-line `DATABASE_URL`
  change in `db/database.py` — untested at any real concurrency.
