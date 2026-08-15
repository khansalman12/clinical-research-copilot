"""
FastAPI accepts a query, returns a job_id immediately, and runs the agent as
a background task. A single agent run can take several seconds (embedding +
up to 2 LLM calls); the API must never hold an HTTP connection open that long.
The client polls GET /result/{job_id} until the job completes.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from db.database import engine, Base
from db.operations import save_run, update_feedback

Base.metadata.create_all(bind=engine)

# In-memory job store — a single-process demo stand-in for Redis.
jobs: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Server starting — DB tables created.")
    yield
    print("[API] Server shutting down.")


app = FastAPI(
    title="Clinical Research Copilot",
    description="Agentic RAG system for clinical research queries",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str


class FeedbackRequest(BaseModel):
    rating: str  # 'up' or 'down'


def run_agent_job(job_id: str, query: str):
    from agent import run_agent  # lazy import so the server starts fast

    jobs[job_id]["status"] = "processing"
    start = time.time()

    try:
        result = run_agent(query)
        latency_ms = int((time.time() - start) * 1000)

        run_id = save_run({**result, "query": query}, latency_ms)

        jobs[job_id].update({
            "status": "done",
            "run_id": run_id,
            "answer": result["answer"],
            "sources": result["sources"],
            "crag_score": result["crag_score"],
            "latency_ms": latency_ms,
        })

    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e)})


@app.post("/query")
def submit_query(request: QueryRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "run_id": None}

    background_tasks.add_task(run_agent_job, job_id, request.query)

    return {"job_id": job_id, "status": "queued"}


@app.get("/result/{job_id}")
def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] in ("queued", "processing"):
        return {"status": job["status"]}

    if job["status"] == "error":
        return {"status": "error", "error": job.get("error")}

    return {
        "status": "done",
        "run_id": job["run_id"],
        "answer": job["answer"],
        "sources": job["sources"],
        "crag_score": job["crag_score"],
        "latency_ms": job["latency_ms"],
    }


@app.post("/feedback/{run_id}")
def submit_feedback(run_id: str, request: FeedbackRequest):
    if request.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")

    updated = update_feedback(run_id, request.rating)

    if not updated:
        raise HTTPException(status_code=404, detail="Run not found")

    return {"status": "feedback recorded", "run_id": run_id, "rating": request.rating}


@app.get("/health")
def health():
    return {"status": "ok"}
