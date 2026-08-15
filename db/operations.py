import json
from db.database import SessionLocal
from db.models import EvalResult


def save_run(result: dict, latency_ms: int) -> str:
    db = SessionLocal()
    try:
        row = EvalResult(
            query=result.get("query", ""),
            query_type=result.get("query_type"),
            crag_score=result.get("crag_score"),
            step_count=result.get("step_count"),
            attempts=result.get("attempts"),
            latency_ms=latency_ms,
            final_answer=result.get("answer"),
            sources=json.dumps(result.get("sources", [])),
            refusal_reason=result.get("refusal_reason"),
            feedback=None,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id
    finally:
        db.close()


def update_feedback(run_id: str, feedback: str) -> bool:
    db = SessionLocal()
    try:
        row = db.query(EvalResult).filter(EvalResult.id == run_id).first()
        if not row:
            return False
        row.feedback = feedback
        db.commit()
        return True
    finally:
        db.close()
