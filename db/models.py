import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, Text, DateTime
from db.database import Base


class EvalResult(Base):
    __tablename__ = "eval_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    query = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=True)

    crag_score = Column(Float, nullable=True)
    step_count = Column(Integer, nullable=True)
    attempts = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)

    final_answer = Column(Text, nullable=True)
    sources = Column(Text, nullable=True)  # JSON-encoded list of PMIDs
    refusal_reason = Column(String(100), nullable=True)

    feedback = Column(String(10), nullable=True)  # 'up', 'down', or NULL

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<EvalResult id={self.id[:8]} crag={self.crag_score} feedback={self.feedback}>"
