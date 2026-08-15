from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict


class CopilotState(TypedDict):
    query: str
    query_type: Optional[str]
    query_attempts: List[str]

    retrieved_chunks: List[Dict[str, Any]]
    crag_score: float

    step_count: int
    started_at: float

    final_answer: Optional[str]
    refusal_reason: Optional[str]
    sources: List[str]
