from typing import List, Optional
from pydantic import BaseModel


class EstimateResponse(BaseModel):
    # ok | no_face | bad_quality | challenge | uncertain | rejected
    status: str
    live: Optional[bool] = None
    liveness_score: Optional[float] = None
    age: Optional[float] = None
    age_range: Optional[List[float]] = None
    confidence: Optional[float] = None
    p_over_threshold: Optional[float] = None
    threshold: Optional[int] = None
    challenge: Optional[str] = None      # blink | turn_left | turn_right
    reasons: List[str] = []
    latency_ms: float = 0.0
