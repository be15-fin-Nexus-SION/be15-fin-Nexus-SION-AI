# models/response.py

from pydantic import BaseModel
from typing import List, Literal

class FPFunctionScore(BaseModel):
    function_name: str
    description: str
    fp_type: Literal["EI", "EO", "EQ", "ILF", "EIF"]
    complexity: Literal["SIMPLE", "MEDIUM", "COMPLEX"]
    estimated_ftr: int
    estimated_det: int
    score: int

class FPInferResponse(BaseModel):
    project_id: int
    functions: List[FPFunctionScore]
    total_fp_score: int
