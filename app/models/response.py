from pydantic import BaseModel
from typing import List, Optional

class FPFunction(BaseModel):
    function_name: str
    description: str
    fp_type: str
    complexity: str
    estimated_det: Optional[int] = None
    estimated_ftr: Optional[int] = None
    score: int


class FPInferResponse(BaseModel):
    project_id: str
    functions: List[FPFunction]
    total_fp_score: int

class FunctionScore(BaseModel):
    functionName: str
    description: str
    fpType: str
    complexity: str
    det: int
    ftrOrRet: int
    stacks: List[str]

class FreelencerFpInferResponse(BaseModel):
    functions: List[FunctionScore]

