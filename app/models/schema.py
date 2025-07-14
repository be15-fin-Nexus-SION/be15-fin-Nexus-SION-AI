from pydantic import BaseModel
from typing import Optional, List

class OCRText(BaseModel):
    ocr_text: str
    request_id: str

class FPResult(BaseModel):
    function_name: str
    description: str
    fp_type: str
    complexity: str
    estimated_det: Optional[int] = None
    estimated_ftr: Optional[int] = None

class RawFunctionBlock(BaseModel):
    function_name: str
    description: str
    stacks: List[str]


