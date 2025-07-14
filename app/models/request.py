# models/request.py

from pydantic import BaseModel
from typing import List, Optional

class OCRItem(BaseModel):
    description: Optional[str] = None  # 선택: "권한 제어 기능" 등
    text: str         # 실제 기능 명세 텍스트

class FPInferRequest(BaseModel):
    project_id: int
    ocr_items: List[OCRItem]

class FPVectorItem(BaseModel):
    function_name: str
    description: str
    fp_type: str
    complexity: str
    det: int
    ftr: int