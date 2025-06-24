from fastapi import APIRouter
from app.models.schema import OCRText, FPResult
from app.services.fp_inference_service import run_fp_inference

router = APIRouter()

@router.post("/fp-infer", response_model=list[FPResult])
def fp_infer(payload: OCRText):
    return run_fp_inference(payload.ocr_text.strip())
