from fastapi import APIRouter, HTTPException

from app.models.request import FPInferRequest
from app.models.response import FPInferResponse
from app.models.schema import OCRText, FPResult
from app.services.fp_inference_service import run_fp_inference
from app.services.fp_scoring import calculate_fp_score

router = APIRouter()

@router.post("/fp-infer", response_model=FPInferResponse)
async def fp_infer(payload: FPInferRequest):
    try:
        all_results = []

        # 1. 각 OCR 문장별 LangChain 추론 수행
        for item in payload.ocr_items:
            if item.text.strip():
                result = await run_fp_inference(item.text.strip())
                all_results.extend(result)

        # 2. FP 점수 계산
        print("OCR 문장별 LangChain 추론 수행완료!")
        print(all_results)
        total_score, scored_functions = calculate_fp_score(all_results)

        return {
            "project_id": payload.project_id,
            "functions": scored_functions,
            "total_fp_score": total_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FP 추론 실패: {str(e)}")
