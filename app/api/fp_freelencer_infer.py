from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.response import FreelencerFpInferResponse, FunctionScore
from app.services.ocr_extractor import extract_function_blocks_from_pdf
import logging
from app.services.fp_inference_service import run_single_fp_inference
import asyncio

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()  # root logger

@router.post("/fp-freelancer-infer", response_model=FreelencerFpInferResponse)
async def infer_freelancer_fp(file: UploadFile = File(...)):
    logger.info("🔥 [1] 라우터 진입 성공")

    try:
        logger.info(f"[START] 프리랜서 FP 추론 요청 수신")
        content = await file.read()
        logger.info(f"PDF 파일 수신 완료 - 바이트 크기: {len(content)}")

        function_blocks = extract_function_blocks_from_pdf(content)
        logger.info(f"OCR 기반 기능 블록 추출 완료 - 총 {len(function_blocks)}개")

        tasks = [
            run_single_fp_inference(block["function_name"], block["description"])
            for block in function_blocks
        ]

        inference_results = await asyncio.gather(*tasks)

        functions = [
            FunctionScore(
                functionName=block["function_name"],
                description=block["description"],
                fpType=result["fpType"],
                complexity=result["complexity"],
                det=result["det"],
                ftrOrRet=result["ftrOrRet"],
                stacks=block["stacks"]
            )
            for block, result in zip(function_blocks, inference_results)
        ]

        return FreelencerFpInferResponse(functions=functions)

    except Exception as e:
        logger.exception("프리랜서 이력서 추론 실패")
        raise HTTPException(status_code=500, detail=f"프리랜서 이력서 추론 실패: {str(e)}")