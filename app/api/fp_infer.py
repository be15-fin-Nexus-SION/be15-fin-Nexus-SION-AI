from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.models.response import FPInferResponse
from app.services.fp_inference_service import run_fp_inference
from app.services.fp_scoring import calculate_fp_score
from pdf2image import convert_from_bytes
import pytesseract
import asyncio
import logging

router = APIRouter()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@router.post("/fp-infer", response_model=FPInferResponse)
async def fp_infer_with_pdf(
        project_id: str = Form(...),
        file: UploadFile = File(...)
):
    try:
        logger.info(f"[START] FP 추론 요청 수신 - project_id: {project_id}")
        content = await file.read()
        logger.info(f"PDF 파일 읽기 완료 - 바이트 크기: {len(content)}")
        images = convert_from_bytes(content)
        logger.info(f"PDF → 이미지 변환 완료 - 총 페이지 수: {len(images)}")

        if not images:
            raise HTTPException(status_code=400, detail="PDF에서 이미지를 추출할 수 없습니다.")

        ocr_texts = []
        for idx, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang="kor+eng")
            sentences = [line.strip() for line in text.split("\n") if line.strip()]
            ocr_texts.extend(sentences)
            logger.info(f"OCR 완료 - 페이지 {idx + 1}, 문장 수: {len(sentences)}")

        logger.info(f"OCR 전체 문장 수: {len(ocr_texts)}")

        semaphore = asyncio.Semaphore(5)

        # 병렬로 처리할 함수 정의 (idx 포함)
        async def safe_run(idx: int, sentence: str):
            async with semaphore:
                logger.info(f"[추론 시작] 문장 {idx + 1}/{len(ocr_texts)}: \"{sentence}\"")
                result = await run_fp_inference(sentence)
                logger.info(f"[추론 완료] 문장 {idx + 1}/{len(ocr_texts)}, 결과 수: {len(result)}")
                return result

        # 인덱스 포함 병렬 작업 생성
        tasks = [safe_run(idx, sentence) for idx, sentence in enumerate(ocr_texts)]
        results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        all_results = []
        for idx, (sentence, result) in enumerate(zip(ocr_texts, results_nested), start=1):
            if isinstance(result, Exception):
                logger.warning(f"FP 추론 실패 - 문장 {idx}/{len(ocr_texts)}, 예외: {result}")
                continue
            all_results.extend(result)
            logger.info(f"FP 추론 완료 - 문장 {idx}/{len(ocr_texts)}, 결과 수: {len(result)}")

        logger.info(f"FP 추론 전체 완료 - 총 결과 수: {len(all_results)}")

        total_score, scored_functions = calculate_fp_score(all_results)
        logger.info(f"FP 점수 계산 완료 - Total Score: {total_score}, Function Count: {len(scored_functions)}")

        logger.info(f"응답 반환 - project_id: {project_id}")


        return FPInferResponse(
            project_id=project_id,
            functions=scored_functions,
            total_fp_score=total_score
        )

    except Exception as e:
        logger.exception("OCR 및 FP 추론 실패")
        raise HTTPException(status_code=500, detail=f"OCR 및 FP 추론 실패: {str(e)}")
