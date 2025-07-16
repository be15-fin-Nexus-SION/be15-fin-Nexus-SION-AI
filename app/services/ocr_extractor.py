from fastapi import HTTPException
from pdf2image import convert_from_bytes
import pytesseract
import logging
from typing import List, Dict
import re
from app.utils.preprocess_image import preprocess_image

import asyncio

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def is_meaningful(sentence: str) -> bool:
    skip_keywords = [
        "요구사항 분류", "표준 요구사항", "고유번호", "테스트 ID",
        "버전", "페이지", "참고", "비고", "목적", "개요", "정의"
    ]
    return (
            len(sentence.strip()) > 10
            and not any(k in sentence for k in skip_keywords)
    )

def is_fp_candidate(sentence: str) -> bool:
    fp_keywords = [
        "등록", "조회", "삭제", "입력", "수정", "출력", "생성", "검색"
        "연동", "저장", "업로드", "다운로드", "로그인"
    ]
    return any(k in sentence for k in fp_keywords)

async def ocr_page(image) -> str:
    custom_config = "--oem 3 --psm 4"
    # OCR 비동기 호출
    text = await asyncio.to_thread(pytesseract.image_to_string, image, "kor+eng", custom_config)
    return text

async def extract_function_sentences_from_pdf(content: bytes) -> List[str]:
    images = convert_from_bytes(content)
    logger.info(f"[OCR] PDF → 이미지 변환 완료 - 페이지 수: {len(images)}")

    if not images:
        raise ValueError("PDF에서 이미지를 추출할 수 없습니다.")

    tasks = [ocr_page(image) for image in images]
    ocr_results = await asyncio.gather(*tasks)

    ocr_texts = []

    for idx, text in enumerate(ocr_results):
        # custom_config = "--oem 3 --psm 4"
        # text = pytesseract.image_to_string(image, lang="kor+eng", config=custom_config)
        # logger.info(f"[OCR TEXT] 페이지 {idx + 1} OCR 결과:\n{text}")

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        grouped_blocks = []
        current_block = []

        for line in lines:
            cleaned = re.sub(r"^[^가-힣a-zA-Z]+", "", line).strip()

            if not cleaned:
                continue

            if current_block:
                grouped_blocks.append(" ".join(current_block))
                current_block = []

            current_block.append(cleaned)

        if current_block:
            grouped_blocks.append(" ".join(current_block))

        logger.info(f"[OCR] 페이지 {idx + 1} 기능 블록 수: {len(grouped_blocks)}")

        for block in grouped_blocks:
            if is_meaningful(block) and is_fp_candidate(block):
                ocr_texts.append(block)

    logger.info(f"[OCR] 최종 기능 후보 문장 수: {len(ocr_texts)}")
    return ocr_texts

def extract_function_blocks_from_pdf(content: bytes) -> List[Dict[str, str]]:
    images = convert_from_bytes(content, dpi=400)
    logger.info(f"[OCR] PDF → 이미지 변환 완료 - 페이지 수: {len(images)}")

    if not images:
        raise HTTPException(status_code=400, detail="PDF에서 이미지를 추출할 수 없습니다.")

    function_blocks = []

    for idx, image in enumerate(images):
        image = preprocess_image(image)

        custom_config = r'--oem 3 --psm 4 -l kor+eng'
        text = pytesseract.image_to_string(image, config=custom_config)

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        current_function = {
            "function_name": "",
            "description": "",
            "stacks": []
        }
        state = None

        for line in lines:
            if line.startswith("기능명:"):
                # 기존 block 저장
                if all(current_function.values()):
                    function_blocks.append(current_function.copy())
                    current_function = {"function_name": "", "description": "", "stacks": []}

                current_function["function_name"] = line.replace("기능명:", "").strip()
                state = 'name'

            elif line.startswith("기능설명:"):
                current_function["description"] = line.replace("기능설명:", "").strip()
                state = 'desc'

            elif line.startswith("사용한 기술스택:"):
                stack_line = line.replace("사용한 기술스택:", "").strip()
                current_function["stacks"] = [s.strip() for s in re.split(r"[,\n]", stack_line) if s.strip()]
                state = 'stack'

            else:
                if state == 'desc':
                    current_function["description"] += " " + line.strip()
                elif state == 'stack':
                    current_function["stacks"].extend([s.strip() for s in re.split(r"[,\n]", line) if s.strip()])

        # 마지막 블록 저장
        if all(current_function.values()):
            function_blocks.append(current_function.copy())

        logger.info(f"[OCR] 페이지 {idx + 1} 기능 블록 수: {len(function_blocks)}")

    logger.info(f"[OCR] 최종 기능 블록 수: {len(function_blocks)}")
    return function_blocks
