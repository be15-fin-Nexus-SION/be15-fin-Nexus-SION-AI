from fastapi import APIRouter, HTTPException
from app.models.request import FPVectorItem
from typing import List
from langchain.schema import Document
from app.chains.vector_store import vectorstore
import logging

router = APIRouter()

logger = logging.getLogger(__name__)

# FastAPI 라우터 정의
@router.post("/fp-embed")
async def embed_fp_vectors(items: List[FPVectorItem]):
    try:
        documents = []

        for item in items:
            documents.append(Document(
                page_content=item.description,
                metadata={
                    "function_name": item.function_name,
                    "fp_type": item.fp_type,
                    "complexity": item.complexity,
                    "estimated_det": item.det,
                    "estimated_ftr": item.ftr,
                }
            ))

        logger.info(f"[VECTOR UPLOAD] 총 {len(documents)}건 벡터 저장 시도 중...")
        vectorstore.add_documents(documents)
        logger.info(f"[VECTOR UPLOAD] 벡터 저장 성공")

        return {"success": True}

    except Exception as e:
        logger.error(f"[VECTOR UPLOAD ERROR] {e}")
        raise HTTPException(status_code=500, detail="벡터 업로드 중 오류 발생")
