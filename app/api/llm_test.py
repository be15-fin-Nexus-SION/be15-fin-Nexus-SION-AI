from fastapi import APIRouter, HTTPException
import asyncio
from app.chains.llm import llm

import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/test-llm")
async def test_llm():
    try:
        result = await asyncio.to_thread(llm.invoke, "테스트 입력")
        return {"result": str(result)}
    except Exception as e:
        logger.exception("LLM 호출 실패")
        raise HTTPException(500, detail=str(e))