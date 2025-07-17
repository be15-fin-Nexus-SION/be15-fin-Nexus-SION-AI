from app.core.config import app
from app.api.fp_infer import router as fp_router
from app.api.fp_embed import router as fp_embed_router
from app.api.fp_freelencer_infer import router as fp_freelancer_infer
from app.api.llm_test import router as llm_test_router
from app.api.qdrant_test import router as qdrant_test_router

import logging
import sys
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
logger.info("FastAPI 서버가 main.py에서 정상 실행 중입니다.")

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI 서버 시작됨, Qdrant 초기화 시작")
    process_and_upload_to_qdrant()
    logger.info("Qdrant 초기화 완료 후 서버 구동 준비 완료")

app.include_router(fp_router)
app.include_router(fp_embed_router)
app.include_router(fp_freelancer_infer)
app.include_router(llm_test_router)
app.include_router(qdrant_test_router)

