from app.core.config import app
from app.api.fp_infer import router as fp_router
from app.api.fp_embed import router as fp_embed_router
from app.api.fp_freelencer_infer import router as fp_freelancer_infer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.include_router(fp_router)
app.include_router(fp_embed_router)
app.include_router(fp_freelancer_infer)

logger.info("✅ FastAPI 서버가 main.py에서 정상 실행 중입니다.")