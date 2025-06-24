from app.core.config import app
from app.api.fp_infer import router as fp_router

app.include_router(fp_router)
