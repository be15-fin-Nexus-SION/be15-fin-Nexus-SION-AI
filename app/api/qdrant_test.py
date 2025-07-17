from fastapi import APIRouter
from qdrant_client import QdrantClient
import os

router = APIRouter()

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT"))

qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

@router.get("/ping-qdrant")
def ping_qdrant():
    try:
        result = qdrant_client.get_collections()
        return {"status": "success", "collections": result}
    except Exception as e:
        return {"status": "fail", "error": str(e)}
