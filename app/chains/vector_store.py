from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import os

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

qdrant_host = os.getenv("QDRANT_HOST", "qdrant.fastapi.svc.cluster.local")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))


client = QdrantClient(host="qdrant-container", port=6333)

vectorstore = Qdrant(
    client=client,
    collection_name="fp_examples",
    embeddings=embedding
)

