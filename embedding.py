from dotenv import load_dotenv
import os
import json

from langchain_community.vectorstores import Qdrant
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient

load_dotenv()

with open("fp_examples.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

docs = [
    Document(
        page_content=ex["description"],
        metadata={
            "function_name": ex["function_name"],
            "fp_type": ex["fp_type"],
            "complexity": ex["complexity"]
        }
    )
    for ex in examples
]

# embedding = OpenAIEmbeddings()
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Qdrant Client 수동 연결
client = QdrantClient(host="localhost", port=6333)

# 컬렉션 삭제 후 재생성 → 항상 최신 상태 보장
client.recreate_collection(
    collection_name="fp_examples",
    vectors_config={
        "size": 384,         # 모델 벡터 차원수 (all-MiniLM-L6-v2는 384차원)
        "distance": "Cosine"
    }
)

vectorstore = Qdrant(
    client=client,
    collection_name="fp_examples",
    embeddings=embedding
)
vectorstore.add_documents(docs)

print("[Qdrant 임베딩 업로드 완료]")
