from dotenv import load_dotenv
import json

from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient

load_dotenv()

# JSON 로딩 및 검증
with open("app/data/fp_examples.json", "r", encoding="utf-8") as f:
    raw_examples = json.load(f)

examples = []
for ex in raw_examples:
    try:
        examples.append(Document(
            page_content = f"{ex['function_name']} - {ex['description']}",
            metadata={
                "function_name": ex["function_name"],
                "fp_type": ex["fp_type"],
                "complexity": ex["complexity"],
                "estimated_det": ex.get("estimated_det"),
                "estimated_ftr": ex.get("estimated_ftr")
            }
        ))
    except KeyError as e:
        print(f"⚠ 예제 스킵 - 필드 누락: {e}, 항목: {ex}")
        continue

# 임베딩 설정
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Qdrant 연결 및 컬렉션 재생성
client = QdrantClient(host="localhost", port=6333)
collection_name = "fp_examples"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config={
        "size": 384,
        "distance": "Cosine"
    }
)

vectorstore = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embedding
)
vectorstore.add_documents(examples)

print("[Qdrant 임베딩 업로드 완료]")
