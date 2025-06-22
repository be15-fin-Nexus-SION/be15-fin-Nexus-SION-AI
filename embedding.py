from dotenv import load_dotenv
import os
import json

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
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

embedding = OpenAIEmbeddings()
client = QdrantClient(host="localhost", port=6333)

vectorstore = Qdrant.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="fp_examples",
    client=client
)

print("[✅ Qdrant 임베딩 업로드 완료]")
