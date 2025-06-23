from dotenv import load_dotenv
import os
import re

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

from qdrant_client import QdrantClient
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json

from prompt_builder import prompt_template

load_dotenv()

app = FastAPI()

class OCRText(BaseModel):
    ocr_text: str
    request_id: str

class FPResult(BaseModel):
    function_name: str
    description: str
    fp_type: str
    complexity: str

# 벡터 DB + 임베딩 설정
# embedding = OpenAIEmbeddings()
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)
vectorstore = Qdrant(client=client, collection_name="fp_examples", embeddings=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG 체인
# llm = ChatOpenAI(model="gpt-4", temperature=0)

# HuggingFaceHub 기반 LLM (예: google/flan-t5-base 등)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.3,
    max_new_tokens=256,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

@app.post("/fp-infer", response_model=List[FPResult])
def fp_infer(payload: OCRText):
    ocr_text = payload.ocr_text.strip()

    # RAG 체인 실행
    raw_output = rag_chain.invoke({"query": ocr_text})
    print("GPT 응답 원문:\n", raw_output)

    # JSON 배열 추출 (정규표현식으로)
    try:
        result_text = raw_output["result"]
        json_block = extract_json_array(result_text)
        parsed = json.loads(json_block)
    except Exception as e:
        print("⚠JSON 파싱 실패:\n", result_text)
        raise HTTPException(status_code=500, detail="GPT 응답 파싱 실패")

    # 항목 검증 및 유연 처리
    result = []
    for item in parsed:
        try:
            result.append(FPResult(
                function_name=item.get("function_name", "").strip(),
                description=item.get("description", "").strip(),
                fp_type=item.get("fp_type", "").strip(),
                complexity=item.get("complexity", "").strip()
            ))
        except Exception as e:
            print("⚠항목 무시 (스킵):", item)
            continue

    return result

def extract_json_array(text: str) -> str:
    """
    텍스트에서 JSON 배열 형태 ([{...}, {...}])만 추출
    """
    match = re.search(r"\[\s*{.*?}\s*\]", text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("JSON 배열 형식을 찾을 수 없습니다.")
