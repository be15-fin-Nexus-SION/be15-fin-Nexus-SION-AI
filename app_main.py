from dotenv import load_dotenv
import os
import re
import json

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI

from prompt_builder import prompt_template
from prompt_fallback import prompt_fallback  # fallback 프롬프트 분리된 파일

# ---------------------- 환경 변수 로드 ----------------------
load_dotenv()
app = FastAPI()

# ---------------------- 입력/출력 모델 ----------------------
class OCRText(BaseModel):
    ocr_text: str
    request_id: str


class FPResult(BaseModel):
    function_name: str
    description: str
    fp_type: str
    complexity: str
    estimated_det: Optional[int] = None
    estimated_ftr: Optional[int] = None

# ---------------------- 벡터 DB 및 LLM 설정 ----------------------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

vectorstore = Qdrant(
    client=client,
    collection_name="fp_examples",
    embeddings=embedding
)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.5,  # 너무 낮으면 unrelated도 포함됨
        "k": 3
    }
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_output_tokens=2048,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

fallback_chain = LLMChain(
    llm=llm,
    prompt=prompt_fallback
)

# ---------------------- FP 추론 API ----------------------
@app.post("/fp-infer", response_model=List[FPResult])
def fp_infer(payload: OCRText):
    ocr_text = payload.ocr_text.strip()
    raw_output = None

    try:
        # 벡터 검색 결과 확인
        search_results = retriever.invoke(ocr_text)
        use_fallback = not search_results or len(search_results) == 0
        print("검색 결과 문서 수:", len(search_results))

        # RAG or fallback 선택 실행
        if use_fallback:
            print("벡터 검색 결과 없음 → fallback 프롬프트 사용")
            raw_output = fallback_chain.invoke({"question": ocr_text})
        else:
            print("벡터 검색 성공 → RAG 실행")
            raw_output = rag_chain.invoke({"query": ocr_text})
            print(raw_output)

            # LLM 응답이 비어있는 경우 fallback 재시도
            result_text = extract_result_text(raw_output)
            print(result_text)
            if not result_text.strip():
                print("RAG 응답 없음 → fallback 실행")
                raw_output = fallback_chain.invoke({"question": ocr_text})

        print("LLM 응답 원문:\n", raw_output)
        result_text = extract_result_text(raw_output)
        parsed = postprocess_llm_output(result_text)

        result = []
        for item in parsed:
            try:
                result.append(FPResult(
                    function_name=item.get("function_name", "").strip(),
                    description=item.get("description", "").strip(),
                    fp_type=item.get("fp_type", "").strip(),
                    complexity=item.get("complexity", "").strip(),
                    estimated_det=int(item.get("estimated_det", 0)) or None,
                    estimated_ftr=int(item.get("estimated_ftr", 0)) or None,
                ))
            except Exception as e:
                print("항목 파싱 실패:", item, "사유:", e)
                continue

        return result

    except Exception as e:
        print("예외 발생:", e)
        return JSONResponse(status_code=500, content={
            "message": "GPT 응답 처리 중 오류 발생",
            "raw_output": str(raw_output) if raw_output else None,
            "error": str(e)
        })


# ---------------------- 헬퍼 함수 ----------------------

def extract_result_text(raw_output) -> str:
    if isinstance(raw_output, dict):
        return raw_output.get("result", "")
    elif isinstance(raw_output, str):
        return raw_output
    else:
        return str(raw_output)


def postprocess_llm_output(text: str) -> List[dict]:
    try:
        # 1. 깨진 따옴표 수정
        cleaned = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

        # 2. ```json 코드블럭 제거
        cleaned = re.sub(r"```json|```", "", cleaned)

        # 3. 복수 JSON 객체 처리
        matches = re.findall(r"{[^{}]*}", cleaned)
        parsed = [json.loads(m) for m in matches if m.strip()]
        return parsed

    except Exception as e:
        print("JSON 파싱 오류:", e)
        return []

