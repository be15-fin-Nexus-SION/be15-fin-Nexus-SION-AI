from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json

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
embedding = OpenAIEmbeddings()
client = QdrantClient(host="localhost", port=6333)
vectorstore = Qdrant(client=client, collection_name="fp_examples", embeddings=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 프롬프트 (RAG 용)
rag_prompt = PromptTemplate.from_template("""
당신은 시스템 분석 전문가입니다. 다음은 시스템 요구사항 문서에서 추출한 문장들입니다.
당신에게 제공된 문맥(근거)을 참고하여, 기능 단위로 설명을 구성해 주세요.

[문맥]
{context}

[추론 대상 문서]
{question}

출력 형식은 다음과 같습니다:
[
  {{
    "function_name": "기능명",
    "description": "기능 설명",
    "fp_type": "EI | EO | EQ | ILF | EIF",
    "complexity": "SIMPLE | MEDIUM | COMPLEX"
  }}
]
""")

# RAG 체인
llm = ChatOpenAI(model="gpt-4", temperature=0)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": rag_prompt}
)

@app.post("/fp-infer", response_model=List[FPResult])
def fp_infer(payload: OCRText):
    ocr_text = payload.ocr_text.strip()
    result = rag_chain.run(query=ocr_text)

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        raise ValueError(f"⚠️ GPT 응답 파싱 실패: \n{result}")
