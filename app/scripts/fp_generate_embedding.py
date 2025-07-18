import os
import json
from dotenv import load_dotenv
from app.services.ocr_extractor import extract_function_sentences_from_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, Document
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from tqdm import tqdm
from app.utils.parser import postprocess_llm_output

load_dotenv()

DATA_DIR = "app/data/requirements"
PDF_FILES = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

llm = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash-lite-preview-06-17",
    temperature=0.8,
    max_output_tokens=2048,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt_template = """
주의: 당신은 절대 <thinking> 모드 또는 하이브리드 추론을 사용하지 마세요. Thinking 기능은 비활성화되었으며 즉시 결과만 반환해야 합니다.

당신은 시스템 분석 및 기능 점수화(Function Point Analysis) 전문가입니다.

아래는 시스템 요구사항 문서에서 추출한 문장과, 관련 문맥(근거)입니다.
이 정보를 기반으로 기능 단위로 분해하고, 각 기능에 대해 IFPUG 기준에 따라 출력 형식에 맞춰 추론해주세요. 
형식 예시를 참고하여 출력 형식에 맞춰 작성해주세요. 

[시스템 요구사항 문서에서 추출한 문장]
{question}

[문맥]
{context}

출력 형식:
[
  {{
    "function_name": "기능명 (간결하게)",
    "description": "기능 설명",
    "fp_type": "EI | EO | EQ | ILF | EIF 중 하나",
    "complexity": "SIMPLE | MEDIUM | COMPLEX 중 하나",
    "estimated_ftr": 추정 FTR 수 (정수, 반드시 숫자),
    "estimated_det": 추정 DET 수 (정수, 반드시 숫자),
  }}
]
 
"""

async def process_and_upload_to_qdrant():
    parsed_fp_results = []

    print(f"📄 처리할 PDF 파일 목록: {PDF_FILES}")

    for file in tqdm(PDF_FILES, desc="PDF 처리 중"):
        print(f"➡️ 현재 파일 처리 시작: {file}")
        with open(os.path.join(DATA_DIR, file), "rb") as f:
            pdf_bytes = f.read()
        print(f"✅ PDF 파일 읽기 완료: {file}")

        sentences = await extract_function_sentences_from_pdf(pdf_bytes)
        print(f"✅ 문장 추출 완료: 총 {len(sentences)} 문장")

        for idx, s in enumerate(tqdm(sentences, desc=f"→ {file} 문장 처리 중", leave=False)):
            try:
                print(f"🟣 문장 {idx+1}/{len(sentences)} 처리 중: {s[:30]}...")

                user_input = prompt_template.format(question=s, context="")
                res = llm.invoke([HumanMessage(content=user_input)])
                print(f"✅ LLM 호출 완료")

                results = postprocess_llm_output(res.content)
                print(f"✅ LLM 응답 파싱 완료")

                if not results:
                    print(f"⚠ JSON 파싱 실패: {s}\n응답 원문:\n{res.content}")
                    continue

                json_obj = results[0]
                json_obj["original_sentence"] = s
                parsed_fp_results.append(json_obj)

            except Exception as e:
                print(f"❌ 문장 처리 중 에러 발생: {e}")
                continue

    print(f"✅ 전체 문장 처리 완료, 총 {len(parsed_fp_results)} 건 결과 저장 시작")

    os.makedirs("app/data", exist_ok=True)
    with open("app/data/fp_generated_dataset.json", "w", encoding="utf-8") as f:
        json.dump(parsed_fp_results, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 저장 완료")

    documents = [
        Document(
            page_content=ex["description"],
            metadata={
                "function_name": ex["function_name"],
                "fp_type": ex["fp_type"],
                "complexity": ex["complexity"],
                "estimated_det": int(ex["estimated_det"]),
                "estimated_ftr": int(ex["estimated_ftr"]),
            }
        )
        for ex in parsed_fp_results
    ]
    print(f"✅ Document 변환 완료: 총 {len(documents)}건")

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"✅ Embedding 모델 로드 완료")

    client = QdrantClient(host="localhost", port=6333)
    print(f"✅ Qdrant 클라이언트 생성 완료")

    client.recreate_collection(
        collection_name="fp_examples",
        vectors_config={"size": 384, "distance": "Cosine"}
    )
    print(f"✅ Qdrant 컬렉션 초기화 완료")

    vectorstore = Qdrant(
        client=client,
        collection_name="fp_examples",
        embeddings=embedding
    )
    vectorstore.add_documents(documents)
    print(f"✅ Qdrant 업로드 완료 - 총 {len(documents)}건 완료됨")

