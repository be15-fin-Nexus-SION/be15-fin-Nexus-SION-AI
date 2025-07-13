import os
import json
from dotenv import load_dotenv
from app.services.ocr_extractor import extract_function_sentences_from_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient

from tqdm import tqdm

from app.utils.parser import postprocess_llm_output

load_dotenv()

DATA_DIR = "app/data/requirements"
PDF_FILES = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    max_output_tokens=2048,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt_template = """
You are an expert in Function Point Analysis (ISO/IEC 20926: IFPUG).
Given the following Korean functional requirement sentence, output a JSON object with:

- function_name (짧은 이름)
- description (요구사항 문장 그대로 또는 자연스러운 설명)
- fp_type (EI, EO, EQ, ILF, EIF 중 하나)
- estimated_det (1 이상 정수)
- estimated_ftr (1 이상 정수)
- complexity (SIMPLE, MEDIUM, COMPLEX 중 하나)

Sentence:
"{sentence}"

Return JSON only. Do not include explanation.
"""

parsed_fp_results = []

for file in tqdm(PDF_FILES, desc="PDF 처리 중"):
    with open(os.path.join(DATA_DIR, file), "rb") as f:
        pdf_bytes = f.read()

    # OCR → 기능 문장 추출
    sentences = extract_function_sentences_from_pdf(pdf_bytes)

    for s in tqdm(sentences, desc=f"→ {file} 문장 처리 중", leave=False):
        try:
            user_input = prompt_template.format(sentence=s)
            res = llm.invoke([HumanMessage(content=user_input)])
            results = postprocess_llm_output(res.content)

            if not results:
                print(f"⚠ JSON 파싱 실패: {s}\n응답 원문:\n{res.content}")
                continue

            json_obj = results[0]
            json_obj["original_sentence"] = s
            parsed_fp_results.append(json_obj)

        except Exception as e:
            print(f"⚠ 문장 처리 실패: {s}\n사유: {e}")
            continue

# 결과 저장
with open("app/data/fp_generated_dataset.json", "w", encoding="utf-8") as f:
    json.dump(parsed_fp_results, f, ensure_ascii=False, indent=2)

# Qdrant에 업로드
documents = []
for ex in parsed_fp_results:
    documents.append(Document(
        page_content=ex["description"],
        metadata={
            "function_name": ex["function_name"],
            "fp_type": ex["fp_type"],
            "complexity": ex["complexity"],
            "estimated_det": int(ex["estimated_det"]),
            "estimated_ftr": int(ex["estimated_ftr"]),
            "original_sentence": ex.get("original_sentence", "")
        }
    ))

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)

client.recreate_collection(
    collection_name="fp_examples",
    vectors_config={"size": 384, "distance": "Cosine"}
)

vectorstore = Qdrant(
    client=client,
    collection_name="fp_examples",
    embeddings=embedding
)
vectorstore.add_documents(documents)

print(f"Qdrant 업로드 완료 - 총 {len(documents)}건")
