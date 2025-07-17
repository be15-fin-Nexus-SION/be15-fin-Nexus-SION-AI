import os
import json
from tqdm import tqdm
from dotenv import load_dotenv

from app.services.ocr_extractor import extract_function_sentences_from_pdf
from app.prompts.prompt_builder import prompt_template
from app.chains.llm import llm
from app.chains.vector_store import vectorstore, client, embedding
from app.utils.parser import postprocess_llm_output

from langchain.schema import HumanMessage, Document

import logging
logger = logging.getLogger(__name__)

def process_and_upload_to_qdrant():
    load_dotenv()

    DATA_DIR = "app/data/requirements"
    PDF_FILES = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    parsed_fp_results = []

    logger.info(f"[STEP 1] {len(PDF_FILES)}개 PDF 파일 처리 시작")

    for file in tqdm(PDF_FILES, desc="PDF 처리 중"):
        with open(os.path.join(DATA_DIR, file), "rb") as f:
            pdf_bytes = f.read()

        sentences = extract_function_sentences_from_pdf(pdf_bytes)

        for s in tqdm(sentences, desc=f"→ {file} 문장 처리 중", leave=False):
            try:
                user_input = prompt_template.format(question=s, context="")
                res = llm.invoke([HumanMessage(content=user_input)])
                results = postprocess_llm_output(res.content)

                if not results:
                    logger.warning(f"⚠ JSON 파싱 실패: {s}")
                    continue

                json_obj = results[0]
                json_obj["original_sentence"] = s
                parsed_fp_results.append(json_obj)

            except Exception as e:
                logger.error(f"⚠ 문장 처리 실패: {s}, 이유: {e}")
                continue

    # JSON 저장
    os.makedirs("app/data", exist_ok=True)
    with open("app/data/fp_generated_dataset.json", "w", encoding="utf-8") as f:
        json.dump(parsed_fp_results, f, ensure_ascii=False, indent=2)

    logger.info(f"[STEP 2] JSON 저장 완료 - 총 {len(parsed_fp_results)}건")

    # Qdrant 업로드
    documents = [
        Document(
            page_content=ex["description"],
            metadata={
                "function_name": ex["function_name"],
                "fp_type": ex["fp_type"],
                "complexity": ex["complexity"],
                "estimated_det": int(ex["estimated_det"]),
                "estimated_ftr": int(ex["estimated_ftr"]),
                "original_sentence": ex.get("original_sentence", "")
            }
        ) for ex in parsed_fp_results
    ]

    client.recreate_collection(
        collection_name="fp_examples",
        vectors_config={"size": 384, "distance": "Cosine"}
    )

    vectorstore.add_documents(documents)
    logger.info(f"[STEP 3] Qdrant 업로드 완료 - 총 {len(documents)}건")
