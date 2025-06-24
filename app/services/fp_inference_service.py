from app.chains.retriever import retriever
from app.chains.chains import rag_chain, fallback_chain
from app.utils.parser import extract_result_text, postprocess_llm_output
from app.models.schema import FPResult

async def run_fp_inference(ocr_text: str) -> list[FPResult]:
    raw_output = None
    search_results = retriever.invoke(ocr_text)
    use_fallback = not search_results or len(search_results) == 0
    print("검색 결과 문서 수:", len(search_results))

    if use_fallback:
        print("벡터 검색 결과 없음 → fallback 프롬프트 사용")
        raw_output = fallback_chain.invoke({"question": ocr_text})
    else:
        print("벡터 검색 성공 → RAG 실행")
        raw_output = rag_chain.invoke({"query": ocr_text})
        result_text = extract_result_text(raw_output)
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

    print("[최종 결과 FPResult 리스트]:")
    for fp in result:
        print(fp.model_dump())  # Pydantic v2 기준, v1이면 .dict()

    return result
