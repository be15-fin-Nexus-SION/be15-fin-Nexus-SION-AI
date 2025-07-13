from app.chains.chains import rag_chain, fallback_chain
from app.utils.parser import extract_result_text, postprocess_llm_output, parse_fp_response_for_inference
from app.models.schema import FPResult
import asyncio

async def run_fp_inference(ocr_text: str) -> list[FPResult]:
    raw_output = await asyncio.to_thread(rag_chain.invoke, {"query": ocr_text})

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
        print(fp.model_dump())

    return result

async def run_single_fp_inference(function_name: str, description: str) -> dict:
    try:
        query = f"{function_name}\n{description}"
        raw_output = await asyncio.to_thread(fallback_chain.invoke, {"query": query})
        print("LLM 응답 원문:\n", raw_output)

        # result_text = extract_result_text(raw_output)
        parsed = parse_fp_response_for_inference(raw_output)

        if not parsed or not isinstance(parsed, list) or not parsed[0]:
            raise ValueError("LLM 응답 파싱 실패")

        item = parsed[0]

        return {
            "fpType": item.get("fp_type", "").strip(),
            "complexity": item.get("complexity", "").strip(),
            "det": int(item.get("estimated_det", 0)) or 0,
            "ftrOrRet": int(item.get("estimated_ftr", 0)) or 0,
        }

    except Exception as e:
        print(f"[ERROR] FP 단일 추론 실패 - 기능명: {function_name}, 사유: {e}")
        return {
            "fpType": "UNKNOWN",
            "complexity": "UNKNOWN",
            "det": 0,
            "ftrOrRet": 0
        }
