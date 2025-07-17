import json
import re
from typing import List

def extract_result_text(raw_output) -> str:
    if isinstance(raw_output, dict):
        return raw_output.get("result", "")
    elif isinstance(raw_output, str):
        return raw_output
    return str(raw_output)

def postprocess_llm_output(text: str) -> List[dict]:
    try:
        cleaned = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        cleaned = re.sub(r"```json|```", "", cleaned)
        matches = re.findall(r"{[^{}]*}", cleaned)
        return [json.loads(m) for m in matches if m.strip()]
    except Exception as e:
        print("JSON 파싱 오류:", e)
        return []

def parse_fp_response_for_inference(raw_output: dict) -> List[dict]:
    try:
        result_text = raw_output.get("result") or raw_output.get("text") or ""

        cleaned = (
            result_text
            .replace("“", '"').replace("”", '"')
            .replace("’", "'").replace("‘", "'")
        )
        cleaned = re.sub(r"```json|```", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # fallback: 객체 단위로만 파싱
            objects = re.findall(r"{[^{}]*}", cleaned)
            return [json.loads(obj) for obj in objects if obj.strip()]

    except Exception as e:
        print("[ERROR] FP 응답 파싱 실패:", e)
        return []
