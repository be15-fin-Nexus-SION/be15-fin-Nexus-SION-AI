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
