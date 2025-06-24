from typing import List, Dict, Tuple, Union
from pathlib import Path
import json

from app.models.schema import FPResult

# FP 점수 룰 JSON 경로
FP_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "fp_weights.json"

# 룰 로딩
try:
    with open(FP_WEIGHTS_PATH, "r", encoding="utf-8") as f:
        FP_WEIGHTS: Dict[str, Dict[str, int]] = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"[에러] FP 점수 룰 파일이 존재하지 않습니다: {FP_WEIGHTS_PATH}")
except json.JSONDecodeError:
    raise RuntimeError(f"[에러] FP 점수 룰 파일이 올바른 JSON 형식이 아닙니다: {FP_WEIGHTS_PATH}")


def _to_dict(item: Union[FPResult, Dict]) -> Dict:
    """FPResult 또는 dict를 dict로 변환"""
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):  # Pydantic v2
        return item.model_dump()
    elif hasattr(item, "dict"):  # Pydantic v1
        return item.dict()
    else:
        raise ValueError(f"[에러] dict로 변환 불가한 타입: {type(item)}")


def calculate_fp_score(functions: List[Union[FPResult, Dict]]) -> Tuple[int, List[Dict]]:
    """
    기능 리스트를 입력 받아 각 기능별 FP 점수 계산 및 총합 반환

    :param functions: 기능 추론 결과 리스트 (function_name, fp_type, complexity 등 포함)
    :return: (총점수, 기능별 점수 포함 리스트)
    """

    total_score = 0
    scored_functions = []

    for index, raw_func in enumerate(functions):
        try:
            func = _to_dict(raw_func)

            fp_type = func.get("fp_type", "").strip().upper()
            complexity = func.get("complexity", "").strip().upper()

            if not fp_type or not complexity:
                print(f"[경고] 인덱스 {index}: fp_type 또는 complexity 누락 → 무시됨 - {func}")
                continue

            score = FP_WEIGHTS.get(fp_type, {}).get(complexity, 0)

            if score == 0:
                print(f"[주의] 인덱스 {index}: 유효하지 않은 조합 (fp_type: {fp_type}, complexity: {complexity}) 또는 점수 없음")

            print(f"[점수 계산] function: {func.get('function_name')} / fp_type: {fp_type} / complexity: {complexity} → score: {score}")

            enriched_func = {
                **func,
                "score": score
            }

            total_score += score
            scored_functions.append(enriched_func)

        except Exception as e:
            print(f"[에러] 인덱스 {index} 기능 점수 계산 실패: {e} → 원본: {raw_func}")
            continue

    print(f"[총 FP 점수]: {total_score}")
    return total_score, scored_functions
