from langchain.prompts import PromptTemplate

prompt_fallback = PromptTemplate.from_template("""
당신은 소프트웨어 요구사항을 기능 단위로 분석하는 전문가입니다.
아래 문장을 보고 각 기능에 대해 가능한 정보를 JSON 형식으로 작성해주세요.

형식 예시:
[
  {{
    "function_name": "비밀번호 변경",
    "description": "사용자가 자신의 비밀번호를 수정할 수 있는 기능",
    "fp_type": "EI",
    "complexity": "SIMPLE",
    "estimated_det": 3,
    "estimated_ftr": 1
  }}
]

선택 가능한 값 안내:
- "fp_type": EI, EO, EQ, ILF, EIF 중에서 선택
- "complexity": SIMPLE, MEDIUM, COMPLEX 중에서 선택
- "estimated_det", "estimated_ftr": 대략적인 숫자 추정 (0 이상 정수)

요구사항 문장:
{query}
""")
