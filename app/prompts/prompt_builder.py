from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("""
당신은 시스템 분석 및 기능 점수화(Function Point Analysis) 전문가입니다.

아래는 시스템 요구사항 문서에서 추출한 문장과, 관련 문맥(근거)입니다.
이 정보를 기반으로 기능 단위로 분해하고, 각 기능에 대해 IFPUG 기준에 따라 다음 항목들을 추론해주세요.

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

형식 예시는 다음과 같습니다:

[
  {{
    "function_name": "로그인 기능",
    "description": "사용자가 ID/PW로 로그인한다",
    "fp_type": "EI" //fp_type은 필수적으로 넣어주세요,
    "complexity": "SIMPLE" //complexity은 필수적으로 넣어주세요,
    "estimated_ftr": 2, // 숫자만 입력. 예: 1, 2, 3 등
    "estimated_ftr": 1, // 숫자만 입력. 예: 1, 2, 3 등
  }}
]

형식은 가능한 맞추되, 정확하지 않아도 좋습니다. 가능한 비슷한 형식으로 JSON 배열로 작성해주세요.

[문맥]
{context}

[추론 대상 문서]
{question}
""")
