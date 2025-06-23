from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("""
당신은 시스템 분석 전문가입니다.

아래는 시스템 요구사항 문서에서 추출한 문장과, 관련 문맥(근거)입니다. 문장을 읽고, 기능 단위로 분해하여 정확한 JSON 형식으로만 출력하세요.

⚠️ 다음 조건을 철저히 지켜주세요:
- 자연어 설명 없이 JSON **배열만 출력**할 것
- 출력 항목은 **항상 아래 4개 필드를 모두 포함**할 것
- JSON 형식은 완전해야 하며, JSON 규칙을 100% 지킬 것

필드 형식:
- function_name: 간결한 기능명 (문자열)
- description: 기능 설명 (문자열)
- fp_type: EI | EO | EQ | ILF | EIF 중 하나
- complexity: SIMPLE | MEDIUM | COMPLEX 중 하나

예시 출력:
[
  {{
    "function_name": "사용자 권한 제어",
    "description": "사용자의 권한에 따라 접근 가능한 기능을 제어합니다.",
    "fp_type": "EI",
    "complexity": "MEDIUM"
  }}
]

[문맥]
{context}

[추론 대상 문서]
{question}
""")
