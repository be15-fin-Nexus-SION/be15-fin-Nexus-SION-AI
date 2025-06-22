from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("""
당신은 시스템 분석 전문가입니다. 다음은 시스템 요구사항 문서에서 추출한 문장들입니다.
당신에게 제공된 문맥(근거)을 참고하여, 기능 단위로 설명을 구성해 주세요.

[문맥]
{context}

[추론 대상 문서]
{question}

출력 형식은 다음과 같습니다:
[
  {{
    "function_name": "기능명",
    "description": "기능 설명",
    "fp_type": "EI | EO | EQ | ILF | EIF",
    "complexity": "SIMPLE | MEDIUM | COMPLEX"
  }}
]
""")