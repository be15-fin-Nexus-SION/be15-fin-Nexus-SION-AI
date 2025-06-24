import requests

test_items = [
    {
        "description": "권한 제어 기능",
        "text": "사용자의 권한에 따라 접근 가능한 기능을 제어한다."
    },
    {
        "description": "비밀번호 변경 기능",
        "text": "사용자는 자신의 비밀번호를 변경할 수 있어야 한다."
    },
    {
        "description": "게시글 작성 및 삭제",
        "text": "관리자는 게시글을 작성하고 삭제할 수 있다."
    },
    {
        "description": "외부 시스템 연동",
        "text": "시스템은 외부 회계 시스템과 연동하여 정산 정보를 교환해야 한다."
    },
    {
        "description": "정보 조회 기능",
        "text": "사용자는 로그인 후 자신의 정보를 조회할 수 있어야 한다."
    }
]

payload = {
    "project_id": 999,
    "ocr_items": test_items
}

res = requests.post("http://localhost:8100/fp-infer", json=payload)

print("=" * 100)
print(f"[TEST] Project ID: {payload['project_id']}")
if res.status_code == 200:
    print("[Success] 응답 결과:")
    print(res.json())
else:
    print(f"[Failed] 상태 코드: {res.status_code}")
    print(res.text)
