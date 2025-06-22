import requests

payload = {
    "ocr_text": "사용자의 권한에 따라 접근 가능한 기능을 제어한다.",
    "request_id": "test-001"
}

res = requests.post("http://localhost:8100/fp-infer", json=payload)

if res.status_code == 200:
    print("[Success] 응답 결과:")
    print(res.json())
else:
    print(f"[Failed] 상태 코드: {res.status_code}")
    print(res.text)
