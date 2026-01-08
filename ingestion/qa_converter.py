from langchain_openai import ChatOpenAI  # OpenAI Chat 모델
from langchain_core.messages import SystemMessage, HumanMessage  # 메시지 타입
from typing import Dict  # 타입 힌트
import json  # JSON 처리
import re    # 정규식 처리

def _extract_json(text: str) -> Dict:
    # LLM 응답에서 JSON만 추출하는 내부 함수
    try:
        return json.loads(text)  # 바로 파싱 시도
    except:
        text = text.replace("```json", "").replace("```", "")  # 코드블록 제거
        match = re.search(r"\{.*\}", text, re.DOTALL)  # JSON 영역 탐색
        if match:
            return json.loads(match.group())  # 추출된 JSON 파싱
    return {}  # 실패 시 빈 dict 반환

def convert_to_qa(item: Dict, llm: ChatOpenAI) -> Dict:
    # 단일 문서를 Q/A로 변환하는 함수

    # 입력 텍스트 구성
    input_text = f"{item.get('title', '')}\n{item.get('content', '')}"

    # 너무 짧은 데이터 필터링
    if len(input_text) < 10:
        return {}

    # 프롬프트 정의
    prompt = f"""
    아래 내용을 바탕으로 질문과 답변 1쌍을 생성해.
    반드시 JSON 형식으로만 출력해.

    {{
      "question": "...",
      "answer": "..."
    }}

    내용:
    {input_text[:1500]}
    """

    # LLM 호출
    response = llm.invoke([
        SystemMessage(content="JSON만 출력"),
        HumanMessage(content=prompt)
    ])

    qa = _extract_json(response.content)  # JSON 파싱

    # 결과 검증
    if "question" in qa and "answer" in qa:
        qa["source"] = item.get("source")  # 출처 추가
        return qa

    return {}
