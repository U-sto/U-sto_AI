import json  # JSON 처리
import re    # 정규식 처리
import traceback
from langchain_openai import ChatOpenAI 
from langchain_core.messages import SystemMessage, HumanMessage  # 메시지 타입
from typing import Dict  # 타입 힌트
from rag.prompt import build_qa_generation_prompt, build_dataset_creation_system_prompt

# MIN_TRUNCATION_RATIO: 텍스트를 자를 때, 마침표(.)를 찾더라도
# 전체 허용 길이의 최소 50% 이상은 유지하도록 보장하는 비율.
MIN_TRUNCATION_RATIO = 0.5

def _extract_json(text: str) -> Dict:
    # LLM 응답에서 JSON만 추출하는 내부 함수
    try:
        return json.loads(text)  # 바로 파싱 시도
    except json.JSONDecodeError:
        # 마크다운 코드블록 제거
        text = text.replace("```json", "").replace("```", "")
        # 중괄호 {} 로 감싸진 부분만 정규식으로 추출
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())  # 추출된 JSON 파싱
    return {}  # 실패 시 빈 dict 반환

def convert_to_qa(item: Dict, llm: ChatOpenAI) -> Dict:
    """
    Convert a single ingested document item into a Q/A pair and attach metadata.
    """

    # 입력 텍스트 구성 (챕터와 제목도 포함하여 AI에게 문맥 제공)
    chapter = item.get('chapter', '')
    title = item.get('title', '')
    content = item.get('content', '')

    # 문맥 정보를 합쳐서 하나의 텍스트로 만듦
    context_text = f"Chapter: {chapter}\nTitle: {title}\nContent: {content}"

    # 너무 짧은 데이터 필터링 (노이즈 제거)
    if len(content) < 10:
        return {}

    # 프롬프트 완성 (템플릿에 데이터 끼워넣기)
    # 2000자 제한을 두어 토큰 비용 절약
    MAX_CONTEXT_CHARS = 2000

    if len(context_text) > MAX_CONTEXT_CHARS:
        # 긴 컨텍스트는 잘라내되, 사용자가 알 수 있도록 로그를 남김
        print(
            f"[Info] Context text truncated from {len(context_text)} to "
            f"{MAX_CONTEXT_CHARS} characters for title='{title}'"
        )
        truncated_context = context_text[:MAX_CONTEXT_CHARS]
        # 가능한 경우 문장 단위(마침표 기준)로 끊어서 부자연스러운 잘림을 최소화
        last_sentence_end = truncated_context.rfind(".")

        # 마침표가 발견되었고, 그 위치가 최소 유지 비율(50%)보다 뒤에 있을 때만 자름
        if last_sentence_end != -1 and last_sentence_end > MAX_CONTEXT_CHARS * MIN_TRUNCATION_RATIO:
            truncated_context = truncated_context[: last_sentence_end + 1]
    else:
        truncated_context = context_text
    
    # import한 함수를 호출하여 프롬프트 템플릿 가져오기
    qa_template = build_qa_generation_prompt()
    final_prompt = qa_template.format(context=truncated_context)

    # LLM 호출
    try:
        # 하드코딩 된 문자열 대신 함수 호출로 변경
        system_msg_content = build_dataset_creation_system_prompt()

        response = llm.invoke([
            SystemMessage(content=system_msg_content),
            HumanMessage(content=final_prompt)
        ])
        
        qa = _extract_json(response.content)  # JSON 파싱
    
    except Exception as e:
        print(f"[Error] LLM 변환 중 치명적인 오류 발생")
        print(f" - 에러 메시지: {e}")
        print(f" - 문제 발생 문서: {item.get('title', '제목 없음')}") # 어떤 문서인지 알려줌
        print("Detailed Traceback:")
        print(traceback.format_exc())
        return {}

    # 결과 검증 및 메타데이터(Metadata) 부착
    # 원본 데이터의 정보를 결과물에 꼬리표로 붙여줍니다.
    if "question" in qa and "answer" in qa:
        qa["source"] = item.get("source")   # 파일명 (loader.py에서 가져옴)
        qa["title"] = title                 # 원본 소제목 (검색 시 활용)
        qa["chapter"] = chapter             # 챕터 정보 (필터링 시 활용)
        
        # category는 위에서 AI가 생성했으므로 그대로 둠 (혹은 강제로 지정 가능)
        if "category" not in qa:
            qa["category"] = "General"
            
        return qa

    return {}