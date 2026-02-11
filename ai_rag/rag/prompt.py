import textwrap
import app.config as config
import zoneinfo
from rag.faq_service import get_relevant_faq_string
from datetime import datetime

def build_question_classifier_prompt():
    """
    질문 분류용 프롬프트 반환
    """
    return textwrap.dedent("""
    다음 질문이
    1) 사내 시스템 데이터(자산, 물품, 상태 조회)에 의존하는 질문인지
    2) 일반적인 설명/절차/정책 질문인지 판단하세요.

    출력은 반드시 아래 중 하나만:
    - NEED_RAG
    - NO_RAG

    질문: {question}
    판단:
    """)

def build_query_refine_prompt():
    """
    사용자 질문을 검색 최적화된 행정 용어 중심 질문으로 변환
    """
    return textwrap.dedent("""
    당신은 대학 행정 시스템 검색 전문가입니다.
    ...
    사용자 질문: {question}
    변환된 질문:
    """)

def build_system_prompt():
    """
    시스템 정체성, 권한, 한계를 정의하는 프롬프트 반환
    """
    return textwrap.dedent("""
    [시스템 정체성]
    - 본 AI는 대학 물품 관리 시스템 전용 AI 챗봇이다.

    [권한]
    - 제공된 Context는 참고 자료로 활용한다.
    - Context가 없는 경우에도 일반적인 절차, 정책, 설명은 가능하다.

    [한계]
    - Context에 없는 정보는 추측하지 않는다.
    - 외부 지식, 일반 상식 사용을 금지한다.
    """)


def build_role_prompt():
    """
    AI의 역할과 응답 스타일을 정의
    """
    return textwrap.dedent("""
    [역할]
    - 대학 행정 담당자를 보조하는 AI 비서 역할

    [응답 규칙]
    - 공손하고 간결한 존댓말 사용
    - 불필요한 설명, 이모지 사용 금지
    """)


def build_safety_prompt():
    """
    환각 및 오동작 방지를 위한 안전 지침 정의
    """
    return textwrap.dedent("""
    [안전 지침]
    - Context 외 정보 사용 금지
    - 모호한 질문에 대해 임의 해석 금지
    - 함수 실행을 직접 시도하지 않는다
    - 필요 시 '함수 호출이 필요함'까지만 판단한다
    """)


def build_function_decision_prompt():
    """
    Function Calling이 필요한 질의 유형과
    자연어 응답으로 처리해야 할 질의 유형을 구분하는 판단 기준 정의
    """
    return textwrap.dedent("""
    [Function Calling 판단 기준]

    다음 경우에는 함수 호출이 필요하다고 판단한다.
    - 특정 물품, 자산, 자산번호, 물품ID가 질문에 포함된 경우
    - '조회', '확인', '상태 알려줘' 등 데이터 요청 표현이 있는 경우
        예) "G2B목록번호 12345678-abcdefg의 상태 확인"
        예) 25년 10월에 구입한 노트북 자산 현황 조회해줘
    다음 경우에는 자연어로 응답한다.
    - 매뉴얼 설명
        예) "처분 절차 설명해줘"
    - 제도, 절차, 정책 설명
        예) "자산의 폐기 기준이 어떻게 되나요?"

    [혼합/모호한 질의 처리 기준]
    - 한 질문에 개별 자산 조회와 정책/절차 설명이 함께 있는 경우:
      1) 개별 자산의 현재 상태/위치/소유자 등 데이터가 필요하면
         → 해당 자산 부분에 대해 함수 호출이 필요하다고 판단한다.
      2) 정책/절차 설명은 자연어로 응답한다.
      예) "G2B목록번호 12345678-abcdefg의 현재 상태랑 노트북 자산 폐기 기준도 알려줘"

    - 특정 자산번호(ex.G2B목록번호/물품고유번호/물품 분류번호/물품 식별번호)가 언급되었으나,
      실제로는 일반 정책만 묻는 경우:
      → 개별 데이터 조회가 필요 없으면 함수 호출을 하지 않는다.
      예) "물품 분류번호 12345678 같은 노트북의 폐기 기준은 뭐야?"

    - 질문이 모호하여 판단이 어려운 경우:
      → 개별 자산의 실시간 데이터가 반드시 필요한 경우에만
        함수 호출이 필요하다고 판단한다.
    """)


# FAQ 데이터를 프롬프트에 주입
def build_faq_prompt(question: str) -> str:
    """
    질문과 관련된 FAQ가 있을 경우에만 프롬프트 섹션을 생성합니다.
    """
    # 서비스 로직 호출 (키워드 매칭 수행)
    faq_data = get_relevant_faq_string(question)
    
    if not faq_data:
        return "" 

    # get_relevant_faq_string가 전체 FAQ 목록을 반환하는 경우
    # (예: "[FAQ 전체 내용 목록]"으로 시작)와 질문 연관 FAQ만 반환하는
    # 경우를 구분하여 안내 문구를 다르게 구성한다.
    is_full_list = faq_data.lstrip().startswith("[FAQ 전체 내용 목록]")
    if is_full_list:
        header = "[FAQ 지식 베이스 (전체 목록)]"
        description = (
            "전체 FAQ 목록이 제공되었습니다.\n"
            "사용자 요청을 충실히 반영하면서, 필요할 경우 아래 FAQ 목록을 참고하여 답변하세요."
        )
    else:
        header = "[FAQ 지식 베이스 (관련 내용)]"
        description = (
            "사용자 질문과 연관된 FAQ 내용이 발견되었습니다.\n"
            "아래 내용을 참고하여 답변하세요."
        )
    return textwrap.dedent(f"""
    {header}
    {description}

    {faq_data}
    """)


def assemble_prompt(context: str, question: str) -> str:
    """
    System / Role / Safety / Function 판단 규칙과
    RAG Context, 사용자 질문을 하나의 프롬프트로 조립
    """
    sections = []

    if config.ENABLE_SYSTEM_PROMPT:
        sections.append(build_system_prompt())

    # 역할 및 응답 스타일은 시스템 전반에 항상 적용되어야 하는 필수 프롬프트이므로
    # 다른 섹션과 달리 별도의 ENABLE_* 플래그 없이 항상 포함한다.
    sections.append(build_role_prompt())

    if config.ENABLE_SAFETY_PROMPT:
        sections.append(build_safety_prompt())

    # FAQ 프롬프트 사용 여부는 다른 ENABLE_* 플래그들과 동일하게 config에서 직접 제어한다.
    if getattr(config, "ENABLE_FAQ_PROMPT", False):
        faq_section = build_faq_prompt(question)
        if faq_section:
            sections.append(faq_section)

    if config.ENABLE_FUNCTION_DECISION_PROMPT:
        sections.append(build_function_decision_prompt())

    sections.append(f"[참고 자료]\n{context}")
    sections.append(f"[질문]\n{question}")

    return "\n\n".join(sections)


# qa_convert.py에서 사용
def build_dataset_creation_system_prompt() -> str:
    """
    [데이터셋 생성] QA 변환 시 AI에게 부여할 역할(System Message) 정의
    """
    return "너는 데이터셋 생성을 돕는 AI야. 반드시 유효한 JSON만 출력해."


# qa_convert.py에서 사용
def build_qa_generation_prompt() -> str:
    """
    [데이터 생성용] 매뉴얼 내용을 바탕으로 QA 쌍을 생성하는 프롬프트 템플릿을 반환합니다.

    Returns:
        str: {context} 플레이스홀더를 포함한 프롬프트 문자열. 
             사용 시 .format(context=...)을 통해 실제 내용을 주입해야 합니다.
    """
    return textwrap.dedent("""
    아래 [내용]을 완벽하게 이해한 뒤, 사용자가 이 정보를 찾기 위해 물어볼 법한 질문(question)과 그에 대한 답변(answer)을 생성해.

    [작성 규칙]
    1. 질문은 "어떻게 해?", "뭐야?" 처럼 대화체로 작성하되, 핵심 키워드(예: 반납, 불용, 처분 등)를 반드시 포함할 것.
    2. 답변은 매뉴얼 내용을 기반으로 상세하게 작성할 것.
    3. 카테고리는 주제를 대표하는 단어 1개(규정, 절차, 시스템, 오류해결 등)로 추출할 것.

    반드시 아래 JSON 형식으로만 출력해:
    {{
      "question": "생성된 질문",
      "answer": "생성된 답변",
      "category": "추출된 카테고리"
    }}

    [내용]:
    {context}
    """)


# Function Calling(Tools) 전용 프롬프트

# KST 타임존 객체는 변하지 않으므로 전역 상수로 한 번만 생성 (메모리 절약 & 속도 향상)
KST = zoneinfo.ZoneInfo("Asia/Seoul")

def build_tool_aware_system_prompt():
    """
    도구(Tools) 사용이 가능한 AI의 **도구 선택/사용 가이드용** 시스템 프롬프트 조각입니다.
    이 프롬프트는 전체 시스템 프롬프트가 아니라, build_role_prompt에서 생성하는 페르소나/역할 프롬프트와 결합되어 사용되는 '도구 선택 로직' 부분만을 담당합니다.
    """
    # 현재 날짜 정보 (수명 계산 등을 위해 필요할 수 있음)
    current_date = datetime.now(KST).strftime("%Y년 %m월 %d일")

    return textwrap.dedent(f"""
    [시스템 설정: 도구(Tools) 사용 및 판단 가이드]
    오늘은 {current_date} 입니다.
    당신은 사용자 질문을 분석하여 **필요한 경우에만** '도구(Tool)'를 선택해야 합니다.
                          
    [판단 기준 1: 도구를 사용해야 하는 경우]
    다음 상황에서는 반드시 적절한 도구를 선택(Call)하세요.
    1. **자산 실시간 정보 조회**: "이 물품의 운용부서 알려줘", "이 물품 취득금액이 얼마야?" 등 DB 데이터가 필요할 때
       -> `get_item_detail_info` 호출
    2. **미래 예측/수명 분석**: "수명 얼마나 남았어?", "교체 주기 알려줘", "언제 고장 나?" 등 분석이 필요할 때
       -> `open_usage_prediction_page` 호출 (직접 계산 금지)
    [판단 기준 2: 도구를 사용하지 않는 경우]
    - "불용 처리 방법 알려줘", "반납 규정이 뭐야?", "물품 등록 절차는?" 등 **업무 절차, 방법, 규정**을 묻는 질문.
    - 위와 같은 질문에서는 제공된 참고 자료(Context)와 일반적인 업무 지식을 활용해 직접 답변하세요.
    - 다만, 위와 같은 질문에 자산의 실시간 정보 조회나 수명 예측이 **함께** 필요한 경우에는, [판단 기준 1]에 따라 해당 목적에 맞는 도구는 병행해서 사용할 수 있습니다.
    """)