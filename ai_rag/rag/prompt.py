import textwrap
import zoneinfo
import sys
from pathlib import Path
from rag.faq_service import get_relevant_faq_string
from datetime import datetime

try:
    import app.config as config
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import app.config as config

def build_question_classifier_prompt():
    """
    질문 분류용 프롬프트 반환
    """
    return textwrap.dedent("""
    당신은 대학 물품 관리 RAG 챗봇의 라우터입니다.
    사용자 질문에 매뉴얼, FAQ, 업무 지식베이스 근거가 필요한지 판단하세요.

    NEED_RAG로 분류할 질문:
    - 물품 취득, 등록, 검수, 운용, 반납, 불용, 처분, 관리전환, 보유현황, 사용주기, 내용연수, G2B, 물품고유번호, 라벨, 승인, 확정
    - 규정, 기준, 절차, 방법, 제약사항, 유의사항, 차이, 정의, 시스템 사용법을 묻는 질문
    - "어떻게 해?", "기준이 뭐야?", "차이가 뭐야?", "절차 알려줘", "왜 안 돼?"처럼 업무 문서 근거가 필요한 질문
    - 대학 물품 관리 업무와 관련된 용어가 하나라도 포함된 질문
    - 특정 자산 조회가 아닌 일반 정책/절차 질문도 반드시 NEED_RAG
    - AI 챗봇, 챗봇, 쳇봇의 기능, 주요 기능, 제한, 사용 예시, FAQ, 좋은 질문, 답변 가능한 질문 묻는 질문 (무조건 NEED_RAG)
                           
    NO_RAG로 분류할 질문:
    - 인사말, 감사 인사, 잡담, 너는 누구야, 질문 다시 써줘 등 지식베이스가 필요 없는 단순 요청
    - 대학 물품 관리와 무관한 일반 상식 질문

    판단 원칙:
    - 업무 도메인 용어가 보이면 기본값은 NEED_RAG입니다.
    - 애매하면 사용자의 업무 답변 안정성을 위해 NEED_RAG로 판단하세요.
    - 설명을 붙이지 말고 라벨 하나만 출력하세요.

    출력은 반드시 아래 중 하나만:
    NEED_RAG
    NO_RAG

    질문: {question}
    판단:
    """)

def build_query_refine_prompt():
    """
    사용자 질문을 검색 최적화된 행정 용어 중심 질문으로 변환
    """
    return textwrap.dedent("""
    당신은 대학 행정 시스템 검색 전문가입니다.
    사용자 질문을 벡터 검색에 적합한 한국어 검색 질의 1문장으로 바꾸세요.

    규칙:
    - 원 질문의 핵심 명사와 업무 용어는 삭제하지 마세요.
    - 물품 취득, 등록, 검수, 운용, 반납, 불용, 처분, 관리전환, 보유현황, 사용주기, 내용연수, G2B, 물품고유번호, 라벨, 승인, 확정 같은 도메인 용어를 보존하세요.
    - 도메인 용어는 행정 매뉴얼의 표준 표현으로 맞추세요: 돌려줌/반환=반납, 사용중단=불용, 폐기/매각/양여=처분, 구매/구입=취득, 대장 입력=등록, 확인/인수=검수, 부서 이관=관리전환.
    - 구어체 표현은 행정 매뉴얼에서 쓰일 법한 표현으로 정리하세요.
      예: "버리는 법" -> "물품 불용 또는 처분 절차"
      예: "돌려주는 법" -> "물품 반납 절차"
      예: "폐기랑 불용 차이" -> "물품 처분과 물품 불용의 차이"
      예: "번호로 찾는 법" -> "물품고유번호 또는 G2B목록번호 조회 방법"
    - G2B목록번호, 물품고유번호, 자산번호, 날짜, 부서명, 물품명은 절대 삭제하지 마세요.
    - 반납/불용/처분/취득처럼 헷갈리는 절차가 함께 나오면 둘 중 하나를 임의로 삭제하지 마세요.
    - 복합 질문은 핵심 업무 용어를 모두 포함한 한 문장으로 압축하세요.
    - 원 질문에 없는 제도명, 화면명, 번호, 조건을 새로 만들지 마세요.
    - 답변이나 설명을 쓰지 말고 검색 질의만 출력하세요.
    - 잘 모르겠으면 원 질문을 거의 그대로 유지하세요.

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
    - 제공된 Context를 최우선 근거로 활용한다.
    - 업무 절차, 정책, 규정, 시스템 사용법은 Context에 근거가 있을 때만 답변한다.

    [한계]
    - Context에 없는 정보는 추측하지 않는다.
    - 외부 지식, 일반 상식 사용을 금지한다.
    - Context에 없는 세부 절차, 버튼명, 화면명, 메뉴명, 예외 조건, 승인 조건은 만들지 않는다.
    - Context가 부족하면 "제공된 매뉴얼 근거만으로는 확인하기 어렵습니다"라고 답한다.
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
    - 서로 다른 절차가 섞여 있으면 Context에서 확인되는 범위만 구분해 답한다
    - 답변에 포함한 단계, 기준, 예외는 반드시 참고 자료에 근거해야 한다
    - 참고 자료에 일부 단계만 있으면, 확인된 단계만 답하고 누락된 단계는 확인되지 않는다고 명시한다
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


def assemble_prompt(
    context: str,
    question: str,
    include_function_decision: bool = False,
    tool_context: str = "",
) -> str:
    """
    System / Role / Safety,
    도구 결과, RAG Context, 사용자 질문을 답변 생성용 프롬프트로 조립한다.

    include_function_decision 인자는 과거 호출부 호환을 위해 남겨두지만,
    답변 생성 단계에서는 도구 호출 판단 기준을 포함하지 않는다.
    """
    sections = []

    if config.ENABLE_SYSTEM_PROMPT:
        sections.append(build_system_prompt())

    # 역할 및 응답 스타일은 시스템 전반에 항상 적용되어야 하는 필수 프롬프트이므로
    # 다른 섹션과 달리 별도의 ENABLE_* 플래그 없이 항상 포함한다.
    sections.append(build_role_prompt())

    if config.ENABLE_SAFETY_PROMPT:
        sections.append(build_safety_prompt())

    # 키워드 기반 FAQ 매칭은 벡터 검색을 보완하는 high precision shortcut이다.
    if getattr(config, "ENABLE_FAQ_PROMPT", False):
        faq_section = build_faq_prompt(question)
        if faq_section:
            sections.append(faq_section)

    if tool_context.strip():
        sections.append(f"[도구 조회 결과]\n{tool_context.strip()}")

    if context.strip():
        sections.append(f"[참고 자료]\n{context}")
    else:
        sections.append("[참고 자료]\n제공된 참고 자료가 없습니다.")

    sections.append(textwrap.dedent("""
    [답변 원칙]
    - 도구 조회 결과와 참고 자료에 근거한 내용만 답변하세요.
    - 도구 조회 결과는 개별 자산의 현재 상태, 금액, 부서 등 실시간 데이터 근거로만 사용하세요.
    - 참고 자료는 정책, 절차, 규정, 시스템 사용법 근거로만 사용하세요.
    - 도구 조회 결과와 참고 자료를 함께 받은 경우, 개별 자산 정보와 정책/절차 설명을 구분해서 답변하세요.
    - 도구 조회가 실패했거나 결과가 없으면 해당 조회 실패를 안내하고, 참고 자료만으로 개별 자산 상태를 추측하지 마세요.
    - doc_type=qa는 질문 의도 매칭 신호로만 보고, 세부 절차와 예외 조건은 doc_type=manual_chunk 또는 doc_type=faq를 근거로 답변하세요.
    - doc_type=manual_chunk와 doc_type=faq의 내용이 충돌하면 doc_type=manual_chunk를 우선하고, FAQ는 보조 근거로만 사용하세요.
    - 참고 자료에서 확인되지 않는 세부 내용은 추측하지 마세요.
    - 참고 자료에 없는 버튼명, 화면명, 메뉴명, 절차 순서, 승인/반려 조건을 새로 만들지 마세요.
    - 근거가 부족한 경우에는 가능한 범위와 확인되지 않는 범위를 분리해서 답변하세요.
    - 절차는 가능한 한 순서대로 정리하세요.
    - 질문과 직접 관련 없는 참고 자료는 답변에 사용하지 마세요.
    """).strip())
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
    - 위와 같은 질문에서는 도구를 호출하지 말고 RAG 검색 단계로 넘기세요.
    - 다만, 위와 같은 질문에 자산의 실시간 정보 조회나 수명 예측이 **함께** 필요한 경우에는, [판단 기준 1]에 따라 해당 목적에 맞는 도구는 병행해서 사용할 수 있습니다.
    """)
