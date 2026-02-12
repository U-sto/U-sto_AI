import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional


logger = logging.getLogger(__name__)


# 경로 설정 (dataset/FAQ/faq_data.json)
BASE_DIR = Path(__file__).resolve().parent.parent
FAQ_FILE_PATH = BASE_DIR / "dataset" / "FAQ" / "faq_data.json"


# 캐싱 및 상태 관리 변수
_FAQ_CACHE_DATA: Optional[List[Dict]] = None  # 파싱된 JSON 데이터 자체를 메모리에 보관
_LAST_MTIME = 0.0                             # 파일 수정 시간
_IS_FILE_MISSING = False                      # 파일 부재 상태 추적 (로그 제어용)


def _ensure_faq_loaded():
    """
    파일 변경 여부를 확인하고(Hot-Reloading), 데이터를 메모리에 로드합니다.
    """
    global _FAQ_CACHE_DATA, _LAST_MTIME, _IS_FILE_MISSING

    # 1. 파일 존재 여부 확인
    if not FAQ_FILE_PATH.exists():
        if not _IS_FILE_MISSING:
            logger.warning(f"FAQ 데이터 파일 없음: {FAQ_FILE_PATH}")
            _IS_FILE_MISSING = True
            _FAQ_CACHE_DATA = []
        # 파일이 존재하지 않을 때는 마지막 수정 시간을 초기화하여
        # 이후 파일이 다시 생성되면 반드시 재로딩되도록 보장합니다.
        _LAST_MTIME = 0.0
        return
    try:
        current_mtime = FAQ_FILE_PATH.stat().st_mtime
        # 파일이 새로 생긴 경우(_IS_FILE_MISSING)에도 강제로 로드합니다.
        if _FAQ_CACHE_DATA is None or _IS_FILE_MISSING or current_mtime > _LAST_MTIME:
            with FAQ_FILE_PATH.open("r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            # 방어 로직: 스키마 검증
            # 1. 최상위 구조가 리스트인지 확인
            if not isinstance(raw_data, list):
                # logger.exception 대신 error 사용, f-string으로 타입 출력
                logger.error(f"FAQ 데이터 형식 오류: List가 아닌 {type(raw_data)} 타입입니다.")
                _FAQ_CACHE_DATA = []
            else:
                valid_data = []
                for idx, item in enumerate(raw_data):
                    if isinstance(item, dict) and "question" in item and "answer" in item:
                        valid_data.append(item)
                    else:
                        logger.warning(f"FAQ {idx}번 항목 스키마 위반으로 무시됨: {item}")
                
                _FAQ_CACHE_DATA = valid_data
            
            _LAST_MTIME = current_mtime
            _IS_FILE_MISSING = False
            
            # 2. 성공 로그: exception 대신 info 사용, f-string으로 건수 출력
            logger.info(f"FAQ 데이터 갱신 완료 (유효 데이터: {len(_FAQ_CACHE_DATA)}건)")

    except json.JSONDecodeError as e:
        logger.error(f"FAQ JSON 파싱 실패 (문법 오류): {e}")
        if _FAQ_CACHE_DATA is None: _FAQ_CACHE_DATA = []
    except Exception as e:
        # 실제 예외가 발생한 지점이므로 여기서는 exception을 사용하여 트레이스백을 남깁니다.
        logger.exception(f"FAQ 로드 중 예상치 못한 시스템 오류 발생: {e}")
        if _FAQ_CACHE_DATA is None: _FAQ_CACHE_DATA = []


def _normalize(text: str) -> str:
    """
    [헬퍼 함수] 텍스트 비교를 위해 공백과 특수문자를 제거하고 소문자로 변환합니다.
    예: "불용 차이?" -> "불용차이"
    """
    # [^a-zA-Z0-9가-힣] : 한글, 영문, 숫자 외의 모든 문자(공백, 기호 등) 제거
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', text).lower()


def get_relevant_faq_string(user_question: str) -> str:
    """
    사용자 질문에 포함된 키워드를 기반으로, 관련 있는 FAQ 항목만 반환합니다.
    (정규화를 통해 띄어쓰기/특수문자 무시하고 매칭)
    """
    _ensure_faq_loaded()
    
    if not _FAQ_CACHE_DATA:
        return ""

    # 1. 사용자 질문 정규화 (공백/기호 제거)
    norm_question = _normalize(user_question)
    
    # 2. [전체 목록 요청 감지]
    # "FAQ 보여줘" -> FAQ 전체 Q&A(질문+답변) 목록을 그대로 보여줌
    list_keywords = ["faq", "자주묻는", "질문리스트", "질문목록"]
    if any(k in norm_question for k in list_keywords):
        formatted_blocks = ["[FAQ 전체 내용 목록]"]
        for item in _FAQ_CACHE_DATA:
            # 각 FAQ 항목의 질문과 답변 쌍을 모두 추가
            formatted_blocks.append(f"Q: {item['question']}\nA: {item['answer']}")
        return "\n\n".join(formatted_blocks)

    # 3. [키워드 매칭]
    matched_items = []
    for item in _FAQ_CACHE_DATA:
        # 키워드 정규화를 1회만 수행하고, 결과를 항목에 캐시
        normalized_keywords = item.get("_normalized_keywords")
        if normalized_keywords is None:
            raw_keywords = item.get("keywords")
            if not isinstance(raw_keywords, list):
                raw_keywords = []
            normalized_keywords = [_normalize(k) for k in raw_keywords]
            item["_normalized_keywords"] = normalized_keywords
        # 이미 정규화된 키워드를 사용해 비교
        for nk in normalized_keywords:
            if nk in norm_question:
                matched_items.append(item)
                break

    if not matched_items:
        return ""

    # [최적화] 리스트에 모은 뒤 join 수행
    formatted_blocks = []
    for item in matched_items:
        formatted_blocks.append(f"Q: {item['question']}\nA: {item['answer']}")
    
    # 각 Q&A 블록 사이에 빈 줄(Double Newline)을 넣어 구분
    return "\n\n".join(formatted_blocks)