# tools.py
import json
import logging
import os
import urllib.parse
import requests
import re
import html
from typing import Optional, Tuple, Dict, Any
from rag import dictionaries

from dotenv import load_dotenv
from langchain_core.tools import tool

# 로거 설정
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# 필수 환경변수 로드
BACKEND_API_URL = os.getenv("BACKEND_API_URL")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL")
API_REQUEST_TIMEOUT = float(os.getenv("API_REQUEST_TIMEOUT", 5))

# 필수 환경변수 검증
missing_vars = []
if not BACKEND_API_URL: missing_vars.append("BACKEND_API_URL")
if not FRONTEND_BASE_URL: missing_vars.append("FRONTEND_BASE_URL")

if missing_vars:
    error_msg = f"Required environment variables are missing: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise RuntimeError(error_msg)


# [최적화] 동의어 조회용 해시 테이블 (O(1))
_SYNONYM_LOOKUP = {k.lower(): v for k, v in dictionaries.KEYWORD_SYNONYMS.items()}


def _get_normalized_keyword(input_str: str) -> Optional[str]:
    """입력값의 동의어를 찾아 표준어로 반환합니다."""
    if not input_str:
        return None
    return _SYNONYM_LOOKUP.get(input_str.strip().lower())


def _apply_smart_correction(
    name: Optional[str], 
    id_val: Optional[str], 
    id_num: Optional[str]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    사용자가 ID 필드에 키워드를 넣었을 때 이를 Name 필드로 이동시키는 보정 로직입니다.
    """
    # 1. asset_id 필드 보정
    if id_val and _get_normalized_keyword(id_val) is not None:
        if not name:
            logger.info(f"[Smart Correction] asset_id '{id_val}' -> asset_name으로 이동")
            name = id_val
        else:
            logger.warning(
                f"[Smart Correction - Conflict] asset_id '{id_val}'가 키워드지만 name이 존재하여 무시함."
            )
        id_val = None  # 이동했거나 충돌났으므로 ID 필드는 비움

    # 2. identification_num 필드 보정
    if id_num and _get_normalized_keyword(id_num) is not None:
        if not name:
            logger.info(f"[Smart Correction] identification_num '{id_num}' -> asset_name으로 이동")
            name = id_num
        else:
            logger.warning(
                f"[Smart Correction - Conflict] id_num '{id_num}'가 키워드지만 name이 존재하여 무시함."
            )
        id_num = None

    return name, id_val, id_num


# =============================================================================
# LangChain Tools
# =============================================================================

@tool
def get_item_detail_info(
    asset_name: Optional[str] = None, 
    asset_id: Optional[str] = None, 
    identification_num: Optional[str] = None
) -> str:
    """
    물품의 G2B목록명, 목록번호, 고유번호를 통해 상세 정보를 조회합니다.
    사용자가 필드를 잘못 입력해도 자동 보정(Smart Correction)을 수행합니다.
    """
    # 1. 기본 정제
    asset_name = asset_name.strip() if asset_name else None
    asset_id = asset_id.strip() if asset_id else None
    identification_num = identification_num.strip() if identification_num else None

    # 2. 스마트 보정 (로직 분리)
    asset_name, asset_id, identification_num = _apply_smart_correction(
        asset_name, asset_id, identification_num
    )

    # 3. 필수값 검증
    if not any((asset_name, asset_id, identification_num)):
        return json.dumps(
            {"error": "검색할 G2B목록명, G2B목록번호, 또는 물품고유번호 중 하나는 필수로 입력해야 합니다."},
            ensure_ascii=False
        )

    # 4. 검색어 표준화 (Synonym -> Standard)
    target_search_name = asset_name
    if asset_name:
        standard_name = _get_normalized_keyword(asset_name)
        if standard_name:
            target_search_name = standard_name
            logger.info(f"[Synonym Match] '{asset_name}' -> '{target_search_name}'")

    # 5. 파라미터 구성
    params = {}
    if identification_num: params["identification_num"] = identification_num
    if asset_id: params["asset_id"] = asset_id
    if target_search_name: params["asset_name"] = target_search_name

    # 6. API 호출
    try:
        # urljoin을 사용하여 경로 결합 안전성 확보
        api_url = urllib.parse.urljoin(BACKEND_API_URL, "/search")
        # 주의: urljoin은 base가 '/'로 끝나지 않으면 마지막 경로를 덮어쓸 수 있으므로, 
        # 명시적으로 슬래시 처리를 하거나 기존 방식 유지. 여기서는 안전하게 기존 로직을 urljoin 스타일로 변경
        # (BACKEND_API_URL이 'http://api.com' 형태라고 가정)
        if not api_url.endswith('/search'): # urljoin 특성상 base 뒤에 /가 없으면 path가 대체될 수 있음 방지
             api_url = f"{BACKEND_API_URL.rstrip('/')}/search"

        response = requests.get(api_url, params=params, timeout=API_REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("results"):
            msg = "조건에 맞는 물품을 찾을 수 없습니다."
            if asset_name and asset_name != target_search_name:
                msg += f" (참고: '{asset_name}' -> '{target_search_name}' 변환 검색)"
            return json.dumps({"message": msg}, ensure_ascii=False)

        return json.dumps(data, ensure_ascii=False)

    except json.JSONDecodeError:
        logger.error("API 응답 JSON 파싱 실패", exc_info=True)
        return json.dumps({"error": "서버 응답 형식이 올바르지 않습니다."}, ensure_ascii=False)

    # 에러 핸들링 (구체적 -> 포괄적 순서 유지)
    except requests.exceptions.Timeout as e:
        logger.error(f"API 요청 시간 초과: {e}")
        return json.dumps({"error": "요청 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."}, ensure_ascii=False)

    except requests.exceptions.ConnectionError as e:
        logger.error(f"API 서버 연결 실패: {e}")
        return json.dumps({"error": "자산 조회 시스템에 연결할 수 없습니다."}, ensure_ascii=False)

    except requests.exceptions.HTTPError as e:
        logger.error(f"API 서버 응답 오류: {e}")
        return json.dumps({"error": "서버 오류가 발생했습니다."}, ensure_ascii=False)

    except requests.exceptions.RequestException as e:
        logger.error(f"API 요청 중 알 수 없는 오류: {e}")
        return json.dumps({"error": "데이터 조회 중 문제가 발생했습니다."}, ensure_ascii=False)


@tool
def open_usage_prediction_page(user_question_context: str) -> str:
    """
    사용자 질문 내용을 바탕으로 [사용주기 예측] 페이지로 이동하는 URL을 생성합니다.
    """
    MAX_CONTEXT_LENGTH = 500
    # 경로 결합 안전성 확보
    base_path = f"{FRONTEND_BASE_URL.rstrip('/')}/prediction/analysis/prediction"

    params = {}
    
    if user_question_context:
        # 문자열 변환 및 정제
        cleaned_context = str(user_question_context).strip()
        cleaned_context = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned_context)
        
        # HTML Escape
        cleaned_context = html.escape(cleaned_context)
        
        # 길이 제한 및 로그
        if len(cleaned_context) > MAX_CONTEXT_LENGTH:
            logger.warning(
                f"[Input Truncation] 입력값 길이({len(cleaned_context)})가 제한을 초과하여 잘렸습니다."
            )
            cleaned_context = cleaned_context[:MAX_CONTEXT_LENGTH]
            
            # 잘린 엔티티 처리
            last_amp = cleaned_context.rfind('&')
            last_semi = cleaned_context.rfind(';')
            if last_amp > last_semi:
                cleaned_context = cleaned_context[:last_amp]
        
        if cleaned_context:
            params['init_prompt'] = cleaned_context

    # URL 생성
    if params:
        query_string = urllib.parse.urlencode(params)
        final_url = f"{base_path}?{query_string}"
    else:
        final_url = base_path

    return json.dumps({
        "action": "navigate",
        "target_url": final_url,
        "guide_msg": "상세 분석을 위해 예측 페이지로 이동합니다."
    }, ensure_ascii=False)