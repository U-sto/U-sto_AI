import json
import logging
import os
import urllib.parse
import requests
import re
from typing import Optional, Tuple

from dotenv import load_dotenv
from langchain_core.tools import tool
from rag.dictionaries import KEYWORD_SYNONYMS, PREDICTION_METADATA

# 로거 설정
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# 필수 환경변수 로드
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# API 요청 타임아웃 (실수형 변환 필요, .env에 적힌 10을 기본값으로 반영)
try:
    API_REQUEST_TIMEOUT = float(os.getenv("API_REQUEST_TIMEOUT", 10))
except ValueError:
    # 혹시라도 숫자가 아닌 이상한 값이 들어오면 10초로 강제 설정
    API_REQUEST_TIMEOUT = 10.0
    logger.warning("API_REQUEST_TIMEOUT 환경변수가 숫자가 아닙니다. 기본값 10초를 사용합니다.")


# [최적화] 동의어 조회용 해시 테이블 (O(1))
_SYNONYM_LOOKUP = {k.lower(): v for k, v in KEYWORD_SYNONYMS.items()}


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
    asset_name = (asset_name.strip() or None) if asset_name else None
    asset_id = (asset_id.strip() or None) if asset_id else None
    identification_num = (identification_num.strip() or None) if identification_num else None

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
        # 소문자 키 기반의 _SYNONYM_LOOKUP 해시 테이블을 이용해 표준명으로 변환
        lookup_key = asset_name.lower()
        standard_name = _SYNONYM_LOOKUP.get(lookup_key, asset_name)
        if standard_name != asset_name:
            target_search_name = standard_name
            logger.info(f"[Synonym Match] '{asset_name}' -> '{target_search_name}'")

    # 5. 파라미터 구성
    params = {}
    if identification_num: params["identification_num"] = identification_num
    if asset_id: params["asset_id"] = asset_id
    if target_search_name: params["asset_name"] = target_search_name

    # 6. API 호출
    response = None 
    
    try:
        # urljoin 대신 f-string 사용 (이전 피드백 반영)
        api_url = f"{BACKEND_API_URL.rstrip('/')}/search"

        response = requests.get(api_url, params=params, timeout=API_REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # JSON 파싱 시도
        try:
            data = response.json()

            # ------------------------------------------------------------------
            # API 결과에 AI 예측 메타데이터 주입 (Enrichment)
            # ------------------------------------------------------------------
            
            # API 결과가 있든 없든, 우리가 가진 '표준명'이 VIP 리스트에 있는지 확인합니다.
            # 만약 API 결과(data) 안에 표준명이 들어있다면 그것을 우선 사용하고,
            # 없다면 요청했던 target_search_name을 사용합니다.
            
            check_name = target_search_name
            results = data.get("results")
            if isinstance(results, list) and results:
                # 결과의 첫 번째 항목의 이름을 가져와서 확인 (API 응답 구조에 따라 조정 필요)
                first_item = results[0]
                if isinstance(first_item, dict):
                    g2b_name = first_item.get("g2b_name")
                    if isinstance(g2b_name, str) and g2b_name.strip():
                        check_name = g2b_name

            # AI 데이터셋(PREDICTION_METADATA)에 있는지 확인
            ai_info = PREDICTION_METADATA.get(check_name)
            
            if ai_info:
                # 결과 JSON에 AI 정보를 추가해줍니다.
                data["ai_capability"] = {
                    "is_predictable": True,
                    "model_code": ai_info["code"],
                    "avg_lifespan": ai_info["lifespan"],
                    "message": "이 물품은 AI 수명 예측 모델 분석이 가능합니다."
                }
            else:
                data["ai_capability"] = {
                    "is_predictable": False,
                    "message": "이 물품은 AI 수명 예측 대상이 아닙니다."
                }
            # ------------------------------------------------------------------

            if not data.get("results"):
                msg = "조건에 맞는 물품을 찾을 수 없습니다."
                if asset_name and asset_name != target_search_name:
                    msg += f" (참고: '{asset_name}' -> '{target_search_name}' 변환 검색)"
                return json.dumps({"message": msg}, ensure_ascii=False)
            
            return json.dumps(data, ensure_ascii=False)

        except json.JSONDecodeError:
            # response가 있으면 내용을 보여주고, 없으면(None이면) "Unknown" 처리
            # (위에서 response = None으로 초기화했으므로 에러 없이 안전하게 실행됨)
            response_preview = response.text[:100] if response else "Unknown"
            
            logger.error(
                f"API 응답 JSON 파싱 실패 (응답 일부: {response_preview!r})",
                exc_info=True
            )
            
            return json.dumps({
                "error": f"서버 응답 형식이 올바르지 않습니다. (응답 일부: {response_preview})"
            }, ensure_ascii=False)

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
        # 잠재적인 XSS를 방지하기 위해 HTML 태그 제거
        cleaned_context = re.sub(r'<[^>]+>', '', cleaned_context)
        
        # 길이 제한 및 로그
        if len(cleaned_context) > MAX_CONTEXT_LENGTH:
            logger.warning(
                f"[Input Truncation] 입력값 길이({len(cleaned_context)})가 제한을 초과하여 잘렸습니다."
            )
            cleaned_context = cleaned_context[:MAX_CONTEXT_LENGTH]
        
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