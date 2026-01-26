import pytest
import json
import urllib.parse
import requests
from unittest.mock import patch, MagicMock
import os

# 모듈 임포트
from rag import dictionaries
from rag.tools import get_item_detail_info, open_usage_prediction_page

# --------------------------------------------------------------------------
# 1. Fixtures (테스트 환경 설정)
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """
    모든 테스트 실행 전 환경변수 주입 (URL, Timeout 등).
    시스템 환경변수에 의존하지 않는 독립적인 테스트 환경을 보장합니다.
    """
    envs = {
        "BACKEND_API_URL": "http://test-backend.com",
        "FRONTEND_BASE_URL": "http://test-frontend.com",
        "API_REQUEST_TIMEOUT": "3.0"
    }
    for key, value in envs.items():
        monkeypatch.setenv(key, value)

@pytest.fixture
def mock_synonyms():
    """
    동의어 사전 Mocking.
    실제 dictionaries.py 파일의 내용과 상관없이 테스트용 데이터를 강제 주입합니다.
    """
    fake_data = {
        "테스트용가짜": "진짜키워드",
        "멍멍이": "강아지",
        "kwd": "keyword"
    }
    with patch.dict(dictionaries.KEYWORD_SYNONYMS, fake_data, clear=True):
        yield

# --------------------------------------------------------------------------
# 2. get_item_detail_info (자산 조회) 테스트
# --------------------------------------------------------------------------

@pytest.mark.parametrize("invalid_input", [
    {"asset_name": None, "asset_id": None, "identification_num": None},  # 전부 None
    {"asset_name": "", "asset_id": "", "identification_num": ""},        # 전부 빈 문자열
    {"asset_name": "   ", "asset_id": None, "identification_num": ""}    # 공백만 존재
])
def test_get_item_validation_error(invalid_input):
    """[Validation] None뿐만 아니라 빈 문자열, 공백 입력 시에도 에러를 반환해야 합니다."""
    result = get_item_detail_info.invoke(invalid_input)
    data = json.loads(result)
    
    assert "error" in data
    # "필수", "입력", "누락" 중 하나는 에러 메시지에 있어야 함
    assert any(k in data["error"] for k in ["필수", "입력", "누락", "missing"])


@pytest.mark.parametrize("input_field, input_value, expected_name", [
    ("asset_id", "테스트용가짜", "진짜키워드"),       # ID 필드에 키워드 입력
    ("identification_num", "멍멍이", "강아지"),    # 관리번호 필드에 키워드 입력
    ("asset_name", "kwd", "keyword")             # 이름 필드에 동의어 입력
])
@patch("rag.tools.requests.get")
def test_get_item_smart_correction_and_synonyms(mock_get, input_field, input_value, expected_name, mock_synonyms):
    """[Smart Correction & Synonym] ID/관리번호 오입력 보정 및 동의어 변환 통합 테스트"""
    # API Mock 설정
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}
    mock_get.return_value = mock_response

    # 동적으로 입력값 생성
    input_data = {key: None for key in ["asset_name", "asset_id", "identification_num"]}
    input_data[input_field] = input_value

    # 실행
    get_item_detail_info.invoke(input_data)

    # 검증: 실제 API로 날아간 파라미터 확인
    assert mock_get.called
    called_params = mock_get.call_args.kwargs.get("params", {})

    # 1. 결과적으로 asset_name은 예상된 '진짜 이름'이어야 함
    assert called_params.get("asset_name") == expected_name
    
    # 2. 오입력된 필드(ID, 관리번호)는 API 요청에서 제거되어야 함 (asset_name일 경우는 제외)
    if input_field != "asset_name":
        assert input_field not in called_params or called_params[input_field] is None


@patch("rag.tools.requests.get")
def test_get_item_correction_conflict_prevention(mock_get, mock_synonyms):
    """[Conflict] ID에 키워드가 있어도, Name에 이미 값이 있다면 Name을 덮어쓰지 않아야 함"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}
    mock_get.return_value = mock_response

    get_item_detail_info.invoke({
        "asset_id": "테스트용가짜",  # 이것만 보면 '진짜키워드'로 바꾸고 싶겠지만
        "asset_name": "사용자입력값"   # 이미 사용자가 명시한 이름이 있음
    })

    called_params = mock_get.call_args.kwargs.get("params", {})
    
    # Name은 보정되지 않고 유지되어야 함
    assert called_params.get("asset_name") == "사용자입력값"
    # 단, 오입력된 ID는 제거되는 것이 안전함
    assert "asset_id" not in called_params or called_params["asset_id"] is None


@pytest.mark.parametrize("exception, error_keywords", [
    (requests.exceptions.Timeout, ["시간", "초과", "timeout"]),
    (requests.exceptions.ConnectionError, ["연결", "네트워크", "connection"]),
    (requests.exceptions.HTTPError, ["오류", "HTTP", "서버"])
])
@patch("rag.tools.requests.get")
def test_get_item_network_errors(mock_get, exception, error_keywords):
    """[Error Handling] 다양한 네트워크 예외 상황을 한 번에 테스트"""
    # HTTPError인 경우 raise_for_status에서 발생하므로 설정 방식이 다름
    if exception == requests.exceptions.HTTPError:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = exception("Server Error")
        mock_get.return_value = mock_response
    else:
        # 그 외(Timeout, ConnectionError)는 get 호출 즉시 발생
        mock_get.side_effect = exception("Network Error")

    result = get_item_detail_info.invoke({"asset_name": "Test"})
    data = json.loads(result)
    
    assert "error" in data
    # 예상되는 키워드 중 하나라도 포함되어 있는지 확인
    assert any(k in data["error"].lower() for k in error_keywords)

# --------------------------------------------------------------------------
# 3. open_usage_prediction_page (페이지 이동) 테스트
# --------------------------------------------------------------------------

def test_prediction_page_url_construction():
    """[URL] 생성된 URL의 도메인, 파라미터 구조 검증"""
    context = "노트북 배터리 수명"
    result = open_usage_prediction_page.invoke({"user_question_context": context})
    data = json.loads(result)
    
    target_url = data["target_url"]
    assert "test-frontend.com" in target_url
    
    # URL 파싱하여 파라미터 정확성 검증
    parsed = urllib.parse.urlparse(target_url)
    qs = urllib.parse.parse_qs(parsed.query)
    
    assert "init_prompt" in qs
    # 인코딩된 값이 다시 디코딩되었을 때 원본과 같은지 확인
    assert qs["init_prompt"][0] == context


def test_prediction_page_security_and_limit():
    """[Security] XSS 방지 태그 제거 및 길이 제한(Truncation) 동시 검증"""
    # 500자 이상의 스크립트 공격 문자열 생성
    malicious_part = "<script>alert(1)</script>"
    long_padding = "A" * 600
    input_text = malicious_part + long_padding # 600자 넘음

    result = open_usage_prediction_page.invoke({"user_question_context": input_text})
    data = json.loads(result)
    
    target_url = data["target_url"]
    
    # URL 디코딩
    decoded_url = urllib.parse.unquote(target_url)
    
    # 1. <script> 태그가 그대로 남아있으면 안 됨 (제거되거나 이스케이프 되어야 함)
    assert "<script>" not in decoded_url
    
    # 2. 파라미터 길이 제한 확인 (URL 전체가 아니라 init_prompt 값의 길이)
    parsed = urllib.parse.urlparse(target_url)
    qs = urllib.parse.parse_qs(parsed.query)
    prompt_value = qs["init_prompt"][0]
    
    # 원본은 600자였지만, 결과는 500자 이하여야 함
    assert len(prompt_value) <= 500