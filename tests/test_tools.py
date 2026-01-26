import pytest
import json
import requests
import urllib.parse
from unittest.mock import patch, MagicMock

# 모듈 임포트
from rag.tools import get_item_detail_info, open_usage_prediction_page
from rag import dictionaries


# 공통 Fixture (환경변수 설정 및 데이터 주입)
@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """
    모든 테스트 실행 전 환경변수를 고정합니다.
    """
    monkeypatch.setenv("BACKEND_API_URL", "http://test-backend.com")
    monkeypatch.setenv("FRONTEND_BASE_URL", "http://test-frontend.com")
    monkeypatch.setenv("API_REQUEST_TIMEOUT", "3.0")  # 타임아웃 테스트용


@pytest.fixture
def mock_synonyms():
    """
    테스트용 동의어 사전을 주입합니다.
    실제 dictionaries.KEYWORD_SYNONYMS를 수정하되, 테스트 후 자동 복구됩니다.
    """
    fake_data = {
        "테스트용가짜": "진짜키워드",
        "멍멍이": "강아지"
    }
    # clear=True: 기존 데이터를 싹 비우고, 오직 fake_data만 남김 (깔끔함)
    with patch.dict(dictionaries.KEYWORD_SYNONYMS, fake_data, clear=True):
        yield


# get_item_detail_info (자산 조회) 테스트
def test_get_item_detail_info_validation_error():
    """필수 입력값이 모두 None일 때 에러 반환 확인"""
    result = get_item_detail_info.invoke({
        "asset_name": None, 
        "asset_id": None, 
        "identification_num": None
    })
    
    data = json.loads(result)
    assert "error" in data
    assert "중 하나는 필수로 입력해야 합니다" in data["error"]


@patch("rag.tools.requests.get")
def test_get_item_smart_correction_asset_id(mock_get, mock_synonyms):
    """
    [Smart Correction] asset_id에 키워드 입력 시 -> asset_name으로 이동 및 변환 확인
    """
    # API 응답 Mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": [{"name": "Success"}]}
    mock_get.return_value = mock_response

    # 실행: ID 필드에 '테스트용가짜'(키워드) 입력
    get_item_detail_info.invoke({"asset_id": "테스트용가짜"})

    # 검증: 실제 API 호출 파라미터
    called_params = mock_get.call_args[1]["params"]
    
    # 1. ID는 비워져야 함
    assert "asset_id" not in called_params
    # 2. Name으로 이동하고 '진짜키워드'로 변환되어야 함
    assert called_params["asset_name"] == "진짜키워드"


@patch("rag.tools.requests.get")
def test_get_item_smart_correction_conflict(mock_get, mock_synonyms):
    """
    [Conflict] asset_id가 키워드인데, asset_name도 이미 있는 경우
    -> asset_id는 버리고 asset_name은 유지하는지 확인
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}
    mock_get.return_value = mock_response

    # 실행: ID는 키워드, Name은 정상 입력
    get_item_detail_info.invoke({
        "asset_id": "테스트용가짜",
        "asset_name": "기존검색어"
    })

    called_params = mock_get.call_args[1]["params"]
    
    # ID는 삭제(Discard)되어야 함
    assert "asset_id" not in called_params
    # Name은 기존 입력값 유지 (덮어씌워지면 안 됨)
    assert called_params["asset_name"] == "기존검색어"


@patch("rag.tools.requests.get")
def test_get_item_synonym_conversion(mock_get, mock_synonyms):
    """단순 이름 검색 시 동의어 변환 동작 확인"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}
    mock_get.return_value = mock_response

    get_item_detail_info.invoke({"asset_name": "멍멍이"})

    called_params = mock_get.call_args[1]["params"]
    assert called_params["asset_name"] == "강아지"


@patch("rag.tools.requests.get")
def test_get_item_smart_correction_identification_num(mock_get, mock_synonyms):
    """
    [Smart Correction] identification_num에 키워드 입력 시 -> asset_name으로 이동 및 변환 확인
    (Coverage Gap: asset_id 뿐만 아니라 identification_num 로직도 검증)
    """
    # API 응답 Mock
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": [{"name": "Success"}]}
    mock_get.return_value = mock_response

    # 실행: 식별번호 필드에 숫자 대신 '테스트용가짜'(키워드) 입력
    # 시나리오: 사용자가 "식별번호가 테스트용가짜인 거 찾아줘"라고 했을 때의 오작동 방지
    get_item_detail_info.invoke({"identification_num": "테스트용가짜"})

    # 검증: 실제 API 호출 파라미터
    called_params = mock_get.call_args[1]["params"]
    
    # 1. 잘못 입력된 identification_num 필드는 제거되어야 함
    assert "identification_num" not in called_params
    
    # 2. 키워드가 asset_name으로 이동하고, 표준어('진짜키워드')로 변환되어야 함
    assert called_params["asset_name"] == "진짜키워드"


@patch("rag.tools.requests.get")
def test_get_item_smart_correction_identification_num_conflict(mock_get, mock_synonyms):
    """
    [Conflict] identification_num이 키워드로 판별되었으나, asset_name도 사용자가 명시한 경우
    -> identification_num은 버리고(Discard), 명시된 asset_name을 최우선으로 유지 확인
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}
    mock_get.return_value = mock_response

    # 실행: 식별번호는 키워드(오입력), 이름은 정상 입력
    get_item_detail_info.invoke({
        "identification_num": "테스트용가짜",  # -> Smart Correction 대상
        "asset_name": "사용자입력값"           # -> 유지되어야 할 값
    })

    called_params = mock_get.call_args[1]["params"]
    
    # 1. identification_num은 오입력이므로 API 요청에서 제외
    assert "identification_num" not in called_params
    
    # 2. asset_name은 Smart Correction 결과('진짜키워드')에 덮어씌워지지 않고, 
    #    사용자가 명시한 '사용자입력값'이 유지되어야 함
    assert called_params["asset_name"] == "사용자입력값"

@patch("rag.tools.requests.get")
def test_get_item_http_error_handling(mock_get):
    """HTTP 500 등 서버 에러 발생 시 처리"""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")
    mock_get.return_value = mock_response

    result = get_item_detail_info.invoke({"asset_name": "ErrorCase"})
    
    data = json.loads(result)
    assert "error" in data
    assert "오류가 발생했습니다" in data["error"]


@patch("rag.tools.requests.get")
def test_get_item_json_decode_error(mock_get):
    """서버가 200 OK지만 HTML 등 잘못된 형식을 줄 때"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    # .json() 호출 시 파싱 에러
    mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    mock_get.return_value = mock_response

    result = get_item_detail_info.invoke({"asset_name": "BadJson"})
    
    data = json.loads(result)
    assert "error" in data
    assert "잘못된 형식" in data["error"]


@patch("rag.tools.requests.get")
def test_get_item_timeout_handling(mock_get, mock_synonyms):
    """
    [Review 1 반영] API 요청 시간 초과(Timeout) 발생 시 처리 확인
    tools.py의 156-161 라인(Timeout 예외 처리)이 제대로 동작하는지 검증
    """
    # 1. Mock 설정: 호출 시 즉시 Timeout 예외 발생
    # 실제 5초, 10초를 기다릴 필요 없이 예외 상황만 시뮬레이션합니다.
    mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

    # 2. 실행
    result = get_item_detail_info.invoke({"asset_name": "SlowAsset"})

    # 3. 검증
    data = json.loads(result)
    assert "error" in data
    
    # 에러 메시지에 '시간', '초과', 'timeout' 등 관련 키워드가 포함되어 있는지 확인
    error_msg = data["error"]
    # 실제 tools.py에서 발생하는 정확한 문구("요청 시간이 초과되었습니다")를 우선 검증
    # 영문 에러(timeout)도 함께 방어적으로 검사
    expected_phrases = ["요청 시간이 초과되었습니다", "timeout", "timed out"]
    
    assert any(phrase in error_msg for phrase in expected_phrases), \
        f"타임아웃 관련 에러 메시지가 아닙니다. 실제 메시지: {error_msg}"


@patch("rag.tools.requests.get")
def test_get_item_connection_error_handling(mock_get, mock_synonyms):
    """
    [Review 2 반영] 네트워크 연결 실패(ConnectionError) 발생 시 처리 확인
    tools.py의 165-170 라인(ConnectionError 예외 처리)이 제대로 동작하는지 검증
    """
    # 1. Mock 설정: 인터넷 끊김, DNS 실패 등의 연결 오류 발생
    mock_get.side_effect = requests.exceptions.ConnectionError("Name resolution failed")

    # 2. 실행
    result = get_item_detail_info.invoke({"asset_name": "DisconnectedAsset"})

    # 3. 검증
    data = json.loads(result)
    assert "error" in data
    
    # 에러 메시지에 '연결', '네트워크', '접속' 등 관련 키워드가 포함되어 있는지 확인
    error_msg = data["error"]
    assert any(keyword in error_msg for keyword in ["연결", "네트워크", "실패", "connection"])


# open_usage_prediction_page (페이지 이동) 테스트
def test_prediction_page_url_structure():
    """기본적인 URL 생성 구조 확인"""
    result = open_usage_prediction_page.invoke({"user_question_context": "노트북 수명"})
    data = json.loads(result)
    
    assert data["action"] == "navigate"
    assert "test-frontend.com" in data["target_url"]
    assert "init_prompt" in data["target_url"]


def test_prediction_page_length_limit():
    """입력값이 500자를 넘을 때 정확히 잘리는지 확인"""
    long_input = "A" * 1000
    result = open_usage_prediction_page.invoke({"user_question_context": long_input})
    data = json.loads(result)
    
    # URL 파싱
    parsed = urllib.parse.urlparse(data["target_url"])
    qs = urllib.parse.parse_qs(parsed.query)
    
    # 값 복원 확인
    truncated = qs["init_prompt"][0]
    assert len(truncated) == 500
    assert truncated == "A" * 500


def test_prediction_page_broken_entity_trimming():
    """
    [Edge Case] 500자 제한으로 인해 HTML 엔티티가 잘리는 경우
    예: '...&amp' 처럼 끝나면 브라우저가 깨지므로 아예 제거해야 함
    """
    # 상황 설정:
    # 496글자 + '&' (1글자) = 497글자 (입력)
    # Escape 후: "A"*496 + "&amp;" (5글자) = 총 501글자
    # 500자 제한으로 잘리면: "A"*496 + "&amp" (마지막 ';'이 탈락된 불완전 엔티티)
    
    prefix = "A" * 496
    problematic_input = prefix + "&" 
    # escape 되면: "A"*496 + "&amp;" (총 길이 501)
    # 잘리면: "A"*496 + "&amp" (마지막 ';' 이 탈락됨)
    
    result = open_usage_prediction_page.invoke({"user_question_context": problematic_input})
    data = json.loads(result)
    
    parsed = urllib.parse.urlparse(data["target_url"])
    qs = urllib.parse.parse_qs(parsed.query)
    final_text = qs["init_prompt"][0]
    
    # 로직에 따르면 불완전한 엔티티(&amp)는 제거되어야 함
    # 따라서 "A"*496 만 남아야 함
    assert final_text == prefix
    assert not final_text.endswith("&amp")
    assert not final_text.endswith("&")


def test_prediction_page_security():
    """XSS 및 제어 문자 제거 확인"""
    # Null Byte(\x00)와 Script 태그
    malicious_input = "He\x00llo<script>"
    
    result = open_usage_prediction_page.invoke({"user_question_context": malicious_input})
    data = json.loads(result)
    
    target_url = data["target_url"]
    decoded = urllib.parse.unquote(target_url)
    
    # 1. Null Byte 제거 확인 (He + llo 가 붙어야 함)
    assert "Hello" in decoded
    assert "\x00" not in decoded
    
    # 2. HTML Escape 확인 (< -> &lt;)
    assert "<script>" not in decoded
    assert "&lt;script&gt;" in decoded