import pytest
import json
import logging
from unittest.mock import patch, MagicMock, call
from langchain_core.messages import AIMessage
from typing import NamedTuple

# 테스트할 대상 함수 임포트
from rag.chain import run_rag_chain

# 테스트 환경 및 Fixture 설정
@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """
    테스트 실행 시 필요한 환경변수를 강제로 주입합니다.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("BACKEND_API_URL", "http://test-backend")
    monkeypatch.setenv("FRONTEND_BASE_URL", "http://test-frontend")


class MockContext(NamedTuple):
    base_llm: MagicMock      # 최종 답변 생성용 (Generator)
    bound_llm: MagicMock     # 도구 판단용 (Router)
    vectordb: MagicMock      # 벡터 DB
    retriever: MagicMock     # 검색기


@pytest.fixture(scope="function")  # [명시적 선언] 함수 단위로 격리됨을 보장
def mock_dependencies():
    """
    RAG 체인의 핵심 의존성을 Mocking합니다.
    - return 대신 yield를 사용하여 Teardown(종료 처리) 시점을 확보했습니다.
    - 테스트가 끝날 때마다 모든 Mock 객체의 상태(호출 기록 등)를 명시적으로 초기화(reset_mock)하여
      테스트 간 간섭(Side Effect)을 원천 차단합니다.
    """
    # 1. Mock 객체 생성
    mock_base_llm = MagicMock(name="BaseLLM")
    mock_bound_llm = MagicMock(name="BoundLLM_WithTools")
    mock_vectordb = MagicMock(name="VectorDB")
    mock_retriever = MagicMock(name="Retriever")

    # 2. LLM 바인딩 관계 설정
    # llm.bind_tools(...) 호출 시 -> mock_bound_llm 반환
    mock_base_llm.bind_tools.return_value = mock_bound_llm
    
    # 설정 메서드 체이닝 방어 (config, fallback, retry 등)
    mock_base_llm.with_config.return_value = mock_base_llm
    mock_base_llm.with_fallbacks.return_value = mock_base_llm
    mock_bound_llm.with_config.return_value = mock_bound_llm
    mock_bound_llm.with_retry.return_value = mock_bound_llm

    # 3. VectorDB -> Retriever 연결
    mock_vectordb.as_retriever.return_value = mock_retriever
    
    # 기본 상태 설정: 검색 결과 없음 (Empty List)
    # 개별 테스트에서 필요 시 side_effect로 덮어씌워서 사용
    mock_retriever.invoke.return_value = []

    # 4. Context 객체 생성
    context = MockContext(
        base_llm=mock_base_llm,
        bound_llm=mock_bound_llm,
        vectordb=mock_vectordb,
        retriever=mock_retriever
    )

    # 5. [Teardown Pattern] 테스트 실행 전 yield -> 테스트 종료 후 reset
    yield context

    # 테스트 종료 후 정리(Clean-up) 단계
    # 다음 테스트를 위해 호출 기록과 설정을 깨끗하게 비웁니다.
    mock_base_llm.reset_mock()
    mock_bound_llm.reset_mock()
    mock_vectordb.reset_mock()
    mock_retriever.reset_mock()


# 테스트 케이스
def test_simple_qa_flow_no_tools(mock_dependencies):
    """
    [Scenario] 도구 호출 없이 일반적인 대화가 진행되는 경우 (RAG 검색으로 폴백)
    """
    ctx = mock_dependencies

    # 1. Router 단계 (bound_llm) - 첫 번째 AI 호출
    # 도구를 사용하지 않겠다는 신호(tool_calls=[])를 명확히 줍니다.
    ctx.bound_llm.invoke.return_value = AIMessage(content="", tool_calls=[])

    # 2. Generator 단계 (base_llm) - 두 번째 AI 호출
    # RAG 컨텍스트를 바탕으로 최종 답변을 생성합니다.
    expected_answer = "반갑습니다. 무엇을 도와드릴까요?"
    ctx.base_llm.invoke.return_value = AIMessage(content=expected_answer)

    # 3. 실행
    result = run_rag_chain(ctx.base_llm, ctx.vectordb, "안녕")

    # 4. 검증
    # - 도구가 없었으므로 Retriever가 호출되어야 함 (RAG 파이프라인)
    ctx.retriever.invoke.assert_called_once()
    # - 최종 답변 확인
    assert result["answer"] == expected_answer


@patch("rag.tools.get_item_detail_info")
def test_tool_execution_search(mock_tool_func, mock_dependencies):
    """
    [Scenario] 도구가 호출되고, 그 결과를 바탕으로 최종 답변을 생성하는 경우
    """
    ctx = mock_dependencies

    # 1. Router 단계 (bound_llm) -> 도구 호출 지시
    tool_call_data = {
        "name": "get_item_detail_info",
        "args": {"asset_name": "맥북"},
        "id": "call_test_1"
    }
    ctx.bound_llm.invoke.return_value = AIMessage(content="", tool_calls=[tool_call_data])

    # 2. 실제 도구(Mock) 동작 설정
    mock_tool_func.invoke.return_value = json.dumps({"result": "재고있음"})

    # 3. Generator 단계 (base_llm) -> 도구 결과를 보고 답변
    ctx.base_llm.invoke.return_value = AIMessage(content="맥북 재고가 있습니다.")

    # 4. 실행
    result = run_rag_chain(ctx.base_llm, ctx.vectordb, "맥북 찾아줘")

    # 5. 검증
    # - 도구가 실행되었으므로 Retriever(검색)는 호출되지 않아야 함 (Smart Logic)
    ctx.retriever.invoke.assert_not_called()
    # - 실제 도구 함수가 올바른 인자로 호출되었는지 확인
    mock_tool_func.assert_called_with({"asset_name": "맥북"})
    # - 최종 답변 확인
    assert result["answer"] == "맥북 재고가 있습니다."


@patch("rag.tools.open_usage_prediction_page")
def test_tool_execution_navigate(mock_nav_tool, mock_dependencies):
    """
    [Scenario] 도구 실행 결과가 '페이지 이동(navigate)'인 경우 -> 즉시 반환 (Early Return)
    """
    ctx = mock_dependencies

    # 1. Router 단계 -> 예측 페이지 도구 호출 지시
    ctx.bound_llm.invoke.return_value = AIMessage(
        content="", 
        tool_calls=[{
            "name": "open_usage_prediction_page",
            "args": {"user_question_context": "수명"},
            "id": "call_nav"
        }]
    )

    # 2. 도구(Mock)가 'navigate' 액션 JSON 반환
    mock_nav_tool.invoke.return_value = json.dumps({
        "action": "navigate",
        "target_url": "https://univ-frontend.com/ai-prediction"
    })

    # 3. 실행
    result = run_rag_chain(ctx.base_llm, ctx.vectordb, "수명 예측해줘")

    # 4. 검증
    # - 결과가 딕셔너리 형태이고 action이 navigate여야 함
    assert isinstance(result, dict)
    assert result.get("action") == "navigate"
    assert result.get("target_url") == "http://test.com/predict"

    # - [중요] 페이지 이동 시에는 최종 답변 생성(Generator) 단계를 건너뛰어야 함
    ctx.base_llm.invoke.assert_not_called()


@patch("rag.tools.get_item_detail_info")
def test_multiple_tool_calls(mock_tool_func, mock_dependencies):
    """
    [Scenario] 한 번에 여러 개의 도구를 호출하는 경우 (Multi-turn tool use)
    """
    ctx = mock_dependencies

    # 1. Router 단계 -> A와 B 두 개 검색 요청
    tool_calls = [
        {"name": "get_item_detail_info", "args": {"asset_name": "A"}, "id": "1"},
        {"name": "get_item_detail_info", "args": {"asset_name": "B"}, "id": "2"}
    ]
    ctx.bound_llm.invoke.return_value = AIMessage(content="", tool_calls=tool_calls)

    # 2. 도구 반환값 (단순 문자열로 설정)
    mock_tool_func.invoke.return_value = '{"status": "ok"}'

    # 3. Generator 단계 -> 최종 답변
    ctx.base_llm.invoke.return_value = AIMessage(content="둘 다 찾음")

    # 4. 실행
    run_rag_chain(ctx.base_llm, ctx.vectordb, "A랑 B 찾아")

    # 5. 검증
    # - 도구가 총 2번 호출되었는지 확인
    assert mock_tool_func.invoke.call_count == 2
    # - 호출 순서 무관하게 인자 확인
    mock_tool_func.invoke.assert_has_calls([
        call({"asset_name": "A"}),
        call({"asset_name": "B"})
    ], any_order=True)


@patch("rag.tools.get_item_detail_info")
def test_fallback_on_tool_error(mock_tool_func, mock_dependencies, caplog):
    """
    [Scenario] 도구 실행 중 예외가 발생하면 
    1. 사용자에겐 RAG 답변으로 폴백(Fallback)하여 부드럽게 응답하고,
    2. [중요] 시스템 로그에는 에러 내용이 확실하게 남는지 검증합니다.
    """
    ctx = mock_dependencies

    # 1. Router 단계 -> 도구 호출 시도
    ctx.bound_llm.invoke.return_value = AIMessage(
        content="", 
        tool_calls=[{"name": "get_item_detail_info", "args": {"asset_name": "Error"}, "id": "err"}]
    )

    # 2. 도구(Mock)가 치명적 예외 발생 (예: 타임아웃, 연결 끊김)
    error_message = "API Connection Failed"
    mock_tool_func.invoke.side_effect = Exception(error_message)

    # 3. Generator 단계 (RAG 폴백 후 답변)
    ctx.base_llm.invoke.return_value = AIMessage(content="RAG 기반 답변입니다.")

    # 4. 실행 및 로그 캡처
    # caplog fixture: 테스트 중에 발생한 로깅 이벤트를 포착합니다.
    # ERROR 레벨 이상의 로그가 제대로 찍히는지 감시합니다.
    with caplog.at_level(logging.ERROR):
        result = run_rag_chain(ctx.base_llm, ctx.vectordb, "검색해줘")

    # 5. 검증
    
    # (1) 사용자 경험(UX): 에러 화면 대신 RAG 답변이 나갔는지
    assert result["answer"] == "RAG 기반 답변입니다."
    
    # (2) 로직 흐름: 도구 실패 후 검색(Retriever)이 실행되었는지
    ctx.retriever.invoke.assert_called_once()

    # (3) 관측 가능성(Observability): 에러가 로그에 남았는지 확인
    # chain.py에서 logger.error(..., exc_info=True)를 사용했으므로,
    # 우리가 발생시킨 예외 메시지("API Connection Failed")가 로그 텍스트에 포함되어야 함.
    assert error_message in caplog.text 
    
    # 추가 검증: chain.py에 작성한 로그 포맷 확인 ("[Tool Execution]" 혹은 "[Tool Error]")
    assert "Tool Execution" in caplog.text or "Tool Error" in caplog.text


def test_fallback_on_unknown_tool(mock_dependencies, caplog):
    """
    [Scenario] LLM이 정의되지 않은(Unknown) 도구를 호출하려고 할 때 ->
    시스템이 멈추거나 에러를 반환하지 않고, 'RAG 검색'으로 유연하게 폴백(Fallback)하는지 확인
    """
    ctx = mock_dependencies

    # 1. Router 단계 -> 없는 도구 이름('make_coffee') 호출
    # LLM이 환각(Hallucination)으로 이상한 도구명을 뱉은 상황 가정
    ctx.bound_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{"name": "make_coffee", "args": {}, "id": "unknown_id"}],
    )

    # 2. Generator 단계 (RAG 폴백 후 답변 예상)
    ctx.base_llm.invoke.return_value = AIMessage(content="RAG 검색 결과로 답변합니다.")

    # 3. 실행 및 로그 캡처
    with caplog.at_level(logging.WARNING):
        result = run_rag_chain(ctx.base_llm, ctx.vectordb, "커피 타줘")

    # 4. 검증
    
    # (1) RAG 폴백 작동 확인
    # 이전 구현과 달리, 도구를 못 찾으면 검색(Retriever)이라도 시도해야 함
    ctx.retriever.invoke.assert_called_once()
    assert result["answer"] == "RAG 검색 결과로 답변합니다."

    # (2) 로그 확인
    # chain.py에서 "정의되지 않은 도구 요청 무시" 경고가 남았는지 확인
    assert "정의되지 않은 도구 요청 무시" in caplog.text
    # "make_coffee"라는 잘못된 도구 이름이 로그에 찍혔는지 확인
    assert "make_coffee" in caplog.text