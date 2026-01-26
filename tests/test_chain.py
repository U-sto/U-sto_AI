import pytest
import json
import logging
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from typing import NamedTuple

# 테스트할 대상 함수
from rag.chain import run_rag_chain

# --------------------------------------------------------------------------
# 1. Fixtures (환경 설정)
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """테스트 환경변수 강제 주입"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("BACKEND_API_URL", "http://test-backend")
    monkeypatch.setenv("FRONTEND_BASE_URL", "http://test-frontend")

class MockContext(NamedTuple):
    base_llm: MagicMock      # Generator
    bound_llm: MagicMock     # Router
    vectordb: MagicMock
    retriever: MagicMock

@pytest.fixture(scope="function")
def mock_dependencies():
    """RAG 체인 의존성 Mocking 및 정리(Teardown)"""
    # 1. Mock 객체 생성
    mock_base_llm = MagicMock(name="BaseLLM")
    mock_bound_llm = MagicMock(name="BoundLLM")
    mock_vectordb = MagicMock(name="VectorDB")
    mock_retriever = MagicMock(name="Retriever")

    # 2. LLM 바인딩 설정
    mock_base_llm.bind_tools.return_value = mock_bound_llm
    
    # 체이닝 메서드 방어
    mock_base_llm.with_config.return_value = mock_base_llm
    mock_base_llm.with_fallbacks.return_value = mock_base_llm
    mock_bound_llm.with_config.return_value = mock_bound_llm
    mock_bound_llm.with_retry.return_value = mock_bound_llm

    # 3. VectorDB -> Retriever 연결
    mock_vectordb.as_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = [] # 기본: 검색 결과 없음

    context = MockContext(
        base_llm=mock_base_llm,
        bound_llm=mock_bound_llm,
        vectordb=mock_vectordb,
        retriever=mock_retriever
    )

    yield context

    # Clean-up
    mock_base_llm.reset_mock()
    mock_bound_llm.reset_mock()
    mock_vectordb.reset_mock()
    mock_retriever.reset_mock()


# --------------------------------------------------------------------------
# 2. Test Cases (완벽 격리 버전)
# --------------------------------------------------------------------------

# [수정 포인트] 모든 테스트에서 앞단 체인(Classifier, Refiner)을 Mocking 하여
# 실제 로직이나 외부 API 호출 없이 오직 'Router'와 'Tool' 로직만 검증하도록 격리함.

@patch("rag.chain.query_refinement_chain")
@patch("rag.chain.question_classifier_chain")
def test_simple_qa_flow_no_tools(mock_classifier, mock_refiner, mock_dependencies):
    """[Scenario] 도구 호출 없음 -> RAG 검색 파이프라인 실행"""
    ctx = mock_dependencies

    # 1. 사전 단계 Mock (통과만 되도록 설정)
    mock_classifier.invoke.return_value = {"category": "general"}
    mock_refiner.invoke.return_value = "안녕"

    # 2. Router: 도구 사용 안 함 (Tool Calls 비어있음)
    ctx.bound_llm.invoke.return_value = AIMessage(content="", tool_calls=[])

    # 3. Generator: RAG 답변 생성
    expected_answer = "반갑습니다."
    ctx.base_llm.invoke.return_value = AIMessage(content=expected_answer)

    # 4. 실행
    result = run_rag_chain(ctx.base_llm, ctx.vectordb, "안녕")

    # 5. 검증
    # - 도구가 없으면 검색(Retriever)이 실행되어야 함
    ctx.retriever.invoke.assert_called_once()
    assert result["answer"] == expected_answer


@patch("rag.chain.query_refinement_chain")
@patch("rag.chain.question_classifier_chain")
@patch("rag.chain.get_item_detail_info")
def test_tool_execution_search(mock_tool_func, mock_classifier, mock_refiner, mock_dependencies):
    """[Scenario] 도구 호출 -> 검색 결과로 답변 (RAG 스킵)"""
    ctx = mock_dependencies

    # 1. 사전 단계 Mock
    mock_classifier.invoke.return_value = {"category": "search"} # 아무 값이나 OK
    mock_refiner.invoke.return_value = "맥북 찾아줘"

    # 2. Router: 도구 호출 지시
    ctx.bound_llm.invoke.return_value = AIMessage(
        content="", 
        tool_calls=[{"name": "get_item_detail_info", "args": {"asset_name": "맥북"}, "id": "call_1"}]
    )

    # 3. Tool 결과
    mock_tool_func.invoke.return_value = json.dumps({"result": "재고있음"})

    # 4. Generator 답변
    ctx.base_llm.invoke.return_value = AIMessage(content="맥북 재고가 있습니다.")

    # 5. 실행
    result = run_rag_chain(ctx.base_llm, ctx.vectordb, "맥북 찾아줘")

    # 6. 검증
    # - 도구 결과가 있으므로 Retriever는 호출되지 않아야 함 (핵심 검증)
    ctx.retriever.invoke.assert_not_called()
    mock_tool_func.invoke.assert_called_with({"asset_name": "맥북"})
    assert result["answer"] == "맥북 재고가 있습니다."


@patch("rag.chain.query_refinement_chain")
@patch("rag.chain.question_classifier_chain")
@patch("rag.chain.open_usage_prediction_page")
def test_tool_execution_navigate(mock_nav_tool, mock_classifier, mock_refiner, mock_dependencies):
    """[Scenario] 도구 결과가 'navigate' -> 즉시 반환 (Early Return)"""
    ctx = mock_dependencies

    # 1. 사전 단계 Mock
    mock_classifier.invoke.return_value = {"category": "predict"}
    mock_refiner.invoke.return_value = "수명 예측해줘"

    # 2. Router
    ctx.bound_llm.invoke.return_value = AIMessage(
        content="", 
        tool_calls=[{"name": "open_usage_prediction_page", "args": {"user_question_context": "수명"}, "id": "nav_1"}]
    )

    # 3. Tool 결과 (Navigate Action)
    mock_nav_tool.invoke.return_value = json.dumps({
        "action": "navigate",
        "target_url": "https://univ-frontend.com/ai-prediction",
        "guide_msg": "이동합니다"
    }, ensure_ascii=False)

    # 4. 실행
    result = run_rag_chain(ctx.base_llm, ctx.vectordb, "수명 예측해줘")

    # 5. 검증
    assert result["action"] == "navigate"
    assert "https://univ-frontend.com/ai-prediction" in result["target_url"]
    
    # - 페이지 이동 시에는 Generator(답변생성) 단계 건너뜀
    ctx.base_llm.invoke.assert_not_called()


@patch("rag.chain.query_refinement_chain")
@patch("rag.chain.question_classifier_chain")
@patch("rag.chain.get_item_detail_info")
def test_multiple_tool_calls(mock_tool_func, mock_classifier, mock_refiner, mock_dependencies):
    """[Scenario] 다중 도구 호출"""
    ctx = mock_dependencies

    # 1. 사전 단계 Mock
    mock_classifier.invoke.return_value = {"category": "search"}
    mock_refiner.invoke.return_value = "A랑 B 찾아"

    # 2. Router: 두 번 호출
    tool_calls = [
        {"name": "get_item_detail_info", "args": {"asset_name": "A"}, "id": "1"},
        {"name": "get_item_detail_info", "args": {"asset_name": "B"}, "id": "2"}
    ]
    ctx.bound_llm.invoke.return_value = AIMessage(content="", tool_calls=tool_calls)

    mock_tool_func.invoke.return_value = '{"status": "ok"}'
    ctx.base_llm.invoke.return_value = AIMessage(content="완료")

    run_rag_chain(ctx.base_llm, ctx.vectordb, "A랑 B 찾아")

    assert mock_tool_func.invoke.call_count == 2


@patch("rag.chain.query_refinement_chain")
@patch("rag.chain.question_classifier_chain")
@patch("rag.chain.get_item_detail_info")
def test_tool_error_handling_by_llm(mock_tool_func, mock_classifier, mock_refiner, mock_dependencies, caplog):
    """
    [Scenario] 도구 실행 중 에러 발생
    Logic: 에러 발생 -> ToolMessage에 에러 담김 -> LLM이 에러를 보고 답변 -> RAG 스킵
    """
    ctx = mock_dependencies

    # 1. 사전 단계 Mock
    mock_classifier.invoke.return_value = {"category": "search"}
    mock_refiner.invoke.return_value = "검색해줘"

    # 2. Router
    ctx.bound_llm.invoke.return_value = AIMessage(
        content="", 
        tool_calls=[{"name": "get_item_detail_info", "args": {"asset_name": "Error"}, "id": "err"}]
    )

    # 3. Tool에서 예외 발생
    error_message = "API Timeout"
    mock_tool_func.invoke.side_effect = Exception(error_message)

    # 4. Generator
    # LLM은 도구 에러 메시지를 보고 상황을 설명하는 답변을 생성함
    ctx.base_llm.invoke.return_value = AIMessage(content="조회 중에 오류가 발생했습니다.")

    # 5. 실행 (로그 캡처 포함)
    with caplog.at_level(logging.ERROR):
        result = run_rag_chain(ctx.base_llm, ctx.vectordb, "검색해줘")

    # 6. 검증
    # (1) 에러가 나도 '도구 실행 결과(에러메시지)'가 존재하므로 RAG 검색은 스킵됨
    ctx.retriever.invoke.assert_not_called()
    
    # (2) 사용자에게는 LLM이 생성한 사과/오류 안내 메시지가 전달됨
    assert result["answer"] == "조회 중에 오류가 발생했습니다."

    # (3) 로그에 에러가 찍혔는지 확인
    assert error_message in caplog.text


@patch("rag.chain.query_refinement_chain")
@patch("rag.chain.question_classifier_chain")
def test_fallback_on_unknown_tool(mock_classifier, mock_refiner, mock_dependencies, caplog):
    """
    [Scenario] 존재하지 않는 도구 호출 -> 무시하고 RAG 검색으로 폴백
    """
    ctx = mock_dependencies

    # 1. 사전 단계 Mock
    mock_classifier.invoke.return_value = {"category": "chat"}
    mock_refiner.invoke.return_value = "커피 타줘"

    # 2. Router: 이상한 도구 호출
    ctx.bound_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{"name": "make_coffee", "args": {}, "id": "unknown"}]
    )

    # 3. Generator: 도구 결과가 없으므로 RAG를 타게 됨 -> RAG 결과 답변
    ctx.base_llm.invoke.return_value = AIMessage(content="RAG 검색 결과입니다.")

    with caplog.at_level(logging.WARNING):
        result = run_rag_chain(ctx.base_llm, ctx.vectordb, "커피 타줘")

    # 4. 검증
    # - 유효한 도구 결과가 없으므로 RAG 검색(Retriever)이 실행되어야 함
    ctx.retriever.invoke.assert_called_once()
    assert result["answer"] == "RAG 검색 결과입니다."
    
    # - 로그 확인
    assert "정의되지 않은 도구" in caplog.text
    assert "make_coffee" in caplog.text