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

@patch("rag.chain.PromptTemplate")
def test_simple_qa_flow_no_tools(mock_prompt_template, mock_dependencies):
    """[Scenario] 도구 호출 없음 -> RAG 검색 파이프라인 실행"""
    ctx = mock_dependencies
    # 1. 사전 단계 Mock (Classifier, Refiner 체인을 PromptTemplate 레벨에서 Mock)
    mock_classifier_chain = MagicMock(name="classifier_chain")
    mock_refiner_chain = MagicMock(name="refiner_chain")
    mock_classifier_chain.invoke.return_value = {"category": "general"}
    mock_refiner_chain.invoke.return_value = "안녕"
     # PromptTemplate.from_template(...)는 실제로 PromptTemplate을 반환하고,
    # 이후 파이프 연산자(|)를 통해 체인이 생성된다고 가정한다.
    # 이를 반영하기 위해 템플릿 Mock을 생성하고 __or__ 연산 결과로 체인 Mock을 반환하도록 설정한다.
    mock_classifier_template = MagicMock(name="classifier_template")
    mock_refiner_template = MagicMock(name="refiner_template")
    mock_classifier_template.__or__.return_value = mock_classifier_chain
    mock_refiner_template.__or__.return_value = mock_refiner_chain
    # run_rag_chain 내부에서 PromptTemplate.from_template(...)가
    # 분류기, 정제기 순으로 두 번 호출된다고 가정하고 각각 템플릿 Mock을 반환
    mock_prompt_template.from_template.side_effect = [
        mock_classifier_template,
        mock_refiner_template,
    ]

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
def test_tool_execution_search(mock_tool, mock_classifier, mock_refiner, mock_dependencies):
    """[Scenario] 도구 호출 -> 검색 결과로 답변 (RAG 스킵)"""
    
    ctx = mock_dependencies

    # ------------------------------------------------------------------
    # 1. 체인(Chain)들의 결과값 직접 지정 (LLM side_effect 대신)
    # ------------------------------------------------------------------
    
    # (1) 분류기: "이건 검색(search)이나 도구 사용이 필요해" 라고 판단
    # 실제 코드가 딕셔너리를 기대하므로 딕셔너리 반환
    mock_classifier.invoke.return_value = {"category": "search"}

    # (2) 정제기: 질문을 깔끔하게 다듬어줌
    mock_refiner.invoke.return_value = "맥북 찾아줘"

    # ------------------------------------------------------------------
    # 2. Router(도구 선택기) 역할인 bound_llm 설정
    # ------------------------------------------------------------------
    # 사용자가 "맥북 찾아줘"라고 했을 때, LLM이 도구를 호출하기로 결정하는 부분
    ctx.bound_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[{
            "name": "get_item_detail_info", 
            "args": {"asset_name": "맥북"}, 
            "id": "call_1"
        }]
    )

    # ------------------------------------------------------------------
    # 3. 도구(Tool)의 실행 결과 설정
    # ------------------------------------------------------------------
    # 실제 도구가 실행되었을 때 뱉을 결과값 (보통 문자열이나 JSON)
    mock_tool.invoke.return_value = json.dumps({"result": "재고 10대 있음"})

    # ------------------------------------------------------------------
    # 4. 실행 (Run)
    # ------------------------------------------------------------------
    # 테스트 대상 함수 실행
    from rag.chain import run_rag_chain  # (import 위치는 파일 상단 권장)
    result = run_rag_chain(ctx.base_llm, ctx.vectordb, "맥북 찾아줘")

    # ------------------------------------------------------------------
    # 5. 검증 (Assert)
    # ------------------------------------------------------------------
    
    # (1) Retriever(벡터 검색)가 호출되지 않았는지 확인 (도구를 썼으니까)
    ctx.retriever.invoke.assert_not_called()
    
    # (2) 도구가 올바른 인자로 호출되었는지 확인
    mock_tool.invoke.assert_called_with({"asset_name": "맥북"})
    
    # (3) 결과값 확인 (result는 보통 딕셔너리거나 문자열)
    # 로직에 따라 도구 결과를 그대로 줄 수도 있고, LLM이 한 번 더 말할 수도 있음.
    # 여기서는 결과에 도구 리턴값이 포함되어 있는지 확인
    print("Debug Result:", result)  # 디버깅용 출력
    
    # 만약 run_rag_chain이 최종적으로 도구 결과를 'answer' 키에 담아준다면:
    assert "재고 10대 있음" in str(result.get("answer", "")) or "재고 10대 있음" in str(result)


@patch("rag.chain.PromptTemplate")
@patch("rag.chain.open_usage_prediction_page")
def test_tool_execution_navigate(mock_nav_tool, mock_prompt_template, mock_dependencies):
    """[Scenario] 도구 결과가 'navigate' -> 즉시 반환 (Early Return)"""
    ctx = mock_dependencies
    # 1. 사전 단계 Mock: Classifier / Refiner 체인을 PromptTemplate를 통해 생성되는 체인으로 가정
    classifier_chain = MagicMock()
    refiner_chain = MagicMock()
    mock_prompt_template.from_template.side_effect = [classifier_chain, refiner_chain]
    classifier_chain.invoke.return_value = {"category": "predict"}
    refiner_chain.invoke.return_value = "수명 예측해줘"

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
def test_multiple_tool_calls(mock_refiner, mock_classifier, mock_tool_func, mock_dependencies):
    """[Scenario] 다중 도구 호출 (A도 찾고 B도 찾아줘)"""
    
    ctx = mock_dependencies

    # 1. 사전 단계 Mock 설정
    # (순서에 맞게 인자명을 수정했으므로, 이제 정확한 객체에 설정이 들어갑니다)
    mock_classifier.invoke.return_value = {"category": "search"}
    mock_refiner.invoke.return_value = "A랑 B 찾아"

    # 2. Router 설정: 도구를 2번 호출하도록 리스트 구성
    tool_calls = [
        {"name": "get_item_detail_info", "args": {"asset_name": "A"}, "id": "call_1"},
        {"name": "get_item_detail_info", "args": {"asset_name": "B"}, "id": "call_2"}
    ]
    
    # bound_llm이 'tool_calls'가 담긴 메시지를 반환하게 함
    ctx.bound_llm.invoke.return_value = AIMessage(content="", tool_calls=tool_calls)

    # 3. Tool 결과 설정
    # 도구가 실행될 때마다 내뱉을 결과 (여기선 둘 다 성공했다고 가정)
    mock_tool_func.invoke.return_value = json.dumps({"status": "ok", "msg": "찾았음"})
    
    # 4. 최종 답변 LLM 설정
    # 도구 결과를 다 모아서 최종적으로 할 말
    ctx.base_llm.invoke.return_value = AIMessage(content="A와 B 모두 찾아서 완료했습니다.")

    # 5. 실행
    from rag.chain import run_rag_chain # 함수 임포트
    run_rag_chain(ctx.base_llm, ctx.vectordb, "A랑 B 찾아")

    # 6. 검증 (Assert)
    # 정말로 도구 함수(mock_tool_func)가 2번 실행되었는지 확인
    assert mock_tool_func.invoke.call_count == 2


@patch("rag.chain.query_refinement_chain")
@patch("rag.chain.question_classifier_chain")
@patch("rag.chain.get_item_detail_info")
def test_tool_error_handling_by_llm(mock_refiner, mock_classifier, mock_tool_func, mock_dependencies, caplog):
    """
    [Scenario] 도구 실행 중 에러 발생 -> LLM이 에러를 인지하고 답변 -> RAG 스킵
    """
    ctx = mock_dependencies

    # ------------------------------------------------------------------
    # 1. 사전 단계 Mock (순서 바뀐 인자에 맞춰 정확히 설정)
    # ------------------------------------------------------------------
    mock_classifier.invoke.return_value = {"category": "search"}
    mock_refiner.invoke.return_value = "검색해줘"

    # ------------------------------------------------------------------
    # 2. Router: 에러가 발생할 도구 호출 지시
    # ------------------------------------------------------------------
    ctx.bound_llm.invoke.return_value = AIMessage(
        content="", 
        tool_calls=[{
            "name": "get_item_detail_info", 
            "args": {"asset_name": "Error"}, 
            "id": "err_1"
        }]
    )

    # ------------------------------------------------------------------
    # 3. Tool에서 예외 발생 설정 (핵심)
    # ------------------------------------------------------------------
    error_message = "API Timeout"
    # 도구를 실행하면 무조건 예외가 터지도록 설정
    mock_tool_func.invoke.side_effect = Exception(error_message)

    # ------------------------------------------------------------------
    # 4. Generator (최종 답변)
    # ------------------------------------------------------------------
    # LLM은 'Tool Message'에 담긴 에러 로그를 보고 사용자에게 상황을 설명
    ctx.base_llm.invoke.return_value = AIMessage(content="조회 중에 오류가 발생했습니다.")

    # ------------------------------------------------------------------
    # 5. 실행 (로그 캡처 포함)
    # ------------------------------------------------------------------
    from rag.chain import run_rag_chain
    
    # 에러 로그를 확인해야 하므로 로그 레벨을 ERROR로 설정
    with caplog.at_level(logging.ERROR):
        result = run_rag_chain(ctx.base_llm, ctx.vectordb, "검색해줘")

    # ------------------------------------------------------------------
    # 6. 검증
    # ------------------------------------------------------------------
    
    # (1) 도구 실행 시도가 있었으므로(비록 실패했지만), RAG 검색은 호출되지 않아야 함
    ctx.retriever.invoke.assert_not_called()
    
    # (2) 사용자에게는 LLM이 생성한 안내 메시지가 나가야 함
    # (주의: result가 문자열인지 딕셔너리인지 실제 리턴 타입에 맞춰 확인 필요)
    if isinstance(result, dict):
        assert result["answer"] == "조회 중에 오류가 발생했습니다."
    else:
        assert str(result) == "조회 중에 오류가 발생했습니다."

    # (3) 로그에 실제 에러 메시지("API Timeout")가 찍혔는지 확인
    # (우리가 코드에서 logger.error(f"... {e}") 처럼 찍었을 것이므로)
    assert error_message in caplog.text


@patch("rag.chain.query_refinement_chain")
@patch("rag.chain.question_classifier_chain")
def test_fallback_on_unknown_tool(mock_classifier, mock_refiner, caplog):
    """
    모르는 도구나 애매한 요청이 들어왔을 때, 
    RAG(검색) 모드로 안전하게 넘어가는지(Fallback) 테스트
    """
    
    # ---------------------------------------------------------------
    # 1. [핵심 수정] Mock 객체에게 '행동 요령(Return Value)' 입력하기
    # ---------------------------------------------------------------
    
    # (1) 분류기(Classifier)가 '이상한 도구'나 '모호한 값'을 뱉는 상황 연출
    # 실제 코드가 .invoke()를 호출하므로, .invoke.return_value를 설정해야 함!
    mock_classifier.invoke.return_value = {
        "intent": "TOOL",
        "tool_name": "unknown_tool_xyz", # 없는 도구 이름
        "param": {}
    }

    # (2) 혹은, 분류기가 에러를 뱉어서 RAG로 넘어가는지 테스트하고 싶다면?
    # mock_classifier.invoke.side_effect = Exception("Classification Error")

    # (3) 정제 체인(Refiner)은 정상적으로 질문을 다듬어준다고 가정
    mock_refiner.invoke.return_value = "정제된 검색 질문"

    # ---------------------------------------------------------------
    # 2. 테스트 대상 함수 실행
    # ---------------------------------------------------------------
    user_query = "이상한 기능 실행해줘"
    result = run_rag_chain(user_query)

    # ---------------------------------------------------------------
    # 3. 결과 검증 (Assert)
    # ---------------------------------------------------------------
    
    # 도구가 없거나 이상하므로, 시스템은 'RAG(검색)'나 '일반 대화'로 빠져야 함.
    # 로그에 "Fallback"이나 "전환" 같은 메시지가 찍혔는지 확인
    assert "Fallback" in caplog.text or "RAG로 전환" in caplog.text
    
    # 혹은 결과값이 에러 없이 텍스트로 잘 나왔는지 확인
    assert isinstance(result, dict)
    assert "answer" in result