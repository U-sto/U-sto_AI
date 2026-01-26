import traceback
import logging
import json

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from vectorstore.retriever import retrieve_docs
from rag.prompt import assemble_prompt, build_question_classifier_prompt, build_query_refine_prompt, build_tool_aware_system_prompt
from rag.tools import get_item_detail_info, open_usage_prediction_page
from rag.reranker import CrossEncoderReranker
from app.config import (
    NO_CONTEXT_RESPONSE, TECHNICAL_ERROR_RESPONSE, SIMILARITY_SCORE_THRESHOLD, TOP_N_CONTEXT, RETRIEVER_TOP_K,
    RERANKER_MODEL_NAME,
    RERANK_CANDIDATE_K,
    RERANK_TOP_N,
    USE_RERANKING,
    RERANK_DEBUG
)

# 로거 설정 (print 대신 사용)
logger = logging.getLogger(__name__)

def run_rag_chain(
    llm,
    vectordb,
    user_query: str,
    retriever_top_k: int = RETRIEVER_TOP_K
):
    # Function Calling (도구 사용) 우선 확인
    try:
        # 1. 도구 정의 및 바인딩
        tools = [get_item_detail_info, open_usage_prediction_page]
        
        # 도구 이름으로 객체를 빠르게 찾기 위한 매핑 (Look-up Optimization)
        tool_map = {}
        for tool in tools:
            tool_name = tool.name
            if tool_name in tool_map:
                logger.error(f"[Tool Registration Error] Duplicate tool name detected: {tool_name}")
                raise ValueError(f"Duplicate tool name detected: {tool_name}")
            tool_map[tool_name] = tool
        
        # LLM에 도구 바인딩 (이 LLM은 도구를 '호출'할 수 있는 상태)
        llm_with_tools = llm.bind_tools(tools)
        
        # 2. 시스템 프롬프트 구성 (도구 사용 가이드라인 주입)
        system_instruction = build_tool_aware_system_prompt()
        
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_query)
        ]
        
        # 3. Router 단계: 도구 사용 여부 판단
        tool_check_response = llm_with_tools.invoke(messages)
        
        # 4. 도구 호출(Tool Calls) 감지 확인
        if tool_check_response.tool_calls:
            logger.info(f"[Tool Check] 도구 사용 감지: {len(tool_check_response.tool_calls)}건")

            # 도구 실행 결과(ToolMessage)를 모아둘 리스트
            tool_messages = []

            # 감지된 모든 도구 호출 순차 실행
            for tool_call in tool_check_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # 안전장치: 알 수 없는 도구 호출 방어
                if tool_name not in tool_map:
                    logger.warning(f"[Tool Execution] 정의되지 않은 도구 요청 무시: {tool_name}")
                    continue

                logger.info(f"[Tool Execution] '{tool_name}' 실행 중... | 인자: {tool_args}")
                
                try:
                    selected_tool = tool_map[tool_name]
                    
                    # [실행] 도구 함수 호출
                    tool_output_str = selected_tool.invoke(tool_args)
                    
                    # (A) 결과 분석: 페이지 이동(Navigate) 여부 체크
                    # 도구 반환값이 JSON 형태인지 확인 (일반 텍스트일 수도 있음)
                    parsed_output = None
                    try:
                        parsed_output = json.loads(tool_output_str)
                    except (json.JSONDecodeError, TypeError):
                        pass # JSON이 아니면 일반 텍스트 결과로 간주

                    # 'navigate' 액션이면 즉시 종료하고 클라이언트에 응답 반환
                    if (
                        isinstance(parsed_output, dict)
                        and parsed_output.get("action") == "navigate"
                        and parsed_output.get("target_url") # target_url 존재 여부만 확인
                    ):
                        logger.info("[Tool Output] 사용주기 AI 예측 페이지 이동 요청 감지 -> 프롬프트 포함하여 반환")
                        return {
                            "answer": "사용주기 AI 예측 페이지로 이동합니다.",
                            "target_url": parsed_output.get("target_url"),
                            "query": user_query, # 여기서 원래 사용자의 질문을 실어줍니다!
                            "action": "navigate"
                        }

                    # (B) 일반 정보 조회(Search) 결과 처리
                    logger.info(f"[Tool Output] 데이터 조회 완료. (메시지 이력에 추가)")
                    
                    # 실행 결과를 ToolMessage 형태로 변환하여 리스트에 추가
                    tool_messages.append(
                        ToolMessage(
                            content=str(tool_output_str), # 문자열 보장
                            tool_call_id=tool_call["id"],
                            name=tool_name
                        )
                    )

                except Exception as e:
                    # 개별 도구 실행 중 에러가 나도 전체 프로세스는 죽지 않도록 방어
                    logger.error(f"[Tool Execution Error] {tool_name} 실행 실패: {e}", exc_info=True)
                    continue

            # 5. 최종 답변 생성 (Generator 단계)
            # 도구 실행 결과가 하나라도 존재할 경우에만 수행
            if tool_messages:
                logger.info(f"[Tool Finalizing] 총 {len(tool_messages)}건의 정보를 바탕으로 답변 생성 중...")

                # 대화 이력 재구성: [시스템, 질문, (도구호출의도), 도구결과1, 도구결과2...]
                history = [
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=user_query),
                    tool_check_response, # AIMessage with tool_calls
                ] + tool_messages
                
                # 최종 답변은 도구가 바인딩되지 않은 순수 'llm'을 사용
                # 이유: llm_with_tools를 쓰면 결과를 보고 또 도구를 호출하려는 루프에 빠질 수 있음.
                final_response = llm.invoke(history)

                # [중요] 도구로 답변을 완료했으므로, 함수를 즉시 종료(return)하여 
                # 아래쪽 RAG 파이프라인이 중복 실행되는 것을 방지합니다.
                return {
                    "answer": final_response.content,
                    "attribution": [], 
                    "source": "tool_execution" # 출처 명시
                }
            
            else:
                # 도구 호출은 있었으나 실질적인 결과(ToolMessage)가 없는 경우
                # 여기서 return 하지 않으므로, 자연스럽게 아래의 RAG 로직으로 흘러갑니다 (Fallback).
                logger.info("[Tool Fallback] 도구 호출이 있었으나 유효한 결과가 없어 RAG 파이프라인으로 넘어갑니다.")

        # 도구 호출이 없는 경우를 명시적으로 처리
        else:
            logger.info("[Tool Fallback] 도구 호출 요청 없음 -> RAG 검색 파이프라인으로 전환하여 답변 시도")

    except Exception as e:
        # 도구 파이프라인 전체 에러 핸들링
        logger.error(f"[Tool System Error] 도구 처리 중 오류 무시 후 RAG 전환 시도: {e}", exc_info=True)


    # 0. 질문 분류 (LLM-first 판단)
    classifier_prompt = PromptTemplate.from_template(
    build_question_classifier_prompt()
)
    classifier_chain = classifier_prompt | llm | StrOutputParser()

    classification = classifier_chain.invoke({"question": user_query}) # 사용자 질문 전달

    use_rag = classification.strip() == "NEED_RAG" # RAG 필요 여부 판단
    
    logger.info(f"[Question Classification] {classification}")

    # A. RAG 필요 없는 질문 → LLM 바로 응답
    if not use_rag:
        prompt = assemble_prompt(
            context="",              # context 없이
            question=user_query
        )

        response = llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return {
            "answer": response.content,
            "attribution": []        # RAG 미사용
        }
    # B. RAG 필요한 경우만 아래 로직 수행
    try:
        # 질문 정제
        refine_prompt = PromptTemplate.from_template(
        build_query_refine_prompt()
        )

        refine_chain = refine_prompt | llm | StrOutputParser()
        
        # LLM에게 검색어 변환 요청
        refined_query = refine_chain.invoke({"question": user_query})
        
        # 로그 확인용 - logging 모듈 사용
        logger.info(f"[Query Refinement] 원본: '{user_query}' -> 변환: '{refined_query}'")

        # 1. Retrieval (검색)
        retrieved_docs = retrieve_docs(
            vectordb=vectordb,
            query=refined_query,   # [CHANGED] user_query -> refined_query
            top_k=retriever_top_k
        )

        # 검색 실패 시 fallback
        if not retrieved_docs:
            context = ""
            attribution = []
        else:
            # 기존 filtering / reranking 로직
            context = ...
            attribution = ...


        # 2️. 유사도 점수 score 기반 필터링
        # threshold 이하 문서만 유지
        filtered_docs = [
            (doc, retrieved_score)
            for doc, retrieved_score in retrieved_docs
            if retrieved_score <= SIMILARITY_SCORE_THRESHOLD
        ]

        # threshold 통과 문서가 없는 경우 fallback
        if not filtered_docs:
            return {
                "answer": NO_CONTEXT_RESPONSE,
                "attribution": []
            }

        # reranking 직전 디버깅
        if USE_RERANKING and RERANK_DEBUG:
            logger.debug("[DEBUG] Retrieval 결과 (정렬 전):")
            for i, (doc, score) in enumerate(filtered_docs[:5]):
                logger.debug(f"  [{i}] doc_id={doc.metadata.get('doc_id')} | score={score:.4f}")
        
        # 기본값 None으로 명시
        rerank_candidates = None

        # 정렬 (Chroma는 score가 낮을수록 유사함 -> 오름차순 정렬)
        filtered_docs.sort(key=lambda x: x[1])  

        # Re-ranking 대상 후보 수 제한
        rerank_candidates = filtered_docs

        if USE_RERANKING:
            rerank_candidates = filtered_docs[:RERANK_CANDIDATE_K]

        # 3. Re-ranking
        if USE_RERANKING:
            if RERANK_DEBUG:
                logger.debug("[DEBUG] Re-ranking 적용")

            reranker = CrossEncoderReranker(RERANKER_MODEL_NAME)
            
            # 중요: Re-ranking도 '변환된 질문(refined_query)'과 문서를 비교해야 정확
            top_docs = reranker.rerank(
                query=refined_query,  # [CHANGED] user_query -> refined_query
                docs_with_scores=rerank_candidates,
                top_n=RERANK_TOP_N
            )

            # Re-ranking 후 결과 확인 로직 (logger 사용)
            if RERANK_DEBUG:
                logger.debug("[DEBUG] Re-ranking 후 최종 선택된 문서:")
                for i, doc in enumerate(top_docs):
                    logger.debug(f"  [{i}] {doc.metadata.get('title', 'No Title')} (ID: {doc.metadata.get('doc_id')})")

        else:
            # Reranking 안 쓰면 상위 N개만 선택
            top_docs = [doc for doc, _ in filtered_docs[:TOP_N_CONTEXT]]

        # 4. Context 구성
        # re-ranking 이후에는 Document 리스트만 사용
        context = "\n\n".join([
            doc.page_content for doc in top_docs 
        ])

        # 5. Chunk Attribution 구성
        attribution = [
            {"doc_id": doc.metadata.get("doc_id")}
            for doc in top_docs
        ]

        # 6. 프롬프트 생성
        prompt = assemble_prompt(
            context=context,
            question=user_query
        )
        
        # 7. LLM 답변 생성
        response = llm.invoke(
            [
                HumanMessage(content=prompt)
            ]
        )

        return {
            "answer": response.content,
            "attribution": attribution
        }

    except Exception as e:
        logger.error(f"[ERROR] LLM Chain failed: {e}")
        logger.error(f"[ERROR] Failed query: {user_query}")
        logger.error(traceback.format_exc())

        return {
            "answer": TECHNICAL_ERROR_RESPONSE,
            "attribution": []
        }

"""
RAG Chain 구성
- Retrieval: Chroma(HNSW) 기반 후보 문서 검색
- Filtering: similarity score threshold 기반 후보 필터링
- Re-ranking: CrossEncoder 기반 의미론적 재채점 및 문서 재정렬
- Context Selection: 상위 chunk 선별
- Generation: context 기반 LLM 응답 생성
- Chunk Attribution: 사용 chunk metadata 추적
"""