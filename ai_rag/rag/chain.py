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

# [설정] 민감 정보 키 목록 정의
# 소문자로 정의하여 대소문자 구분 없이 걸러냅니다.
SENSITIVE_KEYS = {"password", "secret", "token", "auth", "apikey", "ssn", "card_number", "phone"}

# 로거 설정 (print 대신 사용)
logger = logging.getLogger(__name__)


# [최적화] 모듈 레벨 상수 정의 (서버 켜질 때 1번만 실행됨)
# 1. 사용할 도구 목록
TOOLS = [get_item_detail_info, open_usage_prediction_page]

# 2. 도구 이름으로 객체를 빠르게 찾기 위한 매핑 (Look-up Optimization)
TOOL_MAP = {}
for tool in TOOLS:
    if tool.name in TOOL_MAP:
        # 중복된 이름이 있으면 서버 시작 시 에러 발생
        logger.error(f"[Tool Registration Error] Duplicate tool name detected: {tool.name}")
        raise ValueError(f"Duplicate tool name detected: {tool.name}")
    TOOL_MAP[tool.name] = tool


def run_rag_chain(
    llm,
    vectordb,
    user_query: str,
    retriever_top_k: int = RETRIEVER_TOP_K
):
    
    # 1. Function Calling (도구 사용) 시도
    try:
        # [최적화] 매번 리스트 생성 없이 미리 만들어둔 전역 상수 TOOLS 사용
        llm_with_tools = llm.bind_tools(TOOLS)
        
        # 시스템 프롬프트 구성
        system_instruction = build_tool_aware_system_prompt()
        
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_query)
        ]
        
        # Router 단계: 도구 사용 여부 판단
        tool_check_response = llm_with_tools.invoke(messages)
        
        # ----------------------------------------------------------------------
        # 도구 호출(Tool Calls)이 감지된 경우
        # ----------------------------------------------------------------------
        if tool_check_response.tool_calls:
            logger.info(f"[Tool Check] 도구 사용 감지: {len(tool_check_response.tool_calls)}건")
            
            tool_messages = []  # 결과 누적용 리스트

            pending_navigation = None  # 페이지 이동 명령 대기용 변수

            # 감지된 모든 도구 순차 실행
            for tool_call in tool_check_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # [안전장치 1] 정의되지 않은 도구 무시
                if tool_name not in TOOL_MAP:
                    logger.warning(f"[Tool Execution] 정의되지 않은 도구 요청 무시: {tool_name}")
                    continue
                 
                # [안전장치 2] 로깅 보안 (Sanitization + Redaction)
                if isinstance(tool_args, dict):
                    safe_args = {}
                    for k, v in tool_args.items():
                        # [보안] 1. 민감한 키(Key)인지 확인 -> 마스킹 처리
                        if str(k).lower() in SENSITIVE_KEYS:
                            safe_args[k] = "[REDACTED]" # 혹은 "*****"
                        
                        # [기존] 2. 일반 데이터는 길이 제한 (Truncation)
                        else:
                            val_str = str(v).replace("\n", "\\n")
                            safe_args[k] = val_str[:100] + "..." if len(val_str) > 100 else val_str
                
                else:
                    # 딕셔너리가 아닌 경우 (단일 문자열 등) -> 기존 방식 유지
                    safe_args = str(tool_args)[:100].replace("\n", "\\n")
                
                logger.info(f"[Tool Execution] '{tool_name}' 실행 중... | 인자: {safe_args}")
                
                try:
                    # 도구 객체 가져오기 및 실행
                    selected_tool = TOOL_MAP[tool_name]
                    tool_output_str = selected_tool.invoke(tool_args)
                    
                    # 결과 분석: JSON 파싱 시도
                    parsed_output = None
                    try:
                        parsed_output = json.loads(tool_output_str)
                    except (json.JSONDecodeError, TypeError):
                        # JSON 파싱에 실패한 경우, 도구 출력은 일반 텍스트로 처리하기 위해 parsed_output을 None으로 유지합니다.
                        parsed_output = None

                    # -------------------------------------------------------
                    # [Case A] 페이지 이동 (Navigate) -> 임시 저장 (즉시 종료 X)
                    # -------------------------------------------------------
                    if (
                        isinstance(parsed_output, dict)
                        and parsed_output.get("action") == "navigate"
                        and parsed_output.get("target_url")
                    ):
                        logger.info("[Tool Output] 페이지 이동 요청 감지 -> 다른 작업 완료 후 이동 예정")
                        
                        # 즉시 return 하지 않고 변수에 저장해둡니다.
                        pending_navigation = {
                            "answer": parsed_output.get("guide_msg", "페이지로 이동합니다."),
                            "target_url": parsed_output.get("target_url"),
                            "query": user_query,
                            "action": "navigate"
                        }
                        continue  # 다음 도구 처리나 로직으로 넘어감

                    # -------------------------------------------------------
                    # [Case B] 정보 조회 (Search) -> 결과 누적(Append)
                    # -------------------------------------------------------
                    
                    # 타입 체크 및 안전한 문자열 변환 로직
                    final_content = ""
                    
                    if isinstance(tool_output_str, str):
                        # 정상적인 문자열인 경우 그대로 사용
                        final_content = tool_output_str
                    else:
                        # 문자열이 아닌 경우 (예: Dict, List, Int 등)
                        # 1. 코파일럿 지적 반영: 버그를 숨기지 않도록 경고 로그 출력
                        logger.warning(f"[Tool Warning] {tool_name} 도구가 문자열이 아닌 타입({type(tool_output_str)})을 반환했습니다. 자동 변환합니다.")
                        
                        # 2. LLM이 이해하기 쉬운 JSON 형태의 문자열로 변환 (실패 시 일반 str 변환)
                        try:
                            final_content = json.dumps(tool_output_str, ensure_ascii=False)
                        except:
                            final_content = str(tool_output_str)

                    logger.info(f"[Tool Output] 데이터 조회 완료. (메시지 이력에 추가)")
                    
                    tool_messages.append(
                        ToolMessage(
                            content=final_content,  # 검증된 문자열 사용
                            tool_call_id=tool_call["id"],
                            name=tool_name
                        )
                    )

                except Exception as e:
                    # 개별 도구 에러 처리 (멈추지 않고 에러 메시지를 LLM에게 전달)
                    logger.error(f"[Tool Execution Error] {tool_name} 실행 실패: {e}", exc_info=True)
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                        )
                    )

            # ------------------------------------------------------------------
            # 도구 실행 후 최종 답변 생성 (Generator)
            # ------------------------------------------------------------------
            # 도구 메시지가 있거나(OR) 화면 이동 명령이 있다면 RAG를 스킵하고 여기서 처리
            if tool_messages or pending_navigation:
                
                final_answer_text = ""

                # Case A: 도구 실행 결과(데이터)가 있는 경우 -> LLM이 내용을 정리해서 답변
                if tool_messages:
                    logger.info(f"[Tool Finalizing] 총 {len(tool_messages)}건의 정보를 바탕으로 답변 생성 중...")

                    # 대화 이력 재구성: [시스템, 유저질문, (AI의 도구호출), 도구결과1, 도구결과2...]
                    history = [
                        SystemMessage(content=system_instruction),
                        HumanMessage(content=user_query),
                        tool_check_response, 
                    ] + tool_messages
                    
                    # 순수 LLM으로 최종 답변 생성
                    final_response = llm.invoke(history)
                    final_answer_text = final_response.content

                # Case B: 데이터는 없지만 화면 이동 명령만 있는 경우 ("설정 화면으로 가줘")
                else:
                    logger.info("[Tool Finalizing] 데이터 조회 없이 화면 이동만 수행합니다.")
                    final_answer_text = "요청하신 화면으로 이동합니다."

                # --------------------------------------------------------------
                # 결과 반환 처리
                # --------------------------------------------------------------
                
                # 1. 화면 이동 명령이 있는 경우
                if pending_navigation:
                    logger.info(f"[Final Step] 페이지 이동 명령 실행: {pending_navigation.get('target_url')}")
                    
                    # LLM이 만든 답변(혹은 기본 문구)을 이동 명령 패키지에 담음
                    pending_navigation["answer"] = final_answer_text
                    
                    # 이동 명령 반환 (RAG 스킵)
                    return pending_navigation

                # 2. 이동 없이 답변만 있는 경우
                return {
                    "answer": final_answer_text,
                    "attribution": []
                }
            
            # ------------------------------------------------------------------
            # 도구 결과도 없고, 이동 명령도 없는 경우 -> RAG 검색 수행
            # ------------------------------------------------------------------
            else:
                logger.info("[Tool Fallback] 유효한 도구 결과 및 이동 명령 없음 -> RAG로 전환")

        else:
            # 애초에 도구 호출이 필요 없는 질문인 경우 -> RAG로 넘어감
            logger.info("[Tool Fallback] 도구 호출 요청 없음 -> RAG로 전환")

    except Exception as e:
        logger.error(f"[Tool System Error] 도구 처리 중 오류 -> RAG로 전환: {e}", exc_info=True)

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